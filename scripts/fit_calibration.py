"""Fit the range calibration and repeat the reported range checks.

Fitting retains the original endpoint constraints and slope regularisation.
Errors are evaluated against the confirmed inward 2 cm reference positions.
"""

from __future__ import annotations
import argparse
import numpy as np
from scipy.optimize import least_squares
from data_io import ROOT, read_json, recordings, write_json
from uwb_localisation.contact_calibration import apply_calibration, derivative

IDS = {
    2: "centre_facing_a0",
    3: "centre_facing_a2",
    4: "centre_facing_a1",
    5: "near_a0",
    6: "near_a2",
    7: "near_a1",
    14: "midpoint_a0_a1",
    15: "midpoint_a0_a2",
    16: "midpoint_a1_a2",
}
FIT = (2, 3, 5, 6, 7, 14, 15)
CHECK = (4, 16)
CONTACT_ANCHOR = {5: 0, 6: 2, 7: 1}
CONTACT_GROUP = {anchor: number for number, anchor in CONTACT_ANCHOR.items()}
INHERITED_SLOPES = np.array(
    [0.7526185706169173, 0.6448842088750375, 0.6830329761798364]
)


def fitting_rows():
    by_id = {s["meta"]["id"]: s for s in recordings()}
    rows = []
    for number, name in IDS.items():
        session = by_id[name]
        meta = session["meta"]
        xyz = np.asarray(meta["anchors_m"])
        reference = np.asarray(meta["reference_xy_m"])
        fit_point = (
            xyz[CONTACT_ANCHOR[number], :2] if number in CONTACT_ANCHOR else reference
        )
        nominal = np.sqrt(
            np.sum((xyz[:, :2] - fit_point) ** 2, axis=1) + xyz[:, 2] ** 2
        )
        confirmed = np.sqrt(
            np.sum((xyz[:, :2] - reference) ** 2, axis=1) + xyz[:, 2] ** 2
        )
        median = np.nanmedian(session["inputs"], axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(session["inputs"] - median), axis=0)
        rows.append(
            dict(
                number=number,
                id=name,
                raw_median=median,
                nominal_m=nominal,
                reference_m=confirmed,
                raw_mad=mad,
                is_contact=number in CONTACT_ANCHOR,
            )
        )
    return rows


def definitions():
    for weighted in (False, True):
        for penalty in (0.01, 0.1, 1.0):
            yield dict(
                model="contact_affine",
                raw_knot_m=0.0,
                slope_penalty_m2=penalty,
                noise_weighted=weighted,
            )
            for knot in (0.6, 0.8, 1.0):
                yield dict(
                    model="contact_hinge",
                    raw_knot_m=knot,
                    slope_penalty_m2=penalty,
                    noise_weighted=weighted,
                )


def _fit(rows, inherited, definition, contact_m):
    x = np.asarray([r["raw_median"] for r in rows])
    y = np.asarray([r["nominal_m"] for r in rows])
    mad = np.asarray([r["raw_mad"] for r in rows])
    groups = [r["number"] for r in rows]
    contact = np.asarray([r["is_contact"] for r in rows])
    k = definition["raw_knot_m"]
    hinge = definition["model"] == "contact_hinge"
    coefficients = []
    cs = []
    endpoints = []
    for j in range(3):
        own = groups.index(CONTACT_GROUP[j]) if CONTACT_GROUP[j] in groups else None
        x0 = x[own, j] if own is not None else None
        lo = np.maximum(0.0, y[:, j] - contact * contact_m)
        hi = y[:, j] + contact * contact_m
        if own is not None:
            lo[own] = contact_m
            hi[own] = contact_m
        sigma = np.maximum(0.02, mad[:, j] * inherited[j])
        w = 0.03 / sigma if definition["noise_weighted"] else np.ones(len(rows))

        def unpack(p):
            low = p[0]
            high = p[1] if hinge else low
            b = contact_m - low * x0 if x0 is not None else p[-1]
            return low, high, b

        def residual(p):
            low, high, b = unpack(p)
            pred = (
                low * x[:, j] + b + (high - low) * np.maximum(0.0, x[:, j] - k)
                if hinge
                else low * x[:, j] + b
            )
            e = (pred - np.clip(pred, lo, hi)) * w
            ae = np.abs(e)
            robust = np.sign(e) * np.sqrt(
                np.where(ae <= 0.04, e * e, 0.08 * ae - 0.0016)
            )
            return np.r_[
                robust, np.sqrt(definition["slope_penalty_m2"]) * (high - inherited[j])
            ]

        initial = (
            [inherited[j]]
            + ([inherited[j]] if hinge else [])
            + ([-0.05] if own is None else [])
        )
        lower = [0.05] * (2 if hinge else 1) + ([-2.0] if own is None else [])
        upper = [2.0] * (2 if hinge else 1) + ([2.0] if own is None else [])
        fit = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=2000,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        low, high, b = unpack(fit.x)
        c = high - low if hinge else 0.0
        coefficients.append([high, b - c * k])
        cs.append(c)
        endpoints.append(x0)
    return dict(
        coefficients=np.array(coefficients),
        hinge_c=np.array(cs),
        hinge_knot_m=np.full(3, k),
        common_contact_m=contact_m,
        own_contact_raw_medians_m=endpoints,
    )


def score(rows, fit):
    prediction = apply_calibration(np.array([r["raw_median"] for r in rows]), fit)
    reference = np.array([r["reference_m"] for r in rows])
    return dict(
        recordings=[r["id"] for r in rows],
        calibrated_range_medians_m=prediction,
        reference_ranges_m=reference,
        squared_errors_m2=(prediction - reference) ** 2,
    )


def cross_validate(rows, definition, folds):
    results = []
    for omitted in folds:
        keep = [r for r in rows if r["number"] not in omitted]
        check = [r for r in rows if r["number"] in omitted]
        fit = _fit(keep, INHERITED_SLOPES, definition, 0.02)
        results.append(score(check, fit))
    errors = np.concatenate([r["squared_errors_m2"] for r in results])
    return dict(range_rmse_cm=100 * np.sqrt(np.mean(errors)), folds=results)


def uncertainty(rows, fit):
    x = np.array([r["raw_median"] for r in rows])
    y = np.array([r["nominal_m"] for r in rows])
    calibrated = apply_calibration(x, fit)
    random = np.asarray([r["raw_mad"] for r in rows]) * derivative(x, fit)
    group = np.array([r["number"] for r in rows])
    knots, random_sigma, bias_sigma = [], [], []
    for j in range(3):
        own = group == CONTACT_GROUP[j]
        middle = (~own) & (y[:, j] < 0.6)
        far = (~own) & (y[:, j] >= 0.6)
        rand, bias = [], []
        for mask, floor in ((own, 0.02), (middle, 0.015), (far, 0.03)):
            residual = calibrated[mask, j] - y[mask, j]
            if np.array_equal(mask, own):
                residual = calibrated[mask, j] - 0.02
            rand.append(max(float(np.median(random[mask, j])), 0.005))
            bias.append(max(float(np.sqrt(np.mean(residual**2))), floor))
        random_sigma.append(rand)
        bias_sigma.append(bias)
        knots.append([0.05, float(np.median(y[middle, j])), float(np.min(y[far, j]))])
    return dict(
        distance_knots_m=knots, random_sigma_m=random_sigma, bias_sigma_m=bias_sigma
    )


def fit_calibration():
    all_rows = fitting_rows()
    rows = [r for r in all_rows if r["number"] in FIT]
    candidates = []
    for definition in definitions():
        result = cross_validate(rows, definition, [(2,), (3,), (14,), (15,)])
        candidates.append((result["range_rmse_cm"], definition))
    _, definition = min(candidates, key=lambda item: item[0])
    fit = _fit(rows, INHERITED_SLOPES, definition, 0.02)
    reserved = score([r for r in all_rows if r["number"] in CHECK], fit)
    reserved["range_rmse_cm"] = 100 * np.sqrt(np.mean(reserved["squared_errors_m2"]))
    return dict(
        calibration=fit,
        selected_definition=definition,
        inherited_slopes=INHERITED_SLOPES,
        range_uncertainty=uncertainty(rows, fit),
        leave_position_out=cross_validate(rows, definition, [(2, 3), (14,), (15,)]),
        leave_recording_out=cross_validate(rows, definition, [(n,) for n in FIT]),
        reserved_recordings=reserved,
    )


def check_calibration(result, config):
    fitted = result["calibration"]
    for i in range(3):
        anchor = config["anchors"][str(i)]
        expected = [
            anchor["calibration_a"],
            anchor["calibration_b_mm"] / 1000,
            anchor["calibration_hinge_c"],
            anchor["calibration_hinge_knot_m"],
        ]
        actual = [
            *fitted["coefficients"][i],
            fitted["hinge_c"][i],
            fitted["hinge_knot_m"][i],
        ]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    for key, actual in result["range_uncertainty"].items():
        np.testing.assert_allclose(
            actual, config["range_uncertainty"][key], rtol=0, atol=1e-7
        )
    expected = read_json(ROOT / "data/expected_results.json")["calibration_checks_cm"]
    for key, value in expected.items():
        np.testing.assert_allclose(
            result[key]["range_rmse_cm"], value, rtol=0, atol=1e-5
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/final.json")
    parser.add_argument("--output", default=ROOT / "results/calibration.json")
    args = parser.parse_args()
    result = fit_calibration()
    check_calibration(result, read_json(args.config))
    write_json(args.output, result)
    print(f"Calibration checks passed. Results written to {args.output}")
    for key in ("leave_position_out", "leave_recording_out", "reserved_recordings"):
        print(f"{key}: {result[key]['range_rmse_cm']:.2f} cm")


if __name__ == "__main__":
    main()
