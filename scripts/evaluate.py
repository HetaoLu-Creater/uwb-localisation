"""Reprocess the packaged data and calculate the reported numerical metrics."""

from __future__ import annotations
import argparse
import csv
from copy import deepcopy
import numpy as np
from data_io import ROOT, read_json, recordings, write_json
from uwb_localisation.adaptive_ranges import AdaptiveRangeParameters
from uwb_localisation.empirical import EmpiricalParameters
from uwb_localisation.field_boundary import CircularField
from uwb_localisation.output_filter import OutputParameters
from uwb_localisation.reliable_tracker import (
    ReliableRangeTracker,
    ReliablePublisher,
    ReliabilityParameters,
    NearPrecisionParameters,
    calibration_from_config,
)


def replay(recording, config, earlier=False):
    """Run the final method, or disable the two reported refinements."""
    config = deepcopy(config)
    config["conditioned_precision"]["moving_enabled"] = not earlier
    field = CircularField(**config["field_boundary"])
    xyz = np.array(
        [
            [config["anchors"][str(i)][axis] for axis in ("x", "y", "z")]
            for i in range(3)
        ]
    )
    tracker = ReliableRangeTracker(
        xyz,
        config.get("tag_height_m", 0.0),
        calibration_from_config(config),
        EmpiricalParameters(**config["empirical_filter"]),
        AdaptiveRangeParameters(**config["adaptive_ranges"]),
        record_history=False,
        precision_parameters=NearPrecisionParameters(**config["conditioned_precision"]),
        field_boundary=None if earlier else field,
    )
    display = ReliablePublisher(
        OutputParameters(**config["output_filter"]),
        ReliabilityParameters(**config["reliable_output"]),
    )
    tracked, shown = [], []
    for timestamp, raw in zip(recording["time"], recording["inputs"]):
        row = tracker.step(float(timestamp), raw)
        output = display.step(float(timestamp), row)
        tracked.append(row["xy"])
        point = output["xy"]
        shown.append(point if field.accepts(point) else None)
    return tracked, np.array([p if p is not None else [np.nan, np.nan] for p in shown])


def evaluate(config):
    results = []
    for recording in recordings():
        meta = recording["meta"]
        _, positions = replay(recording, config)
        visible = np.isfinite(positions).all(axis=1)
        row = dict(
            id=meta["id"],
            frames=len(positions),
            displayed_frames=int(visible.sum()),
            displayed_percentage=float(100 * visible.mean()),
        )
        if meta["kind"] == "stationary":
            error = positions[visible] - np.asarray(meta["reference_xy_m"])
            row["position_rmse_m"] = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
            settled = positions[visible & (recording["time"] >= 0.5)]
            deviations = settled - np.median(settled, axis=0)
            row["variation_rms_m"] = float(
                np.sqrt(np.mean(np.sum(deviations**2, axis=1)))
            )
        else:
            _, earlier = replay(recording, config, earlier=True)
            row["earlier_displayed_frames"] = int(
                np.isfinite(earlier).all(axis=1).sum()
            )
            row["earlier_displayed_percentage"] = (
                100 * row["earlier_displayed_frames"] / row["frames"]
            )
        results.append(row)
    moving = [row for row in results if "earlier_displayed_frames" in row]
    total = sum(row["frames"] for row in moving)
    final_count = sum(row["displayed_frames"] for row in moving)
    earlier_count = sum(row["earlier_displayed_frames"] for row in moving)
    return dict(
        one_metre=results,
        movement_frames=total,
        final_displayed_frames=final_count,
        earlier_displayed_frames=earlier_count,
        final_displayed_percentage=100 * final_count / total,
        earlier_displayed_percentage=100 * earlier_count / total,
        early_0p6m=evaluate_early(),
    )


def evaluate_early():
    """Check early raw ranges, offset fitting and recorded position outputs."""
    sessions = recordings("early_0p6m")
    fit = read_json(ROOT / "data/early_0p6m/offset_calibration.json")
    training = np.concatenate(
        [
            s["raw"][s["time"] < 5 - 1e-6]
            for s in sessions
            if s["meta"]["calibration_role"] == "offset_fit_first_5s"
        ]
    )
    offset = fit["reference_range_m"] - np.asarray(fit["slopes"]) * np.mean(
        training, axis=0
    )
    rows = []
    for session in sessions:
        row = dict(
            id=session["meta"]["id"],
            raw_range_medians_m=np.median(session["raw"], axis=0).tolist(),
        )
        errors = {}
        for path in sorted(session["path"].glob("positions_*.csv")):
            with path.open(newline="") as stream:
                values = list(csv.DictReader(stream))
            xy = np.array(
                [
                    [float(r["x_m"]), float(r["y_m"])]
                    for r in values
                    if r["x_m"] and r["y_m"]
                ]
            )
            errors[path.stem] = float(
                np.sqrt(
                    np.mean(
                        np.sum((xy - session["meta"]["reference_xy_m"]) ** 2, axis=1)
                    )
                )
            )
        if errors:
            row["position_rmse_m_from_historical_outputs"] = errors
        rows.append(row)
    return dict(offsets_after_m=offset, recordings=rows)


def check_results(result):
    expected = read_json(ROOT / "data/expected_results.json")
    for key in (
        "movement_frames",
        "final_displayed_frames",
        "earlier_displayed_frames",
    ):
        if result[key] != expected[key]:
            raise AssertionError(f"{key}: expected {expected[key]}, got {result[key]}")
    by_id = {row["id"]: row for row in result["one_metre"]}
    for reference in expected["one_metre"]:
        actual = by_id[reference["id"]]
        for key, value in reference.items():
            if key == "id":
                continue
            if not np.isclose(actual[key], value, rtol=0, atol=1e-8):
                raise AssertionError(
                    f"{reference['id']} {key}: expected {value}, got {actual[key]}"
                )
    fit = read_json(ROOT / "data/early_0p6m/offset_calibration.json")
    np.testing.assert_allclose(
        result["early_0p6m"]["offsets_after_m"],
        fit["offsets_after_m"],
        rtol=0,
        atol=1e-12,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/final.json")
    parser.add_argument("--output", default=ROOT / "results/evaluation.json")
    args = parser.parse_args()
    result = evaluate(read_json(args.config))
    check_results(result)
    write_json(args.output, result)
    print(f"Checks passed. Results written to {args.output}")
    print(
        f"Final method: {result['final_displayed_percentage']:.2f}% of movement frames displayed."
    )
    print(
        f"Earlier version: {result['earlier_displayed_percentage']:.2f}% of movement frames displayed."
    )


if __name__ == "__main__":
    main()
