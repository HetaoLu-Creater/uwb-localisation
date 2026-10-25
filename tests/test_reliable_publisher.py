"""Check display gates, held positions and expiry."""

import copy
import json
import numpy as np
import pytest
from uwb_localisation.reliable_tracker import (
    ReliablePublisher,
    ReliabilityParameters,
    ReliableRangeTracker,
)
from uwb_localisation.output_filter import OutputParameters
from uwb_localisation.empirical import EmpiricalTracker


def row(t, *, state="updated", rms=1.0, xy=None):
    return dict(
        xy=[0.2 + t * 0.1, 0.3] if xy is None else xy,
        state=state,
        solution=dict(normalised_rms=rms),
    )


def publisher(**changes):
    return ReliablePublisher(
        OutputParameters(mode="lowpass", lowpass_tau_s=0.06, output_rate_hz=5.0),
        ReliabilityParameters(**changes),
    )


def warmed(p):
    rows = [p.step(float(t), row(t)) for t in np.arange(0.0, 0.181, 0.03)]
    assert rows[-1]["xy"] is not None
    return rows[-1]


def test_withheld_frame_keeps_source_age_and_position_but_updates_fullrate_filter():
    p = publisher()
    previous = warmed(p)
    result = p.step(0.21, row(0.21, state="predicted"))
    assert result["withheld"] and not result["new_output"]
    assert result["reason"] == "prediction_withheld"
    assert result["source_time_s"] == previous["source_time_s"]
    assert result["xy"] == previous["xy"]
    assert result["age_s"] == pytest.approx(0.06)
    assert result["filtered_time_s"] == 0.21
    assert result["filtered_xy"] != previous["filtered_xy"]
    assert result["source_state"] == "updated"
    assert result["inference_state"] == "predicted"


def test_expiration_is_one_clear_event_and_at_inspection_is_pure():
    p = publisher()
    previous = warmed(p)
    out = []
    for t in np.arange(0.21, 0.61, 0.03):
        out.append(p.step(float(t), row(t, state="predicted")))
    expired = [r for r in out if r["reason"] == "expired"]
    assert len(expired) == 1 and expired[0]["new_output"]
    assert expired[0]["state"] == "stale" and expired[0]["xy"] is None
    assert not out[-1]["new_output"]
    assert out[-1]["source_time_s"] == previous["source_time_s"]
    first = p.at(0.7)
    second = p.at(0.8)
    assert not first["new_output"] and not second["new_output"]
    assert p.clock == pytest.approx(0.60)


def test_reacquisition_clears_old_segment_and_warms_up_without_smoothing_across_it():
    p = publisher()
    old = warmed(p)
    lost = p.step(0.21, dict(xy=None, state="lost", solution=None))
    assert lost["new_output"] and lost["xy"] is None
    recovered = p.step(0.24, row(0.24, state="reacquired", xy=[2.0, 3.0]))
    assert recovered["withheld"] and recovered["xy"] is None
    assert recovered["filtered_xy"] == [2.0, 3.0]
    assert recovered["segment_id"] > old["segment_id"]
    for t in np.arange(0.27, 0.401, 0.03):
        last = p.step(float(t), row(t, xy=[2.0, 3.0]))
    assert last["xy"] == [2.0, 3.0]


def test_prefix_does_not_change_when_later_inputs_change():
    times = np.arange(0.0, 1.0, 0.03)
    a = publisher()
    b = publisher()
    for k, t in enumerate(times):
        original = row(t)
        altered = copy.deepcopy(original)
        if k > 18:
            altered["xy"] = [4.0, 5.0]
        ar = a.step(float(t), original)
        br = b.step(float(t), altered)
        if k <= 18:
            assert ar == br


@pytest.mark.parametrize("rms", [float("nan"), float("inf"), -1.0, 4.0])
def test_bad_quality_cannot_publish_even_when_clock_is_due(rms):
    p = publisher()
    warmed(p)
    result = p.step(0.21, row(0.21, rms=rms))
    assert result["withheld"] and not result["new_output"]
    assert result["reason"] == "range_disagreement"


def test_invalid_coordinate_clears_publication_even_when_state_says_updated():
    p = publisher()
    warmed(p)
    result = p.step(0.21, row(0.21, xy=[float("nan"), 0.2]))
    assert result["new_output"] and result["xy"] is None
    assert result["reason"] == "no_position"


@pytest.mark.parametrize(
    "coordinates,expected",
    [
        ([1.4, 0.2], [1.4, 0.2]),
        ([float("nan"), float("inf")], [None, None]),
    ],
)
def test_rejected_position_is_a_json_safe_diagnostic_copy(coordinates, expected):
    rejected = np.asarray(coordinates)
    inference = dict(
        xy=None,
        state="lost",
        solution=None,
        boundary_rejected=True,
        rejected_xy=rejected,
    )
    result = publisher().step(0.0, inference)
    assert result["rejected_xy"] == expected
    assert result["boundary_rejected"] and result["withheld"]
    assert result["reason"] == "outside_field_rejected"
    assert result["xy"] is None
    assert inference["rejected_xy"] is rejected
    np.testing.assert_array_equal(rejected, coordinates)
    json.dumps(result, allow_nan=False)


def test_phase_clock_produces_five_hz_without_restarting_on_actual_sample_time():
    p = publisher()
    times = np.arange(0.0, 3.001, 0.03)
    rows = [p.step(float(t), row(t)) for t in times]
    emitted = [
        t for t, r in zip(times, rows) if r["new_output"] and r["xy"] is not None
    ]
    # One warmup release at .15, then the phase clock .20,.40,...3.0.
    assert len(emitted) == 16
    assert emitted[0] == pytest.approx(0.15)
    np.testing.assert_allclose(
        emitted[1:],
        [
            0.21,
            0.42,
            0.60,
            0.81,
            1.02,
            1.2,
            1.41,
            1.62,
            1.8,
            2.01,
            2.22,
            2.4,
            2.61,
            2.82,
            3.0,
        ],
    )


def test_changing_publication_gate_never_changes_estimator_or_mutates_its_rows():
    xyz = np.array(
        [[0.0, 0.0, 0.0], [np.sqrt(3) / 2, 0.0, 0.0], [np.sqrt(3) / 4, 0.75, 0.0]]
    )
    tracker = EmpiricalTracker(xyz, 0.0)
    strict = publisher(residual_rms_sigma=0.00001)
    relaxed = publisher(residual_rms_sigma=10.0)
    releases = [0, 0]
    for t in np.arange(0.0, 1.0, 0.03):
        xy = np.array([0.3 + 0.05 * t, 0.2])
        measurement = np.linalg.norm(xyz[:, :2] - xy, axis=1) + [0.01, -0.01, 0.02]
        inference = tracker.step(float(t), measurement)
        old = copy.deepcopy(inference)
        for i, p in enumerate((strict, relaxed)):
            r = p.step(float(t), inference)
            releases[i] += r["new_output"] and r["xy"] is not None
        np.testing.assert_array_equal(inference["xy"], old["xy"])
        np.testing.assert_array_equal(inference["covariance"], old["covariance"])
        assert inference["state"] == old["state"]
    assert releases[0] == 0 and releases[1] > 0


@pytest.mark.parametrize("t", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_times_are_rejected_without_poisoning_clock(t):
    p = publisher()
    with pytest.raises(ValueError):
        p.step(t, row(0.0))
    assert p.clock is None
    with pytest.raises(ValueError):
        p.at(t)
    p.step(0.0, row(0.0))


def test_range_covariance_uses_calibration_derivative_at_conditioned_input(monkeypatch):
    xyz = np.array(
        [[0.0, 0.0, 0.0], [np.sqrt(3) / 2, 0.0, 0.0], [np.sqrt(3) / 4, 0.75, 0.0]]
    )
    calibration = dict(
        coefficients=[[0.8, -0.1]] * 3,
        hinge_c=[0.2] * 3,
        hinge_knot_m=[0.3] * 3,
        uncertainty=dict(
            distance_knots_m=[[0.05, 0.5, 0.85]] * 3,
            random_sigma_m=[[0.01, 0.01, 0.01]] * 3,
            bias_sigma_m=[[0.02, 0.02, 0.02]] * 3,
        ),
    )
    tracker = ReliableRangeTracker(xyz, 0.0, calibration)
    controlled = dict(
        conditioned_m=np.array([0.2] * 3),
        variance_m2=np.array([0.04] * 3),
        reset=True,
        diagnostics=[dict(sigma_m=0.2, measurement_accepted=True) for _ in range(3)],
        stationary_state="quiet",
        stationary_confidence=1.0,
        evidence={},
    )
    monkeypatch.setattr(tracker.conditioner, "step", lambda t, z: controlled)
    result = tracker.step(0.0, np.array([0.5] * 3))
    # Current raw is above the .3 m hinge, conditioned input is below it.
    # The calibration slope is .6 at the conditioned value, hence sigma .12.
    np.testing.assert_allclose(tracker.p.range_sigma_m, [0.12] * 3)
    for diagnostic in result["range_diagnostics"]:
        assert diagnostic["weight_unit"] == "m^-2"
        assert diagnostic["confidence_kind"] == "quiet_evidence"
