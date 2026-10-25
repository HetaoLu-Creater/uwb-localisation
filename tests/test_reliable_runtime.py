"""Check live processing against direct tracking and display calculations."""

from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np
import pytest

from uwb_localisation.adaptive_ranges import AdaptiveRangeParameters
from uwb_localisation.empirical import EmpiricalParameters
from uwb_localisation.localisation import anchors_from_config
from uwb_localisation.output_filter import OutputParameters
from uwb_localisation.reliable_tracker import (
    ReliableRangeTracker,
    ReliablePublisher,
    ReliabilityParameters,
    calibration_from_config,
)
from uwb_localisation.runtime_tracker import make_tracker


def config():
    cfg = dict(
        estimator="reliable_range_cv",
        tag_height_m=0.0,
        anchors={
            str(i): dict(
                x=x,
                y=y,
                z=0.0,
                calibration_a=0.8,
                calibration_b_mm=-100.0,
                calibration_hinge_c=0.2,
                calibration_hinge_knot_m=0.3,
            )
            for i, (x, y) in enumerate(
                [(0.0, 0.0), (math.sqrt(3) / 2, 0.0), (math.sqrt(3) / 4, 0.75)]
            )
        },
        empirical_filter=dict(
            process_spectral_density=0.3, huber_k=1.5, innovation_gate_squared=100.0
        ),
        adaptive_ranges=asdict(AdaptiveRangeParameters()),
        output_filter=asdict(
            OutputParameters(mode="lowpass", lowpass_tau_s=0.06, output_rate_hz=5.0)
        ),
        reliable_output=asdict(ReliabilityParameters()),
        range_uncertainty=dict(
            distance_knots_m=[[0.05, 0.5, 0.85]] * 3,
            random_sigma_m=[[0.01, 0.02, 0.04]] * 3,
            bias_sigma_m=[[0.02, 0.025, 0.1]] * 3,
        ),
    )
    return cfg


def ranges(cfg, xy):
    return np.array(
        [
            a.raw_from_calibrated(np.hypot(a.x - xy[0], a.y - xy[1]))
            for a in anchors_from_config(cfg).values()
        ]
    )


def test_live_adapter_matches_direct_causal_filter_and_publisher_through_missing_and_outlier():
    cfg = config()
    live = make_tracker(cfg)
    direct = ReliableRangeTracker(
        live.xyz,
        live.height,
        calibration_from_config(cfg),
        EmpiricalParameters(**cfg["empirical_filter"]),
        AdaptiveRangeParameters(**cfg["adaptive_ranges"]),
        record_history=False,
    )
    publisher = ReliablePublisher(
        OutputParameters(**cfg["output_filter"]),
        ReliabilityParameters(**cfg["reliable_output"]),
    )
    for k, t in enumerate(np.arange(0.0, 4.0, 0.03)):
        raw = ranges(cfg, [0.2 + 0.03 * t, 0.2 + 0.01 * math.sin(t)])
        if k == 55:
            raw[0] += 0.7
        if 70 <= k <= 85:
            raw[1] = np.nan
        row = direct.step(float(t), raw)
        release = publisher.step(float(t), row)
        frame = live.process({i: v for i, v in enumerate(raw)}, float(t))
        for key in (
            "xy",
            "new_output",
            "source_time_s",
            "available_time_s",
            "state",
            "withheld",
            "reason",
            "segment_id",
        ):
            if key == "xy" and release[key] is not None:
                np.testing.assert_array_equal(frame.output_stage[key], release[key])
            else:
                assert frame.output_stage[key] == release[key]
        for j, state in frame.range_states.items():
            assert state.raw_m == pytest.approx(raw[j], nan_ok=True)
            assert state.calibrated_m == pytest.approx(
                row["calibrated_m"][j], nan_ok=True
            )
            assert state.filtered_m == pytest.approx(row["filtered_m"][j], nan_ok=True)
            assert state.variance_m2 == pytest.approx(
                row["effective_range_variance_m2"][j]
            )
            assert state.filter_variance_m2 == pytest.approx(
                row["random_variance_m2"][j]
            )
            assert state.calibration_variance_m2 == pytest.approx(
                row["systematic_variance_m2"][j]
            )
            assert state.accepted == row["range_diagnostics"][j]["measurement_accepted"]
        if row["xy"] is None:
            assert frame.tracking is None
            continue
        result = frame.tracking
        np.testing.assert_array_equal([result.x_m, result.y_m], row["xy"])
        np.testing.assert_array_equal(result.position_covariance_m2, row["covariance"])
        accepted = row["state"] in ("updated", "reacquired")
        assert result.position_measurement_accepted == accepted
        assert result.accepted_range_count == (3 if accepted else 0)
        assert result.position_process_mode == "reliable_range_cv"
        assert all(
            state.used_for_position == accepted for state in frame.range_states.values()
        )
        if accepted:
            expected = (
                np.sqrt(np.sum((live.xyz[:, :2] - row["xy"]) ** 2, axis=1))
                - row["filtered_m"]
            )
            np.testing.assert_allclose(result.solver.residuals_m, expected, atol=1e-12)
            np.testing.assert_allclose(
                result.solver.standardised_residuals,
                expected / np.sqrt(row["effective_range_variance_m2"]),
                atol=1e-12,
            )
        else:
            assert result.solver.residuals_m == ()
    assert live.filter.history == []


def test_rejected_link_is_not_replaced_by_predictor_or_claimed_used_for_position():
    cfg = config()
    live = make_tracker(cfg)
    raw = ranges(cfg, [0.35, 0.2])
    for t in np.arange(0.0, 0.6, 0.03):
        live.process(dict(enumerate(raw)), float(t))
    changed = raw.copy()
    changed[0] += 0.9
    frame = live.process(dict(enumerate(changed)), 0.6)
    assert not frame.range_states[0].accepted
    assert math.isnan(frame.range_states[0].filtered_m)
    assert frame.range_states[1].accepted
    assert not any(s.used_for_position for s in frame.range_states.values())
    assert not frame.tracking.position_measurement_accepted
    assert frame.output_stage["withheld"]
    assert frame.output_stage["reason"] == "prediction_withheld"


def test_publication_clock_does_not_change_measurement_inference():
    cfg = config()
    slow = make_tracker(cfg)
    cfg2 = deepcopy(cfg)
    cfg2["output_filter"]["output_rate_hz"] = 0.0
    full = make_tracker(cfg2)
    slow_events = full_events = 0
    for t in np.arange(0.0, 1.5, 0.03):
        raw = dict(enumerate(ranges(cfg, [0.3 + 0.015 * t, 0.2])))
        a = slow.process(raw, float(t))
        b = full.process(raw, float(t))
        np.testing.assert_array_equal(
            [a.tracking.x_m, a.tracking.y_m], [b.tracking.x_m, b.tracking.y_m]
        )
        assert (
            a.tracking.position_measurement_accepted
            == b.tracking.position_measurement_accepted
        )
        slow_events += a.output_stage["new_output"]
        full_events += b.output_stage["new_output"]
    assert 0 < slow_events < full_events


def test_missing_links_preserve_raw_absence_and_expire_publication():
    cfg = config()
    live = make_tracker(cfg)
    raw = dict(enumerate(ranges(cfg, [0.3, 0.2])))
    for t in np.arange(0.0, 0.6, 0.03):
        frame = live.process(raw, float(t))
    assert frame.output_stage["xy"] is not None
    for t in np.arange(0.6, 1.05, 0.03):
        frame = live.process({}, float(t))
    assert frame.tracking is None and frame.output_stage["xy"] is None
    assert all(
        math.isnan(s.raw_m) and math.isnan(s.filtered_m) and not s.accepted
        for s in frame.range_states.values()
    )


def test_configured_near_precision_override_is_honored_in_production():
    cfg = config()
    cfg["conditioned_precision"] = {"enabled": False}
    unchanged = make_tracker(cfg)
    enabled = deepcopy(cfg)
    enabled["conditioned_precision"]["enabled"] = True
    adjusted = make_tracker(enabled)
    raw = dict(enumerate(ranges(cfg, [0.025, 0.015])))
    for t in np.arange(0.0, 1.5, 0.03):
        baseline = unchanged.process(raw, float(t))
        gain = adjusted.process(raw, float(t))
    assert not unchanged.filter.precision_parameters.enabled
    assert adjusted.filter.precision_parameters.enabled
    assert (
        baseline.output_stage["range_diagnostics"][0]["precision_blend_weight"] == 0.0
    )
    assert gain.output_stage["range_diagnostics"][0]["precision_blend_weight"] == 1.0
    assert (
        gain.range_states[0].filter_variance_m2
        < baseline.range_states[0].filter_variance_m2
    )
    # The near-anchor adjustment retains the systematic uncertainty.
    assert (
        gain.range_states[0].calibration_variance_m2
        == baseline.range_states[0].calibration_variance_m2
    )
