"""Check near-anchor uncertainty for quiet range measurements."""

import numpy as np
import pytest
from uwb_localisation.adaptive_ranges import AdaptiveRangeParameters
from uwb_localisation.empirical import EmpiricalParameters
from uwb_localisation.reliable_tracker import (
    NearPrecisionParameters,
    ReliableRangeTracker,
    near_precision_uncertainty,
)


def calibration():
    return dict(
        coefficients=[[1.0, 0.0]] * 3,
        hinge_c=[0.0] * 3,
        hinge_knot_m=[0.6] * 3,
        uncertainty=dict(
            distance_knots_m=[[0.05, 0.5, 0.85]] * 3,
            random_sigma_m=[[0.035, 0.02, 0.035]] * 3,
            bias_sigma_m=[[0.02, 0.02, 0.12]] * 3,
        ),
    )


def tracker(enabled=True, ranges_enabled=True):
    xyz = np.array(
        [[0.0, 0.0, 0.0], [np.sqrt(3) / 2, 0.0, 0.0], [np.sqrt(3) / 4, 0.75, 0.0]]
    )
    return ReliableRangeTracker(
        xyz,
        0.0,
        calibration(),
        EmpiricalParameters(process_spectral_density=0.03, huber_k=3.0),
        AdaptiveRangeParameters(enabled=ranges_enabled),
        precision_parameters=NearPrecisionParameters(enabled=enabled),
    )


def test_quiet_near_precision_reduces_far_link_pull_with_nonaveraged_bias_floor():
    near = tracker()
    old = tracker(False)
    for t in np.arange(0.0, 3.0, 0.03):
        # Synthetic near-A0 position with inconsistent far-link observations.
        a = near.step(float(t), [0.02, 0.8, 0.81])
        b = old.step(float(t), [0.02, 0.8, 0.81])
    assert np.linalg.norm(a["xy"]) < np.linalg.norm(b["xy"])
    assert a["precision_blend_weight"][0] == 1.0
    assert a["systematic_variance_m2"][0] >= 0.02**2
    assert a["median_effective_samples"][0] <= 0.45 / 0.06 + 1e-12
    assert a["random_variance_m2"][0] < b["random_variance_m2"][0]


def test_middle_far_ranges_are_numerically_identical_with_precision_on_or_off():
    near = tracker()
    old = tracker(False)
    for t in np.arange(0.0, 2.0, 0.03):
        raw = np.array([0.43, 0.76, 0.45]) + 0.004 * np.sin(3 * t)
        a = near.step(float(t), raw)
        b = old.step(float(t), raw)
        for key in (
            "xy",
            "covariance",
            "effective_range_variance_m2",
            "systematic_variance_m2",
        ):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["precision_blend_weight"], np.zeros(3))


def test_missing_conditioner_disables_median_precision_even_if_internal_rule_is_quiet():
    tr = tracker(ranges_enabled=False)
    for t in np.arange(0.0, 2.0, 0.03):
        r = tr.step(float(t), [0.02, 0.8, 0.81])
    np.testing.assert_array_equal(r["precision_blend_weight"], np.zeros(3))
    np.testing.assert_array_equal(r["median_effective_samples"], np.ones(3))


def test_variance_and_bias_transition_are_continuous_at_both_boundaries():
    p = NearPrecisionParameters()

    def apply(d):
        return near_precision_uncertainty(
            [d] * 3,
            [0.03] * 3,
            [0.015] * 3,
            [0.002] * 3,
            quiet=True,
            effective_samples=[7.5] * 3,
            parameters=p,
        )

    for boundary in (0.05, 0.10):
        left = apply(boundary - 1e-9)
        right = apply(boundary + 1e-9)
        np.testing.assert_allclose(left[0], right[0], atol=1e-8, rtol=0)
        np.testing.assert_allclose(left[1], right[1], atol=1e-8, rtol=0)
    assert apply(0.05)[1][0] == pytest.approx(0.02)
    assert apply(0.10)[1][0] == 0.015


def test_actual_window_and_correlation_bound_effective_samples_on_irregular_clock():
    tr = tracker()
    for t in np.arange(0.0, 3.0, 0.09):
        r = tr.step(float(t), [0.02, 0.8, 0.81])
    assert np.all(r["median_effective_samples"] <= 5.0 + 1e-12)
    for j, h in enumerate(tr.conditioner.histories):
        current = [tt for tt, _ in h if tt >= tr.last_t - 0.45]
        assert r["median_effective_samples"][j] <= len(current)


def test_prefix_is_independent_of_later_measurements_and_accepts_no_reference_input():
    a = tracker()
    b = tracker()
    for k, t in enumerate(np.arange(0.0, 2.0, 0.03)):
        raw = [0.02, 0.8, 0.81]
        future = [0.08, 0.7, 0.72] if k > 25 else raw
        ar = a.step(float(t), raw)
        br = b.step(float(t), future)
        if k <= 25:
            np.testing.assert_array_equal(ar["xy"], br["xy"])
            np.testing.assert_array_equal(
                ar["effective_range_variance_m2"], br["effective_range_variance_m2"]
            )
    with pytest.raises(TypeError):
        a.step(2.1, [0.02, 0.8, 0.81], reference_xy=[0.0, 0.0])


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": "false"},
        {"full_range_m": 0.0},
        {"fadeout_range_m": 0.04},
        {"minimum_random_sigma_m": float("nan")},
        {"contact_bias_floor_m": -1.0},
    ],
)
def test_invalid_precision_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        NearPrecisionParameters(**bad)
