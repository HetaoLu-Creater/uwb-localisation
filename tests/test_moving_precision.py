"""Check the configurable near-anchor uncertainty adjustment during movement."""

import numpy as np
import pytest

from uwb_localisation.adaptive_ranges import AdaptiveRangeParameters
from uwb_localisation.reliable_tracker import (
    NearPrecisionParameters,
    near_precision_uncertainty,
    ReliableRangeTracker,
)


def apply(parameters, *, quiet=False, conditioning_enabled=True, neff=(8.0, 8.0, 8.0)):
    return near_precision_uncertainty(
        [0.02, 0.075, 0.20],
        [0.02, 0.03, 0.04],
        [0.02, 0.017, 0.04],
        [0.008, 0.012, 0.015],
        quiet=quiet,
        effective_samples=neff,
        parameters=parameters,
        conditioning_enabled=conditioning_enabled,
    )


def test_moving_uses_posterior_variance_and_keeps_bias_exactly():
    random, bias, weight = apply(NearPrecisionParameters(moving_enabled=True))
    np.testing.assert_allclose(
        random, [0.008, np.sqrt(0.5 * 0.03**2 + 0.5 * 0.012**2), 0.04], atol=1e-14
    )
    np.testing.assert_array_equal(bias, [0.02, 0.017, 0.04])
    np.testing.assert_allclose(weight, [1.0, 0.5, 0.0], atol=1e-14)
    # The moving-range adjustment uses the posterior variance and remains
    # independent of the sample-median count.
    other = apply(
        NearPrecisionParameters(moving_enabled=True), neff=(1000.0, 1000.0, 1000.0)
    )
    np.testing.assert_array_equal(random, other[0])


def test_old_configs_and_disabled_conditioner_cannot_gain_moving_precision():
    for p, enabled in [
        (NearPrecisionParameters(), True),
        (NearPrecisionParameters(enabled=False, moving_enabled=True), True),
        (NearPrecisionParameters(moving_enabled=True), False),
    ]:
        random, bias, w = apply(p, conditioning_enabled=enabled)
        np.testing.assert_array_equal(random, [0.02, 0.03, 0.04])
        np.testing.assert_array_equal(w, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(bias, [0.02, 0.017, 0.04])


def test_quiet_results_are_identical_and_variance_transition_is_continuous():
    old = apply(NearPrecisionParameters(), quiet=True)
    new = apply(NearPrecisionParameters(moving_enabled=True), quiet=True)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)
    p = NearPrecisionParameters(moving_enabled=True)
    for edge in (0.05, 0.10):
        outputs = [
            near_precision_uncertainty(
                [d] * 3,
                [0.03] * 3,
                [0.025] * 3,
                [0.002] * 3,
                quiet=False,
                effective_samples=[1.0] * 3,
                parameters=p,
            )
            for d in (edge - 1e-10, edge + 1e-10)
        ]
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-9)
        assert min(outputs[0][0]) >= 0.005


def test_moving_enabled_requires_explicit_boolean():
    for value in ("true", 1, None):
        with pytest.raises(ValueError):
            NearPrecisionParameters(moving_enabled=value)


def test_disabling_conditioner_blocks_moving_gain_in_actual_tracker():
    xyz = np.array([[0.0, 0.0, 0.0], [0.866, 0.0, 0.0], [0.433, 0.75, 0.0]])
    cal = dict(
        coefficients=[[1.0, 0.0]] * 3,
        hinge_c=[0.0] * 3,
        hinge_knot_m=[0.6] * 3,
        uncertainty=dict(
            distance_knots_m=[[0.05, 0.5, 0.85]] * 3,
            random_sigma_m=[[0.03, 0.02, 0.04]] * 3,
            bias_sigma_m=[[0.02, 0.02, 0.1]] * 3,
        ),
    )
    tracker = ReliableRangeTracker(
        xyz,
        0.0,
        cal,
        range_parameters=AdaptiveRangeParameters(enabled=False),
        precision_parameters=NearPrecisionParameters(moving_enabled=True),
    )
    for t in np.arange(0.0, 1.0, 0.03):
        row = tracker.step(float(t), [0.025, 0.8, 0.81])
    np.testing.assert_array_equal(row["precision_blend_weight"], np.zeros(3))
