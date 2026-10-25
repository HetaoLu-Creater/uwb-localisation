"""Apply monotone range calibration and estimate range uncertainty.

The functions use measured ranges and fitted calibration parameters. The
uncertainty tables describe random variation and persistent range bias.
"""

from __future__ import annotations
import numpy as np


def apply_calibration(raw, calibration):
    raw = np.asarray(raw, float)
    ab = np.asarray(calibration["coefficients"])
    c = np.asarray(calibration["hinge_c"])
    k = np.asarray(calibration["hinge_knot_m"])
    return raw * ab[:, 0] + ab[:, 1] + c * np.maximum(0.0, k - raw)


def inverse_calibration(distance, calibration):
    distance = np.asarray(distance, float)
    ab = np.asarray(calibration["coefficients"])
    c = np.asarray(calibration["hinge_c"])
    k = np.asarray(calibration["hinge_knot_m"])
    boundary = ab[:, 0] * k + ab[:, 1]
    return np.where(
        distance < boundary,
        (distance - ab[:, 1] - c * k) / (ab[:, 0] - c),
        (distance - ab[:, 1]) / ab[:, 0],
    )


def derivative(raw, calibration):
    return np.asarray(calibration["coefficients"])[:, 0] - np.asarray(
        calibration["hinge_c"]
    ) * (np.asarray(raw) < np.asarray(calibration["hinge_knot_m"]))


def range_uncertainty_components(calibrated_ranges, uncertainty):
    """Return random and systematic range uncertainty from the configured tables."""
    r = np.asarray(calibrated_ranges, float)
    shape = r.shape
    flat = r.reshape(-1, 3)
    knots = np.asarray(uncertainty["distance_knots_m"])
    random = np.asarray(uncertainty["random_sigma_m"])
    bias = np.asarray(uncertainty["bias_sigma_m"])
    rv = np.stack(
        [np.interp(flat[:, j], knots[j], random[j] ** 2) for j in range(3)], axis=-1
    )
    bv = np.stack(
        [np.interp(flat[:, j], knots[j], bias[j] ** 2) for j in range(3)], axis=-1
    )
    return np.sqrt(rv).reshape(shape), np.sqrt(bv).reshape(shape)


def range_sigma(
    calibrated_ranges,
    uncertainty,
    *,
    local_noise_sigma=None,
    correlation_variance_scale=1.0,
):
    """Interpolate range variances at the current calibrated range.

    Local noise estimated from recent measurements can increase random variance.
    The fitted systematic variance remains in the combined uncertainty.
    """
    r = np.asarray(calibrated_ranges, float)
    random, bias = range_uncertainty_components(r, uncertainty)
    if local_noise_sigma is not None:
        random = np.maximum(random, np.asarray(local_noise_sigma, float))
    if not np.isfinite(correlation_variance_scale) or correlation_variance_scale < 0:
        raise ValueError("Variance multiplier must be finite and nonnegative")
    sigma = np.sqrt(random**2 * correlation_variance_scale + bias**2)
    return np.where(np.isfinite(r), sigma, np.nan)
