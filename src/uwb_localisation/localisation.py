from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Anchor:
    anchor_id: int
    x: float
    y: float
    z: float
    calibration_a: float = 1.0
    calibration_b_mm: float = 0.0
    calibration_hinge_c: float = 0.0
    calibration_hinge_knot_m: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.calibration_a) or self.calibration_a <= 0.0:
            raise ValueError(
                f"A{self.anchor_id} calibration_a must be a finite positive number"
            )
        if not math.isfinite(self.calibration_b_mm):
            raise ValueError(f"A{self.anchor_id} calibration_b_mm must be finite")
        if (
            not math.isfinite(self.calibration_hinge_c)
            or not math.isfinite(self.calibration_hinge_knot_m)
            or self.calibration_hinge_knot_m < 0
            or self.calibration_a - self.calibration_hinge_c <= 0
        ):
            raise ValueError(
                f"A{self.anchor_id} requires finite, monotone hinge calibration"
            )

    def calibrate(self, measured_m: float) -> float:
        """Convert a raw range to a signed calibrated observation in metres."""
        measured_m = float(measured_m)
        if not math.isfinite(measured_m):
            raise ValueError("measured distance must be finite")
        calibrated_m = (
            self.calibration_a * measured_m
            + self.calibration_b_mm / 1000.0
            + self.calibration_hinge_c
            * max(0.0, self.calibration_hinge_knot_m - measured_m)
        )
        if not math.isfinite(calibrated_m):
            raise ValueError("calibrated distance is not finite")
        return calibrated_m

    def raw_from_calibrated(self, calibrated_m: float) -> float:
        """Invert the continuous monotone calibration for simulation/test tools."""

        calibrated_m = float(calibrated_m)
        if not math.isfinite(calibrated_m):
            raise ValueError("calibrated distance must be finite")
        raw_m = (calibrated_m - self.calibration_b_mm / 1000.0) / self.calibration_a
        if raw_m < self.calibration_hinge_knot_m:
            raw_m = (
                calibrated_m
                - self.calibration_b_mm / 1000.0
                - self.calibration_hinge_c * self.calibration_hinge_knot_m
            ) / (self.calibration_a - self.calibration_hinge_c)
        if not math.isfinite(raw_m):
            raise ValueError("inverse-calibrated distance is not finite")
        return raw_m


@dataclass(frozen=True)
class RobustPosition2D:
    x: float
    y: float
    residual_rms_m: float
    max_residual_m: float
    used_anchor_ids: tuple[int, ...]
    residuals_m: tuple[float, ...]
    robust_weights: tuple[float, ...]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    gdop: float
    condition_number: float
    iterations: int
    converged: bool
    standardised_residuals: tuple[float, ...] = ()
    reduced_chi_squared: float = math.nan
    robust_reduced_cost: float = math.nan


def slant_range_model(
    state: np.ndarray,
    anchor_xyz: np.ndarray,
    tag_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate slant ranges and their position Jacobian at a fixed tag height.

    The range model includes the height difference between each anchor and the tag.
    The solver uses this model for initialisation and subsequent updates.
    """
    delta = state[None, :2] - anchor_xyz[:, :2]
    heights = tag_height - anchor_xyz[:, 2]
    predicted = np.maximum(np.sqrt(np.sum(delta**2, axis=1) + heights**2), 1e-9)
    jacobian = np.zeros((len(anchor_xyz), len(state)))
    jacobian[:, :2] = delta / predicted[:, None]
    return predicted, jacobian


def anchors_from_config(config: dict) -> dict[int, Anchor]:
    anchors: dict[int, Anchor] = {}
    for key, value in config["anchors"].items():
        anchor_id = int(key)
        anchors[anchor_id] = Anchor(
            anchor_id=anchor_id,
            x=float(value["x"]),
            y=float(value["y"]),
            z=float(value["z"]),
            calibration_a=float(value.get("calibration_a", 1.0)),
            calibration_b_mm=float(value.get("calibration_b_mm", 0.0)),
            calibration_hinge_c=float(value.get("calibration_hinge_c", 0.0)),
            calibration_hinge_knot_m=float(value.get("calibration_hinge_knot_m", 0.0)),
        )
    return anchors
