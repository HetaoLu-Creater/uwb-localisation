from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


from .localisation import RobustPosition2D


@dataclass(frozen=True)
class AnchorRangeState:
    """Raw, calibrated and filtered ranges with their uncertainty."""

    anchor_id: int
    input_m: float
    calibrated_m: float
    filtered_m: float
    variance_m2: float
    innovation_sigma: float
    accepted: bool
    reason: str
    measurement_variance_scale: float = 1.0
    freshness_age_s: float = 0.0
    used_for_position: bool = False
    filter_variance_m2: float = math.nan
    calibration_variance_m2: float = 0.0
    calibration_extrapolated: bool = False

    @property
    def raw_m(self) -> float:
        """Return the range supplied to the processing method.

        The live pipeline supplies raw ranges. ``TWRPacket.raw_ranges_mm`` contains
        the original received values in millimetres.
        """

        return self.input_m


@dataclass(frozen=True)
class TrackingResult:
    timestamp_s: float
    raw_x_m: float
    raw_y_m: float
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    residual_rms_m: float
    max_residual_m: float
    gdop: float
    condition_number: float
    position_measurement_accepted: bool
    position_innovation_mahalanobis_squared: float
    position_filter_reason: str
    accepted_range_count: int
    quality: str
    range_states: Mapping[int, AnchorRangeState]
    solver: RobustPosition2D
    position_process_mode: str = "normal"
    position_measurement_variance_scale: float = 1.0
    prediction_age_s: float = 0.0
    position_covariance_m2: tuple[tuple[float, float], tuple[float, float]] = (
        (math.nan, math.nan),
        (math.nan, math.nan),
    )
    calibration_covariance_m2: tuple[tuple[float, float], tuple[float, float]] = (
        (math.nan, math.nan),
        (math.nan, math.nan),
    )
    diagnostic_source: str = "current_ranges"
    # Store held and smoothed display positions separately from the tracking state.
    output_stage: dict | None = None


@dataclass(frozen=True)
class TrackingFrame:
    """Processing results for one decoded UWB frame.

    ``range_states`` contains the processed anchor ranges. ``tracking`` contains
    the position estimate when one is available. Both fields support recording
    and diagnostics for complete and partial measurements.
    """

    timestamp_s: float
    range_states: Mapping[int, AnchorRangeState]
    tracking: TrackingResult | None
    position_error_code: str | None
    position_error: str | None
    # Store display-clear events for frames with an unavailable position.
    output_stage: dict | None = None

    @property
    def has_position(self) -> bool:
        return self.tracking is not None
