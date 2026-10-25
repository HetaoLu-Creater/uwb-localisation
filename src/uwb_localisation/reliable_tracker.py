"""Combine range filtering, calibration and position tracking.

The method processes current and past range measurements. The confidence
score expresses the margin within the display residual threshold.
"""

from dataclasses import dataclass, replace
import math
import numpy as np
from .adaptive_ranges import AdaptiveRangeFilter
from .contact_calibration import (
    apply_calibration,
    derivative,
    range_uncertainty_components,
)
from .empirical import EmpiricalTracker
from .output_filter import CausalOutputFilter, OutputParameters


@dataclass(frozen=True)
class NearPrecisionParameters:
    """Adjust uncertainty for filtered ranges near an anchor.

    The full adjustment applies within 5 cm and fades out by 10 cm.
    """

    enabled: bool = True
    moving_enabled: bool = False
    full_range_m: float = 0.05
    fadeout_range_m: float = 0.10
    minimum_random_sigma_m: float = 0.005
    contact_bias_floor_m: float = 0.02

    def __post_init__(self):
        for name in ("enabled", "moving_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"conditioned_precision.{name} must be boolean")
        for name in (
            "full_range_m",
            "fadeout_range_m",
            "minimum_random_sigma_m",
            "contact_bias_floor_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"conditioned_precision.{name} must be positive and finite"
                )
        if self.fadeout_range_m <= self.full_range_m:
            raise ValueError(
                "conditioned_precision.fadeout_range_m must exceed full_range_m"
            )


def near_precision_uncertainty(
    distance,
    random,
    bias,
    posterior,
    *,
    quiet,
    effective_samples,
    parameters,
    conditioning_enabled=True,
):
    """Adjust near-anchor range variances and retain systematic uncertainty.

    For quiet ranges, pi/(2*N_eff) approximates the variance of a sample median.
    Posterior and minimum random variances bound the adjustment. Systematic
    variance remains separate from the sample count. During motion, the adjustment
    uses the radial constant-velocity posterior variance.
    """
    random = np.asarray(random, float)
    bias = np.asarray(bias, float)
    posterior = np.nan_to_num(np.asarray(posterior, float), nan=0.0)
    weight = np.zeros(3)
    if (
        parameters.enabled
        and conditioning_enabled
        and (quiet or parameters.moving_enabled)
    ):
        weight = np.clip(
            (parameters.fadeout_range_m - np.asarray(distance, float))
            / (parameters.fadeout_range_m - parameters.full_range_m),
            0.0,
            1.0,
        )
        weight = np.nan_to_num(weight, nan=0.0)
    if quiet:
        neff = np.maximum(1.0, np.asarray(effective_samples, float))
        target_random_var = np.maximum(
            parameters.minimum_random_sigma_m**2, random**2 * np.pi / (2 * neff)
        )
        target_bias_var = np.maximum(bias**2, parameters.contact_bias_floor_m**2)
    else:
        target_random_var = np.maximum(
            parameters.minimum_random_sigma_m**2, posterior**2
        )
        target_bias_var = bias**2
    # Apply the adjustment where its blend weight is positive.
    active = weight > 0
    adjusted_random = random.copy()
    adjusted_bias = bias.copy()
    adjusted_random[active] = np.sqrt(
        (1 - weight[active]) * random[active] ** 2
        + weight[active] * target_random_var[active]
    )
    if quiet:
        adjusted_bias[active] = np.sqrt(
            (1 - weight[active]) * bias[active] ** 2
            + weight[active] * target_bias_var[active]
        )
    return np.maximum(adjusted_random, posterior), adjusted_bias, weight


class ReliableRangeTracker(EmpiricalTracker):
    def __init__(
        self,
        xyz,
        height,
        calibration,
        parameters=None,
        range_parameters=None,
        *,
        record_history=True,
        precision_parameters=None,
        field_boundary=None,
    ):
        super().__init__(xyz, height, parameters, record_history=record_history)
        self.field_boundary = field_boundary
        self.base_parameters = self.p
        self.calibration = calibration
        self.conditioner = AdaptiveRangeFilter(range_parameters)
        self.precision_parameters = precision_parameters or NearPrecisionParameters()

    def step(self, t, raw):
        raw = np.asarray(raw, float).copy()
        raw[(raw <= 0) | ~np.isfinite(raw)] = np.nan
        last_range_time = self.conditioner.last_t
        conditioning = self.conditioner.step(t, raw)
        z = apply_calibration(conditioning["conditioned_m"], self.calibration)
        calibrated = apply_calibration(raw, self.calibration)
        # Missing ranges retain their missing-value marker. A finite uncertainty
        # lookup supports sensitivity calculations during prediction.
        lookup = np.where(np.isfinite(z), z, 0.5)
        random, bias = range_uncertainty_components(
            lookup, self.calibration["uncertainty"]
        )
        slopes = derivative(
            np.where(
                np.isfinite(conditioning["conditioned_m"]),
                conditioning["conditioned_m"],
                0.5,
            ),
            self.calibration,
        )
        posterior = np.sqrt(conditioning["variance_m2"]) * slopes
        effective_samples = np.ones(3)
        dt = 0.0 if last_range_time is None else t - last_range_time
        precision_quiet = (
            self.conditioner.parameters.enabled
            and conditioning["stationary_state"] == "quiet"
        )
        if precision_quiet:
            window = self.conditioner.parameters.evidence_window_s
            for j, history in enumerate(self.conditioner.histories):
                points = [tt for tt, _ in history if tt >= t - window]
                if len(points) > 1:
                    observed_span = min(window, points[-1] - points[0])
                    effective_samples[j] = max(
                        1.0,
                        min(
                            len(points),
                            observed_span
                            / max(self.base_parameters.correlation_time_s, dt, 1e-9),
                        ),
                    )
        random, bias, precision_weight = near_precision_uncertainty(
            lookup,
            random,
            bias,
            posterior,
            quiet=precision_quiet,
            effective_samples=effective_samples,
            parameters=self.precision_parameters,
            conditioning_enabled=self.conditioner.parameters.enabled,
        )
        self.p = replace(
            self.base_parameters, range_sigma_m=tuple(random), bias_sigma_m=tuple(bias)
        )
        if conditioning["reset"] and self.x is not None:
            self.x = self.cov = None
            self.candidates.clear()
        row = super().step(t, z)
        for j, d in enumerate(conditioning["diagnostics"]):
            d["raw_sigma_m"] = d["sigma_m"]
            d["sigma_m"] = float(random[j])
            d["bias_sigma_m"] = float(bias[j])
            d["weight"] = (
                float(1 / row["effective_range_variance_m2"][j])
                if d["measurement_accepted"]
                else 0.0
            )
            d["weight_unit"] = "m^-2"
            d["confidence_kind"] = "quiet_evidence"
            d["precision_blend_weight"] = float(precision_weight[j])
            d["median_effective_samples"] = float(effective_samples[j])
        row.update(
            raw_m=raw,
            calibrated_m=calibrated,
            conditioned_m=z,
            range_diagnostics=conditioning["diagnostics"],
            stationary_state=conditioning["stationary_state"],
            stationary_confidence=conditioning["stationary_confidence"],
            range_evidence=conditioning["evidence"],
            range_reset=conditioning["reset"],
            random_variance_m2=random**2 * row["correlation_variance_scale"],
            systematic_variance_m2=bias**2,
            precision_blend_weight=precision_weight,
            median_effective_samples=effective_samples,
        )
        return row


@dataclass(frozen=True)
class ReliabilityParameters:
    warmup_s: float = 0.15
    residual_rms_sigma: float = 2.5
    maximum_source_age_s: float = 0.30
    require_current_update: bool = True

    def __post_init__(self):
        for name in ("warmup_s", "residual_rms_sigma", "maximum_source_age_s"):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be positive and finite")


class ReliablePublisher:
    """Smooth tracked positions and display accepted updates at a lower rate.

    Held positions retain their source time. Expiry and reset events clear the
    position when its supporting measurement becomes too old or tracking resets.
    """

    def __init__(self, output_parameters=None, reliability_parameters=None):
        self.parameters = output_parameters or OutputParameters(
            mode="lowpass", lowpass_tau_s=0.06, output_rate_hz=5.0
        )
        self.reliability = reliability_parameters or ReliabilityParameters()
        self.smoother = CausalOutputFilter(replace(self.parameters, output_rate_hz=0.0))
        self.start = None
        self.next_due = None
        self.last = None
        self.last_time = None
        self.last_state = "lost"
        self.segment_id = -1
        self.clock = None
        self.expiry_reported = False

    def at(self, t):
        if not math.isfinite(t) or (self.clock is not None and t < self.clock):
            raise ValueError("display time must be finite and not precede input")
        age = None if self.last_time is None else t - self.last_time
        visible = (
            self.last is not None
            and age <= self.reliability.maximum_source_age_s + 1e-12
        )
        return dict(
            xy=self.last.tolist() if visible else None,
            new_output=False,
            source_time_s=self.last_time,
            available_time_s=self.last_time,
            age_s=age,
            state=self.last_state
            if visible
            else "stale"
            if self.last is not None
            else "lost",
            source_state=self.last_state,
            segment_id=self.segment_id,
            reason="held" if visible else "stale" if self.last is not None else "lost",
            withheld=False,
            confidence_score=None,
        )

    def step(self, t, row):
        if not math.isfinite(t) or (self.clock is not None and t <= self.clock):
            raise ValueError("timestamps must be finite and increase strictly")
        self.clock = t
        old_segment = self.smoother.segment_id
        smooth = self.smoother.step(
            t, row["xy"], row["state"], reset=row.get("range_reset", False)
        )
        restart = self.smoother.segment_id != old_segment
        clear = False
        invalid_position = smooth["filtered_xy"] is None
        if restart or invalid_position:
            clear = self.last is not None
            self.last = self.last_time = None
            self.last_state = "lost"
            self.start = t
            self.next_due = t
            self.segment_id += 1
            self.expiry_reported = False
        if self.start is None:
            self.start = t
            self.next_due = t
        sol = row.get("solution")
        norm = sol["normalised_rms"] if sol is not None else math.inf
        score = (
            float(1 / (1 + (norm / self.reliability.residual_rms_sigma) ** 2))
            if math.isfinite(norm)
            else 0.0
        )
        if row.get("boundary_rejected"):
            why = "outside_field_rejected"
        elif invalid_position:
            why = "no_position"
        elif t - self.start < self.reliability.warmup_s - 1e-12:
            why = "warmup"
        elif self.reliability.require_current_update and row["state"] not in (
            "updated",
            "reacquired",
        ):
            why = "prediction_withheld"
        elif (
            not math.isfinite(norm)
            or norm < 0
            or norm > self.reliability.residual_rms_sigma
        ):
            why = "range_disagreement"
        else:
            why = None
        due = self.parameters.output_rate_hz == 0 or t + 1e-10 >= self.next_due
        publish = why is None and due
        if publish:
            self.last = np.asarray(smooth["filtered_xy"])
            self.last_time = t
            self.last_state = row["state"]
            self.expiry_reported = False
            if self.parameters.output_rate_hz:
                period = 1 / self.parameters.output_rate_hz
                ticks = math.floor((t - self.start) / period + 1e-9) + 1
                self.next_due = self.start + ticks * period
        result = self.at(t)
        expired = result["state"] == "stale" and not self.expiry_reported
        if expired:
            self.expiry_reported = True
        result.update(
            filtered_xy=smooth["filtered_xy"],
            filtered_time_s=smooth["filtered_time_s"],
            new_output=bool(publish or clear or expired),
            withheld=why is not None,
            confidence_score=score,
            reason="expired"
            if expired
            else why or ("published" if publish else "cadence_hold"),
            inference_state=row["state"],
            cutoff_hz=smooth["cutoff_hz"],
        )
        # Export a diagnostic copy for HTTP snapshots and recordings.
        rejected_xy = row.get("rejected_xy")
        if rejected_xy is not None:
            rejected_xy = [
                float(value) if math.isfinite(value) else None for value in rejected_xy
            ]
        result.update(
            boundary_rejected=bool(row.get("boundary_rejected", False)),
            rejected_xy=rejected_xy,
        )
        return result


def calibration_from_config(config):
    anchors = config["anchors"]
    return dict(
        coefficients=[
            [
                anchors[str(j)].get("calibration_a", 1.0),
                anchors[str(j)].get("calibration_b_mm", 0.0) / 1000,
            ]
            for j in range(3)
        ],
        hinge_c=[anchors[str(j)].get("calibration_hinge_c", 0.0) for j in range(3)],
        hinge_knot_m=[
            anchors[str(j)].get("calibration_hinge_knot_m", 0.0) for j in range(3)
        ],
        uncertainty=config["range_uncertainty"],
    )
