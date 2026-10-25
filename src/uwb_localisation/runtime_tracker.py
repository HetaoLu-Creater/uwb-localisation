"""Shared live and replay adapter for the final method."""

from __future__ import annotations
import math
import numpy as np
from .empirical import EmpiricalParameters
from .localisation import anchors_from_config, RobustPosition2D, slant_range_model
from .tracker import AnchorRangeState, TrackingFrame, TrackingResult
from .output_filter import OutputParameters


def make_tracker(config):
    if config.get("estimator", "reliable_range_cv") != "reliable_range_cv":
        raise ValueError("This package supports the final reliable_range_cv method")
    return ReliableUWBTracker(config)


class ReliableUWBTracker:
    """Live and replay adapter for range filtering and display control.

    ReliableRangeTracker calibrates each raw range once. Per-link acceptance
    identifies readings accepted by the range filter. The position acceptance
    flag identifies a new accepted position update. ``used_for_position`` marks
    the ranges used in that update.
    """

    def __init__(self, config):
        from .adaptive_ranges import AdaptiveRangeParameters
        from .reliable_tracker import (
            ReliableRangeTracker,
            ReliablePublisher,
            ReliabilityParameters,
            NearPrecisionParameters,
            calibration_from_config,
        )

        self.config = config
        self.anchors = anchors_from_config(config)
        self.ids = sorted(self.anchors)
        if self.ids != [0, 1, 2]:
            raise ValueError("reliable_range_cv requires anchors 0,1,2")
        self.parameters = EmpiricalParameters(**config.get("empirical_filter", {}))
        self.xyz = np.array(
            [
                [self.anchors[i].x, self.anchors[i].y, self.anchors[i].z]
                for i in self.ids
            ]
        )
        self.height = float(config.get("tag_height_m", 0.0))
        from .field_boundary import CircularField

        self.filter = ReliableRangeTracker(
            self.xyz,
            self.height,
            calibration_from_config(config),
            self.parameters,
            AdaptiveRangeParameters(**config.get("adaptive_ranges", {})),
            record_history=False,
            precision_parameters=NearPrecisionParameters(
                **config.get("conditioned_precision", {})
            ),
            field_boundary=CircularField(**config.get("field_boundary", {})),
        )
        output = (
            OutputParameters(**config["output_filter"])
            if config.get("output_filter") is not None
            else None
        )
        self.publisher = ReliablePublisher(
            output, ReliabilityParameters(**config.get("reliable_output", {}))
        )
        self.output_filter = self.publisher

    def process(self, raw_ranges_m, timestamp_s):
        raw = np.array([float(raw_ranges_m.get(i, math.nan)) for i in self.ids])
        row = self.filter.step(timestamp_s, raw)
        accepted = row["state"] in ("updated", "reacquired")
        z = np.asarray(row["calibrated_m"])
        conditioned = np.asarray(row["filtered_m"])
        ids = [i for i in self.ids if np.isfinite(conditioned[i])]
        states = {}
        fit = self.config.get("calibration", {}).get("runtime_devices", {})
        for i in self.ids:
            d = row["range_diagnostics"][i]
            limits = fit.get(f"A{i}", {}).get("fit_reference_range_mm")
            extrapolated = bool(
                limits
                and np.isfinite(z[i])
                and not limits[0] / 1000 <= z[i] <= limits[1] / 1000
            )
            reason = d["reason"] + "; position_" + row["reason"]
            if extrapolated:
                reason += "; calibration_extrapolated"
            r = float(row["random_variance_m2"][i])
            c = float(row["systematic_variance_m2"][i])
            used = accepted and i in ids and d["measurement_accepted"]
            states[i] = AnchorRangeState(
                i,
                float(raw[i]),
                float(z[i]),
                float(conditioned[i]),
                r + c,
                math.nan
                if d["innovation_sigma"] is None
                else float(d["innovation_sigma"]),
                bool(d["measurement_accepted"]),
                reason,
                used_for_position=used,
                freshness_age_s=float(d["source_age_s"])
                if d["source_age_s"] is not None
                else math.inf,
                filter_variance_m2=r,
                calibration_variance_m2=c,
                calibration_extrapolated=extrapolated,
                measurement_variance_scale=row["correlation_variance_scale"],
            )
        publication = self.publisher.step(timestamp_s, row)
        publication.update(
            range_diagnostics=row["range_diagnostics"],
            stationary_state=row["stationary_state"],
            stationary_confidence=row["stationary_confidence"],
            range_evidence=row["range_evidence"],
            range_reset=row["range_reset"],
        )
        if row["xy"] is None:
            return TrackingFrame(
                float(timestamp_s),
                states,
                None,
                row["reason"],
                row["reason"],
                publication,
            )
        xy = np.asarray(row["xy"])
        velocity = self.filter.x[2:]
        temporal = np.asarray(row["covariance"])
        bias = np.asarray(row["bias_covariance"])
        total = temporal + bias
        current_ids = ids if accepted else []
        h, j = slant_range_model(xy, self.xyz[current_ids], self.height)
        # Calculate diagnostics from the same filtered ranges used by NLS.
        # Median filtering can make these differ from the calibrated raw ranges.
        residual = h - np.asarray(row["filtered_m"])[current_ids]
        standard = residual / np.sqrt(row["effective_range_variance_m2"][current_ids])
        info = j.T @ j
        gdop = (
            math.sqrt(np.trace(np.linalg.inv(info)))
            if len(current_ids) >= 3 and np.linalg.matrix_rank(info) == 2
            else math.inf
        )
        sol = row["solution"]
        weights = sol["weights"] if accepted and sol is not None else np.array([])
        diagnostic = RobustPosition2D(
            float(xy[0]),
            float(xy[1]),
            float(np.sqrt(np.mean(residual**2))) if len(residual) else math.nan,
            float(np.max(np.abs(residual))) if len(residual) else math.nan,
            tuple(current_ids),
            tuple(residual),
            tuple(weights),
            tuple(map(tuple, total)),
            float(gdop),
            float(np.linalg.cond(info)) if current_ids else math.inf,
            int(sol["iterations"]) if sol else 0,
            bool(sol["converged"]) if sol else False,
            tuple(standard),
            float(standard @ standard) if current_ids else math.nan,
        )
        uncertainty = float(np.sqrt(np.linalg.eigvalsh(total)[-1]))
        quality_config = self.config.get("quality", {})
        quality = (
            "PREDICTED"
            if not accepted
            else (
                "POOR"
                if uncertainty > quality_config.get("maximum_position_std_m", 0.25)
                or diagnostic.residual_rms_m
                > quality_config.get("degraded_residual_m", 0.10)
                or diagnostic.gdop > quality_config.get("maximum_gdop", 5.0)
                else "DEGRADED"
                if not diagnostic.converged
                or row["state"] == "reacquired"
                or diagnostic.residual_rms_m
                > quality_config.get("good_residual_m", 0.04)
                or any(s.calibration_extrapolated for s in states.values())
                else "GOOD"
            )
        )
        tracking = TrackingResult(
            timestamp_s=float(timestamp_s),
            raw_x_m=float(sol["xy"][0]) if sol is not None else math.nan,
            raw_y_m=float(sol["xy"][1]) if sol is not None else math.nan,
            x_m=float(xy[0]),
            y_m=float(xy[1]),
            vx_mps=float(velocity[0]),
            vy_mps=float(velocity[1]),
            residual_rms_m=diagnostic.residual_rms_m,
            max_residual_m=diagnostic.max_residual_m,
            gdop=diagnostic.gdop,
            condition_number=diagnostic.condition_number,
            position_measurement_accepted=accepted,
            position_innovation_mahalanobis_squared=row["nis"]
            if row["nis"] is not None
            else math.nan,
            position_filter_reason=row["reason"],
            accepted_range_count=len(current_ids),
            quality=quality,
            range_states=states,
            solver=diagnostic,
            position_process_mode="reliable_range_cv",
            prediction_age_s=float(row["age_s"] or 0.0),
            position_covariance_m2=tuple(map(tuple, temporal)),
            calibration_covariance_m2=tuple(map(tuple, bias)),
            diagnostic_source="current_ranges" if accepted else "prediction",
            output_stage=publication,
        )
        return TrackingFrame(
            float(timestamp_s), states, tracking, None, None, publication
        )

    def update(self, raw_ranges_m, timestamp_s):
        frame = self.process(raw_ranges_m, timestamp_s)
        if frame.tracking is None:
            raise ValueError(frame.position_error)
        return frame.tracking
