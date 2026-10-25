"""Filter three raw range channels before calibration.

A shared quiet or moving decision adjusts three radial constant-velocity
filters. The decision uses changes in current and past ranges. Accepted
observations form the filtered range output. Predictions are stored separately
for diagnostics.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AdaptiveRangeParameters:
    enabled: bool = True
    noise_window_s: float = 0.9
    evidence_window_s: float = 0.45
    quiet_duration_s: float = 0.36
    minimum_motion_rate_mps: float = 0.04
    minimum_motion_delta_m: float = 0.025
    evidence_sigma: float = 3.0
    moving_links: int = 2
    moving_q_m2_s3: float = 3.0
    quiet_q_m2_s3: float = 0.0001
    quiet_velocity_std_mps: float = 0.015
    noise_floor_m: float = 0.005
    initial_noise_m: float = 0.04
    innovation_gate_sigma: float = 5.0
    reacquire_s: float = 0.12
    reset_gap_s: float = 0.15
    maximum_range_m: float = 5.0

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if key in ("enabled", "moving_links"):
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.moving_links not in (2, 3):
            raise ValueError("moving_links must be 2 or 3")
        if self.quiet_q_m2_s3 > self.moving_q_m2_s3:
            raise ValueError("quiet process noise cannot exceed moving noise")


class AdaptiveRangeFilter:
    def __init__(self, parameters=None):
        self.parameters = parameters or AdaptiveRangeParameters()
        self.last_t = None
        self.histories = [deque() for _ in range(3)]
        self.states = [None] * 3
        self.covariances = [None] * 3
        self.last_accept = [None] * 3
        self.outliers = [deque() for _ in range(3)]
        self.quiet_since = None
        self.quiet = False
        self.quiet_reference = None
        self.segment_id = -1

    def _clear(self):
        self.histories = [deque() for _ in range(3)]
        self.states = [None] * 3
        self.covariances = [None] * 3
        self.last_accept = [None] * 3
        self.outliers = [deque() for _ in range(3)]
        self.quiet_since = None
        self.quiet = False
        self.quiet_reference = None
        self.segment_id += 1

    def _evidence(self, t):
        p = self.parameters
        features = []
        for history in self.histories:
            points = np.asarray(history, float)
            if len(points) < 3:
                features.append(
                    dict(
                        sigma_m=p.initial_noise_m,
                        median_m=None,
                        slope_mps=None,
                        trend_score=0.0,
                        shift_score=0.0,
                        motion=False,
                        ready=False,
                    )
                )
                continue
            differences = np.diff(points[:, 1])
            difference_dt = np.diff(points[:, 0])
            differences = differences[
                (difference_dt > 0) & (difference_dt <= p.reset_gap_s)
            ]
            sigma = (
                1.4826
                * np.median(np.abs(differences - np.median(differences)))
                / math.sqrt(2)
                if len(differences) >= 4
                else p.initial_noise_m
            )
            sigma = max(p.noise_floor_m, float(sigma))
            window = points[points[:, 0] >= t - p.evidence_window_s]
            span = window[-1, 0] - window[0, 0] if len(window) else 0.0
            if len(window) < 5 or span < 0.12 or t - points[-1, 0] > p.reset_gap_s:
                features.append(
                    dict(
                        sigma_m=sigma,
                        median_m=float(np.median(window[:, 1]))
                        if len(window)
                        else None,
                        slope_mps=None,
                        trend_score=0.0,
                        shift_score=0.0,
                        motion=False,
                        ready=False,
                    )
                )
                continue
            tx = window[:, 0] - np.mean(window[:, 0])
            y = window[:, 1]
            slope = float(tx @ (y - y.mean()) / (tx @ tx))
            # A robust regression step reduces the influence of a single spike
            # on the decision to switch to the motion model.
            error = y - (np.median(y - slope * tx) + slope * tx)
            weights = np.minimum(1.0, 2.5 * sigma / np.maximum(np.abs(error), 1e-12))
            design = np.c_[np.ones(len(tx)), tx]
            coeff = np.linalg.lstsq(
                design * weights[:, None] ** 0.5, y * weights**0.5, rcond=None
            )[0]
            slope = float(coeff[1])
            median = float(np.median(y))
            trend_limit = max(
                p.minimum_motion_rate_mps,
                p.evidence_sigma * sigma / math.sqrt(max(tx @ tx, 1e-9)),
            )
            features.append(
                dict(
                    sigma_m=sigma,
                    median_m=median,
                    slope_mps=slope,
                    trend_score=abs(slope) / trend_limit,
                    shift_score=0.0,
                    motion=False,
                    ready=True,
                    samples=len(window),
                )
            )
        if self.quiet_reference is not None and self.quiet:
            for i, f in enumerate(features):
                if f["ready"] and np.isfinite(self.quiet_reference[i]):
                    limit = max(
                        p.minimum_motion_delta_m,
                        p.evidence_sigma * f["sigma_m"] / math.sqrt(f["samples"]),
                    )
                    f["shift_score"] = float(
                        abs(f["median_m"] - self.quiet_reference[i]) / limit
                    )
        # These diagnostics are also served directly as dashboard JSON.
        # Native scalar types keep these diagnostics compatible with JSON.
        for f in features:
            f["motion"] = bool(
                f["ready"] and max(f["trend_score"], f["shift_score"]) > 1.0
            )
        ready = sum(f["ready"] for f in features)
        moving = sum(f["motion"] for f in features)
        released = self.quiet and moving >= p.moving_links
        if ready >= p.moving_links and moving < p.moving_links:
            if self.quiet_since is None:
                self.quiet_since = t
            if t - self.quiet_since >= p.quiet_duration_s:
                if not self.quiet:
                    self.quiet_reference = np.array(
                        [
                            f["median_m"] if f["median_m"] is not None else np.nan
                            for f in features
                        ]
                    )
                self.quiet = True
        else:
            self.quiet_since = None
            self.quiet = False
            self.quiet_reference = None
        confidence = (
            min(1.0, (t - self.quiet_since) / p.quiet_duration_s)
            if self.quiet_since is not None
            else 0.0
        )
        # This score measures how long the quiet-range condition has held.
        return (
            features,
            float(confidence),
            released,
            dict(
                ready_links=ready,
                moving_links=moving,
                required_moving_links=p.moving_links,
                quiet_since_s=self.quiet_since,
                rule_confidence_is_probability=False,
            ),
        )

    def step(self, time_s, raw_ranges_m):
        t = float(time_s)
        raw = np.asarray(raw_ranges_m, float).copy()
        p = self.parameters
        if raw.shape != (3,):
            raise ValueError("three raw ranges are required")
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError("timestamps must increase strictly")
        dt = 0.0 if self.last_t is None else t - self.last_t
        reset = self.last_t is None or dt > p.reset_gap_s + 1e-12
        if reset:
            self._clear()
        self.last_t = t
        raw[(raw <= 0) | (~np.isfinite(raw)) | (raw > p.maximum_range_m)] = np.nan
        for i, h in enumerate(self.histories):
            while h and t - h[0][0] > p.noise_window_s:
                h.popleft()
            if np.isfinite(raw[i]):
                h.append((t, float(raw[i])))
        was_quiet = self.quiet
        evidence, confidence, released, summary = self._evidence(t)
        acquired_quiet = self.quiet and not was_quiet
        conditioned = np.full(3, np.nan)
        prediction = np.full(3, np.nan)
        variance = np.full(3, np.nan)
        diagnostics = []
        for i in range(3):
            sigma = evidence[i]["sigma_m"]
            accepted = False
            state = "missing"
            reason = "no_current_measurement"
            innovation_sigma = None
            if self.states[i] is not None:
                if t - self.last_accept[i] > p.reset_gap_s:
                    self.states[i] = self.covariances[i] = None
                else:
                    f = np.array([[1.0, dt], [0.0, 1.0]])
                    q = p.quiet_q_m2_s3 if self.quiet else p.moving_q_m2_s3
                    q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
                    self.states[i] = f @ self.states[i]
                    self.covariances[i] = f @ self.covariances[i] @ f.T + q
                    if self.quiet:
                        # Sustained quiet ranges support a zero-radial-velocity prior.
                        # Detected movement releases the prior.
                        cov = self.covariances[i]
                        gain = cov[:, 1] / (cov[1, 1] + p.quiet_velocity_std_mps**2)
                        self.states[i] -= gain * self.states[i][1]
                        ikh = np.eye(2)
                        ikh[:, 1] -= gain
                        self.covariances[i] = (
                            ikh @ cov @ ikh.T
                            + np.outer(gain, gain) * p.quiet_velocity_std_mps**2
                        )
            if np.isfinite(raw[i]):
                if (
                    self.states[i] is None
                    or released
                    or (acquired_quiet and evidence[i]["ready"])
                ):
                    recent = [v for tt, v in self.histories[i] if tt >= t - 0.09]
                    initial = (
                        evidence[i]["median_m"]
                        if acquired_quiet and evidence[i]["ready"]
                        else float(np.median(recent))
                        if released and recent
                        else raw[i]
                    )
                    self.states[i] = np.array([initial, 0.0])
                    self.covariances[i] = np.diag([sigma * sigma, 1.0])
                    accepted = True
                    state = (
                        "reacquired"
                        if released
                        else "quiet_acquired"
                        if acquired_quiet
                        else "initialised"
                    )
                    reason = (
                        "coherent_motion_release"
                        if released
                        else "causal_window_median"
                        if acquired_quiet
                        else "fresh_segment"
                    )
                else:
                    residual = raw[i] - self.states[i][0]
                    s = self.covariances[i][0, 0] + sigma * sigma
                    innovation_sigma = abs(float(residual)) / math.sqrt(s)
                    if innovation_sigma <= p.innovation_gate_sigma:
                        cov = self.covariances[i]
                        gain = cov[:, 0] / s
                        # Static calibration is fitted to raw medians. A
                        # quiet-window median therefore also guards against
                        # skewed inlier noise shifting a long-running mean.
                        observation = (
                            evidence[i]["median_m"]
                            if self.quiet and evidence[i]["ready"]
                            else raw[i]
                        )
                        self.states[i] += gain * (observation - self.states[i][0])
                        ikh = np.eye(2)
                        ikh[:, 0] -= gain
                        self.covariances[i] = (
                            ikh @ cov @ ikh.T + np.outer(gain, gain) * sigma * sigma
                        )
                        accepted = True
                        state = "quiet_update" if self.quiet else "moving_update"
                        reason = (
                            "causal_window_median"
                            if self.quiet and evidence[i]["ready"]
                            else "current_measurement"
                        )
                        self.outliers[i].clear()
                    else:
                        h = self.outliers[i]
                        h.append((t, float(raw[i])))
                        while h and t - h[0][0] > 2 * p.reacquire_s:
                            h.popleft()
                        values = np.array([v for _, v in h])
                        stable = np.ptp(values) <= max(4 * sigma, 0.03)
                        if len(h) > 1 and t - h[0][0] >= p.reacquire_s and stable:
                            self.states[i] = np.array([np.median(values), 0.0])
                            self.covariances[i] = np.diag([sigma * sigma, 1.0])
                            accepted = True
                            state = "reacquired"
                            reason = "persistent_link_step"
                            h.clear()
                        else:
                            state = "rejected"
                            reason = "isolated_innovation"
                if accepted:
                    self.last_accept[i] = t
            else:
                self.outliers[i].clear()
            if self.states[i] is not None:
                prediction[i] = self.states[i][0]
                variance[i] = max(p.noise_floor_m**2, self.covariances[i][0, 0])
                if accepted:
                    conditioned[i] = self.states[i][0]
            if not p.enabled:
                conditioned[i] = raw[i]
                variance[i] = sigma * sigma
                accepted = np.isfinite(raw[i])
                state = "passthrough" if accepted else "missing"
                reason = "disabled"
                if accepted:
                    self.last_accept[i] = t
            diagnostics.append(
                dict(
                    state=state,
                    reason=reason,
                    sigma_m=float(sigma),
                    variance_m2=float(variance[i]),
                    measurement_accepted=bool(accepted),
                    innovation_sigma=innovation_sigma,
                    confidence_score=confidence,
                    source_age_s=None
                    if self.last_accept[i] is None
                    else t - self.last_accept[i],
                    weight=1.0 if accepted else 0.0,
                    radial_velocity_mps=None
                    if self.states[i] is None
                    else float(self.states[i][1]),
                )
            )
        return dict(
            conditioned_m=conditioned,
            predicted_m=prediction,
            variance_m2=variance,
            diagnostics=diagnostics,
            stationary_confidence=confidence,
            stationary_state="quiet" if self.quiet else "moving_or_unknown",
            evidence=dict(**summary, links=evidence),
            segment_id=self.segment_id,
            reset=reset,
        )


def condition_ranges(time_s, raw_ranges_m, parameters=None):
    if len(time_s) != len(raw_ranges_m):
        raise ValueError("time/range length mismatch")
    f = AdaptiveRangeFilter(parameters)
    return [f.step(t, z) for t, z in zip(time_s, raw_ranges_m)]
