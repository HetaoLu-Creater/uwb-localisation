"""Estimate planar positions from three calibrated ranges.

Weighted nonlinear least squares provides positions for a constant-velocity
Kalman filter. A configured field boundary rejects estimates outside the
operating area.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np

from .localisation import slant_range_model


def cv_matrices(dt, spectral_density):
    """Continuous white acceleration q [m²/s³], integrated over actual dt."""
    eye = np.eye(2)
    f = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
    q = spectral_density * np.block(
        [[dt**3 / 3 * eye, dt**2 / 2 * eye], [dt**2 / 2 * eye, dt * eye]]
    )
    return f, q


def bias_geometry_budget(xy, xyz, height, parameters):
    """Estimate persistent position uncertainty from weighted local geometry."""
    _, j = slant_range_model(np.asarray(xy), np.asarray(xyz), height)
    w = 1 / (
        np.asarray(parameters.range_sigma_m) ** 2
        + np.asarray(parameters.bias_sigma_m) ** 2
    )
    sensitivity = np.linalg.pinv(j.T @ (w[:, None] * j)) @ (j.T * w)
    return (sensitivity * np.asarray(parameters.bias_sigma_m) ** 2) @ sensitivity.T


def algebraic(xyz, ranges, height=0.0):
    """Difference of squared ranges at a fixed tag height."""
    if len(ranges) != 3 or not np.isfinite(ranges).all():
        return None
    planar2 = ranges**2 - (xyz[:, 2] - height) ** 2
    a = 2 * (xyz[1:, :2] - xyz[0, :2])
    b = (
        planar2[0]
        - planar2[1:]
        + np.sum(xyz[1:, :2] ** 2, axis=1)
        - np.sum(xyz[0, :2] ** 2)
    )
    return np.linalg.solve(a, b)


def range_solution(xyz, z, sigma, height=0.0, huber=1e6, initial=None, iterations=20):
    """Solve weighted range equations with a fixed height and Huber loss.

    Three ranges provide one redundant equation for two coordinates. Residuals
    measure agreement between the ranges and the fitted position.
    """
    if len(z) < 3 or not np.all(np.isfinite(z)):
        return None
    seeds = [algebraic(xyz, z, height), np.mean(xyz[:, :2], axis=0)]
    if initial is not None:
        seeds.append(np.asarray(initial))

    def terms(p):
        h, j = slant_range_model(p, xyz, height)
        u = (h - z) / sigma
        a = np.abs(u)
        w = np.minimum(1.0, huber / np.maximum(a, 1e-12))
        cost = np.sum(np.where(a <= huber, 0.5 * u * u, huber * (a - 0.5 * huber)))
        return h, j, u, w, cost

    p = min(seeds, key=lambda p: terms(p)[-1]).copy()
    converged = False
    for iteration in range(iterations):
        h, j, u, w, cost = terms(p)
        jw = j / sigma[:, None]
        info = jw.T @ (w[:, None] * jw)
        step = np.linalg.solve(info + 1e-6 * np.eye(2), -jw.T @ (w * u))
        if np.linalg.norm(step) < 1e-6:
            converged = True
            break
        for frac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.015625):
            candidate = p + frac * step
            if terms(candidate)[-1] <= cost:
                p = candidate
                break
        else:
            break
    h, j, u, w, cost = terms(p)
    jw = j / sigma[:, None]
    info = jw.T @ (w[:, None] * jw)
    if np.linalg.eigvalsh(info)[0] <= 1e-8:
        return None
    # Residual disagreement increases this approximate position covariance.
    cov = np.linalg.inv(info) * max(1.0, float(np.sum(w * u * u)) / max(len(z) - 2, 1))
    return dict(
        xy=p,
        covariance=cov,
        residual=h - z,
        weights=w,
        normalised_rms=float(np.sqrt(np.mean(u * u))),
        iterations=iteration + 1,
        converged=converged,
    )


@dataclass
class EmpiricalParameters:
    range_sigma_m: tuple = (0.02, 0.02, 0.02)
    bias_sigma_m: tuple = (0.02, 0.02, 0.04)
    correlation_time_s: float = 0.06
    process_spectral_density: float = 0.05
    huber_k: float = 2.0
    innovation_gate_squared: float = 25.0
    residual_gate_sigma: float = 8.0
    maximum_prediction_age_s: float = 0.3
    reacquire_after_s: float = 0.24
    reacquire_consistency_m: float = 0.12
    range_window_s: float = 0.0
    maximum_range_m: float = 3.0
    initial_velocity_std_mps: float = 0.5
    speed_limit_mps: float = 0.0
    max_iterations: int = 20

    def __post_init__(self):
        for key in ("range_sigma_m", "bias_sigma_m"):
            a = np.asarray(getattr(self, key), float)
            if a.shape != (3,) or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(
                    f"{key} must contain three positive finite standard deviations"
                )
        for key in (
            "process_spectral_density",
            "huber_k",
            "innovation_gate_squared",
            "residual_gate_sigma",
            "maximum_prediction_age_s",
            "reacquire_after_s",
            "reacquire_consistency_m",
            "maximum_range_m",
            "initial_velocity_std_mps",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("correlation_time_s", "range_window_s", "speed_limit_mps"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative and finite")
        if int(self.max_iterations) != self.max_iterations or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


class EmpiricalTracker:
    """Track planar position and velocity from weighted range solutions.

    Missing ranges trigger prediction. Consistent measurements allow tracking to
    resume after rejected updates. A reception gap resets the range history.
    History timestamps are in seconds.
    """

    def __init__(self, xyz, height=0.0, parameters=None, *, record_history=True):
        self.xyz = np.asarray(xyz, float)
        self.height = float(height)
        if (
            self.xyz.shape != (3, 3)
            or not np.isfinite(self.xyz).all()
            or not np.isfinite(self.height)
        ):
            raise ValueError(
                "Require three finite anchor XYZ coordinates and finite tag height"
            )
        if np.linalg.matrix_rank(self.xyz[1:, :2] - self.xyz[0, :2]) < 2:
            raise ValueError("Anchors must be noncollinear")
        self.p = parameters or EmpiricalParameters()
        self.x = None
        self.cov = None
        self.last_t = None
        self.last_update = None
        self.had_fix = False
        self.candidates = deque()
        self.ranges = [deque() for _ in range(3)]
        self.history = []
        self.record_history = bool(record_history)
        self.field_boundary = None

    def step(self, t, z):
        t = float(t)
        z = np.asarray(z, float).copy()
        if not np.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError("timestamps must be finite and strictly increasing")
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        z[(np.abs(z) > self.p.maximum_range_m) | ~np.isfinite(z)] = np.nan
        for i, hist in enumerate(self.ranges):
            if self.p.range_window_s > 0:
                while hist and t - hist[0][0] > self.p.range_window_s:
                    hist.popleft()
                if np.isfinite(z[i]):
                    hist.append((t, z[i]))
                    z[i] = np.median([v for _, v in hist])
        f, q = cv_matrices(dt, self.p.process_spectral_density)
        xp, pp = None, None
        if self.x is not None:
            if t - self.last_update > self.p.maximum_prediction_age_s:
                self.x = self.cov = None
            else:
                xp = f @ self.x
                pp = f @ self.cov @ f.T + q
                self.x = xp
                self.cov = pp
        # Increase uncertainty for correlated samples and persistent range bias.
        # The exported uncertainty values keep bias separate.
        factor = max(1.0, self.p.correlation_time_s / max(dt, 0.001)) if dt else 1.0
        sigma = np.sqrt(
            np.asarray(self.p.range_sigma_m) ** 2 * factor
            + np.asarray(self.p.bias_sigma_m) ** 2
        )
        sol = range_solution(
            self.xyz,
            z,
            sigma,
            self.height,
            self.p.huber_k,
            self.x[:2] if self.x is not None else None,
            self.p.max_iterations,
        )
        diagnostic_solution = sol
        rejected_xy = None
        if (
            sol is not None
            and self.field_boundary is not None
            and not self.field_boundary.accepts(sol["xy"])
        ):
            rejected_xy = sol["xy"].copy()
            sol = None
        state = "predicted" if self.x is not None else "lost"
        reason = "missing_ranges"
        nis = None
        reacquired = False
        if sol is not None:
            reason = "range_residual_gate"
            if sol["normalised_rms"] > self.p.residual_gate_sigma:
                self.candidates.clear()
            if sol["normalised_rms"] <= self.p.residual_gate_sigma:
                measurement = sol["xy"]
                r = sol["covariance"]
                if self.x is None:
                    self.x = np.r_[measurement, 0.0, 0.0]
                    self.cov = np.zeros((4, 4))
                    self.cov[:2, :2] = r
                    self.cov[2:, 2:] = np.eye(2) * self.p.initial_velocity_std_mps**2
                    state = "reacquired" if self.had_fix else "updated"
                    reason = "reinitialised" if self.had_fix else "initialised"
                    self.last_update = t
                    self.had_fix = True
                    self.candidates.clear()
                    reacquired = True
                else:
                    innovation = measurement - self.x[:2]
                    s = self.cov[:2, :2] + r
                    nis = float(innovation @ np.linalg.solve(s, innovation))
                    if nis <= self.p.innovation_gate_squared:
                        k = np.linalg.solve(s, self.cov[:2, :]).T
                        self.x = self.x + k @ innovation
                        ikh = np.eye(4)
                        ikh[:, :2] -= k
                        self.cov = ikh @ self.cov @ ikh.T + k @ r @ k.T
                        state = "updated"
                        reason = "measurement"
                        self.last_update = t
                        self.candidates.clear()
                    else:
                        reason = "innovation_gate"
                        self.candidates.append((t, measurement))
                        while (
                            self.candidates
                            and t - self.candidates[0][0]
                            > self.p.reacquire_after_s * 1.5
                        ):
                            self.candidates.popleft()
                        if (
                            len(self.candidates) > 1
                            and t - self.candidates[0][0] >= self.p.reacquire_after_s
                            and np.max(
                                np.linalg.norm(
                                    np.array([v for _, v in self.candidates])
                                    - measurement,
                                    axis=1,
                                )
                            )
                            <= self.p.reacquire_consistency_m
                        ):
                            self.x = np.r_[measurement, 0.0, 0.0]
                            self.cov = np.zeros((4, 4))
                            self.cov[:2, :2] = r
                            self.cov[2:, 2:] = (
                                np.eye(2) * self.p.initial_velocity_std_mps**2
                            )
                            self.last_update = t
                            state = "reacquired"
                            reason = "consistent_reacquisition"
                            reacquired = True
                            self.candidates.clear()
        else:
            self.candidates.clear()
        if rejected_xy is not None:
            reason = "outside_field_rejected"
        # Apply the field boundary to both measurements and predicted states.
        if (
            self.x is not None
            and self.field_boundary is not None
            and not self.field_boundary.accepts(self.x[:2])
        ):
            rejected_xy = self.x[:2].copy()
            self.x = self.cov = None
            self.candidates.clear()
            state = "lost"
            reason = "outside_field_rejected"
        if self.x is not None and self.p.speed_limit_mps > 0:
            speed = np.linalg.norm(self.x[2:])
            if speed > self.p.speed_limit_mps:
                self.x[2:] *= self.p.speed_limit_mps / speed
                reason += ";speed_limited"
        if self.cov is not None:
            self.cov = (self.cov + self.cov.T) / 2
        entry = dict(
            time=t,
            x=None if self.x is None else self.x.copy(),
            p=None if self.cov is None else self.cov.copy(),
            prior=xp,
            prior_cov=pp,
            f=f,
            reset=reacquired,
            state=state,
        )
        if self.record_history:
            self.history.append(entry)
        final_residual = None
        bias_covariance = None
        if self.x is not None:
            h, j = slant_range_model(self.x[:2], self.xyz, self.height)
            valid = np.isfinite(z)
            if valid.any():
                final_residual = float(np.sqrt(np.mean((h[valid] - z[valid]) ** 2)))
            # Estimate persistent position bias from local range sensitivity,
            # separately from the temporal Kalman covariance.
            bias_covariance = bias_geometry_budget(
                self.x[:2], self.xyz, self.height, self.p
            )
        return dict(
            xy=None if self.x is None else self.x[:2].copy(),
            state=state,
            reason=reason,
            residual_m=final_residual,
            measurement_residual_m=None
            if sol is None
            else float(np.sqrt(np.mean(sol["residual"] ** 2))),
            bias_covariance=bias_covariance,
            effective_range_variance_m2=sigma**2,
            correlation_variance_scale=factor,
            nis=nis,
            filtered_m=z,
            covariance=None if self.cov is None else self.cov[:2, :2].copy(),
            solution=diagnostic_solution,
            boundary_rejected=rejected_xy is not None,
            rejected_xy=rejected_xy,
            age_s=None if self.last_update is None else t - self.last_update,
        )
