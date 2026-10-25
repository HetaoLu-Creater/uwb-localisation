"""Smooth tracked positions and control display timing.

Each incoming estimate updates the position smoother. A separate display clock
selects output times. Held positions retain their measurement time so the
caller can clear them when they expire.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OutputParameters:
    mode: str = "none"
    lowpass_tau_s: float = 0.06
    output_rate_hz: float = 0.0  # zero displays every incoming estimate
    reset_gap_s: float = 0.15
    maximum_age_s: float = 0.30

    def __post_init__(self):
        if self.mode not in {"none", "lowpass"}:
            raise ValueError("mode must be none or lowpass")
        for name in ("lowpass_tau_s", "reset_gap_s", "maximum_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("output_rate_hz",):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


class CausalOutputFilter:
    """Smooth one position stream using timestamps in seconds.

    ``step`` accepts increasing source timestamps. ``at`` returns the held position
    at a requested display time using the current filter state. Estimator resets
    start a new segment, identified by ``segment_id``. The display clock starts
    with each segment. The first frame after a scheduled update provides the
    next displayed position.
    """

    def __init__(self, parameters: OutputParameters | None = None):
        self.parameters = parameters or OutputParameters()
        self._clock = None
        self._time = None
        self._raw = None
        self._filtered = None
        self._derivative = np.zeros(2)
        self._cutoff = None
        self._published = None
        self._published_time = None
        self._published_state = "lost"
        self._segment_start = None
        self._next_due = None
        self.segment_id = -1

    def _clear(self):
        self._time = None
        self._raw = None
        self._filtered = None
        self._derivative = np.zeros(2)
        self._cutoff = None
        self._published = None
        self._published_time = None
        self._published_state = "lost"
        self._segment_start = None
        self._next_due = None

    def reset(self):
        """Reset timestamps and filter state for a separate recording."""
        self._clear()
        self._clock = None
        self.segment_id = -1

    def at(self, time_s):
        """Return the held position and its age at the requested display time."""
        t = float(time_s)
        if not math.isfinite(t) or (self._clock is not None and t < self._clock):
            raise ValueError("display time must be finite and not precede latest input")
        age = (
            None if self._published_time is None else max(0.0, t - self._published_time)
        )
        stale = age is not None and age > self.parameters.maximum_age_s + 1e-12
        visible = self._published is not None and not stale
        filtered_fresh = (
            self._time is not None
            and t - self._time <= self.parameters.maximum_age_s + 1e-12
        )
        return dict(
            xy=self._published.tolist() if visible else None,
            filtered_xy=self._filtered.tolist()
            if self._filtered is not None and filtered_fresh
            else None,
            filtered_time_s=self._time,
            new_output=False,
            source_time_s=self._published_time,
            available_time_s=self._published_time,
            age_s=age,
            state=self._published_state if visible else ("stale" if stale else "lost"),
            source_state=self._published_state,
            segment_id=self.segment_id,
            cutoff_hz=self._cutoff,
            reason="stale" if stale else ("held" if visible else "lost"),
        )

    def step(self, time_s, xy, state="updated", *, reset=False):
        t = float(time_s)
        if not math.isfinite(t) or (self._clock is not None and t <= self._clock):
            raise ValueError("source timestamps must be finite and strictly increasing")
        self._clock = t
        p = self.parameters
        value = np.asarray(xy, dtype=float) if xy is not None else None
        valid = value is not None and value.shape == (2,) and np.all(np.isfinite(value))
        if not valid or state in {"lost", "uninitialized", "invalid", "stale"}:
            invalidated = self._published is not None
            self._clear()
            result = self.at(t)
            result.update(new_output=invalidated, reason="lost")
            return result
        gap = self._time is not None and t - self._time > p.reset_gap_s + 1e-12
        restarted = (
            reset or state in {"reacquired", "initialized"} or gap or self._time is None
        )
        if restarted:
            self._clear()
            self.segment_id += 1
            self._filtered = value.copy()
            self._segment_start = t
            self._next_due = t
        else:
            dt = t - self._time
            if p.mode == "none":
                self._filtered = value.copy()
            elif p.mode == "lowpass":
                alpha = -math.expm1(-dt / p.lowpass_tau_s)
                self._filtered += alpha * (value - self._filtered)
                self._cutoff = 1 / (2 * math.pi * p.lowpass_tau_s)
        self._time = t
        self._raw = value.copy()
        publish = restarted or p.output_rate_hz == 0 or t + 1e-10 >= self._next_due
        if publish:
            self._published = self._filtered.copy()
            self._published_time = t
            self._published_state = state
            if p.output_rate_hz:
                period = 1 / p.output_rate_hz
                ticks = math.floor((t - self._segment_start) / period + 1e-9) + 1
                self._next_due = self._segment_start + ticks * period
        result = self.at(t)
        result.update(
            new_output=publish,
            reason="reset" if restarted else ("published" if publish else "held"),
        )
        return result


def filter_trace(time_s, xy, parameters=None, states=None, resets=None):
    """Apply the display filter and return one diagnostic row for each input."""
    if len(time_s) != len(xy):
        raise ValueError("timestamps and coordinates must have matching length")
    states = ["updated"] * len(time_s) if states is None else states
    resets = [False] * len(time_s) if resets is None else resets
    if len(states) != len(time_s) or len(resets) != len(time_s):
        raise ValueError("states and resets must have matching length")
    stage = CausalOutputFilter(parameters)
    return [
        stage.step(t, point, state, reset=reset)
        for t, point, state, reset in zip(time_s, xy, states, resets)
    ]
