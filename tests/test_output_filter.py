"""Check position smoothing, timing and display expiry."""

from dataclasses import replace
import math

import numpy as np
import pytest

from uwb_localisation.output_filter import (
    CausalOutputFilter,
    OutputParameters,
    filter_trace,
)


def test_pass_through_preserves_every_estimate_and_source_state():
    t = [0.0, 0.03, 0.07]
    xy = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    out = filter_trace(t, xy, states=["updated", "predicted", "updated"])
    assert [r["xy"] for r in out] == xy
    assert all(r["new_output"] and r["age_s"] == 0 for r in out)
    assert out[1]["state"] == "predicted"


def test_lowpass_exact_exponential_uses_actual_interval():
    p = OutputParameters(mode="lowpass", lowpass_tau_s=0.2)
    out = filter_trace([0.0, 0.03, 0.13], [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], p)
    assert out[-1]["xy"][0] == pytest.approx(1 - math.exp(-0.13 / 0.2))
    # Splitting a constant-input interval preserves the physical time scale.
    other = filter_trace([0.0, 0.13], [[0.0, 0.0], [1.0, 0.0]], p)
    assert other[-1]["xy"] == pytest.approx(out[-1]["xy"])


@pytest.mark.parametrize("mode", ["lowpass"])
def test_future_points_do_not_change_past_outputs(mode):
    t = np.arange(0.0, 2.0, 0.03)
    xy = np.c_[0.1 * t, np.sin(t)]
    p = OutputParameters(mode=mode, output_rate_hz=10.0)
    short = filter_trace(t[:21], xy[:21], p)
    changed = xy.copy()
    changed[21:] += 30.0
    long = filter_trace(t, changed, p)
    assert short == long[:21]


def test_decimation_keeps_filter_processing_all_frames():
    t = np.arange(0.0, 3.01, 0.03)
    xy = np.c_[t, np.cos(t)]
    p = OutputParameters(mode="lowpass", output_rate_hz=10.0)
    full = filter_trace(t, xy, replace(p, output_rate_hz=0.0))
    sparse = filter_trace(t, xy, p)
    assert [r["filtered_xy"] for r in sparse] == [r["filtered_xy"] for r in full]
    assert sum(r["new_output"] for r in sparse) == 31
    assert sparse[1]["source_time_s"] == 0.0
    assert sparse[1]["available_time_s"] == 0.0
    assert sparse[1]["age_s"] == pytest.approx(0.03)
    assert sparse[1]["xy"] == sparse[0]["xy"]


def test_low_rate_clock_does_not_drift_to_nearest_frame_period():
    t = np.arange(0.0, 9.01, 0.03)
    out = filter_trace(t, np.c_[t, t], OutputParameters(output_rate_hz=10.0))
    published = np.array([r["source_time_s"] for r in out if r["new_output"]])
    assert len(published) == 91
    assert set(np.round(np.diff(published), 2)) == {0.09, 0.12}
    assert np.max(published - np.arange(len(published)) * 0.1) <= 0.030000001


@pytest.mark.parametrize("mode", ["none", "lowpass"])
def test_gap_restarts_without_connecting_old_position(mode):
    stage = CausalOutputFilter(OutputParameters(mode=mode, output_rate_hz=5.0))
    first = stage.step(0.0, [0.0, 0.0])
    stage.step(0.03, [1.0, 1.0])
    returned = stage.step(2.0, [10.0, 10.0])
    assert returned["xy"] == [10.0, 10.0]
    assert returned["reason"] == "reset"
    assert returned["new_output"]
    assert returned["segment_id"] == first["segment_id"] + 1


def test_loss_invalidates_held_point_and_recovery_publishes_immediately():
    stage = CausalOutputFilter(OutputParameters(mode="lowpass", output_rate_hz=5.0))
    stage.step(0.0, [0.0, 0.0])
    loss = stage.step(0.03, None, "lost")
    assert loss["xy"] is None and loss["new_output"]
    assert stage.at(0.04)["xy"] is None
    again = stage.step(0.06, [1.0, 2.0], "reacquired")
    assert again["xy"] == [1.0, 2.0] and again["new_output"]
    assert again["state"] == "reacquired"


def test_reacquisition_resets_even_without_missing_frame():
    out = filter_trace(
        [0.0, 0.03, 0.06],
        [[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]],
        OutputParameters(mode="lowpass"),
        states=["updated", "updated", "reacquired"],
    )
    assert out[-1]["xy"] == [2.0, 2.0]
    assert out[-1]["segment_id"] == 1


def test_no_new_frames_expire_and_inspection_does_not_mutate_filter():
    stage = CausalOutputFilter(OutputParameters(maximum_age_s=0.3))
    stage.step(0.0, [1.0, 2.0])
    assert stage.at(0.3)["xy"] == [1.0, 2.0]
    assert stage.at(0.301)["xy"] is None
    assert stage.at(0.301)["state"] == "stale"
    # The filter clock stays fixed while inspecting a later display time.
    assert stage.step(0.03, [2.0, 3.0])["xy"] == [2.0, 3.0]


def test_low_publication_rate_cannot_expire_fullrate_filter_diagnostics():
    stage = CausalOutputFilter(OutputParameters(output_rate_hz=1.0))
    out = [stage.step(float(t), [float(t), 0.0]) for t in np.arange(0.0, 0.61, 0.03)]
    assert out[-1]["xy"] is None  # the published point is stale
    assert out[-1]["filtered_xy"] == pytest.approx([0.6, 0.0])
    assert out[-1]["filtered_time_s"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "future"},
        {"output_rate_hz": -1},
        {"output_rate_hz": float("nan")},
        {"lowpass_tau_s": 0},
        {"reset_gap_s": 0},
        {"maximum_age_s": float("inf")},
    ],
)
def test_invalid_parameters_rejected(updates):
    with pytest.raises(ValueError):
        OutputParameters(**updates)


def test_duplicate_and_backwards_times_rejected():
    stage = CausalOutputFilter()
    stage.step(1.0, [0.0, 0.0])
    for time in (1.0, 0.5, float("nan")):
        with pytest.raises(ValueError):
            stage.step(time, [1.0, 1.0])


def test_invalid_positions_cannot_propagate_nan():
    stage = CausalOutputFilter()
    stage.step(0.0, [0.0, 0.0])
    assert stage.step(0.03, [float("nan"), 1.0])["xy"] is None
    assert stage.step(0.06, [1.0, 2.0])["xy"] == [1.0, 2.0]


def test_state_lengths_checked_instead_of_silent_truncation():
    with pytest.raises(ValueError):
        filter_trace([0.0, 0.03], [[0.0, 0.0]])
    with pytest.raises(ValueError):
        filter_trace([0.0], [[0.0, 0.0]], states=[])
