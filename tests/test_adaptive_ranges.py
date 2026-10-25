import numpy as np
import pytest

from uwb_localisation.adaptive_ranges import (
    AdaptiveRangeFilter,
    AdaptiveRangeParameters,
    condition_ranges,
)


def test_quiet_ranges_reduce_independent_noise_without_reference_labels():
    rng = np.random.default_rng(94)
    t = np.arange(0.0, 8.0, 0.03)
    raw = np.array([0.8, 0.7, 0.4]) + rng.normal(0, 0.02, (len(t), 3))
    rows = condition_ranges(t, raw)
    filtered = np.array([r["conditioned_m"] for r in rows])
    keep = t >= 1.0
    assert np.all(np.nanstd(filtered[keep], axis=0) < 0.5 * np.std(raw[keep], axis=0))
    assert np.max(np.abs(np.nanmedian(filtered[keep], axis=0) - [0.8, 0.7, 0.4])) < 0.02
    assert np.mean([r["stationary_state"] == "quiet" for r in rows]) > 0.7


def test_no_future_samples_change_past_range_outputs():
    rng = np.random.default_rng(42)
    t = np.arange(0.0, 4.0, 0.03)
    raw = 0.5 + rng.normal(0, 0.02, (len(t), 3))
    short = condition_ranges(t[:51], raw[:51])
    raw[51:] += 0.7
    long = condition_ranges(t, raw)
    for a, b in zip(short, long[:51]):
        np.testing.assert_array_equal(a["conditioned_m"], b["conditioned_m"])
        assert a["diagnostics"] == b["diagnostics"]


def test_single_link_spike_is_missing_not_a_new_prediction_measurement():
    f = AdaptiveRangeFilter()
    for t in np.arange(0.0, 1.0, 0.03):
        f.step(float(t), [0.8, 0.7, 0.5])
    spike = f.step(1.02, [2.0, 0.7, 0.5])
    assert np.isnan(spike["conditioned_m"][0])
    assert spike["diagnostics"][0]["measurement_accepted"] is False
    assert spike["diagnostics"][0]["state"] == "rejected"
    assert abs(spike["predicted_m"][0] - 0.8) < 1e-6
    recovered = f.step(1.05, [0.8, 0.7, 0.5])
    assert recovered["diagnostics"][0]["measurement_accepted"]


def test_coherent_real_change_releases_quiet_prior_without_freezing():
    f = AdaptiveRangeFilter()
    for t in np.arange(0.0, 1.0, 0.03):
        f.step(float(t), [0.8, 0.7, 0.5])
    rows = [f.step(float(t), [0.95, 0.6, 0.6]) for t in np.arange(1.02, 1.35, 0.03)]
    assert any(r["stationary_state"] == "moving_or_unknown" for r in rows)
    assert any(
        any(d["reason"] == "coherent_motion_release" for d in r["diagnostics"])
        for r in rows
    )
    assert np.max(np.abs(rows[-1]["conditioned_m"] - [0.95, 0.6, 0.6])) < 0.02


def test_long_gap_discards_old_range_state():
    f = AdaptiveRangeFilter()
    first = f.step(0.0, [0.3, 0.4, 0.5])
    last = f.step(2.0, [1.0, 1.1, 1.2])
    np.testing.assert_array_equal(last["conditioned_m"], [1.0, 1.1, 1.2])
    assert last["segment_id"] == first["segment_id"] + 1
    assert last["reset"]


def test_missing_channel_can_stay_missing_and_recover_after_history_expires():
    f = AdaptiveRangeFilter()
    for t in np.arange(0.0, 1.0, 0.03):
        f.step(float(t), [0.3, 0.4, 0.5])
    for t in np.arange(1.02, 3.0, 0.03):
        row = f.step(float(t), [np.nan, 0.4, 0.5])
        assert np.isnan(row["conditioned_m"][0])
        assert not row["diagnostics"][0]["measurement_accepted"]
    row = f.step(3.03, [0.8, 0.4, 0.5])
    assert row["conditioned_m"][0] == pytest.approx(0.8)


def test_disabled_mode_preserves_valid_raw_values_and_never_fills_missing():
    t = np.arange(0.0, 0.3, 0.03)
    raw = np.tile([0.3, 0.4, 0.5], (len(t), 1))
    raw[3, 1] = np.nan
    rows = condition_ranges(t, raw, AdaptiveRangeParameters(enabled=False))
    np.testing.assert_equal(np.array([r["conditioned_m"] for r in rows]), raw)


@pytest.mark.parametrize(
    "bad",
    [
        {"noise_window_s": 0},
        {"moving_links": 1},
        {"quiet_q_m2_s3": 10.0},
        {"innovation_gate_sigma": float("nan")},
    ],
)
def test_invalid_parameters_rejected(bad):
    with pytest.raises(ValueError):
        AdaptiveRangeParameters(**bad)


def test_current_invalid_zero_and_duplicate_times_rejected():
    f = AdaptiveRangeFilter()
    r = f.step(0.0, [0.0, 0.3, 0.4])
    assert np.isnan(r["conditioned_m"][0])
    with pytest.raises(ValueError):
        f.step(0.0, [0.2, 0.3, 0.4])


@pytest.mark.parametrize("initial_offset", [-0.25, 0.45])
def test_quiet_acquisition_does_not_lock_in_first_sample_direction(initial_offset):
    t = np.arange(0.0, 6.0, 0.03)
    raw = np.tile([1.2, 1.03, 0.21], (len(t), 1))
    raw[0, 0] += initial_offset
    # Positive skew resembles intermittent excess range; a stationary mean
    # would move calibration away from the raw-median reference.
    raw[7::9, 0] += 0.4
    rows = condition_ranges(t, raw)
    z = np.array([r["conditioned_m"] for r in rows])
    assert any(r["diagnostics"][0]["state"] == "quiet_acquired" for r in rows)
    assert abs(np.nanmedian(z[t > 1.0, 0]) - np.median(raw[t > 1.0, 0])) < 0.01
    assert np.nanstd(z[t > 1.0, 0]) < 0.01


@pytest.mark.parametrize("bad_link,sigma", [(0, 0.234), (2, 0.102)])
def test_high_noise_static_channel_is_stable_without_hiding_its_uncertainty(
    bad_link, sigma
):
    t = np.arange(0.0, 9.0, 0.03)
    rng = np.random.default_rng(101)
    scales = np.array([0.02, 0.02, 0.02])
    scales[bad_link] = sigma
    truth = np.array([1.2, 1.03, 0.8])
    raw = truth + rng.normal(size=(len(t), 3)) * scales
    p = AdaptiveRangeParameters(
        minimum_motion_rate_mps=0.06,
        minimum_motion_delta_m=0.04,
        evidence_sigma=4.0,
        moving_q_m2_s3=10.0,
    )
    rows = condition_ranges(t, raw, p)
    z = np.array([r["conditioned_m"] for r in rows])
    keep = t > 1.0
    assert np.nanstd(z[keep, bad_link]) < 0.2 * np.std(raw[keep, bad_link])
    assert abs(np.nanmedian(z[keep, bad_link]) - truth[bad_link]) < 0.04
    assert (
        np.median([r["diagnostics"][bad_link]["sigma_m"] for r in rows[34:]])
        > 0.6 * sigma
    )


def test_slow_motion_below_instantaneous_velocity_threshold_still_releases_and_progresses():
    t = np.arange(0.0, 9.0, 0.03)
    progress = 0.03 * np.clip(t - 1.0, 0.0, 4.0)
    raw = np.c_[0.3 + progress, 1.0 - progress * 0.5, 0.9 - progress * 0.5]
    p = AdaptiveRangeParameters(
        minimum_motion_rate_mps=0.06,
        minimum_motion_delta_m=0.04,
        evidence_sigma=4.0,
        moving_q_m2_s3=10.0,
    )
    rows = condition_ranges(t, raw, p)
    assert any(
        r["stationary_state"] == "moving_or_unknown"
        for r, tt in zip(rows, t)
        if 2.0 < tt < 5.0
    )
    assert np.max(np.abs(rows[-1]["conditioned_m"] - raw[-1])) < 0.005
