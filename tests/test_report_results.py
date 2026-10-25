"""Check raw data integrity and reproduce the numerical report results."""

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from data_io import read_json
from evaluate import evaluate, check_results
from fit_calibration import fit_calibration, check_calibration


def test_raw_recordings_are_unchanged():
    manifest = read_json(ROOT / "data/recordings.json")
    assert len(manifest) == 20
    for recording in manifest:
        for filename, expected in recording["raw_file_sha256"].items():
            actual = hashlib.sha256(
                (ROOT / recording["path"] / filename).read_bytes()
            ).hexdigest()
            assert actual == expected, (recording["id"], filename)


def test_final_and_earlier_results():
    config = read_json(ROOT / "configs/final.json")
    result = evaluate(config)
    check_results(result)
    stationary = [r for r in result["one_metre"] if "position_rmse_m" in r]
    assert len(stationary) == 9
    assert round(100 * min(r["position_rmse_m"] for r in stationary), 2) == 1.55
    assert round(100 * max(r["position_rmse_m"] for r in stationary), 2) == 6.79


def test_calibration_refit():
    config = read_json(ROOT / "configs/final.json")
    result = fit_calibration()
    check_calibration(result, config)
    assert result["selected_definition"]["model"] == "contact_hinge"


def test_live_and_replay_pipeline_match_numerical_evaluation():
    """The live input path must give the same positions as the report evaluation."""
    import numpy as np
    from data_io import recordings
    from evaluate import replay
    from uwb_localisation.pipeline import (
        ProcessingPipeline,
        iter_recording_chunks,
        replay_recording,
    )

    config = read_json(ROOT / "configs/final.json")
    held_frames = 0
    for recording in recordings():
        tracked, displayed = replay(recording, config)
        rows = replay_recording(recording["path"], config)
        times = np.array([row["time_s"] for row in rows])
        np.testing.assert_array_equal(times, recording["time"])
        assert times[0] == 0.0
        expected_tracked = np.array(
            [p if p is not None else [np.nan, np.nan] for p in tracked]
        )
        actual_tracked = np.array(
            [
                row["tracked_position_m"]
                if row["tracked_position_m"] is not None
                else [np.nan, np.nan]
                for row in rows
            ]
        )
        actual_displayed = np.array(
            [
                row["displayed_position_m"]
                if row["displayed_position_m"] is not None
                else [np.nan, np.nan]
                for row in rows
            ]
        )
        np.testing.assert_allclose(
            actual_tracked, expected_tracked, rtol=0, atol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            actual_displayed, displayed, rtol=0, atol=1e-12, equal_nan=True
        )
        held_frames += sum(
            row["displayed_position_m"] is not None and not row.get("new_display")
            for row in rows
        )

    # A reception gap clears the live display after the measurement expires.
    selected = ROOT / "data/one_metre/from_a2_run1"
    metadata = read_json(selected / "session.json")
    pipeline = ProcessingPipeline(config, metadata["nominal_packet_interval_s"])
    checked_expiry = False
    for chunk, monotonic, epoch in iter_recording_chunks(selected):
        rows = pipeline.feed(chunk, monotonic, epoch)
        for row in rows[-1:]:
            if row["displayed_position_m"] is None:
                continue
            source = row["source_time_s"]
            current = pipeline.snapshot(pipeline.origin_s + row["time_s"])
            np.testing.assert_allclose(
                current["displayed_position_m"],
                row["displayed_position_m"],
                rtol=0,
                atol=1e-12,
            )
            expired = pipeline.snapshot(
                pipeline.origin_s
                + source
                + config["reliable_output"]["maximum_source_age_s"]
                + 1e-6
            )
            assert expired["displayed_position_m"] is None
            assert expired["state"] == "stale"
            checked_expiry = True
            break
        if checked_expiry:
            break
    assert checked_expiry
    assert held_frames > 0
