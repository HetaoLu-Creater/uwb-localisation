"""Check live/replay timing, capture and display behaviour together."""

import argparse
import csv
import json
import struct
import time
from pathlib import Path

import numpy as np

from uwb_localisation.cli import capture
from uwb_localisation.pipeline import ProcessingPipeline, replay_recording
from uwb_localisation.protocol import HEADER, xor_checksum

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / "configs/final.json").read_text())


def frame(index, mask=7, corrupt=False):
    body = bytearray(91)
    struct.pack_into("<IHHBB", body, 0, index * 30, 0, 0, index % 256, mask)
    struct.pack_into("<8I", body, 10, 900, 810, 700, 0, 0, 0, 0, 0)
    prefix = HEADER + bytes([len(body)]) + body
    checksum = xor_checksum(prefix[len(HEADER) :]) ^ int(corrupt)
    return prefix + bytes([checksum]) + b"\r\n"


def test_batched_device_times_and_expiry_without_new_frames():
    pipeline = ProcessingPipeline(configuration())
    rows = pipeline.feed(b"".join(frame(i) for i in range(30)), 100.0, 1000.0)
    np.testing.assert_allclose(np.diff([r["time_s"] for r in rows]), 0.03)
    assert rows[-1]["displayed_position_m"] is not None
    origin = pipeline.origin_s
    source = rows[-1]["source_time_s"]
    assert (
        pipeline.snapshot(origin + source + 0.399)["displayed_position_m"] is not None
    )
    assert pipeline.snapshot(origin + source + 0.401)["displayed_position_m"] is None
    assert (
        pipeline.snapshot(origin + rows[-1]["time_s"] + 0.301)["tracked_position_m"]
        is None
    )
    before = pipeline.tracker.publisher.clock
    pipeline.snapshot(102.0)
    assert pipeline.tracker.publisher.clock == before
    recovered = pipeline.feed(b"".join(frame(i) for i in range(70, 100)), 102.1, 1002.1)
    assert recovered[-1]["displayed_position_m"] is not None


def test_bad_frames_and_missing_ranges_do_not_make_positions():
    pipeline = ProcessingPipeline(configuration())
    rows = pipeline.feed(frame(0, corrupt=True) + frame(1, mask=0), 10.0, 100.0)
    assert len(rows) == 2
    assert not rows[0]["admitted"] and rows[1]["admitted"]
    assert rows[0]["raw_ranges_m"] == [0.9, 0.81, 0.7]
    assert all(row["displayed_position_m"] is None for row in rows)


def test_capture_saves_raw_bytes_even_when_no_position_is_available(
    tmp_path, monkeypatch
):
    chunk = frame(0, mask=0) + frame(1, corrupt=True) + b"incomplete"

    class Serial:
        in_waiting = len(chunk)

        def __init__(self, *_args, **_kwargs):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, _):
            if not self.sent:
                self.sent = True
                return chunk
            time.sleep(0.002)
            return b""

    monkeypatch.setattr("uwb_localisation.cli.serial.Serial", Serial)
    destination = tmp_path / "capture"
    args = argparse.Namespace(
        output=str(destination), port="test", baud=115200, seconds=0.03
    )
    capture(args, configuration())
    assert (destination / "raw_stream.bin").read_bytes() == chunk
    rows = [
        json.loads(line)
        for line in (destination / "outputs.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all(row["displayed_position_m"] is None for row in rows)
    assert replay_recording(destination, configuration()) == rows
    assert (
        json.loads((destination / "session.json").read_text())["status"] == "completed"
    )


def test_replay_uses_same_results_as_live_processing(tmp_path):
    config = configuration()
    chunks = [
        b"".join(frame(i) for i in range(15)),
        b"".join(frame(i) for i in range(15, 30)),
    ]
    (tmp_path / "raw_stream.bin").write_bytes(b"".join(chunks))
    (tmp_path / "session.json").write_text(
        json.dumps({"nominal_packet_interval_s": 0.03})
    )
    pipeline = ProcessingPipeline(config, 0.03)
    live_rows = []
    with (tmp_path / "raw_chunks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "byte_offset",
                "length_bytes",
                "timestamp_monotonic_s",
                "timestamp_epoch_s",
            ]
        )
        offset = 0
        for i, chunk in enumerate(chunks):
            writer.writerow([offset, len(chunk), 20 + i * 0.45, 100 + i * 0.45])
            live_rows.extend(pipeline.feed(chunk, 20 + i * 0.45, 100 + i * 0.45))
            offset += len(chunk)
    assert replay_recording(tmp_path, config) == live_rows
