"""Decode, timestamp and process the same inputs during live use and replay."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .packet_clock import PacketClock, nominal_packet_interval_s
from .protocol import TWRStreamDecoder
from .runtime_tracker import make_tracker


def json_safe(value):
    """Replace nonfinite values with null in browser and recording output."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ProcessingPipeline:
    def __init__(self, config, nominal_interval_s=None):
        self.config = config
        self.decoder = TWRStreamDecoder()
        self.clock = PacketClock(
            nominal_interval_s=(nominal_interval_s or nominal_packet_interval_s(config))
        )
        self.tracker = make_tracker(config)
        self.origin_s = None
        self.last = None
        self.frame_count = 0

    def feed(self, data, host_monotonic_s, host_epoch_s):
        packets = self.decoder.feed(data)
        stamps = self.clock.resolve_batch(
            packets, host_monotonic_s=host_monotonic_s, host_epoch_s=host_epoch_s
        )
        rows = []
        for packet, stamp in zip(packets, stamps):
            if self.origin_s is None:
                self.origin_s = stamp.monotonic_s
            t = stamp.monotonic_s - self.origin_s
            admitted = (
                packet.checksum_ok
                and packet.source_anchor_id == 0
                and packet.tag_id == 0
                and not stamp.duplicate
            )
            row = dict(
                time_s=t,
                timer_ms=packet.timer,
                sequence=packet.sequence,
                checksum_ok=packet.checksum_ok,
                duplicate=stamp.duplicate,
                admitted=admitted,
                raw_ranges_m=[v / 1000 for v in packet.raw_ranges_mm[:3]],
                tracked_position_m=None,
                displayed_position_m=None,
                source_time_s=None,
                state="rejected_frame",
            )
            if admitted:
                frame = self.tracker.process(packet.distances_m(False), t)
                display = frame.output_stage or {}
                row.update(
                    filtered_ranges_m=[
                        frame.range_states[i].filtered_m for i in range(3)
                    ],
                    calibrated_ranges_m=[
                        frame.range_states[i].calibrated_m for i in range(3)
                    ],
                    displayed_position_m=display.get("xy"),
                    source_time_s=display.get("source_time_s"),
                    state=display.get("reason", "no_position"),
                    new_display=display.get("new_output", False),
                )
                if frame.tracking is not None:
                    row["tracked_position_m"] = [frame.tracking.x_m, frame.tracking.y_m]
                    row["tracked_source_time_s"] = t - frame.tracking.prediction_age_s
                self.last = row
            self.frame_count += 1
            rows.append(json_safe(row))
        return rows

    def snapshot(self, host_monotonic_s):
        if self.origin_s is None:
            return dict(
                time_s=0,
                frames=self.frame_count,
                state="waiting_for_frames",
                displayed_position_m=None,
            )
        t = max(
            host_monotonic_s - self.origin_s, self.last["time_s"] if self.last else 0.0
        )
        state = dict(self.last or {})
        if self.tracker.publisher.clock is not None:
            display = self.tracker.publisher.at(max(t, self.tracker.publisher.clock))
            state.update(displayed_position_m=display["xy"], state=display["reason"])
        source = state.get("tracked_source_time_s")
        if (
            source is None
            or t - source
            > self.config["empirical_filter"]["maximum_prediction_age_s"] + 1e-12
        ):
            state["tracked_position_m"] = None
        state.update(time_s=t, frames=self.frame_count)
        return json_safe(state)


def iter_recording_chunks(directory):
    """Read recorded serial bytes in their original chunks and timing."""
    directory = Path(directory)
    raw = (directory / "raw_stream.bin").read_bytes()
    offset = 0
    with (directory / "raw_chunks.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            start, length = int(row["byte_offset"]), int(row["length_bytes"])
            if start != offset or length < 0 or start + length > len(raw):
                raise ValueError("The raw chunk index is not contiguous")
            offset += length
            yield (
                raw[start:offset],
                float(row["timestamp_monotonic_s"]),
                float(row["timestamp_epoch_s"]),
            )
    if offset != len(raw):
        raise ValueError("The chunk index does not cover the raw stream")


def replay_recording(directory, config):
    directory = Path(directory)
    metadata = json.loads((directory / "session.json").read_text())
    pipeline = ProcessingPipeline(config, metadata.get("nominal_packet_interval_s"))
    rows = []
    for chunk, monotonic, epoch in iter_recording_chunks(directory):
        rows.extend(pipeline.feed(chunk, monotonic, epoch))
    return rows
