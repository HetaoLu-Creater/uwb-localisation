"""Read packaged serial bytes and their recorded timestamps."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from uwb_localisation.packet_clock import PacketClock
from uwb_localisation.protocol import TWRStreamDecoder

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def load_recording(path):
    path = Path(path)
    meta = read_json(path / "session.json")
    raw_bytes = (path / "raw_stream.bin").read_bytes()
    decoder = TWRStreamDecoder()
    clock = PacketClock(nominal_interval_s=meta["nominal_packet_interval_s"])
    times, ranges, inputs = [], [], []
    offset = 0
    with (path / "raw_chunks.csv").open(newline="") as stream:
        for chunk in csv.DictReader(stream):
            start, length = int(chunk["byte_offset"]), int(chunk["length_bytes"])
            if start != offset or start + length > len(raw_bytes):
                raise ValueError(f"Invalid byte index in {path}")
            offset += length
            packets = decoder.feed(raw_bytes[start:offset])
            stamps = clock.resolve_batch(
                packets,
                host_monotonic_s=float(chunk["timestamp_monotonic_s"]),
                host_epoch_s=float(chunk["timestamp_epoch_s"]),
            )
            for packet, stamp in zip(packets, stamps):
                raw = np.array(packet.raw_ranges_mm[:3], dtype=float) / 1000
                valid = np.array([bool(packet.valid_mask & (1 << i)) for i in range(3)])
                admitted = (
                    packet.checksum_ok
                    and packet.source_anchor_id == 0
                    and packet.tag_id == 0
                    and not stamp.duplicate
                )
                times.append(stamp.monotonic_s)
                ranges.append(raw)
                inputs.append(np.where(valid & admitted & (raw > 0), raw, np.nan))
    if offset != len(raw_bytes) or not times:
        raise ValueError(f"Incomplete or empty recording: {path}")
    times = np.asarray(times) - times[0]
    if not np.all(np.diff(times) > 0):
        raise ValueError(
            f"Recording contains repeated or unordered frame times: {path}"
        )
    return dict(
        meta=meta, time=times, raw=np.asarray(ranges), inputs=np.asarray(inputs)
    )


def recordings(layout="one_metre"):
    return [
        dict(load_recording(ROOT / spec["path"]), path=ROOT / spec["path"])
        for spec in read_json(ROOT / "data/recordings.json")
        if spec["layout"] == layout
    ]
