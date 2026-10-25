from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass


HEADER = b"CmdM:4"
EXPECTED_BODY_LENGTH = 91


def xor_checksum(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


@dataclass(frozen=True)
class TWRPacket:
    timer: int
    tag_id: int
    source_anchor_id: int
    sequence: int
    valid_mask: int
    raw_ranges_mm: tuple[int, ...]
    kalman_enabled: bool
    filtered_ranges_mm: tuple[int, ...]
    position_enabled: bool
    position_dimension: int
    position_anchor_mask: int
    position_result: int
    position_xyz: tuple[int, int, int]
    checksum_received: int
    checksum_calculated: int
    frame: bytes

    @property
    def checksum_ok(self) -> bool:
        return self.checksum_received == self.checksum_calculated

    def distances_m(self, prefer_filtered: bool = True) -> dict[int, float]:
        values = (
            self.filtered_ranges_mm
            if prefer_filtered and self.kalman_enabled
            else self.raw_ranges_mm
        )
        distances: dict[int, float] = {}
        for anchor_id, value_mm in enumerate(values):
            if self.valid_mask & (1 << anchor_id) and value_mm > 0:
                distances[anchor_id] = value_mm / 1000.0
        return distances


def parse_frame(frame: bytes) -> TWRPacket:
    if not frame.startswith(HEADER):
        raise ValueError("Expected a CmdM:4 TWR frame")
    if len(frame) < len(HEADER) + 1:
        raise ValueError("Incomplete frame")

    body_length = frame[len(HEADER)]
    expected_length = len(HEADER) + 1 + body_length + 1 + 2
    if len(frame) != expected_length:
        raise ValueError(
            f"Invalid frame length: received {len(frame)}, expected {expected_length}"
        )
    if not frame.endswith(b"\r\n"):
        raise ValueError("Frame has no CRLF ending")
    if body_length < EXPECTED_BODY_LENGTH:
        raise ValueError(f"TWR body is too short: {body_length}")

    body_start = len(HEADER) + 1
    body = frame[body_start : body_start + body_length]
    checksum_received = frame[body_start + body_length]
    # The BU03 checksum covers the length byte and the complete frame body.
    checksum_calculated = xor_checksum(frame[len(HEADER) : body_start + body_length])

    timer, tag_id, source_anchor_id, sequence, valid_mask = struct.unpack_from(
        "<IHHBB", body, 0
    )
    raw_ranges_mm = struct.unpack_from("<8I", body, 10)
    kalman_enabled = bool(body[42])
    filtered_ranges_mm = struct.unpack_from("<8I", body, 43)
    position_enabled = bool(body[75])
    position_dimension = body[76]
    position_anchor_mask = body[77]
    position_result = body[78]
    position_xyz = struct.unpack_from("<iii", body, 79)

    return TWRPacket(
        timer=timer,
        tag_id=tag_id,
        source_anchor_id=source_anchor_id,
        sequence=sequence,
        valid_mask=valid_mask,
        raw_ranges_mm=raw_ranges_mm,
        kalman_enabled=kalman_enabled,
        filtered_ranges_mm=filtered_ranges_mm,
        position_enabled=position_enabled,
        position_dimension=position_dimension,
        position_anchor_mask=position_anchor_mask,
        position_result=position_result,
        position_xyz=position_xyz,
        checksum_received=checksum_received,
        checksum_calculated=checksum_calculated,
        frame=frame,
    )


class TWRStreamDecoder:
    def __init__(self, on_noise: Callable[[bytes], None] | None = None) -> None:
        self.buffer = bytearray()
        self.on_noise = on_noise

    def _discard_noise(self, count: int) -> None:
        if count and self.on_noise is not None:
            self.on_noise(bytes(self.buffer[:count]))
        del self.buffer[:count]

    def feed(self, data: bytes) -> list[TWRPacket]:
        self.buffer.extend(data)
        packets: list[TWRPacket] = []

        while True:
            header_index = self.buffer.find(HEADER)
            if header_index < 0:
                # Retain a partial frame header for the next read and pass complete
                # control responses to the callback.
                keep = 0
                for length in range(1, min(len(self.buffer), len(HEADER) - 1) + 1):
                    if self.buffer[-length:] == HEADER[:length]:
                        keep = length
                self._discard_noise(len(self.buffer) - keep)
                break

            if header_index:
                self._discard_noise(header_index)

            if len(self.buffer) < len(HEADER) + 1:
                break

            body_length = self.buffer[len(HEADER)]
            if body_length < 10 or body_length > 240:
                del self.buffer[0]
                continue

            frame_length = len(HEADER) + 1 + body_length + 1 + 2
            if len(self.buffer) < frame_length:
                break

            candidate = bytes(self.buffer[:frame_length])
            if not candidate.endswith(b"\r\n"):
                del self.buffer[0]
                continue

            del self.buffer[:frame_length]
            try:
                packets.append(parse_frame(candidate))
            except ValueError:
                continue

        return packets
