from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .at import nominal_twr_frame_period_ms
from .protocol import TWRPacket


UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
UINT8_MODULUS = 1 << 8

PACKET_CLOCK_METADATA = {
    "device_timer_unit": "millisecond",
    "device_timer_modulus": UINT32_MODULUS,
    "sequence_modulus": UINT8_MODULUS,
    "normal_time_source": "packet.timer",
    "fallback_time_source": "host_monotonic_and_learned_cadence",
    "duplicates_update_filters": False,
    "delayed_retransmissions_update_filters": False,
}


def nominal_packet_interval_s(
    config: Mapping[str, Any], *, default: float = 0.1
) -> float:
    """Return the configured fallback interval for frame timing.

    Frames normally use the BU03 device timer. The fallback interval follows the
    TWR profile during startup, resets and timer stalls.
    """

    fallback = float(config.get("range_filter", {}).get("default_dt_s", default))
    profile = config.get("twr_device_profile", {})
    if isinstance(profile, Mapping) and profile.get("apply_on_configure", False):
        tag_slots = profile.get("tag_slots")
        period_ms = profile.get("estimated_frame_period_ms")
        rate_hz = profile.get("estimated_frame_rate_hz")
        if tag_slots is not None:
            fallback = nominal_twr_frame_period_ms(tag_slots) / 1000.0
        elif period_ms is not None:
            fallback = float(period_ms) / 1000.0
        elif rate_hz is not None:
            fallback = 1.0 / float(rate_hz)
    if not math.isfinite(fallback) or fallback <= 0.0:
        raise ValueError("configured packet interval must be positive and finite")
    return fallback


def packet_clock_metadata(clock: "PacketClock") -> dict[str, Any]:
    """Build session metadata for the actual clock instance in use."""

    return {
        **PACKET_CLOCK_METADATA,
        "nominal_fallback_interval_s": clock.nominal_interval_s,
        "recent_frame_identity_window": clock.recent_identity_window,
    }


@dataclass(frozen=True)
class PacketTimestamp:
    """Host-aligned time and continuity information for one BU03 packet."""

    monotonic_s: float
    epoch_s: float
    source: str
    timer_delta_s: float | None
    sequence_delta: int | None
    missed_packets: int
    duplicate: bool
    timer_wrapped: bool
    sequence_wrapped: bool
    timer_discontinuity: bool


class PacketClock:
    """Align BU03 device time with the host clock.

    The device timer is an unsigned 32-bit millisecond counter. The sequence field
    is an unsigned 8-bit counter used to identify duplicate and missing frames.
    Device time separates frames that arrive together in one serial read. The
    newest frame in the first batch is aligned with the host receive time.

    A stopped, reset or implausible timer uses host monotonic time and the learned
    frame interval. Timer wraparound continues the device timeline. Recognised
    retransmissions retain the preceding filter time.
    """

    def __init__(
        self,
        *,
        timer_hz: float = 1000.0,
        nominal_interval_s: float = 0.1,
        recent_identity_window: int = 256,
    ) -> None:
        if not math.isfinite(timer_hz) or timer_hz <= 0.0:
            raise ValueError("timer_hz must be positive and finite")
        if not math.isfinite(nominal_interval_s) or nominal_interval_s <= 0.0:
            raise ValueError("nominal_interval_s must be positive and finite")
        if recent_identity_window < 2:
            raise ValueError("recent_identity_window must be at least 2")
        self.timer_hz = float(timer_hz)
        self.nominal_interval_s = float(nominal_interval_s)
        self.recent_identity_window = int(recent_identity_window)
        self.reset()

    def reset(self) -> None:
        self._last_timer: int | None = None
        self._last_sequence: int | None = None
        self._last_monotonic_s: float | None = None
        self._last_epoch_s: float | None = None
        self._last_host_receive_s: float | None = None
        self._estimated_interval_s = self.nominal_interval_s
        self._recent_identities: deque[tuple[object, ...]] = deque()
        self._recent_identity_set: set[tuple[object, ...]] = set()

    @staticmethod
    def _packet_identity(packet: TWRPacket) -> tuple[object, ...]:
        if packet.frame:
            return ("frame", bytes(packet.frame))
        # Reconstruct frame identity from payload fields for programmatically
        # created packets.
        return (
            "fields",
            int(packet.timer),
            int(packet.sequence),
            int(packet.tag_id),
            int(packet.source_anchor_id),
            int(packet.valid_mask),
            packet.raw_ranges_mm,
            packet.filtered_ranges_mm,
            packet.position_xyz,
            int(packet.checksum_received),
            int(packet.checksum_calculated),
        )

    def _remember_identity(self, identity: tuple[object, ...]) -> None:
        if len(self._recent_identities) >= self.recent_identity_window:
            expired = self._recent_identities.popleft()
            self._recent_identity_set.discard(expired)
        self._recent_identities.append(identity)
        self._recent_identity_set.add(identity)

    @property
    def estimated_interval_s(self) -> float:
        return self._estimated_interval_s

    def resolve_batch(
        self,
        packets: Iterable[TWRPacket],
        *,
        host_monotonic_s: float | None = None,
        host_epoch_s: float | None = None,
    ) -> list[PacketTimestamp]:
        """Timestamp frames decoded from one serial read.

        ``host_monotonic_s`` and ``host_epoch_s`` give the shared receive time of the
        serial chunk. Device timestamps separate individual frames within the chunk.
        """

        packet_list = list(packets)
        if not packet_list:
            return []
        host_monotonic = (
            time.monotonic() if host_monotonic_s is None else float(host_monotonic_s)
        )
        host_epoch = time.time() if host_epoch_s is None else float(host_epoch_s)
        if not math.isfinite(host_monotonic) or not math.isfinite(host_epoch):
            raise ValueError("host timestamps must be finite")

        initially_empty = self._last_timer is None
        host_elapsed = (
            0.0
            if self._last_host_receive_s is None
            else max(0.0, host_monotonic - self._last_host_receive_s)
        )
        timestamps: list[PacketTimestamp] = []
        fallback_elapsed = host_elapsed
        for packet in packet_list:
            resolved = self._resolve_one(
                packet,
                host_monotonic_s=host_monotonic,
                host_epoch_s=host_epoch,
                host_elapsed_s=fallback_elapsed,
            )
            timestamps.append(resolved)
            # Keep the host elapsed time available for the first new frame.
            if not resolved.duplicate:
                fallback_elapsed = 0.0

        if initially_empty:
            # Align the newest frame in the first batch with its receive time.
            shift_monotonic = host_monotonic - timestamps[-1].monotonic_s
            shift_epoch = host_epoch - timestamps[-1].epoch_s
            timestamps = [
                replace(
                    item,
                    monotonic_s=item.monotonic_s + shift_monotonic,
                    epoch_s=item.epoch_s + shift_epoch,
                )
                for item in timestamps
            ]
            assert self._last_monotonic_s is not None
            assert self._last_epoch_s is not None
            self._last_monotonic_s += shift_monotonic
            self._last_epoch_s += shift_epoch

        if any(not item.duplicate for item in timestamps):
            self._last_host_receive_s = host_monotonic
        return timestamps

    def _resolve_one(
        self,
        packet: TWRPacket,
        *,
        host_monotonic_s: float,
        host_epoch_s: float,
        host_elapsed_s: float,
    ) -> PacketTimestamp:
        timer = int(packet.timer) & (UINT32_MODULUS - 1)
        sequence = int(packet.sequence) & (UINT8_MODULUS - 1)
        identity = self._packet_identity(packet)
        if identity in self._recent_identity_set:
            assert self._last_monotonic_s is not None
            assert self._last_epoch_s is not None
            immediate = timer == self._last_timer and sequence == self._last_sequence
            return PacketTimestamp(
                monotonic_s=self._last_monotonic_s,
                epoch_s=self._last_epoch_s,
                source=("duplicate" if immediate else "duplicate_retransmission"),
                timer_delta_s=0.0 if immediate else None,
                sequence_delta=0 if immediate else None,
                missed_packets=0,
                duplicate=True,
                timer_wrapped=False,
                sequence_wrapped=False,
                timer_discontinuity=False,
            )
        if self._last_timer is None:
            self._last_timer = timer
            self._last_sequence = sequence
            self._last_monotonic_s = host_monotonic_s
            self._last_epoch_s = host_epoch_s
            self._remember_identity(identity)
            return PacketTimestamp(
                monotonic_s=host_monotonic_s,
                epoch_s=host_epoch_s,
                source="host_initial",
                timer_delta_s=None,
                sequence_delta=None,
                missed_packets=0,
                duplicate=False,
                timer_wrapped=False,
                sequence_wrapped=False,
                timer_discontinuity=False,
            )

        assert self._last_sequence is not None
        assert self._last_monotonic_s is not None
        assert self._last_epoch_s is not None
        previous_timer = self._last_timer
        previous_sequence = self._last_sequence
        timer_ticks = (timer - previous_timer) & (UINT32_MODULUS - 1)
        sequence_delta = (sequence - previous_sequence) & (UINT8_MODULUS - 1)

        if timer_ticks == 0 and sequence_delta == 0:
            return PacketTimestamp(
                monotonic_s=self._last_monotonic_s,
                epoch_s=self._last_epoch_s,
                source="duplicate",
                timer_delta_s=0.0,
                sequence_delta=0,
                missed_packets=0,
                duplicate=True,
                timer_wrapped=False,
                sequence_wrapped=False,
                timer_discontinuity=False,
            )

        sequence_steps = sequence_delta if 1 <= sequence_delta <= 128 else 1
        expected_delta_s = self._estimated_interval_s * sequence_steps
        raw_device_delta_s = timer_ticks / self.timer_hz
        plausible_limit_s = max(
            5.0,
            host_elapsed_s * 4.0 + 1.0,
            expected_delta_s * 8.0,
        )
        timer_forward = 0 < timer_ticks < UINT32_HALF_RANGE
        timer_plausible = timer_forward and raw_device_delta_s <= plausible_limit_s

        if timer_plausible:
            delta_s = raw_device_delta_s
            source = "device_timer"
            timer_discontinuity = False
        else:
            fallback = max(host_elapsed_s, expected_delta_s)
            delta_s = max(0.001, fallback)
            source = "host_fallback" if host_elapsed_s > 0.0 else "estimated_cadence"
            timer_discontinuity = True

        missed_packets = 0
        if 1 < sequence_delta <= 128:
            missed_packets = sequence_delta - 1
        timer_wrapped = timer < previous_timer and timer_plausible
        sequence_wrapped = sequence < previous_sequence and 1 <= sequence_delta <= 128
        monotonic_s = self._last_monotonic_s + delta_s
        epoch_s = self._last_epoch_s + delta_s

        if timer_plausible and 1 <= sequence_delta <= 128:
            observed_interval_s = raw_device_delta_s / sequence_delta
            if 0.001 <= observed_interval_s <= 2.0:
                self._estimated_interval_s = (
                    0.8 * self._estimated_interval_s + 0.2 * observed_interval_s
                )

        # Rebase the clock after a timer discontinuity so later frames can
        # resume device timing in serial order.
        self._last_timer = timer
        self._last_sequence = sequence
        self._last_monotonic_s = monotonic_s
        self._last_epoch_s = epoch_s
        self._remember_identity(identity)
        return PacketTimestamp(
            monotonic_s=monotonic_s,
            epoch_s=epoch_s,
            source=source,
            timer_delta_s=(raw_device_delta_s if timer_forward else None),
            sequence_delta=sequence_delta,
            missed_packets=missed_packets,
            duplicate=False,
            timer_wrapped=timer_wrapped,
            sequence_wrapped=sequence_wrapped,
            timer_discontinuity=timer_discontinuity,
        )
