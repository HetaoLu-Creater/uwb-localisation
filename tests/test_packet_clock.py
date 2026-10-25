from __future__ import annotations

import unittest

from uwb_localisation.packet_clock import (
    PacketClock,
    nominal_packet_interval_s,
    packet_clock_metadata,
)
from uwb_localisation.protocol import TWRPacket


def packet(timer: int, sequence: int) -> TWRPacket:
    return TWRPacket(
        timer=timer,
        tag_id=0,
        source_anchor_id=0,
        sequence=sequence,
        valid_mask=0b00000111,
        raw_ranges_mm=(300, 400, 500, 0, 0, 0, 0, 0),
        kalman_enabled=False,
        filtered_ranges_mm=(0, 0, 0, 0, 0, 0, 0, 0),
        position_enabled=False,
        position_dimension=0,
        position_anchor_mask=0,
        position_result=0,
        position_xyz=(0, 0, 0),
        checksum_received=0,
        checksum_calculated=0,
        frame=b"",
    )


class PacketClockTests(unittest.TestCase):
    def test_one_serial_chunk_gets_distinct_device_times(self):
        clock = PacketClock()
        result = clock.resolve_batch(
            [packet(1000, 254), packet(1100, 255), packet(1200, 0)],
            host_monotonic_s=50.0,
            host_epoch_s=1000.0,
        )

        self.assertAlmostEqual(result[0].monotonic_s, 49.8)
        self.assertAlmostEqual(result[1].monotonic_s, 49.9)
        self.assertAlmostEqual(result[2].monotonic_s, 50.0)
        self.assertAlmostEqual(result[2].epoch_s, 1000.0)
        self.assertEqual(result[1].source, "device_timer")
        self.assertTrue(result[2].sequence_wrapped)

    def test_observed_99_100_101_millisecond_jitter_is_preserved(self):
        clock = PacketClock()
        result = clock.resolve_batch(
            [
                packet(1000, 1),
                packet(1099, 2),
                packet(1199, 3),
                packet(1300, 4),
            ],
            host_monotonic_s=10.0,
            host_epoch_s=100.0,
        )

        intervals = [
            result[index].monotonic_s - result[index - 1].monotonic_s
            for index in range(1, len(result))
        ]
        for actual, expected in zip(intervals, (0.099, 0.100, 0.101), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_uint32_timer_wrap_is_a_normal_device_transition(self):
        clock = PacketClock()
        clock.resolve_batch(
            [packet(0xFFFFFFF0, 10)],
            host_monotonic_s=10.0,
            host_epoch_s=100.0,
        )
        result = clock.resolve_batch(
            [packet(84, 11)],
            host_monotonic_s=10.1,
            host_epoch_s=100.1,
        )[0]

        self.assertAlmostEqual(result.monotonic_s, 10.1)
        self.assertAlmostEqual(result.timer_delta_s or 0.0, 0.1)
        self.assertEqual(result.source, "device_timer")
        self.assertTrue(result.timer_wrapped)
        self.assertFalse(result.timer_discontinuity)

    def test_duplicate_does_not_advance_time_and_gap_is_reported(self):
        clock = PacketClock()
        first = packet(1000, 10)
        initial = clock.resolve_batch(
            [first], host_monotonic_s=20.0, host_epoch_s=200.0
        )[0]
        duplicate = clock.resolve_batch(
            [first], host_monotonic_s=20.05, host_epoch_s=200.05
        )[0]
        after_gap = clock.resolve_batch(
            [packet(1400, 14)],
            host_monotonic_s=20.4,
            host_epoch_s=200.4,
        )[0]

        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.source, "duplicate")
        self.assertEqual(duplicate.monotonic_s, initial.monotonic_s)
        self.assertAlmostEqual(after_gap.monotonic_s, 20.4)
        self.assertEqual(after_gap.sequence_delta, 4)
        self.assertEqual(after_gap.missed_packets, 3)

    def test_delayed_retransmission_is_skipped_across_batches(self):
        clock = PacketClock()
        old = packet(1000, 1)
        clock.resolve_batch([old], host_monotonic_s=10.0, host_epoch_s=100.0)
        newest = clock.resolve_batch(
            [packet(1100, 2)],
            host_monotonic_s=10.1,
            host_epoch_s=100.1,
        )[0]
        retransmission = clock.resolve_batch(
            [old], host_monotonic_s=10.2, host_epoch_s=100.2
        )[0]
        recovered = clock.resolve_batch(
            [packet(1200, 3)],
            host_monotonic_s=10.3,
            host_epoch_s=100.3,
        )[0]

        self.assertTrue(retransmission.duplicate)
        self.assertEqual(retransmission.source, "duplicate_retransmission")
        self.assertEqual(retransmission.monotonic_s, newest.monotonic_s)
        self.assertEqual(recovered.source, "device_timer")
        self.assertAlmostEqual(recovered.monotonic_s, 10.2)

    def test_delayed_retransmission_in_first_batch_keeps_newest_time_anchor(self):
        clock = PacketClock()
        old = packet(1000, 1)
        result = clock.resolve_batch(
            [old, packet(1100, 2), old],
            host_monotonic_s=50.0,
            host_epoch_s=500.0,
        )

        self.assertAlmostEqual(result[0].monotonic_s, 49.9)
        self.assertAlmostEqual(result[1].monotonic_s, 50.0)
        self.assertTrue(result[2].duplicate)
        self.assertEqual(result[2].source, "duplicate_retransmission")
        self.assertAlmostEqual(result[2].monotonic_s, 50.0)

    def test_configured_cap3_interval_is_used_for_timer_fallback(self):
        config = {
            "range_filter": {"default_dt_s": 0.03},
            "twr_device_profile": {
                "apply_on_configure": True,
                "tag_slots": 3,
                "estimated_frame_period_ms": 30.0,
                "estimated_frame_rate_hz": 1000.0 / 30.0,
            },
        }
        interval_s = nominal_packet_interval_s(config)
        clock = PacketClock(nominal_interval_s=interval_s)
        initial = clock.resolve_batch(
            [packet(1000, 1)], host_monotonic_s=1.0, host_epoch_s=10.0
        )[0]
        fallback = clock.resolve_batch(
            [packet(1000, 2)], host_monotonic_s=1.01, host_epoch_s=10.01
        )[0]

        self.assertEqual(interval_s, 0.03)
        self.assertEqual(fallback.source, "host_fallback")
        self.assertAlmostEqual(fallback.monotonic_s - initial.monotonic_s, 0.03)
        self.assertEqual(
            packet_clock_metadata(clock)["nominal_fallback_interval_s"],
            0.03,
        )

    def test_reset_or_implausible_timer_uses_host_then_rebases(self):
        clock = PacketClock()
        clock.resolve_batch(
            [packet(5000, 50)],
            host_monotonic_s=30.0,
            host_epoch_s=300.0,
        )
        reset = clock.resolve_batch(
            [packet(100, 0)],
            host_monotonic_s=30.2,
            host_epoch_s=300.2,
        )[0]
        recovered = clock.resolve_batch(
            [packet(200, 1)],
            host_monotonic_s=30.3,
            host_epoch_s=300.3,
        )[0]

        self.assertEqual(reset.source, "host_fallback")
        self.assertTrue(reset.timer_discontinuity)
        self.assertAlmostEqual(reset.monotonic_s, 30.2)
        self.assertEqual(recovered.source, "device_timer")
        self.assertFalse(recovered.timer_discontinuity)
        self.assertAlmostEqual(recovered.monotonic_s, 30.3)

    def test_stopped_and_large_forward_timers_use_host_fallback(self):
        clock = PacketClock()
        clock.resolve_batch(
            [packet(1000, 1)],
            host_monotonic_s=40.0,
            host_epoch_s=400.0,
        )
        stopped = clock.resolve_batch(
            [packet(1000, 2)],
            host_monotonic_s=40.15,
            host_epoch_s=400.15,
        )[0]
        huge_jump = clock.resolve_batch(
            [packet(1_000_000, 3)],
            host_monotonic_s=40.25,
            host_epoch_s=400.25,
        )[0]

        self.assertEqual(stopped.source, "host_fallback")
        self.assertTrue(stopped.timer_discontinuity)
        self.assertAlmostEqual(stopped.monotonic_s, 40.15)
        self.assertEqual(huge_jump.source, "host_fallback")
        self.assertTrue(huge_jump.timer_discontinuity)
        self.assertAlmostEqual(huge_jump.monotonic_s, 40.25)


if __name__ == "__main__":
    unittest.main()
