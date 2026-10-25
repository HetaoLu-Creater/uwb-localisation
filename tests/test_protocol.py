import struct
import unittest

from uwb_localisation.protocol import (
    HEADER,
    TWRStreamDecoder,
    parse_frame,
    xor_checksum,
)


def make_frame() -> bytes:
    body = bytearray(91)
    struct.pack_into("<IHHBB", body, 0, 123456, 0x8834, 0, 7, 0b00000111)
    struct.pack_into("<8I", body, 10, 2000, 3000, 4000, 0, 0, 0, 0, 0)
    body[42] = 1
    struct.pack_into("<8I", body, 43, 2100, 3100, 4100, 0, 0, 0, 0, 0)
    body[75] = 0
    body[76] = 1
    body[77] = 0b00000111
    body[78] = 0
    struct.pack_into("<iii", body, 79, 0, 0, 0)
    prefix = HEADER + bytes([len(body)]) + bytes(body)
    return prefix + bytes([xor_checksum(prefix[len(HEADER) :])]) + b"\r\n"


class ProtocolTests(unittest.TestCase):
    def test_parse_frame(self):
        packet = parse_frame(make_frame())
        self.assertEqual(packet.timer, 123456)
        self.assertEqual(packet.tag_id, 0x8834)
        self.assertEqual(packet.sequence, 7)
        self.assertTrue(packet.checksum_ok)
        self.assertEqual(packet.distances_m(), {0: 2.1, 1: 3.1, 2: 4.1})

    def test_stream_decoder_accepts_partial_chunks(self):
        frame = make_frame()
        decoder = TWRStreamDecoder()
        self.assertEqual(decoder.feed(b"noise" + frame[:20]), [])
        packets = decoder.feed(frame[20:])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0].sequence, 7)


if __name__ == "__main__":
    unittest.main()
