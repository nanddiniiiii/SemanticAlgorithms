import unittest
from src.bitstream import encode_packet,decode_packet,pack_codes,unpack_codes
class Packets(unittest.TestCase):
    def test_all_index_values_and_partial_byte(self):
        for values in [list(range(1024)),[1],[0,1023,4]]:
            self.assertEqual(unpack_codes(pack_codes(values),len(values)),values)
    def test_position_order_roundtrip(self):
        packet=encode_packet([511,0,200],[15,1000,200],[5,9],[7,6])
        self.assertEqual(decode_packet(packet),[([0,200,511],[1000,200,15]),([5,9],[7,6])])
    def test_corruption_detected(self):
        packet=bytearray(encode_packet([0],[1],[511],[1023]));packet[20]^=1
        with self.assertRaisesRegex(ValueError,'CRC'):
            decode_packet(bytes(packet))
    def test_duplicate_position_rejected(self):
        with self.assertRaises(ValueError):
            encode_packet([1,1],[2,3],[0],[4])
