"""
SynapseDB: Unit Tests for Columnar Compression Codecs & Zone Maps.
"""

import unittest
from synapse.types import DataType, Vector
from synapse.encoding import (
    ZoneMap,
    RLECodec,
    DictionaryCodec,
    BitPackingCodec
)


class TestSynapseEncoding(unittest.TestCase):
    def test_zone_map_pruning(self):
        vec = Vector(DataType.INT64, [10, 20, 30, 40, 50])
        zm = ZoneMap.compute(vec)

        self.assertEqual(zm.min_val, 10)
        self.assertEqual(zm.max_val, 50)
        self.assertEqual(zm.null_count, 0)
        self.assertEqual(zm.row_count, 5)

        # Predicate tests
        self.assertTrue(zm.can_match("=", 30))
        self.assertFalse(zm.can_match("=", 100))
        self.assertFalse(zm.can_match("=", 5))

        self.assertTrue(zm.can_match(">", 40))
        self.assertFalse(zm.can_match(">", 55))

        self.assertTrue(zm.can_match("<", 20))
        self.assertFalse(zm.can_match("<", 10))

    def test_rle_codec_integers_and_strings(self):
        # Repeated integers
        ints = [1, 1, 1, 2, 2, 3, 3, 3, 3, 1]
        encoded = RLECodec.encode(ints, DataType.INT64)
        decoded = RLECodec.decode(encoded, DataType.INT64)
        self.assertEqual(decoded, ints)

        # Repeated strings
        strings = ["US", "US", "US", "CA", "CA", "UK"]
        s_encoded = RLECodec.encode(strings, DataType.VARCHAR)
        s_decoded = RLECodec.decode(s_encoded, DataType.VARCHAR)
        self.assertEqual(s_decoded, strings)

    def test_dictionary_codec(self):
        data = ["red", "blue", "red", "green", "blue", "red", "red"]
        dictionary, codes = DictionaryCodec.encode(data)

        self.assertEqual(len(dictionary), 3)
        self.assertIn("red", dictionary)
        self.assertIn("blue", dictionary)
        self.assertIn("green", dictionary)

        decoded = DictionaryCodec.decode(dictionary, codes)
        self.assertEqual(decoded, data)

    def test_bit_packing_codec(self):
        # Sequence of integers with compact delta
        base = 1000
        ints = [base + 1, base + 2, base + 5, base + 12, base + 0, base + 15]
        packed = BitPackingCodec.encode(ints)
        unpacked = BitPackingCodec.decode(packed)

        self.assertEqual(unpacked, ints)

        # Empty
        empty_packed = BitPackingCodec.encode([])
        self.assertEqual(BitPackingCodec.decode(empty_packed), [])


if __name__ == "__main__":
    unittest.main()
