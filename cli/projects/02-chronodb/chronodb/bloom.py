"""
High-performance Bit-Vector Bloom Filter:
Probabilistic data structure for O(1) set membership testing with zero false negatives.
Uses Kirsch-Mitzenmacher double-hashing to generate k independent hash locations
from two base 32-bit hashes without external dependencies.
"""

from __future__ import annotations
import math
import zlib
from typing import Sequence, Optional


class BloomFilter:
    """
    Bit-vector Bloom Filter with optimal sizing and byte serialization.
    """

    def __init__(self, size_bits: int, num_hashes: int):
        self.size_bits = max(size_bits, 64)
        self.num_hashes = max(num_hashes, 1)
        self.num_bytes = (self.size_bits + 7) // 8
        self.bit_array = bytearray(self.num_bytes)
        self.count = 0

    @classmethod
    def create(cls, expected_items: int, false_positive_rate: float = 0.01) -> BloomFilter:
        """
        Computes optimal bit size m and hash count k for given item capacity and error rate:
        m = - (n * ln(p)) / (ln(2)^2)
        k = (m / n) * ln(2)
        """
        n = max(expected_items, 1)
        p = min(max(false_positive_rate, 1e-6), 0.5)

        m = int(- (n * math.log(p)) / (math.log(2) ** 2))
        k = int((m / n) * math.log(2))
        return cls(size_bits=max(m, 64), num_hashes=max(k, 1))

    def _get_hashes(self, key: bytes) -> tuple[int, int]:
        """
        Generates two 32-bit base hashes:
        h1 via CRC32, h2 via FNV-1a 32-bit.
        """
        h1 = zlib.crc32(key) & 0xFFFFFFFF

        # FNV-1a 32-bit hash
        h2 = 2166136261
        for b in key:
            h2 = ((h2 ^ b) * 16777619) & 0xFFFFFFFF
        return h1, h2

    def add(self, key: str | bytes) -> None:
        """Adds key to the Bloom filter."""
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key
        h1, h2 = self._get_hashes(key_bytes)

        m = self.size_bits
        for i in range(self.num_hashes):
            bit_idx = (h1 + i * h2) % m
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            self.bit_array[byte_idx] |= (1 << bit_offset)
        self.count += 1

    def __contains__(self, key: str | bytes) -> bool:
        """
        Checks if key is possibly in the set.
        False: Guaranteed NOT in the set (0% false negatives).
        True: Probably in the set (bounded by false_positive_rate).
        """
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key
        h1, h2 = self._get_hashes(key_bytes)

        m = self.size_bits
        for i in range(self.num_hashes):
            bit_idx = (h1 + i * h2) % m
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            if not (self.bit_array[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def to_bytes(self) -> bytes:
        """
        Serializes filter header and bit array to bytes:
        [size_bits: 4B uint32][num_hashes: 2B uint16][count: 4B uint32][bit_array: N bytes]
        """
        import struct
        header = struct.pack(">IHI", self.size_bits, self.num_hashes, self.count)
        return header + bytes(self.bit_array)

    @classmethod
    def from_bytes(cls, data: bytes) -> BloomFilter:
        """Deserializes filter from bytes."""
        import struct
        size_bits, num_hashes, count = struct.unpack(">IHI", data[:10])
        bf = cls(size_bits=size_bits, num_hashes=num_hashes)
        bf.count = count
        bit_data = data[10:]
        bf.bit_array = bytearray(bit_data)
        return bf
