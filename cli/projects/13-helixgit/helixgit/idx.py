"""
HelixGit: Packfile Index (.idx v2) Engine.
Implements the 256-bucket fan-out table, lexicographical SHA-1 binary search,
CRC32 validation, and packfile offset lookup.
"""

import struct
import binascii
import hashlib
from typing import List, Tuple, Dict, Optional
from .pack import PackObjectInfo

IDX_MAGIC = b"\xfftOc"
IDX_VERSION = 2


class PackIndexWriter:
    """
    Constructs Git .idx v2 index files from packfile object metadata.
    """

    def __init__(self, pack_sha1: bytes) -> None:
        self.pack_sha1 = pack_sha1
        self.entries: List[PackObjectInfo] = []

    def add_entry(self, info: PackObjectInfo) -> None:
        self.entries.append(info)

    def write_index(self) -> bytes:
        """
        Serializes entries into Git .idx v2 binary stream.
        """
        # Sort entries lexicographically by binary 20-byte SHA-1
        sorted_entries = sorted(self.entries, key=lambda e: binascii.unhexlify(e.sha1))
        n = len(sorted_entries)

        out = bytearray()
        # 4 bytes magic + 4 bytes version
        out.extend(IDX_MAGIC)
        out.extend(struct.pack(">I", IDX_VERSION))

        # Build 256-bucket Level-1 fan-out table
        fanout = [0] * 256
        for e in sorted_entries:
            first_byte = binascii.unhexlify(e.sha1)[0]
            fanout[first_byte] += 1

        # Make cumulative
        cum = 0
        fanout_cum = [0] * 256
        for i in range(256):
            cum += fanout[i]
            fanout_cum[i] = cum

        for val in fanout_cum:
            out.extend(struct.pack(">I", val))

        # Table of 20-byte binary SHA-1s in sorted order
        for e in sorted_entries:
            out.extend(binascii.unhexlify(e.sha1))

        # Table of 4-byte CRC32s
        for e in sorted_entries:
            out.extend(struct.pack(">I", e.crc32))

        # Table of 4-byte packfile offsets
        for e in sorted_entries:
            out.extend(struct.pack(">I", e.offset))

        # 20-byte packfile SHA-1
        out.extend(self.pack_sha1)

        # 20-byte index SHA-1 over all preceding bytes
        idx_sha1 = hashlib.sha1(bytes(out)).digest()
        out.extend(idx_sha1)

        return bytes(out)


class PackIndexReader:
    """
    Parses and queries Git .idx v2 files with O(log N) binary search on SHA-1 hashes.
    """

    def __init__(self, idx_data: bytes) -> None:
        self.data = idx_data
        self.num_objects = 0
        self.fanout: List[int] = []
        self._shas_offset = 0
        self._crcs_offset = 0
        self._offsets_offset = 0
        self._parse()

    def _parse(self) -> None:
        if len(self.data) < 1032:  # 8 header + 1024 fanout
            raise ValueError("Corrupt .idx file: smaller than header + fanout table")

        magic = self.data[:4]
        if magic != IDX_MAGIC:
            raise ValueError(f"Invalid .idx magic: {magic!r}")

        version = struct.unpack(">I", self.data[4:8])[0]
        if version != IDX_VERSION:
            raise ValueError(f"Unsupported .idx version: {version}")

        # Parse 256 fanout entries
        self.fanout = list(struct.unpack(">256I", self.data[8:1032]))
        self.num_objects = self.fanout[255]

        # Calculate section offsets
        self._shas_offset = 1032
        self._crcs_offset = self._shas_offset + (self.num_objects * 20)
        self._offsets_offset = self._crcs_offset + (self.num_objects * 4)

        # Expected total size: offsets_offset + (num_objects * 4) + 20 (pack sha) + 20 (idx sha)
        expected_size = self._offsets_offset + (self.num_objects * 4) + 40
        if len(self.data) != expected_size:
            raise ValueError(f"Corrupt .idx file: size mismatch (expected {expected_size}, got {len(self.data)})")

        # Verify trailing SHA-1 checksum
        body = self.data[:-20]
        file_checksum = self.data[-20:]
        calc_checksum = hashlib.sha1(body).digest()
        if file_checksum != calc_checksum:
            raise ValueError("Corrupt .idx file: checksum mismatch")

    def lookup(self, sha1: str) -> Optional[Tuple[int, int]]:
        """
        Finds object offset and CRC32 in O(log N) time using 256-bucket fan-out table.
        Returns (pack_offset, crc32) or None if not found.
        """
        raw_sha = binascii.unhexlify(sha1.lower())
        first_byte = raw_sha[0]

        low = 0 if first_byte == 0 else self.fanout[first_byte - 1]
        high = self.fanout[first_byte]

        # Binary search in range [low, high)
        while low < high:
            mid = (low + high) // 2
            entry_sha = self.data[self._shas_offset + (mid * 20):self._shas_offset + (mid * 20) + 20]
            if entry_sha < raw_sha:
                low = mid + 1
            elif entry_sha > raw_sha:
                high = mid
            else:
                # Found exact match
                crc32 = struct.unpack(">I", self.data[self._crcs_offset + (mid * 4):self._crcs_offset + (mid * 4) + 4])[0]
                offset = struct.unpack(">I", self.data[self._offsets_offset + (mid * 4):self._offsets_offset + (mid * 4) + 4])[0]
                return offset, crc32

        return None
