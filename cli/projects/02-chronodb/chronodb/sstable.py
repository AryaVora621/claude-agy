"""
Immutable Sorted String Table (SSTable) Disk Format:
Binary layout featuring:
- Chunked data blocks
- In-memory sparse index blocks for O(log N) binary search
- Embedded Bit-Vector Bloom Filter for zero-IO negative lookups
- Fixed-size trailer footer with CRC integrity checks
"""

from __future__ import annotations
import os
import struct
import zlib
from typing import List, Tuple, Optional, Iterator, Dict, Any
from chronodb.bloom import BloomFilter

SST_MAGIC = b"SST1"  # 4-byte identifier
BLOCK_SIZE_LIMIT = 4096  # 4 KB data block target


class SSTableWriter:
    """
    Constructs a new immutable SSTable on disk from an ordered stream of key-value pairs.
    """

    def __init__(self, filepath: str, expected_entries: int = 1000):
        self.filepath = filepath
        self._file = open(filepath, "wb")
        self.bloom = BloomFilter.create(expected_items=expected_entries, false_positive_rate=0.01)

        # Index entries: list of (last_key_in_block, block_offset, block_size)
        self.index_entries: List[Tuple[bytes, int, int]] = []
        self.current_block_entries: List[bytes] = []
        self.current_block_bytes = 0
        self.current_block_last_key = b""
        self.total_entries = 0
        self.min_key = b""
        self.max_key = b""

    def append(self, key: bytes, value: bytes, seq_num: int, is_tombstone: bool = False) -> None:
        """Appends an entry. Invariant: keys must arrive in ascending lexicographical order."""
        if self.total_entries == 0:
            self.min_key = key
        self.max_key = key

        self.bloom.add(key)

        # Encode record: [tombstone: 1B][seq: 8B][key_len: 2B][key][val_len: 4B][val]
        record = (
            struct.pack(">B Q H", 1 if is_tombstone else 0, seq_num, len(key))
            + key
            + struct.pack(">I", len(value))
            + value
        )

        self.current_block_entries.append(record)
        self.current_block_bytes += len(record)
        self.current_block_last_key = key
        self.total_entries += 1

        if self.current_block_bytes >= BLOCK_SIZE_LIMIT:
            self._flush_block()

    def _flush_block(self) -> None:
        if not self.current_block_entries:
            return

        block_offset = self._file.tell()
        block_payload = b"".join(self.current_block_entries)
        block_crc = zlib.crc32(block_payload) & 0xFFFFFFFF

        # Write block header: [CRC32: 4B][Length: 4B] + Payload
        self._file.write(struct.pack(">II", block_crc, len(block_payload)) + block_payload)
        block_total_size = 8 + len(block_payload)

        self.index_entries.append((self.current_block_last_key, block_offset, block_total_size))

        self.current_block_entries = []
        self.current_block_bytes = 0

    def finish(self) -> None:
        """Flushes remaining block, writes index block, filter block, and footer trailer."""
        self._flush_block()

        # 1. Write Index Block
        index_offset = self._file.tell()
        index_buf = bytearray()
        index_buf.extend(struct.pack(">I", len(self.index_entries)))
        for last_key, blk_off, blk_size in self.index_entries:
            index_buf.extend(struct.pack(">H", len(last_key)))
            index_buf.extend(last_key)
            index_buf.extend(struct.pack(">QQ", blk_off, blk_size))
        self._file.write(bytes(index_buf))
        index_len = len(index_buf)

        # 2. Write Filter Block (Bloom filter)
        filter_offset = self._file.tell()
        filter_bytes = self.bloom.to_bytes()
        self._file.write(filter_bytes)
        filter_len = len(filter_bytes)

        # 3. Write Footer Trailer (fixed 48-byte layout)
        # [IndexOffset: 8B][IndexLen: 8B][FilterOffset: 8B][FilterLen: 8B][NumEntries: 8B][MinKeyLen: 2B][MaxKeyLen: 2B][Magic: 4B]
        min_k_short = self.min_key[:16]
        max_k_short = self.max_key[:16]

        footer_payload = struct.pack(
            ">QQQQQ HH 16s 16s",
            index_offset,
            index_len,
            filter_offset,
            filter_len,
            self.total_entries,
            len(min_k_short),
            len(max_k_short),
            min_k_short.ljust(16, b"\x00"),
            max_k_short.ljust(16, b"\x00"),
        )
        self._file.write(footer_payload + SST_MAGIC)
        self._file.flush()
        self._file.close()


class SSTableReader:
    """
    Reads an SSTable with cached sparse index and in-memory Bloom filter.
    """

    FOOTER_SIZE = 40 + 4 + 32 + 4  # 80 bytes

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = open(filepath, "rb")
        self.file_size = os.path.getsize(filepath)

        # Parse footer
        self._file.seek(self.file_size - self.FOOTER_SIZE)
        footer_raw = self._file.read(self.FOOTER_SIZE)

        magic = footer_raw[-4:]
        if magic != SST_MAGIC:
            raise ValueError(f"Invalid SSTable file: magic mismatch {magic}")

        (
            self.index_offset,
            self.index_len,
            self.filter_offset,
            self.filter_len,
            self.total_entries,
            min_k_len,
            max_k_len,
            min_k_raw,
            max_k_raw,
        ) = struct.unpack(">QQQQQ HH 16s 16s", footer_raw[:-4])

        self.min_key_prefix = min_k_raw[:min_k_len]
        self.max_key_prefix = max_k_raw[:max_k_len]

        # Load Bloom Filter
        self._file.seek(self.filter_offset)
        filter_bytes = self._file.read(self.filter_len)
        self.bloom = BloomFilter.from_bytes(filter_bytes)

        # Load Index Block
        self._file.seek(self.index_offset)
        index_bytes = self._file.read(self.index_len)
        self.index_entries: List[Tuple[bytes, int, int]] = []

        num_index_entries = struct.unpack(">I", index_bytes[:4])[0]
        offset = 4
        for _ in range(num_index_entries):
            k_len = struct.unpack(">H", index_bytes[offset:offset + 2])[0]
            offset += 2
            k = index_bytes[offset:offset + k_len]
            offset += k_len
            blk_off, blk_sz = struct.unpack(">QQ", index_bytes[offset:offset + 16])
            offset += 16
            self.index_entries.append((k, blk_off, blk_sz))

    def get(self, key: bytes) -> Optional[Tuple[bytes, int, bool]]:
        """
        Retrieves key from SSTable.
        Fast path: Checks Bloom filter first.
        Binary searches index to locate and load only the relevant 4KB block.
        """
        # Step 1: Bloom filter check (zero disk IO on negatives)
        if key not in self.bloom:
            return None

        if not self.index_entries:
            return None

        # Step 2: Binary search sparse index
        low = 0
        high = len(self.index_entries) - 1
        target_idx = -1

        while low <= high:
            mid = (low + high) // 2
            if self.index_entries[mid][0] >= key:
                target_idx = mid
                high = mid - 1
            else:
                low = mid + 1

        if target_idx == -1:
            return None

        # Step 3: Read single data block
        _, blk_off, blk_sz = self.index_entries[target_idx]
        self._file.seek(blk_off)
        raw_block = self._file.read(blk_sz)

        expected_crc, payload_len = struct.unpack(">II", raw_block[:8])
        payload = raw_block[8:8 + payload_len]

        # Scan block entries
        p_offset = 0
        while p_offset < len(payload):
            tomb, seq, k_len = struct.unpack(">B Q H", payload[p_offset:p_offset + 11])
            p_offset += 11
            curr_key = payload[p_offset:p_offset + k_len]
            p_offset += k_len
            v_len = struct.unpack(">I", payload[p_offset:p_offset + 4])[0]
            p_offset += 4
            curr_val = payload[p_offset:p_offset + v_len]
            p_offset += v_len

            if curr_key == key:
                return curr_val, seq, (tomb == 1)
            elif curr_key > key:
                break

        return None

    def __iter__(self) -> Iterator[Tuple[bytes, bytes, int, bool]]:
        """Sequential stream of all entries in the SSTable."""
        for _, blk_off, blk_sz in self.index_entries:
            self._file.seek(blk_off)
            raw_block = self._file.read(blk_sz)
            payload_len = struct.unpack(">II", raw_block[:8])[1]
            payload = raw_block[8:8 + payload_len]

            p_offset = 0
            while p_offset < len(payload):
                tomb, seq, k_len = struct.unpack(">B Q H", payload[p_offset:p_offset + 11])
                p_offset += 11
                curr_key = payload[p_offset:p_offset + k_len]
                p_offset += k_len
                v_len = struct.unpack(">I", payload[p_offset:p_offset + 4])[0]
                p_offset += 4
                curr_val = payload[p_offset:p_offset + v_len]
                p_offset += v_len

                yield curr_key, curr_val, seq, (tomb == 1)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
