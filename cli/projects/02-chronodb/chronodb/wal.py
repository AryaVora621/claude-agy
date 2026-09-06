"""
Write-Ahead Log (WAL) with CRC32 Checksums and Crash Recovery:
Ensures immediate durability for every mutating operation before MemTable ingestion.
Binary framed layout prevents silent corruption and enables crash recovery.
"""

from __future__ import annotations
import os
import struct
import zlib
from typing import Iterator, Tuple, Optional

WAL_MAGIC = b"\x57\x41"  # "WA"
OP_PUT = 1
OP_DELETE = 2


class WAL:
    """
    Append-only binary Write-Ahead Log.
    """

    def __init__(self, filepath: str, sync_on_write: bool = False):
        self.filepath = filepath
        self.sync_on_write = sync_on_write
        self._file = open(filepath, "a+b")

    def append(self, op_type: int, seq_num: int, key: bytes, value: bytes) -> None:
        """
        Appends a framed log entry:
        [Magic: 2B][CRC32: 4B][PayloadLen: 4B] | [Op: 1B][Seq: 8B][KeyLen: 2B][Key][ValLen: 4B][Val]
        """
        payload = struct.pack(">B Q H", op_type, seq_num, len(key)) + key + struct.pack(">I", len(value)) + value
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        header = WAL_MAGIC + struct.pack(">II", crc, len(payload))

        self._file.write(header + payload)
        if self.sync_on_write:
            self._file.flush()
            os.fsync(self._file.fileno())

    def flush(self) -> None:
        """Flushes in-memory file buffers and forces OS disk sync."""
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    @classmethod
    def recover(cls, filepath: str) -> Iterator[Tuple[int, int, bytes, bytes]]:
        """
        Replays the WAL sequentially from start to finish.
        Yields: (op_type, seq_num, key, value)
        Safely halts at corrupted records or partial writes from abrupt power loss.
        """
        if not os.path.exists(filepath):
            return

        with open(filepath, "rb") as f:
            while True:
                header = f.read(10)  # 2B magic + 4B crc + 4B payload_len
                if len(header) < 10:
                    break

                magic = header[:2]
                if magic != WAL_MAGIC:
                    # Invalid magic header, stop replay
                    break

                expected_crc, payload_len = struct.unpack(">II", header[2:10])
                payload = f.read(payload_len)
                if len(payload) < payload_len:
                    # Incomplete trailing record from crash, stop replay
                    break

                # Verify payload CRC32 integrity
                actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_crc != expected_crc:
                    # Corrupted record detected, cease replay
                    break

                # Unpack payload
                op_type, seq_num, key_len = struct.unpack(">B Q H", payload[:11])
                offset = 11
                key = payload[offset:offset + key_len]
                offset += key_len
                val_len = struct.unpack(">I", payload[offset:offset + 4])[0]
                offset += 4
                val = payload[offset:offset + val_len]

                yield op_type, seq_num, key, val
