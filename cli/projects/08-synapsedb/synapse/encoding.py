"""
SynapseDB: Columnar Compression Codecs and Zone Map Statistics.
Implements:
1. Zone Maps (Min/Max/Null-count statistics for query predicate pruning).
2. Run-Length Encoding (RLE) for repeated consecutive values.
3. Dictionary Encoding with compact integer indices for low-cardinality strings.
4. Frame-of-Reference (FoR) Bit-Packing for compact integer storage.
"""

import struct
import math
from typing import List, Tuple, Any, Optional, Sequence
from synapse.types import DataType, Vector


class ZoneMap:
    """
    Column-chunk statistical summary used for Parquet-style predicate pushdown.
    Allows skipping entire chunks of data when predicates cannot possibly match.
    """
    __slots__ = ("min_val", "max_val", "null_count", "row_count", "data_type")

    def __init__(
        self,
        min_val: Any,
        max_val: Any,
        null_count: int,
        row_count: int,
        data_type: DataType
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.null_count = null_count
        self.row_count = row_count
        self.data_type = data_type

    @classmethod
    def compute(cls, vec: Vector) -> "ZoneMap":
        valid_vals = [vec.get(i) for i in range(len(vec)) if not vec.is_null(i)]
        null_count = len(vec) - len(valid_vals)
        row_count = len(vec)

        if not valid_vals:
            return cls(None, None, null_count, row_count, vec.data_type)

        min_val = min(valid_vals)
        max_val = max(valid_vals)
        return cls(min_val, max_val, null_count, row_count, vec.data_type)

    def can_match(self, op: str, value: Any) -> bool:
        """
        Evaluate if a predicate (e.g. col > 100, col = 'NYC') could match any row.
        Returns False if the entire chunk can safely be pruned from disk/scan.
        """
        if self.min_val is None or self.max_val is None:
            # Chunk contains only nulls
            if op == "IS_NULL":
                return self.null_count > 0
            return False

        if op == "=":
            return self.min_val <= value <= self.max_val
        elif op == "!=":
            return not (self.min_val == self.max_val == value)
        elif op == ">":
            return self.max_val > value
        elif op == ">=":
            return self.max_val >= value
        elif op == "<":
            return self.min_val < value
        elif op == "<=":
            return self.min_val <= value
        elif op == "IS_NULL":
            return self.null_count > 0
        elif op == "IS_NOT_NULL":
            return (self.row_count - self.null_count) > 0

        return True  # Default conservative fallback


class RLECodec:
    """
    Run-Length Encoding (RLE) Codec.
    Packs consecutive identical values into (run_count: uint32, value) tuples.
    """

    @staticmethod
    def encode(values: Sequence[Any], data_type: DataType) -> bytes:
        if not values:
            return struct.pack("!I", 0)

        runs: List[Tuple[int, Any]] = []
        current_val = values[0]
        current_count = 1

        for v in values[1:]:
            if v == current_val:
                current_count += 1
            else:
                runs.append((current_count, current_val))
                current_val = v
                current_count = 1
        runs.append((current_count, current_val))

        buf = bytearray()
        # Header: number of runs
        buf.extend(struct.pack("!I", len(runs)))

        for count, val in runs:
            buf.extend(struct.pack("!I", count))
            if data_type == DataType.INT64:
                buf.extend(struct.pack("!q", val if val is not None else 0))
            elif data_type == DataType.FLOAT64:
                buf.extend(struct.pack("!d", val if val is not None else 0.0))
            elif data_type == DataType.BOOLEAN:
                buf.extend(struct.pack("!B", 1 if val else 0))
            else:
                s_bytes = str(val if val is not None else "").encode("utf-8")
                buf.extend(struct.pack("!H", len(s_bytes)))
                buf.extend(s_bytes)

        return bytes(buf)

    @staticmethod
    def decode(raw: bytes, data_type: DataType) -> List[Any]:
        if len(raw) < 4:
            return []
        num_runs = struct.unpack("!I", raw[:4])[0]
        offset = 4
        result = []

        for _ in range(num_runs):
            count = struct.unpack("!I", raw[offset : offset + 4])[0]
            offset += 4

            if data_type == DataType.INT64:
                val = struct.unpack("!q", raw[offset : offset + 8])[0]
                offset += 8
            elif data_type == DataType.FLOAT64:
                val = struct.unpack("!d", raw[offset : offset + 8])[0]
                offset += 8
            elif data_type == DataType.BOOLEAN:
                val = bool(struct.unpack("!B", raw[offset : offset + 1])[0])
                offset += 1
            else:
                s_len = struct.unpack("!H", raw[offset : offset + 2])[0]
                offset += 2
                val = raw[offset : offset + s_len].decode("utf-8")
                offset += s_len

            result.extend([val] * count)

        return result


class DictionaryCodec:
    """
    Dictionary Encoding for string/varchar columns.
    Maps distinct strings to integer symbols (1-byte uint8 if <= 256 distinct, uint16 if <= 65536).
    """

    @staticmethod
    def encode(values: Sequence[str]) -> Tuple[List[str], bytes]:
        unique_terms: List[str] = []
        term_to_id = {}

        for v in values:
            v_str = "" if v is None else str(v)
            if v_str not in term_to_id:
                term_to_id[v_str] = len(unique_terms)
                unique_terms.append(v_str)

        num_terms = len(unique_terms)
        buf = bytearray()
        # Format: 1 byte width flag (1 = uint8, 2 = uint16, 4 = uint32), followed by packed codes
        if num_terms <= 256:
            buf.append(1)
            for v in values:
                v_str = "" if v is None else str(v)
                buf.append(term_to_id[v_str])
        elif num_terms <= 65536:
            buf.append(2)
            for v in values:
                v_str = "" if v is None else str(v)
                buf.extend(struct.pack("!H", term_to_id[v_str]))
        else:
            buf.append(4)
            for v in values:
                v_str = "" if v is None else str(v)
                buf.extend(struct.pack("!I", term_to_id[v_str]))

        return unique_terms, bytes(buf)

    @staticmethod
    def decode(dictionary: List[str], raw_codes: bytes) -> List[str]:
        if not raw_codes:
            return []
        width = raw_codes[0]
        data = raw_codes[1:]
        result = []

        if width == 1:
            for byte_val in data:
                result.append(dictionary[byte_val])
        elif width == 2:
            num_entries = len(data) // 2
            for i in range(num_entries):
                code = struct.unpack("!H", data[i * 2 : (i + 1) * 2])[0]
                result.append(dictionary[code])
        else:
            num_entries = len(data) // 4
            for i in range(num_entries):
                code = struct.unpack("!I", data[i * 4 : (i + 1) * 4])[0]
                result.append(dictionary[code])

        return result


class BitPackingCodec:
    """
    Frame-of-Reference (FoR) Bit-Packing for 64-bit integer columns.
    Subtracts minimum value and packs deltas using the minimum necessary bit-width.
    """

    @staticmethod
    def encode(integers: Sequence[int]) -> bytes:
        if not integers:
            return struct.pack("!qBI", 0, 0, 0)

        min_val = min(integers)
        max_val = max(integers)
        delta_max = max_val - min_val

        # Determine minimum bit-width needed
        if delta_max == 0:
            bit_width = 1
        else:
            bit_width = max(1, math.ceil(math.log2(delta_max + 1)))

        buf = bytearray()
        # Header: base_val (8 bytes), bit_width (1 byte), count (4 bytes)
        buf.extend(struct.pack("!qBI", min_val, bit_width, len(integers)))

        # Pack bits into byte stream
        bit_buffer = 0
        bits_in_buffer = 0

        for val in integers:
            delta = val - min_val
            bit_buffer = (bit_buffer << bit_width) | delta
            bits_in_buffer += bit_width

            while bits_in_buffer >= 8:
                shift = bits_in_buffer - 8
                byte_out = (bit_buffer >> shift) & 0xFF
                buf.append(byte_out)
                bits_in_buffer -= 8
                bit_buffer &= (1 << bits_in_buffer) - 1

        if bits_in_buffer > 0:
            byte_out = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            buf.append(byte_out)

        return bytes(buf)

    @staticmethod
    def decode(raw: bytes) -> List[int]:
        if len(raw) < 13:
            return []

        base_val, bit_width, count = struct.unpack("!qBI", raw[:13])
        if count == 0:
            return []

        data = raw[13:]
        result = []
        bit_buffer = 0
        bits_in_buffer = 0
        byte_idx = 0
        mask = (1 << bit_width) - 1

        for _ in range(count):
            while bits_in_buffer < bit_width and byte_idx < len(data):
                bit_buffer = (bit_buffer << 8) | data[byte_idx]
                bits_in_buffer += 8
                byte_idx += 1

            shift = bits_in_buffer - bit_width
            delta = (bit_buffer >> shift) & mask
            bits_in_buffer -= bit_width
            bit_buffer &= (1 << bits_in_buffer) - 1
            result.append(base_val + delta)

        return result
