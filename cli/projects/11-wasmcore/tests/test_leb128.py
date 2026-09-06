"""
Unit tests for WasmCore LEB128 variable-length integer codecs.
"""

import unittest
import io
from wasmcore.leb128 import (
    decode_u32, decode_i32, decode_u64, decode_i64,
    encode_u32, encode_i32, encode_u64, encode_i64,
    decode_name, encode_name, decode_vec, encode_vec
)
from wasmcore.types import WasmValidationError


class TestLEB128(unittest.TestCase):
    def test_u32_roundtrip(self):
        cases = [0, 1, 42, 127, 128, 255, 256, 16383, 16384, 65535, 65536, 0xFFFFFFFF]
        for val in cases:
            enc = encode_u32(val)
            dec = decode_u32(enc)
            self.assertEqual(dec, val, f"Failed for u32 {val}")

    def test_i32_roundtrip(self):
        cases = [0, 1, -1, 42, -42, 63, 64, -64, -65, 127, -128, 128, 8191, -8192,
                 0x7FFFFFFF, -0x80000000]
        for val in cases:
            enc = encode_i32(val)
            dec = decode_i32(enc)
            self.assertEqual(dec, val, f"Failed for i32 {val}")

    def test_u64_roundtrip(self):
        cases = [0, 1, 128, 0xFFFFFFFF, 0x100000000, 0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
        for val in cases:
            enc = encode_u64(val)
            dec = decode_u64(enc)
            self.assertEqual(dec, val, f"Failed for u64 {val}")

    def test_i64_roundtrip(self):
        cases = [0, 1, -1, 100, -100, 0x7FFFFFFFFFFFFFFF, -0x8000000000000000]
        for val in cases:
            enc = encode_i64(val)
            dec = decode_i64(enc)
            self.assertEqual(dec, val, f"Failed for i64 {val}")

    def test_name_and_vec_roundtrip(self):
        name = "wasm_core_42_🚀"
        enc = encode_name(name)
        stream = io.BytesIO(enc)
        dec = decode_name(stream)
        self.assertEqual(dec, name)

        items = [10, 20, 30, 40]
        vec_bytes = encode_vec(items, encode_u32)
        stream_vec = io.BytesIO(vec_bytes)
        dec_items = decode_vec(stream_vec, decode_u32)
        self.assertEqual(dec_items, items)

    def test_overflow_detection(self):
        # 6 bytes with MSB set causes overflow in 32-bit LEB128
        bad_stream = io.BytesIO(b"\x80\x80\x80\x80\x80\x80\x00")
        with self.assertRaises(WasmValidationError):
            decode_u32(bad_stream)


if __name__ == "__main__":
    unittest.main()
