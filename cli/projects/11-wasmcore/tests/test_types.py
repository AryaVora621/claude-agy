"""
Unit tests for WasmCore types, values, and bitcasts.
"""

import unittest
from wasmcore.types import (
    ValType, FuncType, Limits, Value, WasmTrap,
    to_signed_32, to_unsigned_32, to_signed_64, to_unsigned_64,
    float_to_i32_bits, i32_bits_to_float, double_to_i64_bits, i64_bits_to_double
)


class TestTypes(unittest.TestCase):
    def test_val_type_string(self):
        self.assertEqual(str(ValType.I32), "i32")
        self.assertEqual(str(ValType.I64), "i64")
        self.assertEqual(str(ValType.F32), "f32")
        self.assertEqual(str(ValType.F64), "f64")

    def test_func_type_equality(self):
        ft1 = FuncType([ValType.I32, ValType.I32], [ValType.I32])
        ft2 = FuncType([ValType.I32, ValType.I32], [ValType.I32])
        ft3 = FuncType([ValType.I32], [ValType.I32])
        self.assertEqual(ft1, ft2)
        self.assertNotEqual(ft1, ft3)
        self.assertEqual(hash(ft1), hash(ft2))

    def test_two_complement_conversions(self):
        self.assertEqual(to_signed_32(0xFFFFFFFF), -1)
        self.assertEqual(to_unsigned_32(-1), 0xFFFFFFFF)
        self.assertEqual(to_signed_64(0xFFFFFFFFFFFFFFFF), -1)
        self.assertEqual(to_unsigned_64(-1), 0xFFFFFFFFFFFFFFFF)

    def test_float_bitcasts(self):
        f = 3.1415927410125732  # 32-bit float
        bits = float_to_i32_bits(f)
        f_recovered = i32_bits_to_float(bits)
        self.assertAlmostEqual(f, f_recovered, places=6)

        d = 2.718281828459045
        dbits = double_to_i64_bits(d)
        d_recovered = i64_bits_to_double(dbits)
        self.assertEqual(d, d_recovered)

    def test_value_box(self):
        v_i32 = Value.i32(42)
        self.assertEqual(v_i32.as_i32(), 42)
        self.assertEqual(v_i32.as_u32(), 42)

        v_neg = Value.i32(-1)
        self.assertEqual(v_neg.as_i32(), -1)
        self.assertEqual(v_neg.as_u32(), 0xFFFFFFFF)

        v_f64 = Value.f64(12.34)
        self.assertEqual(v_f64.as_f64(), 12.34)

        with self.assertRaises(WasmTrap):
            v_i32.as_f32()


if __name__ == "__main__":
    unittest.main()
