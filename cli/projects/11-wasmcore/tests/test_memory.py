"""
Unit tests for WasmCore Linear Memory subsystem:
64KB paging, bounds checks, dynamic grow, and integer/float load/store operations.
"""

import unittest
from wasmcore.types import Limits, WasmTrap
from wasmcore.memory import LinearMemory, PAGE_SIZE


class TestLinearMemory(unittest.TestCase):
    def test_initial_allocation_and_paging(self):
        mem = LinearMemory(Limits(min=2, max=4))
        self.assertEqual(mem.pages, 2)
        self.assertEqual(mem.size(), 2)
        self.assertEqual(len(mem.data), 2 * PAGE_SIZE)

    def test_grow_memory(self):
        mem = LinearMemory(Limits(min=1, max=3))
        # Grow by 1 page: returns previous page count 1
        old = mem.grow(1)
        self.assertEqual(old, 1)
        self.assertEqual(mem.pages, 2)

        # Grow by 1 more page: reaches max limit 3
        old2 = mem.grow(1)
        self.assertEqual(old2, 2)
        self.assertEqual(mem.pages, 3)

        # Attempt to grow beyond max limit: returns -1
        fail = mem.grow(1)
        self.assertEqual(fail, -1)
        self.assertEqual(mem.pages, 3)

        # Zero delta returns current page count
        self.assertEqual(mem.grow(0), 3)

        # Negative delta returns -1
        self.assertEqual(mem.grow(-1), -1)

    def test_raw_bytes_read_write(self):
        mem = LinearMemory(Limits(min=1))
        payload = b"WebAssembly MVP Linear Memory"
        mem.write_bytes(100, payload)
        self.assertEqual(mem.read_bytes(100, len(payload)), payload)

        # Out of bounds trap
        with self.assertRaises(WasmTrap):
            mem.read_bytes(PAGE_SIZE - 5, 10)

        with self.assertRaises(WasmTrap):
            mem.write_bytes(PAGE_SIZE - 2, b"ABCD")

    def test_i32_loads_and_stores(self):
        mem = LinearMemory(Limits(min=1))
        mem.store_i32(0, 42)
        self.assertEqual(mem.load_i32(0), 42)

        mem.store_i32(4, -100)
        self.assertEqual(mem.load_i32(4), -100)
        self.assertEqual(mem.load_u32(4), 0xFFFFFF9C)

    def test_sub_word_loads_and_stores(self):
        mem = LinearMemory(Limits(min=1))
        # i8
        mem.store_i8(10, 0xFE)
        self.assertEqual(mem.load_i8_s(10), -2)
        self.assertEqual(mem.load_i8_u(10), 254)

        # i16
        mem.store_i16(20, 0xFFF0)
        self.assertEqual(mem.load_i16_s(20), -16)
        self.assertEqual(mem.load_i16_u(20), 0xFFF0)

    def test_i64_and_float_operations(self):
        mem = LinearMemory(Limits(min=1))
        # i64
        mem.store_i64(30, -1234567890123456)
        self.assertEqual(mem.load_i64(30), -1234567890123456)

        # f32
        mem.store_f32(40, 3.1415927)
        self.assertAlmostEqual(mem.load_f32(40), 3.1415927, places=5)

        # f64
        mem.store_f64(50, 2.718281828459045)
        self.assertAlmostEqual(mem.load_f64(50), 2.718281828459045, places=9)


if __name__ == "__main__":
    unittest.main()
