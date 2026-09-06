"""
SynapseDB: Unit Tests for Columnar Types, Vectors, Bitmaps, and RecordBatches.
"""

import unittest
from synapse.types import (
    DataType,
    Vector,
    RecordBatch,
    SelectionVector,
    NullBitmap
)


class TestSynapseTypes(unittest.TestCase):
    def test_null_bitmap(self):
        bm = NullBitmap(10, all_valid=True)
        for i in range(10):
            self.assertTrue(bm.is_valid(i))

        bm.set_valid(3, False)
        bm.set_valid(7, False)
        self.assertFalse(bm.is_valid(3))
        self.assertFalse(bm.is_valid(7))
        self.assertTrue(bm.is_valid(4))

    def test_primitive_vectors(self):
        # INT64
        int_vec = Vector(DataType.INT64, [10, 20, None, 40])
        self.assertEqual(len(int_vec), 4)
        self.assertEqual(int_vec.get(0), 10)
        self.assertEqual(int_vec.get(1), 20)
        self.assertIsNone(int_vec.get(2))
        self.assertEqual(int_vec.get(3), 40)

        # FLOAT64
        float_vec = Vector(DataType.FLOAT64, [1.5, None, 3.5])
        self.assertEqual(float_vec.get(0), 1.5)
        self.assertIsNone(float_vec.get(1))
        self.assertEqual(float_vec.get(2), 3.5)

        # BOOLEAN
        bool_vec = Vector(DataType.BOOLEAN, [True, False, None, True])
        self.assertTrue(bool_vec.get(0))
        self.assertFalse(bool_vec.get(1))
        self.assertIsNone(bool_vec.get(2))
        self.assertTrue(bool_vec.get(3))

        # VARCHAR
        str_vec = Vector(DataType.VARCHAR, ["apple", "banana", None, "cherry"])
        self.assertEqual(str_vec.get(0), "apple")
        self.assertIsNone(str_vec.get(2))
        self.assertEqual(str_vec.get(3), "cherry")

    def test_selection_vector_and_filtering(self):
        vec = Vector(DataType.INT64, [100, 200, 300, 400, 500])
        sel = SelectionVector([1, 3])
        filtered = vec.filter(sel)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.to_list(), [200, 400])

    def test_record_batch_operations(self):
        v1 = Vector(DataType.INT64, [1, 2, 3, 4, 5])
        v2 = Vector(DataType.VARCHAR, ["a", "b", "c", "d", "e"])
        batch = RecordBatch({"id": v1, "name": v2})

        self.assertEqual(len(batch), 5)
        self.assertEqual(batch.names, ["id", "name"])

        # Slicing
        sliced = batch.slice(1, 3)
        self.assertEqual(len(sliced), 3)
        self.assertEqual(sliced.column("id").to_list(), [2, 3, 4])

        # Projection
        proj = batch.project(["name"])
        self.assertEqual(proj.names, ["name"])
        self.assertEqual(len(proj), 5)

        # Filtering
        sel = SelectionVector([0, 4])
        f_batch = batch.filter(sel)
        self.assertEqual(len(f_batch), 2)
        self.assertEqual(f_batch.column("id").to_list(), [1, 5])
        self.assertEqual(f_batch.column("name").to_list(), ["a", "e"])


if __name__ == "__main__":
    unittest.main()
