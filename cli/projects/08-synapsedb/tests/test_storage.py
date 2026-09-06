"""
SynapseDB: Unit Tests for Columnar Storage, File Persistence, and Pruning.
"""

import unittest
import tempfile
import os
from synapse.types import DataType, Vector, RecordBatch
from synapse.storage import ColumnarTable, TableSchema


class TestSynapseStorage(unittest.TestCase):
    def setUp(self):
        self.schema = TableSchema({
            "id": DataType.INT64,
            "val": DataType.FLOAT64,
            "category": DataType.VARCHAR
        })
        self.table = ColumnarTable("items", self.schema, chunk_size=3)

    def test_insert_and_chunking(self):
        rows = [
            {"id": 1, "val": 10.5, "category": "A"},
            {"id": 2, "val": 20.0, "category": "B"},
            {"id": 3, "val": 30.2, "category": "A"},
            {"id": 4, "val": 40.1, "category": "C"},
            {"id": 5, "val": 50.9, "category": "B"}
        ]
        self.table.insert_rows(rows)
        self.assertEqual(self.table.total_rows, 5)
        self.assertEqual(len(self.table.row_groups), 2)  # chunk_size=3 -> 2 chunks

    def test_zone_map_pruning_scan(self):
        # Chunk 1: IDs 1..3
        # Chunk 2: IDs 4..5
        rows = [
            {"id": 1, "val": 10.5, "category": "A"},
            {"id": 2, "val": 20.0, "category": "B"},
            {"id": 3, "val": 30.2, "category": "A"},
            {"id": 4, "val": 40.1, "category": "C"},
            {"id": 5, "val": 50.9, "category": "B"}
        ]
        self.table.insert_rows(rows)

        # Scan for id > 3 -> chunk 1 should be completely pruned!
        scanned_batches = list(self.table.scan(predicates=[("id", ">", 3)]))
        self.assertEqual(len(scanned_batches), 1)  # Only chunk 2 returned
        self.assertEqual(scanned_batches[0].column("id").to_list(), [4, 5])

    def test_projection_pruning(self):
        rows = [{"id": 1, "val": 1.0, "category": "X"}]
        self.table.insert_rows(rows)

        batches = list(self.table.scan(projection=["category"]))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].names, ["category"])

    def test_file_save_and_load(self):
        rows = [
            {"id": 10, "val": 100.5, "category": "alpha"},
            {"id": 20, "val": 200.5, "category": "beta"},
            {"id": 30, "val": 300.5, "category": "gamma"}
        ]
        self.table.insert_rows(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.synapse")
            self.table.save_to_file(file_path)

            # Load back
            loaded = ColumnarTable.load_from_file(file_path)
            self.assertEqual(loaded.name, "items")
            self.assertEqual(loaded.total_rows, 3)
            self.assertEqual(len(loaded.row_groups), 1)

            batches = list(loaded.scan())
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0].column("id").to_list(), [10, 20, 30])
            self.assertEqual(batches[0].column("category").to_list(), ["alpha", "beta", "gamma"])


if __name__ == "__main__":
    unittest.main()
