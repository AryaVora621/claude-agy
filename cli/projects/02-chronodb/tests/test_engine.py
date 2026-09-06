"""
End-to-End Integration and Durability Tests for ChronoDB:
Tests KV operations, flush thresholds, leveled compaction,
crash recovery from WAL, and hybrid vector search.
"""

import os
import shutil
import tempfile
import unittest
from chronodb.engine import ChronoDB


class TestChronoDBEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="chronodb_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_crud_and_flush_and_compaction(self):
        # Set low memtable limit to force multiple flushes and compaction
        db = ChronoDB(
            data_dir=self.test_dir,
            memtable_limit_bytes=2048,  # 2 KB threshold
            max_l0_tables=3,
        )

        # Write 200 keys
        for i in range(200):
            db.put(f"user:{i}", f"profile_data_payload_for_user_{i}")

        # Verify all keys exist
        for i in range(200):
            val = db.get(f"user:{i}")
            self.assertIsNotNone(val, f"Failed to get user:{i}")
            self.assertEqual(val.decode("utf-8"), f"profile_data_payload_for_user_{i}")

        # Check diagnostics: should have flushed multiple times and compacted
        diag = db.get_diagnostics()
        self.assertGreater(diag["stats"]["flushes"], 0)
        self.assertGreater(diag["stats"]["compactions"], 0)

        # Delete 50 keys
        for i in range(50):
            db.delete(f"user:{i}")

        for i in range(50):
            self.assertIsNone(db.get(f"user:{i}"))

        for i in range(50, 200):
            self.assertIsNotNone(db.get(f"user:{i}"))

        db.close()

    def test_crash_recovery(self):
        """
        Simulates an abrupt process crash where in-memory memtable was NOT flushed to SSTable.
        Verifies that re-opening the database restores un-flushed keys via WAL replay.
        """
        db = ChronoDB(
            data_dir=self.test_dir,
            memtable_limit_bytes=1024 * 1024,  # High limit so no flush happens
        )
        db.put("unflushed_key_1", "alpha")
        db.put("unflushed_key_2", "beta")
        db.put("unflushed_key_3", "gamma")
        db.delete("unflushed_key_2")

        # Simulate crash: do NOT call db.flush() or clean shutdown
        # WAL is synced on write, so file handles close abruptly
        db.wal.flush()
        db.wal._file.close()

        # Re-open database from same directory
        recovered_db = ChronoDB(
            data_dir=self.test_dir,
            memtable_limit_bytes=1024 * 1024,
        )

        self.assertEqual(recovered_db.get("unflushed_key_1"), b"alpha")
        self.assertIsNone(recovered_db.get("unflushed_key_2"))  # Tombstone recovered
        self.assertEqual(recovered_db.get("unflushed_key_3"), b"gamma")

        recovered_db.close()

    def test_hybrid_vector_search(self):
        db = ChronoDB(
            data_dir=self.test_dir,
            vector_dim=4,
            vector_metric="cosine",
        )

        # Insert items with vectors and rich metadata
        db.put_vector("doc:1", [1.0, 0.0, 0.0, 0.0], metadata={"title": "Introduction to AI", "category": "ml"})
        db.put_vector("doc:2", [0.9, 0.1, 0.0, 0.0], metadata={"title": "Deep Learning Systems", "category": "ml"})
        db.put_vector("doc:3", [0.0, 0.0, 1.0, 0.0], metadata={"title": "Distributed Databases", "category": "db"})

        # Query vector close to doc:1 and doc:2
        results = db.search_vector([1.0, 0.05, 0.0, 0.0], k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["key"], "doc:1")
        self.assertEqual(results[0]["metadata"]["category"], "ml")
        self.assertEqual(results[1]["key"], "doc:2")

        db.close()


if __name__ == "__main__":
    unittest.main()
