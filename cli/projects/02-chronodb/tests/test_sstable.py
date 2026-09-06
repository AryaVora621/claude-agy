"""
Unit tests for SSTables and Compaction:
Verifies binary block formatting, sparse index binary searching,
Bloom filter disk filtering, and multi-way merge compaction.
"""

import os
import tempfile
import unittest
from chronodb.sstable import SSTableWriter, SSTableReader
from chronodb.compaction import Compactor


class TestSSTableAndCompaction(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_sstable_write_and_read(self):
        sst_path = os.path.join(self.tmp_dir.name, "00001.sst")
        writer = SSTableWriter(sst_path, expected_entries=500)

        # Write 500 ordered keys
        for i in range(500):
            k = f"key_{i:04d}".encode("utf-8")
            v = f"val_{i:04d}".encode("utf-8")
            writer.append(k, v, seq_num=i + 1, is_tombstone=False)

        writer.finish()

        # Read back
        reader = SSTableReader(sst_path)
        self.assertEqual(reader.total_entries, 500)

        # Test point queries
        res_0 = reader.get(b"key_0000")
        self.assertIsNotNone(res_0)
        self.assertEqual(res_0[0], b"val_0000")
        self.assertEqual(res_0[1], 1)

        res_250 = reader.get(b"key_0250")
        self.assertIsNotNone(res_250)
        self.assertEqual(res_250[0], b"val_0250")

        res_499 = reader.get(b"key_0499")
        self.assertIsNotNone(res_499)
        self.assertEqual(res_499[0], b"val_0499")

        # Test non-existent key
        res_missing = reader.get(b"key_9999")
        self.assertIsNone(res_missing)

        reader.close()

    def test_compaction_multiway_merge(self):
        # Create Table 1: keys A, B, C with older sequence numbers
        t1_path = os.path.join(self.tmp_dir.name, "t1.sst")
        w1 = SSTableWriter(t1_path, expected_entries=10)
        w1.append(b"apple", b"v1_apple", seq_num=1)
        w1.append(b"banana", b"v1_banana", seq_num=2)
        w1.append(b"cherry", b"v1_cherry", seq_num=3)
        w1.finish()

        # Create Table 2: keys B (updated), D (new) with newer sequence numbers
        t2_path = os.path.join(self.tmp_dir.name, "t2.sst")
        w2 = SSTableWriter(t2_path, expected_entries=10)
        w2.append(b"banana", b"v2_banana", seq_num=10)
        w2.append(b"date", b"v1_date", seq_num=11)
        w2.finish()

        # Compact Table 1 and Table 2 into Table 3
        t3_path = os.path.join(self.tmp_dir.name, "t3.sst")
        r1 = SSTableReader(t1_path)
        r2 = SSTableReader(t2_path)

        compactor = Compactor(self.tmp_dir.name)
        compactor.merge_tables([r1, r2], t3_path, is_bottom_level=True)

        r1.close()
        r2.close()

        # Verify merged Table 3
        r3 = SSTableReader(t3_path)
        # Should contain: apple (v1), banana (v2!), cherry (v1), date (v1)
        entries = list(r3)
        self.assertEqual(len(entries), 4)

        k_map = {k: (v, seq) for k, v, seq, tomb in entries}
        self.assertEqual(k_map[b"apple"][0], b"v1_apple")
        self.assertEqual(k_map[b"banana"][0], b"v2_banana")  # Newer seq won!
        self.assertEqual(k_map[b"cherry"][0], b"v1_cherry")
        self.assertEqual(k_map[b"date"][0], b"v1_date")

        r3.close()


if __name__ == "__main__":
    unittest.main()
