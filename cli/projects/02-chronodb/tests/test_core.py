"""
Unit tests for ChronoDB core: Bloom Filter, SkipList MemTable, and WAL crash recovery.
"""

import os
import tempfile
import unittest
from chronodb.bloom import BloomFilter
from chronodb.skiplist import SkipList
from chronodb.wal import WAL, OP_PUT, OP_DELETE


class TestChronoDBCore(unittest.TestCase):

    def test_bloom_filter_accuracy(self):
        bf = BloomFilter.create(expected_items=1000, false_positive_rate=0.01)
        # Insert 1000 keys
        for i in range(1000):
            bf.add(f"user_session_{i}")

        # Zero false negatives
        for i in range(1000):
            self.assertIn(f"user_session_{i}", bf)

        # False positives bounded
        false_positives = 0
        trials = 1000
        for i in range(trials):
            if f"non_existent_{i}" in bf:
                false_positives += 1

        fp_rate = false_positives / trials
        self.assertLess(fp_rate, 0.05, f"False positive rate {fp_rate} exceeded acceptable threshold")

    def test_bloom_filter_serialization(self):
        bf = BloomFilter.create(expected_items=100, false_positive_rate=0.01)
        bf.add("test_key_1")
        bf.add("test_key_2")

        data = bf.to_bytes()
        restored = BloomFilter.from_bytes(data)

        self.assertIn("test_key_1", restored)
        self.assertIn("test_key_2", restored)
        self.assertNotIn("test_key_3", restored)

    def test_skiplist_operations(self):
        sl = SkipList()
        # Insert out of order
        sl.put(b"zebra", b"striped", seq_num=1)
        sl.put(b"apple", b"red", seq_num=2)
        sl.put(b"mango", b"sweet", seq_num=3)

        # Point lookups
        res = sl.get(b"apple")
        self.assertIsNotNone(res)
        self.assertEqual(res[0], b"red")
        self.assertEqual(res[1], 2)

        # Non-existent
        self.assertIsNone(sl.get(b"banana"))

        # Update
        sl.put(b"apple", b"green", seq_num=4)
        self.assertEqual(sl.get(b"apple")[0], b"green")

        # Ordered iteration
        keys = [item[0] for item in sl]
        self.assertEqual(keys, [b"apple", b"mango", b"zebra"])

        # Range scan [b"b", b"p")
        scan_keys = [item[0] for item in sl.scan(start_key=b"b", end_key=b"p")]
        self.assertEqual(scan_keys, [b"mango"])

    def test_wal_durability_and_crash_recovery(self):
        with tempfile.NamedTemporaryFile(suffix=".wal", delete=False) as tmp:
            wal_path = tmp.name

        try:
            wal = WAL(wal_path, sync_on_write=True)
            wal.append(OP_PUT, seq_num=1, key=b"k1", value=b"v1")
            wal.append(OP_PUT, seq_num=2, key=b"k2", value=b"v2")
            wal.append(OP_DELETE, seq_num=3, key=b"k1", value=b"")
            wal.close()

            # Corrupt the file by appending junk at the tail (simulating torn write on power cut)
            with open(wal_path, "ab") as f:
                f.write(b"\x57\x41_INCOMPLETE_TORN_FRAME_DATA")

            # Replay and verify recovery
            recovered = list(WAL.recover(wal_path))
            self.assertEqual(len(recovered), 3)

            op1, seq1, k1, v1 = recovered[0]
            self.assertEqual(op1, OP_PUT)
            self.assertEqual(seq1, 1)
            self.assertEqual(k1, b"k1")
            self.assertEqual(v1, b"v1")

            op3, seq3, k3, v3 = recovered[2]
            self.assertEqual(op3, OP_DELETE)
            self.assertEqual(seq3, 3)
            self.assertEqual(k3, b"k1")
        finally:
            if os.path.exists(wal_path):
                os.remove(wal_path)


if __name__ == "__main__":
    unittest.main()
