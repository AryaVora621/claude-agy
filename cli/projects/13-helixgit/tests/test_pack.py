"""
Unit tests for HelixGit Packfile v2, Delta Compression, and .idx Fan-Out Engine.
"""

import unittest
from helixgit.objects import Blob, Tree, TreeEntry, Commit
from helixgit.pack import (
    create_delta, apply_delta, PackWriter, PackReader,
    OBJ_REF_DELTA, PackObjectInfo
)
from helixgit.idx import PackIndexWriter, PackIndexReader


class TestDeltaCompression(unittest.TestCase):
    def test_delta_roundtrip_simple(self):
        base = b"The quick brown fox jumps over the lazy dog."
        target = b"The quick brown red fox jumps over the extremely lazy dog!"

        delta = create_delta(base, target)
        reconstructed = apply_delta(base, delta)
        self.assertEqual(reconstructed, target)

    def test_delta_roundtrip_large_code(self):
        base = (
            b"def compute_fib(n):\n"
            b"    if n <= 1:\n"
            b"        return n\n"
            b"    return compute_fib(n - 1) + compute_fib(n - 2)\n"
        ) * 50

        # Change a few lines
        target = base.replace(b"compute_fib", b"optimized_fibonacci")
        delta = create_delta(base, target)
        reconstructed = apply_delta(base, delta)
        self.assertEqual(reconstructed, target)
        # Delta should be significantly smaller than target
        self.assertLess(len(delta), len(target))

    def test_delta_invalid_base_raises(self):
        base = b"original base string"
        target = b"target string"
        delta = create_delta(base, target)
        with self.assertRaises(ValueError):
            apply_delta(b"wrong base", delta)


class TestPackfileAndIndex(unittest.TestCase):
    def test_pack_write_and_read(self):
        blob1 = Blob(b"print('Hello, Packfile!')\n")
        blob2 = Blob(b"print('Second module')\n")
        tree = Tree([
            TreeEntry(0o100644, "hello.py", blob1.sha1()),
            TreeEntry(0o100644, "second.py", blob2.sha1())
        ])
        commit = Commit(
            tree=tree.sha1(),
            author_name="Linus Torvalds",
            author_email="torvalds@kernel.org",
            message="Add initial scripts\n"
        )

        writer = PackWriter()
        writer.add_object(blob1)
        writer.add_object(blob2)
        writer.add_object(tree)
        writer.add_object(commit)

        pack_bytes, object_infos = writer.write_pack()
        self.assertTrue(pack_bytes.startswith(b"PACK"))
        self.assertEqual(len(object_infos), 4)

        # Re-read with PackReader
        reader = PackReader(pack_bytes)
        self.assertIn(blob1.sha1(), reader.objects)
        self.assertIn(blob2.sha1(), reader.objects)
        self.assertIn(tree.sha1(), reader.objects)
        self.assertIn(commit.sha1(), reader.objects)

        read_blob1 = reader.objects[blob1.sha1()]
        self.assertEqual(read_blob1.serialize(), b"print('Hello, Packfile!')\n")

        read_commit = reader.objects[commit.sha1()]
        self.assertEqual(read_commit.tree, tree.sha1())

    def test_pack_index_v2_fanout_lookup(self):
        pack_sha1 = b"\xaa" * 20
        idx_writer = PackIndexWriter(pack_sha1)

        # Create dummy entries with various SHA-1s
        test_entries = [
            ("01" + "a" * 38, 12, 100),
            ("01" + "b" * 38, 54, 200),
            ("80" + "c" * 38, 128, 300),
            ("fe" + "d" * 38, 1024, 400),
            ("ff" + "e" * 38, 2048, 500),
        ]

        for sha1, off, crc in test_entries:
            idx_writer.add_entry(PackObjectInfo(
                sha1=sha1,
                type_code=3,
                offset=off,
                size=50,
                crc32=crc
            ))

        idx_bytes = idx_writer.write_index()
        self.assertTrue(idx_bytes.startswith(b"\xfftOc"))

        # Query with PackIndexReader
        reader = PackIndexReader(idx_bytes)
        self.assertEqual(reader.num_objects, len(test_entries))

        for sha1, expected_off, expected_crc in test_entries:
            res = reader.lookup(sha1)
            self.assertIsNotNone(res)
            offset, crc32 = res
            self.assertEqual(offset, expected_off)
            self.assertEqual(crc32, expected_crc)

        # Non-existent SHA returns None
        self.assertIsNone(reader.lookup("00" + "0" * 38))
        self.assertIsNone(reader.lookup("ff" + "0" * 38))


if __name__ == "__main__":
    unittest.main()
