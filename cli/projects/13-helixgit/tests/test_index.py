"""
Unit tests for HelixGit Binary Index (DIRC v2) and Staging Area.
"""

import os
import shutil
import tempfile
import unittest
from helixgit.objects import Blob, Tree
from helixgit.storage import ObjectDatabase
from helixgit.index import Index, IndexEntry, DIRC_MAGIC, INDEX_VERSION


class TestBinaryIndex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.git_dir = os.path.join(self.temp_dir, ".git")
        self.objects_dir = os.path.join(self.git_dir, "objects")
        self.index_file = os.path.join(self.git_dir, "index")
        self.odb = ObjectDatabase(self.objects_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_index_entry_serialization_roundtrip(self):
        entry = IndexEntry(
            path="src/engine/physics.py",
            sha1="e" * 40,
            mode=0o100755,
            ctime_s=1620000000,
            ctime_ns=123456,
            mtime_s=1620000500,
            mtime_ns=654321,
            dev=42,
            ino=1001,
            uid=501,
            gid=20,
            file_size=4096,
            stage=2
        )
        data = entry.serialize()
        # Ensure data length is a multiple of 8
        self.assertEqual(len(data) % 8, 0)

        parsed_entry, next_offset = IndexEntry.deserialize(data, 0)
        self.assertEqual(next_offset, len(data))
        self.assertEqual(parsed_entry.path, "src/engine/physics.py")
        self.assertEqual(parsed_entry.sha1, "e" * 40)
        self.assertEqual(parsed_entry.mode, 0o100755)
        self.assertEqual(parsed_entry.ctime_s, 1620000000)
        self.assertEqual(parsed_entry.ctime_ns, 123456)
        self.assertEqual(parsed_entry.mtime_s, 1620000500)
        self.assertEqual(parsed_entry.mtime_ns, 654321)
        self.assertEqual(parsed_entry.dev, 42)
        self.assertEqual(parsed_entry.ino, 1001)
        self.assertEqual(parsed_entry.file_size, 4096)
        self.assertEqual(parsed_entry.stage, 2)

    def test_index_file_save_and_load(self):
        index = Index()
        index.add_entry(IndexEntry("README.md", "1" * 40, 0o100644, file_size=120))
        index.add_entry(IndexEntry("src/main.py", "2" * 40, 0o100755, file_size=2400))
        index.add_entry(IndexEntry("docs/guide.txt", "3" * 40, 0o100644, file_size=550))

        index.write(self.index_file)
        self.assertTrue(os.path.isfile(self.index_file))

        loaded_index = Index()
        loaded_index.read(self.index_file)

        self.assertEqual(len(loaded_index.entries), 3)
        self.assertIsNotNone(loaded_index.get_entry("README.md"))
        self.assertIsNotNone(loaded_index.get_entry("src/main.py"))
        self.assertIsNotNone(loaded_index.get_entry("docs/guide.txt"))
        self.assertEqual(loaded_index.get_entry("src/main.py").mode, 0o100755)

    def test_multi_stage_conflicts(self):
        index = Index()
        # Add normal staged entry
        index.add_entry(IndexEntry("clean.txt", "a" * 40, stage=0))
        self.assertFalse(index.has_conflicts())

        # Add conflict stages (1: ancestor, 2: ours, 3: theirs)
        index.add_entry(IndexEntry("conflict.py", "b" * 40, stage=1))
        index.add_entry(IndexEntry("conflict.py", "c" * 40, stage=2))
        index.add_entry(IndexEntry("conflict.py", "d" * 40, stage=3))

        self.assertTrue(index.has_conflicts())
        self.assertEqual(index.get_conflicted_paths(), {"conflict.py"})

        # Resolve conflict
        index.remove("conflict.py")
        index.add_entry(IndexEntry("conflict.py", "e" * 40, stage=0))
        self.assertFalse(index.has_conflicts())

    def test_to_tree_and_from_tree_checkout(self):
        # Create physical files in temp directory
        file1 = os.path.join(self.temp_dir, "hello.txt")
        file2 = os.path.join(self.temp_dir, "lib", "util.py")
        file3 = os.path.join(self.temp_dir, "lib", "algo", "sort.py")

        os.makedirs(os.path.dirname(file2), exist_ok=True)
        os.makedirs(os.path.dirname(file3), exist_ok=True)

        with open(file1, "wb") as f:
            f.write(b"Hello Git\n")
        with open(file2, "wb") as f:
            f.write(b"def helper(): pass\n")
        with open(file3, "wb") as f:
            f.write(b"def quicksort(): pass\n")

        index = Index()
        index.stage_file(self.temp_dir, "hello.txt", self.odb)
        index.stage_file(self.temp_dir, "lib/util.py", self.odb)
        index.stage_file(self.temp_dir, "lib/algo/sort.py", self.odb)

        root_tree_sha = index.to_tree(self.odb)
        self.assertTrue(self.odb.exists(root_tree_sha))

        root_tree = self.odb.get_tree(root_tree_sha)
        entry_names = [e.name for e in root_tree.entries]
        self.assertIn("hello.txt", entry_names)
        self.assertIn("lib", entry_names)

        # Re-checkout into fresh target directory
        checkout_dir = tempfile.mkdtemp()
        try:
            new_index = Index()
            new_index.from_tree(root_tree_sha, self.odb, checkout_dir)

            self.assertTrue(os.path.isfile(os.path.join(checkout_dir, "hello.txt")))
            self.assertTrue(os.path.isfile(os.path.join(checkout_dir, "lib", "util.py")))
            self.assertTrue(os.path.isfile(os.path.join(checkout_dir, "lib", "algo", "sort.py")))

            with open(os.path.join(checkout_dir, "lib", "algo", "sort.py"), "rb") as f:
                self.assertEqual(f.read(), b"def quicksort(): pass\n")

            # Check diff against clean worktree
            diff_clean = new_index.diff_worktree(checkout_dir)
            self.assertEqual(diff_clean["modified"], [])
            self.assertEqual(diff_clean["deleted"], [])
            self.assertEqual(diff_clean["untracked"], [])

            # Modify a file, add an untracked file, delete a file
            with open(os.path.join(checkout_dir, "hello.txt"), "wb") as f:
                f.write(b"Modified content!\n")

            with open(os.path.join(checkout_dir, "new_file.md"), "wb") as f:
                f.write(b"# Untracked\n")

            os.remove(os.path.join(checkout_dir, "lib", "util.py"))

            diff_dirty = new_index.diff_worktree(checkout_dir)
            self.assertEqual(diff_dirty["modified"], ["hello.txt"])
            self.assertEqual(diff_dirty["deleted"], ["lib/util.py"])
            self.assertEqual(diff_dirty["untracked"], ["new_file.md"])
        finally:
            shutil.rmtree(checkout_dir)


if __name__ == "__main__":
    unittest.main()
