"""
Unit tests for HelixGit Three-Way Merge Engine, LCA Traversal, and Conflict Resolution.
"""

import os
import shutil
import tempfile
import unittest
from helixgit.objects import Blob, Tree, TreeEntry, Commit
from helixgit.storage import ObjectDatabase
from helixgit.merge import (
    find_lowest_common_ancestor,
    three_way_merge_lines,
    three_way_merge_trees
)


class TestThreeWayMerge(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.odb = ObjectDatabase(os.path.join(self.temp_dir, "objects"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_find_lowest_common_ancestor(self):
        # Build DAG:
        #       c0 (root)
        #       /  \
        #      c1   c2
        #      |    |
        #      c3   c4
        c0 = self.odb.write_commit(tree="0" * 40, parents=[], message="root\n")
        c1 = self.odb.write_commit(tree="1" * 40, parents=[c0], message="c1\n")
        c2 = self.odb.write_commit(tree="2" * 40, parents=[c0], message="c2\n")
        c3 = self.odb.write_commit(tree="3" * 40, parents=[c1], message="c3\n")
        c4 = self.odb.write_commit(tree="4" * 40, parents=[c2], message="c4\n")

        lca = find_lowest_common_ancestor(c3, c4, self.odb)
        self.assertEqual(lca, c0)

        # Direct ancestor
        lca_direct = find_lowest_common_ancestor(c1, c3, self.odb)
        self.assertEqual(lca_direct, c1)

    def test_three_way_merge_lines_clean(self):
        base = ["int a = 1;", "int b = 2;", "return a + b;"]
        # Ours modified first line
        ours = ["int a = 100;", "int b = 2;", "return a + b;"]
        # Theirs identical to base
        theirs = ["int a = 1;", "int b = 2;", "return a + b;"]

        merged, conflict = three_way_merge_lines(base, ours, theirs)
        self.assertFalse(conflict)
        self.assertEqual(merged, ours)

        # Theirs modified second line
        theirs2 = ["int a = 1;", "int b = 200;", "return a + b;"]
        merged2, conflict2 = three_way_merge_lines(base, base, theirs2)
        self.assertFalse(conflict2)
        self.assertEqual(merged2, theirs2)

    def test_three_way_merge_lines_conflict(self):
        base = ["x = 0;"]
        ours = ["x = 1;"]
        theirs = ["x = 2;"]

        merged, conflict = three_way_merge_lines(base, ours, theirs, "feature-A", "feature-B")
        self.assertTrue(conflict)
        merged_text = "\n".join(merged)
        self.assertIn("<<<<<<< feature-A", merged_text)
        self.assertIn("x = 1;", merged_text)
        self.assertIn("=======", merged_text)
        self.assertIn("x = 2;", merged_text)
        self.assertIn(">>>>>>> feature-B", merged_text)

    def test_three_way_merge_trees_disjoint_files(self):
        # Base has file1
        b_file1 = self.odb.write_blob(b"File 1 content\n")
        t_base = self.odb.write_tree([TreeEntry(0o100644, "file1.txt", b_file1)])

        # Ours adds file2
        b_file2 = self.odb.write_blob(b"File 2 from ours\n")
        t_ours = self.odb.write_tree([
            TreeEntry(0o100644, "file1.txt", b_file1),
            TreeEntry(0o100644, "file2.txt", b_file2)
        ])

        # Theirs adds file3
        b_file3 = self.odb.write_blob(b"File 3 from theirs\n")
        t_theirs = self.odb.write_tree([
            TreeEntry(0o100644, "file1.txt", b_file1),
            TreeEntry(0o100644, "file3.txt", b_file3)
        ])

        res = three_way_merge_trees(t_base, t_ours, t_theirs, self.odb)
        self.assertTrue(res.clean)
        self.assertEqual(len(res.conflicts), 0)
        self.assertIn("file1.txt", res.merged_files)
        self.assertIn("file2.txt", res.merged_files)
        self.assertIn("file3.txt", res.merged_files)


if __name__ == "__main__":
    unittest.main()
