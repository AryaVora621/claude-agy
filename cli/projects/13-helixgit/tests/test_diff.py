"""
Unit tests for Myers O((N+M)D) Diff Engine and Unified Diff Formatter.
"""

import unittest
from helixgit.diff import myers_diff, unified_diff, DiffType


class TestMyersDiff(unittest.TestCase):
    def test_diff_identical_sequences(self):
        a = ["line 1", "line 2", "line 3"]
        b = ["line 1", "line 2", "line 3"]
        diff = myers_diff(a, b)
        self.assertEqual(len(diff), 3)
        self.assertTrue(all(d.type == DiffType.KEEP for d in diff))

        udiff = unified_diff(a, b)
        self.assertEqual(udiff, "")

    def test_diff_pure_insertions(self):
        a = ["alpha", "gamma"]
        b = ["alpha", "beta", "gamma"]
        diff = myers_diff(a, b)
        types = [d.type for d in diff]
        self.assertEqual(types, [DiffType.KEEP, DiffType.INSERT, DiffType.KEEP])

        udiff = unified_diff(a, b, "a/file.txt", "b/file.txt")
        self.assertIn("--- a/file.txt", udiff)
        self.assertIn("+++ b/file.txt", udiff)
        self.assertIn("+beta", udiff)

    def test_diff_pure_deletions(self):
        a = ["apple", "banana", "cherry"]
        b = ["apple", "cherry"]
        diff = myers_diff(a, b)
        types = [d.type for d in diff]
        self.assertEqual(types, [DiffType.KEEP, DiffType.DELETE, DiffType.KEEP])

        udiff = unified_diff(a, b)
        self.assertIn("-banana", udiff)

    def test_diff_replacement_and_context(self):
        a = [f"line {i}" for i in range(1, 11)]
        b = [f"line {i}" for i in range(1, 11)]
        b[4] = "line 5 - modified"

        diff = myers_diff(a, b)
        deleted = [d.text for d in diff if d.type == DiffType.DELETE]
        inserted = [d.text for d in diff if d.type == DiffType.INSERT]
        self.assertEqual(deleted, ["line 5"])
        self.assertEqual(inserted, ["line 5 - modified"])

        udiff = unified_diff(a, b, context=2)
        self.assertIn("-line 5", udiff)
        self.assertIn("+line 5 - modified", udiff)
        self.assertIn(" line 4", udiff)
        self.assertIn(" line 6", udiff)


if __name__ == "__main__":
    unittest.main()
