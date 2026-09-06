"""
Unit tests for HelixGit Repository Porcelain, Branching, and End-to-End Workflows.
"""

import os
import shutil
import tempfile
import unittest
from helixgit.repo import Repository


class TestRepositoryPorcelain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo = Repository.init(self.temp_dir, default_branch="main")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_repo_init_structure(self):
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, ".git")))
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, ".git", "objects")))
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, ".git", "refs", "heads")))
        branch, head_sha = self.repo.head_ref()
        self.assertEqual(branch, "main")
        self.assertIsNone(head_sha)

    def test_add_and_commit(self):
        file_path = os.path.join(self.temp_dir, "hello.txt")
        with open(file_path, "w") as f:
            f.write("Hello, HelixGit!\n")

        self.repo.add(["hello.txt"])
        self.assertIsNotNone(self.repo.index.get_entry("hello.txt"))

        commit_sha = self.repo.commit("Initial commit\n", "Arya", "arya@domain.com")
        self.assertEqual(len(commit_sha), 40)

        branch, head_sha = self.repo.head_ref()
        self.assertEqual(head_sha, commit_sha)

        commits = self.repo.log()
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0].message.strip(), "Initial commit")

    def test_branch_and_checkout(self):
        # Create initial commit
        file_path = os.path.join(self.temp_dir, "app.py")
        with open(file_path, "w") as f:
            f.write("print('version 1')\n")
        self.repo.add(["app.py"])
        c1 = self.repo.commit("commit 1")

        # Create feature branch
        self.repo.create_branch("feature")
        branches = self.repo.list_branches()
        self.assertIn("main", branches)
        self.assertIn("feature", branches)

        # Checkout feature
        self.repo.checkout("feature")
        branch, head_sha = self.repo.head_ref()
        self.assertEqual(branch, "feature")
        self.assertEqual(head_sha, c1)

        # Modify on feature branch
        with open(file_path, "w") as f:
            f.write("print('version 2 on feature')\n")
        self.repo.add(["app.py"])
        c2 = self.repo.commit("commit 2 on feature")

        # Checkout back to main: working directory should restore version 1
        self.repo.checkout("main")
        with open(file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "print('version 1')\n")

    def test_fast_forward_merge(self):
        # Commit on main
        f1 = os.path.join(self.temp_dir, "base.txt")
        with open(f1, "w") as f:
            f.write("base\n")
        self.repo.add(["base.txt"])
        c1 = self.repo.commit("base commit")

        # Branch feature and add commit
        self.repo.create_branch("feat")
        self.repo.checkout("feat")
        f2 = os.path.join(self.temp_dir, "feat.txt")
        with open(f2, "w") as f:
            f.write("feat\n")
        self.repo.add(["feat.txt"])
        c2 = self.repo.commit("feat commit")

        # Merge feat into main (fast-forward)
        self.repo.checkout("main")
        msg, conflict = self.repo.merge("feat")
        self.assertFalse(conflict)
        self.assertIn("Fast-forward", msg)

        # Check main now points to c2 and feat.txt exists
        _, head_sha = self.repo.head_ref()
        self.assertEqual(head_sha, c2)
        self.assertTrue(os.path.isfile(os.path.join(self.temp_dir, "feat.txt")))


if __name__ == "__main__":
    unittest.main()
