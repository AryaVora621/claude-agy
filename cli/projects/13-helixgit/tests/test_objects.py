"""
Unit tests for HelixGit Object Model and Loose Storage.
"""

import os
import shutil
import tempfile
import unittest
import hashlib
from helixgit.objects import (
    Blob, Tree, TreeEntry, Commit, Tag, ObjectType, parse_raw_object, compute_sha1
)
from helixgit.storage import LooseObjectStore, ObjectDatabase


class TestGitObjects(unittest.TestCase):
    def test_blob_sha_standard_git_vector(self):
        # Git standard vector: "what is up, doc?"
        # git hash-object: echo -n "what is up, doc?" | git hash-object --stdin
        # Content: "what is up, doc?" (16 bytes)
        # raw: "blob 16\x00what is up, doc?"
        payload = b"what is up, doc?"
        expected_raw = b"blob 16\x00what is up, doc?"
        expected_sha = hashlib.sha1(expected_raw).hexdigest()
        self.assertEqual(expected_sha, "bd9dbf5aae1a3862dd1526723246b20206e5fc37")

        blob = Blob(payload)
        self.assertEqual(blob.serialize(), payload)
        self.assertEqual(blob.raw_data(), expected_raw)
        self.assertEqual(blob.sha1(), "bd9dbf5aae1a3862dd1526723246b20206e5fc37")
        self.assertEqual(blob.text(), "what is up, doc?")

    def test_tree_serialization_and_sorting(self):
        # Git canonical sorting rule: directories sort as if ending with '/'
        # e.g., "foo" (dir) sorts AFTER "foo.c" (file) because "foo/" > "foo.c"
        tree = Tree()
        # Add entries out of order
        tree.add_entry(0o100644, "zebra.txt", "1" * 40)
        tree.add_entry(0o040000, "apple", "2" * 40)
        tree.add_entry(0o100644, "apple.py", "3" * 40)

        # "apple" dir -> sort key "apple/"
        # "apple.py" file -> sort key "apple.py"
        # Since '/' (0x2F) < '.' (0x2E)? In ASCII: '.' is 46, '/' is 47.
        # So "apple." < "apple/" -> "apple.py" comes before "apple" (dir)!
        self.assertEqual(tree.entries[1].sort_key(), "apple/")
        self.assertEqual(tree.entries[2].sort_key(), "apple.py")

        raw_tree_bytes = tree.serialize()
        parsed_tree = Tree.deserialize(raw_tree_bytes)

        self.assertEqual(len(parsed_tree.entries), 3)
        # Verify roundtrip preserves mode, name, sha1
        entry_map = {e.name: (e.mode, e.sha1) for e in parsed_tree.entries}
        self.assertEqual(entry_map["zebra.txt"], (0o100644, "1" * 40))
        self.assertEqual(entry_map["apple"], (0o040000, "2" * 40))
        self.assertEqual(entry_map["apple.py"], (0o100644, "3" * 40))

    def test_commit_serialization_and_roundtrip(self):
        commit = Commit(
            tree="a" * 40,
            parents=["b" * 40, "c" * 40],
            author_name="Ada Lovelace",
            author_email="ada@analytical.engine",
            author_time=1600000000,
            author_tz="+0000",
            committer_name="Charles Babbage",
            committer_email="charles@difference.engine",
            committer_time=1600000005,
            committer_tz="-0500",
            message="Implement first algorithm\n\nNotes on the analytical engine.\n"
        )
        raw_commit = commit.raw_data()
        obj_type, parsed = parse_raw_object(raw_commit)

        self.assertEqual(obj_type, ObjectType.COMMIT)
        self.assertIsInstance(parsed, Commit)
        self.assertEqual(parsed.tree, "a" * 40)
        self.assertEqual(parsed.parents, ["b" * 40, "c" * 40])
        self.assertEqual(parsed.author_name, "Ada Lovelace")
        self.assertEqual(parsed.author_email, "ada@analytical.engine")
        self.assertEqual(parsed.author_time, 1600000000)
        self.assertEqual(parsed.author_tz, "+0000")
        self.assertEqual(parsed.committer_name, "Charles Babbage")
        self.assertEqual(parsed.committer_email, "charles@difference.engine")
        self.assertEqual(parsed.committer_time, 1600000005)
        self.assertEqual(parsed.committer_tz, "-0500")
        self.assertEqual(parsed.message, "Implement first algorithm\n\nNotes on the analytical engine.\n")

    def test_tag_serialization_and_roundtrip(self):
        tag = Tag(
            object_sha="d" * 40,
            type_str="commit",
            tag_name="v1.0.0",
            tagger_name="Linus",
            tagger_email="torvalds@kernel.org",
            tagger_time=1112911993,
            tagger_tz="-0700",
            message="Initial production release\n"
        )
        raw_tag = tag.raw_data()
        obj_type, parsed = parse_raw_object(raw_tag)

        self.assertEqual(obj_type, ObjectType.TAG)
        self.assertIsInstance(parsed, Tag)
        self.assertEqual(parsed.object_sha, "d" * 40)
        self.assertEqual(parsed.type_str, "commit")
        self.assertEqual(parsed.tag_name, "v1.0.0")
        self.assertEqual(parsed.tagger_name, "Linus")
        self.assertEqual(parsed.message, "Initial production release\n")


class TestLooseStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = LooseObjectStore(self.temp_dir)
        self.odb = ObjectDatabase(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_loose_store_write_and_read(self):
        blob = Blob(b"Functional programming in Python\n")
        sha1 = self.store.write(blob)
        self.assertTrue(self.store.exists(sha1))

        # Check path sharding: temp_dir/sha1[:2]/sha1[2:]
        expected_path = os.path.join(self.temp_dir, sha1[:2], sha1[2:])
        self.assertTrue(os.path.isfile(expected_path))

        # Read back raw and typed
        retrieved = self.store.read(sha1)
        self.assertIsInstance(retrieved, Blob)
        self.assertEqual(retrieved.serialize(), b"Functional programming in Python\n")

    def test_odb_caching_and_helpers(self):
        blob_sha = self.odb.write_blob(b"Data stream")
        self.assertTrue(self.odb.exists(blob_sha))

        blob_obj = self.odb.get_blob(blob_sha)
        self.assertEqual(blob_obj.serialize(), b"Data stream")

        # Create tree and commit in ODB
        tree_sha = self.odb.write_tree([
            TreeEntry(0o100644, "stream.txt", blob_sha)
        ])
        tree_obj = self.odb.get_tree(tree_sha)
        self.assertEqual(len(tree_obj.entries), 1)
        self.assertEqual(tree_obj.entries[0].name, "stream.txt")

        commit_sha = self.odb.write_commit(
            tree=tree_sha,
            message="Add stream file\n"
        )
        commit_obj = self.odb.get_commit(commit_sha)
        self.assertEqual(commit_obj.tree, tree_sha)
        self.assertEqual(commit_obj.message, "Add stream file\n")

        # Test listing all shas
        all_shas = self.store.list_all_shas()
        self.assertIn(blob_sha, all_shas)
        self.assertIn(tree_sha, all_shas)
        self.assertIn(commit_sha, all_shas)


if __name__ == "__main__":
    unittest.main()
