"""
HelixGit: Storage Engine for Git Objects.
Implements loose object database storage with zlib compression and 2-character directory sharding.
"""

import os
import zlib
from typing import Optional, Tuple, Dict, Set
from .objects import (
    GitObject, ObjectType, Blob, Tree, Commit, Tag,
    parse_raw_object, compute_sha1
)


class LooseObjectStore:
    """
    Manages filesystem storage and retrieval of loose Git objects.
    Stored at: <git_dir>/objects/{sha1[:2]}/{sha1[2:]}
    """

    def __init__(self, objects_dir: str) -> None:
        self.objects_dir = objects_dir

    def _object_path(self, sha1: str) -> str:
        sha1 = sha1.lower()
        return os.path.join(self.objects_dir, sha1[:2], sha1[2:])

    def exists(self, sha1: str) -> bool:
        """Checks if loose object exists on disk."""
        return os.path.isfile(self._object_path(sha1))

    def write(self, obj: GitObject) -> str:
        """
        Compresses and persists GitObject as a loose file.
        Returns the 40-character hexadecimal SHA-1 digest.
        """
        raw_bytes = obj.raw_data()
        sha1 = compute_sha1(raw_bytes)
        target_path = self._object_path(sha1)

        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            compressed = zlib.compress(raw_bytes, level=zlib.Z_BEST_SPEED)
            # Write atomically using temporary file
            tmp_path = target_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(compressed)
            os.replace(tmp_path, target_path)

        return sha1

    def read_raw(self, sha1: str) -> bytes:
        """Reads and decompresses loose object bytes from disk."""
        target_path = self._object_path(sha1)
        if not os.path.isfile(target_path):
            raise KeyError(f"Git object not found: {sha1}")

        with open(target_path, "rb") as f:
            compressed = f.read()
        return zlib.decompress(compressed)

    def read(self, sha1: str) -> GitObject:
        """Reads, decompresses, and deserializes typed GitObject from disk."""
        raw_bytes = self.read_raw(sha1)
        _, obj = parse_raw_object(raw_bytes)
        return obj

    def list_all_shas(self) -> Set[str]:
        """Scans loose objects directory and returns set of all available SHA-1 hashes."""
        shas: Set[str] = set()
        if not os.path.isdir(self.objects_dir):
            return shas

        for prefix in os.listdir(self.objects_dir):
            prefix_dir = os.path.join(self.objects_dir, prefix)
            if len(prefix) == 2 and os.path.isdir(prefix_dir):
                for suffix in os.listdir(prefix_dir):
                    if len(suffix) == 38:
                        shas.add((prefix + suffix).lower())
        return shas


class ObjectDatabase:
    """
    Unified object database combining loose storage and in-memory caches.
    Can be extended with packfile lookup.
    """

    def __init__(self, objects_dir: str) -> None:
        self.loose = LooseObjectStore(objects_dir)
        self._memory_cache: Dict[str, GitObject] = {}

    def store(self, obj: GitObject) -> str:
        """Saves object to loose storage and populates in-memory cache."""
        sha1 = self.loose.write(obj)
        self._memory_cache[sha1] = obj
        return sha1

    def get(self, sha1: str) -> GitObject:
        """Retrieves object by SHA-1 hash, checking cache first."""
        sha1 = sha1.lower()
        if sha1 in self._memory_cache:
            return self._memory_cache[sha1]
        obj = self.loose.read(sha1)
        self._memory_cache[sha1] = obj
        return obj

    def get_blob(self, sha1: str) -> Blob:
        obj = self.get(sha1)
        if not isinstance(obj, Blob):
            raise TypeError(f"Object {sha1} is {type(obj).__name__}, expected Blob")
        return obj

    def get_tree(self, sha1: str) -> Tree:
        obj = self.get(sha1)
        if not isinstance(obj, Tree):
            raise TypeError(f"Object {sha1} is {type(obj).__name__}, expected Tree")
        return obj

    def get_commit(self, sha1: str) -> Commit:
        obj = self.get(sha1)
        if not isinstance(obj, Commit):
            raise TypeError(f"Object {sha1} is {type(obj).__name__}, expected Commit")
        return obj

    def get_tag(self, sha1: str) -> Tag:
        obj = self.get(sha1)
        if not isinstance(obj, Tag):
            raise TypeError(f"Object {sha1} is {type(obj).__name__}, expected Tag")
        return obj

    def exists(self, sha1: str) -> bool:
        sha1 = sha1.lower()
        return sha1 in self._memory_cache or self.loose.exists(sha1)

    def write_blob(self, data: bytes) -> str:
        blob = Blob(data)
        return self.store(blob)

    def write_tree(self, entries: list) -> str:
        tree = Tree(entries)
        return self.store(tree)

    def write_commit(
        self,
        tree: str,
        parents: Optional[list] = None,
        author_name: str = "HelixGit",
        author_email: str = "helix@git.local",
        author_time: int = 0,
        author_tz: str = "+0000",
        message: str = ""
    ) -> str:
        commit = Commit(
            tree=tree,
            parents=parents or [],
            author_name=author_name,
            author_email=author_email,
            author_time=author_time,
            author_tz=author_tz,
            message=message
        )
        return self.store(commit)
