"""
HelixGit: Git Object Model (Blob, Tree, Commit, Tag).
Implements canonical Git object serialization, deserialization, and SHA-1 hashing.
"""

import hashlib
import binascii
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


class ObjectType(Enum):
    BLOB = "blob"
    TREE = "tree"
    COMMIT = "commit"
    TAG = "tag"


def compute_sha1(data: bytes) -> str:
    """Computes hexadecimal SHA-1 digest."""
    return hashlib.sha1(data).hexdigest()


class GitObject:
    """Abstract base class for all Git repository objects."""
    object_type: ObjectType

    def serialize(self) -> bytes:
        """Serializes object payload into raw bytes."""
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: bytes) -> "GitObject":
        """Reconstructs GitObject from raw payload bytes."""
        raise NotImplementedError

    def raw_data(self) -> bytes:
        """Returns Git loose object format: f'{type} {len}\x00{payload}'."""
        payload = self.serialize()
        header = f"{self.object_type.value} {len(payload)}\x00".encode("ascii")
        return header + payload

    def sha1(self) -> str:
        """Computes Git 40-character hexadecimal content hash."""
        return compute_sha1(self.raw_data())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.sha1()[:8]}>"


class Blob(GitObject):
    """Raw content storage object representing file payloads."""
    object_type = ObjectType.BLOB
    __slots__ = ("data",)

    def __init__(self, data: bytes = b"") -> None:
        self.data = data if isinstance(data, bytes) else data.encode("utf-8")

    def serialize(self) -> bytes:
        return self.data

    @classmethod
    def deserialize(cls, data: bytes) -> "Blob":
        return cls(data)

    def text(self) -> str:
        """Decodes payload as UTF-8 string with fallback."""
        return self.data.decode("utf-8", errors="replace")


class TreeEntry:
    """Single file or directory entry inside a Git Tree."""
    __slots__ = ("mode", "name", "sha1")

    def __init__(self, mode: int, name: str, sha1: str) -> None:
        self.mode = mode
        self.name = name
        self.sha1 = sha1.lower()

    @property
    def is_dir(self) -> bool:
        return (self.mode & 0o170000) == 0o040000

    def sort_key(self) -> str:
        """Git canonical sorting rule: directories sort as if ending with '/'."""
        return self.name + "/" if self.is_dir else self.name

    def __repr__(self) -> str:
        return f"TreeEntry({oct(self.mode)}, {self.name}, {self.sha1[:8]})"


class Tree(GitObject):
    """Directory object containing sorted TreeEntry records."""
    object_type = ObjectType.TREE
    __slots__ = ("entries",)

    def __init__(self, entries: Optional[List[TreeEntry]] = None) -> None:
        self.entries: List[TreeEntry] = list(entries or [])

    def add_entry(self, mode: int, name: str, sha1: str) -> "Tree":
        self.entries.append(TreeEntry(mode, name, sha1))
        return self

    def serialize(self) -> bytes:
        """
        Serializes entries in Git canonical sort order.
        Binary record format: f'{octal_mode} {name}\x00{20_byte_binary_sha1}'
        """
        # Sort according to Git's canonical name sorting rule
        sorted_entries = sorted(self.entries, key=lambda e: e.sort_key())
        out = bytearray()
        for e in sorted_entries:
            mode_str = f"{e.mode:o}".encode("ascii")
            name_bytes = e.name.encode("utf-8")
            raw_sha = binascii.unhexlify(e.sha1)
            out.extend(mode_str)
            out.append(0x20)  # space
            out.extend(name_bytes)
            out.append(0x00)  # NUL
            out.extend(raw_sha)
        return bytes(out)

    @classmethod
    def deserialize(cls, data: bytes) -> "Tree":
        entries: List[TreeEntry] = []
        idx = 0
        n = len(data)

        while idx < n:
            space_idx = data.find(b" ", idx)
            if space_idx == -1:
                break
            mode = int(data[idx:space_idx].decode("ascii"), 8)

            null_idx = data.find(b"\x00", space_idx)
            if null_idx == -1:
                break
            name = data[space_idx + 1:null_idx].decode("utf-8", errors="replace")

            raw_sha = data[null_idx + 1:null_idx + 21]
            if len(raw_sha) < 20:
                break
            sha1 = binascii.hexlify(raw_sha).decode("ascii")

            entries.append(TreeEntry(mode, name, sha1))
            idx = null_idx + 21

        return cls(entries)


class Commit(GitObject):
    """Commit object capturing snapshot tree, parents, author, committer, and log message."""
    object_type = ObjectType.COMMIT
    __slots__ = (
        "tree", "parents", "author_name", "author_email",
        "author_time", "author_tz", "committer_name", "committer_email",
        "committer_time", "committer_tz", "message"
    )

    def __init__(
        self,
        tree: str,
        parents: Optional[List[str]] = None,
        author_name: str = "HelixGit",
        author_email: str = "helix@git.local",
        author_time: int = 0,
        author_tz: str = "+0000",
        committer_name: Optional[str] = None,
        committer_email: Optional[str] = None,
        committer_time: Optional[int] = None,
        committer_tz: Optional[str] = None,
        message: str = ""
    ) -> None:
        self.tree = tree.lower()
        self.parents = [p.lower() for p in (parents or [])]
        self.author_name = author_name
        self.author_email = author_email
        self.author_time = author_time
        self.author_tz = author_tz
        self.committer_name = committer_name or author_name
        self.committer_email = committer_email or author_email
        self.committer_time = committer_time if committer_time is not None else author_time
        self.committer_tz = committer_tz or author_tz
        self.message = message

    def serialize(self) -> bytes:
        lines: List[str] = [f"tree {self.tree}"]
        for p in self.parents:
            lines.append(f"parent {p}")

        lines.append(f"author {self.author_name} <{self.author_email}> {self.author_time} {self.author_tz}")
        lines.append(f"committer {self.committer_name} <{self.committer_email}> {self.committer_time} {self.committer_tz}")
        lines.append("")  # Empty line separator
        lines.append(self.message)

        return "\n".join(lines).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "Commit":
        text = data.decode("utf-8", errors="replace")
        header_part, _, message_part = text.partition("\n\n")

        tree = ""
        parents: List[str] = []
        author_name, author_email = "HelixGit", "helix@git.local"
        author_time, author_tz = 0, "+0000"
        committer_name, committer_email = author_name, author_email
        committer_time, committer_tz = 0, "+0000"

        for line in header_part.splitlines():
            if line.startswith("tree "):
                tree = line[5:].strip()
            elif line.startswith("parent "):
                parents.append(line[7:].strip())
            elif line.startswith("author "):
                # Format: author Name <email> 1234567890 +0000
                parts = line[7:].rsplit(" ", 2)
                if len(parts) == 3:
                    ident, t_str, tz = parts
                    author_time = int(t_str) if t_str.isdigit() else 0
                    author_tz = tz
                    if "<" in ident and ident.endswith(">"):
                        n, e = ident[:-1].split("<", 1)
                        author_name = n.strip()
                        author_email = e.strip()
            elif line.startswith("committer "):
                parts = line[10:].rsplit(" ", 2)
                if len(parts) == 3:
                    ident, t_str, tz = parts
                    committer_time = int(t_str) if t_str.isdigit() else 0
                    committer_tz = tz
                    if "<" in ident and ident.endswith(">"):
                        n, e = ident[:-1].split("<", 1)
                        committer_name = n.strip()
                        committer_email = e.strip()

        return cls(
            tree=tree,
            parents=parents,
            author_name=author_name,
            author_email=author_email,
            author_time=author_time,
            author_tz=author_tz,
            committer_name=committer_name,
            committer_email=committer_email,
            committer_time=committer_time,
            committer_tz=committer_tz,
            message=message_part
        )


class Tag(GitObject):
    """Annotated Tag pointing to an arbitrary Git object with message and tagger."""
    object_type = ObjectType.TAG
    __slots__ = (
        "object_sha", "type_str", "tag_name", "tagger_name",
        "tagger_email", "tagger_time", "tagger_tz", "message"
    )

    def __init__(
        self,
        object_sha: str,
        type_str: str = "commit",
        tag_name: str = "",
        tagger_name: str = "HelixGit",
        tagger_email: str = "helix@git.local",
        tagger_time: int = 0,
        tagger_tz: str = "+0000",
        message: str = ""
    ) -> None:
        self.object_sha = object_sha.lower()
        self.type_str = type_str
        self.tag_name = tag_name
        self.tagger_name = tagger_name
        self.tagger_email = tagger_email
        self.tagger_time = tagger_time
        self.tagger_tz = tagger_tz
        self.message = message

    def serialize(self) -> bytes:
        lines = [
            f"object {self.object_sha}",
            f"type {self.type_str}",
            f"tag {self.tag_name}",
            f"tagger {self.tagger_name} <{self.tagger_email}> {self.tagger_time} {self.tagger_tz}",
            "",
            self.message
        ]
        return "\n".join(lines).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "Tag":
        text = data.decode("utf-8", errors="replace")
        header_part, _, message_part = text.partition("\n\n")

        object_sha = ""
        type_str = "commit"
        tag_name = ""
        tagger_name = "HelixGit"
        tagger_email = "helix@git.local"
        tagger_time = 0
        tagger_tz = "+0000"

        for line in header_part.splitlines():
            if line.startswith("object "):
                object_sha = line[7:].strip()
            elif line.startswith("type "):
                type_str = line[5:].strip()
            elif line.startswith("tag "):
                tag_name = line[4:].strip()
            elif line.startswith("tagger "):
                parts = line[7:].rsplit(" ", 2)
                if len(parts) == 3:
                    ident, t_str, tz = parts
                    tagger_time = int(t_str) if t_str.isdigit() else 0
                    tagger_tz = tz
                    if "<" in ident and ident.endswith(">"):
                        n, e = ident[:-1].split("<", 1)
                        tagger_name = n.strip()
                        tagger_email = e.strip()

        return cls(
            object_sha=object_sha,
            type_str=type_str,
            tag_name=tag_name,
            tagger_name=tagger_name,
            tagger_email=tagger_email,
            tagger_time=tagger_time,
            tagger_tz=tagger_tz,
            message=message_part
        )


def parse_raw_object(raw_data: bytes) -> Tuple[ObjectType, GitObject]:
    """Parses loose Git format f'{type} {len}\x00{payload}' into typed GitObject."""
    null_idx = raw_data.find(b"\x00")
    if null_idx == -1:
        raise ValueError("Invalid Git object: missing header NUL byte")

    header = raw_data[:null_idx].decode("ascii")
    payload = raw_data[null_idx + 1:]

    type_str, size_str = header.split(" ", 1)
    expected_size = int(size_str)
    if len(payload) != expected_size:
        raise ValueError(f"Corrupt Git object: size mismatch (expected {expected_size}, got {len(payload)})")

    if type_str == ObjectType.BLOB.value:
        return ObjectType.BLOB, Blob.deserialize(payload)
    elif type_str == ObjectType.TREE.value:
        return ObjectType.TREE, Tree.deserialize(payload)
    elif type_str == ObjectType.COMMIT.value:
        return ObjectType.COMMIT, Commit.deserialize(payload)
    elif type_str == ObjectType.TAG.value:
        return ObjectType.TAG, Tag.deserialize(payload)
    else:
        raise ValueError(f"Unknown Git object type: {type_str}")
