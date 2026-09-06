"""
HelixGit: Git Binary Index Staging Format (DIRC v2) and Worktree Engine.
Implements the 62-byte stat cache entry format, multi-stage conflict slots,
working tree checkout, and Merkle tree synthesis from staged index entries.
"""

import os
import stat
import struct
import binascii
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from .objects import Tree, TreeEntry, Blob, compute_sha1
from .storage import ObjectDatabase

DIRC_MAGIC = b"DIRC"
INDEX_VERSION = 2
ENTRY_FIXED_LEN = 62  # 10*4 + 20 + 2 bytes


class IndexEntry:
    """
    Represents a single staged entry in the Git binary index (.git/index).
    Contains stat cache metadata for fast change detection without disk content reads.
    """
    __slots__ = (
        "ctime_s", "ctime_ns",
        "mtime_s", "mtime_ns",
        "dev", "ino", "mode", "uid", "gid", "file_size",
        "sha1", "flags", "path", "stage"
    )

    def __init__(
        self,
        path: str,
        sha1: str,
        mode: int = 0o100644,
        ctime_s: int = 0,
        ctime_ns: int = 0,
        mtime_s: int = 0,
        mtime_ns: int = 0,
        dev: int = 0,
        ino: int = 0,
        uid: int = 0,
        gid: int = 0,
        file_size: int = 0,
        stage: int = 0,
        assume_valid: bool = False
    ) -> None:
        self.path = path.replace("\\", "/").strip("/")
        self.sha1 = sha1.lower()
        self.mode = mode
        self.ctime_s = ctime_s
        self.ctime_ns = ctime_ns
        self.mtime_s = mtime_s
        self.mtime_ns = mtime_ns
        self.dev = dev
        self.ino = ino
        self.uid = uid
        self.gid = gid
        self.file_size = file_size
        self.stage = stage

        # Construct 16-bit flags
        # 1 bit assume-valid, 1 bit extended (0 in v2), 2 bits stage, 12 bits path length
        path_len = min(len(self.path.encode("utf-8")), 0xFFF)
        flag_val = (path_len & 0xFFF) | ((stage & 0x3) << 12)
        if assume_valid:
            flag_val |= 0x8000
        self.flags = flag_val

    @property
    def is_dir(self) -> bool:
        return (self.mode & 0o170000) == 0o040000

    def is_modified_by_stat(self, st: os.stat_result) -> bool:
        """
        Fast check: returns True if filesystem stat indicates the file has changed.
        Compares file size and modification time seconds/nanoseconds.
        """
        if self.file_size != (st.st_size & 0xFFFFFFFF):
            return True
        st_mtime_s = int(st.st_mtime)
        if self.mtime_s != st_mtime_s:
            return True
        # Nanoseconds check if available
        st_mtime_ns = getattr(st, "st_mtime_ns", 0) % 1_000_000_000
        if self.mtime_ns != 0 and st_mtime_ns != 0 and self.mtime_ns != st_mtime_ns:
            return True
        # Mode check (executable bit)
        is_exec_idx = bool(self.mode & 0o111)
        is_exec_fs = bool(st.st_mode & 0o111)
        if is_exec_idx != is_exec_fs:
            return True
        return False

    def serialize(self) -> bytes:
        """Serializes entry to binary format with 8-byte aligned NUL padding."""
        raw_sha1 = binascii.unhexlify(self.sha1)
        path_bytes = self.path.encode("utf-8")

        # 10 uint32s: ctime(s, ns), mtime(s, ns), dev, ino, mode, uid, gid, size
        header_data = struct.pack(
            ">10I",
            self.ctime_s & 0xFFFFFFFF,
            self.ctime_ns & 0xFFFFFFFF,
            self.mtime_s & 0xFFFFFFFF,
            self.mtime_ns & 0xFFFFFFFF,
            self.dev & 0xFFFFFFFF,
            self.ino & 0xFFFFFFFF,
            self.mode & 0xFFFFFFFF,
            self.uid & 0xFFFFFFFF,
            self.gid & 0xFFFFFFFF,
            self.file_size & 0xFFFFFFFF
        )

        entry_fixed = header_data + raw_sha1 + struct.pack(">H", self.flags)
        # Entry padding: between 1 and 8 NUL bytes to reach a multiple of 8 bytes
        cur_len = len(entry_fixed) + len(path_bytes)
        pad_len = 8 - (cur_len % 8)
        if pad_len == 0:
            pad_len = 8

        return entry_fixed + path_bytes + (b"\x00" * pad_len)

    @classmethod
    def deserialize(cls, data: bytes, offset: int) -> Tuple["IndexEntry", int]:
        """Deserializes IndexEntry from binary buffer at offset, returning (entry, next_offset)."""
        fixed_size = ENTRY_FIXED_LEN
        header_vals = struct.unpack(">10I", data[offset:offset + 40])
        raw_sha1 = data[offset + 40:offset + 60]
        sha1_str = binascii.hexlify(raw_sha1).decode("ascii")
        flags = struct.unpack(">H", data[offset + 60:offset + 62])[0]

        stage = (flags >> 12) & 0x3
        assume_valid = bool(flags & 0x8000)

        # Read NUL-terminated path
        path_start = offset + 62
        nul_pos = data.find(b"\x00", path_start)
        if nul_pos == -1:
            raise ValueError("Corrupt Git index: entry missing NUL termination")

        path = data[path_start:nul_pos].decode("utf-8", errors="replace")

        # Advance past 1-8 byte NUL padding to 8-byte boundary
        entry_raw_len = (nul_pos - offset) + 1
        pad_len = 8 - (entry_raw_len % 8)
        if pad_len != 8:
            next_offset = nul_pos + 1 + pad_len
        else:
            next_offset = nul_pos + 1

        entry = cls(
            path=path,
            sha1=sha1_str,
            mode=header_vals[6],
            ctime_s=header_vals[0],
            ctime_ns=header_vals[1],
            mtime_s=header_vals[2],
            mtime_ns=header_vals[3],
            dev=header_vals[4],
            ino=header_vals[5],
            uid=header_vals[7],
            gid=header_vals[8],
            file_size=header_vals[9],
            stage=stage,
            assume_valid=assume_valid
        )
        return entry, next_offset

    def __repr__(self) -> str:
        return f"IndexEntry({self.path}, sha={self.sha1[:8]}, stage={self.stage}, mode={oct(self.mode)})"


class Index:
    """
    Git Binary Index (DIRC v2) manager.
    Maintains staged files, multi-stage conflict records, and working tree synchronization.
    """

    def __init__(self) -> None:
        # Key: (path, stage) -> IndexEntry
        self.entries: Dict[Tuple[str, int], IndexEntry] = {}

    def add_entry(self, entry: IndexEntry) -> None:
        self.entries[(entry.path, entry.stage)] = entry

    def get_entry(self, path: str, stage: int = 0) -> Optional[IndexEntry]:
        norm_path = path.replace("\\", "/").strip("/")
        return self.entries.get((norm_path, stage))

    def remove(self, path: str, stage: Optional[int] = None) -> None:
        """Removes entry for path. If stage is None, clears all stages (0, 1, 2, 3)."""
        norm_path = path.replace("\\", "/").strip("/")
        if stage is not None:
            self.entries.pop((norm_path, stage), None)
        else:
            for s in (0, 1, 2, 3):
                self.entries.pop((norm_path, s), None)

    def has_conflicts(self) -> bool:
        """Returns True if any file has non-zero stage entries (unmerged conflicts)."""
        return any(stage > 0 for (_, stage) in self.entries.keys())

    def get_conflicted_paths(self) -> Set[str]:
        """Returns set of all paths that have unmerged conflicts (stage > 0)."""
        return {path for (path, stage) in self.entries.keys() if stage > 0}

    def stage_file(self, worktree_dir: str, rel_path: str, odb: ObjectDatabase) -> IndexEntry:
        """
        Hashes file from disk, stores blob in ODB, captures filesystem stat, and stages entry.
        """
        norm_path = rel_path.replace("\\", "/").strip("/")
        full_path = os.path.join(worktree_dir, norm_path)

        st = os.stat(full_path)
        with open(full_path, "rb") as f:
            content = f.read()

        blob_sha = odb.write_blob(content)

        # Determine mode
        is_exec = bool(st.st_mode & 0o111)
        mode = 0o100755 if is_exec else 0o100644

        st_mtime_ns = getattr(st, "st_mtime_ns", 0) % 1_000_000_000
        st_ctime_ns = getattr(st, "st_ctime_ns", 0) % 1_000_000_000

        entry = IndexEntry(
            path=norm_path,
            sha1=blob_sha,
            mode=mode,
            ctime_s=int(st.st_ctime),
            ctime_ns=st_ctime_ns,
            mtime_s=int(st.st_mtime),
            mtime_ns=st_mtime_ns,
            dev=st.st_dev,
            ino=st.st_ino,
            uid=st.st_uid,
            gid=st.st_gid,
            file_size=len(content),
            stage=0
        )
        # Clear any prior conflicts and stage normal entry
        self.remove(norm_path)
        self.add_entry(entry)
        return entry

    def read(self, index_file: str) -> None:
        """Parses binary .git/index file into in-memory entries dictionary."""
        self.entries.clear()
        if not os.path.isfile(index_file):
            return

        with open(index_file, "rb") as f:
            data = f.read()

        if len(data) < 32:  # 12 header + 20 checksum
            raise ValueError("Corrupt Git index: file is smaller than minimum header + checksum")

        # Verify trailing SHA-1 checksum
        body = data[:-20]
        file_checksum = data[-20:]
        calc_checksum = hashlib.sha1(body).digest()
        if file_checksum != calc_checksum:
            raise ValueError("Corrupt Git index: checksum mismatch")

        magic = data[:4]
        if magic != DIRC_MAGIC:
            raise ValueError(f"Invalid Git index: bad magic {magic!r}")

        version, num_entries = struct.unpack(">2I", data[4:12])
        if version != INDEX_VERSION:
            raise ValueError(f"Unsupported Git index version: {version}")

        offset = 12
        for _ in range(num_entries):
            entry, offset = IndexEntry.deserialize(data, offset)
            self.add_entry(entry)

    def write(self, index_file: str) -> None:
        """Serializes index entries to binary DIRC v2 format with trailing SHA-1 checksum."""
        # Sort entries: Git sorts by path asc, then stage asc
        sorted_keys = sorted(self.entries.keys(), key=lambda k: (k[0], k[1]))

        out = bytearray()
        # 12-byte header
        out.extend(DIRC_MAGIC)
        out.extend(struct.pack(">2I", INDEX_VERSION, len(sorted_keys)))

        for k in sorted_keys:
            entry = self.entries[k]
            out.extend(entry.serialize())

        # Calculate SHA-1 checksum over all preceding bytes
        checksum = hashlib.sha1(bytes(out)).digest()
        out.extend(checksum)

        os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
        tmp_file = index_file + ".tmp"
        with open(tmp_file, "wb") as f:
            f.write(out)
        os.replace(tmp_file, index_file)

    def to_tree(self, odb: ObjectDatabase) -> str:
        """
        Builds a hierarchical Merkle Tree of Git Tree objects from staged stage-0 entries.
        Writes all intermediate subtrees and the root tree to the ODB.
        Returns the 40-character root tree SHA-1.
        """
        if self.has_conflicts():
            raise RuntimeError("Cannot write tree: index contains unmerged conflicts")

        # Construct nested directory structure: {dir_path: {child_name: (mode, sha, is_tree)}}
        # Root is ""
        tree_nodes: Dict[str, Dict[str, Tuple[int, str, bool]]] = {"": {}}

        for (path, stage), entry in self.entries.items():
            if stage != 0:
                continue

            parts = path.split("/")
            # Ensure parent directories exist in tree_nodes
            for i in range(len(parts) - 1):
                parent_dir = "/".join(parts[:i])
                dir_name = parts[i]
                current_dir = "/".join(parts[:i + 1])

                if parent_dir not in tree_nodes:
                    tree_nodes[parent_dir] = {}
                if current_dir not in tree_nodes:
                    tree_nodes[current_dir] = {}

                # Mark directory in parent
                tree_nodes[parent_dir][dir_name] = (0o040000, "", True)

            parent_dir = "/".join(parts[:-1])
            file_name = parts[-1]
            if parent_dir not in tree_nodes:
                tree_nodes[parent_dir] = {}
            tree_nodes[parent_dir][file_name] = (entry.mode, entry.sha1, False)

        # Process trees bottom-up (sorted by path depth descending)
        all_dirs = sorted(tree_nodes.keys(), key=lambda d: len(d.split("/")) if d else 0, reverse=True)

        dir_hashes: Dict[str, str] = {}
        for d in all_dirs:
            entries: List[TreeEntry] = []
            for name, (mode, sha, is_tree) in tree_nodes[d].items():
                if is_tree:
                    child_path = f"{d}/{name}".strip("/")
                    tree_sha = dir_hashes[child_path]
                    entries.append(TreeEntry(0o040000, name, tree_sha))
                else:
                    entries.append(TreeEntry(mode, name, sha))

            tree_obj = Tree(entries)
            tree_sha = odb.store(tree_obj)
            dir_hashes[d] = tree_sha

        return dir_hashes.get("", odb.store(Tree()))

    def from_tree(self, root_sha: str, odb: ObjectDatabase, worktree_dir: Optional[str] = None) -> None:
        """
        Reconstructs the index from a root Git Tree in the ODB.
        Optionally checks out files to worktree_dir with accurate file stats.
        """
        self.entries.clear()

        def _traverse(tree_sha: str, parent_prefix: str) -> None:
            tree = odb.get_tree(tree_sha)
            for entry in tree.entries:
                rel_path = f"{parent_prefix}/{entry.name}".strip("/")
                if entry.is_dir:
                    _traverse(entry.sha1, rel_path)
                else:
                    st_mtime_s, st_mtime_ns = 0, 0
                    st_ctime_s, st_ctime_ns = 0, 0
                    dev, ino, uid, gid = 0, 0, 0, 0
                    file_size = 0

                    if worktree_dir is not None:
                        full_path = os.path.join(worktree_dir, rel_path)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        blob = odb.get_blob(entry.sha1)
                        with open(full_path, "wb") as f:
                            f.write(blob.data)
                        if entry.mode == 0o100755:
                            os.chmod(full_path, 0o755)

                        st = os.stat(full_path)
                        st_mtime_s = int(st.st_mtime)
                        st_mtime_ns = getattr(st, "st_mtime_ns", 0) % 1_000_000_000
                        st_ctime_s = int(st.st_ctime)
                        st_ctime_ns = getattr(st, "st_ctime_ns", 0) % 1_000_000_000
                        dev, ino = st.st_dev, st.st_ino
                        uid, gid = st.st_uid, st.st_gid
                        file_size = len(blob.data)

                    idx_entry = IndexEntry(
                        path=rel_path,
                        sha1=entry.sha1,
                        mode=entry.mode,
                        ctime_s=st_ctime_s,
                        ctime_ns=st_ctime_ns,
                        mtime_s=st_mtime_s,
                        mtime_ns=st_mtime_ns,
                        dev=dev,
                        ino=ino,
                        uid=uid,
                        gid=gid,
                        file_size=file_size,
                        stage=0
                    )
                    self.add_entry(idx_entry)

        if root_sha:
            _traverse(root_sha, "")

    def diff_worktree(self, worktree_dir: str) -> Dict[str, List[str]]:
        """
        Compares the working directory files against staged index entries.
        Returns:
          {
            "modified": [paths changed on disk],
            "deleted": [paths in index but missing from disk],
            "untracked": [paths on disk not present in index]
          }
        """
        modified: List[str] = []
        deleted: List[str] = []
        untracked: List[str] = []

        # Check existing index files
        staged_paths = {path for (path, stage) in self.entries.keys() if stage == 0}

        for path in sorted(staged_paths):
            full_path = os.path.join(worktree_dir, path)
            if not os.path.exists(full_path):
                deleted.append(path)
            else:
                st = os.stat(full_path)
                entry = self.get_entry(path, 0)
                if entry and entry.is_modified_by_stat(st):
                    # Verify by content hash
                    with open(full_path, "rb") as f:
                        data = f.read()
                    content_sha = compute_sha1(Blob(data).raw_data())
                    if content_sha != entry.sha1:
                        modified.append(path)

        # Find untracked files
        for root, dirs, files in os.walk(worktree_dir):
            # Skip .git directory
            if ".git" in dirs:
                dirs.remove(".git")
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, worktree_dir).replace("\\", "/").strip("/")
                if rel_path not in staged_paths:
                    untracked.append(rel_path)

        return {
            "modified": sorted(modified),
            "deleted": sorted(deleted),
            "untracked": sorted(untracked)
        }
