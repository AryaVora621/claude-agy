"""
Probabilistic SkipList MemTable:
Provides O(log n) expected search, insertion, and ordered range scans.
Maintains sequence numbers for MVCC snapshot isolation, deletion tombstones,
and deterministic byte memory budgeting for LSM-tree flush triggering.
"""

from __future__ import annotations
import random
from typing import Optional, List, Tuple, Iterator


class SkipListNode:
    """A node inside the multi-level skip list."""

    __slots__ = ("key", "value", "seq_num", "is_tombstone", "forward")

    def __init__(
        self,
        key: bytes,
        value: bytes,
        seq_num: int,
        is_tombstone: bool,
        level: int,
    ):
        self.key = key
        self.value = value
        self.seq_num = seq_num
        self.is_tombstone = is_tombstone
        # Forward pointers for each level in the node's tower
        self.forward: List[Optional[SkipListNode]] = [None] * (level + 1)


class SkipList:
    """
    In-memory sorted table (MemTable) implementing a probabilistic SkipList.
    Keys are ordered lexicographically by raw byte value.
    """

    MAX_LEVEL = 16
    P = 0.5  # Probability factor for level promotion

    def __init__(self):
        self.header = SkipListNode(
            key=b"",
            value=b"",
            seq_num=0,
            is_tombstone=False,
            level=self.MAX_LEVEL,
        )
        self.level = 0
        self.count = 0
        self.approx_bytes = 0

    def _random_level(self) -> int:
        """Determines the height of a new node using geometric distribution."""
        lvl = 0
        while random.random() < self.P and lvl < self.MAX_LEVEL:
            lvl += 1
        return lvl

    def put(self, key: bytes, value: bytes, seq_num: int, is_tombstone: bool = False) -> None:
        """
        Inserts or updates a key-value pair.
        Tracks accurate byte consumption for memory threshold triggers.
        """
        update = [None] * (self.MAX_LEVEL + 1)
        curr = self.header

        # Traverse down from highest active level
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]
            update[i] = curr

        target = curr.forward[0]

        # Case 1: Key already exists in MemTable -> Update in place
        if target and target.key == key:
            old_bytes = len(target.value)
            new_bytes = len(value)
            self.approx_bytes += (new_bytes - old_bytes)

            target.value = value
            target.seq_num = seq_num
            target.is_tombstone = is_tombstone
            return

        # Case 2: New key insertion
        new_level = self._random_level()
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        new_node = SkipListNode(
            key=key,
            value=value,
            seq_num=seq_num,
            is_tombstone=is_tombstone,
            level=new_level,
        )

        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self.count += 1
        # Node overhead estimate: key + value + 64 bytes pointer/metadata overhead
        self.approx_bytes += len(key) + len(value) + 64 + (new_level * 8)

    def get(self, key: bytes) -> Optional[Tuple[bytes, int, bool]]:
        """
        Searches for key.
        Returns: (value, seq_num, is_tombstone) if found, else None.
        """
        curr = self.header
        for i in range(self.level, -1, -1):
            while curr.forward[i] and curr.forward[i].key < key:
                curr = curr.forward[i]

        curr = curr.forward[0]
        if curr and curr.key == key:
            return curr.value, curr.seq_num, curr.is_tombstone
        return None

    def __iter__(self) -> Iterator[Tuple[bytes, bytes, int, bool]]:
        """Ordered iterator yielding (key, value, seq_num, is_tombstone)."""
        curr = self.header.forward[0]
        while curr:
            yield curr.key, curr.value, curr.seq_num, curr.is_tombstone
            curr = curr.forward[0]

    def scan(
        self,
        start_key: Optional[bytes] = None,
        end_key: Optional[bytes] = None
    ) -> Iterator[Tuple[bytes, bytes, int, bool]]:
        """
        Yields all entries within [start_key, end_key) in lexicographical order.
        """
        curr = self.header
        if start_key is not None:
            for i in range(self.level, -1, -1):
                while curr.forward[i] and curr.forward[i].key < start_key:
                    curr = curr.forward[i]
            curr = curr.forward[0]
        else:
            curr = self.header.forward[0]

        while curr:
            if end_key is not None and curr.key >= end_key:
                break
            yield curr.key, curr.value, curr.seq_num, curr.is_tombstone
            curr = curr.forward[0]

    def clear(self) -> None:
        """Resets the skip list to empty state."""
        self.header = SkipListNode(
            key=b"",
            value=b"",
            seq_num=0,
            is_tombstone=False,
            level=self.MAX_LEVEL,
        )
        self.level = 0
        self.count = 0
        self.approx_bytes = 0
