"""
Leveled Compaction Engine:
Performs multi-way merge sort across SSTables to:
- Bound read amplification by merging overlapping Level 0 tables
- Eliminate superseded sequence numbers
- Reclaim disk storage by purging dead tombstoned keys
"""

from __future__ import annotations
import os
import heapq
from typing import List, Dict, Optional, Tuple, Iterator
from chronodb.sstable import SSTableReader, SSTableWriter


class Compactor:
    """
    Orchestrates leveled merge compaction across SSTables.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def merge_tables(
        self,
        source_readers: List[SSTableReader],
        target_filepath: str,
        is_bottom_level: bool = False
    ) -> str:
        """
        Merges multiple SSTables into a single new consolidated SSTable.
        Source tables are merged in key order. For identical keys, highest seq_num wins.
        If is_bottom_level is True, tombstones are purged completely from disk.
        """
        # Multi-way heap iterator
        # Heap items: (key, -seq_num, value, is_tombstone, iterator_index)
        heap = []
        iterators = [iter(reader) for reader in source_readers]

        for idx, it in enumerate(iterators):
            try:
                k, v, seq, tomb = next(it)
                heapq.heappush(heap, (k, -seq, v, tomb, idx))
            except StopIteration:
                pass

        writer = SSTableWriter(target_filepath, expected_entries=10000)

        last_emitted_key = None

        while heap:
            k, neg_seq, v, tomb, idx = heapq.heappop(heap)
            seq = -neg_seq

            # Advance the iterator that yielded this element
            try:
                next_k, next_v, next_seq, next_tomb = next(iterators[idx])
                heapq.heappush(heap, (next_k, -next_seq, next_v, next_tomb, idx))
            except StopIteration:
                pass

            # If we already emitted a newer version of this key, discard the older version
            if k == last_emitted_key:
                continue

            last_emitted_key = k

            # If it's a tombstone and this is bottom level, we can purge it completely
            if tomb and is_bottom_level:
                continue

            writer.append(k, v, seq, is_tombstone=tomb)

        writer.finish()
        return target_filepath
