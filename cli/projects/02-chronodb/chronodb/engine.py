"""
ChronoDB Storage Engine:
High-performance, embeddable LSM-tree key-value store with native HNSW vector search.
Features WAL crash recovery, SkipList MemTable, SSTables with Bloom filters,
Leveled Compaction, and multi-dimensional vector search.
"""

from __future__ import annotations
import os
import glob
import json
import time
from typing import Optional, Dict, List, Tuple, Any, Iterator
from chronodb.bloom import BloomFilter
from chronodb.skiplist import SkipList
from chronodb.wal import WAL, OP_PUT, OP_DELETE
from chronodb.sstable import SSTableWriter, SSTableReader
from chronodb.compaction import Compactor
from chronodb.hnsw import HNSWIndex


class ChronoDB:
    """
    Unified LSM-Tree & Vector Database Engine.
    """

    def __init__(
        self,
        data_dir: str,
        memtable_limit_bytes: int = 1024 * 1024,  # 1 MB threshold
        max_l0_tables: int = 4,
        vector_dim: Optional[int] = None,
        vector_metric: str = "cosine",
    ):
        self.data_dir = data_dir
        self.memtable_limit = memtable_limit_bytes
        self.max_l0_tables = max_l0_tables
        self.vector_dim = vector_dim

        # Ensure directory tree
        os.makedirs(data_dir, exist_ok=True)
        self.l0_dir = os.path.join(data_dir, "level_0")
        self.l1_dir = os.path.join(data_dir, "level_1")
        os.makedirs(self.l0_dir, exist_ok=True)
        os.makedirs(self.l1_dir, exist_ok=True)

        self.wal_path = os.path.join(data_dir, "active.wal")
        self.seq_num = 0
        self.memtable = SkipList()

        # Compactor
        self.compactor = Compactor(data_dir)

        # Active SSTable Readers by level: level -> list of SSTableReader (newest first for L0)
        self.l0_readers: List[SSTableReader] = []
        self.l1_readers: List[SSTableReader] = []

        # Vector Index
        self.hnsw: Optional[HNSWIndex] = None
        self.hnsw_path = os.path.join(data_dir, "vectors.hnsw.json")
        if vector_dim is not None:
            if os.path.exists(self.hnsw_path):
                self.hnsw = HNSWIndex.load(self.hnsw_path)
            else:
                self.hnsw = HNSWIndex(dim=vector_dim, metric=vector_metric)

        # Metrics
        self.stats = {
            "puts": 0,
            "gets": 0,
            "deletes": 0,
            "flushes": 0,
            "compactions": 0,
            "bloom_hits": 0,
            "bloom_negatives": 0,
        }

        # Initialize existing tables and recover WAL
        self._open_existing_tables()
        self._recover_and_init_wal()

    def _open_existing_tables(self) -> None:
        """Opens and caches readers for all existing SSTables on disk."""
        # Level 0 tables sorted by modification time (newest first)
        l0_files = sorted(glob.glob(os.path.join(self.l0_dir, "*.sst")), key=os.path.getmtime, reverse=True)
        self.l0_readers = [SSTableReader(f) for f in l0_files]

        # Level 1 tables sorted by key prefix
        l1_files = sorted(glob.glob(os.path.join(self.l1_dir, "*.sst")))
        self.l1_readers = [SSTableReader(f) for f in l1_files]

    def _recover_and_init_wal(self) -> None:
        """Replays WAL entries to restore in-memory state after restarts or crashes."""
        if os.path.exists(self.wal_path):
            recovered_count = 0
            for op, seq, k, v in WAL.recover(self.wal_path):
                self.seq_num = max(self.seq_num, seq)
                is_tomb = (op == OP_DELETE)
                self.memtable.put(k, v, seq_num=seq, is_tombstone=is_tomb)
                recovered_count += 1

        # Reopen WAL for live writes
        self.wal = WAL(self.wal_path, sync_on_write=True)

    def put(self, key: str | bytes, value: str | bytes) -> None:
        """Inserts or updates a key-value record."""
        k = key.encode("utf-8") if isinstance(key, str) else key
        v = value.encode("utf-8") if isinstance(value, str) else value

        self.seq_num += 1
        seq = self.seq_num

        # 1. Write to WAL first for durability
        self.wal.append(OP_PUT, seq, k, v)

        # 2. Insert into in-memory SkipList
        self.memtable.put(k, v, seq_num=seq, is_tombstone=False)
        self.stats["puts"] += 1

        # 3. Check flush threshold
        if self.memtable.approx_bytes >= self.memtable_limit:
            self.flush()

    def get(self, key: str | bytes) -> Optional[bytes]:
        """
        Queries the database for key.
        Checks:
        1. MemTable
        2. Level 0 SSTables (newest to oldest)
        3. Level 1 SSTables
        """
        k = key.encode("utf-8") if isinstance(key, str) else key
        self.stats["gets"] += 1

        # 1. Check MemTable
        mem_res = self.memtable.get(k)
        if mem_res is not None:
            val, seq, is_tomb = mem_res
            return None if is_tomb else val

        # 2. Check Level 0 SSTables
        for reader in self.l0_readers:
            res = reader.get(k)
            if res is not None:
                val, seq, is_tomb = res
                self.stats["bloom_hits"] += 1
                return None if is_tomb else val
            else:
                self.stats["bloom_negatives"] += 1

        # 3. Check Level 1 SSTables
        for reader in self.l1_readers:
            res = reader.get(k)
            if res is not None:
                val, seq, is_tomb = res
                self.stats["bloom_hits"] += 1
                return None if is_tomb else val
            else:
                self.stats["bloom_negatives"] += 1

        return None

    def delete(self, key: str | bytes) -> None:
        """Deletes key by writing a tombstone record."""
        k = key.encode("utf-8") if isinstance(key, str) else key
        self.seq_num += 1
        seq = self.seq_num

        # 1. Write tombstone to WAL
        self.wal.append(OP_DELETE, seq, k, b"")

        # 2. Insert tombstone in MemTable
        self.memtable.put(k, b"", seq_num=seq, is_tombstone=True)
        self.stats["deletes"] += 1

        if self.memtable.approx_bytes >= self.memtable_limit:
            self.flush()

    def flush(self) -> None:
        """Flushes MemTable to a new Level 0 SSTable file and rotates the WAL."""
        if self.memtable.count == 0:
            return

        timestamp = int(time.time() * 1000)
        sst_name = f"l0_{timestamp}_{self.seq_num}.sst"
        sst_path = os.path.join(self.l0_dir, sst_name)

        writer = SSTableWriter(sst_path, expected_entries=max(self.memtable.count, 100))
        for k, v, seq, tomb in self.memtable:
            writer.append(k, v, seq_num=seq, is_tombstone=tomb)
        writer.finish()

        # Add new reader to the front of L0 list
        new_reader = SSTableReader(sst_path)
        self.l0_readers.insert(0, new_reader)

        # Clear MemTable and rotate WAL
        self.memtable.clear()
        self.wal.close()
        # Truncate active WAL
        open(self.wal_path, "wb").close()
        self.wal = WAL(self.wal_path, sync_on_write=True)

        self.stats["flushes"] += 1

        # Trigger compaction if Level 0 exceeded table limit
        if len(self.l0_readers) >= self.max_l0_tables:
            self.compact()

    def compact(self) -> None:
        """Merges all Level 0 SSTables into consolidated Level 1 SSTables."""
        if not self.l0_readers:
            return

        timestamp = int(time.time() * 1000)
        out_sst_name = f"l1_{timestamp}_{self.seq_num}.sst"
        out_sst_path = os.path.join(self.l1_dir, out_sst_name)

        # Combine all L0 tables and any overlapping L1 tables
        inputs_to_merge = list(self.l0_readers) + list(self.l1_readers)
        self.compactor.merge_tables(inputs_to_merge, out_sst_path, is_bottom_level=True)

        # Close all compacted readers
        for r in self.l0_readers:
            r.close()
            try:
                os.remove(r.filepath)
            except OSError:
                pass

        for r in self.l1_readers:
            r.close()
            try:
                os.remove(r.filepath)
            except OSError:
                pass

        # Install new Level 1 reader
        self.l0_readers = []
        self.l1_readers = [SSTableReader(out_sst_path)]
        self.stats["compactions"] += 1

    # -------------------------------------------------------------------------
    # Integrated Vector Database Operations
    # -------------------------------------------------------------------------

    def put_vector(self, key: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Indexes a vector in HNSW and stores its associated metadata in the KV store."""
        if self.hnsw is None:
            raise RuntimeError("Vector search not enabled. Provide vector_dim at ChronoDB initialization.")

        # Index in HNSW
        self.hnsw.add(key, vector)

        # Store metadata in KV engine
        meta_payload = json.dumps(metadata or {}).encode("utf-8")
        self.put(f"__vec_meta__{key}", meta_payload)

    def search_vector(self, query: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes approximate nearest neighbor vector search.
        Returns: list of dicts with key, distance, similarity, and stored metadata.
        """
        if self.hnsw is None:
            raise RuntimeError("Vector search not enabled.")

        raw_results = self.hnsw.search(query, k=k)
        results = []

        for node_id, dist in raw_results:
            raw_meta = self.get(f"__vec_meta__{node_id}")
            meta = json.loads(raw_meta.decode("utf-8")) if raw_meta else {}
            results.append({
                "key": node_id,
                "distance": dist,
                "similarity": 1.0 - dist if self.hnsw.metric == "cosine" else 1.0 / (1.0 + dist),
                "metadata": meta,
            })
        return results

    def close(self) -> None:
        """Closes WAL and all SSTable reader file descriptors."""
        self.wal.close()
        for r in self.l0_readers:
            r.close()
        for r in self.l1_readers:
            r.close()
        if self.hnsw is not None:
            self.hnsw.save(self.hnsw_path)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns database diagnostics and operational metrics."""
        disk_bytes = 0
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                disk_bytes += os.path.getsize(os.path.join(root, f))

        return {
            "seq_num": self.seq_num,
            "memtable_keys": self.memtable.count,
            "memtable_approx_bytes": self.memtable.approx_bytes,
            "l0_sstable_count": len(self.l0_readers),
            "l1_sstable_count": len(self.l1_readers),
            "total_disk_bytes": disk_bytes,
            "stats": dict(self.stats),
        }
