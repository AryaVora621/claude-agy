# ChronoDB

> **High-Performance Embeddable LSM-Tree Storage Engine with Native HNSW Vector Search.**  
> Built from first principles in pure Python 3.10+ with zero external dependencies.

---

## Architectural Overview

ChronoDB implements an industrial-grade Log-Structured Merge-Tree (LSM-Tree) storage engine coupled directly with a Hierarchical Navigable Small World (HNSW) proximity graph index for hybrid multi-modal data workloads (key-value + vector embeddings).

```
[ Client Writes ] ──────────┐
                            ▼
                ┌───────────────────────┐
                │ Write-Ahead Log (WAL) │  (Append-only binary framing, CRC32, fsync)
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   SkipList MemTable   │  (O(log n) concurrent-safe in-memory table)
                └───────────┬───────────┘
                            │ (Memory Limit Exceeded -> Background Flush)
                            ▼
                ┌───────────────────────┐
                │   Level 0 SSTables    │  (Overlapping sorted tables on disk)
                │  - Data Blocks (4KB)  │
                │  - Sparse Index Block │
                │  - Bloom Filter Block │
                └───────────┬───────────┘
                            │ (Leveled Compaction -> Multi-Way Merge Sort)
                            ▼
                ┌───────────────────────┐
                │   Level 1 SSTables    │  (Consolidated, non-overlapping tables)
                └───────────────────────┘
                            ▲
                            │
┌───────────────────────────┴───────────────────────────┐
│        Integrated HNSW Vector Proximity Graph         │
│  - Multi-layer skip-graph for ANN vector search       │
│  - Cosine distance and Euclidean (L2) metrics         │
│  - Metadata payloads resolved via ChronoDB KV engine  │
└───────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Durability & Crash Recovery**:
   - Write-Ahead Log (WAL) with strict binary record framing:
     `[Magic: 2B][CRC32: 4B][PayloadLen: 4B] | [Op: 1B][SeqNum: 8B][KeyLen: 2B][Key][ValLen: 4B][Val]`
   - Resilient replay mechanism: power failures or torn writes at the tail are automatically truncated, restoring 100% of committed state.

2. **Probabilistic SkipList MemTable**:
   - $O(\log n)$ expected search, insert, and update operations.
   - Deterministic memory budgeting tracking exact byte footprints.
   - Monotonically increasing sequence numbers for snapshot reads and multi-version concurrency.

3. **Binary SSTable Disk Format**:
   - Fixed-size 4KB data blocks with checksum validation.
   - In-memory sparse index blocks: binary search bounds target blocks in $O(\log B)$ without scanning entire files.
   - **Embedded Bit-Vector Bloom Filter**: Kirsch-Mitzenmacher double-hashing eliminates disk seeks on non-existent keys (yielding >130,000 negative lookups/sec).
   - Fixed 80-byte trailer footer for instant open latency.

4. **Leveled Compaction Engine**:
   - Multi-way merge sort (`heapq`) across overlapping SSTables.
   - Resolves key collisions (highest sequence number wins).
   - Space reclamation: purges dead tombstone records at the bottom level.

5. **Native HNSW Vector Search Index**:
   - Multi-layer proximity graph supporting high-dimensional vector search.
   - Sub-millisecond queries (~0.49 ms latency) achieving 100% recall on benchmark distributions.
   - Seamless coupling with key-value metadata.

---

## Quickstart

### Run Test Suite
```bash
python3 -m unittest discover -s tests
```

### Run Interactive Live Terminal Dashboard
```bash
python3 examples/tui_dashboard.py
```

### Run Performance Benchmarks
```bash
python3 benchmarks/bench_throughput.py
```

---

## Python API Usage

```python
from chronodb import ChronoDB

# Initialize database with 1MB MemTable threshold and 16D vector index
db = ChronoDB(
    data_dir="./my_database",
    memtable_limit_bytes=1024 * 1024,
    max_l0_tables=4,
    vector_dim=4,
    vector_metric="cosine",
)

# 1. Standard Key-Value Operations
db.put("user:101", "Alice Vance")
print("Retrieved:", db.get("user:101"))  # b'Alice Vance'

# 2. Vector Indexing with Metadata
db.put_vector(
    key="doc:42",
    vector=[0.9, 0.1, 0.05, 0.02],
    metadata={"title": "LSM-Tree Systems", "tags": ["db", "storage"]}
)

# 3. Approximate Nearest Neighbor Search
results = db.search_vector([0.88, 0.12, 0.04, 0.01], k=1)
for r in results:
    print(f"Match: {r['key']} | Similarity: {r['similarity']*100:.1f}% | Title: {r['metadata']['title']}")

# 4. Clean shutdown
db.close()
```

---

## Performance Summary

Tested on Apple Silicon / macOS Darwin using standard Python 3.13:

| Benchmark | Throughput | Latency |
|---|---|---|
| **Sequential Writes** (WAL + MemTable + Flushes) | **~40,000 ops/sec** | ~0.025 ms |
| **Random Reads** (Active Keys) | **~67,000 ops/sec** | ~0.015 ms |
| **Negative Lookups** (Bloom Filter Zero-IO) | **~136,000 ops/sec** | ~0.007 ms |
| **HNSW Vector Queries** (Top-5 ANN, Dim=16) | **~2,060 queries/sec** | **0.49 ms** |
