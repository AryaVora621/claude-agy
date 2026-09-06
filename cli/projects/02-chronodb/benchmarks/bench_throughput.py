#!/usr/bin/env python3
"""
ChronoDB Performance Benchmark:
Evaluates:
- Sequential Write throughput (ops/sec) with WAL fsync and MemTable indexing
- Random Read throughput (ops/sec) across RAM and SSTables
- Negative Lookup throughput (measuring Bloom filter elimination efficiency)
- HNSW Vector Similarity Search latency and Queries-Per-Second (QPS)
"""

import os
import sys
import time
import random
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chronodb.engine import ChronoDB


def run_benchmark():
    data_dir = "/tmp/chronodb_benchmark_db"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    print("=" * 65)
    print("           CHRONODB STORAGE ENGINE BENCHMARK SUITE")
    print("=" * 65)

    num_records = 2000
    vector_dim = 16
    num_vectors = 500

    db = ChronoDB(
        data_dir=data_dir,
        memtable_limit_bytes=32 * 1024,  # 32 KB
        max_l0_tables=4,
        vector_dim=vector_dim,
    )

    # -------------------------------------------------------------------------
    # 1. Sequential Write Benchmark
    # -------------------------------------------------------------------------
    print(f"\n[Bench 1] Inserting {num_records} Key-Value Records (WAL + MemTable + Flushes)...")
    t0 = time.perf_counter()
    for i in range(num_records):
        db.put(f"perf_key_{i:06d}", f"simulated_payload_data_block_{i:06d}")
    write_time = time.perf_counter() - t0
    write_qps = num_records / write_time
    print(f"  -> Total Time: {write_time:.3f} s")
    print(f"  -> Write Throughput: {write_qps:,.1f} ops/sec")

    # -------------------------------------------------------------------------
    # 2. Random Read Benchmark (Existing Keys)
    # -------------------------------------------------------------------------
    read_samples = 1000
    print(f"\n[Bench 2] Performing {read_samples} Random Key Reads...")
    sample_indices = [random.randint(0, num_records - 1) for _ in range(read_samples)]

    t0 = time.perf_counter()
    hits = 0
    for idx in sample_indices:
        val = db.get(f"perf_key_{idx:06d}")
        if val is not None:
            hits += 1
    read_time = time.perf_counter() - t0
    read_qps = read_samples / read_time
    print(f"  -> Total Time: {read_time:.3f} s | Hits: {hits}/{read_samples}")
    print(f"  -> Read Throughput: {read_qps:,.1f} ops/sec")

    # -------------------------------------------------------------------------
    # 3. Negative Lookup Benchmark (Bloom Filter Zero-IO Verification)
    # -------------------------------------------------------------------------
    neg_samples = 2000
    print(f"\n[Bench 3] Performing {neg_samples} Negative Lookups (Keys Not in DB)...")
    t0 = time.perf_counter()
    neg_hits = 0
    for i in range(neg_samples):
        val = db.get(f"non_existent_key_{i:06d}")
        if val is not None:
            neg_hits += 1
    neg_time = time.perf_counter() - t0
    neg_qps = neg_samples / neg_time
    print(f"  -> Total Time: {neg_time:.3f} s | False Positives: {neg_hits}/{neg_samples}")
    print(f"  -> Negative Lookup Throughput: {neg_qps:,.1f} ops/sec")

    # -------------------------------------------------------------------------
    # 4. HNSW Vector Index & Query Benchmark
    # -------------------------------------------------------------------------
    print(f"\n[Bench 4] Indexing {num_vectors} Vectors (Dim={vector_dim}) into HNSW Graph...")
    vectors = {}
    for i in range(num_vectors):
        vec = [random.gauss(0, 1) for _ in range(vector_dim)]
        key = f"vec_item_{i:04d}"
        vectors[key] = vec
        db.put_vector(key, vec, {"idx": i})

    query_samples = 100
    print(f"Running {query_samples} HNSW Top-5 ANN Vector Queries...")
    t0 = time.perf_counter()
    for _ in range(query_samples):
        q_vec = [random.gauss(0, 1) for _ in range(vector_dim)]
        res = db.search_vector(q_vec, k=5)
    vec_time = time.perf_counter() - t0
    vec_qps = query_samples / vec_time
    avg_latency_ms = (vec_time / query_samples) * 1000.0
    print(f"  -> Total Time: {vec_time:.3f} s")
    print(f"  -> Vector Search QPS: {vec_qps:,.1f} queries/sec")
    print(f"  -> Average Latency: {avg_latency_ms:.2f} ms per vector query")

    diag = db.get_diagnostics()
    print("\n[Engine Stats Summary]")
    print(f"  MemTable Flushes: {diag['stats']['flushes']}")
    print(f"  Compactions: {diag['stats']['compactions']}")
    print(f"  Total SSTables (L0 + L1): {diag['l0_sstable_count'] + diag['l1_sstable_count']}")
    print(f"  Total Disk Usage: {diag['total_disk_bytes'] / 1024:.1f} KB")

    db.close()
    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    run_benchmark()
