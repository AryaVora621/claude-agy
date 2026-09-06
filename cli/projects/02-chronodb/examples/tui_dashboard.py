#!/usr/bin/env python3
"""
ChronoDB Live Terminal Dashboard & Interactive REPL:
Visualizes the internal LSM-Tree architecture:
- SkipList MemTable capacity gauge
- Level 0 overlapping SSTable layout with Bloom filter counters
- Level 1 consolidated SSTables
- Interactive REPL for key-value mutations and vector similarity queries
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chronodb.engine import ChronoDB


def render_dashboard(db: ChronoDB) -> str:
    diag = db.get_diagnostics()
    stats = diag["stats"]

    mem_bytes = diag["memtable_approx_bytes"]
    mem_limit = db.memtable_limit
    mem_pct = min(100, int((mem_bytes / max(mem_limit, 1)) * 100))
    bar_width = 30
    filled = int((mem_pct / 100) * bar_width)
    mem_bar = "█" * filled + "░" * (bar_width - filled)

    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════════╗")
    lines.append("║                   CHRONODB LSM-TREE & VECTOR ENGINE                  ║")
    lines.append("║        Embeddable Storage Architecture with Native HNSW Vector Index ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Seq Num: #{diag['seq_num']:<8} │ Total Disk: {diag['total_disk_bytes']:<8} bytes │ Puts: {stats['puts']:<6} Gets: {stats['gets']:<6} ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append("║ [1] MEMTABLE (SkipList in RAM)                                       ║")
    lines.append(f"║     Keys: {diag['memtable_keys']:<5} │ Buffer: {mem_bytes:<6} / {mem_limit:<6} B ({mem_pct:>3}%)                  ║")
    lines.append(f"║     Capacity: [{mem_bar}]                      ║")
    lines.append("╟──────────────────────────────────────────────────────────────────────╢")
    lines.append("║ [2] LEVEL 0 (Overlapping SSTables on Disk)                           ║")
    if not db.l0_readers:
        lines.append("║     (No Level 0 tables currently - all data in MemTable or Level 1)  ║")
    else:
        for idx, r in enumerate(db.l0_readers[:4]):
            min_k = r.min_key_prefix.decode("utf-8", errors="replace")[:8]
            max_k = r.max_key_prefix.decode("utf-8", errors="replace")[:8]
            fname = os.path.basename(r.filepath)
            lines.append(f"║     SST #{idx + 1}: {fname:<24} [{min_k:>8} .. {max_k:<8}] Entries: {r.total_entries:<4} ║")

    lines.append("╟──────────────────────────────────────────────────────────────────────╢")
    lines.append("║ [3] LEVEL 1 (Consolidated Merged SSTables)                           ║")
    if not db.l1_readers:
        lines.append("║     (No Level 1 tables - await compaction)                           ║")
    else:
        for idx, r in enumerate(db.l1_readers[:3]):
            min_k = r.min_key_prefix.decode("utf-8", errors="replace")[:8]
            max_k = r.max_key_prefix.decode("utf-8", errors="replace")[:8]
            fname = os.path.basename(r.filepath)
            lines.append(f"║     SST #{idx + 1}: {fname:<24} [{min_k:>8} .. {max_k:<8}] Entries: {r.total_entries:<4} ║")

    lines.append("╟──────────────────────────────────────────────────────────────────────╢")
    lines.append(f"║  Bloom Hits: {stats['bloom_hits']:<5} │ Bloom Filter Discards (0 Disk IO): {stats['bloom_negatives']:<6}     ║")
    lines.append(f"║  MemTable Flushes: {stats['flushes']:<4} │ Compactions Completed: {stats['compactions']:<4}                 ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def run_demo():
    data_dir = "/tmp/chronodb_demo_db"
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)

    print("Scaffolding ChronoDB instance at", data_dir)
    db = ChronoDB(
        data_dir=data_dir,
        memtable_limit_bytes=4096,  # 4KB limit to demonstrate flushes quickly
        max_l0_tables=3,
        vector_dim=4,
    )

    print("\n[Step 1] Ingesting test key-value records...")
    for i in range(120):
        db.put(f"sensor:{i:03d}", f"telemetry_reading_celsius_{i * 0.5:.2f}")

    print("\n[Step 2] Ingesting vector embeddings...")
    db.put_vector("doc:nlp", [0.95, 0.05, 0.02, 0.01], {"topic": "Natural Language Processing"})
    db.put_vector("doc:db", [0.02, 0.03, 0.98, 0.12], {"topic": "LSM Storage Engine Architecture"})
    db.put_vector("doc:compiler", [0.10, 0.88, 0.05, 0.20], {"topic": "SSA Compilers and VMs"})

    print(render_dashboard(db))

    print("\n[Step 3] Querying Key-Value Records:")
    print("-> GET sensor:042 =>", db.get("sensor:042"))
    print("-> GET non_existent =>", db.get("non_existent"))

    print("\n[Step 4] Executing HNSW Vector Similarity Search for query [0.05, 0.02, 0.95, 0.10]:")
    results = db.search_vector([0.05, 0.02, 0.95, 0.10], k=2)
    for rank, r in enumerate(results, 1):
        print(f"  Rank #{rank}: Key={r['key']} | Cosine Similarity={r['similarity'] * 100:.1f}% | Topic='{r['metadata'].get('topic')}'")

    print("\n[Step 5] Triggering Manual Compaction:")
    db.flush()
    db.compact()
    print(render_dashboard(db))

    db.close()
    print("\nDemo completed cleanly.")


if __name__ == "__main__":
    run_demo()
