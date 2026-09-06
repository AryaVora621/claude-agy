#!/usr/bin/env python3
"""
SwarmRaft Consensus Throughput & Latency Benchmarks:
Measures transactions per second (TPS), quorum commit latency percentiles (p50, p95, p99),
and RPC efficiency on multi-node clusters.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarmraft.cluster import RaftCluster


def run_benchmark(node_count: int, num_proposals: int = 50):
    cluster = RaftCluster(
        node_count=node_count,
        election_timeout_range=(0.10, 0.20),
        heartbeat_interval=0.03
    )
    cluster.start()

    try:
        leader = cluster.wait_for_leader(timeout_sec=3.0)
        if not leader:
            print(f"Failed to elect leader for {node_count}-node cluster")
            return

        latencies_ms = []
        rpcs_start = cluster.transport.stats.total_rpcs_sent
        start_time = time.perf_counter()

        successful = 0
        for i in range(num_proposals):
            t0 = time.perf_counter()
            res = cluster.propose({"op": "SET", "key": f"bench_{i}", "val": i}, timeout_sec=2.0)
            t1 = time.perf_counter()
            if res.success:
                successful += 1
                latencies_ms.append((t1 - t0) * 1000.0)

        total_time = time.perf_counter() - start_time
        rpcs_total = cluster.transport.stats.total_rpcs_sent - rpcs_start

        latencies_ms.sort()
        p50 = latencies_ms[int(len(latencies_ms) * 0.50)] if latencies_ms else 0
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0
        p99 = latencies_ms[int(len(latencies_ms) * 0.99)] if latencies_ms else 0
        tps = successful / total_time if total_time > 0 else 0
        rpcs_per_tx = rpcs_total / successful if successful > 0 else 0

        print(f"\n[{node_count}-Node Cluster Results ({num_proposals} proposals)]:")
        print(f"  Success Rate          : {successful}/{num_proposals} ({successful / num_proposals * 100:.1f}%)")
        print(f"  Total Duration        : {total_time * 1000:.2f} ms")
        print(f"  Throughput (TPS)      : {tps:.1f} commits/sec")
        print(f"  Latency p50 (median)  : {p50:.2f} ms")
        print(f"  Latency p95           : {p95:.2f} ms")
        print(f"  Latency p99           : {p99:.2f} ms")
        print(f"  RPCs per Commit       : {rpcs_per_tx:.1f} RPCs/tx")

    finally:
        cluster.stop()


def main():
    print("=" * 70)
    print("       SWARMRAFT CONSENSUS THROUGHPUT & LATENCY BENCHMARKS")
    print("=" * 70)

    print("\nBenchmarking 3-Node Cluster (Quorum = 2)...")
    run_benchmark(node_count=3, num_proposals=60)

    print("\nBenchmarking 5-Node Cluster (Quorum = 3)...")
    run_benchmark(node_count=5, num_proposals=60)

    print("\n" + "=" * 70)
    print(" BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
