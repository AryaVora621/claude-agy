#!/usr/bin/env python3
"""
SwarmRaft Interactive Terminal Dashboard & Cluster Controller.
Demonstrates live Raft consensus, split-brain partitioning,
replicated state machine operations, and node crash/recovery.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarmraft.cluster import RaftCluster
from swarmraft.visualizer import render_cluster_ascii


def main():
    print("Initializing 5-Node SwarmRaft Cluster...")
    cluster = RaftCluster(
        node_count=5,
        election_timeout_range=(0.15, 0.30),
        heartbeat_interval=0.05
    )
    cluster.start()

    try:
        print("Waiting for leader election...")
        leader = cluster.wait_for_leader(timeout_sec=4.0)
        if leader:
            print(f"Leader elected: {leader.node_id} (Term {leader.current_term})\n")
        else:
            print("Leader election timed out.")

        # Display initial dashboard
        print(render_cluster_ascii(cluster))

        # Perform a sequence of automated demonstrations
        time.sleep(1.0)
        print("\n" + "=" * 80)
        print(" [Step 1] Committing Initial Key-Value State via Consensus")
        print("=" * 80)
        proposals = [
            {"op": "SET", "key": "cluster_name", "val": "SwarmAlpha"},
            {"op": "SET", "key": "max_shards", "val": 64},
            {"op": "SET", "key": "status", "val": "HEALTHY"}
        ]
        for p in proposals:
            res = cluster.propose(p, timeout_sec=2.0)
            print(f"  -> Proposed {p['op']} {p['key']}={p.get('val')} : Success={res.success}, Index={res.applied_index}")

        time.sleep(0.5)
        print("\n" + render_cluster_ascii(cluster))

        time.sleep(1.0)
        print("\n" + "=" * 80)
        print(" [Step 2] Chaos Simulation: Injecting Split-Brain Network Partition")
        print(" Partitions: Group A [node_1, node_2]  vs  Group B [node_3, node_4, node_5]")
        print("=" * 80)
        cluster.partition(["node_1", "node_2"], ["node_3", "node_4", "node_5"])

        time.sleep(1.0)
        # Attempt proposal to minority
        n1 = cluster.nodes["node_1"]
        print("  Attempting proposal to minority node_1...")
        res_min = n1.propose({"op": "SET", "key": "split_key", "val": "ghost"}, timeout_sec=0.5)
        print(f"  -> Minority Proposal: Success={res_min.success} (Error: {res_min.error})")

        # Proposal to majority
        print("  Proposing to majority cluster...")
        res_maj = cluster.propose({"op": "SET", "key": "majority_vote", "val": "COMMITTED"}, timeout_sec=2.0)
        print(f"  -> Majority Proposal: Success={res_maj.success}, Index={res_maj.applied_index}")

        time.sleep(0.5)
        print("\n" + render_cluster_ascii(cluster))

        time.sleep(1.0)
        print("\n" + "=" * 80)
        print(" [Step 3] Healing Network Partition & Verifying State Machine Convergence")
        print("=" * 80)
        cluster.heal()
        time.sleep(1.0)
        print(render_cluster_ascii(cluster))

        print("\n" + "=" * 80)
        print(" [Step 4] Simulating Crash and Reboot of Current Leader")
        print("=" * 80)
        curr_leader = cluster.get_leader()
        if curr_leader:
            crashed_id = curr_leader.node_id
            print(f"  Crashing leader {crashed_id}...")
            cluster.crash_node(crashed_id)
            time.sleep(0.5)
            print(f"  Waiting for new leader election among remaining nodes...")
            new_leader = cluster.wait_for_leader(timeout_sec=3.0)
            if new_leader:
                print(f"  New leader elected: {new_leader.node_id} (Term {new_leader.current_term})")
            print(f"  Rebooting crashed node {crashed_id}...")
            cluster.restart_node(crashed_id)
            time.sleep(0.5)
            print("\n" + render_cluster_ascii(cluster))

        print("\n" + "=" * 80)
        print(" CLUSTER DEMONSTRATION COMPLETE")
        print("=" * 80)

    finally:
        cluster.stop()


if __name__ == "__main__":
    main()
