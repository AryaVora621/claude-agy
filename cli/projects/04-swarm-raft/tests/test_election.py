"""
Raft Leader Election Tests:
Verifies single leader invariant, heartbeat suppression, and leader re-election upon failure.
"""

import unittest
import time
from swarmraft.cluster import RaftCluster
from swarmraft.types import NodeRole


class TestLeaderElection(unittest.TestCase):

    def test_single_leader_elected(self):
        cluster = RaftCluster(node_count=3, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader, "A leader should be elected within timeout")

            # Verify election safety: at most one leader in term
            leaders = [n for n in cluster.nodes.values() if n.role == NodeRole.LEADER]
            self.assertEqual(len(leaders), 1, "There should be exactly one leader")
            self.assertGreaterEqual(leader.current_term, 1)

            # Let heartbeats run for 150ms and ensure leader remains stable
            time.sleep(0.15)
            self.assertEqual(leader.role, NodeRole.LEADER)
        finally:
            cluster.stop()

    def test_reelection_after_leader_crash(self):
        cluster = RaftCluster(node_count=5, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            initial_leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(initial_leader)
            initial_term = initial_leader.current_term

            # Crash the current leader
            crashed_id = initial_leader.node_id
            cluster.crash_node(crashed_id)

            # The remaining 4 nodes form a quorum and must elect a new leader in a higher term
            new_leader = cluster.wait_for_leader(timeout_sec=4.0)
            self.assertIsNotNone(new_leader, "Remaining nodes must elect a new leader")
            self.assertNotEqual(new_leader.node_id, crashed_id)
            self.assertGreater(new_leader.current_term, initial_term)
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
