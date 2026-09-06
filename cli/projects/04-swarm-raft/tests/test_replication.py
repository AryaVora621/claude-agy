"""
Log Replication & Linearizability Tests:
Verifies command replication, quorum commit, and state machine convergence.
"""

import unittest
import time
from swarmraft.cluster import RaftCluster
from swarmraft.types import NodeRole


class TestLogReplication(unittest.TestCase):

    def test_basic_command_replication(self):
        cluster = RaftCluster(node_count=3, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            # Propose SET command
            res = cluster.propose({"op": "SET", "key": "user:1", "val": "Alice"})
            self.assertTrue(res.success, f"Proposal failed: {res.error}")
            self.assertEqual(res.applied_index, 1)

            # Wait for replication to propagate to followers
            time.sleep(0.1)

            # Verify state machine on all reachable nodes
            for nid, node in cluster.nodes.items():
                val = node.state_machine.get("user:1")
                self.assertEqual(val, "Alice", f"Node {nid} has incorrect value {val}")
                self.assertEqual(node.commit_index, 1)
        finally:
            cluster.stop()

    def test_multiple_sequential_commands(self):
        cluster = RaftCluster(node_count=3, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            # Send 5 sequential proposals
            for i in range(5):
                res = cluster.propose({"op": "SET", "key": f"key_{i}", "val": f"val_{i}"})
                self.assertTrue(res.success)
                self.assertEqual(res.applied_index, i + 1)

            time.sleep(0.1)

            # Verify all keys exist on all nodes
            for nid, node in cluster.nodes.items():
                self.assertEqual(node.commit_index, 5)
                for i in range(5):
                    self.assertEqual(node.state_machine.get(f"key_{i}"), f"val_{i}")
        finally:
            cluster.stop()

    def test_follower_redirect_behavior(self):
        cluster = RaftCluster(node_count=3, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            # Find a follower
            followers = [n for n in cluster.nodes.values() if n.role == NodeRole.FOLLOWER]
            self.assertGreater(len(followers), 0)
            follower = followers[0]

            # Directly propose to follower - must fail and redirect
            res = follower.propose({"op": "SET", "key": "foo", "val": "bar"})
            self.assertFalse(res.success)
            self.assertEqual(res.leader_id, leader.node_id)
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
