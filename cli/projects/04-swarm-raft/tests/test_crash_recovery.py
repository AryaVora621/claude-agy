"""
Node Crash & Persistence Recovery Tests:
Verifies that nodes save current_term, voted_for, and log entries durably
and restore identically upon reboot after power cuts.
"""

import unittest
import tempfile
import shutil
import time
from swarmraft.cluster import RaftCluster
from swarmraft.types import NodeRole


class TestCrashRecovery(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_leader_crash_and_disk_reboot(self):
        # 3-node cluster with disk persistence
        cluster = RaftCluster(
            node_count=3,
            storage_dir=self.test_dir,
            election_timeout_range=(0.10, 0.20),
            heartbeat_interval=0.03
        )
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            # Propose 3 commands
            for i in range(3):
                res = cluster.propose({"op": "SET", "key": f"persist_{i}", "val": f"val_{i}"})
                self.assertTrue(res.success)

            time.sleep(0.1)

            # Crash the leader
            crashed_id = leader.node_id
            cluster.crash_node(crashed_id)

            # Restart the crashed node from disk
            cluster.restart_node(crashed_id)
            restarted_node = cluster.nodes[crashed_id]

            # Verify persisted term and log
            self.assertGreaterEqual(restarted_node.current_term, leader.current_term)
            self.assertGreaterEqual(restarted_node.last_log_index(), 3)
            for i in range(3):
                self.assertEqual(restarted_node.state_machine.get(f"persist_{i}"), f"val_{i}")
        finally:
            cluster.stop()

    def test_full_cluster_power_cycle(self):
        # 1. Start cluster and commit state
        cluster1 = RaftCluster(
            node_count=3,
            storage_dir=self.test_dir,
            election_timeout_range=(0.10, 0.20),
            heartbeat_interval=0.03
        )
        try:
            cluster1.start()
            leader = cluster1.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            for i in range(5):
                res = cluster1.propose({"op": "SET", "key": f"cycle_{i}", "val": f"data_{i}"})
                self.assertTrue(res.success)

            time.sleep(0.1)
        finally:
            # Full cluster shutdown (simulating data center power outage)
            cluster1.stop()

        # 2. Boot fresh cluster pointing to identical storage files
        cluster2 = RaftCluster(
            node_count=3,
            storage_dir=self.test_dir,
            election_timeout_range=(0.10, 0.20),
            heartbeat_interval=0.03
        )
        try:
            cluster2.start()
            new_leader = cluster2.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(new_leader)

            # Verify all past data restored
            for nid, node in cluster2.nodes.items():
                for i in range(5):
                    self.assertEqual(node.state_machine.get(f"cycle_{i}"), f"data_{i}")

            # New proposal after reboot
            res = cluster2.propose({"op": "SET", "key": "post_reboot", "val": "alive"})
            self.assertTrue(res.success)
            time.sleep(0.1)
            self.assertEqual(new_leader.state_machine.get("post_reboot"), "alive")
        finally:
            cluster2.stop()


if __name__ == "__main__":
    unittest.main()
