"""
Log Compaction & Snapshot Catch-up Tests:
Verifies log pruning, memory bounds, and InstallSnapshot RPC synchronization.
"""

import unittest
import time
from swarmraft.cluster import RaftCluster
from swarmraft.types import NodeRole


class TestLogSnapshotting(unittest.TestCase):

    def test_log_compaction_and_slow_follower_catchup(self):
        # 3-node cluster
        cluster = RaftCluster(node_count=3, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader)

            # Choose a follower (not leader) to isolate
            followers = [nid for nid in cluster.node_ids if nid != leader.node_id]
            slow_follower = followers[0]

            for nid in cluster.node_ids:
                if nid != slow_follower:
                    cluster.transport.block_link(nid, slow_follower)
                    cluster.transport.block_link(slow_follower, nid)

            # Propose 15 commands committed by remaining quorum
            for i in range(15):
                res = cluster.propose({"op": "SET", "key": f"k_{i}", "val": f"v_{i}"})
                self.assertTrue(res.success, f"Proposal {i} failed: {res.error}")

            # Compact the leader's log (keep only up to last 2 entries)
            compacted = leader.compact_log(max_log_length=2)
            self.assertTrue(compacted, "Leader log should have compacted")
            self.assertGreater(leader.last_included_index, 0)
            self.assertLess(len(leader.log), 15)

            # Unblock slow_follower
            for nid in cluster.node_ids:
                if nid != slow_follower:
                    cluster.transport.unblock_link(nid, slow_follower)
                    cluster.transport.unblock_link(slow_follower, nid)

            # Give leader time to send InstallSnapshot to slow_follower
            time.sleep(0.5)

            # Slow follower must now have caught up completely!
            follower_node = cluster.nodes[slow_follower]
            self.assertGreaterEqual(follower_node.last_included_index, leader.last_included_index)
            for i in range(15):
                self.assertEqual(follower_node.state_machine.get(f"k_{i}"), f"v_{i}")
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
