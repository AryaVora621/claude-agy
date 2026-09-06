"""
Chaos Network Partition Tests:
Verifies split-brain resilience:
- Minority partition cannot commit proposals without majority quorum
- Majority partition continues operating and committing transactions
- Upon partition healing, logs are harmonized, uncommitted entries truncated, and state machines converge.
"""

import unittest
import time
from swarmraft.cluster import RaftCluster
from swarmraft.types import NodeRole


class TestNetworkPartition(unittest.TestCase):

    def test_split_brain_partition_and_heal(self):
        # 5-node cluster requires quorum of 3 to commit
        cluster = RaftCluster(node_count=5, election_timeout_range=(0.10, 0.20), heartbeat_interval=0.03)
        try:
            cluster.start()
            leader1 = cluster.wait_for_leader(timeout_sec=3.0)
            self.assertIsNotNone(leader1)

            # 1. Commit initial entry across all 5 nodes
            res = cluster.propose({"op": "SET", "key": "k0", "val": "v0"})
            self.assertTrue(res.success)
            time.sleep(0.1)

            # 2. Partition cluster: put leader1 and 1 peer in minority group
            all_nodes = list(cluster.node_ids)
            l_id = leader1.node_id
            minority_peer = [n for n in all_nodes if n != l_id][0]
            minority = [l_id, minority_peer]
            majority = [n for n in all_nodes if n not in minority]

            # Trigger split-brain partition
            cluster.partition(minority, majority)
            time.sleep(0.1)

            # 3. Try to propose to minority leader: must fail / time out
            min_res = leader1.propose({"op": "SET", "key": "uncommitted", "val": "bad"}, timeout_sec=0.4)
            self.assertFalse(min_res.success, "Minority leader must NOT commit without quorum")

            # 4. Majority partition must elect a new leader and commit new entries
            time.sleep(0.3)
            # Find majority leader
            majority_leaders = [
                cluster.nodes[nid] for nid in majority if cluster.nodes[nid].role == NodeRole.LEADER
            ]
            self.assertGreaterEqual(len(majority_leaders), 1, "Majority partition must elect a leader")
            maj_leader = majority_leaders[0]

            maj_res = maj_leader.propose({"op": "SET", "key": "k1", "val": "v1"}, timeout_sec=1.0)
            self.assertTrue(maj_res.success, "Majority partition must be able to commit")

            # 5. Heal partition!
            cluster.heal()
            time.sleep(0.4)

            # 6. Verify convergence across all 5 nodes
            for nid in all_nodes:
                node = cluster.nodes[nid]
                # 'k0' and 'k1' must exist
                self.assertEqual(node.state_machine.get("k0"), "v0")
                self.assertEqual(node.state_machine.get("k1"), "v1")
                # 'uncommitted' must have been rolled back / truncated!
                self.assertIsNone(node.state_machine.get("uncommitted"))
        finally:
            cluster.stop()


if __name__ == "__main__":
    unittest.main()
