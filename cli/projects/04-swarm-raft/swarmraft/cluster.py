"""
Raft Cluster Coordinator.
Orchestrates multi-node clusters, handles client routing,
and coordinates fault injection and network partitions.
"""

from __future__ import annotations
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from swarmraft.types import NodeRole, ProposalResult
from swarmraft.transport import SimulatedNetwork
from swarmraft.storage import RaftStorage, MemoryStorage, FileStorage
from swarmraft.state_machine import KVStateMachine
from swarmraft.node import RaftNode


class RaftCluster:
    """
    Manages a cluster of Raft nodes operating over a simulated network.
    """

    def __init__(
        self,
        node_ids: Optional[List[str]] = None,
        node_count: int = 5,
        storage_dir: Optional[str] = None,
        election_timeout_range: Tuple[float, float] = (0.15, 0.30),
        heartbeat_interval: float = 0.05,
    ):
        if not node_ids:
            node_ids = [f"node_{i + 1}" for i in range(node_count)]

        self.node_ids = list(node_ids)
        self.transport = SimulatedNetwork()
        self.storage_dir = storage_dir
        self.nodes: Dict[str, RaftNode] = {}
        self._lock = threading.Lock()

        # Instantiate nodes
        for nid in self.node_ids:
            if storage_dir:
                storage = FileStorage(f"{storage_dir}/{nid}_state.json")
            else:
                storage = MemoryStorage()

            node = RaftNode(
                node_id=nid,
                peers=self.node_ids,
                transport=self.transport,
                storage=storage,
                state_machine=KVStateMachine(),
                election_timeout_range=election_timeout_range,
                heartbeat_interval=heartbeat_interval
            )
            self.nodes[nid] = node

    def start(self) -> None:
        """Starts all nodes in the cluster."""
        with self._lock:
            for node in self.nodes.values():
                node.start()

    def stop(self) -> None:
        """Stops all nodes in the cluster."""
        with self._lock:
            for node in self.nodes.values():
                node.stop()

    def get_leader(self) -> Optional[RaftNode]:
        """Returns the currently active leader node, or None."""
        with self._lock:
            for node in self.nodes.values():
                if node.role == NodeRole.LEADER and self.transport.is_node_alive(node.node_id):
                    return node
            return None

    def wait_for_leader(self, timeout_sec: float = 5.0) -> Optional[RaftNode]:
        """Polls until a leader is elected or timeout expires."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            leader = self.get_leader()
            if leader:
                return leader
            time.sleep(0.05)
        return None

    def propose(self, command: Any, timeout_sec: float = 2.0, max_retries: int = 3) -> ProposalResult:
        """
        Sends a client proposal to the cluster, auto-routing to current leader.
        """
        deadline = time.time() + timeout_sec
        attempts = 0

        while attempts < max_retries and time.time() < deadline:
            attempts += 1
            leader = self.get_leader()
            if not leader:
                time.sleep(0.1)
                continue

            remaining_timeout = max(0.2, deadline - time.time())
            res = leader.propose(command, timeout_sec=remaining_timeout)
            if res.success:
                return res

            time.sleep(0.1)

        return ProposalResult(
            success=False,
            error="Failed to propose command: no leader reached or consensus timed out"
        )

    # ==================== Fault Injection API ====================

    def partition(self, group1: List[str], group2: List[str]) -> None:
        """Creates a two-way network partition."""
        self.transport.create_partitions([group1, group2])

    def heal(self) -> None:
        """Heals all network partitions."""
        self.transport.heal_partitions()

    def crash_node(self, node_id: str) -> None:
        """Simulates node crash/power cut."""
        if node_id in self.nodes:
            self.nodes[node_id].stop()

    def restart_node(self, node_id: str) -> None:
        """Restarts a previously crashed node."""
        if node_id in self.nodes:
            old_node = self.nodes[node_id]
            # Create fresh node instance re-reading persisted state
            new_node = RaftNode(
                node_id=node_id,
                peers=self.node_ids,
                transport=self.transport,
                storage=old_node.storage,
                state_machine=KVStateMachine(),
                election_timeout_range=old_node.election_timeout_range,
                heartbeat_interval=old_node.heartbeat_interval
            )
            self.nodes[node_id] = new_node
            new_node.start()
