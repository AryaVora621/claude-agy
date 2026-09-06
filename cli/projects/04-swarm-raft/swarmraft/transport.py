"""
Simulated Network Transport and Partition Simulator.
Allows injecting realistic distributed systems faults:
- Dynamic split-brain network partitions (e.g. {N1, N2} vs {N3, N4, N5})
- Directional link blackholing and latency jitter
- Packet drop rates and node crash/reboot simulation
- Synchronous and asynchronous RPC dispatch
"""

from __future__ import annotations
import random
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class NetworkStats:
    """Telemetry counters for the simulated network."""
    total_rpcs_sent: int = 0
    total_rpcs_delivered: int = 0
    total_rpcs_dropped: int = 0
    total_bytes_transferred: int = 0
    partition_drops: int = 0


class SimulatedNetwork:
    """
    In-memory virtual network bus connecting simulated Raft nodes.
    Supports dynamic partitioning, latency injection, and chaos engineering.
    """

    def __init__(self, seed: Optional[int] = None):
        self._lock = threading.RLock()
        self._handlers: Dict[str, Dict[str, Callable[[Any], Any]]] = {}
        self._alive_nodes: Set[str] = set()

        # Partitions: each node is in a partition group index.
        # Nodes can only communicate if they share the same partition group.
        self._node_partition_map: Dict[str, int] = {}

        # Directional link blocks: (from_id, to_id) -> blocked (bool)
        self._blocked_links: Set[Tuple[str, str]] = set()

        # Fault injection parameters
        self.drop_rate: float = 0.0
        self.base_latency_ms: float = 0.0
        self.jitter_ms: float = 0.0

        # Telemetry
        self.stats = NetworkStats()
        self._rng = random.Random(seed)

    def register_node(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self._handlers:
                self._handlers[node_id] = {}
            self._alive_nodes.add(node_id)
            if node_id not in self._node_partition_map:
                self._node_partition_map[node_id] = 0

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            self._alive_nodes.discard(node_id)
            if node_id in self._handlers:
                del self._handlers[node_id]
            if node_id in self._node_partition_map:
                del self._node_partition_map[node_id]

    def register_handler(self, node_id: str, method: str, handler: Callable[[Any], Any]) -> None:
        with self._lock:
            if node_id not in self._handlers:
                self._handlers[node_id] = {}
            self._handlers[node_id][method] = handler

    def set_node_alive(self, node_id: str, alive: bool) -> None:
        """Simulates node crash or restart."""
        with self._lock:
            if alive:
                self._alive_nodes.add(node_id)
            else:
                self._alive_nodes.discard(node_id)

    def is_node_alive(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._alive_nodes

    # ==================== Partition Management ====================

    def create_partitions(self, groups: List[List[str]]) -> None:
        """
        Splits nodes into disjoint partition sets.
        Nodes within the same group can talk to each other;
        nodes in different groups cannot.
        Example: create_partitions([["node1", "node2"], ["node3", "node4", "node5"]])
        """
        with self._lock:
            self._node_partition_map.clear()
            for group_idx, group in enumerate(groups):
                for node in group:
                    self._node_partition_map[node] = group_idx

    def heal_partitions(self) -> None:
        """Heals all network partitions, restoring full mesh connectivity."""
        with self._lock:
            for node in self._handlers:
                self._node_partition_map[node] = 0
            self._blocked_links.clear()

    def block_link(self, from_id: str, to_id: str) -> None:
        """Blocks a directional link between two nodes."""
        with self._lock:
            self._blocked_links.add((from_id, to_id))

    def unblock_link(self, from_id: str, to_id: str) -> None:
        with self._lock:
            self._blocked_links.discard((from_id, to_id))

    def is_connected(self, from_id: str, to_id: str) -> bool:
        """Returns True if a packet can travel from from_id to to_id."""
        with self._lock:
            if from_id not in self._alive_nodes or to_id not in self._alive_nodes:
                return False
            if (from_id, to_id) in self._blocked_links:
                return False

            part_from = self._node_partition_map.get(from_id, 0)
            part_to = self._node_partition_map.get(to_id, 0)
            return part_from == part_to

    # ==================== RPC Dispatch ====================

    def call(
        self,
        from_id: str,
        to_id: str,
        method: str,
        args: Any,
        timeout_sec: float = 0.5
    ) -> Optional[Any]:
        """
        Synchronously dispatches an RPC from from_id to to_id with timeout.
        Returns the reply object, or None if dropped/partitioned/timed out.
        """
        with self._lock:
            self.stats.total_rpcs_sent += 1

            if not self.is_connected(from_id, to_id):
                self.stats.partition_drops += 1
                self.stats.total_rpcs_dropped += 1
                return None

            if self.drop_rate > 0 and self._rng.random() < self.drop_rate:
                self.stats.total_rpcs_dropped += 1
                return None

            node_handlers = self._handlers.get(to_id)
            if not node_handlers or method not in node_handlers:
                self.stats.total_rpcs_dropped += 1
                return None

            handler = node_handlers[method]

        # Simulate latency if configured
        if self.base_latency_ms > 0:
            jitter = self._rng.uniform(-self.jitter_ms, self.jitter_ms) if self.jitter_ms > 0 else 0
            delay = max(0.0, (self.base_latency_ms + jitter) / 1000.0)
            if delay > timeout_sec:
                with self._lock:
                    self.stats.total_rpcs_dropped += 1
                return None
            time.sleep(delay)

        try:
            reply = handler(args)
        except Exception:
            with self._lock:
                self.stats.total_rpcs_dropped += 1
            return None

        # Check return path connectivity
        with self._lock:
            if not self.is_connected(to_id, from_id):
                self.stats.partition_drops += 1
                self.stats.total_rpcs_dropped += 1
                return None

            self.stats.total_rpcs_delivered += 1

        return reply

    def call_async(
        self,
        from_id: str,
        to_id: str,
        method: str,
        args: Any,
        callback: Callable[[Optional[Any]], None],
        timeout_sec: float = 0.5
    ) -> threading.Thread:
        """Asynchronously dispatches an RPC in a background thread."""
        def _worker():
            reply = self.call(from_id, to_id, method, args, timeout_sec)
            callback(reply)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t
