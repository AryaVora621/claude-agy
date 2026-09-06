"""
Hierarchical Navigable Small World (HNSW) Vector Index:
State-of-the-art multi-layer proximity graph for high-dimensional Approximate Nearest Neighbor (ANN) search.
Implemented purely in standard Python with zero external dependencies.
Supports Cosine Similarity, Euclidean (L2) distance, and dynamic k-NN querying.
"""

from __future__ import annotations
import math
import random
import heapq
import json
import struct
from typing import List, Dict, Tuple, Optional, Set, Callable


def euclidean_distance(u: List[float], v: List[float]) -> float:
    """Computes Euclidean (L2) distance between two vectors."""
    acc = 0.0
    for a, b in zip(u, v):
        diff = a - b
        acc += diff * diff
    return math.sqrt(acc)


def cosine_distance(u: List[float], v: List[float]) -> float:
    """Computes Cosine Distance = 1.0 - Cosine_Similarity."""
    dot = 0.0
    norm_u = 0.0
    norm_v = 0.0
    for a, b in zip(u, v):
        dot += a * b
        norm_u += a * a
        norm_v += b * b

    if norm_u <= 0.0 or norm_v <= 0.0:
        return 1.0

    sim = dot / (math.sqrt(norm_u) * math.sqrt(norm_v))
    # Clamp for numerical stability
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim


class HNSWIndex:
    """
    Hierarchical Navigable Small World Graph Vector Index.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
    ):
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M0 = 2 * M  # Maximum connections for Layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(M)  # Normalization factor for level assignment

        if metric == "cosine":
            self.dist_fn: Callable[[List[float], List[float]], float] = cosine_distance
        elif metric == "l2":
            self.dist_fn = euclidean_distance
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Vector storage: node_id -> vector
        self.vectors: Dict[str, List[float]] = {}
        # Node levels: node_id -> max_level
        self.node_levels: Dict[str, int] = {}
        # Graph adjacency: layer -> node_id -> list of neighbor node_ids
        self.graphs: Dict[int, Dict[str, List[str]]] = {}

        self.entry_point: Optional[str] = None
        self.max_level: int = -1

    def _random_level(self) -> int:
        """Determines the maximum layer a new vector is promoted to."""
        r = random.random()
        if r == 0.0:
            r = 1e-9
        return int(-math.log(r) * self.m_L)

    def add(self, node_id: str, vector: List[float]) -> None:
        """Inserts a high-dimensional vector into the HNSW index."""
        if len(vector) != self.dim:
            raise ValueError(f"Vector dim {len(vector)} does not match index dim {self.dim}")

        self.vectors[node_id] = vector
        new_level = self._random_level()
        self.node_levels[node_id] = new_level

        # Ensure graphs exist up to new_level
        for lvl in range(new_level + 1):
            if lvl not in self.graphs:
                self.graphs[lvl] = {}
            self.graphs[lvl][node_id] = []

        curr_ep = self.entry_point
        curr_max_level = self.max_level

        # Case 1: First node in the index
        if curr_ep is None:
            self.entry_point = node_id
            self.max_level = new_level
            return

        # Phase 1: Search top-down from current max_level to new_level + 1
        curr_dist = self.dist_fn(vector, self.vectors[curr_ep])
        for lvl in range(curr_max_level, new_level, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs.get(lvl, {}).get(curr_ep, []):
                    d = self.dist_fn(vector, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # Phase 2: For levels min(new_level, curr_max_level) down to 0, connect neighbors
        top_conn_level = min(new_level, curr_max_level)
        w_eps = [curr_ep]

        for lvl in range(top_conn_level, -1, -1):
            # Find ef_construction nearest neighbors at this level
            candidates = self._search_layer(vector, w_eps, ef=self.ef_construction, level=lvl)
            m_max = self.M0 if lvl == 0 else self.M

            # Select M nearest neighbors
            candidates.sort(key=lambda x: x[0])
            neighbors = [node for _, node in candidates[:m_max]]
            self.graphs[lvl][node_id] = neighbors

            # Add bidirectional edges
            for n in neighbors:
                n_conns = self.graphs[lvl].setdefault(n, [])
                n_conns.append(node_id)
                # Shrink connections if exceeded m_max
                if len(n_conns) > m_max:
                    # Keep m_max closest
                    scored = [(self.dist_fn(self.vectors[n], self.vectors[other]), other) for other in n_conns]
                    scored.sort(key=lambda x: x[0])
                    self.graphs[lvl][n] = [other for _, other in scored[:m_max]]

            w_eps = [node for _, node in candidates]

        if new_level > self.max_level:
            self.entry_point = node_id
            self.max_level = new_level

    def _search_layer(
        self,
        query: List[float],
        entry_points: List[str],
        ef: int,
        level: int
    ) -> List[Tuple[float, str]]:
        """
        Greedy beam-search traversal on a single layer graph.
        Returns: list of (distance, node_id) tuples.
        """
        visited: Set[str] = set(entry_points)
        # candidates: min-heap of (dist, node_id)
        candidates: List[Tuple[float, str]] = []
        # nearest: max-heap of (-dist, node_id)
        nearest: List[Tuple[float, str]] = []

        for ep in entry_points:
            d = self.dist_fn(query, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(nearest, (-d, ep))

        layer_graph = self.graphs.get(level, {})

        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            furthest_nearest_dist = -nearest[0][0]

            if c_dist > furthest_nearest_dist:
                break

            for neighbor in layer_graph.get(c_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self.dist_fn(query, self.vectors[neighbor])
                    furthest_nearest_dist = -nearest[0][0]

                    if n_dist < furthest_nearest_dist or len(nearest) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(nearest, (-n_dist, neighbor))
                        if len(nearest) > ef:
                            heapq.heappop(nearest)

        return [(-neg_d, node) for neg_d, node in nearest]

    def search(self, query: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Queries the HNSW index for the top-k nearest neighbors.
        Returns: list of (node_id, distance) tuples sorted by ascending distance.
        """
        if self.entry_point is None:
            return []

        curr_ep = self.entry_point
        curr_dist = self.dist_fn(query, self.vectors[curr_ep])

        # Top-down descent through upper layers
        for lvl in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for neighbor in self.graphs.get(lvl, {}).get(curr_ep, []):
                    d = self.dist_fn(query, self.vectors[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_ep = neighbor
                        changed = True

        # Search bottom layer (Layer 0) with ef_search beam width
        candidates = self._search_layer(query, [curr_ep], ef=max(self.ef_search, k), level=0)
        candidates.sort(key=lambda x: x[0])

        return [(node, dist) for dist, node in candidates[:k]]

    def save(self, filepath: str) -> None:
        """Serializes the HNSW index to a JSON file."""
        data = {
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "node_levels": self.node_levels,
            "vectors": self.vectors,
            "graphs": {str(lvl): g for lvl, g in self.graphs.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> HNSWIndex:
        """Restores an HNSW index from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls(
            dim=data["dim"],
            metric=data["metric"],
            M=data["M"],
            ef_construction=data["ef_construction"],
            ef_search=data["ef_search"],
        )
        idx.entry_point = data["entry_point"]
        idx.max_level = data["max_level"]
        idx.node_levels = data["node_levels"]
        idx.vectors = data["vectors"]
        idx.graphs = {int(lvl): g for lvl, g in data["graphs"].items()}
        return idx
