"""
GeoPrism: R*-Tree Spatial Index Engine.
Implements the Beckmann et al. (1990) R*-Tree with overlap minimization,
forced reinsertion on overflow, topological axis splitting, and branch-and-bound k-NN search.
"""

import math
import heapq
from typing import List, Tuple, Optional, Any, Dict, Set, Union
from .aabb import AABB, Point


class RStarNode:
    """Internal or leaf node of the R*-Tree."""
    __slots__ = ("is_leaf", "entries", "aabb", "level", "parent")

    def __init__(self, is_leaf: bool, level: int = 0) -> None:
        self.is_leaf = is_leaf
        self.level = level
        # For leaf: List[Tuple[item, AABB]]
        # For internal: List[Tuple[RStarNode, AABB]]
        self.entries: List[Tuple[Any, AABB]] = []
        self.aabb: Optional[AABB] = None
        self.parent: Optional["RStarNode"] = None

    def update_aabb(self) -> None:
        """Recomputes bounding box enclosing all child entries."""
        if not self.entries:
            self.aabb = None
            return
        box = self.entries[0][1]
        for _, b in self.entries[1:]:
            box = box.union(b)
        self.aabb = box

    def __repr__(self) -> str:
        t = "Leaf" if self.is_leaf else "Internal"
        return f"RStarNode({t}, level={self.level}, count={len(self.entries)}, aabb={self.aabb})"


class RStarTree:
    """
    R*-Tree spatial index for N-dimensional bounding box and point queries.
    """

    def __init__(self, max_entries: int = 16, min_entries: Optional[int] = None, dimensions: int = 2) -> None:
        if max_entries < 4:
            raise ValueError("max_entries must be at least 4")
        self.max_entries = max_entries
        self.min_entries = min_entries if min_entries is not None else max(2, int(0.4 * max_entries))
        self.dimensions = dimensions
        self.root: RStarNode = RStarNode(is_leaf=True, level=0)
        self.size = 0
        self._counter = 0  # Monotonic counter for tie-breaking in priority queues

    def insert(self, item: Any, aabb: AABB) -> None:
        """Inserts an item and its bounding box into the R*-Tree."""
        overflow_levels: Set[int] = set()
        self._insert_entry(item, aabb, target_level=0, overflow_levels=overflow_levels)
        self.size += 1

    def _insert_entry(
        self,
        item: Any,
        aabb: AABB,
        target_level: int,
        overflow_levels: Set[int]
    ) -> None:
        # Choose appropriate subtree
        target_node = self._choose_subtree(self.root, aabb, target_level)
        target_node.entries.append((item, aabb))
        if target_node.is_leaf is False:
            item.parent = target_node
        target_node.update_aabb()

        # Handle overflow if necessary
        if len(target_node.entries) > self.max_entries:
            self._overflow_treatment(target_node, overflow_levels)
        else:
            self._adjust_tree(target_node)

    def _choose_subtree(self, node: RStarNode, aabb: AABB, target_level: int) -> RStarNode:
        """Finds the most suitable node at target_level to insert aabb."""
        if node.level == target_level:
            return node

        # If child nodes are leaves (level == target_level + 1)
        if node.level == target_level + 1:
            best_entry = None
            best_overlap_enlarge = float("inf")
            best_vol_enlarge = float("inf")
            best_vol = float("inf")

            for child_node, child_box in node.entries:
                enlarged_box = child_box.union(aabb)

                # Overlap enlargement: sum of overlaps with other entries
                cur_overlap = 0.0
                new_overlap = 0.0
                for other_node, other_box in node.entries:
                    if other_node is not child_node:
                        cur_overlap += child_box.overlap_volume(other_box)
                        new_overlap += enlarged_box.overlap_volume(other_box)

                overlap_enlarge = new_overlap - cur_overlap
                vol_enlarge = enlarged_box.volume() - child_box.volume()
                vol = child_box.volume()

                # Minimize overlap enlargement, then volume enlargement, then volume
                if (
                    overlap_enlarge < best_overlap_enlarge or
                    (abs(overlap_enlarge - best_overlap_enlarge) < 1e-12 and vol_enlarge < best_vol_enlarge) or
                    (abs(overlap_enlarge - best_overlap_enlarge) < 1e-12 and abs(vol_enlarge - best_vol_enlarge) < 1e-12 and vol < best_vol)
                ):
                    best_overlap_enlarge = overlap_enlarge
                    best_vol_enlarge = vol_enlarge
                    best_vol = vol
                    best_entry = child_node

            return self._choose_subtree(best_entry, aabb, target_level)

        # Child nodes are internal nodes: choose child that minimizes volume enlargement
        best_entry = None
        best_vol_enlarge = float("inf")
        best_vol = float("inf")

        for child_node, child_box in node.entries:
            enlarged = child_box.union(aabb)
            vol_enlarge = enlarged.volume() - child_box.volume()
            vol = child_box.volume()

            if vol_enlarge < best_vol_enlarge or (abs(vol_enlarge - best_vol_enlarge) < 1e-12 and vol < best_vol):
                best_vol_enlarge = vol_enlarge
                best_vol = vol
                best_entry = child_node

        return self._choose_subtree(best_entry, aabb, target_level)

    def _overflow_treatment(self, node: RStarNode, overflow_levels: Set[int]) -> None:
        """
        Applies R*-Tree forced reinsertion if first overflow at this level, else splits.
        """
        if node.level not in overflow_levels and node is not self.root:
            overflow_levels.add(node.level)
            self._reinsert(node, overflow_levels)
        else:
            self._split(node, overflow_levels)

    def _reinsert(self, node: RStarNode, overflow_levels: Set[int]) -> None:
        """Removes the p furthest entries from node center and reinserts them from root."""
        node_center = node.aabb.center()
        # Compute distance from center of each entry to node center
        scored_entries = []
        for item, box in node.entries:
            c = box.center()
            dist_sq = sum((c[d] - node_center[d]) ** 2 for d in range(self.dimensions))
            scored_entries.append((dist_sq, item, box))

        # Sort descending by distance from center
        scored_entries.sort(key=lambda e: e[0], reverse=True)

        # Remove top p = 30% entries
        p = max(1, int(math.ceil(0.30 * self.max_entries)))
        to_reinsert = [(item, box) for (_, item, box) in scored_entries[:p]]
        remaining = [(item, box) for (_, item, box) in scored_entries[p:]]

        node.entries = remaining
        node.update_aabb()
        self._adjust_tree(node)

        # Re-insert from root at the node's original level
        for item, box in to_reinsert:
            self._insert_entry(item, box, target_level=node.level, overflow_levels=overflow_levels)

    def _split(self, node: RStarNode, overflow_levels: Set[int]) -> None:
        """
        Topological split heuristic:
        1. Choose split dimension minimizing sum of margins across all distributions.
        2. Along that dimension, choose distribution minimizing overlap, then volume.
        """
        d_best = self._choose_split_axis(node)
        group1, group2 = self._choose_split_index(node, d_best)

        # Populate current node with group1
        node.entries = group1
        node.update_aabb()
        for child, _ in group1:
            if not node.is_leaf:
                child.parent = node

        # Create new sibling with group2
        new_node = RStarNode(is_leaf=node.is_leaf, level=node.level)
        new_node.entries = group2
        new_node.update_aabb()
        for child, _ in group2:
            if not new_node.is_leaf:
                child.parent = new_node

        # If node is root, create new root
        if node is self.root:
            new_root = RStarNode(is_leaf=False, level=node.level + 1)
            new_root.entries.append((node, node.aabb))
            new_root.entries.append((new_node, new_node.aabb))
            node.parent = new_root
            new_node.parent = new_root
            new_root.update_aabb()
            self.root = new_root
        else:
            parent = node.parent
            parent.entries.append((new_node, new_node.aabb))
            new_node.parent = parent
            # Update parent entry for node
            for i, (child, _) in enumerate(parent.entries):
                if child is node:
                    parent.entries[i] = (node, node.aabb)
                    break
            parent.update_aabb()

            if len(parent.entries) > self.max_entries:
                self._overflow_treatment(parent, overflow_levels)
            else:
                self._adjust_tree(parent)

    def _choose_split_axis(self, node: RStarNode) -> int:
        """Evaluates sum of margins along each dimension and selects the axis with minimum sum."""
        best_axis = 0
        best_margin_sum = float("inf")

        for d in range(self.dimensions):
            # Sort entries by lower bound, then by upper bound
            margin_sum = 0.0
            for sort_key in (lambda e: e[1].lower[d], lambda e: e[1].upper[d]):
                sorted_entries = sorted(node.entries, key=sort_key)
                m = self.min_entries
                M = len(sorted_entries)
                for k in range(m, M - m + 1):
                    # Partition 1: 0..k
                    box1 = sorted_entries[0][1]
                    for _, b in sorted_entries[1:k]:
                        box1 = box1.union(b)

                    # Partition 2: k..M
                    box2 = sorted_entries[k][1]
                    for _, b in sorted_entries[k + 1:]:
                        box2 = box2.union(b)

                    margin_sum += box1.margin() + box2.margin()

            if margin_sum < best_margin_sum:
                best_margin_sum = margin_sum
                best_axis = d

        return best_axis

    def _choose_split_index(self, node: RStarNode, axis: int) -> Tuple[List[Any], List[Any]]:
        """Along the chosen split axis, selects the distribution minimizing overlap then area."""
        best_overlap = float("inf")
        best_total_vol = float("inf")
        best_group1: List[Any] = []
        best_group2: List[Any] = []

        for sort_key in (lambda e: e[1].lower[axis], lambda e: e[1].upper[axis]):
            sorted_entries = sorted(node.entries, key=sort_key)
            m = self.min_entries
            M = len(sorted_entries)

            for k in range(m, M - m + 1):
                g1 = sorted_entries[:k]
                g2 = sorted_entries[k:]

                box1 = g1[0][1]
                for _, b in g1[1:]:
                    box1 = box1.union(b)

                box2 = g2[0][1]
                for _, b in g2[1:]:
                    box2 = box2.union(b)

                overlap = box1.overlap_volume(box2)
                total_vol = box1.volume() + box2.volume()

                if overlap < best_overlap or (abs(overlap - best_overlap) < 1e-12 and total_vol < best_total_vol):
                    best_overlap = overlap
                    best_total_vol = total_vol
                    best_group1 = g1
                    best_group2 = g2

        return best_group1, best_group2

    def _adjust_tree(self, node: Optional[RStarNode]) -> None:
        """Propagates bounding box updates upwards toward root."""
        curr = node
        while curr is not None:
            curr.update_aabb()
            if curr.parent is not None:
                # Update parent's entry bounding box for curr
                for i, (child, _) in enumerate(curr.parent.entries):
                    if child is curr:
                        curr.parent.entries[i] = (curr, curr.aabb)
                        break
            curr = curr.parent

    def search(self, query_box: AABB) -> List[Tuple[Any, AABB]]:
        """Finds all items whose bounding box intersects query_box."""
        results: List[Tuple[Any, AABB]] = []

        def _traverse(node: RStarNode) -> None:
            if node.aabb is None or not node.aabb.intersects(query_box):
                return
            if node.is_leaf:
                for item, box in node.entries:
                    if box.intersects(query_box):
                        results.append((item, box))
            else:
                for child_node, child_box in node.entries:
                    if child_box.intersects(query_box):
                        _traverse(child_node)

        _traverse(self.root)
        return results

    def search_point(self, point: Union[List[float], Tuple[float, ...]]) -> List[Tuple[Any, AABB]]:
        """Finds all items containing the specified point."""
        query_box = AABB.from_point(point, radius=0.0)
        return [res for res in self.search(query_box) if res[1].contains_point(point)]

    def nearest_neighbors(
        self,
        point: Union[List[float], Tuple[float, ...]],
        k: int = 1
    ) -> List[Tuple[float, Any, AABB]]:
        """
        Branch-and-bound Best-First k-Nearest Neighbors search.
        Returns list of (distance, item, aabb) sorted by distance ascending.
        """
        if k <= 0 or self.size == 0 or self.root.aabb is None:
            return []

        # Min-heap storing: (min_dist, counter, is_leaf_entry, object)
        heap: List[Tuple[float, int, bool, Any]] = []
        counter = 0

        # Push root
        root_dist = self.root.aabb.min_dist(point)
        heapq.heappush(heap, (root_dist, counter, False, self.root))
        counter += 1

        results: List[Tuple[float, Any, AABB]] = []

        while heap:
            min_dist, _, is_entry, obj = heapq.heappop(heap)

            # If we already have k items and the closest candidate in heap is further
            # than the furthest item in our results, we can stop!
            if len(results) >= k and min_dist >= results[-1][0]:
                break

            if is_entry:
                item, box = obj
                # Exact distance to point
                dist = box.min_dist(point)
                # Insert into results maintaining sorted order up to k items
                # We do binary insertion
                idx = 0
                while idx < len(results) and results[idx][0] <= dist:
                    idx += 1
                results.insert(idx, (dist, item, box))
                if len(results) > k:
                    results.pop()
            else:
                # obj is an RStarNode
                node: RStarNode = obj
                if node.is_leaf:
                    for item, box in node.entries:
                        d = box.min_dist(point)
                        heapq.heappush(heap, (d, counter, True, (item, box)))
                        counter += 1
                else:
                    for child_node, child_box in node.entries:
                        d = child_box.min_dist(point)
                        heapq.heappush(heap, (d, counter, False, child_node))
                        counter += 1

        return results[:k]
