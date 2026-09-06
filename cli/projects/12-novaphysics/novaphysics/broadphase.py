"""
NovaPhysics: Dynamic AABB Tree Broadphase Collision Pruning.
Uses Surface Area Heuristics (SAH) to maintain a balanced bounding volume hierarchy.
"""

from typing import List, Tuple, Optional, Any, Dict
from .math2d import Vec2
from .shapes import AABB


class TreeNode:
    """Node in the Dynamic AABB bounding volume hierarchy."""
    __slots__ = ("aabb", "user_data", "parent", "left", "right", "height")

    def __init__(self, aabb: AABB, user_data: Any = None) -> None:
        self.aabb = aabb
        self.user_data = user_data
        self.parent: Optional[int] = None
        self.left: Optional[int] = None
        self.right: Optional[int] = None
        self.height: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class DynamicAABBTree:
    """
    Dynamic Bounding Volume Hierarchy (BVH) for accelerated broadphase spatial queries.
    Maintains fattened AABBs and provides O(N log N) pairwise overlap pruning.
    """

    def __init__(self, fat_margin: float = 0.1) -> None:
        self.fat_margin = fat_margin
        self.nodes: Dict[int, TreeNode] = {}
        self.root: Optional[int] = None
        self._next_id: int = 1

    def insert(self, aabb: AABB, user_data: Any) -> int:
        """Insert leaf into BVH with fattened margin; returns node ID."""
        node_id = self._next_id
        self._next_id += 1

        fat_box = aabb.fatten(self.fat_margin)
        new_node = TreeNode(fat_box, user_data)
        self.nodes[node_id] = new_node

        if self.root is None:
            self.root = node_id
            return node_id

        # Find best sibling using surface area heuristic
        leaf_box = fat_box
        sibling = self._find_best_sibling(leaf_box)

        # Create new parent
        old_parent = self.nodes[sibling].parent
        parent_id = self._next_id
        self._next_id += 1

        union_box = leaf_box.union(self.nodes[sibling].aabb)
        parent_node = TreeNode(union_box)
        parent_node.parent = old_parent
        parent_node.left = sibling
        parent_node.right = node_id
        parent_node.height = self.nodes[sibling].height + 1
        self.nodes[parent_id] = parent_node

        self.nodes[sibling].parent = parent_id
        new_node.parent = parent_id

        if old_parent is None:
            self.root = parent_id
        else:
            if self.nodes[old_parent].left == sibling:
                self.nodes[old_parent].left = parent_id
            else:
                self.nodes[old_parent].right = parent_id

        # Walk back up the tree refitting AABBs
        curr = parent_id
        while curr is not None:
            self._sync_node(curr)
            curr = self.nodes[curr].parent

        return node_id

    def remove(self, node_id: int) -> None:
        """Remove leaf from BVH."""
        if node_id not in self.nodes:
            return

        parent_id = self.nodes[node_id].parent
        if parent_id is None:
            # Root leaf
            self.root = None
            del self.nodes[node_id]
            return

        grand_parent = self.nodes[parent_id].parent
        sibling = self.nodes[parent_id].right if self.nodes[parent_id].left == node_id else self.nodes[parent_id].left

        if sibling is not None:
            self.nodes[sibling].parent = grand_parent

        if grand_parent is None:
            self.root = sibling
        else:
            if self.nodes[grand_parent].left == parent_id:
                self.nodes[grand_parent].left = sibling
            else:
                self.nodes[grand_parent].right = sibling

        del self.nodes[node_id]
        del self.nodes[parent_id]

        curr = grand_parent
        while curr is not None:
            self._sync_node(curr)
            curr = self.nodes[curr].parent

    def update(self, node_id: int, aabb: AABB) -> bool:
        """Update leaf AABB. Only re-inserts if true bounding box exceeds fattened margin."""
        node = self.nodes[node_id]
        if node.aabb.contains_aabb(aabb):
            return False
        user_data = node.user_data
        self.remove(node_id)
        # Reinsert
        new_id = self.insert(aabb, user_data)
        # Re-link node_id to preserve identity
        if new_id != node_id:
            self.nodes[node_id] = self.nodes.pop(new_id)
            # Update parent pointer
            p = self.nodes[node_id].parent
            if p is not None:
                if self.nodes[p].left == new_id:
                    self.nodes[p].left = node_id
                else:
                    self.nodes[p].right = node_id
            if self.root == new_id:
                self.root = node_id
        return True

    def _find_best_sibling(self, leaf_box: AABB) -> int:
        """Surface Area Heuristic tree traversal."""
        curr = self.root
        while not self.nodes[curr].is_leaf:
            left = self.nodes[curr].left
            right = self.nodes[curr].right

            area = self.nodes[curr].aabb.perimeter()
            combined_area = self.nodes[curr].aabb.union(leaf_box).perimeter()
            cost = 2.0 * combined_area
            inheritance_cost = 2.0 * (combined_area - area)

            # Left child cost
            left_union = self.nodes[left].aabb.union(leaf_box)
            if self.nodes[left].is_leaf:
                cost_l = left_union.perimeter() + inheritance_cost
            else:
                cost_l = (left_union.perimeter() - self.nodes[left].aabb.perimeter()) + inheritance_cost

            # Right child cost
            right_union = self.nodes[right].aabb.union(leaf_box)
            if self.nodes[right].is_leaf:
                cost_r = right_union.perimeter() + inheritance_cost
            else:
                cost_r = (right_union.perimeter() - self.nodes[right].aabb.perimeter()) + inheritance_cost

            if cost < cost_l and cost < cost_r:
                break

            curr = left if cost_l < cost_r else right

        return curr

    def _sync_node(self, node_id: int) -> None:
        node = self.nodes[node_id]
        l = self.nodes[node.left]
        r = self.nodes[node.right]
        node.aabb = l.aabb.union(r.aabb)
        node.height = 1 + max(l.height, r.height)

    def query(self, aabb: AABB) -> List[Any]:
        """Find all user_data objects whose AABBs overlap the query box."""
        if self.root is None:
            return []

        results = []
        stack = [self.root]

        while stack:
            curr = stack.pop()
            node = self.nodes[curr]
            if node.aabb.overlaps(aabb):
                if node.is_leaf:
                    results.append(node.user_data)
                else:
                    if node.left is not None:
                        stack.append(node.left)
                    if node.right is not None:
                        stack.append(node.right)

        return results

    def query_pairs(self) -> List[Tuple[Any, Any]]:
        """Find all overlapping pairs of leaves in the tree."""
        if self.root is None or self.nodes[self.root].is_leaf:
            return []

        pairs = []
        leaves = [nid for nid, n in self.nodes.items() if n.is_leaf]
        n_leaves = len(leaves)

        # For each leaf, query tree against higher indices
        for i in range(n_leaves):
            id_a = leaves[i]
            body_a = self.nodes[id_a].user_data
            box_a = self.nodes[id_a].aabb
            overlaps = self.query(box_a)
            for body_b in overlaps:
                if id(body_a) < id(body_b):
                    pairs.append((body_a, body_b))

        return pairs
