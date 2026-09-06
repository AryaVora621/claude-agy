"""
Unit tests for NovaPhysics Dynamic AABB Tree broadphase pruning.
"""

import unittest
from novaphysics.math2d import Vec2
from novaphysics.shapes import AABB
from novaphysics.broadphase import DynamicAABBTree


class TestBroadphase(unittest.TestCase):
    def test_insert_and_query(self):
        tree = DynamicAABBTree(fat_margin=0.2)
        box1 = AABB(Vec2(0.0, 0.0), Vec2(2.0, 2.0))
        box2 = AABB(Vec2(5.0, 5.0), Vec2(7.0, 7.0))
        box3 = AABB(Vec2(1.0, 1.0), Vec2(3.0, 3.0))

        id1 = tree.insert(box1, "box1")
        id2 = tree.insert(box2, "box2")
        id3 = tree.insert(box3, "box3")

        # Query box overlapping box1 and box3
        q_box = AABB(Vec2(1.5, 1.5), Vec2(2.5, 2.5))
        overlaps = tree.query(q_box)

        self.assertIn("box1", overlaps)
        self.assertIn("box3", overlaps)
        self.assertNotIn("box2", overlaps)

    def test_query_pairs(self):
        tree = DynamicAABBTree(fat_margin=0.0)
        # boxA and boxB overlap; boxC is distant
        box_a = AABB(Vec2(0.0, 0.0), Vec2(2.0, 2.0))
        box_b = AABB(Vec2(1.0, 1.0), Vec2(3.0, 3.0))
        box_c = AABB(Vec2(10.0, 10.0), Vec2(12.0, 12.0))

        tree.insert(box_a, "A")
        tree.insert(box_b, "B")
        tree.insert(box_c, "C")

        pairs = tree.query_pairs()
        pair_set = {tuple(sorted((p[0], p[1]))) for p in pairs}
        self.assertIn(("A", "B"), pair_set)
        self.assertNotIn(("A", "C"), pair_set)
        self.assertNotIn(("B", "C"), pair_set)

    def test_remove_and_update(self):
        tree = DynamicAABBTree(fat_margin=0.1)
        b = AABB(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
        node_id = tree.insert(b, "item")

        # Query confirms item is present
        self.assertEqual(tree.query(b), ["item"])

        # Remove item
        tree.remove(node_id)
        self.assertEqual(tree.query(b), [])


if __name__ == "__main__":
    unittest.main()
