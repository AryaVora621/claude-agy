"""
Unit tests for GeoPrism R*-Tree Spatial Index and k-NN Search.
"""

import math
import random
import unittest
from geoprism.aabb import AABB
from geoprism.rtree import RStarTree


class TestRStarTree(unittest.TestCase):
    def test_basic_insertion_and_search(self):
        tree = RStarTree(max_entries=4, dimensions=2)

        boxes = [
            ("A", AABB((0, 0), (2, 2))),
            ("B", AABB((5, 5), (7, 7))),
            ("C", AABB((1, 1), (3, 3))),
            ("D", AABB((10, 10), (12, 12))),
            ("E", AABB((2, 0), (4, 2))),
        ]

        for item, box in boxes:
            tree.insert(item, box)

        self.assertEqual(tree.size, 5)

        # Query region [0, 0] to [2.5, 2.5] -> should match A, C, E
        res = tree.search(AABB((0, 0), (2.5, 2.5)))
        matched_items = sorted([item for item, _ in res])
        self.assertEqual(matched_items, ["A", "C", "E"])

        # Query point (1.5, 1.5) -> should contain A and C
        res_pt = tree.search_point((1.5, 1.5))
        matched_pt = sorted([item for item, _ in res_pt])
        self.assertEqual(matched_pt, ["A", "C"])

    def test_knn_search_matches_ground_truth(self):
        # Generate 200 random points in 2D
        random.seed(42)
        tree = RStarTree(max_entries=8, dimensions=2)
        points = []

        for i in range(200):
            x = random.uniform(-100, 100)
            y = random.uniform(-100, 100)
            points.append((f"p{i}", x, y))
            tree.insert(f"p{i}", AABB.from_point((x, y)))

        # Query from origin (0, 0) for k=5 nearest
        query_pt = (0.0, 0.0)
        k = 5

        # Ground truth by brute-force sort
        def distance_sq(pt):
            _, x, y = pt
            return x * x + y * y

        sorted_pts = sorted(points, key=distance_sq)
        ground_truth = [(math.sqrt(distance_sq(pt)), pt[0]) for pt in sorted_pts[:k]]

        # Query R*-Tree
        knn_res = tree.nearest_neighbors(query_pt, k=k)
        self.assertEqual(len(knn_res), k)

        for (d_tree, item_tree, _), (d_true, item_true) in zip(knn_res, ground_truth):
            self.assertEqual(item_tree, item_true)
            self.assertAlmostEqual(d_tree, d_true, places=6)

    def test_3d_spatial_tree(self):
        tree = RStarTree(max_entries=6, dimensions=3)
        b1 = AABB((0, 0, 0), (1, 1, 1))
        b2 = AABB((10, 10, 10), (12, 12, 12))
        b3 = AABB((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))

        tree.insert("cube1", b1)
        tree.insert("cube2", b2)
        tree.insert("cube3", b3)

        res = tree.search(AABB((0, 0, 0), (0.8, 0.8, 0.8)))
        matched = sorted([item for item, _ in res])
        self.assertEqual(matched, ["cube1", "cube3"])

        # Nearest neighbor from (11, 11, 11)
        knn = tree.nearest_neighbors((11, 11, 11), k=1)
        self.assertEqual(len(knn), 1)
        self.assertEqual(knn[0][1], "cube2")


if __name__ == "__main__":
    unittest.main()
