"""
Unit tests for GeoPrism Spatial Analytics & Geospatial Engine.
"""

import math
import random
import unittest
from geoprism.aabb import AABB
from geoprism.rtree import RStarTree
from geoprism.engine import spatial_join, dbscan, h3_hexbin, spatial_kde_raster


class TestEngine(unittest.TestCase):
    def test_spatial_join(self):
        tree_a = RStarTree(max_entries=4, dimensions=2)
        tree_b = RStarTree(max_entries=4, dimensions=2)

        # Tree A boxes
        boxes_a = [
            ("A1", AABB((0, 0), (2, 2))),
            ("A2", AABB((5, 5), (7, 7))),
            ("A3", AABB((10, 10), (12, 12))),
        ]
        for name, box in boxes_a:
            tree_a.insert(name, box)

        # Tree B boxes
        boxes_b = [
            ("B1", AABB((1, 1), (3, 3))),   # Overlaps A1
            ("B2", AABB((6, 6), (8, 8))),   # Overlaps A2
            ("B3", AABB((20, 20), (22, 22))), # Disjoint from all
            ("B4", AABB((0.5, 0.5), (1.5, 1.5))), # Also overlaps A1
        ]
        for name, box in boxes_b:
            tree_b.insert(name, box)

        matches = spatial_join(tree_a, tree_b)
        sorted_matches = sorted(matches)
        expected = sorted([("A1", "B1"), ("A1", "B4"), ("A2", "B2")])
        self.assertEqual(sorted_matches, expected)

    def test_dbscan_clustering(self):
        # Create two distinct clusters and 2 noise points
        cluster_1 = [(1.0, 1.0), (1.2, 1.1), (0.9, 1.3), (1.1, 0.8)]
        cluster_2 = [(20.0, 20.0), (20.3, 19.8), (19.9, 20.2), (20.1, 20.4)]
        noise = [(100.0, 100.0), (-50.0, -50.0)]

        all_points = cluster_1 + cluster_2 + noise
        clusters = dbscan(all_points, eps=1.0, min_pts=3)

        # Must have at least cluster 0, cluster 1, and noise (-1)
        self.assertIn(-1, clusters)
        noise_indices = clusters[-1]
        self.assertIn(8, noise_indices)  # point (100, 100)
        self.assertIn(9, noise_indices)  # point (-50, -50)

        # Non-noise clusters should partition cluster_1 and cluster_2
        non_noise_keys = [k for k in clusters.keys() if k != -1]
        self.assertEqual(len(non_noise_keys), 2)

        c1_set = set(clusters[non_noise_keys[0]])
        c2_set = set(clusters[non_noise_keys[1]])
        if 0 in c2_set:
            c1_set, c2_set = c2_set, c1_set

        self.assertEqual(c1_set, {0, 1, 2, 3})
        self.assertEqual(c2_set, {4, 5, 6, 7})

    def test_h3_hexbin(self):
        coords = [
            (37.7749, -122.4194),
            (37.7750, -122.4195),
            (37.7752, -122.4192),
            (51.5074, -0.1278),
        ]
        bins = h3_hexbin(coords, resolution=6)
        # SF points should fall into the same resolution 6 cell
        self.assertEqual(len(bins), 2)
        counts = sorted(list(bins.values()))
        self.assertEqual(counts, [1, 3])

    def test_spatial_kde_raster(self):
        points = [(10.0, 10.0), (10.1, 10.2), (10.05, 9.95), (20.0, 20.0)]
        grid, bbox = spatial_kde_raster(points, grid_w=30, grid_h=15)

        self.assertEqual(len(grid), 15)
        self.assertEqual(len(grid[0]), 30)

        # Max density value in grid should be 1.0 (normalized)
        max_val = max(max(row) for row in grid)
        self.assertAlmostEqual(max_val, 1.0)
        min_val = min(min(row) for row in grid)
        self.assertGreaterEqual(min_val, 0.0)


if __name__ == "__main__":
    unittest.main()
