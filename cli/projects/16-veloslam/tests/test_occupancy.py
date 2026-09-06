"""
Unit tests for VeloSLAM Probabilistic Occupancy Grid Mapping & LiDAR Raycasting.
"""

import math
import unittest
from veloslam.occupancy import OccupancyGrid, LaserScan, bresenham_line


class TestOccupancyGrid(unittest.TestCase):
    def test_bresenham_line(self):
        line = bresenham_line(0, 0, 5, 0)
        self.assertEqual(len(line), 6)
        self.assertEqual(line[0], (0, 0))
        self.assertEqual(line[-1], (5, 0))

        diag = bresenham_line(0, 0, 3, 3)
        self.assertEqual(diag, [(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_coordinate_transforms(self):
        grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.5, origin_x=0.0, origin_y=0.0)
        self.assertEqual(grid.width_cells, 20)
        self.assertEqual(grid.height_cells, 20)

        gx, gy = grid.world_to_grid(2.2, 3.7)
        self.assertEqual(gx, 4)
        self.assertEqual(gy, 7)

        wx, wy = grid.grid_to_world(4, 7)
        self.assertAlmostEqual(wx, 2.25)
        self.assertAlmostEqual(wy, 3.75)

    def test_initial_prior_probabilities(self):
        grid = OccupancyGrid(width_m=4.0, height_m=4.0, resolution=1.0, origin_x=0.0, origin_y=0.0)
        for gy in range(grid.height_cells):
            for gx in range(grid.width_cells):
                self.assertAlmostEqual(grid.get_probability(gx, gy), 0.5, places=3)
                self.assertFalse(grid.is_occupied(gx, gy))
                self.assertFalse(grid.is_free(gx, gy))

    def test_lidar_scan_update(self):
        grid = OccupancyGrid(
            width_m=10.0, height_m=10.0, resolution=0.2, origin_x=-5.0, origin_y=-5.0
        )
        robot_pose = (0.0, 0.0, 0.0)  # Robot at center, facing East (+X)

        # Single beam directly ahead: obstacle hit at 2.0 meters
        scan = LaserScan(
            ranges=[2.0],
            angle_min=0.0,
            angle_max=0.0,
            angle_increment=0.0,
            range_min=0.1,
            range_max=10.0
        )
        grid.update_with_scan(robot_pose, scan)

        # Origin (0.0, 0.0)
        origin_gx, origin_gy = grid.world_to_grid(0.0, 0.0)
        # Point along ray at x = 1.0m should be Free
        free_gx, free_gy = grid.world_to_grid(1.0, 0.0)
        self.assertTrue(grid.is_free(free_gx, free_gy))
        self.assertLess(grid.get_probability(free_gx, free_gy), 0.4)

        # Obstacle endpoint at x = 2.0m should be Occupied
        hit_gx, hit_gy = grid.world_to_grid(2.0, 0.0)
        self.assertTrue(grid.is_occupied(hit_gx, hit_gy))
        self.assertGreater(grid.get_probability(hit_gx, hit_gy), 0.65)

    def test_inflation_layer(self):
        grid = OccupancyGrid(width_m=6.0, height_m=6.0, resolution=0.5, origin_x=-3.0, origin_y=-3.0)
        # Mark an obstacle cell at center (0, 0)
        cgx, cgy = grid.world_to_grid(0.0, 0.0)
        grid.cells[grid._get_index(cgx, cgy)] = grid.l_max

        # Inflate by 1.0 meter (2 cells in each direction)
        inflated = grid.get_inflated_grid(inflation_radius_m=1.0)

        # The center should remain occupied
        self.assertTrue(inflated.is_occupied(cgx, cgy))

        # 1 cell away (0.5m) should now also be occupied
        adj_gx, adj_gy = grid.world_to_grid(0.5, 0.0)
        self.assertTrue(inflated.is_occupied(adj_gx, adj_gy))

        # 2 cells away (1.0m) should be occupied
        two_gx, two_gy = grid.world_to_grid(1.0, 0.0)
        self.assertTrue(inflated.is_occupied(two_gx, two_gy))

        # 4 cells away (2.0m) should NOT be occupied
        far_gx, far_gy = grid.world_to_grid(2.0, 0.0)
        self.assertFalse(inflated.is_occupied(far_gx, far_gy))


if __name__ == "__main__":
    unittest.main()
