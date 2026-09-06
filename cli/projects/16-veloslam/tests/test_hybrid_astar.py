"""
Unit tests for VeloSLAM Kinodynamic Hybrid A* Motion Planner.
"""

import math
import unittest
from veloslam.occupancy import OccupancyGrid
from veloslam.hybrid_astar import HybridAStar, Pose2D


class TestHybridAStar(unittest.TestCase):
    def test_open_space_plan(self):
        grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution=0.5, origin_x=-10.0, origin_y=-10.0)
        planner = HybridAStar(grid, min_turning_radius=1.0)

        start = Pose2D(0.0, 0.0, 0.0)
        goal = Pose2D(5.0, 0.0, 0.0)

        path = planner.plan(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 5)

        # First waypoint near start
        self.assertAlmostEqual(path[0][0], start.x, delta=0.5)
        self.assertAlmostEqual(path[0][1], start.y, delta=0.5)

        # Last waypoint near goal
        self.assertAlmostEqual(path[-1][0], goal.x, delta=0.5)
        self.assertAlmostEqual(path[-1][1], goal.y, delta=0.5)

    def test_obstacle_avoidance(self):
        grid = OccupancyGrid(width_m=20.0, height_m=20.0, resolution=0.2, origin_x=-10.0, origin_y=-10.0)

        # Place a vertical obstacle wall at x = 3.0 from y = -2.0 to y = 2.0
        for y_m in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
            gx, gy = grid.world_to_grid(3.0, y_m)
            grid.cells[grid._get_index(gx, gy)] = grid.l_max

        planner = HybridAStar(grid, min_turning_radius=1.0, step_size=0.4)

        start = Pose2D(0.0, 0.0, 0.0)
        goal = Pose2D(6.0, 0.0, 0.0)

        path = planner.plan(start, goal, max_iterations=1500)
        self.assertIsNotNone(path)

        # Ensure none of the waypoints intersect the obstacle wall
        for x, y, _ in path:
            self.assertFalse(grid.is_world_occupied(x, y))


if __name__ == "__main__":
    unittest.main()
