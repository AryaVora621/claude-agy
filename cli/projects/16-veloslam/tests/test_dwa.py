"""
Unit tests for VeloSLAM Dynamic Window Approach (DWA) Local Planner.
"""

import math
import unittest
from veloslam.occupancy import OccupancyGrid
from veloslam.dwa import DWAPlanner, DWAConfig, RobotState


class TestDWA(unittest.TestCase):
    def setUp(self):
        self.grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.1, origin_x=-5.0, origin_y=-5.0)
        self.planner = DWAPlanner(DWAConfig())

    def test_dynamic_window_computation(self):
        state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.5, omega=0.0)
        dw = self.planner.calc_dynamic_window(state)
        # v_min, v_max, w_min, w_max
        v_min, v_max, w_min, w_max = dw
        self.assertGreaterEqual(v_min, 0.0)
        self.assertLessEqual(v_max, self.planner.cfg.max_speed)
        self.assertGreaterEqual(v_max, state.v)
        self.assertLessEqual(v_min, state.v)
        self.assertGreater(w_max, 0.0)
        self.assertLess(w_min, 0.0)

    def test_open_space_planning(self):
        # Goal directly ahead at (3.0, 0.0)
        state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)
        goal = (3.0, 0.0)
        best_v, best_w, traj = self.planner.plan(state, goal, self.grid)

        self.assertGreater(best_v, 0.0)
        self.assertAlmostEqual(best_w, 0.0, delta=0.2)
        self.assertGreater(len(traj), 0)

        # Trajectory endpoints should move forward along x-axis
        self.assertGreater(traj[-1][0], 0.0)
        self.assertAlmostEqual(traj[-1][1], 0.0, delta=0.3)

    def test_reactive_obstacle_avoidance(self):
        # Obstacle placed directly in front of the robot at (1.5, 0.0)
        gx, gy = self.grid.world_to_grid(1.5, 0.0)
        for dy in range(-3, 4):
            for dx in range(-1, 2):
                self.grid.cells[self.grid._get_index(gx + dx, gy + dy)] = self.grid.l_max

        state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.2, omega=0.0)
        goal = (4.0, 0.0)

        best_v, best_w, traj = self.planner.plan(state, goal, self.grid)

        # Robot must turn away from center to steer around obstacle
        self.assertNotEqual(best_w, 0.0)

        # Trajectory must have zero collision
        for tx, ty, _ in traj:
            self.assertFalse(self.grid.is_world_occupied(tx, ty))

    def test_trajectory_prediction(self):
        state = RobotState(x=1.0, y=2.0, theta=math.pi / 2.0, v=1.0, omega=0.0)
        traj = self.planner.predict_trajectory(state, v=1.0, omega=0.0)
        self.assertGreater(len(traj), 5)
        # Moving straight in y-direction
        self.assertAlmostEqual(traj[0][0], 1.0, delta=0.1)
        self.assertGreater(traj[-1][1], 2.0)


if __name__ == "__main__":
    unittest.main()
