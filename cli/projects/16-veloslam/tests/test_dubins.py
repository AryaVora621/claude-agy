"""
Unit tests for VeloSLAM Analytical Dubins Path Solver.
"""

import math
import unittest
from veloslam.dubins import dubins_shortest_path, DubinsPath


class TestDubins(unittest.TestCase):
    def test_straight_line_path(self):
        start = (0.0, 0.0, 0.0)
        goal = (10.0, 0.0, 0.0)
        rho = 1.0

        path = dubins_shortest_path(start, goal, rho)
        # Straight path length should be 10.0
        self.assertAlmostEqual(path.length, 10.0, places=3)
        self.assertIn("S", path.word)

        # Check endpoints
        p_start = path.evaluate(0.0)
        p_end = path.evaluate(path.length)

        self.assertAlmostEqual(p_start[0], 0.0, places=3)
        self.assertAlmostEqual(p_start[1], 0.0, places=3)
        self.assertAlmostEqual(p_end[0], 10.0, places=3)
        self.assertAlmostEqual(p_end[1], 0.0, places=3)

    def test_u_turn_path(self):
        rho = 2.0
        start = (0.0, 0.0, 0.0)
        # Goal is 180 degree U-turn: x=0, y=2*rho, heading=pi
        goal = (0.0, 2.0 * rho, math.pi)

        path = dubins_shortest_path(start, goal, rho)
        # Semicircle arc length is pi * rho
        expected_len = math.pi * rho
        self.assertAlmostEqual(path.length, expected_len, delta=0.05)

        p_end = path.evaluate(path.length)
        self.assertAlmostEqual(p_end[0], goal[0], delta=0.05)
        self.assertAlmostEqual(p_end[1], goal[1], delta=0.05)

    def test_path_sampling(self):
        start = (0.0, 0.0, 0.5)
        goal = (5.0, 4.0, -0.5)
        rho = 1.5

        path = dubins_shortest_path(start, goal, rho)
        samples = path.sample_path(step_size=0.1)

        self.assertGreater(len(samples), 10)
        # Start and end match
        self.assertAlmostEqual(samples[0][0], start[0], delta=0.05)
        self.assertAlmostEqual(samples[-1][0], goal[0], delta=0.05)
        self.assertAlmostEqual(samples[-1][1], goal[1], delta=0.05)

        # Consecutive distances between sampled points should be close to step_size
        for i in range(len(samples) - 1):
            d = math.hypot(samples[i+1][0] - samples[i][0], samples[i+1][1] - samples[i][1])
            self.assertLessEqual(d, 0.15)


if __name__ == "__main__":
    unittest.main()
