"""
Unit tests for VeloSLAM Unicode Braille Terminal Visualizer.
"""

import unittest
from veloslam.occupancy import OccupancyGrid, LaserScan
from veloslam.visualizer import render_braille_map


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.grid = OccupancyGrid(width_m=10.0, height_m=10.0, resolution=0.2, origin_x=-5.0, origin_y=-5.0)

    def test_render_empty_map(self):
        output = render_braille_map(self.grid, width_chars=40, height_chars=16, border=True)
        self.assertIn("+", output)
        self.assertIn("-", output)
        lines = output.split("\n")
        self.assertEqual(len(lines), 18)  # 16 + 2 border lines

    def test_render_with_robot_and_path(self):
        robot_pose = (0.0, 0.0, 0.785)  # 45 deg
        path = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.785), (2.0, 2.0, 0.785)]
        landmarks = [(0, 3.0, 2.0, 0.4, 0.2, 0.1)]
        scan = LaserScan(ranges=[2.5] * 16, angle_min=-1.0, angle_max=1.0, range_min=0.1, range_max=5.0)

        output = render_braille_map(
            self.grid,
            robot_pose=robot_pose,
            path=path,
            landmarks=landmarks,
            laser_scan=scan,
            width_chars=40,
            height_chars=16
        )

        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 500)


if __name__ == "__main__":
    unittest.main()
