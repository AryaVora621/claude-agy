"""Unit tests for Differential Flatness and Minimum-Snap Trajectory Generation."""

import math
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.dynamics import QuadrotorParams, Vector3
from avionix.trajectory import (
    AerobaticTrajectories,
    DifferentialFlatness,
    FlatOutput,
    MinimumSnapTrajectory,
    PiecewisePolynomial1D,
    PolynomialSegment,
)


class TestDifferentialFlatness(unittest.TestCase):
    """Test differential flatness mapping from flat outputs to full 6-DOF state."""

    def setUp(self) -> None:
        self.params = QuadrotorParams(mass_kg=1.0)
        self.flatness = DifferentialFlatness(self.params)

    def test_hover_mapping(self) -> None:
        # Stationary flat output
        flat = FlatOutput(
            position=Vector3(0.0, 0.0, 2.0),
            velocity=Vector3(0.0, 0.0, 0.0),
            acceleration=Vector3(0.0, 0.0, 0.0),
            yaw_rad=0.0,
        )
        full = self.flatness.flat_to_full_state(flat, 0.0)

        # Expected collective thrust in hover = m * g
        expected_thrust = self.params.mass_kg * self.params.gravity_m_s2
        self.assertAlmostEqual(full.collective_thrust, expected_thrust, places=4)

        # In hover with zero yaw, body z-axis points up, roll and pitch are zero
        r, p, y = full.attitude.to_euler()
        self.assertAlmostEqual(r, 0.0, places=4)
        self.assertAlmostEqual(p, 0.0, places=4)
        self.assertAlmostEqual(y, 0.0, places=4)

    def test_forward_acceleration_pitch_tilt(self) -> None:
        # Quadrotor accelerating forward (+X) must pitch down (-Y axis tilt in right-handed convention)
        acc_x = 4.0
        flat = FlatOutput(
            position=Vector3(0.0, 0.0, 1.0),
            velocity=Vector3(1.0, 0.0, 0.0),
            acceleration=Vector3(acc_x, 0.0, 0.0),
            yaw_rad=0.0,
        )
        full = self.flatness.flat_to_full_state(flat, 0.0)

        # Thrust magnitude = m * sqrt(a_x^2 + g^2)
        expected_thrust = self.params.mass_kg * math.sqrt(acc_x ** 2 + self.params.gravity_m_s2 ** 2)
        self.assertAlmostEqual(full.collective_thrust, expected_thrust, places=3)

        # Pitch should be non-zero to produce forward thrust component
        r, p, y = full.attitude.to_euler()
        self.assertGreater(abs(p), 0.1)


class TestPiecewisePolynomial(unittest.TestCase):
    """Test quintic polynomial segment calculations and boundary conditions."""

    def test_single_segment_boundary_values(self) -> None:
        # 5th order polynomial from x=0 (at t=0) to x=10 (at t=2) starting and ending at rest
        # Parameterized by normalized tau in [0, 1]
        # p(tau) = 10 * (10 * tau^3 - 15 * tau^4 + 6 * tau^5) = 100 tau^3 - 150 tau^4 + 60 tau^5
        seg = PolynomialSegment(coeffs=[0.0, 0.0, 0.0, 100.0, -150.0, 60.0], duration=2.0)

        pos_0 = seg.eval(0.0)
        vel_0 = seg.eval_deriv(0.0, order=1)
        acc_0 = seg.eval_deriv(0.0, order=2)
        self.assertAlmostEqual(pos_0, 0.0)
        self.assertAlmostEqual(vel_0, 0.0)
        self.assertAlmostEqual(acc_0, 0.0)

        pos_end = seg.eval(1.0)
        vel_end = seg.eval_deriv(1.0, order=1)
        acc_end = seg.eval_deriv(1.0, order=2)
        self.assertAlmostEqual(pos_end, 10.0, places=3)
        self.assertAlmostEqual(vel_end, 0.0, places=3)
        self.assertAlmostEqual(acc_end, 0.0, places=3)


class TestMinimumSnapTrajectory(unittest.TestCase):
    """Test multi-waypoint 3D minimum-snap trajectory solving."""

    def test_two_point_rest_to_rest(self) -> None:
        wps = [Vector3(0.0, 0.0, 0.0), Vector3(5.0, -3.0, 2.0)]
        durations = [3.0]
        traj = MinimumSnapTrajectory(wps, durations)

        self.assertAlmostEqual(traj.total_duration, 3.0)

        # Initial conditions at t=0
        flat_0 = traj.evaluate(0.0)
        self.assertAlmostEqual(flat_0.position.x, 0.0, places=3)
        self.assertAlmostEqual(flat_0.position.y, 0.0, places=3)
        self.assertAlmostEqual(flat_0.position.z, 0.0, places=3)
        self.assertAlmostEqual(flat_0.velocity.norm(), 0.0, places=3)

        # Final conditions at t=3
        flat_end = traj.evaluate(3.0)
        self.assertAlmostEqual(flat_end.position.x, 5.0, places=3)
        self.assertAlmostEqual(flat_end.position.y, -3.0, places=3)
        self.assertAlmostEqual(flat_end.position.z, 2.0, places=3)
        self.assertAlmostEqual(flat_end.velocity.norm(), 0.0, places=3)

    def test_multi_waypoint_continuity(self) -> None:
        # 4 waypoints -> 3 segments
        wps = [
            Vector3(0.0, 0.0, 0.0),
            Vector3(2.0, 1.0, 1.5),
            Vector3(4.0, -1.0, 2.0),
            Vector3(6.0, 0.0, 0.5),
        ]
        durations = [2.0, 2.0, 2.0]
        traj = MinimumSnapTrajectory(wps, durations)

        self.assertAlmostEqual(traj.total_duration, 6.0)

        # Verify C0, C1, C2 continuity across knot at t=2.0
        eps = 1e-4
        left = traj.evaluate(2.0 - eps)
        right = traj.evaluate(2.0 + eps)

        self.assertAlmostEqual(left.position.x, right.position.x, places=2)
        self.assertAlmostEqual(left.position.y, right.position.y, places=2)
        self.assertAlmostEqual(left.position.z, right.position.z, places=2)
        self.assertAlmostEqual(left.velocity.x, right.velocity.x, places=2)
        self.assertAlmostEqual(left.acceleration.x, right.acceleration.x, places=2)


class TestAerobaticTrajectories(unittest.TestCase):
    """Test analytical aerobatic trajectories."""

    def test_figure_eight_properties(self) -> None:
        points = AerobaticTrajectories.figure_eight(
            radius_x=3.0, radius_y=2.0, altitude_z=2.5, num_points=100, loop_duration_s=5.0
        )
        self.assertEqual(len(points), 100)

        # Altitude should oscillate around 2.5m (+-0.5m)
        for pt in points:
            self.assertGreaterEqual(pt.position.z, 2.0 - 1e-4)
            self.assertLessEqual(pt.position.z, 3.0 + 1e-4)

        # Center crossing should happen at x=0, y=0
        has_center = any(abs(pt.position.x) < 0.1 and abs(pt.position.y) < 0.1 for pt in points)
        self.assertTrue(has_center)


if __name__ == "__main__":
    unittest.main()
