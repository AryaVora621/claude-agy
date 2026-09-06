"""Unit tests for Autopilot Mission Executive, Disturbance Observer, and Avionics PFD."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from avionix.autopilot import Autopilot, DisturbanceObserver, FlightMode
from avionix.avionics_pfd import (
    BrailleFlightCanvas,
    render_3d_flight_path,
    render_primary_flight_display,
)
from avionix.dynamics import QuadrotorState, Quaternion, Vector3


class TestBrailleCanvasAndPFD(unittest.TestCase):
    """Test sub-pixel Braille rendering and Primary Flight Display generation."""

    def test_braille_canvas_drawing(self) -> None:
        canvas = BrailleFlightCanvas(char_width=20, char_height=10)
        # Pixel width = 40, height = 40
        canvas.draw_line(0, 0, 39, 39, r=255, g=0, b=0)

        output = canvas.render_to_string()
        # Non-empty rendered ANSI string
        self.assertGreater(len(output), 50)
        # Verify Braille unicode characters (U+2800..U+28FF)
        has_braille = any(0x2800 <= ord(ch) <= 0x28FF for ch in output)
        self.assertTrue(has_braille)

    def test_3d_trajectory_rendering(self) -> None:
        waypoints = [Vector3(0.0, 0.0, 0.0), Vector3(2.0, 2.0, 2.0)]
        actual = [Vector3(0.0, 0.0, 0.0), Vector3(1.0, 1.0, 1.0)]
        state = QuadrotorState(position=Vector3(1.0, 1.0, 1.0))

        rendered = render_3d_flight_path(waypoints, actual, state, char_width=40, char_height=12)
        self.assertIn("3D ORBITAL TRAJECTORY", rendered)
        self.assertIn("┌", rendered)
        self.assertIn("└", rendered)

    def test_primary_flight_display(self) -> None:
        state = QuadrotorState(
            position=Vector3(1.2, -0.5, 4.2),
            velocity=Vector3(2.5, 0.0, 0.8),
            attitude=Quaternion.from_euler(0.1, -0.05, 0.785),
        )
        pfd = render_primary_flight_display(state, target_pos=Vector3(0.0, 0.0, 5.0), flight_mode="AUTO NAV")

        self.assertIn("AUTO NAV", pfd)
        self.assertIn("ALT HOLD", pfd)
        self.assertIn("ROLL:", pfd)
        self.assertIn("PITCH:", pfd)
        self.assertIn("VSI:", pfd)
        self.assertIn("HDG:", pfd)


class TestDisturbanceObserver(unittest.TestCase):
    """Test aerodynamic wind force disturbance observer."""

    def test_dob_wind_force_convergence(self) -> None:
        dob = DisturbanceObserver(mass_kg=1.0, cutoff_freq_hz=10.0)

        # Apply steady 2.0 N lateral wind force
        true_wind = Vector3(2.0, 0.0, 0.0)
        commanded_thrust_w = Vector3(0.0, 0.0, 9.81)

        # Measured accel = (thrust + wind - mg) / m = wind / m = 2.0 m/s^2 along X
        measured_acc = Vector3(2.0, 0.0, 0.0)
        dt = 0.01

        for _ in range(50):
            est = dob.update(measured_acc, commanded_thrust_w, dt)

        # Estimated force should filter towards 2.0 N
        self.assertAlmostEqual(est.x, 2.0, places=2)
        self.assertAlmostEqual(est.y, 0.0, places=2)


class TestAutopilot(unittest.TestCase):
    """Test integrated Autopilot state machine and flight routines."""

    def test_takeoff_hover_transition(self) -> None:
        ap = Autopilot(use_ekf=False, seed=42)
        self.assertEqual(ap.mode, FlightMode.DISARMED)

        ap.takeoff(target_altitude_m=1.5)
        self.assertEqual(ap.mode, FlightMode.TAKEOFF)

        # Run 1.3 seconds of simulation (130 steps) to complete climb
        ap.run_simulation(duration_s=1.3, dt_s=0.01)

        # Altitude should be near 1.5m and mode should transition to HOVER
        self.assertAlmostEqual(ap.true_state.position.z, 1.5, delta=0.25)
        self.assertEqual(ap.mode, FlightMode.HOVER)

        # Telemetry log should have 130 entries
        self.assertEqual(len(ap.log), 130)

        # Compute tracking RMSE
        rmse_p, rmse_v = ap.calculate_tracking_rmse()
        self.assertGreater(rmse_p, 0.0)

    def test_waypoint_navigation_mission(self) -> None:
        ap = Autopilot(use_ekf=False, seed=42)
        ap.arm()
        ap.true_state.position = Vector3(0.0, 0.0, 1.0)

        wps = [
            Vector3(0.0, 0.0, 1.0),
            Vector3(1.0, 1.0, 1.5),
            Vector3(2.0, 0.0, 1.2),
        ]
        durations = [2.0, 2.0]

        ap.fly_waypoints(wps, durations)
        self.assertEqual(ap.mode, FlightMode.WAYPOINT_NAV)

        # Simulate full trajectory
        ap.run_simulation(duration_s=4.1, dt_s=0.01)

        # Quadrotor should reach near final waypoint and transition to HOVER
        self.assertEqual(ap.mode, FlightMode.HOVER)
        self.assertAlmostEqual(ap.true_state.position.x, 2.0, delta=0.35)
        self.assertAlmostEqual(ap.true_state.position.y, 0.0, delta=0.35)


if __name__ == "__main__":
    unittest.main()
