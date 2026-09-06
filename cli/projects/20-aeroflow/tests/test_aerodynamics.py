"""
Unit Tests for AeroFlow Aerodynamic Forces & Strouhal Frequency Estimation.
"""

import unittest
import math
from aeroflow.aerodynamics import AerodynamicForces, AerodynamicTracker


class TestAerodynamics(unittest.TestCase):
    def test_aerodynamic_tracker_recording(self):
        tracker = AerodynamicTracker(
            characteristic_length=10.0,
            freestream_speed=0.1,
            fluid_density=1.0,
            time_step=1.0,
        )

        # Record a force reading: Fx = 0.05, Fy = 0.02
        forces = tracker.record(step=1, fx=0.05, fy=0.02)
        # denom = 0.5 * 1.0 * (0.01) * 10 = 0.05
        # CD = 0.05 / 0.05 = 1.0
        # CL = 0.02 / 0.05 = 0.4
        self.assertAlmostEqual(forces.drag_coeff, 1.0)
        self.assertAlmostEqual(forces.lift_coeff, 0.4)
        self.assertAlmostEqual(forces.lift_to_drag, 0.4)

    def test_strouhal_number_synthetic_oscillation(self):
        # Generate synthetic vortex shedding lift signal: period T = 20 seconds -> f = 0.05 Hz
        # With D = 10 m, U = 2.5 m/s -> St = f * D / U = 0.05 * 10 / 2.5 = 0.20
        tracker = AerodynamicTracker(
            characteristic_length=10.0,
            freestream_speed=2.5,
            fluid_density=1.0,
            time_step=1.0,
        )

        for step in range(160):
            t = float(step)
            # Oscillating lift force with period = 20 steps
            fy = math.sin(2.0 * math.pi * (t / 20.0))
            fx = 2.0 + 0.1 * math.cos(4.0 * math.pi * (t / 20.0))
            forces = tracker.record(step=step, fx=fx, fy=fy)

        # Check estimated Strouhal number
        self.assertIsNotNone(forces.strouhal_number)
        self.assertAlmostEqual(forces.strouhal_number, 0.20, delta=0.02)


if __name__ == "__main__":
    unittest.main()
