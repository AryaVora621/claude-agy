"""
Unit tests for Method of Characteristics Nozzle Design (thermoprop/moc_nozzle.py).
"""

from __future__ import annotations

import unittest

from thermoprop.gas_dynamics import GasProperties
from thermoprop.moc_nozzle import MethodOfCharacteristicsNozzle


class TestMOCNozzle(unittest.TestCase):
    """Test suite for 2D Method of Characteristics supersonic nozzle design."""

    def setUp(self) -> None:
        self.air = GasProperties.air()
        self.moc = MethodOfCharacteristicsNozzle(self.air)

    def test_minimum_length_nozzle_synthesis(self) -> None:
        """Verify MOC contour generation for Mach 2.2 nozzle."""
        target_mach = 2.2
        contour = self.moc.design_minimum_length_nozzle(
            target_exit_mach=target_mach,
            throat_height=0.05,
            num_expansion_waves=8,
        )

        # Throat radius must match input
        self.assertAlmostEqual(contour.throat_radius, 0.05, places=5)
        # Exit radius must be greater than throat radius (diverging)
        self.assertGreater(contour.exit_radius, contour.throat_radius)
        # Expansion ratio > 1.0
        self.assertGreater(contour.expansion_ratio, 1.0)
        # Length must be positive
        self.assertGreater(contour.length, 0.0)

        # Verify wall points monotonically increase in x
        for i in range(1, len(contour.wall_points)):
            x_prev, y_prev = contour.wall_points[i - 1]
            x_curr, y_curr = contour.wall_points[i]
            self.assertGreater(x_curr, x_prev)
            self.assertGreaterEqual(y_curr, y_prev)

    def test_radius_interpolation(self) -> None:
        """Verify radius_at interpolation inside nozzle contour."""
        contour = self.moc.design_minimum_length_nozzle(
            target_exit_mach=2.0,
            throat_height=0.1,
            num_expansion_waves=6,
        )

        r_throat = contour.radius_at(0.0)
        self.assertAlmostEqual(r_throat, 0.1, places=5)

        r_exit = contour.radius_at(contour.length)
        self.assertAlmostEqual(r_exit, contour.exit_radius, places=5)

        # Midpoint radius must lie strictly between throat and exit
        r_mid = contour.radius_at(0.5 * contour.length)
        self.assertGreater(r_mid, r_throat)
        self.assertLess(r_mid, r_exit)

    def test_invalid_parameters_raise_error(self) -> None:
        """Verify boundary condition validations."""
        with self.assertRaises(ValueError):
            self.moc.design_minimum_length_nozzle(target_exit_mach=1.02)

        with self.assertRaises(ValueError):
            self.moc.design_minimum_length_nozzle(target_exit_mach=2.0, num_expansion_waves=2)


if __name__ == "__main__":
    unittest.main()
