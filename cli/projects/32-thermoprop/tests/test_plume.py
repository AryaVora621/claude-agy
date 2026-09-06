"""
Unit tests for Supersonic Exhaust Plume & Shock Engine (thermoprop/plume.py).
"""

from __future__ import annotations

import unittest

from thermoprop.gas_dynamics import GasProperties
from thermoprop.plume import (
    ExhaustPlumeEngine,
    PlumeRegime,
)


class TestExhaustPlume(unittest.TestCase):
    """Test suite for exhaust plume regimes and Mach diamond shock cells."""

    def setUp(self) -> None:
        self.gas = GasProperties.methalox()
        self.plume_engine = ExhaustPlumeEngine(self.gas)

    def test_regime_classification(self) -> None:
        """Verify Summerfield separation and plume adaptation criteria."""
        pa = 101325.0  # 1 atm

        # Pe / Pa <= 0.35 -> Separated flow
        self.assertEqual(
            self.plume_engine.determine_regime(p_exit=0.30 * pa, p_ambient=pa),
            PlumeRegime.SEPARATED,
        )

        # 0.35 < Pe / Pa < 0.95 -> Overexpanded
        self.assertEqual(
            self.plume_engine.determine_regime(p_exit=0.70 * pa, p_ambient=pa),
            PlumeRegime.OVEREXPANDED,
        )

        # 0.95 <= Pe / Pa <= 1.05 -> Ideally expanded
        self.assertEqual(
            self.plume_engine.determine_regime(p_exit=1.0 * pa, p_ambient=pa),
            PlumeRegime.IDEALLY_EXPANDED,
        )

        # Pe / Pa > 1.05 -> Underexpanded
        self.assertEqual(
            self.plume_engine.determine_regime(p_exit=2.5 * pa, p_ambient=pa),
            PlumeRegime.UNDEREXPANDED,
        )

        # Vacuum ambient -> Underexpanded
        self.assertEqual(
            self.plume_engine.determine_regime(p_exit=50000.0, p_ambient=0.0),
            PlumeRegime.UNDEREXPANDED,
        )

    def test_shock_cell_spacing(self) -> None:
        """Verify Prandtl periodic shock cell wavelength formula."""
        exit_d = 1.0
        pc = 80.0e5
        pa = 101325.0

        wavelength = self.plume_engine.calculate_shock_cell_spacing(
            exit_diameter=exit_d,
            chamber_pressure=pc,
            p_ambient=pa,
        )
        self.assertGreater(wavelength, 1.0)
        self.assertLess(wavelength, 10.0)

    def test_simulate_plume_structure(self) -> None:
        """Verify full plume simulation geometry and shock diamonds."""
        plume = self.plume_engine.simulate_plume(
            x_exit=2.0,
            exit_radius=0.5,
            exit_mach=2.8,
            p_exit=60000.0,
            chamber_pressure=70.0e5,
            p_ambient=101325.0,
            plume_length=5.0,
            num_cells=4,
        )

        self.assertEqual(plume.regime, PlumeRegime.OVEREXPANDED)
        self.assertGreater(len(plume.diamonds), 0)
        self.assertGreater(len(plume.shock_lines), 0)
        self.assertGreater(len(plume.boundary_points), 0)

    def test_underexpanded_plume_and_shear_layer(self) -> None:
        """Verify underexpanded high-altitude plume expansion."""
        plume = self.plume_engine.simulate_plume(
            x_exit=3.0,
            exit_radius=0.8,
            exit_mach=3.2,
            p_exit=150000.0,
            chamber_pressure=100.0e5,
            p_ambient=20000.0,  # High altitude low pressure
            plume_length=8.0,
            num_cells=5,
        )

        self.assertEqual(plume.regime, PlumeRegime.UNDEREXPANDED)
        self.assertGreater(plume.pressure_ratio, 1.05)
        # Underexpanded plume spreads outward beyond exit radius
        max_boundary_y = max(pt[1] for pt in plume.boundary_points)
        self.assertGreater(max_boundary_y, 0.8)


if __name__ == "__main__":
    unittest.main()
