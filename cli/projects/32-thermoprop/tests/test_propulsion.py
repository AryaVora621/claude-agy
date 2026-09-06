"""
Unit tests for Rocket Propulsion & Thermochemistry Engine (thermoprop/propulsion.py).
"""

from __future__ import annotations

import math
import unittest

from thermoprop.propulsion import (
    PropellantLibrary,
    RocketPropulsionEngine,
    SEA_LEVEL_PRESSURE,
    STANDARD_GRAVITY,
)


class TestRocketPropulsion(unittest.TestCase):
    """Test suite for rocket engine thermochemistry and performance."""

    def setUp(self) -> None:
        self.methalox = PropellantLibrary.methalox()
        self.engine_methalox = RocketPropulsionEngine(self.methalox)
        self.hydrolox = PropellantLibrary.hydrolox()
        self.engine_hydrolox = RocketPropulsionEngine(self.hydrolox)

    def test_propellant_library_presets(self) -> None:
        """Verify standard propellant properties."""
        self.assertEqual(self.methalox.name, "Methalox")
        self.assertAlmostEqual(self.methalox.mixture_ratio, 3.6)
        self.assertGreater(self.hydrolox.chamber_temperature, 3000.0)

        kerolox = PropellantLibrary.kerolox()
        self.assertEqual(kerolox.name, "Kerolox")
        self.assertAlmostEqual(kerolox.mixture_ratio, 2.6)

        hypergolic = PropellantLibrary.hypergolic()
        self.assertEqual(hypergolic.name, "Hypergolic")
        self.assertAlmostEqual(hypergolic.mixture_ratio, 2.1)

    def test_kerolox_engine_performance(self) -> None:
        """Verify Kerolox engine high sea-level thrust density."""
        kerolox_engine = RocketPropulsionEngine(PropellantLibrary.kerolox())
        state = kerolox_engine.evaluate_engine(
            chamber_pressure=150.0e5,  # 150 bar (typical staged combustion)
            throat_radius=0.15,
            expansion_ratio=25.0,
        )
        self.assertGreater(state.thrust_sea_level, 1.0e6)  # > 1 MN
        self.assertGreater(state.isp_sea_level_seconds, 280.0)

    def test_invalid_engine_parameters(self) -> None:
        """Verify validation of negative pressure, radius, and invalid expansion."""
        with self.assertRaises(ValueError):
            self.engine_methalox.evaluate_engine(chamber_pressure=-10.0, throat_radius=0.1, expansion_ratio=20.0)
        with self.assertRaises(ValueError):
            self.engine_methalox.evaluate_engine(chamber_pressure=50.0e5, throat_radius=-0.1, expansion_ratio=20.0)
        with self.assertRaises(ValueError):
            self.engine_methalox.evaluate_engine(chamber_pressure=50.0e5, throat_radius=0.1, expansion_ratio=0.8)

    def test_characteristic_velocity(self) -> None:
        """Verify characteristic velocity c* for Methalox and Hydrolox."""
        # Methalox typical c* ~ 1700 - 1900 m/s
        c_star_methalox = self.engine_methalox.characteristic_velocity()
        self.assertGreater(c_star_methalox, 1600.0)
        self.assertLess(c_star_methalox, 2100.0)

        # Hydrolox typical c* ~ 2200 - 2500 m/s (high energy / low molecular weight)
        c_star_hydrolox = self.engine_hydrolox.characteristic_velocity()
        self.assertGreater(c_star_hydrolox, 2100.0)

    def test_evaluate_engine_performance(self) -> None:
        """Verify full engine state evaluation for a 100 bar chamber pressure engine."""
        pc = 100.0e5  # 100 bar = 10 MPa
        throat_r = 0.1  # 0.1 m radius -> 0.2 m diameter
        eps = 40.0  # Expansion ratio

        state = self.engine_methalox.evaluate_engine(
            chamber_pressure=pc,
            throat_radius=throat_r,
            expansion_ratio=eps,
        )

        # Choked throat area
        expected_at = math.pi * 0.1 * 0.1
        self.assertAlmostEqual(state.throat_area, expected_at, places=5)
        self.assertAlmostEqual(state.exit_area, expected_at * eps, places=5)

        # Mass flow rate m_dot = P_c * A_t / c*
        self.assertAlmostEqual(
            state.mass_flow_rate,
            (pc * expected_at) / state.characteristic_velocity_c_star,
            places=3,
        )

        # Thrust: Vacuum thrust must exceed Sea-Level thrust
        self.assertGreater(state.thrust_vacuum, state.thrust_sea_level)
        self.assertGreater(state.isp_vacuum_seconds, state.isp_sea_level_seconds)

        # Methalox Isp ranges ~ 350 - 385 s in vacuum for eps=40
        self.assertGreater(state.isp_vacuum_seconds, 340.0)
        self.assertLess(state.isp_vacuum_seconds, 400.0)

        # Thrust coefficient C_F ~ 1.7 - 1.9 for vacuum eps=40
        self.assertGreater(state.thrust_coefficient_vacuum, 1.6)

    def test_tsiolkovsky_delta_v(self) -> None:
        """Verify orbital staging rocket equation delta-v."""
        isp = 350.0  # seconds
        wet_mass = 100000.0  # 100 tonnes
        dry_mass = 10000.0   # 10 tonnes (mass ratio 10:1)

        # delta_v = 350 * 9.80665 * ln(10) ~ 350 * 9.80665 * 2.302585 ~ 7904.8 m/s
        dv = self.engine_methalox.tsiolkovsky_delta_v(isp, wet_mass, dry_mass)
        expected_dv = 350.0 * STANDARD_GRAVITY * math.log(10.0)
        self.assertAlmostEqual(dv, expected_dv, places=2)

        with self.assertRaises(ValueError):
            self.engine_methalox.tsiolkovsky_delta_v(isp, wet_mass=100.0, dry_mass=200.0)


if __name__ == "__main__":
    unittest.main()
