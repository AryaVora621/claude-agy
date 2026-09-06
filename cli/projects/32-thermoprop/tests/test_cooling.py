"""
Unit tests for Regenerative Chamber Cooling & Bartz Heat Transfer (thermoprop/cooling.py).
"""

from __future__ import annotations

import unittest

from thermoprop.cooling import (
    CoolantChannelProperties,
    RegenerativeCoolingEngine,
    WallMaterial,
)
from thermoprop.propulsion import (
    PropellantLibrary,
    RocketPropulsionEngine,
)


class TestRegenerativeCooling(unittest.TestCase):
    """Test suite for rocket chamber aerothermodynamic heat transfer."""

    def setUp(self) -> None:
        prop = PropellantLibrary.methalox()
        engine = RocketPropulsionEngine(prop)
        self.state = engine.evaluate_engine(
            chamber_pressure=80.0e5,  # 80 bar
            throat_radius=0.08,
            expansion_ratio=35.0,
        )
        self.gas = prop.gas
        self.cooling = RegenerativeCoolingEngine(
            engine_state=self.state,
            gas=self.gas,
            material=WallMaterial.copper_cucrzr(thickness_mm=1.5),
        )

    def test_wall_material_properties(self) -> None:
        """Verify copper vs superalloy material thresholds."""
        cucrzr = WallMaterial.copper_cucrzr()
        self.assertAlmostEqual(cucrzr.thermal_conductivity, 320.0)
        self.assertEqual(cucrzr.max_allowable_temperature, 950.0)

        inconel = WallMaterial.inconel_718()
        self.assertAlmostEqual(inconel.thermal_conductivity, 20.0)
        self.assertEqual(inconel.max_allowable_temperature, 1250.0)

    def test_coolant_hydraulic_diameter_and_convection(self) -> None:
        """Verify coolant channel hydraulic diameter and Dittus-Boelter heat transfer."""
        hc = self.cooling.coolant_heat_coefficient()
        self.assertGreater(hc, 1000.0)  # High convective transfer in turbulent cooling channels

    def test_bartz_heat_transfer_coefficient(self) -> None:
        """Verify Bartz convective heat transfer calculation at throat and expansion."""
        # At throat (Mach = 1.0, area_ratio = 1.0)
        hg_throat = self.cooling.bartz_gas_coefficient(
            local_mach=1.0,
            local_area_ratio=1.0,
            throat_radius=0.08,
        )
        self.assertGreater(hg_throat, 500.0)

        # In nozzle divergence (Mach = 2.5, area_ratio = 4.0)
        hg_divergence = self.cooling.bartz_gas_coefficient(
            local_mach=2.5,
            local_area_ratio=4.0,
            throat_radius=0.08,
        )
        # Heat transfer coefficient drops downstream of throat
        self.assertLess(hg_divergence, hg_throat)

    def test_station_thermal_equilibrium(self) -> None:
        """Verify coupled 1D thermal resistance network solution."""
        res = self.cooling.solve_station_thermal_equilibrium(
            x=0.0,
            radius=0.08,
            mach=1.0,
            throat_radius=0.08,
            coolant_temp=120.0,
        )

        # Heat flux must be positive
        self.assertGreater(res.heat_flux, 0.0)
        # Gas wall temp must be higher than coolant wall temp
        self.assertGreater(res.wall_gas_temperature, res.wall_coolant_temperature)
        # Coolant wall temp must be higher than bulk coolant temp
        self.assertGreater(res.wall_coolant_temperature, res.coolant_temperature)
        # Adiabatic recovery temp must exceed wall gas temp
        self.assertGreater(res.adiabatic_wall_temperature, res.wall_gas_temperature)

    def test_inconel_versus_copper_wall_temperature(self) -> None:
        """Verify lower thermal conductivity of Inconel results in hotter gas wall temperature."""
        cooling_inconel = RegenerativeCoolingEngine(
            engine_state=self.state,
            gas=self.gas,
            material=WallMaterial.inconel_718(thickness_mm=1.5),
        )

        res_cu = self.cooling.solve_station_thermal_equilibrium(0.0, 0.08, 1.0, 0.08)
        res_inc = cooling_inconel.solve_station_thermal_equilibrium(0.0, 0.08, 1.0, 0.08)

        # Inconel lower conductivity (20 W/mK vs 320 W/mK) causes higher gas-side wall temperature
        self.assertGreater(res_inc.wall_gas_temperature, res_cu.wall_gas_temperature)


if __name__ == "__main__":
    unittest.main()
