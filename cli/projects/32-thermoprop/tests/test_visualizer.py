"""
Unit tests for Braille Rocket Visualizer & Telemetry HUD (thermoprop/visualizer.py).
"""

from __future__ import annotations

import unittest

from thermoprop.cooling import RegenerativeCoolingEngine, WallMaterial
from thermoprop.gas_dynamics import GasProperties
from thermoprop.moc_nozzle import MethodOfCharacteristicsNozzle
from thermoprop.plume import ExhaustPlumeEngine
from thermoprop.propulsion import PropellantLibrary, RocketPropulsionEngine
from thermoprop.visualizer import BrailleCanvas, RocketVisualizer


class TestVisualizer(unittest.TestCase):
    """Test suite for sub-pixel Braille rendering and HUD formatting."""

    def setUp(self) -> None:
        self.canvas = BrailleCanvas(char_width=40, char_height=20)
        self.viz = RocketVisualizer(self.canvas)

    def test_braille_canvas_primitives(self) -> None:
        """Verify canvas sub-pixel plotting, lines, and string rendering."""
        self.canvas.clear()
        self.canvas.plot_point(0.0, 0.0, color=(255, 100, 50))
        self.canvas.draw_line(-1.0, -1.0, 1.0, 1.0, color=(100, 200, 255))

        out = self.canvas.render_to_string()
        self.assertIsInstance(out, str)
        self.assertIn("\033[38;2;", out)  # Contains ANSI 24-bit TrueColor

    def test_nozzle_and_plume_rendering(self) -> None:
        """Verify nozzle contour and exhaust plume rendering onto Braille canvas."""
        air = GasProperties.air()
        moc = MethodOfCharacteristicsNozzle(air)
        contour = moc.design_minimum_length_nozzle(target_exit_mach=2.0, throat_height=0.5, num_expansion_waves=4)

        plume_engine = ExhaustPlumeEngine(air)
        plume = plume_engine.simulate_plume(
            x_exit=contour.length,
            exit_radius=contour.exit_radius,
            exit_mach=2.0,
            p_exit=80000.0,
            chamber_pressure=50.0e5,
            p_ambient=101325.0,
            plume_length=4.0,
            num_cells=2,
        )

        self.viz.draw_nozzle_contour(contour)
        self.viz.draw_exhaust_plume(plume)
        rendered = self.canvas.render_to_string()
        self.assertGreater(len(rendered), 100)

    def test_telemetry_hud_formatting(self) -> None:
        """Verify rocket telemetry HUD strings and metrics formatting."""
        prop = PropellantLibrary.methalox()
        engine = RocketPropulsionEngine(prop)
        state = engine.evaluate_engine(
            chamber_pressure=100.0e5,
            throat_radius=0.1,
            expansion_ratio=40.0,
        )
        plume_engine = ExhaustPlumeEngine(prop.gas)
        plume = plume_engine.simulate_plume(
            x_exit=1.5,
            exit_radius=0.1 * (40.0**0.5),
            exit_mach=state.exit_mach,
            p_exit=state.exit_pressure,
            chamber_pressure=state.chamber_pressure,
            p_ambient=101325.0,
        )
        cooling = RegenerativeCoolingEngine(state, prop.gas, WallMaterial.copper_cucrzr())
        thermal = cooling.solve_station_thermal_equilibrium(0.0, 0.1, 1.0, 0.1)

        hud = self.viz.format_telemetry_hud(state, plume, thermal)
        self.assertIn("THERMOPROP", hud)
        self.assertIn("COMBUSTION", hud)
        self.assertIn("NOZZLE MOC", hud)
        self.assertIn("PERFORMANCE", hud)
        self.assertIn("HEAT TRANSFER", hud)

    def test_canvas_bounds_clipping(self) -> None:
        """Verify out-of-bounds coordinates are safely clipped without index error."""
        # Plot far outside physical coordinate bounds
        self.canvas.plot_point(100.0, 100.0)
        self.canvas.plot_point(-100.0, -100.0)
        self.canvas.draw_line(-50.0, -50.0, 50.0, 50.0)
        out = self.canvas.render_to_string()
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
