"""
Unit tests for Sub-Pixel Unicode Braille Visualizer & Tokamak Telemetry HUD.
"""

import unittest

from stellarfusion.particles import ParticleState
from stellarfusion.poincare import PoincarePuncture
from stellarfusion.visualizer import BrailleCanvas, TokamakVisualizer


class TestBrailleCanvas(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BrailleCanvas(
            char_width=40,
            char_height=20,
            r_min=1.0,
            r_max=5.0,
            z_min=-2.0,
            z_max=2.0,
        )

    def test_dimensions_and_clearing(self) -> None:
        self.assertEqual(self.canvas.pixel_width, 80)
        self.assertEqual(self.canvas.pixel_height, 80)
        self.canvas.clear()
        self.assertEqual(self.canvas.grid[0][0], 0)

    def test_pixel_and_point_plotting(self) -> None:
        # Plot center coordinate (3.0, 0.0)
        self.canvas.plot_point(3.0, 0.0, color=(255, 200, 100))
        rendered = self.canvas.render_to_string()
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 0)

    def test_line_drawing(self) -> None:
        self.canvas.draw_line(2.0, -1.0, 4.0, 1.0, color=(100, 255, 150))
        rendered = self.canvas.render_to_string()
        # Ensure non-blank Braille characters exist
        has_braille_dots = any(ord(char) > 0x2800 for char in rendered)
        self.assertTrue(has_braille_dots)


class TestTokamakVisualizer(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BrailleCanvas(char_width=50, char_height=25)
        self.viz = TokamakVisualizer(self.canvas)

    def test_temperature_color_scale(self) -> None:
        c_core = self.viz.temperature_color(0.05)
        c_edge = self.viz.temperature_color(0.98)
        self.assertEqual(len(c_core), 3)
        self.assertEqual(len(c_edge), 3)
        self.assertNotEqual(c_core, c_edge)

    def test_drawing_components(self) -> None:
        self.viz.draw_flux_contours(r_axis=3.0, z_axis=0.0, a_minor=0.8, kappa=1.5, num_surfaces=3)
        self.viz.draw_vacuum_vessel(r_axis=3.0, z_axis=0.0, a_wall=1.1, kappa_wall=1.6)

        trajectory = [
            ParticleState(x=3.2, y=0.0, z=0.1, vx=1.0, vy=2.0, vz=0.5),
            ParticleState(x=3.3, y=0.1, z=0.2, vx=1.1, vy=2.0, vz=0.4),
        ]
        self.viz.draw_particle_orbit(trajectory)

        punctures = [PoincarePuncture(r=3.2, z=0.1, turn=1, psi_value=0.5)]
        self.viz.draw_poincare_punctures(punctures)

        output = self.canvas.render_to_string()
        self.assertIn("\033[38;2;", output)

    def test_hud_formatting(self) -> None:
        hud = self.viz.format_telemetry_hud(
            r_0=3.0,
            a_minor=1.0,
            b_0=2.5,
            i_p=1.5e6,
            q_0=1.05,
            q_95=3.20,
            t_core_kev=15.0,
            p_core_kpa=80.0,
            beta_toroidal_pct=3.5,
        )
        self.assertIn("STELLARFUSION", hud)
        self.assertIn("1.500 MA", hud)
        self.assertIn("15.00 keV", hud)


if __name__ == "__main__":
    unittest.main()
