"""Unit Tests for Sub-Pixel Braille Canvas, Butterfly Visualizer, and Telemetry HUD."""

import unittest

from latticeguard.ring import Polynomial
from latticeguard.ind_cpa import PARAMS_512, PARAMS_768, PARAMS_1024
from latticeguard.visualizer import (
    BrailleLatticeCanvas,
    NTTButterflyVisualizer,
    PQCWorkbenchHUD,
    rgb_ansi,
    color_gradient,
)


class TestVisualizer(unittest.TestCase):
    """Test suite for Braille canvas and terminal telemetry HUD."""

    def test_braille_canvas_plotting(self) -> None:
        """Verify sub-pixel Braille plotting and boundary limits."""
        canvas = BrailleLatticeCanvas(char_width=64, char_height=10)
        self.assertEqual(canvas.pixel_width, 128)
        self.assertEqual(canvas.pixel_height, 40)

        # Plot test polynomial
        p = Polynomial([i * 10 for i in range(256)])
        canvas.plot_polynomial(p, centered=True)
        rendered = canvas.render()
        self.assertIn("\033[38;2;", rendered)
        self.assertIn("+", rendered)

    def test_butterfly_diagram(self) -> None:
        """Verify NTT butterfly diagram renders all 7 stages."""
        text = NTTButterflyVisualizer.render_butterfly_summary()
        self.assertIn("Stage 1", text)
        self.assertIn("Stage 7", text)
        self.assertIn("Cooley-Tukey", text)

    def test_pqc_hud_tables(self) -> None:
        """Verify parameter cards for all three parameter sets."""
        for p in [PARAMS_512, PARAMS_768, PARAMS_1024]:
            table = PQCWorkbenchHUD.render_params_table(p)
            self.assertIn(p.name, table)
            self.assertIn("Ring Modulus", table)
            self.assertIn("Ciphertext Size", table)

        p_s = Polynomial([1] * 256)
        p_e = Polynomial([2] * 256)
        p_t = Polynomial([100] * 256)
        noise_hud = PQCWorkbenchHUD.render_noise_telemetry(p_s, p_e, p_t)
        self.assertIn("||f||_inf", noise_hud)


if __name__ == "__main__":
    unittest.main()
