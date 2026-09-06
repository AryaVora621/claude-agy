"""Unit tests for Sub-Pixel Unicode Braille Visualizer and HUD."""

import unittest
from luminawave.grid import Grid2D
from luminawave.fdtd import FDTDSimulator
from luminawave.visualizer import BrailleEMCanvas, PhotonicsWorkbenchHUD


class TestVisualizer(unittest.TestCase):
    def test_braille_dot_map_coverage(self):
        """Verify 2x4 sub-pixel Braille dot matrix offsets."""
        dot_map = BrailleEMCanvas.DOT_MAP
        self.assertEqual(len(dot_map), 4)
        self.assertEqual(len(dot_map[0]), 2)
        # Check standard ISO/IEC 11548 Braille dot bit values
        self.assertEqual(dot_map[0][0], 0x01)
        self.assertEqual(dot_map[1][0], 0x02)
        self.assertEqual(dot_map[2][0], 0x04)
        self.assertEqual(dot_map[0][1], 0x08)
        self.assertEqual(dot_map[1][1], 0x10)
        self.assertEqual(dot_map[2][1], 0x20)
        self.assertEqual(dot_map[3][0], 0x40)
        self.assertEqual(dot_map[3][1], 0x80)

    def test_canvas_rendering_format(self):
        """Verify terminal Braille canvas output lines and non-empty content."""
        grid = Grid2D(nx=60, ny=40)
        # Put some sinusoidal field into Ez
        for y in range(40):
            for x in range(60):
                grid.ez[grid.idx(x, y)] = 10.0 if (x + y) % 6 == 0 else 0.0

        canvas = BrailleEMCanvas.render_field(
            grid, width_chars=30, height_rows=10, use_color=False
        )
        lines = canvas.split("\n")
        self.assertEqual(len(lines), 10)
        for line in lines:
            self.assertEqual(len(line), 30)

    def test_spectrum_sparkline(self):
        """Verify frequency spectrum sparkline plotter."""
        vals = [0.1, 0.3, 0.8, 0.95, 0.8, 0.3, 0.1]
        sparkline = BrailleEMCanvas.plot_spectrum_sparkline(
            vals, width_chars=20, height_rows=4, min_val=0.0, max_val=1.0
        )
        lines = sparkline.split("\n")
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertEqual(len(line), 20)

    def test_workbench_hud_render(self):
        """Verify consolidated PhotonicsWorkbenchHUD rendering."""
        grid = Grid2D(nx=50, ny=40)
        sim = FDTDSimulator(grid)
        hud = PhotonicsWorkbenchHUD(sim)

        output = hud.render(width_chars=60, height_rows=10)
        self.assertIn("LUMINAWAVE 2D FDTD", output)
        self.assertIn("Peak |Ez|", output)
        self.assertIn("Energy", output)


if __name__ == "__main__":
    unittest.main()
