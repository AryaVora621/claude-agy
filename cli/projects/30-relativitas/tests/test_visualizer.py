"""Unit tests for Unicode Braille visualizer, ANSI color output, and telemetry HUD."""

import unittest
from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.raytracer import Camera, FrameBuffer, PixelClass, PixelResult
from relativitas.visualizer import BrailleCanvas, BlackHoleVisualizer


class TestVisualizer(unittest.TestCase):
    """Test suite for sub-pixel Braille rasterization and ANSI dashboards."""

    def test_braille_canvas_subpixels(self) -> None:
        canvas = BrailleCanvas(char_width=4, char_height=3)
        self.assertEqual(canvas.pixel_width, 8)
        self.assertEqual(canvas.pixel_height, 12)

        # Set dot at (0, 0) => bit 0x01
        canvas.set_pixel(0, 0, (255, 100, 50))
        self.assertEqual(canvas.grid[0][0], 0x01)

        # Set dot at (1, 0) => bit 0x08 => combined mask 0x09
        canvas.set_pixel(1, 0, (200, 80, 40))
        self.assertEqual(canvas.grid[0][0], 0x09)

        # Render string contains Braille character
        ansi = canvas.render_ansi(true_color=False)
        self.assertIn(chr(0x2800 + 0x09), ansi)

    def test_black_hole_visualizer_hud(self) -> None:
        metric = SchwarzschildMetric(mass=1.0)
        camera = Camera(r=20.0, theta=1.396, fov_degrees=45.0)

        # Dummy frame buffer
        pixels = [
            [
                PixelResult(
                    screen_x=0, screen_y=0, u=0.0, v=0.0,
                    pixel_class=PixelClass.BLACK_HOLE_SHADOW, rgb=(0, 0, 0),
                    redshift_g=0.0, steps=10, final_r=2.0
                )
            ]
        ]
        fb = FrameBuffer(width=1, height=1, pixels=pixels, total_rays=1, shadow_rays=1, disk_rays=0, escaped_rays=0)

        viz = BlackHoleVisualizer(true_color=False)
        hud = viz.format_telemetry_hud(metric, camera, fb)

        self.assertIn("RELATIVITAS", hud)
        self.assertIn("Schwarzschild", hud)
        self.assertIn("Event Horizon", hud)
        self.assertIn("ISCO", hud)

    def test_kerr_hud(self) -> None:
        kerr = KerrMetric(mass=1.0, spin=0.92)
        camera = Camera(r=25.0, theta=1.4)
        fb = FrameBuffer(width=1, height=1, pixels=[[]], total_rays=0, shadow_rays=0, disk_rays=0, escaped_rays=0)

        viz = BlackHoleVisualizer(true_color=False)
        hud = viz.format_telemetry_hud(kerr, camera, fb)

        self.assertIn("Kerr", hud)
        self.assertIn("0.92", hud)


if __name__ == "__main__":
    unittest.main()
