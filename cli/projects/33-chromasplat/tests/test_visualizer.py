"""Unit tests for BrailleCanvas and SplatVisualizer with telemetry HUD."""

import unittest

from chromasplat.projection import Camera
from chromasplat.renderer import GaussianRenderer
from chromasplat.scene import SceneFactory
from chromasplat.visualizer import BrailleCanvas, SplatVisualizer


class TestVisualizer(unittest.TestCase):
    """Test suite for sub-pixel Unicode Braille terminal rasterizer and HUD."""

    def test_braille_canvas_subpixel_dimensions(self) -> None:
        canvas = BrailleCanvas(char_width=40, char_height=15)
        # 2 pixels per char horizontal, 4 pixels per char vertical
        self.assertEqual(canvas.pixel_width, 80)
        self.assertEqual(canvas.pixel_height, 60)

    def test_visualizer_and_hud_output(self) -> None:
        vis = SplatVisualizer(char_width=40, char_height=15)
        scene = SceneFactory.orbiting_rings(num_planet_splats=10, num_ring_splats=20)
        cam = Camera.orbit(width=vis.pixel_width, height=vis.pixel_height, azimuth_deg=0.0, elevation_deg=0.0, distance=3.0)

        renderer = GaussianRenderer(sh_degree=0)
        res = renderer.render(scene, cam)

        output = vis.render_frame_with_hud(
            scene,
            cam,
            res,
            renderer.last_profile,
            azimuth_deg=0.0,
            elevation_deg=0.0,
            distance=3.0,
        )

        self.assertIn("CHROMASPLAT", output)
        self.assertIn("SaturnianRings", output)
        self.assertIn("FPS:", output)
        lines = output.splitlines()
        self.assertGreater(len(lines), 15)


if __name__ == "__main__":
    unittest.main()
