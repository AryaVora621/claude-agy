"""Unit tests for VolumeRasterizer and front-to-back alpha compositing."""

import unittest

from chromasplat.gaussian import Gaussian3D
from chromasplat.projection import Camera, ProjectionEngine
from chromasplat.rasterizer import VolumeRasterizer


class TestRasterizer(unittest.TestCase):
    """Test suite for volume rasterization, tile binning, and ray accumulation."""

    def setUp(self) -> None:
        self.cam = Camera.orbit(width=64, height=32, azimuth_deg=0.0, elevation_deg=0.0, distance=3.0)
        self.engine = ProjectionEngine()
        self.rasterizer = VolumeRasterizer(tile_size=16)

    def test_direct_vs_tiled_rasterization_agreement(self) -> None:
        g1 = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.2, color_rgb=(1.0, 0.0, 0.0))
        g2 = Gaussian3D.isotropic(position=(0.1, 0.1, -0.3), radius=0.2, color_rgb=(0.0, 1.0, 0.0))
        projected = self.engine.project_scene([g1, g2], self.cam)

        res_direct = self.rasterizer.rasterize(self.cam.width, self.cam.height, projected, use_tiling=False)
        res_tiled = self.rasterizer.rasterize(self.cam.width, self.cam.height, projected, use_tiling=True)

        # Both modes should produce nearly identical images
        max_diff = 0.0
        for y in range(self.cam.height):
            for x in range(self.cam.width):
                c_d = res_direct.colors[y][x]
                c_t = res_tiled.colors[y][x]
                diff = max(abs(c_d[0] - c_t[0]), abs(c_d[1] - c_t[1]), abs(c_d[2] - c_t[2]))
                if diff > max_diff:
                    max_diff = diff

        self.assertLess(max_diff, 0.05)

    def test_opaque_occlusion_and_transmittance(self) -> None:
        # Red foreground sphere covering center
        g_front = Gaussian3D.isotropic(position=(0.0, 0.0, 0.5), radius=0.4, opacity=1.0, color_rgb=(1.0, 0.0, 0.0))
        # Blue background sphere behind it
        g_back = Gaussian3D.isotropic(position=(0.0, 0.0, -0.5), radius=0.4, opacity=1.0, color_rgb=(0.0, 0.0, 1.0))

        projected = self.engine.project_scene([g_front, g_back], self.cam)
        res = self.rasterizer.rasterize(self.cam.width, self.cam.height, projected)

        center_color = res.get_pixel_rgb24(32, 16)
        # Red should dominate completely over blue at center
        self.assertGreater(center_color[0], 200)  # High Red
        self.assertLess(center_color[2], 50)     # Negligible Blue

    def test_ppm_image_export(self) -> None:
        g = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.2, color_rgb=(0.5, 0.8, 0.2))
        projected = self.engine.project_scene([g], self.cam)
        res = self.rasterizer.rasterize(self.cam.width, self.cam.height, projected)

        ppm = res.to_ppm_bytes()
        self.assertTrue(ppm.startswith(b"P6\n64 32\n255\n"))
        expected_len = len(b"P6\n64 32\n255\n") + 64 * 32 * 3
        self.assertEqual(len(ppm), expected_len)


if __name__ == "__main__":
    unittest.main()
