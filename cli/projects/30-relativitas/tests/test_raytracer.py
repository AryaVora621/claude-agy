"""Unit tests for backward curved-spacetime ray tracer and camera projection."""

import math
import unittest
from relativitas.metric import SchwarzschildMetric, KerrMetric
from relativitas.raytracer import RayTracer, Camera, PixelClass


class TestRayTracer(unittest.TestCase):
    """Test suite for pinhole camera projection and backward ray tracing."""

    def setUp(self) -> None:
        self.metric = SchwarzschildMetric(mass=1.0)
        self.raytracer = RayTracer(self.metric, max_steps=350)
        self.camera = Camera(r=20.0, theta=1.396, fov_degrees=45.0)

    def test_camera_setup(self) -> None:
        self.assertEqual(self.camera.r, 20.0)
        self.assertEqual(self.camera.theta, 1.396)
        self.assertAlmostEqual(self.camera.fov_rad, math.radians(45.0), places=6)

    def test_screen_ray_null_condition(self) -> None:
        # For screen center (u=0, v=0), the ray must satisfy exact null condition g_uv p^u p^v = 0
        p = self.raytracer.screen_ray_direction(0.0, 0.0, self.camera)
        norm = self.metric.scalar_norm(self.camera.coords, p)
        self.assertAlmostEqual(norm, 0.0, places=5)
        # Check future-directed: pt > 0
        self.assertGreater(p[0], 0.0)

    def test_off_center_ray_null_condition(self) -> None:
        # For corner screen coords (u=0.5, v=-0.4)
        p = self.raytracer.screen_ray_direction(0.5, -0.4, self.camera)
        norm = self.metric.scalar_norm(self.camera.coords, p)
        self.assertAlmostEqual(norm, 0.0, places=5)
        self.assertGreater(p[0], 0.0)

    def test_single_ray_classification(self) -> None:
        # Center ray points directly at black hole => should hit black hole shadow
        p_res = self.raytracer.trace_single_ray(0.0, 0.0, self.camera)
        self.assertEqual(p_res.pixel_class, PixelClass.BLACK_HOLE_SHADOW)
        self.assertEqual(p_res.rgb, (0, 0, 0))

        # Ray far to the side (u=2.0) should escape to celestial sphere
        p_res_far = self.raytracer.trace_single_ray(2.0, 1.5, self.camera)
        self.assertEqual(p_res_far.pixel_class, PixelClass.CELESTIAL_BACKGROUND)

    def test_framebuffer_render(self) -> None:
        fb = self.raytracer.render_frame(self.camera, width=8, height=6)
        self.assertEqual(fb.width, 8)
        self.assertEqual(fb.height, 6)
        self.assertEqual(fb.total_rays, 48)
        self.assertEqual(len(fb.pixels), 6)
        self.assertEqual(len(fb.pixels[0]), 8)

        # Confirm all rays categorized
        self.assertEqual(
            fb.shadow_rays + fb.disk_rays + fb.escaped_rays,
            fb.total_rays,
        )

        rgb_grid = fb.get_rgb_grid()
        self.assertEqual(len(rgb_grid), 6)
        self.assertEqual(len(rgb_grid[0]), 8)


if __name__ == "__main__":
    unittest.main()
