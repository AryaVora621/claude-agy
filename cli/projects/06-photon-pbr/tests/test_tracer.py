"""
PhotonPBR: Unit Tests for Path Tracer, Camera, and Framebuffer Exporters.
"""

import unittest
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import Sphere, HittableList
from photon.material import Lambertian, DiffuseLight
from photon.camera import Camera
from photon.tracer import PathTracer, ray_color
from photon.framebuffer import Framebuffer


class TestTracerAndCamera(unittest.TestCase):
    def test_ray_color_miss_returns_background(self):
        world = HittableList([])
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        bg = Color(0.2, 0.4, 0.6)

        col = ray_color(ray, world, depth=4, max_depth=4, background=bg)
        self.assertEqual(col.x, bg.x)
        self.assertEqual(col.y, bg.y)
        self.assertEqual(col.z, bg.z)

    def test_ray_color_light_emission(self):
        emit = Color(12.0, 10.0, 8.0)
        light = DiffuseLight(emit)
        sphere = Sphere(Vec3(0, 0, 5), 1.0, light)
        world = HittableList([sphere])

        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        col = ray_color(ray, world, depth=4, max_depth=4, background=Color(0, 0, 0))

        self.assertEqual(col.x, emit.x)
        self.assertEqual(col.y, emit.y)
        self.assertEqual(col.z, emit.z)

    def test_camera_ray_generation(self):
        cam = Camera(
            lookfrom=Vec3(0, 0, 0),
            lookat=Vec3(0, 0, 1),
            vup=Vec3(0, 1, 0),
            vfov_degrees=90.0,
            aspect_ratio=1.0,
            aperture=0.0
        )

        # Center ray (u=0.5, v=0.5) should point directly along +Z
        r_center = cam.get_ray(0.5, 0.5)
        self.assertAlmostEqual(r_center.direction.x, 0.0, places=4)
        self.assertAlmostEqual(r_center.direction.y, 0.0, places=4)
        self.assertAlmostEqual(r_center.direction.z, 1.0, places=4)

    def test_pathtracer_render_dimensions(self):
        tracer = PathTracer(max_depth=3)
        cam = Camera(lookfrom=Vec3(0, 0, -2), lookat=Vec3(0, 0, 0), vup=Vec3(0, 1, 0))
        world = HittableList([Sphere(Vec3(0, 0, 0), 0.5, Lambertian(Color(0.8, 0.8, 0.8)))])

        width = 8
        height = 6
        samples = 2
        pixels = tracer.render(
            world=world,
            camera=cam,
            width=width,
            height=height,
            samples_per_pixel=samples,
            background=Color(0.1, 0.1, 0.1)
        )

        self.assertEqual(len(pixels), height)
        for row in pixels:
            self.assertEqual(len(row), width)
        self.assertEqual(tracer.rays_cast, width * height * samples)

        fb = Framebuffer(width, height, pixels)
        ascii_out = fb.to_ascii(samples_per_pixel=samples)
        self.assertEqual(len(ascii_out.splitlines()), height)

        ansi_out = fb.to_ansi_truecolor(samples_per_pixel=samples)
        self.assertEqual(len(ansi_out.splitlines()), height // 2)


if __name__ == "__main__":
    unittest.main()
