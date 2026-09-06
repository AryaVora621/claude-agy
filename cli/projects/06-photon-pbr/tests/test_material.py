"""
PhotonPBR: Unit Tests for Physically-Based Materials and BRDF Scattering Models.
"""

import unittest
import math
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import HitRecord
from photon.material import Lambertian, Metal, Dielectric, DiffuseLight


class TestMaterials(unittest.TestCase):
    def test_lambertian_scattering(self):
        albedo = Color(0.8, 0.2, 0.1)
        mat = Lambertian(albedo)

        rec = HitRecord(
            point=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            material=mat,
            t=1.0,
            front_face=True
        )
        ray_in = Ray(Vec3(0, 1, 0), Vec3(0, -1, 0))

        did_scatter, attenuation, scattered = mat.scatter(ray_in, rec)
        self.assertTrue(did_scatter)
        self.assertEqual(attenuation.x, albedo.x)
        self.assertEqual(attenuation.y, albedo.y)
        self.assertEqual(attenuation.z, albedo.z)
        self.assertAlmostEqual(scattered.origin.y, 0.0)

    def test_metal_specular_reflection(self):
        albedo = Color(0.9, 0.9, 0.9)
        mat = Metal(albedo, fuzz=0.0)

        # 45 degree incident angle onto flat horizontal surface (normal = (0, 1, 0))
        rec = HitRecord(
            point=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            material=mat,
            t=1.0,
            front_face=True
        )
        ray_in = Ray(Vec3(-1, 1, 0), Vec3(1, -1, 0).normalized())

        did_scatter, attenuation, scattered = mat.scatter(ray_in, rec)
        self.assertTrue(did_scatter)
        self.assertEqual(attenuation.x, albedo.x)
        # Reflected direction should have positive Y
        self.assertGreater(scattered.direction.y, 0.0)
        self.assertAlmostEqual(scattered.direction.x, ray_in.direction.x)
        self.assertAlmostEqual(scattered.direction.y, -ray_in.direction.y)

    def test_dielectric_refraction_and_tir(self):
        glass = Dielectric(index_of_refraction=1.5)

        # Normal perpendicular incidence into glass (air -> glass)
        rec = HitRecord(
            point=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            material=glass,
            t=1.0,
            front_face=True
        )
        ray_in = Ray(Vec3(0, 1, 0), Vec3(0, -1, 0))
        did_scatter, attenuation, scattered = glass.scatter(ray_in, rec)
        self.assertTrue(did_scatter)
        self.assertAlmostEqual(attenuation.x, 1.0)
        self.assertAlmostEqual(attenuation.y, 1.0)
        self.assertAlmostEqual(attenuation.z, 1.0)

        # Steep angle from inside glass to air (glass -> air): front_face = False
        # Normal pointing outward into air is (0, 1, 0), from inside surface normal is (0, -1, 0)
        rec_inside = HitRecord(
            point=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            material=glass,
            t=1.0,
            front_face=False
        )
        shallow_ray = Ray(Vec3(-5, -0.1, 0), Vec3(1, 0.02, 0).normalized())
        did_scatter_tir, _, scattered_tir = glass.scatter(shallow_ray, rec_inside)
        self.assertTrue(did_scatter_tir)

    def test_diffuse_light_emission(self):
        emit_col = Color(10.0, 5.0, 2.0)
        light = DiffuseLight(emit_col)

        rec = HitRecord(
            point=Vec3(0, 5, 0),
            normal=Vec3(0, -1, 0),
            material=light,
            t=2.0
        )
        ray_in = Ray(Vec3(0, 0, 0), Vec3(0, 1, 0))

        # DiffuseLight absorbs incoming rays (does not scatter)
        did_scatter, _, _ = light.scatter(ray_in, rec)
        self.assertFalse(did_scatter)

        # Emits radiant light energy
        emitted = light.emitted(0.5, 0.5, rec.point)
        self.assertEqual(emitted.x, 10.0)
        self.assertEqual(emitted.y, 5.0)
        self.assertEqual(emitted.z, 2.0)


if __name__ == "__main__":
    unittest.main()
