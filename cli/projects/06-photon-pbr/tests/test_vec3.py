"""
PhotonPBR: Unit Tests for Vec3, Ray, and Color Operations.
"""

import unittest
import math
from photon.vec3 import Vec3, Ray, Color


class TestVec3(unittest.TestCase):
    def test_init_and_repr(self):
        v = Vec3(1.0, 2.5, -3.0)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 2.5)
        self.assertEqual(v.z, -3.0)
        self.assertIn("Vec3(1.0000, 2.5000, -3.0000)", repr(v))

    def test_arithmetic(self):
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, -5.0, 6.0)

        # Addition
        add_res = v1 + v2
        self.assertAlmostEqual(add_res.x, 5.0)
        self.assertAlmostEqual(add_res.y, -3.0)
        self.assertAlmostEqual(add_res.z, 9.0)

        # Subtraction
        sub_res = v1 - v2
        self.assertAlmostEqual(sub_res.x, -3.0)
        self.assertAlmostEqual(sub_res.y, 7.0)
        self.assertAlmostEqual(sub_res.z, -3.0)

        # Scalar multiplication
        mul_res = v1 * 2.5
        self.assertAlmostEqual(mul_res.x, 2.5)
        self.assertAlmostEqual(mul_res.y, 5.0)
        self.assertAlmostEqual(mul_res.z, 7.5)

        # Reverse scalar multiplication
        rmul_res = 3.0 * v1
        self.assertAlmostEqual(rmul_res.x, 3.0)
        self.assertAlmostEqual(rmul_res.y, 6.0)
        self.assertAlmostEqual(rmul_res.z, 9.0)

        # Component-wise (Hadamard) multiplication
        hadamard = v1 * Vec3(2.0, 0.5, -1.0)
        self.assertAlmostEqual(hadamard.x, 2.0)
        self.assertAlmostEqual(hadamard.y, 1.0)
        self.assertAlmostEqual(hadamard.z, -3.0)

        # True division
        div_res = v1 / 2.0
        self.assertAlmostEqual(div_res.x, 0.5)
        self.assertAlmostEqual(div_res.y, 1.0)
        self.assertAlmostEqual(div_res.z, 1.5)

        # Negation
        neg_res = -v1
        self.assertAlmostEqual(neg_res.x, -1.0)
        self.assertAlmostEqual(neg_res.y, -2.0)
        self.assertAlmostEqual(neg_res.z, -3.0)

    def test_indexing(self):
        v = Vec3(10.0, 20.0, 30.0)
        self.assertEqual(v[0], 10.0)
        self.assertEqual(v[1], 20.0)
        self.assertEqual(v[2], 30.0)
        with self.assertRaises(IndexError):
            _ = v[3]

    def test_dot_and_cross(self):
        v1 = Vec3(1.0, 0.0, 0.0)
        v2 = Vec3(0.0, 1.0, 0.0)

        # Dot product of orthogonal vectors is 0
        self.assertAlmostEqual(v1.dot(v2), 0.0)
        # Dot product with self is length squared
        self.assertAlmostEqual(v1.dot(v1), 1.0)

        # Cross product of X and Y gives Z
        cross_z = v1.cross(v2)
        self.assertAlmostEqual(cross_z.x, 0.0)
        self.assertAlmostEqual(cross_z.y, 0.0)
        self.assertAlmostEqual(cross_z.z, 1.0)

    def test_length_and_normalize(self):
        v = Vec3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.length_squared(), 25.0)
        self.assertAlmostEqual(v.length(), 5.0)

        u = v.normalized()
        self.assertAlmostEqual(u.length(), 1.0)
        self.assertAlmostEqual(u.x, 0.6)
        self.assertAlmostEqual(u.y, 0.8)
        self.assertAlmostEqual(u.z, 0.0)

        zero_v = Vec3(0, 0, 0)
        self.assertTrue(zero_v.near_zero())
        self.assertEqual(zero_v.normalized().length(), 0.0)

    def test_reflect(self):
        # 45-degree incident ray reflecting off horizontal floor (normal = (0, 1, 0))
        ray_in = Vec3(1.0, -1.0, 0.0).normalized()
        normal = Vec3(0.0, 1.0, 0.0)
        refl = ray_in.reflect(normal)

        self.assertAlmostEqual(refl.x, ray_in.x)
        self.assertAlmostEqual(refl.y, -ray_in.y)
        self.assertAlmostEqual(refl.z, 0.0)

    def test_refract(self):
        # Perpendicular incident ray into medium with eta=1.5 (air to glass)
        incident = Vec3(0.0, -1.0, 0.0)
        normal = Vec3(0.0, 1.0, 0.0)
        success, refracted = incident.refract(normal, 1.0 / 1.5)
        self.assertTrue(success)
        self.assertAlmostEqual(refracted.x, 0.0)
        self.assertAlmostEqual(refracted.y, -1.0)
        self.assertAlmostEqual(refracted.z, 0.0)

        # Shallow angle from glass to air (eta = 1.5): should trigger Total Internal Reflection
        shallow_incident = Vec3(1.0, -0.1, 0.0).normalized()
        tir_success, _ = shallow_incident.refract(normal, 1.5)
        self.assertFalse(tir_success)

    def test_color_rgb_and_tone_mapping(self):
        col = Vec3(0.25, 0.0, 1.0)
        # Gamma 2.0 (square root): sqrt(0.25) = 0.5 -> 256 * 0.5 = 128
        # sqrt(0.0) = 0.0 -> 0
        # sqrt(1.0) = 1.0 -> clamped to 255
        r, g, b = col.to_rgb_bytes(samples_per_pixel=1)
        self.assertEqual(r, 128)
        self.assertEqual(g, 0)
        self.assertEqual(b, 255)

        # Multi-sample averaging: 4 samples of (1.0, 1.0, 1.0) accumulated is (4.0, 4.0, 4.0)
        accum_col = Vec3(4.0, 4.0, 4.0)
        r4, g4, b4 = accum_col.to_rgb_bytes(samples_per_pixel=4)
        self.assertEqual(r4, 255)
        self.assertEqual(g4, 255)
        self.assertEqual(b4, 255)

        # ANSI truecolor code string format
        ansi_str = col.to_ansi_truecolor(samples_per_pixel=1)
        self.assertEqual(ansi_str, "\033[38;2;128;0;255m")


class TestRay(unittest.TestCase):
    def test_ray_evaluation(self):
        orig = Vec3(1.0, 2.0, 3.0)
        direction = Vec3(0.0, 0.0, 2.0)
        r = Ray(orig, direction)

        # Normalized direction: (0, 0, 1)
        self.assertAlmostEqual(r.direction.x, 0.0)
        self.assertAlmostEqual(r.direction.y, 0.0)
        self.assertAlmostEqual(r.direction.z, 1.0)

        p0 = r.at(0.0)
        self.assertAlmostEqual(p0.x, 1.0)
        self.assertAlmostEqual(p0.y, 2.0)
        self.assertAlmostEqual(p0.z, 3.0)

        p5 = r.at(5.0)
        self.assertAlmostEqual(p5.x, 1.0)
        self.assertAlmostEqual(p5.y, 2.0)
        self.assertAlmostEqual(p5.z, 8.0)


if __name__ == "__main__":
    unittest.main()
