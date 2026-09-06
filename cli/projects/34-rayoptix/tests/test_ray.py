"""Unit tests for 3D Vector Ray, Snell Refraction, Reflection and Fresnel."""

import math
import unittest

from rayoptix.ray import (
    Ray,
    fresnel_coefficients,
    reflect_vector,
    refract_vector,
    vec3_cross,
    vec3_dot,
    vec3_norm,
    vec3_normalize,
)


class TestRayOptics(unittest.TestCase):
    """Test suite for 3D ray mechanics and vector Snell law."""

    def test_vector_operations(self) -> None:
        v1 = (1.0, 2.0, 3.0)
        v2 = (4.0, -5.0, 6.0)
        self.assertAlmostEqual(vec3_dot(v1, v2), 12.0)
        norm = vec3_norm((3.0, 4.0, 0.0))
        self.assertAlmostEqual(norm, 5.0)

        u = vec3_normalize((0.0, 0.0, 10.0))
        self.assertEqual(u, (0.0, 0.0, 1.0))

        cross = vec3_cross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(cross[0], 0.0)
        self.assertAlmostEqual(cross[1], 0.0)
        self.assertAlmostEqual(cross[2], 1.0)

    def test_ray_advance(self) -> None:
        ray = Ray(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0))
        adv = ray.advance(10.0, 1.5)
        self.assertAlmostEqual(adv.origin[2], 10.0)
        self.assertAlmostEqual(adv.opl, 15.0)

    def test_snell_refraction_normal_incidence(self) -> None:
        d_in = (0.0, 0.0, 1.0)
        normal = (0.0, 0.0, -1.0)
        d_out = refract_vector(d_in, normal, 1.0, 1.5)
        self.assertIsNotNone(d_out)
        self.assertAlmostEqual(d_out[0], 0.0)
        self.assertAlmostEqual(d_out[1], 0.0)
        self.assertAlmostEqual(d_out[2], 1.0)

    def test_snell_refraction_oblique(self) -> None:
        # 45 deg incidence from air (n=1.0) to glass (n=1.5)
        sin_theta1 = math.sin(math.pi / 4.0)
        d_in = (sin_theta1, 0.0, math.cos(math.pi / 4.0))
        normal = (0.0, 0.0, -1.0)
        d_out = refract_vector(d_in, normal, 1.0, 1.5)
        self.assertIsNotNone(d_out)

        # Expected angle by Snell's law: n1 sin theta1 = n2 sin theta2
        expected_sin_theta2 = (1.0 / 1.5) * sin_theta1
        actual_sin_theta2 = d_out[0]
        self.assertAlmostEqual(actual_sin_theta2, expected_sin_theta2, places=6)

    def test_total_internal_reflection(self) -> None:
        # Ray travelling inside glass (n=1.5) towards air (n=1.0) beyond critical angle
        # Critical angle = arcsin(1.0 / 1.5) = 41.81 degrees
        theta = math.radians(50.0)  # > critical angle
        d_in = (math.sin(theta), 0.0, math.cos(theta))
        normal = (0.0, 0.0, -1.0)
        d_out = refract_vector(d_in, normal, 1.5, 1.0)
        self.assertIsNone(d_out)  # TIR must return None

    def test_reflection_vector(self) -> None:
        theta = math.radians(30.0)
        d_in = (math.sin(theta), 0.0, math.cos(theta))
        normal = (0.0, 0.0, -1.0)
        d_refl = reflect_vector(d_in, normal)
        self.assertAlmostEqual(d_refl[0], math.sin(theta))
        self.assertAlmostEqual(d_refl[2], -math.cos(theta))

    def test_fresnel_coefficients(self) -> None:
        # Normal incidence from air to glass n=1.5
        # R = ((1.5 - 1.0) / (1.5 + 1.0))^2 = (0.5 / 2.5)^2 = 0.2^2 = 0.04 (4%)
        d_in = (0.0, 0.0, 1.0)
        normal = (0.0, 0.0, -1.0)
        r, t = fresnel_coefficients(d_in, normal, 1.0, 1.5)
        self.assertAlmostEqual(r, 0.04, places=4)
        self.assertAlmostEqual(t, 0.96, places=4)

        # TIR Fresnel reflectance must be 1.0
        theta = math.radians(55.0)
        d_in = (math.sin(theta), 0.0, math.cos(theta))
        r_tir, t_tir = fresnel_coefficients(d_in, normal, 1.5, 1.0)
        self.assertEqual(r_tir, 1.0)
        self.assertEqual(t_tir, 0.0)


if __name__ == "__main__":
    unittest.main()
