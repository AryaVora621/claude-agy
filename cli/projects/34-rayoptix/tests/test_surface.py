"""Unit tests for Optical Surfaces: Spherical, Conic, and Even Aspheric Geometries."""

import math
import unittest

from rayoptix.ray import Ray
from rayoptix.surface import OpticalSurface


class TestOpticalSurface(unittest.TestCase):
    """Test suite for optical surface intersection, normal vectors, and aspheric sag."""

    def test_plano_surface(self) -> None:
        surf = OpticalSurface(name="PlanoStop", radius_of_curvature=0.0, semi_diameter=10.0)
        self.assertEqual(surf.curvature, 0.0)
        self.assertEqual(surf.sag(5.0, 5.0), 0.0)

        # Ray towards plano surface at z=0 from z=-10
        ray = Ray(origin=(2.0, 3.0, -10.0), direction=(0.0, 0.0, 1.0))
        hit = surf.intersect(ray)
        self.assertIsNotNone(hit)
        t, pt, normal = hit
        self.assertAlmostEqual(t, 10.0)
        self.assertAlmostEqual(pt[0], 2.0)
        self.assertAlmostEqual(pt[1], 3.0)
        self.assertAlmostEqual(pt[2], 0.0)
        self.assertEqual(normal, (0.0, 0.0, -1.0))

    def test_spherical_sag_analytical(self) -> None:
        R = 100.0
        surf = OpticalSurface(radius_of_curvature=R, conic_constant=0.0)
        # Analytical sag: z = R - sqrt(R^2 - r^2)
        r = 10.0
        z_expected = R - math.sqrt(R * R - r * r)
        z_calc = surf.sag(r, 0.0)
        self.assertAlmostEqual(z_calc, z_expected, places=7)

    def test_conic_paraboloid_sag(self) -> None:
        # For paraboloid k = -1: z = c r^2 / (1 + 1) = c r^2 / 2 = r^2 / (2R)
        R = 50.0
        surf = OpticalSurface(radius_of_curvature=R, conic_constant=-1.0)
        r = 8.0
        z_expected = (r * r) / (2.0 * R)
        self.assertAlmostEqual(surf.sag(r, 0.0), z_expected, places=7)

    def test_spherical_normal_vector(self) -> None:
        R = 50.0
        surf = OpticalSurface(radius_of_curvature=R, conic_constant=0.0)
        # Normal at vertex (0, 0) should be strictly (0, 0, -1)
        norm_v, _ = surf.normal_and_sag(0.0, 0.0)
        self.assertAlmostEqual(norm_v[0], 0.0)
        self.assertAlmostEqual(norm_v[1], 0.0)
        self.assertAlmostEqual(norm_v[2], -1.0)

        # Off-axis point: normal should point toward incident medium (-z)
        x = 10.0
        norm_off, z = surf.normal_and_sag(x, 0.0)
        # For convex surface z(x) > 0, tangent is (1, 0, dz/dx)
        # Normal with negative z is (dz/dx, 0, -1) normalized
        dz_dx = x / math.sqrt(R * R - x * x)
        mag = math.sqrt(1.0 + dz_dx * dz_dx)
        expected_norm = (dz_dx / mag, 0.0, -1.0 / mag)
        self.assertAlmostEqual(norm_off[0], expected_norm[0], places=5)
        self.assertAlmostEqual(norm_off[2], expected_norm[2], places=5)

    def test_spherical_ray_intersection(self) -> None:
        R = 25.0
        surf = OpticalSurface(radius_of_curvature=R)
        # Ray launched on-axis from z = -10
        ray = Ray(origin=(0.0, 0.0, -10.0), direction=(0.0, 0.0, 1.0))
        hit = surf.intersect(ray)
        self.assertIsNotNone(hit)
        t, pt, normal = hit
        self.assertAlmostEqual(t, 10.0)
        self.assertAlmostEqual(pt[2], 0.0)

        # Off-axis ray parallel to z-axis at height y = 5
        ray_off = Ray(origin=(0.0, 5.0, -10.0), direction=(0.0, 0.0, 1.0))
        hit_off = surf.intersect(ray_off)
        self.assertIsNotNone(hit_off)
        t_off, pt_off, _ = hit_off
        z_expected = R - math.sqrt(R * R - 25.0)
        self.assertAlmostEqual(pt_off[2], z_expected, places=6)
        self.assertAlmostEqual(t_off, 10.0 + z_expected, places=6)

    def test_aspheric_surface_intersection(self) -> None:
        # Spherical base with 4th-order polynomial deformation: z += 1e-5 * r^4
        R = 40.0
        alpha2 = 1e-5
        surf = OpticalSurface(radius_of_curvature=R, aspheric_coefficients=[alpha2])

        r = 6.0
        z_expected = (R - math.sqrt(R * R - r * r)) + alpha2 * (r ** 4)
        self.assertAlmostEqual(surf.sag(r, 0.0), z_expected, places=7)

        # Trace ray at height y = 6
        ray = Ray(origin=(0.0, r, -5.0), direction=(0.0, 0.0, 1.0))
        hit = surf.intersect(ray)
        self.assertIsNotNone(hit)
        t, pt, _ = hit
        self.assertAlmostEqual(pt[2], z_expected, places=6)
        self.assertAlmostEqual(t, 5.0 + z_expected, places=6)

    def test_vignetting_aperture(self) -> None:
        surf = OpticalSurface(radius_of_curvature=0.0, semi_diameter=5.0)
        ray_inside = Ray(origin=(0.0, 3.0, -2.0), direction=(0.0, 0.0, 1.0))
        ray_outside = Ray(origin=(0.0, 7.0, -2.0), direction=(0.0, 0.0, 1.0))

        res_in = surf.trace_ray(ray_inside, surface_index=1, n_incident=1.0, n_transmitted=1.5)
        res_out = surf.trace_ray(ray_outside, surface_index=1, n_incident=1.0, n_transmitted=1.5)

        self.assertFalse(res_in.is_vignetted)
        self.assertIsNotNone(res_in.outgoing_ray)

        self.assertTrue(res_out.is_vignetted)
        self.assertIsNone(res_out.outgoing_ray)


if __name__ == "__main__":
    unittest.main()
