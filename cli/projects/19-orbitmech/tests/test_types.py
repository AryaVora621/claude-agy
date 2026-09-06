"""
Unit tests for OrbitMech fundamental types and 3D vector mathematics.
"""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    OrbitType,
    EARTH,
    SUN,
)


class TestVector3(unittest.TestCase):
    def test_vector_arithmetic(self):
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)

        # Addition and subtraction
        v_add = v1 + v2
        self.assertAlmostEqual(v_add.x, 5.0)
        self.assertAlmostEqual(v_add.y, 7.0)
        self.assertAlmostEqual(v_add.z, 9.0)

        v_sub = v2 - v1
        self.assertAlmostEqual(v_sub.x, 3.0)
        self.assertAlmostEqual(v_sub.y, 3.0)
        self.assertAlmostEqual(v_sub.z, 3.0)

        # Scaling and division
        v_scale = v1 * 2.5
        self.assertAlmostEqual(v_scale.x, 2.5)
        self.assertAlmostEqual(v_scale.y, 5.0)
        self.assertAlmostEqual(v_scale.z, 7.5)

        v_div = v2 / 2.0
        self.assertAlmostEqual(v_div.x, 2.0)
        self.assertAlmostEqual(v_div.y, 2.5)
        self.assertAlmostEqual(v_div.z, 3.0)

        # Negation
        v_neg = -v1
        self.assertAlmostEqual(v_neg.x, -1.0)
        self.assertAlmostEqual(v_neg.y, -2.0)
        self.assertAlmostEqual(v_neg.z, -3.0)

    def test_dot_and_cross_product(self):
        # Basis unit vectors
        i_hat = Vector3(1.0, 0.0, 0.0)
        j_hat = Vector3(0.0, 1.0, 0.0)
        k_hat = Vector3(0.0, 0.0, 1.0)

        self.assertAlmostEqual(i_hat.dot(j_hat), 0.0)
        self.assertAlmostEqual(i_hat.dot(i_hat), 1.0)

        # Cross product: i x j = k
        cross_ij = i_hat.cross(j_hat)
        self.assertAlmostEqual(cross_ij.x, 0.0)
        self.assertAlmostEqual(cross_ij.y, 0.0)
        self.assertAlmostEqual(cross_ij.z, 1.0)

        # j x k = i
        cross_jk = j_hat.cross(k_hat)
        self.assertAlmostEqual(cross_jk.x, 1.0)
        self.assertAlmostEqual(cross_jk.y, 0.0)
        self.assertAlmostEqual(cross_jk.z, 0.0)

    def test_norm_and_normalization(self):
        v = Vector3(3.0, 4.0, 12.0)
        self.assertAlmostEqual(v.norm_squared(), 169.0)
        self.assertAlmostEqual(v.norm(), 13.0)

        unit_v = v.normalized()
        self.assertAlmostEqual(unit_v.norm(), 1.0)
        self.assertAlmostEqual(unit_v.x, 3.0 / 13.0)

        # Zero vector handling
        v_zero = Vector3.zero()
        self.assertEqual(v_zero.normalized(), Vector3(0.0, 0.0, 0.0))

    def test_angles_and_distance(self):
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 10.0, 0.0)
        self.assertAlmostEqual(v1.angle_to(v2), math.pi / 2.0)
        self.assertAlmostEqual(v1.distance_to(v2), math.sqrt(101.0))


class TestOrbitalElements(unittest.TestCase):
    def test_leo_orbital_properties(self):
        # 400 km circular Low Earth Orbit
        r_leo = EARTH.radius + 400.0
        coe = ClassicalOrbitalElements(
            a=r_leo,
            e=0.0001,
            i=math.radians(51.6),  # ISS inclination
            raan=0.0,
            arg_peri=0.0,
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        self.assertEqual(coe.orbit_type, OrbitType.CIRCULAR)
        self.assertAlmostEqual(coe.periapsis_radius, r_leo * (1 - 0.0001), places=3)
        self.assertAlmostEqual(coe.apoapsis_radius, r_leo * (1 + 0.0001), places=3)
        # ISS period ~ 92.5 minutes
        self.assertAlmostEqual(coe.period / 60.0, 92.56, delta=0.5)

    def test_geostationary_orbit_period(self):
        # GEO radius ~ 42,164 km
        r_geo = 42164.14
        coe = ClassicalOrbitalElements(
            a=r_geo,
            e=0.0,
            i=0.0,
            raan=0.0,
            arg_peri=0.0,
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        # Period should be one sidereal day ~ 86164 seconds
        self.assertAlmostEqual(coe.period, 86164.1, delta=5.0)


if __name__ == "__main__":
    unittest.main()
