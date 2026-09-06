"""
PhotonPBR: Unit Tests for Geometric Primitives and Ray Intersection Algorithms.
Tests AABB Slab Method, Quadric Sphere Intersection, and Möller-Trumbore Triangle Algorithm.
"""

import unittest
import math
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import AABB, Sphere, Triangle, HittableList, HitRecord


class MockMaterial:
    def __init__(self, name: str = "mock"):
        self.name = name

    def scatter(self, ray_in, rec):
        return False, Color(0, 0, 0), Ray(Vec3(), Vec3(1, 0, 0))

    def emitted(self, u, v, p):
        return Color(0, 0, 0)


class TestAABB(unittest.TestCase):
    def test_construction_and_ordering(self):
        # Even if min and max inputs are inverted, AABB correctly orders them
        box = AABB(Vec3(5, -2, 10), Vec3(-5, 8, 3))
        self.assertEqual(box.min_pt.x, -5.0)
        self.assertEqual(box.min_pt.y, -2.0)
        self.assertEqual(box.min_pt.z, 3.0)
        self.assertEqual(box.max_pt.x, 5.0)
        self.assertEqual(box.max_pt.y, 8.0)
        self.assertEqual(box.max_pt.z, 10.0)

    def test_surface_area(self):
        box = AABB(Vec3(0, 0, 0), Vec3(2, 3, 4))
        # dx=2, dy=3, dz=4 -> 2 * (2*3 + 3*4 + 4*2) = 2 * (6 + 12 + 8) = 52
        self.assertAlmostEqual(box.surface_area(), 52.0)

    def test_slab_hit_and_miss(self):
        box = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))

        # Direct hit through center along Z axis
        ray_hit = Ray(Vec3(0, 0, -5), Vec3(0, 0, 1))
        self.assertTrue(box.hit(ray_hit, 0.001, 100.0))

        # Ray pointing away from box
        ray_away = Ray(Vec3(0, 0, -5), Vec3(0, 0, -1))
        self.assertFalse(box.hit(ray_away, 0.001, 100.0))

        # Ray parallel and offset outside box
        ray_miss = Ray(Vec3(2, 0, -5), Vec3(0, 0, 1))
        self.assertFalse(box.hit(ray_miss, 0.001, 100.0))

        # Interval bounded before box
        self.assertFalse(box.hit(ray_hit, 0.001, 2.0))

    def test_surrounding_box(self):
        b1 = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        b2 = AABB(Vec3(-2, 0.5, 0.5), Vec3(0.5, 3, 2))
        surr = AABB.surrounding_box(b1, b2)
        self.assertEqual(surr.min_pt.x, -2.0)
        self.assertEqual(surr.min_pt.y, 0.0)
        self.assertEqual(surr.min_pt.z, 0.0)
        self.assertEqual(surr.max_pt.x, 1.0)
        self.assertEqual(surr.max_pt.y, 3.0)
        self.assertEqual(surr.max_pt.z, 2.0)


class TestSphere(unittest.TestCase):
    def test_sphere_hit_and_normal(self):
        mat = MockMaterial("sphere_mat")
        sphere = Sphere(Vec3(0, 0, 5), 1.0, mat)

        # Center direct hit
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        rec = sphere.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 4.0)
        self.assertAlmostEqual(rec.point.z, 4.0)
        self.assertTrue(rec.front_face)
        # Normal points outward back toward ray origin (0, 0, -1)
        self.assertAlmostEqual(rec.normal.x, 0.0)
        self.assertAlmostEqual(rec.normal.y, 0.0)
        self.assertAlmostEqual(rec.normal.z, -1.0)
        self.assertEqual(rec.material, mat)

    def test_sphere_inside_hit(self):
        mat = MockMaterial()
        sphere = Sphere(Vec3(0, 0, 0), 2.0, mat)

        # Ray originating inside sphere
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 1, 0))
        rec = sphere.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 2.0)
        # From inside, front_face is False, normal points inward
        self.assertFalse(rec.front_face)
        self.assertAlmostEqual(rec.normal.y, -1.0)

    def test_sphere_miss(self):
        mat = MockMaterial()
        sphere = Sphere(Vec3(0, 5, 0), 1.0, mat)

        # Ray aimed along Z axis, missing sphere above at Y=5
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        rec = sphere.hit(ray, 0.001, 100.0)
        self.assertIsNone(rec)


class TestTriangle(unittest.TestCase):
    def test_moller_trumbore_hit(self):
        mat = MockMaterial("tri_mat")
        # Triangle in XY plane at Z=5, winding counter-clockwise toward -Z
        v0 = Vec3(-1, -1, 5)
        v1 = Vec3(0, 1, 5)
        v2 = Vec3(1, -1, 5)
        tri = Triangle(v0, v1, v2, mat)

        # Ray through centroid of triangle
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        rec = tri.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 5.0)
        self.assertAlmostEqual(rec.point.x, 0.0)
        self.assertAlmostEqual(rec.point.y, 0.0)
        self.assertAlmostEqual(rec.point.z, 5.0)
        self.assertTrue(rec.front_face)
        # Normal faces -Z
        self.assertAlmostEqual(rec.normal.z, -1.0)
        self.assertGreaterEqual(rec.u, 0.0)
        self.assertGreaterEqual(rec.v, 0.0)
        self.assertLessEqual(rec.u + rec.v, 1.0)

    def test_triangle_miss_outside_edges(self):
        mat = MockMaterial()
        v0 = Vec3(0, 0, 5)
        v1 = Vec3(2, 0, 5)
        v2 = Vec3(0, 2, 5)
        tri = Triangle(v0, v1, v2, mat)

        # Ray hitting (2, 2, 5) which is outside hypotenuse (u+v > 1)
        ray_miss = Ray(Vec3(2, 2, 0), Vec3(0, 0, 1))
        self.assertIsNone(tri.hit(ray_miss, 0.001, 100.0))

        # Ray hitting (-0.5, 0.5, 5) which is outside u < 0
        ray_miss2 = Ray(Vec3(-0.5, 0.5, 0), Vec3(0, 0, 1))
        self.assertIsNone(tri.hit(ray_miss2, 0.001, 100.0))

    def test_triangle_parallel_ray(self):
        mat = MockMaterial()
        v0 = Vec3(0, 0, 5)
        v1 = Vec3(1, 0, 5)
        v2 = Vec3(0, 1, 5)
        tri = Triangle(v0, v1, v2, mat)

        # Ray parallel to XY plane (direction (1, 0, 0))
        ray_par = Ray(Vec3(0, 0, 5), Vec3(1, 0, 0))
        self.assertIsNone(tri.hit(ray_par, 0.001, 100.0))


class TestHittableList(unittest.TestCase):
    def test_closest_hit_selection(self):
        mat1 = MockMaterial("near")
        mat2 = MockMaterial("far")
        s_near = Sphere(Vec3(0, 0, 3), 1.0, mat1)
        s_far = Sphere(Vec3(0, 0, 7), 1.0, mat2)

        world = HittableList([s_far, s_near])
        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))

        rec = world.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 2.0)
        self.assertEqual(rec.material.name, "near")


if __name__ == "__main__":
    unittest.main()
