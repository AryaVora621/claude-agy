"""
PhotonPBR: Unit Tests for Surface Area Heuristic (SAH) BVH Acceleration Tree.
Verifies equivalence between O(log N) BVH traversal and O(N) naive linear scanning.
"""

import unittest
import random
from photon.vec3 import Vec3, Ray, Color
from photon.geometry import Sphere, Triangle, HittableList
from photon.bvh import BVHNode


class MockMat:
    def __init__(self, idx: int):
        self.idx = idx

    def scatter(self, ray_in, rec):
        return False, Color(0, 0, 0), Ray(Vec3(), Vec3(1, 0, 0))

    def emitted(self, u, v, p):
        return Color(0, 0, 0)


class TestBVH(unittest.TestCase):
    def test_single_object_bvh(self):
        mat = MockMat(1)
        sphere = Sphere(Vec3(0, 0, 5), 1.0, mat)
        bvh = BVHNode([sphere])

        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        rec = bvh.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 4.0)

    def test_two_objects_bvh(self):
        mat1 = MockMat(1)
        mat2 = MockMat(2)
        s1 = Sphere(Vec3(0, 0, 4), 1.0, mat1)
        s2 = Sphere(Vec3(0, 0, 8), 1.0, mat2)
        bvh = BVHNode([s1, s2])

        ray = Ray(Vec3(0, 0, 0), Vec3(0, 0, 1))
        rec = bvh.hit(ray, 0.001, 100.0)
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.t, 3.0)
        self.assertEqual(rec.material.idx, 1)

    def test_equivalence_with_linear_hittable_list(self):
        # Create a scene with 16 distributed spheres and triangles
        random.seed(42)
        primitives = []
        for i in range(10):
            center = Vec3(
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(5, 30)
            )
            primitives.append(Sphere(center, random.uniform(0.5, 2.0), MockMat(i)))

        for i in range(6):
            base = Vec3(
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(5, 30)
            )
            v0 = base
            v1 = base + Vec3(random.uniform(1, 3), 0, 0)
            v2 = base + Vec3(0, random.uniform(1, 3), 0)
            primitives.append(Triangle(v0, v1, v2, MockMat(10 + i)))

        naive_world = HittableList(list(primitives))
        bvh_world = BVHNode(list(primitives))

        # Cast 50 rays into the scene from various directions
        for _ in range(50):
            target = Vec3(
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(5, 30)
            )
            orig = Vec3(0, 0, 0)
            ray = Ray(orig, (target - orig).normalized())

            naive_hit = naive_world.hit(ray, 0.001, 1000.0)
            bvh_hit = bvh_world.hit(ray, 0.001, 1000.0)

            if naive_hit is None:
                self.assertIsNone(bvh_hit)
            else:
                self.assertIsNotNone(bvh_hit)
                self.assertAlmostEqual(naive_hit.t, bvh_hit.t, places=4)
                self.assertAlmostEqual(naive_hit.point.x, bvh_hit.point.x, places=4)
                self.assertAlmostEqual(naive_hit.point.y, bvh_hit.point.y, places=4)
                self.assertAlmostEqual(naive_hit.point.z, bvh_hit.point.z, places=4)
                self.assertEqual(naive_hit.material.idx, bvh_hit.material.idx)


if __name__ == "__main__":
    unittest.main()
