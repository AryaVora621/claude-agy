"""
Unit tests for NovaPhysics Verlet particle and cloth simulation.
"""

import math
import unittest
from novaphysics.math2d import Vec2
from novaphysics.cloth import VerletParticle, DistanceConstraint, ClothMesh


class TestCloth(unittest.TestCase):
    def test_verlet_particle_integration(self):
        """Free-falling Verlet particle under gravity."""
        p = VerletParticle(Vec2(0.0, 10.0), mass=1.0)
        gravity = Vec2(0.0, -9.81)

        dt = 0.1
        # Apply gravity force = m * g = (0, -9.81)
        for _ in range(5):
            p.apply_force(gravity * 1.0)
            p.integrate(dt, damping=0.0)

        # In discrete Verlet integration: y_n = y_0 - (n*(n+1)/2) * g * dt^2
        # For n=5, dt=0.1: delta = 15 * 9.81 * 0.01 = 1.4715 -> expected = 8.5285
        expected_y = 10.0 - 15 * 9.81 * 0.01
        self.assertAlmostEqual(p.position.y, expected_y, places=4)

    def test_distance_constraint_relaxation(self):
        """Two particles with distance constraint pulled apart should relax back to rest length."""
        p1 = VerletParticle(Vec2(0.0, 0.0), mass=1.0, pinned=True)
        # Displaced to distance 3.0, rest length 2.0
        p2 = VerletParticle(Vec2(3.0, 0.0), mass=1.0, pinned=False)

        c = DistanceConstraint(p1, p2, stiffness=1.0, rest_length=2.0)
        c.relax()

        # p1 is pinned, so p2 should move to (2.0, 0.0)
        self.assertAlmostEqual((p2.position - p1.position).length(), 2.0, delta=1e-4)
        self.assertAlmostEqual(p2.position.x, 2.0, delta=1e-4)

    def test_cloth_mesh_hanging(self):
        """Cloth pinned at top corners hangs stably under gravity."""
        cloth = ClothMesh(
            cols=6,
            rows=6,
            origin=Vec2(-1.5, 3.0),
            spacing=0.5,
            pin_corners=True,
            iterations=5
        )

        dt = 1.0 / 60.0
        for _ in range(60):
            cloth.step(dt)

        # Top corners should remain pinned at initial positions
        tl = cloth.particles[0][0]
        tr = cloth.particles[0][5]
        self.assertEqual(tl.position.x, -1.5)
        self.assertEqual(tl.position.y, 3.0)
        self.assertEqual(tr.position.x, 1.0)
        self.assertEqual(tr.position.y, 3.0)

        # Center particles should have sagged downwards
        center = cloth.particles[3][3]
        self.assertLess(center.position.y, 3.0 - 3 * 0.5)

    def test_cloth_sphere_collision(self):
        """Spherical obstacle pushes cloth particles outside."""
        cloth = ClothMesh(cols=5, rows=5, origin=Vec2(-1.0, 3.0), spacing=0.5, pin_top_row=True)
        sphere_center = Vec2(0.0, 2.0)
        sphere_radius = 0.8

        dt = 1.0 / 60.0
        for _ in range(30):
            cloth.step(dt)
            cloth.apply_obstacle_sphere(sphere_center, sphere_radius)

        # Confirm no non-pinned particle penetrates the sphere
        for p in cloth.get_all_particles():
            if not p.pinned:
                dist = (p.position - sphere_center).length()
                self.assertGreaterEqual(dist, sphere_radius - 0.01)

    def test_cloth_tear(self):
        """Tearing at a location severs constraints."""
        cloth = ClothMesh(cols=5, rows=5, origin=Vec2(0.0, 0.0), spacing=0.5)
        initial_active = sum(1 for c in cloth.constraints if not c.is_broken)
        torn = cloth.tear_at(Vec2(0.5, -0.5), radius=0.4)

        self.assertGreater(torn, 0)
        active_after = sum(1 for c in cloth.constraints if not c.is_broken)
        self.assertEqual(active_after, initial_active - torn)


if __name__ == "__main__":
    unittest.main()
