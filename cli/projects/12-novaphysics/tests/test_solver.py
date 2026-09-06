"""
Unit tests for NovaPhysics Sequential Impulse Solver and World simulation.
"""

import math
import unittest
from novaphysics.math2d import Vec2
from novaphysics.shapes import Box, Circle
from novaphysics.body import RigidBody, BodyType, Material
from novaphysics.world import World


class TestSolver(unittest.TestCase):
    def test_ball_drop_on_ground(self):
        """Dynamic circle falling onto a static floor should settle and not penetrate."""
        world = World(gravity=Vec2(0.0, -9.81), sub_steps=2, velocity_iterations=10)

        # Static floor at y = 0.0 with half-height 0.5 (top at y = 0.5)
        ground = RigidBody(Box(20.0, 1.0), position=Vec2(0.0, 0.0), body_type=BodyType.STATIC)
        world.add_body(ground)

        # Dynamic ball at y = 3.0, radius 0.5 (inelastic material)
        ball_mat = Material(density=1.0, restitution=0.0, dynamic_friction=0.5)
        ball = RigidBody(Circle(0.5), position=Vec2(0.0, 3.0), material=ball_mat)
        world.add_body(ball)

        # Simulate 1.0 second (60 frames at 1/60s)
        dt = 1.0 / 60.0
        for _ in range(60):
            world.step(dt)

        # Ball should have settled on top of ground (ground top = 0.5, ball center = 0.5 + 0.5 = 1.0)
        self.assertAlmostEqual(ball.position.y, 1.0, delta=0.05)
        self.assertAlmostEqual(ball.velocity.y, 0.0, delta=0.1)

    def test_elastic_collision_momentum_conservation(self):
        """Two identical elastic balls colliding head-on should swap velocities."""
        world = World(gravity=Vec2(0.0, 0.0), velocity_iterations=10)

        mat = Material(density=1.0, restitution=1.0, dynamic_friction=0.0)
        b1 = RigidBody(Circle(0.5), position=Vec2(-2.0, 0.0), material=mat)
        b2 = RigidBody(Circle(0.5), position=Vec2(2.0, 0.0), material=mat)

        b1.velocity = Vec2(3.0, 0.0)
        b2.velocity = Vec2(-3.0, 0.0)

        world.add_body(b1)
        world.add_body(b2)

        # Step until collision occurs
        dt = 1.0 / 60.0
        for _ in range(60):
            world.step(dt)

        # Total linear momentum should be conserved (~0.0)
        total_p_x = b1.mass * b1.velocity.x + b2.mass * b2.velocity.x
        self.assertAlmostEqual(total_p_x, 0.0, places=2)
        # Velocities should be reversed
        self.assertAlmostEqual(b1.velocity.x, -3.0, delta=0.2)
        self.assertAlmostEqual(b2.velocity.x, 3.0, delta=0.2)

    def test_box_stacking_stability(self):
        """Two stacked boxes under gravity should remain stably stacked."""
        world = World(gravity=Vec2(0.0, -9.81), sub_steps=4, velocity_iterations=12, warm_starting=True)

        ground = RigidBody(Box(10.0, 1.0), position=Vec2(0.0, 0.0), body_type=BodyType.STATIC)
        world.add_body(ground)

        # Box 1 resting on ground (ground top is y=0.5, box 1 is 1x1, center at y=1.0)
        box1 = RigidBody(Box(1.0, 1.0), position=Vec2(0.0, 1.0))
        # Box 2 resting on Box 1 (box 1 top is y=1.5, box 2 center at y=2.0)
        box2 = RigidBody(Box(1.0, 1.0), position=Vec2(0.0, 2.0))

        world.add_body(box1)
        world.add_body(box2)

        dt = 1.0 / 60.0
        for _ in range(120):  # 2 seconds of simulation
            world.step(dt)

        # Both boxes should remain near their equilibrium positions
        self.assertAlmostEqual(box1.position.y, 1.0, delta=0.05)
        self.assertAlmostEqual(box2.position.y, 2.0, delta=0.08)
        self.assertAlmostEqual(box1.velocity.y, 0.0, delta=0.05)
        self.assertAlmostEqual(box2.velocity.y, 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
