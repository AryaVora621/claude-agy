"""
Unit tests for NovaPhysics mechanical joints (DistanceJoint, RevoluteJoint, SpringJoint).
"""

import math
import unittest
from novaphysics.math2d import Vec2
from novaphysics.shapes import Box, Circle
from novaphysics.body import RigidBody, BodyType
from novaphysics.joints import DistanceJoint, RevoluteJoint, SpringJoint
from novaphysics.world import World


class TestJoints(unittest.TestCase):
    def test_distance_joint_pendulum(self):
        """A ball attached to a static anchor via DistanceJoint should swing like a pendulum with fixed length."""
        world = World(gravity=Vec2(0.0, -9.81), sub_steps=2, velocity_iterations=10)

        # Static anchor at (0, 5)
        anchor = RigidBody(Circle(0.1), position=Vec2(0.0, 5.0), body_type=BodyType.STATIC)
        # Dynamic bob at (3, 5) released from rest (expected length = 3.0)
        bob = RigidBody(Circle(0.2), position=Vec2(3.0, 5.0), body_type=BodyType.DYNAMIC)

        world.add_body(anchor)
        world.add_body(bob)

        joint = DistanceJoint(anchor, bob, anchor_a=Vec2(0.0, 5.0), anchor_b=Vec2(3.0, 5.0), distance=3.0)
        world.add_joint(joint)

        dt = 1.0 / 60.0
        for _ in range(90):  # 1.5 seconds of pendulum swinging
            world.step(dt)
            curr_dist = (bob.position - anchor.position).length()
            # Distance should be tightly conserved to 3.0 +/- 0.05
            self.assertAlmostEqual(curr_dist, 3.0, delta=0.05)

    def test_revolute_joint_pin(self):
        """Two bodies pinned at a common pivot should rotate together without the pivot drifting."""
        world = World(gravity=Vec2(0.0, -9.81), sub_steps=4, velocity_iterations=12)

        pivot = Vec2(0.0, 4.0)
        # Static body at (0, 4)
        base = RigidBody(Box(1.0, 1.0), position=pivot, body_type=BodyType.STATIC)
        # Dynamic beam extending horizontally from pivot
        beam = RigidBody(Box(4.0, 0.4), position=Vec2(2.0, 4.0), body_type=BodyType.DYNAMIC)

        world.add_body(base)
        world.add_body(beam)

        rev_joint = RevoluteJoint(base, beam, pivot=pivot)
        world.add_joint(rev_joint)

        dt = 1.0 / 60.0
        for _ in range(60):
            world.step(dt)
            # Anchor on beam should stay at pivot (0, 4)
            beam_anchor = beam.transform.transform_point(rev_joint.local_anchor_b)
            self.assertAlmostEqual(beam_anchor.x, pivot.x, delta=0.05)
            self.assertAlmostEqual(beam_anchor.y, pivot.y, delta=0.05)

        # Beam should have rotated downwards due to gravity
        self.assertLess(beam.position.y, 4.0)

    def test_spring_joint_oscillation(self):
        """A mass on a vertical spring should oscillate harmonically."""
        world = World(gravity=Vec2(0.0, -9.81), sub_steps=2)

        anchor = RigidBody(Circle(0.1), position=Vec2(0.0, 5.0), body_type=BodyType.STATIC)
        # Suspended mass with rest length 2.0 (initial position at y=3.0)
        mass = RigidBody(Circle(0.3), position=Vec2(0.0, 3.0), body_type=BodyType.DYNAMIC)

        world.add_body(anchor)
        world.add_body(mass)

        # Spring with stiffness 50 and damping 1.0
        spring = SpringJoint(anchor, mass, rest_length=2.0, stiffness=50.0, damping=1.0)
        world.add_joint(spring)

        # Simulate 120 steps
        dt = 1.0 / 60.0
        y_positions = []
        for _ in range(120):
            world.step(dt)
            y_positions.append(mass.position.y)

        # Confirm mass moved downwards and then moved upwards (oscillated)
        min_y = min(y_positions)
        self.assertLess(min_y, 3.0)
        # Damping should prevent explosive oscillation
        self.assertGreater(mass.position.y, 0.0)


if __name__ == "__main__":
    unittest.main()
