"""
NovaPhysics: Physics Joints and Mechanical Constraints.
Implements DistanceJoint, RevoluteJoint (pin/hinge), and SpringJoint.
"""

import math
from typing import Optional
from .math2d import Vec2, Mat22
from .body import RigidBody


class Joint:
    """Abstract base class for rigid body constraints."""
    __slots__ = ("body_a", "body_b", "collide_connected")

    def __init__(self, body_a: RigidBody, body_b: RigidBody, collide_connected: bool = False) -> None:
        self.body_a = body_a
        self.body_b = body_b
        self.collide_connected = collide_connected

    def pre_solve(self, dt: float, inv_dt: float) -> None:
        pass

    def solve_velocity(self, dt: float, inv_dt: float) -> None:
        pass


class DistanceJoint(Joint):
    """
    Maintains a fixed Euclidean distance between two anchor points on two rigid bodies.
    Ideal for linkages, pendulums, and suspension rods.
    """
    __slots__ = (
        "local_anchor_a", "local_anchor_b", "distance",
        "r_a", "r_b", "u", "mass", "impulse", "bias_factor"
    )

    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        anchor_a: Optional[Vec2] = None,
        anchor_b: Optional[Vec2] = None,
        distance: Optional[float] = None,
        bias_factor: float = 0.2
    ) -> None:
        super().__init__(body_a, body_b)
        # If world anchors not supplied, use current centers of mass
        if anchor_a is None:
            self.local_anchor_a = Vec2(0.0, 0.0)
        else:
            self.local_anchor_a = body_a.transform.inverse_transform_point(anchor_a)

        if anchor_b is None:
            self.local_anchor_b = Vec2(0.0, 0.0)
        else:
            self.local_anchor_b = body_b.transform.inverse_transform_point(anchor_b)

        world_a = body_a.transform.transform_point(self.local_anchor_a)
        world_b = body_b.transform.transform_point(self.local_anchor_b)

        if distance is None:
            self.distance = (world_b - world_a).length()
        else:
            self.distance = float(distance)

        self.r_a = Vec2(0.0, 0.0)
        self.r_b = Vec2(0.0, 0.0)
        self.u = Vec2(0.0, 1.0)
        self.mass = 0.0
        self.impulse = 0.0
        self.bias_factor = bias_factor

    def pre_solve(self, dt: float, inv_dt: float) -> None:
        b_a = self.body_a
        b_b = self.body_b

        # Compute world-space anchor offsets from body positions
        world_a = b_a.transform.transform_point(self.local_anchor_a)
        world_b = b_b.transform.transform_point(self.local_anchor_b)
        self.r_a = world_a - b_a.position
        self.r_b = world_b - b_b.position

        # Direction vector along the joint
        diff = world_b - world_a
        d = diff.length()
        if d > 1e-6:
            self.u = diff * (1.0 / d)
        else:
            self.u = Vec2(0.0, 0.0)

        # Constraint Jacobian effective mass: 1 / (J * M^-1 * J^T)
        cr_a = self.r_a.cross(self.u)
        cr_b = self.r_b.cross(self.u)
        k = b_a.inv_mass + b_b.inv_mass + (cr_a * cr_a * b_a.inv_inertia) + (cr_b * cr_b * b_b.inv_inertia)
        self.mass = 1.0 / k if k > 1e-9 else 0.0

        # Warm starting: apply accumulated impulse along joint axis
        p = self.u * self.impulse
        b_a.apply_impulse(-p, world_a)
        b_b.apply_impulse(p, world_b)

    def solve_velocity(self, dt: float, inv_dt: float) -> None:
        b_a = self.body_a
        b_b = self.body_b

        world_a = b_a.position + self.r_a
        world_b = b_b.position + self.r_b

        # Relative velocity at anchors
        v_a = b_a.get_velocity_at_world_point(world_a)
        v_b = b_b.get_velocity_at_world_point(world_b)
        v_rel = v_b - v_a

        c_dot = self.u.dot(v_rel)

        # Position correction bias via Baumgarte stabilization
        current_distance = (world_b - world_a).length()
        c = current_distance - self.distance
        bias = self.bias_factor * inv_dt * c

        # Compute corrective impulse
        delta_impulse = -self.mass * (c_dot + bias)
        self.impulse += delta_impulse

        p = self.u * delta_impulse
        b_a.apply_impulse(-p, world_a)
        b_b.apply_impulse(p, world_b)


class RevoluteJoint(Joint):
    """
    Revolute (hinge / pin) joint that fixes two bodies together at a common pivot point,
    allowing free relative angular rotation around the z-axis.
    """
    __slots__ = (
        "local_anchor_a", "local_anchor_b",
        "r_a", "r_b", "k_matrix", "impulse", "bias_factor"
    )

    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        pivot: Vec2,
        bias_factor: float = 0.2
    ) -> None:
        super().__init__(body_a, body_b)
        self.local_anchor_a = body_a.transform.inverse_transform_point(pivot)
        self.local_anchor_b = body_b.transform.inverse_transform_point(pivot)

        self.r_a = Vec2(0.0, 0.0)
        self.r_b = Vec2(0.0, 0.0)
        self.k_matrix = Mat22.identity()
        self.impulse = Vec2(0.0, 0.0)
        self.bias_factor = bias_factor

    def pre_solve(self, dt: float, inv_dt: float) -> None:
        b_a = self.body_a
        b_b = self.body_b

        world_a = b_a.transform.transform_point(self.local_anchor_a)
        world_b = b_b.transform.transform_point(self.local_anchor_b)
        self.r_a = world_a - b_a.position
        self.r_b = world_b - b_b.position

        # Effective mass matrix K = J M^-1 J^T (2x2 matrix)
        # K_11 = m_a^-1 + m_b^-1 + r_ay^2 * I_a^-1 + r_by^2 * I_b^-1
        # K_12 = -r_ax * r_ay * I_a^-1 - r_bx * r_by * I_b^-1
        # K_22 = m_a^-1 + m_b^-1 + r_ax^2 * I_a^-1 + r_bx^2 * I_b^-1
        m_inv_sum = b_a.inv_mass + b_b.inv_mass

        k11 = m_inv_sum + (self.r_a.y * self.r_a.y * b_a.inv_inertia) + (self.r_b.y * self.r_b.y * b_b.inv_inertia)
        k12 = -(self.r_a.x * self.r_a.y * b_a.inv_inertia) - (self.r_b.x * self.r_b.y * b_b.inv_inertia)
        k21 = k12
        k22 = m_inv_sum + (self.r_a.x * self.r_a.x * b_a.inv_inertia) + (self.r_b.x * self.r_b.x * b_b.inv_inertia)

        # Invert 2x2 matrix
        det = k11 * k22 - k12 * k21
        if abs(det) > 1e-9:
            inv_det = 1.0 / det
            self.k_matrix = Mat22(
                Vec2(k22 * inv_det, -k21 * inv_det),
                Vec2(-k12 * inv_det, k11 * inv_det)
            )
        else:
            self.k_matrix = Mat22(Vec2(0.0, 0.0), Vec2(0.0, 0.0))

        # Warm starting
        world_pivot = (world_a + world_b) * 0.5
        b_a.apply_impulse(-self.impulse, world_pivot)
        b_b.apply_impulse(self.impulse, world_pivot)

    def solve_velocity(self, dt: float, inv_dt: float) -> None:
        b_a = self.body_a
        b_b = self.body_b

        world_a = b_a.position + self.r_a
        world_b = b_b.position + self.r_b

        v_a = b_a.get_velocity_at_world_point(world_a)
        v_b = b_b.get_velocity_at_world_point(world_b)
        v_rel = v_b - v_a

        # Position error between anchors
        pos_error = world_b - world_a
        bias = pos_error * (self.bias_factor * inv_dt)

        c_dot = v_rel + bias
        delta_impulse = -self.k_matrix.mul_vec(c_dot)
        self.impulse = self.impulse + delta_impulse

        world_pivot = (world_a + world_b) * 0.5
        b_a.apply_impulse(-delta_impulse, world_pivot)
        b_b.apply_impulse(delta_impulse, world_pivot)


class SpringJoint(Joint):
    """
    Harmonic distance spring with stiffness and viscous damping.
    Forces are applied directly during velocity integration.
    """
    __slots__ = (
        "local_anchor_a", "local_anchor_b", "rest_length",
        "stiffness", "damping"
    )

    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        anchor_a: Optional[Vec2] = None,
        anchor_b: Optional[Vec2] = None,
        rest_length: Optional[float] = None,
        stiffness: float = 100.0,
        damping: float = 5.0
    ) -> None:
        super().__init__(body_a, body_b)
        if anchor_a is None:
            self.local_anchor_a = Vec2(0.0, 0.0)
        else:
            self.local_anchor_a = body_a.transform.inverse_transform_point(anchor_a)

        if anchor_b is None:
            self.local_anchor_b = Vec2(0.0, 0.0)
        else:
            self.local_anchor_b = body_b.transform.inverse_transform_point(anchor_b)

        world_a = body_a.transform.transform_point(self.local_anchor_a)
        world_b = body_b.transform.transform_point(self.local_anchor_b)

        if rest_length is None:
            self.rest_length = (world_b - world_a).length()
        else:
            self.rest_length = float(rest_length)

        self.stiffness = float(stiffness)
        self.damping = float(damping)

    def pre_solve(self, dt: float, inv_dt: float) -> None:
        pass

    def solve_velocity(self, dt: float, inv_dt: float) -> None:
        b_a = self.body_a
        b_b = self.body_b

        world_a = b_a.transform.transform_point(self.local_anchor_a)
        world_b = b_b.transform.transform_point(self.local_anchor_b)

        diff = world_b - world_a
        current_len = diff.length()
        if current_len < 1e-6:
            return

        direction = diff * (1.0 / current_len)
        delta = current_len - self.rest_length

        # Relative velocity
        v_a = b_a.get_velocity_at_world_point(world_a)
        v_b = b_b.get_velocity_at_world_point(world_b)
        rel_vel = v_b - v_a
        v_spring = rel_vel.dot(direction)

        # Hooke's Law + viscous damping
        force_mag = -self.stiffness * delta - self.damping * v_spring
        force = direction * force_mag
        impulse = force * dt

        b_a.apply_impulse(-impulse, world_a)
        b_b.apply_impulse(impulse, world_b)
