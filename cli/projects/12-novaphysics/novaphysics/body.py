"""
NovaPhysics: Rigid Body Dynamics and Material Properties.
Encapsulates state, mass properties, forces, impulses, and coordinate conversions.
"""

import math
from enum import Enum, auto
from typing import Optional, Any
from .math2d import Vec2, Transform2D
from .shapes import Shape, AABB


class BodyType(Enum):
    STATIC = auto()     # Zero mass, non-movable obstacle (walls, floors)
    KINEMATIC = auto()  # Infinite mass, velocity-driven, unaffected by forces
    DYNAMIC = auto()    # Standard movable body simulated with forces & impulses


class Material:
    """Surface physical characteristics determining collision response."""
    __slots__ = ("restitution", "static_friction", "dynamic_friction", "density")

    def __init__(
        self,
        restitution: float = 0.2,
        static_friction: float = 0.5,
        dynamic_friction: float = 0.3,
        density: float = 1.0
    ) -> None:
        self.restitution = max(0.0, min(1.0, float(restitution)))
        self.static_friction = max(0.0, float(static_friction))
        self.dynamic_friction = max(0.0, float(dynamic_friction))
        self.density = max(1e-4, float(density))


class RigidBody:
    """Rigid body physics entity with position, velocity, mass, and shape."""
    __slots__ = (
        "id", "name", "body_type", "shape", "material",
        "position", "velocity", "force",
        "angle", "angular_velocity", "torque",
        "mass", "inv_mass", "inertia", "inv_inertia",
        "transform", "aabb", "broadphase_id",
        "linear_damping", "angular_damping", "user_data"
    )

    _id_counter = 0

    def __init__(
        self,
        shape: Shape,
        position: Vec2 = Vec2(0.0, 0.0),
        angle: float = 0.0,
        body_type: BodyType = BodyType.DYNAMIC,
        material: Optional[Material] = None,
        name: Optional[str] = None
    ) -> None:
        RigidBody._id_counter += 1
        self.id = RigidBody._id_counter
        self.name = name or f"body_{self.id}"
        self.body_type = body_type
        self.shape = shape
        self.material = material or Material()

        # Positional state
        self.position = position
        self.velocity = Vec2(0.0, 0.0)
        self.force = Vec2(0.0, 0.0)

        # Rotational state
        self.angle = float(angle)
        self.angular_velocity = 0.0
        self.torque = 0.0

        # Transform and AABB
        self.transform = Transform2D(self.position, self.angle)
        self.aabb = self.shape.compute_aabb(self.transform)
        self.broadphase_id: Optional[int] = None

        # Damping factors
        self.linear_damping = 0.001
        self.angular_damping = 0.01
        self.user_data: Any = None

        # Mass properties
        self.mass = 0.0
        self.inv_mass = 0.0
        self.inertia = 0.0
        self.inv_inertia = 0.0
        self._calculate_mass()

    def __repr__(self) -> str:
        return f"RigidBody({self.name}, {self.body_type.name}, pos={self.position})"

    def _calculate_mass(self) -> None:
        if self.body_type == BodyType.STATIC or self.body_type == BodyType.KINEMATIC:
            self.mass = 0.0
            self.inv_mass = 0.0
            self.inertia = 0.0
            self.inv_inertia = 0.0
        else:
            m, i, center = self.shape.compute_mass(self.material.density)
            self.mass = m
            self.inv_mass = 1.0 / m if m > 0 else 0.0
            self.inertia = i
            self.inv_inertia = 1.0 / i if i > 0 else 0.0

    def set_mass(self, mass: float) -> None:
        """Explicitly override body mass."""
        if mass <= 0.0:
            self.mass = 0.0
            self.inv_mass = 0.0
        else:
            self.mass = float(mass)
            self.inv_mass = 1.0 / mass

    def update_transform(self) -> None:
        """Sync coordinate transform with position and angle."""
        self.transform.position = self.position
        self.transform.set_angle(self.angle)
        self.aabb = self.shape.compute_aabb(self.transform)

    def get_aabb(self) -> AABB:
        """Returns the current world-space axis-aligned bounding box."""
        return self.aabb

    def apply_force(self, force: Vec2, world_point: Optional[Vec2] = None) -> None:
        """Apply force in Newtons. If world_point is specified, also generates torque."""
        if self.body_type != BodyType.DYNAMIC:
            return
        self.force = self.force + force
        if world_point is not None:
            r = world_point - self.position
            self.torque += r.cross(force)

    def apply_torque(self, torque: float) -> None:
        if self.body_type != BodyType.DYNAMIC:
            return
        self.torque += torque

    def apply_impulse(self, impulse: Vec2, world_point: Optional[Vec2] = None) -> None:
        """Apply instantaneous linear impulse (N*s), updating velocities directly."""
        if self.body_type != BodyType.DYNAMIC:
            return
        self.velocity = self.velocity + impulse * self.inv_mass
        if world_point is not None:
            r = world_point - self.position
            self.angular_velocity += self.inv_inertia * r.cross(impulse)

    def apply_angular_impulse(self, angular_impulse: float) -> None:
        if self.body_type != BodyType.DYNAMIC:
            return
        self.angular_velocity += self.inv_inertia * angular_impulse

    def get_world_point(self, local_pt: Vec2) -> Vec2:
        return self.transform.transform_point(local_pt)

    def get_local_point(self, world_pt: Vec2) -> Vec2:
        return self.transform.inverse_transform_point(world_pt)

    def get_world_vector(self, local_vec: Vec2) -> Vec2:
        return self.transform.transform_vector(local_vec)

    def get_velocity_at_world_point(self, world_pt: Vec2) -> Vec2:
        """Total velocity at world point: v + omega x r."""
        r = world_pt - self.position
        # omega x r in 2D = Vec2(-omega * r.y, omega * r.x)
        tangent_v = Vec2(-self.angular_velocity * r.y, self.angular_velocity * r.x)
        return self.velocity + tangent_v

    def integrate_forces(self, gravity: Vec2, dt: float) -> None:
        """Update linear and angular velocities from forces, torques, and gravity."""
        if self.body_type != BodyType.DYNAMIC:
            return

        # Linear acceleration: F/m + g
        accel = self.force * self.inv_mass + gravity
        self.velocity = self.velocity + accel * dt

        # Angular acceleration: tau / I
        alpha = self.torque * self.inv_inertia
        self.angular_velocity += alpha * dt

        # Apply velocity damping
        self.velocity = self.velocity * (1.0 / (1.0 + self.linear_damping * dt))
        self.angular_velocity *= (1.0 / (1.0 + self.angular_damping * dt))

    def integrate_velocities(self, dt: float) -> None:
        """Update position and angle using current velocities."""
        if self.body_type == BodyType.STATIC:
            return

        self.position = self.position + self.velocity * dt
        self.angle += self.angular_velocity * dt
        self.update_transform()

    def clear_forces(self) -> None:
        """Reset external applied forces and torques."""
        self.force = Vec2(0.0, 0.0)
        self.torque = 0.0
