"""
NovaPhysics: High-Performance 2D Rigid Body Dynamics, GJK/EPA Collision Engine & Constraint Solver.
Zero external dependencies. Pure Python standard library.
"""

from .math2d import Vec2, Mat22, Transform2D, scalar_cross_vec
from .shapes import Shape, ShapeType, AABB, Circle, Polygon, Box
from .body import RigidBody, BodyType, Material
from .broadphase import DynamicAABBTree
from .gjk import gjk_intersect, support_minkowski
from .epa import epa_penetration
from .contact import (
    ContactPoint,
    ContactManifold,
    find_collision,
    collide_circle_circle,
    collide_circle_polygon,
    collide_convex_gjk_epa
)
from .joints import Joint, DistanceJoint, RevoluteJoint, SpringJoint
from .solver import ContactConstraint, SequentialImpulseSolver
from .world import World
from .cloth import VerletParticle, DistanceConstraint, ClothMesh
from .visualizer import BrailleCanvas, PhysicsRenderer

__version__ = "1.0.0"

__all__ = [
    "Vec2",
    "Mat22",
    "Transform2D",
    "scalar_cross_vec",
    "Shape",
    "ShapeType",
    "AABB",
    "Circle",
    "Polygon",
    "Box",
    "RigidBody",
    "BodyType",
    "Material",
    "DynamicAABBTree",
    "gjk_intersect",
    "support_minkowski",
    "epa_penetration",
    "ContactPoint",
    "ContactManifold",
    "find_collision",
    "collide_circle_circle",
    "collide_circle_polygon",
    "collide_convex_gjk_epa",
    "Joint",
    "DistanceJoint",
    "RevoluteJoint",
    "SpringJoint",
    "ContactConstraint",
    "SequentialImpulseSolver",
    "World",
    "VerletParticle",
    "DistanceConstraint",
    "ClothMesh",
    "BrailleCanvas",
    "PhysicsRenderer",
]
