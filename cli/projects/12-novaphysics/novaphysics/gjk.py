"""
NovaPhysics: Gilbert-Johnson-Keerthi (GJK) Convex Collision Detection Algorithm.
Determines whether two convex shapes intersect by evolving a simplex in Minkowski difference space.
"""

from typing import Tuple, List, Optional
from .math2d import Vec2
from .shapes import Shape
from .body import RigidBody


def support_minkowski(body_a: RigidBody, body_b: RigidBody, direction: Vec2) -> Vec2:
    """Minkowski difference support point: S_A(d) - S_B(-d)."""
    p_a = body_a.shape.support(direction, body_a.transform)
    p_b = body_b.shape.support(-direction, body_b.transform)
    return p_a - p_b


def gjk_intersect(body_a: RigidBody, body_b: RigidBody) -> Tuple[bool, List[Vec2]]:
    """
    Executes 2D GJK algorithm between two convex rigid bodies.
    Returns (is_colliding, simplex).
    If colliding, simplex contains 3 points enclosing the origin (ready for EPA).
    """
    # 1. Initial search direction: vector between body positions
    initial_dir = body_b.position - body_a.position
    if initial_dir.length_sq() < 1e-9:
        initial_dir = Vec2(1.0, 0.0)

    # First simplex point
    simplex: List[Vec2] = [support_minkowski(body_a, body_b, initial_dir)]

    # Next search direction is towards the origin
    d = -simplex[0]
    if d.length_sq() < 1e-9:
        d = Vec2(1.0, 0.0)

    max_iterations = 32
    for _ in range(max_iterations):
        # New support point in direction d
        a = support_minkowski(body_a, body_b, d)

        # If support point does not cross origin along d, no collision is possible
        if a.dot(d) < 0.0:
            return False, simplex

        simplex.append(a)

        # Evolve simplex
        if len(simplex) == 2:
            # Line simplex: [b, a] where a is the newest point
            b = simplex[0]
            ab = b - a
            ao = -a

            # Perpendicular to ab pointing towards origin
            # In 2D: normal can be (-ab.y, ab.x) or (ab.y, -ab.x)
            n1 = Vec2(-ab.y, ab.x)
            if n1.dot(ao) > 0.0:
                d = n1
            else:
                d = -n1

        elif len(simplex) == 3:
            # Triangle simplex: [c, b, a] where a is newest point
            c, b, a = simplex[0], simplex[1], simplex[2]
            ab = b - a
            ac = c - a
            ao = -a

            # Outward normal of edge ab
            ab_norm = Vec2(-ab.y, ab.x)
            if ab_norm.dot(c - a) > 0.0:
                ab_norm = -ab_norm

            # Outward normal of edge ac
            ac_norm = Vec2(-ac.y, ac.x)
            if ac_norm.dot(b - a) > 0.0:
                ac_norm = -ac_norm

            # Check if origin is outside edge ab
            if ab_norm.dot(ao) > 0.0:
                # Remove c
                simplex = [b, a]
                d = ab_norm
            # Check if origin is outside edge ac
            elif ac_norm.dot(ao) > 0.0:
                # Remove b
                simplex = [c, a]
                d = ac_norm
            else:
                # Origin is inside both edges -> origin is enclosed inside triangle!
                return True, simplex

    return False, simplex
