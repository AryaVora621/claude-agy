"""
NovaPhysics: Expanding Polytope Algorithm (EPA).
Determines exact penetration depth and contact normal for intersecting convex shapes.
"""

from typing import List, Tuple
from .math2d import Vec2
from .body import RigidBody
from .gjk import support_minkowski


def epa_penetration(
    body_a: RigidBody,
    body_b: RigidBody,
    simplex: List[Vec2],
    tolerance: float = 1e-4,
    max_iterations: int = 32
) -> Tuple[float, Vec2]:
    """
    Computes penetration depth and contact normal pointing from body A to body B.
    Takes 3-vertex simplex enclosing origin from GJK.
    """
    if len(simplex) < 3:
        # Fallback if degenerate simplex
        diff = body_b.position - body_a.position
        dist = diff.length()
        normal = diff.normalized() if dist > 1e-6 else Vec2(0.0, 1.0)
        return max(0.01, dist), normal

    polytope = list(simplex)

    # Ensure counter-clockwise winding
    e1 = polytope[1] - polytope[0]
    e2 = polytope[2] - polytope[0]
    if e1.cross(e2) < 0.0:
        polytope.reverse()

    for _ in range(max_iterations):
        # 1. Find edge closest to origin
        min_dist = float("inf")
        min_index = 0
        min_normal = Vec2(0.0, 1.0)

        n = len(polytope)
        for i in range(n):
            j = (i + 1) % n
            edge = polytope[j] - polytope[i]

            # Outward normal for CCW winding: (edge.y, -edge.x)
            normal = Vec2(edge.y, -edge.x).normalized()
            dist = normal.dot(polytope[i])

            if dist < 0.0:
                dist = -dist
                normal = -normal

            if dist < min_dist:
                min_dist = dist
                min_index = j
                min_normal = normal

        # 2. Get support point along closest edge normal
        support = support_minkowski(body_a, body_b, min_normal)
        dist = min_normal.dot(support)

        # 3. Check convergence
        if dist - min_dist < tolerance:
            # Polytope has reached boundary of Minkowski difference
            # min_normal points outwards from A towards B
            return min_dist, min_normal

        # 4. Insert new support point, splitting the closest edge
        polytope.insert(min_index, support)

    return min_dist, min_normal
