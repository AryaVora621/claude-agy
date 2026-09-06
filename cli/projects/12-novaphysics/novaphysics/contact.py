"""
NovaPhysics: Collision Contact Manifolds and Feature Clipping.
Extracts contact points, penetration depths, and combined material properties.
"""

import math
from typing import List, Optional, Tuple
from .math2d import Vec2, Transform2D
from .shapes import ShapeType, Circle, Polygon
from .body import RigidBody
from .gjk import gjk_intersect
from .epa import epa_penetration


class ContactPoint:
    """Individual contact point on a contact manifold."""
    __slots__ = (
        "position", "penetration", "r_a", "r_b",
        "normal_impulse", "tangent_impulse", "normal_mass", "tangent_mass",
        "velocity_bias", "position_bias"
    )

    def __init__(self, position: Vec2, penetration: float) -> None:
        self.position = position
        self.penetration = penetration
        self.r_a = Vec2(0.0, 0.0)  # Vector from body A COM to contact
        self.r_b = Vec2(0.0, 0.0)  # Vector from body B COM to contact
        self.normal_impulse = 0.0
        self.tangent_impulse = 0.0
        self.normal_mass = 0.0
        self.tangent_mass = 0.0
        self.velocity_bias = 0.0
        self.position_bias = 0.0


class ContactManifold:
    """Manifold representing geometric contact between two rigid bodies."""
    __slots__ = (
        "body_a", "body_b", "normal", "points",
        "restitution", "static_friction", "dynamic_friction"
    )

    def __init__(
        self,
        body_a: RigidBody,
        body_b: RigidBody,
        normal: Vec2,
        points: List[ContactPoint]
    ) -> None:
        self.body_a = body_a
        self.body_b = body_b
        self.normal = normal  # Points from A to B
        self.points = points

        # Combined material properties
        mat_a = body_a.material
        mat_b = body_b.material
        self.restitution = min(mat_a.restitution, mat_b.restitution)
        self.static_friction = math.sqrt(mat_a.static_friction * mat_b.static_friction)
        self.dynamic_friction = math.sqrt(mat_a.dynamic_friction * mat_b.dynamic_friction)


def clip_segment_to_line(v_in: List[Vec2], normal: Vec2, offset: float) -> List[Vec2]:
    """Clips line segment against half-plane normal . x <= offset."""
    v_out: List[Vec2] = []
    if len(v_in) < 2:
        return v_in
    d0 = normal.dot(v_in[0]) - offset
    d1 = normal.dot(v_in[1]) - offset
    if d0 <= 0.0:
        v_out.append(v_in[0])
    if d1 <= 0.0:
        v_out.append(v_in[1])
    if (d0 > 0.0 and d1 <= 0.0) or (d0 <= 0.0 and d1 > 0.0):
        t = d0 / (d0 - d1)
        v_out.append(v_in[0] + (v_in[1] - v_in[0]) * t)
    return v_out


def find_best_edge(poly: Polygon, transform: Transform2D, normal: Vec2) -> Tuple[Vec2, Vec2, Vec2]:
    """
    Finds edge on polygon whose outward normal is most aligned with normal.
    Returns (v1_world, v2_world, edge_normal_world).
    """
    # Transform normal into polygon local orientation
    rot = transform.rotation
    normal_local = Vec2(
        rot.col1.x * normal.x + rot.col1.y * normal.y,
        rot.col2.x * normal.x + rot.col2.y * normal.y
    )
    best_dot = -float("inf")
    best_index = 0
    for i, n in enumerate(poly.normals):
        dot = n.dot(normal_local)
        if dot > best_dot:
            best_dot = dot
            best_index = i

    v1_local = poly.vertices[best_index]
    v2_local = poly.vertices[(best_index + 1) % len(poly.vertices)]
    v1_world = transform.transform_point(v1_local)
    v2_world = transform.transform_point(v2_local)
    edge_normal = transform.transform_vector(poly.normals[best_index])
    return v1_world, v2_world, edge_normal


def collide_circle_circle(body_a: RigidBody, body_b: RigidBody) -> Optional[ContactManifold]:
    """Analytical collision between two circles."""
    circ_a: Circle = body_a.shape  # type: ignore
    circ_b: Circle = body_b.shape  # type: ignore

    pos_a = body_a.transform.transform_point(circ_a.center)
    pos_b = body_b.transform.transform_point(circ_b.center)

    diff = pos_b - pos_a
    dist_sq = diff.length_sq()
    radius_sum = circ_a.radius + circ_b.radius

    if dist_sq >= radius_sum * radius_sum:
        return None

    dist = math.sqrt(dist_sq)
    if dist < 1e-9:
        normal = Vec2(0.0, 1.0)
        penetration = radius_sum
        contact_pos = pos_a
    else:
        normal = diff * (1.0 / dist)
        penetration = radius_sum - dist
        contact_pos = pos_a + normal * (circ_a.radius - penetration * 0.5)

    cp = ContactPoint(contact_pos, penetration)
    return ContactManifold(body_a, body_b, normal, [cp])


def collide_circle_polygon(body_circle: RigidBody, body_poly: RigidBody) -> Optional[ContactManifold]:
    """Collision between Circle (body_a) and Polygon (body_b). Normal points from circle to polygon."""
    circ: Circle = body_circle.shape  # type: ignore
    poly: Polygon = body_poly.shape   # type: ignore

    center_world = body_circle.transform.transform_point(circ.center)
    center_local = body_poly.transform.inverse_transform_point(center_world)

    separation = -float("inf")
    normal_index = 0

    for i in range(len(poly.vertices)):
        s = poly.normals[i].dot(center_local - poly.vertices[i])
        if s > circ.radius:
            return None
        if s > separation:
            separation = s
            normal_index = i

    v1 = poly.vertices[normal_index]
    v2 = poly.vertices[(normal_index + 1) % len(poly.vertices)]

    # Check if center is within edge segment bounds
    if separation < 1e-9:
        normal_world = body_poly.transform.transform_vector(poly.normals[normal_index])
        penetration = circ.radius - separation
        contact_pos = center_world - normal_world * (circ.radius - penetration * 0.5)
        cp = ContactPoint(contact_pos, penetration)
        return ContactManifold(body_circle, body_poly, -normal_world, [cp])

    u1 = (center_local - v1).dot(v2 - v1)
    u2 = (center_local - v2).dot(v1 - v2)

    if u1 <= 0.0:
        diff = center_local - v1
        dist_sq = diff.length_sq()
        if dist_sq > circ.radius * circ.radius:
            return None
        dist = math.sqrt(dist_sq)
        normal_world = body_poly.transform.transform_vector(diff * (1.0 / dist if dist > 1e-9 else 1.0))
        penetration = circ.radius - dist
        contact_pos = body_poly.transform.transform_point(v1)
        cp = ContactPoint(contact_pos, penetration)
        return ContactManifold(body_circle, body_poly, -normal_world, [cp])

    elif u2 <= 0.0:
        diff = center_local - v2
        dist_sq = diff.length_sq()
        if dist_sq > circ.radius * circ.radius:
            return None
        dist = math.sqrt(dist_sq)
        normal_world = body_poly.transform.transform_vector(diff * (1.0 / dist if dist > 1e-9 else 1.0))
        penetration = circ.radius - dist
        contact_pos = body_poly.transform.transform_point(v2)
        cp = ContactPoint(contact_pos, penetration)
        return ContactManifold(body_circle, body_poly, -normal_world, [cp])

    else:
        normal_world = body_poly.transform.transform_vector(poly.normals[normal_index])
        penetration = circ.radius - separation
        contact_pos = center_world - normal_world * (circ.radius - penetration * 0.5)
        cp = ContactPoint(contact_pos, penetration)
        return ContactManifold(body_circle, body_poly, -normal_world, [cp])


def collide_convex_gjk_epa(body_a: RigidBody, body_b: RigidBody) -> Optional[ContactManifold]:
    """General convex polygon vs polygon collision using GJK, EPA, and feature edge clipping."""
    colliding, simplex = gjk_intersect(body_a, body_b)
    if not colliding:
        return None

    penetration, normal = epa_penetration(body_a, body_b, simplex)
    if penetration <= 1e-6:
        return None

    poly_a: Polygon = body_a.shape  # type: ignore
    poly_b: Polygon = body_b.shape  # type: ignore

    # Find edge on body A aligned with normal and on body B aligned with -normal
    e1_a, e2_a, n_a = find_best_edge(poly_a, body_a.transform, normal)
    e1_b, e2_b, n_b = find_best_edge(poly_b, body_b.transform, -normal)

    # Reference edge is more perpendicular to collision normal
    if abs(n_a.dot(normal)) >= abs(n_b.dot(-normal)):
        ref_v1, ref_v2, ref_n = e1_a, e2_a, n_a
        inc_v1, inc_v2 = e1_b, e2_b
    else:
        ref_v1, ref_v2, ref_n = e1_b, e2_b, n_b
        inc_v1, inc_v2 = e1_a, e2_a

    ref_tangent = (ref_v2 - ref_v1).normalized()
    sp1_offset = (-ref_tangent).dot(ref_v1)
    sp2_offset = ref_tangent.dot(ref_v2)

    clipped = clip_segment_to_line([inc_v1, inc_v2], -ref_tangent, sp1_offset)
    if len(clipped) >= 2:
        clipped = clip_segment_to_line(clipped, ref_tangent, sp2_offset)
    else:
        clipped = clip_segment_to_line([inc_v1, inc_v2], ref_tangent, sp2_offset)

    contact_points: List[ContactPoint] = []
    ref_offset = ref_n.dot(ref_v1)

    for pt in clipped:
        sep = ref_n.dot(pt) - ref_offset
        if sep <= 0.0:
            pt_pen = -sep
            contact_pos = pt - ref_n * (sep * 0.5)
            contact_points.append(ContactPoint(contact_pos, pt_pen))

    # Fallback if edge clipping generated no penetrating points
    if not contact_points:
        p_a = body_a.shape.support(normal, body_a.transform)
        p_b = body_b.shape.support(-normal, body_b.transform)
        contact_points.append(ContactPoint((p_a + p_b) * 0.5, penetration))

    return ContactManifold(body_a, body_b, normal, contact_points)


def find_collision(body_a: RigidBody, body_b: RigidBody) -> Optional[ContactManifold]:
    """Dispatch collision detection to optimal specialized routine or GJK/EPA."""
    type_a = body_a.shape.shape_type
    type_b = body_b.shape.shape_type

    if type_a == ShapeType.CIRCLE and type_b == ShapeType.CIRCLE:
        return collide_circle_circle(body_a, body_b)
    elif type_a == ShapeType.CIRCLE and type_b == ShapeType.POLYGON:
        return collide_circle_polygon(body_a, body_b)
    elif type_a == ShapeType.POLYGON and type_b == ShapeType.CIRCLE:
        # Swap and flip normal
        manifold = collide_circle_polygon(body_b, body_a)
        if manifold:
            manifold.body_a, manifold.body_b = body_a, body_b
            manifold.normal = -manifold.normal
        return manifold
    else:
        return collide_convex_gjk_epa(body_a, body_b)
