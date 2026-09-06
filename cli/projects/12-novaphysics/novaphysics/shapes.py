"""
NovaPhysics: Geometric Shapes and Axis-Aligned Bounding Boxes.
Computes mass properties, moments of inertia, AABBs, and Minkowski support mappings.
"""

import math
from enum import Enum, auto
from typing import List, Tuple
from .math2d import Vec2, Transform2D


class ShapeType(Enum):
    CIRCLE = auto()
    POLYGON = auto()


class AABB:
    """Axis-Aligned Bounding Box for spatial pruning and broadphase tracking."""
    __slots__ = ("lower_bound", "upper_bound")

    def __init__(self, lower_bound: Vec2, upper_bound: Vec2) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __repr__(self) -> str:
        return f"AABB({self.lower_bound}, {self.upper_bound})"

    def overlaps(self, other: "AABB") -> bool:
        """Check if this AABB overlaps another."""
        if self.upper_bound.x < other.lower_bound.x or self.lower_bound.x > other.upper_bound.x:
            return False
        if self.upper_bound.y < other.lower_bound.y or self.lower_bound.y > other.upper_bound.y:
            return False
        return True

    def contains_point(self, pt: Vec2) -> bool:
        return (self.lower_bound.x <= pt.x <= self.upper_bound.x and
                self.lower_bound.y <= pt.y <= self.upper_bound.y)

    def contains_aabb(self, other: "AABB") -> bool:
        return (self.lower_bound.x <= other.lower_bound.x and
                self.lower_bound.y <= other.lower_bound.y and
                self.upper_bound.x >= other.upper_bound.x and
                self.upper_bound.y >= other.upper_bound.y)

    def perimeter(self) -> float:
        """Surface area heuristic (perimeter for 2D bounding boxes)."""
        w = self.upper_bound.x - self.lower_bound.x
        h = self.upper_bound.y - self.lower_bound.y
        return 2.0 * (w + h)

    def center(self) -> Vec2:
        return (self.lower_bound + self.upper_bound) * 0.5

    def union(self, other: "AABB") -> "AABB":
        """Smallest enclosing AABB covering both boxes."""
        return AABB(
            Vec2(min(self.lower_bound.x, other.lower_bound.x),
                 min(self.lower_bound.y, other.lower_bound.y)),
            Vec2(max(self.upper_bound.x, other.upper_bound.x),
                 max(self.upper_bound.y, other.upper_bound.y))
        )

    def fatten(self, margin: float) -> "AABB":
        """Expand bounds by a safety margin to reduce broadphase tree updates."""
        return AABB(
            Vec2(self.lower_bound.x - margin, self.lower_bound.y - margin),
            Vec2(self.upper_bound.x + margin, self.upper_bound.y + margin)
        )


class Shape:
    """Abstract geometric shape with mass and support functions."""
    __slots__ = ("shape_type",)

    def __init__(self, shape_type: ShapeType) -> None:
        self.shape_type = shape_type

    def compute_mass(self, density: float) -> Tuple[float, float, Vec2]:
        """Compute (mass, moment_of_inertia, center_of_mass)."""
        raise NotImplementedError

    def compute_aabb(self, transform: Transform2D) -> AABB:
        """Compute world-space AABB."""
        raise NotImplementedError

    def support(self, direction: Vec2, transform: Transform2D) -> Vec2:
        """Minkowski support mapping: furthest point along direction in world space."""
        raise NotImplementedError


class Circle(Shape):
    """Circle shape defined by radius and local center."""
    __slots__ = ("radius", "center")

    def __init__(self, radius: float, center: Vec2 = Vec2(0.0, 0.0)) -> None:
        super().__init__(ShapeType.CIRCLE)
        self.radius = float(radius)
        self.center = center

    def __repr__(self) -> str:
        return f"Circle(r={self.radius:.2f})"

    def compute_mass(self, density: float) -> Tuple[float, float, Vec2]:
        area = math.pi * self.radius * self.radius
        mass = density * area
        # Inertia of a solid disc: 0.5 * m * r^2
        inertia = 0.5 * mass * self.radius * self.radius
        return mass, inertia, self.center

    def compute_aabb(self, transform: Transform2D) -> AABB:
        world_center = transform.transform_point(self.center)
        r_vec = Vec2(self.radius, self.radius)
        return AABB(world_center - r_vec, world_center + r_vec)

    def support(self, direction: Vec2, transform: Transform2D) -> Vec2:
        world_center = transform.transform_point(self.center)
        d_norm = direction.normalized()
        return world_center + d_norm * self.radius


class Polygon(Shape):
    """Convex polygon with counter-clockwise oriented vertices."""
    __slots__ = ("vertices", "normals")

    def __init__(self, vertices: List[Vec2]) -> None:
        super().__init__(ShapeType.POLYGON)
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        # 1. Ensure counter-clockwise winding
        area_sum = 0.0
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            area_sum += v1.cross(v2)

        if area_sum < 0:
            # Clockwise: reverse vertices to make CCW
            vertices = list(reversed(vertices))

        # 2. Compute centroid and center vertices around origin
        cx = 0.0
        cy = 0.0
        area2 = 0.0
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            cross = v1.cross(v2)
            area2 += cross
            cx += (v1.x + v2.x) * cross
            cy += (v1.y + v2.y) * cross

        if abs(area2) > 1e-9:
            centroid = Vec2(cx / (3.0 * area2), cy / (3.0 * area2))
            self.vertices = [v - centroid for v in vertices]
        else:
            self.vertices = list(vertices)

        # 3. Compute outward edge unit normals
        self.normals: List[Vec2] = []
        for i in range(len(self.vertices)):
            edge = self.vertices[(i + 1) % len(self.vertices)] - self.vertices[i]
            # Normal perpendicular to edge pointing outward
            normal = Vec2(edge.y, -edge.x).normalized()
            self.normals.append(normal)

    def __repr__(self) -> str:
        return f"Polygon({len(self.vertices)} vertices)"

    def compute_mass(self, density: float) -> Tuple[float, float, Vec2]:
        """Compute mass and moment of inertia via polygon triangulation integration."""
        area = 0.0
        inertia = 0.0
        center = Vec2(0.0, 0.0)
        inv3 = 1.0 / 3.0

        n = len(self.vertices)
        for i in range(n):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % n]
            tri_area2 = p1.cross(p2)
            tri_area = 0.5 * tri_area2
            area += tri_area

            # Integral over triangle for moment of inertia about origin:
            # (p1.x^2 + p1.x*p2.x + p2.x^2 + p1.y^2 + p1.y*p2.y + p2.y^2) / 6
            d_sq = (p1.x * p1.x + p1.x * p2.x + p2.x * p2.x +
                    p1.y * p1.y + p1.y * p2.y + p2.y * p2.y)
            inertia += (0.25 * inv3 * tri_area2) * d_sq

        mass = density * area
        inertia = density * inertia
        return mass, max(inertia, 1e-5), center

    def compute_aabb(self, transform: Transform2D) -> AABB:
        world_verts = [transform.transform_point(v) for v in self.vertices]
        min_x = min(v.x for v in world_verts)
        min_y = min(v.y for v in world_verts)
        max_x = max(v.x for v in world_verts)
        max_y = max(v.y for v in world_verts)
        return AABB(Vec2(min_x, min_y), Vec2(max_x, max_y))

    def support(self, direction: Vec2, transform: Transform2D) -> Vec2:
        """Find vertex furthest along direction in world space."""
        # Convert search direction to body local space
        local_dir = transform.inverse_transform_vector(direction)
        best_vert = self.vertices[0]
        best_dot = best_vert.dot(local_dir)

        for i in range(1, len(self.vertices)):
            v = self.vertices[i]
            d = v.dot(local_dir)
            if d > best_dot:
                best_dot = d
                best_vert = v

        return transform.transform_point(best_vert)


class Box(Polygon):
    """Oriented rectangular bounding box helper."""

    def __init__(self, width: float, height: float) -> None:
        hw = width * 0.5
        hh = height * 0.5
        verts = [
            Vec2(-hw, -hh),
            Vec2(hw, -hh),
            Vec2(hw, hh),
            Vec2(-hw, hh)
        ]
        super().__init__(verts)
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Box({self.width:.2f}x{self.height:.2f})"
