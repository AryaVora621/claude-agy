"""
NovaPhysics: 2D Vector, Matrix, and Rigid Body Transform Mathematics.
Zero-dependency, high-performance linear algebra primitives.
"""

import math
from typing import Union, Tuple


class Vec2:
    """Two-dimensional vector with comprehensive geometric and algebraic operations."""
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Vec2({self.x:.4f}, {self.y:.4f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return False
        return math.isclose(self.x, other.x, abs_tol=1e-7) and math.isclose(self.y, other.y, abs_tol=1e-7)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Union[int, float]) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Union[int, float]) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: Union[int, float]) -> "Vec2":
        inv = 1.0 / scalar
        return Vec2(self.x * inv, self.y * inv)

    def dot(self, other: "Vec2") -> float:
        """Standard Euclidean inner dot product."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        """2D cross product: z-component of 3D cross product (scalar)."""
        return self.x * other.y - self.y * other.x

    def cross_scalar(self, scalar: float) -> "Vec2":
        """Cross product of 2D vector with a scalar perpendicular vector (v x s)."""
        return Vec2(scalar * self.y, -scalar * self.x)

    def length_sq(self) -> float:
        """Squared Euclidean length."""
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        """Euclidean magnitude (norm)."""
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        """Unit length vector in same direction; returns Vec2(0, 0) if length is 0."""
        l = self.length()
        if l < 1e-9:
            return Vec2(0.0, 0.0)
        inv = 1.0 / l
        return Vec2(self.x * inv, self.y * inv)

    def perp(self) -> "Vec2":
        """Perpendicular counter-clockwise vector (-y, x)."""
        return Vec2(-self.y, self.x)

    def distance_to(self, other: "Vec2") -> float:
        """Distance to another point."""
        return (self - other).length()

    def distance_sq_to(self, other: "Vec2") -> float:
        """Squared distance to another point."""
        return (self - other).length_sq()

    def rotated(self, radians: float) -> "Vec2":
        """Rotate vector counter-clockwise by radians."""
        c = math.cos(radians)
        s = math.sin(radians)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


def scalar_cross_vec(s: float, v: Vec2) -> Vec2:
    """Cross product of scalar z-component with 2D vector (s x v = Vec2(-s * v.y, s * v.x))."""
    return Vec2(-s * v.y, s * v.x)


class Mat22:
    """2x2 matrix for 2D orientation and moment of inertia calculations."""
    __slots__ = ("col1", "col2")

    def __init__(self, col1: Vec2, col2: Vec2) -> None:
        self.col1 = col1
        self.col2 = col2

    @classmethod
    def from_angle(cls, radians: float) -> "Mat22":
        """Construct rotation matrix from angle in radians."""
        c = math.cos(radians)
        s = math.sin(radians)
        return cls(Vec2(c, s), Vec2(-s, c))

    @classmethod
    def identity(cls) -> "Mat22":
        return cls(Vec2(1.0, 0.0), Vec2(0.0, 1.0))

    def __repr__(self) -> str:
        return f"Mat22([{self.col1.x:.3f}, {self.col2.x:.3f}], [{self.col1.y:.3f}, {self.col2.y:.3f}])"

    def mul_vec(self, v: Vec2) -> Vec2:
        """Matrix-vector multiplication M * v."""
        return Vec2(
            self.col1.x * v.x + self.col2.x * v.y,
            self.col1.y * v.x + self.col2.y * v.y
        )

    def mul_mat(self, other: "Mat22") -> "Mat22":
        """Matrix-matrix multiplication M * other."""
        return Mat22(self.mul_vec(other.col1), self.mul_vec(other.col2))

    def transpose(self) -> "Mat22":
        return Mat22(Vec2(self.col1.x, self.col2.x), Vec2(self.col1.y, self.col2.y))

    def determinant(self) -> float:
        return self.col1.x * self.col2.y - self.col2.x * self.col1.y

    def inverted(self) -> "Mat22":
        det = self.determinant()
        if abs(det) < 1e-12:
            return Mat22.identity()
        inv_det = 1.0 / det
        return Mat22(
            Vec2(self.col2.y * inv_det, -self.col1.y * inv_det),
            Vec2(-self.col2.x * inv_det, self.col1.x * inv_det)
        )


class Transform2D:
    """Rigid body coordinate transform representing translation and rotation."""
    __slots__ = ("position", "rotation", "angle")

    def __init__(self, position: Vec2, angle: float = 0.0) -> None:
        self.position = position
        self.angle = angle
        self.rotation = Mat22.from_angle(angle)

    def set_angle(self, angle: float) -> None:
        self.angle = angle
        self.rotation = Mat22.from_angle(angle)

    def transform_point(self, local_pt: Vec2) -> Vec2:
        """Convert point from local body space to world space."""
        return self.rotation.mul_vec(local_pt) + self.position

    def inverse_transform_point(self, world_pt: Vec2) -> Vec2:
        """Convert point from world space to local body space."""
        diff = world_pt - self.position
        return self.rotation.transpose().mul_vec(diff)

    def transform_vector(self, local_vec: Vec2) -> Vec2:
        """Rotate vector from local body space to world space."""
        return self.rotation.mul_vec(local_vec)

    def inverse_transform_vector(self, world_vec: Vec2) -> Vec2:
        """Rotate vector from world space to local body space."""
        return self.rotation.transpose().mul_vec(world_vec)
