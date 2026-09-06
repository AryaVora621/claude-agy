"""
Solida: 3D Differential Geometry, Vector & Matrix Transformations, Quaternions, Planes & Bounds.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional, Iterable, Sequence


class Vector3D:
    """Immutable 3D Cartesian vector with full vector space operations."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, name, value):
        raise AttributeError("Vector3D is immutable")

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3D):
            return False
        return (
            abs(self.x - other.x) < 1e-9
            and abs(self.y - other.y) < 1e-9
            and abs(self.z - other.z) < 1e-9
        )

    def __hash__(self) -> int:
        return hash((round(self.x, 8), round(self.y, 8), round(self.z, 8)))

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        s = float(scalar)
        return Vector3D(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, scalar: float) -> Vector3D:
        s = float(scalar)
        return Vector3D(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, scalar: float) -> Vector3D:
        s = float(scalar)
        if s == 0.0:
            raise ZeroDivisionError("Division of Vector3D by zero")
        inv = 1.0 / s
        return Vector3D(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: Vector3D) -> float:
        """Scalar dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """Vector cross product."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_sq(self) -> float:
        """Squared Euclidean norm."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        """Euclidean magnitude / norm."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vector3D:
        """Unit direction vector."""
        n = self.norm()
        if n < 1e-15:
            return Vector3D(0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Vector3D(self.x * inv, self.y * inv, self.z * inv)

    def distance_to(self, other: Vector3D) -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def distance_sq_to(self, other: Vector3D) -> float:
        """Squared distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx * dx + dy * dy + dz * dz

    def lerp(self, other: Vector3D, t: float) -> Vector3D:
        """Linear interpolation between self and other."""
        return Vector3D(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z),
        )

    def angle_to(self, other: Vector3D) -> float:
        """Angle between two vectors in radians [0, pi]."""
        d = self.dot(other)
        m = self.norm() * other.norm()
        if m < 1e-15:
            return 0.0
        cos_theta = max(-1.0, min(1.0, d / m))
        return math.acos(cos_theta)

    def project_onto(self, target: Vector3D) -> Vector3D:
        """Orthogonal projection of self onto target vector."""
        t_sq = target.norm_sq()
        if t_sq < 1e-15:
            return Vector3D(0.0, 0.0, 0.0)
        scale = self.dot(target) / t_sq
        return target * scale

    def reject_from(self, target: Vector3D) -> Vector3D:
        """Component of self orthogonal to target vector."""
        return self - self.project_onto(target)

    def reflect(self, normal: Vector3D) -> Vector3D:
        """Reflection across a surface normal."""
        return self - normal * (2.0 * self.dot(normal))

    def is_zero(self, tol: float = 1e-9) -> bool:
        """Returns True if vector magnitude is within tolerance of zero."""
        return self.norm_sq() <= tol * tol

    def approx_eq(self, other: Vector3D, tol: float = 1e-6) -> bool:
        """Returns True if two vectors are equal within numerical tolerance."""
        return (
            abs(self.x - other.x) <= tol
            and abs(self.y - other.y) <= tol
            and abs(self.z - other.z) <= tol
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Matrix4x4:
    """4x4 Transformation Matrix in row-major representation."""
    __slots__ = ("m",)

    def __init__(self, elements: Optional[Sequence[float]] = None) -> None:
        if elements is None:
            # Identity matrix
            self.m = (
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            )
        else:
            if len(elements) != 16:
                raise ValueError(f"Matrix4x4 requires 16 elements, got {len(elements)}")
            self.m = tuple(float(x) for x in elements)

    def __repr__(self) -> str:
        lines = []
        for row in range(4):
            r = [f"{self.m[row * 4 + col]:8.4f}" for col in range(4)]
            lines.append(" ".join(r))
        return "Matrix4x4(\n  " + "\n  ".join(lines) + "\n)"

    def __getitem__(self, index: Tuple[int, int]) -> float:
        row, col = index
        return self.m[row * 4 + col]

    @property
    def data(self) -> List[List[float]]:
        return [[self.m[r * 4 + c] for c in range(4)] for r in range(4)]

    @classmethod
    def identity(cls) -> Matrix4x4:
        return cls()

    @classmethod
    def translation(cls, tx: float, ty: float, tz: float) -> Matrix4x4:
        return cls((
            1.0, 0.0, 0.0, float(tx),
            0.0, 1.0, 0.0, float(ty),
            0.0, 0.0, 1.0, float(tz),
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def scaling(cls, sx: float, sy: float, sz: float) -> Matrix4x4:
        return cls((
            float(sx), 0.0, 0.0, 0.0,
            0.0, float(sy), 0.0, 0.0,
            0.0, 0.0, float(sz), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_x(cls, angle_rad: float) -> Matrix4x4:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return cls((
            1.0, 0.0, 0.0, 0.0,
            0.0,   c,  -s, 0.0,
            0.0,   s,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_y(cls, angle_rad: float) -> Matrix4x4:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return cls((
              c, 0.0,   s, 0.0,
            0.0, 1.0, 0.0, 0.0,
             -s, 0.0,   c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_z(cls, angle_rad: float) -> Matrix4x4:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return cls((
              c,  -s, 0.0, 0.0,
              s,   c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_axis_angle(cls, axis: Vector3D, angle_rad: float) -> Matrix4x4:
        """Rodrigues rotation matrix about an arbitrary unit axis."""
        u = axis.normalized()
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        t = 1.0 - c

        m00 = t * u.x * u.x + c
        m01 = t * u.x * u.y - s * u.z
        m02 = t * u.x * u.z + s * u.y
        m10 = t * u.x * u.y + s * u.z
        m11 = t * u.y * u.y + c
        m12 = t * u.y * u.z - s * u.x
        m20 = t * u.x * u.z - s * u.y
        m21 = t * u.y * u.z + s * u.x
        m22 = t * u.z * u.z + c

        return cls((
            m00, m01, m02, 0.0,
            m10, m11, m12, 0.0,
            m20, m21, m22, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def look_at(cls, eye: Vector3D, target: Vector3D, up: Vector3D) -> Matrix4x4:
        """View matrix oriented from eye to target."""
        f = (target - eye).normalized()
        s = f.cross(up).normalized()
        u = s.cross(f)

        return cls((
             s.x,  s.y,  s.z, -s.dot(eye),
             u.x,  u.y,  u.z, -u.dot(eye),
            -f.x, -f.y, -f.z,  f.dot(eye),
             0.0,  0.0,  0.0,  1.0,
        ))

    def __matmul__(self, other: Matrix4x4) -> Matrix4x4:
        """Matrix multiplication."""
        a = self.m
        b = other.m
        res = [0.0] * 16
        for r in range(4):
            r4 = r * 4
            for c in range(4):
                res[r4 + c] = (
                    a[r4 + 0] * b[0 * 4 + c]
                    + a[r4 + 1] * b[1 * 4 + c]
                    + a[r4 + 2] * b[2 * 4 + c]
                    + a[r4 + 3] * b[3 * 4 + c]
                )
        return Matrix4x4(res)

    def transform_point(self, p: Vector3D) -> Vector3D:
        """Transforms a 3D point (w = 1.0) with perspective divide."""
        m = self.m
        w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15]
        inv_w = 1.0 / w if abs(w) > 1e-15 else 1.0
        return Vector3D(
            (m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3]) * inv_w,
            (m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7]) * inv_w,
            (m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]) * inv_w,
        )

    def transform_vector(self, v: Vector3D) -> Vector3D:
        """Transforms a direction vector (w = 0.0, translations ignored)."""
        m = self.m
        return Vector3D(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z,
        )

    def transpose(self) -> Matrix4x4:
        m = self.m
        return Matrix4x4((
            m[0], m[4], m[8],  m[12],
            m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15],
        ))

    def determinant(self) -> float:
        """Full 4x4 analytical determinant."""
        m = self.m
        a = m[0] * (
            m[5] * (m[10] * m[15] - m[11] * m[14])
            - m[6] * (m[9] * m[15] - m[11] * m[13])
            + m[7] * (m[9] * m[14] - m[10] * m[13])
        )
        b = m[1] * (
            m[4] * (m[10] * m[15] - m[11] * m[14])
            - m[6] * (m[8] * m[15] - m[11] * m[12])
            + m[7] * (m[8] * m[14] - m[10] * m[12])
        )
        c = m[2] * (
            m[4] * (m[9] * m[15] - m[11] * m[13])
            - m[5] * (m[8] * m[15] - m[11] * m[12])
            + m[7] * (m[8] * m[13] - m[9] * m[12])
        )
        d = m[3] * (
            m[4] * (m[9] * m[14] - m[10] * m[13])
            - m[5] * (m[8] * m[14] - m[10] * m[12])
            + m[6] * (m[8] * m[13] - m[9] * m[12])
        )
        return a - b + c - d

    def inverse(self) -> Matrix4x4:
        """Analytical 4x4 matrix inversion using Laplace cofactor expansions."""
        m = self.m
        det = self.determinant()
        if abs(det) < 1e-15:
            raise ValueError("Matrix4x4 is singular, cannot invert")
        inv_det = 1.0 / det

        # Sub-determinants for 2x2 blocks
        b00 = m[0] * m[5] - m[1] * m[4]
        b01 = m[0] * m[6] - m[2] * m[4]
        b02 = m[0] * m[7] - m[3] * m[4]
        b03 = m[1] * m[6] - m[2] * m[5]
        b04 = m[1] * m[7] - m[3] * m[5]
        b05 = m[2] * m[7] - m[3] * m[6]
        b06 = m[8] * m[13] - m[9] * m[12]
        b07 = m[8] * m[14] - m[10] * m[12]
        b08 = m[8] * m[15] - m[11] * m[12]
        b09 = m[9] * m[14] - m[10] * m[13]
        b10 = m[9] * m[15] - m[11] * m[13]
        b11 = m[10] * m[15] - m[11] * m[14]

        inv = [
            (m[5] * b11 - m[6] * b10 + m[7] * b09) * inv_det,
            (-m[1] * b11 + m[2] * b10 - m[3] * b09) * inv_det,
            (m[13] * b05 - m[14] * b04 + m[15] * b03) * inv_det,
            (-m[9] * b05 + m[10] * b04 - m[11] * b03) * inv_det,

            (-m[4] * b11 + m[6] * b08 - m[7] * b07) * inv_det,
            (m[0] * b11 - m[2] * b08 + m[3] * b07) * inv_det,
            (-m[12] * b05 + m[14] * b02 - m[15] * b01) * inv_det,
            (m[8] * b05 - m[10] * b02 + m[11] * b01) * inv_det,

            (m[4] * b10 - m[5] * b08 + m[7] * b06) * inv_det,
            (-m[0] * b10 + m[1] * b08 - m[3] * b06) * inv_det,
            (m[12] * b04 - m[13] * b02 + m[15] * b00) * inv_det,
            (-m[8] * b04 + m[9] * b02 - m[11] * b00) * inv_det,

            (-m[4] * b09 + m[5] * b07 - m[6] * b06) * inv_det,
            (m[0] * b09 - m[1] * b07 + m[2] * b06) * inv_det,
            (-m[12] * b03 + m[13] * b01 - m[14] * b00) * inv_det,
            (m[8] * b03 - m[9] * b01 + m[10] * b00) * inv_det,
        ]
        return Matrix4x4(inv)


class Quaternion:
    """Unit quaternion for spatial rotations."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        object.__setattr__(self, "w", float(w))
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, name, value):
        raise AttributeError("Quaternion is immutable")

    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.4f}, x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: Vector3D, angle_rad: float) -> Quaternion:
        u = axis.normalized()
        half = 0.5 * angle_rad
        s = math.sin(half)
        return cls(math.cos(half), u.x * s, u.y * s, u.z * s)

    def norm(self) -> float:
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Quaternion:
        n = self.norm()
        if n < 1e-15:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Quaternion(self.w * inv, self.x * inv, self.y * inv, self.z * inv)

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate_vector(self, v: Vector3D) -> Vector3D:
        """Rotate vector v using q * v * q^*."""
        qv = Quaternion(0.0, v.x, v.y, v.z)
        qr = (self * qv) * self.conjugate()
        return Vector3D(qr.x, qr.y, qr.z)

    def to_matrix(self) -> Matrix4x4:
        """Convert quaternion to 4x4 rotation matrix."""
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return Matrix4x4((
            1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),       2.0 * (xz + wy), 0.0,
                  2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),       2.0 * (yz - wx), 0.0,
                  2.0 * (xz - wy),       2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0,
                              0.0,                   0.0,                   0.0, 1.0,
        ))

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation between self and other."""
        q1 = self.normalized()
        q2 = other.normalized()

        dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

        # If negative dot, invert one quaternion to take shortest path
        if dot < 0.0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot

        if dot > 0.9995:
            # Linear interpolation for very close orientations
            res = Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
            )
            return res.normalized()

        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return Quaternion(
            s0 * q1.w + s1 * q2.w,
            s0 * q1.x + s1 * q2.x,
            s0 * q1.y + s1 * q2.y,
            s0 * q1.z + s1 * q2.z,
        ).normalized()


class Ray3D:
    """Parametric 3D ray r(t) = origin + t * direction (t >= 0)."""
    __slots__ = ("origin", "direction")

    def __init__(self, origin: Vector3D, direction: Vector3D) -> None:
        self.origin = origin
        self.direction = direction.normalized()

    def point_at(self, t: float) -> Vector3D:
        return self.origin + self.direction * t

    def intersect_sphere(self, center: Vector3D, radius: float) -> Optional[Tuple[float, float]]:
        """Finds entry and exit t parameters for sphere intersection."""
        oc = self.origin - center
        b = oc.dot(self.direction)
        c = oc.dot(oc) - radius * radius
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc
        return (t1, t2)

    def intersect_triangle(
        self, v0: Vector3D, v1: Vector3D, v2: Vector3D, epsilon: float = 1e-9
    ) -> Optional[Tuple[float, float, float]]:
        """Möller-Trumbore ray-triangle intersection. Returns (t, u, v)."""
        edge1 = v1 - v0
        edge2 = v2 - v0
        pvec = self.direction.cross(edge2)
        det = edge1.dot(pvec)

        if abs(det) < epsilon:
            return None

        inv_det = 1.0 / det
        tvec = self.origin - v0
        u = tvec.dot(pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None

        qvec = tvec.cross(edge1)
        v = self.direction.dot(qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return None

        t = edge2.dot(qvec) * inv_det
        if t < epsilon:
            return None

        return (t, u, v)

    def intersect_plane(self, plane: Plane) -> Optional[Vector3D]:
        t = plane.intersect_ray(self)
        return self.point_at(t) if t is not None else None


class Plane:
    """Infinite plane defined by unit normal n and offset d: n . x + d = 0."""
    __slots__ = ("normal", "d")

    def __init__(self, normal: Vector3D, d: float) -> None:
        n = normal.normalized()
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "d", float(d))

    def __setattr__(self, name, value):
        raise AttributeError("Plane is immutable")

    def __repr__(self) -> str:
        return f"Plane(n={self.normal}, d={self.d:.6f})"

    @classmethod
    def from_point_and_normal(cls, point: Vector3D, normal: Vector3D) -> Plane:
        n = normal.normalized()
        d = -n.dot(point)
        return cls(n, d)

    @classmethod
    def from_three_points(cls, p1: Vector3D, p2: Vector3D, p3: Vector3D) -> Plane:
        v1 = p2 - p1
        v2 = p3 - p1
        normal = v1.cross(v2)
        if normal.norm_sq() < 1e-15:
            raise ValueError("Collinear points cannot define a plane")
        return cls.from_point_and_normal(p1, normal)

    def signed_distance(self, point: Vector3D) -> float:
        """Signed perpendicular distance: >0 is front, <0 is back, 0 is on-plane."""
        return self.normal.dot(point) + self.d

    def project_point(self, point: Vector3D) -> Vector3D:
        """Orthogonal projection of point onto plane."""
        dist = self.signed_distance(point)
        return point - self.normal * dist

    def intersect_ray(self, ray: Ray3D, epsilon: float = 1e-9) -> Optional[float]:
        """Calculates distance t where ray hits plane."""
        denom = self.normal.dot(ray.direction)
        if abs(denom) < epsilon:
            return None
        t = -(self.normal.dot(ray.origin) + self.d) / denom
        return t if t >= 0.0 else None

    def clip_polygon(
        self, polygon: Sequence[Vector3D], epsilon: float = 1e-6
    ) -> Tuple[List[Vector3D], List[Vector3D]]:
        """
        Sutherland-Hodgman style clipping of a convex 3D polygon by this plane.
        Returns (front_polygon, back_polygon).
        """
        front: List[Vector3D] = []
        back: List[Vector3D] = []
        num_verts = len(polygon)
        if num_verts < 3:
            return (front, back)

        distances = [self.signed_distance(v) for v in polygon]

        for i in range(num_verts):
            curr_v = polygon[i]
            next_v = polygon[(i + 1) % num_verts]
            curr_d = distances[i]
            next_d = distances[(i + 1) % num_verts]

            if curr_d >= -epsilon:
                front.append(curr_v)
            if curr_d <= epsilon:
                back.append(curr_v)

            # Check if edge crosses the plane
            if (curr_d > epsilon and next_d < -epsilon) or (curr_d < -epsilon and next_d > epsilon):
                t = curr_d / (curr_d - next_d)
                inter = curr_v.lerp(next_v, t)
                front.append(inter)
                back.append(inter)

        return (front, back)


class BoundingBox3D:
    """Axis-Aligned Bounding Box (AABB)."""
    __slots__ = ("min_pt", "max_pt")

    def __init__(self, min_pt: Optional[Vector3D] = None, max_pt: Optional[Vector3D] = None) -> None:
        if min_pt is None or max_pt is None:
            self.min_pt = Vector3D(float("inf"), float("inf"), float("inf"))
            self.max_pt = Vector3D(-float("inf"), -float("inf"), -float("inf"))
        else:
            self.min_pt = Vector3D(
                min(min_pt.x, max_pt.x),
                min(min_pt.y, max_pt.y),
                min(min_pt.z, max_pt.z),
            )
            self.max_pt = Vector3D(
                max(min_pt.x, max_pt.x),
                max(min_pt.y, max_pt.y),
                max(min_pt.z, max_pt.z),
            )

    def include_point(self, p: Vector3D) -> None:
        """Expands bounding box to include point p."""
        self.min_pt = Vector3D(
            min(self.min_pt.x, p.x),
            min(self.min_pt.y, p.y),
            min(self.min_pt.z, p.z),
        )
        self.max_pt = Vector3D(
            max(self.max_pt.x, p.x),
            max(self.max_pt.y, p.y),
            max(self.max_pt.z, p.z),
        )

    def include_box(self, other: BoundingBox3D) -> None:
        """Expands bounding box to enclose other bounding box."""
        self.include_point(other.min_pt)
        self.include_point(other.max_pt)

    def __repr__(self) -> str:
        return f"BoundingBox3D(min={self.min_pt}, max={self.max_pt})"

    @classmethod
    def from_points(cls, points: Iterable[Vector3D]) -> BoundingBox3D:
        pts = list(points)
        if not pts:
            return cls(Vector3D(0, 0, 0), Vector3D(0, 0, 0))
        min_x = min(p.x for p in pts)
        min_y = min(p.y for p in pts)
        min_z = min(p.z for p in pts)
        max_x = max(p.x for p in pts)
        max_y = max(p.y for p in pts)
        max_z = max(p.z for p in pts)
        return cls(Vector3D(min_x, min_y, min_z), Vector3D(max_x, max_y, max_z))

    def center(self) -> Vector3D:
        return (self.min_pt + self.max_pt) * 0.5

    def extents(self) -> Vector3D:
        return self.max_pt - self.min_pt

    def diagonal(self) -> float:
        return (self.max_pt - self.min_pt).norm()

    def volume(self) -> float:
        ext = self.extents()
        return max(0.0, ext.x) * max(0.0, ext.y) * max(0.0, ext.z)

    def contains_point(self, p: Vector3D, tol: float = 1e-7) -> bool:
        return (
            self.min_pt.x - tol <= p.x <= self.max_pt.x + tol
            and self.min_pt.y - tol <= p.y <= self.max_pt.y + tol
            and self.min_pt.z - tol <= p.z <= self.max_pt.z + tol
        )

    def intersects(self, other: BoundingBox3D) -> bool:
        return not (
            self.max_pt.x < other.min_pt.x or self.min_pt.x > other.max_pt.x
            or self.max_pt.y < other.min_pt.y or self.min_pt.y > other.max_pt.y
            or self.max_pt.z < other.min_pt.z or self.min_pt.z > other.max_pt.z
        )

    def union(self, other: BoundingBox3D) -> BoundingBox3D:
        return BoundingBox3D(
            Vector3D(
                min(self.min_pt.x, other.min_pt.x),
                min(self.min_pt.y, other.min_pt.y),
                min(self.min_pt.z, other.min_pt.z),
            ),
            Vector3D(
                max(self.max_pt.x, other.max_pt.x),
                max(self.max_pt.y, other.max_pt.y),
                max(self.max_pt.z, other.max_pt.z),
            ),
        )

    def intersect_ray(self, ray: Ray3D) -> Optional[Tuple[float, float]]:
        """Kay-Kajiya slab method for ray-AABB intersection."""
        t_min = -float("inf")
        t_max = float("inf")

        for i in range(3):
            orig = (ray.origin.x, ray.origin.y, ray.origin.z)[i]
            d = (ray.direction.x, ray.direction.y, ray.direction.z)[i]
            b_min = (self.min_pt.x, self.min_pt.y, self.min_pt.z)[i]
            b_max = (self.max_pt.x, self.max_pt.y, self.max_pt.z)[i]

            if abs(d) < 1e-15:
                if orig < b_min or orig > b_max:
                    return None
            else:
                inv_d = 1.0 / d
                t1 = (b_min - orig) * inv_d
                t2 = (b_max - orig) * inv_d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None

        if t_max < 0.0:
            return None

        return (max(0.0, t_min), t_max)
