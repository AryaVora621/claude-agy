"""
PhotonPBR: 3D Vector, Ray, and Color Primitives.
Provides high-performance vector algebra, reflection, Snell's law refraction,
cosine-weighted hemisphere sampling, and gamma-corrected color mapping.
"""

import math
import random
from typing import Tuple, Union


class Vec3:
    """3D Vector with Cartesian coordinates (x, y, z)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Union["Vec3", float, int]) -> "Vec3":
        if isinstance(other, Vec3):
            # Component-wise Hadamard product
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        val = float(other)
        return Vec3(self.x * val, self.y * val, self.z * val)

    def __rmul__(self, other: Union[float, int]) -> "Vec3":
        val = float(other)
        return Vec3(self.x * val, self.y * val, self.z * val)

    def __truediv__(self, other: Union[float, int]) -> "Vec3":
        val = float(other)
        inv = 1.0 / val
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z
        raise IndexError(f"Vec3 index {idx} out of range [0..2]")

    def dot(self, other: "Vec3") -> float:
        """Scalar dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        """Vector cross product."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        """Return unit vector along same direction."""
        l = self.length()
        if l < 1e-12:
            return Vec3(0, 0, 0)
        inv_l = 1.0 / l
        return Vec3(self.x * inv_l, self.y * inv_l, self.z * inv_l)

    def near_zero(self) -> bool:
        """Check if vector is close to zero in all dimensions."""
        s = 1e-8
        return (abs(self.x) < s) and (abs(self.y) < s) and (abs(self.z) < s)

    def reflect(self, normal: "Vec3") -> "Vec3":
        """Reflect vector across surface normal: v - 2*(v . n)*n."""
        return self - 2.0 * self.dot(normal) * normal

    def refract(self, normal: "Vec3", etai_over_etat: float) -> Tuple[bool, "Vec3"]:
        """
        Refract vector using Snell's Law.
        Returns (success, refracted_vec). If Total Internal Reflection occurs, success is False.
        """
        cos_theta = min((-self).dot(normal), 1.0)
        r_out_perp = etai_over_etat * (self + cos_theta * normal)
        perp_len_sq = r_out_perp.length_squared()
        if perp_len_sq > 1.0:
            # Total internal reflection: cannot refract
            return False, Vec3(0, 0, 0)
        r_out_parallel = -math.sqrt(abs(1.0 - perp_len_sq)) * normal
        return True, (r_out_perp + r_out_parallel)

    @staticmethod
    def random(min_val: float = 0.0, max_val: float = 1.0) -> "Vec3":
        return Vec3(
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val),
            random.uniform(min_val, max_val)
        )

    @staticmethod
    def random_in_unit_sphere() -> "Vec3":
        """Rejection sampling for points strictly inside the unit sphere."""
        while True:
            p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            if p.length_squared() < 1.0:
                return p

    @staticmethod
    def random_unit_vector() -> "Vec3":
        """Cosine-weighted diffuse reflection direction."""
        return Vec3.random_in_unit_sphere().normalized()

    @staticmethod
    def random_in_unit_disk() -> "Vec3":
        """Rejection sampling for thin-lens camera depth-of-field sampling."""
        while True:
            p = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), 0.0)
            if p.length_squared() < 1.0:
                return p

    def to_rgb_bytes(self, samples_per_pixel: int = 1) -> Tuple[int, int, int]:
        """Apply sample averaging and gamma 2.0 correction (sqrt tone mapping)."""
        scale = 1.0 / max(1, samples_per_pixel)
        r = self.x * scale
        g = self.y * scale
        b = self.z * scale

        # Gamma 2 correction
        r = math.sqrt(max(0.0, r))
        g = math.sqrt(max(0.0, g))
        b = math.sqrt(max(0.0, b))

        # Clamp to [0, 255]
        ir = int(256 * max(0.0, min(0.999, r)))
        ig = int(256 * max(0.0, min(0.999, g)))
        ib = int(256 * max(0.0, min(0.999, b)))
        return ir, ig, ib

    def to_ansi_truecolor(self, samples_per_pixel: int = 1) -> str:
        """Format 24-bit TrueColor ANSI escape sequence for terminal block characters."""
        ir, ig, ib = self.to_rgb_bytes(samples_per_pixel)
        return f"\033[38;2;{ir};{ig};{ib}m"


class Ray:
    """Parametric ray defined by origin and direction: P(t) = origin + t * dir."""
    __slots__ = ("origin", "direction")

    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalized()

    def at(self, t: float) -> Vec3:
        """Evaluate point along ray at distance parameter t."""
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(orig={self.origin}, dir={self.direction})"


class Color(Vec3):
    """HDR Color vector (r, g, b) with tone-mapping and ANSI terminal output."""
    pass
