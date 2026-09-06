"""3D Gaussian Splatting representation and 3D covariance matrix.

Provides exact quaternion rotations, scale transformations, and positive
semi-definite 3D spatial covariance matrix formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion representing 3D spatial orientation: w + xi + yj + zk."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def norm(self) -> float:
        """Compute Euclidean norm of quaternion components."""
        return math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Quaternion:
        """Return unit quaternion normalized to unit magnitude."""
        n = self.norm()
        if n < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Quaternion(self.w * inv, self.x * inv, self.y * inv, self.z * inv)

    @classmethod
    def identity(cls) -> Quaternion:
        """Construct identity rotation quaternion."""
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: Tuple[float, float, float], angle_rad: float) -> Quaternion:
        """Construct rotation quaternion from unit axis and angle in radians."""
        ax, ay, az = axis
        axis_len = math.sqrt(ax * ax + ay * ay + az * az)
        if axis_len < 1e-12:
            return cls.identity()
        inv_len = 1.0 / axis_len
        ux, uy, uz = ax * inv_len, ay * inv_len, az * inv_len
        half = 0.5 * angle_rad
        sin_half = math.sin(half)
        cos_half = math.cos(half)
        return cls(cos_half, ux * sin_half, uy * sin_half, uz * sin_half).normalized()

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> Quaternion:
        """Construct rotation quaternion from Euler angles (in radians: pitch=X, yaw=Y, roll=Z)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return cls(w, x, y, z).normalized()

    def multiply(self, other: Quaternion) -> Quaternion:
        """Hamilton product of two quaternions: self * other."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion(w, x, y, z).normalized()

    def to_rotation_matrix(self) -> Tuple[Tuple[float, float, float], ...]:
        """Convert normalized quaternion to 3x3 orthonormal rotation matrix."""
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        r00 = 1.0 - 2.0 * (yy + zz)
        r01 = 2.0 * (xy - wz)
        r02 = 2.0 * (xz + wy)

        r10 = 2.0 * (xy + wz)
        r11 = 1.0 - 2.0 * (xx + zz)
        r12 = 2.0 * (yz - wx)

        r20 = 2.0 * (xz - wy)
        r21 = 2.0 * (yz + wx)
        r22 = 1.0 - 2.0 * (xx + yy)

        return (
            (r00, r01, r02),
            (r10, r11, r12),
            (r20, r21, r22),
        )


@dataclass
class Gaussian3D:
    """3D Gaussian parameterization for volume radiance fields.

    Attributes:
        position: (x, y, z) spatial center in world coordinates.
        scale: (sx, sy, sz) half-extents along local principal axes.
        rotation: Quaternion orientation governing principal axes alignment.
        opacity: Scalar opacity alpha in [0.0, 1.0].
        sh_coeffs: Spherical harmonics coefficients (Tuple of (r, g, b) tuples).
            For degree 0: 1 coefficient.
            For degree 1: 4 coefficients.
            For degree 2: 9 coefficients.
            For degree 3: 16 coefficients.
    """

    position: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    opacity: float = 1.0
    sh_coeffs: Tuple[Tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)

    def __post_init__(self) -> None:
        # Enforce positive scale elements to preserve non-degeneracy
        sx = max(1e-6, abs(self.scale[0]))
        sy = max(1e-6, abs(self.scale[1]))
        sz = max(1e-6, abs(self.scale[2]))
        self.scale = (sx, sy, sz)

        # Clamp opacity to valid probability range
        self.opacity = max(0.0, min(1.0, self.opacity))

    def compute_covariance_3d(self) -> Tuple[Tuple[float, float, float], ...]:
        """Compute the 3x3 symmetric positive semi-definite covariance matrix Sigma.

        Sigma = R * S * S^T * R^T = M * M^T where M = R * S.
        Sigma_ik = sum_j (R_ij * R_kj * s_j^2).
        """
        r = self.rotation.to_rotation_matrix()
        sx2 = self.scale[0] * self.scale[0]
        sy2 = self.scale[1] * self.scale[1]
        sz2 = self.scale[2] * self.scale[2]

        # Row 0
        r00, r01, r02 = r[0]
        r10, r11, r12 = r[1]
        r20, r21, r22 = r[2]

        sig00 = r00 * r00 * sx2 + r01 * r01 * sy2 + r02 * r02 * sz2
        sig01 = r00 * r10 * sx2 + r01 * r11 * sy2 + r02 * r12 * sz2
        sig02 = r00 * r20 * sx2 + r01 * r21 * sy2 + r02 * r22 * sz2

        sig11 = r10 * r10 * sx2 + r11 * r11 * sy2 + r12 * r12 * sz2
        sig12 = r10 * r20 * sx2 + r11 * r21 * sy2 + r12 * r22 * sz2

        sig22 = r20 * r20 * sx2 + r21 * r21 * sy2 + r22 * r22 * sz2

        return (
            (sig00, sig01, sig02),
            (sig01, sig11, sig12),
            (sig02, sig12, sig22),
        )

    def volume(self) -> float:
        """Compute 3-sigma ellipsoid volume (4/3 * pi * sx * sy * sz)."""
        return (4.0 / 3.0) * math.pi * self.scale[0] * self.scale[1] * self.scale[2]

    def translate(self, dx: float, dy: float, dz: float) -> Gaussian3D:
        """Return translated copy of Gaussian."""
        px, py, pz = self.position
        return Gaussian3D(
            position=(px + dx, py + dy, pz + dz),
            scale=self.scale,
            rotation=self.rotation,
            opacity=self.opacity,
            sh_coeffs=self.sh_coeffs,
        )

    def scale_uniform(self, factor: float) -> Gaussian3D:
        """Return uniformly scaled copy of Gaussian."""
        f = max(1e-6, abs(factor))
        return Gaussian3D(
            position=self.position,
            scale=(self.scale[0] * f, self.scale[1] * f, self.scale[2] * f),
            rotation=self.rotation,
            opacity=self.opacity,
            sh_coeffs=self.sh_coeffs,
        )

    @classmethod
    def isotropic(
        cls,
        position: Tuple[float, float, float],
        radius: float,
        opacity: float = 1.0,
        color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Gaussian3D:
        """Construct an isotropic (spherical) 3D Gaussian with base degree-0 color."""
        c0 = 0.28209479177387814  # Y_0^0 factor
        # Since color = c_00 * Y_0^0 + 0.5, we solve for c_00:
        # c_00 = (color - 0.5) / Y_0^0
        inv_c0 = 1.0 / c0
        sh_0 = (
            (color_rgb[0] - 0.5) * inv_c0,
            (color_rgb[1] - 0.5) * inv_c0,
            (color_rgb[2] - 0.5) * inv_c0,
        )
        return cls(
            position=position,
            scale=(radius, radius, radius),
            rotation=Quaternion.identity(),
            opacity=opacity,
            sh_coeffs=(sh_0,),
        )
