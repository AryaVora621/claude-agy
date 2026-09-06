"""Camera model and 2D perspective projection (EWA Splatting).

Implements pinhole camera geometry, look-at viewing transformations, projective
Jacobian evaluation, 2D screen covariance matrix synthesis with low-pass Gaussian
filtering, and eigenvalue-based 3-sigma screen bounding radius extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from chromasplat.gaussian import Gaussian3D
from chromasplat.spherical_harmonics import eval_sh


@dataclass
class Camera:
    """Pinhole perspective camera with look-at extrinsics.

    Attributes:
        width: Image sensor width in pixels.
        height: Image sensor height in pixels.
        fov_y_rad: Vertical field of view in radians.
        position: (x, y, z) world coordinates of camera center.
        target: (x, y, z) world coordinates of look-at point.
        up: (x, y, z) world up direction vector.
        z_near: Near clipping distance in camera space.
        z_far: Far clipping distance in camera space.
    """

    width: int
    height: int
    fov_y_rad: float = math.radians(60.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    z_near: float = 0.05
    z_far: float = 100.0

    @property
    def focal_y(self) -> float:
        """Focal length along Y axis in pixels."""
        return self.height / (2.0 * math.tan(0.5 * self.fov_y_rad))

    @property
    def focal_x(self) -> float:
        """Focal length along X axis in pixels (assuming square pixels)."""
        return self.focal_y

    @property
    def center_x(self) -> float:
        """Principal point X coordinate on image plane."""
        return 0.5 * self.width

    @property
    def center_y(self) -> float:
        """Principal point Y coordinate on image plane."""
        return 0.5 * self.height

    def get_view_basis(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute camera orthonormal basis vectors (right, down, forward).

        Returns:
            Tuple of (right_vector, down_vector, forward_vector) in world coordinates.
        """
        px, py, pz = self.position
        tx, ty, tz = self.target
        fx, fy, fz = tx - px, ty - py, tz - pz
        f_len = math.sqrt(fx * fx + fy * fy + fz * fz)
        if f_len < 1e-12:
            fx, fy, fz = 0.0, 0.0, 1.0
        else:
            inv = 1.0 / f_len
            fx, fy, fz = fx * inv, fy * inv, fz * inv

        # Right vector = forward x up
        ux, uy, uz = self.up
        rx = fy * uz - fz * uy
        ry = fz * ux - fx * uz
        rz = fx * uy - fy * ux
        r_len = math.sqrt(rx * rx + ry * ry + rz * rz)
        if r_len < 1e-12:
            rx, ry, rz = 1.0, 0.0, 0.0
        else:
            inv = 1.0 / r_len
            rx, ry, rz = rx * inv, ry * inv, rz * inv

        # Down vector = -(right x forward) = forward x right
        # Image row coordinates increase downwards
        dx = fy * rz - fz * ry
        dy = fz * rx - fx * rz
        dz = fx * ry - fy * rx

        return (
            (rx, ry, rz),
            (dx, dy, dz),
            (fx, fy, fz),
        )

    def world_to_camera_point(self, pt: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Transform point from world space to camera coordinates (tx, ty, tz)."""
        right, down, forward = self.get_view_basis()
        dx = pt[0] - self.position[0]
        dy = pt[1] - self.position[1]
        dz = pt[2] - self.position[2]

        tx = dx * right[0] + dy * right[1] + dz * right[2]
        ty = dx * down[0] + dy * down[1] + dz * down[2]
        tz = dx * forward[0] + dy * forward[1] + dz * forward[2]
        return (tx, ty, tz)

    @classmethod
    def orbit(
        cls,
        width: int,
        height: int,
        azimuth_deg: float,
        elevation_deg: float,
        distance: float,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        fov_y_deg: float = 60.0,
    ) -> Camera:
        """Construct orbital camera centered on target."""
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(max(-89.9, min(89.9, elevation_deg)))
        d = max(0.01, distance)

        cx = target[0] + d * math.cos(el_rad) * math.sin(az_rad)
        cy = target[1] + d * math.sin(el_rad)
        cz = target[2] + d * math.cos(el_rad) * math.cos(az_rad)

        return cls(
            width=width,
            height=height,
            fov_y_rad=math.radians(fov_y_deg),
            position=(cx, cy, cz),
            target=target,
            up=(0.0, 1.0, 0.0),
        )


@dataclass
class ProjectedGaussian2D:
    """Projected 2D Gaussian in screen space with 2x2 covariance and bounding radius.

    Attributes:
        gaussian_idx: Index of source 3D Gaussian.
        u: Projected screen horizontal center in pixels.
        v: Projected screen vertical center in pixels.
        depth: Camera-space distance tz (for depth sorting).
        cov_a: Element (0, 0) of 2D screen covariance matrix.
        cov_b: Element (0, 1) and (1, 0) of 2D screen covariance matrix.
        cov_c: Element (1, 1) of 2D screen covariance matrix.
        inv_cov_a: Element (0, 0) of inverted 2D covariance.
        inv_cov_b: Element (0, 1) and (1, 0) of inverted 2D covariance.
        inv_cov_c: Element (1, 1) of inverted 2D covariance.
        radius: 3-sigma bounding radius in pixels.
        color: View-dependent RGB radiance tuple in [0.0, 1.0].
        opacity: Base Gaussian opacity alpha in [0.0, 1.0].
    """

    gaussian_idx: int
    u: float
    v: float
    depth: float
    cov_a: float
    cov_b: float
    cov_c: float
    inv_cov_a: float
    inv_cov_b: float
    inv_cov_c: float
    radius: float
    color: Tuple[float, float, float]
    opacity: float


class ProjectionEngine:
    """Performs perspective projection and EWA 2D covariance synthesis."""

    def __init__(self, anti_aliasing_filter: float = 0.3) -> None:
        """Initialize projection engine.

        Args:
            anti_aliasing_filter: Low-pass filter variance nu added to diagonal of
                2D screen covariance to avoid sub-pixel aliasing.
        """
        self.nu = max(0.01, anti_aliasing_filter)

    def project_single(
        self,
        gaussian: Gaussian3D,
        camera: Camera,
        view_basis: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]],
        gaussian_idx: int = 0,
        sh_degree: int = 0,
    ) -> Optional[ProjectedGaussian2D]:
        """Project a single 3D Gaussian to 2D screen space.

        Returns:
            ProjectedGaussian2D if in front of camera and inside screen margin, else None.
        """
        right, down, forward = view_basis

        # 1. Transform spatial center to camera space
        dx = gaussian.position[0] - camera.position[0]
        dy = gaussian.position[1] - camera.position[1]
        dz = gaussian.position[2] - camera.position[2]

        tx = dx * right[0] + dy * right[1] + dz * right[2]
        ty = dx * down[0] + dy * down[1] + dz * down[2]
        tz = dx * forward[0] + dy * forward[1] + dz * forward[2]

        # Near plane clipping
        if tz <= camera.z_near or tz >= camera.z_far:
            return None

        # 2. Projected screen coordinates
        inv_tz = 1.0 / tz
        inv_tz2 = inv_tz * inv_tz
        fx = camera.focal_x
        fy = camera.focal_y
        u = fx * tx * inv_tz + camera.center_x
        v = fy * ty * inv_tz + camera.center_y

        # 3. Transform 3D covariance to camera coordinates: Sigma_cam = R_view * Sigma * R_view^T
        cov3d = gaussian.compute_covariance_3d()
        r0, r1, r2 = right, down, forward

        # Intermediate M = R_view * Sigma (3x3)
        m00 = r0[0] * cov3d[0][0] + r0[1] * cov3d[1][0] + r0[2] * cov3d[2][0]
        m01 = r0[0] * cov3d[0][1] + r0[1] * cov3d[1][1] + r0[2] * cov3d[2][1]
        m02 = r0[0] * cov3d[0][2] + r0[1] * cov3d[1][2] + r0[2] * cov3d[2][2]

        m10 = r1[0] * cov3d[0][0] + r1[1] * cov3d[1][0] + r1[2] * cov3d[2][0]
        m11 = r1[0] * cov3d[0][1] + r1[1] * cov3d[1][1] + r1[2] * cov3d[2][1]
        m12 = r1[0] * cov3d[0][2] + r1[1] * cov3d[1][2] + r1[2] * cov3d[2][2]

        m20 = r2[0] * cov3d[0][0] + r2[1] * cov3d[1][0] + r2[2] * cov3d[2][0]
        m21 = r2[0] * cov3d[0][1] + r2[1] * cov3d[1][1] + r2[2] * cov3d[2][1]
        m22 = r2[0] * cov3d[0][2] + r2[1] * cov3d[1][2] + r2[2] * cov3d[2][2]

        # Sigma_cam = M * R_view^T
        c_cam00 = m00 * r0[0] + m01 * r0[1] + m02 * r0[2]
        c_cam01 = m00 * r1[0] + m01 * r1[1] + m02 * r1[2]
        c_cam02 = m00 * r2[0] + m01 * r2[1] + m02 * r2[2]

        c_cam11 = m10 * r1[0] + m11 * r1[1] + m12 * r1[2]
        c_cam12 = m10 * r2[0] + m11 * r2[1] + m12 * r2[2]

        c_cam22 = m20 * r2[0] + m21 * r2[1] + m22 * r2[2]

        # 4. Projective Jacobian J (2x3):
        # J_0 = [fx / tz, 0, -fx * tx / tz^2]
        # J_1 = [0, fy / tz, -fy * ty / tz^2]
        j00 = fx * inv_tz
        j02 = -fx * tx * inv_tz2

        j11 = fy * inv_tz
        j12 = -fy * ty * inv_tz2

        # 5. Sigma_2D = J * Sigma_cam * J^T
        # Row 0 of J * Sigma_cam:
        # [j00 * c_cam00 + j02 * c_cam02, j00 * c_cam01 + j02 * c_cam12, j00 * c_cam02 + j02 * c_cam22]
        p00 = j00 * c_cam00 + j02 * c_cam02
        p01 = j00 * c_cam01 + j02 * c_cam12
        p02 = j00 * c_cam02 + j02 * c_cam22

        # Row 1 of J * Sigma_cam:
        # [j11 * c_cam01 + j12 * c_cam02, j11 * c_cam11 + j12 * c_cam12, j11 * c_cam12 + j12 * c_cam22]
        p10 = j11 * c_cam01 + j12 * c_cam02
        p11 = j11 * c_cam11 + j12 * c_cam12
        p12 = j11 * c_cam12 + j12 * c_cam22

        cov_a = p00 * j00 + p02 * j02 + self.nu
        cov_b = p01 * j11 + p02 * j12
        cov_c = p10 * 0.0 + p11 * j11 + p12 * j12 + self.nu

        # 6. Inversion of 2x2 matrix: det = cov_a * cov_c - cov_b^2
        det = cov_a * cov_c - cov_b * cov_b
        if det < 1e-12:
            return None
        inv_det = 1.0 / det
        inv_cov_a = cov_c * inv_det
        inv_cov_b = -cov_b * inv_det
        inv_cov_c = cov_a * inv_det

        # 7. Screen-space 3-sigma bounding radius from eigenvalues
        mid = 0.5 * (cov_a + cov_c)
        diff = 0.5 * (cov_a - cov_c)
        disc = math.sqrt(max(0.0, diff * diff + cov_b * cov_b))
        lambda_max = mid + disc
        radius = math.ceil(3.0 * math.sqrt(max(1.0, lambda_max)))

        # Screen-space bounding box check
        if u + radius < 0 or u - radius >= camera.width or v + radius < 0 or v - radius >= camera.height:
            return None

        # 8. View-dependent color via spherical harmonics
        # Ray direction from camera to Gaussian center
        dir_cam = (-dx, -dy, -dz)
        color = eval_sh(sh_degree, gaussian.sh_coeffs, dir_cam)

        return ProjectedGaussian2D(
            gaussian_idx=gaussian_idx,
            u=u,
            v=v,
            depth=tz,
            cov_a=cov_a,
            cov_b=cov_b,
            cov_c=cov_c,
            inv_cov_a=inv_cov_a,
            inv_cov_b=inv_cov_b,
            inv_cov_c=inv_cov_c,
            radius=radius,
            color=color,
            opacity=gaussian.opacity,
        )

    def project_scene(
        self,
        gaussians: Sequence[Gaussian3D],
        camera: Camera,
        sh_degree: int = 0,
    ) -> List[ProjectedGaussian2D]:
        """Project an entire collection of 3D Gaussians into 2D screen space.

        Args:
            gaussians: Sequence of 3D Gaussians.
            camera: Perspective pinhole camera.
            sh_degree: Maximum spherical harmonics degree for directional color.

        Returns:
            List of projected 2D Gaussians sorted by depth tz ascending (front to back).
        """
        view_basis = camera.get_view_basis()
        projected: List[ProjectedGaussian2D] = []

        for idx, g in enumerate(gaussians):
            p = self.project_single(g, camera, view_basis, gaussian_idx=idx, sh_degree=sh_degree)
            if p is not None:
                projected.append(p)

        # Sort front-to-back by camera depth for volume accumulation
        projected.sort(key=lambda item: item.depth)
        return projected
