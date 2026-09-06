"""Curved Spacetime Backward Ray Tracer and Camera Engine.

Simulates gravitational lensing, black hole shadows, and warped accretion disks:
- Observer camera positioned at arbitrary Boyer-Lindquist coordinates (r_cam, theta_cam, phi_cam)
- Orthonormal tetrad projection converting screen pixels to exact null 4-momenta (g_uv p^u p^v = 0)
- Backward ray tracing through curved spacetime via the adaptive RK4 geodesic integrator
- Multi-component scene rendering: black hole shadow, equatorial accretion disk, and celestial sphere
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

from relativitas.metric import SpacetimeMetric
from relativitas.geodesic import GeodesicIntegrator, GeodesicState, GeodesicStatus
from relativitas.accretion import AccretionDisk, RadiativeTransferResult


class PixelClass(Enum):
    """Classification of a rendered screen pixel."""
    BLACK_HOLE_SHADOW = "BLACK_HOLE_SHADOW"
    ACCRETION_DISK = "ACCRETION_DISK"
    CELESTIAL_BACKGROUND = "CELESTIAL_BACKGROUND"


@dataclass
class PixelResult:
    """Telemetry and photometric data for a single screen ray."""
    screen_x: int
    screen_y: int
    u: float
    v: float
    pixel_class: PixelClass
    rgb: Tuple[int, int, int]
    redshift_g: float
    steps: int
    final_r: float
    disk_result: Optional[RadiativeTransferResult] = None


@dataclass
class FrameBuffer:
    """2D rasterized image buffer and physical telemetry grid."""
    width: int
    height: int
    pixels: List[List[PixelResult]]
    total_rays: int
    shadow_rays: int
    disk_rays: int
    escaped_rays: int

    def get_rgb_grid(self) -> List[List[Tuple[int, int, int]]]:
        """Extract 2D array of (R, G, B) tuples."""
        return [[p.rgb for p in row] for row in self.pixels]

    def get_redshift_grid(self) -> List[List[float]]:
        """Extract 2D array of redshift factors g."""
        return [[p.redshift_g for p in row] for row in self.pixels]


class Camera:
    """Virtual pinhole observer camera situated in curved spacetime."""

    def __init__(
        self,
        r: float = 20.0,
        theta: float = 1.396,  # 80 degrees inclination (near edge-on)
        phi: float = 0.0,
        fov_degrees: float = 50.0,
    ) -> None:
        self.r = r
        self.theta = theta
        self.phi = phi
        self.fov_degrees = fov_degrees
        self.fov_rad = math.radians(fov_degrees)

    @property
    def coords(self) -> Tuple[float, float, float, float]:
        """Camera 4-position x^u = (t=0, r, theta, phi)."""
        return (0.0, self.r, self.theta, self.phi)


class RayTracer:
    """Curved-spacetime backward ray tracer."""

    def __init__(
        self,
        metric: SpacetimeMetric,
        disk: Optional[AccretionDisk] = None,
        max_steps: int = 400,
        horizon_buffer: float = 0.04,
        escape_radius: float = 35.0,
    ) -> None:
        self.metric = metric
        self.disk = disk if disk is not None else AccretionDisk(metric)
        self.integrator = GeodesicIntegrator(
            metric=metric,
            is_null=True,
            horizon_buffer=horizon_buffer,
            escape_radius=escape_radius,
            max_steps=max_steps,
            base_step_size=0.12,
        )

    def screen_ray_direction(
        self,
        u: float,
        v: float,
        camera: Camera,
    ) -> Tuple[float, float, float, float]:
        """Compute initial null 4-momentum p^u at the camera position for screen coords (u, v).

        Normalized screen coordinates: u in [-1, 1] (horizontal), v in [-1, 1] (vertical).
        Uses asymptotic locally flat Minkowski tetrad aligned with spherical coordinates:
        e_(r) points radially outward => camera ray points inward (-dr)
        e_(phi) points in direction of increasing phi => horizontal (+dphi)
        e_(theta) points south => vertical (+dtheta)
        """
        tan_half_fov = math.tan(0.5 * camera.fov_rad)

        # Direction in local orthonormal camera frame:
        # z: pointing towards black hole (-r direction)
        # x: pointing right (+phi direction)
        # y: pointing up (-theta direction)
        nx = u * tan_half_fov
        ny = -v * tan_half_fov
        nz = -1.0  # Pointing inward towards the black hole

        # Spatial normalization: |n| = 1
        n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= n_len
        ny /= n_len
        nz /= n_len

        # Convert to Boyer-Lindquist coordinate momentum components:
        # dr = nz / sqrt(g_rr)
        # dtheta = ny / sqrt(g_th_th)
        # dphi = nx / (sqrt(g_phi_phi))
        x_cam = camera.coords
        g = self.metric.metric_tensor(x_cam)

        pr = nz / math.sqrt(max(1e-9, g[1][1]))
        ptheta = ny / math.sqrt(max(1e-9, g[2][2]))
        pphi = nx / math.sqrt(max(1e-9, g[3][3]))

        # Determine pt from the null condition: g_uv p^u p^v = 0
        # g_tt (pt)^2 + 2 g_t_phi pt pphi + (g_rr pr^2 + g_th_th ptheta^2 + g_phi_phi pphi^2) = 0
        # A (pt)^2 + B pt + C = 0
        a = g[0][0]
        b = 2.0 * g[0][3] * pphi
        c = g[1][1] * pr * pr + g[2][2] * ptheta * ptheta + g[3][3] * pphi * pphi

        discriminant = max(0.0, b * b - 4.0 * a * c)
        # Since g_tt < 0, a < 0. To get positive pt (future-directed):
        # pt = (-b - sqrt(discriminant)) / (2*a)
        pt = (-b - math.sqrt(discriminant)) / (2.0 * a)

        return (pt, pr, ptheta, pphi)

    def trace_single_ray(
        self,
        u: float,
        v: float,
        camera: Camera,
        screen_x: int = 0,
        screen_y: int = 0,
    ) -> PixelResult:
        """Trace a single light ray backward from the screen into the black hole system."""
        p_init = self.screen_ray_direction(u, v, camera)
        initial_state = GeodesicState(
            t=0.0,
            r=camera.r,
            theta=camera.theta,
            phi=camera.phi,
            pt=p_init[0],
            pr=p_init[1],
            ptheta=p_init[2],
            pphi=p_init[3],
        )

        # Integrate backward through spacetime
        res = self.integrator.integrate(
            initial_state,
            backward=False,
            stop_on_disk=True,
            disk_r_in=self.disk.r_in,
            disk_r_out=self.disk.r_out,
        )

        if res.status == GeodesicStatus.HORIZON_CAPTURED:
            return PixelResult(
                screen_x=screen_x,
                screen_y=screen_y,
                u=u,
                v=v,
                pixel_class=PixelClass.BLACK_HOLE_SHADOW,
                rgb=(0, 0, 0),
                redshift_g=0.0,
                steps=res.steps,
                final_r=res.final_state.r,
            )

        elif res.status == GeodesicStatus.DISK_INTERSECTED and res.disk_intersections:
            crossing = res.disk_intersections[0]
            rad_res = self.disk.evaluate_radiative_transfer(
                r=crossing.r_cross,
                phi=crossing.phi_cross,
                p_contra=crossing.state.momentum,
            )
            return PixelResult(
                screen_x=screen_x,
                screen_y=screen_y,
                u=u,
                v=v,
                pixel_class=PixelClass.ACCRETION_DISK,
                rgb=rad_res.rgb,
                redshift_g=rad_res.redshift_factor_g,
                steps=res.steps,
                final_r=crossing.r_cross,
                disk_result=rad_res,
            )

        else:
            # Ray escaped to celestial sphere: render subtle background star field or cosmic glow
            phi_esc = res.final_state.phi
            th_esc = res.final_state.theta

            # Star field / cosmic grid simulation based on escape angles
            grid_val = (math.sin(10.0 * phi_esc) * math.sin(10.0 * th_esc))**12
            if grid_val > 0.85:
                rgb = (180, 200, 255)  # Distant background star
            else:
                rgb = (8, 8, 16)  # Deep cosmic void

            return PixelResult(
                screen_x=screen_x,
                screen_y=screen_y,
                u=u,
                v=v,
                pixel_class=PixelClass.CELESTIAL_BACKGROUND,
                rgb=rgb,
                redshift_g=1.0,
                steps=res.steps,
                final_r=res.final_state.r,
            )

    def render_frame(
        self,
        camera: Camera,
        width: int = 80,
        height: int = 40,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> FrameBuffer:
        """Render a complete 2D image grid by tracing rays across the screen plane."""
        pixels: List[List[PixelResult]] = []
        shadow_count = 0
        disk_count = 0
        escape_count = 0

        aspect_ratio = width / height

        for y in range(height):
            row: List[PixelResult] = []
            # Map y in [0, height-1] to v in [-1, 1]
            v = 1.0 - 2.0 * (y + 0.5) / height

            for x in range(width):
                # Map x in [0, width-1] to u in [-1, 1], corrected for aspect ratio
                u = (2.0 * (x + 0.5) / width - 1.0) * aspect_ratio

                p_res = self.trace_single_ray(u, v, camera, screen_x=x, screen_y=y)
                row.append(p_res)

                if p_res.pixel_class == PixelClass.BLACK_HOLE_SHADOW:
                    shadow_count += 1
                elif p_res.pixel_class == PixelClass.ACCRETION_DISK:
                    disk_count += 1
                else:
                    escape_count += 1

            pixels.append(row)
            if progress_callback is not None:
                progress_callback(y + 1, height)

        total_rays = width * height
        return FrameBuffer(
            width=width,
            height=height,
            pixels=pixels,
            total_rays=total_rays,
            shadow_rays=shadow_count,
            disk_rays=disk_count,
            escaped_rays=escape_count,
        )
