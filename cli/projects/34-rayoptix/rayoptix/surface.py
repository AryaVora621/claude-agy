"""Optical Surfaces: Spherical, Conic, and Even Aspheric Geometries.

Implements exact analytical ray-surface intersections for conic sections,
high-precision Newton-Raphson solvers for high-order even aspheres,
exact surface normal gradients, and aperture vignetting auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

from rayoptix.ray import (
    Ray,
    RayIntersection,
    fresnel_coefficients,
    reflect_vector,
    refract_vector,
    vec3_dot,
    vec3_norm,
    vec3_normalize,
    vec3_scale,
)


@dataclass
class OpticalSurface:
    """Optical interface with geometry, thickness, material, and aperture bounds."""

    name: str = "Surface"
    radius_of_curvature: float = 0.0  # 0.0 represents flat/plano (R = infinity)
    conic_constant: float = 0.0  # k: 0=sphere, -1=parabola, <-1=hyperbola, >0=oblate
    aspheric_coefficients: List[float] = field(default_factory=list)  # [alpha2, alpha4, alpha6, ...]
    thickness: float = 0.0  # Distance along z-axis to next surface vertex
    material_name: str = "AIR"  # Medium following this surface
    semi_diameter: float = 25.0  # Maximum clear aperture radius (vignetting limit)
    inner_diameter: float = 0.0  # Optional central obscuration radius
    is_stop: bool = False  # Is this surface the system aperture stop?
    is_mirror: bool = False  # Is this surface a specular reflective mirror?
    z_position: float = 0.0  # Absolute z-coordinate of surface vertex in global system

    @property
    def curvature(self) -> float:
        """Vertex curvature c = 1 / R (0.0 for plano)."""
        if abs(self.radius_of_curvature) < 1e-12:
            return 0.0
        return 1.0 / self.radius_of_curvature

    def sag(self, x: float, y: float) -> float:
        """Compute surface sagitta z(x, y) relative to surface vertex at (0, 0, 0)."""
        r2 = x * x + y * y
        c = self.curvature
        k = self.conic_constant

        # 1. Conic base sag
        if abs(c) < 1e-12:
            z_conic = 0.0
        else:
            denom_arg = 1.0 - (1.0 + k) * c * c * r2
            if denom_arg < 0.0:
                # Outside physical surface bounds
                return float("nan")
            z_conic = (c * r2) / (1.0 + math.sqrt(denom_arg))

        # 2. Even polynomial aspheric deformation: sum(alpha_i * r^(2*(i+1)))
        z_asph = 0.0
        if self.aspheric_coefficients:
            r_pow = r2  # r^2
            for alpha in self.aspheric_coefficients:
                r_pow *= r2  # r^4, r^6, r^8...
                z_asph += alpha * r_pow

        return z_conic + z_asph

    def normal_and_sag(self, x: float, y: float) -> Tuple[Tuple[float, float, float], float]:
        """Compute outward unit normal vector and sagitta at (x, y).

        The normal vector points toward the incident medium (negative z direction
        for a standard forward-facing surface).
        """
        r2 = x * x + y * y
        r = math.sqrt(r2)
        c = self.curvature
        k = self.conic_constant

        if abs(c) < 1e-12 and not self.aspheric_coefficients:
            return ((0.0, 0.0, -1.0), 0.0)

        # dz_conic / dr
        if abs(c) < 1e-12:
            z_conic = 0.0
            dz_conic_dr = 0.0
        else:
            denom_arg = 1.0 - (1.0 + k) * c * c * r2
            if denom_arg <= 0.0:
                return ((0.0, 0.0, -1.0), float("nan"))
            sqrt_denom = math.sqrt(denom_arg)
            z_conic = (c * r2) / (1.0 + sqrt_denom)
            dz_conic_dr = (c * r) / sqrt_denom

        # dz_asph / dr
        z_asph = 0.0
        dz_asph_dr = 0.0
        if self.aspheric_coefficients:
            r_pow = r2
            for i, alpha in enumerate(self.aspheric_coefficients):
                order = 2 * (i + 2)  # 4, 6, 8...
                r_pow *= r2
                z_asph += alpha * r_pow
                dz_asph_dr += order * alpha * (r ** (order - 1))

        dz_dr = dz_conic_dr + dz_asph_dr

        if r < 1e-12:
            dz_dx = 0.0
            dz_dy = 0.0
        else:
            inv_r = 1.0 / r
            dz_dx = dz_dr * x * inv_r
            dz_dy = dz_dr * y * inv_r

        # Normal gradient: (-dz/dx, -dz/dy, 1), normalized and oriented towards -z
        mag = math.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
        inv_mag = 1.0 / mag
        normal = (dz_dx * inv_mag, dz_dy * inv_mag, -inv_mag)
        return (normal, z_conic + z_asph)

    def intersect(self, ray: Ray) -> Optional[Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]]:
        """Find ray intersection parameter t, 3D point, and surface normal.

        Coordinates are evaluated relative to surface vertex (0, 0, 0).
        Returns None if ray misses surface or intersects behind its origin.
        """
        ox, oy, oz = ray.origin
        dx, dy, dz = ray.direction
        c = self.curvature
        k = self.conic_constant

        # Case 1: Plano surface (flat mirror, stop, image plane)
        if abs(c) < 1e-12 and not self.aspheric_coefficients:
            if abs(dz) < 1e-14:
                return None
            t = -oz / dz
            if t <= 1e-9:
                return None
            px = ox + t * dx
            py = oy + t * dy
            pz = 0.0
            normal = (0.0, 0.0, -1.0 if dz > 0.0 else 1.0)
            return (t, (px, py, pz), normal)

        # Case 2: Conic / Spherical surface (analytical quadratic formula)
        # Equation: c (x^2 + y^2) - 2 z + (1+k) c z^2 = 0
        # Substitute (ox + t dx, oy + t dy, oz + t dz):
        d_perp_sq = dx * dx + dy * dy
        o_perp_sq = ox * ox + oy * oy
        o_dot_d_perp = ox * dx + oy * dy

        quad_a = c * (d_perp_sq + (1.0 + k) * dz * dz)
        quad_b = 2.0 * (c * (o_dot_d_perp + (1.0 + k) * oz * dz) - dz)
        quad_c = c * (o_perp_sq + (1.0 + k) * oz * oz) - 2.0 * oz

        t_guess: Optional[float] = None

        if abs(quad_a) < 1e-14:
            if abs(quad_b) > 1e-14:
                t_lin = -quad_c / quad_b
                if t_lin > 1e-9:
                    t_guess = t_lin
        else:
            discriminant = quad_b * quad_b - 4.0 * quad_a * quad_c
            if discriminant >= 0.0:
                sqrt_disc = math.sqrt(discriminant)
                # Standard numerically stable quadratic root finding
                t1 = (-quad_b - sqrt_disc) / (2.0 * quad_a)
                t2 = (-quad_b + sqrt_disc) / (2.0 * quad_a)

                # Pick smallest positive root
                valid_roots = [t for t in (t1, t2) if t > 1e-9]
                if valid_roots:
                    t_guess = min(valid_roots)

        if t_guess is None:
            return None

        # Case 3: Pure Conic surface without aspheric polynomials
        if not self.aspheric_coefficients:
            t = t_guess
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz
            normal, _ = self.normal_and_sag(px, py)
            # Ensure normal opposes ray direction: dot(d, normal) < 0
            if vec3_dot(ray.direction, normal) > 0.0:
                normal = vec3_scale(normal, -1.0)
            return (t, (px, py, pz), normal)

        # Case 4: High-Order Even Aspheric surface (Newton-Raphson refinement)
        t = t_guess
        max_iter = 12
        tol = 1e-12

        for _ in range(max_iter):
            px = ox + t * dx
            py = oy + t * dy
            pz = oz + t * dz

            z_surf = self.sag(px, py)
            if math.isnan(z_surf):
                return None

            f_val = pz - z_surf
            if abs(f_val) < tol:
                break

            # Derivative of f(t) = (oz + t dz) - z_surf(px, py)
            normal, _ = self.normal_and_sag(px, py)
            # Normal is proportional to (dz/dx, dz/dy, -1)
            # df/dt = dz - (dz/dx * dx + dz/dy * dy)
            # Using dot product with normal:
            # normal = (nx, ny, nz) with nz < 0
            df_dt = dz - (-normal[0] / normal[2] * dx + -normal[1] / normal[2] * dy)
            if abs(df_dt) < 1e-14:
                break

            delta_t = f_val / df_dt
            t -= delta_t

            if t <= 1e-9:
                return None

        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        normal, _ = self.normal_and_sag(px, py)
        if vec3_dot(ray.direction, normal) > 0.0:
            normal = vec3_scale(normal, -1.0)

        return (t, (px, py, pz), normal)

    def trace_ray(
        self,
        ray: Ray,
        surface_index: int,
        n_incident: float,
        n_transmitted: float,
    ) -> RayIntersection:
        """Trace an optical ray through this surface and compute refraction/reflection."""
        hit = self.intersect(ray)

        if hit is None:
            # Ray completely missed surface geometry
            return RayIntersection(
                surface_index=surface_index,
                t=0.0,
                point=ray.origin,
                normal=(0.0, 0.0, -1.0),
                incident_ray=ray,
                outgoing_ray=None,
                is_vignetted=True,
            )

        t, point, normal = hit
        r_hit = math.sqrt(point[0] * point[0] + point[1] * point[1])

        # Vignetting / Clear aperture check
        is_vignetted = (r_hit > self.semi_diameter) or (r_hit < self.inner_diameter)

        # Refraction or Reflection
        if self.is_mirror:
            d_out = reflect_vector(ray.direction, normal)
            is_tir = False
            r_pow, t_pow = (1.0, 0.0)
        else:
            d_out = refract_vector(ray.direction, normal, n_incident, n_transmitted)
            is_tir = d_out is None
            if is_tir:
                r_pow, t_pow = (1.0, 0.0)
                d_out = reflect_vector(ray.direction, normal)
            else:
                r_pow, t_pow = fresnel_coefficients(ray.direction, normal, n_incident, n_transmitted)

        outgoing_ray: Optional[Ray] = None
        if d_out is not None and not is_vignetted:
            delta_opl = t * abs(n_incident)
            outgoing_ray = Ray(
                origin=point,
                direction=d_out,
                wavelength_nm=ray.wavelength_nm,
                opl=ray.opl + delta_opl,
                intensity=ray.intensity * (t_pow if not self.is_mirror and not is_tir else r_pow),
                is_blocked=False,
            )

        return RayIntersection(
            surface_index=surface_index,
            t=t,
            point=point,
            normal=normal,
            incident_ray=ray,
            outgoing_ray=outgoing_ray,
            is_tir=is_tir,
            is_vignetted=is_vignetted,
        )
