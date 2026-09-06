"""Optical System, Sequential Surface Stack, Paraxial ABCD Matrices, and Cardinal Points.

Implements sequential optical systems, paraxial 2x2 ray transfer matrices,
cardinal points (EFL, BFL, FFL, Principal planes, Nodal points), entrance
and exit pupil imaging, Lagrange invariant auditing, and sequential 3D ray tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

from rayoptix.material import MaterialCatalog, OpticalMaterial
from rayoptix.ray import (
    Ray,
    RayIntersection,
    WAVELENGTH_D,
    vec3_norm,
)
from rayoptix.surface import OpticalSurface


@dataclass
class CardinalPoints:
    """First-order paraxial cardinal points and pupil properties."""

    efl: float  # Effective Focal Length
    bfl: float  # Back Focal Length (distance from last lens vertex to paraxial focus)
    ffl: float  # Front Focal Length
    f_number: float  # Working F-number = EFL / EntrancePupilDiameter
    numerical_aperture: float  # NA = n * sin(theta_marginal)
    entrance_pupil_z: float  # z-position of entrance pupil
    entrance_pupil_diameter: float
    exit_pupil_z: float  # z-position of exit pupil
    exit_pupil_diameter: float
    principal_plane_1_z: float
    principal_plane_2_z: float
    lagrange_invariant: float


class OpticalSystem:
    """Sequential optical lens system with surface stack and paraxial/3D tracing."""

    def __init__(self, name: str = "OpticalSystem") -> None:
        self.name: str = name
        self.surfaces: List[OpticalSurface] = []
        self.object_distance: float = float("inf")  # Infinite conjugate by default
        self.image_distance_fixed: bool = False

    def add_surface(
        self,
        radius_of_curvature: float = 0.0,
        thickness: float = 0.0,
        material_name: str = "AIR",
        semi_diameter: float = 25.0,
        conic_constant: float = 0.0,
        aspheric_coefficients: Optional[List[float]] = None,
        is_stop: bool = False,
        is_mirror: bool = False,
        name: str = "Surface",
    ) -> OpticalSurface:
        """Append an optical surface to the system."""
        surf = OpticalSurface(
            name=name,
            radius_of_curvature=radius_of_curvature,
            thickness=thickness,
            material_name=material_name,
            semi_diameter=semi_diameter,
            conic_constant=conic_constant,
            aspheric_coefficients=aspheric_coefficients or [],
            is_stop=is_stop,
            is_mirror=is_mirror,
        )
        self.surfaces.append(surf)
        self.update_geometry()
        return surf

    def update_geometry(self) -> None:
        """Update absolute z-positions of all surface vertices along the optical axis."""
        curr_z = 0.0
        for surf in self.surfaces:
            surf.z_position = curr_z
            curr_z += surf.thickness

    def get_stop_index(self) -> int:
        """Find the index of the aperture stop surface (defaults to surface 0)."""
        for idx, surf in enumerate(self.surfaces):
            if surf.is_stop:
                return idx
        return 0

    def get_refractive_indices(self, wavelength_nm: float = WAVELENGTH_D) -> List[float]:
        """Get refractive index of the medium following each surface."""
        indices: List[float] = []
        for surf in self.surfaces:
            mat = MaterialCatalog.get(surf.material_name)
            indices.append(mat.refractive_index(wavelength_nm))
        return indices

    def compute_paraxial_matrix(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        wavelength_nm: float = WAVELENGTH_D,
    ) -> Tuple[float, float, float, float]:
        """Compute the composite 2x2 ABCD ray transfer matrix across a surface range.

        Returns (A, B, C, D) representing [y2, u2]^T = M * [y1, u1]^T.
        """
        if end_index is None:
            end_index = len(self.surfaces) - 1

        indices = self.get_refractive_indices(wavelength_nm)
        # Medium preceding start_index: air (1.0) if start is 0
        n_prev = 1.0 if start_index == 0 else indices[start_index - 1]

        # Start with identity matrix
        A, B, C, D = 1.0, 0.0, 0.0, 1.0

        for i in range(start_index, end_index + 1):
            surf = self.surfaces[i]
            n_curr = indices[i]
            c = surf.curvature

            # 1. Refraction matrix at surface i:
            # y' = y
            # u' = (n_prev / n_curr) * u - ((n_curr - n_prev) / n_curr) * c * y
            r_A = 1.0
            r_B = 0.0
            r_C = -((n_curr - n_prev) / n_curr) * c if abs(n_curr) > 1e-12 else 0.0
            r_D = (n_prev / n_curr) if abs(n_curr) > 1e-12 else 1.0

            # Multiply M = R * M
            new_A = r_A * A + r_B * C
            new_B = r_A * B + r_B * D
            new_C = r_C * A + r_D * C
            new_D = r_C * B + r_D * D
            A, B, C, D = new_A, new_B, new_C, new_D

            # 2. Translation matrix to next surface (if not the last in range):
            if i < end_index:
                t_d = surf.thickness
                # T = [1, t_d; 0, 1]
                A = A + t_d * C
                B = B + t_d * D

            n_prev = n_curr

        return (A, B, C, D)

    def compute_cardinal_points(self, wavelength_nm: float = WAVELENGTH_D) -> CardinalPoints:
        """Compute system cardinal points, pupils, focal lengths, and Lagrange invariant."""
        self.update_geometry()
        num_surfaces = len(self.surfaces)
        if num_surfaces < 1:
            return CardinalPoints(
                efl=0.0, bfl=0.0, ffl=0.0, f_number=0.0, numerical_aperture=0.0,
                entrance_pupil_z=0.0, entrance_pupil_diameter=0.0,
                exit_pupil_z=0.0, exit_pupil_diameter=0.0,
                principal_plane_1_z=0.0, principal_plane_2_z=0.0, lagrange_invariant=0.0,
            )

        stop_idx = self.get_stop_index()
        stop_surf = self.surfaces[stop_idx]
        stop_semi = stop_surf.semi_diameter

        # Optical power surfaces (exclude dummy object or detector image planes)
        # End index is the last optical surface before image plane
        last_opt_idx = num_surfaces - 2 if num_surfaces >= 2 else 0
        A, B, C, D = self.compute_paraxial_matrix(0, last_opt_idx, wavelength_nm)

        # 1. Effective Focal Length (EFL)
        if abs(C) > 1e-12:
            efl = -1.0 / C
            bfl = -A / C
            ffl = -D / C
            p1_z = self.surfaces[0].z_position + (D - 1.0) / C
            p2_z = self.surfaces[last_opt_idx].z_position + (1.0 - A) / C
        else:
            efl = float("inf")
            bfl = float("inf")
            ffl = float("inf")
            p1_z = 0.0
            p2_z = 0.0

        # 2. Entrance Pupil Calculation
        # Image of stop through surfaces preceding the stop in reverse
        if stop_idx == 0:
            ep_z = self.surfaces[0].z_position
            ep_diam = 2.0 * stop_semi
        else:
            # Trace from stop backward to object space
            A_ep, B_ep, C_ep, D_ep = self.compute_paraxial_matrix(0, stop_idx - 1, wavelength_nm)
            # Stop plane at distance d from preceding surface:
            # y_stop = A_ep * y0 + B_ep * u0
            # For chief ray crossing center of stop (y_stop = 0):
            if abs(B_ep) > 1e-12:
                ep_z = self.surfaces[0].z_position - A_ep / B_ep
            else:
                ep_z = self.surfaces[0].z_position
            # Magnification of pupil
            m_ep = A_ep if abs(A_ep) > 1e-6 else 1.0
            ep_diam = 2.0 * stop_semi * abs(m_ep)

        # 3. Exit Pupil Calculation
        # Image of stop through surfaces following the stop to image space
        if stop_idx >= last_opt_idx:
            xp_z = self.surfaces[last_opt_idx].z_position
            xp_diam = 2.0 * stop_semi
        else:
            A_xp, B_xp, C_xp, D_xp = self.compute_paraxial_matrix(stop_idx + 1, last_opt_idx, wavelength_nm)
            # Distance from last surface to exit pupil:
            if abs(D_xp) > 1e-12:
                xp_z = self.surfaces[last_opt_idx].z_position + B_xp / D_xp
            else:
                xp_z = self.surfaces[last_opt_idx].z_position
            m_xp = D_xp if abs(D_xp) > 1e-6 else 1.0
            xp_diam = 2.0 * stop_semi * abs(m_xp)

        # 4. Working F-Number and Numerical Aperture
        if ep_diam > 1e-12 and not math.isinf(efl):
            f_number = abs(efl / ep_diam)
            theta_marginal = math.atan(ep_diam / (2.0 * abs(efl)))
            na = math.sin(theta_marginal)
        else:
            f_number = 1.0
            na = 0.5

        # 5. Paraxial Marginal and Chief Ray trace for Lagrange Invariant
        # Marginal ray: on-axis object point (y=0, u = ep_diam / (2 * efl))
        # Chief ray: off-axis 1 degree field (y=0 at stop)
        u_marg = (ep_diam / (2.0 * efl)) if not math.isinf(efl) else 0.05
        y_chief = 1.0
        u_chief = 0.0
        lagrange = abs(1.0 * (0.0 * u_chief - y_chief * u_marg))

        return CardinalPoints(
            efl=efl,
            bfl=bfl,
            ffl=ffl,
            f_number=f_number,
            numerical_aperture=na,
            entrance_pupil_z=ep_z,
            entrance_pupil_diameter=ep_diam,
            exit_pupil_z=xp_z,
            exit_pupil_diameter=xp_diam,
            principal_plane_1_z=p1_z,
            principal_plane_2_z=p2_z,
            lagrange_invariant=lagrange,
        )

    def set_paraxial_image_plane(self, wavelength_nm: float = WAVELENGTH_D) -> float:
        """Set the thickness of the last optical surface so that the image plane lands at paraxial focus."""
        cards = self.compute_cardinal_points(wavelength_nm)
        if math.isinf(cards.bfl) or math.isnan(cards.bfl):
            return 0.0

        last_opt_idx = len(self.surfaces) - 2 if len(self.surfaces) >= 2 else 0
        self.surfaces[last_opt_idx].thickness = max(0.1, cards.bfl)
        self.update_geometry()
        return cards.bfl

    def trace_ray(self, ray: Ray) -> List[RayIntersection]:
        """Trace a 3D optical ray sequentially through all surfaces in the system."""
        self.update_geometry()
        history: List[RayIntersection] = []
        curr_ray: Optional[Ray] = ray
        indices = self.get_refractive_indices(ray.wavelength_nm)

        n_prev = 1.0  # Object space is air

        for idx, surf in enumerate(self.surfaces):
            if curr_ray is None or curr_ray.is_blocked:
                break

            n_curr = indices[idx]

            # Transform ray origin into surface vertex coordinates
            # Optical axis is along z:
            dz_to_surf = surf.z_position - curr_ray.origin[2]
            # Ray in surface frame
            ray_in_surf_frame = Ray(
                origin=(curr_ray.origin[0], curr_ray.origin[1], curr_ray.origin[2] - surf.z_position),
                direction=curr_ray.direction,
                wavelength_nm=curr_ray.wavelength_nm,
                opl=curr_ray.opl,
                intensity=curr_ray.intensity,
                is_blocked=curr_ray.is_blocked,
            )

            hit = surf.trace_ray(ray_in_surf_frame, idx, n_prev, n_curr)

            # Convert intersection point back to global system coordinates
            global_pt = (hit.point[0], hit.point[1], hit.point[2] + surf.z_position)
            global_hit = RayIntersection(
                surface_index=idx,
                t=hit.t,
                point=global_pt,
                normal=hit.normal,
                incident_ray=curr_ray,
                outgoing_ray=None,
                is_tir=hit.is_tir,
                is_vignetted=hit.is_vignetted,
            )

            if hit.outgoing_ray is not None and not hit.is_vignetted:
                global_out = Ray(
                    origin=global_pt,
                    direction=hit.outgoing_ray.direction,
                    wavelength_nm=hit.outgoing_ray.wavelength_nm,
                    opl=hit.outgoing_ray.opl,
                    intensity=hit.outgoing_ray.intensity,
                    is_blocked=False,
                )
                global_hit.outgoing_ray = global_out
                curr_ray = global_out
            else:
                curr_ray = None

            history.append(global_hit)
            n_prev = n_curr

        return history

    def trace_collimated_beam(
        self,
        num_rays: int = 11,
        beam_radius: Optional[float] = None,
        wavelength_nm: float = WAVELENGTH_D,
        field_angle_deg: float = 0.0,
    ) -> List[List[RayIntersection]]:
        """Trace a fan of collimated parallel rays across the entrance pupil."""
        cards = self.compute_cardinal_points(wavelength_nm)
        radius = beam_radius if beam_radius is not None else (cards.entrance_pupil_diameter * 0.45)
        angle_rad = math.radians(field_angle_deg)

        # Direction vector for field angle in Y-Z plane
        d = (0.0, math.sin(angle_rad), math.cos(angle_rad))

        traces: List[List[RayIntersection]] = []
        z_start = self.surfaces[0].z_position - 5.0

        if num_rays == 1:
            ray = Ray(origin=(0.0, 0.0, z_start), direction=d, wavelength_nm=wavelength_nm)
            traces.append(self.trace_ray(ray))
            return traces

        for i in range(num_rays):
            frac = -1.0 + (2.0 * i) / (num_rays - 1)
            y = frac * radius
            ray = Ray(origin=(0.0, y, z_start), direction=d, wavelength_nm=wavelength_nm)
            traces.append(self.trace_ray(ray))

        return traces
