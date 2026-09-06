"""3D Vector Ray Tracing and Snell Law of Refraction and Reflection.

Provides exact vector formulations for optical rays, 3D refraction with
total internal reflection (TIR) detection, mirror reflection, and Fresnel
power transmission and reflectance coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

# Standard Fraunhofer spectral wavelengths in nanometers
WAVELENGTH_D: float = 587.5618  # Helium yellow d-line (primary reference)
WAVELENGTH_F: float = 486.1327  # Hydrogen blue F-line
WAVELENGTH_C: float = 656.2725  # Hydrogen red C-line
WAVELENGTH_E: float = 546.0740  # Mercury green e-line
WAVELENGTH_G: float = 435.8343  # Mercury violet g-line


def vec3_norm(v: Tuple[float, float, float]) -> float:
    """Compute Euclidean norm of a 3D vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length."""
    mag = vec3_norm(v)
    if mag < 1e-15:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / mag
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def vec3_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Compute scalar dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Compute vector cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vec3_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Vector addition."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Vector subtraction."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_scale(v: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
    """Scalar vector multiplication."""
    return (v[0] * s, v[1] * s, v[2] * s)


@dataclass(slots=True)
class Ray:
    """3D optical ray with position, direction, wavelength, and path length."""

    origin: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    wavelength_nm: float = WAVELENGTH_D
    opl: float = 0.0
    intensity: float = 1.0
    is_blocked: bool = False

    def __post_init__(self) -> None:
        """Ensure direction vector is strictly normalized to unit length."""
        mag = vec3_norm(self.direction)
        if abs(mag - 1.0) > 1e-12:
            self.direction = vec3_normalize(self.direction)

    def point_at(self, distance: float) -> Tuple[float, float, float]:
        """Compute Cartesian coordinates at parameter distance t along the ray."""
        return (
            self.origin[0] + self.direction[0] * distance,
            self.origin[1] + self.direction[1] * distance,
            self.origin[2] + self.direction[2] * distance,
        )

    def advance(self, distance: float, medium_refractive_index: float) -> Ray:
        """Return a new Ray advanced by distance through a medium of given refractive index."""
        new_origin = self.point_at(distance)
        delta_opl = distance * abs(medium_refractive_index)
        return Ray(
            origin=new_origin,
            direction=self.direction,
            wavelength_nm=self.wavelength_nm,
            opl=self.opl + delta_opl,
            intensity=self.intensity,
            is_blocked=self.is_blocked,
        )


def refract_vector(
    d_incident: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    n1: float,
    n2: float,
) -> Optional[Tuple[float, float, float]]:
    """Compute refracted unit direction vector using exact 3D vector Snell law.

    Args:
        d_incident: Unit direction vector of incident ray.
        normal: Unit surface normal pointing toward incident medium (cos_theta1 > 0).
        n1: Refractive index of incident medium.
        n2: Refractive index of transmitted medium.

    Returns:
        Refracted unit direction vector, or None if total internal reflection (TIR) occurs.
    """
    if abs(n2) < 1e-12:
        return None

    # Handle mirror reflection when n2 has opposite sign of n1
    if (n1 > 0.0 and n2 < 0.0) or (n1 < 0.0 and n2 > 0.0):
        return reflect_vector(d_incident, normal)

    mu = n1 / n2
    cos_theta1 = -vec3_dot(d_incident, normal)

    # If ray strikes surface from the backside, flip normal so cos_theta1 > 0
    actual_normal = normal
    if cos_theta1 < 0.0:
        cos_theta1 = -cos_theta1
        actual_normal = vec3_scale(normal, -1.0)

    sin2_theta2 = mu * mu * (1.0 - cos_theta1 * cos_theta1)
    if sin2_theta2 > 1.0:
        # Total internal reflection
        return None

    cos_theta2 = math.sqrt(max(0.0, 1.0 - sin2_theta2))
    factor = mu * cos_theta1 - cos_theta2

    d_refracted = (
        mu * d_incident[0] + factor * actual_normal[0],
        mu * d_incident[1] + factor * actual_normal[1],
        mu * d_incident[2] + factor * actual_normal[2],
    )
    return vec3_normalize(d_refracted)


def reflect_vector(
    d_incident: Tuple[float, float, float],
    normal: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Compute reflected unit direction vector from a specular surface.

    Args:
        d_incident: Unit direction vector of incident ray.
        normal: Unit surface normal.

    Returns:
        Specularly reflected unit direction vector.
    """
    cos_theta1 = -vec3_dot(d_incident, normal)
    actual_normal = normal
    if cos_theta1 < 0.0:
        cos_theta1 = -cos_theta1
        actual_normal = vec3_scale(normal, -1.0)

    d_reflected = (
        d_incident[0] + 2.0 * cos_theta1 * actual_normal[0],
        d_incident[1] + 2.0 * cos_theta1 * actual_normal[1],
        d_incident[2] + 2.0 * cos_theta1 * actual_normal[2],
    )
    return vec3_normalize(d_reflected)


def fresnel_coefficients(
    d_incident: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    n1: float,
    n2: float,
) -> Tuple[float, float]:
    """Compute unpolarized Fresnel power reflection and transmission coefficients.

    Args:
        d_incident: Unit direction vector of incident ray.
        normal: Unit surface normal.
        n1: Incident medium refractive index.
        n2: Transmitted medium refractive index.

    Returns:
        Tuple (reflectance R, transmittance T) where R + T = 1.0.
    """
    if abs(n2) < 1e-12 or (n1 > 0.0 and n2 < 0.0) or (n1 < 0.0 and n2 > 0.0):
        # Ideal mirror
        return (1.0, 0.0)

    cos_theta1 = abs(vec3_dot(d_incident, normal))
    mu = n1 / n2
    sin2_theta2 = mu * mu * max(0.0, 1.0 - cos_theta1 * cos_theta1)

    if sin2_theta2 >= 1.0:
        # Total internal reflection
        return (1.0, 0.0)

    cos_theta2 = math.sqrt(max(0.0, 1.0 - sin2_theta2))

    # Perpendicular (s-polarization) and parallel (p-polarization) amplitude coefficients
    denom_s = n1 * cos_theta1 + n2 * cos_theta2
    r_s = (n1 * cos_theta1 - n2 * cos_theta2) / max(1e-15, denom_s)

    denom_p = n2 * cos_theta1 + n1 * cos_theta2
    r_p = (n2 * cos_theta1 - n1 * cos_theta2) / max(1e-15, denom_p)

    r_power = 0.5 * (r_s * r_s + r_p * r_p)
    r_power = min(1.0, max(0.0, r_power))
    t_power = 1.0 - r_power
    return (r_power, t_power)


@dataclass(slots=True)
class RayIntersection:
    """Record of an optical ray intersection with an optical surface."""

    surface_index: int
    t: float
    point: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    incident_ray: Ray
    outgoing_ray: Optional[Ray] = None
    is_tir: bool = False
    is_vignetted: bool = False
