"""
OrbitMech: Fundamental Astrodynamics Data Types & 3D Vector Mathematics.
Zero external dependencies. Pure standard library implementation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union


class OrbitType(Enum):
    """Classification of orbital conic trajectory."""
    CIRCULAR = "circular"
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"


@dataclass(frozen=True)
class Vector3:
    """
    Immutable 3-dimensional Euclidean vector in Cartesian space.
    Provides vector arithmetic, dot products, cross products, and norms.
    """
    x: float
    y: float
    z: float

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Union[int, float]) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Union[int, float]) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]) -> Vector3:
        if scalar == 0.0:
            raise ZeroDivisionError("Vector3 cannot be divided by zero scalar")
        inv = 1.0 / scalar
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, other: Vector3) -> float:
        """Euclidean inner product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Right-handed 3D vector cross product."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_squared(self) -> float:
        """Squared L2 norm."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        """Euclidean L2 length."""
        return math.sqrt(self.norm_squared())

    def normalized(self) -> Vector3:
        """Unit vector in direction of self."""
        mag = self.norm()
        if mag < 1e-15:
            return Vector3(0.0, 0.0, 0.0)
        return self / mag

    def distance_to(self, other: Vector3) -> float:
        """Euclidean distance between two position vectors."""
        return (self - other).norm()

    def angle_to(self, other: Vector3) -> float:
        """Angular separation in radians between two vectors in [0, pi]."""
        n1 = self.norm()
        n2 = other.norm()
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0
        cos_theta = max(-1.0, min(1.0, self.dot(other) / (n1 * n2)))
        return math.acos(cos_theta)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def zero(cls) -> Vector3:
        return cls(0.0, 0.0, 0.0)

    def __repr__(self) -> str:
        return f"Vector3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


@dataclass(frozen=True)
class StateVector:
    """
    Instantaneous kinematic state vector: position (km) and velocity (km/s).
    """
    r: Vector3
    v: Vector3
    time: float = 0.0  # Seconds from epoch

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector in km/s."""
        return self.v.norm()

    @property
    def radius(self) -> float:
        """Magnitude of position vector in km."""
        return self.r.norm()


@dataclass(frozen=True)
class CelestialBody:
    """
    Physical and gravitational characteristics of a central attracting body.
    All gravitational units are in km and seconds.
    """
    name: str
    mu: float           # Gravitational parameter (km^3 / s^2)
    radius: float       # Mean equatorial radius (km)
    j2: float = 0.0     # Second zonal harmonic coefficient (oblateness)
    mass: float = 0.0   # Mass in kg
    soi_radius: float = float("inf")  # Sphere of Influence radius (km)


# Standard Astrodynamical Planetary Constants (NASA JPL Horizons values)
EARTH = CelestialBody(
    name="Earth",
    mu=398600.4418,          # km^3 / s^2
    radius=6378.137,         # km
    j2=1.08263e-3,           # Oblateness coefficient
    mass=5.9722e24,          # kg
    soi_radius=925000.0,     # km (~0.925M km)
)

SUN = CelestialBody(
    name="Sun",
    mu=132712440018.0,       # km^3 / s^2
    radius=696340.0,         # km
    j2=2.0e-7,
    mass=1.9885e30,          # kg
)

MOON = CelestialBody(
    name="Moon",
    mu=4902.8000,            # km^3 / s^2
    radius=1737.4,           # km
    j2=2.027e-4,
    mass=7.342e22,           # kg
    soi_radius=66100.0,      # km
)

MARS = CelestialBody(
    name="Mars",
    mu=42828.3752,           # km^3 / s^2
    radius=3396.2,           # km
    j2=1.96045e-3,
    mass=6.4171e23,          # kg
    soi_radius=577000.0,     # km
)

VENUS = CelestialBody(
    name="Venus",
    mu=324859.0,             # km^3 / s^2
    radius=6051.8,           # km
    j2=4.458e-6,
    mass=4.8675e24,          # kg
    soi_radius=616000.0,     # km
)

JUPITER = CelestialBody(
    name="Jupiter",
    mu=126686534.0,          # km^3 / s^2
    radius=71492.0,          # km
    j2=1.4736e-2,
    mass=1.8982e27,          # kg
    soi_radius=48200000.0,   # km
)

# Standard Astronomical Unit in km
AU_KM = 149597870.7

# Speed of light in km/s
SPEED_OF_LIGHT_KMS = 299792.458


@dataclass
class ClassicalOrbitalElements:
    """
    Keplerian classical orbital elements describing a conic section in 3D:
      a: Semi-major axis (km) (negative for hyperbolas)
      e: Eccentricity (dimensionless, >= 0)
      i: Inclination (radians, [0, pi])
      raan: Longitude of the ascending node Omega (radians, [0, 2*pi))
      arg_peri: Argument of periapsis omega (radians, [0, 2*pi))
      true_anomaly: True anomaly nu (radians, [0, 2*pi))
      mu: Gravitational parameter of central body (km^3 / s^2)
    """
    a: float
    e: float
    i: float
    raan: float
    arg_peri: float
    true_anomaly: float
    mu: float = EARTH.mu

    @property
    def orbit_type(self) -> OrbitType:
        """Determine conic orbit classification based on eccentricity."""
        if self.e < 1e-3:
            return OrbitType.CIRCULAR
        elif self.e < 1.0:
            return OrbitType.ELLIPTIC
        elif abs(self.e - 1.0) <= 1e-4:
            return OrbitType.PARABOLIC
        else:
            return OrbitType.HYPERBOLIC

    @property
    def periapsis_radius(self) -> float:
        """Distance from center of attraction to periapsis (km)."""
        return self.a * (1.0 - self.e)

    @property
    def apoapsis_radius(self) -> float:
        """Distance from center of attraction to apoapsis (km) for bound orbits."""
        if self.e >= 1.0:
            return float("inf")
        return self.a * (1.0 + self.e)

    @property
    def semi_latus_rectum(self) -> float:
        """Semi-latus rectum parameter p = a * (1 - e^2) in km."""
        return self.a * (1.0 - self.e * self.e)

    @property
    def period(self) -> float:
        """Orbital period in seconds for closed elliptic orbits."""
        if self.e >= 1.0 or self.a <= 0:
            return float("inf")
        return 2.0 * math.pi * math.sqrt((self.a ** 3) / self.mu)

    @property
    def mean_motion(self) -> float:
        """Mean angular rate n in radians/second."""
        if self.a <= 0:
            return 0.0
        return math.sqrt(self.mu / (self.a ** 3))

    @property
    def specific_energy(self) -> float:
        """Specific mechanical energy E = -mu / (2*a) in km^2 / s^2."""
        return -self.mu / (2.0 * self.a)

    @property
    def angular_momentum(self) -> float:
        """Magnitude of specific angular momentum h in km^2 / s."""
        p = self.semi_latus_rectum
        return math.sqrt(abs(self.mu * p))

    def summary(self) -> str:
        """Human-readable orbital element summary table."""
        return (
            f"Orbit Type:         {self.orbit_type.value.upper()}\n"
            f"Semi-Major Axis:    {self.a:,.2f} km\n"
            f"Eccentricity:       {self.e:.6f}\n"
            f"Inclination:        {math.degrees(self.i):.3f}°\n"
            f"RAAN (Ω):           {math.degrees(self.raan):.3f}°\n"
            f"Arg of Periapsis (ω): {math.degrees(self.arg_peri):.3f}°\n"
            f"True Anomaly (ν):   {math.degrees(self.true_anomaly):.3f}°\n"
            f"Periapsis Radius:   {self.periapsis_radius:,.2f} km\n"
            f"Apoapsis Radius:    {self.apoapsis_radius:,.2f} km\n"
            f"Period:             {self.period / 60.0:,.2f} min ({self.period / 3600.0:.2f} hr)"
        )
