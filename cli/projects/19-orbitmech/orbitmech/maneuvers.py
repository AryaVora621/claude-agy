"""
OrbitMech: Orbital Maneuvers, Interplanetary Transfers & Gravity Assists.
Provides Hohmann transfer, Bi-elliptic 3-impulse transfer, Plane change optimization,
and Planetary Gravity Assist (Hyperbolic Flyby) kinematics.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

from orbitmech.types import (
    Vector3,
    StateVector,
    CelestialBody,
    EARTH,
    SUN,
)


@dataclass(frozen=True)
class HohmannTransferResult:
    """
    Characteristics of a two-impulse coplanar Hohmann transfer between circular orbits.
      r1: Initial circular orbit radius (km)
      r2: Final target circular orbit radius (km)
      delta_v1: First burn magnitude at r1 (km/s)
      delta_v2: Second burn magnitude at r2 (km/s)
      total_delta_v: Sum of both impulse magnitudes (km/s)
      time_of_flight: Half-period duration of transfer ellipse (seconds)
      transfer_semi_major_axis: Semi-major axis of transfer ellipse (km)
      phase_angle_deg: Required phase angle between departure and arrival bodies (degrees)
    """
    r1: float
    r2: float
    delta_v1: float
    delta_v2: float
    total_delta_v: float
    time_of_flight: float
    transfer_semi_major_axis: float
    phase_angle_deg: float


@dataclass(frozen=True)
class BiEllipticTransferResult:
    """
    Characteristics of a three-impulse bi-elliptic transfer via intermediate radius rb.
      r1: Initial circular orbit radius (km)
      r2: Final target circular orbit radius (km)
      rb: Intermediate apoapsis radius (km)
      delta_v1: First burn at r1 (km/s)
      delta_v2: Second burn at rb (km/s)
      delta_v3: Third burn at r2 (km/s)
      total_delta_v: Sum of all three burns (km/s)
      time_of_flight: Total duration tau1 + tau2 (seconds)
      is_more_efficient_than_hohmann: True if total_delta_v < hohmann_total_delta_v
    """
    r1: float
    r2: float
    rb: float
    delta_v1: float
    delta_v2: float
    delta_v3: float
    total_delta_v: float
    time_of_flight: float
    is_more_efficient_than_hohmann: bool


@dataclass(frozen=True)
class PlaneChangeResult:
    """
    Impulse requirements for changing orbital inclination by delta_i.
      delta_i_deg: Inclination change in degrees
      v_initial: Orbital speed at time of burn (km/s)
      delta_v: Magnitude of impulsive velocity change required (km/s)
    """
    delta_i_deg: float
    v_initial: float
    delta_v: float


@dataclass(frozen=True)
class GravityAssistResult:
    """
    Kinematic outcome of an unpowered hyperbolic planetary gravity assist flyby.
      v_inf_in: Excess approach velocity vector relative to planet (km/s)
      v_inf_out: Excess departure velocity vector relative to planet (km/s)
      v_sc_in: Incoming heliocentric spacecraft velocity vector (km/s)
      v_sc_out: Outgoing heliocentric spacecraft velocity vector (km/s)
      delta_v_helio: Net vector change in heliocentric velocity (km/s)
      delta_v_mag: Magnitude of heliocentric velocity boost or brake (km/s)
      bending_angle_deg: Hyperbolic turning angle delta in degrees
      periapsis_altitude: Closest approach altitude above planetary surface (km)
      periapsis_radius: Closest approach radius from planetary center (km)
    """
    v_inf_in: Vector3
    v_inf_out: Vector3
    v_sc_in: Vector3
    v_sc_out: Vector3
    delta_v_helio: Vector3
    delta_v_mag: float
    bending_angle_deg: float
    periapsis_altitude: float
    periapsis_radius: float


def hohmann_transfer(
    r1: float,
    r2: float,
    mu: float = EARTH.mu,
) -> HohmannTransferResult:
    """
    Compute optimal two-impulse coplanar Hohmann transfer between circular orbits of radius r1 and r2.
    """
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbital radii must be strictly positive")

    # Circular velocities
    v_c1 = math.sqrt(mu / r1)
    v_c2 = math.sqrt(mu / r2)

    # Transfer ellipse
    a_tx = 0.5 * (r1 + r2)
    v_tx1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a_tx))
    v_tx2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a_tx))

    # Impulses
    dv1 = abs(v_tx1 - v_c1)
    dv2 = abs(v_c2 - v_tx2)
    total_dv = dv1 + dv2

    # Half-period transfer duration: tau = pi * sqrt(a_tx^3 / mu)
    time_of_flight = math.pi * math.sqrt((a_tx ** 3) / mu)

    # Lead angle / phase angle for planetary rendezvous
    # Angular velocity of target orbit: omega_2 = sqrt(mu / r2^3)
    omega2 = math.sqrt(mu / (r2 ** 3))
    # Angle traversed by target during flight: theta_target = omega2 * tof
    # Required phase angle at departure: phi = pi - omega2 * tof
    phase_rad = math.pi - omega2 * time_of_flight
    # Normalize to [0, 2*pi)
    phase_rad = phase_rad % (2.0 * math.pi)

    return HohmannTransferResult(
        r1=r1,
        r2=r2,
        delta_v1=dv1,
        delta_v2=dv2,
        total_delta_v=total_dv,
        time_of_flight=time_of_flight,
        transfer_semi_major_axis=a_tx,
        phase_angle_deg=math.degrees(phase_rad),
    )


def bi_elliptic_transfer(
    r1: float,
    r2: float,
    rb: float,
    mu: float = EARTH.mu,
) -> BiEllipticTransferResult:
    """
    Compute three-impulse bi-elliptic transfer via intermediate apoapsis radius rb.
    For radius ratios r2 / r1 > 11.9387, a bi-elliptic transfer with large rb
    requires less total Delta-V than a Hohmann transfer, at the expense of longer flight time.
    """
    if rb < max(r1, r2):
        raise ValueError("Intermediate radius rb must be greater than or equal to both r1 and r2")

    v_c1 = math.sqrt(mu / r1)
    v_c2 = math.sqrt(mu / r2)

    # Ellipse 1: periapsis r1, apoapsis rb
    a1 = 0.5 * (r1 + rb)
    v_1_tx1 = math.sqrt(mu * (2.0 / r1 - 1.0 / a1))
    v_b_tx1 = math.sqrt(mu * (2.0 / rb - 1.0 / a1))
    dv1 = abs(v_1_tx1 - v_c1)

    # Ellipse 2: periapsis r2, apoapsis rb
    a2 = 0.5 * (r2 + rb)
    v_b_tx2 = math.sqrt(mu * (2.0 / rb - 1.0 / a2))
    v_2_tx2 = math.sqrt(mu * (2.0 / r2 - 1.0 / a2))
    dv2 = abs(v_b_tx2 - v_b_tx1)

    # Final circularization at r2
    dv3 = abs(v_c2 - v_2_tx2)
    total_dv = dv1 + dv2 + dv3

    # Total duration: tau1 + tau2
    tof = math.pi * math.sqrt((a1 ** 3) / mu) + math.pi * math.sqrt((a2 ** 3) / mu)

    # Compare against standard Hohmann
    hohmann_res = hohmann_transfer(r1, r2, mu=mu)
    is_better = total_dv < hohmann_res.total_delta_v

    return BiEllipticTransferResult(
        r1=r1,
        r2=r2,
        rb=rb,
        delta_v1=dv1,
        delta_v2=dv2,
        delta_v3=dv3,
        total_delta_v=total_dv,
        time_of_flight=tof,
        is_more_efficient_than_hohmann=is_better,
    )


def plane_change_maneuver(
    delta_i_deg: float,
    v_initial: float,
) -> PlaneChangeResult:
    """
    Compute impulse required for a pure orbital inclination change:
      Delta_V = 2 * v * sin(Delta_i / 2)
    """
    delta_i_rad = math.radians(abs(delta_i_deg))
    dv = 2.0 * v_initial * math.sin(delta_i_rad * 0.5)
    return PlaneChangeResult(
        delta_i_deg=delta_i_deg,
        v_initial=v_initial,
        delta_v=dv,
    )


def gravity_assist_flyby(
    v_sc_in: Vector3,
    v_planet: Vector3,
    planet: CelestialBody,
    periapsis_altitude: float,
    flyby_plane_normal: Optional[Vector3] = None,
) -> GravityAssistResult:
    """
    Compute planetary hyperbolic flyby gravity assist kinematics.
      v_sc_in: Spacecraft incoming heliocentric velocity vector (km/s)
      v_planet: Planet heliocentric orbital velocity vector (km/s)
      planet: Celestial attracting body performing the flyby
      periapsis_altitude: Closest approach distance above planet's surface (km)
    """
    rp = planet.radius + periapsis_altitude
    if rp <= planet.radius:
        raise ValueError("Periapsis radius penetrates planetary surface")

    # Hyperbolic excess arrival velocity vector relative to planet
    v_inf_in = v_sc_in - v_planet
    v_inf_mag = v_inf_in.norm()

    if v_inf_mag < 1e-6:
        raise ValueError("Excess hyperbolic velocity too small for meaningful flyby")

    # Bending angle: sin(delta/2) = 1 / (1 + (rp * v_inf^2 / mu))
    denom = 1.0 + (rp * (v_inf_mag ** 2)) / planet.mu
    sin_half_delta = 1.0 / denom
    half_delta = math.asin(max(0.0, min(1.0, sin_half_delta)))
    delta_rad = 2.0 * half_delta

    # Define rotation plane for deflection
    if flyby_plane_normal is None:
        # Default: rotate in the plane formed by v_inf_in and v_planet
        cross_plane = v_inf_in.cross(v_planet)
        if cross_plane.norm() < 1e-6:
            # Collinear, pick perpendicular axis
            flyby_plane_normal = Vector3(0.0, 0.0, 1.0)
        else:
            flyby_plane_normal = cross_plane.normalized()
    else:
        flyby_plane_normal = flyby_plane_normal.normalized()

    # Rodrigues' rotation formula rotating v_inf_in by angle delta_rad around flyby_plane_normal:
    # v_rot = v*cos(d) + (k x v)*sin(d) + k*(k.v)*(1 - cos(d))
    k = flyby_plane_normal
    v = v_inf_in
    cos_d = math.cos(delta_rad)
    sin_d = math.sin(delta_rad)
    k_cross_v = k.cross(v)
    k_dot_v = k.dot(v)

    v_inf_out = v * cos_d + k_cross_v * sin_d + k * (k_dot_v * (1.0 - cos_d))

    # Outgoing heliocentric spacecraft velocity
    v_sc_out = v_planet + v_inf_out

    # Net heliocentric velocity boost vector
    delta_v_helio = v_sc_out - v_sc_in
    delta_v_mag = delta_v_helio.norm()

    return GravityAssistResult(
        v_inf_in=v_inf_in,
        v_inf_out=v_inf_out,
        v_sc_in=v_sc_in,
        v_sc_out=v_sc_out,
        delta_v_helio=delta_v_helio,
        delta_v_mag=delta_v_mag,
        bending_angle_deg=math.degrees(delta_rad),
        periapsis_altitude=periapsis_altitude,
        periapsis_radius=rp,
    )
