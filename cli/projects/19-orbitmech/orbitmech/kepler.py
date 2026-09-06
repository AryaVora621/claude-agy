"""
OrbitMech: Keplerian Two-Body Dynamics & Classical Orbital Elements Engine.
Provides state vector <-> orbital elements transformations, anomaly conversions,
and Danby high-order root solving for Kepler's equation.
"""

from __future__ import annotations
import math
from typing import Tuple

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    OrbitType,
    EARTH,
)


def true_to_eccentric_anomaly(nu: float, e: float) -> float:
    """
    Convert true anomaly nu to eccentric anomaly E for elliptic orbits (e < 1.0)
    or hyperbolic anomaly F for hyperbolic orbits (e > 1.0).
    Result is wrapped into [0, 2*pi).
    """
    if e < 1.0:
        # Standard half-angle identity avoiding quadrant ambiguities
        sin_half_nu = math.sin(nu * 0.5)
        cos_half_nu = math.cos(nu * 0.5)
        E = 2.0 * math.atan2(
            math.sqrt(max(0.0, 1.0 - e)) * sin_half_nu,
            math.sqrt(1.0 + e) * cos_half_nu,
        )
        return E % (2.0 * math.pi)
    elif e > 1.0:
        # Hyperbolic anomaly F
        cos_nu = math.cos(nu)
        arg = (e + cos_nu) / (1.0 + e * cos_nu)
        # Bounded between 1 and inf
        arg = max(1.0, arg)
        F = math.acosh(arg)
        return F if nu >= 0 else -F
    else:
        # Parabolic anomaly
        return math.tan(nu * 0.5)


def eccentric_to_true_anomaly(E: float, e: float) -> float:
    """
    Convert eccentric anomaly E to true anomaly nu for elliptic orbits (e < 1.0)
    or hyperbolic anomaly F to nu for hyperbolic orbits (e > 1.0).
    Result is wrapped into [0, 2*pi).
    """
    if e < 1.0:
        sin_half_E = math.sin(E * 0.5)
        cos_half_E = math.cos(E * 0.5)
        nu = 2.0 * math.atan2(
            math.sqrt(1.0 + e) * sin_half_E,
            math.sqrt(max(0.0, 1.0 - e)) * cos_half_E,
        )
        return nu % (2.0 * math.pi)
    elif e > 1.0:
        # Hyperbolic anomaly F -> nu
        sinh_F = math.sinh(E)
        cosh_F = math.cosh(E)
        cos_nu = (cosh_F - e) / (1.0 - e * cosh_F)
        cos_nu = max(-1.0, min(1.0, cos_nu))
        nu = math.acos(cos_nu)
        return nu if E >= 0 else 2.0 * math.pi - nu
    else:
        # Parabolic anomaly
        return 2.0 * math.atan(E)


def eccentric_to_mean_anomaly(E: float, e: float) -> float:
    """
    Calculate mean anomaly M from eccentric anomaly E using Kepler's relation M = E - e*sin(E).
    """
    if e < 1.0:
        return (E - e * math.sin(E)) % (2.0 * math.pi)
    elif e > 1.0:
        return e * math.sinh(E) - E
    else:
        # Barker's equation for parabolic orbit
        return E + (E ** 3) / 3.0


def solve_kepler_equation(
    M: float,
    e: float,
    tolerance: float = 1e-13,
    max_iter: int = 50,
) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.
    Uses Danby's 3rd-order Householder/Halley method with Danby starting guess,
    achieving machine-precision convergence in 2-3 iterations across all e in [0, 0.99999].
    """
    # Normalize M to [0, 2*pi)
    two_pi = 2.0 * math.pi
    M = M % two_pi

    # Danby initial guess
    sin_M = math.sin(M)
    E = M + 0.85 * e * (1.0 if sin_M >= 0 else -1.0)

    for _ in range(max_iter):
        sin_E = math.sin(E)
        cos_E = math.cos(E)
        f = E - e * sin_E - M
        f_prime = 1.0 - e * cos_E
        f_double_prime = e * sin_E
        f_triple_prime = e * cos_E

        if abs(f) < tolerance:
            break

        # Danby high-order update corrections
        delta_1 = -f / f_prime
        delta_2 = -f / (f_prime + 0.5 * delta_1 * f_double_prime)
        delta_3 = -f / (
            f_prime
            + 0.5 * delta_2 * f_double_prime
            + (1.0 / 6.0) * (delta_2 ** 2) * f_triple_prime
        )
        E += delta_3

    return E % two_pi


def state_to_orbital_elements(
    state: StateVector,
    mu: float = EARTH.mu,
) -> ClassicalOrbitalElements:
    """
    Convert 3D Cartesian position and velocity state vector to classical Keplerian elements:
    semi-major axis (a), eccentricity (e), inclination (i), RAAN (Omega),
    argument of periapsis (omega), and true anomaly (nu).
    """
    r_vec = state.r
    v_vec = state.v
    r = r_vec.norm()
    v = v_vec.norm()

    if r < 1e-6 or v < 1e-12:
        raise ValueError("Degenerate state vector: position or velocity near zero")

    # 1. Specific Angular Momentum h = r x v
    h_vec = r_vec.cross(v_vec)
    h = h_vec.norm()

    # 2. Specific Mechanical Energy E = v^2/2 - mu/r
    energy = 0.5 * (v * v) - (mu / r)

    # 3. Semi-Major Axis a = -mu / (2*E)
    if abs(energy) < 1e-10:
        # Parabolic orbit
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # 4. Eccentricity Vector e = ((v^2 - mu/r)*r - (r.v)*v) / mu
    rdotv = r_vec.dot(v_vec)
    e_vec = (r_vec * (v * v - (mu / r)) - v_vec * rdotv) / mu
    e = e_vec.norm()

    # 5. Inclination i = acos(h_z / h)
    i = math.acos(max(-1.0, min(1.0, h_vec.z / h)))

    # 6. Node Vector n = k_hat x h = (-h_y, h_x, 0)
    n_vec = Vector3(-h_vec.y, h_vec.x, 0.0)
    n = n_vec.norm()

    # 7. Longitude of Ascending Node (RAAN) Omega
    if n > 1e-11:
        raan = math.acos(max(-1.0, min(1.0, n_vec.x / n)))
        if n_vec.y < 0:
            raan = 2.0 * math.pi - raan
    else:
        # Equatorial orbit: RAAN is undefined, conventionally set to 0
        raan = 0.0

    # 8. Argument of Periapsis omega
    if n > 1e-11 and e > 1e-10:
        cos_omega = n_vec.dot(e_vec) / (n * e)
        omega = math.acos(max(-1.0, min(1.0, cos_omega)))
        if e_vec.z < 0:
            omega = 2.0 * math.pi - omega
    elif e > 1e-10:
        # Equatorial eccentric orbit: longitude of periapsis pi = omega
        omega = math.acos(max(-1.0, min(1.0, e_vec.x / e)))
        if e_vec.y < 0:
            omega = 2.0 * math.pi - omega
    else:
        # Circular orbit: omega is undefined, conventionally set to 0
        omega = 0.0

    # 9. True Anomaly nu
    if e > 1e-10:
        cos_nu = e_vec.dot(r_vec) / (e * r)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if rdotv < 0:
            nu = 2.0 * math.pi - nu
    else:
        # Circular orbit: argument of latitude u = nu + omega
        if n > 1e-11:
            cos_u = n_vec.dot(r_vec) / (n * r)
            u = math.acos(max(-1.0, min(1.0, cos_u)))
            if r_vec.z < 0:
                u = 2.0 * math.pi - u
            nu = u
        else:
            # Equatorial circular: true longitude lambda = nu
            nu = math.atan2(r_vec.y, r_vec.x)
            if nu < 0:
                nu += 2.0 * math.pi

    return ClassicalOrbitalElements(
        a=a,
        e=e,
        i=i,
        raan=raan,
        arg_peri=omega,
        true_anomaly=nu,
        mu=mu,
    )


def orbital_elements_to_state(
    elements: ClassicalOrbitalElements,
) -> StateVector:
    """
    Convert classical Keplerian orbital elements to Cartesian state vector (r, v).
    Computes position and velocity in the perifocal (P, Q, W) orbital plane frame,
    then transforms to equatorial inertial frame via 3-1-3 Euler rotation matrix.
    """
    mu = elements.mu
    e = elements.e
    nu = elements.true_anomaly

    # Semi-latus rectum parameter p = a * (1 - e^2)
    p = elements.semi_latus_rectum
    if abs(p) < 1e-10:
        raise ValueError("Degenerate orbit: semi-latus rectum near zero")

    # Radial distance r = p / (1 + e*cos(nu))
    denom = 1.0 + e * math.cos(nu)
    if denom <= 0:
        raise ValueError(f"True anomaly {nu:.3f} rad is not realizable on this conic (1 + e*cos(nu) <= 0)")
    r_mag = p / denom

    # Perifocal frame coordinates
    sin_nu = math.sin(nu)
    cos_nu = math.cos(nu)
    r_peri_x = r_mag * cos_nu
    r_peri_y = r_mag * sin_nu

    speed_factor = math.sqrt(abs(mu / p))
    v_peri_x = -speed_factor * sin_nu
    v_peri_y = speed_factor * (e + cos_nu)

    # Transformation from Perifocal (P, Q, W) to Equatorial Inertial (X, Y, Z)
    # Rotation angles: Omega (raan), i (inclination), omega (arg_peri)
    cos_raan = math.cos(elements.raan)
    sin_raan = math.sin(elements.raan)
    cos_i = math.cos(elements.i)
    sin_i = math.sin(elements.i)
    cos_w = math.cos(elements.arg_peri)
    sin_w = math.sin(elements.arg_peri)

    # Unit vectors P and Q expressed in equatorial Cartesian coordinates
    # P vector points toward periapsis
    Px = cos_raan * cos_w - sin_raan * sin_w * cos_i
    Py = sin_raan * cos_w + cos_raan * sin_w * cos_i
    Pz = sin_w * sin_i

    # Q vector points in direction of motion at periapsis (nu = 90 deg)
    Qx = -cos_raan * sin_w - sin_raan * cos_w * cos_i
    Qy = -sin_raan * sin_w + cos_raan * cos_w * cos_i
    Qz = cos_w * sin_i

    # Inertial position and velocity vectors
    rx = r_peri_x * Px + r_peri_y * Qx
    ry = r_peri_x * Py + r_peri_y * Qy
    rz = r_peri_x * Pz + r_peri_y * Qz

    vx = v_peri_x * Px + v_peri_y * Qx
    vy = v_peri_x * Py + v_peri_y * Qy
    vz = v_peri_x * Pz + v_peri_y * Qz

    return StateVector(r=Vector3(rx, ry, rz), v=Vector3(vx, vy, vz))


def propagate_keplerian(
    state: StateVector,
    dt: float,
    mu: float = EARTH.mu,
) -> StateVector:
    """
    Analytically propagate an initial state vector forward or backward by dt seconds
    under unperturbed two-body Keplerian motion.
    Solves Kepler's equation for the future true anomaly, then transforms back to Cartesian coordinates.
    """
    elements = state_to_orbital_elements(state, mu=mu)
    if elements.e >= 1.0:
        # For parabolic/hyperbolic orbits, caller should use universal variable propagator
        raise NotImplementedError("Analytic Keplerian propagation for e >= 1.0 requires universal variable solver")

    # 1. Current eccentric anomaly and mean anomaly
    E0 = true_to_eccentric_anomaly(elements.true_anomaly, elements.e)
    M0 = eccentric_to_mean_anomaly(E0, elements.e)

    # 2. Advance mean anomaly M(t) = M0 + n*dt
    n = elements.mean_motion
    M_target = M0 + n * dt

    # 3. Solve Kepler's equation for future eccentric anomaly E(t)
    E_target = solve_kepler_equation(M_target, elements.e)

    # 4. Convert future E(t) to future true anomaly nu(t)
    nu_target = eccentric_to_true_anomaly(E_target, elements.e)

    # 5. Build updated orbital elements and compute Cartesian state vector
    target_elements = ClassicalOrbitalElements(
        a=elements.a,
        e=elements.e,
        i=elements.i,
        raan=elements.raan,
        arg_peri=elements.arg_peri,
        true_anomaly=nu_target,
        mu=mu,
    )
    result_state = orbital_elements_to_state(target_elements)
    return StateVector(r=result_state.r, v=result_state.v, time=state.time + dt)
