"""
OrbitMech: Universal Variable Formulation & Lambert Boundary Value Problem Solver.
Provides Stumpff functions, universal Kepler propagation across all conics,
and robust Lambert targeting using Bate-Mueller-White and Battin algorithms.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
)


def stumpff_c2(z: float) -> float:
    """
    Evaluate Stumpff function c_2(z) = (1 - cos(sqrt(z))) / z for z > 0,
    (cosh(sqrt(-z)) - 1) / (-z) for z < 0, and 1/2 at z = 0.
    Uses Taylor series expansion for |z| < 1e-4 to maintain numerical precision.
    """
    if abs(z) < 1e-4:
        # Taylor series: 1/2! - z/4! + z^2/6! - z^3/8! + ...
        return (
            0.5
            - z / 24.0
            + (z * z) / 720.0
            - (z ** 3) / 40320.0
            + (z ** 4) / 3628800.0
        )
    elif z > 0:
        sqrt_z = math.sqrt(z)
        return (1.0 - math.cos(sqrt_z)) / z
    else:
        sqrt_neg_z = math.sqrt(-z)
        return (math.cosh(sqrt_neg_z) - 1.0) / (-z)


def stumpff_c3(z: float) -> float:
    """
    Evaluate Stumpff function c_3(z) = (sqrt(z) - sin(sqrt(z))) / (z^(3/2)) for z > 0,
    (sinh(sqrt(-z)) - sqrt(-z)) / ((-z)^(3/2)) for z < 0, and 1/6 at z = 0.
    Uses Taylor series expansion for |z| < 1e-4.
    """
    if abs(z) < 1e-4:
        # Taylor series: 1/3! - z/5! + z^2/7! - z^3/9! + ...
        return (
            (1.0 / 6.0)
            - z / 120.0
            + (z * z) / 5040.0
            - (z ** 3) / 362880.0
            + (z ** 4) / 39916800.0
        )
    elif z > 0:
        sqrt_z = math.sqrt(z)
        return (sqrt_z - math.sin(sqrt_z)) / (z * sqrt_z)
    else:
        sqrt_neg_z = math.sqrt(-z)
        return (math.sinh(sqrt_neg_z) - sqrt_neg_z) / (-z * sqrt_neg_z)


def propagate_universal(
    state: StateVector,
    dt: float,
    mu: float = EARTH.mu,
    tolerance: float = 1e-12,
    max_iter: int = 50,
) -> StateVector:
    """
    Propagate an initial orbital state by dt seconds using the universal variable formulation.
    Valid for all conic trajectories: circular, elliptic, parabolic, and hyperbolic.
    Solves for universal anomaly chi via Newton-Raphson iteration, then evaluates
    Lagrange f, g, f_dot, g_dot coefficients.
    """
    r0_vec = state.r
    v0_vec = state.v
    r0 = r0_vec.norm()
    v0 = v0_vec.norm()
    sqrt_mu = math.sqrt(mu)

    # Reciprocal of semi-major axis: alpha = 1/a = 2/r0 - v0^2/mu
    alpha = (2.0 / r0) - ((v0 * v0) / mu)
    r0_dot_v0 = r0_vec.dot(v0_vec)

    # Initial guess for universal anomaly chi
    if alpha > 1e-6:
        # Elliptic orbit: chi ~ sqrt(mu) * dt * alpha
        chi = sqrt_mu * dt * alpha
    elif abs(alpha) <= 1e-6:
        # Parabolic orbit
        h_vec = r0_vec.cross(v0_vec)
        h = h_vec.norm()
        p = (h * h) / mu
        s = 0.5 * math.atan2(1.0, 3.0 * math.sqrt(mu / (p ** 3)) * dt)
        w = math.atan((math.tan(s)) ** (1.0 / 3.0))
        chi = math.sqrt(p) * 2.0 / math.tan(2.0 * w)
    else:
        # Hyperbolic orbit: alpha < 0
        a = 1.0 / alpha
        term = -2.0 * mu * dt / (a * (r0_dot_v0 + math.copysign(1.0, dt) * math.sqrt(-mu * a) * (1.0 - r0 * alpha)))
        chi = math.copysign(1.0, dt) * math.sqrt(-a) * math.log(max(1e-12, abs(term) + 1.0))

    # Newton-Raphson iteration to solve for chi
    for _ in range(max_iter):
        chi2 = chi * chi
        z = alpha * chi2
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)

        # Universal Kepler function: f(chi)
        f_val = (
            (r0_dot_v0 / sqrt_mu) * chi2 * c2
            + (1.0 - alpha * r0) * (chi * chi2) * c3
            + r0 * chi
            - sqrt_mu * dt
        )

        if abs(f_val) < tolerance:
            break

        # Derivative: f'(chi)
        f_prime = (
            (r0_dot_v0 / sqrt_mu) * chi * (1.0 - z * c3)
            + (1.0 - alpha * r0) * chi2 * c2
            + r0
        )

        if abs(f_prime) < 1e-14:
            break

        chi -= f_val / f_prime

    # Compute Lagrange f, g, f_dot, g_dot coefficients
    chi2 = chi * chi
    z = alpha * chi2
    c2 = stumpff_c2(z)
    c3 = stumpff_c3(z)

    f = 1.0 - (chi2 / r0) * c2
    g = dt - ((chi * chi2) / sqrt_mu) * c3

    r_vec = r0_vec * f + v0_vec * g
    r = r_vec.norm()

    f_dot = (sqrt_mu / (r * r0)) * (z * chi * c3 - chi)
    g_dot = 1.0 - (chi2 / r) * c2

    v_vec = r0_vec * f_dot + v0_vec * g_dot

    return StateVector(r=r_vec, v=v_vec, time=state.time + dt)


@dataclass(frozen=True)
class LambertSolution:
    """
    Output of Lambert's boundary value targeting problem.
      v1: Initial departure velocity vector (km/s)
      v2: Final arrival velocity vector (km/s)
      time_of_flight: Trajectory duration in seconds
      semi_major_axis: Semi-major axis of transfer orbit (km)
      transfer_angle: Angle swept by transfer in radians
      revolutions: Number of complete orbits before arrival (default 0)
    """
    v1: Vector3
    v2: Vector3
    time_of_flight: float
    semi_major_axis: float
    transfer_angle: float
    revolutions: int = 0


def solve_lambert(
    r1_vec: Vector3,
    r2_vec: Vector3,
    time_of_flight: float,
    mu: float = EARTH.mu,
    prograde: bool = True,
    short_way: bool = True,
    tolerance: float = 1e-10,
    max_iter: int = 100,
) -> LambertSolution:
    """
    Solve Lambert's boundary value problem: find the orbital trajectory connecting
    initial position r1 to final position r2 in elapsed duration time_of_flight.
    Uses universal variables with Stumpff functions to handle all conic regimes uniformly.
    """
    if time_of_flight <= 0:
        raise ValueError("Time of flight must be strictly positive")

    r1 = r1_vec.norm()
    r2 = r2_vec.norm()
    if r1 < 1e-3 or r2 < 1e-3:
        raise ValueError("Degenerate position vector near coordinate origin")

    sqrt_mu = math.sqrt(mu)
    cos_dtheta = max(-1.0, min(1.0, r1_vec.dot(r2_vec) / (r1 * r2)))

    # Determine transfer angle delta_theta
    cross_12 = r1_vec.cross(r2_vec)
    if prograde:
        if cross_12.z >= 0:
            dtheta = math.acos(cos_dtheta)
        else:
            dtheta = 2.0 * math.pi - math.acos(cos_dtheta)
    else:
        if cross_12.z < 0:
            dtheta = math.acos(cos_dtheta)
        else:
            dtheta = 2.0 * math.pi - math.acos(cos_dtheta)

    if short_way and dtheta > math.pi:
        dtheta = 2.0 * math.pi - dtheta
    elif not short_way and dtheta < math.pi:
        dtheta = 2.0 * math.pi - dtheta

    # Geometric constant A
    sin_dtheta = math.sin(dtheta)
    denom = 1.0 - cos_dtheta
    if abs(denom) < 1e-12:
        raise ValueError("Collinear position vectors (delta_theta = 0 or 2*pi) cannot form a unique plane")

    A = sin_dtheta * math.sqrt((r1 * r2) / denom)

    # Function evaluating y(z) and time_of_flight(z)
    def tof_equation(z: float) -> Tuple[float, float]:
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)
        if c2 <= 0:
            return 0.0, float("inf")
        y = r1 + r2 + A * (z * c3 - 1.0) / math.sqrt(c2)
        if y < 0:
            # Physically unrealizable for this z
            return y, -1.0
        chi = math.sqrt(y / c2)
        t = (chi ** 3 * c3 + A * math.sqrt(y)) / sqrt_mu
        return y, t

    # Bracket the root z where tof_equation(z) - time_of_flight = 0.
    # Note: t(z) is strictly monotonically increasing with z.
    # Upper bound for single revolution: z < 4*pi^2 ~ 39.478
    z_high = 4.0 * (math.pi ** 2)

    # Find a valid lower bound z_low where y(z) > 0
    z_low = -4.0 * (math.pi ** 2)
    while tof_equation(z_low)[0] <= 0 and z_low < 0:
        z_low *= 0.5

    # Bisection iteration
    z = 0.0
    y_final = 0.0
    for _ in range(max_iter):
        z = 0.5 * (z_low + z_high)
        y, t_calc = tof_equation(z)
        if y <= 0:
            z_low = z
            continue

        error = t_calc - time_of_flight
        if abs(error) < tolerance:
            y_final = y
            break

        if error < 0:
            # Calculated time is shorter than required -> need larger z
            z_low = z
        else:
            # Calculated time is longer than required -> need smaller z
            z_high = z

        y_final = y
    else:
        y_final, _ = tof_equation(z)

    # Recover Lagrange coefficients f, g, g_dot
    f = 1.0 - (y_final / r1)
    g = A * math.sqrt(y_final / mu)
    g_dot = 1.0 - (y_final / r2)

    v1 = (r2_vec - r1_vec * f) / g
    v2 = (r2_vec * g_dot - r1_vec) / g

    # Semi-major axis from energy
    v1_sq = v1.norm_squared()
    energy = 0.5 * v1_sq - (mu / r1)
    semi_major_axis = -mu / (2.0 * energy) if abs(energy) > 1e-9 else float("inf")

    return LambertSolution(
        v1=v1,
        v2=v2,
        time_of_flight=time_of_flight,
        semi_major_axis=semi_major_axis,
        transfer_angle=dtheta,
    )
