"""
OrbitMech: High-Order Numerical Propagators & Astrodynamical Perturbation Modeling.
Provides J2 zonal harmonic oblateness, atmospheric drag, solar radiation pressure,
Symplectic Störmer-Verlet (energy-conserving), and adaptive RK45 integrators.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional

from orbitmech.types import (
    Vector3,
    StateVector,
    CelestialBody,
    EARTH,
    SUN,
    AU_KM,
)


@dataclass(frozen=True)
class SpacecraftProperties:
    """
    Physical properties of an orbiting spacecraft.
      mass: Wet mass in kilograms
      cross_section: Drag and radiation surface area in m^2
      drag_coefficient: Dimensionless aerodynamic drag coefficient Cd (default 2.2)
      reflectivity: Dimensionless SRP reflectivity Cr in [1.0, 2.0] (default 1.3)
    """
    mass: float = 1000.0         # kg
    cross_section: float = 10.0  # m^2
    drag_coefficient: float = 2.2
    reflectivity: float = 1.3


@dataclass(frozen=True)
class PerturbationConfig:
    """
    Flags controlling active orbital perturbation models.
    """
    enable_j2: bool = True
    enable_drag: bool = False
    enable_srp: bool = False
    body: CelestialBody = EARTH
    spacecraft: SpacecraftProperties = field(default_factory=SpacecraftProperties)


def atmospheric_density_earth(altitude_km: float) -> float:
    """
    Approximate exponential atmospheric density model for Earth in kg/km^3.
    Altitude in kilometers. Bounded below 800 km.
    """
    if altitude_km < 0:
        altitude_km = 0.0
    if altitude_km > 1000.0:
        return 0.0

    # Scale height parameters: rho_0 = 1.225 kg/m^3 = 1.225e9 kg/km^3
    # Effective scale height H ~ 7.5 km in troposphere, increasing with altitude
    h = altitude_km
    if h < 25.0:
        rho_m3 = 1.225 * math.exp(-h / 7.2)
    elif h < 100.0:
        rho_m3 = 3.899e-2 * math.exp(-(h - 25.0) / 6.5)
    elif h < 300.0:
        rho_m3 = 5.297e-7 * math.exp(-(h - 100.0) / 28.0)
    elif h < 600.0:
        rho_m3 = 4.0e-11 * math.exp(-(h - 300.0) / 55.0)
    else:
        rho_m3 = 1.0e-13 * math.exp(-(h - 600.0) / 80.0)

    # Convert kg/m^3 to kg/km^3: 1 m^3 = 1e-9 km^3 -> multiply by 1e9
    return rho_m3 * 1e9


def compute_acceleration(
    r_vec: Vector3,
    v_vec: Vector3,
    config: PerturbationConfig,
) -> Vector3:
    """
    Compute net acceleration (km/s^2) acting on spacecraft given position and velocity:
      a_net = a_gravity + a_J2 + a_drag + a_srp
    """
    body = config.body
    mu = body.mu
    r = r_vec.norm()

    if r < 1e-3:
        return Vector3.zero()

    # 1. Central Body Point-Mass Gravitational Acceleration: a = -mu/r^3 * r
    inv_r3 = 1.0 / (r * r * r)
    a_net = -r_vec * (mu * inv_r3)

    # 2. J2 Oblateness (Zonal Harmonic) Perturbation
    if config.enable_j2 and body.j2 != 0.0 and body.radius > 0:
        R = body.radius
        j2 = body.j2
        z = r_vec.z
        z_sq = z * z
        r_sq = r * r
        inv_r5 = inv_r3 / r_sq

        factor = 1.5 * j2 * mu * (R * R) * inv_r5
        z_ratio = 5.0 * (z_sq / r_sq)

        ax_j2 = factor * r_vec.x * (z_ratio - 1.0)
        ay_j2 = factor * r_vec.y * (z_ratio - 1.0)
        az_j2 = factor * r_vec.z * (z_ratio - 3.0)

        a_net = a_net + Vector3(ax_j2, ay_j2, az_j2)

    # 3. Atmospheric Drag Perturbation: a_drag = -0.5 * rho * (Cd*A/m) * v_rel * v_rel_vec
    if config.enable_drag and body.radius > 0:
        altitude = r - body.radius
        if altitude < 1000.0:
            rho = atmospheric_density_earth(altitude)
            # Area in km^2: cross_section (m^2) * 1e-6
            area_km2 = config.spacecraft.cross_section * 1e-6
            mass_kg = config.spacecraft.mass
            cd = config.spacecraft.drag_coefficient

            # Atmosphere co-rotation velocity: v_rel = v - omega x r
            # Earth rotation rate ~ 7.292115e-5 rad/s
            omega_earth = Vector3(0.0, 0.0, 7.292115e-5)
            v_rel = v_vec - omega_earth.cross(r_vec)
            v_rel_mag = v_rel.norm()

            if v_rel_mag > 1e-6 and rho > 0:
                drag_factor = 0.5 * rho * (cd * area_km2 / mass_kg) * v_rel_mag
                a_drag = -v_rel * drag_factor
                a_net = a_net + a_drag

    # 4. Solar Radiation Pressure (SRP)
    if config.enable_srp:
        # Nominal solar pressure P0 = 4.56e-6 N/m^2 = 4.56e-6 kg / (m * s^2)
        # In km units: 1 N = 1 kg * m / s^2 = 1e-3 kg * km / s^2
        # Area in km^2: A_m2 * 1e-6
        area_m2 = config.spacecraft.cross_section
        mass_kg = config.spacecraft.mass
        cr = config.spacecraft.reflectivity
        p0_kms = 4.56e-9  # kN / m^2 = kg * km / (s^2 * m^2)

        srp_accel_mag = (p0_kms * cr * area_m2) / mass_kg
        # Assuming Sun lies along +X axis at 1 AU for canonical model
        sun_dir = Vector3(1.0, 0.0, 0.0)
        a_net = a_net + (sun_dir * srp_accel_mag)

    return a_net


@dataclass
class PropagationResult:
    """
    Trajectory time-series produced by numerical orbit propagation.
    """
    trajectory: List[StateVector]
    times: List[float]
    energies: List[float]
    final_state: StateVector
    step_count: int

    @property
    def total_duration(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    @property
    def max_energy_drift(self) -> float:
        """Maximum absolute deviation in specific mechanical energy from initial value."""
        if not self.energies:
            return 0.0
        e0 = self.energies[0]
        return max(abs(e - e0) for e in self.energies)


def propagate_symplectic_verlet(
    initial_state: StateVector,
    duration: float,
    dt: float,
    config: Optional[PerturbationConfig] = None,
) -> PropagationResult:
    """
    Geometric Symplectic Störmer-Verlet (Velocity-Verlet / Leapfrog) numerical propagator.
    Strictly preserves the symplectic 2-form and maintains bounded Hamiltonian energy
    oscillation without secular artificial dissipation or accumulation over thousands of orbits.
    """
    if config is None:
        config = PerturbationConfig(enable_j2=False, enable_drag=False, enable_srp=False)

    if dt <= 0:
        raise ValueError("Time step dt must be positive")

    num_steps = max(1, int(math.ceil(abs(duration) / dt)))
    step_dt = math.copysign(dt, duration)

    r_curr = initial_state.r
    v_curr = initial_state.v
    t_curr = initial_state.time

    trajectory: List[StateVector] = [initial_state]
    times: List[float] = [t_curr]
    e0 = 0.5 * v_curr.norm_squared() - (config.body.mu / r_curr.norm())
    energies: List[float] = [e0]

    # Initial acceleration: a_0 = a(r_0)
    a_curr = compute_acceleration(r_curr, v_curr, config)

    half_dt = 0.5 * step_dt
    dt_sq_half = 0.5 * step_dt * step_dt

    for _ in range(num_steps):
        # 1. Update position: r_{n+1} = r_n + v_n*dt + 0.5*a_n*dt^2
        r_next = r_curr + v_curr * step_dt + a_curr * dt_sq_half

        # 2. Evaluate acceleration at new position: a_{n+1} = a(r_{n+1})
        # For velocity-independent conservative forces (gravity + J2)
        v_estimate = v_curr + a_curr * half_dt
        a_next = compute_acceleration(r_next, v_estimate, config)

        # 3. Update velocity: v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt
        v_next = v_curr + (a_curr + a_next) * half_dt
        t_curr += step_dt

        r_curr = r_next
        v_curr = v_next
        a_curr = a_next

        current_energy = 0.5 * v_curr.norm_squared() - (config.body.mu / r_curr.norm())
        current_state = StateVector(r=r_curr, v=v_curr, time=t_curr)

        trajectory.append(current_state)
        times.append(t_curr)
        energies.append(current_energy)

    return PropagationResult(
        trajectory=trajectory,
        times=times,
        energies=energies,
        final_state=trajectory[-1],
        step_count=num_steps,
    )


def propagate_rk45(
    initial_state: StateVector,
    duration: float,
    initial_dt: float = 60.0,
    min_dt: float = 0.01,
    max_dt: float = 300.0,
    tolerance: float = 1e-9,
    config: Optional[PerturbationConfig] = None,
) -> PropagationResult:
    """
    Adaptive step-size Runge-Kutta-Fehlberg 4(5) (Dormand-Prince / Fehlberg) integrator.
    Monitors local truncation error at every step and dynamically adjusts time step dt
    to maintain bounded integration error across changing orbital curvature (periapsis vs apoapsis).
    """
    if config is None:
        config = PerturbationConfig(enable_j2=True)

    # Cash-Karp Runge-Kutta Fehlberg coefficients
    c = [0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0]
    a = [
        [],
        [1.0 / 5.0],
        [3.0 / 40.0, 9.0 / 40.0],
        [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0],
        [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0],
        [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0],
    ]
    # 5th-order weights
    b5 = [37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0]
    # 4th-order weights
    b4 = [2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0]

    r_curr = initial_state.r
    v_curr = initial_state.v
    t_curr = initial_state.time
    t_end = t_curr + duration

    dt = initial_dt
    trajectory: List[StateVector] = [initial_state]
    times: List[float] = [t_curr]
    e0 = 0.5 * v_curr.norm_squared() - (config.body.mu / r_curr.norm())
    energies: List[float] = [e0]
    total_steps = 0

    while (t_curr < t_end and duration > 0) or (t_curr > t_end and duration < 0):
        # Don't step past t_end
        if abs(t_end - t_curr) < abs(dt):
            dt = t_end - t_curr

        # Compute 6 stages: k_r = v, k_v = acceleration
        k_r: List[Vector3] = []
        k_v: List[Vector3] = []

        for i in range(6):
            r_stage = r_curr
            v_stage = v_curr
            for j in range(i):
                r_stage = r_stage + k_r[j] * (a[i][j] * dt)
                v_stage = v_stage + k_v[j] * (a[i][j] * dt)

            accel_stage = compute_acceleration(r_stage, v_stage, config)
            k_r.append(v_stage)
            k_v.append(accel_stage)

        # Candidate 5th-order and 4th-order solutions
        r5 = r_curr
        v5 = v_curr
        r4 = r_curr
        v4 = v_curr

        for i in range(6):
            r5 = r5 + k_r[i] * (b5[i] * dt)
            v5 = v5 + k_v[i] * (b5[i] * dt)
            r4 = r4 + k_r[i] * (b4[i] * dt)
            v4 = v4 + k_v[i] * (b4[i] * dt)

        # Estimate local truncation error
        err_r = (r5 - r4).norm()
        err_v = (v5 - v4).norm()
        error = max(err_r / (r_curr.norm() + 1.0), err_v / (v_curr.norm() + 1.0))

        # Check error tolerance
        if error <= tolerance or abs(dt) <= min_dt:
            # Step accepted
            r_curr = r5
            v_curr = v5
            t_curr += dt
            total_steps += 1

            current_energy = 0.5 * v_curr.norm_squared() - (config.body.mu / r_curr.norm())
            current_state = StateVector(r=r_curr, v=v_curr, time=t_curr)
            trajectory.append(current_state)
            times.append(t_curr)
            energies.append(current_energy)

        # Adapt step size for next step (or retry if rejected)
        if error > 0:
            scale = 0.9 * ((tolerance / error) ** 0.2)
            scale = max(0.2, min(2.0, scale))
            dt = math.copysign(max(min_dt, min(max_dt, abs(dt) * scale)), duration)
        else:
            dt = math.copysign(min(max_dt, abs(dt) * 1.5), duration)

    return PropagationResult(
        trajectory=trajectory,
        times=times,
        energies=energies,
        final_state=trajectory[-1],
        step_count=total_steps,
    )
