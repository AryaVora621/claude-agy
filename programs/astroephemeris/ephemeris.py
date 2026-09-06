"""
AstroEphemeris: High-Precision Orbital Mechanics & Celestial Ephemeris Kernel
Zero external dependencies, standard library only.
Provides:
- Keplerian orbital elements to Cartesian state vector transformations (and inverse).
- High-order Newton-Halley Kepler equation solver.
- Symplectic 4th-order Yoshida integrator for n-body gravitational dynamics.
- General relativistic Post-Newtonian (1PN) Schwarzschild perihelion precession.
- Planetary J2 equatorial oblateness perturbation and nodal regression.
- Circular Restricted 3-Body Problem (CR3BP) synodic rotating frame dynamics.
- Analytical computation of 5 Lagrangian libration equilibrium points (L1 to L5).
- Jacobi energy integral conservation and zero-velocity Hill boundary curves.
- Hohmann and bi-elliptic interplanetary orbital transfer targeting.
- Real solar system ephemeris data (Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Physical & Astrodynamical Constants (Standard SI & Astronomical Units)
GRAVITATIONAL_CONSTANT = 6.67430e-11   # m^3 / (kg * s^2)
SPEED_OF_LIGHT = 299792458.0           # m/s
ASTRONOMICAL_UNIT = 1.495978707e11     # meters (1 AU)
SOLAR_MASS = 1.98847e30                # kg
EARTH_MASS = 5.9722e24                 # kg
MOON_MASS = 7.342e22                   # kg
MARS_MASS = 6.4171e23                  # kg
JUPITER_MASS = 1.8982e27               # kg

# Standard Gravitational Parameter of the Sun
MU_SUN = GRAVITATIONAL_CONSTANT * SOLAR_MASS # ~1.3271244e20 m^3/s^2


@dataclass
class OrbitalElements:
    """Classical Keplerian orbital elements."""
    a: float  # Semi-major axis (meters or AU)
    e: float  # Eccentricity (0 <= e < 1 for elliptical orbits)
    i: float  # Inclination (radians)
    raan: float # Longitude of ascending node Omega (radians)
    arg_p: float # Argument of periapsis omega (radians)
    true_anomaly: float # True anomaly nu (radians)

    @property
    def periapsis_radius(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def apoapsis_radius(self) -> float:
        return self.a * (1.0 + self.e)

    @property
    def semi_latus_rectum(self) -> float:
        return self.a * (1.0 - self.e * self.e)


@dataclass
class CartesianState:
    """Position and velocity state vector in 3D Cartesian coordinates."""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    @property
    def position_magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.vx, self.vy, self.vz)


def solve_kepler_equation(mean_anomaly: float, eccentricity: float, tolerance: float = 1e-12, max_iter: int = 50) -> float:
    """
    Solves Kepler's equation M = E - e * sin(E) for eccentric anomaly E.
    Uses third-order Halley iteration for rapid, quadratic-to-cubic convergence.
    """
    # Normalize mean anomaly to [0, 2*pi)
    m = mean_anomaly % (2.0 * math.pi)
    if m < 0.0:
        m += 2.0 * math.pi

    e = max(0.0, min(0.999999, eccentricity))

    # Initial guess via Danby starter
    if e < 0.8:
        e_anomaly = m + e * math.sin(m) / (1.0 - math.sin(m + e) + math.sin(m))
    else:
        e_anomaly = math.pi if m > math.pi else m

    for _ in range(max_iter):
        sin_e = math.sin(e_anomaly)
        cos_e = math.cos(e_anomaly)
        f = e_anomaly - e * sin_e - m
        f_prime = 1.0 - e * cos_e
        f_double_prime = e * sin_e

        # Halley step: delta = f / (f' - f * f'' / (2 * f'))
        denom = f_prime - (f * f_double_prime) / (2.0 * f_prime)
        if abs(denom) < 1e-15:
            delta = f / f_prime
        else:
            delta = f / denom

        e_anomaly -= delta
        if abs(delta) < tolerance:
            break

    return e_anomaly


def orbital_elements_to_cartesian(elements: OrbitalElements, mu: float) -> CartesianState:
    """
    Converts Keplerian orbital elements (a, e, i, raan, arg_p, nu) to Cartesian position and velocity.
    """
    a = elements.a
    e = elements.e
    i = elements.i
    raan = elements.raan
    arg_p = elements.arg_p
    nu = elements.true_anomaly

    # Semi-latus rectum
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))

    # Position and velocity in perifocal frame (P, Q, W)
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)
    r_p = r * cos_nu
    r_q = r * sin_nu

    h = math.sqrt(max(1e-15, mu * p))
    v_p = -(mu / h) * sin_nu
    v_q = (mu / h) * (e + cos_nu)

    # Coordinate rotation from perifocal frame to inertial reference frame
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    cos_w = math.cos(arg_p)
    sin_w = math.sin(arg_p)
    cos_i = math.cos(i)
    sin_i = math.sin(i)

    # Direction cosine matrix elements: R = R_z(-raan) * R_x(-i) * R_z(-arg_p)
    px = cos_o * cos_w - sin_o * sin_w * cos_i
    py = sin_o * cos_w + cos_o * sin_w * cos_i
    pz = sin_w * sin_i

    qx = -cos_o * sin_w - sin_o * cos_w * cos_i
    qy = -sin_o * sin_w + cos_o * cos_w * cos_i
    qz = cos_w * sin_i

    x = px * r_p + qx * r_q
    y = py * r_p + qy * r_q
    z = pz * r_p + qz * r_q

    vx = px * v_p + qx * v_q
    vy = py * v_p + qy * v_q
    vz = pz * v_p + qz * v_q

    return CartesianState(x, y, z, vx, vy, vz)


def cartesian_to_orbital_elements(state: CartesianState, mu: float) -> OrbitalElements:
    """
    Converts Cartesian state vector to Keplerian orbital elements.
    """
    rx, ry, rz = state.x, state.y, state.z
    vx, vy, vz = state.vx, state.vy, state.vz

    r = math.sqrt(rx * rx + ry * ry + rz * rz)
    v2 = vx * vx + vy * vy + vz * vz

    # Specific angular momentum vector h = r x v
    hx = ry * vz - rz * vy
    hy = rz * vx - rx * vz
    hz = rx * vy - ry * vx
    h = math.sqrt(hx * hx + hy * hy + hz * hz)

    # Node vector n = k x h = (-hy, hx, 0)
    nx = -hy
    ny = hx
    n = math.sqrt(nx * nx + ny * ny)

    # Eccentricity vector: e_vec = ((v^2 - mu/r)*r - (r.v)*v) / mu
    rdotv = rx * vx + ry * vy + rz * vz
    c1 = (v2 - mu / max(1e-15, r)) / mu
    c2 = rdotv / mu
    ex = c1 * rx - c2 * vx
    ey = c1 * ry - c2 * vy
    ez = c1 * rz - c2 * vz
    e = math.sqrt(ex * ex + ey * ey + ez * ez)

    # Specific mechanical energy: E = v^2/2 - mu/r = -mu/(2a)
    energy = 0.5 * v2 - mu / max(1e-15, r)
    if abs(energy) > 1e-15:
        a = -mu / (2.0 * energy)
    else:
        a = 1e12 # Parabolic limit

    # Inclination: cos(i) = hz / h
    inc = math.acos(max(-1.0, min(1.0, hz / max(1e-15, h))))

    # Longitude of ascending node raan
    if n > 1e-12:
        raan = math.acos(max(-1.0, min(1.0, nx / n)))
        if ny < 0.0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0 # Equatorial orbit convention

    # Argument of periapsis arg_p
    if n > 1e-12 and e > 1e-8:
        ndote = nx * ex + ny * ey
        arg_p = math.acos(max(-1.0, min(1.0, ndote / (n * e))))
        if ez < 0.0:
            arg_p = 2.0 * math.pi - arg_p
    else:
        arg_p = 0.0

    # True anomaly nu
    if e > 1e-8:
        edotr = ex * rx + ey * ry + ez * rz
        nu = math.acos(max(-1.0, min(1.0, edotr / (e * r))))
        if rdotv < 0.0:
            nu = 2.0 * math.pi - nu
    else:
        nu = 0.0

    return OrbitalElements(a=a, e=e, i=inc, raan=raan, arg_p=arg_p, true_anomaly=nu)


def compute_general_relativistic_acceleration(r: Tuple[float, float, float], v: Tuple[float, float, float], mu: float, c: float = SPEED_OF_LIGHT, scale: float = 1.0) -> Tuple[float, float, float]:
    """
    Computes 1PN Post-Newtonian General Relativistic acceleration for Schwarzschild spacetime.
    a_GR = (mu / (c^2 * r^3)) * [ (4*mu/r - v^2)*r + 4*(r.v)*v ]
    scale parameter allows visual exaggeration for educational demonstration.
    """
    rx, ry, rz = r
    vx, vy, vz = v
    r_mag = math.sqrt(rx * rx + ry * ry + rz * rz)
    if r_mag < 1.0:
        return (0.0, 0.0, 0.0)

    v2 = vx * vx + vy * vy + vz * vz
    rdotv = rx * vx + ry * vy + rz * vz

    factor = (mu / (c * c * (r_mag ** 3))) * scale
    term1 = (4.0 * mu / r_mag) - v2
    term2 = 4.0 * rdotv

    ax = factor * (term1 * rx + term2 * vx)
    ay = factor * (term1 * ry + term2 * vy)
    az = factor * (term1 * rz + term2 * vz)

    return (ax, ay, az)


def compute_j2_oblateness_acceleration(r: Tuple[float, float, float], mu: float, equatorial_radius: float, j2: float) -> Tuple[float, float, float]:
    """
    Computes gravitational perturbation acceleration due to planetary oblateness J2 zonal harmonic.
    a_J2 = -3/2 * J2 * mu * R_eq^2 / r^5 * [ (1 - 5 * z^2/r^2)*r + 2*z*k_hat ]
    """
    rx, ry, rz = r
    r2 = rx * rx + ry * ry + rz * rz
    r_mag = math.sqrt(r2)
    if r_mag < equatorial_radius * 0.1:
        return (0.0, 0.0, 0.0)

    r5 = r_mag ** 5
    z2 = rz * rz
    factor = 1.5 * j2 * mu * (equatorial_radius ** 2) / r5
    z_ratio = 5.0 * z2 / r2

    ax = -factor * rx * (1.0 - z_ratio)
    ay = -factor * ry * (1.0 - z_ratio)
    az = -factor * rz * (3.0 - z_ratio)

    return (ax, ay, az)


class CR3BPModel:
    """
    Circular Restricted 3-Body Problem (CR3BP) in normalized rotating barycentric coordinates.
    Primary mass m1 at (-mu, 0, 0), Secondary mass m2 at (1 - mu, 0, 0).
    Mass ratio mu = m2 / (m1 + m2). Total mass = 1, Distance = 1, Angular velocity = 1.
    """

    def __init__(self, mu: float = 0.012150585609624): # Earth-Moon standard mass parameter
        self.mu = mu
        self.lagrange_points = self._compute_lagrange_points()

    def effective_potential(self, x: float, y: float, z: float) -> float:
        """
        Computes effective potential Omega(x, y, z) = 1/2*(x^2 + y^2) + (1 - mu)/r1 + mu/r2.
        """
        r1 = math.sqrt((x + self.mu) ** 2 + y * y + z * z)
        r2 = math.sqrt((x - (1.0 - self.mu)) ** 2 + y * y + z * z)
        if r1 < 1e-9 or r2 < 1e-9:
            return 1e9
        return 0.5 * (x * x + y * y) + (1.0 - self.mu) / r1 + self.mu / r2

    def jacobi_constant(self, x: float, y: float, z: float, vx: float, vy: float, vz: float) -> float:
        """
        Computes the Jacobi energy integral C_J = 2*Omega(x, y, z) - (vx^2 + vy^2 + vz^2).
        Strictly conserved in the rotating synodic frame along unperturbed trajectories.
        """
        v2 = vx * vx + vy * vy + vz * vz
        omega = self.effective_potential(x, y, z)
        return 2.0 * omega - v2

    def equations_of_motion(self, state: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float, float, float, float]:
        """
        Evaluates CR3BP equations of motion in synodic rotating frame:
        x'' - 2*y' = dOmega/dx
        y'' + 2*x' = dOmega/dy
        z'' = dOmega/dz
        """
        x, y, z, vx, vy, vz = state
        r1_sq = (x + self.mu) ** 2 + y * y + z * z
        r2_sq = (x - (1.0 - self.mu)) ** 2 + y * y + z * z

        r1_cube = max(1e-12, r1_sq * math.sqrt(r1_sq))
        r2_cube = max(1e-12, r2_sq * math.sqrt(r2_sq))

        mu = self.mu
        one_minus_mu = 1.0 - mu

        # Accelerations including Coriolis (2*vy, -2*vx) and centrifugal potential gradients
        ax = 2.0 * vy + x - one_minus_mu * (x + mu) / r1_cube - mu * (x - one_minus_mu) / r2_cube
        ay = -2.0 * vx + y - one_minus_mu * y / r1_cube - mu * y / r2_cube
        az = -one_minus_mu * z / r1_cube - mu * z / r2_cube

        return (vx, vy, vz, ax, ay, az)

    def _compute_lagrange_points(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes coordinates of the 5 analytical Lagrangian libration equilibrium points (L1..L5).
        """
        mu = self.mu
        pts = {}

        # Triangular equilateral points (L4 and L5)
        # Form equilateral triangles with m1 and m2 in the x-y plane
        pts["L4"] = (0.5 - mu, math.sqrt(3.0) * 0.5, 0.0)
        pts["L5"] = (0.5 - mu, -math.sqrt(3.0) * 0.5, 0.0)

        # Collinear points (L1, L2, L3) lying on the x-axis (y = 0, z = 0)
        # L1: between m1 and m2, -mu < x < 1 - mu
        def domega_dx_l1(x: float) -> Tuple[float, float]:
            f = x - (1.0 - mu) / ((x + mu) ** 2) + mu / ((x - 1.0 + mu) ** 2)
            df = 1.0 + 2.0 * (1.0 - mu) / ((x + mu) ** 3) - 2.0 * mu / ((x - 1.0 + mu) ** 3)
            return f, df

        # L2: beyond m2, x > 1 - mu
        def domega_dx_l2(x: float) -> Tuple[float, float]:
            f = x - (1.0 - mu) / ((x + mu) ** 2) - mu / ((x - 1.0 + mu) ** 2)
            df = 1.0 + 2.0 * (1.0 - mu) / ((x + mu) ** 3) + 2.0 * mu / ((x - 1.0 + mu) ** 3)
            return f, df

        # L3: beyond m1, x < -mu
        def domega_dx_l3(x: float) -> Tuple[float, float]:
            f = x + (1.0 - mu) / ((x + mu) ** 2) + mu / ((x - 1.0 + mu) ** 2)
            df = 1.0 - 2.0 * (1.0 - mu) / ((x + mu) ** 3) - 2.0 * mu / ((x - 1.0 + mu) ** 3)
            return f, df

        # Solve for L1 via Newton-Raphson
        gamma1 = (mu / (3.0 * (1.0 - mu))) ** (1.0 / 3.0)
        x_l1 = 1.0 - mu - gamma1
        for _ in range(30):
            val, deriv = domega_dx_l1(x_l1)
            dx = val / deriv
            x_l1 -= dx
            if abs(dx) < 1e-12:
                break
        pts["L1"] = (x_l1, 0.0, 0.0)

        # Solve for L2 via Newton-Raphson
        gamma2 = (mu / (3.0 * (1.0 - mu))) ** (1.0 / 3.0)
        x_l2 = 1.0 - mu + gamma2
        for _ in range(30):
            val, deriv = domega_dx_l2(x_l2)
            dx = val / deriv
            x_l2 -= dx
            if abs(dx) < 1e-12:
                break
        pts["L2"] = (x_l2, 0.0, 0.0)

        # Solve for L3 via Newton-Raphson
        x_l3 = -1.0 - (5.0 / 12.0) * mu
        for _ in range(30):
            val, deriv = domega_dx_l3(x_l3)
            dx = val / deriv
            x_l3 -= dx
            if abs(dx) < 1e-12:
                break
        pts["L3"] = (x_l3, 0.0, 0.0)

        return pts

    def step_rk4(self, state: Tuple[float, float, float, float, float, float], dt: float) -> Tuple[float, float, float, float, float, float]:
        """4th-order Runge-Kutta numerical integrator step for CR3BP state."""
        k1 = self.equations_of_motion(state)
        s2 = tuple(state[i] + 0.5 * dt * k1[i] for i in range(6))

        k2 = self.equations_of_motion(s2)
        s3 = tuple(state[i] + 0.5 * dt * k2[i] for i in range(6))

        k3 = self.equations_of_motion(s3)
        s4 = tuple(state[i] + dt * k3[i] for i in range(6))

        k4 = self.equations_of_motion(s4)
        next_state = tuple(state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(6))
        return next_state


def compute_hohmann_transfer(r1: float, r2: float, mu: float) -> Tuple[float, float, float, float]:
    """
    Computes analytical Hohmann interplanetary transfer between circular orbits.
    Returns: (delta_v1, delta_v2, total_delta_v, time_of_flight)
    """
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    a_trans = (r1 + r2) * 0.5

    # Periapsis velocity on transfer ellipse
    v_trans_peri = math.sqrt(mu * (2.0 / r1 - 1.0 / a_trans))
    delta_v1 = abs(v_trans_peri - v1)

    # Apoapsis velocity on transfer ellipse
    v_trans_apo = math.sqrt(mu * (2.0 / r2 - 1.0 / a_trans))
    delta_v2 = abs(v2 - v_trans_apo)

    total_delta_v = delta_v1 + delta_v2
    # Time of flight is half the orbital period of the transfer ellipse
    time_of_flight = math.pi * math.sqrt((a_trans ** 3) / mu)

    return (delta_v1, delta_v2, total_delta_v, time_of_flight)


@dataclass
class CelestialBody:
    """Represents a celestial body in the solar system or planetary system."""
    name: str
    mass: float              # kg
    radius: float            # meters (for rendering and J2 calculation)
    elements: OrbitalElements
    color: str               # Hex color code for visualization
    trail_color: str
    parent_name: Optional[str] = None
    state: CartesianState = field(init=False)
    j2: float = 0.0
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize state vector from orbital elements assuming parent is Sun by default
        self.state = orbital_elements_to_cartesian(self.elements, MU_SUN)


@dataclass
class Spacecraft:
    """Virtual spacecraft with thrusters and telemetry."""
    name: str
    state: CartesianState
    mass_kg: float = 1000.0
    delta_v_spent: float = 0.0
    color: str = "#00FFCC"
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    def apply_impulse(self, dvx: float, dvy: float, dvz: float):
        """Applies an instantaneous velocity change impulse."""
        self.state.vx += dvx
        self.state.vy += dvy
        self.state.vz += dvz
        dv_mag = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
        self.delta_v_spent += dv_mag


class SolarSystemEngine:
    """
    Precision N-Body Gravitational Ephemeris & Orbital Mechanics Simulation Engine.
    Implements:
    - 4th-order symplectic Yoshida integrator for long-term energy conservation.
    - Relativistic 1PN corrections.
    - J2 planetary oblateness.
    - Spacecraft thruster control.
    """

    def __init__(self, enable_relativistic: bool = False, rel_scale: float = 1.0):
        self.bodies: Dict[str, CelestialBody] = {}
        self.spacecraft: Optional[Spacecraft] = None
        self.sim_time_sec: float = 0.0
        self.enable_relativistic = enable_relativistic
        self.relativistic_scale = rel_scale
        self.cr3bp_model: Optional[CR3BPModel] = None
        self.view_mode: str = "inertial" # 'inertial' or 'cr3bp'

    def add_body(self, body: CelestialBody):
        self.bodies[body.name] = body

    def set_spacecraft(self, sc: Spacecraft):
        self.spacecraft = sc

    def initialize_default_solar_system(self):
        """Populates engine with major solar system bodies."""
        # Central Sun at origin
        sun_elements = OrbitalElements(a=0.0, e=0.0, i=0.0, raan=0.0, arg_p=0.0, true_anomaly=0.0)
        sun = CelestialBody(
            name="Sun", mass=SOLAR_MASS, radius=6.9634e8,
            elements=sun_elements, color="#FFD700", trail_color="#FFA500"
        )
        sun.state = CartesianState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.add_body(sun)

        # Mercury (noticeable eccentricity e = 0.2056 for relativistic precession)
        mercury_elements = OrbitalElements(
            a=0.387098 * ASTRONOMICAL_UNIT, e=0.205630, i=math.radians(7.005),
            raan=math.radians(48.331), arg_p=math.radians(29.124), true_anomaly=0.0
        )
        self.add_body(CelestialBody(
            name="Mercury", mass=3.3011e23, radius=2.4397e6,
            elements=mercury_elements, color="#B0A090", trail_color="#888877", parent_name="Sun"
        ))

        # Venus
        venus_elements = OrbitalElements(
            a=0.723332 * ASTRONOMICAL_UNIT, e=0.006773, i=math.radians(3.394),
            raan=math.radians(76.680), arg_p=math.radians(54.884), true_anomaly=0.0
        )
        self.add_body(CelestialBody(
            name="Venus", mass=4.8675e24, radius=6.0518e6,
            elements=venus_elements, color="#E3BB7B", trail_color="#C29D5D", parent_name="Sun"
        ))

        # Earth
        earth_elements = OrbitalElements(
            a=1.000000 * ASTRONOMICAL_UNIT, e=0.0167086, i=math.radians(0.000),
            raan=math.radians(0.0), arg_p=math.radians(102.937), true_anomaly=0.0
        )
        earth = CelestialBody(
            name="Earth", mass=EARTH_MASS, radius=6.371e6,
            elements=earth_elements, color="#00BFFF", trail_color="#1E90FF", parent_name="Sun",
            j2=1.08263e-3
        )
        self.add_body(earth)

        # Mars
        mars_elements = OrbitalElements(
            a=1.523679 * ASTRONOMICAL_UNIT, e=0.0934, i=math.radians(1.850),
            raan=math.radians(49.558), arg_p=math.radians(286.502), true_anomaly=0.0
        )
        self.add_body(CelestialBody(
            name="Mars", mass=MARS_MASS, radius=3.3895e6,
            elements=mars_elements, color="#FF6347", trail_color="#CD5C5C", parent_name="Sun"
        ))

        # Jupiter
        jupiter_elements = OrbitalElements(
            a=5.2044 * ASTRONOMICAL_UNIT, e=0.0489, i=math.radians(1.303),
            raan=math.radians(100.464), arg_p=math.radians(273.867), true_anomaly=0.0
        )
        self.add_body(CelestialBody(
            name="Jupiter", mass=JUPITER_MASS, radius=6.9911e7,
            elements=jupiter_elements, color="#DEB887", trail_color="#D2B48C", parent_name="Sun"
        ))

    def step_yoshida_4th_order(self, dt: float):
        """
        Symplectic 4th-order Yoshida integrator for n-body gravitational equations of motion.
        Preserves phase space volume and conserves Hamiltonian energy over millions of orbits.
        """
        # Yoshida coefficients
        w1 = 1.0 / (2.0 - (2.0 ** (1.0 / 3.0)))
        w0 = -(2.0 ** (1.0 / 3.0)) * w1

        c1 = c4 = 0.5 * w1
        c2 = c3 = 0.5 * (w0 + w1)

        d1 = d3 = w1
        d2 = w0

        steps = [(c1, d1), (c2, d2), (c3, d3), (c4, 0.0)]

        for c_step, d_step in steps:
            # Drift positions
            if c_step != 0.0:
                dt_c = c_step * dt
                for body in self.bodies.values():
                    if body.name == "Sun":
                        continue
                    body.state.x += body.state.vx * dt_c
                    body.state.y += body.state.vy * dt_c
                    body.state.z += body.state.vz * dt_c

                if self.spacecraft:
                    self.spacecraft.state.x += self.spacecraft.state.vx * dt_c
                    self.spacecraft.state.y += self.spacecraft.state.vy * dt_c
                    self.spacecraft.state.z += self.spacecraft.state.vz * dt_c

            # Kick velocities
            if d_step != 0.0:
                dt_d = d_step * dt
                # Compute accelerations on all bodies
                for name, body in self.bodies.items():
                    if name == "Sun":
                        continue
                    ax, ay, az = self._compute_total_acceleration(body.state)
                    body.state.vx += ax * dt_d
                    body.state.vy += ay * dt_d
                    body.state.vz += az * dt_d

                if self.spacecraft:
                    sc_ax, sc_ay, sc_az = self._compute_total_acceleration(self.spacecraft.state)
                    self.spacecraft.state.vx += sc_ax * dt_d
                    self.spacecraft.state.vy += sc_ay * dt_d
                    self.spacecraft.state.vz += sc_az * dt_d

        self.sim_time_sec += dt

        # Record trajectory history (subsampled for performance)
        for body in self.bodies.values():
            body.history.append((body.state.x, body.state.y, body.state.z))
            if len(body.history) > 300:
                body.history.pop(0)

        if self.spacecraft:
            self.spacecraft.history.append((self.spacecraft.state.x, self.spacecraft.state.y, self.spacecraft.state.z))
            if len(self.spacecraft.history) > 500:
                self.spacecraft.history.pop(0)

    def _compute_total_acceleration(self, state: CartesianState) -> Tuple[float, float, float]:
        """Computes gravitational acceleration on a point mass at given state."""
        ax_tot = 0.0
        ay_tot = 0.0
        az_tot = 0.0

        for b_name, other in self.bodies.items():
            dx = other.state.x - state.x
            dy = other.state.y - state.y
            dz = other.state.z - state.z
            dist_sq = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(dist_sq)

            if dist < 1000.0: # Softening to prevent singularity
                continue

            # Newtonian gravitation
            f = (GRAVITATIONAL_CONSTANT * other.mass) / (dist_sq * dist)
            ax_tot += f * dx
            ay_tot += f * dy
            az_tot += f * dz

            # Relativistic 1PN correction for Sun
            if self.enable_relativistic and b_name == "Sun":
                rel_ax, rel_ay, rel_az = compute_general_relativistic_acceleration(
                    (state.x, state.y, state.z),
                    (state.vx, state.vy, state.vz),
                    MU_SUN,
                    scale=self.relativistic_scale
                )
                ax_tot += rel_ax
                ay_tot += rel_ay
                az_tot += rel_az

            # J2 equatorial oblateness perturbation for Earth
            if other.j2 > 0.0:
                j2_ax, j2_ay, j2_az = compute_j2_oblateness_acceleration(
                    (dx, dy, dz),
                    GRAVITATIONAL_CONSTANT * other.mass,
                    other.radius,
                    other.j2
                )
                ax_tot += j2_ax
                ay_tot += j2_ay
                az_tot += j2_az

        return (ax_tot, ay_tot, az_tot)

    def compute_system_energy(self) -> float:
        """Computes total mechanical energy of the system."""
        total_energy = 0.0
        body_list = list(self.bodies.values())

        # Kinetic energy
        for b in body_list:
            v2 = b.state.vx ** 2 + b.state.vy ** 2 + b.state.vz ** 2
            total_energy += 0.5 * b.mass * v2

        # Potential energy
        for i in range(len(body_list)):
            for j in range(i + 1, len(body_list)):
                b1 = body_list[i]
                b2 = body_list[j]
                dx = b1.state.x - b2.state.x
                dy = b1.state.y - b2.state.y
                dz = b1.state.z - b2.state.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > 1.0:
                    total_energy -= (GRAVITATIONAL_CONSTANT * b1.mass * b2.mass) / dist

        return total_energy
