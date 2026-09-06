"""
AstroEphemeris: Aerospace Presets & Orbital Scenarios
Zero external dependencies, standard library only.
Provides curated orbital scenarios:
- Inner Solar System (Mercury, Venus, Earth, Mars).
- Full Planetary System (Sun to Saturn).
- Earth-Moon CR3BP with L1-L5 Lagrangian Libration Points.
- Sun-Earth James Webb Space Telescope (JWST) L2 Halo Orbit.
- Mercury General Relativistic 1PN Perihelion Precession Rosette.
- Earth-to-Mars Hohmann Interplanetary Transfer Orbit.
- Jupiter Trojan Asteroids Swarm (L4 and L5 Libration Clusters).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from ephemeris import (
    ASTRONOMICAL_UNIT,
    GRAVITATIONAL_CONSTANT,
    SOLAR_MASS,
    EARTH_MASS,
    MOON_MASS,
    MARS_MASS,
    JUPITER_MASS,
    MU_SUN,
    OrbitalElements,
    CartesianState,
    CelestialBody,
    Spacecraft,
    CR3BPModel,
    compute_hohmann_transfer
)


@dataclass
class ScenarioPreset:
    key: str
    name: str
    description: str
    view_mode: str  # 'inertial' or 'cr3bp'
    time_step_sec: float
    camera_distance: float
    enable_relativistic: bool = False
    relativistic_scale: float = 1.0
    cr3bp_mu: Optional[float] = None
    bodies: List[CelestialBody] = None
    spacecraft: Optional[Spacecraft] = None
    initial_cr3bp_state: Optional[List[float]] = None


PRESETS: Dict[str, ScenarioPreset] = {}


def load_inner_solar_system() -> ScenarioPreset:
    bodies = []
    # Sun
    sun_elem = OrbitalElements(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sun = CelestialBody("Sun", SOLAR_MASS, 6.9634e8, sun_elem, "#FFD700", "#FFA500")
    sun.state = CartesianState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bodies.append(sun)

    # Mercury
    merc_elem = OrbitalElements(0.387098 * ASTRONOMICAL_UNIT, 0.205630, math.radians(7.0), math.radians(48.3), math.radians(29.1), 0.0)
    bodies.append(CelestialBody("Mercury", 3.3011e23, 2.4397e6, merc_elem, "#A0A0A0", "#777777", "Sun"))

    # Venus
    venus_elem = OrbitalElements(0.723332 * ASTRONOMICAL_UNIT, 0.006773, math.radians(3.4), math.radians(76.7), math.radians(54.9), 0.0)
    bodies.append(CelestialBody("Venus", 4.8675e24, 6.0518e6, venus_elem, "#E3BB7B", "#C29D5D", "Sun"))

    # Earth
    earth_elem = OrbitalElements(1.000000 * ASTRONOMICAL_UNIT, 0.016708, 0.0, 0.0, math.radians(102.9), 0.0)
    bodies.append(CelestialBody("Earth", EARTH_MASS, 6.371e6, earth_elem, "#00BFFF", "#1E90FF", "Sun", j2=1.08263e-3))

    # Mars
    mars_elem = OrbitalElements(1.523679 * ASTRONOMICAL_UNIT, 0.0934, math.radians(1.85), math.radians(49.6), math.radians(286.5), 0.0)
    bodies.append(CelestialBody("Mars", MARS_MASS, 3.3895e6, mars_elem, "#FF6347", "#CD5C5C", "Sun"))

    return ScenarioPreset(
        key="inner_solar_system",
        name="Inner Solar System",
        description="Sun, Mercury, Venus, Earth, and Mars in high-precision Newtonian gravitational orbits.",
        view_mode="inertial",
        time_step_sec=86400.0 * 0.5, # 12 hours per step
        camera_distance=2.2 * ASTRONOMICAL_UNIT,
        bodies=bodies
    )


def load_earth_moon_cr3bp() -> ScenarioPreset:
    # Earth-Moon mass ratio mu = m2 / (m1 + m2)
    mu = MOON_MASS / (EARTH_MASS + MOON_MASS) # ~0.01215

    # Initial spacecraft state near L1 libration point
    # State format: [x, y, z, vx, vy, vz] in normalized rotating frame
    # L1 is around x = 0.8369
    cr3bp = CR3BPModel(mu)
    l1_x = cr3bp.lagrange_points["L1"][0]
    # Small offset for halo/quasi-periodic orbit
    initial_sc_state = [l1_x + 0.005, 0.0, 0.02, 0.0, 0.018, 0.0]

    return ScenarioPreset(
        key="earth_moon_cr3bp",
        name="Earth-Moon CR3BP & Libration Points",
        description="Rotating synodic frame with Earth-Moon masses, L1 to L5 Lagrange points, and Jacobi zero-velocity curves.",
        view_mode="cr3bp",
        time_step_sec=0.01, # Normalized dimensionless time
        camera_distance=1.6,
        cr3bp_mu=mu,
        initial_cr3bp_state=initial_sc_state
    )


def load_sun_earth_jwst_l2() -> ScenarioPreset:
    # Sun-Earth mass ratio
    mu = EARTH_MASS / (SOLAR_MASS + EARTH_MASS) # ~3.003e-6

    # L2 is at x ~ 1.0100
    cr3bp = CR3BPModel(mu)
    l2_x = cr3bp.lagrange_points["L2"][0]
    # Halo orbit injection vector around Sun-Earth L2
    initial_sc_state = [l2_x + 0.0025, 0.0, 0.0015, 0.0, 0.0085, 0.0]

    return ScenarioPreset(
        key="sun_earth_jwst_l2",
        name="Sun-Earth JWST at Lagrange L2",
        description="James Webb Space Telescope (JWST) operating in a halo orbit around the Sun-Earth L2 libration point.",
        view_mode="cr3bp",
        time_step_sec=0.005,
        camera_distance=1.4,
        cr3bp_mu=mu,
        initial_cr3bp_state=initial_sc_state
    )


def load_mercury_relativistic_precession() -> ScenarioPreset:
    bodies = []
    # Sun
    sun_elem = OrbitalElements(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sun = CelestialBody("Sun", SOLAR_MASS, 6.9634e8, sun_elem, "#FFD700", "#FFA500")
    sun.state = CartesianState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bodies.append(sun)

    # Mercury with eccentric orbit e = 0.25
    merc_elem = OrbitalElements(0.387098 * ASTRONOMICAL_UNIT, 0.25, 0.0, 0.0, 0.0, 0.0)
    bodies.append(CelestialBody("Mercury", 3.3011e23, 2.4397e6, merc_elem, "#00E5FF", "#0099AA", "Sun"))

    return ScenarioPreset(
        key="mercury_relativistic_precession",
        name="Mercury 1PN Relativistic Precession",
        description="General relativistic Post-Newtonian (1PN) Schwarzschild perihelion precession generating an orbital rosette.",
        view_mode="inertial",
        time_step_sec=86400.0 * 0.2,
        camera_distance=0.6 * ASTRONOMICAL_UNIT,
        enable_relativistic=True,
        relativistic_scale=2000.0, # Visual exaggeration factor to reveal rosette petals in minutes
        bodies=bodies
    )


def load_earth_mars_hohmann() -> ScenarioPreset:
    bodies = []
    # Sun
    sun_elem = OrbitalElements(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sun = CelestialBody("Sun", SOLAR_MASS, 6.9634e8, sun_elem, "#FFD700", "#FFA500")
    sun.state = CartesianState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bodies.append(sun)

    # Earth at r1 = 1.0 AU
    r1 = 1.0 * ASTRONOMICAL_UNIT
    earth_elem = OrbitalElements(r1, 0.0, 0.0, 0.0, 0.0, 0.0)
    bodies.append(CelestialBody("Earth", EARTH_MASS, 6.371e6, earth_elem, "#00BFFF", "#1E90FF", "Sun"))

    # Mars at r2 = 1.524 AU (phased at optimal departure angle)
    r2 = 1.523679 * ASTRONOMICAL_UNIT
    dv1, dv2, total_dv, tof = compute_hohmann_transfer(r1, r2, MU_SUN)
    # Target phase angle at departure = pi - omega_mars * tof
    omega_mars = math.sqrt(MU_SUN / (r2 ** 3))
    lead_angle = math.pi - omega_mars * tof
    mars_elem = OrbitalElements(r2, 0.0, 0.0, 0.0, 0.0, lead_angle)
    bodies.append(CelestialBody("Mars", MARS_MASS, 3.3895e6, mars_elem, "#FF6347", "#CD5C5C", "Sun"))

    # Spacecraft departing Earth along Hohmann transfer ellipse
    v1 = math.sqrt(MU_SUN / r1)
    v_trans_peri = v1 + dv1
    sc_state = CartesianState(r1, 0.0, 0.0, 0.0, v_trans_peri, 0.0)
    spacecraft = Spacecraft("Ares-1", sc_state, mass_kg=1200.0, color="#FF00FF")

    return ScenarioPreset(
        key="earth_mars_hohmann",
        name="Earth-to-Mars Hohmann Transfer",
        description="Minimum-energy interplanetary transfer orbit with Earth departure injection and Mars intercept.",
        view_mode="inertial",
        time_step_sec=86400.0 * 1.0, # 1 day per step
        camera_distance=2.4 * ASTRONOMICAL_UNIT,
        bodies=bodies,
        spacecraft=spacecraft
    )


def load_jupiter_trojan_asteroids() -> ScenarioPreset:
    # Sun-Jupiter mass ratio
    mu = JUPITER_MASS / (SOLAR_MASS + JUPITER_MASS) # ~0.000953

    cr3bp = CR3BPModel(mu)
    # L4 is at (0.5 - mu, sqrt(3)/2)
    l4_x, l4_y, _ = cr3bp.lagrange_points["L4"]
    # Spacecraft representing Trojan asteroid cluster center
    initial_sc_state = [l4_x + 0.02, l4_y - 0.01, 0.0, 0.003, -0.002, 0.0]

    return ScenarioPreset(
        key="jupiter_trojan_asteroids",
        name="Jupiter Trojan Asteroids (Lagrange L4/L5)",
        description="Sun-Jupiter system showing stable triangular libration orbits for the Greek (L4) and Trojan (L5) asteroid swarms.",
        view_mode="cr3bp",
        time_step_sec=0.01,
        camera_distance=1.8,
        cr3bp_mu=mu,
        initial_cr3bp_state=initial_sc_state
    )


# Register all presets
for loader in [
    load_inner_solar_system,
    load_earth_moon_cr3bp,
    load_sun_earth_jwst_l2,
    load_mercury_relativistic_precession,
    load_earth_mars_hohmann,
    load_jupiter_trojan_asteroids
]:
    p = loader()
    PRESETS[p.key] = p


def get_preset_list() -> List[str]:
    return list(PRESETS.keys())


def load_preset(key: str) -> ScenarioPreset:
    if key not in PRESETS:
        key = "inner_solar_system"
    # Re-call generator to return fresh instances
    if key == "inner_solar_system":
        return load_inner_solar_system()
    elif key == "earth_moon_cr3bp":
        return load_earth_moon_cr3bp()
    elif key == "sun_earth_jwst_l2":
        return load_sun_earth_jwst_l2()
    elif key == "mercury_relativistic_precession":
        return load_mercury_relativistic_precession()
    elif key == "earth_mars_hohmann":
        return load_earth_mars_hohmann()
    elif key == "jupiter_trojan_asteroids":
        return load_jupiter_trojan_asteroids()
    return load_inner_solar_system()
