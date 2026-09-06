"""Gravitas Celestial System Presets.

Pre-configured gravitationally stable choreographies, chaotic N-body benchmarks,
and astrophysical orbital dynamics scenarios.
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Callable
from .physics import GravitasPhysics


def setup_sol_system(sim: GravitasPhysics) -> None:
    """Inner and outer Solar System core with scaled mass and orbital velocity."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.05
    sim.enable_relativity = False

    # Central Sun
    sim.add_body(
        name="Sol",
        mass=1000.0,
        radius=2.4,
        color="#F59E0B",
        pos=[0.0, 0.0, 0.0],
        vel=[0.0, 0.0, 0.0],
        fixed=False
    )

    # Planets: (name, mass, radius, color, distance, inclination_deg)
    planets = [
        ("Mercury", 0.05, 0.6, "#9CA3AF", 4.0, 7.0),
        ("Venus", 0.82, 0.9, "#FBBF24", 7.0, 3.4),
        ("Earth", 1.00, 1.0, "#06B6D4", 10.0, 0.0),
        ("Mars", 0.11, 0.7, "#EF4444", 15.0, 1.8),
        ("Jupiter", 317.0, 2.0, "#D97706", 26.0, 1.3),
        ("Saturn", 95.0, 1.6, "#FCD34D", 38.0, 2.5),
    ]

    for name, mass, rad, col, dist, inc in planets:
        # Circular orbit velocity v = sqrt(G * M / r)
        v_circ = math.sqrt(sim.G * 1000.0 / dist)
        inc_rad = math.radians(inc)
        pos = [dist, 0.0, 0.0]
        vel = [0.0, v_circ * math.cos(inc_rad), v_circ * math.sin(inc_rad)]
        sim.add_body(name, mass, rad, col, pos, vel)


def setup_figure_eight(sim: GravitasPhysics) -> None:
    """Stable 3-body Figure-Eight choreography discovered by Chenciner & Montgomery (2000)."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.001
    sim.enable_relativity = False

    m = 100.0
    # Canonical initial conditions for figure-8 orbit
    x1, y1 = 0.97000436, -0.24308753
    x2, y2 = -x1, -y1
    x3, y3 = 0.0, 0.0

    vx3, vy3 = -2.0 * -0.93240737, -2.0 * -0.86473146
    vx1, vy1 = -0.93240737, -0.86473146
    vx2, vy2 = vx1, vy1

    # Scale to our simulation unit
    scale_pos = 12.0
    scale_vel = 3.2

    sim.add_body("Body Alpha", m, 1.1, "#06B6D4", [x1 * scale_pos, y1 * scale_pos, 0.0], [vx1 * scale_vel, vy1 * scale_vel, 0.0])
    sim.add_body("Body Beta", m, 1.1, "#8B5CF6", [x2 * scale_pos, y2 * scale_pos, 0.0], [vx2 * scale_vel, vy2 * scale_vel, 0.0])
    sim.add_body("Body Gamma", m, 1.1, "#10B981", [x3 * scale_pos, y3 * scale_pos, 0.0], [vx3 * scale_vel, vy3 * scale_vel, 0.0])


def setup_lagrange_trojans(sim: GravitasPhysics) -> None:
    """Star and massive Jovian companion with stable L4 & L5 Trojan asteroid swarms."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.08
    sim.enable_relativity = False

    m_star = 1200.0
    m_jovian = 60.0
    r_orbit = 22.0

    # Primary Star
    sim.add_body("Helios", m_star, 2.6, "#F59E0B", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    # Jovian companion
    v_jovian = math.sqrt(sim.G * (m_star + m_jovian) / r_orbit)
    sim.add_body("Titanus", m_jovian, 1.5, "#8B5CF6", [r_orbit, 0.0, 0.0], [0.0, v_jovian, 0.0])

    # L4 and L5 are at equilateral triangle vertices (+60 deg and -60 deg from planet)
    angles_deg = [60.0, -60.0]
    names = ["L4 Trojan", "L5 Greek"]
    colors = ["#10B981", "#06B6D4"]

    rng = random.Random(42)
    for l_idx, base_deg in enumerate(angles_deg):
        for i in range(16):
            # Cluster around equilibrium point
            d_deg = base_deg + rng.gauss(0.0, 3.5)
            r_ast = r_orbit + rng.gauss(0.0, 0.7)
            rad = math.radians(d_deg)

            x = r_ast * math.cos(rad)
            y = r_ast * math.sin(rad)
            z = rng.gauss(0.0, 0.3)

            # Circular velocity perpendicular to radius
            v = math.sqrt(sim.G * m_star / r_ast)
            vx = -v * math.sin(rad) + rng.gauss(0.0, 0.05)
            vy = v * math.cos(rad) + rng.gauss(0.0, 0.05)
            vz = rng.gauss(0.0, 0.02)

            sim.add_body(
                name=f"{names[l_idx]} {i+1}",
                mass=0.01,
                radius=0.35,
                color=colors[l_idx],
                pos=[x, y, z],
                vel=[vx, vy, vz]
            )


def setup_pythagorean_three_body(sim: GravitasPhysics) -> None:
    """Burrau's (1913) classical Pythagorean three-body problem with masses 3, 4, 5."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.05
    sim.enable_relativity = False

    scale = 3.5
    # Right-triangle vertices with side lengths 3, 4, 5
    # Mass 3 opposite to side 3, Mass 4 opposite to side 4, Mass 5 opposite to hypotenuse
    sim.add_body("Mass 3 (Red)", 30.0, 0.9, "#EF4444", [1.0 * scale, -1.0 * scale, 0.0], [0.0, 0.0, 0.0])
    sim.add_body("Mass 4 (Green)", 40.0, 1.0, "#10B981", [-2.0 * scale, -1.0 * scale, 0.0], [0.0, 0.0, 0.0])
    sim.add_body("Mass 5 (Blue)", 50.0, 1.1, "#06B6D4", [1.0 * scale, 3.0 * scale, 0.0], [0.0, 0.0, 0.0])


def setup_galactic_merger(sim: GravitasPhysics) -> None:
    """Two interacting galaxies colliding with tidal bridges and tails."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.2
    sim.enable_relativity = False

    # Galaxy A: Centered at (-18, -6, 0), moving right
    core_a_mass = 800.0
    pos_a = [-18.0, -6.0, 0.0]
    vel_a = [1.8, 0.8, 0.0]
    sim.add_body("Core Andromeda", core_a_mass, 2.0, "#06B6D4", pos_a, vel_a)

    # Disk stars for Galaxy A
    rng = random.Random(1337)
    for ring_r in [3.0, 5.0, 7.0, 9.0]:
        num_stars = int(ring_r * 4)
        v_orb = math.sqrt(sim.G * core_a_mass / ring_r)
        for s in range(num_stars):
            theta = (2.0 * math.pi * s) / num_stars + rng.uniform(-0.1, 0.1)
            x = pos_a[0] + ring_r * math.cos(theta)
            y = pos_a[1] + ring_r * math.sin(theta)
            z = pos_a[2] + rng.gauss(0.0, 0.15)
            vx = vel_a[0] - v_orb * math.sin(theta)
            vy = vel_a[1] + v_orb * math.cos(theta)
            vz = vel_a[2]
            sim.add_body(f"Star A-{s}", 0.005, 0.3, "#7DD3FC", [x, y, z], [vx, vy, vz])

    # Galaxy B: Centered at (18, 6, 4), inclined 30 deg, moving left
    core_b_mass = 600.0
    pos_b = [18.0, 6.0, 4.0]
    vel_b = [-1.8, -0.8, 0.0]
    sim.add_body("Core Triangulum", core_b_mass, 1.8, "#8B5CF6", pos_b, vel_b)

    # Disk stars for Galaxy B with tilt
    tilt = math.radians(35.0)
    for ring_r in [2.5, 4.5, 6.5]:
        num_stars = int(ring_r * 4)
        v_orb = math.sqrt(sim.G * core_b_mass / ring_r)
        for s in range(num_stars):
            theta = (2.0 * math.pi * s) / num_stars + rng.uniform(-0.1, 0.1)
            # Local coordinates
            lx = ring_r * math.cos(theta)
            ly = ring_r * math.sin(theta) * math.cos(tilt)
            lz = ring_r * math.sin(theta) * math.sin(tilt)

            lvx = -v_orb * math.sin(theta)
            lvy = v_orb * math.cos(theta) * math.cos(tilt)
            lvz = v_orb * math.cos(theta) * math.sin(tilt)

            x = pos_b[0] + lx
            y = pos_b[1] + ly
            z = pos_b[2] + lz + rng.gauss(0.0, 0.15)

            vx = vel_b[0] + lvx
            vy = vel_b[1] + lvy
            vz = vel_b[2] + lvz

            sim.add_body(f"Star B-{s}", 0.005, 0.3, "#C4B5FD", [x, y, z], [vx, vy, vz])


def setup_relativistic_rosette(sim: GravitasPhysics) -> None:
    """Supermassive black hole with relativistic perihelion advance (Schwarzschild rosette)."""
    sim.clear()
    sim.G = 1.0
    sim.softening = 0.01
    sim.c = 25.0
    sim.enable_relativity = True

    m_hole = 2500.0
    sim.add_body("Sagittarius A*", m_hole, 2.2, "#111827", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], fixed=True)

    # Highly eccentric star orbit: periapsis = 3.0, apoapsis = 18.0
    # Semi-major axis a = (18 + 3)/2 = 10.5
    # Velocity at periapsis = sqrt(G*M * (2/r - 1/a))
    r_peri = 3.2
    a = 11.0
    v_peri = math.sqrt(sim.G * m_hole * (2.0 / r_peri - 1.0 / a))

    sim.add_body(
        name="Star S2",
        mass=1.0,
        radius=0.9,
        color="#F43F5E",
        pos=[r_peri, 0.0, 0.0],
        vel=[0.0, v_peri, 0.0]
    )


PRESETS: Dict[str, Callable[[GravitasPhysics], None]] = {
    "Sol System Core": setup_sol_system,
    "Figure-8 Choreography": setup_figure_eight,
    "Lagrangian Trojan Resonance": setup_lagrange_trojans,
    "Pythagorean 3-Body Problem": setup_pythagorean_three_body,
    "Galactic Disk Merger": setup_galactic_merger,
    "Relativistic Rosette": setup_relativistic_rosette,
}
