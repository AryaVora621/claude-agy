"""Gravitas Orbital Dynamics & Physics Engine.

Zero-dependency celestial mechanics engine implementing 4th-order symplectic
Yoshida integration, Velocity Verlet, pairwise Newtonian gravity with Plummer
softening, Post-Newtonian relativistic perihelion precession, and inelastic
momentum-conserving collision coalescence.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Body:
    """Represents a celestial body in 3D Euclidean space."""
    id: int
    name: str
    mass: float
    radius: float
    color: str
    pos: List[float]  # [x, y, z]
    vel: List[float]  # [vx, vy, vz]
    acc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    trail: deque = field(default_factory=lambda: deque(maxlen=200))
    fixed: bool = False
    alive: bool = True

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy T = 0.5 * m * v^2."""
        v_sq = self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2
        return 0.5 * self.mass * v_sq

    def speed(self) -> float:
        """Calculate scalar velocity magnitude."""
        return math.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)

    def clone(self) -> Body:
        """Create an isolated deep copy of the body."""
        b = Body(
            id=self.id,
            name=self.name,
            mass=self.mass,
            radius=self.radius,
            color=self.color,
            pos=list(self.pos),
            vel=list(self.vel),
            acc=list(self.acc),
            fixed=self.fixed,
            alive=self.alive
        )
        b.trail = deque(list(self.trail), maxlen=self.trail.maxlen)
        return b


class CollisionEvent:
    """Telemetry data describing a collision between two bodies."""
    def __init__(self, pos: List[float], energy_released: float, survivor_name: str):
        self.pos = list(pos)
        self.energy_released = energy_released
        self.survivor_name = survivor_name
        self.age = 0.0


class GravitasPhysics:
    """High-precision N-body gravitational dynamics engine."""

    # 4th-order Yoshida symplectic integrator constants
    _CBRT2 = 2.0 ** (1.0 / 3.0)
    _W0 = -_CBRT2 / (2.0 - _CBRT2)
    _W1 = 1.0 / (2.0 - _CBRT2)
    YOSHIDA_C = [_W1 / 2.0, (_W0 + _W1) / 2.0, (_W0 + _W1) / 2.0, _W1 / 2.0]
    YOSHIDA_D = [_W1, _W0, _W1]

    def __init__(self, g_constant: float = 1.0, softening: float = 0.1, speed_of_light: float = 50.0):
        self.G = g_constant
        self.softening = softening
        self.c = speed_of_light
        self.enable_relativity = False
        self.enable_collisions = True
        self.bodies: List[Body] = []
        self.collision_events: List[CollisionEvent] = []
        self.next_body_id = 1
        self.sim_time = 0.0
        self.initial_energy: Optional[float] = None

    def add_body(self, name: str, mass: float, radius: float, color: str,
                 pos: List[float], vel: List[float], fixed: bool = False) -> Body:
        """Register a new celestial body in the simulation."""
        body = Body(
            id=self.next_body_id,
            name=name,
            mass=max(mass, 1e-9),
            radius=max(radius, 0.2),
            color=color,
            pos=list(pos),
            vel=list(vel),
            fixed=fixed
        )
        body.trail.append(list(body.pos))
        self.bodies.append(body)
        self.next_body_id += 1
        return body

    def clear(self) -> None:
        """Remove all bodies and reset simulation clock."""
        self.bodies.clear()
        self.collision_events.clear()
        self.sim_time = 0.0
        self.initial_energy = None

    def compute_accelerations(self) -> None:
        """Compute mutual gravitational accelerations for all active bodies."""
        n = len(self.bodies)
        # Reset accelerations
        for b in self.bodies:
            b.acc = [0.0, 0.0, 0.0]

        eps_sq = self.softening ** 2
        c_sq = self.c ** 2

        for i in range(n):
            bi = self.bodies[i]
            if not bi.alive:
                continue

            for j in range(i + 1, n):
                bj = self.bodies[j]
                if not bj.alive or (bi.fixed and bj.fixed):
                    continue

                dx = bj.pos[0] - bi.pos[0]
                dy = bj.pos[1] - bi.pos[1]
                dz = bj.pos[2] - bi.pos[2]

                r_sq = dx*dx + dy*dy + dz*dz
                dist_soft_sq = r_sq + eps_sq
                dist_soft = math.sqrt(dist_soft_sq)
                inv_cube = 1.0 / (dist_soft_sq * dist_soft)

                # Post-Newtonian 1PN-like effective potential correction
                # Produces authentic precession of perihelion
                rel_factor = 1.0
                if self.enable_relativity and r_sq > 1e-6:
                    rel_factor += (3.0 * self.G * (bi.mass + bj.mass)) / (c_sq * math.sqrt(r_sq))

                force_mag = self.G * bi.mass * bj.mass * inv_cube * rel_factor

                fx = force_mag * dx
                fy = force_mag * dy
                fz = force_mag * dz

                if not bi.fixed:
                    bi.acc[0] += fx / bi.mass
                    bi.acc[1] += fy / bi.mass
                    bi.acc[2] += fz / bi.mass

                if not bj.fixed:
                    bj.acc[0] -= fx / bj.mass
                    bj.acc[1] -= fy / bj.mass
                    bj.acc[2] -= fz / bj.mass

    def step_verlet(self, dt: float) -> None:
        """Symplectic Velocity Verlet integration step."""
        if not self.bodies or dt <= 0:
            return

        dt_half = 0.5 * dt

        # Step 1: Update positions with current velocities and accelerations
        for b in self.bodies:
            if not b.alive or b.fixed:
                continue
            b.pos[0] += b.vel[0] * dt + 0.5 * b.acc[0] * dt * dt
            b.pos[1] += b.vel[1] * dt + 0.5 * b.acc[1] * dt * dt
            b.pos[2] += b.vel[2] * dt + 0.5 * b.acc[2] * dt * dt
            # Intermediate half-step velocity
            b.vel[0] += b.acc[0] * dt_half
            b.vel[1] += b.acc[1] * dt_half
            b.vel[2] += b.acc[2] * dt_half

        # Step 2: Compute new accelerations at new positions
        self.compute_accelerations()

        # Step 3: Complete velocity update
        for b in self.bodies:
            if not b.alive or b.fixed:
                continue
            b.vel[0] += b.acc[0] * dt_half
            b.vel[1] += b.acc[1] * dt_half
            b.vel[2] += b.acc[2] * dt_half

        self._post_step_updates(dt)

    def step_yoshida(self, dt: float) -> None:
        """4th-order symplectic Yoshida integrator for ultra-low Hamiltonian drift."""
        if not self.bodies or dt <= 0:
            return

        c = self.YOSHIDA_C
        d = self.YOSHIDA_D

        for i in range(3):
            # Drift positions
            c_dt = c[i] * dt
            for b in self.bodies:
                if b.alive and not b.fixed:
                    b.pos[0] += b.vel[0] * c_dt
                    b.pos[1] += b.vel[1] * c_dt
                    b.pos[2] += b.vel[2] * c_dt

            # Kick velocities with new accelerations
            self.compute_accelerations()
            d_dt = d[i] * dt
            for b in self.bodies:
                if b.alive and not b.fixed:
                    b.vel[0] += b.acc[0] * d_dt
                    b.vel[1] += b.acc[1] * d_dt
                    b.vel[2] += b.acc[2] * d_dt

        # Final drift
        c_dt = c[3] * dt
        for b in self.bodies:
            if b.alive and not b.fixed:
                b.pos[0] += b.vel[0] * c_dt
                b.pos[1] += b.vel[1] * c_dt
                b.pos[2] += b.vel[2] * c_dt

        self.compute_accelerations()
        self._post_step_updates(dt)

    def _post_step_updates(self, dt: float) -> None:
        """Update body trails, check collisions, and advance clock."""
        self.sim_time += dt

        # Record trail points
        for b in self.bodies:
            if b.alive:
                b.trail.append(list(b.pos))

        # Check and resolve inelastic collisions
        if self.enable_collisions:
            self._handle_collisions()

        # Update collision effect ages
        self.collision_events = [ev for ev in self.collision_events if ev.age < 1.5]
        for ev in self.collision_events:
            ev.age += dt

        # Cache initial energy if uninitialized
        if self.initial_energy is None and len(self.bodies) > 0:
            self.initial_energy = self.total_energy()

    def _handle_collisions(self) -> None:
        """Inelastic collision resolution preserving linear momentum and mass."""
        n = len(self.bodies)
        merged_ids = set()

        for i in range(n):
            bi = self.bodies[i]
            if not bi.alive or bi.id in merged_ids:
                continue

            for j in range(i + 1, n):
                bj = self.bodies[j]
                if not bj.alive or bj.id in merged_ids:
                    continue

                dx = bj.pos[0] - bi.pos[0]
                dy = bj.pos[1] - bi.pos[1]
                dz = bj.pos[2] - bi.pos[2]
                dist_sq = dx*dx + dy*dy + dz*dz
                contact_dist = bi.radius + bj.radius

                if dist_sq <= contact_dist * contact_dist:
                    # Inelastic merge: primary is the more massive body
                    if bi.mass >= bj.mass:
                        primary, secondary = bi, bj
                    else:
                        primary, secondary = bj, bi

                    total_mass = primary.mass + secondary.mass
                    old_energy = primary.kinetic_energy() + secondary.kinetic_energy()

                    # Inelastic momentum conservation: v = (m1*v1 + m2*v2) / (m1+m2)
                    new_vx = (primary.mass * primary.vel[0] + secondary.mass * secondary.vel[0]) / total_mass
                    new_vy = (primary.mass * primary.vel[1] + secondary.mass * secondary.vel[1]) / total_mass
                    new_vz = (primary.mass * primary.vel[2] + secondary.mass * secondary.vel[2]) / total_mass

                    # Center of mass position
                    new_x = (primary.mass * primary.pos[0] + secondary.mass * secondary.pos[0]) / total_mass
                    new_y = (primary.mass * primary.pos[1] + secondary.mass * secondary.pos[1]) / total_mass
                    new_z = (primary.mass * primary.pos[2] + secondary.mass * secondary.pos[2]) / total_mass

                    # Sphere volume conservation: R_new = (R1^3 + R2^3)^(1/3)
                    new_radius = (primary.radius**3 + secondary.radius**3) ** (1.0 / 3.0)

                    primary.mass = total_mass
                    primary.radius = new_radius
                    primary.pos = [new_x, new_y, new_z]
                    primary.vel = [new_vx, new_vy, new_vz]

                    secondary.alive = False
                    merged_ids.add(secondary.id)

                    new_energy = primary.kinetic_energy()
                    energy_released = max(0.0, old_energy - new_energy)

                    self.collision_events.append(
                        CollisionEvent([new_x, new_y, new_z], energy_released, primary.name)
                    )

        # Remove dead bodies from active list
        self.bodies = [b for b in self.bodies if b.alive]

    def kinetic_energy(self) -> float:
        """Total kinetic energy of the system."""
        return sum(b.kinetic_energy() for b in self.bodies if b.alive)

    def potential_energy(self) -> float:
        """Total gravitational potential energy of pairwise interactions."""
        u = 0.0
        n = len(self.bodies)
        eps_sq = self.softening ** 2

        for i in range(n):
            bi = self.bodies[i]
            if not bi.alive:
                continue
            for j in range(i + 1, n):
                bj = self.bodies[j]
                if not bj.alive:
                    continue
                dx = bj.pos[0] - bi.pos[0]
                dy = bj.pos[1] - bi.pos[1]
                dz = bj.pos[2] - bi.pos[2]
                r = math.sqrt(dx*dx + dy*dy + dz*dz + eps_sq)
                u -= self.G * bi.mass * bj.mass / r

        return u

    def total_energy(self) -> float:
        """Total Hamiltonian energy H = T + U."""
        return self.kinetic_energy() + self.potential_energy()

    def energy_drift(self) -> float:
        """Fractional energy drift relative to initial state |(E - E0) / E0|."""
        if self.initial_energy is None or abs(self.initial_energy) < 1e-12:
            return 0.0
        current = self.total_energy()
        return abs((current - self.initial_energy) / self.initial_energy)

    def center_of_mass(self) -> Tuple[float, float, float]:
        """Compute center of mass position (X_cm, Y_cm, Z_cm)."""
        tot_m = sum(b.mass for b in self.bodies if b.alive)
        if tot_m <= 0:
            return (0.0, 0.0, 0.0)
        cx = sum(b.mass * b.pos[0] for b in self.bodies if b.alive) / tot_m
        cy = sum(b.mass * b.pos[1] for b in self.bodies if b.alive) / tot_m
        cz = sum(b.mass * b.pos[2] for b in self.bodies if b.alive) / tot_m
        return (cx, cy, cz)

    def linear_momentum(self) -> Tuple[float, float, float]:
        """Compute total net linear momentum (P_x, P_y, P_z)."""
        px = sum(b.mass * b.vel[0] for b in self.bodies if b.alive)
        py = sum(b.mass * b.vel[1] for b in self.bodies if b.alive)
        pz = sum(b.mass * b.vel[2] for b in self.bodies if b.alive)
        return (px, py, pz)

    def angular_momentum(self) -> Tuple[float, float, float]:
        """Compute total net angular momentum vector L = sum(r x p)."""
        lx, ly, lz = 0.0, 0.0, 0.0
        for b in self.bodies:
            if not b.alive:
                continue
            px = b.mass * b.vel[0]
            py = b.mass * b.vel[1]
            pz = b.mass * b.vel[2]
            # Cross product r x p
            lx += b.pos[1] * pz - b.pos[2] * py
            ly += b.pos[2] * px - b.pos[0] * pz
            lz += b.pos[0] * py - b.pos[1] * px
        return (lx, ly, lz)
