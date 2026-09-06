"""
NovaPhysics: Verlet Particle Mechanics & Position-Based Dynamics (PBD) Cloth Simulation.
Features structural, shear, and bending constraints, tearable linkages, obstacle collisions, and wind forces.
"""

import math
from typing import List, Optional, Tuple, Set
from .math2d import Vec2


class VerletParticle:
    """Particle integrated via Verlet algorithm with numerical damping."""
    __slots__ = ("position", "old_position", "acceleration", "inv_mass", "pinned")

    def __init__(self, position: Vec2, mass: float = 1.0, pinned: bool = False) -> None:
        self.position = Vec2(position.x, position.y)
        self.old_position = Vec2(position.x, position.y)
        self.acceleration = Vec2(0.0, 0.0)
        self.pinned = pinned
        if pinned or mass <= 0.0:
            self.inv_mass = 0.0
        else:
            self.inv_mass = 1.0 / float(mass)

    @property
    def velocity(self) -> Vec2:
        return self.position - self.old_position

    def apply_force(self, force: Vec2) -> None:
        if not self.pinned:
            self.acceleration = self.acceleration + force * self.inv_mass

    def integrate(self, dt: float, damping: float = 0.01) -> None:
        """Position Verlet integration step: x_next = x + (x - x_old)*(1 - damping) + a*dt^2."""
        if self.pinned:
            self.acceleration = Vec2(0.0, 0.0)
            return

        vel = (self.position - self.old_position) * (1.0 - damping)
        new_pos = self.position + vel + self.acceleration * (dt * dt)
        self.old_position = self.position
        self.position = new_pos
        self.acceleration = Vec2(0.0, 0.0)


class DistanceConstraint:
    """Distance relaxation constraint between two Verlet particles with tearable limit."""
    __slots__ = ("p1", "p2", "rest_length", "stiffness", "tear_ratio", "is_broken")

    def __init__(
        self,
        p1: VerletParticle,
        p2: VerletParticle,
        stiffness: float = 1.0,
        tear_ratio: float = 3.0,
        rest_length: Optional[float] = None
    ) -> None:
        self.p1 = p1
        self.p2 = p2
        self.stiffness = min(1.0, max(0.0, stiffness))
        self.tear_ratio = tear_ratio
        self.is_broken = False

        if rest_length is None:
            self.rest_length = (p2.position - p1.position).length()
        else:
            self.rest_length = float(rest_length)

    def relax(self) -> bool:
        """Enforce distance constraint. Returns False if constraint tore and broke."""
        if self.is_broken:
            return False

        p1 = self.p1
        p2 = self.p2
        w1 = p1.inv_mass
        w2 = p2.inv_mass
        w_sum = w1 + w2

        if w_sum < 1e-9:
            return True

        diff = p2.position - p1.position
        dist = diff.length()
        if dist < 1e-6:
            return True

        # Check for tearing
        if self.tear_ratio > 0.0 and dist > self.rest_length * self.tear_ratio:
            self.is_broken = True
            return False

        # Constraint delta: C = dist - rest_length
        factor = ((dist - self.rest_length) / dist) * self.stiffness
        correction = diff * factor

        if not p1.pinned:
            p1.position = p1.position + correction * (w1 / w_sum)
        if not p2.pinned:
            p2.position = p2.position - correction * (w2 / w_sum)

        return True


class ClothMesh:
    """
    2D Position-Based Dynamics (PBD) cloth mesh.
    Configured with structural, shear, and bending springs, tearing, wind turbulence, and obstacle repulsion.
    """
    __slots__ = (
        "cols", "rows", "particles", "constraints", "gravity", "damping",
        "iterations", "wind"
    )

    def __init__(
        self,
        cols: int,
        rows: int,
        origin: Vec2,
        spacing: float = 0.5,
        mass_per_particle: float = 0.1,
        pin_top_row: bool = False,
        pin_corners: bool = True,
        gravity: Vec2 = Vec2(0.0, -9.81),
        damping: float = 0.01,
        iterations: int = 4
    ) -> None:
        self.cols = cols
        self.rows = rows
        self.gravity = gravity
        self.damping = damping
        self.iterations = max(1, iterations)
        self.wind = Vec2(0.0, 0.0)

        self.particles: List[List[VerletParticle]] = []
        self.constraints: List[DistanceConstraint] = []

        # Create particles
        for r in range(rows):
            row_particles: List[VerletParticle] = []
            for c in range(cols):
                pos = origin + Vec2(c * spacing, -r * spacing)
                is_pinned = False
                if pin_top_row and r == 0:
                    is_pinned = True
                elif pin_corners and r == 0 and (c == 0 or c == cols - 1):
                    is_pinned = True

                p = VerletParticle(pos, mass=mass_per_particle, pinned=is_pinned)
                row_particles.append(p)
            self.particles.append(row_particles)

        # Create structural constraints (horizontal & vertical)
        for r in range(rows):
            for c in range(cols):
                curr = self.particles[r][c]
                # Right neighbor
                if c + 1 < cols:
                    self.constraints.append(DistanceConstraint(curr, self.particles[r][c + 1], stiffness=0.95))
                # Bottom neighbor
                if r + 1 < rows:
                    self.constraints.append(DistanceConstraint(curr, self.particles[r + 1][c], stiffness=0.95))

        # Create shear constraints (diagonals) for in-plane shear resistance
        for r in range(rows - 1):
            for c in range(cols - 1):
                p_tl = self.particles[r][c]
                p_tr = self.particles[r][c + 1]
                p_bl = self.particles[r + 1][c]
                p_br = self.particles[r + 1][c + 1]
                self.constraints.append(DistanceConstraint(p_tl, p_br, stiffness=0.85))
                self.constraints.append(DistanceConstraint(p_tr, p_bl, stiffness=0.85))

        # Create bending constraints (distance-2 neighbors) for fold resistance
        for r in range(rows):
            for c in range(cols):
                curr = self.particles[r][c]
                if c + 2 < cols:
                    self.constraints.append(DistanceConstraint(curr, self.particles[r][c + 2], stiffness=0.6))
                if r + 2 < rows:
                    self.constraints.append(DistanceConstraint(curr, self.particles[r + 2][c], stiffness=0.6))

    def get_all_particles(self) -> List[VerletParticle]:
        return [p for row in self.particles for p in row]

    def set_wind(self, wind: Vec2) -> None:
        self.wind = wind

    def apply_obstacle_sphere(self, center: Vec2, radius: float) -> None:
        """Pushes cloth particles outside a circular obstacle of given center and radius."""
        r_sq = radius * radius
        for row in self.particles:
            for p in row:
                if p.pinned:
                    continue
                diff = p.position - center
                d_sq = diff.length_sq()
                if d_sq < r_sq:
                    d = math.sqrt(d_sq)
                    if d > 1e-6:
                        normal = diff * (1.0 / d)
                    else:
                        normal = Vec2(0.0, 1.0)
                    p.position = center + normal * (radius + 0.001)

    def tear_at(self, world_point: Vec2, radius: float = 0.5) -> int:
        """Sever all distance constraints near a given world point (cloth cut/tear)."""
        torn_count = 0
        r_sq = radius * radius
        for c in self.constraints:
            if not c.is_broken:
                mid = (c.p1.position + c.p2.position) * 0.5
                if (mid - world_point).length_sq() <= r_sq:
                    c.is_broken = True
                    torn_count += 1
        return torn_count

    def step(self, dt: float) -> None:
        """Simulates one time step of cloth physics."""
        all_particles = self.get_all_particles()

        # 1. Apply gravity, wind forces, and integrate positions
        for p in all_particles:
            if not p.pinned:
                p.apply_force(self.gravity * (1.0 / p.inv_mass if p.inv_mass > 0 else 1.0))
                if self.wind.length_sq() > 1e-6:
                    p.apply_force(self.wind)
            p.integrate(dt, damping=self.damping)

        # 2. Relax distance constraints over Gauss-Seidel iterations
        for _ in range(self.iterations):
            for c in self.constraints:
                c.relax()
