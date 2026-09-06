"""
Atomix: Symplectic Integrators and SHAKE Constraint Engine.
Implements:
  - Velocity Verlet symplectic time integration
  - Leapfrog Verlet integrator
  - SHAKE holonomic bond length constraint solver
  - RATTLE velocity projection for constrained rigid bonds
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional, Callable

from atomix.types import Vector3D, Atom, Bond, SimulationBox


class SHAKEConstraintSolver:
    """
    Iterative SHAKE algorithm for holonomic distance constraints.
    Enforces |r_i - r_j|^2 = d_0^2 for rigid bonds (e.g. water O-H bonds).
    """
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 100) -> None:
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def solve_positions(
        self,
        atoms: List[Atom],
        old_positions: List[Vector3D],
        rigid_bonds: List[Bond],
        box: SimulationBox,
    ) -> bool:
        """
        Iteratively adjust atomic positions to satisfy rigid bond lengths.
        Returns True if converged within max_iterations.
        """
        if not rigid_bonds:
            return True

        converged = False
        for iteration in range(self.max_iterations):
            max_dev = 0.0

            for bond in rigid_bonds:
                i = bond.atom1_id
                j = bond.atom2_id
                target_dist = bond.length_eq
                target_sq = target_dist * target_dist

                # Current distance vector using minimum image convention
                r_ij_curr = box.minimum_image_vector(atoms[i].position, atoms[j].position)
                curr_sq = r_ij_curr.norm_sq()

                diff = curr_sq - target_sq
                rel_dev = abs(diff) / target_sq
                if rel_dev > max_dev:
                    max_dev = rel_dev

                if rel_dev > self.tolerance:
                    # Old reference vector from previous unconstrained time step
                    r_ij_old = box.minimum_image_vector(old_positions[i], old_positions[j])

                    inv_m_i = 1.0 / atoms[i].mass
                    inv_m_j = 1.0 / atoms[j].mass
                    inv_m_sum = inv_m_i + inv_m_j

                    dot_old_curr = r_ij_old.dot(r_ij_curr)
                    if abs(dot_old_curr) < 1e-12:
                        continue

                    # Lagrange multiplier step
                    delta = diff / (2.0 * inv_m_sum * dot_old_curr)

                    # Update coordinates (when diff > 0, move i towards j and j towards i)
                    corr_i = (+delta * inv_m_i) * r_ij_old
                    corr_j = (-delta * inv_m_j) * r_ij_old

                    atoms[i].position = atoms[i].position + corr_i
                    atoms[j].position = atoms[j].position + corr_j

            if max_dev <= self.tolerance:
                converged = True
                break

        return converged

    def solve_velocities(
        self,
        atoms: List[Atom],
        rigid_bonds: List[Bond],
        box: SimulationBox,
    ) -> None:
        """
        RATTLE velocity projection step: ensures relative velocities
        are perpendicular to bond vectors (dr/dt . r_ij = 0).
        """
        if not rigid_bonds:
            return

        for iteration in range(self.max_iterations):
            max_v_proj = 0.0

            for bond in rigid_bonds:
                i = bond.atom1_id
                j = bond.atom2_id

                r_ij = box.minimum_image_vector(atoms[i].position, atoms[j].position)
                r_sq = r_ij.norm_sq()
                if r_sq < 1e-12:
                    continue

                v_ij = atoms[j].velocity - atoms[i].velocity
                v_proj = r_ij.dot(v_ij)
                abs_v_proj = abs(v_proj)
                if abs_v_proj > max_v_proj:
                    max_v_proj = abs_v_proj

                if abs_v_proj > self.tolerance:
                    inv_m_i = 1.0 / atoms[i].mass
                    inv_m_j = 1.0 / atoms[j].mass
                    k_v = v_proj / ((inv_m_i + inv_m_j) * r_sq)

                    atoms[i].velocity = atoms[i].velocity + (k_v * inv_m_i) * r_ij
                    atoms[j].velocity = atoms[j].velocity - (k_v * inv_m_j) * r_ij

            if max_v_proj <= self.tolerance:
                break


class VelocityVerletIntegrator:
    """
    Standard two-stage Velocity Verlet symplectic integrator.
    Stage 1:
      r(t + dt) = r(t) + v(t)*dt + 0.5*(F(t)/m)*dt^2
      v(t + dt/2) = v(t) + 0.5*(F(t)/m)*dt
    Stage 2 (after force recalculation):
      v(t + dt) = v(t + dt/2) + 0.5*(F(t + dt)/m)*dt
    """
    def __init__(
        self,
        timestep: float = 0.001,
        enable_shake: bool = False,
        shake_tol: float = 1e-6,
    ) -> None:
        if timestep <= 0.0:
            raise ValueError(f"Timestep must be strictly positive, got {timestep}")
        self.dt = float(timestep)
        self.dt_half = 0.5 * self.dt
        self.dt_sq_half = 0.5 * self.dt * self.dt
        self.enable_shake = enable_shake
        self.shake = SHAKEConstraintSolver(tolerance=shake_tol) if enable_shake else None

    def step_stage1(
        self,
        atoms: List[Atom],
        box: SimulationBox,
        rigid_bonds: Optional[List[Bond]] = None,
    ) -> List[Vector3D]:
        """
        First integration phase: update coordinates and half-step velocities.
        Returns reference positions before displacement (for SHAKE).
        """
        old_positions = [atom.position for atom in atoms]

        for atom in atoms:
            inv_m = 1.0 / atom.mass
            acc = atom.force * inv_m

            # Position update: r + v*dt + 0.5*a*dt^2
            new_pos = atom.position + (atom.velocity * self.dt) + (acc * self.dt_sq_half)
            atom.position = box.wrap_position(new_pos)

            # Half-step velocity: v + 0.5*a*dt
            atom.velocity = atom.velocity + (acc * self.dt_half)

        # Enforce holonomic constraints if enabled
        if self.enable_shake and self.shake and rigid_bonds:
            self.shake.solve_positions(atoms, old_positions, rigid_bonds, box)

        return old_positions

    def step_stage2(
        self,
        atoms: List[Atom],
        box: SimulationBox,
        rigid_bonds: Optional[List[Bond]] = None,
    ) -> None:
        """
        Second integration phase: update velocities to full step using newly calculated forces.
        """
        for atom in atoms:
            inv_m = 1.0 / atom.mass
            acc = atom.force * inv_m
            # Complete velocity update: v(t + dt/2) + 0.5*a(t + dt)*dt
            atom.velocity = atom.velocity + (acc * self.dt_half)

        if self.enable_shake and self.shake and rigid_bonds:
            self.shake.solve_velocities(atoms, rigid_bonds, box)
