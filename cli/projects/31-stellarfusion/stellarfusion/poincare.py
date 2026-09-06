"""
Poincaré Surface-of-Section & Resonant Magnetic Perturbation (RMP) Engine.

Traces 3D helical magnetic field lines through tokamak magnetic topologies,
computes poloidal plane intersections (Poincaré punctures) across toroidal transits,
and models resonant magnetic perturbations (RMP) that form magnetic island chains
(O-points and X-points) and chaotic ergodic edge layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .magnetic import MagneticFieldEvaluator


@dataclass(frozen=True)
class PoincarePuncture:
    """Poloidal cross-section puncture coordinates and transit counter."""

    r: float
    z: float
    turn: int
    psi_value: float


@dataclass
class ResonantMagneticPerturbation:
    """
    Helical magnetic perturbation delta_psi(R, Z, phi) = epsilon * cos(m * theta - n * phi).

    Models external correction coils or error fields that induce magnetic reconnection
    and island chains on rational surfaces where q = m / n.
    """

    m_mode: int = 2  # Poloidal mode number
    n_mode: int = 1  # Toroidal mode number
    amplitude: float = 1e-3  # Dimensionless perturbation amplitude epsilon
    r_axis: float = 3.0
    z_axis: float = 0.0

    def evaluate_perturbation(self, r: float, z: float, phi: float) -> Tuple[float, float]:
        """
        Evaluate additional magnetic field components (delta_B_R, delta_B_Z).
            delta_B_R = -(1/R) * d(delta_psi)/dZ
            delta_B_Z = (1/R) * d(delta_psi)/dR
        """
        if self.amplitude == 0.0:
            return 0.0, 0.0

        rel_r = r - self.r_axis
        rel_z = z - self.z_axis
        theta = math.atan2(rel_z, rel_r)

        phase = self.m_mode * theta - self.n_mode * phi
        cos_phase = math.cos(phase)
        sin_phase = math.sin(phase)

        dist_sq = rel_r * rel_r + rel_z * rel_z
        dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6

        # Radial envelope factor (peaked at resonance radius)
        envelope = math.exp(-0.5 * (dist - 0.5) ** 2 / 0.1)

        # Derivatives of theta: dtheta/dR = -rel_z / dist_sq, dtheta/dZ = rel_r / dist_sq
        dtheta_dr = -rel_z / dist_sq
        dtheta_dz = rel_r / dist_sq

        dpsi_dr = -self.amplitude * envelope * self.m_mode * sin_phase * dtheta_dr
        dpsi_dz = -self.amplitude * envelope * self.m_mode * sin_phase * dtheta_dz

        delta_br = -(1.0 / r) * dpsi_dz
        delta_bz = (1.0 / r) * dpsi_dr

        return delta_br, delta_bz


class PoincareFieldTracer:
    """
    Integrates 3D magnetic field line trajectory equations:
        dR / dphi = R * B_R / B_phi
        dZ / dphi = R * B_Z / B_phi
    using 4th-order Runge-Kutta (RK4) integration with toroidal puncture detection.
    """

    def __init__(
        self,
        evaluator: MagneticFieldEvaluator,
        rmp: Optional[ResonantMagneticPerturbation] = None,
    ) -> None:
        self.evaluator = evaluator
        self.rmp = rmp

    def field_line_derivatives(
        self, r: float, z: float, phi: float
    ) -> Tuple[float, float]:
        """Compute (dR/dphi, dZ/dphi)."""
        mag = self.evaluator.evaluate_at(r, z)
        br = mag.b_r
        bphi = mag.b_phi
        bz = mag.b_z

        if self.rmp is not None:
            delta_br, delta_bz = self.rmp.evaluate_perturbation(r, z, phi)
            br += delta_br
            bz += delta_bz

        if abs(bphi) < 1e-12:
            return 0.0, 0.0

        dr_dphi = (r * br) / bphi
        dz_dphi = (r * bz) / bphi
        return dr_dphi, dz_dphi

    def rk4_step(
        self, r: float, z: float, phi: float, dphi: float
    ) -> Tuple[float, float, float]:
        """Advance one RK4 step of toroidal angle increment dphi."""
        half_dphi = 0.5 * dphi

        k1_r, k1_z = self.field_line_derivatives(r, z, phi)

        r2 = r + half_dphi * k1_r
        z2 = z + half_dphi * k1_z
        phi2 = phi + half_dphi
        k2_r, k2_z = self.field_line_derivatives(r2, z2, phi2)

        r3 = r + half_dphi * k2_r
        z3 = z + half_dphi * k2_z
        phi3 = phi + half_dphi
        k3_r, k3_z = self.field_line_derivatives(r3, z3, phi3)

        r4 = r + dphi * k3_r
        z4 = z + dphi * k3_z
        phi4 = phi + dphi
        k4_r, k4_z = self.field_line_derivatives(r4, z4, phi4)

        r_next = r + (dphi / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        z_next = z + (dphi / 6.0) * (k1_z + 2.0 * k2_z + 2.0 * k3_z + k4_z)
        phi_next = phi + dphi

        return r_next, z_next, phi_next

    def trace_puncture_series(
        self,
        r_start: float,
        z_start: float,
        num_turns: int = 100,
        steps_per_turn: int = 64,
    ) -> List[PoincarePuncture]:
        """
        Trace a single magnetic field line for num_turns toroidal transits,
        recording Poincaré section punctures at phi = 0 (mod 2*pi).
        """
        punctuations: List[PoincarePuncture] = []
        dphi = (2.0 * math.pi) / steps_per_turn

        r_curr = r_start
        z_curr = z_start
        phi_curr = 0.0

        # Initial puncture at turn 0
        psi_init = self.evaluator.equilibrium.interpolate_psi(r_curr, z_curr) if hasattr(
            self.evaluator.equilibrium, "interpolate_psi"
        ) else self.evaluator.equilibrium.psi(r_curr, z_curr)
        punctuations.append(PoincarePuncture(r_curr, z_curr, 0, psi_init))

        for turn in range(1, num_turns + 1):
            for _ in range(steps_per_turn):
                r_curr, z_curr, phi_curr = self.rk4_step(r_curr, z_curr, phi_curr, dphi)

            # Record puncture at completion of one full 2*pi toroidal revolution
            psi_curr = self.evaluator.equilibrium.interpolate_psi(r_curr, z_curr) if hasattr(
                self.evaluator.equilibrium, "interpolate_psi"
            ) else self.evaluator.equilibrium.psi(r_curr, z_curr)

            punctuations.append(PoincarePuncture(r_curr, z_curr, turn, psi_curr))

        return punctuations

    def generate_multisurface_map(
        self,
        seed_radii: List[float],
        z_seed: float = 0.0,
        num_turns: int = 60,
        steps_per_turn: int = 64,
    ) -> List[List[PoincarePuncture]]:
        """
        Generate a multi-surface Poincaré map starting from multiple radial seeds.

        Reveals nested KAM invariant surfaces, island chains, and stochastic regions.
        """
        surfaces: List[List[PoincarePuncture]] = []
        for r_seed in seed_radii:
            surface = self.trace_puncture_series(
                r_start=r_seed,
                z_start=z_seed,
                num_turns=num_turns,
                steps_per_turn=steps_per_turn,
            )
            surfaces.append(surface)
        return surfaces
