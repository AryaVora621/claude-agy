"""
Atomix: Interatomic Non-Bonded and Intramolecular Bonded Force Fields.
Implements:
  - Lennard-Jones (12-6) with shifted-force cutoff and Lorentz-Berthelot mixing
  - Coulombic electrostatics with reaction-field/shifted potential
  - Harmonic bond stretching
  - Harmonic valence angle bending with analytical chain-rule gradients
  - Periodic torsional dihedrals with torque projection
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional, Dict

from atomix.types import (
    Vector3D,
    Atom,
    Bond,
    Angle,
    Dihedral,
    COULOMB_CONSTANT_KJ,
)


class LennardJonesPotential:
    """
    Truncated and shifted Lennard-Jones 12-6 pairwise potential.
    V(r) = 4*eps * [(sig/r)^12 - (sig/r)^6]
    Includes Lorentz-Berthelot mixing rules and analytical force evaluation.
    """
    def __init__(self, cutoff: float = 2.5, shifted: bool = True) -> None:
        if cutoff <= 0.0:
            raise ValueError(f"Cutoff radius must be strictly positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self.cutoff_sq = self.cutoff * self.cutoff
        self.shifted = shifted

    @staticmethod
    def mix_parameters(sigma1: float, epsilon1: float, sigma2: float, epsilon2: float) -> Tuple[float, float]:
        """
        Lorentz-Berthelot arithmetic mean for sigma and geometric mean for epsilon.
        """
        sig_12 = 0.5 * (sigma1 + sigma2)
        eps_12 = math.sqrt(epsilon1 * epsilon2)
        return (sig_12, eps_12)

    def evaluate_pair(
        self,
        r_vec: Vector3D,
        sigma: float,
        epsilon: float,
    ) -> Tuple[float, Vector3D, float]:
        """
        Compute pairwise LJ energy, force on atom 1, and virial scalar term.
        r_vec = r2 - r1 (minimum image separation vector).
        Returns: (energy, force_on_atom1, virial_contribution)
        """
        r_sq = r_vec.norm_sq()
        if r_sq >= self.cutoff_sq or r_sq < 1e-12:
            return (0.0, Vector3D(0.0, 0.0, 0.0), 0.0)

        r = math.sqrt(r_sq)
        inv_r = 1.0 / r
        inv_r_sq = inv_r * inv_r

        s = sigma * inv_r
        s2 = s * s
        s6 = s2 * s2 * s2
        s12 = s6 * s6

        # Standard unshifted potential
        v_raw = 4.0 * epsilon * (s12 - s6)

        # Shift potential so V(r_cutoff) = 0
        if self.shifted:
            s_c = sigma / self.cutoff
            s_c2 = s_c * s_c
            s_c6 = s_c2 * s_c2 * s_c2
            s_c12 = s_c6 * s_c6
            v_cutoff = 4.0 * epsilon * (s_c12 - s_c6)
            energy = v_raw - v_cutoff
        else:
            energy = v_raw

        # Force magnitude factor: -dV/dr * (1/r)
        # dV/dr = -24*eps/r * (2*s12 - s6)
        # F1 = + dV/dr * (r_vec / r) = - 24*eps * (2*s12 - s6) * inv_r_sq * r_vec
        f_coeff = -24.0 * epsilon * (2.0 * s12 - s6) * inv_r_sq
        force_on_1 = Vector3D(
            f_coeff * r_vec.x,
            f_coeff * r_vec.y,
            f_coeff * r_vec.z,
        )

        # Pairwise scalar virial: r_vec . F2 = - r_vec . F1 = 24*eps*(2*s12 - s6)
        virial = 24.0 * epsilon * (2.0 * s12 - s6)

        return (energy, force_on_1, virial)


class CoulombPotential:
    """
    Pairwise Coulombic electrostatic potential with shifted potential cutoff.
    V(r) = f_coulomb * (q1 * q2 / r)
    """
    def __init__(self, cutoff: float = 12.0, coulomb_constant: float = COULOMB_CONSTANT_KJ) -> None:
        if cutoff <= 0.0:
            raise ValueError(f"Cutoff must be positive, got {cutoff}")
        self.cutoff = float(cutoff)
        self.cutoff_sq = self.cutoff * self.cutoff
        self.coulomb_constant = float(coulomb_constant)
        self.inv_cutoff = 1.0 / self.cutoff

    def evaluate_pair(
        self,
        r_vec: Vector3D,
        q1: float,
        q2: float,
    ) -> Tuple[float, Vector3D, float]:
        """
        Compute Coulomb energy and force on atom 1.
        r_vec = r2 - r1.
        Returns: (energy, force_on_atom1, virial)
        """
        q_prod = q1 * q2
        if abs(q_prod) < 1e-9:
            return (0.0, Vector3D(0.0, 0.0, 0.0), 0.0)

        r_sq = r_vec.norm_sq()
        if r_sq >= self.cutoff_sq or r_sq < 1e-12:
            return (0.0, Vector3D(0.0, 0.0, 0.0), 0.0)

        r = math.sqrt(r_sq)
        inv_r = 1.0 / r
        inv_r3 = inv_r * inv_r * inv_r

        pref = self.coulomb_constant * q_prod
        # Shifted Coulomb potential to avoid energy jump at boundary
        energy = pref * (inv_r - self.inv_cutoff)

        # F1 = - pref * inv_r3 * r_vec
        f_factor = -pref * inv_r3
        force_on_1 = Vector3D(
            f_factor * r_vec.x,
            f_factor * r_vec.y,
            f_factor * r_vec.z,
        )

        virial = pref * inv_r
        return (energy, force_on_1, virial)


class HarmonicBondPotential:
    """
    Harmonic bond stretching potential.
    V(r) = 0.5 * k_spring * (r - r_0)^2
    """
    @staticmethod
    def evaluate(
        pos1: Vector3D,
        pos2: Vector3D,
        bond: Bond,
    ) -> Tuple[float, Vector3D, Vector3D]:
        """
        Compute bond energy and restoring forces on atom 1 and atom 2.
        Returns: (energy, force_on_1, force_on_2)
        """
        # Bond displacement vector r12 = pos2 - pos1
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        r_sq = dx * dx + dy * dy + dz * dz
        r = math.sqrt(max(1e-12, r_sq))

        delta_r = r - bond.length_eq
        energy = 0.5 * bond.k_spring * delta_r * delta_r

        # Force magnitude: F = k_spring * (r - r_0) / r
        # F1 points towards pos2 along +r12
        f_scale = bond.k_spring * delta_r / r
        f1 = Vector3D(f_scale * dx, f_scale * dy, f_scale * dz)
        f2 = Vector3D(-f1.x, -f1.y, -f1.z)

        return (energy, f1, f2)


class HarmonicAnglePotential:
    """
    Valence angle bending potential centered at vertex atom j:
    i - j - k
    V(theta) = 0.5 * k_angle * (theta - theta_0)^2
    """
    @staticmethod
    def evaluate(
        pos_i: Vector3D,
        pos_j: Vector3D,
        pos_k: Vector3D,
        angle: Angle,
    ) -> Tuple[float, Vector3D, Vector3D, Vector3D]:
        """
        Compute angle bending energy and analytical forces on atoms i, j, k.
        Returns: (energy, force_i, force_j, force_k)
        """
        # Vectors from central vertex j to arms i and k
        v_ji = pos_i - pos_j
        v_jk = pos_k - pos_j

        r_ji = v_ji.norm()
        r_jk = v_jk.norm()
        if r_ji < 1e-8 or r_jk < 1e-8:
            return (0.0, Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0))

        u_ji = v_ji / r_ji
        u_jk = v_jk / r_jk

        cos_theta = u_ji.dot(u_jk)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.acos(cos_theta)

        delta_theta = theta - angle.theta_eq
        energy = 0.5 * angle.k_angle * delta_theta * delta_theta

        sin_theta = math.sqrt(max(1e-12, 1.0 - cos_theta * cos_theta))
        d_energy = angle.k_angle * delta_theta

        # Forces on end atoms from analytical gradient:
        # Fi = (dE / sin_theta) * (u_jk - cos_theta * u_ji) / r_ji
        coeff_i = (d_energy / sin_theta) / r_ji
        f_i = coeff_i * (u_jk - cos_theta * u_ji)

        coeff_k = (d_energy / sin_theta) / r_jk
        f_k = coeff_k * (u_ji - cos_theta * u_jk)

        # By translational invariance of isolated angle potential
        f_j = -(f_i + f_k)

        return (energy, f_i, f_j, f_k)


class PeriodicDihedralPotential:
    """
    Periodic torsional potential for 4 bonded atoms: i - j - k - l.
    V(phi) = k_dihedral * [1 + cos(periodicity * phi - phase_rad)]
    """
    @staticmethod
    def evaluate(
        pos_i: Vector3D,
        pos_j: Vector3D,
        pos_k: Vector3D,
        pos_l: Vector3D,
        dihedral: Dihedral,
    ) -> Tuple[float, Vector3D, Vector3D, Vector3D, Vector3D]:
        """
        Compute torsional dihedral energy and forces on atoms i, j, k, l.
        Returns: (energy, f_i, f_j, f_k, f_l)
        """
        b1 = pos_j - pos_i
        b2 = pos_k - pos_j
        b3 = pos_l - pos_k

        b2_len = b2.norm()
        if b2_len < 1e-8:
            z = Vector3D(0.0, 0.0, 0.0)
            return (0.0, z, z, z, z)

        # Normal vectors to planes (i, j, k) and (j, k, l)
        n1 = b1.cross(b2)
        n2 = b2.cross(b3)

        n1_sq = n1.norm_sq()
        n2_sq = n2.norm_sq()
        if n1_sq < 1e-12 or n2_sq < 1e-12:
            z = Vector3D(0.0, 0.0, 0.0)
            return (0.0, z, z, z, z)

        # Dihedral angle calculation using atan2 for full 2*pi range
        u_b2 = b2 / b2_len
        cos_phi = n1.dot(n2) / math.sqrt(n1_sq * n2_sq)
        cos_phi = max(-1.0, min(1.0, cos_phi))
        sin_phi = (n1.cross(n2)).dot(u_b2) / math.sqrt(n1_sq * n2_sq)

        phi = math.atan2(sin_phi, cos_phi)

        n = dihedral.periodicity
        arg = n * phi - dihedral.phase_rad
        energy = dihedral.k_dihedral * (1.0 + math.cos(arg))

        # Torque derivative: dV/dphi = -n * k_dihedral * sin(n*phi - delta)
        d_v_dphi = -float(n) * dihedral.k_dihedral * math.sin(arg)

        # Analytical Blondel-Karplus force projection
        f_i = (-d_v_dphi * b2_len / n1_sq) * n1
        f_l = (+d_v_dphi * b2_len / n2_sq) * n2

        # Lever arm distribution onto inner hinge atoms j and k
        dot_12 = b1.dot(b2) / (b2_len * b2_len)
        dot_32 = b3.dot(b2) / (b2_len * b2_len)

        f_j = -f_i + (dot_12 * f_i) - (dot_32 * f_l)
        f_k = -(f_i + f_j + f_l)

        return (energy, f_i, f_j, f_k, f_l)
