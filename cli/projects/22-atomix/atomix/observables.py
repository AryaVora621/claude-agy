"""
Atomix: Thermodynamic Observables, Radial Distribution Function, and Structural Metrics.
Implements:
  - Instantaneous kinetic energy, potential energy, temperature, and virial pressure
  - Radial Distribution Function g(r) with shell volume normalization
  - Mean Squared Displacement (MSD) and Einstein self-diffusion coefficient D
  - Radius of Gyration Rg and Root Mean Square Deviation (RMSD)
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from atomix.types import Vector3D, Atom, SimulationBox, KB_REDUCED


@dataclass
class ThermodynamicState:
    """
    Snapshot of physical and thermodynamic properties at an instant in time.
    """
    step: int
    time: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    temperature: float
    pressure: float
    virial: float
    volume: float
    density: float

    def __repr__(self) -> str:
        return (
            f"Step={self.step:6d} | Time={self.time:8.3f} | "
            f"Temp={self.temperature:7.3f} | Pres={self.pressure:8.3f} | "
            f"E_tot={self.total_energy:10.3f} | E_pot={self.potential_energy:10.3f} | "
            f"E_kin={self.kinetic_energy:10.3f}"
        )


def compute_kinetic_energy(atoms: List[Atom]) -> float:
    """Compute total kinetic energy E_k = 0.5 * sum(m_i * |v_i|^2)."""
    return 0.5 * sum(a.mass * a.velocity.norm_sq() for a in atoms)


def compute_temperature(
    atoms: List[Atom],
    kb: float = KB_REDUCED,
    n_constraints: int = 0,
    remove_drift: bool = True,
) -> float:
    """
    Compute instantaneous kinetic temperature: T = 2*E_k / (N_df * k_B).
    Accounts for degrees of freedom lost to constraints and COM drift.
    """
    n_atoms = len(atoms)
    if n_atoms == 0:
        return 0.0

    n_dofs = 3 * n_atoms - n_constraints
    if remove_drift and n_atoms > 1:
        n_dofs -= 3
    n_dofs = max(1, n_dofs)

    ek = compute_kinetic_energy(atoms)
    return (2.0 * ek) / (n_dofs * kb)


def compute_virial_pressure(
    atoms: List[Atom],
    box: SimulationBox,
    kinetic_energy: float,
    virial_sum: float,
) -> float:
    """
    Compute instantaneous pressure using the Clausius virial theorem:
    P = (2*E_k + W) / (3*V), where W is the total pairwise virial.
    """
    vol = box.volume
    if vol < 1e-12:
        return 0.0
    return (2.0 * kinetic_energy + virial_sum) / (3.0 * vol)


class RadialDistributionFunction:
    """
    Accumulator for the Radial Distribution Function g(r).
    Quantifies local liquid/crystal structure and coordination shells.
    """
    def __init__(self, r_max: float, n_bins: int = 100) -> None:
        if r_max <= 0.0 or n_bins <= 0:
            raise ValueError("r_max and n_bins must be positive")
        self.r_max = float(r_max)
        self.n_bins = int(n_bins)
        self.dr = self.r_max / self.n_bins
        self.inv_dr = 1.0 / self.dr

        self.hist = [0] * self.n_bins
        self.num_samples = 0
        self.total_pairs_sampled = 0

    def sample(self, atoms: List[Atom], box: SimulationBox) -> None:
        """Accumulate pair distances into radial bins."""
        n = len(atoms)
        if n < 2:
            return

        for i in range(n):
            pos_i = atoms[i].position
            for j in range(i + 1, n):
                r_vec = box.minimum_image_vector(pos_i, atoms[j].position)
                r = r_vec.norm()
                if r < self.r_max:
                    bin_idx = int(r * self.inv_dr)
                    if 0 <= bin_idx < self.n_bins:
                        # Count both (i,j) and (j,i) pairs
                        self.hist[bin_idx] += 2

        self.num_samples += 1
        self.total_pairs_sampled += n * (n - 1)

    def get_distribution(self, box: SimulationBox, num_atoms: int) -> Tuple[List[float], List[float]]:
        """
        Normalize accumulated histogram into g(r).
        Returns: (r_centers, g_values)
        """
        if self.num_samples == 0 or num_atoms < 2:
            r_centers = [(i + 0.5) * self.dr for i in range(self.n_bins)]
            return (r_centers, [0.0] * self.n_bins)

        bulk_density = num_atoms / box.volume
        r_centers: List[float] = []
        g_vals: List[float] = []

        for i in range(self.n_bins):
            r_low = i * self.dr
            r_high = (i + 1) * self.dr
            r_mid = 0.5 * (r_low + r_high)
            r_centers.append(r_mid)

            # Spherical shell volume: 4/3 * pi * (r_high^3 - r_low^3)
            shell_vol = (4.0 / 3.0) * math.pi * (r_high**3 - r_low**3)
            # Expected number of ideal gas atoms in this shell
            n_ideal = bulk_density * shell_vol

            # Normalized g(r) = observed_count / (N_atoms * N_samples * n_ideal)
            norm = float(num_atoms * self.num_samples * n_ideal)
            g = self.hist[i] / norm if norm > 1e-12 else 0.0
            g_vals.append(g)

        return (r_centers, g_vals)


class MSDTracker:
    """
    Tracks unwrapped particle trajectories to compute Mean Squared Displacement (MSD)
    and self-diffusion coefficient D via Einstein's relation: MSD(t) = 6*D*t.
    """
    def __init__(self, initial_atoms: List[Atom]) -> None:
        self.initial_positions = [atom.position for atom in initial_atoms]
        # Track previous wrapped positions to detect boundary crossings
        self.prev_wrapped = [atom.position for atom in initial_atoms]
        # Cumulative box displacement offset vectors
        self.box_crossings = [Vector3D(0.0, 0.0, 0.0) for _ in initial_atoms]

    def update(self, atoms: List[Atom], box: SimulationBox) -> float:
        """
        Update unwrapped positions by detecting periodic jumps.
        Returns current instantaneous MSD.
        """
        n = len(atoms)
        if n == 0:
            return 0.0

        total_sq_disp = 0.0
        for i in range(n):
            curr = atoms[i].position
            prev = self.prev_wrapped[i]

            # Detect periodic boundary crossing jump: |dx| > L / 2
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dz = curr.z - prev.z

            shift_x = 0.0
            shift_y = 0.0
            shift_z = 0.0

            if dx > 0.5 * box.lx:
                shift_x = -box.lx
            elif dx < -0.5 * box.lx:
                shift_x = box.lx

            if dy > 0.5 * box.ly:
                shift_y = -box.ly
            elif dy < -0.5 * box.ly:
                shift_y = box.ly

            if dz > 0.5 * box.lz:
                shift_z = -box.lz
            elif dz < -0.5 * box.lz:
                shift_z = box.lz

            # Accumulate offset
            self.box_crossings[i] = self.box_crossings[i] + Vector3D(shift_x, shift_y, shift_z)
            self.prev_wrapped[i] = curr

            # Unwrapped position: wrapped_coord + accumulated_box_shifts
            unwrapped = curr + self.box_crossings[i]
            disp = unwrapped - self.initial_positions[i]
            total_sq_disp += disp.norm_sq()

        return total_sq_disp / n

    @staticmethod
    def calculate_diffusion_coefficient(msd: float, elapsed_time: float) -> float:
        """Einstein relation: D = MSD / (6 * t)."""
        if elapsed_time <= 1e-12:
            return 0.0
        return msd / (6.0 * elapsed_time)


def compute_radius_of_gyration(atoms: List[Atom]) -> float:
    """
    Compute radius of gyration R_g:
    R_g = sqrt((1 / M) * sum(m_i * |r_i - r_com|^2))
    Measures spatial compactness of polymers or biomolecules.
    """
    if not atoms:
        return 0.0

    total_mass = sum(a.mass for a in atoms)
    com_x = sum(a.mass * a.position.x for a in atoms) / total_mass
    com_y = sum(a.mass * a.position.y for a in atoms) / total_mass
    com_z = sum(a.mass * a.position.z for a in atoms) / total_mass
    r_com = Vector3D(com_x, com_y, com_z)

    sum_sq = sum(a.mass * (a.position - r_com).norm_sq() for a in atoms)
    return math.sqrt(sum_sq / total_mass)


def compute_rmsd(current_atoms: List[Atom], reference_positions: List[Vector3D]) -> float:
    """
    Compute coordinate Root Mean Square Deviation against a reference structure.
    RMSD = sqrt((1 / N) * sum(|r_i - r_ref_i|^2))
    """
    n = min(len(current_atoms), len(reference_positions))
    if n == 0:
        return 0.0

    sum_sq = sum((current_atoms[i].position - reference_positions[i]).norm_sq() for i in range(n))
    return math.sqrt(sum_sq / n)
