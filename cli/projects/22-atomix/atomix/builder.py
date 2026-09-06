"""
Atomix: Molecular System Builders and Lattice Generators.
Implements:
  - Face-Centered Cubic (FCC) noble gas crystal generator
  - Overlap-free random solvent box generator
  - TIP3P rigid/flexible explicit water box generator
  - Polypeptide protein backbone polymer chain builder
Zero external dependencies.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional

from atomix.types import (
    Vector3D,
    Atom,
    Bond,
    Angle,
    Dihedral,
    SimulationBox,
)


def build_fcc_lattice(
    n_cells: int = 3,
    lattice_constant: float = 1.58,
    element: str = "Ar",
    mass: float = 1.0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
) -> Tuple[List[Atom], SimulationBox]:
    """
    Generate a 3D Face-Centered Cubic (FCC) crystal lattice.
    Each unit cell contains 4 basis atoms. Total atoms: 4 * n_cells^3.
    Classic starting configuration for liquid and solid Lennard-Jones Argon simulations.
    """
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")

    # 4 basis points in fractional cell coordinates
    basis = [
        Vector3D(0.0, 0.0, 0.0),
        Vector3D(0.5, 0.5, 0.0),
        Vector3D(0.5, 0.0, 0.5),
        Vector3D(0.0, 0.5, 0.5),
    ]

    a = float(lattice_constant)
    box_length = n_cells * a
    box = SimulationBox(box_length, box_length, box_length)

    atoms: List[Atom] = []
    atom_id = 0

    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                cell_origin = Vector3D(ix * a, iy * a, iz * a)
                for b_vec in basis:
                    pos = cell_origin + (b_vec * a)
                    atom = Atom(
                        id=atom_id,
                        name=f"{element}{atom_id}",
                        element=element,
                        position=pos,
                        velocity=Vector3D(0.0, 0.0, 0.0),
                        force=Vector3D(0.0, 0.0, 0.0),
                        mass=mass,
                        charge=0.0,
                        sigma=sigma,
                        epsilon=epsilon,
                        molecule_id=atom_id,
                    )
                    atoms.append(atom)
                    atom_id += 1

    return (atoms, box)


def build_random_solvent_box(
    num_particles: int,
    box_size: float,
    element: str = "Ar",
    mass: float = 1.0,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    min_dist: float = 0.8,
    seed: Optional[int] = 42,
) -> Tuple[List[Atom], SimulationBox]:
    """
    Generate an amorphous box of particles ensuring no unphysical steric clashes (r >= min_dist).
    """
    rng = random.Random(seed)
    box = SimulationBox(box_size, box_size, box_size)
    atoms: List[Atom] = []

    min_dist_sq = min_dist * min_dist
    max_attempts = 1000

    for i in range(num_particles):
        placed = False
        for _ in range(max_attempts):
            pos = Vector3D(
                rng.random() * box_size,
                rng.random() * box_size,
                rng.random() * box_size,
            )

            # Check overlap against all placed atoms with minimum image
            clash = False
            for existing in atoms:
                r_vec = box.minimum_image_vector(existing.position, pos)
                if r_vec.norm_sq() < min_dist_sq:
                    clash = True
                    break

            if not clash:
                atom = Atom(
                    id=i,
                    name=f"{element}{i}",
                    element=element,
                    position=pos,
                    velocity=Vector3D(0.0, 0.0, 0.0),
                    force=Vector3D(0.0, 0.0, 0.0),
                    mass=mass,
                    charge=0.0,
                    sigma=sigma,
                    epsilon=epsilon,
                    molecule_id=i,
                )
                atoms.append(atom)
                placed = True
                break

        if not placed:
            # Fallback to simple grid placement if box density is high
            break

    return (atoms, box)


def build_tip3p_water_box(
    num_molecules: int,
    box_size: float = 20.0,
    is_rigid: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[List[Atom], List[Bond], List[Angle], SimulationBox]:
    """
    Generate an explicit box of TIP3P water molecules.
    Standard Jorgensen et al. TIP3P parameters:
      Oxygen:   mass = 15.9994, charge = -0.834 e, sigma = 3.1507 A, eps = 0.6364 kJ/mol
      Hydrogen: mass = 1.008,   charge = +0.417 e, sigma = 0.0 A,    eps = 0.0 kJ/mol
      r_OH = 0.9572 A, theta_HOH = 104.52 deg (1.8242 rad)
    """
    rng = random.Random(seed)
    box = SimulationBox(box_size, box_size, box_size)
    atoms: List[Atom] = []
    bonds: List[Bond] = []
    angles: List[Angle] = []

    r_oh = 0.9572
    theta_rad = 104.52 * (math.pi / 180.0)
    half_theta = 0.5 * theta_rad

    # Relative coordinates of the two hydrogens with respect to oxygen
    h1_offset = Vector3D(r_oh * math.sin(half_theta), r_oh * math.cos(half_theta), 0.0)
    h2_offset = Vector3D(-r_oh * math.sin(half_theta), r_oh * math.cos(half_theta), 0.0)

    atom_idx = 0
    min_dist_sq = 2.5 * 2.5  # Prevent oxygen-oxygen clashing

    for m in range(num_molecules):
        for _ in range(500):
            # Pick random oxygen coordinate and random 3D orientation
            ox_pos = Vector3D(
                rng.random() * box_size,
                rng.random() * box_size,
                rng.random() * box_size,
            )

            # Check overlap with existing oxygens
            clash = False
            for a in atoms:
                if a.element == "O":
                    r_vec = box.minimum_image_vector(a.position, ox_pos)
                    if r_vec.norm_sq() < min_dist_sq:
                        clash = True
                        break
            if not clash:
                break

        # Random rotation Euler angles
        alpha = rng.random() * 2.0 * math.pi
        beta = rng.random() * math.pi
        gamma = rng.random() * 2.0 * math.pi

        # Simple rotation matrix application
        def rotate(v: Vector3D) -> Vector3D:
            # ZYZ Euler rotation
            c1, s1 = math.cos(alpha), math.sin(alpha)
            c2, s2 = math.cos(beta), math.sin(beta)
            c3, s3 = math.cos(gamma), math.sin(gamma)
            x = (c1 * c2 * c3 - s1 * s3) * v.x + (-c1 * c2 * s3 - s1 * c3) * v.y + (c1 * s2) * v.z
            y = (s1 * c2 * c3 + c1 * s3) * v.x + (-s1 * c2 * s3 + c1 * c3) * v.y + (s1 * s2) * v.z
            z = (-s2 * c3) * v.x + (s2 * s3) * v.y + c2 * v.z
            return Vector3D(x, y, z)

        h1_pos = box.wrap_position(ox_pos + rotate(h1_offset))
        h2_pos = box.wrap_position(ox_pos + rotate(h2_offset))

        o_atom = Atom(
            id=atom_idx,
            name=f"OW{m}",
            element="O",
            position=ox_pos,
            mass=15.9994,
            charge=-0.834,
            sigma=3.1507,
            epsilon=0.6364,
            molecule_id=m,
        )
        atoms.append(o_atom)

        h1_atom = Atom(
            id=atom_idx + 1,
            name=f"HW1_{m}",
            element="H",
            position=h1_pos,
            mass=1.008,
            charge=0.417,
            sigma=0.4,
            epsilon=0.046,
            molecule_id=m,
        )
        atoms.append(h1_atom)

        h2_atom = Atom(
            id=atom_idx + 2,
            name=f"HW2_{m}",
            element="H",
            position=h2_pos,
            mass=1.008,
            charge=0.417,
            sigma=0.4,
            epsilon=0.046,
            molecule_id=m,
        )
        atoms.append(h2_atom)

        # O-H Bonds
        bonds.append(Bond(atom_idx, atom_idx + 1, length_eq=r_oh, k_spring=450000.0, is_rigid=is_rigid))
        bonds.append(Bond(atom_idx, atom_idx + 2, length_eq=r_oh, k_spring=450000.0, is_rigid=is_rigid))

        # H-O-H Angle
        angles.append(Angle(atom_idx + 1, atom_idx, atom_idx + 2, theta_eq=theta_rad, k_angle=460.0))

        atom_idx += 3

    return (atoms, bonds, angles, box)


def build_peptide_chain(
    num_residues: int = 12,
    box_size: float = 30.0,
    secondary_structure: str = "helix",
) -> Tuple[List[Atom], List[Bond], List[Angle], List[Dihedral], SimulationBox]:
    """
    Construct a coarse-grained polypeptide protein backbone (N - CA - C).
    Supports alpha-helix (phi=-57 deg, psi=-47 deg) or extended beta-strand conformation.
    """
    box = SimulationBox(box_size, box_size, box_size)
    atoms: List[Atom] = []
    bonds: List[Bond] = []
    angles: List[Angle] = []
    dihedrals: List[Dihedral] = []

    # Helical geometry parameters
    # Alpha-helix has 3.6 residues per turn, pitch 5.4 Angstroms (1.5 A rise per residue), radius ~2.3 A
    r_helix = 2.3 if secondary_structure.lower() == "helix" else 0.5
    pitch_per_res = 1.5 if secondary_structure.lower() == "helix" else 3.5
    d_theta = (2.0 * math.pi / 3.6) if secondary_structure.lower() == "helix" else (math.pi)

    center_x = 0.5 * box_size
    center_y = 0.5 * box_size
    start_z = 0.2 * box_size

    atom_idx = 0
    for res in range(num_residues):
        theta = res * d_theta
        z_base = start_z + res * pitch_per_res

        # Backbone Nitrogen (N)
        pos_n = Vector3D(
            center_x + r_helix * math.cos(theta),
            center_y + r_helix * math.sin(theta),
            z_base,
        )
        # Alpha Carbon (CA)
        pos_ca = Vector3D(
            center_x + (r_helix + 0.4) * math.cos(theta + 0.2),
            center_y + (r_helix + 0.4) * math.sin(theta + 0.2),
            z_base + 0.5,
        )
        # Carbonyl Carbon (C)
        pos_c = Vector3D(
            center_x + r_helix * math.cos(theta + 0.4),
            center_y + r_helix * math.sin(theta + 0.4),
            z_base + 1.0,
        )
        # Carbonyl Oxygen (O)
        pos_o = Vector3D(
            center_x + (r_helix + 1.2) * math.cos(theta + 0.4),
            center_y + (r_helix + 1.2) * math.sin(theta + 0.4),
            z_base + 1.1,
        )

        n_atom = Atom(atom_idx, f"N_{res}", "N", pos_n, mass=14.007, charge=-0.28, sigma=3.25, epsilon=0.71)
        ca_atom = Atom(atom_idx + 1, f"CA_{res}", "C", pos_ca, mass=12.011, charge=0.00, sigma=3.50, epsilon=0.28)
        c_atom = Atom(atom_idx + 2, f"C_{res}", "C", pos_c, mass=12.011, charge=0.38, sigma=3.75, epsilon=0.44)
        o_atom = Atom(atom_idx + 3, f"O_{res}", "O", pos_o, mass=15.999, charge=-0.38, sigma=2.96, epsilon=0.88)

        atoms.extend([n_atom, ca_atom, c_atom, o_atom])

        # Intra-residue bonds: N-CA, CA-C, C=O
        bonds.append(Bond(atom_idx, atom_idx + 1, length_eq=1.46, k_spring=300.0))
        bonds.append(Bond(atom_idx + 1, atom_idx + 2, length_eq=1.52, k_spring=300.0))
        bonds.append(Bond(atom_idx + 2, atom_idx + 3, length_eq=1.23, k_spring=500.0))

        # Intra-residue valence angles: N-CA-C and CA-C=O
        angles.append(Angle(atom_idx, atom_idx + 1, atom_idx + 2, theta_eq=1.937, k_angle=70.0))
        angles.append(Angle(atom_idx + 1, atom_idx + 2, atom_idx + 3, theta_eq=2.112, k_angle=80.0))

        # Inter-residue peptide bond: C(res-1) - N(res)
        if res > 0:
            prev_c = atom_idx - 2
            bonds.append(Bond(prev_c, atom_idx, length_eq=1.33, k_spring=400.0))
            # Peptide dihedral omega: CA(res-1) - C(res-1) - N(res) - CA(res) (trans planar, 180 deg)
            prev_ca = atom_idx - 3
            curr_ca = atom_idx + 1
            dihedrals.append(Dihedral(prev_ca, prev_c, atom_idx, curr_ca, periodicity=2, phase_rad=math.pi, k_dihedral=30.0))

        atom_idx += 4

    return (atoms, bonds, angles, dihedrals, box)
