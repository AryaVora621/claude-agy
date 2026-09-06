"""
Atomix: Core Data Structures, Spatial Vectors, and Physical Types.
Defines:
  - Vector3D with algebraic methods (dot, cross, norm, normalization)
  - Atom representation with coordinates, velocities, forces, and force field parameters
  - Bond, Angle, and Dihedral topology records
  - SimulationBox with Periodic Boundary Conditions and Minimum Image Convention
  - Physical constants in reduced and real biochemical units
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional


# Physical Constants in Standard Real Units:
# Distance: Angstroms (A), Time: picoseconds (ps), Mass: g/mol (amu), Energy: kJ/mol
KB_REAL_KJ = 0.008314462618  # Boltzmann constant in kJ / (mol * K)
KB_REDUCED = 1.0              # In standard dimensionless Lennard-Jones units

# Coulomb electrostatic constant f = 1 / (4 * pi * eps_0)
# Yields energy in kJ/mol for charges in elementary units e and distances in Angstroms
COULOMB_CONSTANT_KJ = 1389.35456

# Standard CPK element color scheme in 24-bit RGB (R, G, B)
CPK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "H": (240, 240, 245),   # White / Silver
    "C": (85, 90, 100),     # Slate Gray / Charcoal
    "N": (50, 110, 235),    # Sapphire Blue
    "O": (230, 45, 45),     # Crimson Red
    "S": (240, 205, 35),    # Golden Yellow
    "P": (245, 135, 30),    # Amber Orange
    "AR": (40, 205, 205),   # Cyan for Argon noble gas
    "HE": (180, 230, 245),  # Light Sky Blue
    "NA": (145, 75, 230),   # Violet Sodium
    "CL": (35, 200, 75),    # Green Chlorine
    "FE": (210, 115, 40),   # Rust Orange Iron
    "CA": (120, 120, 140),  # Dark Silver Calcium
    "DEFAULT": (160, 165, 175),
}


class Vector3D:
    """
    Immutable 3-component Euclidean spatial vector.
    Methods are optimized for fast inner-loop execution in molecular force loops.
    """
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, key, value):
        raise AttributeError("Vector3D instances are immutable.")

    def __repr__(self) -> str:
        return f"Vector3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> Vector3D:
        inv = 1.0 / scalar
        return Vector3D(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    def dot(self, other: Vector3D) -> float:
        """Euclidean inner product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3D) -> Vector3D:
        """3D vector cross product."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm_sq(self) -> float:
        """Square of Euclidean L2 norm."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        """Euclidean L2 norm."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vector3D:
        """Return unit vector. Returns zero vector if magnitude is near zero."""
        n = self.norm()
        if n < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Vector3D(self.x * inv, self.y * inv, self.z * inv)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Atom:
    """
    Physical representation of an atom or pseudo-atom site.
    """
    id: int
    name: str
    element: str
    position: Vector3D
    velocity: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    force: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    mass: float = 1.0
    charge: float = 0.0        # In elementary units e
    sigma: float = 1.0         # LJ distance parameter (Angstroms or reduced units)
    epsilon: float = 1.0       # LJ potential well depth (kJ/mol or reduced units)
    molecule_id: int = 0

    @property
    def cpk_color(self) -> Tuple[int, int, int]:
        """Lookup standard CPK RGB color based on element symbol."""
        el_upper = self.element.upper()
        return CPK_COLORS.get(el_upper, CPK_COLORS["DEFAULT"])


@dataclass(frozen=True)
class Bond:
    """
    Covalent chemical bond between two atoms.
    """
    atom1_id: int
    atom2_id: int
    length_eq: float          # Equilibrium bond length r_0
    k_spring: float           # Spring force constant k_b
    is_rigid: bool = False    # True if enforced by SHAKE algorithm rather than harmonic potential


@dataclass(frozen=True)
class Angle:
    """
    Valence angle bending term between three connected atoms (atom1 - vertex - atom3).
    """
    atom1_id: int
    vertex_id: int
    atom3_id: int
    theta_eq: float           # Equilibrium angle in radians
    k_angle: float            # Angular spring constant k_theta


@dataclass(frozen=True)
class Dihedral:
    """
    Proper or improper torsional dihedral term (atom1 - atom2 - atom3 - atom4).
    V(phi) = k_dihedral * [1 + cos(periodicity * phi - phase_rad)]
    """
    atom1_id: int
    atom2_id: int
    atom3_id: int
    atom4_id: int
    periodicity: int          # Multiplicity n (typically 1, 2, 3, or 4)
    phase_rad: float          # Phase offset delta_n in radians
    k_dihedral: float         # Torsional barrier height k_phi


class SimulationBox:
    """
    Orthogonal 3D periodic simulation cell [0, lx] x [0, ly] x [0, lz].
    Implements Periodic Boundary Conditions (PBC) and Minimum Image Convention.
    """
    def __init__(self, lx: float, ly: float, lz: float) -> None:
        if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
            raise ValueError(f"Simulation box dimensions must be strictly positive: ({lx}, {ly}, {lz})")
        self.lx = float(lx)
        self.ly = float(ly)
        self.lz = float(lz)
        self.inv_lx = 1.0 / self.lx
        self.inv_ly = 1.0 / self.ly
        self.inv_lz = 1.0 / self.lz

    @property
    def volume(self) -> float:
        """Volume of the orthogonal cell."""
        return self.lx * self.ly * self.lz

    def wrap_position(self, pos: Vector3D) -> Vector3D:
        """
        Fold an atomic coordinate back into the primary box [0, L).
        Uses math.floor to ensure correct handling of negative coordinates.
        """
        wx = pos.x - self.lx * math.floor(pos.x * self.inv_lx)
        wy = pos.y - self.ly * math.floor(pos.y * self.inv_ly)
        wz = pos.z - self.lz * math.floor(pos.z * self.inv_lz)
        return Vector3D(wx, wy, wz)

    def minimum_image_vector(self, r1: Vector3D, r2: Vector3D) -> Vector3D:
        """
        Compute minimum image separation vector r12 = r2 - r1.
        Shifts the displacement to lie within [-L/2, +L/2].
        """
        dx = r2.x - r1.x
        dy = r2.y - r1.y
        dz = r2.z - r1.z

        # Rounding maps offsets beyond L/2 back across the periodic boundary
        dx -= self.lx * round(dx * self.inv_lx)
        dy -= self.ly * round(dy * self.inv_ly)
        dz -= self.lz * round(dz * self.inv_lz)

        return Vector3D(dx, dy, dz)

    def scale(self, scale_factor: float) -> None:
        """
        Isotropically rescale the simulation box dimensions (used by barostats).
        """
        if scale_factor <= 0.0:
            raise ValueError(f"Scale factor must be positive, got {scale_factor}")
        self.lx *= scale_factor
        self.ly *= scale_factor
        self.lz *= scale_factor
        self.inv_lx = 1.0 / self.lx
        self.inv_ly = 1.0 / self.ly
        self.inv_lz = 1.0 / self.lz
