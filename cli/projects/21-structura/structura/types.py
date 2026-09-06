"""
Structura: Core Types, Material Constitutive Models, and Stress/Strain Tensors.
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict


@dataclass
class Node2D:
    """
    2D nodal point with spatial coordinates and global degree-of-freedom indices.
    """
    id: int
    x: float
    y: float
    dof_indices: Tuple[int, ...] = field(default_factory=tuple)

    def distance_to(self, other: Node2D) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class Material:
    """
    Linear isotropic elastic material properties.
    """
    name: str = "Structural Steel"
    elastic_modulus: float = 210e9       # Young's modulus E (Pascals)
    poissons_ratio: float = 0.30         # Poisson's ratio nu (dimensionless)
    density: float = 7850.0              # Density rho (kg/m^3)
    yield_strength: float = 250e6        # Yield strength sigma_y (Pascals)
    thickness: float = 1.0               # Member / plate thickness t (meters)

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G = E / (2 * (1 + nu))."""
        return self.elastic_modulus / (2.0 * (1.0 + self.poissons_ratio))

    def get_elasticity_matrix(self, plane_strain: bool = False) -> List[List[float]]:
        """
        Compute 3x3 constitutive elasticity matrix D for 2D continuum problems.
        Relates stress vector [sigma_x, sigma_y, tau_xy]^T to strain [eps_x, eps_y, gamma_xy]^T.
        """
        e = self.elastic_modulus
        nu = self.poissons_ratio
        g = self.shear_modulus

        if not plane_strain:
            # Plane Stress (sigma_z = 0, thin plates)
            factor = e / (1.0 - nu * nu)
            return [
                [factor, nu * factor, 0.0],
                [nu * factor, factor, 0.0],
                [0.0, 0.0, g],
            ]
        else:
            # Plane Strain (eps_z = 0, thick structures)
            factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
            return [
                [(1.0 - nu) * factor, nu * factor, 0.0],
                [nu * factor, (1.0 - nu) * factor, 0.0],
                [0.0, 0.0, g],
            ]


# Predefined standard engineering materials
STRUCTURAL_STEEL = Material(
    name="Structural Steel (A36)",
    elastic_modulus=200e9,
    poissons_ratio=0.30,
    density=7850.0,
    yield_strength=250e6,
    thickness=0.01,
)

ALUMINUM_6061 = Material(
    name="Aluminum 6061-T6",
    elastic_modulus=69e9,
    poissons_ratio=0.33,
    density=2700.0,
    yield_strength=276e6,
    thickness=0.01,
)

TITANIUM_TI6AL4V = Material(
    name="Titanium Grade 5 (Ti-6Al-4V)",
    elastic_modulus=114e9,
    poissons_ratio=0.34,
    density=4430.0,
    yield_strength=880e6,
    thickness=0.01,
)

STRUCTURAL_CONCRETE = Material(
    name="Structural Concrete",
    elastic_modulus=30e9,
    poissons_ratio=0.20,
    density=2400.0,
    yield_strength=30e6,
    thickness=0.10,
)


@dataclass
class BoundaryCondition:
    """
    Prescribed displacement or rotation constraint at a node degree of freedom.
    dof: 0 for Ux, 1 for Uy, 2 for Theta_z (bending rotation).
    """
    node_id: int
    dof: int
    prescribed_value: float = 0.0


@dataclass
class NodalLoad:
    """
    External concentrated force or moment applied at a node.
    """
    node_id: int
    fx: float = 0.0
    fy: float = 0.0
    moment: float = 0.0


@dataclass
class StressTensor2D:
    """
    2D Cauchy stress tensor state (sigma_x, sigma_y, tau_xy).
    All values in Pascals (N/m^2).
    """
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    tau_xy: float = 0.0

    def principal_stresses(self) -> Tuple[float, float]:
        """
        Compute major and minor principal stresses (sigma_1, sigma_2) where sigma_1 >= sigma_2.
        """
        avg = 0.5 * (self.sigma_x + self.sigma_y)
        diff = 0.5 * (self.sigma_x - self.sigma_y)
        radius = math.sqrt(diff * diff + self.tau_xy * self.tau_xy)
        return (avg + radius, avg - radius)

    def max_shear_stress(self) -> float:
        """
        Compute maximum in-plane shear stress tau_max = (sigma_1 - sigma_2) / 2 = Mohr circle radius.
        """
        diff = 0.5 * (self.sigma_x - self.sigma_y)
        return math.sqrt(diff * diff + self.tau_xy * self.tau_xy)

    def von_mises(self) -> float:
        """
        Compute Von Mises equivalent yield stress:
        sigma_v = sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2 + 3*tau_xy^2)
        """
        term1 = self.sigma_x * self.sigma_x - self.sigma_x * self.sigma_y + self.sigma_y * self.sigma_y
        term2 = 3.0 * self.tau_xy * self.tau_xy
        val = term1 + term2
        return math.sqrt(val) if val > 0.0 else 0.0

    def principal_angle_rad(self) -> float:
        """Angle theta_p of major principal stress axis relative to x-axis."""
        return 0.5 * math.atan2(2.0 * self.tau_xy, self.sigma_x - self.sigma_y)


@dataclass
class StrainTensor2D:
    """
    2D engineering strain tensor state (eps_x, eps_y, gamma_xy).
    """
    eps_x: float = 0.0
    eps_y: float = 0.0
    gamma_xy: float = 0.0

    def principal_strains(self) -> Tuple[float, float]:
        """Compute major and minor principal strains."""
        avg = 0.5 * (self.eps_x + self.eps_y)
        diff = 0.5 * (self.eps_x - self.eps_y)
        radius = math.sqrt(diff * diff + 0.25 * self.gamma_xy * self.gamma_xy)
        return (avg + radius, avg - radius)
