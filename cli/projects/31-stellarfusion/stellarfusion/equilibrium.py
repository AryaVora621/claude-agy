"""
Grad-Shafranov 2D Magnetohydrodynamic (MHD) Plasma Equilibrium Engine.

Solves the non-linear 2D elliptic partial differential equation for axisymmetric
toroidal plasma equilibria in tokamak geometries:
    Delta* psi = -mu_0 * R^2 * dp/dpsi - F * dF/dpsi

where:
    Delta* psi = d^2 psi / dR^2 - (1/R) * dpsi/dR + d^2 psi / dZ^2
    psi(R, Z) is the poloidal magnetic flux function
    p(psi) is the scalar kinetic plasma pressure profile
    F(psi) = R * B_phi is the poloidal current stream function

Includes exact Solovev analytical benchmarks, successive over-relaxation (SOR)
multigrid solvers, and magnetic axis locator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


# Physical constants (SI units)
MU_0: float = 4.0 * math.pi * 1e-7  # Vacuum magnetic permeability (H/m)
ELEMENTARY_CHARGE: float = 1.602176634e-19  # Elementary charge (C)
KILO_ELECTRON_VOLT: float = 1.602176634e-16  # 1 keV in Joules


@dataclass(frozen=True)
class Grid2D:
    """Two-dimensional cylindrical coordinate grid (R, Z) for toroidal geometries."""

    nr: int
    nz: int
    r_min: float
    r_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if self.nr < 5 or self.nz < 5:
            raise ValueError("Grid resolution must be at least 5x5 nodes.")
        if self.r_min <= 0.0:
            raise ValueError("R_min must be strictly positive in cylindrical coordinates.")
        if self.r_max <= self.r_min:
            raise ValueError("R_max must be greater than R_min.")
        if self.z_max <= self.z_min:
            raise ValueError("Z_max must be greater than Z_min.")

    @property
    def dr(self) -> float:
        return (self.r_max - self.r_min) / (self.nr - 1)

    @property
    def dz(self) -> float:
        return (self.z_max - self.z_min) / (self.nz - 1)

    def r_at(self, i: int) -> float:
        """Major radial coordinate at radial index i."""
        return self.r_min + i * self.dr

    def z_at(self, j: int) -> float:
        """Vertical coordinate at vertical index j."""
        return self.z_min + j * self.dz

    def index_of_r(self, r: float) -> float:
        """Continuous index corresponding to radial position r."""
        return (r - self.r_min) / self.dr

    def index_of_z(self, z: float) -> float:
        """Continuous index corresponding to vertical position z."""
        return (z - self.z_min) / self.dz

    def contains(self, r: float, z: float) -> bool:
        """Check if coordinates lie within the computational domain."""
        return self.r_min <= r <= self.r_max and self.z_min <= z <= self.z_max


class EquilibriumProfile:
    """
    Parametric kinetic pressure and poloidal current stream function profiles.

    Functions are parameterized against normalized poloidal magnetic flux:
        psi_N = (psi - psi_axis) / (psi_edge - psi_axis) in [0, 1]
    """

    def __init__(
        self,
        p_0: float = 50.0e3,
        f_0: float = 10.0,
        alpha_p: float = 2.0,
        beta_p: float = 1.0,
        alpha_f: float = 1.0,
        diamagnetic_factor: float = 0.05,
    ) -> None:
        # Central kinetic pressure in Pascals (e.g. 50 kPa)
        self.p_0 = p_0
        # Vacuum toroidal magnetic field parameter F_0 = R_0 * B_0 (T*m)
        self.f_0 = f_0
        # Pressure profile peaking exponents
        self.alpha_p = alpha_p
        self.beta_p = beta_p
        self.alpha_f = alpha_f
        self.diamagnetic_factor = diamagnetic_factor

    def pressure(self, psi_n: float) -> float:
        """Kinetic plasma pressure p(psi_N)."""
        if psi_n < 0.0:
            return self.p_0
        if psi_n >= 1.0:
            return 0.0
        return self.p_0 * ((1.0 - psi_n**self.alpha_p) ** self.beta_p)

    def dp_dpsi(self, psi_n: float, delta_psi: float) -> float:
        """Derivative of pressure with respect to dimensional flux dp/dpsi."""
        if delta_psi == 0.0:
            return 0.0
        if psi_n <= 0.0 or psi_n >= 1.0:
            return 0.0
        # Chain rule: dp/dpsi = (dp/dpsi_N) / (psi_edge - psi_axis)
        dp_dpsin = -self.p_0 * self.beta_p * ((1.0 - psi_n**self.alpha_p) ** (self.beta_p - 1.0)) * (
            self.alpha_p * (psi_n ** (self.alpha_p - 1.0))
        )
        return dp_dpsin / delta_psi

    def f_stream(self, psi_n: float, diamagnetic_factor: Optional[float] = None) -> float:
        """Poloidal current stream function F(psi_N) = R * B_phi."""
        if psi_n < 0.0:
            psi_n = 0.0
        if psi_n > 1.0:
            psi_n = 1.0
        factor = self.diamagnetic_factor if diamagnetic_factor is None else diamagnetic_factor
        # Diamagnetic reduction in plasma core
        return self.f_0 * (1.0 - factor * (1.0 - psi_n**self.alpha_f))

    def f_df_dpsi(self, psi_n: float, delta_psi: float, diamagnetic_factor: Optional[float] = None) -> float:
        """F * dF/dpsi term in the Grad-Shafranov RHS."""
        if delta_psi == 0.0:
            return 0.0
        if psi_n <= 0.0 or psi_n >= 1.0:
            return 0.0
        factor = self.diamagnetic_factor if diamagnetic_factor is None else diamagnetic_factor
        f_val = self.f_stream(psi_n, factor)
        df_dpsin = self.f_0 * factor * self.alpha_f * (psi_n ** (self.alpha_f - 1.0))
        df_dpsi = df_dpsin / delta_psi
        return f_val * df_dpsi


class SolovevEquilibrium:
    """
    Exact analytical Solovev equilibrium solution to the Grad-Shafranov equation.

    Solovev solution with elongation kappa and triangularity delta:
        psi(R, Z) = (psi_0 / R_0^4) * [ R^2 * Z^2 + (kappa^2 / 4) * (R^2 - R_0^2)^2 ]

    Yields exact analytical verification of the Shafranov operator:
        Delta* psi = (2 * (1 + kappa^2) * psi_0 / R_0^4) * R^2
    which corresponds to a pure pressure-driven equilibrium with constant p' and F*F' = 0.
    """

    def __init__(
        self,
        r_0: float = 3.0,
        z_0: float = 0.0,
        kappa: float = 1.7,
        psi_0: float = 1.0,
        b_0: float = 2.5,
    ) -> None:
        self.r_0 = r_0
        self.z_0 = z_0
        self.kappa = kappa
        self.psi_0 = psi_0
        self.b_0 = b_0  # Toroidal field at major radius R_0

    def psi(self, r: float, z: float) -> float:
        """Poloidal magnetic flux at (r, z)."""
        rel_z = z - self.z_0
        scale = self.psi_0 / (self.r_0**4)
        term1 = r * r * rel_z * rel_z
        diff_r2 = r * r - self.r_0 * self.r_0
        term2 = (self.kappa * self.kappa / 4.0) * (diff_r2 * diff_r2)
        return scale * (term1 + term2)

    def dpsi_dr(self, r: float, z: float) -> float:
        """First radial derivative dpsi/dr."""
        rel_z = z - self.z_0
        scale = self.psi_0 / (self.r_0**4)
        diff_r2 = r * r - self.r_0 * self.r_0
        return scale * (2.0 * r * rel_z * rel_z + self.kappa * self.kappa * diff_r2 * r)

    def dpsi_dz(self, r: float, z: float) -> float:
        """First vertical derivative dpsi/dz."""
        rel_z = z - self.z_0
        scale = self.psi_0 / (self.r_0**4)
        return scale * (2.0 * r * r * rel_z)

    def d2psi_dr2(self, r: float, z: float) -> float:
        """Second radial derivative d^2 psi / dr^2."""
        rel_z = z - self.z_0
        scale = self.psi_0 / (self.r_0**4)
        diff_r2 = r * r - self.r_0 * self.r_0
        return scale * (2.0 * rel_z * rel_z + self.kappa * self.kappa * (diff_r2 + 2.0 * r * r))

    def d2psi_dz2(self, r: float, z: float) -> float:
        """Second vertical derivative d^2 psi / dz^2."""
        scale = self.psi_0 / (self.r_0**4)
        return scale * (2.0 * r * r)

    def shafranov_operator(self, r: float, z: float) -> float:
        """
        Analytical Shafranov operator Delta* psi:
            d^2 psi / dR^2 - (1/R) * dpsi/dR + d^2 psi / dZ^2
        """
        d2r = self.d2psi_dr2(r, z)
        d1r = self.dpsi_dr(r, z)
        d2z = self.d2psi_dz2(r, z)
        return d2r - (1.0 / r) * d1r + d2z

    def exact_rhs(self, r: float) -> float:
        """Exact theoretical analytical RHS: 2 * (1 + kappa^2) * psi_0 * R^2 / R_0^4."""
        return (2.0 * (1.0 + self.kappa * self.kappa) * self.psi_0 / (self.r_0**4)) * (r * r)

    def magnetic_field(self, r: float, z: float) -> Tuple[float, float, float]:
        """
        Calculate analytical magnetic field vector (B_R, B_phi, B_Z).
            B_R = -(1/R) * dpsi/dz
            B_phi = R_0 * B_0 / R
            B_Z = (1/R) * dpsi/dr
        """
        br = -(1.0 / r) * self.dpsi_dz(r, z)
        bphi = (self.r_0 * self.b_0) / r
        bz = (1.0 / r) * self.dpsi_dr(r, z)
        return br, bphi, bz


class GradShafranovSolver:
    """
    Finite-difference 2D Grad-Shafranov Partial Differential Equation Solver.

    Discretizes the elliptic operator Delta* on a cylindrical grid (R, Z) using
    second-order central differences with successive over-relaxation (SOR) and
    Picard source iteration.
    """

    def __init__(
        self,
        grid: Grid2D,
        profile: Optional[EquilibriumProfile] = None,
        target_ip: float = 1.0e6,  # Target plasma current in Amperes (1 MA)
    ) -> None:
        self.grid = grid
        self.profile = profile or EquilibriumProfile()
        self.target_ip = target_ip

        # 2D Poloidal flux array [nr][nz]
        self.psi: List[List[float]] = [[0.0 for _ in range(grid.nz)] for _ in range(grid.nr)]
        # Toroidal current density array J_phi [nr][nz] in A/m^2
        self.j_phi: List[List[float]] = [[0.0 for _ in range(grid.nz)] for _ in range(grid.nr)]

        # Magnetic axis position and flux extremum
        self.r_axis: float = 0.5 * (grid.r_min + grid.r_max)
        self.z_axis: float = 0.5 * (grid.z_min + grid.z_max)
        self.psi_axis: float = 0.0
        self.psi_edge: float = 1.0

    def initialize_with_solovev(self, solovev: SolovevEquilibrium) -> None:
        """Seed the flux field with an analytical Solovev equilibrium."""
        for i in range(self.grid.nr):
            r = self.grid.r_at(i)
            for j in range(self.grid.nz):
                z = self.grid.z_at(j)
                self.psi[i][j] = solovev.psi(r, z)
        self.update_magnetic_axis()

    def update_magnetic_axis(self) -> None:
        """Locate the magnetic axis (minimum/maximum of poloidal flux psi)."""
        min_psi = float("inf")
        min_i, min_j = 0, 0

        # Search interior nodes for the flux extremum
        for i in range(1, self.grid.nr - 1):
            for j in range(1, self.grid.nz - 1):
                val = self.psi[i][j]
                if val < min_psi:
                    min_psi = val
                    min_i, min_j = i, j

        self.psi_axis = min_psi
        self.r_axis = self.grid.r_at(min_i)
        self.z_axis = self.grid.z_at(min_j)

        # Edge flux taken as average of Dirichlet boundary values
        edge_sum = 0.0
        count = 0
        for i in range(self.grid.nr):
            edge_sum += self.psi[i][0] + self.psi[i][self.grid.nz - 1]
            count += 2
        for j in range(1, self.grid.nz - 1):
            edge_sum += self.psi[0][j] + self.psi[self.grid.nr - 1][j]
            count += 2
        self.psi_edge = edge_sum / max(1, count)

    def normalized_flux(self, psi_val: float) -> float:
        """Compute normalized flux coordinate psi_N in [0, 1]."""
        denom = self.psi_edge - self.psi_axis
        if abs(denom) < 1e-12:
            return 0.0
        val = (psi_val - self.psi_axis) / denom
        return max(0.0, min(1.0, val))

    def evaluate_shafranov_operator_at(self, i: int, j: int) -> float:
        """
        Evaluate finite-difference Shafranov operator Delta* psi at grid node (i, j).
            Delta* psi = d^2 psi / dR^2 - (1/R) * dpsi/dR + d^2 psi / dZ^2
        """
        if i <= 0 or i >= self.grid.nr - 1 or j <= 0 or j >= self.grid.nz - 1:
            return 0.0

        r = self.grid.r_at(i)
        dr = self.grid.dr
        dz = self.grid.dz

        d2r = (self.psi[i + 1][j] - 2.0 * self.psi[i][j] + self.psi[i - 1][j]) / (dr * dr)
        d1r = (self.psi[i + 1][j] - self.psi[i - 1][j]) / (2.0 * dr)
        d2z = (self.psi[i][j + 1] - 2.0 * self.psi[i][j] + self.psi[i][j - 1]) / (dz * dz)

        return d2r - (1.0 / r) * d1r + d2z

    def solve_step(self, omega: float = 1.4) -> float:
        """
        Perform one Successive Over-Relaxation (SOR) relaxation sweep.

        Returns the maximum flux change (L_infinity residual) for convergence checking.
        """
        dr = self.grid.dr
        dz = self.grid.dz
        dr2 = dr * dr
        dz2 = dz * dz
        denom_center = 2.0 * (1.0 / dr2 + 1.0 / dz2)

        delta_psi = self.psi_edge - self.psi_axis
        max_residual = 0.0

        for i in range(1, self.grid.nr - 1):
            r = self.grid.r_at(i)
            c_r_plus = 1.0 / dr2 - 1.0 / (2.0 * r * dr)
            c_r_minus = 1.0 / dr2 + 1.0 / (2.0 * r * dr)
            c_z = 1.0 / dz2

            for j in range(1, self.grid.nz - 1):
                psi_curr = self.psi[i][j]
                psi_n = self.normalized_flux(psi_curr)

                # Grad-Shafranov RHS: -mu_0 * R^2 * dp/dpsi - F * dF/dpsi
                dp = self.profile.dp_dpsi(psi_n, delta_psi)
                f_df = self.profile.f_df_dpsi(psi_n, delta_psi)
                rhs = -MU_0 * (r * r) * dp - f_df

                # Toroidal current density J_phi = -(1 / (mu_0 * R)) * Delta* psi
                # In equilibrium Delta* psi = RHS, so J_phi = -RHS / (mu_0 * R)
                self.j_phi[i][j] = -rhs / (MU_0 * r) if abs(MU_0 * r) > 1e-12 else 0.0

                stencil_sum = (
                    c_r_plus * self.psi[i + 1][j]
                    + c_r_minus * self.psi[i - 1][j]
                    + c_z * (self.psi[i][j + 1] + self.psi[i][j - 1])
                    - rhs
                )
                psi_new = stencil_sum / denom_center
                psi_relaxed = (1.0 - omega) * psi_curr + omega * psi_new

                diff = abs(psi_relaxed - psi_curr)
                if diff > max_residual:
                    max_residual = diff

                self.psi[i][j] = psi_relaxed

        self.update_magnetic_axis()
        return max_residual

    def solve(
        self,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        omega: float = 1.4,
    ) -> int:
        """
        Iterate SOR solver until convergence or max iterations reached.

        Returns the total number of iterations executed.
        """
        for it in range(1, max_iterations + 1):
            residual = self.solve_step(omega=omega)
            if residual < tolerance:
                return it
        return max_iterations

    def interpolate_psi(self, r: float, z: float) -> float:
        """Bilinear interpolation of poloidal flux psi at arbitrary coordinates (r, z)."""
        if not self.grid.contains(r, z):
            return self.psi_edge

        i_float = self.grid.index_of_r(r)
        j_float = self.grid.index_of_z(z)

        i0 = int(math.floor(i_float))
        j0 = int(math.floor(j_float))
        i1 = min(self.grid.nr - 1, i0 + 1)
        j1 = min(self.grid.nz - 1, j0 + 1)

        wr = i_float - i0
        wz = j_float - j0

        p00 = self.psi[i0][j0]
        p10 = self.psi[i1][j0]
        p01 = self.psi[i0][j1]
        p11 = self.psi[i1][j1]

        val = (
            (1.0 - wr) * (1.0 - wz) * p00
            + wr * (1.0 - wz) * p10
            + (1.0 - wr) * wz * p01
            + wr * wz * p11
        )
        return val

    def compute_plasma_current(self) -> float:
        """Integrate total toroidal plasma current I_p = double_integral J_phi dR dZ."""
        total_ip = 0.0
        area_cell = self.grid.dr * self.grid.dz
        for i in range(1, self.grid.nr - 1):
            for j in range(1, self.grid.nz - 1):
                total_ip += self.j_phi[i][j] * area_cell
        return total_ip

    def compute_stored_thermal_energy(self) -> float:
        """Integrate total plasma thermal energy W_th = (3/2) * integral p * 2*pi*R dR dZ."""
        energy = 0.0
        area_cell = self.grid.dr * self.grid.dz
        for i in range(1, self.grid.nr - 1):
            r = self.grid.r_at(i)
            two_pi_r = 2.0 * math.pi * r
            for j in range(1, self.grid.nz - 1):
                psi_val = self.psi[i][j]
                psi_n = self.normalized_flux(psi_val)
                p = self.profile.pressure(psi_n)
                energy += 1.5 * p * two_pi_r * area_cell
        return energy
