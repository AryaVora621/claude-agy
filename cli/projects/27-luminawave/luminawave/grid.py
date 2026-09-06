"""2D Yee Staggered Grid, Material Distributions, and Courant Stability.

Provides:
1. Physical constants in SI and normalized units
2. 2D Yee lattice spatial discretization for TM_z electromagnetic fields (Ez, Hx, Hy)
3. Spatially varying dielectric permittivity (eps_r), permeability (mu_r), and conductivity (sigma)
4. Courant-Friedrichs-Lewy (CFL) numerical stability verification
5. High-performance flat buffer storage and update coefficient computation
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# Fundamental Physical Constants (CODATA 2018 / SI Units)
C0: float = 299792458.0            # Speed of light in vacuum (m/s)
EPSILON_0: float = 8.8541878128e-12 # Vacuum permittivity (F/m)
MU_0: float = 1.25663706212e-6     # Vacuum permeability (H/m)
ETA_0: float = math.sqrt(MU_0 / EPSILON_0) # Free-space wave impedance (~376.73 Ohms)


@dataclass
class GridDimensions:
    """Dimensions and spatial resolution of a 2D Yee simulation domain."""
    nx: int
    ny: int
    dx: float
    dy: float

    @property
    def width(self) -> float:
        """Physical domain width in meters."""
        return self.nx * self.dx

    @property
    def height(self) -> float:
        """Physical domain height in meters."""
        return self.ny * self.dy


class Grid2D:
    """2D Yee staggered finite-difference lattice for TM_z electromagnetic fields.

    In TM_z polarization:
    - Ez is defined at integer nodes (i, j)
    - Hx is defined at (i, j + 1/2)
    - Hy is defined at (i + 1/2, j)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 50e-9,
        dy: float = 50e-9,
        courant_factor: float = 0.70,
    ) -> None:
        """Initialize 2D Yee grid.

        Args:
            nx: Number of spatial cells along x.
            ny: Number of spatial cells along y.
            dx: Spatial step size along x in meters (default: 50 nm).
            dy: Spatial step size along y in meters (default: 50 nm).
            courant_factor: Courant CFL stability factor S <= 1.0 (default: 0.70).
        """
        if nx < 4 or ny < 4:
            raise ValueError(f"Grid dimensions must be at least 4x4, got {nx}x{ny}")
        if dx <= 0.0 or dy <= 0.0:
            raise ValueError(f"Spatial steps must be positive, got dx={dx}, dy={dy}")
        if courant_factor <= 0.0 or courant_factor > 1.0:
            raise ValueError(f"Courant factor must be in (0, 1.0], got {courant_factor}")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.courant_factor = courant_factor
        self.dims = GridDimensions(nx=nx, ny=ny, dx=dx, dy=dy)

        # 2D CFL stability condition: dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
        max_dt = 1.0 / (C0 * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy)))
        self.dt = courant_factor * max_dt

        self.total_cells = nx * ny

        # Primary electromagnetic field components (flat 1D arrays for cache locality)
        self.ez: List[float] = [0.0] * self.total_cells
        self.hx: List[float] = [0.0] * self.total_cells
        self.hy: List[float] = [0.0] * self.total_cells

        # Material properties
        self.eps_r: List[float] = [1.0] * self.total_cells  # Relative permittivity (n^2)
        self.mu_r: List[float] = [1.0] * self.total_cells   # Relative permeability
        self.sigma: List[float] = [0.0] * self.total_cells  # Electric conductivity (S/m)

        # Precomputed FDTD leapfrog update coefficients
        self.ca: List[float] = [1.0] * self.total_cells
        self.cb: List[float] = [0.0] * self.total_cells
        self.dbx: float = self.dt / (MU_0 * dy)
        self.dby: float = self.dt / (MU_0 * dx)

        self.recompute_coefficients()

    def idx(self, x: int, y: int) -> int:
        """Convert 2D (x, y) coordinates to flat 1D buffer index."""
        return y * self.nx + x

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if (x, y) is within grid boundaries."""
        return 0 <= x < self.nx and 0 <= y < self.ny

    def recompute_coefficients(self) -> None:
        """Recompute Maxwell update coefficients after modifying material parameters."""
        dt = self.dt
        eps0 = EPSILON_0
        for i in range(self.total_cells):
            eps = eps0 * self.eps_r[i]
            sig = self.sigma[i]
            denom = 2.0 * eps + sig * dt
            if denom > 0.0:
                self.ca[i] = (2.0 * eps - sig * dt) / denom
                self.cb[i] = (2.0 * dt) / denom
            else:
                self.ca[i] = 1.0
                self.cb[i] = dt / eps0

    def set_material_point(
        self,
        x: int,
        y: int,
        eps_r: float = 1.0,
        mu_r: float = 1.0,
        sigma: float = 0.0,
    ) -> None:
        """Set material parameters at a specific grid cell."""
        if not self.in_bounds(x, y):
            return
        idx = self.idx(x, y)
        eps = max(1.0, eps_r)
        self.eps_r[idx] = eps
        self.mu_r[idx] = max(1.0, mu_r)
        self.sigma[idx] = max(0.0, sigma)

        # Update local update coefficients
        eps_val = EPSILON_0 * eps
        dt = self.dt
        sig_val = self.sigma[idx]
        denom = 2.0 * eps_val + sig_val * dt
        if denom > 0.0:
            self.ca[idx] = (2.0 * eps_val - sig_val * dt) / denom
            self.cb[idx] = (2.0 * dt) / denom
        else:
            self.ca[idx] = 1.0
            self.cb[idx] = dt / (EPSILON_0 * eps)

    def set_refractive_index(self, x: int, y: int, n: float) -> None:
        """Set refractive index n at a cell (eps_r = n^2 for non-magnetic media)."""
        self.set_material_point(x, y, eps_r=n * n)

    def get_refractive_index(self, x: int, y: int) -> float:
        """Retrieve refractive index at a cell."""
        if not self.in_bounds(x, y):
            return 1.0
        return math.sqrt(self.eps_r[self.idx(x, y)])

    def add_rectangle(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        eps_r: float = 1.0,
        sigma: float = 0.0,
    ) -> None:
        """Add a rectangular dielectric slab to the simulation domain."""
        xmin = max(0, min(x1, x2))
        xmax = min(self.nx - 1, max(x1, x2))
        ymin = max(0, min(y1, y2))
        ymax = min(self.ny - 1, max(y1, y2))

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                idx = self.idx(x, y)
                self.eps_r[idx] = eps_r
                self.sigma[idx] = sigma

        self.recompute_coefficients()

    def add_circle(
        self,
        cx: int,
        cy: int,
        radius: int,
        eps_r: float = 1.0,
        sigma: float = 0.0,
    ) -> None:
        """Add a circular dielectric cylinder or rod to the simulation domain."""
        r2 = radius * radius
        xmin = max(0, cx - radius)
        xmax = min(self.nx - 1, cx + radius)
        ymin = max(0, cy - radius)
        ymax = min(self.ny - 1, cy + radius)

        for y in range(ymin, ymax + 1):
            dy = y - cy
            for x in range(xmin, xmax + 1):
                dx = x - cx
                if dx * dx + dy * dy <= r2:
                    idx = self.idx(x, y)
                    self.eps_r[idx] = eps_r
                    self.sigma[idx] = sigma

        self.recompute_coefficients()

    def add_ring(
        self,
        cx: int,
        cy: int,
        r_inner: int,
        r_outer: int,
        eps_r: float = 1.0,
        sigma: float = 0.0,
    ) -> None:
        """Add an annular dielectric micro-ring resonator to the simulation domain."""
        r_in2 = r_inner * r_inner
        r_out2 = r_outer * r_outer
        xmin = max(0, cx - r_outer)
        xmax = min(self.nx - 1, cx + r_outer)
        ymin = max(0, cy - r_outer)
        ymax = min(self.ny - 1, cy + r_outer)

        for y in range(ymin, ymax + 1):
            dy = y - cy
            for x in range(xmin, xmax + 1):
                dx = x - cx
                dist2 = dx * dx + dy * dy
                if r_in2 <= dist2 <= r_out2:
                    idx = self.idx(x, y)
                    self.eps_r[idx] = eps_r
                    self.sigma[idx] = sigma

        self.recompute_coefficients()

    def reset_fields(self) -> None:
        """Zero out all electromagnetic field components."""
        for i in range(self.total_cells):
            self.ez[i] = 0.0
            self.hx[i] = 0.0
            self.hy[i] = 0.0

    def total_energy(self) -> Tuple[float, float, float]:
        """Compute total stored electromagnetic energy (Joules/meter in 2D).

        Returns:
            Tuple of (electric_energy, magnetic_energy, total_energy).
        """
        cell_area = self.dx * self.dy
        u_e = 0.0
        u_h = 0.0

        for y in range(self.ny):
            for x in range(self.nx):
                idx = self.idx(x, y)
                # Electric energy: 1/2 * eps0 * eps_r * Ez^2
                u_e += 0.5 * EPSILON_0 * self.eps_r[idx] * (self.ez[idx] ** 2) * cell_area
                # Magnetic energy: 1/2 * mu0 * (Hx^2 + Hy^2)
                u_h += 0.5 * MU_0 * (self.hx[idx] ** 2 + self.hy[idx] ** 2) * cell_area

        return u_e, u_h, u_e + u_h
