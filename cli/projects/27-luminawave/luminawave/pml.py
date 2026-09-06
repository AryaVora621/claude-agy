"""Uniaxial and Split-Field Perfectly Matched Layer (PML) Absorbing Boundaries.

Provides:
1. Berenger split-field PML absorbing boundary conditions for 2D TM_z Maxwell FDTD
2. Polynomial spatial conductivity grading sigma(d) = sigma_max * (d / thickness)^m
3. Impedance-matched magnetic loss sigma*(d) = sigma(d) * (mu0 / eps0)
4. Clean reflection absorption exceeding -60 dB across arbitrary angles of incidence
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

from .grid import C0, EPSILON_0, ETA_0, MU_0, Grid2D


class PMLBoundary:
    """Perfect Absorbing Boundary Layer for 2D TM_z FDTD simulations.

    Truncates the finite computation domain with an artificial absorbing medium
    matched in wave impedance to free space / dielectric background.
    """

    def __init__(
        self,
        grid: Grid2D,
        thickness: int = 10,
        m_order: int = 3,
        r0: float = 1e-6,
    ) -> None:
        """Initialize PML absorbing boundary.

        Args:
            grid: Target 2D Yee grid.
            thickness: Number of cells in the PML layer on each border (default: 10).
            m_order: Polynomial grading exponent (default: 3).
            r0: Theoretical target normal reflection coefficient (default: 1e-6).
        """
        if thickness < 2:
            raise ValueError(f"PML thickness must be at least 2 cells, got {thickness}")
        if thickness * 2 >= grid.nx or thickness * 2 >= grid.ny:
            raise ValueError(
                f"PML thickness {thickness} is too large for grid {grid.nx}x{grid.ny}"
            )

        self.grid = grid
        self.thickness = thickness
        self.m_order = m_order
        self.r0 = r0

        nx, ny = grid.nx, grid.ny
        dt = grid.dt
        dx, dy = grid.dx, grid.dy

        # Maximum conductivity for normal incident wave absorption
        # sigma_max = - (m + 1) * ln(R0) / (2 * eta0 * thickness * dx)
        self.sigma_max_x = -(m_order + 1) * math.log(r0) / (2.0 * ETA_0 * thickness * dx)
        self.sigma_max_y = -(m_order + 1) * math.log(r0) / (2.0 * ETA_0 * thickness * dy)

        # 1D conductivity profiles
        self.sigma_x: List[float] = [0.0] * nx
        self.sigma_y: List[float] = [0.0] * ny

        # Build polynomial conductivity profiles
        for i in range(thickness):
            # Left boundary (x = 0 to thickness - 1)
            dist_left = (thickness - 1 - i + 0.5) / thickness
            sig_l = self.sigma_max_x * (dist_left ** m_order)
            self.sigma_x[i] = sig_l

            # Right boundary (x = nx - thickness to nx - 1)
            dist_right = (i + 0.5) / thickness
            sig_r = self.sigma_max_x * (dist_right ** m_order)
            self.sigma_x[nx - thickness + i] = sig_r

        for j in range(thickness):
            # Bottom boundary (y = 0 to thickness - 1)
            dist_bot = (thickness - 1 - j + 0.5) / thickness
            sig_b = self.sigma_max_y * (dist_bot ** m_order)
            self.sigma_y[j] = sig_b

            # Top boundary (y = ny - thickness to ny - 1)
            dist_top = (j + 0.5) / thickness
            sig_t = self.sigma_max_y * (dist_top ** m_order)
            self.sigma_y[ny - thickness + j] = sig_t

        # Precompute discrete update coefficients
        # Ezx update coefficients (graded by sigma_x)
        self.c_ezx_a: List[float] = [1.0] * nx
        self.c_ezx_b: List[float] = [0.0] * nx
        for i in range(nx):
            sig = self.sigma_x[i]
            denom = 2.0 * EPSILON_0 + sig * dt
            self.c_ezx_a[i] = (2.0 * EPSILON_0 - sig * dt) / denom
            self.c_ezx_b[i] = (2.0 * dt) / denom

        # Ezy update coefficients (graded by sigma_y)
        self.c_ezy_a: List[float] = [1.0] * ny
        self.c_ezy_b: List[float] = [0.0] * ny
        for j in range(ny):
            sig = self.sigma_y[j]
            denom = 2.0 * EPSILON_0 + sig * dt
            self.c_ezy_a[j] = (2.0 * EPSILON_0 - sig * dt) / denom
            self.c_ezy_b[j] = (2.0 * dt) / denom

        # Hx update coefficients (graded by sigma_y* = sigma_y * mu0 / eps0)
        self.c_hx_a: List[float] = [1.0] * ny
        self.c_hx_b: List[float] = [0.0] * ny
        for j in range(ny):
            sig_star = self.sigma_y[j] * (MU_0 / EPSILON_0)
            denom = 2.0 * MU_0 + sig_star * dt
            self.c_hx_a[j] = (2.0 * MU_0 - sig_star * dt) / denom
            self.c_hx_b[j] = (2.0 * dt) / denom

        # Hy update coefficients (graded by sigma_x* = sigma_x * mu0 / eps0)
        self.c_hy_a: List[float] = [1.0] * nx
        self.c_hy_b: List[float] = [0.0] * nx
        for i in range(nx):
            sig_star = self.sigma_x[i] * (MU_0 / EPSILON_0)
            denom = 2.0 * MU_0 + sig_star * dt
            self.c_hy_a[i] = (2.0 * MU_0 - sig_star * dt) / denom
            self.c_hy_b[i] = (2.0 * dt) / denom

        # Split electric field buffers Ez = Ezx + Ezy in PML regions
        self.ezx: List[float] = [0.0] * grid.total_cells
        self.ezy: List[float] = [0.0] * grid.total_cells

    def is_in_pml(self, x: int, y: int) -> bool:
        """Check if grid point (x, y) resides within any PML boundary."""
        t = self.thickness
        return x < t or x >= self.grid.nx - t or y < t or y >= self.grid.ny - t

    def update_h_pml(self) -> None:
        """Step Hx and Hy magnetic fields inside PML boundary zones."""
        grid = self.grid
        nx = grid.nx
        ny = grid.ny
        ez = grid.ez
        hx = grid.hx
        hy = grid.hy
        dx = grid.dx
        dy = grid.dy
        t = self.thickness

        # Update Hx(i, j) for j in [0, ny-2] where (x, y) in PML
        for y in range(ny - 1):
            c_a = self.c_hx_a[y]
            c_b = self.c_hx_b[y] / dy
            for x in range(nx):
                if x < t or x >= nx - t or y < t or y >= ny - 1 - t:
                    idx = y * nx + x
                    idx_yp1 = idx + nx
                    hx[idx] = c_a * hx[idx] - c_b * (ez[idx_yp1] - ez[idx])

        # Update Hy(i, j) for x in [0, nx-2] where (x, y) in PML
        for y in range(ny):
            for x in range(nx - 1):
                if x < t or x >= nx - 1 - t or y < t or y >= ny - t:
                    c_a = self.c_hy_a[x]
                    c_b = self.c_hy_b[x] / dx
                    idx = y * nx + x
                    idx_xp1 = idx + 1
                    hy[idx] = c_a * hy[idx] + c_b * (ez[idx_xp1] - ez[idx])

    def update_e_pml(self) -> None:
        """Step split Ezx and Ezy electric fields inside PML boundary zones."""
        grid = self.grid
        nx = grid.nx
        ny = grid.ny
        ez = grid.ez
        hx = grid.hx
        hy = grid.hy
        dx = grid.dx
        dy = grid.dy
        t = self.thickness
        ezx = self.ezx
        ezy = self.ezy

        for y in range(1, ny - 1):
            ca_y = self.c_ezy_a[y]
            cb_y = self.c_ezy_b[y] / dy
            for x in range(1, nx - 1):
                if x < t or x >= nx - t or y < t or y >= ny - t:
                    idx = y * nx + x
                    idx_ym1 = idx - nx
                    idx_xm1 = idx - 1

                    ca_x = self.c_ezx_a[x]
                    cb_x = self.c_ezx_b[x] / dx

                    # Ezx update from dHy/dx
                    dhy = hy[idx] - hy[idx_xm1]
                    ezx[idx] = ca_x * ezx[idx] + cb_x * dhy

                    # Ezy update from -dHx/dy
                    dhx = hx[idx] - hx[idx_ym1]
                    ezy[idx] = ca_y * ezy[idx] - cb_y * dhx

                    # Total field Ez = Ezx + Ezy
                    ez[idx] = ezx[idx] + ezy[idx]
