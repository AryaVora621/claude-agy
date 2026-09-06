"""Maxwell 2D Finite-Difference Time-Domain (FDTD) Simulation Engine.

Provides:
1. Leapfrog time-stepping of Maxwell's curl equations in 2D TM_z polarization
2. Seamless integration with UPML absorbing boundaries
3. Multiple simultaneous optical excitation sources
4. On-the-fly energy and stability tracking
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple

from .grid import C0, EPSILON_0, MU_0, Grid2D
from .pml import PMLBoundary
from .sources import PointSource, WaveguideModeSource


class FDTDSimulator:
    """Core 2D FDTD Electrodynamics Solver."""

    def __init__(
        self,
        grid: Grid2D,
        pml_thickness: int = 10,
        enable_pml: bool = True,
    ) -> None:
        """Initialize FDTD simulation engine.

        Args:
            grid: Underlying 2D Yee grid with material assignments.
            pml_thickness: Number of boundary cells reserved for PML absorption.
            enable_pml: If True, applies Berenger split-field PML absorbing boundaries.
        """
        self.grid = grid
        self.enable_pml = enable_pml
        self.pml: Optional[PMLBoundary] = None
        if enable_pml:
            self.pml = PMLBoundary(grid, thickness=pml_thickness)

        self.sources: List[Any] = []
        self.monitors: List[Any] = []

        self.step_count: int = 0
        self.time: float = 0.0  # Physical time in seconds

        # Interior update bounds (excluding PML if enabled)
        t = pml_thickness if enable_pml else 0
        self.interior_x_min = max(1, t)
        self.interior_x_max = min(grid.nx - 1, grid.nx - t)
        self.interior_y_min = max(1, t)
        self.interior_y_max = min(grid.ny - 1, grid.ny - t)

    def add_source(self, source: Any) -> None:
        """Register an optical excitation source with the simulation."""
        self.sources.append(source)

    def add_monitor(self, monitor: Any) -> None:
        """Register a field or DFT flux monitor with the simulation."""
        self.monitors.append(monitor)

    def step(self) -> None:
        """Execute one complete leapfrog time step (Delta t)."""
        grid = self.grid
        nx = grid.nx
        ny = grid.ny
        ez = grid.ez
        hx = grid.hx
        hy = grid.hy
        dx = grid.dx
        dy = grid.dy
        dt = grid.dt
        dbx = grid.dbx
        dby = grid.dby

        # ------------------------------------------------------------------
        # 1. Update Hx and Hy magnetic fields (Half-step n -> n + 1/2)
        # ------------------------------------------------------------------
        # Interior Hx: Hx(i, j) -= dt / (mu * dy) * (Ez(i, j+1) - Ez(i, j))
        y_max_h = ny - 1 if not self.enable_pml else self.interior_y_max
        y_min_h = 0 if not self.enable_pml else self.interior_y_min
        x_min_h = 0 if not self.enable_pml else self.interior_x_min
        x_max_h = nx if not self.enable_pml else self.interior_x_max

        for y in range(y_min_h, y_max_h):
            base = y * nx
            base_yp1 = base + nx
            for x in range(x_min_h, x_max_h):
                idx = base + x
                hx[idx] -= dbx * (ez[base_yp1 + x] - ez[idx])

        # Interior Hy: Hy(i, j) += dt / (mu * dx) * (Ez(i+1, j) - Ez(i, j))
        x_max_hy = nx - 1 if not self.enable_pml else self.interior_x_max
        for y in range(y_min_h, y_max_h + (0 if self.enable_pml else 1)):
            base = y * nx
            for x in range(x_min_h, x_max_hy):
                idx = base + x
                hy[idx] += dby * (ez[idx + 1] - ez[idx])

        # Update H in PML regions
        if self.pml is not None:
            self.pml.update_h_pml()

        # Advance time by half step
        self.time += 0.5 * dt

        # ------------------------------------------------------------------
        # 2. Update Ez electric fields (Half-step n + 1/2 -> n + 1)
        # ------------------------------------------------------------------
        ca = grid.ca
        cb = grid.cb
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        x_min_e = self.interior_x_min
        x_max_e = self.interior_x_max
        y_min_e = self.interior_y_min
        y_max_e = self.interior_y_max

        for y in range(y_min_e, y_max_e):
            base = y * nx
            base_ym1 = base - nx
            for x in range(x_min_e, x_max_e):
                idx = base + x
                # Curl of H: dHy/dx - dHx/dy
                dhy = (hy[idx] - hy[idx - 1]) * inv_dx
                dhx = (hx[idx] - hx[base_ym1 + x]) * inv_dy
                ez[idx] = ca[idx] * ez[idx] + cb[idx] * (dhy - dhx)

        # Update Ez in PML regions
        if self.pml is not None:
            self.pml.update_e_pml()

        # Advance time by remaining half step
        self.time += 0.5 * dt
        self.step_count += 1

        # ------------------------------------------------------------------
        # 3. Inject Sources at time t
        # ------------------------------------------------------------------
        for src in self.sources:
            src.inject(grid, self.time)

        # ------------------------------------------------------------------
        # 4. Sample Monitors at time t
        # ------------------------------------------------------------------
        for mon in self.monitors:
            mon.sample(grid, self.step_count, self.time)

    def run(self, steps: int) -> None:
        """Run simulation for a fixed number of time steps."""
        for _ in range(steps):
            self.step()

    def get_max_fields(self) -> Tuple[float, float]:
        """Compute maximum absolute Ez and H field values across grid."""
        max_ez = 0.0
        max_h2 = 0.0
        ez = self.grid.ez
        hx = self.grid.hx
        hy = self.grid.hy
        for i in range(self.grid.total_cells):
            a_ez = abs(ez[i])
            if a_ez > max_ez:
                max_ez = a_ez
            h2 = hx[i] * hx[i] + hy[i] * hy[i]
            if h2 > max_h2:
                max_h2 = h2
        return max_ez, math.sqrt(max_h2)

    def is_stable(self, threshold: float = 1e6) -> bool:
        """Verify whether electromagnetic fields remain numerically bounded."""
        max_ez, max_h = self.get_max_fields()
        return not (math.isnan(max_ez) or math.isinf(max_ez) or max_ez > threshold)
