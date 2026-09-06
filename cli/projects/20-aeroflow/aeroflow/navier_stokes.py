"""
AeroFlow: Eulerian Incompressible Navier-Stokes Solver.
Implements Chorin's fractional step projection method:
  1. Semi-Lagrangian characteristic backtracing advection (unconditionally stable).
  2. Implicit viscous diffusion via Jacobi relaxation.
  3. Pressure Poisson equation via Red-Black Gauss-Seidel with SOR.
  4. Divergence-free velocity projection enforcing div(u) = 0.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

from aeroflow.types import (
    Vector2D,
    Grid2D,
    VectorField2D,
    ObstacleMask,
    FluidParams,
)


class NavierStokesSolver:
    """
    Incompressible Navier-Stokes flow solver on 2D Cartesian grid.
    """
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float = 1.0,
        dt: float = 0.1,
        viscosity: float = 0.001,
        density: float = 1.0,
        inflow_velocity: float = 1.0,
        obstacle: Optional[ObstacleMask] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.size = nx * ny
        self.dx = dx
        self.dt = dt
        self.viscosity = viscosity
        self.density = density
        self.inflow_velocity = inflow_velocity

        if obstacle is not None:
            self.obstacle = obstacle
        else:
            self.obstacle = ObstacleMask(nx, ny)

        # Primary fields
        self.velocity = VectorField2D(nx, ny, dx, u_init=inflow_velocity, v_init=0.0)
        self.pressure = Grid2D(nx, ny, dx, initial_value=0.0)

        # Intermediate scratch buffers
        self.u_star = Grid2D(nx, ny, dx, initial_value=inflow_velocity)
        self.v_star = Grid2D(nx, ny, dx, initial_value=0.0)
        self.divergence = Grid2D(nx, ny, dx, initial_value=0.0)

        # Simulation step count & physical time
        self.step_count: int = 0
        self.time: float = 0.0

        # Apply initial boundary conditions
        self.apply_boundary_conditions(self.velocity.u, self.velocity.v)

    def advect(self) -> None:
        """
        Semi-Lagrangian advection step for momentum:
          u^* = u(x - u*dt)
          v^* = v(x - u*dt)
        Unconditionally stable backward characteristic tracing with bilinear interpolation.
        """
        nx, ny = self.nx, self.ny
        dt = self.dt
        u = self.velocity.u
        v = self.velocity.v
        u_star = self.u_star.data
        v_star = self.v_star.data
        is_solid = self.obstacle.mask

        for y in range(ny):
            y_offset = y * nx
            for x in range(nx):
                idx = y_offset + x
                if is_solid[idx]:
                    u_star[idx] = 0.0
                    v_star[idx] = 0.0
                    continue

                vx = u.get(x, y)
                vy = v.get(x, y)

                # Backtrace position along velocity characteristic
                prev_x = x - vx * dt
                prev_y = y - vy * dt

                # Sample previous velocities
                u_star[idx] = u.sample_bilinear(prev_x, prev_y)
                v_star[idx] = v.sample_bilinear(prev_x, prev_y)

    def diffuse(self, iterations: int = 15) -> None:
        """
        Implicit viscous diffusion step:
          (I - nu * dt * Laplacian) u^{**} = u^*
        Solved via iterative Jacobi relaxation.
        """
        if self.viscosity <= 0.0:
            return  # Inviscid Euler flow

        nx, ny = self.nx, self.ny
        alpha = (self.viscosity * self.dt) / (self.dx * self.dx)
        denom = 1.0 / (1.0 + 4.0 * alpha)
        is_solid = self.obstacle.mask

        u_temp = list(self.u_star.data)
        v_temp = list(self.v_star.data)
        u_src = self.u_star.data
        v_src = self.v_star.data

        for _ in range(iterations):
            for y in range(ny):
                y_offset = y * nx
                y_up = max(0, y - 1) * nx
                y_down = min(ny - 1, y + 1) * nx

                for x in range(nx):
                    idx = y_offset + x
                    if is_solid[idx]:
                        u_temp[idx] = 0.0
                        v_temp[idx] = 0.0
                        continue

                    x_left = max(0, x - 1)
                    x_right = min(nx - 1, x + 1)

                    sum_u = (u_temp[y_offset + x_left] + u_temp[y_offset + x_right] +
                             u_temp[y_up + x] + u_temp[y_down + x])
                    sum_v = (v_temp[y_offset + x_left] + v_temp[y_offset + x_right] +
                             v_temp[y_up + x] + v_temp[y_down + x])

                    u_temp[idx] = (u_src[idx] + alpha * sum_u) * denom
                    v_temp[idx] = (v_src[idx] + alpha * sum_v) * denom

        self.u_star.data = u_temp
        self.v_star.data = v_temp

    def compute_divergence(self) -> None:
        """
        Compute divergence of intermediate velocity field:
          div = (du*/dx + dv*/dy)
        """
        nx, ny = self.nx, self.ny
        inv_2dx = 0.5 / self.dx
        u_data = self.u_star.data
        v_data = self.v_star.data
        div_data = self.divergence.data
        is_solid = self.obstacle.mask

        for y in range(ny):
            y_offset = y * nx
            y_prev = max(0, y - 1) * nx
            y_next = min(ny - 1, y + 1) * nx

            for x in range(nx):
                idx = y_offset + x
                if is_solid[idx]:
                    div_data[idx] = 0.0
                    continue

                x_prev = max(0, x - 1)
                x_next = min(nx - 1, x + 1)

                du_dx = (u_data[y_offset + x_next] - u_data[y_offset + x_prev]) * inv_2dx
                dv_dy = (v_data[y_next + x] - v_data[y_prev + x]) * inv_2dx

                div_data[idx] = du_dx + dv_dy

    def solve_pressure_poisson(self, max_iterations: int = 40, sor_omega: float = 1.6) -> float:
        """
        Solve pressure Poisson equation:
          Laplacian(p) = (rho / dt) * div(u*)
        Uses Red-Black Gauss-Seidel iteration with Successive Over-Relaxation (SOR).
        Returns final maximum residual norm.
        """
        nx, ny = self.nx, self.ny
        dx_sq = self.dx * self.dx
        rhs_scale = (self.density / self.dt) * dx_sq
        p = self.pressure.data
        div = self.divergence.data
        is_solid = self.obstacle.mask

        max_residual = 0.0

        for _ in range(max_iterations):
            max_residual = 0.0

            # Pass 0: Red cells (x + y is even)
            # Pass 1: Black cells (x + y is odd)
            for color in (0, 1):
                for y in range(ny):
                    y_offset = y * nx
                    y_prev = max(0, y - 1) * nx
                    y_next = min(ny - 1, y + 1) * nx

                    for x in range(nx):
                        if (x + y) % 2 != color:
                            continue

                        idx = y_offset + x
                        if is_solid[idx]:
                            p[idx] = 0.0
                            continue

                        x_prev = max(0, x - 1)
                        x_next = min(nx - 1, x + 1)

                        # Neighbor pressure values
                        p_left = p[y_offset + x_prev]
                        p_right = p[y_offset + x_next]
                        p_bot = p[y_prev + x]
                        p_top = p[y_next + x]

                        b = rhs_scale * div[idx]
                        p_new = 0.25 * (p_left + p_right + p_bot + p_top - b)

                        # SOR update
                        diff = p_new - p[idx]
                        p[idx] += sor_omega * diff

                        abs_diff = abs(diff)
                        if abs_diff > max_residual:
                            max_residual = abs_diff

        return max_residual

    def project_velocity(self) -> None:
        """
        Project velocity to divergence-free field:
          u^{n+1} = u^* - (dt / rho) * grad(p)
        Enforces exact incompressibility div(u) = 0.
        """
        nx, ny = self.nx, self.ny
        grad_scale = (self.dt / (2.0 * self.density * self.dx))
        u_data = self.velocity.u.data
        v_data = self.velocity.v.data
        u_star = self.u_star.data
        v_star = self.v_star.data
        p = self.pressure.data
        is_solid = self.obstacle.mask

        for y in range(ny):
            y_offset = y * nx
            y_prev = max(0, y - 1) * nx
            y_next = min(ny - 1, y + 1) * nx

            for x in range(nx):
                idx = y_offset + x
                if is_solid[idx]:
                    u_data[idx] = 0.0
                    v_data[idx] = 0.0
                    continue

                x_prev = max(0, x - 1)
                x_next = min(nx - 1, x + 1)

                dp_dx = p[y_offset + x_next] - p[y_offset + x_prev]
                dp_dy = p[y_next + x] - p[y_prev + x]

                u_data[idx] = u_star[idx] - grad_scale * dp_dx
                v_data[idx] = v_star[idx] - grad_scale * dp_dy

    def apply_boundary_conditions(self, u_grid: Grid2D, v_grid: Grid2D) -> None:
        """
        Apply inflow at x=0, outflow at x=nx-1, and no-slip walls at y=0, y=ny-1.
        """
        nx, ny = self.nx, self.ny
        u = u_grid.data
        v = v_grid.data
        u_in = self.inflow_velocity

        # Inflow at x = 0
        for y in range(ny):
            idx = y * nx
            u[idx] = u_in
            v[idx] = 0.0

        # Outflow at x = nx - 1: Neumann zero-gradient
        for y in range(ny):
            idx_out = y * nx + (nx - 1)
            idx_prev = y * nx + (nx - 2)
            u[idx_out] = u[idx_prev]
            v[idx_out] = v[idx_prev]

        # Top and bottom no-slip walls
        for x in range(nx):
            u[x] = 0.0
            v[x] = 0.0
            u[(ny - 1) * nx + x] = 0.0
            v[(ny - 1) * nx + x] = 0.0

    def step(self) -> None:
        """Execute one complete Navier-Stokes fractional step integration."""
        self.advect()
        self.diffuse(iterations=10)
        self.apply_boundary_conditions(self.u_star, self.v_star)
        self.compute_divergence()
        self.solve_pressure_poisson(max_iterations=30, sor_omega=1.6)
        self.project_velocity()
        self.apply_boundary_conditions(self.velocity.u, self.velocity.v)

        self.step_count += 1
        self.time += self.dt

    def get_max_divergence(self, interior_only: bool = True) -> float:
        """
        Compute maximum absolute divergence across fluid cells.
        When interior_only is True, evaluates interior cells away from boundary clamps.
        """
        div = self.velocity.compute_divergence()
        is_solid = self.obstacle.mask
        max_div = 0.0
        nx, ny = self.nx, self.ny
        y_start = 2 if interior_only else 0
        y_end = ny - 2 if interior_only else ny
        x_start = 2 if interior_only else 0
        x_end = nx - 2 if interior_only else nx

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                idx = y * nx + x
                if not is_solid[idx]:
                    val = abs(div.data[idx])
                    if val > max_div:
                        max_div = val
        return max_div

    def get_vorticity(self) -> Grid2D:
        """Compute vorticity curl(u) = dv/dx - du/dy."""
        return self.velocity.compute_vorticity()
