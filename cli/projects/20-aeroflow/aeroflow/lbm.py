"""
AeroFlow: D2Q9 Lattice Boltzmann Method (LBM) Fluid Dynamics Solver.
Implements BGK collision operator, streaming, Zou-He boundary conditions,
half-way bounce-back on curved/arbitrary solid obstacles, and momentum exchange.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any

from aeroflow.types import (
    Vector2D,
    Grid2D,
    VectorField2D,
    ObstacleMask,
    FluidParams,
)


class LatticeD2Q9:
    """
    Standard 2D 9-velocity discrete lattice (D2Q9).
    Discrete directions:
      0: ( 0,  0)  [rest, weight 4/9]
      1: ( 1,  0)  [right, weight 1/9]
      2: ( 0,  1)  [up, weight 1/9]
      3: (-1,  0)  [left, weight 1/9]
      4: ( 0, -1)  [down, weight 1/9]
      5: ( 1,  1)  [up-right, weight 1/36]
      6: (-1,  1)  [up-left, weight 1/36]
      7: (-1, -1)  [down-left, weight 1/36]
      8: ( 1, -1)  [down-right, weight 1/36]
    """
    CX: Tuple[int, ...] = (0, 1, 0, -1, 0, 1, -1, -1, 1)
    CY: Tuple[int, ...] = (0, 0, 1, 0, -1, 1, 1, -1, -1)
    WEIGHTS: Tuple[float, ...] = (
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    )
    OPPOSITE: Tuple[int, ...] = (0, 3, 4, 1, 2, 7, 8, 5, 6)
    CS_SQ: float = 1.0 / 3.0
    INV_CS_SQ: float = 3.0
    INV_2CS_FOURTH: float = 4.5
    INV_2CS_SQ: float = 1.5


class LBMSolver:
    """
    Lattice Boltzmann D2Q9 flow solver with BGK single-relaxation-time collision.
    """
    def __init__(
        self,
        nx: int,
        ny: int,
        viscosity: float = 0.02,
        inflow_velocity: float = 0.08,
        obstacle: Optional[ObstacleMask] = None,
        top_bottom_no_slip: bool = True,
        left_right_no_slip: bool = False,
        periodic_x: bool = False,
    ):
        self.nx = nx
        self.ny = ny
        self.size = nx * ny
        self.viscosity = viscosity
        self.inflow_velocity = inflow_velocity
        self.top_bottom_no_slip = top_bottom_no_slip
        self.left_right_no_slip = left_right_no_slip
        self.periodic_x = periodic_x

        # Relaxation time tau = 3*nu + 0.5, omega = 1 / tau
        self.tau = 3.0 * viscosity + 0.5
        if self.tau <= 0.5:
            self.tau = 0.501  # Numerical stability floor
        self.omega = 1.0 / self.tau

        # Obstacle mask
        if obstacle is not None:
            self.obstacle = obstacle
        else:
            self.obstacle = ObstacleMask(nx, ny)

        # 9 discrete distribution functions f[i][idx]
        self.f: List[List[float]] = [[0.0] * self.size for _ in range(9)]
        self.f_star: List[List[float]] = [[0.0] * self.size for _ in range(9)]

        # Macroscopic quantities
        self.rho = Grid2D(nx, ny, 1.0, 1.0)
        self.velocity = VectorField2D(nx, ny, 1.0, u_init=inflow_velocity, v_init=0.0)

        # Force accumulation on obstacles (Momentum Exchange Method)
        self.drag_force: float = 0.0
        self.lift_force: float = 0.0

        # Step count
        self.time_step: int = 0

        # Initialize to equilibrium with inflow velocity
        self.reset_to_equilibrium(inflow_velocity, 0.0, 1.0)

    def reset_to_equilibrium(self, u_init: float, v_init: float, rho_init: float = 1.0) -> None:
        """Initialize all lattice nodes to Maxwell-Boltzmann equilibrium distribution."""
        w = LatticeD2Q9.WEIGHTS
        cx = LatticeD2Q9.CX
        cy = LatticeD2Q9.CY

        for idx in range(self.size):
            u_sq = u_init * u_init + v_init * v_init
            for i in range(9):
                c_u = cx[i] * u_init + cy[i] * v_init
                feq = w[i] * rho_init * (1.0 + 3.0 * c_u + 4.5 * c_u * c_u - 1.5 * u_sq)
                self.f[i][idx] = feq
                self.f_star[i][idx] = feq

        self.rho.fill(rho_init)
        self.velocity.u.fill(u_init)
        self.velocity.v.fill(v_init)
        self.time_step = 0
        self.drag_force = 0.0
        self.lift_force = 0.0

    def compute_macroscopic(self) -> None:
        """
        Recover macroscopic fluid density rho and velocity (u, v) from distributions:
          rho = sum(f_i)
          rho * u = sum(f_i * cx_i)
          rho * v = sum(f_i * cy_i)
        """
        f0, f1, f2, f3, f4, f5, f6, f7, f8 = self.f
        rho_data = self.rho.data
        u_data = self.velocity.u.data
        v_data = self.velocity.v.data
        is_solid = self.obstacle.mask

        for idx in range(self.size):
            if is_solid[idx]:
                rho_data[idx] = 1.0
                u_data[idx] = 0.0
                v_data[idx] = 0.0
                continue

            # Density
            r = (f0[idx] + f1[idx] + f2[idx] + f3[idx] + f4[idx] +
                 f5[idx] + f6[idx] + f7[idx] + f8[idx])
            rho_data[idx] = r

            inv_r = 1.0 / r if r > 1e-12 else 1.0
            # Momentum
            u = ((f1[idx] + f5[idx] + f8[idx]) - (f3[idx] + f6[idx] + f7[idx])) * inv_r
            v = ((f2[idx] + f5[idx] + f6[idx]) - (f4[idx] + f7[idx] + f8[idx])) * inv_r

            u_data[idx] = u
            v_data[idx] = v

    def collide(self) -> None:
        """
        BGK collision step:
          f_i^* = f_i - omega * (f_i - f_i^eq)
        Evaluated on all fluid nodes.
        """
        w = LatticeD2Q9.WEIGHTS
        cx = LatticeD2Q9.CX
        cy = LatticeD2Q9.CY
        omega = self.omega
        is_solid = self.obstacle.mask

        rho_data = self.rho.data
        u_data = self.velocity.u.data
        v_data = self.velocity.v.data

        for idx in range(self.size):
            if is_solid[idx]:
                continue

            r = rho_data[idx]
            u = u_data[idx]
            v = v_data[idx]
            u_sq = u * u + v * v

            for i in range(9):
                c_u = cx[i] * u + cy[i] * v
                feq = w[i] * r * (1.0 + 3.0 * c_u + 4.5 * c_u * c_u - 1.5 * u_sq)
                self.f_star[i][idx] = self.f[i][idx] - omega * (self.f[i][idx] - feq)

    def stream_and_bounce(self) -> None:
        """
        Stream distributions to neighboring cells:
          f_i(x + cx_i, y + cy_i) = f_i^*(x, y)
        Includes half-way bounce-back on obstacle boundary nodes and top/bottom walls,
        accumulating aerodynamic drag and lift forces via momentum exchange.
        """
        nx, ny = self.nx, self.ny
        cx = LatticeD2Q9.CX
        cy = LatticeD2Q9.CY
        opp = LatticeD2Q9.OPPOSITE
        is_solid = self.obstacle.mask

        f = self.f
        f_star = self.f_star

        total_fx = 0.0
        total_fy = 0.0

        # Stream interior nodes
        for y in range(ny):
            y_offset = y * nx
            for x in range(nx):
                curr_idx = y_offset + x
                if is_solid[curr_idx]:
                    continue

                for i in range(9):
                    next_x = x + cx[i]
                    next_y = y + cy[i]

                    # Wall bounce-back on top/bottom boundary
                    if self.top_bottom_no_slip and (next_y < 0 or next_y >= ny):
                        # Bounce back to same cell in opposite direction
                        f[opp[i]][curr_idx] = f_star[i][curr_idx]
                        continue

                    # Periodic in y if not top_bottom_no_slip
                    next_y = (next_y + ny) % ny

                    # Wall bounce-back on left/right boundary
                    if self.left_right_no_slip and (next_x < 0 or next_x >= nx):
                        f[opp[i]][curr_idx] = f_star[i][curr_idx]
                        continue

                    # Periodic in x if requested
                    if self.periodic_x:
                        next_x = (next_x + nx) % nx
                    elif next_x < 0 or next_x >= nx:
                        continue  # Handled in boundary conditions

                    next_idx = next_y * nx + next_x

                    # Solid obstacle bounce-back with momentum exchange
                    if is_solid[next_idx]:
                        # Momentum transferred to obstacle = e_i * (f_star_i + f_opp)
                        f_reflected = f_star[i][curr_idx]
                        f[opp[i]][curr_idx] = f_reflected

                        # Momentum exchange: Force = delta_p / delta_t
                        total_fx += cx[i] * (f_star[i][curr_idx] + f_reflected)
                        total_fy += cy[i] * (f_star[i][curr_idx] + f_reflected)
                    else:
                        # Standard streaming to fluid neighbor
                        f[i][next_idx] = f_star[i][curr_idx]

        self.drag_force = total_fx
        self.lift_force = total_fy

    def apply_boundary_conditions(self) -> None:
        """
        Apply Zou-He velocity inlet at x=0 and convective/open outlet at x=nx-1.
        Adds tiny transverse velocity perturbation to trigger natural vortex shedding.
        """
        nx, ny = self.nx, self.ny
        f = self.f
        w = LatticeD2Q9.WEIGHTS

        # Inflow at x = 0: Zou-He velocity boundary condition
        # u = inflow_velocity, v = slight perturbation
        u_in = self.inflow_velocity
        t = self.time_step

        for y in range(ny):
            idx = y * nx  # x = 0

            # Subtle asymmetric perturbation to break symmetry and induce vortex street
            v_perturb = 0.001 * math.sin(2.0 * math.pi * (y / ny) + t * 0.05)
            # Density at inlet from known incoming populations
            # rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - u_in)
            rho_in = (f[0][idx] + f[2][idx] + f[4][idx] +
                      2.0 * (f[3][idx] + f[6][idx] + f[7][idx])) / (1.0 - u_in)

            # Zou-He unknown distributions for right-facing directions (1, 5, 8)
            # f1 = f3 + (2/3) * rho * u
            f[1][idx] = f[3][idx] + (2.0 / 3.0) * rho_in * u_in
            # f5 = f7 - 0.5*(f2 - f4) + 0.5*rho*v + (1/6)*rho*u
            f[5][idx] = (f[7][idx] - 0.5 * (f[2][idx] - f[4][idx]) +
                         0.5 * rho_in * v_perturb + (1.0 / 6.0) * rho_in * u_in)
            # f8 = f6 + 0.5*(f2 - f4) - 0.5*rho*v + (1/6)*rho*u
            f[8][idx] = (f[6][idx] + 0.5 * (f[2][idx] - f[4][idx]) -
                         0.5 * rho_in * v_perturb + (1.0 / 6.0) * rho_in * u_in)

        # Outflow at x = nx - 1: Zero-gradient open convective extrapolation
        for y in range(ny):
            idx_out = y * nx + (nx - 1)
            idx_prev = y * nx + (nx - 2)
            for i in range(9):
                f[i][idx_out] = f[i][idx_prev]

    def step(self) -> None:
        """Advance simulation by one discrete Lattice Boltzmann time step."""
        self.collide()
        self.stream_and_bounce()
        self.apply_boundary_conditions()
        self.compute_macroscopic()
        self.time_step += 1

    def run_steps(self, num_steps: int) -> None:
        """Run multiple consecutive time steps."""
        for _ in range(num_steps):
            self.step()

    def get_aerodynamic_coefficients(
        self,
        characteristic_length: float = 10.0,
        ref_density: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Compute non-dimensional Drag (CD) and Lift (CL) coefficients:
          CD = 2 * F_drag / (rho * U_inf^2 * D)
          CL = 2 * F_lift / (rho * U_inf^2 * D)
        """
        u_inf = self.inflow_velocity
        denom = 0.5 * ref_density * (u_inf * u_inf) * characteristic_length
        if denom <= 1e-12:
            return 0.0, 0.0
        cd = self.drag_force / denom
        cl = self.lift_force / denom
        return cd, cl

    def get_vorticity(self) -> Grid2D:
        """Compute curl of velocity field: omega = dv/dx - du/dy."""
        return self.velocity.compute_vorticity()
