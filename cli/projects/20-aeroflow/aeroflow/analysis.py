"""
AeroFlow: Flow Field Analysis, Streamlines, Q-Criterion & Enstrophy.
Zero external dependencies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from aeroflow.types import Vector2D, Grid2D, VectorField2D, ObstacleMask


@dataclass
class FlowStatistics:
    """
    Comprehensive hydrodynamic flow metrics.
    """
    kinetic_energy: float
    enstrophy: float
    max_velocity: float
    max_vorticity: float
    mean_pressure: float
    reynolds_number: float


def compute_stream_function(
    vorticity: Grid2D,
    obstacle: ObstacleMask,
    max_iterations: int = 50,
) -> Grid2D:
    """
    Compute scalar stream function psi by solving Poisson equation:
      Laplacian(psi) = -vorticity
    Where u = d(psi)/dy and v = -d(psi)/dx.
    """
    nx, ny = vorticity.nx, vorticity.ny
    psi = Grid2D(nx, ny, dx=vorticity.dx, initial_value=0.0)
    psi_data = psi.data
    vort_data = vorticity.data
    is_solid = obstacle.mask
    dx_sq = vorticity.dx * vorticity.dx

    for _ in range(max_iterations):
        for y in range(1, ny - 1):
            y_offset = y * nx
            y_up = (y + 1) * nx
            y_down = (y - 1) * nx

            for x in range(1, nx - 1):
                idx = y_offset + x
                if is_solid[idx]:
                    psi_data[idx] = 0.0
                    continue

                # 5-point discrete Laplacian relaxation
                sum_neighbors = (psi_data[y_offset + (x + 1)] +
                                 psi_data[y_offset + (x - 1)] +
                                 psi_data[y_up + x] +
                                 psi_data[y_down + x])
                rhs = dx_sq * vort_data[idx]
                psi_data[idx] = 0.25 * (sum_neighbors + rhs)

    return psi


def trace_streamline(
    velocity: VectorField2D,
    obstacle: ObstacleMask,
    seed_x: float,
    seed_y: float,
    max_steps: int = 150,
    dt: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Trace a streamline path from a seed location forward using 4th-order Runge-Kutta (RK4).
    """
    points: List[Tuple[float, float]] = [(seed_x, seed_y)]
    curr_x = seed_x
    curr_y = seed_y
    nx, ny = velocity.nx, velocity.ny

    for _ in range(max_steps):
        # Stop if out of bounds or inside obstacle
        ix = int(curr_x)
        iy = int(curr_y)
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny or obstacle.is_solid(ix, iy):
            break

        # RK4 integration
        # k1
        v1 = velocity.sample_bilinear(curr_x, curr_y)
        if v1.norm_sq() < 1e-8:
            break

        # k2
        x2 = curr_x + 0.5 * dt * v1.x
        y2 = curr_y + 0.5 * dt * v1.y
        v2 = velocity.sample_bilinear(x2, y2)

        # k3
        x3 = curr_x + 0.5 * dt * v2.x
        y3 = curr_y + 0.5 * dt * v2.y
        v3 = velocity.sample_bilinear(x3, y3)

        # k4
        x4 = curr_x + dt * v3.x
        y4 = curr_y + dt * v3.y
        v4 = velocity.sample_bilinear(x4, y4)

        next_x = curr_x + (dt / 6.0) * (v1.x + 2.0 * v2.x + 2.0 * v3.x + v4.x)
        next_y = curr_y + (dt / 6.0) * (v1.y + 2.0 * v2.y + 2.0 * v3.y + v4.y)

        # Stop if no progression
        if abs(next_x - curr_x) < 1e-4 and abs(next_y - curr_y) < 1e-4:
            break

        points.append((next_x, next_y))
        curr_x = next_x
        curr_y = next_y

    return points


def compute_enstrophy(vorticity: Grid2D) -> float:
    """
    Compute domain total integrated enstrophy:
      E = 0.5 * integral(omega^2 dA)
    """
    total = 0.0
    vort_data = vorticity.data
    for val in vort_data:
        total += val * val
    return 0.5 * total * (vorticity.dx * vorticity.dx)


def compute_q_criterion(velocity: VectorField2D, out_q: Optional[Grid2D] = None) -> Grid2D:
    """
    Compute Hunt's Q-criterion for coherent vortex core identification:
      Q = 0.5 * (||Omega||^2 - ||S||^2)
    where Omega is the vorticity tensor and S is the strain-rate tensor.
    Positive Q regions represent rotation-dominated vortex cores.
    In 2D: Q = -0.5 * [(du/dx)^2 + (dv/dy)^2 + 2*(du/dy)*(dv/dx)].
    """
    if out_q is None:
        out_q = Grid2D(velocity.nx, velocity.ny, velocity.dx)

    nx, ny = velocity.nx, velocity.ny
    inv_2dx = 0.5 / velocity.dx
    u_data = velocity.u.data
    v_data = velocity.v.data
    q_data = out_q.data

    for y in range(ny):
        y_offset = y * nx
        y_prev = max(0, y - 1) * nx
        y_next = min(ny - 1, y + 1) * nx

        for x in range(nx):
            idx = y_offset + x
            x_prev = max(0, x - 1)
            x_next = min(nx - 1, x + 1)

            du_dx = (u_data[y_offset + x_next] - u_data[y_offset + x_prev]) * inv_2dx
            du_dy = (u_data[y_next + x] - u_data[y_prev + x]) * inv_2dx
            dv_dx = (v_data[y_offset + x_next] - v_data[y_offset + x_prev]) * inv_2dx
            dv_dy = (v_data[y_next + x] - v_data[y_prev + x]) * inv_2dx

            # S = 0.5 * (grad u + grad u^T), Omega = 0.5 * (grad u - grad u^T)
            # Q = 0.5 * (||Omega||^2 - ||S||^2)
            s11 = du_dx
            s22 = dv_dy
            s12 = 0.5 * (du_dy + dv_dx)
            omega12 = 0.5 * (dv_dx - du_dy)

            norm_omega_sq = 2.0 * (omega12 * omega12)
            norm_s_sq = s11 * s11 + s22 * s22 + 2.0 * (s12 * s12)

            q_val = 0.5 * (norm_omega_sq - norm_s_sq)
            q_data[idx] = q_val

    return out_q
