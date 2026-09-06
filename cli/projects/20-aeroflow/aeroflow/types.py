"""
AeroFlow: Fundamental Data Structures & Geometric Types for 2D CFD.
Zero external dependencies. High-performance flat memory layouts.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Sequence


@dataclass(frozen=True)
class Vector2D:
    """Immutable 2D Euclidean vector."""
    x: float
    y: float

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        inv = 1.0 / scalar
        return Vector2D(self.x * inv, self.y * inv)

    def __neg__(self) -> Vector2D:
        return Vector2D(-self.x, -self.y)

    def dot(self, other: Vector2D) -> float:
        return self.x * other.x + self.y * other.y

    def cross_2d(self, other: Vector2D) -> float:
        """2D cross product magnitude (z-component): x1*y2 - y1*x2."""
        return self.x * other.y - self.y * other.x

    def norm_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def normalized(self) -> Vector2D:
        n = self.norm()
        if n == 0.0:
            return Vector2D(0.0, 0.0)
        return self / n


class Grid2D:
    """
    High-performance 2D continuous/discrete scalar field on structured cartesian grid.
    Stored as a contiguous 1D flat list in row-major order: index = y * nx + x.
    """
    __slots__ = ("nx", "ny", "dx", "data")

    def __init__(self, nx: int, ny: int, dx: float = 1.0, initial_value: float = 0.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.data: List[float] = [initial_value] * (nx * ny)

    def get(self, x: int, y: int) -> float:
        """Get cell value with periodic or clamped boundary protection."""
        cx = max(0, min(self.nx - 1, x))
        cy = max(0, min(self.ny - 1, y))
        return self.data[cy * self.nx + cx]

    def set(self, x: int, y: int, value: float) -> None:
        """Set cell value at coordinate (x, y)."""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            self.data[y * self.nx + x] = value

    def sample_bilinear(self, px: float, py: float) -> float:
        """
        Sample continuous coordinate (px, py) using bilinear interpolation.
        Essential for semi-Lagrangian fluid backtracing advection.
        """
        # Clamp to domain bounds
        px = max(0.0, min(float(self.nx - 1), px))
        py = max(0.0, min(float(self.ny - 1), py))

        x0 = int(px)
        y0 = int(py)
        x1 = min(x0 + 1, self.nx - 1)
        y1 = min(y0 + 1, self.ny - 1)

        fx = px - x0
        fy = py - y0

        idx00 = y0 * self.nx + x0
        idx10 = y0 * self.nx + x1
        idx01 = y1 * self.nx + x0
        idx11 = y1 * self.nx + x1

        top = (1.0 - fx) * self.data[idx00] + fx * self.data[idx10]
        bot = (1.0 - fx) * self.data[idx01] + fx * self.data[idx11]

        return (1.0 - fy) * top + fy * bot

    def fill(self, value: float) -> None:
        """Reset all cells to uniform scalar value."""
        self.data = [value] * len(self.data)

    def copy(self) -> Grid2D:
        """Create deep copy of scalar grid."""
        clone = Grid2D(self.nx, self.ny, self.dx)
        clone.data = list(self.data)
        return clone

    def min_max(self) -> Tuple[float, float]:
        """Compute minimum and maximum values across the grid."""
        return min(self.data), max(self.data)

    def mean(self) -> float:
        """Compute arithmetic mean across all grid cells."""
        return sum(self.data) / len(self.data)


class VectorField2D:
    """
    2D velocity vector field containing horizontal (u) and vertical (v) components.
    """
    __slots__ = ("nx", "ny", "dx", "u", "v")

    def __init__(self, nx: int, ny: int, dx: float = 1.0, u_init: float = 0.0, v_init: float = 0.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.u = Grid2D(nx, ny, dx, u_init)
        self.v = Grid2D(nx, ny, dx, v_init)

    def get_vector(self, x: int, y: int) -> Vector2D:
        return Vector2D(self.u.get(x, y), self.v.get(x, y))

    def set_vector(self, x: int, y: int, vec: Vector2D) -> None:
        self.u.set(x, y, vec.x)
        self.v.set(x, y, vec.y)

    def sample_bilinear(self, px: float, py: float) -> Vector2D:
        """Bilinear velocity interpolation at arbitrary fractional position."""
        return Vector2D(self.u.sample_bilinear(px, py), self.v.sample_bilinear(px, py))

    def compute_divergence(self, out_div: Optional[Grid2D] = None) -> Grid2D:
        """
        Compute discrete velocity divergence: div(u) = du/dx + dv/dy.
        Uses central differences in interior, one-sided at boundaries.
        """
        if out_div is None:
            out_div = Grid2D(self.nx, self.ny, self.dx)

        inv_2dx = 0.5 / self.dx
        nx, ny = self.nx, self.ny
        u_data = self.u.data
        v_data = self.v.data
        div_data = out_div.data

        for y in range(ny):
            y_offset = y * nx
            y_prev_offset = max(0, y - 1) * nx
            y_next_offset = min(ny - 1, y + 1) * nx

            for x in range(nx):
                # du/dx
                x_prev = max(0, x - 1)
                x_next = min(nx - 1, x + 1)
                du_dx = (u_data[y_offset + x_next] - u_data[y_offset + x_prev]) * inv_2dx

                # dv/dy
                dv_dy = (v_data[y_next_offset + x] - v_data[y_prev_offset + x]) * inv_2dx

                div_data[y_offset + x] = du_dx + dv_dy

        return out_div

    def compute_vorticity(self, out_vort: Optional[Grid2D] = None) -> Grid2D:
        """
        Compute discrete 2D scalar vorticity: omega = dv/dx - du/dy.
        Positive indicates counter-clockwise circulation, negative indicates clockwise.
        """
        if out_vort is None:
            out_vort = Grid2D(self.nx, self.ny, self.dx)

        inv_2dx = 0.5 / self.dx
        nx, ny = self.nx, self.ny
        u_data = self.u.data
        v_data = self.v.data
        vort_data = out_vort.data

        for y in range(ny):
            y_offset = y * nx
            y_prev_offset = max(0, y - 1) * nx
            y_next_offset = min(ny - 1, y + 1) * nx

            for x in range(nx):
                x_prev = max(0, x - 1)
                x_next = min(nx - 1, x + 1)

                dv_dx = (v_data[y_offset + x_next] - v_data[y_offset + x_prev]) * inv_2dx
                du_dy = (u_data[y_next_offset + x] - u_data[y_prev_offset + x]) * inv_2dx

                vort_data[y_offset + x] = dv_dx - du_dy

        return out_vort

    def compute_kinetic_energy(self) -> float:
        """Total integrated kinetic energy: 0.5 * sum(u^2 + v^2) * dx^2."""
        total = 0.0
        u_data = self.u.data
        v_data = self.v.data
        for i in range(len(u_data)):
            total += u_data[i] * u_data[i] + v_data[i] * v_data[i]
        return 0.5 * total * (self.dx * self.dx)


class ObstacleMask:
    """
    2D solid obstacle boolean mask for no-slip solid geometries.
    True represents a solid wall node, False represents fluid flow.
    """
    __slots__ = ("nx", "ny", "mask", "boundary_nodes")

    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.mask: List[bool] = [False] * (nx * ny)
        self.boundary_nodes: List[Tuple[int, int]] = []

    def set_solid(self, x: int, y: int, is_solid: bool = True) -> None:
        if 0 <= x < self.nx and 0 <= y < self.ny:
            self.mask[y * self.nx + x] = is_solid

    def is_solid(self, x: int, y: int) -> bool:
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return self.mask[y * self.nx + x]
        return True  # Domain exterior treated as solid boundary

    def update_boundary_list(self) -> None:
        """Find all solid nodes that have at least one adjacent fluid node."""
        self.boundary_nodes.clear()
        nx, ny = self.nx, self.ny
        for y in range(ny):
            for x in range(nx):
                if self.is_solid(x, y):
                    # Check 4-connectivity for adjacent fluid
                    if (not self.is_solid(x - 1, y) or
                        not self.is_solid(x + 1, y) or
                        not self.is_solid(x, y - 1) or
                        not self.is_solid(x, y + 1)):
                        self.boundary_nodes.append((x, y))


@dataclass
class FluidParams:
    """
    Physical and numerical parameters for fluid simulations.
    """
    viscosity: float = 0.01          # Kinematic viscosity (nu = mu / rho)
    density: float = 1.0             # Reference fluid density (rho_0)
    inflow_velocity: float = 0.1     # Freestream inflow velocity (U_inf)
    reynolds_number: Optional[float] = None

    def compute_reynolds(self, characteristic_length: float) -> float:
        """Compute non-dimensional Reynolds number: Re = (U * L) / nu."""
        if self.viscosity <= 0:
            return float("inf")
        self.reynolds_number = (self.inflow_velocity * characteristic_length) / self.viscosity
        return self.reynolds_number
