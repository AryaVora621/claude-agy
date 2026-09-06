"""
Two-Dimensional Method of Characteristics (MOC) Supersonic Nozzle Design Engine.

Synthesizes shock-free minimum-length supersonic De Laval nozzle contours (MLN)
by solving the hyperbolic 2D irrotational potential flow equations along
C+ and C- characteristic Mach lines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .gas_dynamics import CompressibleFlowEngine, GasProperties


@dataclass(frozen=True)
class CharacteristicPoint:
    """State and spatial coordinates at a Method of Characteristics node."""

    x: float
    y: float
    theta: float  # Streamline angle in radians
    nu: float  # Prandtl-Meyer angle in radians
    mach: float
    mu: float  # Mach angle in radians

    @property
    def k_plus(self) -> float:
        """Riemann invariant along C+ characteristic: theta - nu."""
        return self.theta - self.nu

    @property
    def k_minus(self) -> float:
        """Riemann invariant along C- characteristic: theta + nu."""
        return self.theta + self.nu


@dataclass
class NozzleContour:
    """Designed supersonic De Laval nozzle geometry and flow properties."""

    throat_radius: float
    exit_radius: float
    length: float
    exit_mach: float
    expansion_ratio: float
    wall_points: List[Tuple[float, float]]
    characteristic_mesh: List[List[CharacteristicPoint]]

    def radius_at(self, x: float) -> float:
        """Linear interpolation of nozzle radius y(x) at axial station x."""
        if x <= self.wall_points[0][0]:
            return self.wall_points[0][1]
        if x >= self.wall_points[-1][0]:
            return self.wall_points[-1][1]

        # Binary search for interval
        low = 0
        high = len(self.wall_points) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if self.wall_points[mid][0] <= x:
                low = mid
            else:
                high = mid

        x0, y0 = self.wall_points[low]
        x1, y1 = self.wall_points[high]
        factor = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
        return y0 + factor * (y1 - y0)


class MethodOfCharacteristicsNozzle:
    """
    Method of Characteristics (MOC) 2D supersonic nozzle designer.
    """

    def __init__(self, gas: Optional[GasProperties] = None) -> None:
        self.gas = gas or GasProperties.air()
        self.flow = CompressibleFlowEngine(self.gas)

    def design_minimum_length_nozzle(
        self,
        target_exit_mach: float,
        throat_height: float = 1.0,
        num_expansion_waves: int = 15,
    ) -> NozzleContour:
        """
        Synthesize a 2D minimum-length supersonic nozzle (MLN) for target exit Mach number.

        Maximum wall turning angle: theta_max = 0.5 * nu(M_exit).
        """
        if target_exit_mach <= 1.05:
            raise ValueError("Target exit Mach must be strictly greater than 1.05.")
        if num_expansion_waves < 3:
            raise ValueError("Must specify at least 3 expansion characteristic waves.")

        # 1. Total Prandtl-Meyer expansion angle required to reach target exit Mach
        nu_exit = self.flow.prandtl_meyer_nu(target_exit_mach)
        theta_max = 0.5 * nu_exit

        # Discretize initial expansion fan at throat corner (x=0, y=throat_height)
        # Each characteristic ray i has theta_i = i * theta_max / N and nu_i = theta_i
        d_theta = theta_max / num_expansion_waves

        mesh: List[List[CharacteristicPoint]] = []

        # Characteristic lines originating from expansion fan
        # Level 0: Points along the initial characteristics radiating from throat
        first_line: List[CharacteristicPoint] = []
        for i in range(1, num_expansion_waves + 1):
            theta_i = i * d_theta
            nu_i = theta_i
            mach_i = self.flow.mach_from_prandtl_meyer(nu_i)
            mu_i = self.flow.mach_angle(mach_i)

            # Ray originates at throat lip (0, throat_height)
            # Propagate to axis or first interior line
            pt = CharacteristicPoint(
                x=0.0,
                y=throat_height,
                theta=theta_i,
                nu=nu_i,
                mach=mach_i,
                mu=mu_i,
            )
            first_line.append(pt)

        # Build characteristic net using unit processes
        current_line: List[CharacteristicPoint] = []

        # Intersect first expansion fan with the centerline (y = 0)
        # Centerline unit process: theta = 0, nu = theta_orig + nu_orig
        for i, pt_orig in enumerate(first_line):
            if i == 0:
                # First point meets centerline
                theta_axis = 0.0
                nu_axis = pt_orig.theta + pt_orig.nu
                mach_axis = self.flow.mach_from_prandtl_meyer(nu_axis)
                mu_axis = self.flow.mach_angle(mach_axis)

                # Slope of C- line from throat corner to axis
                wave_angle = 0.5 * ((pt_orig.theta - pt_orig.mu) + (theta_axis - mu_axis))
                slope = math.tan(wave_angle)
                # y_axis = 0 = y_orig + slope * (x_axis - x_orig)
                x_axis = pt_orig.x - pt_orig.y / slope if abs(slope) > 1e-9 else pt_orig.x + 1.0

                current_line.append(
                    CharacteristicPoint(
                        x=x_axis,
                        y=0.0,
                        theta=theta_axis,
                        nu=nu_axis,
                        mach=mach_axis,
                        mu=mu_axis,
                    )
                )
            else:
                # Interior point: intersection of C+ from current_line[-1] and C- from pt_orig
                pt_a = current_line[-1]  # on C+
                pt_b = pt_orig          # on C-

                theta_c = 0.5 * (pt_b.k_minus + pt_a.k_plus)
                nu_c = 0.5 * (pt_b.k_minus - pt_a.k_plus)
                mach_c = self.flow.mach_from_prandtl_meyer(nu_c)
                mu_c = self.flow.mach_angle(mach_c)

                # Characteristic slopes
                slope_plus = math.tan(0.5 * ((pt_a.theta + pt_a.mu) + (theta_c + mu_c)))
                slope_minus = math.tan(0.5 * ((pt_b.theta - pt_b.mu) + (theta_c - mu_c)))

                # Intersect lines y - y_a = m_plus * (x - x_a) and y - y_b = m_minus * (x - x_b)
                denom = slope_plus - slope_minus
                if abs(denom) > 1e-9:
                    x_c = (pt_b.y - pt_a.y + slope_plus * pt_a.x - slope_minus * pt_b.x) / denom
                    y_c = pt_a.y + slope_plus * (x_c - pt_a.x)
                else:
                    x_c = 0.5 * (pt_a.x + pt_b.x)
                    y_c = 0.5 * (pt_a.y + pt_b.y)

                current_line.append(
                    CharacteristicPoint(
                        x=x_c,
                        y=max(0.0, y_c),
                        theta=theta_c,
                        nu=nu_c,
                        mach=mach_c,
                        mu=mu_c,
                    )
                )

        mesh.append(current_line)

        # 2. Straightening section & Wall synthesis
        # The wall starts at the throat (0, throat_height)
        wall_points: List[Tuple[float, float]] = [(0.0, throat_height)]
        theta_wall_prev = theta_max

        # Trace subsequent characteristic lines until the entire kernel reaches uniform flow
        active_line = current_line
        while len(active_line) > 1:
            next_line: List[CharacteristicPoint] = []
            for j in range(1, len(active_line)):
                pt_a = next_line[-1] if next_line else active_line[0]
                pt_b = active_line[j]

                theta_c = 0.5 * (pt_b.k_minus + pt_a.k_plus)
                nu_c = 0.5 * (pt_b.k_minus - pt_a.k_plus)
                mach_c = self.flow.mach_from_prandtl_meyer(nu_c)
                mu_c = self.flow.mach_angle(mach_c)

                slope_plus = math.tan(0.5 * ((pt_a.theta + pt_a.mu) + (theta_c + mu_c)))
                slope_minus = math.tan(0.5 * ((pt_b.theta - pt_b.mu) + (theta_c - mu_c)))

                denom = slope_plus - slope_minus
                if abs(denom) > 1e-9:
                    x_c = (pt_b.y - pt_a.y + slope_plus * pt_a.x - slope_minus * pt_b.x) / denom
                    y_c = pt_a.y + slope_plus * (x_c - pt_a.x)
                else:
                    x_c = 0.5 * (pt_a.x + pt_b.x)
                    y_c = 0.5 * (pt_a.y + pt_b.y)

                next_line.append(
                    CharacteristicPoint(
                        x=x_c,
                        y=max(0.0, y_c),
                        theta=theta_c,
                        nu=nu_c,
                        mach=mach_c,
                        mu=mu_c,
                    )
                )

            # Compute wall point intersecting C+ from last point of active_line
            pt_wall_in = active_line[-1]
            theta_wall = max(0.0, pt_wall_in.theta)
            # Wall slope matches average streamline angle between stations
            slope_wall = math.tan(max(0.0, 0.5 * (theta_wall_prev + theta_wall)))
            slope_c_plus = math.tan(pt_wall_in.theta + pt_wall_in.mu)

            x_w_prev, y_w_prev = wall_points[-1]
            denom_w = slope_c_plus - slope_wall
            if abs(denom_w) > 1e-9:
                x_w = (y_w_prev - pt_wall_in.y + slope_c_plus * pt_wall_in.x - slope_wall * x_w_prev) / denom_w
                y_w = y_w_prev + slope_wall * (x_w - x_w_prev)
            else:
                x_w = x_w_prev + 0.5
                y_w = y_w_prev + slope_wall * 0.5

            wall_points.append((max(x_w_prev + 1e-4, x_w), max(y_w_prev, y_w)))
            theta_wall_prev = theta_wall
            mesh.append(next_line)
            active_line = next_line

        # Ensure exit point reaches exit lip
        exit_radius = wall_points[-1][1]
        exit_length = wall_points[-1][0]
        expansion_ratio = exit_radius / throat_height

        return NozzleContour(
            throat_radius=throat_height,
            exit_radius=exit_radius,
            length=exit_length,
            exit_mach=target_exit_mach,
            expansion_ratio=expansion_ratio,
            wall_points=wall_points,
            characteristic_mesh=mesh,
        )
