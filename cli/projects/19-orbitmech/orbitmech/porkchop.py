"""
OrbitMech: Interplanetary Porkchop Plot Generator & Trajectory Optimization.
Computes departure characteristic energy C3, arrival Delta-V, and launch windows
by solving Lambert's problem across departure and arrival date grids.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    SUN,
    AU_KM,
)
from orbitmech.kepler import orbital_elements_to_state
from orbitmech.lambert import solve_lambert


@dataclass(frozen=True)
class PorkchopPoint:
    """
    Trajectory parameters for a single departure-arrival date pair.
      departure_day: Day index from reference epoch
      arrival_day: Day index from reference epoch
      time_of_flight_days: Flight duration in days
      c3_km2_s2: Departure characteristic energy (km^2 / s^2)
      v_inf_dep: Hyperbolic excess departure speed (km/s)
      v_inf_arr: Hyperbolic excess arrival speed (km/s)
      total_delta_v: Sum of departure and arrival excess velocities (km/s)
    """
    departure_day: float
    arrival_day: float
    time_of_flight_days: float
    c3_km2_s2: float
    v_inf_dep: float
    v_inf_arr: float
    total_delta_v: float


@dataclass
class PorkchopGrid:
    """
    2D evaluation grid of interplanetary transfer opportunities.
    """
    departure_days: List[float]
    arrival_days: List[float]
    grid: List[List[PorkchopPoint]]
    min_c3_point: PorkchopPoint
    min_dv_point: PorkchopPoint

    def get_contour_matrix(self, field_name: str = "total_delta_v") -> List[List[float]]:
        """
        Extract 2D scalar float matrix for contour plotting.
        field_name can be 'total_delta_v', 'c3_km2_s2', or 'v_inf_dep'.
        """
        matrix: List[List[float]] = []
        for row in self.grid:
            matrix_row: List[float] = []
            for pt in row:
                val = getattr(pt, field_name, 0.0)
                matrix_row.append(val)
            matrix.append(matrix_row)
        return matrix

    def summary(self) -> str:
        return (
            f"=== PORKCHOP LAUNCH WINDOW OPTIMIZATION ===\n"
            f"Departure Range:  Day {self.departure_days[0]:.1f} to {self.departure_days[-1]:.1f}\n"
            f"Arrival Range:    Day {self.arrival_days[0]:.1f} to {self.arrival_days[-1]:.1f}\n"
            f"Grid Resolution:  {len(self.departure_days)} x {len(self.arrival_days)} ({len(self.departure_days)*len(self.arrival_days)} Lambert solutions)\n"
            f"\n--- Minimum Total Delta-V Trajectory ---\n"
            f"  Optimal Departure: Day {self.min_dv_point.departure_day:.1f}\n"
            f"  Optimal Arrival:   Day {self.min_dv_point.arrival_day:.1f}\n"
            f"  Time of Flight:    {self.min_dv_point.time_of_flight_days:.1f} days\n"
            f"  Departure C3:      {self.min_dv_point.c3_km2_s2:.2f} km^2/s^2\n"
            f"  V_inf Departure:   {self.min_dv_point.v_inf_dep:.3f} km/s\n"
            f"  V_inf Arrival:     {self.min_dv_point.v_inf_arr:.3f} km/s\n"
            f"  Total Excess ΔV:   {self.min_dv_point.total_delta_v:.3f} km/s\n"
            f"\n--- Minimum Departure Energy (C3) Trajectory ---\n"
            f"  Departure C3:      {self.min_c3_point.c3_km2_s2:.2f} km^2/s^2 (V_inf = {self.min_c3_point.v_inf_dep:.3f} km/s)\n"
            f"  Time of Flight:    {self.min_c3_point.time_of_flight_days:.1f} days"
        )


def approximate_planet_state(
    semi_major_axis_au: float,
    eccentricity: float,
    epoch_day: float,
    orbital_period_days: float,
    initial_mean_anomaly_deg: float = 0.0,
    inclination_deg: float = 0.0,
) -> StateVector:
    """
    Approximate heliocentric planetary state vector at epoch_day
    using classical two-body Keplerian mechanics around the Sun.
    """
    a_km = semi_major_axis_au * AU_KM
    mean_motion_deg_per_day = 360.0 / orbital_period_days
    M_deg = (initial_mean_anomaly_deg + mean_motion_deg_per_day * epoch_day) % 360.0

    # Approximate true anomaly for low eccentricity: nu ~ M + 2e*sin(M)
    M_rad = math.radians(M_deg)
    nu_rad = M_rad + 2.0 * eccentricity * math.sin(M_rad)

    elements = ClassicalOrbitalElements(
        a=a_km,
        e=eccentricity,
        i=math.radians(inclination_deg),
        raan=0.0,
        arg_peri=0.0,
        true_anomaly=nu_rad % (2.0 * math.pi),
        mu=SUN.mu,
    )
    return orbital_elements_to_state(elements)


def generate_porkchop_grid(
    dep_start_day: float,
    dep_end_day: float,
    dep_steps: int,
    arr_start_day: float,
    arr_end_day: float,
    arr_steps: int,
    r_planet1_fn: Optional[Any] = None,
    r_planet2_fn: Optional[Any] = None,
    mu: float = SUN.mu,
) -> PorkchopGrid:
    """
    Generate Porkchop plot grid across departure days and arrival days.
    By default, models canonical Earth -> Mars interplanetary transfer window.
    """
    # Default planetary ephemeris models: Earth (1.0 AU) and Mars (1.524 AU)
    if r_planet1_fn is None:
        # Earth: a=1.000 AU, e=0.0167, T=365.25 days
        r_planet1_fn = lambda day: approximate_planet_state(1.0, 0.0167, day, 365.25, initial_mean_anomaly_deg=0.0)
    if r_planet2_fn is None:
        # Mars: a=1.524 AU, e=0.0934, T=686.98 days, initial phase offset ~44 deg for 2026 window
        r_planet2_fn = lambda day: approximate_planet_state(1.524, 0.0934, day, 686.98, initial_mean_anomaly_deg=44.0, inclination_deg=1.85)

    dep_step_size = (dep_end_day - dep_start_day) / max(1, dep_steps - 1)
    arr_step_size = (arr_end_day - arr_start_day) / max(1, arr_steps - 1)

    departure_days = [dep_start_day + i * dep_step_size for i in range(dep_steps)]
    arrival_days = [arr_start_day + j * arr_step_size for j in range(arr_steps)]

    grid: List[List[PorkchopPoint]] = []
    min_c3_pt: Optional[PorkchopPoint] = None
    min_dv_pt: Optional[PorkchopPoint] = None

    for dep_day in departure_days:
        state_dep = r_planet1_fn(dep_day)
        row: List[PorkchopPoint] = []
        for arr_day in arrival_days:
            tof_days = arr_day - dep_day
            if tof_days <= 10.0:
                # Unrealistically short transfer, penalty point
                pt = PorkchopPoint(
                    departure_day=dep_day,
                    arrival_day=arr_day,
                    time_of_flight_days=tof_days,
                    c3_km2_s2=999.0,
                    v_inf_dep=99.0,
                    v_inf_arr=99.0,
                    total_delta_v=198.0,
                )
                row.append(pt)
                continue

            state_arr = r_planet2_fn(arr_day)
            tof_seconds = tof_days * 86400.0

            try:
                # Solve Lambert problem
                lambert_sol = solve_lambert(
                    r1_vec=state_dep.r,
                    r2_vec=state_arr.r,
                    time_of_flight=tof_seconds,
                    mu=mu,
                    prograde=True,
                    short_way=True,
                )
                # Excess departure velocity
                v_inf_dep_vec = lambert_sol.v1 - state_dep.v
                v_inf_dep = v_inf_dep_vec.norm()
                c3 = v_inf_dep * v_inf_dep

                # Excess arrival velocity
                v_inf_arr_vec = lambert_sol.v2 - state_arr.v
                v_inf_arr = v_inf_arr_vec.norm()

                total_dv = v_inf_dep + v_inf_arr

                pt = PorkchopPoint(
                    departure_day=dep_day,
                    arrival_day=arr_day,
                    time_of_flight_days=tof_days,
                    c3_km2_s2=c3,
                    v_inf_dep=v_inf_dep,
                    v_inf_arr=v_inf_arr,
                    total_delta_v=total_dv,
                )
            except Exception:
                # Fallback on singularity or non-convergence
                pt = PorkchopPoint(
                    departure_day=dep_day,
                    arrival_day=arr_day,
                    time_of_flight_days=tof_days,
                    c3_km2_s2=999.0,
                    v_inf_dep=99.0,
                    v_inf_arr=99.0,
                    total_delta_v=198.0,
                )

            row.append(pt)

            if min_c3_pt is None or pt.c3_km2_s2 < min_c3_pt.c3_km2_s2:
                min_c3_pt = pt
            if min_dv_pt is None or pt.total_delta_v < min_dv_pt.total_delta_v:
                min_dv_pt = pt

        grid.append(row)

    # Fallbacks if grid was degenerate
    if min_c3_pt is None:
        min_c3_pt = grid[0][0]
    if min_dv_pt is None:
        min_dv_pt = grid[0][0]

    return PorkchopGrid(
        departure_days=departure_days,
        arrival_days=arrival_days,
        grid=grid,
        min_c3_point=min_c3_pt,
        min_dv_point=min_dv_pt,
    )
