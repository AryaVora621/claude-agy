"""
AeroFlow: Aerodynamic Force Analysis & Vortex Dynamics.
Calculates Drag (CD), Lift (CL), Lift-to-Drag Ratio (L/D), surface pressure integration,
and Strouhal number (vortex shedding frequency).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from aeroflow.types import Vector2D, Grid2D, ObstacleMask


@dataclass
class AerodynamicForces:
    """
    Dimensional and non-dimensional aerodynamic forces acting on an immersed body.
    """
    drag_force: float = 0.0          # F_x in flow direction
    lift_force: float = 0.0          # F_y perpendicular to flow
    drag_coeff: float = 0.0          # C_D = 2 * F_D / (rho * U^2 * D)
    lift_coeff: float = 0.0          # C_L = 2 * F_L / (rho * U^2 * D)
    lift_to_drag: float = 0.0        # C_L / C_D
    strouhal_number: Optional[float] = None  # St = f * D / U

    def summary(self) -> str:
        st_str = f"{self.strouhal_number:.4f}" if self.strouhal_number is not None else "N/A"
        return (
            f"Drag Coeff (CD): {self.drag_coeff:.4f} | "
            f"Lift Coeff (CL): {self.lift_coeff:.4f} | "
            f"L/D: {self.lift_to_drag:.3f} | "
            f"Strouhal (St): {st_str}"
        )


class AerodynamicTracker:
    """
    Monitors aerodynamic force histories, computes time averages, RMS fluctuations,
    and extracts vortex shedding frequency via discrete Fourier analysis / peak counting.
    """
    def __init__(
        self,
        characteristic_length: float = 10.0,
        freestream_speed: float = 0.1,
        fluid_density: float = 1.0,
        time_step: float = 1.0,
    ):
        self.characteristic_length = characteristic_length
        self.freestream_speed = freestream_speed
        self.fluid_density = fluid_density
        self.time_step = time_step

        self.time_history: List[float] = []
        self.cl_history: List[float] = []
        self.cd_history: List[float] = []

    def record(self, step: int, fx: float, fy: float) -> AerodynamicForces:
        """
        Record instantaneous force measurement and compute current aerodynamic coefficients.
        """
        t = step * self.time_step
        u = self.freestream_speed
        d = self.characteristic_length
        rho = self.fluid_density

        denom = 0.5 * rho * (u * u) * d
        cd = fx / denom if denom > 1e-12 else 0.0
        cl = fy / denom if denom > 1e-12 else 0.0
        l_over_d = cl / cd if abs(cd) > 1e-6 else 0.0

        self.time_history.append(t)
        self.cd_history.append(cd)
        self.cl_history.append(cl)

        st = self.estimate_strouhal_number()
        return AerodynamicForces(
            drag_force=fx,
            lift_force=fy,
            drag_coeff=cd,
            lift_coeff=cl,
            lift_to_drag=l_over_d,
            strouhal_number=st,
        )

    def estimate_strouhal_number(self, min_cycles: int = 2) -> Optional[float]:
        """
        Estimate Strouhal number from oscillatory lift coefficient zero-crossings:
          St = f * D / U
        Requires at least `min_cycles` periodic oscillations in lift history.
        """
        if len(self.cl_history) < 40:
            return None

        # Look at the latter half of the signal to skip initial startup transients
        n = len(self.cl_history)
        recent_cl = self.cl_history[n // 2:]
        recent_t = self.time_history[n // 2:]

        # Mean-center the signal
        mean_cl = sum(recent_cl) / len(recent_cl)
        centered = [c - mean_cl for c in recent_cl]

        # Detect upward zero-crossings
        crossings: List[float] = []
        for i in range(len(centered) - 1):
            if centered[i] <= 0.0 and centered[i + 1] > 0.0:
                # Linear interpolation for fractional crossing time
                t0, t1 = recent_t[i], recent_t[i + 1]
                v0, v1 = centered[i], centered[i + 1]
                t_cross = t0 + (-v0) * (t1 - t0) / (v1 - v0) if (v1 - v0) != 0 else t0
                crossings.append(t_cross)

        if len(crossings) < (min_cycles + 1):
            return None

        # Calculate average period T between consecutive crossings
        periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
        avg_period = sum(periods) / len(periods)

        if avg_period <= 1e-6:
            return None

        freq = 1.0 / avg_period
        strouhal = (freq * self.characteristic_length) / self.freestream_speed
        return strouhal

    def get_statistics(self) -> Tuple[float, float, float]:
        """
        Returns (mean_cd, mean_cl, rms_cl_fluctuation) over steady-state signal.
        """
        if not self.cd_history:
            return 0.0, 0.0, 0.0

        n = len(self.cd_history)
        sample_cd = self.cd_history[n // 2:]
        sample_cl = self.cl_history[n // 2:]

        mean_cd = sum(sample_cd) / len(sample_cd)
        mean_cl = sum(sample_cl) / len(sample_cl)
        variance_cl = sum((c - mean_cl) ** 2 for c in sample_cl) / len(sample_cl)
        rms_cl = math.sqrt(variance_cl)

        return mean_cd, mean_cl, rms_cl
