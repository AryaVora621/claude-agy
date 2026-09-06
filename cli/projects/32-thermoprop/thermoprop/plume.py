"""
Supersonic Exhaust Plume & Mach Diamond Shock Structure Engine.

Models:
  - Nozzle adaptation regimes: Overexpanded (Pe < Pa), Adapted (Pe = Pa), Underexpanded (Pe > Pa)
  - Summerfield flow separation criterion (Pe / Pa <= 0.35)
  - Periodic shock cell (Mach diamond) wavelength L = 1.306 * De * sqrt(Mj^2 - 1)
  - Oblique shock reflections, Mach stems, and diamond vertex geometry
  - Viscous turbulent shear layer boundary spreading
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .gas_dynamics import CompressibleFlowEngine, GasProperties


class PlumeRegime(Enum):
    """Atmospheric pressure adaptation classification of supersonic rocket exhaust."""

    SEPARATED = "Separated Flow (Severe Overexpansion)"
    OVEREXPANDED = "Overexpanded (Oblique Lip Shock Waves)"
    IDEALLY_EXPANDED = "Ideally Expanded (Adapted Matched Pressure)"
    UNDEREXPANDED = "Underexpanded (Prandtl-Meyer Lip Expansion Fan)"


@dataclass(frozen=True)
class MachDiamondCell:
    """Coordinates and geometry of an individual periodic Mach diamond shock cell."""

    index: int
    axial_start: float
    axial_center: float  # Centerline diamond vertex or Mach disk
    axial_end: float
    width: float
    mach_stem_radius: float


@dataclass
class PlumeStructure:
    """Spatial geometry of supersonic exhaust jet, shock cells, and shear layer."""

    regime: PlumeRegime
    pressure_ratio: float  # P_e / P_a
    cell_wavelength: float  # Spacing between successive diamonds
    diamonds: List[MachDiamondCell]
    boundary_points: List[Tuple[float, float]]  # (x, y) upper shear layer
    shock_lines: List[List[Tuple[float, float]]]  # (x, y) shock segments


class ExhaustPlumeEngine:
    """
    Computes supersonic exhaust plume flowfields, shock reflections, and Mach diamonds.
    """

    def __init__(self, gas: Optional[GasProperties] = None) -> None:
        self.gas = gas or GasProperties.methalox()
        self.flow = CompressibleFlowEngine(self.gas)

    def determine_regime(self, p_exit: float, p_ambient: float) -> PlumeRegime:
        """Classify plume adaptation regime based on pressure ratio Pe / Pa."""
        if p_ambient <= 1e-4:
            return PlumeRegime.UNDEREXPANDED  # Vacuum expansion

        ratio = p_exit / p_ambient
        if ratio <= 0.35:
            return PlumeRegime.SEPARATED
        if ratio < 0.95:
            return PlumeRegime.OVEREXPANDED
        if ratio <= 1.05:
            return PlumeRegime.IDEALLY_EXPANDED
        return PlumeRegime.UNDEREXPANDED

    def calculate_shock_cell_spacing(
        self,
        exit_diameter: float,
        chamber_pressure: float,
        p_ambient: float,
    ) -> float:
        """
        Prandtl's formula for periodic shock cell wavelength L:
            L = 1.306 * D_e * sqrt(M_j^2 - 1)
        where M_j is the fully expanded jet Mach number matching ambient pressure.
        """
        if p_ambient <= 1e-4:
            return exit_diameter * 4.0

        ratio = chamber_pressure / p_ambient
        gamma = self.gas.gamma
        gm1 = gamma - 1.0

        if ratio <= 1.0:
            return exit_diameter * 1.5

        # Isentropic fully expanded jet Mach M_j: P0 / Pa = (1 + 0.5 * (gamma - 1) * M_j^2) ^ (gamma / (gamma - 1))
        bracket = ratio ** (gm1 / gamma) - 1.0
        m_j_sq = (2.0 / gm1) * max(0.0, bracket)
        m_j = math.sqrt(max(1.001, m_j_sq))

        wavelength = 1.306 * exit_diameter * math.sqrt(m_j * m_j - 1.0)
        return wavelength

    def simulate_plume(
        self,
        x_exit: float,
        exit_radius: float,
        exit_mach: float,
        p_exit: float,
        chamber_pressure: float,
        p_ambient: float,
        plume_length: float = 8.0,
        num_cells: int = 5,
    ) -> PlumeStructure:
        """
        Generate spatial geometry of supersonic plume boundaries and internal Mach diamond shock cells.
        """
        regime = self.determine_regime(p_exit, p_ambient)
        pressure_ratio = p_exit / p_ambient if p_ambient > 1e-4 else 100.0
        d_e = 2.0 * exit_radius

        wavelength = self.calculate_shock_cell_spacing(d_e, chamber_pressure, p_ambient)

        diamonds: List[MachDiamondCell] = []
        shock_lines: List[List[Tuple[float, float]]] = []

        # Generate periodic diamond cells
        curr_x = x_exit
        for k in range(1, num_cells + 1):
            cell_start = curr_x
            cell_center = curr_x + 0.5 * wavelength
            cell_end = curr_x + wavelength

            if cell_start >= x_exit + plume_length:
                break

            # Oblique shock lines forming the diamond
            # Lip to centerline
            shock_top_down = [(cell_start, exit_radius), (cell_center, 0.0)]
            shock_bottom_up = [(cell_start, -exit_radius), (cell_center, 0.0)]
            # Centerline reflection to outer shear layer
            shock_reflect_up = [(cell_center, 0.0), (cell_end, exit_radius * 0.95)]
            shock_reflect_down = [(cell_center, 0.0), (cell_end, -exit_radius * 0.95)]

            shock_lines.extend([shock_top_down, shock_bottom_up, shock_reflect_up, shock_reflect_down])

            # Mach stem disk radius if overexpanded
            stem_radius = exit_radius * 0.15 if regime == PlumeRegime.OVEREXPANDED else 0.0
            if stem_radius > 0.0:
                # Vertical Mach disk
                shock_lines.append([(cell_center, -stem_radius), (cell_center, stem_radius)])

            diamonds.append(
                MachDiamondCell(
                    index=k,
                    axial_start=cell_start,
                    axial_center=cell_center,
                    axial_end=cell_end,
                    width=wavelength,
                    mach_stem_radius=stem_radius,
                )
            )
            curr_x = cell_end

        # Generate jet shear layer boundary y_boundary(x)
        boundary_points: List[Tuple[float, float]] = []
        num_boundary_steps = 80
        dx = plume_length / num_boundary_steps

        for step in range(num_boundary_steps + 1):
            x = x_exit + step * dx
            rel_x = x - x_exit

            # Turbulent mixing layer spreading: dr/dx ~ 0.09
            turbulent_spread = 0.09 * rel_x

            # Periodic wave modulation along shock cells
            cell_phase = (2.0 * math.pi * rel_x) / wavelength if wavelength > 0.0 else 0.0
            wave_mod = 0.12 * exit_radius * math.cos(cell_phase)

            # Pressure mismatch initial deflection
            if regime == PlumeRegime.UNDEREXPANDED:
                initial_deflect = exit_radius * 0.3 * (1.0 - math.exp(-rel_x / (0.5 * d_e)))
            elif regime == PlumeRegime.OVEREXPANDED:
                initial_deflect = -exit_radius * 0.15 * (1.0 - math.exp(-rel_x / (0.5 * d_e)))
            else:
                initial_deflect = 0.0

            y_b = exit_radius + initial_deflect + wave_mod + turbulent_spread
            boundary_points.append((x, max(0.1 * exit_radius, y_b)))

        return PlumeStructure(
            regime=regime,
            pressure_ratio=pressure_ratio,
            cell_wavelength=wavelength,
            diamonds=diamonds,
            boundary_points=boundary_points,
            shock_lines=shock_lines,
        )
