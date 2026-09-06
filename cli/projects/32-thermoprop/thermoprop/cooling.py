"""
Regenerative Thrust Chamber Cooling & Bartz Convective Heat Transfer Engine.

Models:
  - Gas-side convective heat transfer coefficient h_g via Bartz empirical correlation
  - Adiabatic wall recovery temperature T_aw with boundary layer recovery factor
  - Wall thermal conduction resistance through copper-alloy or superalloy liners
  - Coolant-side convective heat transfer h_c via Dittus-Boelter turbulent pipe flow
  - Coupled 1D thermal resistance network yielding gas wall and coolant wall temperatures
  - Coolant enthalpy uptake and bulk temperature rise along regenerative jacket channels
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .gas_dynamics import GasProperties
from .propulsion import RocketEngineState


@dataclass(frozen=True)
class WallMaterial:
    """Thermal and structural properties of the thrust chamber liner wall."""

    name: str
    thermal_conductivity: float  # W / (m * K)
    thickness: float  # meters (liner thickness)
    max_allowable_temperature: float  # Kelvin

    @classmethod
    def copper_cucrzr(cls, thickness_mm: float = 1.5) -> WallMaterial:
        """High-conductivity Copper-Chromium-Zirconium liner."""
        return cls(
            name="CuCrZr Alloy",
            thermal_conductivity=320.0,
            thickness=thickness_mm * 1e-3,
            max_allowable_temperature=950.0,
        )

    @classmethod
    def inconel_718(cls, thickness_mm: float = 2.0) -> WallMaterial:
        """Nickel-based Inconel 718 superalloy."""
        return cls(
            name="Inconel 718",
            thermal_conductivity=20.0,
            thickness=thickness_mm * 1e-3,
            max_allowable_temperature=1250.0,
        )


@dataclass(frozen=True)
class CoolantChannelProperties:
    """Regenerative cooling channel geometry and fluid thermophysics."""

    num_channels: int
    channel_width: float  # meters
    channel_height: float  # meters
    coolant_mass_flow: float  # kg / s
    coolant_cp: float  # J / (kg * K)
    coolant_thermal_conductivity: float  # W / (m * K)
    coolant_viscosity: float  # Pa * s (dynamic viscosity)
    coolant_prandtl: float

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter D_h = 4 * Area / Perimeter."""
        area = self.channel_width * self.channel_height
        perim = 2.0 * (self.channel_width + self.channel_height)
        return (4.0 * area) / perim if perim > 0.0 else 1e-3


@dataclass(frozen=True)
class ThermalStationResult:
    """Coupled aerothermodynamic heat transfer state at an axial station."""

    x: float
    radius: float
    mach: float
    gas_temperature: float
    adiabatic_wall_temperature: float
    heat_flux: float  # W / m^2
    gas_heat_transfer_coeff: float  # W / (m^2 * K)
    coolant_heat_transfer_coeff: float  # W / (m^2 * K)
    wall_gas_temperature: float  # K
    wall_coolant_temperature: float  # K
    coolant_temperature: float  # K
    safety_margin_kelvin: float  # T_max - T_wg


class RegenerativeCoolingEngine:
    """
    Solves Bartz convective gas-side correlation coupled to 1D coolant heat exchange.
    """

    def __init__(
        self,
        engine_state: RocketEngineState,
        gas: GasProperties,
        material: Optional[WallMaterial] = None,
        channel: Optional[CoolantChannelProperties] = None,
    ) -> None:
        self.engine_state = engine_state
        self.gas = gas
        self.material = material or WallMaterial.copper_cucrzr()

        # Default regenerative channel configuration (e.g. subcooled liquid methane/hydrogen)
        self.channel = channel or CoolantChannelProperties(
            num_channels=80,
            channel_width=1.5e-3,
            channel_height=3.0e-3,
            coolant_mass_flow=0.7 * engine_state.mass_flow_rate,
            coolant_cp=3500.0,
            coolant_thermal_conductivity=0.18,
            coolant_viscosity=8.0e-5,
            coolant_prandtl=1.5,
        )

    def bartz_gas_coefficient(
        self,
        local_mach: float,
        local_area_ratio: float,
        throat_radius: float,
        t_wall_gas_guess: float = 700.0,
    ) -> float:
        """
        Evaluate convective heat transfer coefficient h_g (W / (m^2 * K)) via Bartz equation:
            h_g = [ 0.026 / (D_t^0.2) ] * [ mu^0.2 * cp / Pr^0.6 ] * [ P_c / c* ]^0.8 * [ D_t / r_c ]^0.1 * [ A_t / A ]^0.9 * sigma
        """
        pc = self.engine_state.chamber_pressure
        tc = self.engine_state.chamber_temperature
        c_star = self.engine_state.characteristic_velocity_c_star
        d_t = 2.0 * throat_radius
        r_c = 1.5 * throat_radius  # Throat curvature radius

        # Core combustion gas transport approximations
        # Viscosity scaling mu ~ mu_ref * (T / T_ref)^0.6
        mu_core = 8.5e-5  # Pa * s at flame temperature
        cp_core = self.gas.cp
        pr_core = 0.82

        # Boundary layer correction factor sigma
        gamma = self.gas.gamma
        mach_factor = 1.0 + 0.5 * (gamma - 1.0) * local_mach * local_mach
        temp_ratio = t_wall_gas_guess / tc

        sigma_bracket = 0.5 * temp_ratio * mach_factor + 0.5
        sigma = (sigma_bracket ** -0.68) * (mach_factor ** -0.12)

        area_factor = (1.0 / local_area_ratio) ** 0.9 if local_area_ratio >= 1.0 else 1.0

        term_dt = 0.026 / (d_t**0.2)
        term_fluid = (mu_core**0.2) * cp_core / (pr_core**0.6)
        term_pc = (pc / c_star) ** 0.8
        term_curv = (d_t / r_c) ** 0.1

        hg = term_dt * term_fluid * term_pc * term_curv * area_factor * sigma
        return hg

    def coolant_heat_coefficient(self) -> float:
        """
        Evaluate coolant-side convective heat transfer coefficient h_c via Dittus-Boelter:
            Nu = 0.023 * Re^0.8 * Pr^0.4
            h_c = Nu * k / D_h
        """
        d_h = self.channel.hydraulic_diameter
        area_single = self.channel.channel_width * self.channel.channel_height
        total_flow_area = self.channel.num_channels * area_single

        # Coolant mass velocity G = m_dot / A_flow
        g_coolant = self.channel.coolant_mass_flow / total_flow_area if total_flow_area > 0.0 else 1.0
        re = (g_coolant * d_h) / self.channel.coolant_viscosity

        nu = 0.023 * (re**0.8) * (self.channel.coolant_prandtl**0.4)
        hc = (nu * self.channel.coolant_thermal_conductivity) / d_h
        return hc

    def solve_station_thermal_equilibrium(
        self,
        x: float,
        radius: float,
        mach: float,
        throat_radius: float,
        coolant_temp: float = 120.0,
    ) -> ThermalStationResult:
        """
        Solve coupled 1D thermal resistance network at axial station x:
            q = (T_aw - T_c) / (1/h_g + t_w/k_w + 1/h_c)
        """
        area_ratio = (radius * radius) / (throat_radius * throat_radius)
        tc = self.engine_state.chamber_temperature
        gamma = self.gas.gamma

        # Static gas temperature
        mach_factor = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
        t_gas = tc / mach_factor

        # Recovery temperature T_aw = T_gas * (1 + r * 0.5 * (gamma - 1) * M^2)
        # Turbulent recovery factor r = Pr^(1/3) ~ 0.90
        recovery_factor = 0.90
        t_aw = t_gas * (1.0 + recovery_factor * 0.5 * (gamma - 1.0) * mach * mach)

        # Coolant convective coefficient
        h_c = self.coolant_heat_coefficient()
        r_wall = self.material.thickness / self.material.thermal_conductivity

        # Iterate on wall temperature for Bartz sigma factor convergence
        t_wg = 750.0
        h_g = self.bartz_gas_coefficient(mach, area_ratio, throat_radius, t_wg)

        for _ in range(5):
            r_total = (1.0 / h_g) + r_wall + (1.0 / h_c)
            q = (t_aw - coolant_temp) / r_total
            t_wg = t_aw - q / h_g
            t_wc = coolant_temp + q / h_c
            h_g = self.bartz_gas_coefficient(mach, area_ratio, throat_radius, t_wg)

        safety_margin = self.material.max_allowable_temperature - t_wg

        return ThermalStationResult(
            x=x,
            radius=radius,
            mach=mach,
            gas_temperature=t_gas,
            adiabatic_wall_temperature=t_aw,
            heat_flux=q,
            gas_heat_transfer_coeff=h_g,
            coolant_heat_transfer_coeff=h_c,
            wall_gas_temperature=t_wg,
            wall_coolant_temperature=t_wc,
            coolant_temperature=coolant_temp,
            safety_margin_kelvin=safety_margin,
        )
