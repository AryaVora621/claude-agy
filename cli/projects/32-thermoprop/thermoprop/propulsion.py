"""
Rocket Propulsion Thermochemistry & Nozzle Performance Engine.

Evaluates ideal and actual rocket propulsion metrics:
  - Characteristic exhaust velocity c*
  - Thrust coefficient C_F (vacuum, sea level, and arbitrary ambient pressure)
  - Specific impulse Isp (vacuum and sea level) in seconds
  - Mass flow rate m_dot and chamber choking
  - Tsiolkovsky delta-v orbital staging calculations
  - Propellant combustion presets (Hydrolox, Methalox, Kerolox, Hypergolic)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .gas_dynamics import CompressibleFlowEngine, GasProperties


# Standard gravitational acceleration (m / s^2)
STANDARD_GRAVITY: float = 9.80665

# Standard sea-level atmospheric pressure (Pa)
SEA_LEVEL_PRESSURE: float = 101325.0


@dataclass(frozen=True)
class PropellantData:
    """Thermochemical combustion properties of a liquid rocket propellant pair."""

    name: str
    oxidizer: str
    fuel: str
    mixture_ratio: float  # O/F mass ratio
    chamber_temperature: float  # Kelvin (adiabatic flame temp)
    gas: GasProperties


class PropellantLibrary:
    """Predefined high-performance rocket propellant combustion combinations."""

    @classmethod
    def hydrolox(cls) -> PropellantData:
        """Liquid Oxygen / Liquid Hydrogen (high specific impulse)."""
        return PropellantData(
            name="Hydrolox",
            oxidizer="LOX",
            fuel="LH2",
            mixture_ratio=6.0,
            chamber_temperature=3600.0,
            gas=GasProperties.hydrolox(),
        )

    @classmethod
    def methalox(cls) -> PropellantData:
        """Liquid Oxygen / Liquid Methane (clean-burning reusable)."""
        return PropellantData(
            name="Methalox",
            oxidizer="LOX",
            fuel="LCH4",
            mixture_ratio=3.6,
            chamber_temperature=3550.0,
            gas=GasProperties.methalox(),
        )

    @classmethod
    def kerolox(cls) -> PropellantData:
        """Liquid Oxygen / RP-1 Kerosene (high booster thrust density)."""
        return PropellantData(
            name="Kerolox",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
            chamber_temperature=3700.0,
            gas=GasProperties.kerolox(),
        )

    @classmethod
    def hypergolic(cls) -> PropellantData:
        """Dinitrogen Tetroxide / Unsymmetrical Dimethylhydrazine (storable)."""
        return PropellantData(
            name="Hypergolic",
            oxidizer="N2O4",
            fuel="UDMH",
            mixture_ratio=2.1,
            chamber_temperature=3400.0,
            gas=GasProperties(gamma=1.25, molecular_weight=21.0, name="N2O4/UDMH"),
        )


@dataclass(frozen=True)
class RocketEngineState:
    """Complete operating state and propulsion metrics of a rocket engine."""

    chamber_pressure: float  # Pa
    chamber_temperature: float  # K
    throat_area: float  # m^2
    exit_area: float  # m^2
    expansion_ratio: float  # A_e / A_t
    exit_mach: float
    exit_pressure: float  # Pa
    exit_temperature: float  # K
    exit_velocity: float  # m / s
    mass_flow_rate: float  # kg / s
    characteristic_velocity_c_star: float  # m / s
    thrust_coefficient_vacuum: float
    thrust_coefficient_sea_level: float
    thrust_vacuum: float  # N
    thrust_sea_level: float  # N
    isp_vacuum_seconds: float  # s
    isp_sea_level_seconds: float  # s


class RocketPropulsionEngine:
    """
    Evaluates thrust, mass flow, c*, and Isp across atmospheric altitudes.
    """

    def __init__(self, propellant: Optional[PropellantData] = None) -> None:
        self.propellant = propellant or PropellantLibrary.methalox()
        self.flow = CompressibleFlowEngine(self.propellant.gas)

    def characteristic_velocity(self, chamber_temp: Optional[float] = None) -> float:
        """
        Ideal characteristic exhaust velocity c* in m/s:
            c* = sqrt(gamma * R * T_c) / ( gamma * sqrt( [2 / (gamma + 1)] ^ ((gamma + 1) / (gamma - 1)) ) )
        """
        tc = chamber_temp or self.propellant.chamber_temperature
        gamma = self.propellant.gas.gamma
        r_gas = self.propellant.gas.gas_constant

        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        exponent = gp1 / gm1

        gamma_factor = math.sqrt(gamma * (2.0 / gp1) ** exponent)
        c_star = math.sqrt(gamma * r_gas * tc) / gamma_factor
        return c_star

    def evaluate_engine(
        self,
        chamber_pressure: float,
        throat_radius: float,
        expansion_ratio: float,
        chamber_temperature: Optional[float] = None,
    ) -> RocketEngineState:
        """
        Compute propulsion parameters for a rocket engine thrust chamber.
        """
        if chamber_pressure <= 0.0:
            raise ValueError("Chamber pressure must be strictly positive.")
        if throat_radius <= 0.0:
            raise ValueError("Throat radius must be strictly positive.")
        if expansion_ratio < 1.0:
            raise ValueError("Expansion ratio must be >= 1.0.")

        tc = chamber_temperature or self.propellant.chamber_temperature
        throat_area = math.pi * throat_radius * throat_radius
        exit_area = expansion_ratio * throat_area

        c_star = self.characteristic_velocity(tc)
        mass_flow = (chamber_pressure * throat_area) / c_star

        # Exit flow parameters from area-Mach expansion
        exit_mach = self.flow.mach_from_area_ratio(expansion_ratio, supersonic=True)
        exit_state = self.flow.isentropic_state_from_stagnation(
            mach=exit_mach,
            t_total=tc,
            p_total=chamber_pressure,
        )

        pe = exit_state.pressure
        te = exit_state.temperature
        ve = exit_state.velocity

        # Vacuum thrust: F_vac = m_dot * v_e + P_e * A_e
        thrust_vac = mass_flow * ve + pe * exit_area
        cf_vac = thrust_vac / (chamber_pressure * throat_area)
        isp_vac = thrust_vac / (mass_flow * STANDARD_GRAVITY)

        # Sea-level thrust: F_sl = m_dot * v_e + (P_e - P_sl) * A_e
        thrust_sl = mass_flow * ve + (pe - SEA_LEVEL_PRESSURE) * exit_area
        cf_sl = thrust_sl / (chamber_pressure * throat_area)
        isp_sl = thrust_sl / (mass_flow * STANDARD_GRAVITY) if thrust_sl > 0.0 else 0.0

        return RocketEngineState(
            chamber_pressure=chamber_pressure,
            chamber_temperature=tc,
            throat_area=throat_area,
            exit_area=exit_area,
            expansion_ratio=expansion_ratio,
            exit_mach=exit_mach,
            exit_pressure=pe,
            exit_temperature=te,
            exit_velocity=ve,
            mass_flow_rate=mass_flow,
            characteristic_velocity_c_star=c_star,
            thrust_coefficient_vacuum=cf_vac,
            thrust_coefficient_sea_level=cf_sl,
            thrust_vacuum=thrust_vac,
            thrust_sea_level=thrust_sl,
            isp_vacuum_seconds=isp_vac,
            isp_sea_level_seconds=isp_sl,
        )

    def tsiolkovsky_delta_v(
        self,
        isp_seconds: float,
        wet_mass: float,
        dry_mass: float,
    ) -> float:
        """
        Ideal orbital delta-v via Tsiolkovsky rocket equation:
            delta_v = Isp * g_0 * ln(m_0 / m_f)
        """
        if wet_mass <= dry_mass:
            raise ValueError("Wet mass must exceed dry mass.")
        if dry_mass <= 0.0:
            raise ValueError("Dry mass must be positive.")

        return isp_seconds * STANDARD_GRAVITY * math.log(wet_mass / dry_mass)
