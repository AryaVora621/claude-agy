"""
Compressible Gas Dynamics, Aerothermodynamics & Shock Wave Engine.

Implements exact 1D/2D compressible flow physics:
  - Isentropic stagnation to static relations (T, P, rho, acoustic velocity a)
  - Area-Mach relation A/A*(M) with high-order Newton/Halley inversion
  - Normal shock wave Rankine-Hugoniot jump conditions
  - Oblique shock theta-beta-Mach relation (weak and strong shock solutions)
  - Prandtl-Meyer supersonic expansion fan function nu(M) and inverse solver
  - Mach angle mu(M) = arcsin(1/M)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# Universal gas constant (J / (kmol * K))
UNIVERSAL_GAS_CONSTANT: float = 8314.462618


@dataclass(frozen=True)
class GasProperties:
    """Thermodynamic properties of a compressible gas species."""

    gamma: float  # Ratio of specific heats (cp / cv)
    molecular_weight: float  # kg / kmol (g / mol)
    name: str = "Ideal Gas"

    def __post_init__(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError("Ratio of specific heats gamma must be greater than 1.0.")
        if self.molecular_weight <= 0.0:
            raise ValueError("Molecular weight must be strictly positive.")

    @property
    def gas_constant(self) -> float:
        """Specific gas constant R = R_universal / M_w in J / (kg * K)."""
        return UNIVERSAL_GAS_CONSTANT / self.molecular_weight

    @property
    def cp(self) -> float:
        """Specific heat at constant pressure in J / (kg * K)."""
        return (self.gamma * self.gas_constant) / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        """Specific heat at constant volume in J / (kg * K)."""
        return self.gas_constant / (self.gamma - 1.0)

    def speed_of_sound(self, temperature: float) -> float:
        """Speed of sound a = sqrt(gamma * R * T) in m/s."""
        if temperature <= 0.0:
            raise ValueError("Temperature must be positive in Kelvin.")
        return math.sqrt(self.gamma * self.gas_constant * temperature)

    @classmethod
    def air(cls) -> GasProperties:
        return cls(gamma=1.40, molecular_weight=28.9647, name="Air")

    @classmethod
    def hydrolox(cls) -> GasProperties:
        """LOX / LH2 combustion gas (hydrogen rich)."""
        return cls(gamma=1.22, molecular_weight=13.5, name="LOX/LH2 Exhaust")

    @classmethod
    def methalox(cls) -> GasProperties:
        """LOX / LCH4 combustion gas."""
        return cls(gamma=1.20, molecular_weight=20.0, name="LOX/LCH4 Exhaust")

    @classmethod
    def kerolox(cls) -> GasProperties:
        """LOX / RP-1 combustion gas."""
        return cls(gamma=1.24, molecular_weight=23.5, name="LOX/RP-1 Exhaust")


@dataclass(frozen=True)
class IsentropicFlowState:
    """Local compressible aerodynamic flow state."""

    mach: float
    temperature: float  # Kelvin
    pressure: float  # Pascals
    density: float  # kg / m^3
    velocity: float  # m / s
    speed_of_sound: float  # m / s
    total_temperature: float  # Kelvin (stagnation)
    total_pressure: float  # Pascals (stagnation)
    dynamic_pressure: float  # Pascals (0.5 * rho * v^2)


# Alias for backward compatibility
FlowState = IsentropicFlowState


@dataclass(frozen=True)
class NormalShockState:
    """Aerodynamic jump state across a normal shock wave."""

    mach_upstream: float
    mach_downstream: float
    pressure_ratio: float
    temperature_ratio: float
    density_ratio: float
    stagnation_pressure_ratio: float
    entropy_change: float  # J / (kg * K)
    p_downstream: float  # Pa
    t_downstream: float  # K


@dataclass(frozen=True)
class ObliqueShockState:
    """Aerodynamic state across an oblique shock wave."""

    mach_upstream: float
    deflection_angle: float  # theta in radians
    wave_angle: float  # beta in radians
    mach_downstream: float
    pressure_ratio: float
    temperature_ratio: float
    density_ratio: float
    stagnation_pressure_ratio: float
    p_downstream: float  # Pa
    t_downstream: float  # K


class CompressibleFlowEngine:
    """
    Evaluates exact 1D isentropic compressible flow relations and shock waves.
    """

    def __init__(self, gas: Optional[GasProperties] = None) -> None:
        self.gas = gas or GasProperties.air()

    def temperature_ratio(self, mach: float) -> float:
        """Static to stagnation temperature ratio T / T0 = 1 / (1 + 0.5 * (gamma - 1) * M^2)."""
        gamma = self.gas.gamma
        return 1.0 / (1.0 + 0.5 * (gamma - 1.0) * mach * mach)

    def pressure_ratio(self, mach: float) -> float:
        """Static to stagnation pressure ratio P / P0 = (T / T0) ^ (gamma / (gamma - 1))."""
        t_rat = self.temperature_ratio(mach)
        gamma = self.gas.gamma
        return t_rat ** (gamma / (gamma - 1.0))

    def density_ratio(self, mach: float) -> float:
        """Static to stagnation density ratio rho / rho0 = (T / T0) ^ (1 / (gamma - 1))."""
        t_rat = self.temperature_ratio(mach)
        gamma = self.gas.gamma
        return t_rat ** (1.0 / (gamma - 1.0))

    def isentropic_state_from_stagnation(
        self,
        mach: float,
        t_total: float,
        p_total: float,
    ) -> IsentropicFlowState:
        """
        Compute static flow parameters from total stagnation conditions (T_0, P_0) and Mach number.
        """
        if mach < 0.0:
            raise ValueError("Mach number must be non-negative.")

        gamma = self.gas.gamma
        r_gas = self.gas.gas_constant

        t_factor = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
        t_static = t_total / t_factor

        p_factor = t_factor ** (gamma / (gamma - 1.0))
        p_static = p_total / p_factor

        rho_static = p_static / (r_gas * t_static)
        speed_of_sound = math.sqrt(gamma * r_gas * t_static)
        velocity = mach * speed_of_sound
        dynamic_p = 0.5 * rho_static * velocity * velocity

        return IsentropicFlowState(
            mach=mach,
            temperature=t_static,
            pressure=p_static,
            density=rho_static,
            velocity=velocity,
            speed_of_sound=speed_of_sound,
            total_temperature=t_total,
            total_pressure=p_total,
            dynamic_pressure=dynamic_p,
        )

    def area_ratio(self, mach: float) -> float:
        """
        Compute isentropic cross-sectional area ratio A / A*(M):
            A / A* = (1 / M) * [ (2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M^2) ] ^ ((gamma + 1) / (2 * (gamma - 1)))
        """
        if mach <= 0.0:
            return float("inf")
        if abs(mach - 1.0) < 1e-12:
            return 1.0

        gamma = self.gas.gamma
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        exponent = gp1 / (2.0 * gm1)

        bracket = (2.0 / gp1) * (1.0 + 0.5 * gm1 * mach * mach)
        return (1.0 / mach) * (bracket**exponent)

    def area_ratio_from_mach(self, mach: float) -> float:
        """Alias for area_ratio(mach)."""
        return self.area_ratio(mach)

    def mach_from_area_ratio(
        self,
        area_ratio: float,
        supersonic: bool = True,
        tolerance: float = 1e-9,
        max_iter: int = 50,
    ) -> float:
        """
        Inverse Area-Mach relation: find M given A / A* using Halley's root-finding method.
        """
        if area_ratio < 1.0:
            raise ValueError(f"Area ratio A/A* cannot be less than 1.0 (got {area_ratio}).")
        if abs(area_ratio - 1.0) < 1e-12:
            return 1.0

        gamma = self.gas.gamma
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        exp_factor = gp1 / (2.0 * gm1)

        # Initial seed
        if supersonic:
            mach = 1.0 + math.sqrt(2.0 / gp1) * math.sqrt(area_ratio - 1.0) if area_ratio < 2.0 else (
                (gp1 / 2.0) ** (exp_factor / 2.0) * area_ratio ** (gm1 / gp1)
            )
            mach = max(1.001, mach)
        else:
            mach = max(0.001, min(0.999, 1.0 / area_ratio))

        # Halley's method for f(M) = area_ratio(M) - target = 0
        for _ in range(max_iter):
            val = self.area_ratio(mach)
            res = val - area_ratio
            if abs(res) < tolerance:
                return mach

            denom_deriv = mach * (1.0 + 0.5 * gm1 * mach * mach)
            f_prime = val * (mach * mach - 1.0) / denom_deriv

            if abs(f_prime) < 1e-12:
                break

            d_denom = 1.0 + 1.5 * gm1 * mach * mach
            f_double_prime = (
                f_prime * (mach * mach - 1.0) / denom_deriv
                + val * (2.0 * mach * denom_deriv - (mach * mach - 1.0) * d_denom) / (denom_deriv * denom_deriv)
            )

            halley_denom = f_prime - 0.5 * res * (f_double_prime / f_prime)
            delta = res / halley_denom if abs(halley_denom) > 1e-14 else res / f_prime

            mach_next = mach - delta

            if supersonic and mach_next <= 1.0:
                mach_next = 1.0 + 0.5 * (mach - 1.0)
            elif not supersonic and (mach_next <= 0.0 or mach_next >= 1.0):
                mach_next = 0.5 * mach

            if abs(mach_next - mach) < tolerance:
                return mach_next
            mach = mach_next

        return mach

    def normal_shock_jump(self, mach_upstream: float) -> Tuple[float, float, float, float, float]:
        """
        Rankine-Hugoniot normal shock wave jump relations for M1 >= 1.0.

        Returns:
            (M2, P2/P1, T2/T1, rho2/rho1, P02/P01)
        """
        if mach_upstream < 1.0:
            raise ValueError(f"Normal shock requires upstream Mach >= 1.0 (got {mach_upstream}).")
        if abs(mach_upstream - 1.0) < 1e-12:
            return 1.0, 1.0, 1.0, 1.0, 1.0

        gamma = self.gas.gamma
        m1_sq = mach_upstream * mach_upstream
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0

        m2_sq = (2.0 + gm1 * m1_sq) / (2.0 * gamma * m1_sq - gm1)
        m2 = math.sqrt(max(0.0, m2_sq))

        p_ratio = 1.0 + (2.0 * gamma / gp1) * (m1_sq - 1.0)
        rho_ratio = (gp1 * m1_sq) / (2.0 + gm1 * m1_sq)
        t_ratio = p_ratio / rho_ratio

        term1 = (gp1 * m1_sq / (2.0 + gm1 * m1_sq)) ** (gamma / gm1)
        term2 = (gp1 / (2.0 * gamma * m1_sq - gm1)) ** (1.0 / gm1)
        p0_ratio = term1 * term2

        return m2, p_ratio, t_ratio, rho_ratio, p0_ratio

    def normal_shock(
        self,
        mach_upstream: float,
        p_upstream: float = 101325.0,
        t_upstream: float = 288.15,
    ) -> NormalShockState:
        """Compute full thermodynamic state downstream of a normal shock wave."""
        m2, p_rat, t_rat, rho_rat, p0_rat = self.normal_shock_jump(mach_upstream)
        # Entropy change delta_s = cp * ln(T2/T1) - R * ln(P2/P1)
        cp = self.gas.cp
        r_gas = self.gas.gas_constant
        delta_s = cp * math.log(t_rat) - r_gas * math.log(p_rat)

        return NormalShockState(
            mach_upstream=mach_upstream,
            mach_downstream=m2,
            pressure_ratio=p_rat,
            temperature_ratio=t_rat,
            density_ratio=rho_rat,
            stagnation_pressure_ratio=p0_rat,
            entropy_change=delta_s,
            p_downstream=p_upstream * p_rat,
            t_downstream=t_upstream * t_rat,
        )

    def oblique_shock_beta(
        self,
        mach_1: float,
        theta: float,
        weak_shock: bool = True,
        strong_shock: Optional[bool] = None,
        max_iter: int = 50,
    ) -> float:
        """
        Solve theta-beta-Mach relation for wave angle beta given upstream Mach M1 and deflection angle theta.
            tan(theta) = 2 * cot(beta) * [ (M1^2 * sin^2(beta) - 1) / (M1^2 * (gamma + cos(2*beta)) + 2) ]
        """
        if strong_shock is not None:
            weak_shock = not strong_shock

        if mach_1 <= 1.0:
            raise ValueError("Oblique shock requires supersonic upstream Mach > 1.0.")
        if theta <= 0.0:
            return math.asin(1.0 / mach_1)

        gamma = self.gas.gamma
        mu = math.asin(1.0 / mach_1)

        def theta_of_beta(b: float) -> float:
            sin_b = math.sin(b)
            cot_b = 1.0 / math.tan(b)
            num = mach_1 * mach_1 * sin_b * sin_b - 1.0
            den = mach_1 * mach_1 * (gamma + math.cos(2.0 * b)) + 2.0
            if den <= 0.0:
                return 0.0
            tan_th = 2.0 * cot_b * (num / den)
            return math.atan(max(0.0, tan_th))

        # Determine maximum detachment angle theta_max by scanning between mu and pi/2
        n_scan = 40
        step = (0.5 * math.pi - mu) / n_scan
        theta_max = 0.0
        beta_max = mu
        for k in range(n_scan + 1):
            b_test = mu + k * step
            th_test = theta_of_beta(b_test)
            if th_test > theta_max:
                theta_max = th_test
                beta_max = b_test

        if theta > theta_max:
            raise ValueError(
                f"Deflection angle {math.degrees(theta):.2f}° exceeds maximum detachment angle "
                f"{math.degrees(theta_max):.2f}° for Mach {mach_1:.2f}."
            )

        # Search interval:
        # Weak shock is between mu and beta_max
        # Strong shock is between beta_max and pi/2
        if weak_shock:
            low = mu
            high = beta_max
        else:
            low = beta_max
            high = 0.5 * math.pi

        b_mid = 0.5 * (low + high)
        for _ in range(max_iter):
            th_curr = theta_of_beta(b_mid)
            if abs(th_curr - theta) < 1e-8:
                return b_mid
            if weak_shock:
                if th_curr < theta:
                    low = b_mid
                else:
                    high = b_mid
            else:
                if th_curr > theta:
                    low = b_mid
                else:
                    high = b_mid
            b_mid = 0.5 * (low + high)

        return b_mid

    def oblique_shock(
        self,
        mach_upstream: float,
        theta: float,
        p_upstream: float = 101325.0,
        t_upstream: float = 288.15,
        weak_shock: bool = True,
    ) -> ObliqueShockState:
        """Compute complete flow state across an oblique shock wave."""
        beta = self.oblique_shock_beta(mach_upstream, theta, weak_shock=weak_shock)
        # Normal component of upstream Mach: M_n1 = M1 * sin(beta)
        m_n1 = mach_upstream * math.sin(beta)
        m_n2, p_rat, t_rat, rho_rat, p0_rat = self.normal_shock_jump(m_n1)

        # Downstream total Mach: M2 = M_n2 / sin(beta - theta)
        sin_beta_minus_theta = math.sin(beta - theta)
        m2 = m_n2 / sin_beta_minus_theta if sin_beta_minus_theta > 1e-6 else m_n2

        return ObliqueShockState(
            mach_upstream=mach_upstream,
            deflection_angle=theta,
            wave_angle=beta,
            mach_downstream=m2,
            pressure_ratio=p_rat,
            temperature_ratio=t_rat,
            density_ratio=rho_rat,
            stagnation_pressure_ratio=p0_rat,
            p_downstream=p_upstream * p_rat,
            t_downstream=t_upstream * t_rat,
        )

    def prandtl_meyer_nu(self, mach: float) -> float:
        """
        Prandtl-Meyer supersonic expansion fan function nu(M) in radians:
            nu(M) = sqrt((gamma + 1)/(gamma - 1)) * arctan(sqrt(((gamma - 1)/(gamma + 1))*(M^2 - 1))) - arctan(sqrt(M^2 - 1))
        """
        if mach <= 1.0:
            return 0.0

        gamma = self.gas.gamma
        gp1 = gamma + 1.0
        gm1 = gamma - 1.0
        coeff = math.sqrt(gp1 / gm1)

        m_term = math.sqrt(mach * mach - 1.0)
        angle1 = math.atan(math.sqrt(gm1 / gp1) * m_term)
        angle2 = math.atan(m_term)

        return coeff * angle1 - angle2

    def mach_from_prandtl_meyer(
        self,
        nu_radians: float,
        tolerance: float = 1e-9,
        max_iter: int = 50,
    ) -> float:
        """
        Invert Prandtl-Meyer function: find Mach number M given expansion angle nu in radians.
        """
        if nu_radians <= 0.0:
            return 1.0

        gamma = self.gas.gamma
        nu_inf = 0.5 * math.pi * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
        if nu_radians >= nu_inf:
            return float("inf")

        mach = 1.0 + (nu_radians / 0.5) ** (2.0 / 3.0)

        for _ in range(max_iter):
            val = self.prandtl_meyer_nu(mach)
            res = val - nu_radians
            if abs(res) < tolerance:
                return mach

            denom = mach * (1.0 + 0.5 * (gamma - 1.0) * mach * mach)
            dnu_dm = math.sqrt(mach * mach - 1.0) / denom if mach > 1.0 else 1e-6
            mach_next = mach - res / dnu_dm
            if mach_next <= 1.0:
                mach_next = 1.0 + 0.5 * (mach - 1.0)
            if abs(mach_next - mach) < tolerance:
                return mach_next
            mach = mach_next

        return mach

    def mach_angle(self, mach: float) -> float:
        """Mach wave angle mu = arcsin(1 / M) in radians."""
        if mach < 1.0:
            raise ValueError(f"Mach angle defined only for supersonic flow M >= 1.0 (got {mach}).")
        return math.asin(1.0 / mach)
