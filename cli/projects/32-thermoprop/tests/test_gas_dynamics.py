"""
Unit tests for Compressible Gas Dynamics Engine (thermoprop/gas_dynamics.py).
"""

from __future__ import annotations

import math
import unittest

from thermoprop.gas_dynamics import (
    CompressibleFlowEngine,
    GasProperties,
)


class TestGasDynamics(unittest.TestCase):
    """Test suite for 1D/2D compressible flow relations, shocks, and expansion."""

    def setUp(self) -> None:
        self.air = GasProperties.air()
        self.flow_air = CompressibleFlowEngine(self.air)
        self.methalox = GasProperties.methalox()
        self.flow_methalox = CompressibleFlowEngine(self.methalox)

    def test_gas_properties(self) -> None:
        """Verify specific gas constant, heat capacities, and sound speed."""
        # Air: R = 8314.4626 / 28.9647 ~ 287.05 J/(kg*K)
        self.assertAlmostEqual(self.air.gas_constant, 287.05, delta=0.5)
        # cp - cv = R
        self.assertAlmostEqual(self.air.cp - self.air.cv, self.air.gas_constant, delta=1e-5)
        # gamma = cp / cv = 1.4
        self.assertAlmostEqual(self.air.cp / self.air.cv, 1.4, places=4)
        # Speed of sound at 300 K: sqrt(1.4 * 287.05 * 300) ~ 347.2 m/s
        a_300 = self.air.speed_of_sound(300.0)
        self.assertAlmostEqual(a_300, 347.2, delta=0.5)

    def test_isentropic_relations_at_mach_zero_and_one(self) -> None:
        """Verify stagnation ratios at M=0 (ambient) and M=1 (choked sonic throat)."""
        # M = 0: T/T0 = 1, P/P0 = 1, rho/rho0 = 1
        t_ratio_0 = self.flow_air.temperature_ratio(0.0)
        p_ratio_0 = self.flow_air.pressure_ratio(0.0)
        rho_ratio_0 = self.flow_air.density_ratio(0.0)
        self.assertAlmostEqual(t_ratio_0, 1.0, places=6)
        self.assertAlmostEqual(p_ratio_0, 1.0, places=6)
        self.assertAlmostEqual(rho_ratio_0, 1.0, places=6)

        # M = 1 for gamma = 1.4:
        # T* / T0 = 2 / (gamma + 1) = 2 / 2.4 = 0.83333
        # P* / P0 = (2 / 2.4) ^ (1.4 / 0.4) = (0.83333) ^ 3.5 = 0.52828
        t_ratio_1 = self.flow_air.temperature_ratio(1.0)
        p_ratio_1 = self.flow_air.pressure_ratio(1.0)
        self.assertAlmostEqual(t_ratio_1, 2.0 / 2.4, places=5)
        self.assertAlmostEqual(p_ratio_1, 0.52828, delta=1e-4)

    def test_area_mach_relation(self) -> None:
        """Verify area ratio A/A* at M=1 and root finding for both subsonic & supersonic branches."""
        # At M = 1, A/A* == 1.0
        ar_1 = self.flow_air.area_ratio_from_mach(1.0)
        self.assertAlmostEqual(ar_1, 1.0, places=6)

        # Area ratio for M = 2.5
        target_ar = self.flow_air.area_ratio_from_mach(2.5)
        self.assertGreater(target_ar, 2.0)

        # Solve for supersonic Mach from area ratio
        m_supersonic = self.flow_air.mach_from_area_ratio(target_ar, supersonic=True)
        self.assertAlmostEqual(m_supersonic, 2.5, places=5)

        # Solve for subsonic Mach from area ratio
        m_subsonic = self.flow_air.mach_from_area_ratio(target_ar, supersonic=False)
        self.assertLess(m_subsonic, 1.0)
        ar_sub = self.flow_air.area_ratio_from_mach(m_subsonic)
        self.assertAlmostEqual(ar_sub, target_ar, places=5)

        with self.assertRaises(ValueError):
            self.flow_air.mach_from_area_ratio(0.8)

    def test_normal_shock_wave(self) -> None:
        """Verify Rankine-Hugoniot normal shock relations."""
        # For M1 = 2.0, gamma = 1.4:
        # M2 = sqrt((1 + 0.2*4) / (1.4*4 - 0.2)) = sqrt(1.8 / 5.4) = sqrt(1/3) ~ 0.57735
        # P2 / P1 = 1 + (2*1.4/2.4)*(4 - 1) = 1 + (2.8/2.4)*3 = 1 + 3.5 = 4.5
        # T2 / T1 = 1.6875
        shock = self.flow_air.normal_shock(mach_upstream=2.0, p_upstream=100000.0, t_upstream=300.0)
        self.assertAlmostEqual(shock.mach_downstream, 1.0 / math.sqrt(3.0), places=4)
        self.assertAlmostEqual(shock.pressure_ratio, 4.5, places=4)
        self.assertAlmostEqual(shock.temperature_ratio, 1.6875, places=3)
        self.assertLess(shock.stagnation_pressure_ratio, 1.0)  # Total pressure loss / entropy rise
        self.assertGreater(shock.entropy_change, 0.0)

        # Normal shock requires M1 >= 1.0
        with self.assertRaises(ValueError):
            self.flow_air.normal_shock(mach_upstream=0.8, p_upstream=100000.0, t_upstream=300.0)

    def test_oblique_shock_and_theta_beta_mach(self) -> None:
        """Verify oblique shock wave angle and deflection relations."""
        mach = 3.0
        theta_rad = math.radians(15.0)

        # Solve for weak oblique shock wave angle beta
        beta = self.flow_air.oblique_shock_beta(mach, theta_rad, strong_shock=False)
        self.assertGreater(beta, self.flow_air.mach_angle(mach))
        self.assertLess(beta, math.pi / 2.0)

        oblique = self.flow_air.oblique_shock(mach, theta_rad, p_upstream=101325.0, t_upstream=288.15)
        self.assertGreater(oblique.mach_downstream, 1.0)  # Weak shock remains supersonic for small theta
        self.assertGreater(oblique.pressure_ratio, 1.0)

        # Deflection exceeding maximum shock angle raises ValueError
        with self.assertRaises(ValueError):
            self.flow_air.oblique_shock_beta(1.5, math.radians(45.0))

    def test_strong_shock_solution(self) -> None:
        """Verify strong shock solution has steeper wave angle beta and subsonic downstream Mach."""
        mach = 3.0
        theta_rad = math.radians(15.0)

        beta_weak = self.flow_air.oblique_shock_beta(mach, theta_rad, weak_shock=True)
        beta_strong = self.flow_air.oblique_shock_beta(mach, theta_rad, weak_shock=False)

        # Strong shock wave angle is steeper
        self.assertGreater(beta_strong, beta_weak)
        self.assertLessEqual(beta_strong, math.pi / 2.0)

        oblique_strong = self.flow_air.oblique_shock(mach, theta_rad, weak_shock=False)
        # Strong oblique shock creates subsonic downstream flow
        self.assertLess(oblique_strong.mach_downstream, 1.0)
        # Strong shock creates higher pressure ratio than weak shock
        oblique_weak = self.flow_air.oblique_shock(mach, theta_rad, weak_shock=True)
        self.assertGreater(oblique_strong.pressure_ratio, oblique_weak.pressure_ratio)

    def test_hypersonic_expansion_and_mach_angle(self) -> None:
        """Verify high Mach number behavior and Mach angle limits."""
        mu_5 = self.flow_air.mach_angle(5.0)
        # mu = arcsin(0.2) ~ 0.20136 rad ~ 11.54 deg
        self.assertAlmostEqual(math.degrees(mu_5), 11.54, delta=0.05)

        # Mach angle raises error for subsonic
        with self.assertRaises(ValueError):
            self.flow_air.mach_angle(0.9)

        # Isentropic state at hypersonic M=6
        state_6 = self.flow_air.isentropic_state_from_stagnation(6.0, t_total=2000.0, p_total=50.0e5)
        self.assertGreater(state_6.velocity, 1500.0)
        self.assertLess(state_6.temperature, 300.0)

    def test_prandtl_meyer_expansion(self) -> None:
        """Verify Prandtl-Meyer expansion angle and inversion."""
        # At M = 1, nu = 0
        nu_1 = self.flow_air.prandtl_meyer_nu(1.0)
        self.assertAlmostEqual(nu_1, 0.0, places=6)

        # For M = 2.0, nu ~ 26.38 deg = 0.4604 rad
        nu_2 = self.flow_air.prandtl_meyer_nu(2.0)
        self.assertAlmostEqual(math.degrees(nu_2), 26.38, delta=0.1)

        # Inversion: M from nu
        m_recov = self.flow_air.mach_from_prandtl_meyer(nu_2)
        self.assertAlmostEqual(m_recov, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
