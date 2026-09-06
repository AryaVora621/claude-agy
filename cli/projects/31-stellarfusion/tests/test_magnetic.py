"""
Unit tests for Magnetic Field Topology & Safety Factor Engine.
"""

import math
import unittest

from stellarfusion.equilibrium import (
    EquilibriumProfile,
    GradShafranovSolver,
    Grid2D,
    SolovevEquilibrium,
)
from stellarfusion.magnetic import (
    MagneticField,
    MagneticFieldEvaluator,
    SafetyFactorCalculator,
)


class TestMagneticField(unittest.TestCase):
    def test_field_properties_and_magnitude(self) -> None:
        field = MagneticField(b_r=3.0, b_phi=4.0, b_z=0.0)
        self.assertAlmostEqual(field.magnitude, 5.0)
        self.assertAlmostEqual(field.poloidal_magnitude, 3.0)
        self.assertAlmostEqual(field.pitch_angle, math.atan2(3.0, 4.0))

    def test_cartesian_transformation(self) -> None:
        field = MagneticField(b_r=2.0, b_phi=3.0, b_z=1.0)
        # At phi = 0: bx = b_r, by = b_phi, bz = b_z
        bx, by, bz = field.to_cartesian(0.0)
        self.assertAlmostEqual(bx, 2.0)
        self.assertAlmostEqual(by, 3.0)
        self.assertAlmostEqual(bz, 1.0)

        # At phi = pi / 2: bx = -b_phi, by = b_r
        bx_pi2, by_pi2, bz_pi2 = field.to_cartesian(0.5 * math.pi)
        self.assertAlmostEqual(bx_pi2, -3.0)
        self.assertAlmostEqual(by_pi2, 2.0)
        self.assertAlmostEqual(bz_pi2, 1.0)


class TestMagneticFieldEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.5, psi_0=1.0, b_0=2.0)
        self.evaluator = MagneticFieldEvaluator(self.solovev)

    def test_divergence_free_condition(self) -> None:
        # div(B) must vanish identically in cylindrical coordinates
        sample_points = [(2.5, 0.3), (3.0, 0.8), (3.5, -0.4), (2.8, -0.6)]
        for r, z in sample_points:
            div_b = self.evaluator.verify_divergence_free(r, z, h=1e-4)
            self.assertLess(div_b, 1e-4)

    def test_numerical_evaluator_consistency(self) -> None:
        grid = Grid2D(nr=31, nz=31, r_min=1.5, r_max=4.5, z_min=-1.5, z_max=1.5)
        solver = GradShafranovSolver(grid=grid)
        solver.initialize_with_solovev(self.solovev)

        profile = EquilibriumProfile(
            f_0=self.solovev.r_0 * self.solovev.b_0,
            diamagnetic_factor=0.0,
        )
        num_evaluator = MagneticFieldEvaluator(solver, profile=profile)
        b_exact = self.evaluator.evaluate_at(3.4, 0.3)
        b_num = num_evaluator.evaluate_at(3.4, 0.3)

        # Numerical gradient should match analytical to within ~2%
        self.assertAlmostEqual(b_num.b_phi, b_exact.b_phi, delta=0.1)
        self.assertAlmostEqual(b_num.b_r, b_exact.b_r, delta=0.05)
        self.assertAlmostEqual(b_num.b_z, b_exact.b_z, delta=0.05)


class TestSafetyFactorCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.5, psi_0=1.0, b_0=2.5)
        self.evaluator = MagneticFieldEvaluator(self.solovev)
        self.q_calc = SafetyFactorCalculator(
            self.evaluator,
            r_axis=3.0,
            z_axis=0.0,
            minor_radius_a=0.8,
            kappa=1.5,
        )

    def test_safety_factor_positive_and_monotonic(self) -> None:
        q_core = self.q_calc.compute_q_at_minor_radius(0.1, num_samples=64)
        q_edge = self.q_calc.compute_q_at_minor_radius(0.6, num_samples=64)

        self.assertGreater(q_core, 0.0)
        self.assertGreater(q_edge, 0.0)
        # Tokamak safety factor normally increases outward
        self.assertGreaterEqual(q_edge, q_core)

    def test_profile_computation_and_shear(self) -> None:
        profile = self.q_calc.compute_profile(num_points=5)
        self.assertEqual(len(profile), 5)
        for rho, q_val, shear in profile:
            self.assertGreater(rho, 0.0)
            self.assertLessEqual(rho, 1.0)
            self.assertGreater(q_val, 0.0)


if __name__ == "__main__":
    unittest.main()
