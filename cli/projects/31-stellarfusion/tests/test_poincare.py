"""
Unit tests for Poincaré Surface-of-Section & Field Line Tracer.
"""

import math
import unittest

from stellarfusion.equilibrium import SolovevEquilibrium
from stellarfusion.magnetic import MagneticFieldEvaluator
from stellarfusion.poincare import (
    PoincareFieldTracer,
    PoincarePuncture,
    ResonantMagneticPerturbation,
)


class TestPoincareFieldTracer(unittest.TestCase):
    def setUp(self) -> None:
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.5, psi_0=1.0, b_0=2.5)
        self.evaluator = MagneticFieldEvaluator(self.solovev)
        self.tracer = PoincareFieldTracer(self.evaluator)

    def test_field_line_derivatives(self) -> None:
        dr, dz = self.tracer.field_line_derivatives(r=3.2, z=0.1, phi=0.0)
        self.assertIsInstance(dr, float)
        self.assertIsInstance(dz, float)

    def test_field_line_flux_conservation(self) -> None:
        # Along an unperturbed magnetic field line, B . grad(psi) = 0, so psi is conserved
        punctures = self.tracer.trace_puncture_series(
            r_start=3.3,
            z_start=0.0,
            num_turns=10,
            steps_per_turn=32,
        )
        self.assertEqual(len(punctures), 11)

        psi_initial = punctures[0].psi_value
        for p in punctures[1:]:
            # Numerical integration maintains constant flux to high precision
            rel_error = abs(p.psi_value - psi_initial) / max(1e-6, abs(psi_initial))
            self.assertLess(rel_error, 0.02)

    def test_multisurface_map_generation(self) -> None:
        seed_radii = [3.2, 3.4, 3.6]
        surfaces = self.tracer.generate_multisurface_map(
            seed_radii=seed_radii,
            z_seed=0.0,
            num_turns=5,
            steps_per_turn=16,
        )
        self.assertEqual(len(surfaces), 3)
        for s in surfaces:
            self.assertEqual(len(s), 6)  # Turn 0 + 5 turns


class TestResonantMagneticPerturbation(unittest.TestCase):
    def test_rmp_evaluation(self) -> None:
        rmp = ResonantMagneticPerturbation(m_mode=2, n_mode=1, amplitude=1e-3, r_axis=3.0, z_axis=0.0)
        d_br, d_bz = rmp.evaluate_perturbation(3.3, 0.2, 0.0)
        self.assertIsInstance(d_br, float)
        self.assertIsInstance(d_bz, float)


if __name__ == "__main__":
    unittest.main()
