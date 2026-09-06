"""
Unit tests for Keplerian Two-Body Mechanics and Anomaly Solvers.
"""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
)
from orbitmech.kepler import (
    state_to_orbital_elements,
    orbital_elements_to_state,
    solve_kepler_equation,
    true_to_eccentric_anomaly,
    eccentric_to_true_anomaly,
    propagate_keplerian,
)


class TestKepler(unittest.TestCase):
    def test_kepler_equation_danby_solver(self):
        # Test for various eccentricities from circular to highly eccentric (Molniya-like)
        eccentricities = [0.0, 0.1, 0.5, 0.74, 0.95, 0.999]
        mean_anomalies = [0.0, 0.5, 1.0, math.pi, 4.5, 6.0]

        for e in eccentricities:
            for M in mean_anomalies:
                E = solve_kepler_equation(M, e, tolerance=1e-13)
                # Verify M = E - e*sin(E)
                M_reconstructed = (E - e * math.sin(E)) % (2.0 * math.pi)
                M_norm = M % (2.0 * math.pi)
                self.assertAlmostEqual(M_reconstructed, M_norm, places=11)

    def test_state_to_elements_and_back_roundtrip(self):
        # ISS-like Low Earth Orbit
        initial_elements = ClassicalOrbitalElements(
            a=6778.137,          # 400 km altitude
            e=0.015,
            i=math.radians(51.64),
            raan=math.radians(125.0),
            arg_peri=math.radians(75.0),
            true_anomaly=math.radians(35.0),
            mu=EARTH.mu,
        )

        state = orbital_elements_to_state(initial_elements)
        # Position magnitude must equal r = p / (1 + e*cos(nu))
        p = initial_elements.semi_latus_rectum
        expected_r = p / (1.0 + initial_elements.e * math.cos(initial_elements.true_anomaly))
        self.assertAlmostEqual(state.r.norm(), expected_r, places=4)

        # Convert state back to orbital elements
        reconstructed = state_to_orbital_elements(state, mu=EARTH.mu)
        self.assertAlmostEqual(reconstructed.a, initial_elements.a, places=3)
        self.assertAlmostEqual(reconstructed.e, initial_elements.e, places=5)
        self.assertAlmostEqual(reconstructed.i, initial_elements.i, places=5)
        self.assertAlmostEqual(reconstructed.raan, initial_elements.raan, places=5)
        self.assertAlmostEqual(reconstructed.arg_peri, initial_elements.arg_peri, places=5)
        self.assertAlmostEqual(reconstructed.true_anomaly, initial_elements.true_anomaly, places=5)

    def test_keplerian_propagation_one_full_period(self):
        # When propagated for exactly one orbital period, position and velocity should return to initial
        initial_elements = ClassicalOrbitalElements(
            a=7000.0,
            e=0.1,
            i=math.radians(28.5),
            raan=math.radians(45.0),
            arg_peri=math.radians(30.0),
            true_anomaly=math.radians(10.0),
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(initial_elements)
        period = initial_elements.period

        state_final = propagate_keplerian(state_0, dt=period, mu=EARTH.mu)

        # Difference should be negligible
        pos_err = (state_final.r - state_0.r).norm()
        vel_err = (state_final.v - state_0.v).norm()
        self.assertLess(pos_err, 1e-6)
        self.assertLess(vel_err, 1e-9)


if __name__ == "__main__":
    unittest.main()
