"""
Unit tests for Universal Variable Kepler Propagation & Lambert's Problem Solver.
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
    orbital_elements_to_state,
    state_to_orbital_elements,
    propagate_keplerian,
)
from orbitmech.lambert import (
    stumpff_c2,
    stumpff_c3,
    propagate_universal,
    solve_lambert,
)


class TestLambert(unittest.TestCase):
    def test_stumpff_limits(self):
        # Near zero Taylor expansions
        self.assertAlmostEqual(stumpff_c2(0.0), 0.5, places=10)
        self.assertAlmostEqual(stumpff_c3(0.0), 1.0 / 6.0, places=10)
        self.assertAlmostEqual(stumpff_c2(1e-7), 0.5, places=8)
        self.assertAlmostEqual(stumpff_c3(1e-7), 1.0 / 6.0, places=8)

        # Known values: z = pi^2 -> c2(pi^2) = (1 - cos(pi)) / pi^2 = 2 / pi^2
        self.assertAlmostEqual(stumpff_c2(math.pi ** 2), 2.0 / (math.pi ** 2), places=7)

    def test_universal_propagation_matches_kepler_analytic(self):
        # Elliptic orbit: compare universal variable propagation against classical Kepler equation
        initial_elements = ClassicalOrbitalElements(
            a=8000.0,
            e=0.25,
            i=math.radians(30.0),
            raan=math.radians(60.0),
            arg_peri=math.radians(45.0),
            true_anomaly=math.radians(15.0),
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(initial_elements)
        dt = 1800.0  # 30 minutes

        state_analytic = propagate_keplerian(state_0, dt=dt, mu=EARTH.mu)
        state_universal = propagate_universal(state_0, dt=dt, mu=EARTH.mu)

        pos_diff = (state_analytic.r - state_universal.r).norm()
        vel_diff = (state_analytic.v - state_universal.v).norm()

        self.assertLess(pos_diff, 1e-4)  # Less than 0.1 mm position error
        self.assertLess(vel_diff, 1e-7)  # Less than 0.1 um/s velocity error

    def test_lambert_targeting_solution(self):
        # Generate two position vectors along a known orbit
        elements = ClassicalOrbitalElements(
            a=7500.0,
            e=0.15,
            i=math.radians(20.0),
            raan=math.radians(10.0),
            arg_peri=math.radians(40.0),
            true_anomaly=math.radians(0.0),
            mu=EARTH.mu,
        )
        state_1 = orbital_elements_to_state(elements)
        dt = 1200.0  # 20 minutes

        state_2 = propagate_universal(state_1, dt=dt, mu=EARTH.mu)

        # Solve Lambert problem from r1 to r2 given dt
        sol = solve_lambert(
            r1_vec=state_1.r,
            r2_vec=state_2.r,
            time_of_flight=dt,
            mu=EARTH.mu,
            prograde=True,
            short_way=True,
        )

        # Departure velocity v1 must match true orbital velocity at state_1
        v1_diff = (sol.v1 - state_1.v).norm()
        v2_diff = (sol.v2 - state_2.v).norm()

        self.assertLess(v1_diff, 1e-3)
        self.assertLess(v2_diff, 1e-3)
        self.assertAlmostEqual(sol.semi_major_axis, elements.a, delta=5.0)

    def test_universal_hyperbolic_propagation(self):
        # Hyperbolic escape orbit (speed > escape speed sqrt(2*mu/r))
        r0 = Vector3(7000.0, 0.0, 0.0)
        v_esc = math.sqrt(2.0 * EARTH.mu / 7000.0)
        # 2 km/s excess speed
        v0 = Vector3(0.0, v_esc + 2.0, 0.0)
        state_hyp = StateVector(r=r0, v=v0)

        # Propagate 2 hours into deep space
        dt = 7200.0
        state_future = propagate_universal(state_hyp, dt=dt, mu=EARTH.mu)

        # Energy must remain conserved
        e0 = 0.5 * v0.norm_squared() - EARTH.mu / r0.norm()
        e_future = 0.5 * state_future.v.norm_squared() - EARTH.mu / state_future.r.norm()
        self.assertGreater(state_future.r.norm(), 7000.0)
        self.assertAlmostEqual(e0, e_future, places=6)


if __name__ == "__main__":
    unittest.main()
