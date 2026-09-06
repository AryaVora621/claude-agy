"""
Unit tests for OrbitMech Numerical Propagators & Perturbation Modeling.
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
)
from orbitmech.propagator import (
    PerturbationConfig,
    SpacecraftProperties,
    compute_acceleration,
    propagate_symplectic_verlet,
    propagate_rk45,
    atmospheric_density_earth,
)


class TestPropagator(unittest.TestCase):
    def test_symplectic_verlet_energy_conservation(self):
        # Propagate unperturbed LEO orbit for 3 full revolutions
        elements = ClassicalOrbitalElements(
            a=6800.0,
            e=0.05,
            i=math.radians(28.5),
            raan=0.0,
            arg_peri=0.0,
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(elements)
        period = elements.period
        duration = 3.0 * period

        # Symplectic integrator with 10 second time step
        res = propagate_symplectic_verlet(
            initial_state=state_0,
            duration=duration,
            dt=10.0,
            config=PerturbationConfig(enable_j2=False, enable_drag=False),
        )

        # Max energy deviation across 3 full orbits must be bounded
        initial_e = res.energies[0]
        max_drift = res.max_energy_drift
        relative_drift = max_drift / abs(initial_e)

        self.assertLess(relative_drift, 1e-5)
        self.assertGreater(len(res.trajectory), 100)

    def test_j2_nodal_precession_rate(self):
        # 45-degree inclined circular orbit
        r_orbit = 7000.0
        inc_deg = 45.0
        elements = ClassicalOrbitalElements(
            a=r_orbit,
            e=0.001,
            i=math.radians(inc_deg),
            raan=math.radians(50.0),
            arg_peri=0.0,
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(elements)

        # Theoretical secular nodal precession rate:
        # dOmega/dt = -1.5 * J2 * (R/p)^2 * n * cos(i)
        p = elements.semi_latus_rectum
        n = elements.mean_motion
        d_omega_dt_theory = (
            -1.5
            * EARTH.j2
            * ((EARTH.radius / p) ** 2)
            * n
            * math.cos(elements.i)
        )

        # Propagate for 6 hours
        sim_duration = 6.0 * 3600.0
        expected_delta_raan = d_omega_dt_theory * sim_duration

        config = PerturbationConfig(enable_j2=True, enable_drag=False)
        res = propagate_rk45(
            initial_state=state_0,
            duration=sim_duration,
            initial_dt=30.0,
            tolerance=1e-8,
            config=config,
        )

        final_elements = state_to_orbital_elements(res.final_state, mu=EARTH.mu)
        measured_delta_raan = final_elements.raan - elements.raan

        # Nodal precession should be negative (westward regression of nodes)
        self.assertLess(measured_delta_raan, 0.0)
        # Should match theoretical secular rate to within 2.5%
        self.assertAlmostEqual(measured_delta_raan, expected_delta_raan, delta=abs(expected_delta_raan) * 0.025)

    def test_atmospheric_drag_orbital_decay(self):
        # Very low LEO circular orbit at 220 km altitude experiencing heavy drag
        r_orbit = EARTH.radius + 220.0
        elements = ClassicalOrbitalElements(
            a=r_orbit,
            e=0.0005,
            i=math.radians(45.0),
            raan=0.0,
            arg_peri=0.0,
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(elements)

        config = PerturbationConfig(
            enable_j2=False,
            enable_drag=True,
            spacecraft=SpacecraftProperties(mass=500.0, cross_section=25.0, drag_coefficient=2.2),
        )

        # Propagate for 2 orbits (~3 hours)
        duration = 2.0 * elements.period
        res = propagate_rk45(
            initial_state=state_0,
            duration=duration,
            initial_dt=15.0,
            config=config,
        )

        final_elements = state_to_orbital_elements(res.final_state, mu=EARTH.mu)
        # Drag must extract energy, decreasing semi-major axis
        self.assertLess(final_elements.a, elements.a)
        # Final energy must be lower than initial energy (more negative)
        self.assertLess(res.energies[-1], res.energies[0])

    def test_rk45_adaptive_stepping_on_eccentric_orbit(self):
        # Molniya-like eccentric orbit (e = 0.70)
        elements = ClassicalOrbitalElements(
            a=26600.0,
            e=0.70,
            i=math.radians(63.4),  # Critical inclination
            raan=0.0,
            arg_peri=math.radians(270.0),
            true_anomaly=0.0,
            mu=EARTH.mu,
        )
        state_0 = orbital_elements_to_state(elements)

        # Propagate for 1 period
        res = propagate_rk45(
            initial_state=state_0,
            duration=elements.period,
            initial_dt=60.0,
            tolerance=1e-8,
            config=PerturbationConfig(enable_j2=False),
        )

        # Energy conservation in RK45 for high eccentricity
        rel_energy_err = res.max_energy_drift / abs(res.energies[0])
        self.assertLess(rel_energy_err, 1e-4)


if __name__ == "__main__":
    unittest.main()
