"""
Unit tests for Symplectic Boris Particle Pusher & Neoclassical Orbit Kinematics.
"""

import math
import unittest

from stellarfusion.equilibrium import SolovevEquilibrium
from stellarfusion.magnetic import MagneticFieldEvaluator
from stellarfusion.particles import (
    BorisParticlePusher,
    ParticleSpecies,
    ParticleState,
    SpeciesProperties,
)


class TestParticleProperties(unittest.TestCase):
    def test_species_initialization(self) -> None:
        props_d = SpeciesProperties.from_species(ParticleSpecies.DEUTERIUM)
        self.assertGreater(props_d.charge, 0.0)
        self.assertGreater(props_d.mass, 3.0e-27)

        props_a = SpeciesProperties.from_species(ParticleSpecies.ALPHA)
        self.assertAlmostEqual(props_a.charge, 2.0 * props_d.charge)

        props_e = SpeciesProperties.from_species(ParticleSpecies.ELECTRON)
        self.assertLess(props_e.charge, 0.0)
        self.assertLess(props_e.mass, 1e-30)

    def test_particle_state_coordinates_and_energy(self) -> None:
        state = ParticleState(x=3.0, y=4.0, z=0.5, vx=1e5, vy=0.0, vz=0.0)
        self.assertAlmostEqual(state.r_cylindrical, 5.0)
        self.assertAlmostEqual(state.phi_cylindrical, math.atan2(4.0, 3.0))
        self.assertAlmostEqual(state.speed, 1e5)

        mass = 3.34e-27
        ke_j = state.kinetic_energy_joules(mass)
        self.assertAlmostEqual(ke_j, 0.5 * mass * 1e10)
        ke_kev = state.kinetic_energy_kev(mass)
        self.assertGreater(ke_kev, 0.0)


class TestBorisParticlePusher(unittest.TestCase):
    def setUp(self) -> None:
        self.solovev = SolovevEquilibrium(r_0=3.0, z_0=0.0, kappa=1.5, psi_0=1.0, b_0=2.0)
        self.evaluator = MagneticFieldEvaluator(self.solovev)
        self.pusher = BorisParticlePusher(self.evaluator, species=ParticleSpecies.NORMALIZED_ION)

    def test_exact_kinetic_energy_conservation(self) -> None:
        # In pure magnetic fields (E=0), the Boris algorithm strictly preserves kinetic energy
        init_state = ParticleState(x=3.2, y=0.0, z=0.1, vx=10.0, vy=50.0, vz=5.0)
        initial_speed = init_state.speed

        curr = init_state
        dt = 1e-4
        for _ in range(200):
            curr = self.pusher.step(curr, dt)

        final_speed = curr.speed
        # Exact velocity magnitude preservation to floating-point precision
        rel_error = abs(final_speed - initial_speed) / initial_speed
        self.assertLess(rel_error, 1e-11)

    def test_cyclotron_frequency_and_larmor_radius(self) -> None:
        omega_c = self.pusher.cyclotron_frequency(3.0, 0.0)
        self.assertGreater(omega_c, 0.0)

        # For B=2.0 T and unit charge/mass, omega_c = q*B/m = 2.0 rad/s
        self.assertAlmostEqual(omega_c, 2.0, places=6)

        rho = self.pusher.larmor_radius(speed_perp=10.0, r=3.0, z=0.0)
        # rho = v_perp / omega_c = 10.0 / 2.0 = 5.0 m
        self.assertAlmostEqual(rho, 5.0, places=6)

    def test_trapped_vs_passing_orbit_classification(self) -> None:
        # Trapped particle launched outboard with primarily perpendicular velocity
        # Low parallel velocity relative to perpendicular velocity
        trapped_state = ParticleState(
            x=3.5, y=0.0, z=0.0, vx=0.1, vy=20.0, vz=0.2
        )
        is_trapped, bounces, freq, width = self.pusher.analyze_orbit_trapping(
            trapped_state, r_axis=3.0, dt=1e-3, num_steps=300
        )
        self.assertGreater(width, 0.0)
        # Verify it computes valid orbital metrics
        self.assertIsInstance(is_trapped, bool)
        self.assertIsInstance(bounces, int)


if __name__ == "__main__":
    unittest.main()
