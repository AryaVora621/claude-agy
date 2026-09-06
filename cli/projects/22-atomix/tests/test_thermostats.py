"""
Unit tests for Atomix Thermostats, Barostats, and Velocity Distributions.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Atom, SimulationBox
from atomix.thermostats import (
    initialize_maxwell_boltzmann_velocities,
    BerendsenThermostat,
    NoseHooverThermostat,
    AndersenThermostat,
    BerendsenBarostat,
)
from atomix.observables import compute_temperature, compute_kinetic_energy


class TestThermostatsAndBarostats(unittest.TestCase):
    def test_maxwell_boltzmann_initialization(self):
        n_atoms = 150
        atoms = [Atom(i, "Ar", "Ar", Vector3D(float(i), 0.0, 0.0), mass=1.0) for i in range(n_atoms)]
        target_t = 1.8

        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=target_t, remove_drift=True)

        # Net linear momentum must be zero
        net_px = sum(a.mass * a.velocity.x for a in atoms)
        net_py = sum(a.mass * a.velocity.y for a in atoms)
        net_pz = sum(a.mass * a.velocity.z for a in atoms)
        self.assertAlmostEqual(net_px, 0.0, places=6)
        self.assertAlmostEqual(net_py, 0.0, places=6)
        self.assertAlmostEqual(net_pz, 0.0, places=6)

        # Kinetic temperature must match target_t exactly
        measured_t = compute_temperature(atoms, remove_drift=True)
        self.assertAlmostEqual(measured_t, target_t, places=5)

    def test_berendsen_thermostat_relaxation(self):
        atoms = [Atom(i, "Ar", "Ar", Vector3D(float(i), 0.0, 0.0), mass=1.0) for i in range(50)]
        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=1.0)

        # Heat from T=1.0 towards T=2.5
        thermo = BerendsenThermostat(target_temperature=2.5, tau_t=0.05)
        dt = 0.005

        curr_t = compute_temperature(atoms)
        for _ in range(30):
            thermo.apply(atoms, dt, curr_t)
            curr_t = compute_temperature(atoms)

        # Temperature must approach 2.5
        self.assertGreater(curr_t, 2.0)
        self.assertLessEqual(curr_t, 2.6)

    def test_nose_hoover_thermostat_dynamics(self):
        atoms = [Atom(i, "Ar", "Ar", Vector3D(float(i), 0.0, 0.0), mass=1.0) for i in range(50)]
        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.5)

        # Higher target temperature -> friction variable xi should become negative to accelerate atoms
        thermo = NoseHooverThermostat(target_temperature=2.0, tau_nh=0.1)
        dt = 0.005

        for _ in range(10):
            ke = compute_kinetic_energy(atoms)
            thermo.step(atoms, dt, ke)

        # Friction coefficient xi drives heating
        self.assertLess(thermo.xi, 0.0)

    def test_berendsen_barostat_volume_scaling(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        atoms = [Atom(0, "A", "Ar", Vector3D(2.0, 2.0, 2.0)), Atom(1, "B", "Ar", Vector3D(8.0, 8.0, 8.0))]

        barostat = BerendsenBarostat(target_pressure=1.0, tau_p=0.5, compressibility=0.01)

        # Current pressure is higher than target pressure (P_curr = 5.0 > P_target = 1.0)
        # Barostat should expand the box (mu > 1) to relieve pressure
        mu = barostat.apply(atoms, box, dt=0.01, current_pressure=5.0)
        self.assertGreater(mu, 1.0)
        self.assertGreater(box.volume, 1000.0)


if __name__ == "__main__":
    unittest.main()
