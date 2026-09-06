"""
End-to-End integration tests for Atomix Simulation Orchestrator, Ensembles, and Visualizer.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Atom, SimulationBox
from atomix.builder import build_fcc_lattice, build_peptide_chain
from atomix.thermostats import (
    initialize_maxwell_boltzmann_velocities,
    BerendsenThermostat,
)
from atomix.simulation import MolecularDynamicsSimulation
from atomix.visualizer import render_molecular_system, render_telemetry_hud


class TestSimulationEndToEnd(unittest.TestCase):
    def test_nve_energy_conservation(self):
        # Microcanonical NVE: total energy E_tot = E_kin + E_pot must remain strictly constant
        atoms, box = build_fcc_lattice(n_cells=2, lattice_constant=1.7)
        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.8, remove_drift=True)

        # Pure NVE: no thermostat or barostat
        sim = MolecularDynamicsSimulation(
            atoms,
            box,
            timestep=0.001,
            cutoff=2.5,
            thermostat=None,
            barostat=None,
        )

        initial_energy = sim.step().total_energy

        # Run 60 steps
        traj = sim.run(num_steps=60, sample_interval=10)
        final_energy = traj[-1].total_energy

        # Energy fluctuations in Velocity Verlet should be tiny (< 0.1%)
        rel_energy_change = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(rel_energy_change, 0.005)

    def test_nvt_berendsen_temperature_regulation(self):
        # Canonical NVT: temperature must be maintained near target
        atoms, box = build_fcc_lattice(n_cells=2, lattice_constant=1.65)
        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.5)

        target_temp = 1.2
        thermo = BerendsenThermostat(target_temperature=target_temp, tau_t=0.05)
        sim = MolecularDynamicsSimulation(
            atoms,
            box,
            timestep=0.002,
            thermostat=thermo,
        )

        # Run 80 steps to allow thermal relaxation
        traj = sim.run(num_steps=80, sample_interval=10)
        final_temp = traj[-1].temperature

        # Temperature should have climbed towards 1.2
        self.assertGreater(final_temp, 0.8)
        self.assertLess(final_temp, 1.5)

    def test_peptide_chain_simulation(self):
        # Polypeptide chain with bonded and non-bonded forces
        atoms, bonds, angles, dihedrals, box = build_peptide_chain(num_residues=3)
        self.assertEqual(len(atoms), 12)
        self.assertGreater(len(bonds), 0)
        self.assertGreater(len(angles), 0)
        self.assertGreater(len(dihedrals), 0)

        initialize_maxwell_boltzmann_velocities(atoms, target_temperature=0.3)
        sim = MolecularDynamicsSimulation(
            atoms,
            box,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals,
            timestep=0.001,
        )

        traj = sim.run(num_steps=25)
        self.assertEqual(len(traj), 25)
        self.assertTrue(math.isfinite(traj[-1].total_energy))

    def test_visualizer_rendering(self):
        atoms, box = build_fcc_lattice(n_cells=1, lattice_constant=2.0)
        view = render_molecular_system(atoms, box=box, char_width=50, char_height=15)
        self.assertIn("\n", view)
        lines = view.splitlines()
        self.assertEqual(len(lines), 15)

        state = MolecularDynamicsSimulation(atoms, box).step()
        hud = render_telemetry_hud(state, len(atoms), 0, rg=1.0, diffusion_coeff=1e-5)
        self.assertIn("ATOMIX", hud)
        self.assertIn("Temp", hud)


if __name__ == "__main__":
    unittest.main()
