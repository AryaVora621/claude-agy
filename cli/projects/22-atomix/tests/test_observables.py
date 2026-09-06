"""
Unit tests for Atomix Thermodynamic Observables, RDF, MSD, and Structural Metrics.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Atom, SimulationBox
from atomix.observables import (
    compute_kinetic_energy,
    compute_temperature,
    compute_virial_pressure,
    compute_radius_of_gyration,
    compute_rmsd,
    RadialDistributionFunction,
    MSDTracker,
)


class TestObservables(unittest.TestCase):
    def test_energies_and_temperature(self):
        atoms = [
            Atom(0, "A", "Ar", Vector3D(0, 0, 0), velocity=Vector3D(2.0, 0.0, 0.0), mass=2.0),
            Atom(1, "B", "Ar", Vector3D(1, 0, 0), velocity=Vector3D(0.0, 1.0, 0.0), mass=4.0),
        ]
        # E_k = 0.5 * [2 * (4) + 4 * (1)] = 0.5 * [8 + 4] = 6.0
        ke = compute_kinetic_energy(atoms)
        self.assertAlmostEqual(ke, 6.0)

        # N_dofs = 3 * 2 - 3 = 3 -> T = 2 * 6.0 / (3 * 1.0) = 4.0
        temp = compute_temperature(atoms, kb=1.0, remove_drift=True)
        self.assertAlmostEqual(temp, 4.0)

    def test_virial_pressure(self):
        box = SimulationBox(10.0, 10.0, 10.0) # V = 1000
        atoms = [Atom(0, "A", "Ar", Vector3D(0, 0, 0))]
        ke = 300.0
        virial = 900.0
        # P = (2 * 300 + 900) / (3 * 1000) = 1500 / 3000 = 0.5
        p = compute_virial_pressure(atoms, box, ke, virial)
        self.assertAlmostEqual(p, 0.5)

    def test_radius_of_gyration_and_rmsd(self):
        # Symmetrical 4-atom square in xy plane of side length 2
        atoms = [
            Atom(0, "C0", "C", Vector3D(1.0, 1.0, 0.0), mass=1.0),
            Atom(1, "C1", "C", Vector3D(-1.0, 1.0, 0.0), mass=1.0),
            Atom(2, "C2", "C", Vector3D(-1.0, -1.0, 0.0), mass=1.0),
            Atom(3, "C3", "C", Vector3D(1.0, -1.0, 0.0), mass=1.0),
        ]
        # Center of mass is (0, 0, 0). Each atom distance from COM is sqrt(1 + 1) = sqrt(2).
        # R_g should be sqrt(2) ~ 1.414213
        rg = compute_radius_of_gyration(atoms)
        self.assertAlmostEqual(rg, math.sqrt(2.0), places=5)

        # RMSD against perturbed coordinates
        ref_positions = [a.position for a in atoms]
        atoms[0].position = Vector3D(1.1, 1.0, 0.0)  # shift by 0.1
        # RMSD = sqrt( (0.1)^2 / 4 ) = 0.1 / 2 = 0.05
        rmsd = compute_rmsd(atoms, ref_positions)
        self.assertAlmostEqual(rmsd, 0.05, places=5)

    def test_radial_distribution_function(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        # Place pairs at exact distance 2.0
        atoms = [
            Atom(0, "A0", "Ar", Vector3D(5.0, 5.0, 5.0)),
            Atom(1, "A1", "Ar", Vector3D(7.0, 5.0, 5.0)), # r = 2.0
        ]
        rdf = RadialDistributionFunction(r_max=5.0, n_bins=10)
        rdf.sample(atoms, box)

        r_centers, g_vals = rdf.get_distribution(box, num_atoms=2)
        self.assertEqual(len(r_centers), 10)

        # Bin covering r=2.0 (dr = 0.5, so bin 4 covers [2.0, 2.5)) must be non-zero
        self.assertGreater(g_vals[4], 0.0)
        # Bins far away must be zero
        self.assertEqual(g_vals[0], 0.0)
        self.assertEqual(g_vals[9], 0.0)

    def test_msd_tracker_with_pbc_wrapping(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        a0 = Atom(0, "A0", "Ar", Vector3D(9.0, 5.0, 5.0))
        tracker = MSDTracker([a0])

        # Atom moves across positive periodic boundary: 9.0 -> 10.5 wrapped to 0.5
        a0.position = Vector3D(0.5, 5.0, 5.0)
        msd = tracker.update([a0], box)

        # True displacement is 1.5, so MSD should be (1.5)^2 = 2.25
        self.assertAlmostEqual(msd, 2.25, places=5)

        # Diffusion coefficient D = MSD / (6 * t)
        d = MSDTracker.calculate_diffusion_coefficient(msd, elapsed_time=1.5)
        self.assertAlmostEqual(d, 2.25 / 9.0, places=5)


if __name__ == "__main__":
    unittest.main()
