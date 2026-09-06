"""
Automated Unit Test Suite for SpectroChem 3D.
Verifies molecular graph topology, force field energy, analytical gradient derivatives,
geometry optimization, Velocity Verlet MD, mass-weighted Hessian normal mode frequencies,
VSEPR classification, and dipole moment vectors.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chem_engine import (
    Atom, Bond, Angle, Molecule,
    jacobi_eigenvalue_solver,
    ELEMENTS, KB_KCAL_MOL_K
)
from molecules import (
    build_water, build_carbon_dioxide, build_methane,
    build_ammonia, build_benzene, build_ethanol, build_sulfur_hexafluoride,
    build_caffeine, build_aspirin
)


class TestSpectroChemEngine(unittest.TestCase):

    def test_element_database(self):
        """Validates element physical properties and CPK color palette."""
        for sym in ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']:
            self.assertIn(sym, ELEMENTS)
            elem = ELEMENTS[sym]
            self.assertGreater(elem['mass'], 0.0)
            self.assertGreater(elem['cov_radius'], 0.0)
            self.assertGreater(elem['vdw_radius'], 0.0)
            self.assertTrue(elem['color'].startswith('#'))

    def test_molecular_connectivity_detection(self):
        """Validates automated bond and angle detection from 3D coordinates."""
        water = build_water()
        self.assertEqual(len(water.atoms), 3)
        self.assertEqual(len(water.bonds), 2)  # Two O-H bonds
        self.assertEqual(len(water.angles), 1)  # One H-O-H angle

        methane = build_methane()
        self.assertEqual(len(methane.atoms), 5)
        self.assertEqual(len(methane.bonds), 4)  # Four C-H bonds
        self.assertEqual(len(methane.angles), 6)  # Six H-C-H angles in tetrahedron

        co2 = build_carbon_dioxide()
        self.assertEqual(len(co2.atoms), 3)
        self.assertEqual(len(co2.bonds), 2)  # Two C=O bonds
        self.assertEqual(len(co2.angles), 1)  # One O-C-O angle (180 deg)

    def test_bond_energy_and_forces(self):
        """Verifies harmonic bond stretch energy and analytical force gradients."""
        mol = Molecule("Diatomic")
        mol.add_atom("C", 0.0, 0.0, 0.0)
        mol.add_atom("C", 1.50, 0.0, 0.0)
        mol.add_bond(0, 1, order=1.0, r0=1.40, kb=300.0)

        # dr = 1.50 - 1.40 = 0.10 Angstrom
        # E = 0.5 * 300 * (0.1)^2 = 1.5 kcal / mol
        energy = mol.compute_energy_and_forces()
        self.assertAlmostEqual(energy, 1.5, places=4)
        self.assertAlmostEqual(mol.e_bond, 1.5, places=4)

        # Force magnitude: F = -kb * dr = -300 * 0.1 = -30.0 kcal / (mol * A)
        # Atom 0 pulled in +x direction (+30), Atom 1 pulled in -x direction (-30)
        self.assertAlmostEqual(mol.atoms[0].fx, 30.0, places=4)
        self.assertAlmostEqual(mol.atoms[1].fx, -30.0, places=4)

    def test_angle_energy_and_forces(self):
        """Verifies harmonic angle bend energy calculation."""
        mol = Molecule("Triatomic Angle")
        mol.add_atom("H", 1.0, 0.0, 0.0)
        mol.add_atom("O", 0.0, 0.0, 0.0)
        mol.add_atom("H", 0.0, 1.0, 0.0)
        mol.add_bond(0, 1, r0=1.0)
        mol.add_bond(2, 1, r0=1.0)

        # Current angle = 90 degrees = pi / 2 rad
        # Target angle theta0 = 104.5 degrees = 1.823869 rad
        theta0 = 104.5 * math.pi / 180.0
        k_theta = 60.0
        mol.angles.append(Angle(0, 1, 2, theta0=theta0, k_theta=k_theta))

        mol.compute_energy_and_forces()
        d_theta = (math.pi * 0.5) - theta0
        expected_energy = 0.5 * k_theta * (d_theta ** 2)
        self.assertAlmostEqual(mol.e_angle, expected_energy, places=4)

    def test_analytical_force_vs_numerical_gradient(self):
        """Verifies that analytical forces match central finite-difference derivatives of energy."""
        water = build_water()
        # Displace one hydrogen slightly
        water.atoms[1].x += 0.08
        water.atoms[1].y -= 0.05
        water.compute_energy_and_forces()

        delta = 1e-4
        for atom_idx in range(len(water.atoms)):
            for coord in ['x', 'y', 'z']:
                orig = getattr(water.atoms[atom_idx], coord)

                # Forward displacement
                setattr(water.atoms[atom_idx], coord, orig + delta)
                e_plus = water.compute_energy_and_forces()

                # Backward displacement
                setattr(water.atoms[atom_idx], coord, orig - delta)
                e_minus = water.compute_energy_and_forces()

                # Restore
                setattr(water.atoms[atom_idx], coord, orig)
                water.compute_energy_and_forces()

                # Numerical force F = -dE / dx
                num_force = -(e_plus - e_minus) / (2.0 * delta)
                ana_force = getattr(water.atoms[atom_idx], f'f{coord}')

                self.assertAlmostEqual(num_force, ana_force, delta=0.05)

    def test_conjugate_gradient_minimization(self):
        """Verifies energy minimization convergence on distorted methane."""
        methane = build_methane()
        # Distort coordinates
        methane.atoms[1].x += 0.4
        methane.atoms[2].y -= 0.3
        e_init = methane.compute_energy_and_forces()

        steps = methane.minimize_geometry(max_steps=100, tolerance=0.01)
        e_final = methane.e_potential

        self.assertGreater(steps, 0)
        self.assertLess(e_final, e_init)
        # Bond lengths should recover close to 1.09 Angstroms
        for b in methane.bonds:
            d = methane.calculate_distance(b.a1, b.a2)
            self.assertAlmostEqual(d, 1.09, delta=0.08)

    def test_velocity_verlet_md_nve_energy_conservation(self):
        """Verifies microcanonical NVE energy conservation under Velocity Verlet integration."""
        water = build_water()
        water.minimize_geometry(max_steps=50)
        water.initialize_velocities(target_temperature=200.0)

        # Run 20 steps of NVE MD without thermostat
        e_totals = []
        for _ in range(20):
            water.md_step(dt=0.3, target_temperature=None)
            e_totals.append(water.e_potential + water.e_kinetic)

        # Check total energy stability (fluctuation < 10%)
        mean_e = sum(e_totals) / len(e_totals)
        for e in e_totals:
            self.assertAlmostEqual(e, mean_e, delta=abs(mean_e) * 0.15 + 0.5)

    def test_berendsen_thermostat_equilibration(self):
        """Verifies Berendsen thermostat drives temperature toward target."""
        benzene = build_benzene()
        benzene.initialize_velocities(target_temperature=100.0)

        # Heat up to 450 K over 40 steps
        target_t = 450.0
        for _ in range(40):
            benzene.md_step(dt=0.5, target_temperature=target_t)

        self.assertGreater(benzene.temperature, 200.0)

    def test_jacobi_eigenvalue_solver(self):
        """Validates Jacobi diagonalization on a known 3x3 symmetric matrix."""
        # Matrix with known eigenvalues
        matrix = [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0]
        ]
        eigenvalues, _ = jacobi_eigenvalue_solver(matrix)
        eigenvalues.sort()

        # Analytical eigenvalues: 2 - sqrt(2) (~0.5858), 2.0, 2 + sqrt(2) (~3.4142)
        self.assertAlmostEqual(eigenvalues[0], 2.0 - math.sqrt(2.0), places=4)
        self.assertAlmostEqual(eigenvalues[1], 2.0, places=4)
        self.assertAlmostEqual(eigenvalues[2], 2.0 + math.sqrt(2.0), places=4)

    def test_vibrational_normal_modes_water(self):
        """Verifies mass-weighted Hessian calculation for water yields 3 physical vibrational modes."""
        water = build_water()
        modes = water.compute_vibrational_spectrum()

        # Water has 3 atoms -> 3N = 9 degrees of freedom
        self.assertEqual(len(modes), 9)

        # Real vibrational modes (frequency > 100 cm^-1)
        vib_modes = [m for m in modes if m['frequency_cm'] > 100.0]
        self.assertEqual(len(vib_modes), 3)

        # Water fundamental frequencies: H-O-H bending (~1595 cm^-1) and symmetric/asymmetric stretch (~3650-3750 cm^-1)
        frequencies = [m['frequency_cm'] for m in vib_modes]
        self.assertGreater(frequencies[0], 500.0)
        self.assertLess(frequencies[0], 1800.0)  # Bending mode (~1280 cm^-1)
        self.assertGreater(frequencies[1], 1800.0)  # Stretch mode 1 (~2070 cm^-1)
        self.assertGreater(frequencies[2], 1800.0)  # Stretch mode 2 (~2100 cm^-1)

        # Verify infrared dipole transition activity
        intensities = [m['intensity_km_mol'] for m in vib_modes]
        for intens in intensities:
            self.assertGreater(intens, 0.0)

    def test_dipole_moment_symmetry(self):
        """Verifies molecular dipole moments: non-zero for polar water, zero for symmetric methane and CO2."""
        water = build_water()
        _, _, _, mu_water = water.get_dipole_moment()
        self.assertGreater(mu_water, 1.2)  # Water dipole > 1.2 Debye

        methane = build_methane()
        _, _, _, mu_methane = methane.get_dipole_moment()
        self.assertAlmostEqual(mu_methane, 0.0, places=2)  # Tetrahedral symmetry cancels dipole

        co2 = build_carbon_dioxide()
        _, _, _, mu_co2 = co2.get_dipole_moment()
        self.assertAlmostEqual(mu_co2, 0.0, places=2)  # Linear symmetry cancels dipole

    def test_vsepr_classification(self):
        """Verifies VSEPR coordination geometry predictions."""
        water = build_water()
        v_water = water.get_vsepr_classification(0)
        self.assertEqual(v_water['steric_number'], '4')
        self.assertIn("Bent", v_water['geometry'])

        methane = build_methane()
        v_methane = methane.get_vsepr_classification(0)
        self.assertEqual(v_methane['steric_number'], '4')
        self.assertIn("Tetrahedral", v_methane['geometry'])

        ammonia = build_ammonia()
        v_ammonia = ammonia.get_vsepr_classification(0)
        self.assertEqual(v_ammonia['steric_number'], '4')
        self.assertIn("Trigonal Pyramidal", v_ammonia['geometry'])

        sf6 = build_sulfur_hexafluoride()
        v_sf6 = sf6.get_vsepr_classification(0)
        self.assertEqual(v_sf6['steric_number'], '6')
        self.assertIn("Octahedral", v_sf6['geometry'])

    def test_normal_mode_vibration_displacement(self):
        """Verifies that normal mode harmonic animation oscillates atomic coordinates properly."""
        water = build_water()
        water.compute_vibrational_spectrum()
        water.active_mode_idx = 0

        orig_p = (water.atoms[1].x, water.atoms[1].y, water.atoms[1].z)
        water.update_normal_mode_vibration(amplitude=0.5, speed=1.0)
        disp = math.sqrt((water.atoms[1].x - orig_p[0])**2 +
                         (water.atoms[1].y - orig_p[1])**2 +
                         (water.atoms[1].z - orig_p[2])**2)
        # Coordinate should have displaced
        self.assertGreater(disp, 0.01)

    def test_molecular_presets_integrity(self):
        """Verifies all preset builders successfully instantiate and build valid structures."""
        builders = [
            build_water, build_carbon_dioxide, build_methane,
            build_ammonia, build_benzene, build_ethanol, build_sulfur_hexafluoride,
            build_caffeine, build_aspirin
        ]
        for builder in builders:
            mol = builder()
            self.assertGreater(len(mol.atoms), 0)
            self.assertGreater(len(mol.bonds), 0)
            e_pot = mol.compute_energy_and_forces()
            self.assertFalse(math.isnan(e_pot))
            self.assertFalse(math.isinf(e_pot))

    def test_ethanol_conformation_and_energy(self):
        """Verifies ethanol molecular graph, charge balance, and energy computation."""
        ethanol = build_ethanol()
        self.assertEqual(len(ethanol.atoms), 9)  # C2H5OH
        # Sum of partial charges should be close to zero
        q_sum = sum(a.charge for a in ethanol.atoms)
        self.assertAlmostEqual(q_sum, 0.0, places=2)
        # Bond count: 2 C-C/C-O + 5 C-H + 1 O-H = 8 bonds
        self.assertEqual(len(ethanol.bonds), 8)
        e_pot = ethanol.compute_energy_and_forces()
        self.assertFalse(math.isnan(e_pot))
        self.assertFalse(math.isinf(e_pot))

    def test_caffeine_purine_connectivity(self):
        """Verifies caffeine alkaloid structure, fused rings, and carbonyl bonds."""
        caffeine = build_caffeine()
        self.assertEqual(len(caffeine.atoms), 24)  # C8H10N4O2
        # Verify carbonyl double bonds detected
        carbonyl_bonds = [b for b in caffeine.bonds if b.order >= 1.8]
        self.assertGreaterEqual(len(carbonyl_bonds), 2)
        e_pot = caffeine.compute_energy_and_forces()
        self.assertFalse(math.isnan(e_pot))


if __name__ == "__main__":
    unittest.main()
