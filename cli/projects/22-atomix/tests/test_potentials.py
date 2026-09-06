"""
Unit tests for Atomix Force Fields and Potentials.
Tests:
  - Lennard-Jones 12-6 potential minimum, shifted cutoff, and force signs
  - Coulombic electrostatics with like/opposite charge behavior
  - Harmonic bond stretching and momentum conservation
  - Harmonic angle bending and 3-body momentum conservation
  - Periodic dihedral torsions and 4-body momentum conservation
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Bond, Angle, Dihedral
from atomix.potentials import (
    LennardJonesPotential,
    CoulombPotential,
    HarmonicBondPotential,
    HarmonicAnglePotential,
    PeriodicDihedralPotential,
)


class TestLennardJonesPotential(unittest.TestCase):
    def test_lj_minimum_and_forces(self):
        sigma = 1.0
        epsilon = 1.0
        r_min = 2.0 ** (1.0 / 6.0) * sigma  # ~1.122462

        # Unshifted potential at minimum must equal -epsilon
        lj_unshifted = LennardJonesPotential(cutoff=3.0, shifted=False)
        e_min, f_min, _ = lj_unshifted.evaluate_pair(Vector3D(r_min, 0.0, 0.0), sigma, epsilon)
        self.assertAlmostEqual(e_min, -epsilon, places=5)
        self.assertAlmostEqual(f_min.norm(), 0.0, places=5)

        # Repulsive regime: r < r_min -> force on atom 1 must push in -r direction
        e_rep, f_rep, _ = lj_unshifted.evaluate_pair(Vector3D(0.9, 0.0, 0.0), sigma, epsilon)
        self.assertGreater(e_rep, 0.0)
        self.assertLess(f_rep.x, 0.0)  # Pushes atom 1 to the left away from atom 2

        # Attractive regime: r > r_min -> force on atom 1 must pull in +r direction
        e_att, f_att, _ = lj_unshifted.evaluate_pair(Vector3D(1.5, 0.0, 0.0), sigma, epsilon)
        self.assertLess(e_att, 0.0)
        self.assertGreater(f_att.x, 0.0)  # Pulls atom 1 to the right towards atom 2

    def test_lj_shifted_cutoff(self):
        cutoff = 2.5
        lj_shifted = LennardJonesPotential(cutoff=cutoff, shifted=True)
        # Exactly at cutoff radius, energy must be zero
        e_cut, f_cut, _ = lj_shifted.evaluate_pair(Vector3D(cutoff, 0.0, 0.0), 1.0, 1.0)
        self.assertAlmostEqual(e_cut, 0.0, places=6)

        # Beyond cutoff, energy and force must be zero
        e_beyond, f_beyond, _ = lj_shifted.evaluate_pair(Vector3D(cutoff + 0.1, 0.0, 0.0), 1.0, 1.0)
        self.assertEqual(e_beyond, 0.0)
        self.assertEqual(f_beyond.norm(), 0.0)

    def test_lorentz_berthelot_mixing(self):
        s1, e1 = 1.0, 2.0
        s2, e2 = 2.0, 8.0
        s_mix, e_mix = LennardJonesPotential.mix_parameters(s1, e1, s2, e2)
        self.assertAlmostEqual(s_mix, 1.5)              # Arithmetic mean
        self.assertAlmostEqual(e_mix, math.sqrt(16.0))  # Geometric mean


class TestCoulombPotential(unittest.TestCase):
    def test_coulomb_repulsion_and_attraction(self):
        coulomb = CoulombPotential(cutoff=10.0, coulomb_constant=100.0)

        # Like charges (repulsion)
        e_rep, f1_rep, _ = coulomb.evaluate_pair(Vector3D(2.0, 0.0, 0.0), +1.0, +1.0)
        self.assertGreater(e_rep, 0.0)
        self.assertLess(f1_rep.x, 0.0)  # Repels atom 1 away from atom 2

        # Opposite charges (attraction)
        e_att, f1_att, _ = coulomb.evaluate_pair(Vector3D(2.0, 0.0, 0.0), +1.0, -1.0)
        self.assertLess(e_att, 0.0)
        self.assertGreater(f1_att.x, 0.0)  # Attracts atom 1 towards atom 2

        # At cutoff, shifted potential energy must be 0
        e_cut, _, _ = coulomb.evaluate_pair(Vector3D(10.0, 0.0, 0.0), +1.0, -1.0)
        self.assertAlmostEqual(e_cut, 0.0, places=6)


class TestBondedPotentials(unittest.TestCase):
    def test_harmonic_bond(self):
        bond = Bond(0, 1, length_eq=1.5, k_spring=200.0)

        # Equilibrium position -> zero energy, zero force
        p1 = Vector3D(0.0, 0.0, 0.0)
        p2_eq = Vector3D(1.5, 0.0, 0.0)
        e_eq, f1_eq, f2_eq = HarmonicBondPotential.evaluate(p1, p2_eq, bond)
        self.assertAlmostEqual(e_eq, 0.0)
        self.assertAlmostEqual(f1_eq.norm(), 0.0)
        self.assertAlmostEqual(f2_eq.norm(), 0.0)

        # Stretched bond -> restoring forces pull atoms together
        p2_stretch = Vector3D(1.7, 0.0, 0.0)
        e_str, f1_str, f2_str = HarmonicBondPotential.evaluate(p1, p2_stretch, bond)
        self.assertGreater(e_str, 0.0)
        self.assertGreater(f1_str.x, 0.0)  # Atom 1 pulled to right
        self.assertLess(f2_str.x, 0.0)     # Atom 2 pulled to left

        # Newton's third law
        f_net = f1_str + f2_str
        self.assertAlmostEqual(f_net.norm(), 0.0)

    def test_harmonic_angle(self):
        angle = Angle(0, 1, 2, theta_eq=0.5 * math.pi, k_angle=100.0)  # 90 degrees

        # 90 degrees conformation
        pi_eq = Vector3D(1.0, 0.0, 0.0)
        pj_eq = Vector3D(0.0, 0.0, 0.0)
        pk_eq = Vector3D(0.0, 1.0, 0.0)
        e_eq, fi_eq, fj_eq, fk_eq = HarmonicAnglePotential.evaluate(pi_eq, pj_eq, pk_eq, angle)
        self.assertAlmostEqual(e_eq, 0.0, places=5)
        self.assertAlmostEqual(fi_eq.norm(), 0.0, places=5)

        # Distorted 120 degree angle
        pk_dist = Vector3D(-0.5, math.sqrt(3) / 2.0, 0.0)
        e_dist, fi, fj, fk = HarmonicAnglePotential.evaluate(pi_eq, pj_eq, pk_dist, angle)
        self.assertGreater(e_dist, 0.0)

        # Translational invariance: total force on triplet must sum to zero
        f_total = fi + fj + fk
        self.assertAlmostEqual(f_total.norm(), 0.0, places=5)

    def test_periodic_dihedral(self):
        dih = Dihedral(0, 1, 2, 3, periodicity=2, phase_rad=0.0, k_dihedral=25.0)

        p1 = Vector3D(0.0, 1.0, 0.0)
        p2 = Vector3D(0.0, 0.0, 0.0)
        p3 = Vector3D(1.0, 0.0, 0.0)
        p4 = Vector3D(1.0, 1.0, 0.5)

        e, f1, f2, f3, f4 = PeriodicDihedralPotential.evaluate(p1, p2, p3, p4, dih)
        self.assertGreater(e, 0.0)

        # Exact momentum conservation across 4-atom torsion
        f_net = f1 + f2 + f3 + f4
        self.assertAlmostEqual(f_net.norm(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
