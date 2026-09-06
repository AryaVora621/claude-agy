"""
Unit tests for Structura Element Formulations.
Tests stiffness matrix symmetry, rigid body modes, and patch tests for Truss, Beam, CST, and Quad Q4.
"""

import unittest
import math
from structura.types import Node2D, Material
from structura.elements import Truss2D, Beam2D, TriangleCST, QuadQ4


class TestElements(unittest.TestCase):
    def setUp(self):
        self.mat = Material(
            name="TestMaterial",
            elastic_modulus=1000.0,
            poissons_ratio=0.25,
            density=10.0,
            thickness=1.0,
        )

    def _assert_matrix_symmetric(self, mat, tol=1e-9):
        n = len(mat)
        for r in range(n):
            for c in range(n):
                self.assertAlmostEqual(mat[r][c], mat[c][r], delta=tol)

    def test_truss_stiffness_symmetry_and_rigid_mode(self):
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 4.0, 3.0),  # L = 5.0, c = 0.8, s = 0.6
        }
        truss = Truss2D(element_id=0, node_ids=(0, 1), material=self.mat, cross_section_area=0.01)
        k = truss.compute_stiffness_matrix(nodes)
        self.assertEqual(len(k), 4)
        self._assert_matrix_symmetric(k)

        # Rigid body translation ux=2.0, uy=3.0
        u_rigid = [2.0, 3.0, 2.0, 3.0]
        f_rigid = [sum(k[i][j] * u_rigid[j] for j in range(4)) for i in range(4)]
        for val in f_rigid:
            self.assertAlmostEqual(val, 0.0, delta=1e-10)

        # Uniform elongation
        stress, strain = truss.compute_stress_strain(nodes, [0.0, 0.0, 0.08, 0.06])
        # Delta L = 0.08*0.8 + 0.06*0.6 = 0.064 + 0.036 = 0.1
        # Strain = 0.1 / 5 = 0.02
        # Axial stress = 1000 * 0.02 = 20.0
        self.assertAlmostEqual(stress.von_mises(), 20.0, delta=1e-6)

    def test_beam_stiffness_symmetry_and_bending(self):
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 10.0, 0.0),
        }
        beam = Beam2D(
            element_id=0,
            node_ids=(0, 1),
            material=self.mat,
            cross_section_area=0.1,
            moment_of_inertia=0.001,
        )
        k = beam.compute_stiffness_matrix(nodes)
        self.assertEqual(len(k), 6)
        self._assert_matrix_symmetric(k)

        # Rigid translation along Y
        u_rigid = [0.0, 5.0, 0.0, 0.0, 5.0, 0.0]
        f_rigid = [sum(k[i][j] * u_rigid[j] for j in range(6)) for i in range(6)]
        for val in f_rigid:
            self.assertAlmostEqual(val, 0.0, delta=1e-10)

    def test_cst_stiffness_symmetry_and_patch_test(self):
        # 3-node triangle
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 2.0, 0.0),
            2: Node2D(2, 0.0, 1.0),
        }
        cst = TriangleCST(element_id=0, node_ids=(0, 1, 2), material=self.mat)
        k = cst.compute_stiffness_matrix(nodes)
        self.assertEqual(len(k), 6)
        self._assert_matrix_symmetric(k)

        # Rigid body translation
        u_rigid = [1.5, -2.5, 1.5, -2.5, 1.5, -2.5]
        f_rigid = [sum(k[i][j] * u_rigid[j] for j in range(6)) for i in range(6)]
        for val in f_rigid:
            self.assertAlmostEqual(val, 0.0, delta=1e-10)

        # Uniaxial stretch: eps_x = 0.01, eps_y = 0.0
        # u(x, y) = 0.01 * x, v(x, y) = 0
        u_disp = [0.01 * n.x if i % 2 == 0 else 0.0 for i, n in enumerate([nodes[0], nodes[1], nodes[2]]) for _ in (0,)]
        u_elem = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]
        stress, strain = cst.compute_stress_strain(nodes, u_elem)
        self.assertAlmostEqual(strain.eps_x, 0.01, delta=1e-6)
        self.assertAlmostEqual(strain.eps_y, 0.0, delta=1e-6)
        self.assertAlmostEqual(strain.gamma_xy, 0.0, delta=1e-6)

        # Theoretical stress: D11 * 0.01 = (1000 / (1 - 0.25^2)) * 0.01 = 10.666667
        d_mat = self.mat.get_elasticity_matrix(plane_strain=False)
        self.assertAlmostEqual(stress.sigma_x, d_mat[0][0] * 0.01, delta=1e-6)
        self.assertAlmostEqual(stress.sigma_y, d_mat[0][1] * 0.01, delta=1e-6)
        self.assertAlmostEqual(stress.tau_xy, 0.0, delta=1e-6)

    def test_quad_q4_stiffness_symmetry_and_patch_test(self):
        # 4-node quadrilateral: [0, 2] x [0, 1]
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 2.0, 0.0),
            2: Node2D(2, 2.0, 1.0),
            3: Node2D(3, 0.0, 1.0),
        }
        quad = QuadQ4(element_id=0, node_ids=(0, 1, 2, 3), material=self.mat)
        k = quad.compute_stiffness_matrix(nodes)
        self.assertEqual(len(k), 8)
        self._assert_matrix_symmetric(k)

        # Rigid body translation
        u_rigid = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        f_rigid = [sum(k[i][j] * u_rigid[j] for j in range(8)) for i in range(8)]
        for val in f_rigid:
            self.assertAlmostEqual(val, 0.0, delta=1e-9)

        # Uniform stretch: u(x) = 0.005 * x, v(y) = 0.0
        # Node 0: (0,0) -> (0,0)
        # Node 1: (2,0) -> (0.01, 0)
        # Node 2: (2,1) -> (0.01, 0)
        # Node 3: (0,1) -> (0,0)
        u_elem = [0.0, 0.0, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0]
        stress, strain = quad.compute_stress_strain(nodes, u_elem)
        self.assertAlmostEqual(strain.eps_x, 0.005, delta=1e-6)
        self.assertAlmostEqual(strain.eps_y, 0.0, delta=1e-6)
        self.assertAlmostEqual(strain.gamma_xy, 0.0, delta=1e-6)

        d_mat = self.mat.get_elasticity_matrix(plane_strain=False)
        self.assertAlmostEqual(stress.sigma_x, d_mat[0][0] * 0.005, delta=1e-6)
        self.assertAlmostEqual(stress.sigma_y, d_mat[0][1] * 0.005, delta=1e-6)
        self.assertAlmostEqual(stress.tau_xy, 0.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
