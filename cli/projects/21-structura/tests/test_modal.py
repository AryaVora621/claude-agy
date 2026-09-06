"""
Unit tests for Structura ModalSolver.
Tests cantilever beam natural resonant frequencies vs Euler-Bernoulli vibration theory and mode orthogonality.
"""

import unittest
import math
from structura.types import Node2D, Material
from structura.elements import Beam2D
from structura.mesh import Mesh2D
from structura.modal import ModalSolver


class TestModal(unittest.TestCase):
    def test_cantilever_beam_vibration_frequencies(self):
        # Clamped-free beam: L = 3.0 m, E = 200 GPa, rho = 7850 kg/m^3
        # Rectangular cross-section: width b = 0.05 m, height h = 0.10 m
        # A = b * h = 0.005 m^2
        # I = (b * h^3) / 12 = 0.05 * 1e-3 / 12 = 4.166667e-6 m^4
        # Mass per unit length m_bar = rho * A = 7850 * 0.005 = 39.25 kg/m
        # Exact Euler-Bernoulli 1st natural frequency:
        # omega_1 = 3.5160 * sqrt(E * I / (m_bar * L^4))
        # f_1 = omega_1 / (2 * pi)
        l = 3.0
        e = 200e9
        rho = 7850.0
        b = 0.05
        h = 0.10
        a = b * h
        i_z = (b * (h ** 3)) / 12.0
        m_bar = rho * a

        exact_omega_1 = 3.5160 * math.sqrt((e * i_z) / (m_bar * (l ** 4)))
        exact_f_1 = exact_omega_1 / (2.0 * math.pi)

        # Discretize beam into 10 elements
        n_elems = 10
        dx = l / n_elems
        mat = Material(elastic_modulus=e, poissons_ratio=0.3, density=rho, thickness=b)

        mesh = Mesh2D()
        for i in range(n_elems + 1):
            mesh.add_node(i * dx, 0.0)

        for i in range(n_elems):
            mesh.add_element(Beam2D(i, (i, i + 1), mat, cross_section_area=a, moment_of_inertia=i_z))

        # Clamp root node 0 (Ux=0, Uy=0, Theta=0)
        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        mesh.add_boundary_condition(0, 2, 0.0)

        solver = ModalSolver(mesh)
        res = solver.solve(num_modes=3)

        self.assertEqual(len(res.modes), 3)

        # Mode 1 frequency should match analytical within 5%
        f1_computed = res.modes[0].frequency_hz
        rel_error = abs(f1_computed - exact_f_1) / exact_f_1
        self.assertLess(rel_error, 0.05)

        # Frequencies must increase: f1 < f2 < f3
        self.assertLess(res.modes[0].frequency_hz, res.modes[1].frequency_hz)
        self.assertLess(res.modes[1].frequency_hz, res.modes[2].frequency_hz)

    def test_mode_orthogonality(self):
        # 6-element beam
        l = 2.0
        n_elems = 6
        dx = l / n_elems
        mat = Material(elastic_modulus=100e9, poissons_ratio=0.3, density=2500.0)

        mesh = Mesh2D()
        for i in range(n_elems + 1):
            mesh.add_node(i * dx, 0.0)

        for i in range(n_elems):
            mesh.add_element(Beam2D(i, (i, i + 1), mat, cross_section_area=0.01, moment_of_inertia=1e-5))

        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        mesh.add_boundary_condition(0, 2, 0.0)

        solver = ModalSolver(mesh)
        res = solver.solve(num_modes=3)

        m_diag = solver.assemble_lumped_mass_vector()
        phi1 = res.modes[0].eigenvector
        phi2 = res.modes[1].eigenvector

        # phi1^T * M * phi2 should be close to zero relative to norm
        cross_prod = sum(phi1[i] * m_diag[i] * phi2[i] for i in range(len(m_diag)))
        norm1 = math.sqrt(sum(phi1[i] * m_diag[i] * phi1[i] for i in range(len(m_diag))))
        norm2 = math.sqrt(sum(phi2[i] * m_diag[i] * phi2[i] for i in range(len(m_diag))))
        rel_ortho = abs(cross_prod) / (norm1 * norm2)

        self.assertLess(rel_ortho, 0.05)


if __name__ == "__main__":
    unittest.main()
