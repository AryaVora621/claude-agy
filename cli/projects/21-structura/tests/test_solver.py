"""
Unit tests for Structura FEASolver.
Tests cantilever beam bending vs Euler-Bernoulli theory, reaction equilibrium, and truss bridge mechanics.
"""

import unittest
import math
from structura.types import Node2D, Material, STRUCTURAL_STEEL
from structura.elements import Beam2D, Truss2D, QuadQ4
from structura.mesh import Mesh2D, generate_rectangular_mesh, generate_truss_bridge_mesh
from structura.solver import FEASolver


class TestSolver(unittest.TestCase):
    def test_cantilever_beam_analytical_deflection(self):
        # 1-element or 2-element Euler-Bernoulli beam
        # Length L = 2.0 m, E = 200 GPa, I = 1e-4 m^4, A = 0.01 m^2
        # End load P = -10,000 N (downward)
        # Analytical tip deflection: delta = P * L^3 / (3 * E * I)
        l = 2.0
        e = 200e9
        i_z = 1e-4
        a = 0.01
        p_load = -10000.0

        mat = Material(elastic_modulus=e, poissons_ratio=0.3, thickness=1.0)
        mesh = Mesh2D()
        mesh.add_node(0.0, 0.0)  # Node 0: fixed support
        mesh.add_node(1.0, 0.0)  # Node 1: intermediate
        mesh.add_node(2.0, 0.0)  # Node 2: tip

        mesh.add_element(Beam2D(0, (0, 1), mat, cross_section_area=a, moment_of_inertia=i_z))
        mesh.add_element(Beam2D(1, (1, 2), mat, cross_section_area=a, moment_of_inertia=i_z))

        # Clamp node 0 (Ux=0, Uy=0, Theta=0)
        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        mesh.add_boundary_condition(0, 2, 0.0)

        # Apply end load at node 2
        mesh.add_nodal_load(2, fy=p_load)

        solver = FEASolver(mesh)
        res = solver.solve()

        # Analytical tip deflection
        exact_tip_defl = (p_load * (l ** 3)) / (3.0 * e * i_z)  # -10000 * 8 / (6e7) = -0.0013333 m
        computed_tip_defl = res.displacements[3 * 2 + 1]  # Uy at node 2

        self.assertAlmostEqual(computed_tip_defl, exact_tip_defl, delta=abs(exact_tip_defl) * 0.01)

        # Reaction force at root node 0 must equal -p_load = +10,000 N
        r_root = res.reactions.get(0)
        self.assertIsNotNone(r_root)
        self.assertAlmostEqual(r_root[1], -p_load, delta=1.0)

    def test_truss_bridge_reaction_equilibrium(self):
        # 4-bay truss bridge, 20m span, 4m height
        mesh = generate_truss_bridge_mesh(span=20.0, height=4.0, num_bays=4)

        # Fix bottom left node 0 (pin: Ux=0, Uy=0)
        mesh.add_boundary_condition(0, 0, 0.0)
        mesh.add_boundary_condition(0, 1, 0.0)
        # Fix bottom right node 4 (roller: Uy=0)
        mesh.add_boundary_condition(4, 1, 0.0)

        # Downward load at midspan node 2: Fy = -50,000 N
        mesh.add_nodal_load(2, fy=-50000.0)

        solver = FEASolver(mesh)
        res = solver.solve()

        # Symmetry: reactions at node 0 and node 4 should each carry half the load (+25,000 N)
        r0 = res.reactions.get(0)
        r4 = res.reactions.get(4)
        self.assertIsNotNone(r0)
        self.assertIsNotNone(r4)
        self.assertAlmostEqual(r0[1], 25000.0, delta=10.0)
        self.assertAlmostEqual(r4[1], 25000.0, delta=10.0)

        # Total equilibrium sum Fy = 0
        total_ry = r0[1] + r4[1]
        self.assertAlmostEqual(total_ry, 50000.0, delta=10.0)

    def test_2d_continuum_cantilever_quads(self):
        # Cantilever beam with 2D QuadQ4 elements: L = 6.0, H = 1.0, 12x4 mesh
        mesh = generate_rectangular_mesh(length=6.0, height=1.0, nx=12, ny=4, material=STRUCTURAL_STEEL, elem_type="quad")

        # Clamp all nodes at x = 0
        left_nodes = mesh.find_nodes_at_x(0.0)
        for nid in left_nodes:
            mesh.add_boundary_condition(nid, 0, 0.0)
            mesh.add_boundary_condition(nid, 1, 0.0)

        # Apply downward shear load distributed along right edge x = 6.0
        right_nodes = mesh.find_nodes_at_x(6.0)
        total_p = -20000.0
        load_per_node = total_p / len(right_nodes)
        for nid in right_nodes:
            mesh.add_nodal_load(nid, fy=load_per_node)

        solver = FEASolver(mesh)
        res = solver.solve()

        self.assertGreater(res.max_displacement, 0.0)
        self.assertGreater(res.max_von_mises_stress, 0.0)
        self.assertLess(res.residual_norm, 1e-6)

        # Sum of Y-reactions at root must equal -total_p = +20,000 N
        total_rx = sum(r[0] for r in res.reactions.values())
        total_ry = sum(r[1] for r in res.reactions.values())
        self.assertAlmostEqual(total_rx, 0.0, delta=1.0)
        self.assertAlmostEqual(total_ry, 20000.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
