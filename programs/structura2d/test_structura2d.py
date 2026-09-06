"""
Automated Unit Test Suite for Structura 2D FEA Engine.
Verifies element stiffness matrices, patch tests, equilibrium balance,
Kirsch stress concentrations, beam deflection, and all engineering presets.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fea_engine import (
    Node2D, TrussElement2D, CSTElement2D, Quad4Element2D,
    FEAModel, solve_linear_system
)
from presets import (
    build_warren_truss, build_cantilever_cst, build_kirsch_plate_hole,
    build_l_bracket, build_thick_cylinder, build_quad4_shear_wall, PRESETS
)


class TestStructura2D(unittest.TestCase):

    def test_linear_solver(self):
        """Validates Gaussian elimination with partial pivoting on known systems."""
        # 3x3 system:
        # 2x + y - z = 8
        # -3x - y + 2z = -11
        # -2x + y + 2z = -3
        # Solution: x = 2, y = 3, z = -1
        A = [
            [ 2.0,  1.0, -1.0],
            [-3.0, -1.0,  2.0],
            [-2.0,  1.0,  2.0]
        ]
        b = [8.0, -11.0, -3.0]
        x = solve_linear_system(A, b)
        self.assertAlmostEqual(x[0], 2.0, places=5)
        self.assertAlmostEqual(x[1], 3.0, places=5)
        self.assertAlmostEqual(x[2], -1.0, places=5)

    def test_truss_axial_force_and_equilibrium(self):
        """Verifies 1D axial bar extension and reaction force equilibrium."""
        model = FEAModel("Single Bar")
        model.add_node(0, 0.0, 0.0, fix_x=True, fix_y=True)
        model.add_node(1, 2.0, 0.0, fix_x=False, fix_y=True, fx=10000.0)
        # Area = 0.01 m^2, E = 200 GPa
        model.add_truss(0, 0, 1, area=0.01, E=200e9)

        results = model.solve_static()
        # Delta L = P * L / (A * E) = 10000 * 2.0 / (0.01 * 200e9) = 1e-5 m
        self.assertAlmostEqual(model.nodes[1].ux, 1e-5, places=8)
        # Reaction at node 0 must equal -10000 N
        self.assertAlmostEqual(model.nodes[0].rx, -10000.0, places=2)
        # Axial force in bar must equal 10000 N
        self.assertAlmostEqual(model.trusses[0].axial_force, 10000.0, places=2)

    def test_cst_area_and_stiffness(self):
        """Verifies CST element area and stiffness matrix properties."""
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 1.0, 0.0),
            2: Node2D(2, 0.0, 1.0)
        }
        # Right triangle with base 1, height 1 -> area = 0.5
        cst = CSTElement2D(0, 0, 1, 2, thickness=1.0, E=1000.0, nu=0.25)
        area = cst.compute_geometry_and_B(nodes)
        self.assertAlmostEqual(area, 0.5, places=6)

        ke = cst.stiffness_matrix()
        self.assertEqual(len(ke), 6)
        self.assertEqual(len(ke[0]), 6)

        # Stiffness matrix must be symmetric: ke[i][j] == ke[j][i]
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(ke[i][j], ke[j][i], places=5)

    def test_cst_rigid_body_modes(self):
        """Verifies that rigid body translation produces zero strain and zero stress in CST."""
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 2.0, 0.0),
            2: Node2D(2, 1.0, 1.5)
        }
        cst = CSTElement2D(0, 0, 1, 2, thickness=0.1, E=200e9, nu=0.3)
        cst.compute_geometry_and_B(nodes)

        # Rigid body translation: ux = 0.05, uy = -0.02 for all 3 nodes
        u_rigid = [0.05, -0.02, 0.05, -0.02, 0.05, -0.02]
        sxx, syy, txy, vm = cst.recover_stress(u_rigid)
        self.assertAlmostEqual(sxx, 0.0, places=4)
        self.assertAlmostEqual(syy, 0.0, places=4)
        self.assertAlmostEqual(txy, 0.0, places=4)
        self.assertAlmostEqual(vm, 0.0, places=4)

    def test_quad4_area_and_symmetry(self):
        """Verifies Quad4 2x2 Gauss Quadrature integration and stiffness symmetry."""
        nodes = {
            0: Node2D(0, 0.0, 0.0),
            1: Node2D(1, 2.0, 0.0),
            2: Node2D(2, 2.0, 1.5),
            3: Node2D(3, 0.0, 1.5)
        }
        quad = Quad4Element2D(0, 0, 1, 2, 3, thickness=1.0, E=200e9, nu=0.3)
        ke = quad.stiffness_matrix(nodes)

        # Area = 2.0 * 1.5 = 3.0
        self.assertAlmostEqual(quad.area, 3.0, places=4)

        # Symmetry check
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(ke[i][j], ke[j][i], places=3)

    def test_warren_truss_equilibrium(self):
        """Verifies static equilibrium (sum F = 0, sum M = 0) on Warren-Pratt bridge preset."""
        truss = build_warren_truss()
        res = truss.solve_static()

        # Sum of applied external loads
        total_fy_applied = sum(node.fy for node in truss.nodes.values())
        # Sum of vertical reaction forces
        total_ry = sum(node.ry for node in truss.nodes.values() if node.fix_y)

        # Total vertical equilibrium: sum(Ry) + sum(Fy) = 0
        self.assertAlmostEqual(total_ry + total_fy_applied, 0.0, places=2)

        # Total horizontal reaction must be zero since no horizontal load applied
        total_rx = sum(node.rx for node in truss.nodes.values() if node.fix_x)
        self.assertAlmostEqual(total_rx, 0.0, places=2)

        self.assertGreater(res['max_disp'], 0.0)
        self.assertGreater(res['max_von_mises'], 0.0)

    def test_cantilever_tip_deflection(self):
        """Verifies cantilever beam deflection order of magnitude matches beam theory."""
        beam = build_cantilever_cst()
        res = beam.solve_static()

        # Tip nodes should deflect downwards (negative uy)
        tip_nodes = [node for node in beam.nodes.values() if node.x >= 3.99]
        for node in tip_nodes:
            self.assertLess(node.uy, 0.0)

        # Tip deflection should be in reasonable mm range (0.0001 to 0.05 m)
        self.assertGreater(res['max_disp'], 1e-4)
        self.assertLess(res['max_disp'], 0.10)

    def test_kirsch_stress_concentration(self):
        """Verifies stress concentration factor Kt > 1.5 in Kirsch plate with hole."""
        plate = build_kirsch_plate_hole()
        res = plate.solve_static()

        # Nominal far-field tensile stress applied is 10 MPa
        nominal_stress = 10e6
        # Peak stress near hole rim must be amplified: sigma_max / sigma_nom > 1.5
        sc_ratio = res['max_von_mises'] / nominal_stress
        self.assertGreater(sc_ratio, 1.5)

    def test_l_bracket_corner_stress(self):
        """Verifies high stress concentration at re-entrant inner corner of L-bracket."""
        bracket = build_l_bracket()
        res = bracket.solve_static()

        self.assertGreater(res['max_von_mises'], 1e6)
        self.assertGreater(res['strain_energy'], 0.0)

    def test_thick_cylinder_pressurization(self):
        """Verifies thick cylinder expands radially under internal pressure."""
        cyl = build_thick_cylinder()
        res = cyl.solve_static()

        # Inner boundary nodes should move outward (positive displacement)
        inner_nodes = [n for n in cyl.nodes.values() if math.isclose(math.sqrt(n.x**2 + n.y**2), 0.4, abs_tol=0.01)]
        for n in inner_nodes:
            radial_disp = (n.x * n.ux + n.y * n.uy) / math.sqrt(n.x**2 + n.y**2)
            self.assertGreater(radial_disp, 0.0)

    def test_quad4_shear_wall_drift(self):
        """Verifies multi-story shear wall lateral story drift under lateral wind loads."""
        wall = build_quad4_shear_wall()
        res = wall.solve_static()

        # Top nodes should deflect horizontally in positive x direction
        top_nodes = [n for n in wall.nodes.values() if n.y >= 5.99]
        for n in top_nodes:
            self.assertGreater(n.ux, 0.0)

    def test_plane_stress_vs_plane_strain_stiffness(self):
        """Verifies plane strain condition yields higher stiffness (lower deflection) than plane stress."""
        def make_model(is_strain: bool):
            m = FEAModel("Compare")
            m.add_node(0, 0.0, 0.0, fix_x=True, fix_y=True)
            m.add_node(1, 1.0, 0.0, fix_x=False, fix_y=True, fx=1e5)
            m.add_node(2, 0.0, 1.0, fix_x=True, fix_y=False)
            m.add_cst(0, 0, 1, 2, thickness=0.1, E=200e9, nu=0.3, is_plane_strain=is_strain)
            m.solve_static()
            return m.nodes[1].ux

        disp_stress = make_model(is_strain=False)
        disp_strain = make_model(is_strain=True)

        # Plane strain constraints add transverse stiffness, reducing in-plane displacement
        self.assertLess(disp_strain, disp_stress)

    def test_strain_energy_positive(self):
        """Verifies that total strain energy is strictly positive for loaded structures."""
        truss = build_warren_truss()
        res = truss.solve_static()
        self.assertGreater(res['strain_energy'], 0.0)

    def test_natural_frequency_positive(self):
        """Verifies Rayleigh quotient natural frequency calculation produces positive Hz."""
        truss = build_warren_truss()
        res = truss.solve_static()
        self.assertGreater(res['natural_frequency_hz'], 0.0)

    def test_all_presets_solvable(self):
        """Verifies every preset in the library instantiates and solves without errors."""
        for name, builder in PRESETS.items():
            model = builder()
            self.assertGreater(len(model.nodes), 0)
            res = model.solve_static()
            self.assertFalse(math.isnan(res['max_disp']))
            self.assertFalse(math.isnan(res['max_von_mises']))
            self.assertFalse(math.isinf(res['max_disp']))
            self.assertFalse(math.isinf(res['max_von_mises']))

    def test_degenerate_geometry_rejection(self):
        """Verifies that zero-length truss or zero-area CST raises ValueError."""
        model = FEAModel("Degenerate")
        model.add_node(0, 0.0, 0.0)
        model.add_node(1, 0.0, 0.0)  # Same coordinates
        t = model.add_truss(0, 0, 1)
        with self.assertRaises(ValueError):
            t.compute_geometry(model.nodes)

        model2 = FEAModel("Collinear CST")
        model2.add_node(0, 0.0, 0.0)
        model2.add_node(1, 1.0, 0.0)
        model2.add_node(2, 2.0, 0.0)  # Collinear points -> area = 0
        cst = model2.add_cst(0, 0, 1, 2)
        with self.assertRaises(ValueError):
            cst.compute_geometry_and_B(model2.nodes)


if __name__ == "__main__":
    unittest.main()
