"""
Unit tests for Structura Mesh2D container and parametric generators.
"""

import unittest
import math
from structura.mesh import (
    Mesh2D,
    generate_rectangular_mesh,
    generate_truss_bridge_mesh,
    generate_perforated_plate_mesh,
)
from structura.types import STRUCTURAL_STEEL


class TestMesh(unittest.TestCase):
    def test_rectangular_quad_mesh(self):
        # 10x4 grid -> (10+1)*(4+1) = 55 nodes, 40 quad elements
        mesh = generate_rectangular_mesh(length=10.0, height=4.0, nx=10, ny=4, elem_type="quad")
        self.assertEqual(mesh.num_nodes, 55)
        self.assertEqual(mesh.num_elements, 40)
        self.assertEqual(mesh.num_dofs, 110)

        min_x, max_x, min_y, max_y = mesh.get_bounds()
        self.assertAlmostEqual(min_x, 0.0)
        self.assertAlmostEqual(max_x, 10.0)
        self.assertAlmostEqual(min_y, 0.0)
        self.assertAlmostEqual(max_y, 4.0)

        # Boundary node finding
        left_nodes = mesh.find_nodes_at_x(0.0)
        self.assertEqual(len(left_nodes), 5)

    def test_rectangular_tri_mesh(self):
        # 5x2 grid -> (5+1)*(2+1) = 18 nodes, 5*2*2 = 20 CST triangles
        mesh = generate_rectangular_mesh(length=5.0, height=2.0, nx=5, ny=2, elem_type="tri")
        self.assertEqual(mesh.num_nodes, 18)
        self.assertEqual(mesh.num_elements, 20)

    def test_truss_bridge_mesh(self):
        # 4 bays
        mesh = generate_truss_bridge_mesh(span=20.0, height=4.0, num_bays=4)
        # 5 bottom nodes + 5 top nodes = 10 nodes
        self.assertEqual(mesh.num_nodes, 10)
        # 4 bottom + 4 top + 5 vertical + 4 diagonal = 17 members
        self.assertEqual(mesh.num_elements, 17)

    def test_perforated_plate_mesh(self):
        mesh = generate_perforated_plate_mesh(width=2.0, height=2.0, hole_radius=0.5, n_radial=4, n_tangential=6)
        # (4 + 1) * (6 + 1) = 35 nodes, 4 * 6 = 24 elements
        self.assertEqual(mesh.num_nodes, 35)
        self.assertEqual(mesh.num_elements, 24)

        # Verify inner radius is 0.5
        for i in range(7):
            n = mesh.nodes[i]
            r = math.sqrt(n.x * n.x + n.y * n.y)
            self.assertAlmostEqual(r, 0.5, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
