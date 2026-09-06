"""
Unit tests for Solida Manifold Half-Edge B-Rep Topology and Euler Invariants.
"""

import unittest
from solida.geometry import Vector3D
from solida.brep import (
    Vertex,
    HalfEdge,
    Edge,
    Loop,
    Face,
    Shell,
    Solid,
    build_solid_from_polygons,
)


class TestBRep(unittest.TestCase):
    def test_cube_topology_and_euler_invariants(self):
        # 8 vertices for unit cube [0, 1]^3
        p000 = Vector3D(0, 0, 0)
        p100 = Vector3D(1, 0, 0)
        p110 = Vector3D(1, 1, 0)
        p010 = Vector3D(0, 1, 0)
        p001 = Vector3D(0, 0, 1)
        p101 = Vector3D(1, 0, 1)
        p111 = Vector3D(1, 1, 1)
        p011 = Vector3D(0, 1, 1)

        polys = [
            [p000, p010, p110, p100],  # Bottom (-Z)
            [p001, p101, p111, p011],  # Top (+Z)
            [p000, p100, p101, p001],  # Front (-Y)
            [p110, p010, p011, p111],  # Back (+Y)
            [p000, p001, p011, p010],  # Left (-X)
            [p100, p110, p111, p101],  # Right (+X)
        ]

        cube = build_solid_from_polygons(polys, name="UnitCube")

        # Euler characteristic: V - E + F = 8 - 12 + 6 = 2
        v_count = len(cube.vertices)
        e_count = len(cube.edges)
        f_count = len(cube.faces)

        self.assertEqual(v_count, 8)
        self.assertEqual(e_count, 12)
        self.assertEqual(f_count, 6)
        self.assertEqual(cube.outer_shell.euler_characteristic(), 2)

        # Manifold integrity: every edge must have exactly 2 mated half-edges
        for edge in cube.edges:
            self.assertTrue(edge.is_manifold)
            self.assertIsNotNone(edge.half_edge.twin)
            self.assertEqual(edge.half_edge.twin.twin, edge.half_edge)

        # Exact Divergence Theorem Volume
        vol = cube.volume()
        self.assertAlmostEqual(vol, 1.0, places=5)

        # Surface area: 6 faces * 1.0 = 6.0
        area = cube.surface_area()
        self.assertAlmostEqual(area, 6.0, places=5)

        # Centroid: (0.5, 0.5, 0.5)
        c = cube.center_of_mass()
        self.assertAlmostEqual(c.x, 0.5, places=5)
        self.assertAlmostEqual(c.y, 0.5, places=5)
        self.assertAlmostEqual(c.z, 0.5, places=5)

    def test_tetrahedron_euler_invariant(self):
        # 4 vertices: (0,0,0), (1,0,0), (0.5, sqrt(3)/2, 0), (0.5, sqrt(3)/6, sqrt(6)/3)
        p0 = Vector3D(0, 0, 0)
        p1 = Vector3D(2, 0, 0)
        p2 = Vector3D(1, 1.732, 0)
        p3 = Vector3D(1, 0.577, 1.633)

        polys = [
            [p0, p2, p1],  # Base
            [p0, p1, p3],  # Face 1
            [p1, p2, p3],  # Face 2
            [p2, p0, p3],  # Face 3
        ]
        tet = build_solid_from_polygons(polys, name="Tetrahedron")

        # V = 4, E = 6, F = 4 -> chi = 4 - 6 + 4 = 2
        self.assertEqual(len(tet.vertices), 4)
        self.assertEqual(len(tet.edges), 6)
        self.assertEqual(len(tet.faces), 4)
        self.assertEqual(tet.outer_shell.euler_characteristic(), 2)
        self.assertGreater(tet.volume(), 0.0)


if __name__ == "__main__":
    unittest.main()
