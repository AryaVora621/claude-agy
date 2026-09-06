"""
Unit tests for Atomix Vector3D, Atom, Bond, Angle, Dihedral, and SimulationBox.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import (
    Vector3D,
    Atom,
    Bond,
    Angle,
    Dihedral,
    SimulationBox,
    CPK_COLORS,
)


class TestVector3D(unittest.TestCase):
    def test_vector_arithmetic(self):
        v1 = Vector3D(1.0, 2.0, 3.0)
        v2 = Vector3D(4.0, 5.0, 6.0)

        # Addition and Subtraction
        v_sum = v1 + v2
        self.assertAlmostEqual(v_sum.x, 5.0)
        self.assertAlmostEqual(v_sum.y, 7.0)
        self.assertAlmostEqual(v_sum.z, 9.0)

        v_diff = v2 - v1
        self.assertAlmostEqual(v_diff.x, 3.0)
        self.assertAlmostEqual(v_diff.y, 3.0)
        self.assertAlmostEqual(v_diff.z, 3.0)

        # Multiplication and Division
        v_mul = v1 * 2.5
        self.assertAlmostEqual(v_mul.x, 2.5)
        self.assertAlmostEqual(v_mul.y, 5.0)
        self.assertAlmostEqual(v_mul.z, 7.5)

        v_div = v2 / 2.0
        self.assertAlmostEqual(v_div.x, 2.0)
        self.assertAlmostEqual(v_div.y, 2.5)
        self.assertAlmostEqual(v_div.z, 3.0)

        # Negation
        v_neg = -v1
        self.assertAlmostEqual(v_neg.x, -1.0)
        self.assertAlmostEqual(v_neg.y, -2.0)
        self.assertAlmostEqual(v_neg.z, -3.0)

    def test_vector_products_and_norms(self):
        v1 = Vector3D(1.0, 0.0, 0.0)
        v2 = Vector3D(0.0, 1.0, 0.0)

        # Dot product
        self.assertAlmostEqual(v1.dot(v2), 0.0)
        self.assertAlmostEqual(v1.dot(v1), 1.0)

        # Cross product (x cross y = z)
        v_cross = v1.cross(v2)
        self.assertAlmostEqual(v_cross.x, 0.0)
        self.assertAlmostEqual(v_cross.y, 0.0)
        self.assertAlmostEqual(v_cross.z, 1.0)

        # Norm and Normalization
        v3 = Vector3D(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v3.norm_sq(), 25.0)
        self.assertAlmostEqual(v3.norm(), 5.0)
        u3 = v3.normalized()
        self.assertAlmostEqual(u3.x, 0.6)
        self.assertAlmostEqual(u3.y, 0.8)
        self.assertAlmostEqual(u3.z, 0.0)
        self.assertAlmostEqual(u3.norm(), 1.0)

        # Zero vector normalization
        zero = Vector3D(0.0, 0.0, 0.0).normalized()
        self.assertAlmostEqual(zero.norm(), 0.0)


class TestSimulationBox(unittest.TestCase):
    def test_pbc_wrapping(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        self.assertAlmostEqual(box.volume, 1000.0)

        # Inside box
        p1 = Vector3D(5.0, 3.0, 8.0)
        w1 = box.wrap_position(p1)
        self.assertAlmostEqual(w1.x, 5.0)
        self.assertAlmostEqual(w1.y, 3.0)
        self.assertAlmostEqual(w1.z, 8.0)

        # Negative coordinates
        p2 = Vector3D(-2.0, -15.0, 12.0)
        w2 = box.wrap_position(p2)
        self.assertAlmostEqual(w2.x, 8.0)
        self.assertAlmostEqual(w2.y, 5.0)
        self.assertAlmostEqual(w2.z, 2.0)

    def test_minimum_image_convention(self):
        box = SimulationBox(10.0, 10.0, 10.0)

        # Two points separated across the periodic boundary
        p1 = Vector3D(1.0, 5.0, 5.0)
        p2 = Vector3D(9.0, 5.0, 5.0)

        # r12 = p2 - p1 = (8, 0, 0) -> minimum image is (-2, 0, 0)
        r12 = box.minimum_image_vector(p1, p2)
        self.assertAlmostEqual(r12.x, -2.0)
        self.assertAlmostEqual(r12.y, 0.0)
        self.assertAlmostEqual(r12.z, 0.0)

    def test_box_scaling(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        box.scale(1.1)
        self.assertAlmostEqual(box.lx, 11.0)
        self.assertAlmostEqual(box.ly, 11.0)
        self.assertAlmostEqual(box.lz, 11.0)
        self.assertAlmostEqual(box.volume, 1331.0)


class TestAtomAndCPK(unittest.TestCase):
    def test_atom_properties(self):
        atom_c = Atom(0, "CA", "C", Vector3D(1.0, 2.0, 3.0), mass=12.011)
        atom_o = Atom(1, "O", "O", Vector3D(2.0, 2.0, 3.0), mass=15.999)
        atom_n = Atom(2, "N", "N", Vector3D(3.0, 2.0, 3.0), mass=14.007)

        self.assertEqual(atom_c.cpk_color, CPK_COLORS["C"])
        self.assertEqual(atom_o.cpk_color, CPK_COLORS["O"])
        self.assertEqual(atom_n.cpk_color, CPK_COLORS["N"])


if __name__ == "__main__":
    unittest.main()
