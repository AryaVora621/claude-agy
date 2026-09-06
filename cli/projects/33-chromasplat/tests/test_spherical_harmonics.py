"""Unit tests for Real Spherical Harmonics directional color evaluation."""

import math
import unittest

from chromasplat.spherical_harmonics import (
    C0,
    create_specular_sh,
    eval_sh,
    eval_sh_basis,
    rgb_to_sh_deg0,
    sh_deg0_to_rgb,
)


class TestSphericalHarmonics(unittest.TestCase):
    """Test suite for spherical harmonics basis functions and radiance evaluation."""

    def test_basis_lengths_by_degree(self) -> None:
        dirs = (0.577, 0.577, 0.577)
        self.assertEqual(len(eval_sh_basis(0, dirs)), 1)
        self.assertEqual(len(eval_sh_basis(1, dirs)), 4)
        self.assertEqual(len(eval_sh_basis(2, dirs)), 9)
        self.assertEqual(len(eval_sh_basis(3, dirs)), 16)

    def test_degree_0_constant(self) -> None:
        b1 = eval_sh_basis(0, (1.0, 0.0, 0.0))
        b2 = eval_sh_basis(0, (0.0, 1.0, 0.0))
        b3 = eval_sh_basis(0, (-0.5, 0.7, 0.2))
        self.assertAlmostEqual(b1[0], C0)
        self.assertAlmostEqual(b2[0], C0)
        self.assertAlmostEqual(b3[0], C0)

    def test_degree_1_antisymmetry(self) -> None:
        d = (0.3, -0.6, 0.74)
        neg_d = (-d[0], -d[1], -d[2])
        b_pos = eval_sh_basis(1, d)
        b_neg = eval_sh_basis(1, neg_d)

        # Degree 1 basis functions are odd: Y_1^m(-d) = -Y_1^m(d)
        for i in range(1, 4):
            self.assertAlmostEqual(b_pos[i], -b_neg[i], places=6)

    def test_degree_2_symmetry(self) -> None:
        d = (0.3, -0.6, 0.74)
        neg_d = (-d[0], -d[1], -d[2])
        b_pos = eval_sh_basis(2, d)
        b_neg = eval_sh_basis(2, neg_d)

        # Degree 2 basis functions are even: Y_2^m(-d) = Y_2^m(d)
        for i in range(4, 9):
            self.assertAlmostEqual(b_pos[i], b_neg[i], places=6)

    def test_degree_3_antisymmetry(self) -> None:
        d = (0.3, -0.6, 0.74)
        neg_d = (-d[0], -d[1], -d[2])
        b_pos = eval_sh_basis(3, d)
        b_neg = eval_sh_basis(3, neg_d)

        # Degree 3 basis functions are odd: Y_3^m(-d) = -Y_3^m(d)
        for i in range(9, 16):
            self.assertAlmostEqual(b_pos[i], -b_neg[i], places=6)

    def test_rgb_to_sh_roundtrip(self) -> None:
        colors = [
            (0.1, 0.5, 0.9),
            (0.0, 1.0, 0.0),
            (0.8, 0.2, 0.3),
        ]
        for c in colors:
            sh0 = rgb_to_sh_deg0(c)
            c_rec = sh_deg0_to_rgb(sh0)
            self.assertAlmostEqual(c[0], c_rec[0], places=5)
            self.assertAlmostEqual(c[1], c_rec[1], places=5)
            self.assertAlmostEqual(c[2], c_rec[2], places=5)

    def test_specular_sh_directional_highlight(self) -> None:
        base = (0.2, 0.2, 0.2)
        highlight = (1.0, 1.0, 1.0)
        spec_dir = (0.0, 0.0, 1.0)

        sh = create_specular_sh(base, highlight, spec_dir, degree=2, shininess=0.6)
        color_towards = eval_sh(2, sh, (0.0, 0.0, 1.0))
        color_opposite = eval_sh(2, sh, (0.0, 0.0, -1.0))

        # Color aligned with specular lobe should be much brighter
        self.assertGreater(color_towards[0], color_opposite[0] + 0.3)
        self.assertGreater(color_towards[1], color_opposite[1] + 0.3)
        self.assertGreater(color_towards[2], color_opposite[2] + 0.3)


if __name__ == "__main__":
    unittest.main()
