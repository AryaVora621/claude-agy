"""
Unit tests for Structura core types, materials, and stress/strain tensors.
"""

import unittest
import math
from structura.types import (
    Node2D,
    Material,
    STRUCTURAL_STEEL,
    ALUMINUM_6061,
    StressTensor2D,
    StrainTensor2D,
    BoundaryCondition,
    NodalLoad,
)


class TestTypes(unittest.TestCase):
    def test_node_distance(self):
        n1 = Node2D(0, 0.0, 0.0)
        n2 = Node2D(1, 3.0, 4.0)
        self.assertAlmostEqual(n1.distance_to(n2), 5.0)

    def test_material_elasticity_matrix_plane_stress(self):
        mat = Material(elastic_modulus=100.0, poissons_ratio=0.25, thickness=1.0)
        d = mat.get_elasticity_matrix(plane_strain=False)
        # factor = 100 / (1 - 0.25^2) = 100 / 0.9375 = 106.666667
        # D[0][0] = factor, D[0][1] = 0.25 * factor = 26.666667
        # G = 100 / (2 * 1.25) = 40.0
        self.assertAlmostEqual(d[0][0], 100.0 / 0.9375)
        self.assertAlmostEqual(d[0][1], 25.0 / 0.9375)
        self.assertAlmostEqual(d[1][0], 25.0 / 0.9375)
        self.assertAlmostEqual(d[1][1], 100.0 / 0.9375)
        self.assertAlmostEqual(d[2][2], 40.0)
        self.assertEqual(d[0][2], 0.0)
        self.assertEqual(d[2][0], 0.0)

    def test_material_elasticity_matrix_plane_strain(self):
        mat = Material(elastic_modulus=100.0, poissons_ratio=0.25, thickness=1.0)
        d = mat.get_elasticity_matrix(plane_strain=True)
        # factor = 100 / (1.25 * 0.5) = 100 / 0.625 = 160.0
        # D[0][0] = (1 - 0.25) * 160 = 120.0
        # D[0][1] = 0.25 * 160 = 40.0
        # G = 40.0
        self.assertAlmostEqual(d[0][0], 120.0)
        self.assertAlmostEqual(d[0][1], 40.0)
        self.assertAlmostEqual(d[1][1], 120.0)
        self.assertAlmostEqual(d[2][2], 40.0)

    def test_stress_tensor_invariants(self):
        # Uniaxial tension sigma_x = 100, sigma_y = 0, tau_xy = 0
        s1 = StressTensor2D(sigma_x=100.0, sigma_y=0.0, tau_xy=0.0)
        p1, p2 = s1.principal_stresses()
        self.assertAlmostEqual(p1, 100.0)
        self.assertAlmostEqual(p2, 0.0)
        self.assertAlmostEqual(s1.von_mises(), 100.0)
        self.assertAlmostEqual(s1.max_shear_stress(), 50.0)

        # Pure shear tau_xy = 50, sigma_x = 0, sigma_y = 0
        s2 = StressTensor2D(sigma_x=0.0, sigma_y=0.0, tau_xy=50.0)
        p1, p2 = s2.principal_stresses()
        self.assertAlmostEqual(p1, 50.0)
        self.assertAlmostEqual(p2, -50.0)
        # Von Mises for pure shear is sqrt(3) * tau = 50 * sqrt(3) ~ 86.6025
        self.assertAlmostEqual(s2.von_mises(), 50.0 * math.sqrt(3.0))
        self.assertAlmostEqual(s2.max_shear_stress(), 50.0)

    def test_strain_tensor_principal(self):
        strain = StrainTensor2D(eps_x=0.002, eps_y=0.001, gamma_xy=0.0)
        e1, e2 = strain.principal_strains()
        self.assertAlmostEqual(e1, 0.002)
        self.assertAlmostEqual(e2, 0.001)


if __name__ == "__main__":
    unittest.main()
