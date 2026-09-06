"""Unit tests for Optical System, Paraxial ABCD Matrices, and Cardinal Points."""

import math
import unittest

from rayoptix.material import ConstantIndexMaterial, MaterialCatalog
from rayoptix.ray import Ray
from rayoptix.system import OpticalSystem


class TestOpticalSystem(unittest.TestCase):
    """Test suite for sequential lens stack, paraxial optics, and 3D ray tracing."""

    def setUp(self) -> None:
        # Register a simple model glass with fixed index 1.5
        MaterialCatalog.register(ConstantIndexMaterial(name="GLASS15", fixed_index=1.5))

    def test_thin_lens_lensmakers_formula(self) -> None:
        # Plano-convex singlet: R1 = 50 mm, R2 = flat, thickness = 0.001 mm (thin), n = 1.5
        # 1/f = (n - 1) * (1/R1 - 1/R2) = (0.5) * (1/50 - 0) = 1/100 mm => f = 100 mm
        sys = OpticalSystem("ThinPlanoConvex")
        sys.add_surface(radius_of_curvature=50.0, thickness=0.001, material_name="GLASS15", semi_diameter=15.0)
        sys.add_surface(radius_of_curvature=0.0, thickness=0.0, material_name="AIR", semi_diameter=15.0)
        sys.add_surface(radius_of_curvature=0.0, thickness=0.0, material_name="AIR", semi_diameter=15.0, name="Image")

        cards = sys.compute_cardinal_points()
        self.assertAlmostEqual(cards.efl, 100.0, places=2)
        self.assertAlmostEqual(cards.bfl, 100.0, places=2)

    def test_paraxial_matrix_determinant(self) -> None:
        # Composite optical system in air must have det(M) == 1.0
        sys = OpticalSystem("ThickDoublet")
        sys.add_surface(radius_of_curvature=60.0, thickness=8.0, material_name="N-BK7", semi_diameter=20.0)
        sys.add_surface(radius_of_curvature=-40.0, thickness=3.0, material_name="N-SF11", semi_diameter=20.0)
        sys.add_surface(radius_of_curvature=-120.0, thickness=50.0, material_name="AIR", semi_diameter=20.0)

        A, B, C, D = sys.compute_paraxial_matrix(0, 2)
        det_M = A * D - B * C
        self.assertAlmostEqual(det_M, 1.0, places=6)

    def test_paraxial_focus_ray_trace(self) -> None:
        # Build plano-convex lens with R = 50 mm, thickness = 5.0 mm, n = 1.5
        sys = OpticalSystem("ThickPlanoConvex")
        sys.add_surface(radius_of_curvature=50.0, thickness=5.0, material_name="GLASS15", semi_diameter=15.0)
        sys.add_surface(radius_of_curvature=0.0, thickness=0.0, material_name="AIR", semi_diameter=15.0)
        sys.add_surface(radius_of_curvature=0.0, thickness=0.0, material_name="AIR", semi_diameter=15.0, name="Image")

        # Position image plane at paraxial focus
        bfl = sys.set_paraxial_image_plane()
        self.assertGreater(bfl, 90.0)
        self.assertLess(bfl, 105.0)

        # Trace a paraxial ray very close to axis (y = 0.1 mm)
        ray = Ray(origin=(0.0, 0.1, -10.0), direction=(0.0, 0.0, 1.0))
        history = sys.trace_ray(ray)
        self.assertEqual(len(history), 3)

        # Image plane hit should be extremely close to y = 0.0
        img_hit = history[-1]
        self.assertAlmostEqual(img_hit.point[1], 0.0, places=4)

    def test_collimated_fan_trace(self) -> None:
        sys = OpticalSystem("SimpleSinglet")
        sys.add_surface(radius_of_curvature=80.0, thickness=6.0, material_name="N-BK7", semi_diameter=18.0)
        sys.add_surface(radius_of_curvature=-80.0, thickness=100.0, material_name="AIR", semi_diameter=18.0)
        sys.add_surface(radius_of_curvature=0.0, thickness=0.0, material_name="AIR", semi_diameter=25.0, name="Image")

        fan = sys.trace_collimated_beam(num_rays=7, beam_radius=10.0)
        self.assertEqual(len(fan), 7)
        for trace in fan:
            self.assertGreaterEqual(len(trace), 2)
            self.assertFalse(trace[0].is_vignetted)


if __name__ == "__main__":
    unittest.main()
