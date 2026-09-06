"""Unit tests for Camera model and 2D EWA perspective projection."""

import math
import unittest

from chromasplat.gaussian import Gaussian3D
from chromasplat.projection import Camera, ProjectionEngine


class TestProjection(unittest.TestCase):
    """Test suite for Camera extrinsics, intrinsics, and 2D EWA projection."""

    def test_camera_orthonormal_view_basis(self) -> None:
        cam = Camera.orbit(
            width=100,
            height=60,
            azimuth_deg=45.0,
            elevation_deg=30.0,
            distance=5.0,
            target=(0.0, 0.0, 0.0),
        )
        right, down, forward = cam.get_view_basis()

        # Check unit lengths
        for vec in (right, down, forward):
            length = math.sqrt(sum(x * x for x in vec))
            self.assertAlmostEqual(length, 1.0, places=5)

        # Check orthogonality: dot products == 0
        dot_rd = sum(r * d for r, d in zip(right, down))
        dot_rf = sum(r * f for r, f in zip(right, forward))
        dot_df = sum(d * f for d, f in zip(down, forward))

        self.assertAlmostEqual(dot_rd, 0.0, places=5)
        self.assertAlmostEqual(dot_rf, 0.0, places=5)
        self.assertAlmostEqual(dot_df, 0.0, places=5)

    def test_camera_world_to_camera_point(self) -> None:
        cam = Camera(
            width=100,
            height=60,
            position=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        )
        # Target at origin should have camera coordinates (0, 0, 5.0)
        tx, ty, tz = cam.world_to_camera_point((0.0, 0.0, 0.0))
        self.assertAlmostEqual(tx, 0.0, places=5)
        self.assertAlmostEqual(ty, 0.0, places=5)
        self.assertAlmostEqual(tz, 5.0, places=5)

    def test_projection_near_plane_clipping(self) -> None:
        cam = Camera(
            width=100,
            height=60,
            position=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            z_near=1.0,
        )
        engine = ProjectionEngine()

        # Point behind camera (z = 6.0 in world, behind eye at 5.0 looking at 0)
        g_behind = Gaussian3D.isotropic(position=(0.0, 0.0, 6.0), radius=0.2)
        proj = engine.project_scene([g_behind], cam)
        self.assertEqual(len(proj), 0)

        # Point in front of camera
        g_front = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.2)
        proj = engine.project_scene([g_front], cam)
        self.assertEqual(len(proj), 1)
        self.assertAlmostEqual(proj[0].depth, 5.0, places=5)

    def test_screen_covariance_properties(self) -> None:
        cam = Camera.orbit(width=120, height=80, azimuth_deg=0.0, elevation_deg=0.0, distance=4.0)
        engine = ProjectionEngine(anti_aliasing_filter=0.3)
        g = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.25)

        proj = engine.project_single(g, cam, cam.get_view_basis())
        self.assertIsNotNone(proj)
        assert proj is not None

        # 2D Covariance matrix: [[cov_a, cov_b], [cov_b, cov_c]]
        # Check positive definiteness and positive determinant
        det = proj.cov_a * proj.cov_c - proj.cov_b * proj.cov_b
        self.assertGreater(det, 0.0)
        self.assertGreater(proj.cov_a, 0.0)
        self.assertGreater(proj.cov_c, 0.0)
        self.assertGreater(proj.radius, 1.0)

    def test_depth_sorting_order(self) -> None:
        cam = Camera(width=80, height=40, position=(0.0, 0.0, 10.0), target=(0.0, 0.0, 0.0))
        engine = ProjectionEngine()

        g_far = Gaussian3D.isotropic(position=(0.0, 0.0, -2.0), radius=0.1)  # tz = 12.0
        g_mid = Gaussian3D.isotropic(position=(0.0, 0.0, 0.0), radius=0.1)   # tz = 10.0
        g_near = Gaussian3D.isotropic(position=(0.0, 0.0, 4.0), radius=0.1)  # tz = 6.0

        projected = engine.project_scene([g_far, g_near, g_mid], cam)
        self.assertEqual(len(projected), 3)
        # Should be sorted front-to-back: near, mid, far
        self.assertAlmostEqual(projected[0].depth, 6.0, places=4)
        self.assertAlmostEqual(projected[1].depth, 10.0, places=4)
        self.assertAlmostEqual(projected[2].depth, 12.0, places=4)


if __name__ == "__main__":
    unittest.main()
