"""Unit tests for high-level GaussianRenderer and turntable animation."""

import unittest

from chromasplat.projection import Camera
from chromasplat.renderer import GaussianRenderer
from chromasplat.scene import SceneFactory


class TestRenderer(unittest.TestCase):
    """Test suite for GaussianRenderer pipeline and performance telemetry."""

    def setUp(self) -> None:
        self.scene = SceneFactory.orbiting_rings(num_planet_splats=20, num_ring_splats=40)
        self.renderer = GaussianRenderer(sh_degree=1)
        self.cam = Camera.orbit(width=48, height=24, azimuth_deg=45.0, elevation_deg=20.0, distance=3.0)

    def test_pipeline_execution_and_telemetry(self) -> None:
        res = self.renderer.render(self.scene, self.cam)
        self.assertEqual(res.width, 48)
        self.assertEqual(res.height, 24)

        prof = self.renderer.last_profile
        self.assertIsNotNone(prof)
        assert prof is not None
        self.assertGreater(prof.num_visible_gaussians, 0)
        self.assertGreater(prof.fps, 0.0)
        self.assertGreater(prof.total_time_ms, 0.0)

    def test_turntable_frames(self) -> None:
        frames = self.renderer.render_turntable(
            self.scene,
            width=32,
            height=16,
            num_frames=4,
            distance=3.0,
        )
        self.assertEqual(len(frames), 4)
        for frame in frames:
            self.assertEqual(frame.width, 32)
            self.assertEqual(frame.height, 16)


if __name__ == "__main__":
    unittest.main()
