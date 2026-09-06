"""Unit tests for Scene representation, PLY format serialization, and procedural factory."""

import tempfile
import unittest
from pathlib import Path

from chromasplat.gaussian import Gaussian3D
from chromasplat.scene import GaussianScene, PLYCodec, SceneFactory


class TestScene(unittest.TestCase):
    """Test suite for GaussianScene, PLY codecs, and procedural generators."""

    def test_scene_bounds_and_centroid(self) -> None:
        scene = GaussianScene()
        scene.add(Gaussian3D.isotropic(position=(-1.0, 0.0, 2.0), radius=0.1))
        scene.add(Gaussian3D.isotropic(position=(3.0, 4.0, -2.0), radius=0.1))

        bounds = scene.get_bounds()
        self.assertEqual(bounds[0], (-1.0, 0.0, -2.0))
        self.assertEqual(bounds[1], (3.0, 4.0, 2.0))

        center = scene.get_center()
        self.assertAlmostEqual(center[0], 1.0)
        self.assertAlmostEqual(center[1], 2.0)
        self.assertAlmostEqual(center[2], 0.0)

    def test_scene_translation_and_scaling(self) -> None:
        scene = GaussianScene()
        scene.add(Gaussian3D.isotropic(position=(1.0, 1.0, 1.0), radius=0.5))
        scene.translate(2.0, -1.0, 3.0)
        self.assertEqual(scene.gaussians[0].position, (3.0, 0.0, 4.0))

    def test_scene_pruning(self) -> None:
        scene = GaussianScene()
        scene.add(Gaussian3D.isotropic(position=(0, 0, 0), radius=0.1, opacity=0.9))
        scene.add(Gaussian3D.isotropic(position=(1, 1, 1), radius=0.1, opacity=0.001))  # Transparent
        scene.add(Gaussian3D.isotropic(position=(2, 2, 2), radius=50.0, opacity=0.9))  # Excessively huge

        pruned = scene.prune(min_opacity=0.01, max_scale=20.0)
        self.assertEqual(pruned, 2)
        self.assertEqual(len(scene), 1)

    def test_ply_ascii_roundtrip(self) -> None:
        scene = SceneFactory.orbiting_rings(num_planet_splats=20, num_ring_splats=40)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_scene.ply"
            PLYCodec.save_ascii(scene, file_path)
            self.assertTrue(file_path.exists())

            loaded_scene = PLYCodec.load_ascii(file_path)
            self.assertEqual(len(loaded_scene), len(scene))

            # Verify positions preserved
            for orig, rec in zip(scene.gaussians, loaded_scene.gaussians):
                for a, b in zip(orig.position, rec.position):
                    self.assertAlmostEqual(a, b, places=4)

    def test_procedural_factories(self) -> None:
        rings = SceneFactory.orbiting_rings(num_planet_splats=30, num_ring_splats=50)
        self.assertGreater(rings.count, 50)

        box = SceneFactory.cornell_box()
        self.assertGreater(box.count, 200)

        dna = SceneFactory.dna_double_helix(num_turns=2.0)
        self.assertGreater(dna.count, 100)


if __name__ == "__main__":
    unittest.main()
