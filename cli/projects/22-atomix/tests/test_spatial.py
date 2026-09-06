"""
Unit tests for Atomix Spatial Partitioning, Linked Cell Lists, and Verlet Lists.
Zero external dependencies.
"""

import math
import unittest
from atomix.types import Vector3D, Atom, SimulationBox
from atomix.spatial import LinkedCellList, VerletNeighborList


class TestSpatialPartitioning(unittest.TestCase):
    def test_linked_cell_vs_brute_force(self):
        # Create a 3D box with multiple atoms
        box = SimulationBox(12.0, 12.0, 12.0)
        cutoff = 2.5

        atoms = [
            Atom(0, "A0", "Ar", Vector3D(1.0, 1.0, 1.0)),
            Atom(1, "A1", "Ar", Vector3D(2.0, 1.0, 1.0)),  # distance 1.0
            Atom(2, "A2", "Ar", Vector3D(11.5, 1.0, 1.0)), # distance 1.5 across PBC
            Atom(3, "A3", "Ar", Vector3D(6.0, 6.0, 6.0)),  # far away
            Atom(4, "A4", "Ar", Vector3D(7.0, 6.0, 6.0)),  # distance 1.0 from A3
        ]

        # Brute-force pair search
        brute_pairs = set()
        n = len(atoms)
        for i in range(n):
            for j in range(i + 1, n):
                r12 = box.minimum_image_vector(atoms[i].position, atoms[j].position)
                if r12.norm() < cutoff:
                    brute_pairs.add((i, j))

        # Linked Cell list pair search
        cell_list = LinkedCellList(box, cutoff=cutoff)
        found_pairs = cell_list.find_all_pairs(atoms)
        cell_pair_set = set((p[0], p[1]) for p in found_pairs)

        self.assertEqual(cell_pair_set, brute_pairs)
        self.assertIn((0, 1), cell_pair_set)
        self.assertIn((0, 2), cell_pair_set)
        self.assertIn((3, 4), cell_pair_set)
        self.assertNotIn((0, 3), cell_pair_set)

    def test_excluded_pairs(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        cutoff = 2.5
        atoms = [
            Atom(0, "A0", "Ar", Vector3D(1.0, 1.0, 1.0)),
            Atom(1, "A1", "Ar", Vector3D(2.0, 1.0, 1.0)),
            Atom(2, "A2", "Ar", Vector3D(1.5, 2.0, 1.0)),
        ]

        cell_list = LinkedCellList(box, cutoff=cutoff)
        excluded = {(0, 1)}

        pairs = cell_list.find_all_pairs(atoms, excluded_pairs=excluded)
        pair_set = set((p[0], p[1]) for p in pairs)

        self.assertNotIn((0, 1), pair_set)
        self.assertIn((0, 2), pair_set)
        self.assertIn((1, 2), pair_set)

    def test_verlet_neighbor_list_rebuild(self):
        box = SimulationBox(10.0, 10.0, 10.0)
        cutoff = 2.0
        skin = 0.5
        verlet = VerletNeighborList(box, cutoff=cutoff, skin=skin)

        atoms = [
            Atom(0, "A0", "Ar", Vector3D(2.0, 2.0, 2.0)),
            Atom(1, "A1", "Ar", Vector3D(3.5, 2.0, 2.0)), # dist = 1.5 < cutoff
        ]

        # First query triggers build
        pairs1 = verlet.get_active_pairs(atoms)
        self.assertEqual(len(pairs1), 1)
        self.assertFalse(verlet.needs_rebuild(atoms))

        # Small displacement (< skin / 2) -> does not need rebuild
        atoms[0].position = Vector3D(2.1, 2.0, 2.0)
        self.assertFalse(verlet.needs_rebuild(atoms))

        # Large displacement (> skin / 2) -> needs rebuild
        atoms[0].position = Vector3D(3.0, 2.0, 2.0)
        self.assertTrue(verlet.needs_rebuild(atoms))

        # After get_active_pairs, list is automatically rebuilt
        pairs2 = verlet.get_active_pairs(atoms)
        self.assertFalse(verlet.needs_rebuild(atoms))
        self.assertEqual(len(pairs2), 1)


if __name__ == "__main__":
    unittest.main()
