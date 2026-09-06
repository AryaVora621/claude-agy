"""
Unit tests for NucleoCore BWT & FM-Index Subsystem.
"""

import unittest
from nucleocore.fm_index import FMIndex, build_suffix_array


class TestFMIndex(unittest.TestCase):
    def test_suffix_array(self):
        text = "banana$"
        sa = build_suffix_array(text)
        # Suffixes sorted: $, a$, ana$, anana$, banana$, na$, nana$
        # Indices: 6, 5, 3, 1, 0, 4, 2
        self.assertEqual(sa, [6, 5, 3, 1, 0, 4, 2])

    def test_bwt_and_reconstruction(self):
        ref = "ACGTACGTAGCTAGCTA"
        index = FMIndex(ref, occ_stride=4, sa_sample_rate=4)
        reconstructed = index.reconstruct_text()
        self.assertEqual(reconstructed, ref)

    def test_exact_search_count_and_locate(self):
        # Multiple occurrences of "TAG"
        ref = "GATTAGACATTAGAGATTAGA"
        index = FMIndex(ref, occ_stride=8, sa_sample_rate=4)

        # Count occurrences
        self.assertEqual(index.count("TAG"), 3)
        self.assertEqual(index.count("GATT"), 2)
        self.assertEqual(index.count("ZZZ"), 0)

        # Locate positions
        locs = index.locate("TAG")
        for loc in locs:
            self.assertEqual(ref[loc:loc + 3], "TAG")

    def test_inexact_search_mismatches(self):
        ref = "AACCGGTTAACCGGTT"
        index = FMIndex(ref, occ_stride=4, sa_sample_rate=2)

        # Query "AATCGG" has 1 mismatch against "AACCGG" at pos 0 and 8
        results = index.search_inexact("AATCGG", max_mismatches=1)
        positions = [pos for pos, mm in results]
        self.assertIn(0, positions)
        self.assertIn(8, positions)


if __name__ == "__main__":
    unittest.main()
