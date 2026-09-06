"""
Unit tests for NucleoCore Types, DNA helpers, and CIGAR string handlers.
"""

import unittest
from nucleocore.types import (
    reverse_complement,
    gc_content,
    kmer_spectrum,
    canonical_kmer,
    CIGAR,
    ScoringMatrix,
)


class TestTypes(unittest.TestCase):
    def test_reverse_complement_standard(self):
        seq = "ATGC"
        self.assertEqual(reverse_complement(seq), "GCAT")

    def test_reverse_complement_degenerate(self):
        seq = "ATGC-N-RYSWKMBDHV"
        rc = reverse_complement("ACGTNR")
        self.assertEqual(rc, "YNACGT")

    def test_gc_content(self):
        self.assertAlmostEqual(gc_content("ATGC"), 0.5)
        self.assertAlmostEqual(gc_content("GGCC"), 1.0)
        self.assertAlmostEqual(gc_content("AATT"), 0.0)
        self.assertAlmostEqual(gc_content(""), 0.0)

    def test_kmer_spectrum(self):
        seq = "ACGTG"
        kmers = kmer_spectrum(seq, 3)
        self.assertEqual(kmers, ["ACG", "CGT", "GTG"])
        self.assertEqual(kmer_spectrum("AC", 3), [])

    def test_canonical_kmer(self):
        # "ATG" rev comp is "CAT", "ATG" < "CAT" -> "ATG"
        self.assertEqual(canonical_kmer("ATG"), "ATG")
        # "CAT" rev comp is "ATG", "ATG" < "CAT" -> "ATG"
        self.assertEqual(canonical_kmer("CAT"), "ATG")

    def test_cigar_operations(self):
        cigar = CIGAR.from_string("10M2I5M1D3M")
        self.assertEqual(cigar.to_string(), "10M2I5M1D3M")
        # Reference length: M (10+5+3) + D (1) = 19
        self.assertEqual(cigar.reference_length(), 19)
        # Query length: M (10+5+3) + I (2) = 20
        self.assertEqual(cigar.query_length(), 20)

    def test_cigar_incremental_add(self):
        cigar = CIGAR()
        cigar.add("M", 5)
        cigar.add("M", 3)
        cigar.add("I", 2)
        self.assertEqual(cigar.to_string(), "8M2I")

    def test_scoring_matrix(self):
        sm = ScoringMatrix(match=2, mismatch=-2, transition=-1, transversion=-3)
        self.assertEqual(sm.score("A", "A"), 2)
        # Transition: A <-> G
        self.assertEqual(sm.score("A", "G"), -1)
        # Transversion: A <-> C
        self.assertEqual(sm.score("A", "C"), -3)


if __name__ == "__main__":
    unittest.main()
