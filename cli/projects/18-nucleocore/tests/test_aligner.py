"""
Unit tests for NucleoCore Pairwise Sequence Alignment Engine.
"""

import unittest
from nucleocore.aligner import align_pairwise, AlignmentMode
from nucleocore.types import ScoringMatrix


class TestAligner(unittest.TestCase):
    def test_global_exact_match(self):
        seq = "ATGCATGC"
        res = align_pairwise(seq, seq, mode=AlignmentMode.GLOBAL)
        self.assertEqual(res.score, len(seq) * 2)
        self.assertEqual(res.aligned_seq1, seq)
        self.assertEqual(res.aligned_seq2, seq)
        self.assertEqual(res.identity, 1.0)
        self.assertEqual(res.cigar.to_string(), "8M")

    def test_global_with_gap(self):
        seq1 = "ATGCATGC"
        seq2 = "ATG---GC"
        # Align with gap
        res = align_pairwise(seq1, "ATGGC", mode=AlignmentMode.GLOBAL, gap_open=3, gap_extend=1)
        self.assertIn("D", res.cigar.to_string())
        self.assertEqual(res.matches, 5)

    def test_local_alignment_smith_waterman(self):
        # Local homology embedded in noisy flanking sequences
        seq1 = "NNNNNNATGCATGCATNNNNNN"
        seq2 = "ZZZZZZATGCATGCATZZZZZZ"
        res = align_pairwise(seq1, seq2, mode=AlignmentMode.LOCAL)
        self.assertEqual(res.aligned_seq1, "ATGCATGCAT")
        self.assertEqual(res.aligned_seq2, "ATGCATGCAT")
        self.assertEqual(res.score, 20)
        self.assertEqual(res.identity, 1.0)

    def test_format_alignment_output(self):
        seq1 = "ATGCATGC"
        seq2 = "ATGGATGC"
        res = align_pairwise(seq1, seq2, mode=AlignmentMode.GLOBAL)
        formatted = res.format_alignment()
        self.assertIn("Alignment [GLOBAL]", formatted)
        self.assertIn("Seq1", formatted)
        self.assertIn("Seq2", formatted)


if __name__ == "__main__":
    unittest.main()
