"""
Unit tests for NucleoCore De Bruijn Graph De Novo Genome Assembler.
"""

import unittest
from nucleocore.types import kmer_spectrum
from nucleocore.assembler import DeBruijnGraph, calculate_assembly_stats


class TestAssembler(unittest.TestCase):
    def test_dbg_linear_assembly(self):
        genome = "ATGCGTACGTTAGCTACTA"
        k = 5
        read_len = 10
        reads = [genome[i:i + read_len] for i in range(len(genome) - read_len + 1)]

        dbg = DeBruijnGraph(k=k)
        dbg.add_reads(reads)

        contigs = dbg.build_contigs()
        self.assertTrue(len(contigs) >= 1)
        self.assertEqual(contigs[0], genome)

    def test_tip_clipping(self):
        dbg = DeBruijnGraph(k=4)
        # Main trunk: AAAA -> AAAC -> AACT -> ACTG
        dbg.add_kmer("AAAC", count=10)
        dbg.add_kmer("AACT", count=10)
        dbg.add_kmer("ACTG", count=10)

        # Erroneous tip off AAAA: AAAG (dead end)
        dbg.add_kmer("AAAG", count=1)

        self.assertEqual(dbg.out_degree("AAA"), 2)
        clipped = dbg.clip_tips()
        self.assertGreaterEqual(clipped, 1)
        # Dead end tip removed
        self.assertEqual(dbg.out_degree("AAA"), 1)

    def test_bubble_popping_heterozygous_snp(self):
        # Biological case: wild-type vs SNP substitution
        seq1 = "AACGTATT"  # wild-type (coverage 10)
        seq2 = "AACATATT"  # SNP at pos 3: G -> A (coverage 2)

        k = 4
        dbg = DeBruijnGraph(k=k)
        for km in kmer_spectrum(seq1, k):
            dbg.add_kmer(km, count=10)
        for km in kmer_spectrum(seq2, k):
            dbg.add_kmer(km, count=2)

        # Before popping: fork at AAC and merge at TAT
        self.assertEqual(dbg.out_degree("AAC"), 2)
        self.assertEqual(dbg.in_degree("TAT"), 2)

        popped = dbg.pop_bubbles()
        self.assertEqual(popped, 1)

        # After popping: alternate path removed, AAC out-degree is 1, TAT in-degree is 1
        self.assertEqual(dbg.out_degree("AAC"), 1)
        self.assertEqual(dbg.in_degree("TAT"), 1)

        # Assembles cleanly into the dominant wild-type sequence
        contigs = dbg.build_contigs()
        self.assertEqual(contigs[0], seq1)

    def test_assembly_stats_n50_l50(self):
        contigs = [
            "A" * 1000,
            "C" * 800,
            "G" * 600,
            "T" * 400,
            "A" * 200,
        ]
        # Total length = 3000. 50% = 1500.
        # Contig 1 (1000) + Contig 2 (800) = 1800 >= 1500.
        # N50 = 800, L50 = 2.
        stats = calculate_assembly_stats(contigs)
        self.assertEqual(stats.contig_count, 5)
        self.assertEqual(stats.total_length, 3000)
        self.assertEqual(stats.max_length, 1000)
        self.assertEqual(stats.n50, 800)
        self.assertEqual(stats.l50, 2)
        summary = stats.summary()
        self.assertIn("N50:          800 bp", summary)


if __name__ == "__main__":
    unittest.main()
