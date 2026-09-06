#!/usr/bin/env python3
"""
NucleoCore: Computational Genomics & Sequence Assembly Interactive Laboratory.
Demonstrates:
  1. Full-text FM-Index backward search on viral genomes.
  2. Optimal pairwise sequence alignment with Gotoh affine gap penalties.
  3. De novo genome assembly using topological De Bruijn graphs with tip clipping & bubble popping.
  4. Epigenetic CpG island discovery using log-space Viterbi HMM decoding.
  5. Sub-pixel Unicode Braille dot-plot homology matrix and genomic read depth track.
"""

import sys
import os
import time
import random

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nucleocore.types import reverse_complement, gc_content, kmer_spectrum
from nucleocore.fm_index import FMIndex
from nucleocore.aligner import align_pairwise, AlignmentMode
from nucleocore.assembler import DeBruijnGraph, calculate_assembly_stats
from nucleocore.hmm import detect_cpg_islands, ProfileHMM
from nucleocore.visualizer import (
    colorize_sequence,
    generate_dot_plot,
    render_coverage_depth_track,
    render_genomics_summary,
)


def run_genomics_lab():
    print("\n" + "=" * 76)
    print("      NUCLEOCORE: COMPUTATIONAL GENOMICS & SEQUENCE ASSEMBLY LAB")
    print("   First-Principles Genomics, FM-Index, Gotoh Alignment & De Bruijn DBG")
    print("=" * 76)

    # 1. Synthetic Viral Genome (Spike snippet)
    random.seed(42)
    spike_motif = "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACTCAATTACCCCCTGCATACACTAATTCTTTCACACGTGGTGTTTATTACCCTGACAAAGTTTTCAGATCCTCAGTTTTACATTCAACTCAGGACTTGTTCTTACCTTTCTTTTCCAATGTTACTTGGTTCCATGCTATACATGTCTCTGGGACCAATGGTACTAAGAGGTTTGATAACCCTGTCCTACCA"
    flank_left = "".join(random.choice(["A", "C", "G", "T"]) for _ in range(300))
    flank_right = "".join(random.choice(["A", "C", "G", "T"]) for _ in range(300))
    viral_genome = flank_left + spike_motif + flank_right

    # Simulated coverage depth
    sim_coverage = [random.randint(15, 30) for _ in range(len(viral_genome))]
    for i in range(len(flank_left), len(flank_left) + len(spike_motif)):
        sim_coverage[i] += random.randint(25, 45) # Higher depth over spike

    print("\n[1] Reference Viral Genome & Epigenetic Landscape:")
    print(render_genomics_summary("SARS-CoV-2 Spike Snippet (Synthetic)", viral_genome, sim_coverage))

    # 2. FM-Index Exact & Inexact Search
    print("\n[2] FM-Index Construction & Sub-Millisecond Pattern Search:")
    t0 = time.perf_counter()
    index = FMIndex(viral_genome, occ_stride=16, sa_sample_rate=8)
    elapsed_index = time.perf_counter() - t0
    print(f" -> FM-Index constructed over {len(viral_genome):,} bp in {elapsed_index*1000:.2f} ms")

    target_motif = "TGTTTTATTGCCACTAGTCT"
    t0 = time.perf_counter_ns()
    count = index.count(target_motif)
    positions = index.locate(target_motif)
    elapsed_ns = time.perf_counter_ns() - t0

    print(f" -> Query Motif: '{target_motif}' ({len(target_motif)} bp)")
    print(f"    Occurrences: {count} | 0-indexed Reference Coordinate: {positions}")
    print(f"    Query Latency: {elapsed_ns / 1000:.2f} µs")

    # Inexact query with 1 mutation
    mutated_query = target_motif[:10] + ("A" if target_motif[10] != "A" else "C") + target_motif[11:]
    inexact_res = index.search_inexact(mutated_query, max_mismatches=1)
    print(f" -> Inexact Search (1 mismatch): '{mutated_query}' -> Found at {inexact_res}")

    # 3. Gotoh Affine Gap Sequence Alignment
    print("\n[3] Gotoh Pairwise Alignment (Needleman-Wunsch with Affine Gaps):")
    ref_sub = spike_motif[:70]
    # Introduce 1 deletion of 3 bp and 1 insertion of 2 bp and 1 SNP
    var_sub = ref_sub[:15] + ref_sub[18:35] + "GG" + ref_sub[35:50] + "A" + ref_sub[51:]

    alignment = align_pairwise(ref_sub, var_sub, mode=AlignmentMode.GLOBAL, gap_open=4, gap_extend=1)
    print(alignment.format_alignment(line_width=60))

    # 4. De Novo Genome Assembly via De Bruijn Graph
    print("\n[4] De Novo Genome Assembly from Shotgun Reads with Bubble Popping:")
    # Simulate 200 reads of length 60 with 1 heterozygous SNP bubble
    read_len = 50
    reads = []
    for _ in range(120):
        st = random.randint(0, len(spike_motif) - read_len)
        reads.append(spike_motif[st:st + read_len])

    # Inject minor SNP allele
    snp_pos = 100
    mutant_spike = spike_motif[:snp_pos] + ("G" if spike_motif[snp_pos] != "G" else "A") + spike_motif[snp_pos + 1:]
    for _ in range(25):
        st = max(0, min(len(mutant_spike) - read_len, snp_pos - 20 + random.randint(0, 10)))
        reads.append(mutant_spike[st:st + read_len])

    dbg = DeBruijnGraph(k=15)
    dbg.add_reads(reads)
    initial_nodes = len(dbg.get_all_nodes())
    print(f" -> Initial De Bruijn Graph: {initial_nodes:,} nodes from {len(reads)} reads")

    clipped = dbg.clip_tips()
    popped = dbg.pop_bubbles()
    print(f" -> Topological Cleaning: {clipped} dead-end tips clipped, {popped} SNP bubbles popped")

    contigs = dbg.build_contigs()
    stats = calculate_assembly_stats(contigs)
    print(f" -> Assembly Output: {len(contigs)} contigs | Longest Contig: {stats.max_length:,} bp (Target: {len(spike_motif):,} bp)")
    print(f"    N50: {stats.n50:,} bp | L50: {stats.l50} | Assembly GC: {stats.gc_fraction*100:.2f}%")

    # 5. CpG Island Segmentation with Viterbi Decoding
    print("\n[5] Epigenetic CpG Island Discovery via 2-State HMM (Viterbi):")
    # Insert high-density CG island inside AT background
    test_seq = "ATATATATAT" * 10 + "CGCGCGCGCG" * 8 + "ATATATATAT" * 10
    islands = detect_cpg_islands(test_seq)
    print(f" -> Analyzed {len(test_seq)} bp chromosome fragment")
    for start, end in islands:
        island_seq = test_seq[start:end]
        print(f"    Identified CpG Island at [{start}:{end}] ({end-start} bp, GC: {gc_content(island_seq)*100:.1f}%)")
        print(f"    Sequence: {colorize_sequence(island_seq[:40])}...")

    # 6. Sub-Pixel Unicode Braille Dot-Plot Homology Matrix
    print("\n[6] Sub-Pixel Unicode Braille Sequence Homology Dot-Plot:")
    variant_b = spike_motif[:120]
    # Variant with a 15bp tandem duplication
    variant_dup = variant_b[:50] + variant_b[35:50] + variant_b[50:]
    dot_plot_str = generate_dot_plot(variant_b, variant_dup, width_chars=54, height_chars=14, window=3, min_match=2)
    print(dot_plot_str)

    print("\n" + "=" * 76)
    print("            GENOMICS LABORATORY COMPLETED SUCCESSFULLY")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_genomics_lab()
