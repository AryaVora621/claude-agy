"""
NucleoCore: High-Performance Computational Genomics & Sequence Assembly Benchmarks.
Measures FM-Index backward search throughput, Gotoh DP cell updates,
De Bruijn Graph contig assembly, Viterbi decoding, and Braille dot-plot rasterization.
"""

import sys
import os
import time
import random
from typing import List

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nucleocore.types import reverse_complement, kmer_spectrum
from nucleocore.fm_index import FMIndex
from nucleocore.aligner import align_pairwise, AlignmentMode
from nucleocore.assembler import DeBruijnGraph, calculate_assembly_stats
from nucleocore.hmm import create_cpg_island_detector
from nucleocore.visualizer import generate_dot_plot


def generate_random_dna(length: int) -> str:
    bases = ["A", "C", "G", "T"]
    return "".join(random.choice(bases) for _ in range(length))


def bench_fm_index():
    print("\n--- 1. FM-Index Construction & Backward Search Throughput ---")
    ref_len = 100_000
    ref_dna = generate_random_dna(ref_len)

    t0 = time.perf_counter()
    index = FMIndex(ref_dna, occ_stride=32, sa_sample_rate=16)
    elapsed_build = time.perf_counter() - t0
    build_rate = ref_len / elapsed_build
    print(f"FM-Index Built: {ref_len:,} bp in {elapsed_build*1000:.2f} ms ({build_rate:,.0f} bp/sec)")

    # Benchmark exact pattern queries
    query_len = 20
    num_queries = 20_000
    # Sample real patterns from the reference
    queries = [
        ref_dna[i:i + query_len]
        for i in [random.randint(0, ref_len - query_len) for _ in range(num_queries)]
    ]

    t0 = time.perf_counter()
    match_count = 0
    for q in queries:
        l, r = index.backward_search_range(q)
        if l < r:
            match_count += 1
    elapsed_query = time.perf_counter() - t0
    queries_sec = num_queries / elapsed_query
    us_per_query = (elapsed_query / num_queries) * 1e6
    print(f"Backward Search: {num_queries:,} queries in {elapsed_query*1000:.2f} ms ({queries_sec:,.0f} queries/sec, {us_per_query:.2f} µs/query)")


def bench_gotoh_alignment():
    print("\n--- 2. Pairwise Alignment with Gotoh Affine Gaps (DP Cell Updates) ---")
    trials = 100
    len1 = 200
    len2 = 200
    total_cells = trials * len1 * len2

    seq1_list = [generate_random_dna(len1) for _ in range(trials)]
    seq2_list = [generate_random_dna(len2) for _ in range(trials)]

    t0 = time.perf_counter()
    for s1, s2 in zip(seq1_list, seq2_list):
        align_pairwise(s1, s2, mode=AlignmentMode.GLOBAL)
    elapsed = time.perf_counter() - t0
    cells_per_sec = total_cells / elapsed
    aligns_per_sec = trials / elapsed
    print(f"Gotoh Global Alignment: {trials} alignments ({total_cells:,} DP cells) in {elapsed*1000:.2f} ms")
    print(f"Throughput: {cells_per_sec:,.0f} DP cells/sec ({aligns_per_sec:,.0f} alignments/sec)")


def bench_debruijn_assembly():
    print("\n--- 3. De Bruijn Graph Ingestion & Contig Assembly ---")
    genome_len = 5000
    genome = generate_random_dna(genome_len)
    read_len = 100
    coverage = 20
    num_reads = (genome_len * coverage) // read_len

    reads = [
        genome[start:start + read_len]
        for start in [random.randint(0, genome_len - read_len) for _ in range(num_reads)]
    ]

    t0 = time.perf_counter()
    dbg = DeBruijnGraph(k=21)
    dbg.add_reads(reads)
    elapsed_ingest = time.perf_counter() - t0
    kmers_per_sec = (num_reads * (read_len - 21 + 1)) / elapsed_ingest
    print(f"DBG Ingestion: {num_reads:,} reads ({len(dbg.get_all_nodes()):,} unique nodes) in {elapsed_ingest*1000:.2f} ms ({kmers_per_sec:,.0f} kmers/sec)")

    t0 = time.perf_counter()
    dbg.clip_tips()
    dbg.pop_bubbles()
    contigs = dbg.build_contigs()
    elapsed_assembly = time.perf_counter() - t0
    stats = calculate_assembly_stats(contigs)
    print(f"Contig Assembly: {len(contigs)} contigs (N50: {stats.n50:,} bp, Total: {stats.total_length:,} bp) assembled in {elapsed_assembly*1000:.2f} ms")


def bench_viterbi_decoding():
    print("\n--- 4. Profile HMM & Viterbi Decoding Throughput ---")
    detector = create_cpg_island_detector()
    seq_len = 25_000
    seq = generate_random_dna(seq_len)

    t0 = time.perf_counter()
    score, path = detector.viterbi(seq)
    elapsed = time.perf_counter() - t0
    states_per_sec = seq_len / elapsed
    print(f"Viterbi Decoding: {seq_len:,} bases in {elapsed*1000:.2f} ms ({states_per_sec:,.0f} bases/sec)")


def bench_braille_dot_plot():
    print("\n--- 5. Sub-Pixel Unicode Braille Dot-Plot Rasterization ---")
    s1 = generate_random_dna(500)
    s2 = generate_random_dna(500)

    t0 = time.perf_counter()
    count = 50
    for _ in range(count):
        generate_dot_plot(s1, s2, width_chars=60, height_chars=20)
    elapsed = time.perf_counter() - t0
    plots_sec = count / elapsed
    print(f"Dot-Plot Rasterization: {count} plots in {elapsed*1000:.2f} ms ({plots_sec:,.0f} plots/sec)")


def run_all_benchmarks():
    print("=" * 68)
    print("      NUCLEOCORE: COMPUTATIONAL GENOMICS BENCHMARK SUITE")
    print("=" * 68)
    bench_fm_index()
    bench_gotoh_alignment()
    bench_debruijn_assembly()
    bench_viterbi_decoding()
    bench_braille_dot_plot()
    print("\n" + "=" * 68)
    print("      ALL NUCLEOCORE BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 68)


if __name__ == "__main__":
    run_all_benchmarks()
