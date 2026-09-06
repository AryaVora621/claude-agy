# NucleoCore: Computational Genomics & De Novo Sequence Assembly Engine

[![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib%20only)-success)]()

NucleoCore is an ultra-fast, zero-dependency computational genomics and bioinformatics engine implemented entirely from first principles in the pure Python standard library. It provides high-performance algorithms for full-genome exact and inexact search, optimal pairwise sequence alignment with affine gap costs, de novo genome assembly from short sequencing reads using topological De Bruijn graphs, probabilistic sequence modeling via profile Hidden Markov Models, and terminal visualization with sub-pixel Braille dot-plots:

* **Burrows-Wheeler Transform (BWT) & FM-Index**: Full-text Minute index enabling sub-millisecond exact string matching via Ferragina-Manzini backward search in $O(m)$ time (independent of reference genome length $N$), with checkpointed Occurrence tables and sampled Suffix Arrays for memory-bounded coordinate resolution.
* **Gotoh Affine Gap Pairwise Alignment**: Dynamic programming alignment engine implementing Gotoh's three-matrix recurrence ($M, I_x, I_y$) for Needleman-Wunsch (Global), Smith-Waterman (Local), and Semi-Global alignments in $O(N \cdot M)$ time with standardized CIGAR string generation.
* **De Novo Genome Assembly via De Bruijn Graphs (DBG)**: Builds directed $(k-1)$-mer adjacency graphs with $k$-mer coverage tracking, performing automated topological error correction:
  * **Tip Clipping**: Identifies and prunes short dead-end chains caused by sequencing errors near read termini.
  * **Bubble Popping**: Detects and merges divergent/reconvergent branches of identical length caused by heterozygous single nucleotide polymorphisms (SNPs) and sequencing noise.
  * **Contig Synthesis**: Traverses maximal non-branching paths to reconstruct contiguous chromosomal contigs with automated $N50$ and $L50$ quality metrics.
* **Profile Hidden Markov Models & Viterbi Decoding**: Log-space Viterbi algorithm for maximum-likelihood state sequence decoding, Forward-Backward posterior probability evaluation, and a calibrated 2-state epigenetic CpG island segmenter.
* **Sub-Pixel Unicode Braille Dot-Plot Visualizer**: High-resolution $2 \times 4$ sub-pixel Braille canvas (`U+2800..U+28FF`) rendering sequence homology matrices (identifying matches, insertions, deletions, and inversions) with 24-bit TrueColor ANSI nucleotide palettes and genomic read depth tracks.

---

## Architectural Systems & Theoretical Foundations

### 1. Burrows-Wheeler Transform & FM-Index
Modern genomic read aligners (e.g. Bowtie, BWA) rely on the FM-Index to index multi-gigabase genomes:
* **Suffix Array (SA)**:
  $$\text{SA}[i] = \text{start index of } i\text{-th lexicographically smallest suffix of } T\$$$
* **BWT Construction**:
  $$B[i] = \begin{cases} T[\text{SA}[i] - 1] & \text{if } \text{SA}[i] > 0 \\ \$ & \text{if } \text{SA}[i] = 0 \end{cases}$$
* **LF-Mapping (Last-to-First)**:
  The $i$-th occurrence of character $c$ in the Last column ($B$) corresponds to the exact same character in the First column ($F$):
  $$LF(i) = C[B[i]] + \text{Occ}(B[i], i)$$
  where $C[c]$ is the cumulative count of characters in $T\$$ strictly smaller than $c$, and $\text{Occ}(c, i)$ is the number of occurrences of $c$ in $B[0 \dots i-1]$.
* **Ferragina-Manzini Backward Search**:
  Given pattern $P = p_0 p_1 \dots p_{m-1}$:
  $$\text{Initialize } [l, r) \leftarrow [0, N+1)$$
  $$\text{For } k = m-1 \text{ down to } 0:$$
  $$l \leftarrow C[p_k] + \text{Occ}(p_k, l), \quad r \leftarrow C[p_k] + \text{Occ}(p_k, r)$$
  Search completes in $O(m)$ character rank queries with zero dependence on genome length $N$.

### 2. Gotoh Pairwise Alignment with Affine Gaps
Biological insertions and deletions occur in multi-base events. Gotoh (1982) decomposed the affine gap cost $W(k) = \text{open} + k \cdot \text{extend}$ into three coupled DP matrices:
* **Recurrence Relations**:
  $$I_x(i, j) = \max \begin{cases} M(i-1, j) - \text{open} - \text{extend} \\ I_x(i-1, j) - \text{extend} \end{cases}$$
  $$I_y(i, j) = \max \begin{cases} M(i, j-1) - \text{open} - \text{extend} \\ I_y(i, j-1) - \text{extend} \end{cases}$$
  $$M(i, j) = \text{score}(S_1[i], S_2[j]) + \max(M(i-1, j-1), I_x(i-1, j-1), I_y(i-1, j-1))$$
* **Smith-Waterman Local Alignment**:
  $$M(i, j) = \max(0, \text{Gotoh Recurrence})$$
  Traceback begins at $\arg\max_{i, j} M(i, j)$ and terminates when score reaches 0.

### 3. De Novo Genome Assembly via De Bruijn Graphs
* **$k$-mer Graph Formulation**:
  * Vertices: Unique $(k-1)$-mers.
  * Directed Edges: Observed $k$-mers directed from prefix $(k-1)$-mer to suffix $(k-1)$-mer.
* **Topological Error Cleaning**:
  * **Tip Clipping**: Removes paths of length $\le 2k$ ending in out-degree 0 or in-degree 0.
  * **Bubble Popping**: Identifies parallel paths between common fork and merge vertices, evaluating sequence similarity and collapsing lower-coverage alternative alleles.
* **Assembly Evaluation ($N50$ / $L50$)**:
  * $N50$: Contig length such that contigs of equal or greater length contain $\ge 50\%$ of total assembly length:
    $$\sum_{i=1}^{L_{50}} \text{len}(C_i) \ge \frac{1}{2} \sum \text{len}(C)$$

### 4. Profile Hidden Markov Models & Epigenetics
* **Log-Space Viterbi Algorithm**:
  Prevents floating-point underflow on long sequences:
  $$V_s(t) = \max_r \left( V_r(t-1) + \ln A_{r \to s} \right) + \ln E_s(y_t)$$
* **Epigenetic CpG Island Detection**:
  Two-state model modeling unmethylated CpG-dense promoters ($+$) against background genomic DNA ($-$) with elevated $G+C$ emission probabilities.

---

## Performance Benchmarks

Executed on Apple Silicon (Python 3.13 standard library, single-threaded):

| Subsystem | Operation | Measured Performance |
|:---|:---|:---:|
| **FM-Index Construction** | 100,000 bp DNA index build | **47,180 bp/sec** |
| **FM-Index Backward Search** | 20,000 exact 20-mer queries | **42,256 queries/sec (23.67 µs/query)** |
| **Gotoh Pairwise Alignment** | 4,000,000 DP cell evaluations | **2,291,096 DP cells/sec** |
| **De Bruijn Graph Ingestion** | 1,000 reads into $k$-mer graph | **3,487,960 kmers/sec** |
| **Contig Synthesis** | Maximal non-branching path assembly | **4.71 ms / assembly (4,994 bp assembled)** |
| **Viterbi HMM Decoding** | 25,000 bp state sequence decoding | **1,674,719 bases/sec** |
| **Braille Dot-Plot Canvas** | Homology matrix rasterization | **276 full plots/sec** |

---

## Project Structure

```
projects/18-nucleocore/
├── nucleocore/
│   ├── __init__.py           # Unified public API exports
│   ├── types.py              # DNA types, reverse complement, CIGAR, Scoring Matrix
│   ├── fm_index.py           # BWT, Suffix Array, LF-mapping, Occ table, Backward Search
│   ├── aligner.py            # Gotoh Affine Gap Needleman-Wunsch & Smith-Waterman
│   ├── assembler.py          # De Bruijn Graph, Tip Clipping, Bubble Popping, Contigs, N50
│   ├── hmm.py                # Profile HMM, Viterbi Decoding, CpG Island Segmenter
│   └── visualizer.py         # ANSI TrueColor Colorizer, Braille Dot-Plot, Coverage Depth
├── tests/
│   ├── test_types.py         # Reverse complement, GC content, CIGAR parsing
│   ├── test_fm_index.py      # BWT, inverse BWT, Occ rank, exact/inexact search
│   ├── test_aligner.py       # Global/local Gotoh alignment, affine gap penalties
│   ├── test_assembler.py     # k-mer spectrum, DBG assembly, bubble popping, N50
│   ├── test_hmm.py           # Profile HMM construction, Viterbi state decoding
│   └── test_visualizer.py    # Braille dot-plot generation, ANSI color sequences
├── benchmarks/
│   └── bench_genomics.py     # Microbenchmarks: BWT search, Gotoh DP cells, DBG, Viterbi
├── examples/
│   └── genomics_lab.py       # Interactive genomics lab: viral search, assembly, dot-plot
├── PLAN.md                   # Architectural design blueprint
├── TASK_QUEUE.md             # Autonomous task tracking
├── CHECKPOINT_LAST.md        # State synchronization
└── README.md                 # System documentation
```

---

## Quick Start & Usage Examples

### 1. Sub-Millisecond Pattern Search via FM-Index

```python
from nucleocore.fm_index import FMIndex

# Build FM-Index on reference sequence
reference = "GATTAGACATTAGAGATTAGA"
index = FMIndex(reference, occ_stride=16, sa_sample_rate=8)

# Count and locate occurrences in O(m) time
motif = "TAG"
count = index.count(motif)
positions = index.locate(motif)
print(f"Motif '{motif}' occurs {count} times at positions: {positions}")
# Positions: [3, 10, 17]
```

### 2. Gotoh Pairwise Alignment with CIGAR String

```python
from nucleocore.aligner import align_pairwise, AlignmentMode

seq1 = "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAG"
seq2 = "ATGTTTGTTTTTC---TTTTATTGCCACTAGTCTCGGTAGTCAG"

alignment = align_pairwise(seq1, seq2, mode=AlignmentMode.GLOBAL, gap_open=4, gap_extend=1)
print(f"Score: {alignment.score} | Identity: {alignment.identity*100:.1f}%")
print(f"CIGAR: {alignment.cigar.to_string()}")
print(alignment.format_alignment())
```

### 3. De Novo Genome Assembly from Sequencing Reads

```python
from nucleocore.assembler import DeBruijnGraph, calculate_assembly_stats

# Instantiate De Bruijn Graph with k=21
dbg = DeBruijnGraph(k=21)
dbg.add_reads(sequencing_reads)

# Perform topological error correction
dbg.clip_tips()
dbg.pop_bubbles()

# Assemble contigs
contigs = dbg.build_contigs()
stats = calculate_assembly_stats(contigs)
print(stats.summary())
```

### 4. Epigenetic CpG Island Discovery via Viterbi HMM

```python
from nucleocore.hmm import detect_cpg_islands

chromosome_seq = "ATATATATAT" * 10 + "CGCGCGCGCG" * 10 + "ATATATATAT" * 10
islands = detect_cpg_islands(chromosome_seq)
for start, end in islands:
    print(f"CpG Island predicted at [{start}:{end}] (Length: {end - start} bp)")
```

---

## Verification & Interactive Demos

### Run Full Test Suite
```bash
python3 -m unittest discover -s tests
# Ran 29 tests in 0.002s -> OK
```

### Run Performance Benchmarks
```bash
python3 benchmarks/bench_genomics.py
```

### Run Interactive Genomics Laboratory
```bash
python3 examples/genomics_lab.py
```
Demonstrates viral genome indexing, sub-millisecond motif location, Gotoh pairwise alignment with CIGAR strings, de novo assembly with bubble popping, CpG island segmentation, and sub-pixel Unicode Braille dot-plots.
