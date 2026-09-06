# Architectural Plan: Project 18 - NucleoCore Computational Genomics & Sequence Assembly Engine

## 1. Overview & Core Philosophy

NucleoCore is a zero-dependency, portfolio-grade computational genomics and bioinformatics engine built from first principles in the pure Python standard library. It provides high-performance algorithms for full-genome exact/inexact search, optimal pairwise alignment with affine gap costs, de novo genome assembly using topological De Bruijn graphs, probabilistic sequence modeling with profile Hidden Markov Models, and terminal visualization with sub-pixel Braille dot-plots.

---

## 2. Mathematical Formulations & Data Structures

### 2.1 Burrows-Wheeler Transform & FM-Index (`nucleocore/fm_index.py`)
- **Suffix Array**: Given text $T$ of length $N$ with sentinel character `$` (lexicographically smallest, ordinal 0):
  $$\text{SA}[i] = \text{start index of } i\text{-th lexicographically smallest suffix of } T\$$$
- **BWT Construction**:
  $$B[i] = \begin{cases} T[\text{SA}[i] - 1] & \text{if } \text{SA}[i] > 0 \\ \$ & \text{if } \text{SA}[i] = 0 \end{cases}$$
- **First Column ($F$) & Count Table ($C$)**:
  $$C[c] = \text{total characters in } T\$ \text{ strictly smaller than } c$$
- **Checkpointed Occurrence Table ($\text{Occ}(c, i)$)**:
  $$\text{Occ}(c, i) = \sum_{k=0}^{i-1} \mathbb{I}(B[k] == c)$$
  Checkpointed every $K$ rows (e.g. $K = 32$) with fast linear scan across small intervals for $O(1)$ rank queries with compact memory.
- **Ferragina-Manzini Backward Search**:
  For query pattern $P = p_0 p_1 \dots p_{m-1}$, search proceeds backwards from $p_{m-1}$ to $p_0$:
  $$\text{Start with } [l, r) \leftarrow [0, N+1)$$
  $$\text{For } k = m-1 \text{ down to } 0:$$
  $$l \leftarrow C[p_k] + \text{Occ}(p_k, l)$$
  $$r \leftarrow C[p_k] + \text{Occ}(p_k, r)$$
  $$\text{If } l \ge r, \text{ pattern does not occur in text.}$$
  Search complexity: $O(m)$ character lookups, completely independent of reference genome length $N$.
- **Sampled Suffix Array**:
  Store $\text{SA}[i]$ only when $\text{SA}[i] \equiv 0 \pmod{S_{\text{sa}}}$. When an exact match interval $[l, r)$ is found, walk backwards using the LF-mapping $LF(i) = C[B[i]] + \text{Occ}(B[i], i)$ until a sampled index is reached, reconstructing the genomic offset in $O(S_{\text{sa}})$ steps.

### 2.2 Dynamic Programming Pairwise Alignment with Gotoh Affine Gaps (`nucleocore/aligner.py`)
- **Affine Gap Model**:
  $$W(k) = \text{gap\_open} + (k-1) \cdot \text{gap\_extend}$$
- **Gotoh Three-Matrix Recurrence**:
  $$I_x(i, j) = \max \begin{cases} M(i-1, j) - \text{gap\_open} \\ I_x(i-1, j) - \text{gap\_extend} \end{cases}$$
  $$I_y(i, j) = \max \begin{cases} M(i, j-1) - \text{gap\_open} \\ I_y(i, j-1) - \text{gap\_extend} \end{cases}$$
  $$M(i, j) = \text{score}(S_1[i], S_2[j]) + \max \begin{cases} M(i-1, j-1) \\ I_x(i-1, j-1) \\ I_y(i-1, j-1) \end{cases}$$
- **Global Alignment (Needleman-Wunsch-Gotoh)**:
  - Boundary conditions: $M(i, 0) = -(\text{open} + (i-1)\text{extend})$, $M(0, j) = -(\text{open} + (j-1)\text{extend})$.
  - Traceback from $(N, M)$ to $(0, 0)$.
- **Local Alignment (Smith-Waterman-Gotoh)**:
  - $M(i, j) = \max(0, \text{Gotoh Recurrence})$.
  - Traceback starts from $\arg\max_{i, j} M(i, j)$ and terminates when score reaches 0.
- **CIGAR String Generation**:
  Emits standardized alignment operations: Match/Mismatch (`M`), Insertion (`I`), Deletion (`D`).

### 2.3 De Novo Genome Assembly via De Bruijn Graphs (`nucleocore/assembler.py`)
- **$k$-mer Graph**:
  - Vertices: All unique $(k-1)$-mers observed as prefixes and suffixes of $k$-mers in read dataset.
  - Directed Edges: $k$-mers connecting prefix $(k-1)$-mer to suffix $(k-1)$-mer with observed read count (coverage weight).
- **Topological Graph Simplification**:
  1. **Tip Clipping**: Traverses dead-end chains (in-degree 0 and out-degree 1, or out-degree 0 and in-degree 1). If chain length $< 2k$ and mean coverage is low, prune from graph.
  2. **Bubble Popping**: Identifies branching vertices where two paths diverge and reconnect at an identical target vertex within length $L \le 2k$. If path sequences have high similarity, collapse the lower-coverage alternate path into the dominant path.
- **Contig Synthesis**:
  - Maximal non-branching path traversal: Follows chains of nodes with in-degree 1 and out-degree 1.
  - Outputs assembled contigs and computes quality metrics: $N50$, $L50$, total length, contig count.

### 2.4 Profile Hidden Markov Models & Viterbi Decoding (`nucleocore/hmm.py`)
- **Profile HMM Topology**:
  - Sequence length $L$ with Match ($M_j$), Insert ($I_j$), and non-emitting Delete ($D_j$) states for $j \in [1, L]$.
  - Parameterized by transition probabilities $A_{u \to v}$ and emission probabilities $E_u(c)$.
- **Log-Space Viterbi Decoding**:
  - Transforms probabilities to log-space to prevent underflow: $a_{uv} = \ln A_{uv}, e_u(c) = \ln E_u(c)$.
  - Dynamic programming state maximization:
    $$V_u(t) = \max_v \left( V_v(t-1) + a_{vu} \right) + e_u(s_t)$$
  - Returns maximum-likelihood state sequence and log-odds score.
- **CpG Island Discovery HMM**:
  - Two-state model ($+$ island, $-$ background) with dinucleotide emission frequencies. Identifies high-density CG dinucleotide clusters in genomic DNA.

### 2.5 Sub-Pixel Unicode Braille Dot-Plot & Alignment Visualizer (`nucleocore/visualizer.py`)
- **ANSI Color Palette**:
  - Adenine ($A$): Emerald Green (`#2ecc71`)
  - Cytosine ($C$): Cobalt Blue (`#3498db`)
  - Guanine ($G$): Amber Gold (`#f39c12`)
  - Thymine ($T$): Crimson Red (`#e74c3c`)
- **Unicode Braille Dot-Plot Canvas**:
  - $2 \times 4$ dot matrix (`U+2800..U+28FF`) representing sequence homology comparison between two genomes.
  - Visualizes direct sequence identity, insertions (horizontal shifts), deletions (vertical shifts), and inversions (antiparallel diagonals).
- **Genomic Coverage Depth Track**:
  - Read mapping depth rendered as high-resolution Unicode bar chart (` `, `▂`, `▃`, `▄`, `▅`, `▆`, `▇`, `█`).

---

## 3. Directory Layout & Module Plan

```
projects/18-nucleocore/
├── nucleocore/
│   ├── __init__.py           # Unified public API exports
│   ├── types.py              # Nucleotides, DNA utilities, CIGAR, Scoring Matrix
│   ├── fm_index.py           # BWT, Suffix Array, LF-Mapping, Occ Table, Backward Search
│   ├── aligner.py            # Gotoh Affine Gap Needleman-Wunsch & Smith-Waterman
│   ├── assembler.py          # De Bruijn Graph, Tip Clipping, Bubble Popping, Contigs, N50
│   ├── hmm.py                # Profile HMM, Viterbi Decoding, CpG Island Detector
│   └── visualizer.py         # ANSI Colorizer, Braille Dot-Plot, Coverage Depth Track
├── tests/
│   ├── test_types.py         # Reverse complement, GC content, CIGAR parsing
│   ├── test_fm_index.py      # BWT, inverse BWT, Occ rank, exact backward search
│   ├── test_aligner.py       # Global/local Gotoh alignment, affine gap penalties
│   ├── test_assembler.py     # k-mer spectrum, DBG assembly, bubble popping, N50
│   ├── test_hmm.py           # Profile HMM construction, Viterbi state decoding
│   └── test_visualizer.py    # Braille dot-plot generation, ANSI color sequences
├── benchmarks/
│   └── bench_genomics.py     # Microbenchmarks: BWT search, Gotoh DP cells, DBG, Viterbi
├── examples/
│   └── genomics_lab.py       # Interactive lab: viral genome search, assembly, Braille plot
├── PLAN.md                   # Architectural specification
├── TASK_QUEUE.md             # Autonomous task tracking
├── CHECKPOINT_LAST.md        # State synchronization
└── README.md                 # System documentation
```

---

## 4. Verification & Done Criteria
1. Zero external dependencies (pure Python standard library only).
2. All 18+ unit tests passing cleanly with 100% mathematical and biological accuracy.
3. Microbenchmarks demonstrating high throughput (millions of base pairs searched/sec, fast DP cell updates).
4. Interactive terminal laboratory (`examples/genomics_lab.py`).
5. Seamless integration with master `showcase.py`.
