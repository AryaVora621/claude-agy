"""
NucleoCore: De Bruijn Graph (DBG) De Novo Genome Assembler.
Features k-mer spectrum extraction, directed De Bruijn graph construction,
topological tip clipping, bubble popping for heterozygous SNPs,
maximal non-branching path contig generation, and N50/L50 quality metrics.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from nucleocore.types import kmer_spectrum, canonical_kmer, gc_content


class DBGEdge:
    """Directed edge in De Bruijn graph representing an observed k-mer."""
    __slots__ = ("kmer", "coverage")

    def __init__(self, kmer: str, coverage: int = 1):
        self.kmer = kmer
        self.coverage = coverage


class DeBruijnGraph:
    """
    De Bruijn Graph (DBG) constructed from sequencing reads.
    Vertices are (k-1)-mers, directed edges are k-mers with coverage weights.
    """
    def __init__(self, k: int):
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        # Adjacency: node -> dict of {neighbor: DBGEdge}
        self.out_edges: Dict[str, Dict[str, DBGEdge]] = defaultdict(dict)
        self.in_edges: Dict[str, Dict[str, DBGEdge]] = defaultdict(dict)

    def add_kmer(self, kmer: str, count: int = 1) -> None:
        """Adds an observed k-mer into the graph, updating edge coverage."""
        if len(kmer) != self.k:
            raise ValueError(f"k-mer length must be {self.k}, got {len(kmer)}")
        u = kmer[:-1]
        v = kmer[1:]

        if v in self.out_edges[u]:
            self.out_edges[u][v].coverage += count
        else:
            edge = DBGEdge(kmer, count)
            self.out_edges[u][v] = edge
            self.in_edges[v][u] = edge

    def add_reads(self, reads: List[str]) -> None:
        """Populates graph with k-mers extracted from all input reads."""
        for read in reads:
            kmers = kmer_spectrum(read, self.k)
            for km in kmers:
                self.add_kmer(km)

    def remove_edge(self, u: str, v: str) -> None:
        """Removes directed edge u -> v from the graph."""
        if u in self.out_edges and v in self.out_edges[u]:
            del self.out_edges[u][v]
            if not self.out_edges[u]:
                del self.out_edges[u]
        if v in self.in_edges and u in self.in_edges[v]:
            del self.in_edges[v][u]
            if not self.in_edges[v]:
                del self.in_edges[v]

    def in_degree(self, node: str) -> int:
        return len(self.in_edges.get(node, {}))

    def out_degree(self, node: str) -> int:
        return len(self.out_edges.get(node, {}))

    def get_all_nodes(self) -> Set[str]:
        nodes = set(self.out_edges.keys())
        nodes.update(self.in_edges.keys())
        return nodes

    def prune_low_coverage(self, min_coverage: int = 2) -> int:
        """Prunes edges with coverage strictly below min_coverage."""
        pruned = 0
        to_remove = []
        for u, neighbors in list(self.out_edges.items()):
            for v, edge in list(neighbors.items()):
                if edge.coverage < min_coverage:
                    to_remove.append((u, v))
        for u, v in to_remove:
            self.remove_edge(u, v)
            pruned += 1
        return pruned

    def clip_tips(self, max_tip_length: Optional[int] = None) -> int:
        """
        Prunes dead-end tip paths of length <= max_tip_length.
        Tips are short dead-end branches caused by sequencing errors at read ends.
        """
        if max_tip_length is None:
            max_tip_length = 2 * self.k

        clipped = 0
        # Forward tips (nodes with out_degree == 0 and in_degree == 1)
        nodes_to_check = [node for node in self.get_all_nodes() if self.out_degree(node) == 0 and self.in_degree(node) == 1]

        for tip in nodes_to_check:
            path = [tip]
            curr = tip
            valid_tip = True
            for _ in range(max_tip_length):
                if self.in_degree(curr) == 1 and self.out_degree(curr) <= 1:
                    parent = next(iter(self.in_edges[curr].keys()))
                    if self.out_degree(parent) > 1:
                        # Found fork point!
                        path.append(parent)
                        break
                    path.append(parent)
                    curr = parent
                else:
                    valid_tip = False
                    break
            else:
                valid_tip = False

            if valid_tip and len(path) > 1:
                # Remove edges along tip path
                for i in range(len(path) - 1):
                    child = path[i]
                    parent = path[i + 1]
                    self.remove_edge(parent, child)
                    clipped += 1

        return clipped

    def pop_bubbles(self, max_bubble_length: Optional[int] = None) -> int:
        """
        Identifies and merges bubbles: two alternate paths between common source
        and target nodes with identical length and sequence similarity.
        Collapses lower-coverage path into dominant path.
        """
        if max_bubble_length is None:
            max_bubble_length = 2 * self.k

        popped = 0
        # Find nodes with out_degree >= 2
        for u in list(self.out_edges.keys()):
            neighbors = list(self.out_edges[u].keys())
            if len(neighbors) < 2:
                continue

            # Compare pairs of divergent branches
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    v1 = neighbors[i]
                    v2 = neighbors[j]

                    # Trace path 1
                    p1 = [u, v1]
                    curr1 = v1
                    while self.in_degree(curr1) == 1 and self.out_degree(curr1) == 1 and len(p1) <= max_bubble_length:
                        curr1 = next(iter(self.out_edges[curr1].keys()))
                        p1.append(curr1)

                    # Trace path 2
                    p2 = [u, v2]
                    curr2 = v2
                    while self.in_degree(curr2) == 1 and self.out_degree(curr2) == 1 and len(p2) <= max_bubble_length:
                        curr2 = next(iter(self.out_edges[curr2].keys()))
                        p2.append(curr2)

                    # Check if paths reconverge at the same node
                    if curr1 == curr2 and curr1 != u and len(p1) == len(p2):
                        # Calculate mean coverage of each path
                        cov1 = sum(self.out_edges[p1[k]][p1[k+1]].coverage for k in range(len(p1) - 1))
                        cov2 = sum(self.out_edges[p2[k]][p2[k+1]].coverage for k in range(len(p2) - 1))

                        # Remove the path with lower coverage
                        path_to_remove = p2 if cov1 >= cov2 else p1
                        for k in range(len(path_to_remove) - 1):
                            self.remove_edge(path_to_remove[k], path_to_remove[k+1])
                        popped += 1
                        break
        return popped

    def build_contigs(self) -> List[str]:
        """
        Assembles continuous contigs from maximal non-branching paths in the graph.
        """
        contigs: List[str] = []
        visited_edges: Set[Tuple[str, str]] = set()

        all_nodes = self.get_all_nodes()

        # 1. Start from branching or boundary nodes (in_degree != 1 or out_degree != 1)
        start_nodes = [node for node in all_nodes if not (self.in_degree(node) == 1 and self.out_degree(node) == 1)]

        for start in start_nodes:
            for neighbor in list(self.out_edges.get(start, {}).keys()):
                if (start, neighbor) in visited_edges:
                    continue

                # Walk forward along non-branching path
                path = [start, neighbor]
                visited_edges.add((start, neighbor))
                curr = neighbor

                while self.in_degree(curr) == 1 and self.out_degree(curr) == 1:
                    nxt = next(iter(self.out_edges[curr].keys()))
                    if (curr, nxt) in visited_edges:
                        break
                    visited_edges.add((curr, nxt))
                    path.append(nxt)
                    curr = nxt

                # Spell out contig sequence from node path
                seq = path[0] + "".join(node[-1] for node in path[1:])
                contigs.append(seq)

        # 2. Check for isolated cycles
        for u in all_nodes:
            for v in list(self.out_edges.get(u, {}).keys()):
                if (u, v) not in visited_edges:
                    path = [u, v]
                    visited_edges.add((u, v))
                    curr = v
                    while self.in_degree(curr) == 1 and self.out_degree(curr) == 1:
                        nxt = next(iter(self.out_edges[curr].keys()))
                        if (curr, nxt) in visited_edges:
                            break
                        visited_edges.add((curr, nxt))
                        path.append(nxt)
                        curr = nxt

                    seq = path[0] + "".join(node[-1] for node in path[1:])
                    contigs.append(seq)

        # Sort contigs by length descending
        contigs.sort(key=len, reverse=True)
        return contigs


class AssemblyStats:
    """Assembly quality metrics."""
    __slots__ = ("contig_count", "total_length", "max_length", "mean_length", "n50", "l50", "gc_fraction")

    def __init__(
        self,
        contig_count: int,
        total_length: int,
        max_length: int,
        mean_length: float,
        n50: int,
        l50: int,
        gc_fraction: float
    ):
        self.contig_count = contig_count
        self.total_length = total_length
        self.max_length = max_length
        self.mean_length = mean_length
        self.n50 = n50
        self.l50 = l50
        self.gc_fraction = gc_fraction

    def summary(self) -> str:
        return (
            f"Assembly Metrics:\n"
            f"  Contigs:      {self.contig_count}\n"
            f"  Total Length: {self.total_length:,} bp\n"
            f"  Max Contig:   {self.max_length:,} bp\n"
            f"  Mean Contig:  {self.mean_length:,.1f} bp\n"
            f"  N50:          {self.n50:,} bp\n"
            f"  L50:          {self.l50}\n"
            f"  GC Content:   {self.gc_fraction * 100:.2f}%"
        )


def calculate_assembly_stats(contigs: List[str]) -> AssemblyStats:
    """
    Computes standard genome assembly metrics: N50, L50, Max length, Total length, GC content.
    """
    if not contigs:
        return AssemblyStats(0, 0, 0, 0.0, 0, 0, 0.0)

    lengths = sorted([len(c) for c in contigs], reverse=True)
    total_len = sum(lengths)
    max_len = lengths[0]
    mean_len = total_len / len(lengths)

    # Calculate N50 and L50
    half_total = total_len / 2.0
    running_sum = 0
    n50 = lengths[0]
    l50 = 1
    for idx, l in enumerate(lengths, start=1):
        running_sum += l
        if running_sum >= half_total:
            n50 = l
            l50 = idx
            break

    # Aggregate GC content
    total_gc = sum(c.upper().count("G") + c.upper().count("C") for c in contigs)
    gc_frac = total_gc / total_len if total_len > 0 else 0.0

    return AssemblyStats(
        contig_count=len(contigs),
        total_length=total_len,
        max_length=max_len,
        mean_length=mean_len,
        n50=n50,
        l50=l50,
        gc_fraction=gc_frac
    )
