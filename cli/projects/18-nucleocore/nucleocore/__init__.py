"""
NucleoCore: Autonomous Computational Genomics & Sequence Assembly Engine.
Pure Python standard library implementation with zero external dependencies.
"""

from nucleocore.types import (
    reverse_complement,
    gc_content,
    kmer_spectrum,
    canonical_kmer,
    CIGAROpType,
    CIGARElement,
    CIGAR,
    ScoringMatrix,
)
from nucleocore.fm_index import (
    FMIndex,
    build_suffix_array,
)
from nucleocore.aligner import (
    AlignmentMode,
    AlignmentResult,
    align_pairwise,
)
from nucleocore.assembler import (
    DeBruijnGraph,
    DBGEdge,
    AssemblyStats,
    calculate_assembly_stats,
)
from nucleocore.hmm import (
    HiddenMarkovModel,
    ProfileHMM,
    create_cpg_island_detector,
    detect_cpg_islands,
)
from nucleocore.visualizer import (
    colorize_sequence,
    BrailleDotPlotCanvas,
    generate_dot_plot,
    render_coverage_depth_track,
    render_genomics_summary,
)

__all__ = [
    "reverse_complement",
    "gc_content",
    "kmer_spectrum",
    "canonical_kmer",
    "CIGAROpType",
    "CIGARElement",
    "CIGAR",
    "ScoringMatrix",
    "FMIndex",
    "build_suffix_array",
    "AlignmentMode",
    "AlignmentResult",
    "align_pairwise",
    "DeBruijnGraph",
    "DBGEdge",
    "AssemblyStats",
    "calculate_assembly_stats",
    "HiddenMarkovModel",
    "ProfileHMM",
    "create_cpg_island_detector",
    "detect_cpg_islands",
    "colorize_sequence",
    "BrailleDotPlotCanvas",
    "generate_dot_plot",
    "render_coverage_depth_track",
    "render_genomics_summary",
]
