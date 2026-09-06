"""LogicCraft: EDA Logic Synthesis, Technology Mapping & Static Timing Analysis.

Pure Python 3.10+ implementation of an industrial-grade digital logic synthesis
and timing closure flow:
- Reduced Ordered Binary Decision Diagrams (ROBDD) with canonical form equivalence checking
- And-Inverter Graphs (AIG) with two-level structural hashing (strashing)
- Standard Cell Library & Liberty NLDM Timing Models with 2D bilinear interpolation
- Technology Mapping via Dynamic Programming Tree Covering (DAGON algorithm)
- Static Timing Analysis (STA) with Arrival Time (AT), Required Arrival Time (RAT), and Slack
- Sub-Pixel Unicode Braille Circuit Visualization and Timing HUD
"""

from .bdd import BDDManager, BDDNode
from .aig import AIGGraph, AIGNode, lit_node, lit_is_inv, lit_not, make_lit
from .liberty import (
    LibertyLibrary,
    StandardCell,
    TimingArc,
    LookupTable2D,
    get_default_library,
)
from .netlist import Netlist, CellInstance, Net
from .techmap import TechMapper, MatchResult
from .sta import StaticTimingAnalyzer, TimingReport, PathSegment
from .visualizer import BrailleCanvas, CircuitVisualizer

__all__ = [
    "BDDManager",
    "BDDNode",
    "AIGGraph",
    "AIGNode",
    "lit_node",
    "lit_is_inv",
    "lit_not",
    "make_lit",
    "LibertyLibrary",
    "StandardCell",
    "TimingArc",
    "LookupTable2D",
    "get_default_library",
    "Netlist",
    "CellInstance",
    "Net",
    "TechMapper",
    "MatchResult",
    "StaticTimingAnalyzer",
    "TimingReport",
    "PathSegment",
    "BrailleCanvas",
    "CircuitVisualizer",
]
