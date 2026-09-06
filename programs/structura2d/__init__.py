"""
Structura 2D: Standalone Desktop Finite Element Analysis & Continuum Mechanics Studio.
"""

from .fea_engine import (
    Node2D,
    TrussElement2D,
    CSTElement2D,
    Quad4Element2D,
    FEAModel,
    solve_linear_system
)
from .presets import PRESETS

__all__ = [
    "Node2D",
    "TrussElement2D",
    "CSTElement2D",
    "Quad4Element2D",
    "FEAModel",
    "solve_linear_system",
    "PRESETS"
]
