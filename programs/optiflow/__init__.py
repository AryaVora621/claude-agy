"""
OptiFlow 2D Package
Zero external dependencies. Pure Python standard library.
"""

from .cfd_engine import CFDEngine2D, Particle
from .presets import AERO_PRESETS

__all__ = [
    "CFDEngine2D",
    "Particle",
    "AERO_PRESETS"
]
