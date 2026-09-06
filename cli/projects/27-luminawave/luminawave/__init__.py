"""LuminaWave: Computational Electromagnetics, 2D FDTD Maxwell Solver & Silicon Photonics Engine."""

from .grid import (
    C0,
    EPSILON_0,
    ETA_0,
    MU_0,
    Grid2D,
    GridDimensions,
)
from .pml import PMLBoundary
from .sources import (
    InjectionMode,
    OpticalSource,
    PointSource,
    SourceWaveform,
    WaveguideModeSource,
)
from .fdtd import FDTDSimulator
from .photonics import (
    INDEX_AIR,
    INDEX_SILICON,
    INDEX_SILICA,
    INDEX_SI_NITRIDE,
    PhotonicCircuitBuilder,
    PhotonicPort,
)
from .monitors import (
    LineDFTMonitor,
    PointTimeMonitor,
    ResonanceMetrics,
    SParameterAnalyzer,
)
from .visualizer import (
    BrailleEMCanvas,
    PhotonicsWorkbenchHUD,
)

__all__ = [
    "C0",
    "EPSILON_0",
    "MU_0",
    "ETA_0",
    "Grid2D",
    "GridDimensions",
    "PMLBoundary",
    "SourceWaveform",
    "InjectionMode",
    "OpticalSource",
    "PointSource",
    "WaveguideModeSource",
    "FDTDSimulator",
    "INDEX_AIR",
    "INDEX_SILICON",
    "INDEX_SILICA",
    "INDEX_SI_NITRIDE",
    "PhotonicCircuitBuilder",
    "PhotonicPort",
    "PointTimeMonitor",
    "LineDFTMonitor",
    "ResonanceMetrics",
    "SParameterAnalyzer",
    "BrailleEMCanvas",
    "PhotonicsWorkbenchHUD",
]
