"""
StellarFusion: Magnetohydrodynamic (MHD) Plasma Equilibrium & Tokamak Confinement Engine.

A pure Python standard library simulation engine for toroidal plasma equilibria,
relativistic Boris particle dynamics, neoclassical banana orbits, Poincaré surface
puncture maps, safety factor evaluation, and sub-pixel Unicode Braille visualization.
"""

from .equilibrium import (
    ELEMENTARY_CHARGE,
    KILO_ELECTRON_VOLT,
    MU_0,
    EquilibriumProfile,
    Grid2D,
    GradShafranovSolver,
    SolovevEquilibrium,
)
from .magnetic import (
    MagneticField,
    MagneticFieldEvaluator,
    SafetyFactorCalculator,
)
from .particles import (
    BorisParticlePusher,
    ParticleSpecies,
    ParticleState,
    SpeciesProperties,
)
from .poincare import (
    PoincareFieldTracer,
    PoincarePuncture,
    ResonantMagneticPerturbation,
)
from .visualizer import (
    BrailleCanvas,
    TokamakVisualizer,
)

__version__ = "0.1.0"

__all__ = [
    "MU_0",
    "ELEMENTARY_CHARGE",
    "KILO_ELECTRON_VOLT",
    "Grid2D",
    "EquilibriumProfile",
    "SolovevEquilibrium",
    "GradShafranovSolver",
    "MagneticField",
    "MagneticFieldEvaluator",
    "SafetyFactorCalculator",
    "BorisParticlePusher",
    "ParticleSpecies",
    "ParticleState",
    "SpeciesProperties",
    "PoincareFieldTracer",
    "PoincarePuncture",
    "ResonantMagneticPerturbation",
    "BrailleCanvas",
    "TokamakVisualizer",
]
