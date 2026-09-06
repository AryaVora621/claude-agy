"""Relativitas - General Relativity Curved-Spacetime Geodesics & Black Hole Accretion Engine.

Pure Python 3.10+ standard library implementation of:
- Schwarzschild and Kerr curved spacetime metrics in Boyer-Lindquist coordinates
- Christoffel connection symbols and 4-momentum invariants
- 8-state first-order geodesic RK4 integrator for null and timelike particles
- Keplerian accretion disk kinematics, relativistic Doppler boosting, and gravitational redshift
- Backward curved-spacetime ray tracer for black hole shadows and lensed photon rings
- Sub-pixel Unicode Braille visualizer with 24-bit TrueColor ANSI output
"""

from relativitas.metric import (
    SpacetimeMetric,
    SchwarzschildMetric,
    KerrMetric,
)
from relativitas.geodesic import (
    GeodesicIntegrator,
    GeodesicState,
    GeodesicStatus,
    GeodesicResult,
    DiskIntersection,
)
from relativitas.accretion import (
    AccretionDisk,
    RadiativeTransferResult,
)
from relativitas.raytracer import (
    Camera,
    RayTracer,
    PixelClass,
    PixelResult,
    FrameBuffer,
)
from relativitas.visualizer import (
    BrailleCanvas,
    BlackHoleVisualizer,
)

__version__ = "1.0.0"
__all__ = [
    "SpacetimeMetric",
    "SchwarzschildMetric",
    "KerrMetric",
    "GeodesicIntegrator",
    "GeodesicState",
    "GeodesicStatus",
    "GeodesicResult",
    "DiskIntersection",
    "AccretionDisk",
    "RadiativeTransferResult",
    "Camera",
    "RayTracer",
    "PixelClass",
    "PixelResult",
    "FrameBuffer",
    "BrailleCanvas",
    "BlackHoleVisualizer",
]
