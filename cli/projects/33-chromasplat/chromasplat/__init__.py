"""ChromaSplat: 3D Gaussian Splatting, Radiance Fields & Real-Time Volume Rendering Engine.

A high-performance, zero-dependency, pure Python 3.10+ standard library implementation
of 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) and volumetric radiance fields.
"""

from __future__ import annotations

from chromasplat.gaussian import Gaussian3D, Quaternion
from chromasplat.projection import Camera, ProjectedGaussian2D, ProjectionEngine
from chromasplat.rasterizer import RenderResult, VolumeRasterizer
from chromasplat.renderer import GaussianRenderer, RendererProfile
from chromasplat.scene import GaussianScene, PLYCodec, SceneFactory
from chromasplat.spherical_harmonics import (
    create_specular_sh,
    eval_sh,
    eval_sh_basis,
    rgb_to_sh_deg0,
    sh_deg0_to_rgb,
)
from chromasplat.visualizer import BrailleCanvas, SplatVisualizer

__version__ = "1.0.0"

__all__ = [
    "Gaussian3D",
    "Quaternion",
    "Camera",
    "ProjectedGaussian2D",
    "ProjectionEngine",
    "RenderResult",
    "VolumeRasterizer",
    "GaussianRenderer",
    "RendererProfile",
    "GaussianScene",
    "PLYCodec",
    "SceneFactory",
    "eval_sh",
    "eval_sh_basis",
    "rgb_to_sh_deg0",
    "sh_deg0_to_rgb",
    "create_specular_sh",
    "BrailleCanvas",
    "SplatVisualizer",
]
