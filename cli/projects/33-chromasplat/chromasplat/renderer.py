"""High-level 3D Gaussian Splatting rendering orchestrator.

Coordinates camera projection, view frustum culling, spherical harmonics directional
shading, tile-based spatial rasterization, and multi-view turntable animation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from chromasplat.projection import Camera, ProjectedGaussian2D, ProjectionEngine
from chromasplat.rasterizer import RenderResult, VolumeRasterizer
from chromasplat.scene import GaussianScene


@dataclass
class RendererProfile:
    """Performance telemetry for a single render frame.

    Attributes:
        num_scene_gaussians: Total primitives in scene.
        num_visible_gaussians: Primitives surviving frustum culling.
        projection_time_ms: Projection and EWA 2D covariance evaluation time in ms.
        rasterization_time_ms: Volume rasterization and compositing time in ms.
        total_time_ms: End-to-end frame time in ms.
        fps: Instantaneous frames per second throughput.
    """

    num_scene_gaussians: int
    num_visible_gaussians: int
    projection_time_ms: float
    rasterization_time_ms: float
    total_time_ms: float
    fps: float


class GaussianRenderer:
    """Orchestrator for 3D Gaussian Splatting rendering pipeline."""

    def __init__(
        self,
        sh_degree: int = 2,
        anti_aliasing_filter: float = 0.3,
        tile_size: int = 16,
        background_color: Tuple[float, float, float] = (0.04, 0.04, 0.06),
    ) -> None:
        """Initialize renderer.

        Args:
            sh_degree: Spherical harmonics degree for directional color (0 to 3).
            anti_aliasing_filter: Screen covariance low-pass filter variance.
            tile_size: Screen binning tile size in pixels.
            background_color: Ambient background RGB tuple in [0.0, 1.0].
        """
        self.sh_degree = max(0, min(3, sh_degree))
        self.projection_engine = ProjectionEngine(anti_aliasing_filter=anti_aliasing_filter)
        self.rasterizer = VolumeRasterizer(
            tile_size=tile_size,
            background_color=background_color,
        )
        self.last_profile: Optional[RendererProfile] = None

    def render(
        self,
        scene: GaussianScene,
        camera: Camera,
        use_tiling: bool = True,
    ) -> RenderResult:
        """Render a GaussianScene from the perspective of a Camera.

        Args:
            scene: Collection of 3D Gaussians.
            camera: Perspective pinhole camera with defined view parameters.
            use_tiling: Enable spatial tile binning.

        Returns:
            RenderResult containing color buffer, depth buffer, and timing telemetry.
        """
        t_start = time.perf_counter()

        # 1. Perspective projection & 2D covariance synthesis
        t_proj_0 = time.perf_counter()
        projected = self.projection_engine.project_scene(
            gaussians=scene.gaussians,
            camera=camera,
            sh_degree=self.sh_degree,
        )
        proj_time_ms = (time.perf_counter() - t_proj_0) * 1000.0

        # 2. Volume alpha compositing
        t_rast_0 = time.perf_counter()
        result = self.rasterizer.rasterize(
            width=camera.width,
            height=camera.height,
            projected_gaussians=projected,
            use_tiling=use_tiling,
        )
        rast_time_ms = (time.perf_counter() - t_rast_0) * 1000.0

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / max(0.01, total_time_ms)

        self.last_profile = RendererProfile(
            num_scene_gaussians=len(scene.gaussians),
            num_visible_gaussians=len(projected),
            projection_time_ms=proj_time_ms,
            rasterization_time_ms=rast_time_ms,
            total_time_ms=total_time_ms,
            fps=fps,
        )

        return result

    def render_turntable(
        self,
        scene: GaussianScene,
        width: int,
        height: int,
        num_frames: int = 8,
        distance: float = 3.5,
        elevation_deg: float = 20.0,
        fov_y_deg: float = 60.0,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> List[RenderResult]:
        """Render orbital 360-degree rotation animation frames around scene target."""
        frames: List[RenderResult] = []
        for f in range(num_frames):
            azimuth = (360.0 / num_frames) * f
            cam = Camera.orbit(
                width=width,
                height=height,
                azimuth_deg=azimuth,
                elevation_deg=elevation_deg,
                distance=distance,
                target=target,
                fov_y_deg=fov_y_deg,
            )
            res = self.render(scene, cam)
            frames.append(res)
        return frames
