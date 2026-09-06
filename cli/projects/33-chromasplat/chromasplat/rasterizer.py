"""Tiled and depth-sorted alpha-compositing volume rasterizer.

Implements front-to-back volume rendering with 2D Gaussian Mahalanobis quadratic
evaluation, spatial tile binning, ray transmittance tracking, and early ray
saturation termination.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from chromasplat.projection import ProjectedGaussian2D


@dataclass
class RenderResult:
    """Output image and auxiliary buffers produced by volume rasterizer.

    Attributes:
        width: Screen width in pixels.
        height: Screen height in pixels.
        colors: 2D array of (r, g, b) floating point values in [0.0, 1.0], indexed [y][x].
        depths: 2D array of camera-space depth values in meters, indexed [y][x].
        transmittances: 2D array of remaining ray transmittance in [0.0, 1.0], indexed [y][x].
        num_gaussians: Number of visible Gaussians processed.
        render_time_ms: Wall-clock rasterization execution time in milliseconds.
    """

    width: int
    height: int
    colors: List[List[Tuple[float, float, float]]]
    depths: List[List[float]]
    transmittances: List[List[float]]
    num_gaussians: int
    render_time_ms: float

    def get_pixel_rgb24(self, x: int, y: int) -> Tuple[int, int, int]:
        """Return 8-bit integer RGB tuple in [0, 255] for pixel (x, y)."""
        clamped_x = max(0, min(self.width - 1, x))
        clamped_y = max(0, min(self.height - 1, y))
        r, g, b = self.colors[clamped_y][clamped_x]
        ir = max(0, min(255, int(r * 255.0 + 0.5)))
        ig = max(0, min(255, int(g * 255.0 + 0.5)))
        ib = max(0, min(255, int(b * 255.0 + 0.5)))
        return (ir, ig, ib)

    def to_ppm_bytes(self) -> bytes:
        """Export rendered color buffer as binary PPM (P6) image format."""
        header = f"P6\n{self.width} {self.height}\n255\n".encode("ascii")
        pixel_data = bytearray(self.width * self.height * 3)
        ptr = 0
        for y in range(self.height):
            for x in range(self.width):
                r, g, b = self.get_pixel_rgb24(x, y)
                pixel_data[ptr] = r
                pixel_data[ptr + 1] = g
                pixel_data[ptr + 2] = b
                ptr += 3
        return header + bytes(pixel_data)


class VolumeRasterizer:
    """Front-to-back volumetric radiance rasterizer for 2D projected Gaussians."""

    def __init__(
        self,
        tile_size: int = 16,
        early_exit_transmittance: float = 1e-4,
        background_color: Tuple[float, float, float] = (0.05, 0.05, 0.07),
    ) -> None:
        """Initialize volume rasterizer.

        Args:
            tile_size: Square tile dimension in pixels for screen binning.
            early_exit_transmittance: Ray transmittance threshold below which
                accumulation terminates (early ray termination).
            background_color: Ambient background RGB tuple in [0.0, 1.0].
        """
        self.tile_size = max(4, tile_size)
        self.t_min = early_exit_transmittance
        self.bg_color = background_color

    def rasterize(
        self,
        width: int,
        height: int,
        projected_gaussians: Sequence[ProjectedGaussian2D],
        use_tiling: bool = True,
    ) -> RenderResult:
        """Rasterize sorted 2D Gaussians to color and depth buffers.

        Args:
            width: Framebuffer width in pixels.
            height: Framebuffer height in pixels.
            projected_gaussians: Sequence of projected 2D Gaussians, sorted front-to-back.
            use_tiling: If True, uses spatial 2D tile binning.

        Returns:
            RenderResult containing color buffer, depth buffer, and timing telemetry.
        """
        t0 = time.perf_counter()

        # Initialize accumulation buffers
        # Pre-allocate 1D flat buffers for cache efficiency, reshape into 2D at return
        num_pixels = width * height
        buf_r = [0.0] * num_pixels
        buf_g = [0.0] * num_pixels
        buf_b = [0.0] * num_pixels
        buf_d = [0.0] * num_pixels
        buf_t = [1.0] * num_pixels  # Transmittance starts at 1.0 (fully transparent)

        if use_tiling and len(projected_gaussians) > 8:
            self._rasterize_tiled(
                width,
                height,
                projected_gaussians,
                buf_r,
                buf_g,
                buf_b,
                buf_d,
                buf_t,
            )
        else:
            self._rasterize_direct(
                width,
                height,
                projected_gaussians,
                buf_r,
                buf_g,
                buf_b,
                buf_d,
                buf_t,
            )

        # Composite background color onto remaining transmittance
        bg_r, bg_g, bg_b = self.bg_color
        for i in range(num_pixels):
            t_rem = buf_t[i]
            if t_rem > 0.0:
                buf_r[i] += bg_r * t_rem
                buf_g[i] += bg_g * t_rem
                buf_b[i] += bg_b * t_rem

            # Normalize depth by accumulated opacity (1 - T)
            opacity_acc = 1.0 - t_rem
            if opacity_acc > 1e-4:
                buf_d[i] /= opacity_acc

        # Reshape 1D buffers into 2D rows for clean user consumption
        colors: List[List[Tuple[float, float, float]]] = []
        depths: List[List[float]] = []
        transmittances: List[List[float]] = []

        for y in range(height):
            row_start = y * width
            row_end = row_start + width
            row_colors = [
                (
                    max(0.0, min(1.0, buf_r[idx])),
                    max(0.0, min(1.0, buf_g[idx])),
                    max(0.0, min(1.0, buf_b[idx])),
                )
                for idx in range(row_start, row_end)
            ]
            colors.append(row_colors)
            depths.append(buf_d[row_start:row_end])
            transmittances.append(buf_t[row_start:row_end])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RenderResult(
            width=width,
            height=height,
            colors=colors,
            depths=depths,
            transmittances=transmittances,
            num_gaussians=len(projected_gaussians),
            render_time_ms=elapsed_ms,
        )

    def _rasterize_direct(
        self,
        width: int,
        height: int,
        gaussians: Sequence[ProjectedGaussian2D],
        buf_r: List[float],
        buf_g: List[float],
        buf_b: List[float],
        buf_d: List[float],
        buf_t: List[float],
    ) -> None:
        """Direct bounding-box evaluation across pixels."""
        t_min = self.t_min

        for g in gaussians:
            u, v = g.u, g.v
            rad = g.radius
            x_min = max(0, int(u - rad))
            x_max = min(width - 1, int(u + rad))
            y_min = max(0, int(v - rad))
            y_max = min(height - 1, int(v + rad))

            if x_min > x_max or y_min > y_max:
                continue

            inv_a = g.inv_cov_a
            inv_b = g.inv_cov_b
            inv_c = g.inv_cov_c
            cr, cg, cb = g.color
            depth = g.depth
            base_opacity = g.opacity

            for py in range(y_min, y_max + 1):
                row_offset = py * width
                dy = py - v
                dy_term = inv_c * dy * dy
                two_inv_b_dy = 2.0 * inv_b * dy

                for px in range(x_min, x_max + 1):
                    pix_idx = row_offset + px
                    t_curr = buf_t[pix_idx]
                    if t_curr < t_min:
                        continue

                    dx = px - u
                    # Quadratic form: delta^T * Sigma_2D^-1 * delta
                    mahalanobis_sq = inv_a * dx * dx + two_inv_b_dy * dx + dy_term
                    if mahalanobis_sq > 9.0:
                        # Outside 3-sigma confidence ellipse
                        continue

                    # Gaussian spatial evaluation
                    weight_gauss = math.exp(-0.5 * mahalanobis_sq)
                    alpha = min(0.99, base_opacity * weight_gauss)
                    if alpha < 0.0039:  # < 1/255
                        continue

                    # Front-to-back alpha compositing
                    weight_comp = alpha * t_curr
                    buf_r[pix_idx] += cr * weight_comp
                    buf_g[pix_idx] += cg * weight_comp
                    buf_b[pix_idx] += cb * weight_comp
                    buf_d[pix_idx] += depth * weight_comp
                    buf_t[pix_idx] = t_curr * (1.0 - alpha)

    def _rasterize_tiled(
        self,
        width: int,
        height: int,
        gaussians: Sequence[ProjectedGaussian2D],
        buf_r: List[float],
        buf_g: List[float],
        buf_b: List[float],
        buf_d: List[float],
        buf_t: List[float],
    ) -> None:
        """Spatial tile-binned rasterization."""
        ts = self.tile_size
        num_tiles_x = (width + ts - 1) // ts
        num_tiles_y = (height + ts - 1) // ts
        num_tiles = num_tiles_x * num_tiles_y

        # Assign Gaussian indices to intersecting tiles
        tile_bins: List[List[ProjectedGaussian2D]] = [[] for _ in range(num_tiles)]

        for g in gaussians:
            u, v = g.u, g.v
            rad = g.radius

            tx_min = max(0, int((u - rad) // ts))
            tx_max = min(num_tiles_x - 1, int((u + rad) // ts))
            ty_min = max(0, int((v - rad) // ts))
            ty_max = min(num_tiles_y - 1, int((v + rad) // ts))

            for ty in range(ty_min, ty_max + 1):
                ty_offset = ty * num_tiles_x
                for tx in range(tx_min, tx_max + 1):
                    tile_bins[ty_offset + tx].append(g)

        # Process each tile independently
        t_min = self.t_min

        for ty in range(num_tiles_y):
            tile_y_start = ty * ts
            tile_y_end = min(height, tile_y_start + ts)

            for tx in range(num_tiles_x):
                tile_idx = ty * num_tiles_x + tx
                bin_gaussians = tile_bins[tile_idx]
                if not bin_gaussians:
                    continue

                tile_x_start = tx * ts
                tile_x_end = min(width, tile_x_start + ts)

                for g in bin_gaussians:
                    u, v = g.u, g.v
                    rad = g.radius

                    px_min = max(tile_x_start, int(u - rad))
                    px_max = min(tile_x_end - 1, int(u + rad))
                    py_min = max(tile_y_start, int(v - rad))
                    py_max = min(tile_y_end - 1, int(v + rad))

                    if px_min > px_max or py_min > py_max:
                        continue

                    inv_a = g.inv_cov_a
                    inv_b = g.inv_cov_b
                    inv_c = g.inv_cov_c
                    cr, cg, cb = g.color
                    depth = g.depth
                    base_opacity = g.opacity

                    for py in range(py_min, py_max + 1):
                        row_offset = py * width
                        dy = py - v
                        dy_term = inv_c * dy * dy
                        two_inv_b_dy = 2.0 * inv_b * dy

                        for px in range(px_min, px_max + 1):
                            pix_idx = row_offset + px
                            t_curr = buf_t[pix_idx]
                            if t_curr < t_min:
                                continue

                            dx = px - u
                            mahalanobis_sq = inv_a * dx * dx + two_inv_b_dy * dx + dy_term
                            if mahalanobis_sq > 9.0:
                                continue

                            weight_gauss = math.exp(-0.5 * mahalanobis_sq)
                            alpha = min(0.99, base_opacity * weight_gauss)
                            if alpha < 0.0039:
                                continue

                            weight_comp = alpha * t_curr
                            buf_r[pix_idx] += cr * weight_comp
                            buf_g[pix_idx] += cg * weight_comp
                            buf_b[pix_idx] += cb * weight_comp
                            buf_d[pix_idx] += depth * weight_comp
                            buf_t[pix_idx] = t_curr * (1.0 - alpha)
