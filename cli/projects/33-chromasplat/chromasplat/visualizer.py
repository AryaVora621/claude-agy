"""Sub-pixel Unicode Braille visualizer and 24-bit TrueColor telemetry HUD.

Translates floating point 3D Gaussian radiance fields into 2x4 sub-pixel
Unicode Braille terminal art (U+2800..U+28FF) with 24-bit ANSI RGB color
shading and comprehensive real-time graphics telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from chromasplat.projection import Camera
from chromasplat.rasterizer import RenderResult
from chromasplat.renderer import RendererProfile
from chromasplat.scene import GaussianScene

# ANSI TrueColor terminal formatting escapes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_GREEN = "\033[32m"
C_MAGENTA = "\033[35m"

# 2x4 Braille sub-pixel dot bit offsets
BRAILLE_DOT_MAP = (
    (0x01, 0x08),  # dy = 0: (x=0 -> bit 0, x=1 -> bit 3)
    (0x02, 0x10),  # dy = 1: (x=0 -> bit 1, x=1 -> bit 4)
    (0x04, 0x20),  # dy = 2: (x=0 -> bit 2, x=1 -> bit 5)
    (0x40, 0x80),  # dy = 3: (x=0 -> bit 6, x=1 -> bit 7)
)


def truecolor_fg(r: int, g: int, b: int) -> str:
    """Generate ANSI escape sequence for 24-bit TrueColor foreground."""
    cr = max(0, min(255, r))
    cg = max(0, min(255, g))
    cb = max(0, min(255, b))
    return f"\033[38;2;{cr};{cg};{cb}m"


class BrailleCanvas:
    """Sub-pixel terminal canvas where each character represents a 2x4 pixel block."""

    def __init__(self, char_width: int, char_height: int) -> None:
        """Initialize canvas with character cell dimensions.

        Args:
            char_width: Width in terminal character columns.
            char_height: Height in terminal character rows.
        """
        self.char_width = max(10, char_width)
        self.char_height = max(5, char_height)
        self.pixel_width = self.char_width * 2
        self.pixel_height = self.char_height * 4

    def rasterize_render_result(
        self,
        render_result: RenderResult,
        luminance_threshold: float = 0.06,
    ) -> str:
        """Convert RenderResult pixel buffer into ANSI TrueColor Braille string.

        Args:
            render_result: RenderResult from GaussianRenderer.
            luminance_threshold: Minimum luminance (0.0 to 1.0) to activate a sub-pixel dot.

        Returns:
            Formatted multiline ANSI string with Braille characters and TrueColor escapes.
        """
        buf_w = render_result.width
        buf_h = render_result.height
        lines: List[str] = []

        for cy in range(self.char_height):
            line_chunks: List[str] = []
            py_base = cy * 4

            for cx in range(self.char_width):
                px_base = cx * 2

                braille_code = 0
                active_dots = 0
                sum_r = 0.0
                sum_g = 0.0
                sum_b = 0.0

                # Sample 2x4 sub-pixel neighborhood
                for dy in range(4):
                    py = py_base + dy
                    if py >= buf_h:
                        continue

                    for dx in range(2):
                        px = px_base + dx
                        if px >= buf_w:
                            continue

                        # Sample color and compute relative luminance
                        r, g, b = render_result.colors[py][px]
                        lum = 0.299 * r + 0.587 * g + 0.114 * b

                        if lum >= luminance_threshold:
                            braille_code |= BRAILLE_DOT_MAP[dy][dx]
                            active_dots += 1
                            sum_r += r
                            sum_g += g
                            sum_b += b

                if active_dots > 0:
                    inv = 1.0 / active_dots
                    avg_r = int(sum_r * inv * 255.0 + 0.5)
                    avg_g = int(sum_g * inv * 255.0 + 0.5)
                    avg_b = int(sum_b * inv * 255.0 + 0.5)

                    char = chr(0x2800 + braille_code)
                    line_chunks.append(f"{truecolor_fg(avg_r, avg_g, avg_b)}{char}")
                else:
                    # Empty cell rendered as space
                    line_chunks.append(" ")

            line_chunks.append(C_RESET)
            lines.append("".join(line_chunks))

        return "\n".join(lines)


class SplatVisualizer:
    """Combines sub-pixel Braille rendering with real-time HUD telemetry."""

    def __init__(self, char_width: int = 80, char_height: int = 30) -> None:
        """Initialize visualizer.

        Args:
            char_width: Screen width in terminal character columns.
            char_height: Screen height in terminal character rows.
        """
        self.canvas = BrailleCanvas(char_width=char_width, char_height=char_height)

    @property
    def pixel_width(self) -> int:
        """Required framebuffer width for 2x4 sub-pixel sampling."""
        return self.canvas.pixel_width

    @property
    def pixel_height(self) -> int:
        """Required framebuffer height for 2x4 sub-pixel sampling."""
        return self.canvas.pixel_height

    def render_frame_with_hud(
        self,
        scene: GaussianScene,
        camera: Camera,
        render_result: RenderResult,
        profile: Optional[RendererProfile] = None,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        distance: float = 3.0,
    ) -> str:
        """Render complete interactive frame including telemetry header and footer.

        Args:
            scene: Active GaussianScene.
            camera: Active camera model.
            render_result: Rasterized pixel result.
            profile: Optional renderer performance telemetry.
            azimuth_deg: Camera orbit azimuth in degrees.
            elevation_deg: Camera orbit elevation in degrees.
            distance: Camera distance in meters.

        Returns:
            Complete ANSI formatted terminal visualizer string.
        """
        cw = self.canvas.char_width
        separator = "=" * cw
        sub_sep = "-" * cw

        # Header HUD
        title = "CHROMASPLAT: 3D GAUSSIAN SPLATTING & RADIANCE FIELD ENGINE"
        header_lines = [
            f"{C_BOLD}{C_CYAN}{separator}{C_RESET}",
            f"{C_BOLD}{title:^{cw}}{C_RESET}",
            f"{C_DIM}{sub_sep}{C_RESET}",
            (
                f"{C_BOLD}Scene:{C_RESET} {scene.name:<18} "
                f"{C_BOLD}Gaussians:{C_RESET} {render_result.num_gaussians}/{len(scene)} "
                f"{C_BOLD}Resolution:{C_RESET} {self.pixel_width}x{self.pixel_height} ({cw}x{self.canvas.char_height})"
            ),
            (
                f"{C_BOLD}Camera Orbit:{C_RESET} Azimuth: {azimuth_deg:5.1f} deg | "
                f"Elevation: {elevation_deg:5.1f} deg | Distance: {distance:4.2f} m | "
                f"FOV: {math.degrees(camera.fov_y_rad):4.1f} deg"
            ),
            f"{C_DIM}{sub_sep}{C_RESET}",
        ]

        # Canvas rasterization
        canvas_str = self.canvas.rasterize_render_result(render_result)

        # Performance & Telemetry Footer
        total_ms = profile.total_time_ms if profile else render_result.render_time_ms
        fps = profile.fps if profile else (1000.0 / max(0.01, total_ms))
        proj_ms = profile.projection_time_ms if profile else 0.0
        rast_ms = profile.rasterization_time_ms if profile else render_result.render_time_ms

        footer_lines = [
            f"{C_DIM}{sub_sep}{C_RESET}",
            (
                f"{C_BOLD}{C_GREEN}FPS:{C_RESET} {fps:5.1f}  "
                f"{C_BOLD}Total Frame:{C_RESET} {total_ms:5.2f} ms  "
                f"{C_BOLD}Projection:{C_RESET} {proj_ms:4.2f} ms  "
                f"{C_BOLD}Rasterization:{C_RESET} {rast_ms:4.2f} ms"
            ),
            (
                f"{C_DIM}Features: EWA Perspective Jacobian | SH Directional Radiance | "
                f"Tile Binned Compositing | Sub-Pixel Braille{C_RESET}"
            ),
            f"{C_BOLD}{C_CYAN}{separator}{C_RESET}",
        ]

        full_display = "\n".join(header_lines) + "\n" + canvas_str + "\n" + "\n".join(footer_lines)
        return full_display
