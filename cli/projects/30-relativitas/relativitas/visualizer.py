"""Sub-Pixel Unicode Braille Visualizer & Telemetry HUD.

Renders curved spacetime black hole simulations and gravitational lensing to the terminal:
- 2x4 sub-pixel Unicode Braille dot matrix canvas (U+2800..U+28FF)
- 24-bit TrueColor ANSI escape colorization
- Gravitational redshift factor g and Doppler boosting colorimetry
- Real-time telemetry HUD displaying spacetime metrics, horizon radii, and ray tracing statistics
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

from relativitas.metric import SpacetimeMetric, KerrMetric, SchwarzschildMetric
from relativitas.raytracer import Camera, FrameBuffer, PixelClass, PixelResult


class BrailleCanvas:
    """2x4 sub-pixel dot matrix canvas utilizing Unicode Braille characters (U+2800..U+28FF)."""

    # Braille bit weights for the standard 2x4 grid:
    # (x=0, y=0): 0x01   (x=1, y=0): 0x08
    # (x=0, y=1): 0x02   (x=1, y=1): 0x10
    # (x=0, y=2): 0x04   (x=1, y=2): 0x20
    # (x=0, y=3): 0x40   (x=1, y=3): 0x80
    DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(self, char_width: int, char_height: int) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        # Dot activation grid and RGB color grid
        self.grid: List[List[int]] = [[0 for _ in range(char_width)] for _ in range(char_height)]
        self.colors: List[List[Tuple[int, int, int]]] = [
            [(0, 0, 0) for _ in range(char_width)] for _ in range(char_height)
        ]

    def set_pixel(self, px: int, py: int, rgb: Tuple[int, int, int], active: bool = True) -> None:
        """Set a single sub-pixel dot at high-resolution coordinate (px, py)."""
        if not (0 <= px < self.pixel_width and 0 <= py < self.pixel_height):
            return

        cx = px // 2
        cy = py // 4
        dx = px % 2
        dy = py % 4

        if active:
            self.grid[cy][cx] |= self.DOT_MAP[dy][dx]
            # Accumulate / blend color
            curr_r, curr_g, curr_b = self.colors[cy][cx]
            self.colors[cy][cx] = (
                max(curr_r, rgb[0]),
                max(curr_g, rgb[1]),
                max(curr_b, rgb[2]),
            )

    def render_ansi(self, true_color: bool = True) -> str:
        """Rasterize the Braille canvas to a formatted ANSI text string."""
        lines: List[str] = []
        for cy in range(self.char_height):
            row_chars: List[str] = []
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask) if mask > 0 else " "
                if true_color and mask > 0:
                    r, g, b = self.colors[cy][cx]
                    row_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    row_chars.append(char)
            lines.append("".join(row_chars))
        return "\n".join(lines)


class BlackHoleVisualizer:
    """High-level terminal renderer for black hole simulations and telemetry HUD."""

    def __init__(self, true_color: bool = True) -> None:
        self.true_color = true_color

    def rasterize_frame(self, frame: FrameBuffer) -> str:
        """Convert a ray-traced FrameBuffer into a sub-pixel Unicode Braille visualization."""
        # Frame dimensions match pixel resolution
        char_width = max(1, frame.width // 2)
        char_height = max(1, frame.height // 4)
        canvas = BrailleCanvas(char_width, char_height)

        for y in range(frame.height):
            for x in range(frame.width):
                p = frame.pixels[y][x]
                if p.pixel_class == PixelClass.BLACK_HOLE_SHADOW:
                    # Shadow is completely dark void
                    continue
                elif p.pixel_class == PixelClass.ACCRETION_DISK:
                    # Active glowing disk pixel
                    canvas.set_pixel(x, y, p.rgb, active=True)
                elif p.pixel_class == PixelClass.CELESTIAL_BACKGROUND:
                    # Render background stars if bright enough
                    if p.rgb[0] > 100:
                        canvas.set_pixel(x, y, p.rgb, active=True)

        return canvas.render_ansi(self.true_color)

    def render_ascii_intensity_map(self, frame: FrameBuffer) -> str:
        """Render frame using classical ASCII density glyphs based on observed intensity."""
        ramp = " .:-=+*#%@"
        lines: List[str] = []
        for y in range(frame.height):
            row_chars: List[str] = []
            for x in range(frame.width):
                p = frame.pixels[y][x]
                if p.pixel_class == PixelClass.BLACK_HOLE_SHADOW:
                    row_chars.append(" ")
                elif p.pixel_class == PixelClass.ACCRETION_DISK:
                    idx = int(min(len(ramp) - 1, max(1, p.redshift_g * 4.0)))
                    r, g, b = p.rgb
                    if self.true_color:
                        row_chars.append(f"\033[38;2;{r};{g};{b}m{ramp[idx]}\033[0m")
                    else:
                        row_chars.append(ramp[idx])
                else:
                    row_chars.append("." if p.rgb[0] > 100 else " ")
            lines.append("".join(row_chars))
        return "\n".join(lines)

    def format_telemetry_hud(
        self,
        metric: SpacetimeMetric,
        camera: Camera,
        frame: FrameBuffer,
    ) -> str:
        """Format an informative telemetry dashboard."""
        is_kerr = isinstance(metric, KerrMetric)
        spacetime_type = f"Kerr (Spin a/M={metric.spin:.2f})" if is_kerr else "Schwarzschild (Static)"
        r_h = metric.horizon_radius()
        r_isco = metric.isco_radius()

        r_ergo = metric.ergosphere_radius(camera.theta) if is_kerr else r_h

        shadow_pct = (frame.shadow_rays / max(1, frame.total_rays)) * 100.0
        disk_pct = (frame.disk_rays / max(1, frame.total_rays)) * 100.0
        escape_pct = (frame.escaped_rays / max(1, frame.total_rays)) * 100.0

        inc_deg = math.degrees(camera.theta)

        hud = [
            "+" + "-" * 78 + "+",
            f"| RELATIVITAS: GENERAL RELATIVITY BLACK HOLE ACCRETION ENGINE {' ' * 16}|",
            "+" + "-" * 78 + "+",
            f"| Spacetime Manifold: {spacetime_type:<24} Mass M: {metric.mass:.2f} G=c=1 {' ' * 13}|",
            f"| Event Horizon r_+: {r_h:.4f}M   Ergosphere r_ergo: {r_ergo:.4f}M   ISCO: {r_isco:.4f}M |",
            f"| Camera: r={camera.r:.1f}M  Inclination={inc_deg:.1f} deg  FOV={camera.fov_degrees:.1f} deg {' ' * 20}|",
            f"| Rays: Total={frame.total_rays}  Shadow={shadow_pct:.1f}%  Disk={disk_pct:.1f}%  Escape={escape_pct:.1f}% {' ' * 9}|",
            "+" + "-" * 78 + "+",
            "| Relativistic Color Spectral Doppler Shift:                                  |",
            "|  - [BLUE-WHITE] Approaching Gas (g > 1.2, Relativistic Beaming I ~ g^4)    |",
            "|  - [GOLDEN-YELLOW] Keplerian Rest Emission (g ~ 1.0)                        |",
            "|  - [CRIMSON-RED] Receding Gas / Gravitational Redshift (g < 0.8)             |",
            "|  - [DEEP VOID] Event Horizon Shadow                                         |",
            "+" + "-" * 78 + "+",
        ]
        return "\n".join(hud)
