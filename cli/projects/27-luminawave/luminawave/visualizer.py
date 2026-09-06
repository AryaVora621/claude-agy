"""Sub-Pixel Unicode Braille Electromagnetic Field Visualizer and Telemetry HUD.

Provides:
1. 2x4 dot sub-pixel Unicode Braille canvas (U+2800..U+28FF)
2. 24-bit TrueColor ANSI gradient rendering for positive/negative Ez electric fields
3. Dielectric structure overlay (waveguide core boundary shading)
4. Sub-pixel Braille transmission spectrum curve plotter
5. Interactive optical telemetry HUD
"""

from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

from .grid import Grid2D


class BrailleEMCanvas:
    """Sub-pixel 2x4 Unicode Braille canvas for electromagnetic wave plotting."""

    # Standard 2x4 Unicode Braille dot matrix bit offsets
    # Col 0: 0x01, 0x02, 0x04, 0x40
    # Col 1: 0x08, 0x10, 0x20, 0x80
    DOT_MAP = (
        (0x01, 0x08),
        (0x02, 0x10),
        (0x04, 0x20),
        (0x40, 0x80),
    )

    @staticmethod
    def render_field(
        grid: Grid2D,
        width_chars: int = 70,
        height_rows: int = 24,
        max_val: Optional[float] = None,
        show_structure: bool = True,
        use_color: bool = True,
    ) -> str:
        """Render 2D electric field Ez on a sub-pixel Unicode Braille canvas.

        Args:
            grid: 2D Yee grid containing Ez and material maps.
            width_chars: Target terminal display width in character cells.
            height_rows: Target terminal display height in rows.
            max_val: Optional field normalization limit (defaults to auto-peak).
            show_structure: If True, renders dielectric waveguide walls.
            use_color: If True, emits 24-bit TrueColor ANSI color escapes.

        Returns:
            Multi-line string formatted for terminal display.
        """
        # Determine dynamic peak field amplitude for contrast scaling
        if max_val is None or max_val <= 0.0:
            peak = max(abs(v) for v in grid.ez)
            max_val = max(1e-6, peak)

        # Each character cell represents a 2x4 sub-pixel block
        sub_w = width_chars * 2
        sub_h = height_rows * 4

        scale_x = grid.nx / sub_w
        scale_y = grid.ny / sub_h

        lines: List[str] = []

        # ANSI color codes
        RESET = "\033[0m" if use_color else ""

        for row in range(height_rows):
            line_chars: List[str] = []
            for col in range(width_chars):
                braille_code = 0
                avg_ez = 0.0
                has_dielectric = False
                active_subpixels = 0

                # Sample the 2x4 sub-pixel block
                for dy in range(4):
                    sub_y = row * 4 + dy
                    gy = min(grid.ny - 1, int(sub_y * scale_y))
                    for dx in range(2):
                        sub_x = col * 2 + dx
                        gx = min(grid.nx - 1, int(sub_x * scale_x))

                        idx = grid.idx(gx, gy)
                        val = grid.ez[idx]
                        norm = val / max_val
                        avg_ez += val

                        # Check dielectric structure
                        if grid.eps_r[idx] > 1.5:
                            has_dielectric = True

                        # Dot activates if absolute normalized amplitude exceeds threshold
                        if abs(norm) > 0.12:
                            braille_code |= BrailleEMCanvas.DOT_MAP[dy][dx]
                            active_subpixels += 1
                        elif show_structure and has_dielectric and (gx % 4 == 0 or gy % 4 == 0):
                            # Subtle outline dots for dielectric waveguide
                            braille_code |= BrailleEMCanvas.DOT_MAP[dy][dx]

                # If no dots activated, display empty braille or faint dielectric marker
                if braille_code == 0:
                    char = " " if not (show_structure and has_dielectric) else "·"
                else:
                    char = chr(0x2800 + braille_code)

                if use_color:
                    # Color based on dominant polarity and intensity
                    norm_avg = avg_ez / (8.0 * max_val)
                    intensity = min(1.0, abs(norm_avg) * 1.5)

                    if intensity < 0.05 and has_dielectric and show_structure:
                        # Gray dielectric waveguide core
                        color_code = "\033[38;2;90;90;120m"
                    elif norm_avg > 0:
                        # Positive Ez: Red to Bright Yellow
                        r = int(200 + 55 * intensity)
                        g = int(60 + 195 * intensity)
                        b = 30
                        color_code = f"\033[38;2;{r};{g};{b}m"
                    else:
                        # Negative Ez: Deep Blue to Bright Cyan
                        r = 30
                        g = int(80 + 175 * intensity)
                        b = int(200 + 55 * intensity)
                        color_code = f"\033[38;2;{r};{g};{b}m"

                    line_chars.append(f"{color_code}{char}{RESET}")
                else:
                    line_chars.append(char)

            lines.append("".join(line_chars))

        return "\n".join(lines)

    @staticmethod
    def plot_spectrum_sparkline(
        values: Sequence[float],
        width_chars: int = 60,
        height_rows: int = 6,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> str:
        """Plot frequency transmission spectrum curve using sub-pixel Braille dots.

        Args:
            values: Transmission values across frequency bins.
            width_chars: Display width in character columns.
            height_rows: Display height in character rows.
            min_val: Minimum y-axis value (e.g. 0.0 for 0% transmission).
            max_val: Maximum y-axis value (e.g. 1.0 for 100% transmission).

        Returns:
            Multi-line string formatted as a Braille curve.
        """
        if not values or len(values) < 2:
            return "(insufficient data for spectrum plot)"

        sub_w = width_chars * 2
        sub_h = height_rows * 4
        y_range = max(1e-12, max_val - min_val)

        # Allocate sub-pixel grid
        grid = [[0 for _ in range(sub_w)] for _ in range(sub_h)]

        # Map values across columns
        n_vals = len(values)
        for sx in range(sub_w):
            v_idx = min(n_vals - 1, int(sx * (n_vals - 1) / (sub_w - 1)))
            val = values[v_idx]
            norm_y = (val - min_val) / y_range
            sy = min(sub_h - 1, max(0, int(norm_y * (sub_h - 1))))
            # Invert y so 0 is at bottom
            grid[sub_h - 1 - sy][sx] = 1

        # Render Braille characters
        lines: List[str] = []
        for r in range(height_rows):
            row_chars: List[str] = []
            for c in range(width_chars):
                code = 0
                for dy in range(4):
                    for dx in range(2):
                        if grid[r * 4 + dy][c * 2 + dx]:
                            code |= BrailleEMCanvas.DOT_MAP[dy][dx]
                row_chars.append(chr(0x2800 + code) if code != 0 else " ")
            lines.append("".join(row_chars))

        return "\n".join(lines)


class PhotonicsWorkbenchHUD:
    """Consolidated terminal hardware visualizer and optical telemetry HUD."""

    def __init__(self, sim: Any) -> None:
        self.sim = sim

    def render(self, width_chars: int = 76, height_rows: int = 20) -> str:
        """Render complete terminal dashboard with HUD and field visualization."""
        sim = self.sim
        grid = sim.grid
        fs_time = sim.time * 1e15  # Femtoseconds
        max_ez, max_h = sim.get_max_fields()
        ue, uh, utotal = grid.total_energy()

        lines: List[str] = []
        lines.append("\033[1m\033[36m" + "=" * width_chars + "\033[0m")
        lines.append(
            f"\033[1m\033[36m  LUMINAWAVE 2D FDTD : COMPUTATIONAL SILICON PHOTONICS HUD\033[0m"
        )
        lines.append(
            f"  Step: {sim.step_count:06d} | Time: {fs_time:8.2f} fs | dt: {grid.dt*1e15:.3f} fs | CFL: {grid.courant_factor:.2f}"
        )
        lines.append("\033[1m\033[36m" + "=" * width_chars + "\033[0m")

        # Telemetry panel
        lines.append(
            f"| Peak |Ez|: {max_ez:9.3e} V/m | Peak |H|: {max_h:9.3e} A/m | Grid: {grid.nx}x{grid.ny} cells |"
        )
        lines.append(
            f"| Energy  : {utotal:9.3e} J/m (Electric: {ue:8.2e} J/m, Magnetic: {uh:8.2e} J/m) |"
        )
        lines.append("+" + "-" * (width_chars - 2) + "+")

        # Render 2D Braille electric field
        field_canvas = BrailleEMCanvas.render_field(
            grid, width_chars=width_chars - 2, height_rows=height_rows, use_color=True
        )
        for line in field_canvas.split("\n"):
            lines.append(f"|{line}|")

        lines.append("+" + "-" * (width_chars - 2) + "+")
        lines.append(
            f"  Palette: \033[38;2;255;100;30m+Ez (Red/Yellow)\033[0m | \033[38;2;30;120;255m-Ez (Blue/Cyan)\033[0m | \033[38;2;90;90;120mDielectric Core (Gray)\033[0m"
        )
        lines.append("\033[1m\033[36m" + "=" * width_chars + "\033[0m")

        return "\n".join(lines)
