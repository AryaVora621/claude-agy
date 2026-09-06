"""
AeroFlow: Sub-Pixel Unicode Braille Flow Visualizer & Velocity Vector Renderer.
Zero external dependencies. Renders vorticity heatmaps, solid obstacles,
streamlines, and telemetry HUD in 24-bit TrueColor ANSI.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Any

from aeroflow.types import Grid2D, VectorField2D, ObstacleMask, Vector2D
from aeroflow.aerodynamics import AerodynamicForces


class ANSI:
    """ANSI terminal styling codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Obstacle styling
    OBSTACLE = "\033[38;2;160;170;185m\033[48;2;40;45;55m"

    # Vorticity Palette (Positive: CCW Vortices -> Cyan/Blue; Negative: CW Vortices -> Red/Orange)
    POS_CYAN = "\033[38;2;70;220;255m"
    POS_BLUE = "\033[38;2;30;130;250m"
    NEG_RED = "\033[38;2;255;70;70m"
    NEG_ORANGE = "\033[38;2;255;150;40m"
    STREAM_GREEN = "\033[38;2;80;240;130m"
    NEUTRAL_GRAY = "\033[38;2;80;90;105m"


class BrailleFlowCanvas:
    """
    Sub-pixel 2x4 Unicode Braille graphics canvas (U+2800..U+28FF).
    Provides 2x horizontal and 4x vertical resolution relative to terminal character cells.
    """
    DOT_MASKS = [
        [0x01, 0x08],  # row 0
        [0x02, 0x10],  # row 1
        [0x04, 0x20],  # row 2
        [0x40, 0x80],  # row 3
    ]

    def __init__(self, char_width: int = 70, char_height: int = 24):
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.grid = [[0 for _ in range(char_width)] for _ in range(char_height)]
        self.colors = [["" for _ in range(char_width)] for _ in range(char_height)]
        self.solid_cells = [[False for _ in range(char_width)] for _ in range(char_height)]

    def set_pixel(self, px: int, py: int, color: str = "") -> None:
        """Activate a single sub-pixel dot."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            cx = px // 2
            cy = py // 4
            dot_col = px % 2
            dot_row = py % 4
            self.grid[cy][cx] |= self.DOT_MASKS[dot_row][dot_col]
            if color:
                self.colors[cy][cx] = color

    def mark_solid(self, cx: int, cy: int) -> None:
        """Mark an entire character cell as a solid obstacle."""
        if 0 <= cx < self.char_width and 0 <= cy < self.char_height:
            self.solid_cells[cy][cx] = True

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: str = "") -> None:
        """Bresenham line drawing in sub-pixel coordinates."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def render(self) -> List[str]:
        """Render canvas to lines of ANSI terminal characters."""
        lines: List[str] = []
        for cy in range(self.char_height):
            parts: List[str] = []
            curr_color = ""

            for cx in range(self.char_width):
                if self.solid_cells[cy][cx]:
                    if curr_color != ANSI.OBSTACLE:
                        parts.append(ANSI.OBSTACLE)
                        curr_color = ANSI.OBSTACLE
                    parts.append("█")
                    continue

                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask) if mask > 0 else " "
                cell_color = self.colors[cy][cx]

                if cell_color != curr_color:
                    if cell_color:
                        parts.append(cell_color)
                    else:
                        parts.append(ANSI.RESET)
                    curr_color = cell_color

                parts.append(char)

            if curr_color:
                parts.append(ANSI.RESET)
            lines.append("".join(parts))
        return lines


def render_vorticity_field(
    vorticity: Grid2D,
    obstacle: ObstacleMask,
    char_width: int = 70,
    char_height: int = 24,
    vort_threshold: float = 0.005,
    streamlines: Optional[List[List[Tuple[float, float]]]] = None,
) -> str:
    """
    Render 2D vorticity field into a sub-pixel Unicode Braille terminal graphic.
    """
    canvas = BrailleFlowCanvas(char_width=char_width, char_height=char_height)
    nx, ny = vorticity.nx, vorticity.ny
    pw, ph = canvas.pixel_width, canvas.pixel_height

    # 1. Rasterize solid obstacles
    for cy in range(char_height):
        # Map character cell center to grid coordinates
        gy = int((cy + 0.5) * (ny / char_height))
        for cx in range(char_width):
            gx = int((cx + 0.5) * (nx / char_width))
            if obstacle.is_solid(gx, gy):
                canvas.mark_solid(cx, cy)

    # 2. Render sub-pixel vorticity dots
    for py in range(ph):
        gy = int((py + 0.5) * (ny / ph))
        for px in range(pw):
            gx = int((px + 0.5) * (nx / pw))

            if obstacle.is_solid(gx, gy):
                continue

            vort = vorticity.get(gx, gy)
            abs_vort = abs(vort)

            if abs_vort > vort_threshold:
                if vort > 0:
                    color = ANSI.POS_CYAN if abs_vort > 2.0 * vort_threshold else ANSI.POS_BLUE
                else:
                    color = ANSI.NEG_RED if abs_vort > 2.0 * vort_threshold else ANSI.NEG_ORANGE
                canvas.set_pixel(px, py, color)

    # 3. Render streamlines if provided
    if streamlines:
        for stream in streamlines:
            for i in range(len(stream) - 1):
                p0 = stream[i]
                p1 = stream[i + 1]
                px0 = int(p0[0] * (pw / nx))
                py0 = int(p0[1] * (ph / ny))
                px1 = int(p1[0] * (pw / nx))
                py1 = int(p1[1] * (ph / ny))
                canvas.draw_line(px0, py0, px1, py1, ANSI.STREAM_GREEN)

    # Wrap in framed box
    lines = canvas.render()
    out: List[str] = []
    out.append(f"┌{'─' * char_width}┐")
    for row in lines:
        out.append(f"│{row}│")
    out.append(f"└{'─' * char_width}┘")
    return "\n".join(out)


def render_velocity_vector_field(
    velocity: VectorField2D,
    obstacle: ObstacleMask,
    char_width: int = 70,
    char_height: int = 24,
) -> str:
    """
    Render directional velocity glyph arrows across downsampled terminal grid.
    """
    ARROW_GLYPHS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    nx, ny = velocity.nx, velocity.ny
    out: List[str] = []
    out.append(f"┌{'─' * char_width}┐")

    for cy in range(char_height):
        row_chars: List[str] = []
        gy = int((cy + 0.5) * (ny / char_height))

        for cx in range(char_width):
            gx = int((cx + 0.5) * (nx / char_width))

            if obstacle.is_solid(gx, gy):
                row_chars.append(f"{ANSI.OBSTACLE}█{ANSI.RESET}")
                continue

            u = velocity.u.get(gx, gy)
            v = velocity.v.get(gx, gy)
            speed = math.sqrt(u * u + v * v)

            if speed < 0.01:
                row_chars.append("·")
            else:
                # Calculate angle in degrees [0, 360)
                angle_deg = math.degrees(math.atan2(v, u)) % 360.0
                sector = int((angle_deg + 22.5) / 45.0) % 8
                glyph = ARROW_GLYPHS[sector]

                # Color by speed
                if speed > 0.12:
                    row_chars.append(f"{ANSI.POS_CYAN}{glyph}{ANSI.RESET}")
                elif speed > 0.06:
                    row_chars.append(f"{ANSI.STREAM_GREEN}{glyph}{ANSI.RESET}")
                else:
                    row_chars.append(f"{ANSI.NEUTRAL_GRAY}{glyph}{ANSI.RESET}")

        out.append(f"│{''.join(row_chars)}│")

    out.append(f"└{'─' * char_width}┘")
    return "\n".join(out)


def render_cfd_telemetry_hud(
    step: int,
    reynolds_number: float,
    forces: AerodynamicForces,
    solver_name: str = "D2Q9 LBM (BGK)",
    max_velocity: float = 0.0,
    enstrophy: float = 0.0,
) -> str:
    """
    Format CFD simulation telemetry HUD.
    """
    st_str = f"{forces.strouhal_number:.4f}" if forces.strouhal_number is not None else "TRANS"
    hud = (
        f"┌───────────────────────── CFD TELEMETRY HUD ─────────────────────────┐\n"
        f"│ Solver Model:    {solver_name:<16} Reynolds Number: Re = {reynolds_number:<10,.0f} │\n"
        f"│ Iteration Step:  {step:<16} Max Flow Speed:   U_max = {max_velocity:<8.4f} │\n"
        f"│ Drag Coeff (CD): {forces.drag_coeff:<16.4f} Lift Coeff (CL):   C_L = {forces.lift_coeff:<10.4f} │\n"
        f"│ Lift/Drag Ratio: {forces.lift_to_drag:<16.3f} Strouhal Number:  St  = {st_str:<10} │\n"
        f"│ Domain Enstrophy: {enstrophy:<15.4f} Vortex Shedding:   {'ACTIVE' if forces.strouhal_number else 'DEVELOPING':<10} │\n"
        f"└─────────────────────────────────────────────────────────────────────┘"
    )
    return hud
