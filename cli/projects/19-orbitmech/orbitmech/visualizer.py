"""
OrbitMech: 3D Camera Projection, Sub-Pixel Unicode Braille Orbit & Porkchop Visualizer.
Zero external dependencies. Renders orbital trajectories in TrueColor ANSI.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from orbitmech.types import (
    Vector3,
    StateVector,
    ClassicalOrbitalElements,
    EARTH,
)


class ANSI:
    """ANSI terminal styling sequences."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Celestial Palette
    SUN_GOLD = "\033[38;2;255;200;40m"
    EARTH_BLUE = "\033[38;2;60;140;240m"
    MARS_RUST = "\033[38;2;220;80;50m"
    JUPITER_AMBER = "\033[38;2;230;160;70m"
    MOON_SILVER = "\033[38;2;200;200;210m"
    ORBIT_CYAN = "\033[38;2;80;220;240m"
    TRANSFER_GREEN = "\033[38;2;60;230;120m"
    GRID_GRAY = "\033[38;2;70;80;95m"
    ALERT_RED = "\033[38;2;250;60;60m"
    STAR_WHITE = "\033[38;2;240;245;255m"


class BrailleCanvas:
    """
    Sub-pixel 2x4 Unicode Braille graphics canvas (U+2800..U+28FF).
    Provides 2x horizontal and 4x vertical resolution relative to terminal characters.
    """
    # Standard Braille dot bitmask offsets
    DOT_MASKS = [
        [0x01, 0x08],  # row 0: left (1), right (8)
        [0x02, 0x10],  # row 1: left (2), right (16)
        [0x04, 0x20],  # row 2: left (4), right (32)
        [0x40, 0x80],  # row 3: left (64), right (128)
    ]

    def __init__(self, char_width: int = 60, char_height: int = 24):
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        # 2D array of Braille bitmasks
        self.grid = [[0 for _ in range(char_width)] for _ in range(char_height)]
        # 2D array of ANSI color strings per character cell
        self.colors = [["" for _ in range(char_width)] for _ in range(char_height)]

    def set_pixel(self, px: int, py: int, color: str = "") -> None:
        """Turn on a single sub-pixel dot at (px, py)."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            cx = px // 2
            cy = py // 4
            dot_col = px % 2
            dot_row = py % 4
            self.grid[cy][cx] |= self.DOT_MASKS[dot_row][dot_col]
            if color:
                self.colors[cy][cx] = color

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: str = "") -> None:
        """Bresenham's integer line algorithm in sub-pixel coordinates."""
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
        """Render canvas into an array of formatted terminal character strings."""
        lines: List[str] = []
        for cy in range(self.char_height):
            line_parts: List[str] = []
            current_color = ""
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask)
                cell_color = self.colors[cy][cx]

                if cell_color != current_color:
                    if cell_color:
                        line_parts.append(cell_color)
                    else:
                        line_parts.append(ANSI.RESET)
                    current_color = cell_color

                line_parts.append(char)

            if current_color:
                line_parts.append(ANSI.RESET)
            lines.append("".join(line_parts))
        return lines


@dataclass
class Camera3D:
    """
    3D-to-2D viewing camera with azimuth and elevation perspective.
      azimuth_deg: View azimuth yaw angle around Z-axis in degrees
      elevation_deg: View elevation pitch angle above XY-plane in degrees
      scale: Physical distance per pixel (e.g. km/pixel or AU/pixel)
    """
    azimuth_deg: float = 35.0
    elevation_deg: float = 25.0
    scale: float = 250.0  # km per sub-pixel

    def project(
        self,
        point: Vector3,
        center_x: int,
        center_y: int,
    ) -> Tuple[int, int, float]:
        """
        Project 3D point (x, y, z) into 2D canvas pixel coordinates (px, py).
        Returns (px, py, depth_z).
        """
        psi = math.radians(self.azimuth_deg)
        theta = math.radians(self.elevation_deg)

        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # Camera frame transformation
        xc = -point.x * sin_psi + point.y * cos_psi
        yc = -point.x * cos_psi * sin_theta - point.y * sin_psi * sin_theta + point.z * cos_theta
        zc = point.x * cos_psi * cos_theta + point.y * sin_psi * cos_theta + point.z * sin_theta

        # Screen coordinates (Y inverted for terminal raster)
        px = int(center_x + (xc / self.scale))
        py = int(center_y - (yc / self.scale))

        return px, py, zc


def render_orbit_scene(
    trajectory: List[StateVector],
    central_body: Any = EARTH,
    camera: Optional[Camera3D] = None,
    canvas_chars_x: int = 64,
    canvas_chars_y: int = 22,
    orbit_color: str = ANSI.ORBIT_CYAN,
    show_reference_ring: bool = True,
) -> str:
    """
    Render 3D orbital trajectory into a sub-pixel Unicode Braille terminal view.
    """
    canvas = BrailleCanvas(char_width=canvas_chars_x, char_height=canvas_chars_y)
    center_px = canvas.pixel_width // 2
    center_py = canvas.pixel_height // 2

    # Auto-fit scale if not specified
    if camera is None:
        max_r = max((s.r.norm() for s in trajectory), default=8000.0)
        scale = (max_r * 1.25) / (min(center_px, center_py))
        camera = Camera3D(azimuth_deg=40.0, elevation_deg=25.0, scale=scale)

    # 1. Draw central body at center origin
    body_r_px = max(2, int(central_body.radius / camera.scale))
    for angle_deg in range(0, 360, 10):
        rad = math.radians(angle_deg)
        bx = int(center_px + body_r_px * math.cos(rad))
        by = int(center_py - body_r_px * math.sin(rad) * math.sin(math.radians(camera.elevation_deg + 45.0)))
        canvas.set_pixel(bx, by, ANSI.EARTH_BLUE)

    # 2. Draw equatorial reference ring (dotted)
    if show_reference_ring:
        ref_radius = central_body.radius * 1.5
        for deg in range(0, 360, 6):
            pt = Vector3(ref_radius * math.cos(math.radians(deg)), ref_radius * math.sin(math.radians(deg)), 0.0)
            px, py, _ = camera.project(pt, center_px, center_py)
            canvas.set_pixel(px, py, ANSI.GRID_GRAY)

    # 3. Draw trajectory line segments
    prev_px, prev_py = None, None
    for state in trajectory:
        px, py, _ = camera.project(state.r, center_px, center_py)
        if prev_px is not None:
            canvas.draw_line(prev_px, prev_py, px, py, orbit_color)
        prev_px, prev_py = px, py

    # 4. Mark current position with distinct star
    if trajectory:
        last_s = trajectory[-1]
        sc_px, sc_py, _ = camera.project(last_s.r, center_px, center_py)
        canvas.set_pixel(sc_px, sc_py, ANSI.STAR_WHITE)
        canvas.set_pixel(sc_px + 1, sc_py, ANSI.STAR_WHITE)
        canvas.set_pixel(sc_px - 1, sc_py, ANSI.STAR_WHITE)
        canvas.set_pixel(sc_px, sc_py + 1, ANSI.STAR_WHITE)
        canvas.set_pixel(sc_px, sc_py - 1, ANSI.STAR_WHITE)

    # Render lines with border frame
    lines = canvas.render()
    w = canvas_chars_x
    out: List[str] = []
    out.append(f"┌{'─' * w}┐")
    for row in lines:
        out.append(f"│{row}│")
    out.append(f"└{'─' * w}┘")
    return "\n".join(out)


def render_porkchop_contour(
    grid_matrix: List[List[float]],
    dep_days: List[float],
    arr_days: List[float],
    min_point_idx: Tuple[int, int],
    title: str = "INTERPLANETARY PORKCHOP PLOT (TOTAL ΔV km/s)",
) -> str:
    """
    Render 2D Porkchop launch opportunity contour grid in formatted ANSI terminal output.
    """
    rows = len(grid_matrix)
    cols = len(grid_matrix[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return "Empty Grid"

    # Find value range with realistic thresholding for launch window contours
    valid_vals = [val for r in grid_matrix for val in r if val < 90.0]
    min_val = min(valid_vals) if valid_vals else 0.0
    # Bound dynamic range around the basin of minimum Delta-V
    max_val = min(max(valid_vals) if valid_vals else 20.0, min_val + 5.0)

    # Gradient symbols: best to worst (low Delta-V to high Delta-V)
    GRADIENT_SYMBOLS = ["·", "░", "▒", "▓", "█"]

    out: List[str] = []
    out.append(f"\n{ANSI.BOLD}{title}{ANSI.RESET}")
    out.append(f"  Dep Days: {dep_days[0]:.0f} to {dep_days[-1]:.0f} | Arr Days: {arr_days[0]:.0f} to {arr_days[-1]:.0f}")
    out.append(f"  Optimal Target Marker: {ANSI.TRANSFER_GREEN}[★]{ANSI.RESET} (Min Val: {min_val:.2f} km/s)")
    out.append("  ┌" + "──" * cols + "┐")

    for i in range(rows):
        row_chars = []
        for j in range(cols):
            val = grid_matrix[i][j]
            if (i, j) == min_point_idx:
                row_chars.append(f"{ANSI.TRANSFER_GREEN}★ {ANSI.RESET}")
            elif val >= 90.0:
                row_chars.append(f"{ANSI.GRID_GRAY}··{ANSI.RESET}")
            else:
                ratio = (val - min_val) / max(1e-5, max_val - min_val)
                idx = min(len(GRADIENT_SYMBOLS) - 1, int(ratio * len(GRADIENT_SYMBOLS)))
                sym = GRADIENT_SYMBOLS[idx]
                if ratio < 0.25:
                    color = ANSI.TRANSFER_GREEN
                elif ratio < 0.55:
                    color = ANSI.ORBIT_CYAN
                elif ratio < 0.80:
                    color = ANSI.JUPITER_AMBER
                else:
                    color = ANSI.ALERT_RED
                row_chars.append(f"{color}{sym}{sym}{ANSI.RESET}")
        out.append(f"  │{''.join(row_chars)}│")

    out.append("  └" + "──" * cols + "┘\n")
    return "\n".join(out)


def render_mission_telemetry_hud(
    state: StateVector,
    elements: ClassicalOrbitalElements,
    central_body: Any = EARTH,
) -> str:
    """
    Format spacecraft instantaneous orbital telemetry HUD.
    """
    r_mag = state.radius
    v_mag = state.speed
    altitude = r_mag - central_body.radius
    specific_energy = elements.specific_energy
    period_str = f"{elements.period / 60.0:,.1f} min" if elements.period < 1e9 else "ESCAPE"

    hud = (
        f"┌───────────────────────── MISSION TELEMETRY HUD ─────────────────────────┐\n"
        f"│ Central Body:    {central_body.name:<12} Orbit Type:       {elements.orbit_type.value.upper():<16} │\n"
        f"│ Radial Altitude: {altitude:,.2f} km      Inertial Velocity: {v_mag:.4f} km/s           │\n"
        f"│ Semi-Major Axis: {elements.a:,.2f} km      Eccentricity:      {elements.e:.6f}           │\n"
        f"│ Inclination:     {math.degrees(elements.i):.3f}°            RAAN (Ω):          {math.degrees(elements.raan):.3f}°           │\n"
        f"│ Arg Periapsis:   {math.degrees(elements.arg_peri):.3f}°            True Anomaly:      {math.degrees(elements.true_anomaly):.3f}°           │\n"
        f"│ Orbital Period:  {period_str:<16} Mechanical Energy: {specific_energy:,.3f} km²/s²     │\n"
        f"└─────────────────────────────────────────────────────────────────────────┘"
    )
    return hud
