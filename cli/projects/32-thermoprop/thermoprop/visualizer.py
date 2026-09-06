"""
Sub-Pixel Unicode Braille Rocket Nozzle Visualizer & Propulsion HUD.

Renders high-resolution 2D De Laval nozzle geometries, supersonic exhaust plumes,
oblique shock reflections, and Mach diamond shock cells using a 2x4 sub-pixel
Unicode Braille dot matrix (U+2800..U+28FF) with 24-bit TrueColor ANSI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .cooling import ThermalStationResult
from .moc_nozzle import NozzleContour
from .plume import PlumeStructure
from .propulsion import RocketEngineState


# ANSI formatting constants
RESET_SEQ: str = "\033[0m"
BOLD_SEQ: str = "\033[1m"


def truecolor_fg(r: int, g: int, b: int) -> str:
    """ANSI 24-bit TrueColor foreground escape sequence."""
    return f"\033[38;2;{r};{g};{b}m"


class BrailleCanvas:
    """
    Sub-pixel dot matrix canvas using Unicode Braille Patterns (U+2800..U+28FF).
    """

    DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(
        self,
        char_width: int = 76,
        char_height: int = 30,
        x_min: float = -1.5,
        x_max: float = 12.0,
        y_min: float = -3.5,
        y_max: float = 3.5,
    ) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.grid: List[List[int]] = [[0 for _ in range(char_width)] for _ in range(char_height)]
        self.color_grid: List[List[Tuple[int, int, int]]] = [
            [(160, 160, 170) for _ in range(char_width)] for _ in range(char_height)
        ]

    def clear(self) -> None:
        """Clear all canvas sub-pixels and reset colors."""
        for cy in range(self.char_height):
            for cx in range(self.char_width):
                self.grid[cy][cx] = 0
                self.color_grid[cy][cx] = (160, 160, 170)

    def coord_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert physical coordinates (x, y) to canvas pixels (px, py)."""
        norm_x = (x - self.x_min) / (self.x_max - self.x_min)
        norm_y = (self.y_max - y) / (self.y_max - self.y_min)

        px = int(norm_x * (self.pixel_width - 1))
        py = int(norm_y * (self.pixel_height - 1))
        return px, py

    def set_pixel(
        self,
        px: int,
        py: int,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Set an individual sub-pixel dot with optional TrueColor RGB."""
        if px < 0 or px >= self.pixel_width or py < 0 or py >= self.pixel_height:
            return

        cx = px // 2
        cy = py // 4
        sub_x = px % 2
        sub_y = py % 4

        mask = self.DOT_MAP[sub_y][sub_x]
        self.grid[cy][cx] |= mask

        if color is not None:
            self.color_grid[cy][cx] = color

    def plot_point(
        self,
        x: float,
        y: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Plot physical point (x, y)."""
        px, py = self.coord_to_pixel(x, y)
        self.set_pixel(px, py, color)

    def draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw line segment between physical coordinates via Bresenham algorithm."""
        px1, py1 = self.coord_to_pixel(x1, y1)
        px2, py2 = self.coord_to_pixel(x2, y2)

        dx = abs(px2 - px1)
        dy = abs(py2 - py1)
        sx = 1 if px1 < px2 else -1
        sy = 1 if py1 < py2 else -1
        err = dx - dy

        cur_x, cur_y = px1, py1
        while True:
            self.set_pixel(cur_x, cur_y, color)
            if cur_x == px2 and cur_y == py2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cur_x += sx
            if e2 < dx:
                err += dx
                cur_y += sy

    def render_to_string(self) -> str:
        """Render Braille matrix to ANSI terminal string."""
        lines: List[str] = []
        for cy in range(self.char_height):
            line_parts: List[str] = []
            for cx in range(self.char_width):
                code = self.grid[cy][cx]
                if code == 0:
                    line_parts.append(" ")
                else:
                    char = chr(0x2800 + code)
                    r, g, b = self.color_grid[cy][cx]
                    line_parts.append(f"{truecolor_fg(r, g, b)}{char}")
            line_parts.append(RESET_SEQ)
            lines.append("".join(line_parts))
        return "\n".join(lines)


class RocketVisualizer:
    """
    Visualizer for rocket nozzles, supersonic plumes, and propulsion HUD.
    """

    def __init__(self, canvas: Optional[BrailleCanvas] = None) -> None:
        self.canvas = canvas or BrailleCanvas()

    def draw_nozzle_contour(
        self,
        contour: NozzleContour,
        color: Tuple[int, int, int] = (180, 185, 200),  # Steel metallic
    ) -> None:
        """Draw converging-diverging De Laval nozzle walls (top and bottom)."""
        r_t = contour.throat_radius
        # Converging chamber section (x in [-1.2, 0])
        r_chamber = 2.2 * r_t
        x_chamber_start = -1.5 * r_t

        # Draw converging arc
        steps = 20
        dx = (0.0 - x_chamber_start) / steps
        for i in range(steps):
            x1 = x_chamber_start + i * dx
            x2 = x_chamber_start + (i + 1) * dx
            # Smooth cubic contraction to throat
            t1 = (x1 - x_chamber_start) / (0.0 - x_chamber_start)
            t2 = (x2 - x_chamber_start) / (0.0 - x_chamber_start)
            y1 = r_chamber + (r_t - r_chamber) * (3.0 * t1 * t1 - 2.0 * t1 * t1 * t1)
            y2 = r_chamber + (r_t - r_chamber) * (3.0 * t2 * t2 - 2.0 * t2 * t2 * t2)

            self.canvas.draw_line(x1, y1, x2, y2, color=color)
            self.canvas.draw_line(x1, -y1, x2, -y2, color=color)

        # Draw supersonic diverging wall points
        pts = contour.wall_points
        for i in range(1, len(pts)):
            x1, y1 = pts[i - 1]
            x2, y2 = pts[i]
            self.canvas.draw_line(x1, y1, x2, y2, color=color)
            self.canvas.draw_line(x1, -y1, x2, -y2, color=color)

        # Centerline dashed axis
        cx_start = x_chamber_start
        cx_end = pts[-1][0]
        c_step = 0.2
        c_curr = cx_start
        while c_curr < cx_end:
            self.canvas.draw_line(c_curr, 0.0, c_curr + 0.1, 0.0, color=(90, 95, 110))
            c_curr += c_step

    def draw_exhaust_plume(
        self,
        plume: PlumeStructure,
        boundary_color: Tuple[int, int, int] = (60, 160, 255),  # Cyan shear layer
        shock_color: Tuple[int, int, int] = (255, 230, 50),     # Bright yellow shock lines
    ) -> None:
        """Render supersonic exhaust plume boundaries and Mach diamond shock cells."""
        # Draw shear layer boundary (top and bottom)
        pts = plume.boundary_points
        for i in range(1, len(pts)):
            x1, y1 = pts[i - 1]
            x2, y2 = pts[i]
            self.canvas.draw_line(x1, y1, x2, y2, color=boundary_color)
            self.canvas.draw_line(x1, -y1, x2, -y2, color=boundary_color)

        # Draw Mach diamond shock reflection lines
        for line in plume.shock_lines:
            if len(line) >= 2:
                p1 = line[0]
                p2 = line[1]
                self.canvas.draw_line(p1[0], p1[1], p2[0], p2[1], color=shock_color)

    def format_telemetry_hud(
        self,
        engine: RocketEngineState,
        plume: PlumeStructure,
        thermal: Optional[ThermalStationResult] = None,
    ) -> str:
        """Format rocket propulsion and aerothermodynamic telemetry HUD."""
        pc_bar = engine.chamber_pressure / 1.0e5
        pe_kpa = engine.exit_pressure / 1.0e3
        thrust_vac_kn = engine.thrust_vacuum / 1.0e3
        thrust_sl_kn = engine.thrust_sea_level / 1.0e3

        q_flux_mw = (thermal.heat_flux / 1.0e6) if thermal else 0.0
        t_wg = thermal.wall_gas_temperature if thermal else 0.0
        margin = thermal.safety_margin_kelvin if thermal else 0.0

        lines = [
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
            f"{BOLD_SEQ}      THERMOPROP : SUPERSONIC NOZZLE & ROCKET PROPULSION HUD         {RESET_SEQ}",
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
            f" [COMBUSTION]    Chamber Pressure Pc : {pc_bar:6.2f} bar     Chamber Temp Tc : {engine.chamber_temperature:6.1f} K",
            f"                 Mass Flow Rate m_dot: {engine.mass_flow_rate:6.2f} kg/s    Char. Velocity c*: {engine.characteristic_velocity_c_star:6.1f} m/s",
            f" [NOZZLE MOC]    Expansion Ratio eps : {engine.expansion_ratio:6.2f}        Exit Mach Me    : {engine.exit_mach:6.2f}",
            f"                 Exit Velocity ve    : {engine.exit_velocity:6.1f} m/s    Exit Pressure Pe: {pe_kpa:6.2f} kPa",
            f" [PERFORMANCE]   Thrust (Vacuum)     : {thrust_vac_kn:6.2f} kN     Thrust (Sea Lvl): {thrust_sl_kn:6.2f} kN",
            f"                 Specific Impulse Vac: {engine.isp_vacuum_seconds:6.1f} s      Isp Sea Level   : {engine.isp_sea_level_seconds:6.1f} s",
            f" [EXHAUST PLUME] Regime              : {plume.regime.value}",
            f"                 Pressure Ratio Pe/Pa: {plume.pressure_ratio:6.3f}      Diamond Spacing : {plume.cell_wavelength:6.2f} m",
            f" [HEAT TRANSFER] Peak Wall Heat Flux : {q_flux_mw:6.2f} MW/m2   Liner Temp Twg  : {t_wg:6.1f} K",
            f"                 Liner Safety Margin : {margin:6.1f} K (CuCrZr threshold 950 K)",
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
        ]
        return "\n".join(lines)
