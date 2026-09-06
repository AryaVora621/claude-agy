"""
Sub-Pixel Unicode Braille Tokamak Visualizer & Telemetry HUD.

Renders high-resolution 2D poloidal cross-sections (R, Z) of magnetic flux surfaces,
Poincaré surface-of-section puncture plots, and neoclassical banana drift orbits
using a 2x4 sub-pixel Braille dot matrix (U+2800..U+28FF) with 24-bit TrueColor ANSI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .equilibrium import Grid2D
from .particles import ParticleState
from .poincare import PoincarePuncture


# ANSI 24-bit TrueColor sequences
RESET_SEQ: str = "\033[0m"
BOLD_SEQ: str = "\033[1m"


def truecolor_fg(r: int, g: int, b: int) -> str:
    """Generate ANSI escape sequence for 24-bit foreground color."""
    return f"\033[38;2;{r};{g};{b}m"


class BrailleCanvas:
    """
    Sub-pixel dot matrix canvas using Unicode Braille Patterns (U+2800..U+28FF).

    Each text character cell represents a 2x4 sub-pixel dot matrix:
        Dot 1: (0, 0) [0x01]    Dot 4: (1, 0) [0x08]
        Dot 2: (0, 1) [0x02]    Dot 5: (1, 1) [0x10]
        Dot 3: (0, 2) [0x04]    Dot 6: (1, 2) [0x20]
        Dot 7: (0, 3) [0x40]    Dot 8: (1, 3) [0x80]
    """

    DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(
        self,
        char_width: int = 70,
        char_height: int = 30,
        r_min: float = 1.5,
        r_max: float = 4.5,
        z_min: float = -2.0,
        z_max: float = 2.0,
    ) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max

        # 2D arrays for character cells [char_h][char_w]
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

    def coord_to_pixel(self, r: float, z: float) -> Tuple[int, int]:
        """Map physical coordinates (R, Z) to pixel indices (px, py)."""
        norm_r = (r - self.r_min) / (self.r_max - self.r_min)
        # Invert vertical axis so positive Z points upward on terminal screen
        norm_z = (self.z_max - z) / (self.z_max - self.z_min)

        px = int(norm_r * (self.pixel_width - 1))
        py = int(norm_z * (self.pixel_height - 1))
        return px, py

    def set_pixel(
        self,
        px: int,
        py: int,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Set a single sub-pixel dot with optional TrueColor RGB."""
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
        r: float,
        z: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Plot a physical coordinate (R, Z) on the Braille canvas."""
        px, py = self.coord_to_pixel(r, z)
        self.set_pixel(px, py, color)

    def draw_line(
        self,
        r1: float,
        z1: float,
        r2: float,
        z2: float,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw a line segment between two physical points using Bresenham algorithm."""
        px1, py1 = self.coord_to_pixel(r1, z1)
        px2, py2 = self.coord_to_pixel(r2, z2)

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
        """Render the Braille canvas to a formatted ANSI terminal string."""
        lines: List[str] = []
        for cy in range(self.char_height):
            line_parts: List[str] = []
            for cx in range(self.char_width):
                code = self.grid[cy][cx]
                braille_char = chr(0x2800 + code)
                r, g, b = self.color_grid[cy][cx]
                color_seq = truecolor_fg(r, g, b)
                line_parts.append(f"{color_seq}{braille_char}")
            line_parts.append(RESET_SEQ)
            lines.append("".join(line_parts))
        return "\n".join(lines)


class TokamakVisualizer:
    """
    High-level visualizer for tokamak equilibria, orbits, and Poincaré maps.
    """

    def __init__(self, canvas: Optional[BrailleCanvas] = None) -> None:
        self.canvas = canvas or BrailleCanvas()

    def temperature_color(self, psi_n: float) -> Tuple[int, int, int]:
        """
        Map normalized flux psi_N to 24-bit TrueColor plasma temperature gradient.
            psi_N = 0.0: Core (hot gold-white, > 15 keV)
            psi_N = 0.3: Confinement region (hot amber-orange, ~10 keV)
            psi_N = 0.7: Mid-radius (neon magenta, ~5 keV)
            psi_N = 0.95: Pedestal edge (cyan, ~1 keV)
            psi_N = 1.0: Scrape-off layer (slate blue, cold)
        """
        if psi_n <= 0.1:
            return (255, 245, 200)  # Core white-gold
        if psi_n <= 0.35:
            return (255, 160, 40)  # Amber orange
        if psi_n <= 0.65:
            return (220, 60, 170)  # Plasma magenta
        if psi_n <= 0.88:
            return (50, 160, 240)  # Confinement blue
        return (70, 90, 120)  # Edge slate

    def draw_flux_contours(
        self,
        r_axis: float,
        z_axis: float,
        a_minor: float,
        kappa: float,
        num_surfaces: int = 7,
        num_angles: int = 100,
    ) -> None:
        """Draw nested elliptical flux surfaces psi_N = const."""
        d_theta = (2.0 * math.pi) / num_angles
        for s in range(1, num_surfaces + 1):
            rho = (s / num_surfaces)
            r_minor = rho * a_minor
            psi_n = rho * rho
            color = self.temperature_color(psi_n)

            # Draw closed contour ring
            for k in range(num_angles):
                t1 = k * d_theta
                t2 = (k + 1) * d_theta
                r1 = r_axis + r_minor * math.cos(t1)
                z1 = z_axis + kappa * r_minor * math.sin(t1)
                r2 = r_axis + r_minor * math.cos(t2)
                z2 = z_axis + kappa * r_minor * math.sin(t2)
                self.canvas.draw_line(r1, z1, r2, z2, color=color)

        # Draw magnetic axis point
        self.canvas.plot_point(r_axis, z_axis, color=(255, 255, 255))

    def draw_particle_orbit(
        self,
        trajectory: List[ParticleState],
        color: Tuple[int, int, int] = (100, 255, 120),  # Neon emerald green
    ) -> None:
        """Plot charged particle poloidal trajectory (R, Z)."""
        if len(trajectory) < 2:
            return
        for i in range(1, len(trajectory)):
            p1 = trajectory[i - 1]
            p2 = trajectory[i]
            self.canvas.draw_line(p1.r_cylindrical, p1.z, p2.r_cylindrical, p2.z, color=color)

    def draw_poincare_punctures(
        self,
        punctures: List[PoincarePuncture],
        color: Tuple[int, int, int] = (255, 220, 50),  # Bright gold
    ) -> None:
        """Plot scattered Poincaré puncture points."""
        for p in punctures:
            self.canvas.plot_point(p.r, p.z, color=color)

    def draw_vacuum_vessel(
        self,
        r_axis: float,
        z_axis: float,
        a_wall: float,
        kappa_wall: float,
        num_angles: int = 120,
    ) -> None:
        """Draw the surrounding vacuum vessel first wall in slate gray."""
        color = (110, 115, 130)
        d_theta = (2.0 * math.pi) / num_angles
        for k in range(num_angles):
            t1 = k * d_theta
            t2 = (k + 1) * d_theta
            r1 = r_axis + a_wall * math.cos(t1)
            z1 = z_axis + kappa_wall * a_wall * math.sin(t1)
            r2 = r_axis + a_wall * math.cos(t2)
            z2 = z_axis + kappa_wall * a_wall * math.sin(t2)
            self.canvas.draw_line(r1, z1, r2, z2, color=color)

    def format_telemetry_hud(
        self,
        r_0: float,
        a_minor: float,
        b_0: float,
        i_p: float,
        q_0: float,
        q_95: float,
        t_core_kev: float,
        p_core_kpa: float,
        beta_toroidal_pct: float,
        confinement_time_s: float = 2.4,
        plasma_density_m3: float = 1.0e20,
    ) -> str:
        """
        Format high-density tokamak fusion reactor telemetry HUD.
        """
        aspect_ratio = r_0 / a_minor if a_minor > 0.0 else 0.0
        # Lawson triple product: n_e * T_i * tau_E in 10^20 keV * s / m^3
        lawson_product = (plasma_density_m3 / 1e20) * t_core_kev * confinement_time_s
        lawson_ignition_target = 30.0  # ~3.0 * 10^21 keV s / m^3
        ignition_margin = (lawson_product / lawson_ignition_target) * 100.0

        i_p_ma = i_p / 1.0e6

        lines = [
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
            f"{BOLD_SEQ}       STELLARFUSION : TOKAMAK EQUILIBRIUM & CONFINEMENT HUD         {RESET_SEQ}",
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
            f" [GEOMETRY]      Major Radius R0 : {r_0:6.3f} m    Minor Radius a  : {a_minor:6.3f} m",
            f"                 Aspect Ratio A  : {aspect_ratio:6.3f}      Elongation k    :  1.700",
            f" [MAGNETICS]     Toroidal Field  : {b_0:6.3f} T    Plasma Current  : {i_p_ma:6.3f} MA",
            f"                 Safety Factor q0: {q_0:6.3f}      Edge Safety q95 : {q_95:6.3f}",
            f" [THERMODYNAMICS]Core Temp T0    : {t_core_kev:6.2f} keV  Core Pressure p0: {p_core_kpa:6.2f} kPa",
            f"                 Toroidal Beta   : {beta_toroidal_pct:6.2f} %    Energy Tau_E    : {confinement_time_s:6.2f} s",
            f" [LAWSON CRIT.]  Triple Product  : {lawson_product:6.2f} x10^20 keV*s/m3",
            f"                 Ignition Margin : {ignition_margin:6.1f} % of D-T burning threshold",
            f"{BOLD_SEQ}======================================================================{RESET_SEQ}",
        ]
        return "\n".join(lines)
