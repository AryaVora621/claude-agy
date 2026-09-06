"""Silicon Photonic Devices and Integrated Circuit Generators.

Provides:
1. Standard optical material indices (Silicon, Silica, Silicon Nitride, Air)
2. Dielectric strip waveguides and low-loss 90-degree bends
3. 2x2 evanescent directional couplers and 3dB optical power splitters
4. Micro-ring resonators (bus-to-ring evanescent coupling)
5. Mach-Zehnder Interferometers (MZI) with dual optical arms
6. Photonic crystal line-defect waveguides with 2D bandgap confinement
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .grid import Grid2D


# Standard Optical Refractive Indices at 1.55 um (Telecom C-Band)
INDEX_AIR: float = 1.0000
INDEX_SILICON: float = 3.4764       # Silicon core (SOI platform)
INDEX_SILICA: float = 1.4440        # SiO2 cladding
INDEX_SI_NITRIDE: float = 1.9960    # Si3N4 low-loss core


@dataclass
class PhotonicPort:
    """Optical input or output port location and waveguide cross-section."""
    name: str
    x: int
    y: int
    width_cells: int
    is_vertical: bool = True  # True if port cross-section is vertical line (waveguide runs horizontally)


class PhotonicCircuitBuilder:
    """Factory helper for generating standard Silicon Photonic Integrated Circuits."""

    @staticmethod
    def add_straight_waveguide(
        grid: Grid2D,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int = 6,
        n_core: float = INDEX_SILICON,
    ) -> None:
        """Add horizontal or vertical straight dielectric strip waveguide."""
        eps_r = n_core * n_core
        if x1 == x2:
            # Vertical waveguide
            ymin, ymax = min(y1, y2), max(y1, y2)
            half = width // 2
            grid.add_rectangle(x1 - half, ymin, x1 + half, ymax, eps_r=eps_r)
        elif y1 == y2:
            # Horizontal waveguide
            xmin, xmax = min(x1, x2), max(x1, x2)
            half = width // 2
            grid.add_rectangle(xmin, y1 - half, xmax, y1 + half, eps_r=eps_r)
        else:
            # General line with rasterized width
            dx = x2 - x1
            dy = y2 - y1
            length = int(math.hypot(dx, dy))
            half = width // 2
            for step in range(length + 1):
                t = step / length if length > 0 else 0
                cx = int(x1 + t * dx)
                cy = int(y1 + t * dy)
                grid.add_circle(cx, cy, radius=half, eps_r=eps_r)

    @staticmethod
    def add_waveguide_bend(
        grid: Grid2D,
        cx: int,
        cy: int,
        radius: int,
        width: int = 6,
        start_angle_deg: float = 0.0,
        end_angle_deg: float = 90.0,
        n_core: float = INDEX_SILICON,
    ) -> None:
        """Add circular dielectric waveguide bend."""
        eps_r = n_core * n_core
        r_inner = max(0, radius - width // 2)
        r_outer = radius + width // 2
        r_in2 = r_inner * r_inner
        r_out2 = r_outer * r_outer

        rad_start = math.radians(min(start_angle_deg, end_angle_deg))
        rad_end = math.radians(max(start_angle_deg, end_angle_deg))

        xmin = max(0, cx - r_outer)
        xmax = min(grid.nx - 1, cx + r_outer)
        ymin = max(0, cy - r_outer)
        ymax = min(grid.ny - 1, cy + r_outer)

        for y in range(ymin, ymax + 1):
            dy = y - cy
            for x in range(xmin, xmax + 1):
                dx = x - cx
                dist2 = dx * dx + dy * dy
                if r_in2 <= dist2 <= r_out2:
                    angle = math.atan2(dy, dx)
                    if angle < 0:
                        angle += 2 * math.pi
                    if rad_start <= angle <= rad_end:
                        idx = grid.idx(x, y)
                        grid.eps_r[idx] = eps_r

        grid.recompute_coefficients()

    @staticmethod
    def build_directional_coupler(
        grid: Grid2D,
        y_center: int = 40,
        waveguide_width: int = 6,
        coupling_gap: int = 3,
        coupling_length: int = 30,
        n_core: float = INDEX_SILICON,
    ) -> Tuple[List[PhotonicPort], List[PhotonicPort]]:
        """Build 2x2 Directional Coupler with input/output S-bends.

        Returns:
            Tuple of (input_ports, output_ports).
        """
        eps_r = n_core * n_core
        nx = grid.nx
        half_w = waveguide_width // 2

        # In coupling region:
        # Top waveguide center
        y_top = y_center + half_w + coupling_gap // 2 + 1
        # Bottom waveguide center
        y_bot = y_center - half_w - coupling_gap // 2 - 1

        x_start_c = (nx - coupling_length) // 2
        x_end_c = x_start_c + coupling_length

        # 1. Straight coupling region
        grid.add_rectangle(x_start_c, y_top - half_w, x_end_c, y_top + half_w, eps_r=eps_r)
        grid.add_rectangle(x_start_c, y_bot - half_w, x_end_c, y_bot + half_w, eps_r=eps_r)

        # 2. Input straight access waveguides (left)
        grid.add_rectangle(0, y_top + 8 - half_w, x_start_c - 10, y_top + 8 + half_w, eps_r=eps_r)
        grid.add_rectangle(0, y_bot - 8 - half_w, x_start_c - 10, y_bot - 8 + half_w, eps_r=eps_r)

        # Connect with diagonal/curved transitions
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_start_c - 10, y_top + 8, x_start_c, y_top, width=waveguide_width, n_core=n_core
        )
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_start_c - 10, y_bot - 8, x_start_c, y_bot, width=waveguide_width, n_core=n_core
        )

        # 3. Output straight access waveguides (right)
        grid.add_rectangle(x_end_c + 10, y_top + 8 - half_w, nx - 1, y_top + 8 + half_w, eps_r=eps_r)
        grid.add_rectangle(x_end_c + 10, y_bot - 8 - half_w, nx - 1, y_bot - 8 + half_w, eps_r=eps_r)

        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_end_c, y_top, x_end_c + 10, y_top + 8, width=waveguide_width, n_core=n_core
        )
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_end_c, y_bot, x_end_c + 10, y_bot - 8, width=waveguide_width, n_core=n_core
        )

        inputs = [
            PhotonicPort("In_Top", 15, y_top + 8, waveguide_width, is_vertical=True),
            PhotonicPort("In_Bot", 15, y_bot - 8, waveguide_width, is_vertical=True),
        ]
        outputs = [
            PhotonicPort("Out_Bar", nx - 15, y_top + 8, waveguide_width, is_vertical=True),
            PhotonicPort("Out_Cross", nx - 15, y_bot - 8, waveguide_width, is_vertical=True),
        ]
        return inputs, outputs

    @staticmethod
    def build_ring_resonator(
        grid: Grid2D,
        cx: int = 60,
        cy: int = 50,
        radius: int = 16,
        ring_width: int = 5,
        bus_y: int = 25,
        bus_width: int = 5,
        n_core: float = INDEX_SILICON,
    ) -> Tuple[PhotonicPort, PhotonicPort]:
        """Build Micro-Ring Resonator coupled to straight bus waveguide.

        Returns:
            Tuple of (input_port, through_port).
        """
        eps_r = n_core * n_core
        nx = grid.nx

        # 1. Straight bus waveguide across full width
        half_bus = bus_width // 2
        grid.add_rectangle(0, bus_y - half_bus, nx - 1, bus_y + half_bus, eps_r=eps_r)

        # 2. Dielectric Ring
        r_inner = radius - ring_width // 2
        r_outer = radius + ring_width // 2
        grid.add_ring(cx, cy, r_inner, r_outer, eps_r=eps_r)

        in_port = PhotonicPort("In_Bus", 15, bus_y, bus_width, is_vertical=True)
        through_port = PhotonicPort("Through_Bus", nx - 15, bus_y, bus_width, is_vertical=True)
        return in_port, through_port

    @staticmethod
    def build_mach_zehnder(
        grid: Grid2D,
        y_center: int = 40,
        arm_spacing: int = 18,
        arm_length: int = 35,
        waveguide_width: int = 5,
        n_core: float = INDEX_SILICON,
        phase_shift_index_delta: float = 0.0,
    ) -> Tuple[PhotonicPort, PhotonicPort]:
        """Build Mach-Zehnder Interferometer (MZI) with two optical arms.

        Args:
            phase_shift_index_delta: Optional refractive index shift delta_n in top arm.

        Returns:
            Tuple of (input_port, output_port).
        """
        eps_r = n_core * n_core
        nx = grid.nx
        half_w = waveguide_width // 2

        x_start = (nx - arm_length) // 2
        x_end = x_start + arm_length

        y_top = y_center + arm_spacing // 2
        y_bot = y_center - arm_spacing // 2

        # 1. Input waveguide (left)
        grid.add_rectangle(0, y_center - half_w, x_start - 12, y_center + half_w, eps_r=eps_r)

        # Y-splitter transitions to top and bottom arms
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_start - 12, y_center, x_start, y_top, width=waveguide_width, n_core=n_core
        )
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_start - 12, y_center, x_start, y_bot, width=waveguide_width, n_core=n_core
        )

        # 2. Top and bottom arms
        n_top = n_core + phase_shift_index_delta
        eps_top = n_top * n_top
        grid.add_rectangle(x_start, y_top - half_w, x_end, y_top + half_w, eps_r=eps_top)
        grid.add_rectangle(x_start, y_bot - half_w, x_end, y_bot + half_w, eps_r=eps_r)

        # 3. Y-combiner transitions back to center output
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_end, y_top, x_end + 12, y_center, width=waveguide_width, n_core=n_core
        )
        PhotonicCircuitBuilder.add_straight_waveguide(
            grid, x_end, y_bot, x_end + 12, y_center, width=waveguide_width, n_core=n_core
        )

        # Output waveguide (right)
        grid.add_rectangle(x_end + 12, y_center - half_w, nx - 1, y_center + half_w, eps_r=eps_r)

        in_port = PhotonicPort("In", 12, y_center, waveguide_width, is_vertical=True)
        out_port = PhotonicPort("Out", nx - 12, y_center, waveguide_width, is_vertical=True)
        return in_port, out_port

    @staticmethod
    def build_photonic_crystal_waveguide(
        grid: Grid2D,
        pitch: int = 8,
        rod_radius: int = 2,
        n_rod: float = INDEX_SILICON,
        defect_row: int = 5,
    ) -> Tuple[PhotonicPort, PhotonicPort]:
        """Build 2D periodic dielectric rod photonic crystal with W1 line defect waveguide."""
        eps_r = n_rod * n_rod
        nx, ny = grid.nx, grid.ny

        n_rows = ny // pitch
        n_cols = nx // pitch

        for row in range(1, n_rows):
            if row == defect_row:
                # W1 Defect: missing row forms optical waveguide
                continue
            cy = row * pitch
            for col in range(1, n_cols):
                cx = col * pitch
                grid.add_circle(cx, cy, radius=rod_radius, eps_r=eps_r)

        y_guide = defect_row * pitch
        in_port = PhotonicPort("In_PBG", 12, y_guide, pitch, is_vertical=True)
        out_port = PhotonicPort("Out_PBG", nx - 12, y_guide, pitch, is_vertical=True)
        return in_port, out_port
