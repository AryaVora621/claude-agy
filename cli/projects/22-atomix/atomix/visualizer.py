"""
Atomix: 3D Sub-Pixel Unicode Braille Molecular Viewer and Telemetry HUD.
Implements:
  - 3D camera projection with azimuth/elevation orbital rotation
  - 2x4 sub-pixel Braille canvas (U+2800..U+28FF) for bonds and wireframes
  - Depth-sorted 24-bit TrueColor CPK element spheres and bonds
  - Live thermodynamic dashboard HUD and energy sparkline
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

from atomix.types import Vector3D, Atom, Bond, SimulationBox, CPK_COLORS
from atomix.observables import ThermodynamicState


# Braille Unicode sub-pixel dot bitmask layout (2 columns x 4 rows):
# (0,0): 0x01 | (1,0): 0x08
# (0,1): 0x02 | (1,1): 0x10
# (0,2): 0x04 | (1,2): 0x20
# (0,3): 0x40 | (1,3): 0x80
BRAILLE_DOTS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


class BrailleCanvas3D:
    """
    Sub-pixel Braille 3D drawing canvas with 24-bit ANSI TrueColor support.
    """
    def __init__(self, char_width: int = 70, char_height: int = 22) -> None:
        self.char_w = char_width
        self.char_h = char_height
        self.pixel_w = char_width * 2
        self.pixel_h = char_height * 4

        self.grid = [[0 for _ in range(self.char_w)] for _ in range(self.char_h)]
        self.color_grid: List[List[Optional[Tuple[int, int, int]]]] = [
            [None for _ in range(self.char_w)] for _ in range(self.char_h)
        ]
        self.z_buffer = [[-float("inf") for _ in range(self.char_w)] for _ in range(self.char_h)]

    def set_pixel(self, px: int, py: int, z: float, color: Tuple[int, int, int]) -> None:
        """Light up a single sub-pixel with depth testing."""
        if 0 <= px < self.pixel_w and 0 <= py < self.pixel_h:
            cx = px // 2
            cy = py // 4
            sub_x = px % 2
            sub_y = py % 4

            self.grid[cy][cx] |= BRAILLE_DOTS[sub_y][sub_x]
            if z >= self.z_buffer[cy][cx]:
                self.z_buffer[cy][cx] = z
                self.color_grid[cy][cx] = color

    def draw_line_3d(
        self,
        p0: Tuple[int, int, float],
        p1: Tuple[int, int, float],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a depth-interpolated 3D line using Bresenham's algorithm."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        steps = max(dx, dy)
        step_count = 0

        while True:
            t = (step_count / steps) if steps > 0 else 0.0
            z = z0 + t * (z1 - z0)
            self.set_pixel(x0, y0, z, color)

            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            step_count += 1

    def render(self) -> str:
        """Render Braille grid to ANSI string with TrueColor escape codes."""
        lines: List[str] = []
        for cy in range(self.char_h):
            row_chars: List[str] = []
            for cx in range(self.char_w):
                mask = self.grid[cy][cx]
                color = self.color_grid[cy][cx]
                char = chr(0x2800 + mask) if mask > 0 else " "
                if mask > 0 and color is not None:
                    r, g, b = color
                    row_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        return "\n".join(lines)


def render_molecular_system(
    atoms: List[Atom],
    bonds: Optional[List[Bond]] = None,
    box: Optional[SimulationBox] = None,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    char_width: int = 70,
    char_height: int = 22,
) -> str:
    """
    Project 3D molecular coordinates and bonds onto a 2x4 sub-pixel Braille canvas.
    Colors atoms according to IUPAC CPK conventions.
    """
    if not atoms:
        return "[Empty System]"

    canvas = BrailleCanvas3D(char_width, char_height)

    # Compute center of geometry for camera targeting
    xs = [a.position.x for a in atoms]
    ys = [a.position.y for a in atoms]
    zs = [a.position.z for a in atoms]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))

    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)

    # 3D Euler rotation angles in radians
    phi = azimuth_deg * (math.pi / 180.0)
    theta = elevation_deg * (math.pi / 180.0)

    cos_p, sin_p = math.cos(phi), math.sin(phi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def project(pos: Vector3D) -> Tuple[int, int, float]:
        # Center relative coordinates
        rx = pos.x - cx
        ry = pos.y - cy
        rz = pos.z - cz

        # Rotate around Y (azimuth) then X (elevation)
        x_rot = rx * cos_p + rz * sin_p
        z_tmp = -rx * sin_p + rz * cos_p
        y_rot = ry * cos_t - z_tmp * sin_t
        z_rot = ry * sin_t + z_tmp * cos_t

        # Scale to canvas pixels with margin
        scale_x = (canvas.pixel_w * 0.42) / (0.5 * span)
        scale_y = (canvas.pixel_h * 0.42) / (0.5 * span)
        scale = min(scale_x, scale_y)

        px = int(canvas.pixel_w * 0.5 + x_rot * scale)
        py = int(canvas.pixel_h * 0.5 - y_rot * scale)
        return (px, py, z_rot)

    # 1. Draw periodic box wireframe if provided
    if box:
        box_corners = [
            Vector3D(0, 0, 0), Vector3D(box.lx, 0, 0),
            Vector3D(box.lx, box.ly, 0), Vector3D(0, box.ly, 0),
            Vector3D(0, 0, box.lz), Vector3D(box.lx, 0, box.lz),
            Vector3D(box.lx, box.ly, box.lz), Vector3D(0, box.ly, box.lz),
        ]
        box_proj = [project(p) for p in box_corners]
        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        dim_box_color = (65, 70, 80)
        for e1, e2 in box_edges:
            canvas.draw_line_3d(box_proj[e1], box_proj[e2], dim_box_color)

    # 2. Draw covalent bonds
    if bonds:
        atom_pos_map = {a.id: project(a.position) for a in atoms}
        bond_color = (130, 135, 145)
        for bond in bonds:
            if bond.atom1_id in atom_pos_map and bond.atom2_id in atom_pos_map:
                p1 = atom_pos_map[bond.atom1_id]
                p2 = atom_pos_map[bond.atom2_id]
                canvas.draw_line_3d(p1, p2, bond_color)

    # 3. Draw atom spheres/dots with CPK element colors
    for atom in atoms:
        px, py, z = project(atom.position)
        color = atom.cpk_color

        # Small 2x2 cluster for visual presence
        canvas.set_pixel(px, py, z, color)
        canvas.set_pixel(px + 1, py, z, color)
        canvas.set_pixel(px, py + 1, z, color)
        canvas.set_pixel(px + 1, py + 1, z, color)

    return canvas.render()


def render_telemetry_hud(
    state: ThermodynamicState,
    num_atoms: int,
    num_bonds: int,
    rg: float = 0.0,
    diffusion_coeff: float = 0.0,
    energy_history: Optional[List[float]] = None,
    width: int = 70,
) -> str:
    """
    Render a clean B2B terminal telemetry HUD with physical units and mini sparkline.
    """
    cyan = "\033[38;2;0;210;225m"
    gold = "\033[38;2;245;185;45m"
    green = "\033[38;2;80;220;100m"
    bold = "\033[1m"
    reset = "\033[0m"

    title = f"{bold}{cyan}[ATOMIX: MOLECULAR DYNAMICS & STATISTICAL MECHANICS ENGINE]{reset}"
    step_str = f"Step: {bold}{state.step:6d}{reset} | Time: {state.time:7.3f} ps"
    temp_str = f"Temp: {bold}{gold}{state.temperature:7.2f} K{reset}"
    pres_str = f"Pres: {state.pressure:8.2f} bar"
    dens_str = f"Density: {state.density:6.3f} g/cm3"

    e_tot_str = f"E_tot: {bold}{green}{state.total_energy:10.2f}{reset} kJ/mol"
    e_pot_str = f"E_pot: {state.potential_energy:10.2f} kJ/mol"
    e_kin_str = f"E_kin: {state.kinetic_energy:10.2f} kJ/mol"

    geom_str = f"Atoms: {num_atoms:4d} | Bonds: {num_bonds:4d} | R_gyr: {rg:6.2f} A | D: {diffusion_coeff:8.2e}"

    lines = [
        f"{title}",
        f"{'=' * width}",
        f"{step_str} | {temp_str} | {pres_str}",
        f"{e_tot_str} | {e_pot_str} | {e_kin_str}",
        f"{geom_str} | {dens_str}",
        f"{'-' * width}",
    ]

    # Sparkline of recent total energy if history exists
    if energy_history and len(energy_history) >= 4:
        recent = energy_history[-min(width - 15, len(energy_history)):]
        min_e = min(recent)
        max_e = max(recent)
        span = max(1e-6, max_e - min_e)
        # Unicode 8-level vertical spark blocks:   ▂ ▃ ▄ ▅ ▆ ▇ █
        spark_blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        spark_str = ""
        for val in recent:
            frac = (val - min_e) / span
            idx = min(7, int(frac * 8))
            spark_str += spark_blocks[idx]
        lines.append(f"Energy Trend: [{spark_str}] ({min_e:.1f} .. {max_e:.1f} kJ/mol)")
        lines.append(f"{'=' * width}")

    return "\n".join(lines)
