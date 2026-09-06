"""
Structura: Sub-Pixel Unicode Braille Finite Element Visualizer.
Implements:
  - BrailleFEACanvas (2x4 sub-pixel U+2800..U+28FF resolution)
  - render_mesh_deformation (undeformed vs magnified deformed mesh)
  - render_stress_field (24-bit TrueColor ANSI Von Mises contour heatmap)
  - render_fea_telemetry_hud (structural telemetry display)
Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

from structura.types import Node2D
from structura.mesh import Mesh2D
from structura.solver import FEAResult
from structura.modal import ModalResult


def turbo_colormap(val: float) -> Tuple[int, int, int]:
    """
    Map normalized scalar in [0.0, 1.0] to 24-bit RGB TrueColor.
    Smooth transition: Blue (low) -> Cyan -> Green -> Yellow -> Red (peak stress).
    """
    v = max(0.0, min(1.0, val))
    if v < 0.25:
        # Blue to Cyan
        t = v / 0.25
        r = int(20 * (1 - t) + 0 * t)
        g = int(30 * (1 - t) + 200 * t)
        b = int(180 * (1 - t) + 240 * t)
    elif v < 0.50:
        # Cyan to Green
        t = (v - 0.25) / 0.25
        r = int(0 * (1 - t) + 20 * t)
        g = int(200 * (1 - t) + 230 * t)
        b = int(240 * (1 - t) + 40 * t)
    elif v < 0.75:
        # Green to Yellow
        t = (v - 0.50) / 0.25
        r = int(20 * (1 - t) + 245 * t)
        g = int(230 * (1 - t) + 220 * t)
        b = int(40 * (1 - t) + 10 * t)
    else:
        # Yellow to Red
        t = (v - 0.75) / 0.25
        r = int(245 * (1 - t) + 255 * t)
        g = int(220 * (1 - t) + 30 * t)
        b = int(10 * (1 - t) + 20 * t)
    return (r, g, b)


class BrailleFEACanvas:
    """
    2D Terminal Canvas using Unicode Braille Patterns (U+2800..U+28FF).
    Each character cell represents a 2x4 sub-pixel grid:
      dot 1: (0, 0)   dot 4: (1, 0)
      dot 2: (0, 1)   dot 5: (1, 1)
      dot 3: (0, 2)   dot 6: (1, 2)
      dot 7: (0, 3)   dot 8: (1, 3)
    """
    _DOT_MAP = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(self, char_width: int = 72, char_height: int = 20):
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.grid = [[0] * char_width for _ in range(char_height)]
        self.colors: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    def clear(self) -> None:
        for r in range(self.char_height):
            for c in range(self.char_width):
                self.grid[r][c] = 0
        self.colors.clear()

    def set_pixel(self, px: int, py: int, color_rgb: Optional[Tuple[int, int, int]] = None) -> None:
        """Set a single sub-pixel."""
        if px < 0 or px >= self.pixel_width or py < 0 or py >= self.pixel_height:
            return
        cx = px // 2
        cy = py // 4
        sub_x = px % 2
        sub_y = py % 4

        self.grid[cy][cx] |= self._DOT_MAP[sub_y][sub_x]
        if color_rgb:
            self.colors[(cx, cy)] = color_rgb

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Bresenham line algorithm on sub-pixel grid."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while True:
            self.set_pixel(curr_x, curr_y, color_rgb)
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

    def render(self, use_color: bool = True) -> List[str]:
        """Render canvas lines."""
        lines = []
        reset = "\033[0m"

        for cy in range(self.char_height):
            row_str = []
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask)
                if use_color and (cx, cy) in self.colors:
                    r, g, b = self.colors[(cx, cy)]
                    row_str.append(f"\033[38;2;{r};{g};{b}m{char}{reset}")
                else:
                    row_str.append(char)
            lines.append("".join(row_str))
        return lines


def render_mesh_deformation(
    mesh: Mesh2D,
    displacements: List[float],
    magnification: Optional[float] = None,
    char_width: int = 72,
    char_height: int = 20,
    element_stresses: Optional[List[float]] = None,
) -> str:
    """
    Render undeformed vs deformed mesh wireframe on a sub-pixel Unicode Braille canvas.
    """
    canvas = BrailleFEACanvas(char_width, char_height)
    min_x, max_x, min_y, max_y = mesh.get_bounds()

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    has_beams = any(len(elem.get_dofs(mesh.nodes)) == 6 and elem.num_nodes == 2 for elem in mesh.elements)
    dofs_per_node = 3 if has_beams else 2

    # Compute max displacement for automated scaling
    max_disp = 0.0
    for nid in mesh.nodes.keys():
        base = dofs_per_node * nid
        ux = displacements[base]
        uy = displacements[base + 1]
        mag = math.sqrt(ux * ux + uy * uy)
        if mag > max_disp:
            max_disp = mag

    if magnification is None:
        # Scale displacement so peak deflection is ~15% of mesh domain span
        target_pix_def = 0.15 * min(span_x, span_y)
        magnification = (target_pix_def / max_disp) if max_disp > 1e-12 else 1.0

    margin_x = 4
    margin_y = 4
    draw_w = canvas.pixel_width - 2 * margin_x
    draw_h = canvas.pixel_height - 2 * margin_y

    def world_to_pix(wx: float, wy: float) -> Tuple[int, int]:
        norm_x = (wx - min_x) / span_x
        norm_y = (wy - min_y) / span_y
        px = int(margin_x + norm_x * draw_w)
        # Flip Y so y=0 is at bottom
        py = int((canvas.pixel_height - 1 - margin_y) - norm_y * draw_h)
        return (px, py)

    # 1. Draw undeformed mesh wireframe in dim gray (100, 100, 100)
    dim_gray = (90, 95, 105)
    for elem in mesh.elements:
        nids = elem.node_ids
        for i in range(len(nids)):
            n_start = mesh.nodes[nids[i]]
            n_end = mesh.nodes[nids[(i + 1) % len(nids)]]
            p0 = world_to_pix(n_start.x, n_start.y)
            p1 = world_to_pix(n_end.x, n_end.y)
            canvas.draw_line(p0[0], p0[1], p1[0], p1[1], dim_gray)

    # 2. Draw deformed mesh wireframe color-coded by stress or cyan
    max_stress = max(element_stresses) if element_stresses else 1.0
    max_stress = max(1e-6, max_stress)

    for idx, elem in enumerate(mesh.elements):
        nids = elem.node_ids
        color = (0, 230, 240)  # Default vibrant cyan
        if element_stresses:
            val = element_stresses[idx] / max_stress
            color = turbo_colormap(val)

        for i in range(len(nids)):
            nid_start = nids[i]
            nid_end = nids[(i + 1) % len(nids)]

            n_start = mesh.nodes[nid_start]
            n_end = mesh.nodes[nid_end]

            base_s = dofs_per_node * nid_start
            base_e = dofs_per_node * nid_end

            x_s = n_start.x + magnification * displacements[base_s]
            y_s = n_start.y + magnification * displacements[base_s + 1]

            x_e = n_end.x + magnification * displacements[base_e]
            y_e = n_end.y + magnification * displacements[base_e + 1]

            p0 = world_to_pix(x_s, y_s)
            p1 = world_to_pix(x_e, y_e)
            canvas.draw_line(p0[0], p0[1], p1[0], p1[1], color)

    rendered_lines = canvas.render(use_color=True)
    scale_str = f"Deformation Scale: {magnification:.1f}x | Dim: Undeformed | Color: Deformed"
    rendered_lines.insert(0, f"┌{'─' * (char_width - 2)}┐")
    rendered_lines.append(f"└{'─' * (char_width - 2)}┘")
    rendered_lines.append(f"  {scale_str}")
    return "\n".join(rendered_lines)


def render_stress_field(
    mesh: Mesh2D,
    nodal_stresses: Dict[int, float],
    char_width: int = 72,
    char_height: int = 20,
) -> str:
    """
    Render 24-bit TrueColor ANSI Von Mises stress contour heatmap.
    """
    canvas = BrailleFEACanvas(char_width, char_height)
    min_x, max_x, min_y, max_y = mesh.get_bounds()
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)

    max_stress = max(nodal_stresses.values()) if nodal_stresses else 1.0
    max_stress = max(1e-6, max_stress)

    margin_x = 4
    margin_y = 4
    draw_w = canvas.pixel_width - 2 * margin_x
    draw_h = canvas.pixel_height - 2 * margin_y

    def world_to_pix(wx: float, wy: float) -> Tuple[int, int]:
        norm_x = (wx - min_x) / span_x
        norm_y = (wy - min_y) / span_y
        px = int(margin_x + norm_x * draw_w)
        py = int((canvas.pixel_height - 1 - margin_y) - norm_y * draw_h)
        return (px, py)

    # Render element wireframe with interpolated nodal stress color
    for elem in mesh.elements:
        nids = elem.node_ids
        for i in range(len(nids)):
            nid_start = nids[i]
            nid_end = nids[(i + 1) % len(nids)]

            ns = mesh.nodes[nid_start]
            ne = mesh.nodes[nid_end]

            p0 = world_to_pix(ns.x, ns.y)
            p1 = world_to_pix(ne.x, ne.y)

            s_avg = 0.5 * (nodal_stresses.get(nid_start, 0.0) + nodal_stresses.get(nid_end, 0.0))
            norm_val = s_avg / max_stress
            color = turbo_colormap(norm_val)

            canvas.draw_line(p0[0], p0[1], p1[0], p1[1], color)

    rendered_lines = canvas.render(use_color=True)
    header = f"┌{'─' * (char_width - 2)}┐"
    footer = f"└{'─' * (char_width - 2)}┘"

    # Colorbar legend
    legend = (
        "  \033[38;2;20;30;180m■\033[0m Low (0%) "
        "\033[38;2;0;200;220m■\033[0m 25% "
        "\033[38;2;20;230;40m■\033[0m 50% "
        "\033[38;2;245;220;10m■\033[0m 75% "
        f"\033[38;2;255;30;20m■\033[0m Peak ({max_stress / 1e6:.1f} MPa)"
    )

    rendered_lines.insert(0, header)
    rendered_lines.append(footer)
    rendered_lines.append(legend)
    return "\n".join(rendered_lines)


def render_fea_telemetry_hud(
    mesh: Mesh2D,
    result: FEAResult,
    modal_result: Optional[ModalResult] = None,
    title: str = "FINITE ELEMENT ANALYSIS TELEMETRY",
) -> str:
    """
    Format a clean engineering telemetry HUD for structural analysis results.
    """
    elem_types = list(set(e.__class__.__name__ for e in mesh.elements))
    type_str = ", ".join(elem_types)

    disp_mm = result.max_displacement * 1000.0
    vm_mpa = result.max_von_mises_stress / 1e6

    fos_str = f"{result.min_factor_of_safety:.2f}" if result.min_factor_of_safety < 1000.0 else "> 1000"
    compliance_j = result.strain_energy

    f1_str = f"{modal_result.fundamental_frequency_hz:.2f} Hz" if modal_result and modal_result.modes else "N/A"

    lines = [
        "╔══════════════════════════════════════════════════════════════════════════════╗",
        f"║   {title.center(74)}   ║",
        "╠══════════════════════════════════════════════════════════════════════════════╣",
        f"║  Element Types: {type_str:<18}  Nodes: {mesh.num_nodes:<6} Elements: {mesh.num_elements:<6} DOFs: {mesh.num_dofs:<6} ║",
        f"║  Max Deflection: {disp_mm:>8.3f} mm       Peak Stress: {vm_mpa:>9.2f} MPa  Safety Factor: {fos_str:>6} ║",
        f"║  Strain Energy:  {compliance_j:>8.3f} J        PCG Iters:   {result.solver_iterations:>6}     Residual: {result.residual_norm:>10.2e} ║",
        f"║  Resonance (f1): {f1_str:<14}     Reactions:   {len(result.reactions):>6} nodes evaluated             ║",
        "╚══════════════════════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)
