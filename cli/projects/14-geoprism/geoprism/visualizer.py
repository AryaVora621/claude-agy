"""
GeoPrism: High-Resolution Unicode Braille & Terminal Visualizer.
Renders spatial bounding boxes, Delaunay triangulations, Voronoi diagrams,
and continuous density heatmaps into standard ANSI terminals using 2x4 Braille sub-pixels.
"""

import math
from typing import List, Tuple, Optional, Dict
from geoprism.aabb import AABB

Point2D = Tuple[float, float]

# Braille bitmask mapping for (x in 0..1, y in 0..3)
# Row 0: 0x01, 0x08
# Row 1: 0x02, 0x10
# Row 2: 0x04, 0x20
# Row 3: 0x40, 0x80
BRAILLE_DOTS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


class BrailleCanvas:
    """
    High-resolution 2D terminal canvas utilizing Unicode Braille characters (U+2800..U+28FF).
    Provides 2x horizontal and 4x vertical sub-pixel resolution per terminal character.
    """

    def __init__(
        self,
        char_width: int = 60,
        char_height: int = 25,
        bbox: Optional[AABB] = None
    ) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        # Sub-pixel buffer: True/False
        self.grid = [[0 for _ in range(self.pixel_width)] for _ in range(self.pixel_height)]
        self.bbox = bbox or AABB((0.0, 0.0), (float(self.pixel_width), float(self.pixel_height)))

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Maps world coordinates to discrete canvas pixel indices."""
        x_min, y_min = self.bbox.lower[0], self.bbox.lower[1]
        x_max, y_max = self.bbox.upper[0], self.bbox.upper[1]

        dx = max(1e-9, x_max - x_min)
        dy = max(1e-9, y_max - y_min)

        norm_x = (x - x_min) / dx
        # Invert Y so highest world Y is row 0
        norm_y = (y_max - y) / dy

        px = int(norm_x * (self.pixel_width - 1))
        py = int(norm_y * (self.pixel_height - 1))

        px = max(0, min(self.pixel_width - 1, px))
        py = max(0, min(self.pixel_height - 1, py))
        return px, py

    def set_pixel(self, px: int, py: int) -> None:
        """Sets a sub-pixel in canvas."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            self.grid[py][px] = 1

    def draw_point(self, x: float, y: float, radius: int = 1) -> None:
        """Draws a point at world coordinates with pixel radius."""
        cx, cy = self.world_to_pixel(x, y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    self.set_pixel(cx + dx, cy + dy)

    def draw_line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Draws a line segment using Bresenham's algorithm in pixel space."""
        px0, py0 = self.world_to_pixel(x0, y0)
        px1, py1 = self.world_to_pixel(x1, y1)

        dx = abs(px1 - px0)
        dy = -abs(py1 - py0)
        sx = 1 if px0 < px1 else -1
        sy = 1 if py0 < py1 else -1
        err = dx + dy

        x, y = px0, py0
        while True:
            self.set_pixel(x, y)
            if x == px1 and y == py1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def draw_box(self, box: AABB) -> None:
        """Draws an Axis-Aligned Bounding Box rectangle."""
        x0, y0 = box.lower[0], box.lower[1]
        x1, y1 = box.upper[0], box.upper[1]
        self.draw_line(x0, y0, x1, y0)
        self.draw_line(x1, y0, x1, y1)
        self.draw_line(x1, y1, x0, y1)
        self.draw_line(x0, y1, x0, y0)

    def draw_polygon(self, vertices: List[Point2D]) -> None:
        """Draws closed polygon perimeter."""
        n = len(vertices)
        if n < 2:
            return
        for i in range(n):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % n]
            self.draw_line(p0[0], p0[1], p1[0], p1[1])

    def render(self, border: bool = True) -> str:
        """
        Renders the pixel buffer into Unicode Braille characters.
        """
        lines: List[str] = []
        if border:
            lines.append("+" + "-" * self.char_width + "+")

        for cy in range(self.char_height):
            row_chars: List[str] = []
            for cx in range(self.char_width):
                code = 0
                for by in range(4):
                    py = cy * 4 + by
                    for bx in range(2):
                        px = cx * 2 + bx
                        if self.grid[py][px]:
                            code |= BRAILLE_DOTS[by][bx]
                char = chr(0x2800 + code)
                row_chars.append(char)
            content = "".join(row_chars)
            lines.append(f"|{content}|" if border else content)

        if border:
            lines.append("+" + "-" * self.char_width + "+")

        return "\n".join(lines)


# ANSI 24-bit color ramp for thermal density (Cold Blue -> Cyan -> Green -> Orange -> Bright Red)
COLOR_RAMP = [
    (0.00, (30, 30, 60)),     # Dark Blue
    (0.20, (0, 150, 255)),    # Cyan / Blue
    (0.40, (0, 220, 120)),    # Green
    (0.65, (240, 200, 0)),    # Yellow
    (0.85, (255, 100, 0)),    # Orange
    (1.00, (255, 30, 30)),    # Vivid Red
]

ASCII_RAMP = " .:-=+*#%@"


def render_heatmap(
    grid: List[List[float]],
    use_color: bool = True,
    border: bool = True
) -> str:
    """
    Renders a 2D scalar density grid into an ANSI color terminal heatmap.
    """
    if not grid or not grid[0]:
        return ""

    h = len(grid)
    w = len(grid[0])
    lines: List[str] = []

    if border:
        lines.append("+" + "-" * w + "+")

    for row in grid:
        row_chars: List[str] = []
        for val in row:
            clamped = max(0.0, min(1.0, val))
            # Pick character
            ascii_idx = int(clamped * (len(ASCII_RAMP) - 1))
            ch = ASCII_RAMP[ascii_idx]

            if use_color:
                # Interpolate color in ramp
                r, g, b = (30, 30, 60)
                for i in range(len(COLOR_RAMP) - 1):
                    t0, c0 = COLOR_RAMP[i]
                    t1, c1 = COLOR_RAMP[i + 1]
                    if t0 <= clamped <= t1:
                        factor = (clamped - t0) / max(1e-9, t1 - t0)
                        r = int(c0[0] + factor * (c1[0] - c0[0]))
                        g = int(c0[1] + factor * (c1[1] - c0[1]))
                        b = int(c0[2] + factor * (c1[2] - c0[2]))
                        break
                colored_ch = f"\033[38;2;{r};{g};{b}m{ch}\033[0m"
                row_chars.append(colored_ch)
            else:
                row_chars.append(ch)

        content = "".join(row_chars)
        lines.append(f"|{content}|" if border else content)

    if border:
        lines.append("+" + "-" * w + "+")

    return "\n".join(lines)
