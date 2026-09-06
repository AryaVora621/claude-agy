"""
Solida: 3D Sub-Pixel Unicode Braille CAD Visualizer & Telemetry HUD.
Uses 2x4 sub-pixel Braille canvas (U+2800..U+28FF) with depth-buffering,
3D orbital camera, wireframe & Lambertian shaded rasterization, and CAD telemetry HUD.
Pure Python standard library. Zero external dependencies.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Sequence
from solida.geometry import Vector3D, Matrix4x4, BoundingBox3D
from solida.brep import Solid
from solida.tessellation import tessellate_solid, Facet3D


BRAILLE_BASE = 0x2800
# Dot offset bitmasks for 2x4 Braille grid
DOT_MASKS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


class Camera3D:
    """3D Orbital Camera with perspective or orthographic projection."""
    __slots__ = ("target", "distance", "azimuth_rad", "elevation_rad", "is_perspective", "fov_rad")

    def __init__(
        self,
        target: Optional[Vector3D] = None,
        distance: float = 10.0,
        azimuth_deg: float = 35.0,
        elevation_deg: float = 25.0,
        is_perspective: bool = False,
        fov_deg: float = 45.0,
    ) -> None:
        self.target = Vector3D(0, 0, 0) if target is None else target
        self.distance = max(0.1, float(distance))
        self.azimuth_rad = math.radians(float(azimuth_deg))
        self.elevation_rad = math.radians(float(elevation_deg))
        self.is_perspective = bool(is_perspective)
        self.fov_rad = math.radians(float(fov_deg))

    def eye_position(self) -> Vector3D:
        """Calculates 3D eye position from orbital angles."""
        cos_el = math.cos(self.elevation_rad)
        sin_el = math.sin(self.elevation_rad)
        cos_az = math.cos(self.azimuth_rad)
        sin_az = math.sin(self.azimuth_rad)

        offset = Vector3D(
            self.distance * cos_el * sin_az,
            self.distance * cos_el * cos_az,
            self.distance * sin_el,
        )
        return self.target + offset

    def view_matrix(self) -> Matrix4x4:
        eye = self.eye_position()
        up = Vector3D(0, 0, 1)
        if abs(math.cos(self.elevation_rad)) < 1e-4:
            up = Vector3D(0, 1, 0)
        return Matrix4x4.look_at(eye, self.target, up)


class BrailleCADCanvas:
    """
    Sub-pixel 2x4 Unicode Braille graphics buffer with floating-point depth buffer.
    """
    __slots__ = ("char_width", "char_height", "width", "height", "grid", "z_buffer", "color_buffer")

    def __init__(self, char_width: int = 70, char_height: int = 22) -> None:
        self.char_width = max(10, int(char_width))
        self.char_height = max(6, int(char_height))
        # Sub-pixel dimensions (2 dots wide, 4 dots high per char)
        self.width = self.char_width * 2
        self.height = self.char_height * 4

        # Char grid stores bitmask for U+2800
        self.grid = [[0] * self.char_width for _ in range(self.char_height)]
        # Depth buffer (higher z is closer to viewer)
        self.z_buffer = [[-float("inf")] * self.width for _ in range(self.height)]
        # RGB color buffer: (r, g, b) per char cell
        self.color_buffer: List[List[Optional[Tuple[int, int, int]]]] = [
            [None] * self.char_width for _ in range(self.char_height)
        ]

    def set_subpixel(
        self,
        px: int,
        py: int,
        z: float,
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> bool:
        """Sets a sub-pixel dot if depth test passes."""
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return False

        if z < self.z_buffer[py][px]:
            return False

        self.z_buffer[py][px] = z
        cx = px // 2
        cy = py // 4
        dx = px % 2
        dy = py % 4

        self.grid[cy][cx] |= DOT_MASKS[dy][dx]
        if color_rgb is not None:
            self.color_buffer[cy][cx] = color_rgb
        return True

    def draw_line_3d(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Bresenham 3D line rasterizer on sub-pixel grid with depth interpolation."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        steps = int(max(abs(dx), abs(dy), 1.0))

        step_x = dx / steps
        step_y = dy / steps
        step_z = dz / steps

        curr_x = x0
        curr_y = y0
        curr_z = z0

        for _ in range(steps + 1):
            px = int(round(curr_x))
            py = int(round(curr_y))
            self.set_subpixel(px, py, curr_z, color_rgb)
            curr_x += step_x
            curr_y += step_y
            curr_z += step_z

    def render_to_string(self, use_ansi_color: bool = True) -> str:
        """Encodes sub-pixel grid into terminal string with TrueColor ANSI colors."""
        lines: List[str] = []
        for cy in range(self.char_height):
            row_chars: List[str] = []
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(BRAILLE_BASE + mask)
                color = self.color_buffer[cy][cx]

                if use_ansi_color and color is not None and mask > 0:
                    r, g, b = color
                    row_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    row_chars.append(char)
            lines.append("".join(row_chars))
        return "\n".join(lines)


def project_point_to_screen(
    point: Vector3D,
    camera: Camera3D,
    canvas_width: int,
    canvas_height: int,
    bbox: BoundingBox3D,
) -> Tuple[float, float, float]:
    """
    Transforms 3D world coordinate into 2D sub-pixel screen space with depth.
    """
    view_mat = camera.view_matrix()
    p_cam = view_mat.transform_point(point)

    diag = max(bbox.diagonal(), 0.1)
    scale = (min(canvas_width, canvas_height) * 0.42) / (diag * 0.6)

    # Invert Y for screen rasterization (Y grows downward in terminal)
    sx = canvas_width * 0.5 + p_cam.x * scale
    sy = canvas_height * 0.5 - p_cam.y * scale
    sz = p_cam.z
    return (sx, sy, sz)


def render_solid_cad(
    solid: Solid,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    char_width: int = 68,
    char_height: int = 18,
    mode: str = "wireframe",
    use_ansi_color: bool = True,
) -> str:
    """
    Renders 3D B-Rep CAD solid to sub-pixel Unicode Braille terminal graphic.
    Modes:
      - 'wireframe': Renders topological edges with depth testing.
      - 'shaded': Renders Lambertian diffuse lit facets with 24-bit TrueColor cyan/amber lighting.
    """
    bbox = solid.bounding_box()
    center = bbox.center()
    diag = bbox.diagonal()
    cam = Camera3D(
        target=center,
        distance=diag * 2.5,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )

    canvas = BrailleCADCanvas(char_width=char_width, char_height=char_height)
    w_sub = canvas.width
    h_sub = canvas.height

    if mode == "shaded":
        # Key light source
        light_dir = Vector3D(0.5, 0.6, 0.8).normalized()
        facets = tessellate_solid(solid)

        for f in facets:
            p0_s = project_point_to_screen(f.v0, cam, w_sub, h_sub, bbox)
            p1_s = project_point_to_screen(f.v1, cam, w_sub, h_sub, bbox)
            p2_s = project_point_to_screen(f.v2, cam, w_sub, h_sub, bbox)

            # Lambertian shading: diffuse = max(0.2, n . l)
            diffuse = max(0.25, f.normal.dot(light_dir))
            r = int(min(255, 30 + diffuse * 180))
            g = int(min(255, 140 + diffuse * 115))
            b = int(min(255, 200 + diffuse * 55))
            col = (r, g, b)

            # Draw triangle outline
            canvas.draw_line_3d(p0_s, p1_s, col)
            canvas.draw_line_3d(p1_s, p2_s, col)
            canvas.draw_line_3d(p2_s, p0_s, col)

            # Draw interior median lines for surface density
            pm = (f.v0 + f.v1) * 0.5
            pm_s = project_point_to_screen(pm, cam, w_sub, h_sub, bbox)
            canvas.draw_line_3d(pm_s, p2_s, col)

    else:
        # Wireframe mode: render all manifold B-Rep boundary edges
        edge_color = (0, 220, 255)  # Cyan
        for edge in solid.edges:
            v0 = edge.half_edge.origin.point
            v1 = edge.half_edge.target.point
            p0_s = project_point_to_screen(v0, cam, w_sub, h_sub, bbox)
            p1_s = project_point_to_screen(v1, cam, w_sub, h_sub, bbox)
            canvas.draw_line_3d(p0_s, p1_s, edge_color)

    return canvas.render_to_string(use_ansi_color=use_ansi_color)


def render_cad_hud(
    solid: Solid,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 25.0,
    width: int = 76,
) -> str:
    """
    Renders CAD engineering telemetry HUD with volume, surface area,
    Euler characteristic invariants, and bounding extents.
    """
    vol = solid.volume()
    area = solid.surface_area()
    cm = solid.center_of_mass()
    bbox = solid.bounding_box()
    ext = bbox.extents()

    v_count = len(solid.vertices)
    e_count = len(solid.edges)
    f_count = len(solid.faces)
    chi = solid.outer_shell.euler_characteristic()
    is_watertight = all(edge.is_manifold for edge in solid.edges)
    manifold_str = "2-MANIFOLD [OK]" if is_watertight else "NON-MANIFOLD [WARN]"

    top_border = "+" + "=" * (width - 2) + "+"
    bottom_border = "+" + "=" * (width - 2) + "+"
    sep = "|" + "-" * (width - 2) + "|"

    lines: List[str] = [
        top_border,
        f"|  SOLIDA CAD MODEL TELEMETRY: {solid.name.upper():<40}  |",
        sep,
        f"|  Volume:       {vol:14.4f} mm3  |  Area:        {area:14.4f} mm2  |",
        f"|  Centroid:    ({cm.x:6.2f}, {cm.y:6.2f}, {cm.z:6.2f})  |  Bounding Ext: ({ext.x:5.1f}x{ext.y:5.1f}x{ext.z:5.1f})   |",
        sep,
        f"|  Topology:     V={v_count:<4} E={e_count:<4} F={f_count:<4}   |  Euler Chi:    chi={chi:<3} (V-E+F={chi})   |",
        f"|  Integrity:    {manifold_str:<18}    |  Camera:       Az={azimuth_deg:4.0f} deg El={elevation_deg:3.0f} deg  |",
        bottom_border,
    ]
    return "\n".join(lines)
