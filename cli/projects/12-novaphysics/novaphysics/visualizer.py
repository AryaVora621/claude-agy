"""
NovaPhysics: High-Performance ANSI Terminal Physics Visualizer.
Supports sub-pixel Unicode Braille rasterization (2x4 pixels per character),
rigid body polygon rendering, joint linkage visualization, contact point overlays, and live telemetry.
"""

import math
import sys
import time
from typing import List, Optional, Tuple
from .math2d import Vec2
from .shapes import ShapeType, Circle, Polygon, Box
from .body import RigidBody, BodyType
from .world import World
from .joints import DistanceJoint, RevoluteJoint, SpringJoint
from .cloth import ClothMesh

# Braille bit mapping (col, row):
# (0,0): 0x01, (0,1): 0x02, (0,2): 0x04, (0,3): 0x40
# (1,0): 0x08, (1,1): 0x10, (1,2): 0x20, (1,3): 0x80
BRAILLE_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80]
]


class BrailleCanvas:
    """Sub-pixel terminal canvas offering 2x4 sub-pixels per character cell."""
    def __init__(self, char_width: int = 80, char_height: int = 30) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4
        self.grid = [[0 for _ in range(char_width)] for _ in range(char_height)]

    def clear(self) -> None:
        for r in range(self.char_height):
            for c in range(self.char_width):
                self.grid[r][c] = 0

    def set_pixel(self, px: int, py: int) -> None:
        """Sets a sub-pixel at (px, py) with origin at bottom-left."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            # Flip py so y=0 is bottom
            screen_y = (self.pixel_height - 1) - py
            char_c = px // 2
            char_r = screen_y // 4
            sub_c = px % 2
            sub_r = screen_y % 4
            self.grid[char_r][char_c] |= BRAILLE_MAP[sub_r][sub_c]

    def draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham's sub-pixel line rasterization algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while True:
            self.set_pixel(curr_x, curr_y)
            if curr_x == x1 and curr_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

    def draw_circle(self, cx: int, cy: int, radius: int) -> None:
        """Midpoint circle rasterization algorithm."""
        x = radius
        y = 0
        err = 0

        while x >= y:
            self.set_pixel(cx + x, cy + y)
            self.set_pixel(cx + y, cy + x)
            self.set_pixel(cx - y, cy + x)
            self.set_pixel(cx - x, cy + y)
            self.set_pixel(cx - x, cy - y)
            self.set_pixel(cx - y, cy - x)
            self.set_pixel(cx + y, cy - x)
            self.set_pixel(cx + x, cy - y)

            if err <= 0:
                y += 1
                err += 2 * y + 1
            if err > 0:
                x -= 1
                err -= 2 * x + 1

    def render(self) -> str:
        """Converts internal bit grid to Unicode Braille string."""
        lines = []
        for r in range(self.char_height):
            line_chars = []
            for c in range(self.char_width):
                bits = self.grid[r][c]
                if bits == 0:
                    line_chars.append(" ")
                else:
                    line_chars.append(chr(0x2800 + bits))
            lines.append("".join(line_chars))
        return "\n".join(lines)


class PhysicsRenderer:
    """High-level physics scene rasterizer mapping world coordinates to terminal screen."""
    def __init__(
        self,
        char_width: int = 76,
        char_height: int = 24,
        view_center: Vec2 = Vec2(0.0, 4.0),
        view_scale: float = 12.0
    ) -> None:
        self.canvas = BrailleCanvas(char_width, char_height)
        self.view_center = view_center
        self.view_scale = view_scale  # Pixels per world meter

    def world_to_pixel(self, world_pt: Vec2) -> Tuple[int, int]:
        """Convert world coordinates (meters) to sub-pixel canvas coordinates."""
        half_w = self.canvas.pixel_width * 0.5
        half_h = self.canvas.pixel_height * 0.5

        px = int(half_w + (world_pt.x - self.view_center.x) * self.view_scale)
        py = int(half_h + (world_pt.y - self.view_center.y) * self.view_scale)
        return px, py

    def render_world(self, world: World, cloth: Optional[ClothMesh] = None) -> str:
        """Renders complete world state including rigid bodies, joints, contacts, and cloth."""
        self.canvas.clear()

        # 1. Render Rigid Bodies
        for body in world.bodies:
            shape = body.shape
            if shape.shape_type == ShapeType.CIRCLE:
                c_circ: Circle = shape  # type: ignore
                center_world = body.transform.transform_point(c_circ.center)
                cx, cy = self.world_to_pixel(center_world)
                r_pix = max(1, int(c_circ.radius * self.view_scale))
                self.canvas.draw_circle(cx, cy, r_pix)

                # Draw angle orientation spoke
                rim_pt = center_world + Vec2(math.cos(body.angle), math.sin(body.angle)) * c_circ.radius
                rx, ry = self.world_to_pixel(rim_pt)
                self.canvas.draw_line(cx, cy, rx, ry)

            elif shape.shape_type == ShapeType.POLYGON:
                poly: Polygon = shape  # type: ignore
                verts_world = [body.transform.transform_point(v) for v in poly.vertices]
                pix_pts = [self.world_to_pixel(pt) for pt in verts_world]

                n = len(pix_pts)
                for i in range(n):
                    x0, y0 = pix_pts[i]
                    x1, y1 = pix_pts[(i + 1) % n]
                    self.canvas.draw_line(x0, y0, x1, y1)

        # 2. Render Joints
        for joint in world.joints:
            if isinstance(joint, (DistanceJoint, SpringJoint)):
                w_a = joint.body_a.transform.transform_point(joint.local_anchor_a)
                w_b = joint.body_b.transform.transform_point(joint.local_anchor_b)
                x0, y0 = self.world_to_pixel(w_a)
                x1, y1 = self.world_to_pixel(w_b)
                self.canvas.draw_line(x0, y0, x1, y1)
            elif isinstance(joint, RevoluteJoint):
                w_pivot = joint.body_a.transform.transform_point(joint.local_anchor_a)
                px, py = self.world_to_pixel(w_pivot)
                self.canvas.draw_circle(px, py, 2)

        # 3. Render Cloth if present
        if cloth is not None:
            for c in cloth.constraints:
                if not c.is_broken:
                    x0, y0 = self.world_to_pixel(c.p1.position)
                    x1, y1 = self.world_to_pixel(c.p2.position)
                    self.canvas.draw_line(x0, y0, x1, y1)

        # Build telemetry banner
        ascii_grid = self.canvas.render()
        kinetic_energy = sum(
            0.5 * b.mass * b.velocity.length_sq() + 0.5 * b.inertia * (b.angular_velocity ** 2)
            for b in world.bodies if b.body_type == BodyType.DYNAMIC
        )

        header = (
            f"\033[1;36m┌─── NovaPhysics Engine ────────────────────────────────────────────────────────┐\033[0m\n"
            f"\033[1;36m│\033[0m Time: {world.time:6.2f}s │ Step: {world.step_count:5d} │ Bodies: {len(world.bodies):2d} │ "
            f"Joints: {len(world.joints):2d} │ KE: {kinetic_energy:8.2f} J \033[1;36m│\033[0m\n"
            f"\033[1;36m├─── Canvas (Braille 2x4 Sub-pixel) ──────────────────────────────────────────┤\033[0m"
        )
        footer = f"\033[1;36m└─────────────────────────────────────────────────────────────────────────────┘\033[0m"

        return f"{header}\n{ascii_grid}\n{footer}"
