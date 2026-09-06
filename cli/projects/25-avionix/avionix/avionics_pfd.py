"""Electronic Flight Instrument System (EFIS) Primary Flight Display & 3D Flight Visualizer.

Implements:
1. Sub-pixel Unicode Braille Canvas (2x4 dots per character cell, U+2800..U+28FF)
   with 24-bit TrueColor ANSI color escapes.
2. 3D orbital perspective trajectory renderer showing reference path and quadrotor body axes.
3. Full Primary Flight Display (PFD) with artificial horizon, pitch ladder, roll pointer,
   calibrated airspeed tape, altimeter tape, vertical speed indicator, and heading tape.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple

from .dynamics import Matrix3x3, QuadrotorState, Quaternion, Vector3


class BrailleFlightCanvas:
    """High-density 2x4 sub-pixel Unicode Braille graphics buffer."""

    # Standard Unicode 2x4 Braille dot bit values
    # dot(0,0)=0x01, dot(0,1)=0x02, dot(0,2)=0x04, dot(0,3)=0x40
    # dot(1,0)=0x08, dot(1,1)=0x10, dot(1,2)=0x20, dot(1,3)=0x80
    PIXEL_MASKS = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(self, char_width: int = 50, char_height: int = 20) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.grid = [[0] * char_width for _ in range(char_height)]
        self.colors = [[(180, 180, 180)] * char_width for _ in range(char_height)]

    def clear(self) -> None:
        """Clear all pixels and reset colors."""
        for r in range(self.char_height):
            for c in range(self.char_width):
                self.grid[r][c] = 0
                self.colors[r][c] = (180, 180, 180)

    def set_pixel(self, x: int, y: int, r: int = 255, g: int = 255, b: int = 255) -> None:
        """Plot a single sub-pixel in canvas coordinates (0, 0 is top-left)."""
        if 0 <= x < self.pixel_width and 0 <= y < self.pixel_height:
            char_x = x // 2
            char_y = y // 4
            sub_x = x % 2
            sub_y = y % 4
            self.grid[char_y][char_x] |= self.PIXEL_MASKS[sub_y][sub_x]
            self.colors[char_y][char_x] = (r, g, b)

    def draw_line(
        self, x0: int, y0: int, x1: int, y1: int, r: int = 255, g: int = 255, b: int = 255
    ) -> None:
        """Bresenham line algorithm for sub-pixel accuracy."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            self.set_pixel(x, y, r, g, b)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def render_to_string(self) -> str:
        """Serialize grid to ANSI TrueColor terminal string."""
        lines = []
        for r in range(self.char_height):
            row_chars = []
            for c in range(self.char_width):
                mask = self.grid[r][c]
                char = chr(0x2800 + mask) if mask != 0 else " "
                cr, cg, cb = self.colors[r][c]
                if mask != 0:
                    row_chars.append(f"\033[38;2;{cr};{cg};{cb}m{char}\033[0m")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        return "\n".join(lines)


def render_3d_flight_path(
    waypoints: List[Vector3],
    actual_path: List[Vector3],
    current_state: QuadrotorState,
    char_width: int = 55,
    char_height: int = 16,
    azimuth_deg: float = 45.0,
    elevation_deg: float = 30.0,
) -> str:
    """Render 3D orbital perspective projection of reference and flown trajectory."""
    canvas = BrailleFlightCanvas(char_width, char_height)

    # All points for bounding box computation
    all_points = waypoints + actual_path + [current_state.position]
    min_x = min(p.x for p in all_points) - 1.0
    max_x = max(p.x for p in all_points) + 1.0
    min_y = min(p.y for p in all_points) - 1.0
    max_y = max(p.y for p in all_points) + 1.0
    min_z = min(p.z for p in all_points) - 0.5
    max_z = max(p.z for p in all_points) + 1.0

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    cz = 0.5 * (min_z + max_z)
    span = max(max_x - min_x, max_y - min_y, max_z - min_z, 2.0)

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cos_az, sin_az = math.cos(az), math.sin(az)
    cos_el, sin_el = math.cos(el), math.sin(el)

    def project(pt: Vector3) -> Tuple[int, int]:
        # Centered coordinate
        dx = pt.x - cx
        dy = pt.y - cy
        dz = pt.z - cz

        # Camera rotation: azimuth then elevation
        x_rot = -sin_az * dx + cos_az * dy
        y_depth = cos_az * dx + sin_az * dy
        z_cam = -sin_el * y_depth + cos_el * dz

        # Orthographic/axonometric projection scaled to canvas
        scale = min(canvas.pixel_width, canvas.pixel_height) / span * 0.75
        screen_x = int(canvas.pixel_width * 0.5 + x_rot * scale)
        screen_y = int(canvas.pixel_height * 0.5 - z_cam * scale)
        return screen_x, screen_y

    # 1. Draw reference waypoints as dashed or cyan segments
    for i in range(len(waypoints) - 1):
        p0 = project(waypoints[i])
        p1 = project(waypoints[i + 1])
        canvas.draw_line(p0[0], p0[1], p1[0], p1[1], r=60, g=180, b=255)

    # 2. Draw actual flown path in vibrant green
    for i in range(len(actual_path) - 1):
        p0 = project(actual_path[i])
        p1 = project(actual_path[i + 1])
        canvas.draw_line(p0[0], p0[1], p1[0], p1[1], r=80, g=255, b=120)

    # 3. Draw current quadrotor body arms (X-configuration)
    pos = current_state.position
    R = current_state.attitude.to_rotation_matrix()
    d = 0.6  # Visual arm length
    arm1 = pos + R.dot_vec(Vector3(d * 0.707, d * 0.707, 0.0))
    arm2 = pos + R.dot_vec(Vector3(-d * 0.707, d * 0.707, 0.0))
    arm3 = pos + R.dot_vec(Vector3(-d * 0.707, -d * 0.707, 0.0))
    arm4 = pos + R.dot_vec(Vector3(d * 0.707, -d * 0.707, 0.0))

    center_px = project(pos)
    a1_px = project(arm1)
    a2_px = project(arm2)
    a3_px = project(arm3)
    a4_px = project(arm4)

    # Front arms in orange, rear arms in red
    canvas.draw_line(center_px[0], center_px[1], a1_px[0], a1_px[1], r=255, g=140, b=50)
    canvas.draw_line(center_px[0], center_px[1], a4_px[0], a4_px[1], r=255, g=140, b=50)
    canvas.draw_line(center_px[0], center_px[1], a2_px[0], a2_px[1], r=255, g=60, b=60)
    canvas.draw_line(center_px[0], center_px[1], a3_px[0], a3_px[1], r=255, g=60, b=60)

    header = (
        f"┌{'─' * (char_width - 2)}┐\n"
        f"│ 3D ORBITAL TRAJECTORY (AZ: {azimuth_deg:.0f}° | EL: {elevation_deg:.0f}°) {' ' * (char_width - 39)}│\n"
        f"├{'─' * (char_width - 2)}┤"
    )
    footer = f"└{'─' * (char_width - 2)}┘"
    return f"{header}\n{canvas.render_to_string()}\n{footer}"


def render_primary_flight_display(
    state: QuadrotorState,
    target_pos: Optional[Vector3] = None,
    flight_mode: str = "NAV SE(3)",
    display_width: int = 74,
) -> str:
    """Render standard EFIS Primary Flight Display (PFD) with artificial horizon."""
    roll_rad, pitch_rad, yaw_rad = state.attitude.to_euler()
    roll_deg = math.degrees(roll_rad)
    pitch_deg = math.degrees(pitch_rad)
    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0

    airspeed_ms = state.velocity.norm()
    altitude_m = state.position.z
    vsi_ms = state.velocity.z

    target_alt = target_pos.z if target_pos else altitude_m
    alt_error = target_alt - altitude_m

    lines = []
    # 1. Flight Mode Annunciator (FMA) Top Bar
    fma = f"│ [ {flight_mode:^12} ]   [ ALT HOLD: {target_alt:5.1f}m ]   [ ARMED: AUTO ] │"
    lines.append(f"┌{'─' * (display_width - 2)}┐")
    lines.append(f"{fma:^{display_width}}")
    lines.append(f"├{'─' * 14}┬{'─' * 42}┬{'─' * 14}┤")

    # 2. Main Instrument Area (Airspeed Tape | Artificial Horizon | Altimeter Tape)
    horizon_rows = 11
    horizon_cols = 42

    for r in range(horizon_rows):
        # Y coordinate relative to center reticle (-5 to +5)
        y_rel = 5 - r

        # Left Airspeed Tape
        cas_val = max(0.0, airspeed_ms + (y_rel * 1.5))
        spd_tape = f"│ {cas_val:5.1f} m/s " if r % 2 == 0 else "│   ───    "
        if y_rel == 0:
            spd_tape = f"│>\033[1;33m{airspeed_ms:5.1f}\033[0m m/s "

        # Center Artificial Horizon calculation
        # Horizon line equation: y = tan(roll) * x + pitch_offset
        tan_r = math.tan(roll_rad) if abs(roll_rad) < math.radians(85) else 100.0
        pitch_pix_offset = pitch_deg * 0.35

        horizon_line = []
        for c in range(horizon_cols):
            x_rel = c - 21
            y_line = tan_r * (x_rel * 0.3) + pitch_pix_offset

            if abs(y_rel - y_line) < 0.6:
                # Horizon divider line (White)
                horizon_line.append("\033[1;37m━\033[0m")
            elif y_rel > y_line:
                # Sky (Blue)
                # Check for pitch ladder ticks (+10, +20)
                if abs(y_rel - (pitch_pix_offset + 3)) < 0.3 and abs(x_rel) < 6:
                    horizon_line.append("\033[1;37m+\033[0m")
                else:
                    horizon_line.append("\033[38;2;60;140;240m░\033[0m")
            else:
                # Ground (Brown)
                # Check for negative pitch ladder ticks (-10, -20)
                if abs(y_rel - (pitch_pix_offset - 3)) < 0.3 and abs(x_rel) < 6:
                    horizon_line.append("\033[1;37m-\033[0m")
                else:
                    horizon_line.append("\033[38;2;140;90;50m░\033[0m")

        # Overlay central boresight reticle at row 5 (y_rel == 0)
        if y_rel == 0:
            horizon_line[17] = "\033[1;33m>\033[0m"
            horizon_line[18] = "\033[1;33m─\033[0m"
            horizon_line[19] = "\033[1;33m─\033[0m"
            horizon_line[20] = "\033[1;33m●\033[0m"
            horizon_line[21] = "\033[1;33m─\033[0m"
            horizon_line[22] = "\033[1;33m─\033[0m"
            horizon_line[23] = "\033[1;33m<\033[0m"

        center_horizon = "".join(horizon_line)

        # Right Altimeter Tape
        alt_val = max(0.0, altitude_m + (y_rel * 2.0))
        alt_tape = f" {alt_val:6.1f}m │" if r % 2 == 0 else "   ───    │"
        if y_rel == 0:
            alt_tape = f"<\033[1;33m{altitude_m:6.1f}\033[0m m│"

        lines.append(f"{spd_tape}│{center_horizon}│{alt_tape}")

    # 3. Attitude Readings Line & Vertical Speed
    lines.append(f"├{'─' * 14}┴{'─' * 42}┴{'─' * 14}┤")
    roll_str = f"ROLL: {roll_deg:+5.1f}°"
    pitch_str = f"PITCH: {pitch_deg:+5.1f}°"
    vsi_str = f"VSI: {vsi_ms:+4.1f} m/s"
    lines.append(f"│  {roll_str:<18}   {pitch_str:<18}   {vsi_str:<22} │")

    # 4. Heading Tape / Compass Ribbon
    lines.append(f"├{'─' * (display_width - 2)}┤")
    hdg_center = int(yaw_deg)
    tape_chars = []
    for offset in range(-15, 16):
        h = (hdg_center + offset) % 360
        if offset == 0:
            tape_chars.append(f"\033[1;33m[{h:03d}]\033[0m")
        elif h % 15 == 0:
            # Cardinal points
            cardinal = {0: "N", 90: "E", 180: "S", 270: "W"}.get(h, f"{h//10}")
            tape_chars.append(cardinal)
        elif h % 5 == 0:
            tape_chars.append("·")
        else:
            tape_chars.append(" ")

    heading_ribbon = "".join(tape_chars[:35])
    lines.append(f"│ HDG: ─── {heading_ribbon:^54} ─── │")

    # 5. Bottom Telemetry Status Bar
    lines.append(f"├{'─' * (display_width - 2)}┤")
    rpm_info = f"M1: {state.rotor_speeds[0]:.0f} | M2: {state.rotor_speeds[1]:.0f} | M3: {state.rotor_speeds[2]:.0f} | M4: {state.rotor_speeds[3]:.0f} rad/s"
    pos_info = f"POS: X:{state.position.x:+5.2f}m | Y:{state.position.y:+5.2f}m | Z:{state.position.z:+5.2f}m"
    lines.append(f"│  {pos_info:<33}  {rpm_info:<33} │")
    lines.append(f"└{'─' * (display_width - 2)}┘")

    return "\n".join(lines)
