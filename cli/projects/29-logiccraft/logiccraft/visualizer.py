"""Sub-Pixel Braille Circuit Graph Visualizer & Static Timing Analysis HUD.

Features:
1. Unicode Braille (U+2800..U+28FF) 2x4 sub-pixel canvas for delay curves and path profiles.
2. Endpoint slack distribution histogram with ANSI color coding (green/red).
3. Critical path timing waterfall diagram displaying arrival time progression.
4. Comprehensive EDA synthesis & timing closure telemetry HUD.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from .netlist import Netlist
from .sta import PathSegment, TimingReport


class BrailleCanvas:
    """High-resolution 2x4 sub-pixel canvas using Unicode Braille patterns."""

    def __init__(self, char_width: int = 70, char_height: int = 15) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4
        # Grid of active pixel states: [y][x]
        self.pixels: List[List[bool]] = [
            [False] * self.pixel_width for _ in range(self.pixel_height)
        ]

    def set_pixel(self, x: int, y: int) -> None:
        """Set a single sub-pixel."""
        if 0 <= x < self.pixel_width and 0 <= y < self.pixel_height:
            self.pixels[y][x] = True

    def draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Draw line between two pixel coordinates via Bresenham algorithm."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.set_pixel(x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def render(self) -> str:
        """Encode sub-pixels into Unicode Braille characters."""
        # Braille dot offsets:
        # col 0: (row 0: 0x01, row 1: 0x02, row 2: 0x04, row 3: 0x40)
        # col 1: (row 0: 0x08, row 1: 0x10, row 2: 0x20, row 3: 0x80)
        dot_weights = [
            [(0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40)],
            [(1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80)],
        ]

        lines: List[str] = []
        for cy in range(self.char_height):
            row_chars: List[str] = []
            for cx in range(self.char_width):
                code = 0x2800
                px_base = cx * 2
                py_base = cy * 4

                for col_idx in (0, 1):
                    px = px_base + col_idx
                    if px >= self.pixel_width:
                        continue
                    for dx_o, dy_o, weight in dot_weights[col_idx]:
                        py = py_base + dy_o
                        if py < self.pixel_height and self.pixels[py][px]:
                            code |= weight

                row_chars.append(chr(code))
            lines.append("".join(row_chars))
        return "\n".join(lines)


class CircuitVisualizer:
    """Visualizer for mapped netlists and timing performance."""

    @staticmethod
    def render_timing_hud(
        netlist: Netlist,
        report: TimingReport,
        target_name: str = "LogicCraft Circuit",
    ) -> str:
        """Render ANSI telemetry dashboard for synthesis and STA results."""
        c_cyan = "\033[96m"
        c_green = "\033[92m"
        c_yellow = "\033[93m"
        c_red = "\033[91m"
        c_bold = "\033[1m"
        c_dim = "\033[2m"
        c_reset = "\033[0m"

        status_color = c_green if report.is_timing_met else c_red
        status_text = "TIMING MET (PASSED)" if report.is_timing_met else "TIMING VIOLATED (FAILED)"

        lines: List[str] = []
        lines.append(f"{c_bold}{c_cyan}┌{'─' * 74}┐{c_reset}")
        title = f"LOGICCRAFT EDA SYNTHESIS & STA TELEMETRY: {target_name}"
        lines.append(f"{c_bold}{c_cyan}│ {c_reset}{c_bold}{title.ljust(72)}{c_cyan}│{c_reset}")
        lines.append(f"{c_bold}{c_cyan}├{'─' * 74}┤{c_reset}")

        # Physical Summary
        lines.append(
            f"{c_cyan}│{c_reset}  Module Name:       {c_bold}{netlist.name.ljust(18)}{c_reset}"
            f"Physical Area:     {c_bold}{f'{netlist.total_area:.2f} um^2'.ljust(18)}{c_reset}{c_cyan}│{c_reset}"
        )
        lines.append(
            f"{c_cyan}│{c_reset}  Standard Cells:    {c_bold}{str(netlist.num_cells).ljust(18)}{c_reset}"
            f"Leakage Power:     {c_bold}{f'{netlist.total_leakage_power:.2f} nW'.ljust(18)}{c_reset}{c_cyan}│{c_reset}"
        )
        lines.append(
            f"{c_cyan}│{c_reset}  Electrical Nets:   {c_bold}{str(netlist.num_nets).ljust(18)}{c_reset}"
            f"I/O Port Count:    {c_bold}{f'{len(netlist.inputs)} In / {len(netlist.outputs)} Out'.ljust(18)}{c_reset}{c_cyan}│{c_reset}"
        )

        lines.append(f"{c_bold}{c_cyan}├{'─' * 74}┤{c_reset}")

        # Timing Summary
        lines.append(
            f"{c_cyan}│{c_reset}  Clock Period:      {c_bold}{f'{report.clock_period:.1f} ps'.ljust(18)}{c_reset}"
            f"Max Operating Freq:{c_bold}{c_green}{f'{report.max_frequency_ghz:.2f} GHz'.ljust(18)}{c_reset}{c_cyan}│{c_reset}"
        )
        wns_color = c_green if report.worst_negative_slack >= 0 else c_red
        tns_color = c_green if report.total_negative_slack >= 0 else c_red
        lines.append(
            f"{c_cyan}│{c_reset}  Worst Slack (WNS): {c_bold}{wns_color}{f'{report.worst_negative_slack:.2f} ps'.ljust(18)}{c_reset}"
            f"Critical Path Delay:{c_bold}{f'{report.max_delay:.2f} ps'.ljust(17)}{c_reset}{c_cyan}│{c_reset}"
        )
        lines.append(
            f"{c_cyan}│{c_reset}  Total Slack (TNS): {c_bold}{tns_color}{f'{report.total_negative_slack:.2f} ps'.ljust(18)}{c_reset}"
            f"Timing Status:     {c_bold}{status_color}{status_text.ljust(18)}{c_reset}{c_cyan}│{c_reset}"
        )

        lines.append(f"{c_bold}{c_cyan}└{'─' * 74}┘{c_reset}")
        return "\n".join(lines)

    @staticmethod
    def render_critical_path_waterfall(report: TimingReport, max_width: int = 70) -> str:
        """Render step-by-step timing delay waterfall for the critical path."""
        c_cyan = "\033[96m"
        c_yellow = "\033[93m"
        c_bold = "\033[1m"
        c_dim = "\033[2m"
        c_reset = "\033[0m"

        if not report.critical_path:
            return "No critical path extracted."

        lines: List[str] = []
        lines.append(f"{c_bold}CRITICAL PATH TIMING WATERFALL:{c_reset}")
        header = f"{'Pin / Port':<22} {'Cell Type':<12} {'Incr (ps)':<11} {'Arrival (ps)':<14} {'Timeline'}"
        lines.append(f"{c_dim}{header}{c_reset}")
        lines.append(f"{c_dim}{'─' * 76}{c_reset}")

        clock_t = max(report.clock_period, report.max_delay * 1.05)

        for seg in report.critical_path:
            pin_short = seg.pin[-20:] if len(seg.pin) > 20 else seg.pin
            incr_str = f"+{seg.edge_delay:.1f}" if seg.edge_delay > 0 else "0.0"
            at_str = f"{seg.arrival_time:.1f}"

            # Calculate bar position
            bar_len = 20
            pos = int((seg.arrival_time / clock_t) * bar_len)
            pos = max(0, min(bar_len - 1, pos))
            bar = " " * pos + "█" + "░" * (bar_len - 1 - pos)

            line = f"{pin_short:<22} {seg.cell_type:<12} {incr_str:<11} {at_str:<14} [{bar}]"
            lines.append(line)

        lines.append(f"{c_dim}{'─' * 76}{c_reset}")
        return "\n".join(lines)

    @staticmethod
    def render_delay_curve_braille(delays: List[float], width: int = 60, height: int = 8) -> str:
        """Plot progressive delay curve in sub-pixel Unicode Braille."""
        if not delays:
            return ""

        canvas = BrailleCanvas(char_width=width, char_height=height)
        max_d = max(delays) if max(delays) > 0 else 1.0
        n = len(delays)

        prev_px = 0
        prev_py = canvas.pixel_height - 1 - int((delays[0] / max_d) * (canvas.pixel_height - 1))

        for i in range(1, n):
            curr_px = int((i / (n - 1)) * (canvas.pixel_width - 1))
            curr_py = canvas.pixel_height - 1 - int((delays[i] / max_d) * (canvas.pixel_height - 1))
            curr_py = max(0, min(canvas.pixel_height - 1, curr_py))
            canvas.draw_line(prev_px, prev_py, curr_px, curr_py)
            prev_px, prev_py = curr_px, curr_py

        return canvas.render()
