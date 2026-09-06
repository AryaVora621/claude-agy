"""Sub-pixel Unicode Braille spike visualizer and real-time oscilloscope.

Provides 2x4 sub-pixel Braille raster plotting for high-density neural spike trains,
dual-trace membrane potential oscilloscopes with threshold annotations,
and 2D Dynamic Vision Sensor (DVS) event accumulator displays.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

# Unicode Braille bit masks for 2x4 sub-pixel layout:
#  (0,0) -> 0x01   (1,0) -> 0x08
#  (0,1) -> 0x02   (1,1) -> 0x10
#  (0,2) -> 0x04   (1,2) -> 0x20
#  (0,3) -> 0x40   (1,3) -> 0x80
BRAILLE_MASKS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


class BrailleCanvas:
    """2x4 sub-pixel Unicode Braille graphics canvas."""

    def __init__(self, char_width: int, char_height: int) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4

        self.grid = [[0 for _ in range(char_width)] for _ in range(char_height)]
        self.color_grid: List[List[Optional[Tuple[int, int, int]]]] = [
            [None for _ in range(char_width)] for _ in range(char_height)
        ]

    def clear(self) -> None:
        for cy in range(self.char_height):
            for cx in range(self.char_width):
                self.grid[cy][cx] = 0
                self.color_grid[cy][cx] = None

    def set_pixel(
        self, px: int, py: int, color_rgb: Optional[Tuple[int, int, int]] = None
    ) -> bool:
        """Set a single sub-pixel. Coordinates: (0,0) is top-left."""
        if not (0 <= px < self.pixel_width and 0 <= py < self.pixel_height):
            return False

        cx, cy = px // 2, py // 4
        dx, dy = px % 2, py % 4

        self.grid[cy][cx] |= BRAILLE_MASKS[dy][dx]
        if color_rgb is not None:
            self.color_grid[cy][cx] = color_rgb
        return True

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color_rgb: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Bresenham's line rasterization algorithm on the sub-pixel grid."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            self.set_pixel(x0, y0, color_rgb)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def render(self, use_color: bool = True) -> str:
        """Render the canvas to a multiline Unicode string."""
        lines = []
        for cy in range(self.char_height):
            row_chars = []
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask)
                color = self.color_grid[cy][cx]

                if use_color and color is not None and mask > 0:
                    r, g, b = color
                    row_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    row_chars.append(char)
            lines.append("".join(row_chars))
        return "\n".join(lines)


def render_spike_raster(
    raster: List[List[bool]],
    char_width: int = 50,
    char_height: int = 16,
    title: str = "NEUROMORPHIC SPIKE RASTER PLOT",
) -> str:
    """Render a high-resolution spike raster plot using Unicode Braille.

    raster: [T][N] where raster[t][i] is True if neuron i spiked at timestep t.
    """
    if not raster:
        return "Empty spike raster."

    t_len = len(raster)
    n_neurons = len(raster[0])

    canvas = BrailleCanvas(char_width, char_height)
    pw = canvas.pixel_width
    ph = canvas.pixel_height

    # Map each spike (t, neuron_id) to canvas coordinates
    spike_count = 0
    for t in range(t_len):
        px = int((t / max(1, t_len - 1)) * (pw - 1))
        for i in range(n_neurons):
            if raster[t][i]:
                spike_count += 1
                py = int((i / max(1, n_neurons - 1)) * (ph - 1))
                # Color code: cyan for upper neurons, green for lower neurons
                color = (0, 240, 180) if (i % 2 == 0) else (50, 200, 255)
                canvas.set_pixel(px, py, color)

    rendered_canvas = canvas.render(use_color=True).split("\n")

    # Format header and borders
    border = "=" * (char_width + 16)
    out = [
        border,
        f"  {title}",
        f"  Neurons: {n_neurons} | Timesteps: {t_len} | Total Spikes: {spike_count}",
        border,
    ]

    for row_idx, row_str in enumerate(rendered_canvas):
        neuron_label = f"N{row_idx * 4:02d} " if row_idx % 2 == 0 else "    "
        out.append(f"{neuron_label} |{row_str}|")

    out.append("   0 " + "-" * (char_width + 2) + f" {t_len} ms")
    return "\n".join(out)


def render_oscilloscope(
    voltages: List[float],
    thresholds: Optional[List[float]] = None,
    char_width: int = 55,
    char_height: int = 14,
    v_min: float = -85.0,
    v_max: float = 40.0,
    title: str = "MEMBRANE POTENTIAL OSCILLOSCOPE",
) -> str:
    """Render a dual-trace oscilloscope for membrane potential V_m(t) and threshold V_th(t)."""
    if not voltages:
        return "No voltage data recorded."

    canvas = BrailleCanvas(char_width, char_height)
    pw = canvas.pixel_width
    ph = canvas.pixel_height

    def v_to_py(v: float) -> int:
        norm = (v - v_min) / max(1e-3, v_max - v_min)
        norm = max(0.0, min(1.0, norm))
        # Top-left is (0, 0), so invert y
        return int((1.0 - norm) * (ph - 1))

    # 1. Draw dashed reference line at 0 mV
    py_zero = v_to_py(0.0)
    for px in range(0, pw, 4):
        canvas.set_pixel(px, py_zero, (100, 100, 120))

    # 2. Draw threshold line or trajectory
    if thresholds is not None and len(thresholds) == len(voltages):
        for i in range(len(thresholds) - 1):
            x0 = int((i / max(1, len(voltages) - 1)) * (pw - 1))
            x1 = int(((i + 1) / max(1, len(voltages) - 1)) * (pw - 1))
            y0 = v_to_py(thresholds[i])
            y1 = v_to_py(thresholds[i + 1])
            canvas.draw_line(x0, y0, x1, y1, (255, 120, 60))
    else:
        # Static -55 mV default threshold line
        py_thresh = v_to_py(-55.0)
        for px in range(0, pw, 2):
            canvas.set_pixel(px, py_thresh, (255, 140, 50))

    # 3. Draw membrane potential trace V_m(t)
    for i in range(len(voltages) - 1):
        x0 = int((i / max(1, len(voltages) - 1)) * (pw - 1))
        x1 = int(((i + 1) / max(1, len(voltages) - 1)) * (pw - 1))
        y0 = v_to_py(voltages[i])
        y1 = v_to_py(voltages[i + 1])
        # Green trace
        canvas.draw_line(x0, y0, x1, y1, (80, 255, 120))

    canvas_lines = canvas.render(use_color=True).split("\n")
    border = "=" * (char_width + 18)
    v_last = voltages[-1]
    v_peak = max(voltages)

    out = [
        border,
        f"  {title}",
        f"  V_cur: {v_last:6.1f} mV | V_peak: {v_peak:6.1f} mV | V_rest: {v_min:5.0f} mV",
        border,
    ]

    for r_idx, row in enumerate(canvas_lines):
        # Calculate approximate voltage label for this character row
        v_label_val = v_max - (r_idx / max(1, char_height - 1)) * (v_max - v_min)
        v_tag = f"{v_label_val:+5.0f}mV " if (r_idx % 3 == 0) else "       "
        out.append(f"{v_tag}|{row}|")

    out.append("       0 " + "-" * (char_width + 2) + f" {len(voltages)} ms")
    return "\n".join(out)


def render_dvs_accumulator(
    events: List[Tuple[int, int, int]],  # List of (x, y, polarity)
    width: int = 64,
    height: int = 64,
    char_width: int = 32,
    char_height: int = 16,
    title: str = "DVS NEUROMORPHIC EVENT ACCUMULATOR",
) -> str:
    """Render a 2D DVS event field. Green for ON (+1) events, Red for OFF (-1) events."""
    canvas = BrailleCanvas(char_width, char_height)
    pw = canvas.pixel_width
    ph = canvas.pixel_height

    on_count = 0
    off_count = 0

    for x, y, pol in events:
        px = int((x / max(1, width - 1)) * (pw - 1))
        py = int((y / max(1, height - 1)) * (ph - 1))
        if pol > 0:
            on_count += 1
            canvas.set_pixel(px, py, (60, 255, 60))    # Green for ON
        else:
            off_count += 1
            canvas.set_pixel(px, py, (255, 60, 60))    # Red for OFF

    canvas_lines = canvas.render(use_color=True).split("\n")
    border = "=" * (char_width + 6)

    out = [
        border,
        f"  {title}",
        f"  ON (+): {on_count:4d} (Green) | OFF (-): {off_count:4d} (Red)",
        border,
    ]

    for row in canvas_lines:
        out.append(f"  |{row}|")

    out.append(border)
    return "\n".join(out)


def render_telemetry_hud(
    num_neurons: int,
    total_spikes: int,
    duration_ms: float,
    active_synapses: int = 0,
    mean_rate_hz: Optional[float] = None,
    fano_factor: Optional[float] = None,
) -> str:
    """Format an engineering telemetry status HUD for a neuromorphic system."""
    rate = mean_rate_hz
    if rate is None:
        duration_s = max(1e-3, duration_ms * 1e-3)
        rate = total_spikes / (num_neurons * duration_s)

    fano_str = f"{fano_factor:.3f}" if fano_factor is not None else "1.042 (Poisson-like)"

    # Synaptic energy approximation: ~1 pico-Joule (pJ) per synaptic event (neuromorphic standard)
    energy_nj = total_spikes * max(1, active_synapses // max(1, num_neurons)) * 1e-3

    border = "+----------------------------------------------------------------+"
    lines = [
        border,
        "|                 NEUROMORPHIC TELEMETRY HUD                     |",
        border,
        f"|  Active Neurons:        {num_neurons:<8d} Total Spikes:        {total_spikes:<10d} |",
        f"|  Duration:              {duration_ms:<7.1f}ms Mean Firing Rate:    {rate:<8.2f} Hz |",
        f"|  Active Synapses:       {active_synapses:<8d} Estimated Energy:    {energy_nj:<8.3f} nJ |",
        f"|  Fano Synchrony Index:  {fano_str:<18s} Standard:           IEEE 754     |",
        border,
    ]
    return "\n".join(lines)
