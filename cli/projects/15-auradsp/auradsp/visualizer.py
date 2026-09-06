"""
AuraDSP: High-Resolution Unicode Braille Spectrogram Waterfall & Terminal Visualizer.
Renders continuous audio spectrograms and frequency bars into ANSI terminals
using 2x4 sub-pixel Braille characters and 24-bit TrueColor spectral palettes.
"""

import math
from typing import List, Tuple

# Unicode Braille 2x4 dot bitmasks
# Dot indices:
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

# Spectral color ramp (Dark Navy -> Purple -> Magenta -> Amber Orange -> Bright White)
SPECTRAL_RAMP = [
    (0.00, (15, 15, 45)),      # Deep Navy (silence)
    (0.20, (60, 20, 110)),     # Dark Violet
    (0.40, (160, 30, 130)),    # Purple-Magenta
    (0.65, (235, 80, 50)),     # Coral Red
    (0.85, (255, 190, 20)),    # Amber Gold
    (1.00, (255, 255, 220)),   # Bright Yellow/White (peak)
]

BLOCK_CHARS = "  ▂▃▄▅▆▇█"


def _interpolate_color(val: float) -> Tuple[int, int, int]:
    clamped = max(0.0, min(1.0, val))
    for i in range(len(SPECTRAL_RAMP) - 1):
        t0, c0 = SPECTRAL_RAMP[i]
        t1, c1 = SPECTRAL_RAMP[i + 1]
        if t0 <= clamped <= t1:
            factor = (clamped - t0) / max(1e-9, t1 - t0)
            r = int(c0[0] + factor * (c1[0] - c0[0]))
            g = int(c0[1] + factor * (c1[1] - c0[1]))
            b = int(c0[2] + factor * (c1[2] - c0[2]))
            return r, g, b
    return SPECTRAL_RAMP[-1][1]


def render_braille_waterfall(
    matrix: List[List[float]],
    width_chars: int = 60,
    height_chars: int = 18,
    threshold: float = 0.25,
    use_color: bool = True,
    border: bool = True
) -> str:
    """
    Renders a 2D spectrogram matrix (time x frequency) into a sub-pixel Braille waterfall.
    Time progresses left-to-right (X axis).
    Frequency increases bottom-to-top (Y axis).
    """
    if not matrix or not matrix[0]:
        return ""

    num_time = len(matrix)
    num_freq = len(matrix[0])

    pixel_width = width_chars * 2
    pixel_height = height_chars * 4

    # Sample sub-pixel grid
    pixel_grid = [[0 for _ in range(pixel_width)] for _ in range(pixel_height)]
    pixel_energy = [[0.0 for _ in range(pixel_width)] for _ in range(pixel_height)]

    for py in range(pixel_height):
        # py = 0 is highest frequency, py = pixel_height - 1 is lowest frequency
        norm_y = 1.0 - (py / max(1, pixel_height - 1))
        # Logarithmic-like frequency mapping for better bass/mid resolution
        freq_idx = int((norm_y ** 1.3) * (num_freq - 1))
        freq_idx = max(0, min(num_freq - 1, freq_idx))

        for px in range(pixel_width):
            norm_x = px / max(1, pixel_width - 1)
            time_idx = int(norm_x * (num_time - 1))
            time_idx = max(0, min(num_time - 1, time_idx))

            intensity = matrix[time_idx][freq_idx]
            pixel_energy[py][px] = intensity
            if intensity >= threshold:
                pixel_grid[py][px] = 1

    lines: List[str] = []
    if border:
        lines.append("+" + "-" * width_chars + "+")

    for cy in range(height_chars):
        row_chars: List[str] = []
        for cx in range(width_chars):
            code = 0
            cell_energy = 0.0
            for by in range(4):
                py = cy * 4 + by
                for bx in range(2):
                    px = cx * 2 + bx
                    if pixel_grid[py][px]:
                        code |= BRAILLE_DOTS[by][bx]
                    cell_energy = max(cell_energy, pixel_energy[py][px])

            char = chr(0x2800 + code)
            if use_color:
                r, g, b = _interpolate_color(cell_energy)
                row_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
            else:
                row_chars.append(char)

        content = "".join(row_chars)
        lines.append(f"|{content}|" if border else content)

    if border:
        lines.append("+" + "-" * width_chars + "+")

    return "\n".join(lines)


def render_spectrum_bars(
    spectrum_db: List[float],
    freqs: List[float],
    num_bars: int = 40,
    bar_height: int = 8,
    min_db: float = -80.0,
    max_db: float = 0.0
) -> str:
    """
    Renders an instantaneous frequency spectrum bar chart into ASCII/Unicode blocks.
    """
    if not spectrum_db or len(spectrum_db) < num_bars:
        return ""

    db_range = max(1.0, max_db - min_db)
    # Bin down to num_bars
    step = len(spectrum_db) / num_bars
    bar_heights: List[float] = []

    for b in range(num_bars):
        start_idx = int(b * step)
        end_idx = max(start_idx + 1, int((b + 1) * step))
        chunk = spectrum_db[start_idx:end_idx]
        avg_db = sum(chunk) / len(chunk) if chunk else min_db
        norm = (avg_db - min_db) / db_range
        bar_heights.append(max(0.0, min(1.0, norm)))

    lines: List[str] = []
    num_levels = len(BLOCK_CHARS) - 1

    for row in range(bar_height - 1, -1, -1):
        row_min = row / bar_height
        row_max = (row + 1) / bar_height
        row_chars: List[str] = []

        for h in bar_heights:
            if h >= row_max:
                row_chars.append(BLOCK_CHARS[-1])
            elif h <= row_min:
                row_chars.append(" ")
            else:
                frac = (h - row_min) / (row_max - row_min)
                idx = int(frac * num_levels)
                row_chars.append(BLOCK_CHARS[idx])

        lines.append("".join(row_chars))

    lines.append("-" * num_bars)
    return "\n".join(lines)
