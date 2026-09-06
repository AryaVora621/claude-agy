"""
NucleoCore: Terminal Alignment, ANSI Colorizer & Sub-Pixel Braille Dot-Plot Visualizer.
Zero external dependencies. Renders high-resolution sequence homology matrices
using Unicode Braille patterns (U+2800..U+28FF) and 24-bit TrueColor ANSI styling.
"""

from typing import List, Tuple, Optional
from nucleocore.types import gc_content


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# 24-Bit ANSI TrueColor Palettes
COLOR_A = "\033[38;2;46;204;113m"   # Emerald Green
COLOR_C = "\033[38;2;52;152;219m"   # Cobalt Blue
COLOR_G = "\033[38;2;243;156;18m"   # Amber Gold
COLOR_T = "\033[38;2;231;76;60m"    # Crimson Red
COLOR_GAP = "\033[38;2;120;120;120m"# Muted Gray
COLOR_BG = "\033[48;2;20;25;35m"    # Dark Slate
COLOR_CYAN = "\033[38;2;0;220;220m"
COLOR_WHITE = "\033[38;2;245;245;245m"

NUCLEOTIDE_COLORS = {
    "A": COLOR_A, "a": COLOR_A,
    "C": COLOR_C, "c": COLOR_C,
    "G": COLOR_G, "g": COLOR_G,
    "T": COLOR_T, "t": COLOR_T,
    "-": COLOR_GAP,
}


def colorize_sequence(seq: str) -> str:
    """Colorizes a DNA sequence with TrueColor ANSI base highlighting."""
    out = []
    for ch in seq:
        color = NUCLEOTIDE_COLORS.get(ch, COLOR_GAP)
        out.append(f"{color}{ch}{RESET}")
    return "".join(out)


class BrailleDotPlotCanvas:
    """
    Sub-pixel 2x4 dot rasterizer using Unicode Braille patterns (U+2800..U+28FF).
    A canvas of width W and height H characters renders 2*W by 4*H dots.
    """
    # Braille dot bitmask offsets:
    # Col 0: (r0: 0x01, r1: 0x02, r2: 0x04, r3: 0x40)
    # Col 1: (r0: 0x08, r1: 0x10, r2: 0x20, r3: 0x80)
    PIXEL_MASKS = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def __init__(self, char_width: int = 60, char_height: int = 20):
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2
        self.pixel_height = char_height * 4
        self.grid = [[0] * char_width for _ in range(char_height)]

    def set_pixel(self, px: int, py: int) -> None:
        """Sets a sub-pixel dot at coordinate (px, py)."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            cx = px // 2
            cy = py // 4
            sub_x = px % 2
            sub_y = py % 4
            self.grid[cy][cx] |= self.PIXEL_MASKS[sub_y][sub_x]

    def render(self) -> List[str]:
        """Renders the grid as a list of strings with Unicode Braille glyphs."""
        lines = []
        for row in self.grid:
            chars = [chr(0x2800 + mask) for mask in row]
            lines.append("".join(chars))
        return lines


def generate_dot_plot(
    seq1: str,
    seq2: str,
    width_chars: int = 60,
    height_chars: int = 20,
    window: int = 3,
    min_match: int = 2
) -> str:
    """
    Generates an ASCII/Braille dot-plot homology matrix comparing seq1 against seq2.
    Diagonals represent homologous matching regions.
    """
    canvas = BrailleDotPlotCanvas(width_chars, height_chars)

    n1 = len(seq1)
    n2 = len(seq2)
    if n1 == 0 or n2 == 0:
        return ""

    pw = canvas.pixel_width
    ph = canvas.pixel_height

    # Rasterize homology matches
    for py in range(ph):
        idx2 = int(py * (n2 - window) / ph) if n2 > window else 0
        kmer2 = seq2[idx2:idx2 + window].upper()
        for px in range(pw):
            idx1 = int(px * (n1 - window) / pw) if n1 > window else 0
            kmer1 = seq1[idx1:idx1 + window].upper()

            # Compare window matches
            matches = sum(1 for a, b in zip(kmer1, kmer2) if a == b)
            if matches >= min_match:
                canvas.set_pixel(px, py)

    braille_lines = canvas.render()

    header = f"{BOLD}{COLOR_CYAN}  SEQ1 ({len(seq1):,} bp) vs SEQ2 ({len(seq2):,} bp) | Homology Dot-Plot{RESET}"
    border_top = "  ┌" + "─" * width_chars + "┐"
    border_bot = "  └" + "─" * width_chars + "┘"

    out = [header, border_top]
    for idx, b_line in enumerate(braille_lines):
        y_label = f"{(idx * 4 * n2 // ph):4d}│" if idx % 5 == 0 else "    │"
        out.append(f"{DIM}{y_label}{RESET}{COLOR_WHITE}{b_line}{RESET}│")
    out.append(border_bot)
    out.append(f"    0{' ' * (width_chars - 10)}{len(seq1):,} bp ->")

    return "\n".join(out)


def render_coverage_depth_track(coverage_depths: List[int], width_chars: int = 60) -> str:
    """
    Renders a genomic read coverage depth track using Unicode block characters.
    """
    if not coverage_depths:
        return "No coverage data"

    blocks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    total_len = len(coverage_depths)
    max_cov = max(coverage_depths) if coverage_depths else 1
    if max_cov == 0:
        max_cov = 1

    # Bin into width_chars columns
    step = total_len / width_chars
    binned_cov = []
    for i in range(width_chars):
        start = int(i * step)
        end = max(start + 1, int((i + 1) * step))
        mean_c = sum(coverage_depths[start:end]) / (end - start)
        binned_cov.append(mean_c)

    track_chars = []
    for c in binned_cov:
        normalized = min(1.0, max(0.0, c / max_cov))
        b_idx = int(round(normalized * 8))
        track_chars.append(blocks[b_idx])

    mean_cov = sum(coverage_depths) / total_len
    header = f"{BOLD}Genomic Read Depth Coverage Track{RESET} (Mean: {mean_cov:.1f}x | Max: {max_cov}x)"
    chart_line = "".join(track_chars)

    return f"{header}\n[{COLOR_CYAN}{chart_line}{RESET}]"


def render_genomics_summary(
    genome_name: str,
    sequence: str,
    coverage: Optional[List[int]] = None
) -> str:
    """
    Produces a terminal summary card for a genomic sequence with nucleotide stats,
    GC content, and optional coverage visualization.
    """
    seq_len = len(sequence)
    gc = gc_content(sequence)
    count_a = sequence.upper().count("A")
    count_c = sequence.upper().count("C")
    count_g = sequence.upper().count("G")
    count_t = sequence.upper().count("T")

    lines = []
    lines.append(f"╔{'═' * 66}╗")
    lines.append(f"║ {BOLD}{genome_name:<64}{RESET} ║")
    lines.append(f"╠{'═' * 66}╣")
    lines.append(f"║ Total Bases: {seq_len:<12,d} | GC Content: {gc*100:<6.2f}%{' ' * 23}║")
    lines.append(f"║ A: {COLOR_A}{count_a:<8,d}{RESET} | C: {COLOR_C}{count_c:<8,d}{RESET} | G: {COLOR_G}{count_g:<8,d}{RESET} | T: {COLOR_T}{count_t:<8,d}{RESET} ║")
    lines.append(f"╚{'═' * 66}╝")

    if coverage:
        lines.append("")
        lines.append(render_coverage_depth_track(coverage, width_chars=66))

    return "\n".join(lines)
