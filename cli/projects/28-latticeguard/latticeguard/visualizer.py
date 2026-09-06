"""Sub-Pixel Braille Lattice Visualizer, NTT Butterfly Network, and PQC HUD.

Features:
1. BrailleLatticeCanvas: 2x4 sub-pixel Unicode Braille (U+2800..U+28FF) canvas
   mapping 256 polynomial coefficients onto exactly 128 character columns (256 sub-pixels)
   and configurable vertical amplitude resolution.
2. NTTButterflyVisualizer: Diagrammatic flow of the 7-stage Cooley-Tukey NTT
   factorization showing modular twiddle factors (powers of zeta).
3. PQCWorkbenchHUD: Rich ANSI terminal dashboard displaying security level,
   key/ciphertext bandwidths, noise margins, and decapsulation state.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple

from .ring import KYBER_N, KYBER_Q, Polynomial, PolyVec
from .ind_cpa import MLKEMParams


# Sub-pixel Braille dot offsets: (dx, dy) where dx in {0, 1}, dy in {0, 1, 2, 3}
BRAILLE_DOT_MAP = {
    (0, 0): 0x01,
    (0, 1): 0x02,
    (0, 2): 0x04,
    (1, 0): 0x08,
    (1, 1): 0x10,
    (1, 2): 0x20,
    (0, 3): 0x40,
    (1, 3): 0x80,
}


def rgb_ansi(r: int, g: int, b: int, text: str) -> str:
    """Format string with 24-bit TrueColor ANSI foreground."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def color_gradient(val: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
    """Calculate RGB gradient from cyan (low) to purple/violet (mid) to gold/amber (high)."""
    if max_val <= min_val:
        return (60, 200, 240)
    ratio = max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))
    if ratio < 0.5:
        # Cyan to Magenta
        t = ratio * 2.0
        r = int(40 + t * 180)
        g = int(220 - t * 160)
        b = int(250)
    else:
        # Magenta to Gold
        t = (ratio - 0.5) * 2.0
        r = int(220 + t * 35)
        g = int(60 + t * 150)
        b = int(250 - t * 210)
    return (r, g, b)


class BrailleLatticeCanvas:
    """Sub-pixel 2x4 Unicode Braille graph renderer for 256-degree polynomials."""

    def __init__(self, char_width: int = 128, char_height: int = 14) -> None:
        self.char_width = char_width
        self.char_height = char_height
        self.pixel_width = char_width * 2    # 256 sub-pixels (exactly 1 per coefficient!)
        self.pixel_height = char_height * 4  # 4 sub-pixels per Braille character row
        self.grid: List[List[int]] = [[0] * self.char_width for _ in range(self.char_height)]
        self.colors: List[List[Tuple[int, int, int]]] = [
            [(160, 160, 180)] * self.char_width for _ in range(self.char_height)
        ]

    def clear(self) -> None:
        """Reset all Braille grid cells to zero."""
        for r in range(self.char_height):
            for c in range(self.char_width):
                self.grid[r][c] = 0
                self.colors[r][c] = (160, 160, 180)

    def set_pixel(self, px: int, py: int, rgb: Tuple[int, int, int] | None = None) -> None:
        """Set a single sub-pixel in the canvas."""
        if 0 <= px < self.pixel_width and 0 <= py < self.pixel_height:
            cx = px // 2
            cy = py // 4
            dx = px % 2
            dy = py % 4
            dot = BRAILLE_DOT_MAP.get((dx, dy), 0)
            self.grid[cy][cx] |= dot
            if rgb:
                self.colors[cy][cx] = rgb

    def plot_polynomial(
        self,
        poly: Polynomial,
        centered: bool = True,
        line: bool = True,
        rgb: Tuple[int, int, int] | None = None,
    ) -> None:
        """Plot a degree-255 polynomial across the 256 sub-pixel width.

        Args:
            poly: Polynomial in R_q.
            centered: If True, plot centered values in [-q/2, q/2]. If False, plot [0, q-1].
            line: If True, connect consecutive coefficients with a vertical line.
            rgb: Custom RGB color override.
        """
        coeffs = poly.to_centered_list() if centered else list(poly.coeffs)
        min_v = -KYBER_Q // 2 if centered else 0
        max_v = KYBER_Q // 2 if centered else KYBER_Q - 1

        prev_py = None
        for i in range(min(KYBER_N, self.pixel_width)):
            val = coeffs[i]
            ratio = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.5
            ratio = max(0.0, min(1.0, ratio))
            # Invert py so higher values appear near the top
            py = int((1.0 - ratio) * (self.pixel_height - 1))

            c_rgb = rgb if rgb else color_gradient(val, min_v, max_v)

            if line and prev_py is not None:
                step = 1 if py >= prev_py else -1
                for y in range(prev_py, py + step, step):
                    self.set_pixel(i, y, c_rgb)
            else:
                self.set_pixel(i, py, c_rgb)

            prev_py = py

    def render(self, show_border: bool = True) -> str:
        """Render the canvas to ANSI terminal text."""
        lines: List[str] = []
        border_horiz = "+" + "-" * self.char_width + "+"

        if show_border:
            lines.append(rgb_ansi(100, 110, 130, border_horiz))

        for cy in range(self.char_height):
            row_str = []
            for cx in range(self.char_width):
                mask = self.grid[cy][cx]
                char = chr(0x2800 + mask) if mask != 0 else " "
                r, g, b = self.colors[cy][cx]
                row_str.append(rgb_ansi(r, g, b, char))
            content = "".join(row_str)
            if show_border:
                lines.append(f"{rgb_ansi(100, 110, 130, '|')} {content} {rgb_ansi(100, 110, 130, '|')}")
            else:
                lines.append(content)

        if show_border:
            lines.append(rgb_ansi(100, 110, 130, border_horiz))

        return "\n".join(lines)


class NTTButterflyVisualizer:
    """ASCII diagrammatic visualizer for the 7-stage Cooley-Tukey NTT butterfly."""

    @staticmethod
    def render_butterfly_summary() -> str:
        """Generate a compact, clear diagram of the 7-stage NTT factorization."""
        stages = [
            ("Stage 1", "len=128", "1 butterfly pair of 128 coeffs", "zeta^1"),
            ("Stage 2", "len=64",  "2 butterfly pairs of 64 coeffs",  "zeta^2, zeta^3"),
            ("Stage 3", "len=32",  "4 butterfly pairs of 32 coeffs",  "zeta^4 .. zeta^7"),
            ("Stage 4", "len=16",  "8 butterfly pairs of 16 coeffs",  "zeta^8 .. zeta^15"),
            ("Stage 5", "len=8",   "16 butterfly pairs of 8 coeffs",  "zeta^16 .. zeta^31"),
            ("Stage 6", "len=4",   "32 butterfly pairs of 4 coeffs",  "zeta^32 .. zeta^63"),
            ("Stage 7", "len=2",   "64 butterfly pairs of 2 coeffs",  "zeta^64 .. zeta^127"),
        ]

        out = []
        out.append(rgb_ansi(240, 200, 60, "=== Cooley-Tukey NTT 7-Stage Decimation Network (n=256, q=3329) ==="))
        out.append(rgb_ansi(140, 160, 190, "Polynomial quotient ring: R_q = Z_q[X] / (X^256 + 1) with primitive zeta = 17 mod 3329\n"))

        for name, length, desc, roots in stages:
            stage_box = (
                f"  [{rgb_ansi(60, 210, 240, name)}]  "
                f"Block: {rgb_ansi(200, 140, 255, length):<9} | "
                f"Structure: {desc:<35} | "
                f"Twiddles: {rgb_ansi(255, 180, 80, roots)}"
            )
            out.append(stage_box)

        out.append("\n" + rgb_ansi(120, 220, 140, "BaseCaseMultiply: 128 modular products in Z_q[X]/(X^2 - zeta^(2*bitrev(i)+1))"))
        out.append(rgb_ansi(160, 170, 185, "Complexity: O(n log n) = 1,792 butterfly operations vs 65,536 classical operations (36.6x reduction)"))
        return "\n".join(out)


class PQCWorkbenchHUD:
    """Rich terminal telemetry dashboard and parameter inspector."""

    @staticmethod
    def render_params_table(params: MLKEMParams) -> str:
        """Render formatted ASCII card of cryptographic parameters and key sizes."""
        lines = []
        title = f"LATTICEGUARD PQC TELEMETRY: {params.name} (NIST FIPS 203)"
        border = "+" + "=" * 68 + "+"
        lines.append(rgb_ansi(240, 200, 60, border))
        lines.append(f"| {rgb_ansi(255, 255, 255, title):<76} |")
        lines.append(rgb_ansi(240, 200, 60, border))

        rows = [
            ("Cryptographic Primitive", "Module-LWE Key Encapsulation (ML-KEM)"),
            ("Security Category", f"NIST Level {params.k} (Equivalent to AES-{64 * params.k})"),
            ("Ring Modulus (q)", f"{KYBER_Q} (Prime, q = 1 mod 256)"),
            ("Polynomial Degree (n)", f"{KYBER_N} coefficients"),
            ("Module Rank (k)", f"{params.k} x {params.k} polynomial matrix"),
            ("Secret Noise Parameter (eta1)", f"{params.eta1} (CBD variance = {params.eta1 / 2.0:.1f})"),
            ("Encryption Noise Parameter (eta2)", f"{params.eta2} (CBD variance = {params.eta2 / 2.0:.1f})"),
            ("Vector Compression (du)", f"{params.du} bits / coefficient"),
            ("Scalar Compression (dv)", f"{params.dv} bits / coefficient"),
            ("Public Key Size (ek)", f"{params.pk_bytes_len:,} bytes ({params.pk_bytes_len * 8:,} bits)"),
            ("Secret Key Size (dk)", f"{768 * params.k + 96:,} bytes (CPA + FO implicit rejection)"),
            ("Ciphertext Size (ct)", f"{params.ct_bytes_len:,} bytes"),
            ("Shared Secret Key (K)", "32 bytes (256 bits)"),
        ]

        for label, val in rows:
            line_str = f"|  {rgb_ansi(140, 170, 210, f'{label:<34}')}: {rgb_ansi(230, 230, 240, f'{val:<28}')} |"
            lines.append(line_str)

        lines.append(rgb_ansi(240, 200, 60, border))
        return "\n".join(lines)

    @staticmethod
    def render_noise_telemetry(poly_s: Polynomial, poly_e: Polynomial, poly_t: Polynomial) -> str:
        """Render noise distribution and coefficient norms."""
        lines = []
        lines.append(rgb_ansi(80, 220, 200, "--- Ring Polynomial Noise Telemetry ---"))
        stats = [
            ("Secret Poly (s_0)", poly_s.infinity_norm, poly_s.l1_norm, poly_s.energy),
            ("Error Poly (e_0)", poly_e.infinity_norm, poly_e.l1_norm, poly_e.energy),
            ("Public Key Poly (t_0)", poly_t.infinity_norm, poly_t.l1_norm, poly_t.energy),
        ]
        lines.append(f"  {'Polynomial':<22} | {'||f||_inf':<10} | {'||f||_1':<10} | {'RMS Energy':<12}")
        lines.append("  " + "-" * 60)
        for name, inf_norm, l1_norm, energy in stats:
            lines.append(
                f"  {rgb_ansi(200, 190, 230, f'{name:<22}')} | "
                f"{inf_norm:<10} | "
                f"{l1_norm:<10} | "
                f"{energy:<12.2f}"
            )
        return "\n".join(lines)
