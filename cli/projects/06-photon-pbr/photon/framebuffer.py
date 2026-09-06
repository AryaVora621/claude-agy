"""
PhotonPBR: Terminal Framebuffer and Image Exporters.
Provides ANSI 24-bit TrueColor terminal rendering with half-block character packing,
high-contrast ASCII luminance art, and Netpbm PPM image output.
"""

from typing import List, Tuple
from photon.vec3 import Color


class Framebuffer:
    """Stores a 2D matrix of HDR pixel colors and exports to multiple representations."""

    ASCII_RAMP = " .:-=+*#%@"

    def __init__(self, width: int, height: int, pixels: List[List[Color]]):
        self.width = width
        self.height = height
        self.pixels = pixels

    def to_ascii(self, samples_per_pixel: int = 1) -> str:
        """Render high-contrast ASCII terminal representation using luminance ramp."""
        lines = []
        ramp_len = len(self.ASCII_RAMP)

        for row in self.pixels:
            chars = []
            for col in row:
                r, g, b = col.to_rgb_bytes(samples_per_pixel)
                # Perceptual relative luminance formula (ITU-R BT.709)
                luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                idx = min(ramp_len - 1, int(luminance * ramp_len))
                chars.append(self.ASCII_RAMP[idx])
            lines.append("".join(chars))

        return "\n".join(lines)

    def to_ansi_truecolor(self, samples_per_pixel: int = 1) -> str:
        """
        Render 24-bit TrueColor to terminal using half-block character '▀'.
        Packs two vertical pixels into one terminal row for square aspect ratio.
        """
        lines = []
        # Step by 2 rows at a time
        for j in range(0, self.height, 2):
            line_parts = []
            top_row = self.pixels[j]
            bottom_row = self.pixels[j + 1] if j + 1 < self.height else None

            for i in range(self.width):
                top_col = top_row[i]
                tr, tg, tb = top_col.to_rgb_bytes(samples_per_pixel)

                if bottom_row is not None:
                    bottom_col = bottom_row[i]
                    br, bg, bb = bottom_col.to_rgb_bytes(samples_per_pixel)
                    # Foreground: top pixel, Background: bottom pixel
                    ansi_code = f"\033[38;2;{tr};{tg};{tb}m\033[48;2;{br};{bg};{bb}m▀\033[0m"
                else:
                    ansi_code = f"\033[38;2;{tr};{tg};{tb}m▀\033[0m"

                line_parts.append(ansi_code)
            lines.append("".join(line_parts))

        return "\n".join(lines)

    def save_ppm(self, filepath: str, samples_per_pixel: int = 1) -> None:
        """Save image to Netpbm PPM (P3 text) format."""
        with open(filepath, "w", encoding="ascii") as f:
            f.write(f"P3\n{self.width} {self.height}\n255\n")
            for row in self.pixels:
                for col in row:
                    r, g, b = col.to_rgb_bytes(samples_per_pixel)
                    f.write(f"{r} {g} {b}\n")
