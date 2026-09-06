"""
Unit tests for NucleoCore Terminal Visualizer.
"""

import unittest
from nucleocore.visualizer import (
    colorize_sequence,
    BrailleDotPlotCanvas,
    generate_dot_plot,
    render_coverage_depth_track,
    render_genomics_summary,
)


class TestVisualizer(unittest.TestCase):
    def test_colorize_sequence(self):
        seq = "ACGT-"
        colored = colorize_sequence(seq)
        self.assertIn("A", colored)
        self.assertIn("\033[", colored)  # ANSI escape code present

    def test_braille_canvas_pixels(self):
        canvas = BrailleDotPlotCanvas(char_width=10, char_height=5)
        # Set individual sub-pixel
        canvas.set_pixel(0, 0)
        lines = canvas.render()
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(lines[0]), 10)
        # First char has bit 0x1 set -> chr(0x2800 + 1) = '⠁'
        self.assertEqual(lines[0][0], "⠁")

    def test_generate_dot_plot(self):
        seq1 = "ACGTACGTACGT"
        seq2 = "ACGTACGTACGT"
        plot = generate_dot_plot(seq1, seq2, width_chars=20, height_chars=8)
        self.assertIn("Homology Dot-Plot", plot)
        self.assertIn("┌", plot)
        self.assertIn("└", plot)

    def test_render_coverage_depth_track(self):
        depths = [10, 20, 30, 40, 50, 40, 30, 20, 10]
        track = render_coverage_depth_track(depths, width_chars=30)
        self.assertIn("Genomic Read Depth Coverage Track", track)
        self.assertIn("[", track)
        self.assertIn("]", track)

    def test_render_genomics_summary(self):
        summary = render_genomics_summary("SARS-CoV-2 Spike", "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAG", coverage=[5]*43)
        self.assertIn("SARS-CoV-2 Spike", summary)
        self.assertIn("GC Content", summary)
        self.assertIn("Total Bases", summary)


if __name__ == "__main__":
    unittest.main()
