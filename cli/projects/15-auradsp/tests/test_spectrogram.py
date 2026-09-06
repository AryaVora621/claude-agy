"""
Unit tests for AuraDSP STFT Spectrogram, Spectral Metrics, and Braille Visualizer.
"""

import math
import random
import unittest
from auradsp.spectrogram import stft, spectrogram_matrix, spectral_centroid, spectral_flatness
from auradsp.visualizer import render_braille_waterfall, render_spectrum_bars
from auradsp.fft import fft, fft_frequencies, magnitude_spectrum, hann_window, apply_window


class TestSpectrogram(unittest.TestCase):
    def test_stft_dimensions(self):
        sample_rate = 44100.0
        frame_size = 256
        hop_size = 128
        n_samples = 2048

        # Dual tone signal
        signal = [
            math.sin(2.0 * math.pi * 440.0 * i / sample_rate) +
            math.sin(2.0 * math.pi * 880.0 * i / sample_rate)
            for i in range(n_samples)
        ]

        frames = stft(signal, frame_size=frame_size, hop_size=hop_size)
        expected_frames = 1 + (n_samples - frame_size) // hop_size
        self.assertEqual(len(frames), expected_frames)

        for frame in frames:
            self.assertEqual(len(frame), frame_size)

    def test_spectrogram_matrix_normalized(self):
        sample_rate = 22050.0
        frame_size = 128
        hop_size = 64
        n_samples = 1024

        signal = [0.8 * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate) for i in range(n_samples)]
        matrix, times, freqs = spectrogram_matrix(
            signal,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            min_db=-60.0
        )

        expected_frames = 1 + (n_samples - frame_size) // hop_size
        expected_bins = frame_size // 2 + 1

        self.assertEqual(len(matrix), expected_frames)
        self.assertEqual(len(times), expected_frames)
        self.assertEqual(len(matrix[0]), expected_bins)
        self.assertEqual(len(freqs), expected_bins)

        # All values in matrix should be normalized between 0.0 and 1.0
        for row in matrix:
            for val in row:
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)

    def test_spectral_centroid(self):
        sample_rate = 44100.0
        n = 1024
        freqs = fft_frequencies(n, sample_rate)
        win = hann_window(n)

        # Tone 1: 500 Hz (windowed to eliminate rectangular spectral leakage)
        tone_500 = [math.sin(2.0 * math.pi * 500.0 * i / sample_rate) for i in range(n)]
        mags_500 = magnitude_spectrum(fft(apply_window(tone_500, win)))
        centroid_500 = spectral_centroid(mags_500, freqs)

        # Tone 2: 2500 Hz
        tone_2500 = [math.sin(2.0 * math.pi * 2500.0 * i / sample_rate) for i in range(n)]
        mags_2500 = magnitude_spectrum(fft(apply_window(tone_2500, win)))
        centroid_2500 = spectral_centroid(mags_2500, freqs)

        # Centroids should closely reflect the fundamental frequencies
        self.assertAlmostEqual(centroid_500, 500.0, delta=20.0)
        self.assertAlmostEqual(centroid_2500, 2500.0, delta=20.0)
        self.assertGreater(centroid_2500, centroid_500)

    def test_spectral_flatness(self):
        sample_rate = 44100.0
        n = 1024
        win = hann_window(n)

        # Tonal signal: pure sine wave (windowed)
        tone = [math.sin(2.0 * math.pi * 440.0 * i / sample_rate) for i in range(n)]
        tone_mags = magnitude_spectrum(fft(apply_window(tone, win)))
        flatness_tone = spectral_flatness(tone_mags)

        # Noise signal: random uniform values
        random.seed(42)
        noise = [random.uniform(-1.0, 1.0) for _ in range(n)]
        noise_mags = magnitude_spectrum(fft(noise))
        flatness_noise = spectral_flatness(noise_mags)

        # Tone should have low spectral flatness (near 0)
        # Noise should have significantly higher spectral flatness
        self.assertLess(flatness_tone, 0.05)
        self.assertGreater(flatness_noise, 0.5)
        self.assertGreater(flatness_noise, flatness_tone * 100.0)

    def test_braille_waterfall_rendering(self):
        # Create a mock 10x16 spectrogram matrix
        matrix = [[(i + j) / 26.0 for j in range(16)] for i in range(10)]

        rendered_colored = render_braille_waterfall(
            matrix, width_chars=30, height_chars=8, threshold=0.3, use_color=True, border=True
        )
        self.assertIn("\033[38;2;", rendered_colored)
        self.assertIn("+", rendered_colored)

        rendered_mono = render_braille_waterfall(
            matrix, width_chars=30, height_chars=8, threshold=0.3, use_color=False, border=False
        )
        self.assertNotIn("\033[38;2;", rendered_mono)
        self.assertNotIn("+", rendered_mono)
        lines = rendered_mono.strip().split("\n")
        self.assertEqual(len(lines), 8)
        self.assertEqual(len(lines[0]), 30)

    def test_spectrum_bars_rendering(self):
        spectrum_db = [-80.0 + (i * 2.0) for i in range(40)]
        freqs = [i * 100.0 for i in range(40)]

        bars = render_spectrum_bars(spectrum_db, freqs, num_bars=20, bar_height=5)
        self.assertTrue(len(bars) > 0)
        lines = bars.split("\n")
        self.assertEqual(len(lines), 6)  # 5 height rows + separator line


if __name__ == "__main__":
    unittest.main()
