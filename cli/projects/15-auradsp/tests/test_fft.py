"""
Unit tests for AuraDSP FFT, IFFT, DFT, and Spectral Analysis.
"""

import math
import random
import unittest
from auradsp.fft import (
    fft, ifft, dft, idft,
    hann_window, hamming_window, blackman_window, apply_window,
    magnitude_spectrum, power_spectrum_db, fft_frequencies,
    is_power_of_two, next_power_of_two
)


class TestFFT(unittest.TestCase):
    def test_power_of_two_helpers(self):
        self.assertTrue(is_power_of_two(1))
        self.assertTrue(is_power_of_two(2))
        self.assertTrue(is_power_of_two(1024))
        self.assertFalse(is_power_of_two(0))
        self.assertFalse(is_power_of_two(3))
        self.assertFalse(is_power_of_two(1000))

        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(5), 8)
        self.assertEqual(next_power_of_two(500), 512)
        self.assertEqual(next_power_of_two(1024), 1024)

    def test_fft_matches_dft(self):
        # Compare FFT against reference O(N^2) DFT for sizes 8, 16, 64
        random.seed(42)
        for n in [8, 16, 64]:
            signal = [random.uniform(-1.0, 1.0) for _ in range(n)]
            fft_res = fft(signal)
            dft_res = dft(signal)

            self.assertEqual(len(fft_res), n)
            for k in range(n):
                diff = abs(fft_res[k] - dft_res[k])
                self.assertLess(diff, 1e-10, f"Mismatch at N={n}, bin {k}: {diff}")

    def test_ifft_roundtrip(self):
        random.seed(1337)
        for n in [16, 128, 256]:
            original = [random.uniform(-2.0, 2.0) + 1j * random.uniform(-2.0, 2.0) for _ in range(n)]
            transformed = fft(original)
            reconstructed = ifft(transformed)

            for i in range(n):
                diff = abs(original[i] - reconstructed[i])
                self.assertLess(diff, 1e-10, f"Roundtrip error at idx {i}: {diff}")

    def test_parseval_theorem(self):
        # Sum of squared time amplitudes must equal (1/N) * sum of squared frequency magnitudes
        random.seed(99)
        n = 128
        signal = [random.uniform(-1.0, 1.0) for _ in range(n)]

        time_energy = sum(abs(x) ** 2 for x in signal)
        freq_spectrum = fft(signal)
        freq_energy = sum(abs(X) ** 2 for X in freq_spectrum) / n

        self.assertAlmostEqual(time_energy, freq_energy, places=8)

    def test_impulse_and_dc_response(self):
        n = 64
        # Unit impulse at n=0
        impulse = [1.0] + [0.0] * (n - 1)
        spec = fft(impulse)
        # All frequency bins should be exactly 1.0
        for val in spec:
            self.assertAlmostEqual(abs(val), 1.0, places=9)

        # Constant DC signal
        dc = [3.5] * n
        spec_dc = fft(dc)
        # Bin 0 should be 3.5 * n, all others 0
        self.assertAlmostEqual(abs(spec_dc[0]), 3.5 * n, places=9)
        for val in spec_dc[1:]:
            self.assertAlmostEqual(abs(val), 0.0, places=9)

    def test_frequency_bin_peak_detection(self):
        sample_rate = 8000.0
        n = 512
        target_freq = 1000.0  # 1 kHz

        # Generate sine wave: sin(2 * pi * f * t)
        signal = [math.sin(2.0 * math.pi * target_freq * i / sample_rate) for i in range(n)]
        spectrum = fft(signal)
        mags = magnitude_spectrum(spectrum)
        freqs = fft_frequencies(n, sample_rate)

        # Find peak bin index
        peak_idx = max(range(len(mags)), key=lambda i: mags[i])
        detected_freq = freqs[peak_idx]

        # Should match within bin width resolution
        bin_width = sample_rate / n  # 15.625 Hz
        self.assertLessEqual(abs(detected_freq - target_freq), bin_width)

    def test_window_functions(self):
        n = 64
        h = hann_window(n)
        hm = hamming_window(n)
        bk = blackman_window(n)

        self.assertEqual(len(h), n)
        self.assertEqual(len(hm), n)
        self.assertEqual(len(bk), n)

        # Symmetry test: w[i] == w[n - 1 - i]
        for i in range(n // 2):
            self.assertAlmostEqual(h[i], h[n - 1 - i], places=9)
            self.assertAlmostEqual(hm[i], hm[n - 1 - i], places=9)
            self.assertAlmostEqual(bk[i], bk[n - 1 - i], places=9)

        # Hann endpoints should be 0.0
        self.assertAlmostEqual(h[0], 0.0, places=9)
        self.assertAlmostEqual(h[-1], 0.0, places=9)

    def test_zero_padding(self):
        # 10 samples padded to 16
        signal = [1.0] * 10
        with self.assertRaises(ValueError):
            fft(signal, pad_to_pow2=False)

        res = fft(signal, pad_to_pow2=True)
        self.assertEqual(len(res), 16)


if __name__ == "__main__":
    unittest.main()
