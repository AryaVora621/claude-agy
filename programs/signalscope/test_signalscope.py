"""Comprehensive test suite for SignalScope DSP engine and desktop GUI."""

import unittest
import math
import cmath
import os
import sys
import wave
import tempfile
from typing import List

# Ensure local module directory is in pythonpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .dsp import (
        fft,
        next_power_of_two,
        apply_window,
        ChannelParams,
        BiquadFilter,
        SignalEngine
    )
except ImportError:
    from dsp import (
        fft,
        next_power_of_two,
        apply_window,
        ChannelParams,
        BiquadFilter,
        SignalEngine
    )


class TestCooleyTukeyFFT(unittest.TestCase):
    """Verify in-place Radix-2 Cooley-Tukey FFT mathematical accuracy."""

    def test_next_power_of_two(self):
        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(2), 2)
        self.assertEqual(next_power_of_two(3), 4)
        self.assertEqual(next_power_of_two(7), 8)
        self.assertEqual(next_power_of_two(511), 512)
        self.assertEqual(next_power_of_two(1024), 1024)

    def test_impulse_response_flat_spectrum(self):
        """FFT of a unit impulse delta[0] = 1 must yield constant magnitude 1 across all bins."""
        impulse = [1.0 + 0j] + [0.0 + 0j] * 15
        spectrum = fft(impulse)
        self.assertEqual(len(spectrum), 16)
        for val in spectrum:
            self.assertAlmostEqual(abs(val), 1.0, places=5)

    def test_pure_tone_bin_identification(self):
        """A pure sinusoidal tone at bin k must concentrate maximum energy at bin k and N-k."""
        n = 64
        k = 8  # 8 complete cycles
        sine_wave = [cmath.exp(2j * math.pi * k * i / n) for i in range(n)]
        spectrum = fft(sine_wave)

        # In complex exponential, bin k should be N (64), others near zero
        for idx, val in enumerate(spectrum):
            if idx == k:
                self.assertAlmostEqual(abs(val), 64.0, places=4)
            else:
                self.assertLess(abs(val), 1e-4)

    def test_parsevals_theorem(self):
        """Energy in time domain must equal energy in frequency domain (divided by N)."""
        signal = [math.sin(2.0 * math.pi * 3 * i / 32) + 0.5 * math.cos(2.0 * math.pi * 7 * i / 32) for i in range(32)]
        time_energy = sum(s * s for s in signal)

        spectrum = fft([complex(s, 0.0) for s in signal])
        freq_energy = sum(abs(c) ** 2 for c in spectrum) / len(spectrum)

        self.assertAlmostEqual(time_energy, freq_energy, places=4)


class TestSpectralWindows(unittest.TestCase):
    """Verify windowing attenuation at sample endpoints."""

    def test_hanning_window_boundary_taper(self):
        samples = [1.0] * 64
        windowed = apply_window(samples, "Hanning")
        self.assertAlmostEqual(windowed[0], 0.0, places=5)
        self.assertAlmostEqual(windowed[-1], 0.0, places=5)
        self.assertAlmostEqual(windowed[31], 1.0, delta=0.05)

    def test_blackman_window_boundary_taper(self):
        samples = [1.0] * 64
        windowed = apply_window(samples, "Blackman")
        self.assertAlmostEqual(windowed[0], 0.0, places=5)
        self.assertAlmostEqual(windowed[-1], 0.0, places=5)


class TestBiquadFilter(unittest.TestCase):
    """Verify 2nd-order IIR biquad filtering characteristics."""

    def test_lowpass_attenuation(self):
        """Lowpass filter should preserve low frequencies and attenuate high frequencies."""
        sr = 44100.0
        flt = BiquadFilter(filter_type="Lowpass", cutoff=500.0, q_factor=0.707, sample_rate=sr)

        # Low frequency test: 100 Hz
        low_f = 100.0
        low_samples = [math.sin(2.0 * math.pi * low_f * i / sr) for i in range(1000)]
        flt.reset_state()
        low_out = [flt.process_sample(s) for s in low_samples][500:]
        amp_low = max(low_out) - min(low_out)

        # High frequency test: 5000 Hz
        high_f = 5000.0
        high_samples = [math.sin(2.0 * math.pi * high_f * i / sr) for i in range(1000)]
        flt.reset_state()
        high_out = [flt.process_sample(s) for s in high_samples][500:]
        amp_high = max(high_out) - min(high_out)

        # High frequency should be significantly attenuated compared to low frequency
        self.assertGreater(amp_low, 1.6)  # Near 2.0 Vpp
        self.assertLess(amp_high, 0.25)   # Substantially attenuated (>18 dB)

    def test_highpass_attenuation(self):
        """Highpass filter should attenuate low frequencies and pass high frequencies."""
        sr = 44100.0
        flt = BiquadFilter(filter_type="Highpass", cutoff=3000.0, q_factor=0.707, sample_rate=sr)

        # Low frequency test: 100 Hz
        low_f = 100.0
        low_samples = [math.sin(2.0 * math.pi * low_f * i / sr) for i in range(1000)]
        flt.reset_state()
        low_out = [flt.process_sample(s) for s in low_samples][500:]
        amp_low = max(low_out) - min(low_out)

        # High frequency test: 6000 Hz
        high_f = 6000.0
        high_samples = [math.sin(2.0 * math.pi * high_f * i / sr) for i in range(1000)]
        flt.reset_state()
        high_out = [flt.process_sample(s) for s in high_samples][500:]
        amp_high = max(high_out) - min(high_out)

        self.assertLess(amp_low, 0.15)
        self.assertGreater(amp_high, 1.6)


class TestSignalEngine(unittest.TestCase):
    """Verify waveform synthesis, triggering, metrics, and audio export."""

    def setUp(self):
        self.engine = SignalEngine(sample_rate=44100)

    def test_sine_metrics_ideal_vrms(self):
        """For a pure sine wave with amplitude A=1.0, Vpp=2.0 and Vrms=A/sqrt(2) ~ 0.7071."""
        self.engine.ch1.waveform = "Sine"
        self.engine.ch1.frequency = 440.0
        self.engine.ch1.amplitude = 1.0

        # Generate 1 second of audio
        buf1, _, _ = self.engine.generate_buffers(44100)
        metrics = self.engine.analyze_signal_metrics(buf1)

        self.assertAlmostEqual(metrics["vpp"], 2.0, places=2)
        self.assertAlmostEqual(metrics["vrms"], 1.0 / math.sqrt(2.0), places=2)
        self.assertAlmostEqual(metrics["freq"], 440.0, delta=2.0)

    def test_square_wave_amplitudes(self):
        """Square wave must strictly transition between +amplitude and -amplitude."""
        self.engine.ch1.waveform = "Square"
        self.engine.ch1.amplitude = 0.8
        self.engine.ch1.frequency = 100.0

        buf1, _, _ = self.engine.generate_buffers(1000)
        for val in buf1:
            self.assertTrue(abs(val - 0.8) < 1e-5 or abs(val - (-0.8)) < 1e-5)

    def test_trigger_edge_detection(self):
        """Rising trigger must identify zero-crossing index with positive slope."""
        signal = [-0.5, -0.2, 0.1, 0.6, 0.9, 0.4, -0.1, -0.4, 0.2, 0.7]
        idx = self.engine.find_trigger_index(signal, level=0.0, slope="Rising")
        self.assertEqual(idx, 2)  # -0.2 to +0.1 crossing

        idx_fall = self.engine.find_trigger_index(signal, level=0.0, slope="Falling")
        self.assertEqual(idx_fall, 6)  # +0.4 to -0.1 crossing

    def test_spectrum_peak_frequency(self):
        """FFT spectrum of a 1000 Hz sine tone should locate peak within 1 bin resolution."""
        self.engine.ch1.waveform = "Sine"
        self.engine.ch1.frequency = 1000.0
        self.engine.ch1.amplitude = 1.0
        self.engine.ch2.amplitude = 0.0

        samples = 1024
        buf1, _, mix = self.engine.generate_buffers(samples)
        freqs, mags_db = self.engine.compute_spectrum(buf1, window_type="Hanning")

        max_idx = mags_db.index(max(mags_db))
        peak_freq = freqs[max_idx]

        # Resolution df = 44100 / 1024 ~ 43 Hz
        self.assertAlmostEqual(peak_freq, 1000.0, delta=45.0)

    def test_wav_audio_export(self):
        """Verify 16-bit PCM mono WAV file generation with proper headers."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name

        try:
            self.engine.export_wav(wav_path, duration_sec=0.1)
            self.assertTrue(os.path.exists(wav_path))

            # Inspect WAV headers
            with wave.open(wav_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)  # 16-bit
                self.assertEqual(wf.getframerate(), 44100)
                self.assertEqual(wf.getnframes(), 4410)  # 0.1s * 44100
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


class TestSignalScopeGUI(unittest.TestCase):
    """Verify GUI desktop window instantiation, theme styling, and state changes."""

    def test_gui_modes_and_controls(self):
        try:
            import tkinter as tk
            from .signalscope import SignalScopeApp
        except ImportError:
            self.skipTest("Tkinter not available")

        root = tk.Tk()
        root.withdraw()
        try:
            app = SignalScopeApp(root)
            self.assertEqual(app.display_mode, "Oscilloscope")

            app.set_display_mode("Spectrum")
            self.assertEqual(app.display_mode, "Spectrum")

            app.set_display_mode("Lissajous")
            self.assertEqual(app.display_mode, "Lissajous")

            app.toggle_run()
            self.assertFalse(app.running)
            app.toggle_run()
            self.assertTrue(app.running)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
