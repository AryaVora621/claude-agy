"""
Unit tests for AuraDSP Digital Biquad IIR and FIR Filter Design.
"""

import math
import unittest
from auradsp.filters import BiquadFilter, CascadeFilter, FIRFilter


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 44100.0

    def test_low_pass_butterworth_response(self):
        cutoff = 1000.0
        lpf = BiquadFilter.low_pass(self.sample_rate, cutoff)

        self.assertTrue(lpf.is_stable())

        # Check frequency response
        test_freqs = [10.0, cutoff, 5000.0, 15000.0]
        mags_db, _ = lpf.frequency_response(test_freqs)

        # Passband: near DC should be ~0 dB
        self.assertAlmostEqual(mags_db[0], 0.0, delta=0.1)

        # Cutoff frequency: Butterworth 3dB down point (-3.01 dB)
        self.assertAlmostEqual(mags_db[1], -3.01, delta=0.2)

        # Stopband: 5 kHz and 15 kHz must have strong attenuation
        self.assertLess(mags_db[2], -20.0)
        self.assertLess(mags_db[3], -40.0)

    def test_high_pass_butterworth_response(self):
        cutoff = 1000.0
        hpf = BiquadFilter.high_pass(self.sample_rate, cutoff)

        self.assertTrue(hpf.is_stable())

        test_freqs = [10.0, cutoff, 10000.0]
        mags_db, _ = hpf.frequency_response(test_freqs)

        # Stopband: near DC heavily attenuated
        self.assertLess(mags_db[0], -40.0)

        # Cutoff: ~ -3 dB
        self.assertAlmostEqual(mags_db[1], -3.01, delta=0.2)

        # Passband: high frequencies ~ 0 dB
        self.assertAlmostEqual(mags_db[2], 0.0, delta=0.2)

    def test_band_pass_response(self):
        center = 2000.0
        bpf = BiquadFilter.band_pass(self.sample_rate, center, q=2.0)

        self.assertTrue(bpf.is_stable())

        test_freqs = [100.0, center, 10000.0]
        mags_db, _ = bpf.frequency_response(test_freqs)

        # At center frequency, gain should be exactly 0 dB
        self.assertAlmostEqual(mags_db[1], 0.0, delta=0.1)

        # Away from center, should be attenuated
        self.assertLess(mags_db[0], -20.0)
        self.assertLess(mags_db[2], -15.0)

    def test_notch_filter_response(self):
        notch_freq = 1000.0
        notch = BiquadFilter.notch(self.sample_rate, notch_freq, q=10.0)

        self.assertTrue(notch.is_stable())

        test_freqs = [50.0, notch_freq, 10000.0]
        mags_db, _ = notch.frequency_response(test_freqs)

        # Passband at 50 Hz and 10 kHz should be ~0 dB
        self.assertAlmostEqual(mags_db[0], 0.0, delta=0.1)
        self.assertAlmostEqual(mags_db[2], 0.0, delta=0.1)

        # At notch frequency, should have deep rejection (< -40 dB)
        self.assertLess(mags_db[1], -40.0)

    def test_peaking_eq(self):
        center = 1500.0
        gain_db = 6.0
        eq = BiquadFilter.peaking_eq(self.sample_rate, center, gain_db=gain_db, q=1.5)

        self.assertTrue(eq.is_stable())

        test_freqs = [50.0, center, 15000.0]
        mags_db, _ = eq.frequency_response(test_freqs)

        # At center, gain should match target gain_db
        self.assertAlmostEqual(mags_db[1], gain_db, delta=0.1)

        # Baseline should be 0 dB away from center
        self.assertAlmostEqual(mags_db[0], 0.0, delta=0.5)
        self.assertAlmostEqual(mags_db[2], 0.0, delta=0.5)

    def test_cascade_4th_order_butterworth(self):
        cutoff = 1000.0
        cascade = CascadeFilter.butterworth_4th_order_low_pass(self.sample_rate, cutoff)

        # Test filtering an impulse
        impulse = [1.0] + [0.0] * 127
        output = cascade.process(impulse)
        self.assertEqual(len(output), 128)

        # Output must be finite and decay towards 0
        self.assertTrue(all(math.isfinite(x) for x in output))
        self.assertLess(abs(output[-1]), 0.01)

    def test_fir_windowed_sinc(self):
        cutoff = 2000.0
        fir = FIRFilter.windowed_sinc_low_pass(self.sample_rate, cutoff, num_taps=51)

        # DC response test (constant 1.0 signal)
        dc_signal = [1.0] * 100
        output = fir.process(dc_signal)
        # After filter delay settles (around tap 25), output should be ~1.0
        self.assertAlmostEqual(output[50], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
