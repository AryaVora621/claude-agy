"""
Unit tests for AuraDSP RIFF WAV Audio Codec.
"""

import math
import os
import tempfile
import unittest
from auradsp.wav import encode_wav_bytes, decode_wav_bytes, write_wav, read_wav


class TestWAV(unittest.TestCase):
    def test_mono_wav_roundtrip(self):
        sample_rate = 44100
        num_samples = 4410  # 0.1s
        # 440 Hz sine wave
        original = [math.sin(2.0 * math.pi * 440.0 * i / sample_rate) for i in range(num_samples)]

        wav_bytes = encode_wav_bytes(original, sample_rate)
        self.assertGreater(len(wav_bytes), 44)

        decoded, sr_out = decode_wav_bytes(wav_bytes)
        self.assertEqual(sr_out, sample_rate)
        self.assertEqual(len(decoded), num_samples)

        # 16-bit quantization introduces max error of 1 / 32767 ~= 3e-5
        for orig, dec in zip(original, decoded):
            self.assertAlmostEqual(orig, dec, delta=1e-4)

    def test_stereo_wav_roundtrip(self):
        sample_rate = 22050
        num_samples = 2205
        left = [0.5 * math.sin(2.0 * math.pi * 300.0 * i / sample_rate) for i in range(num_samples)]
        right = [0.5 * math.sin(2.0 * math.pi * 600.0 * i / sample_rate) for i in range(num_samples)]

        wav_bytes = encode_wav_bytes((left, right), sample_rate)
        decoded_mono, sr_out = decode_wav_bytes(wav_bytes)

        self.assertEqual(sr_out, sample_rate)
        self.assertEqual(len(decoded_mono), num_samples)

    def test_file_io_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_path = tf.name

        try:
            samples = [0.25 * math.sin(0.1 * i) for i in range(1000)]
            write_wav(temp_path, samples, sample_rate=48000)

            rec_samples, sr = read_wav(temp_path)
            self.assertEqual(sr, 48000)
            self.assertEqual(len(rec_samples), 1000)
            for s1, s2 in zip(samples, rec_samples):
                self.assertAlmostEqual(s1, s2, delta=1e-4)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
