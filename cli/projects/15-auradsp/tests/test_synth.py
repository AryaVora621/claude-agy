"""
Unit tests for AuraDSP Synthesizer, Oscillators, ADSR Envelopes, and Polyphony.
"""

import math
import unittest
from auradsp.synth import Oscillator, ADSREnvelope, LFO, PolyphonicSynth, midi_to_freq


class TestSynth(unittest.TestCase):
    def test_midi_to_freq(self):
        # A4 = MIDI note 69 = 440.0 Hz
        self.assertAlmostEqual(midi_to_freq(69), 440.0, places=4)
        # A5 = MIDI note 81 = 880.0 Hz
        self.assertAlmostEqual(midi_to_freq(81), 880.0, places=4)
        # Middle C = C4 = MIDI note 60 = ~261.63 Hz
        self.assertAlmostEqual(midi_to_freq(60), 261.6256, places=2)

    def test_oscillator_waveforms(self):
        sr = 44100.0
        n_samples = 1000

        for wf in ["sine", "saw", "square", "triangle", "noise"]:
            osc = Oscillator(waveform=wf, frequency=440.0, sample_rate=sr)
            samples = osc.generate(n_samples)
            self.assertEqual(len(samples), n_samples)
            # All samples must stay strictly within [-1.0, 1.0]
            for s in samples:
                self.assertGreaterEqual(s, -1.0)
                self.assertLessEqual(s, 1.0)

    def test_adsr_envelope(self):
        sr = 1000.0  # 1 kHz sample rate for clear step verification
        env = ADSREnvelope(
            attack_time=0.1,    # 100 samples
            decay_time=0.1,     # 100 samples
            sustain_level=0.5,  # level 0.5
            release_time=0.2,   # 200 samples
            sample_rate=sr
        )

        total_duration = 1.0  # 1000 samples
        curve = env.generate_curve(total_duration, gate_duration=0.8)

        self.assertEqual(len(curve), 1000)
        # Starts near 0
        self.assertAlmostEqual(curve[0], 0.0, places=2)
        # Peaks at attack end (sample 100) near 1.0
        self.assertAlmostEqual(curve[100], 1.0, delta=0.05)
        # Reaches sustain level around sample 300
        self.assertAlmostEqual(curve[300], 0.5, delta=0.05)
        # Ends at 0 after release
        self.assertAlmostEqual(curve[-1], 0.0, delta=0.05)

    def test_polyphonic_chord_and_limiter(self):
        synth = PolyphonicSynth(sample_rate=44100.0)
        # C Major triad: C4 (261.63), E4 (329.63), G4 (392.00)
        freqs = [261.63, 329.63, 392.00]
        chord = synth.synthesize_chord(freqs, duration=0.1, waveform="saw")

        self.assertEqual(len(chord), 4410)
        # Soft clipping limiter prevents overflow past 1.0
        for s in chord:
            self.assertGreater(s, -1.0)
            self.assertLess(s, 1.0)


if __name__ == "__main__":
    unittest.main()
