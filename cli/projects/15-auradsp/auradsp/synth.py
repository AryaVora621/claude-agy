"""
AuraDSP: Sound Synthesizer & Envelope Generation Engine.
Implements continuous-phase polyphonic oscillators (Sine, Sawtooth, Square/PWM, Triangle, Noise),
4-stage exponential ADSR envelope generators, LFO modulation, and MIDI frequency translation.
"""

import math
import random
from typing import List, Optional, Callable


def midi_to_freq(midi_note: int) -> float:
    """Converts a standard MIDI note number (0-127, where 69 is Concert A 440 Hz) to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


class Oscillator:
    """
    Continuous-phase waveform generator supporting Sine, Sawtooth, Square (with PWM),
    Triangle, and White Noise.
    """

    def __init__(
        self,
        waveform: str = "sine",
        frequency: float = 440.0,
        sample_rate: float = 44100.0,
        duty_cycle: float = 0.5
    ) -> None:
        self.waveform = waveform.lower()
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.duty_cycle = duty_cycle  # For square wave pulse width
        self.phase = 0.0

    def reset(self) -> None:
        """Resets phase accumulator."""
        self.phase = 0.0

    def next_sample(self, freq_offset: float = 0.0, phase_offset: float = 0.0) -> float:
        """
        Advances oscillator phase and returns the next output sample in [-1.0, 1.0].
        Supports frequency modulation (FM) and phase modulation (PM).
        """
        eff_phase = (self.phase + phase_offset) % 1.0
        wf = self.waveform

        if wf == "sine":
            val = math.sin(2.0 * math.pi * eff_phase)
        elif wf == "saw" or wf == "sawtooth":
            # Bipolar sawtooth from -1.0 to 1.0
            val = 2.0 * eff_phase - 1.0
        elif wf == "square":
            val = 1.0 if eff_phase < self.duty_cycle else -1.0
        elif wf == "triangle":
            # Triangle from -1.0 to 1.0
            val = 4.0 * abs(eff_phase - 0.5) - 1.0
        elif wf == "noise":
            val = random.uniform(-1.0, 1.0)
        else:
            val = math.sin(2.0 * math.pi * eff_phase)

        # Advance phase
        eff_freq = max(0.0, self.frequency + freq_offset)
        phase_increment = eff_freq / self.sample_rate
        self.phase = (self.phase + phase_increment) % 1.0
        return val

    def generate(self, num_samples: int) -> List[float]:
        """Generates a buffer of consecutive samples."""
        return [self.next_sample() for _ in range(num_samples)]


class ADSREnvelope:
    """
    4-Stage Exponential ADSR (Attack, Decay, Sustain, Release) Envelope Generator.
    """

    def __init__(
        self,
        attack_time: float = 0.01,   # seconds
        decay_time: float = 0.1,     # seconds
        sustain_level: float = 0.7,  # level in [0.0, 1.0]
        release_time: float = 0.2,   # seconds
        sample_rate: float = 44100.0
    ) -> None:
        self.attack_time = max(1e-4, attack_time)
        self.decay_time = max(1e-4, decay_time)
        self.sustain_level = max(0.0, min(1.0, sustain_level))
        self.release_time = max(1e-4, release_time)
        self.sample_rate = sample_rate

    def generate_curve(self, total_duration: float, gate_duration: Optional[float] = None) -> List[float]:
        """
        Renders the full envelope curve for a note of total_duration seconds.
        If gate_duration is not provided, gate opens for (total_duration - release_time).
        """
        if gate_duration is None:
            gate_duration = max(0.0, total_duration - self.release_time)

        num_samples = int(total_duration * self.sample_rate)
        envelope = [0.0] * num_samples

        dt = 1.0 / self.sample_rate
        current_val = 0.0
        t = 0.0

        for i in range(num_samples):
            if t < gate_duration:
                # Key is held down: Attack -> Decay -> Sustain
                if t < self.attack_time:
                    # Attack phase (smooth rise)
                    progress = t / self.attack_time
                    current_val = progress
                elif t < (self.attack_time + self.decay_time):
                    # Decay phase (exponential approach to sustain level)
                    decay_t = t - self.attack_time
                    progress = decay_t / self.decay_time
                    current_val = 1.0 - (1.0 - self.sustain_level) * (1.0 - math.exp(-3.0 * progress))
                else:
                    # Sustain phase
                    current_val = self.sustain_level
            else:
                # Key released: Release phase
                rel_t = t - gate_duration
                progress = min(1.0, rel_t / self.release_time)
                # Exponential decay toward zero
                current_val = self.sustain_level * math.exp(-4.0 * progress)
                if progress >= 1.0:
                    current_val = 0.0

            envelope[i] = max(0.0, min(1.0, current_val))
            t += dt

        return envelope


class LFO:
    """Low-Frequency Oscillator for vibrato, tremolo, and filter modulation."""

    def __init__(
        self,
        rate_hz: float = 5.0,
        depth: float = 1.0,
        waveform: str = "sine",
        sample_rate: float = 44100.0
    ) -> None:
        self.osc = Oscillator(waveform=waveform, frequency=rate_hz, sample_rate=sample_rate)
        self.depth = depth

    def next_val(self) -> float:
        """Returns next modulation value scaled by depth."""
        return self.osc.next_sample() * self.depth


class PolyphonicSynth:
    """
    Polyphonic Synthesizer Voice Mixer with soft-saturation limiter.
    """

    def __init__(self, sample_rate: float = 44100.0) -> None:
        self.sample_rate = sample_rate

    def synthesize_note(
        self,
        frequency: float,
        duration: float,
        waveform: str = "saw",
        envelope: Optional[ADSREnvelope] = None,
        vibrato_lfo: Optional[LFO] = None,
        amplitude: float = 0.8
    ) -> List[float]:
        """Synthesizes a single musical tone with optional vibrato and envelope."""
        num_samples = int(duration * self.sample_rate)
        osc = Oscillator(waveform=waveform, frequency=frequency, sample_rate=self.sample_rate)

        if envelope is None:
            envelope = ADSREnvelope(attack_time=0.02, decay_time=0.1, sustain_level=0.7, release_time=0.1, sample_rate=self.sample_rate)

        env_curve = envelope.generate_curve(duration)
        samples: List[float] = []

        for i in range(num_samples):
            fm_offset = vibrato_lfo.next_val() if vibrato_lfo else 0.0
            raw_sample = osc.next_sample(freq_offset=fm_offset)
            env_val = env_curve[i] if i < len(env_curve) else 0.0
            samples.append(raw_sample * env_val * amplitude)

        return samples

    def synthesize_chord(
        self,
        frequencies: List[float],
        duration: float,
        waveform: str = "sine",
        envelope: Optional[ADSREnvelope] = None,
        amplitude: float = 0.8
    ) -> List[float]:
        """Synthesizes multiple simultaneous notes mixed and soft-clipped."""
        if not frequencies:
            return [0.0] * int(duration * self.sample_rate)

        num_samples = int(duration * self.sample_rate)
        mixed = [0.0] * num_samples
        voice_gain = amplitude / len(frequencies)

        for freq in frequencies:
            voice_samples = self.synthesize_note(
                frequency=freq,
                duration=duration,
                waveform=waveform,
                envelope=envelope,
                amplitude=voice_gain
            )
            for i in range(num_samples):
                mixed[i] += voice_samples[i]

        # Soft-clipping saturation limiter: tanh(x)
        return [math.tanh(x) for x in mixed]
