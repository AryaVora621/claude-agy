"""Spike encoding and decoding schemes for neuromorphic computing.

Implements rate coding (Poisson, Bernoulli), temporal coding (Time-to-First-Spike / TTFS,
Rank Order Coding / ROC, Phase coding), temporal delta modulation,
and decoders (rate averaging, exponential convolution filter, first-spike winner-take-all).
"""

from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class SpikeEncoder(ABC):
    """Abstract base class for spike encoders."""

    @abstractmethod
    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        """Encode continuous values into a spike raster [num_timesteps][num_neurons]."""
        pass


class PoissonEncoder(SpikeEncoder):
    """Poisson rate encoder.

    Spike probability per step is P(spike) = rate * dt, where rate in Hz is
    proportional to input magnitude.
    """

    def __init__(self, max_rate_hz: float = 100.0, min_rate_hz: float = 0.0) -> None:
        self.max_rate_hz = max_rate_hz
        self.min_rate_hz = min_rate_hz

    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        dt_s = dt_ms * 1e-3
        num_neurons = len(values)
        raster: List[List[bool]] = []

        # Convert values to instantaneous firing rates in Hz
        rates = [
            self.min_rate_hz + max(0.0, min(1.0, v)) * (self.max_rate_hz - self.min_rate_hz)
            for v in values
        ]

        for _ in range(num_timesteps):
            step_spikes = []
            for r in rates:
                prob = r * dt_s
                step_spikes.append(random.random() < prob)
            raster.append(step_spikes)

        return raster


class BernoulliEncoder(SpikeEncoder):
    """Bernoulli encoder where normalized input value is the probability of firing."""

    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        num_neurons = len(values)
        clamped = [max(0.0, min(1.0, v)) for v in values]
        raster: List[List[bool]] = []

        for _ in range(num_timesteps):
            step_spikes = [random.random() < p for p in clamped]
            raster.append(step_spikes)

        return raster


class TTFSEncoder(SpikeEncoder):
    """Time-to-First-Spike (TTFS) latency encoder.

    High intensity values fire earliest in time; zero or low intensity values
    fire latest or not at all.
    """

    def __init__(self, tau_ms: float = 15.0, threshold: float = 0.05) -> None:
        self.tau_ms = tau_ms
        self.threshold = threshold

    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        num_neurons = len(values)
        raster: List[List[bool]] = [
            [False for _ in range(num_neurons)] for _ in range(num_timesteps)
        ]

        for i, val in enumerate(values):
            v_clamped = max(0.0, min(1.0, val))
            if v_clamped < self.threshold:
                continue

            # Logarithmic latency mapping: t = -tau * ln(v)
            latency_ms = -self.tau_ms * math.log(max(1e-4, v_clamped))
            step_idx = int(latency_ms / dt_ms)
            if 0 <= step_idx < num_timesteps:
                raster[step_idx][i] = True

        return raster


LatencyEncoder = TTFSEncoder


class RankOrderEncoder(SpikeEncoder):
    """Rank Order Coding (Thorpe 1998).

    Emits spikes sequentially ordered by input magnitude: highest magnitude fires
    at step 0, second highest at step 1, etc.
    """

    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        num_neurons = len(values)
        raster: List[List[bool]] = [
            [False for _ in range(num_neurons)] for _ in range(num_timesteps)
        ]

        # Rank indices by descending value
        indexed_values = sorted(enumerate(values), key=lambda item: item[1], reverse=True)

        for step, (neuron_idx, val) in enumerate(indexed_values):
            if step < num_timesteps and val > 0.0:
                raster[step][neuron_idx] = True

        return raster


class PhaseEncoder(SpikeEncoder):
    """Phase encoder locking spike timing to a reference oscillatory cycle."""

    def __init__(self, cycle_period_steps: int = 10) -> None:
        self.cycle_period_steps = max(1, cycle_period_steps)

    def encode(
        self,
        values: List[float],
        num_timesteps: int,
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        num_neurons = len(values)
        raster: List[List[bool]] = [
            [False for _ in range(num_neurons)] for _ in range(num_timesteps)
        ]

        # Map value in [0, 1] to phase offset within the cycle [0..cycle_period_steps-1]
        for i, v in enumerate(values):
            norm_v = max(0.0, min(1.0, v))
            offset = int((1.0 - norm_v) * (self.cycle_period_steps - 1))
            for t in range(num_timesteps):
                if (t % self.cycle_period_steps) == offset:
                    raster[t][i] = True

        return raster


class DeltaEncoder:
    """Temporal delta modulator generating bipolar spikes on signal changes."""

    def __init__(self, delta_threshold: float = 0.1) -> None:
        self.delta_threshold = delta_threshold
        self.last_values: List[float] = []

    def encode_step(self, current_values: List[float]) -> Tuple[List[bool], List[bool]]:
        """Compute ON (increase) and OFF (decrease) spikes for this step.

        Returns (on_spikes, off_spikes).
        """
        if not self.last_values:
            self.last_values = list(current_values)
            return [False] * len(current_values), [False] * len(current_values)

        on_spikes = []
        off_spikes = []

        for i, val in enumerate(current_values):
            diff = val - self.last_values[i]
            if diff >= self.delta_threshold:
                on_spikes.append(True)
                off_spikes.append(False)
                self.last_values[i] = val
            elif diff <= -self.delta_threshold:
                on_spikes.append(False)
                off_spikes.append(True)
                self.last_values[i] = val
            else:
                on_spikes.append(False)
                off_spikes.append(False)

        return on_spikes, off_spikes


class RateDecoder:
    """Decodes spike rasters into normalized firing rates or total spike counts."""

    @staticmethod
    def decode(raster: List[List[bool]]) -> List[float]:
        """Return average firing rate (spikes per timestep) for each neuron."""
        if not raster:
            return []
        num_timesteps = len(raster)
        num_neurons = len(raster[0])

        counts = [0.0] * num_neurons
        for step in raster:
            for j in range(num_neurons):
                if step[j]:
                    counts[j] += 1.0

        return [c / num_timesteps for c in counts]


class ExponentialFilterDecoder:
    """Convolves spike trains with an exponential filter to reconstruct continuous analog signals."""

    def __init__(self, tau_filter_ms: float = 20.0, dt_ms: float = 1.0) -> None:
        self.tau_filter_ms = tau_filter_ms
        self.dt_ms = dt_ms
        self.decay = math.exp(-dt_ms / tau_filter_ms)

    def decode(self, raster: List[List[bool]]) -> List[List[float]]:
        """Returns continuous analog trace [num_timesteps][num_neurons]."""
        if not raster:
            return []
        num_timesteps = len(raster)
        num_neurons = len(raster[0])

        traces: List[List[float]] = []
        current_state = [0.0] * num_neurons

        for step in raster:
            for j in range(num_neurons):
                spike_weight = 1.0 if step[j] else 0.0
                current_state[j] = current_state[j] * self.decay + spike_weight
            traces.append(list(current_state))

        return traces


class FirstSpikeWinnerDecoder:
    """Winner-Take-All (WTA) classification based on earliest firing neuron."""

    @staticmethod
    def decode(raster: List[List[bool]]) -> Tuple[Optional[int], Optional[int]]:
        """Returns (winning_neuron_id, first_spike_timestep), or (None, None) if no spike occurred."""
        for t, step in enumerate(raster):
            for neuron_id, spiked in enumerate(step):
                if spiked:
                    return neuron_id, t
        return None, None
