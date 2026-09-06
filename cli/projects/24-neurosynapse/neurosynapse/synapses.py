"""Synaptic dynamics and Spike-Timing-Dependent Plasticity (STDP) learning rules.

Implements exponential and alpha-function post-synaptic currents,
online pair-based asymmetric STDP with eligibility traces,
triplet-based STDP for frequency-dependent burst sensitivity,
and homeostatic multi-synaptic weight normalization.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class SynapseType(str, Enum):
    """Kinetics of the post-synaptic current (PSC)."""
    EXPONENTIAL = "exponential"
    ALPHA = "alpha"
    INSTANTANEOUS = "instantaneous"


@dataclass
class STDPConfig:
    """Configuration parameters for Spike-Timing-Dependent Plasticity."""
    a_plus: float = 0.01          # LTP learning rate (potentiation magnitude)
    a_minus: float = 0.0105       # LTD learning rate (depression magnitude, typically slightly > a_plus)
    tau_plus_ms: float = 20.0     # Time constant for pre-synaptic trace / LTP window
    tau_minus_ms: float = 20.0    # Time constant for post-synaptic trace / LTD window
    w_min: float = 0.0            # Lower synaptic weight bound
    w_max: float = 1.0            # Upper synaptic weight bound
    soft_bounds: bool = False     # Use multiplicative soft bounds instead of hard clamping


class Synapse:
    """Individual synaptic connection between two neurons."""

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 1.0,
        delay_ms: float = 1.0,
        tau_syn_ms: float = 5.0,
        synapse_type: SynapseType = SynapseType.EXPONENTIAL,
        stdp_config: Optional[STDPConfig] = None,
    ) -> None:
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = weight
        self.delay_ms = delay_ms
        self.tau_syn_ms = tau_syn_ms
        self.synapse_type = synapse_type
        self.stdp_config = stdp_config

        # Post-synaptic current state variables
        self.i_syn: float = 0.0
        self.g_aux: float = 0.0  # Auxiliary variable for alpha-function kinetics

        # Online trace variables for STDP
        self.trace_pre: float = 0.0
        self.trace_post: float = 0.0

        # Delayed spike arrival queue: list of (arrival_time_ms, spike_weight)
        self._delay_queue: List[Tuple[float, float]] = []

    def on_pre_spike(self, current_time_ms: float) -> None:
        """Called when the presynaptic neuron fires an action potential."""
        arrival_time = current_time_ms + self.delay_ms
        self._delay_queue.append((arrival_time, self.weight))

        # Update presynaptic eligibility trace for STDP
        if self.stdp_config is not None:
            self.trace_pre += 1.0
            # Pre-spike arrives while post trace is active -> Long-Term Depression (LTD)
            cfg = self.stdp_config
            dw = -cfg.a_minus * self.trace_post
            self._apply_dw(dw)

    def on_post_spike(self, current_time_ms: float) -> None:
        """Called when the postsynaptic neuron fires an action potential."""
        if self.stdp_config is not None:
            self.trace_post += 1.0
            # Post-spike fires while pre trace is active -> Long-Term Potentiation (LTP)
            cfg = self.stdp_config
            dw = cfg.a_plus * self.trace_pre
            self._apply_dw(dw)

    def _apply_dw(self, dw: float) -> None:
        """Apply weight delta with hard or soft bounding."""
        cfg = self.stdp_config
        if cfg is None:
            return

        if cfg.soft_bounds:
            if dw > 0.0:
                self.weight += dw * (cfg.w_max - self.weight)
            else:
                self.weight += dw * (self.weight - cfg.w_min)
        else:
            self.weight += dw

        self.weight = max(cfg.w_min, min(cfg.w_max, self.weight))

    def step(self, dt_ms: float, current_time_ms: float) -> float:
        """Advance synaptic conductance and decay STDP traces by dt_ms.

        Returns the post-synaptic current (i_syn) delivered to the target neuron.
        """
        # Decay STDP traces
        if self.stdp_config is not None:
            decay_pre = math.exp(-dt_ms / self.stdp_config.tau_plus_ms)
            decay_post = math.exp(-dt_ms / self.stdp_config.tau_minus_ms)
            self.trace_pre *= decay_pre
            self.trace_post *= decay_post

        # Process arriving spikes from delay queue
        arrived_weight = 0.0
        remaining_queue = []
        for arrival_time, w in self._delay_queue:
            if arrival_time <= current_time_ms + 1e-9:
                arrived_weight += w
            else:
                remaining_queue.append((arrival_time, w))
        self._delay_queue = remaining_queue

        # Update PSC kinetics
        if self.synapse_type == SynapseType.INSTANTANEOUS:
            self.i_syn = arrived_weight
        elif self.synapse_type == SynapseType.EXPONENTIAL:
            decay = math.exp(-dt_ms / self.tau_syn_ms)
            self.i_syn = self.i_syn * decay + arrived_weight
        elif self.synapse_type == SynapseType.ALPHA:
            decay = math.exp(-dt_ms / self.tau_syn_ms)
            # dg/dt = -g / tau; dI/dt = (g - I) / tau
            self.g_aux = self.g_aux * decay + arrived_weight * (math.e / self.tau_syn_ms)
            self.i_syn = self.i_syn * decay + self.g_aux * (dt_ms / self.tau_syn_ms)

        return self.i_syn

    def reset(self) -> None:
        """Reset synaptic dynamics and queues."""
        self.i_syn = 0.0
        self.g_aux = 0.0
        self.trace_pre = 0.0
        self.trace_post = 0.0
        self._delay_queue.clear()


@dataclass
class TripletSTDPConfig:
    """Parameters for Pfister-Gerstner (2006) triplet STDP model."""
    a2_plus: float = 0.005       # 2-spike LTP magnitude
    a3_plus: float = 0.015       # 3-spike LTP magnitude
    a2_minus: float = 0.007      # 2-spike LTD magnitude
    a3_minus: float = 0.000      # 3-spike LTD magnitude
    tau_plus_ms: float = 16.8    # Fast pre trace
    tau_x_ms: float = 101.0      # Slow pre trace
    tau_minus_ms: float = 33.7   # Fast post trace
    tau_y_ms: float = 125.0      # Slow post trace
    w_min: float = 0.0
    w_max: float = 1.0


class TripletSynapse:
    """Synapse governed by the Pfister-Gerstner Triplet STDP learning rule."""

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 0.5,
        config: Optional[TripletSTDPConfig] = None,
        tau_syn_ms: float = 5.0,
    ) -> None:
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = weight
        self.config = config or TripletSTDPConfig()
        self.tau_syn_ms = tau_syn_ms

        # Two pre traces: r1 (fast), r2 (slow)
        self.r1: float = 0.0
        self.r2: float = 0.0

        # Two post traces: o1 (fast), o2 (slow)
        self.o1: float = 0.0
        self.o2: float = 0.0

        self.i_syn: float = 0.0

    def on_pre_spike(self) -> None:
        # Pre-spike causes depression dependent on post fast trace o1
        dw = -self.o1 * (self.config.a2_minus + self.config.a3_minus * self.r2)
        self.weight = max(self.config.w_min, min(self.config.w_max, self.weight + dw))
        self.r1 += 1.0
        self.r2 += 1.0
        self.i_syn += self.weight

    def on_post_spike(self) -> None:
        # Post-spike causes potentiation dependent on pre fast trace r1 and post slow trace o2
        dw = self.r1 * (self.config.a2_plus + self.config.a3_plus * self.o2)
        self.weight = max(self.config.w_min, min(self.config.w_max, self.weight + dw))
        self.o1 += 1.0
        self.o2 += 1.0

    def step(self, dt_ms: float) -> float:
        decay_r1 = math.exp(-dt_ms / self.config.tau_plus_ms)
        decay_r2 = math.exp(-dt_ms / self.config.tau_x_ms)
        decay_o1 = math.exp(-dt_ms / self.config.tau_minus_ms)
        decay_o2 = math.exp(-dt_ms / self.config.tau_y_ms)
        decay_syn = math.exp(-dt_ms / self.tau_syn_ms)

        self.r1 *= decay_r1
        self.r2 *= decay_r2
        self.o1 *= decay_o1
        self.o2 *= decay_o2
        self.i_syn *= decay_syn

        return self.i_syn


class SynapticMatrix:
    """Fully-connected or sparse synaptic projection matrix between two populations.

    Dimensions: num_pre x num_post.
    Provides vectorized synaptic current calculation and homeostatic normalization.
    """

    def __init__(
        self,
        num_pre: int,
        num_post: int,
        initial_weight: float = 0.2,
        tau_syn_ms: float = 5.0,
        stdp_config: Optional[STDPConfig] = None,
    ) -> None:
        self.num_pre = num_pre
        self.num_post = num_post
        self.tau_syn_ms = tau_syn_ms
        self.stdp_config = stdp_config

        # 2D weight matrix: weights[pre_idx][post_idx]
        self.weights: List[List[float]] = [
            [initial_weight for _ in range(num_post)] for _ in range(num_pre)
        ]

        # Post-synaptic currents for each post neuron
        self.post_currents: List[float] = [0.0] * num_post

        # Online trace vectors
        self.traces_pre: List[float] = [0.0] * num_pre
        self.traces_post: List[float] = [0.0] * num_post

    def step(
        self,
        dt_ms: float,
        pre_spikes: List[bool],
        post_spikes: Optional[List[bool]] = None,
    ) -> List[float]:
        """Step synaptic currents and update STDP weights given spike trains."""
        # 1. Decay synaptic currents
        decay_syn = math.exp(-dt_ms / self.tau_syn_ms)
        for j in range(self.num_post):
            self.post_currents[j] *= decay_syn

        # 2. Decay STDP traces
        if self.stdp_config is not None:
            decay_pre = math.exp(-dt_ms / self.stdp_config.tau_plus_ms)
            decay_post = math.exp(-dt_ms / self.stdp_config.tau_minus_ms)
            for i in range(self.num_pre):
                self.traces_pre[i] *= decay_pre
            for j in range(self.num_post):
                self.traces_post[j] *= decay_post

        # 3. Process presynaptic spikes
        for i, spiked in enumerate(pre_spikes):
            if spiked:
                if self.stdp_config is not None:
                    self.traces_pre[i] += 1.0
                    # LTD: Pre-spike depresses weights where post was active
                    cfg = self.stdp_config
                    for j in range(self.num_post):
                        dw = -cfg.a_minus * self.traces_post[j]
                        self._apply_weight_delta(i, j, dw)

                # Inject current into postsynaptic neurons
                for j in range(self.num_post):
                    self.post_currents[j] += self.weights[i][j]

        # 4. Process postsynaptic spikes (LTP)
        if post_spikes is not None and self.stdp_config is not None:
            cfg = self.stdp_config
            for j, post_spiked in enumerate(post_spikes):
                if post_spiked:
                    self.traces_post[j] += 1.0
                    # LTP: Post-spike potentiates weights where pre was active
                    for i in range(self.num_pre):
                        dw = cfg.a_plus * self.traces_pre[i]
                        self._apply_weight_delta(i, j, dw)

        return list(self.post_currents)

    def _apply_weight_delta(self, i: int, j: int, dw: float) -> None:
        cfg = self.stdp_config
        if cfg is None:
            return

        w = self.weights[i][j]
        if cfg.soft_bounds:
            if dw > 0.0:
                w += dw * (cfg.w_max - w)
            else:
                w += dw * (w - cfg.w_min)
        else:
            w += dw

        self.weights[i][j] = max(cfg.w_min, min(cfg.w_max, w))

    def normalize_weights_homeostatic(self, target_sum: float = 1.0) -> None:
        """Multiplicatively scale incoming synaptic weights for each postsynaptic neuron.

        Ensures sum_i |w_ij| = target_sum, preventing runaway excitation or silencing.
        """
        for j in range(self.num_post):
            current_sum = sum(abs(self.weights[i][j]) for i in range(self.num_pre))
            if current_sum > 1e-9:
                factor = target_sum / current_sum
                for i in range(self.num_pre):
                    self.weights[i][j] *= factor
