"""Surrogate gradient backpropagation through time (Spike-BPTT) engine.

Solves the non-differentiability of biological action potentials by replacing
the Heaviside step derivative with continuous surrogate functions during the backward pass.
Includes autograd trajectory unrolling, recurrent temporal credit assignment,
and pure Python SGD/Adam optimizers.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class SurrogateFunction(ABC):
    """Abstract base class for surrogate gradient functions."""

    @abstractmethod
    def forward(self, v: float, v_thresh: float) -> float:
        """Compute the forward action potential (0.0 or 1.0)."""
        pass

    @abstractmethod
    def derivative(self, v: float, v_thresh: float) -> float:
        """Compute the surrogate derivative dS/dV."""
        pass


class FastSigmoidSurrogate(SurrogateFunction):
    """Fast Sigmoid surrogate gradient:

    sigma'(x) = 1 / (1 + k * |x|)^2 where x = V - V_thresh.
    """

    def __init__(self, k: float = 25.0) -> None:
        self.k = k

    def forward(self, v: float, v_thresh: float) -> float:
        return 1.0 if v >= v_thresh else 0.0

    def derivative(self, v: float, v_thresh: float) -> float:
        x = v - v_thresh
        denom = 1.0 + self.k * abs(x)
        return 1.0 / (denom * denom)


class ArcTanSurrogate(SurrogateFunction):
    """ArcTan surrogate gradient:

    sigma'(x) = 1 / (pi * (1 + (pi * k * x)^2)) where x = V - V_thresh.
    """

    def __init__(self, k: float = 10.0) -> None:
        self.k = k

    def forward(self, v: float, v_thresh: float) -> float:
        return 1.0 if v >= v_thresh else 0.0

    def derivative(self, v: float, v_thresh: float) -> float:
        x = v - v_thresh
        arg = math.pi * self.k * x
        return (self.k) / (1.0 + arg * arg)


class TriangularSurrogate(SurrogateFunction):
    """Triangular (piecewise linear) surrogate gradient:

    sigma'(x) = max(0, 1 - |x| / gamma) / gamma.
    """

    def __init__(self, gamma: float = 0.5) -> None:
        self.gamma = gamma

    def forward(self, v: float, v_thresh: float) -> float:
        return 1.0 if v >= v_thresh else 0.0

    def derivative(self, v: float, v_thresh: float) -> float:
        x = abs(v - v_thresh)
        if x > self.gamma:
            return 0.0
        return (1.0 - x / self.gamma) / self.gamma


class SpikeBPTTTape:
    """Computational tape tracking forward simulation for Spike-BPTT."""

    def __init__(
        self,
        surrogate: Optional[SurrogateFunction] = None,
        decay_factor: float = 0.9,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
    ) -> None:
        self.surrogate = surrogate or FastSigmoidSurrogate()
        self.decay_factor = decay_factor
        self.v_thresh = v_thresh
        self.v_reset = v_reset

        # Unrolled trajectory histories: list across time steps t = 0..T-1
        self.inputs_t: List[List[float]] = []        # Shape at t: [in_features]
        self.voltages_t: List[List[float]] = []      # Shape at t: [out_features]
        self.spikes_t: List[List[float]] = []        # Shape at t: [out_features]

    def record_step(
        self,
        inputs: List[float],
        voltages: List[float],
        spikes: List[float],
    ) -> None:
        """Record one forward simulation timestep."""
        self.inputs_t.append(list(inputs))
        self.voltages_t.append(list(voltages))
        self.spikes_t.append(list(spikes))

    def clear(self) -> None:
        """Clear the unrolled trajectory."""
        self.inputs_t.clear()
        self.voltages_t.clear()
        self.spikes_t.clear()


class SpikingDenseLayer:
    """Trainable fully-connected spiking layer with LIF dynamics and BPTT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        decay_factor: float = 0.9,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        surrogate: Optional[SurrogateFunction] = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.decay_factor = decay_factor
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.surrogate = surrogate or FastSigmoidSurrogate()

        # Weight initialization (He / Xavier scale)
        scale = math.sqrt(2.0 / (in_features + out_features))
        import random
        self.weights: List[List[float]] = [
            [(random.random() * 2.0 - 1.0) * scale for _ in range(out_features)]
            for _ in range(in_features)
        ]
        self.biases: List[float] = [0.0] * out_features

        # Gradients
        self.grad_weights: List[List[float]] = [
            [0.0 for _ in range(out_features)] for _ in range(in_features)
        ]
        self.grad_biases: List[float] = [0.0] * out_features

        # State tracking
        self.voltages: List[float] = [v_reset] * out_features
        self.tape = SpikeBPTTTape(
            surrogate=self.surrogate,
            decay_factor=decay_factor,
            v_thresh=v_thresh,
            v_reset=v_reset,
        )

    def forward_step(self, x: List[float]) -> List[float]:
        """Compute one time step forward:

        V(t) = decay * V(t-1) * (1 - S(t-1)) + W^T * X(t) + b
        S(t) = Theta(V(t) - V_thresh)
        """
        out_spikes = [0.0] * self.out_features
        new_voltages = [0.0] * self.out_features

        for j in range(self.out_features):
            # Affine synaptic input
            synaptic_in = self.biases[j]
            for i in range(self.in_features):
                synaptic_in += x[i] * self.weights[i][j]

            # Leaky integration (with soft reset from prior step's voltage)
            v_prev = self.voltages[j]
            v_cur = self.decay_factor * v_prev + synaptic_in

            # Spike generation via surrogate threshold
            s_cur = self.surrogate.forward(v_cur, self.v_thresh)

            # Reset voltage if spiked
            if s_cur > 0.5:
                self.voltages[j] = self.v_reset
            else:
                self.voltages[j] = v_cur

            new_voltages[j] = v_cur
            out_spikes[j] = s_cur

        self.tape.record_step(x, new_voltages, out_spikes)
        return out_spikes

    def forward_sequence(self, sequence: List[List[float]]) -> List[List[float]]:
        """Process an entire time series input and return output spike sequence."""
        self.reset_state()
        self.tape.clear()
        outputs = []
        for x_t in sequence:
            outputs.append(self.forward_step(x_t))
        return outputs

    def backward(self, grad_output_spikes: List[List[float]]) -> List[List[float]]:
        """Backpropagate surrogate gradients through time (Spike-BPTT).

        grad_output_spikes: list across time steps t = 0..T-1 of dL/dS_t.
        Returns grad_inputs across time steps t = 0..T-1 of dL/dX_t.
        """
        t_len = len(self.tape.inputs_t)
        if t_len == 0:
            return []

        # Zero existing gradients
        for i in range(self.in_features):
            for j in range(self.out_features):
                self.grad_weights[i][j] = 0.0
        for j in range(self.out_features):
            self.grad_biases[j] = 0.0

        grad_inputs = [[0.0] * self.in_features for _ in range(t_len)]

        # Temporal backward accumulation
        # dL_dV_next holds accumulated gradient flowing backward in time through decay
        dl_dv_next = [0.0] * self.out_features

        for t in reversed(range(t_len)):
            x_t = self.tape.inputs_t[t]
            v_t = self.tape.voltages_t[t]
            s_t = self.tape.spikes_t[t]
            dl_ds_t = grad_output_spikes[t]

            dl_dv_cur = [0.0] * self.out_features

            for j in range(self.out_features):
                # Gradient of spike generation: dS/dV via surrogate derivative
                surrogate_grad = self.surrogate.derivative(v_t[j], self.v_thresh)

                # Total gradient with respect to membrane potential at time t:
                # dl/dv_t = (dl/ds_t * dS/dV) + (dl/dv_{t+1} * decay * (1 - s_t))
                reset_mask = 0.0 if s_t[j] > 0.5 else 1.0
                dl_dv = dl_ds_t[j] * surrogate_grad + dl_dv_next[j] * self.decay_factor * reset_mask
                dl_dv_cur[j] = dl_dv

                # Accumulate weight and bias gradients
                for i in range(self.in_features):
                    self.grad_weights[i][j] += x_t[i] * dl_dv
                    grad_inputs[t][i] += dl_dv * self.weights[i][j]

                self.grad_biases[j] += dl_dv

            dl_dv_next = dl_dv_cur

        return grad_inputs

    def reset_state(self) -> None:
        """Reset membrane potential to initial baseline."""
        self.voltages = [self.v_reset] * self.out_features


class SGDOptimizer:
    """Stochastic Gradient Descent optimizer with momentum and weight decay."""

    def __init__(
        self,
        layers: List[SpikingDenseLayer],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        self.layers = layers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Velocity buffers for weights and biases
        self.v_w: List[List[List[float]]] = [
            [[0.0 for _ in range(layer.out_features)] for _ in range(layer.in_features)]
            for layer in layers
        ]
        self.v_b: List[List[float]] = [
            [0.0 for _ in range(layer.out_features)] for layer in layers
        ]

    def step(self) -> None:
        """Update layer parameters using accumulated gradients."""
        for l_idx, layer in enumerate(self.layers):
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    grad = layer.grad_weights[i][j] + self.weight_decay * layer.weights[i][j]
                    self.v_w[l_idx][i][j] = (
                        self.momentum * self.v_w[l_idx][i][j] + self.lr * grad
                    )
                    layer.weights[i][j] -= self.v_w[l_idx][i][j]

            for j in range(layer.out_features):
                grad = layer.grad_biases[j]
                self.v_b[l_idx][j] = (
                    self.momentum * self.v_b[l_idx][j] + self.lr * grad
                )
                layer.biases[j] -= self.v_b[l_idx][j]


class AdamOptimizer:
    """Adam (Adaptive Moment Estimation) optimizer for Spiking Neural Networks."""

    def __init__(
        self,
        layers: List[SpikingDenseLayer],
        lr: float = 0.005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

        self.m_w: List[List[List[float]]] = [
            [[0.0 for _ in range(layer.out_features)] for _ in range(layer.in_features)]
            for layer in layers
        ]
        self.v_w: List[List[List[float]]] = [
            [[0.0 for _ in range(layer.out_features)] for _ in range(layer.in_features)]
            for layer in layers
        ]

        self.m_b: List[List[float]] = [
            [0.0 for _ in range(layer.out_features)] for layer in layers
        ]
        self.v_b: List[List[float]] = [
            [0.0 for _ in range(layer.out_features)] for layer in layers
        ]

    def step(self) -> None:
        self.step_count += 1
        bias_corr1 = 1.0 - (self.beta1 ** self.step_count)
        bias_corr2 = 1.0 - (self.beta2 ** self.step_count)

        for l_idx, layer in enumerate(self.layers):
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    g = layer.grad_weights[i][j]
                    self.m_w[l_idx][i][j] = self.beta1 * self.m_w[l_idx][i][j] + (1.0 - self.beta1) * g
                    self.v_w[l_idx][i][j] = self.beta2 * self.v_w[l_idx][i][j] + (1.0 - self.beta2) * (g * g)

                    m_hat = self.m_w[l_idx][i][j] / bias_corr1
                    v_hat = self.v_w[l_idx][i][j] / bias_corr2
                    layer.weights[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            for j in range(layer.out_features):
                g = layer.grad_biases[j]
                self.m_b[l_idx][j] = self.beta1 * self.m_b[l_idx][j] + (1.0 - self.beta1) * g
                self.v_b[l_idx][j] = self.beta2 * self.v_b[l_idx][j] + (1.0 - self.beta2) * (g * g)

                m_hat = self.m_b[l_idx][j] / bias_corr1
                v_hat = self.v_b[l_idx][j] / bias_corr2
                layer.biases[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def mean_squared_rate_loss(
    spike_sequence: List[List[float]],
    target_rates: List[float],
) -> Tuple[float, List[List[float]]]:
    """Compute MSE loss on total spike counts and calculate output gradients.

    Returns (loss_scalar, grad_spikes_t) where grad_spikes_t is [T][num_neurons].
    """
    t_len = len(spike_sequence)
    num_neurons = len(target_rates)

    # Calculate actual firing rates
    actual_rates = [0.0] * num_neurons
    for s_t in spike_sequence:
        for j in range(num_neurons):
            actual_rates[j] += s_t[j]
    for j in range(num_neurons):
        actual_rates[j] /= max(1, t_len)

    # Compute MSE loss
    loss = 0.0
    rate_diff = [0.0] * num_neurons
    for j in range(num_neurons):
        diff = actual_rates[j] - target_rates[j]
        loss += 0.5 * diff * diff
        rate_diff[j] = diff / max(1, t_len)

    # Distribute gradient equally across all time steps
    grad_spikes = [[rate_diff[j] for j in range(num_neurons)] for _ in range(t_len)]

    return loss, grad_spikes
