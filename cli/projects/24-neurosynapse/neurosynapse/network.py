"""Spiking Neural Network (SNN) topologies and dual-mode execution engines.

Supports feedforward architectures, Liquid State Machine (LSM) recurrent reservoirs
with Dale's principle and distance-dependent connectivity, as well as both
synchronous time-stepped and asynchronous event-driven (priority queue) simulators.
"""

from __future__ import annotations
import heapq
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .neurons import LIFNeuron, SpikingNeuron
from .synapses import STDPConfig, SynapticMatrix


@dataclass(order=True)
class EventQueueItem:
    """Item for the asynchronous event-driven priority queue."""
    time_ms: float
    event_type: str      # 'spike' or 'stimulus'
    source_id: int
    target_id: int
    weight: float


class SpikingNetwork:
    """General spiking neural network graph composed of neurons and synaptic connections."""

    def __init__(self, dt_ms: float = 1.0) -> None:
        self.dt_ms = dt_ms
        self.current_time_ms: float = 0.0
        self.neurons: List[SpikingNeuron] = []
        self.synaptic_matrices: List[Tuple[int, int, SynapticMatrix]] = []  # (src_offset, dst_offset, matrix)
        self.adjacency: Dict[int, List[Tuple[int, float, float]]] = {}     # src -> [(dst, weight, delay)]

        # Global spike history: list of (time_ms, neuron_id)
        self.spike_history: List[Tuple[float, int]] = []

    def add_neuron(self, neuron: SpikingNeuron) -> int:
        """Add a neuron to the network and return its unique global ID."""
        neuron.neuron_id = len(self.neurons)
        self.neurons.append(neuron)
        self.adjacency[neuron.neuron_id] = []
        return neuron.neuron_id

    def add_synapse(
        self, pre_id: int, post_id: int, weight: float, delay_ms: float = 1.0
    ) -> None:
        """Add a directed synaptic connection between two neurons."""
        if pre_id not in self.adjacency:
            self.adjacency[pre_id] = []
        self.adjacency[pre_id].append((post_id, weight, delay_ms))

    def step_synchronous(
        self, external_currents: Optional[List[float]] = None
    ) -> List[bool]:
        """Advance the entire network by one synchronous timestep dt_ms."""
        num_neurons = len(self.neurons)
        input_currents = list(external_currents) if external_currents else [0.0] * num_neurons

        # Check spikes from previous step and deliver instantaneous/decayed currents
        # For simplicity in synchronous graph step, we accumulate into input_currents
        spikes = []
        for i, neuron in enumerate(self.neurons):
            spiked = neuron.step(self.dt_ms, input_currents[i], self.current_time_ms)
            spikes.append(spiked)
            if spiked:
                self.spike_history.append((self.current_time_ms, i))

        self.current_time_ms += self.dt_ms
        return spikes

    def reset(self) -> None:
        """Reset all neurons and state in the network."""
        for neuron in self.neurons:
            neuron.reset()
        self.current_time_ms = 0.0
        self.spike_history.clear()


class EventDrivenSimulator:
    """Asynchronous, event-driven priority queue spiking simulation engine.

    Only updates synapses when action potentials actually fire.
    Computational complexity scales with O(spikes) rather than O(N^2 * T).
    """

    def __init__(self, network: SpikingNetwork) -> None:
        self.network = network
        self.event_queue: List[EventQueueItem] = []
        self.recorded_spikes: List[Tuple[float, int]] = []

    def schedule_stimulus(
        self, time_ms: float, target_id: int, current_amplitude: float
    ) -> None:
        """Inject external stimulus into a target neuron at a specific future timestamp."""
        item = EventQueueItem(
            time_ms=time_ms,
            event_type="stimulus",
            source_id=-1,
            target_id=target_id,
            weight=current_amplitude,
        )
        heapq.heappush(self.event_queue, item)

    def run_until(self, end_time_ms: float) -> List[Tuple[float, int]]:
        """Process events up to end_time_ms in strict chronological order."""
        while self.event_queue and self.event_queue[0].time_ms <= end_time_ms:
            item = heapq.heappop(self.event_queue)
            t = item.time_ms
            neuron = self.network.neurons[item.target_id]

            # Step neuron with the arriving impulse
            # In event-driven LIF, an arriving weight acts as a momentary delta current
            spiked = neuron.step(dt_ms=1.0, i_inj=item.weight, current_time_ms=t)

            if spiked:
                self.recorded_spikes.append((t, item.target_id))
                # Propagate action potential to all downstream synaptic targets
                for target_id, weight, delay in self.network.adjacency.get(item.target_id, []):
                    arrival_time = t + delay
                    spike_item = EventQueueItem(
                        time_ms=arrival_time,
                        event_type="spike",
                        source_id=item.target_id,
                        target_id=target_id,
                        weight=weight,
                    )
                    heapq.heappush(self.event_queue, spike_item)

        return list(self.recorded_spikes)


class LiquidStateMachine:
    """Recurrent Spiking Neural Network Reservoir (Maass et al. 2002).

    Features a 3D cortical microcircuit layout with Dale's Principle (80% excitatory,
    20% inhibitory), distance-dependent connectivity, and a trainable linear readout.
    """

    def __init__(
        self,
        grid_dim: Tuple[int, int, int] = (4, 4, 3),  # 48 neurons total
        inhibitory_ratio: float = 0.2,
        connection_prob_c: float = 0.3,
        lambda_dist: float = 2.0,
        tau_m_ms: float = 20.0,
    ) -> None:
        self.grid_dim = grid_dim
        self.nx, self.ny, self.nz = grid_dim
        self.total_neurons = self.nx * self.ny * self.nz
        self.inhibitory_ratio = inhibitory_ratio

        # Assign 3D coordinates and neuron types (Dale's principle)
        self.coords: List[Tuple[int, int, int]] = []
        self.is_inhibitory: List[bool] = []
        self.neurons: List[LIFNeuron] = []

        for z in range(self.nz):
            for y in range(self.ny):
                for x in range(self.nx):
                    idx = len(self.neurons)
                    is_inh = random.random() < inhibitory_ratio
                    self.coords.append((x, y, z))
                    self.is_inhibitory.append(is_inh)

                    # Inhibitory neurons typically have faster membrane dynamics
                    tau = tau_m_ms * 0.5 if is_inh else tau_m_ms
                    self.neurons.append(LIFNeuron(neuron_id=idx, tau_m_ms=tau))

        # Recurrent synaptic weight matrix [total_neurons x total_neurons]
        self.weights: List[List[float]] = [
            [0.0 for _ in range(self.total_neurons)] for _ in range(self.total_neurons)
        ]

        # Distance-dependent connectivity: P(u, v) = C * exp(-D(u, v)^2 / lambda^2)
        for i in range(self.total_neurons):
            x1, y1, z1 = self.coords[i]
            for j in range(self.total_neurons):
                if i == j:
                    continue
                x2, y2, z2 = self.coords[j]
                d_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
                prob = connection_prob_c * math.exp(-d_sq / (lambda_dist ** 2))

                if random.random() < prob:
                    # Dale's principle: Excitatory weights > 0, Inhibitory weights < 0
                    if self.is_inhibitory[i]:
                        self.weights[i][j] = -random.uniform(0.3, 0.8)
                    else:
                        self.weights[i][j] = random.uniform(0.2, 0.6)

        # Spectral radius scaling to maintain liquid state at edge of chaos
        self._scale_spectral_radius(target_radius=0.9)

        # Readout weights (initialized to 0.0, trained via ridge regression)
        self.readout_weights: List[List[float]] = []  # [num_neurons x num_classes]

    def _scale_spectral_radius(self, target_radius: float = 0.9) -> None:
        """Approximate spectral radius scaling via Gershgorin circle row-sum bound."""
        max_row_sum = 0.0
        for i in range(self.total_neurons):
            row_sum = sum(abs(self.weights[i][j]) for j in range(self.total_neurons))
            if row_sum > max_row_sum:
                max_row_sum = row_sum

        if max_row_sum > 1e-6:
            factor = target_radius / max_row_sum
            for i in range(self.total_neurons):
                for j in range(self.total_neurons):
                    self.weights[i][j] *= factor

    def simulate(
        self,
        input_spike_trains: List[List[bool]],
        input_weights: List[List[float]],
        dt_ms: float = 1.0,
    ) -> List[List[bool]]:
        """Simulate the liquid state machine for T timesteps.

        input_spike_trains: [T][num_inputs]
        input_weights: [num_inputs][total_neurons]
        Returns state raster [T][total_neurons].
        """
        for neuron in self.neurons:
            neuron.reset()

        num_inputs = len(input_weights)
        t_len = len(input_spike_trains)
        raster: List[List[bool]] = []

        cur_reservoir_spikes = [False] * self.total_neurons

        for t in range(t_len):
            step_inputs = input_spike_trains[t]
            currents = [0.0] * self.total_neurons

            # Feedforward input currents
            for in_idx in range(num_inputs):
                if step_inputs[in_idx]:
                    for r_idx in range(self.total_neurons):
                        currents[r_idx] += input_weights[in_idx][r_idx]

            # Recurrent currents from previous step's reservoir spikes
            for i in range(self.total_neurons):
                if cur_reservoir_spikes[i]:
                    for j in range(self.total_neurons):
                        currents[j] += self.weights[i][j]

            # Step all neurons
            new_spikes = []
            for i, neuron in enumerate(self.neurons):
                new_spikes.append(neuron.step(dt_ms, currents[i], current_time_ms=t * dt_ms))

            cur_reservoir_spikes = new_spikes
            raster.append(new_spikes)

        return raster

    def extract_liquid_state(
        self, state_raster: List[List[bool]], decay_factor: float = 0.95
    ) -> List[float]:
        """Low-pass filter the spike raster to extract the continuous liquid state vector."""
        state_vec = [0.0] * self.total_neurons
        for step in state_raster:
            for j in range(self.total_neurons):
                spike_val = 1.0 if step[j] else 0.0
                state_vec[j] = state_vec[j] * decay_factor + spike_val
        return state_vec

    def train_readout(
        self,
        training_states: List[List[float]],
        target_labels: List[int],
        num_classes: int,
        reg_lambda: float = 1e-3,
    ) -> None:
        """Train linear readout weights using regularized ridge regression."""
        n_samples = len(training_states)
        n_features = self.total_neurons

        # Convert targets to one-hot: Y is [n_samples x num_classes]
        y_targets = [[0.0] * num_classes for _ in range(n_samples)]
        for i, label in enumerate(target_labels):
            if 0 <= label < num_classes:
                y_targets[i][label] = 1.0

        # Normal equations: (X^T X + lambda * I) W = X^T Y
        # Compute A = X^T X + lambda * I [n_features x n_features]
        a_mat = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                dot = sum(training_states[k][i] * training_states[k][j] for k in range(n_samples))
                a_mat[i][j] = dot
            a_mat[i][i] += reg_lambda

        # Compute B = X^T Y [n_features x num_classes]
        b_mat = [[0.0 for _ in range(num_classes)] for _ in range(n_features)]
        for i in range(n_features):
            for c in range(num_classes):
                b_mat[i][c] = sum(training_states[k][i] * y_targets[k][c] for k in range(n_samples))

        # Solve via Gauss-Jordan elimination
        self.readout_weights = self._solve_linear_system(a_mat, b_mat)

    def predict(self, state_vector: List[float]) -> int:
        """Predict the class label for a liquid state vector."""
        if not self.readout_weights:
            return 0

        num_classes = len(self.readout_weights[0])
        scores = [0.0] * num_classes
        for c in range(num_classes):
            for i, val in enumerate(state_vector):
                scores[c] += val * self.readout_weights[i][c]

        return max(range(num_classes), key=lambda c: scores[c])

    @staticmethod
    def _solve_linear_system(
        a: List[List[float]], b: List[List[float]]
    ) -> List[List[float]]:
        """Solve A * X = B for matrix X using Gauss-Jordan elimination with partial pivoting."""
        n = len(a)
        m = len(b[0])

        # Form augmented matrix [A | B]
        aug = [list(a[i]) + list(b[i]) for i in range(n)]

        for col in range(n):
            # Pivot selection
            max_row = col
            max_val = abs(aug[col][col])
            for r in range(col + 1, n):
                if abs(aug[r][col]) > max_val:
                    max_val = abs(aug[r][col])
                    max_row = r

            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                pivot = 1e-6

            # Normalize pivot row
            for c in range(col, n + m):
                aug[col][c] /= pivot

            # Eliminate column entries in other rows
            for r in range(n):
                if r != col:
                    factor = aug[r][col]
                    for c in range(col, n + m):
                        aug[r][c] -= factor * aug[col][c]

        # Extract X from augmented matrix
        x_res = [[aug[i][n + c] for c in range(m)] for i in range(n)]
        return x_res
