"""High-level unified orchestrator and flagship neuromorphic applications.

Provides high-level workflows for:
1. Spiking Audio Spectrogram Classifier (trained via Surrogate Gradient BPTT).
2. Liquid State Machine (LSM) Cortical Reservoir classifying dynamical systems.
3. Unsupervised STDP Feature Extractor with lateral inhibition and homeostasis.
4. Neuromorphic DVS Event Stream Processor & Optical Flow Tracker.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple

from .dvs import DVSStimulusGenerator, DVSStream, OpticalFlowEstimator
from .encoding import PoissonEncoder, RateDecoder
from .learning import AdamOptimizer, SpikingDenseLayer, mean_squared_rate_loss
from .network import LiquidStateMachine, SpikingNetwork
from .neurons import HodgkinHuxleyNeuron, IzhikevichNeuron, LIFNeuron
from .synapses import STDPConfig, SynapticMatrix
from .visualizer import (
    render_dvs_accumulator,
    render_oscilloscope,
    render_spike_raster,
    render_telemetry_hud,
)


class SpikingAudioClassifier:
    """Spiking Neural Network classifying frequency patterns using Spike-BPTT."""

    def __init__(
        self, num_freq_bins: int = 8, num_hidden: int = 16, num_classes: int = 2
    ) -> None:
        self.num_freq_bins = num_freq_bins
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.layer1 = SpikingDenseLayer(num_freq_bins, num_hidden, decay_factor=0.85, v_thresh=1.0)
        self.layer2 = SpikingDenseLayer(num_hidden, num_classes, decay_factor=0.90, v_thresh=1.0)
        self.optimizer = AdamOptimizer([self.layer1, self.layer2], lr=0.03)
        self.encoder = PoissonEncoder(max_rate_hz=100.0)

    def generate_synthetic_audio(
        self, class_id: int, num_timesteps: int = 20
    ) -> List[List[float]]:
        """Generate synthetic cochlear spectrogram patterns (Class 0: Low-pitch, Class 1: High-pitch)."""
        pattern = []
        for t in range(num_timesteps):
            row = [0.0] * self.num_freq_bins
            if class_id == 0:
                # Energy concentrated in lower frequency channels (0..3)
                for b in range(min(4, self.num_freq_bins)):
                    row[b] = 0.7 + 0.2 * math.sin(t * 0.5 + b)
            else:
                # Energy concentrated in higher frequency channels (4..7)
                for b in range(max(0, self.num_freq_bins - 4), self.num_freq_bins):
                    row[b] = 0.7 + 0.2 * math.sin(t * 0.5 + b)
            pattern.append(row)
        return pattern

    def forward(self, input_sequence: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
        """Run two-layer forward spiking inference."""
        h_spikes = self.layer1.forward_sequence(input_sequence)
        out_spikes = self.layer2.forward_sequence(h_spikes)
        return h_spikes, out_spikes

    def train_step(
        self, input_sequence: List[List[float]], target_class: int
    ) -> float:
        """One step of Surrogate Gradient BPTT training."""
        h_spikes, out_spikes = self.forward(input_sequence)

        # Target rate: 0.7 for target class, 0.05 for non-target classes
        target_rates = [0.05] * self.num_classes
        target_rates[target_class] = 0.7

        loss, grad_out = mean_squared_rate_loss(out_spikes, target_rates)

        grad_h = self.layer2.backward(grad_out)
        self.layer1.backward(grad_h)
        self.optimizer.step()

        return loss

    def predict(self, input_sequence: List[List[float]]) -> int:
        """Predict class label with highest output firing rate."""
        _, out_spikes = self.forward(input_sequence)
        counts = [0.0] * self.num_classes
        for step in out_spikes:
            for c in range(self.num_classes):
                counts[c] += step[c]
        return max(range(self.num_classes), key=lambda c: counts[c])


class UnsupervisedSTDPFeatureExtractor:
    """Unsupervised competitive feature learner using pair-based STDP and homeostasis."""

    def __init__(self, num_inputs: int = 16, num_outputs: int = 4) -> None:
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        cfg = STDPConfig(
            a_plus=0.02,
            a_minus=0.022,
            tau_plus_ms=20.0,
            tau_minus_ms=20.0,
            w_min=0.0,
            w_max=1.0,
            soft_bounds=True,
        )
        self.synapses = SynapticMatrix(
            num_pre=num_inputs,
            num_post=num_outputs,
            initial_weight=0.25,
            stdp_config=cfg,
        )
        self.output_neurons = [
            LIFNeuron(neuron_id=i, v_thresh=-55.0, v_reset=-70.0, tau_m_ms=15.0)
            for i in range(num_outputs)
        ]

    def present_pattern(
        self, pattern_spikes: List[List[bool]], dt_ms: float = 1.0
    ) -> List[List[bool]]:
        """Present input pattern spikes over time and update weights via STDP."""
        t_len = len(pattern_spikes)
        output_raster: List[List[bool]] = []

        for t in range(t_len):
            pre_step = pattern_spikes[t]

            # Compute input currents from synaptic matrix
            currents = self.synapses.step(dt_ms, pre_spikes=pre_step)

            # Lateral inhibition: Winner-Take-All competition
            step_out_spikes = []
            max_v = -1e9
            winner_idx = -1

            # Step neurons and check who fires
            for i, n in enumerate(self.output_neurons):
                spiked = n.step(dt_ms, currents[i] * 5.0, current_time_ms=t * dt_ms)
                if spiked and n.v > max_v:
                    max_v = n.v
                    winner_idx = i

            for i in range(self.num_outputs):
                step_out_spikes.append(i == winner_idx)

            # Re-update synaptic matrix with actual post spikes (LTP for winner)
            if winner_idx != -1:
                self.synapses.step(dt_ms, pre_spikes=[False] * self.num_inputs, post_spikes=step_out_spikes)

            output_raster.append(step_out_spikes)

        # Apply homeostatic normalization after pattern presentation
        self.synapses.normalize_weights_homeostatic(target_sum=1.5)
        return output_raster


class NeuromorphicEngine:
    """Unified top-level orchestrator for the NeuroSynapse platform."""

    @staticmethod
    def run_audio_classifier_demo() -> Dict[str, str]:
        """Run the Spiking Audio Classifier demo and return visualization outputs."""
        classifier = SpikingAudioClassifier(num_freq_bins=8, num_hidden=12, num_classes=2)

        # Train on synthetic patterns
        train_losses = []
        for epoch in range(15):
            # Class 0: Low pitch
            p0 = classifier.generate_synthetic_audio(0, num_timesteps=15)
            l0 = classifier.train_step(p0, target_class=0)
            # Class 1: High pitch
            p1 = classifier.generate_synthetic_audio(1, num_timesteps=15)
            l1 = classifier.train_step(p1, target_class=1)
            train_losses.append((l0 + l1) * 0.5)

        # Test on fresh sample
        test_p0 = classifier.generate_synthetic_audio(0, num_timesteps=20)
        pred0 = classifier.predict(test_p0)
        h_spikes, out_spikes = classifier.forward(test_p0)

        # Render outputs
        raster_display = render_spike_raster(
            [[bool(s > 0.5) for s in step] for step in h_spikes],
            char_width=45,
            char_height=10,
            title="AUDIO SNN HIDDEN LAYER SPIKE RASTER",
        )
        hud_display = render_telemetry_hud(
            num_neurons=12,
            total_spikes=sum(int(sum(step)) for step in h_spikes),
            duration_ms=20.0,
            active_synapses=8 * 12 + 12 * 2,
        )

        return {
            "prediction": f"Class {pred0} (Expected: 0)",
            "initial_loss": f"{train_losses[0]:.4f}",
            "final_loss": f"{train_losses[-1]:.4f}",
            "raster": raster_display,
            "hud": hud_display,
        }

    @staticmethod
    def run_lsm_chaos_demo() -> Dict[str, str]:
        """Run Liquid State Machine classifying chaotic attractor trajectories."""
        lsm = LiquidStateMachine(grid_dim=(3, 3, 2), inhibitory_ratio=0.2)  # 18 neurons

        # Generate two dynamical trajectories: Period-2 oscillation vs High-freq burst
        def make_trajectory(mode: int, t_steps: int = 40) -> List[List[bool]]:
            traj = []
            for t in range(t_steps):
                if mode == 0:
                    # Alternating pulse
                    traj.append([t % 4 == 0, t % 8 == 0])
                else:
                    # High frequency burst
                    traj.append([t % 2 == 0, t % 3 == 0])
            return traj

        input_weights = [[12.0] * lsm.total_neurons, [8.0] * lsm.total_neurons]

        # Collect training states
        states = []
        labels = []
        for _ in range(8):
            r0 = lsm.simulate(make_trajectory(0), input_weights)
            states.append(lsm.extract_liquid_state(r0))
            labels.append(0)

            r1 = lsm.simulate(make_trajectory(1), input_weights)
            states.append(lsm.extract_liquid_state(r1))
            labels.append(1)

        lsm.train_readout(states, labels, num_classes=2)

        # Test
        test_r = lsm.simulate(make_trajectory(0), input_weights)
        test_s = lsm.extract_liquid_state(test_r)
        pred = lsm.predict(test_s)

        raster_display = render_spike_raster(
            test_r,
            char_width=45,
            char_height=12,
            title="LSM 3D CORTICAL RESERVOIR SPIKE RASTER",
        )

        return {
            "reservoir_size": str(lsm.total_neurons),
            "prediction": f"Class {pred} (Expected: 0)",
            "raster": raster_display,
        }

    @staticmethod
    def run_dvs_optical_flow_demo() -> Dict[str, str]:
        """Run neuromorphic event-based optical flow tracking on moving visual stimuli."""
        stream = DVSStimulusGenerator.moving_vertical_bar(
            width=32, height=32, speed_px_s=60.0, duration_s=0.1
        )
        estimator = OpticalFlowEstimator(width=32, height=32)

        flows = []
        events_for_acc = []
        for ev in stream.events:
            events_for_acc.append((ev.x, ev.y, ev.polarity))
            f = estimator.estimate_flow(ev)
            if f is not None:
                flows.append(f)

        avg_vx = sum(f[0] for f in flows) / max(1, len(flows))
        avg_vy = sum(f[1] for f in flows) / max(1, len(flows))

        dvs_display = render_dvs_accumulator(
            events_for_acc,
            width=32,
            height=32,
            char_width=24,
            char_height=12,
            title="DVS RETINA EVENT FIELD",
        )

        return {
            "total_events": str(len(stream)),
            "flow_vectors_computed": str(len(flows)),
            "estimated_velocity": f"vx={avg_vx:.1f} px/s, vy={avg_vy:.1f} px/s (Ground Truth: 60.0 px/s)",
            "canvas": dvs_display,
        }
