"""High-throughput microbenchmarks for NeuroSynapse subsystems.

Measures operational throughput and latency across:
1. LIF Neuron Integration
2. Izhikevich 2D Dynamical Integration
3. Hodgkin-Huxley 4-State Conductance RK4 Integration
4. Synaptic Matrix & Online STDP Plasticity
5. Surrogate Gradient Spike-BPTT Forward & Backward
6. DVS Spatio-Temporal Event Filtering
7. Event-Based Plane-Fitting Optical Flow
8. Sub-Pixel Unicode Braille Rasterizer Rendering
"""

import time
from typing import Callable, Tuple

from neurosynapse.dvs import (
    BackgroundActivityFilter,
    DVSEvent,
    DVSStimulusGenerator,
    OpticalFlowEstimator,
    RefractoryFilter,
)
from neurosynapse.learning import SpikingDenseLayer
from neurosynapse.neurons import HodgkinHuxleyNeuron, IzhikevichNeuron, LIFNeuron
from neurosynapse.synapses import STDPConfig, SynapticMatrix
from neurosynapse.visualizer import render_spike_raster


def benchmark(name: str, fn: Callable[[], int], iterations: int = 5) -> Tuple[float, float]:
    """Run benchmark function over multiple iterations and report throughput and latency."""
    # Warmup
    fn()

    times = []
    total_ops = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        ops = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        total_ops += ops

    total_time = sum(times)
    throughput = total_ops / total_time
    avg_latency_us = (total_time / total_ops) * 1e6
    return throughput, avg_latency_us


def bench_lif_neuron() -> int:
    n = LIFNeuron()
    steps = 50000
    for t in range(steps):
        n.step(dt_ms=1.0, i_inj=5.0, current_time_ms=float(t))
    return steps


def bench_izhikevich_neuron() -> int:
    n = IzhikevichNeuron.regular_spiking()
    steps = 40000
    for t in range(steps):
        n.step(dt_ms=0.5, i_inj=10.0, current_time_ms=t * 0.5)
    return steps


def bench_hodgkin_huxley() -> int:
    hh = HodgkinHuxleyNeuron()
    steps = 15000
    for t in range(steps):
        hh.step(dt_ms=0.02, i_inj=10.0, current_time_ms=t * 0.02)
    return steps


def bench_synaptic_matrix() -> int:
    cfg = STDPConfig(a_plus=0.01, a_minus=0.012)
    mat = SynapticMatrix(num_pre=32, num_post=32, initial_weight=0.2, stdp_config=cfg)
    steps = 1000
    pre = [i % 4 == 0 for i in range(32)]
    post = [i % 3 == 0 for i in range(32)]
    for _ in range(steps):
        mat.step(dt_ms=1.0, pre_spikes=pre, post_spikes=post)
    return steps * 32 * 32  # Synapse-steps


def bench_spike_bptt() -> int:
    layer = SpikingDenseLayer(in_features=16, out_features=16, decay_factor=0.85)
    inputs = [[0.5] * 16 for _ in range(25)]
    grad_out = [[0.1] * 16 for _ in range(25)]
    runs = 100
    for _ in range(runs):
        layer.forward_sequence(inputs)
        layer.backward(grad_out)
    return runs * 25  # Timesteps


def bench_dvs_filtering() -> int:
    stream = DVSStimulusGenerator.moving_vertical_bar(width=32, height=32, speed_px_s=50.0, duration_s=0.5)
    rf = RefractoryFilter(32, 32, tau_ref_us=5000)
    baf = BackgroundActivityFilter(32, 32, window_us=10000)
    for ev in stream.events:
        if rf.filter_event(ev):
            baf.filter_event(ev)
    return len(stream.events)


def bench_optical_flow() -> int:
    stream = DVSStimulusGenerator.moving_vertical_bar(width=32, height=32, speed_px_s=50.0, duration_s=0.2)
    estimator = OpticalFlowEstimator(width=32, height=32)
    for ev in stream.events:
        estimator.estimate_flow(ev)
    return len(stream.events)


def bench_braille_raster() -> int:
    raster = [[(t + i) % 7 == 0 for i in range(32)] for t in range(50)]
    frames = 200
    for _ in range(frames):
        render_spike_raster(raster, char_width=40, char_height=12)
    return frames


def run_all_benchmarks() -> None:
    print("=" * 72)
    print("          NEUROSYNAPSE NEUROMORPHIC PERFORMANCE BENCHMARKS")
    print("=" * 72)
    print(f"{'Benchmark Operation':<36} | {'Throughput':<18} | {'Latency':<12}")
    print("-" * 72)

    benchmarks = [
        ("LIF Neuron Integration", bench_lif_neuron, "steps/sec"),
        ("Izhikevich 2D Integration", bench_izhikevich_neuron, "steps/sec"),
        ("Hodgkin-Huxley RK4 Integration", bench_hodgkin_huxley, "steps/sec"),
        ("Synaptic Matrix + STDP Plasticity", bench_synaptic_matrix, "syn-steps/sec"),
        ("Spike-BPTT Forward & Backward", bench_spike_bptt, "timesteps/sec"),
        ("DVS Spatio-Temporal Filtering", bench_dvs_filtering, "events/sec"),
        ("Event-Based Optical Flow", bench_optical_flow, "events/sec"),
        ("Braille Spike Raster Rendering", bench_braille_raster, "frames/sec"),
    ]

    for label, fn, unit in benchmarks:
        tp, lat = benchmark(label, fn, iterations=4)
        print(f"{label:<36} | {tp:>10,.1f} {unit:<7} | {lat:>8.2f} μs")

    print("=" * 72)


if __name__ == "__main__":
    run_all_benchmarks()
