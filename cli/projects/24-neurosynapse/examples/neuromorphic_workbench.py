"""Interactive Terminal Neuromorphic Workbench.

Demonstrates:
1. Spiking Audio Spectrogram Classifier (Surrogate Gradient BPTT)
2. Liquid State Machine (LSM) Cortical Reservoir
3. Unsupervised STDP Feature Discovery with Lateral Inhibition
4. Neuromorphic DVS Event-Based Vision & Optical Flow
5. Dual-Trace Biophysical Membrane Potential Oscilloscope
"""

import argparse
import math
import sys
import time

from neurosynapse.dvs import DVSStimulusGenerator, OpticalFlowEstimator
from neurosynapse.engine import (
    NeuromorphicEngine,
    SpikingAudioClassifier,
    UnsupervisedSTDPFeatureExtractor,
)
from neurosynapse.neurons import HodgkinHuxleyNeuron, IzhikevichNeuron, LIFNeuron
from neurosynapse.visualizer import (
    render_dvs_accumulator,
    render_oscilloscope,
    render_spike_raster,
    render_telemetry_hud,
)


def demo_oscilloscope() -> None:
    print("\n" + "=" * 76)
    print("      DEMO 1: BIOPHYSICAL MEMBRANE POTENTIAL DUAL-TRACE OSCILLOSCOPE")
    print("=" * 76)
    print("Simulating Hodgkin-Huxley 4-variable biophysical action potential dynamics...")

    hh = HodgkinHuxleyNeuron()
    voltages = []
    # Inject 10 micro-A/cm^2 current pulse for 30 ms
    for t in range(1200):
        t_ms = t * 0.025
        i_inj = 10.0 if 5.0 <= t_ms <= 25.0 else 0.0
        hh.step(dt_ms=0.025, i_inj=i_inj, current_time_ms=t_ms)
        voltages.append(hh.v)

    # Downsample for terminal display
    downsampled = voltages[::5]
    print(render_oscilloscope(
        downsampled,
        char_width=55,
        char_height=14,
        v_min=-85.0,
        v_max=40.0,
        title="HODGKIN-HUXLEY ACTION POTENTIAL OSCILLOSCOPE (RK4)",
    ))


def demo_audio_classifier() -> None:
    print("\n" + "=" * 76)
    print("      DEMO 2: SPIKING AUDIO CLASSIFIER (SURROGATE GRADIENT BPTT)")
    print("=" * 76)
    print("Training 2-layer SNN on auditory cochleagram frequency patterns...")

    res = NeuromorphicEngine.run_audio_classifier_demo()
    print(f"Training Loss: {res['initial_loss']} -> {res['final_loss']}")
    print(f"Inference Prediction: {res['prediction']}")
    print(res["raster"])
    print(res["hud"])


def demo_lsm() -> None:
    print("\n" + "=" * 76)
    print("      DEMO 3: LIQUID STATE MACHINE (LSM) 3D CORTICAL RESERVOIR")
    print("=" * 76)
    print("Simulating 3D cortical microcircuit with Dale's Principle (80% Exc, 20% Inh)...")

    res = NeuromorphicEngine.run_lsm_chaos_demo()
    print(f"Reservoir Neurons: {res['reservoir_size']}")
    print(f"Readout Prediction: {res['prediction']}")
    print(res["raster"])


def demo_stdp() -> None:
    print("\n" + "=" * 76)
    print("      DEMO 4: UNSUPERVISED STDP FEATURE EXTRACTION & LATERAL INHIBITION")
    print("=" * 76)
    print("Presenting alternating horizontal vs vertical pattern spike trains...")

    extractor = UnsupervisedSTDPFeatureExtractor(num_inputs=16, num_outputs=4)

    # Pattern A: Top half inputs active
    pat_a = [[i < 8 for i in range(16)] for _ in range(20)]
    # Pattern B: Bottom half inputs active
    pat_b = [[i >= 8 for i in range(16)] for _ in range(20)]

    for epoch in range(10):
        extractor.present_pattern(pat_a)
        extractor.present_pattern(pat_b)

    # Test Pattern A response
    out_a = extractor.present_pattern(pat_a)
    raster_display = render_spike_raster(
        out_a,
        char_width=45,
        char_height=8,
        title="STDP COMPETITIVE OUTPUT SPIKE RASTER",
    )
    print(raster_display)
    print("\nLearned Post-Synaptic Weight Profiles (4 Output Neurons):")
    for j in range(4):
        w_top = sum(extractor.synapses.weights[i][j] for i in range(8))
        w_bot = sum(extractor.synapses.weights[i][j] for i in range(8, 16))
        print(f"  Neuron {j}: Top Inputs Weight={w_top:.2f} | Bottom Inputs Weight={w_bot:.2f}")


def demo_dvs() -> None:
    print("\n" + "=" * 76)
    print("      DEMO 5: DVS EVENT-BASED VISION & PLANE-FITTING OPTICAL FLOW")
    print("=" * 76)
    print("Ingesting asynchronous Address-Event Representation (AER) packets...")

    res = NeuromorphicEngine.run_dvs_optical_flow_demo()
    print(f"Total DVS Events: {res['total_events']}")
    print(f"Flow Vectors: {res['flow_vectors_computed']}")
    print(f"Calculated Velocity: {res['estimated_velocity']}")
    print(res["canvas"])


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroSynapse Terminal Neuromorphic Workbench")
    parser.add_argument(
        "--mode",
        choices=["all", "oscilloscope", "audio", "lsm", "stdp", "dvs"],
        default="all",
        help="Workbench demo module to execute",
    )
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                 NEUROSYNAPSE NEUROMORPHIC WORKBENCH                        ║")
    print("║      First-Principles Spiking Neural Networks & Event-Based Vision         ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    if args.mode in ("all", "oscilloscope"):
        demo_oscilloscope()
    if args.mode in ("all", "audio"):
        demo_audio_classifier()
    if args.mode in ("all", "lsm"):
        demo_lsm()
    if args.mode in ("all", "stdp"):
        demo_stdp()
    if args.mode in ("all", "dvs"):
        demo_dvs()

    print("\n[NeuroSynapse Demo Completed Successfully]\n")


if __name__ == "__main__":
    main()
