"""
Curated Neurobiological Presets for NeuroSim Desktop Studio.
Standard library Python: zero external dependencies.
"""

from typing import Tuple, Dict, Any
from programs.neurosim.biophys_engine import (
    BiophysicalNeuron,
    Synapse,
    MultiCompartmentNeuron,
    NeuralCircuit,
)


def preset_giant_squid_axon() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    Hodgkin-Huxley (1952) Giant Squid Axon.
    Canonical single action potential generation and refractory recovery
    under tonic depolarizing current injection.
    """
    circuit = NeuralCircuit()
    soma = BiophysicalNeuron(0, label="Giant Squid Axon", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3)
    soma.i_inj = 9.5  # Sustained depolarizing current
    circuit.add_neuron(soma)

    metadata = {
        "name": "Hodgkin-Huxley Giant Squid Axon",
        "description": "Canonical 1952 conductance-based action potential spike train with fast Na+ activation and delayed K+ rectifier.",
        "primary_id": 0,
        "default_dt": 0.02,
        "voltage_range": (-90.0, 55.0),
    }
    return circuit, metadata


def preset_anode_break() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    Anode Break Excitation (Post-Inhibitory Rebound).
    Hyperpolarizing current clamp removes Na+ channel inactivation (h increases),
    such that sudden release triggers a rebound action potential without depolarizing drive.
    """
    circuit = NeuralCircuit()
    soma = BiophysicalNeuron(0, label="Rebound Neuron", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3)
    soma.i_inj = -4.5  # Hyperpolarizing hold
    circuit.add_neuron(soma)

    metadata = {
        "name": "Anode Break Excitation (Rebound Spiking)",
        "description": "Post-inhibitory rebound: release from hyperpolarization unblocks Na+ h-gates, firing an autonomous spike.",
        "primary_id": 0,
        "default_dt": 0.02,
        "voltage_range": (-95.0, 45.0),
    }
    return circuit, metadata


def preset_ping_gamma_oscillations() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    PING (Pyramidal-Interneuron Network Gamma) Cortical Oscillations.
    Reciprocal E-I microcircuit generating synchronous 40 Hz gamma rhythms:
    Pyramidal cell excites Interneuron via AMPA; Interneuron suppresses Pyramidal cell via GABA_A.
    """
    circuit = NeuralCircuit()

    # Pyramidal Excitatory Neuron
    pyr = BiophysicalNeuron(0, label="Pyramidal (E)", g_na_bar=120.0, g_k_bar=36.0, g_l=0.28)
    pyr.i_inj = 12.0  # Tonic drive to pyramidal cell
    circuit.add_neuron(pyr)

    # Fast-Spiking Basket Interneuron
    inte = BiophysicalNeuron(1, label="Basket Interneuron (I)", g_na_bar=140.0, g_k_bar=45.0, g_l=0.35)
    inte.i_inj = 1.0
    circuit.add_neuron(inte)

    # Reciprocal Chemical Synapses
    syn_e_to_i = Synapse(pre_id=0, post_id=1, syn_type="AMPA", weight=1.8, delay_ms=1.2)
    syn_i_to_e = Synapse(pre_id=1, post_id=0, syn_type="GABA", weight=2.4, delay_ms=1.5)
    circuit.add_synapse(syn_e_to_i)
    circuit.add_synapse(syn_i_to_e)

    metadata = {
        "name": "PING Cortical Gamma Oscillations (40 Hz)",
        "description": "Reciprocal Pyramidal-Interneuron microcircuit pacing coherent 40 Hz gamma rhythms via rhythmic feedback inhibition.",
        "primary_id": 0,
        "secondary_id": 1,
        "default_dt": 0.02,
        "voltage_range": (-85.0, 50.0),
    }
    return circuit, metadata


def preset_half_center_cpg() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    Locomotor Central Pattern Generator (CPG) Half-Center Oscillator.
    Bilateral reciprocal inhibition between left and right motor pools producing
    alternating rhythmic locomotor phases (swimming / walking gait).
    """
    circuit = NeuralCircuit()

    left_motor = BiophysicalNeuron(0, label="Left Motor Pool", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3)
    left_motor.i_inj = 11.5
    circuit.add_neuron(left_motor)

    right_motor = BiophysicalNeuron(1, label="Right Motor Pool", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3)
    right_motor.i_inj = 11.5
    right_motor.v = -70.0  # Slight phase asymmetry to trigger alternation
    circuit.add_neuron(right_motor)

    # Mutually inhibitory GABA synapses with spike frequency adaptation
    syn_l_to_r = Synapse(pre_id=0, post_id=1, syn_type="GABA", weight=2.2, delay_ms=1.0)
    syn_r_to_l = Synapse(pre_id=1, post_id=0, syn_type="GABA", weight=2.2, delay_ms=1.0)
    circuit.add_synapse(syn_l_to_r)
    circuit.add_synapse(syn_r_to_l)

    metadata = {
        "name": "Locomotor Half-Center CPG Oscillator",
        "description": "Bilateral reciprocal inhibition pacing alternating left-right motor burst phases for locomotion.",
        "primary_id": 0,
        "secondary_id": 1,
        "default_dt": 0.02,
        "voltage_range": (-85.0, 50.0),
    }
    return circuit, metadata


def preset_dendritic_backpropagation() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    Multi-Compartment Dendritic Cable & Back-Propagating Action Potential (bAP).
    Rall (1959) compartmental model: Soma connected to Basal Dendrite, Apical Trunk,
    and Apical Tuft via axial resistance R_a. Action potential initiated in soma
    back-propagates into distal dendrites with amplitude attenuation and latency.
    """
    circuit = NeuralCircuit()
    mc = MultiCompartmentNeuron(g_axial=0.18)
    mc.compartments[0].i_inj = 14.0  # Current injected into Soma only
    circuit.set_multi_compartment(mc)

    # Mirror soma to main neuron pool for unified graphing
    circuit.add_neuron(mc.compartments[0])

    metadata = {
        "name": "Dendritic Cable & Back-Propagating Action Potential",
        "description": "Multi-compartment Rall cable model: somatic action potential back-propagates into apical dendritic tree with axial decay.",
        "primary_id": 0,
        "is_multicompartment": True,
        "default_dt": 0.02,
        "voltage_range": (-80.0, 50.0),
    }
    return circuit, metadata


def preset_thalamic_bursting() -> Tuple[NeuralCircuit, Dict[str, Any]]:
    """
    Thalamocortical Relay Neuron Bursting vs Tonic Spiking.
    Low-threshold T-type calcium channels (g_T = 2.4 mS/cm^2) generate burst discharges
    when de-inactivated at hyperpolarized resting states, transitioning to single-spike
    tonic transmission when depolarized.
    """
    circuit = NeuralCircuit()
    thalamic = BiophysicalNeuron(
        0,
        label="Thalamocortical Relay",
        g_na_bar=120.0,
        g_k_bar=36.0,
        g_l=0.32,
        g_t_bar=2.4,  # T-type Ca2+ channel
        e_ca=120.0,
        v_init=-72.0,
    )
    thalamic.i_inj = 3.8
    circuit.add_neuron(thalamic)

    metadata = {
        "name": "Thalamocortical Bursting vs Tonic Spiking",
        "description": "Low-threshold T-type Ca2+ channel kinetics generating burst discharges during sleep states and tonic spikes when aroused.",
        "primary_id": 0,
        "default_dt": 0.02,
        "voltage_range": (-90.0, 55.0),
    }
    return circuit, metadata


PRESETS = {
    "giant_squid": preset_giant_squid_axon,
    "anode_break": preset_anode_break,
    "ping_gamma": preset_ping_gamma_oscillations,
    "half_center": preset_half_center_cpg,
    "dendritic": preset_dendritic_backpropagation,
    "thalamic": preset_thalamic_bursting,
}
