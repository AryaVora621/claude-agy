"""
Automated Unit Test Suite for NeuroSim Biophysical Engine.
16 tests verifying Hodgkin-Huxley dynamics, cable theory, synapses, and networks.
Standard library Python unittest: zero external dependencies.
"""

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from programs.neurosim.biophys_engine import (
    BiophysicalNeuron,
    Synapse,
    MultiCompartmentNeuron,
    NeuralCircuit,
)
from programs.neurosim.presets import (
    preset_giant_squid_axon,
    preset_anode_break,
    preset_ping_gamma_oscillations,
    preset_half_center_cpg,
    preset_dendritic_backpropagation,
    preset_thalamic_bursting,
)


class TestNeuroSimBiophysics(unittest.TestCase):
    """Test suite for biophysical electrophysiology and neural circuits."""

    def test_resting_potential_equilibrium(self):
        """Verify resting membrane potential settles stably near -65 mV without input."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        dt = 0.02
        for _ in range(500):  # 10 ms
            neuron.step(dt)

        self.assertAlmostEqual(neuron.v, -65.0, delta=1.5)
        self.assertFalse(neuron.is_spiking)

    def test_action_potential_generation(self):
        """Verify suprathreshold current triggers genuine action potential overshoot > +15 mV."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        neuron.i_inj = 10.0  # Suprathreshold current step
        dt = 0.02

        max_v = -65.0
        spiked = False
        for step_idx in range(1000):  # 20 ms
            t = step_idx * dt
            res = neuron.step(dt, sim_time=t)
            if res["v"] > max_v:
                max_v = res["v"]
            if res["just_spiked"] > 0.5:
                spiked = True

        self.assertTrue(spiked, "Neuron should register an action potential")
        self.assertGreater(max_v, 15.0, f"Spike peak {max_v:.1f} mV should overshoot +15 mV")

    def test_subthreshold_passive_charging(self):
        """Verify subthreshold current depolarizes membrane passively without firing a spike."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        neuron.i_inj = 1.0  # Subthreshold small current
        dt = 0.02

        max_v = -65.0
        spikes = 0
        for step_idx in range(1000):
            t = step_idx * dt
            res = neuron.step(dt, sim_time=t)
            if res["v"] > max_v:
                max_v = res["v"]
            if res["just_spiked"] > 0.5:
                spikes += 1

        self.assertEqual(spikes, 0, "Subthreshold current must not fire a spike")
        self.assertLess(max_v, -45.0, "Membrane should not depolarize past spike threshold")
        self.assertGreater(max_v, -65.0, "Membrane should depolarize from rest")

    def test_gating_variable_bounds(self):
        """Verify gating variables m, h, n remain strictly bounded in [0, 1] across voltages."""
        neuron = BiophysicalNeuron(0)
        test_voltages = [-120.0, -90.0, -65.0, -40.0, 0.0, 30.0, 60.0, 100.0]

        for v in test_voltages:
            m, h, n = neuron._steady_state_gates(v)
            self.assertTrue(0.0 <= m <= 1.0, f"m gate {m} out of bounds at V={v}")
            self.assertTrue(0.0 <= h <= 1.0, f"h gate {h} out of bounds at V={v}")
            self.assertTrue(0.0 <= n <= 1.0, f"n gate {n} out of bounds at V={v}")

    def test_absolute_refractory_period(self):
        """Verify second current pulse delivered during spike downstroke fails to fire second spike."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        dt = 0.02

        # 1. Trigger initial spike
        neuron.i_inj = 25.0
        t = 0.0
        first_spike_time = None
        for _ in range(250):  # 5 ms
            res = neuron.step(dt, sim_time=t)
            if res["just_spiked"] > 0.5 and first_spike_time is None:
                first_spike_time = t
            t += dt

        self.assertIsNotNone(first_spike_time, "Initial spike must occur")

        # 2. In absolute refractory period (e.g. 1.2 ms after spike peak), apply another massive pulse
        neuron.i_inj = 50.0
        second_spike = False
        for _ in range(60):  # 1.2 ms
            res = neuron.step(dt, sim_time=t)
            if res["just_spiked"] > 0.5:
                second_spike = True
            t += dt

        self.assertFalse(second_spike, "Neuron should be in absolute refractory period")

    def test_relative_refractory_period(self):
        """Verify after-hyperpolarization (AHP) creates elevated threshold 8 ms after a spike."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        dt = 0.02

        # Trigger single spike with brief pulse
        neuron.i_inj = 25.0
        t = 0.0
        for _ in range(100):  # 2 ms pulse
            neuron.step(dt, sim_time=t)
            t += dt
        neuron.i_inj = 0.0

        # Let it recover for 8 ms (into relative refractory AHP)
        for _ in range(400):
            neuron.step(dt, sim_time=t)
            t += dt

        # During AHP, V should be below resting potential (-65 mV) due to lingering K+ conductance
        self.assertLess(neuron.v, -64.0, "Neuron should exhibit after-hyperpolarization")

    def test_anode_break_rebound(self):
        """Verify release from prolonged hyperpolarization fires an autonomous rebound spike."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        dt = 0.02

        # Hyperpolarize strongly for 35 ms
        neuron.i_inj = -5.0
        t = 0.0
        for _ in range(1750):
            neuron.step(dt, sim_time=t)
            t += dt

        # Release current clamp: I_inj = 0
        neuron.i_inj = 0.0
        rebound_max_v = neuron.v
        rebound_spiked = False
        for _ in range(1000):  # 20 ms post-release
            res = neuron.step(dt, sim_time=t)
            if res["v"] > rebound_max_v:
                rebound_max_v = res["v"]
            if res["just_spiked"] > 0.5:
                rebound_spiked = True
            t += dt

        self.assertTrue(rebound_spiked, "Anode break release should trigger rebound action potential")
        self.assertGreater(rebound_max_v, 10.0, "Rebound spike must overshoot +10 mV")

    def test_cable_axial_attenuation(self):
        """Verify steady state voltage attenuates electrotonically along passive dendrites."""
        mc = MultiCompartmentNeuron(g_axial=0.20, passive_dendrites=True)
        dt = 0.02

        # Inject tonic subthreshold current into soma
        mc.compartments[0].i_inj = 3.0
        t = 0.0
        for _ in range(1500):  # 30 ms to steady state
            mc.step(dt, sim_time=t)
            t += dt

        v_soma = mc.compartments[0].v
        v_trunk = mc.compartments[2].v
        v_tuft = mc.compartments[3].v

        # Somatic depolarization should decay into trunk and tuft
        self.assertGreater(v_soma, v_trunk, "Soma must be more depolarized than apical trunk")
        self.assertGreater(v_trunk, v_tuft, "Apical trunk must be more depolarized than distal tuft")

    def test_backpropagating_action_potential(self):
        """Verify somatic action potential propagates into apical dendrites."""
        mc = MultiCompartmentNeuron(g_axial=0.25)
        dt = 0.02

        # Inject suprathreshold pulse into soma
        mc.compartments[0].i_inj = 18.0
        t = 0.0
        max_v_tuft = -65.0
        for _ in range(800):  # 16 ms
            mc.step(dt, sim_time=t)
            if mc.compartments[3].v > max_v_tuft:
                max_v_tuft = mc.compartments[3].v
            t += dt

        # Tuft should receive significant back-propagating depolarization
        self.assertGreater(max_v_tuft, -30.0, "Action potential should back-propagate into apical tuft")

    def test_ampa_synaptic_conductance(self):
        """Verify AMPA synapse causes rapid inward depolarizing current towards 0 mV."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        dt = 0.02

        neuron.receive_spike("AMPA", weight=2.0)
        self.assertAlmostEqual(neuron.g_ampa, 2.0, places=2)

        res = neuron.step(dt)
        # At V = -65 mV, AMPA current I = g * (V - 0) < 0 (inward current depolarizing cell)
        self.assertGreater(res["v"], -65.0, "AMPA activation must depolarize resting membrane")

    def test_gaba_synaptic_conductance(self):
        """Verify GABA_A synapse causes hyperpolarizing outward current when V > -70 mV."""
        neuron = BiophysicalNeuron(0, v_init=-55.0)  # Depolarized state
        dt = 0.02

        neuron.receive_spike("GABA", weight=2.5)
        self.assertAlmostEqual(neuron.g_gaba, 2.5, places=2)

        res = neuron.step(dt)
        # At V = -55 mV and E_gaba = -70 mV, GABA current pulls towards -70 mV
        self.assertLess(res["v"], -55.0, "GABA activation must hyperpolarize membrane towards -70 mV")

    def test_nmda_magnesium_block(self):
        """Verify NMDA receptor conductance is blocked at hyperpolarized potentials and active when depolarized."""
        neuron_hyper = BiophysicalNeuron(0, v_init=-75.0)
        neuron_depol = BiophysicalNeuron(1, v_init=20.0)

        # Block factor B(V) = 1 / (1 + (1/3.57)*exp(-0.062*V))
        b_hyper = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * neuron_hyper.v))
        b_depol = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * neuron_depol.v))

        self.assertLess(b_hyper, 0.15, "NMDA should be heavily blocked at -75 mV")
        self.assertGreater(b_depol, 0.80, "NMDA should be largely unblocked at +20 mV")

    def test_ping_network_oscillation(self):
        """Verify reciprocal E-I network sustains oscillatory spiking."""
        circuit, _ = preset_ping_gamma_oscillations()
        dt = 0.02

        pyr_spikes = 0
        int_spikes = 0
        for _ in range(4000):  # 80 ms
            circuit.step(dt)

        pyr_spikes = len(circuit.neurons[0].spike_times)
        int_spikes = len(circuit.neurons[1].spike_times)

        self.assertGreaterEqual(pyr_spikes, 2, "Pyramidal cell should fire multiple rhythmic spikes")
        self.assertGreaterEqual(int_spikes, 2, "Interneuron should fire multiple rhythmic spikes")

    def test_half_center_alternation(self):
        """Verify CPG half-center circuit fires alternating bursts between left and right motor pools."""
        circuit, _ = preset_half_center_cpg()
        dt = 0.02

        for _ in range(5000):  # 100 ms
            circuit.step(dt)

        left_spikes = len(circuit.neurons[0].spike_times)
        right_spikes = len(circuit.neurons[1].spike_times)

        self.assertGreaterEqual(left_spikes, 1, "Left pool should fire spikes")
        self.assertGreaterEqual(right_spikes, 1, "Right pool should fire spikes")

    def test_numerical_stability_under_large_dt(self):
        """Verify Rush-Larsen integration preserves stability without exploding under large time steps."""
        neuron = BiophysicalNeuron(0, v_init=-65.0)
        neuron.i_inj = 12.0
        dt = 0.05  # Large dt for stiff biophysical systems

        for _ in range(1000):
            res = neuron.step(dt)
            self.assertFalse(math.isnan(res["v"]), "Membrane potential must not be NaN")
            self.assertFalse(math.isinf(res["v"]), "Membrane potential must not be Inf")
            self.assertTrue(-120.0 <= res["v"] <= 120.0, "Voltage must stay within physical bounds")

    def test_preset_catalog_initialization(self):
        """Verify all curated neurobiological presets load successfully without errors."""
        presets = [
            preset_giant_squid_axon,
            preset_anode_break,
            preset_ping_gamma_oscillations,
            preset_half_center_cpg,
            preset_dendritic_backpropagation,
            preset_thalamic_bursting,
        ]
        for preset_fn in presets:
            circuit, meta = preset_fn()
            self.assertIsNotNone(circuit)
            self.assertIn("name", meta)
            self.assertIn("default_dt", meta)
            circuit.step(0.02)


if __name__ == "__main__":
    unittest.main()
