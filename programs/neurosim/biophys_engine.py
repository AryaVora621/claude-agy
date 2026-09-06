"""
NeuroSim Biophysical Electrophysiology & Neural Circuit Engine.
First-principles numerical simulation of excitable neuronal membranes,
multi-compartment cable theory, and synaptic receptor networks.
Standard library Python only: zero external dependencies.
"""

import math
from typing import List, Dict, Tuple, Optional


class BiophysicalNeuron:
    """
    Biophysical neuron compartment implementing the Hodgkin-Huxley (1952)
    conductance-based model with optional low-threshold T-type calcium channels.
    """

    def __init__(
        self,
        neuron_id: int,
        label: str = "Soma",
        c_m: float = 1.0,
        g_na_bar: float = 120.0,
        e_na: float = 50.0,
        g_k_bar: float = 36.0,
        e_k: float = -77.0,
        g_l: float = 0.3,
        e_l: float = -54.387,
        g_t_bar: float = 0.0,
        e_ca: float = 120.0,
        v_init: float = -65.0,
    ):
        self.neuron_id = neuron_id
        self.label = label
        self.c_m = c_m
        self.g_na_bar = g_na_bar
        self.e_na = e_na
        self.g_k_bar = g_k_bar
        self.e_k = e_k
        self.g_l = g_l
        self.e_l = e_l
        self.g_t_bar = g_t_bar
        self.e_ca = e_ca

        # State variables
        self.v = v_init
        self.v_prev = v_init
        self.i_inj = 0.0

        # Initialize gating variables to steady-state at v_init
        self.m, self.h, self.n = self._steady_state_gates(v_init)
        self.m_t, self.h_t = self._steady_state_t_gates(v_init)

        # Synaptic conductances (mS/cm^2)
        self.g_ampa = 0.0
        self.g_nmda = 0.0
        self.g_gaba = 0.0

        # Synaptic reversal potentials (mV)
        self.e_ampa = 0.0
        self.e_nmda = 0.0
        self.e_gaba = -70.0

        # Synaptic decay time constants (ms)
        self.tau_ampa = 2.5
        self.tau_nmda = 60.0
        self.tau_gaba = 7.0

        # Spike detection
        self.spike_threshold = 0.0
        self.is_spiking = False
        self.spike_times: List[float] = []
        self.last_spike_time = -1e9

    def _alpha_m(self, v: float) -> float:
        x = v + 40.0
        if abs(x) < 1e-5:
            return 1.0
        return (0.1 * x) / (1.0 - math.exp(-x / 10.0))

    def _beta_m(self, v: float) -> float:
        return 4.0 * math.exp(-(v + 65.0) / 18.0)

    def _alpha_h(self, v: float) -> float:
        return 0.07 * math.exp(-(v + 65.0) / 20.0)

    def _beta_h(self, v: float) -> float:
        return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))

    def _alpha_n(self, v: float) -> float:
        x = v + 55.0
        if abs(x) < 1e-5:
            return 0.1
        return (0.01 * x) / (1.0 - math.exp(-x / 10.0))

    def _beta_n(self, v: float) -> float:
        return 0.125 * math.exp(-(v + 65.0) / 80.0)

    def _steady_state_gates(self, v: float) -> Tuple[float, float, float]:
        a_m = self._alpha_m(v)
        b_m = self._beta_m(v)
        m_inf = a_m / (a_m + b_m)

        a_h = self._alpha_h(v)
        b_h = self._beta_h(v)
        h_inf = a_h / (a_h + b_h)

        a_n = self._alpha_n(v)
        b_n = self._beta_n(v)
        n_inf = a_n / (a_n + b_n)

        return m_inf, h_inf, n_inf

    def _steady_state_t_gates(self, v: float) -> Tuple[float, float]:
        m_inf = 1.0 / (1.0 + math.exp(-(v + 57.0) / 6.2))
        h_inf = 1.0 / (1.0 + math.exp((v + 81.0) / 4.0))
        return m_inf, h_inf

    def step(self, dt: float, i_axial: float = 0.0, sim_time: float = 0.0) -> Dict[str, float]:
        """
        Advance biophysical state by time step dt (ms) using Rush-Larsen exponential
        integration for gating variables and Runge-Kutta / Euler for membrane potential.
        """
        v_curr = self.v
        self.v_prev = v_curr

        # 1. Update Hodgkin-Huxley gating variables via Rush-Larsen exponential rule
        a_m = self._alpha_m(v_curr)
        b_m = self._beta_m(v_curr)
        tau_m = 1.0 / (a_m + b_m)
        m_inf = a_m * tau_m
        self.m = m_inf + (self.m - m_inf) * math.exp(-dt / tau_m)

        a_h = self._alpha_h(v_curr)
        b_h = self._beta_h(v_curr)
        tau_h = 1.0 / (a_h + b_h)
        h_inf = a_h * tau_h
        self.h = h_inf + (self.h - h_inf) * math.exp(-dt / tau_h)

        a_n = self._alpha_n(v_curr)
        b_n = self._beta_n(v_curr)
        tau_n = 1.0 / (a_n + b_n)
        n_inf = a_n * tau_n
        self.n = n_inf + (self.n - n_inf) * math.exp(-dt / tau_n)

        # 2. Update T-type calcium channels if enabled
        if self.g_t_bar > 0.001:
            m_t_inf = 1.0 / (1.0 + math.exp(-(v_curr + 57.0) / 6.2))
            tau_mt = 0.6 + 1.0 / (math.exp((v_curr + 27.0) / 10.0) + math.exp(-(v_curr + 102.0) / 15.0))
            self.m_t = m_t_inf + (self.m_t - m_t_inf) * math.exp(-dt / tau_mt)

            h_t_inf = 1.0 / (1.0 + math.exp((v_curr + 81.0) / 4.0))
            tau_ht = 22.7 + 0.27 / (math.exp((v_curr + 48.0) / 4.0) + math.exp(-(v_curr + 407.0) / 50.0))
            self.h_t = h_t_inf + (self.h_t - h_t_inf) * math.exp(-dt / tau_ht)

        # 3. Calculate intrinsic ionic currents (microA / cm^2)
        i_na = self.g_na_bar * (self.m ** 3) * self.h * (v_curr - self.e_na)
        i_k = self.g_k_bar * (self.n ** 4) * (v_curr - self.e_k)
        i_l = self.g_l * (v_curr - self.e_l)

        i_t = 0.0
        if self.g_t_bar > 0.001:
            i_t = self.g_t_bar * (self.m_t ** 2) * self.h_t * (v_curr - self.e_ca)

        # 4. Calculate synaptic currents
        i_ampa = self.g_ampa * (v_curr - self.e_ampa)

        # Voltage-dependent magnesium block for NMDA receptors (Jahr & Stevens 1990)
        b_mg = 1.0 / (1.0 + (1.0 / 3.57) * math.exp(-0.062 * v_curr))
        i_nmda = self.g_nmda * b_mg * (v_curr - self.e_nmda)

        i_gaba = self.g_gaba * (v_curr - self.e_gaba)
        i_syn = i_ampa + i_nmda + i_gaba

        # 5. Membrane potential integration: C_m dV/dt = I_inj - I_ionic - I_syn + I_axial
        dv_dt = (self.i_inj - i_na - i_k - i_l - i_t - i_syn + i_axial) / self.c_m
        self.v += dv_dt * dt

        # Prevent numerical divergence under non-physiological extreme inputs
        if self.v > 120.0:
            self.v = 120.0
        elif self.v < -120.0:
            self.v = -120.0

        # 6. Exponential decay of synaptic conductances
        self.g_ampa *= math.exp(-dt / self.tau_ampa)
        self.g_nmda *= math.exp(-dt / self.tau_nmda)
        self.g_gaba *= math.exp(-dt / self.tau_gaba)

        # 7. Action potential spike detection
        just_spiked = False
        if not self.is_spiking and self.v >= self.spike_threshold and self.v_prev < self.spike_threshold:
            self.is_spiking = True
            just_spiked = True
            self.spike_times.append(sim_time)
            self.last_spike_time = sim_time
        elif self.is_spiking and self.v < self.spike_threshold - 10.0:
            self.is_spiking = False

        return {
            "v": self.v,
            "m": self.m,
            "h": self.h,
            "n": self.n,
            "i_na": i_na,
            "i_k": i_k,
            "i_l": i_l,
            "i_t": i_t,
            "i_syn": i_syn,
            "just_spiked": 1.0 if just_spiked else 0.0,
        }

    def receive_spike(self, synapse_type: str, weight: float):
        """Deliver synaptic transmission packet upon presynaptic action potential."""
        if synapse_type == "AMPA":
            self.g_ampa += weight
        elif synapse_type == "NMDA":
            self.g_nmda += weight
        elif synapse_type == "GABA":
            self.g_gaba += weight


class Synapse:
    """Directed chemical synapse connecting presynaptic and postsynaptic neurons."""

    def __init__(
        self,
        pre_id: int,
        post_id: int,
        syn_type: str = "AMPA",
        weight: float = 0.5,
        delay_ms: float = 1.0,
    ):
        self.pre_id = pre_id
        self.post_id = post_id
        self.syn_type = syn_type
        self.weight = weight
        self.delay_ms = delay_ms
        self.pending_spikes: List[float] = []

    def queue_spike(self, sim_time: float):
        """Schedule synaptic arrival at sim_time + delay_ms."""
        self.pending_spikes.append(sim_time + self.delay_ms)

    def check_deliveries(self, sim_time: float) -> List[float]:
        """Return and clear spikes whose arrival time has passed."""
        delivered = [t for t in self.pending_spikes if sim_time >= t]
        self.pending_spikes = [t for t in self.pending_spikes if sim_time < t]
        return delivered


class MultiCompartmentNeuron:
    """
    Multi-compartment dendritic tree cable model (Rall 1959).
    Includes Soma + Basal Dendrite + Apical Trunk + Apical Tuft compartments
    coupled via axial resistance R_a.
    """

    def __init__(self, g_axial: float = 0.20, passive_dendrites: bool = False):
        self.g_axial = g_axial
        if passive_dendrites:
            self.compartments: List[BiophysicalNeuron] = [
                BiophysicalNeuron(0, label="Soma", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3, e_l=-54.387),
                BiophysicalNeuron(1, label="Basal Dendrite", g_na_bar=0.0, g_k_bar=0.0, g_l=0.15, e_l=-65.0),
                BiophysicalNeuron(2, label="Apical Trunk", g_na_bar=0.0, g_k_bar=0.0, g_l=0.15, e_l=-65.0),
                BiophysicalNeuron(3, label="Apical Tuft", g_na_bar=0.0, g_k_bar=0.0, g_l=0.15, e_l=-65.0),
            ]
        else:
            self.compartments: List[BiophysicalNeuron] = [
                BiophysicalNeuron(0, label="Soma", g_na_bar=120.0, g_k_bar=36.0, g_l=0.3, e_l=-54.387),
                BiophysicalNeuron(1, label="Basal Dendrite", g_na_bar=20.0, g_k_bar=10.0, g_l=0.3, e_l=-54.387),
                BiophysicalNeuron(2, label="Apical Trunk", g_na_bar=40.0, g_k_bar=15.0, g_l=0.3, e_l=-54.387),
                BiophysicalNeuron(3, label="Apical Tuft", g_na_bar=20.0, g_k_bar=10.0, g_l=0.3, e_l=-54.387),
            ]
        # Adjacency: Soma(0) connected to Basal(1) and Apical Trunk(2); Trunk(2) to Tuft(3)
        self.topology = [
            (0, 1),
            (0, 2),
            (2, 3),
        ]

    def step(self, dt: float, sim_time: float) -> Dict[str, float]:
        # Compute axial currents between connected compartments
        n_comp = len(self.compartments)
        i_axials = [0.0] * n_comp

        for c1_idx, c2_idx in self.topology:
            v1 = self.compartments[c1_idx].v
            v2 = self.compartments[c2_idx].v
            # Current flows from high to low potential: I_axial = g_axial * (V_neighbor - V_here)
            i_12 = self.g_axial * (v2 - v1)
            i_21 = self.g_axial * (v1 - v2)
            i_axials[c1_idx] += i_12
            i_axials[c2_idx] += i_21

        results = {}
        for idx, comp in enumerate(self.compartments):
            res = comp.step(dt, i_axial=i_axials[idx], sim_time=sim_time)
            results[f"v_{comp.label}"] = res["v"]

        return results


class NeuralCircuit:
    """
    Biophysical neural network circuit simulator managing populations of
    Hodgkin-Huxley neurons, chemical synapses, and multi-compartment cells.
    """

    def __init__(self):
        self.neurons: Dict[int, BiophysicalNeuron] = {}
        self.synapses: List[Synapse] = []
        self.multi_compartment: Optional[MultiCompartmentNeuron] = None
        self.sim_time = 0.0
        self.history_time: List[float] = []
        self.history_v: Dict[int, List[float]] = {}
        self.max_history = 2000

    def add_neuron(self, neuron: BiophysicalNeuron):
        self.neurons[neuron.neuron_id] = neuron
        self.history_v[neuron.neuron_id] = []

    def add_synapse(self, synapse: Synapse):
        self.synapses.append(synapse)

    def set_multi_compartment(self, mc: MultiCompartmentNeuron):
        self.multi_compartment = mc

    def step(self, dt: float):
        """Advance the whole network circuit by dt (ms)."""
        self.sim_time += dt

        # 1. Step individual neurons and check for spikes
        new_spikes = []
        for n_id, neuron in self.neurons.items():
            res = neuron.step(dt, sim_time=self.sim_time)
            if res["just_spiked"] > 0.5:
                new_spikes.append(n_id)

            # Record history
            hist = self.history_v[n_id]
            hist.append(res["v"])
            if len(hist) > self.max_history:
                hist.pop(0)

        # 2. Step multi-compartment cell if present
        if self.multi_compartment:
            self.multi_compartment.step(dt, self.sim_time)

        # 3. Queue new spikes in outgoing synapses
        for syn in self.synapses:
            if syn.pre_id in new_spikes:
                syn.queue_spike(self.sim_time)

        # 4. Deliver mature synaptic transmissions
        for syn in self.synapses:
            delivered = syn.check_deliveries(self.sim_time)
            if delivered and syn.post_id in self.neurons:
                post_neuron = self.neurons[syn.post_id]
                for _ in delivered:
                    post_neuron.receive_spike(syn.syn_type, syn.weight)

        # Record timeline
        self.history_time.append(self.sim_time)
        if len(self.history_time) > self.max_history:
            self.history_time.pop(0)

    def reset(self):
        """Reset network simulation time and state."""
        self.sim_time = 0.0
        self.history_time.clear()
        for n_id in self.history_v:
            self.history_v[n_id].clear()
        for syn in self.synapses:
            syn.pending_spikes.clear()
