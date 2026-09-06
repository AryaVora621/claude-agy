"""Biophysical and phenomenological spiking neuron models.

Implements Leaky Integrate-and-Fire (LIF) with adaptive threshold,
the Izhikevich 2D dynamical system with 8 cortical firing presets,
and the classic Hodgkin-Huxley 4-variable conductance model integrated via RK4.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SpikeEvent:
    """Record of a single action potential event."""
    neuron_id: int
    time_ms: float
    voltage: float = 0.0


class SpikingNeuron(ABC):
    """Abstract base class for spiking neurons."""

    def __init__(
        self,
        neuron_id: int = 0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -55.0,
    ) -> None:
        self.neuron_id = neuron_id
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.v = v_rest
        self.last_spike_time_ms: float = -1e9
        self.spike_count: int = 0

    @abstractmethod
    def step(self, dt_ms: float, i_inj: float, current_time_ms: float = 0.0) -> bool:
        """Advance the neuron simulation by dt_ms given injected current i_inj.

        Returns True if an action potential (spike) was triggered.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the neuron to its initial baseline resting state."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, float]:
        """Return a dictionary of the neuron's current state variables."""
        pass


class LIFNeuron(SpikingNeuron):
    """Leaky Integrate-and-Fire (LIF) neuron with adaptive threshold.

    Dynamics:
        tau_m * dV/dt = -(V - V_rest) + R * I_inj
        V_th(t) = V_th0 + theta(t)
        tau_theta * dtheta/dt = -theta

    When V >= V_th:
        - Spike is emitted
        - V is clamped to V_reset for refractory period tau_ref
        - theta is incremented by delta_theta (homeostatic adaptation)
    """

    def __init__(
        self,
        neuron_id: int = 0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        v_thresh: float = -55.0,
        tau_m_ms: float = 20.0,
        r_membrane: float = 10.0,  # Mega-ohms
        tau_ref_ms: float = 2.0,
        adaptive_threshold: bool = False,
        tau_theta_ms: float = 50.0,
        delta_theta_mv: float = 5.0,
    ) -> None:
        super().__init__(neuron_id, v_rest, v_reset, v_thresh)
        self.tau_m_ms = tau_m_ms
        self.r_membrane = r_membrane
        self.tau_ref_ms = tau_ref_ms
        self.adaptive_threshold = adaptive_threshold
        self.tau_theta_ms = tau_theta_ms
        self.delta_theta_mv = delta_theta_mv

        self.v_thresh_base = v_thresh
        self.theta_mv: float = 0.0
        self.refractory_remaining_ms: float = 0.0

    def step(self, dt_ms: float, i_inj: float, current_time_ms: float = 0.0) -> bool:
        # Handle adaptive threshold decay
        if self.adaptive_threshold:
            decay_theta = math.exp(-dt_ms / self.tau_theta_ms)
            self.theta_mv *= decay_theta
            self.v_thresh = self.v_thresh_base + self.theta_mv
        else:
            self.v_thresh = self.v_thresh_base

        # Check refractory period
        if self.refractory_remaining_ms > 0.0:
            self.refractory_remaining_ms -= dt_ms
            self.v = self.v_reset
            return False

        # Analytical exponential decay with steady-state asymptotic target
        # V_target = V_rest + R * I_inj
        v_inf = self.v_rest + self.r_membrane * i_inj
        decay = math.exp(-dt_ms / self.tau_m_ms)
        self.v = v_inf + (self.v - v_inf) * decay

        # Threshold check
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.refractory_remaining_ms = self.tau_ref_ms
            self.last_spike_time_ms = current_time_ms
            self.spike_count += 1
            if self.adaptive_threshold:
                self.theta_mv += self.delta_theta_mv
                self.v_thresh = self.v_thresh_base + self.theta_mv
            return True

        return False

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta_mv = 0.0
        self.v_thresh = self.v_thresh_base
        self.refractory_remaining_ms = 0.0
        self.last_spike_time_ms = -1e9
        self.spike_count = 0

    def get_state(self) -> Dict[str, float]:
        return {
            "v": self.v,
            "v_thresh": self.v_thresh,
            "theta": self.theta_mv,
            "refractory": self.refractory_remaining_ms,
        }


class IzhikevichNeuron(SpikingNeuron):
    """Izhikevich 2D phenomenological dynamical spiking neuron model.

    Differential Equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

    Reset condition:
        if v >= 30 mV:
            v <- c
            u <- u + d

    Parameters:
        a: time scale of the recovery variable u
        b: sensitivity of u to the subthreshold fluctuations of v
        c: after-spike reset value of v
        d: after-spike reset increment of u
    """

    def __init__(
        self,
        neuron_id: int = 0,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_init: Optional[float] = None,
        u_init: Optional[float] = None,
        peak_mv: float = 30.0,
    ) -> None:
        v_start = v_init if v_init is not None else c
        super().__init__(neuron_id, v_rest=v_start, v_reset=c, v_thresh=peak_mv)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.peak_mv = peak_mv

        self.v = v_start
        self.u = u_init if u_init is not None else b * self.v
        self.is_spiking_now: bool = False

    def step(self, dt_ms: float, i_inj: float, current_time_ms: float = 0.0) -> bool:
        # Check if reset from previous step's spike is needed
        if self.is_spiking_now:
            self.v = self.c
            self.u += self.d
            self.is_spiking_now = False

        # Two half-step sub-integrations for numerical stability (Izhikevich recommended)
        half_dt = dt_ms * 0.5

        # Sub-step 1
        dv1 = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i_inj) * half_dt
        self.v += dv1
        dv2 = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i_inj) * half_dt
        self.v += dv2

        # Recovery variable integration
        du = self.a * (self.b * self.v - self.u) * dt_ms
        self.u += du

        # Peak detection
        if self.v >= self.peak_mv:
            self.is_spiking_now = True
            self.last_spike_time_ms = current_time_ms
            self.spike_count += 1
            # Artificially clamp to peak for visual recording on this step
            self.v = self.peak_mv
            return True

        return False

    def reset(self) -> None:
        self.v = self.c
        self.u = self.b * self.v
        self.is_spiking_now = False
        self.last_spike_time_ms = -1e9
        self.spike_count = 0

    def get_state(self) -> Dict[str, float]:
        return {
            "v": self.v,
            "u": self.u,
            "spiking": 1.0 if self.is_spiking_now else 0.0,
        }

    @classmethod
    def regular_spiking(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Cortical excitatory regular spiking (RS) pyramidal neuron."""
        return cls(neuron_id=neuron_id, a=0.02, b=0.2, c=-65.0, d=8.0)

    @classmethod
    def intrinsically_bursting(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Intrinsically bursting (IB) layer 5 cortical neuron."""
        return cls(neuron_id=neuron_id, a=0.02, b=0.2, c=-55.0, d=4.0)

    @classmethod
    def chattering(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Chattering (CH) cortical interneuron firing high-frequency bursts."""
        return cls(neuron_id=neuron_id, a=0.02, b=0.2, c=-50.0, d=2.0)

    @classmethod
    def fast_spiking(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Inhibitory fast spiking (FS) basket / chandelier interneuron."""
        return cls(neuron_id=neuron_id, a=0.1, b=0.2, c=-65.0, d=2.0)

    @classmethod
    def low_threshold_spiking(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Low-threshold spiking (LTS) interneuron."""
        return cls(neuron_id=neuron_id, a=0.02, b=0.25, c=-65.0, d=2.0)

    @classmethod
    def thalamocortical(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Thalamocortical (TC) relay neuron with tonic and burst modes."""
        return cls(neuron_id=neuron_id, a=0.02, b=0.25, c=-65.0, d=0.05)

    @classmethod
    def resonator(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Resonator (RZ) neuron exhibiting subthreshold oscillatory resonance."""
        return cls(neuron_id=neuron_id, a=0.1, b=0.26, c=-65.0, d=2.0)

    @classmethod
    def accommodating(cls, neuron_id: int = 0) -> IzhikevichNeuron:
        """Accommodating (ACC) neuron showing strong spike-frequency adaptation."""
        return cls(neuron_id=neuron_id, a=0.02, b=1.0, c=-55.0, d=4.0)


class HodgkinHuxleyNeuron(SpikingNeuron):
    """Hodgkin-Huxley biophysical 4-variable conductance model (1952).

    Equations:
        C_m * dV/dt = I_inj - I_Na - I_K - I_L
        I_Na = g_Na_max * m^3 * h * (V - E_Na)
        I_K  = g_K_max  * n^4 * (V - E_K)
        I_L  = g_L      * (V - E_L)

    Gating Kinetics (dm/dt, dh/dt, dn/dt):
        dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x

    Integrated using 4th-Order Runge-Kutta (RK4) for maximum numerical fidelity.
    """

    def __init__(
        self,
        neuron_id: int = 0,
        c_m: float = 1.0,           # micro-Farad / cm^2
        g_na: float = 120.0,        # milli-Siemens / cm^2
        e_na: float = 50.0,         # mV
        g_k: float = 36.0,          # milli-Siemens / cm^2
        e_k: float = -77.0,         # mV
        g_l: float = 0.3,           # milli-Siemens / cm^2
        e_l: float = -54.387,       # mV
        v_init: float = -65.0,      # mV
    ) -> None:
        super().__init__(neuron_id, v_rest=v_init, v_reset=-65.0, v_thresh=0.0)
        self.c_m = c_m
        self.g_na = g_na
        self.e_na = e_na
        self.g_k = g_k
        self.e_k = e_k
        self.g_l = g_l
        self.e_l = e_l
        self.v_init = v_init

        self.v = v_init
        self.m = self._alpha_m(self.v) / (self._alpha_m(self.v) + self._beta_m(self.v))
        self.h = self._alpha_h(self.v) / (self._alpha_h(self.v) + self._beta_h(self.v))
        self.n = self._alpha_n(self.v) / (self._alpha_n(self.v) + self._beta_n(self.v))

        self.prev_v: float = v_init
        self.in_spike: bool = False

    @staticmethod
    def _alpha_m(v: float) -> float:
        # Avoid division by zero at V = -40 mV
        diff = v + 40.0
        if abs(diff) < 1e-6:
            return 1.0
        return 0.1 * diff / (1.0 - math.exp(-diff / 10.0))

    @staticmethod
    def _beta_m(v: float) -> float:
        return 4.0 * math.exp(-(v + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(v: float) -> float:
        return 0.07 * math.exp(-(v + 65.0) / 20.0)

    @staticmethod
    def _beta_h(v: float) -> float:
        return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(v: float) -> float:
        # Avoid division by zero at V = -55 mV
        diff = v + 55.0
        if abs(diff) < 1e-6:
            return 0.1
        return 0.01 * diff / (1.0 - math.exp(-diff / 10.0))

    @staticmethod
    def _beta_n(v: float) -> float:
        return 0.125 * math.exp(-(v + 65.0) / 80.0)

    def _derivatives(
        self, v: float, m: float, h: float, n: float, i_inj: float
    ) -> Tuple[float, float, float, float]:
        """Compute time derivatives (dv/dt, dm/dt, dh/dt, dn/dt)."""
        i_na = self.g_na * (m ** 3) * h * (v - self.e_na)
        i_k = self.g_k * (n ** 4) * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        dv_dt = (i_inj - i_na - i_k - i_l) / self.c_m
        dm_dt = self._alpha_m(v) * (1.0 - m) - self._beta_m(v) * m
        dh_dt = self._alpha_h(v) * (1.0 - h) - self._beta_h(v) * h
        dn_dt = self._alpha_n(v) * (1.0 - n) - self._beta_n(v) * n

        return dv_dt, dm_dt, dh_dt, dn_dt

    def step(self, dt_ms: float, i_inj: float, current_time_ms: float = 0.0) -> bool:
        # 4th-Order Runge-Kutta (RK4) integration
        v0, m0, h0, n0 = self.v, self.m, self.h, self.n

        k1_v, k1_m, k1_h, k1_n = self._derivatives(v0, m0, h0, n0, i_inj)

        v_mid1 = v0 + 0.5 * dt_ms * k1_v
        m_mid1 = m0 + 0.5 * dt_ms * k1_m
        h_mid1 = h0 + 0.5 * dt_ms * k1_h
        n_mid1 = n0 + 0.5 * dt_ms * k1_n
        k2_v, k2_m, k2_h, k2_n = self._derivatives(v_mid1, m_mid1, h_mid1, n_mid1, i_inj)

        v_mid2 = v0 + 0.5 * dt_ms * k2_v
        m_mid2 = m0 + 0.5 * dt_ms * k2_m
        h_mid2 = h0 + 0.5 * dt_ms * k2_h
        n_mid2 = n0 + 0.5 * dt_ms * k2_n
        k3_v, k3_m, k3_h, k3_n = self._derivatives(v_mid2, m_mid2, h_mid2, n_mid2, i_inj)

        v_end = v0 + dt_ms * k3_v
        m_end = m0 + dt_ms * k3_m
        h_end = h0 + dt_ms * k3_h
        n_end = n0 + dt_ms * k3_n
        k4_v, k4_m, k4_h, k4_n = self._derivatives(v_end, m_end, h_end, n_end, i_inj)

        self.prev_v = self.v
        self.v += (dt_ms / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        self.m += (dt_ms / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
        self.h += (dt_ms / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
        self.n += (dt_ms / 6.0) * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)

        # Boundary clamping for gating probabilities in [0, 1]
        self.m = max(0.0, min(1.0, self.m))
        self.h = max(0.0, min(1.0, self.h))
        self.n = max(0.0, min(1.0, self.n))

        # Upward zero-crossing detection for action potential firing
        spike_detected = False
        if not self.in_spike and self.prev_v < self.v_thresh and self.v >= self.v_thresh:
            spike_detected = True
            self.in_spike = True
            self.last_spike_time_ms = current_time_ms
            self.spike_count += 1
        elif self.in_spike and self.v < self.v_thresh - 10.0:
            self.in_spike = False

        return spike_detected

    def reset(self) -> None:
        self.v = self.v_init
        self.m = self._alpha_m(self.v) / (self._alpha_m(self.v) + self._beta_m(self.v))
        self.h = self._alpha_h(self.v) / (self._alpha_h(self.v) + self._beta_h(self.v))
        self.n = self._alpha_n(self.v) / (self._alpha_n(self.v) + self._beta_n(self.v))
        self.prev_v = self.v_init
        self.in_spike = False
        self.last_spike_time_ms = -1e9
        self.spike_count = 0

    def get_state(self) -> Dict[str, float]:
        return {
            "v": self.v,
            "m": self.m,
            "h": self.h,
            "n": self.n,
            "in_spike": 1.0 if self.in_spike else 0.0,
        }


class NeuronPopulation:
    """A collection of spiking neurons simulated in lockstep."""

    def __init__(self, neurons: List[SpikingNeuron]) -> None:
        self.neurons = neurons
        self.size = len(neurons)

    def step(
        self, dt_ms: float, currents: List[float], current_time_ms: float = 0.0
    ) -> List[bool]:
        """Step all neurons with the corresponding input currents.

        Returns a boolean list of spike flags for each neuron.
        """
        if len(currents) != self.size:
            raise ValueError(f"Expected {self.size} currents, received {len(currents)}")

        spikes = []
        for neuron, current in zip(self.neurons, currents):
            spikes.append(neuron.step(dt_ms, current, current_time_ms))
        return spikes

    def reset(self) -> None:
        """Reset all neurons in the population."""
        for neuron in self.neurons:
            neuron.reset()

    @property
    def voltages(self) -> List[float]:
        """Return current membrane voltages of all neurons."""
        return [neuron.v for neuron in self.neurons]

    @property
    def total_spikes(self) -> int:
        """Return cumulative spike count across all neurons."""
        return sum(neuron.spike_count for neuron in self.neurons)

    @classmethod
    def create_lif_population(
        cls,
        size: int,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        v_thresh: float = -55.0,
        tau_m_ms: float = 20.0,
        adaptive_threshold: bool = False,
    ) -> NeuronPopulation:
        neurons = [
            LIFNeuron(
                neuron_id=i,
                v_rest=v_rest,
                v_reset=v_reset,
                v_thresh=v_thresh,
                tau_m_ms=tau_m_ms,
                adaptive_threshold=adaptive_threshold,
            )
            for i in range(size)
        ]
        return cls(neurons)

    @classmethod
    def create_izhikevich_population(
        cls, size: int, preset: str = "regular_spiking"
    ) -> NeuronPopulation:
        neurons = []
        for i in range(size):
            if preset == "regular_spiking":
                n = IzhikevichNeuron.regular_spiking(neuron_id=i)
            elif preset == "fast_spiking":
                n = IzhikevichNeuron.fast_spiking(neuron_id=i)
            elif preset == "chattering":
                n = IzhikevichNeuron.chattering(neuron_id=i)
            elif preset == "intrinsically_bursting":
                n = IzhikevichNeuron.intrinsically_bursting(neuron_id=i)
            else:
                n = IzhikevichNeuron(neuron_id=i)
            neurons.append(n)
        return cls(neurons)
