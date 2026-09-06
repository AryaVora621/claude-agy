"""Unit tests for synaptic dynamics, post-synaptic currents, and STDP learning rules."""

import unittest
from neurosynapse.synapses import (
    STDPConfig,
    Synapse,
    SynapseType,
    SynapticMatrix,
    TripletSTDPConfig,
    TripletSynapse,
)


class TestSynapsesSTDP(unittest.TestCase):
    """Test suite for post-synaptic currents and STDP plasticity."""

    def test_exponential_psc_kinetics(self) -> None:
        """Verify exponential rise and decay of post-synaptic currents."""
        syn = Synapse(pre_id=0, post_id=1, weight=1.0, delay_ms=0.0, tau_syn_ms=10.0, synapse_type=SynapseType.EXPONENTIAL)
        # Pre spike at t=0
        syn.on_pre_spike(0.0)
        i_now = syn.step(dt_ms=1.0, current_time_ms=0.0)
        self.assertAlmostEqual(i_now, 1.0, places=3)

        # Decay over 10 ms
        i_later = syn.step(dt_ms=10.0, current_time_ms=10.0)
        self.assertLess(i_later, i_now)
        self.assertAlmostEqual(i_later, i_now * 0.3678, delta=0.05)

    def test_alpha_psc_kinetics(self) -> None:
        """Verify alpha function post-synaptic current kinetics."""
        syn = Synapse(pre_id=0, post_id=1, weight=1.0, delay_ms=0.0, tau_syn_ms=5.0, synapse_type=SynapseType.ALPHA)
        syn.on_pre_spike(0.0)
        trace = []
        for t in range(20):
            trace.append(syn.step(dt_ms=1.0, current_time_ms=float(t)))
        # Alpha profile rises then falls
        peak_idx = trace.index(max(trace))
        self.assertGreater(peak_idx, 0)
        self.assertLess(peak_idx, 15)

    def test_pair_stdp_ltp(self) -> None:
        """Verify Long-Term Potentiation (LTP) when pre-spike precedes post-spike."""
        cfg = STDPConfig(a_plus=0.05, a_minus=0.05, tau_plus_ms=20.0, w_min=0.0, w_max=1.0)
        syn = Synapse(pre_id=0, post_id=1, weight=0.5, stdp_config=cfg)

        # Pre fires at t=0 ms
        syn.on_pre_spike(0.0)
        syn.step(dt_ms=5.0, current_time_ms=5.0)

        # Post fires at t=5 ms (Delta t = +5 ms > 0 -> LTP)
        syn.on_post_spike(5.0)
        self.assertGreater(syn.weight, 0.5)

    def test_pair_stdp_ltd(self) -> None:
        """Verify Long-Term Depression (LTD) when post-spike precedes pre-spike."""
        cfg = STDPConfig(a_plus=0.05, a_minus=0.05, tau_minus_ms=20.0, w_min=0.0, w_max=1.0)
        syn = Synapse(pre_id=0, post_id=1, weight=0.5, stdp_config=cfg)

        # Post fires at t=0 ms
        syn.on_post_spike(0.0)
        syn.step(dt_ms=5.0, current_time_ms=5.0)

        # Pre fires at t=5 ms (Delta t = -5 ms < 0 -> LTD)
        syn.on_pre_spike(5.0)
        self.assertLess(syn.weight, 0.5)

    def test_triplet_stdp(self) -> None:
        """Verify Triplet STDP rules for frequency-dependent plasticity."""
        cfg = TripletSTDPConfig(a2_plus=0.01, a3_plus=0.02, w_min=0.0, w_max=1.0)
        syn = TripletSynapse(pre_id=0, post_id=1, weight=0.4, config=cfg)

        # Triplet sequence: Pre -> Post -> Post
        syn.on_pre_spike()
        syn.step(dt_ms=2.0)
        syn.on_post_spike()
        syn.step(dt_ms=2.0)
        syn.on_post_spike()

        self.assertGreater(syn.weight, 0.4)

    def test_synaptic_matrix_homeostasis(self) -> None:
        """Verify homeostatic normalization scales weights proportionally."""
        mat = SynapticMatrix(num_pre=3, num_post=2, initial_weight=0.5)
        # Target column sum = 1.0
        mat.normalize_weights_homeostatic(target_sum=1.0)

        col0_sum = sum(mat.weights[i][0] for i in range(3))
        col1_sum = sum(mat.weights[i][1] for i in range(3))
        self.assertAlmostEqual(col0_sum, 1.0, places=5)
        self.assertAlmostEqual(col1_sum, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
