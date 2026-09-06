"""Unit tests for biophysical and phenomenological spiking neuron models."""

import unittest
from neurosynapse.neurons import (
    HodgkinHuxleyNeuron,
    IzhikevichNeuron,
    LIFNeuron,
    NeuronPopulation,
)


class TestNeuronModels(unittest.TestCase):
    """Test suite for LIF, Izhikevich, and Hodgkin-Huxley neuron dynamics."""

    def test_lif_subthreshold_integration(self) -> None:
        """Verify leaky integration towards steady state voltage."""
        neuron = LIFNeuron(v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0, tau_m_ms=20.0, r_membrane=10.0)
        # Without input current, membrane remains at v_rest
        for _ in range(20):
            spiked = neuron.step(dt_ms=1.0, i_inj=0.0)
            self.assertFalse(spiked)
        self.assertAlmostEqual(neuron.v, -70.0, places=2)

        # Inject subthreshold current (I = 1.0 nA -> V_inf = -70 + 10 = -60 mV < -55 mV)
        for _ in range(100):
            spiked = neuron.step(dt_ms=1.0, i_inj=1.0)
            self.assertFalse(spiked)
        self.assertAlmostEqual(neuron.v, -60.0, delta=0.5)

    def test_lif_action_potential_generation(self) -> None:
        """Verify action potential emission and refractory clamping."""
        neuron = LIFNeuron(v_rest=-70.0, v_reset=-75.0, v_thresh=-55.0, tau_m_ms=20.0, r_membrane=10.0, tau_ref_ms=3.0)
        # Inject suprathreshold current (I = 5.0 nA -> V_inf = -20 mV)
        spikes = []
        for t in range(50):
            spikes.append(neuron.step(dt_ms=1.0, i_inj=5.0, current_time_ms=float(t)))

        total_spikes = sum(1 for s in spikes if s)
        self.assertGreater(total_spikes, 2)
        self.assertEqual(neuron.spike_count, total_spikes)

    def test_lif_adaptive_threshold(self) -> None:
        """Verify adaptive threshold elevates upon firing and decays back to baseline."""
        neuron = LIFNeuron(
            v_thresh=-55.0,
            adaptive_threshold=True,
            tau_theta_ms=20.0,
            delta_theta_mv=8.0,
        )
        self.assertEqual(neuron.v_thresh, -55.0)

        # Force a spike
        neuron.v = -50.0
        spiked = neuron.step(dt_ms=1.0, i_inj=5.0)
        self.assertTrue(spiked)
        self.assertGreater(neuron.v_thresh, -55.0)

        # Let threshold decay without input
        for _ in range(100):
            neuron.step(dt_ms=1.0, i_inj=0.0)
        self.assertAlmostEqual(neuron.v_thresh, -55.0, delta=0.2)

    def test_izhikevich_regular_spiking(self) -> None:
        """Verify Izhikevich Regular Spiking (RS) neuron firing."""
        neuron = IzhikevichNeuron.regular_spiking()
        spikes = []
        for t in range(100):
            spikes.append(neuron.step(dt_ms=0.5, i_inj=10.0, current_time_ms=t * 0.5))

        total_spikes = sum(1 for s in spikes if s)
        self.assertGreater(total_spikes, 1)
        self.assertEqual(neuron.spike_count, total_spikes)

    def test_izhikevich_fast_spiking(self) -> None:
        """Verify Izhikevich Fast Spiking (FS) interneuron higher frequency."""
        rs = IzhikevichNeuron.regular_spiking()
        fs = IzhikevichNeuron.fast_spiking()

        for t in range(200):
            rs.step(dt_ms=0.5, i_inj=10.0, current_time_ms=t * 0.5)
            fs.step(dt_ms=0.5, i_inj=10.0, current_time_ms=t * 0.5)

        # Fast spiking interneurons fire at higher or comparable rates with less adaptation
        self.assertGreaterEqual(fs.spike_count, rs.spike_count)

    def test_hodgkin_huxley_action_potential(self) -> None:
        """Verify Hodgkin-Huxley 4-variable biophysical action potential."""
        hh = HodgkinHuxleyNeuron()
        self.assertAlmostEqual(hh.v, -65.0, places=1)

        # Inject suprathreshold pulse of 10 micro-A/cm^2
        spikes = []
        voltages = []
        for t in range(1000):  # 20 ms at dt=0.02 ms
            spiked = hh.step(dt_ms=0.02, i_inj=10.0, current_time_ms=t * 0.02)
            spikes.append(spiked)
            voltages.append(hh.v)

        total_spikes = sum(1 for s in spikes if s)
        self.assertGreater(total_spikes, 0)
        # Action potential overshoot should exceed +15 mV
        self.assertGreater(max(voltages), 15.0)

    def test_neuron_population(self) -> None:
        """Verify population batch stepping and voltage queries."""
        pop = NeuronPopulation.create_lif_population(size=4)
        currents = [0.0, 2.0, 5.0, 10.0]
        spikes = pop.step(dt_ms=1.0, currents=currents)
        self.assertEqual(len(spikes), 4)
        self.assertEqual(len(pop.voltages), 4)


if __name__ == "__main__":
    unittest.main()
