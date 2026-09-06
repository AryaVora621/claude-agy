"""Unit tests for network topologies, event queue, and Liquid State Machines."""

import unittest
from neurosynapse.network import (
    EventDrivenSimulator,
    LiquidStateMachine,
    SpikingNetwork,
)
from neurosynapse.neurons import LIFNeuron


class TestNetworkTopologies(unittest.TestCase):
    """Test suite for synchronous/asynchronous network execution and LSM reservoirs."""

    def test_synchronous_network_execution(self) -> None:
        """Verify synchronous multi-neuron stepping."""
        net = SpikingNetwork(dt_ms=1.0)
        net.add_neuron(LIFNeuron(neuron_id=0, v_thresh=-55.0))
        net.add_neuron(LIFNeuron(neuron_id=1, v_thresh=-55.0))

        # Drive neuron 0 with suprathreshold current (10.0), neuron 1 with zero current
        spikes = []
        for _ in range(30):
            step_spikes = net.step_synchronous([10.0, 0.0])
            spikes.append(step_spikes)

        # Neuron 0 should fire multiple times; neuron 1 should remain silent
        n0_fires = sum(1 for s in spikes if s[0])
        n1_fires = sum(1 for s in spikes if s[1])
        self.assertGreater(n0_fires, 0)
        self.assertEqual(n1_fires, 0)

    def test_event_driven_priority_queue(self) -> None:
        """Verify min-heap event queue dispatches spikes across axonal delays."""
        net = SpikingNetwork()
        n0 = net.add_neuron(LIFNeuron(neuron_id=0))
        n1 = net.add_neuron(LIFNeuron(neuron_id=1))
        net.add_synapse(pre_id=0, post_id=1, weight=50.0, delay_ms=3.0)

        sim = EventDrivenSimulator(net)
        sim.schedule_stimulus(time_ms=0.0, target_id=0, current_amplitude=50.0)
        recorded = sim.run_until(end_time_ms=10.0)

        # Expect neuron 0 at t=0.0, and neuron 1 at t=3.0
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0], (0.0, 0))
        self.assertEqual(recorded[1], (3.0, 1))

    def test_liquid_state_machine(self) -> None:
        """Verify 3D Liquid State Machine reservoir and ridge regression readout."""
        lsm = LiquidStateMachine(grid_dim=(2, 2, 2))  # 8 neurons
        self.assertEqual(lsm.total_neurons, 8)

        # Verify Dale's principle: excitatory weights >= 0, inhibitory weights <= 0
        for i in range(lsm.total_neurons):
            for j in range(lsm.total_neurons):
                w = lsm.weights[i][j]
                if lsm.is_inhibitory[i]:
                    self.assertLessEqual(w, 0.0)
                else:
                    self.assertGreaterEqual(w, 0.0)

        # Test simulation and state vector extraction
        input_spikes = [[True if t % 2 == 0 else False] for t in range(15)]
        input_weights = [[0.5] * lsm.total_neurons]
        raster = lsm.simulate(input_spikes, input_weights)
        state_vec = lsm.extract_liquid_state(raster)

        self.assertEqual(len(raster), 15)
        self.assertEqual(len(state_vec), 8)


if __name__ == "__main__":
    unittest.main()
