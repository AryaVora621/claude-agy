"""Unit tests for neuromorphic spike encoders and decoders."""

import unittest
from neurosynapse.encoding import (
    BernoulliEncoder,
    DeltaEncoder,
    ExponentialFilterDecoder,
    FirstSpikeWinnerDecoder,
    PhaseEncoder,
    PoissonEncoder,
    RankOrderEncoder,
    RateDecoder,
    TTFSEncoder,
)


class TestEncodingDecoding(unittest.TestCase):
    """Test suite for spike encoders and continuous decoders."""

    def test_poisson_rate_encoding(self) -> None:
        """Verify Poisson encoder produces spike rates proportional to stimulus."""
        pe = PoissonEncoder(max_rate_hz=200.0)
        values = [1.0, 0.5, 0.0]
        raster = pe.encode(values, num_timesteps=200, dt_ms=1.0)
        rates = RateDecoder.decode(raster)

        self.assertGreater(rates[0], rates[1])
        self.assertEqual(rates[2], 0.0)

    def test_ttfs_latency_encoding(self) -> None:
        """Verify Time-to-First-Spike orders spikes by descending magnitude."""
        ttfs = TTFSEncoder(tau_ms=10.0)
        values = [0.95, 0.5, 0.1]
        raster = ttfs.encode(values, num_timesteps=50, dt_ms=1.0)

        # Find first spike step for each neuron
        steps = []
        for i in range(3):
            step_found = None
            for t in range(50):
                if raster[t][i]:
                    step_found = t
                    break
            steps.append(step_found)

        self.assertIsNotNone(steps[0])
        self.assertIsNotNone(steps[1])
        self.assertLess(steps[0], steps[1])

    def test_rank_order_encoding(self) -> None:
        """Verify Rank Order Coding fires in strict rank order."""
        roc = RankOrderEncoder()
        values = [0.2, 0.9, 0.5]
        raster = roc.encode(values, num_timesteps=5)

        # Highest (0.9, idx 1) fires at step 0
        self.assertTrue(raster[0][1])
        # Second highest (0.5, idx 2) fires at step 1
        self.assertTrue(raster[1][2])
        # Third highest (0.2, idx 0) fires at step 2
        self.assertTrue(raster[2][0])

    def test_phase_encoding(self) -> None:
        """Verify phase encoder locks spike timing to reference cycle."""
        pe = PhaseEncoder(cycle_period_steps=10)
        values = [1.0, 0.0]
        raster = pe.encode(values, num_timesteps=30)

        # Value 1.0 -> offset 0
        self.assertTrue(raster[0][0])
        self.assertTrue(raster[10][0])
        self.assertTrue(raster[20][0])

        # Value 0.0 -> offset 9
        self.assertTrue(raster[9][1])
        self.assertTrue(raster[19][1])
        self.assertTrue(raster[29][1])

    def test_delta_modulation_encoding(self) -> None:
        """Verify temporal delta modulator emits ON/OFF spikes on transitions."""
        delta = DeltaEncoder(delta_threshold=0.2)
        on0, off0 = delta.encode_step([1.0])
        # No spikes on initialization step
        self.assertFalse(on0[0])
        self.assertFalse(off0[0])

        # Increase by +0.3 -> ON spike
        on1, off1 = delta.encode_step([1.3])
        self.assertTrue(on1[0])
        self.assertFalse(off1[0])

        # Decrease by -0.4 -> OFF spike
        on2, off2 = delta.encode_step([0.9])
        self.assertFalse(on2[0])
        self.assertTrue(off2[0])

    def test_exponential_filter_decoding(self) -> None:
        """Verify continuous analog trace reconstruction from spikes."""
        decoder = ExponentialFilterDecoder(tau_filter_ms=10.0, dt_ms=1.0)
        raster = [[True], [False], [False], [False], [False]]
        traces = decoder.decode(raster)

        self.assertEqual(len(traces), 5)
        self.assertAlmostEqual(traces[0][0], 1.0, places=3)
        # Trace should decay smoothly
        self.assertLess(traces[4][0], traces[0][0])

    def test_first_spike_winner_decoding(self) -> None:
        """Verify first spike Winner-Take-All decoder."""
        raster = [
            [False, False, False],
            [False, True, False],   # Neuron 1 fires at t=1
            [True, False, False],    # Neuron 0 fires at t=2
        ]
        winner, time_step = FirstSpikeWinnerDecoder.decode(raster)
        self.assertEqual(winner, 1)
        self.assertEqual(time_step, 1)


if __name__ == "__main__":
    unittest.main()
