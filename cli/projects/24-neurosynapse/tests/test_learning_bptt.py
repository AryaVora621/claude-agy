"""Unit tests for surrogate gradient learning and Spike-BPTT."""

import unittest
from neurosynapse.learning import (
    AdamOptimizer,
    ArcTanSurrogate,
    FastSigmoidSurrogate,
    SGDOptimizer,
    SpikingDenseLayer,
    TriangularSurrogate,
    mean_squared_rate_loss,
)


class TestLearningBPTT(unittest.TestCase):
    """Test suite for surrogate derivatives and temporal gradient descent."""

    def test_surrogate_derivatives(self) -> None:
        """Verify analytical surrogate derivatives are positive and symmetric."""
        surrogates = [
            FastSigmoidSurrogate(k=10.0),
            ArcTanSurrogate(k=5.0),
            TriangularSurrogate(gamma=1.0),
        ]

        for s in surrogates:
            # At threshold (V = V_th), derivative is maximal and positive
            d_center = s.derivative(v=1.0, v_thresh=1.0)
            self.assertGreater(d_center, 0.0)

            # Away from threshold, derivative is non-negative and smaller
            d_offset_pos = s.derivative(v=1.5, v_thresh=1.0)
            d_offset_neg = s.derivative(v=0.5, v_thresh=1.0)
            self.assertGreaterEqual(d_offset_pos, 0.0)
            self.assertAlmostEqual(d_offset_pos, d_offset_neg, delta=1e-4)
            self.assertLess(d_offset_pos, d_center)

    def test_spiking_dense_forward_backward(self) -> None:
        """Verify forward unrolling and backward surrogate gradient accumulation."""
        layer = SpikingDenseLayer(in_features=3, out_features=2, decay_factor=0.8, v_thresh=1.0)
        inputs = [[1.0, 0.5, 0.0] for _ in range(5)]

        spikes = layer.forward_sequence(inputs)
        self.assertEqual(len(spikes), 5)
        self.assertEqual(len(spikes[0]), 2)

        # Backward pass with synthetic target gradients
        grad_out = [[0.1, -0.1] for _ in range(5)]
        grad_in = layer.backward(grad_out)

        self.assertEqual(len(grad_in), 5)
        self.assertEqual(len(grad_in[0]), 3)

        # Gradients must be computed and non-zero
        w_grad_sum = sum(abs(layer.grad_weights[i][j]) for i in range(3) for j in range(2))
        self.assertGreater(w_grad_sum, 0.0)

    def test_adam_and_sgd_optimizers(self) -> None:
        """Verify Adam and SGD optimizers update layer weights."""
        layer1 = SpikingDenseLayer(in_features=2, out_features=2)
        layer2 = SpikingDenseLayer(in_features=2, out_features=2)

        sgd = SGDOptimizer([layer1], lr=0.1)
        adam = AdamOptimizer([layer2], lr=0.1)

        # Set artificial gradients
        layer1.grad_weights = [[1.0, 1.0], [1.0, 1.0]]
        layer2.grad_weights = [[1.0, 1.0], [1.0, 1.0]]

        w1_before = [list(row) for row in layer1.weights]
        w2_before = [list(row) for row in layer2.weights]

        sgd.step()
        adam.step()

        self.assertNotEqual(layer1.weights, w1_before)
        self.assertNotEqual(layer2.weights, w2_before)

    def test_mean_squared_rate_loss(self) -> None:
        """Verify rate loss calculation and gradient distribution."""
        spikes = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # Actual rates: [0.5, 0.0]
        targets = [0.5, 0.0]
        loss, grad = mean_squared_rate_loss(spikes, targets)
        self.assertAlmostEqual(loss, 0.0, places=5)

        targets_diff = [1.0, 0.0]
        loss2, grad2 = mean_squared_rate_loss(spikes, targets_diff)
        self.assertGreater(loss2, 0.0)
        self.assertEqual(len(grad2), 4)


if __name__ == "__main__":
    unittest.main()
