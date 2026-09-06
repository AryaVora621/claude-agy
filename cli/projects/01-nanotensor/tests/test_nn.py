"""
Unit tests and mathematical verification of neural network components and KV-cache equivalence.
"""

import unittest
import math
from nanotensor.tensor import Tensor
from nanotensor.nn import Linear, RMSNorm, SwiGLU, Embedding, MultiHeadAttention, TransformerBlock
from nanotensor.transformer import NanoTransformer, cross_entropy_loss
from tests.test_autograd import grad_check


class TestNeuralNetworkModules(unittest.TestCase):

    def test_rmsnorm_gradcheck(self):
        norm = RMSNorm(4)
        x = Tensor([[1.0, -2.0, 3.0, 0.5]], requires_grad=True)
        ok = grad_check(lambda t: norm(t).sum(), x)
        self.assertTrue(ok)

    def test_linear_forward_backward(self):
        lin = Linear(3, 2)
        x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = lin(x)
        self.assertEqual(y.shape, (1, 2))
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(lin.weight.grad)

    def test_embedding_forward_backward(self):
        emb = Embedding(10, 4)
        indices = [1, 3, 1]
        out = emb(indices)
        self.assertEqual(out.shape, (3, 4))
        loss = out.sum()
        loss.backward()
        # Row 1 was accessed twice, row 3 once, other rows 0
        w_grad = emb.weight.grad
        self.assertAlmostEqual(w_grad._data[1 * 4 + 0], 2.0)
        self.assertAlmostEqual(w_grad._data[3 * 4 + 0], 1.0)
        self.assertAlmostEqual(w_grad._data[0 * 4 + 0], 0.0)

    def test_swiglu_forward(self):
        mlp = SwiGLU(4, 8)
        x = Tensor([[1.0, 0.5, -1.0, 2.0]])
        out = mlp(x)
        self.assertEqual(out.shape, (1, 4))

    def test_kv_cache_equivalence(self):
        """
        Critical test: Verifies that step-by-step KV-cache generation
        produces identical logits to full sequence non-cached forward pass.
        """
        dim = 16
        heads = 2
        vocab = 20
        model = NanoTransformer(vocab_size=vocab, dim=dim, num_layers=2, num_heads=heads)

        prompt = [2, 5, 8]
        next_tok = 11

        # 1. Full non-cached forward pass with [2, 5, 8, 11]
        full_tokens = [prompt + [next_tok]]
        logits_full, _ = model.forward(full_tokens, start_pos=0, kv_caches=None)
        # Logits of the final token (pos 3)
        final_logits_full = logits_full._data[-vocab:]

        # 2. Cached step-by-step forward pass
        caches = [{} for _ in range(model.num_layers)]
        # Prefill prompt
        model.forward([prompt], start_pos=0, kv_caches=caches)
        # Step with next_tok
        logits_step, _ = model.forward([[next_tok]], start_pos=len(prompt), kv_caches=caches)
        final_logits_cached = logits_step._data[-vocab:]

        # Verify numerical equivalence
        for i in range(vocab):
            self.assertAlmostEqual(
                final_logits_full[i],
                final_logits_cached[i],
                places=4,
                msg=f"KV-cache mismatch at index {i}"
            )

    def test_cross_entropy_loss(self):
        logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
        targets = [0, 1]
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # Probabilities should be high for target classes, so loss should be low (< 1.0)
        self.assertLess(loss._data[0], 1.0)


if __name__ == "__main__":
    unittest.main()
