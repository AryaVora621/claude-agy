"""
Full NanoTransformer Model Architecture:
Features pre-norm Transformer decoder blocks, tied embeddings,
KV-cache iterative text generation, temperature/top-k sampling,
and numerically stable Cross-Entropy loss computation.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional, Dict, Any, Callable
from nanotensor.tensor import Tensor
from nanotensor.nn import Module, Embedding, RMSNorm, Linear, TransformerBlock


def cross_entropy_loss(logits: Tensor, targets: List[int]) -> Tensor:
    """
    Computes numerically stable cross-entropy loss:
    Loss = - (1/N) * sum_i ( log_softmax(logits_i)[targets_i] )
    With exact gradient backpropagation:
    dL/d(logit_j) = (softmax(logit_j) - 1(j == target)) / N
    """
    c_logits = logits.contiguous()
    N = len(targets)
    vocab_size = c_logits.shape[-1]
    outer_len = c_logits.size // vocab_size

    if outer_len != N:
        raise ValueError(f"Number of target labels ({N}) must match outer logits dimension ({outer_len})")

    probs_data = [0.0] * c_logits.size
    total_loss = 0.0

    for i in range(N):
        offset = i * vocab_size
        target_idx = targets[i]

        # Max subtraction for numerical stability
        max_val = -float("inf")
        for v in range(vocab_size):
            val = c_logits._data[offset + v]
            if val > max_val:
                max_val = val

        sum_exp = 0.0
        for v in range(vocab_size):
            e = math.exp(c_logits._data[offset + v] - max_val)
            probs_data[offset + v] = e
            sum_exp += e

        inv_sum = 1.0 / sum_exp
        for v in range(vocab_size):
            probs_data[offset + v] *= inv_sum

        target_prob = probs_data[offset + target_idx]
        total_loss += -math.log(max(target_prob, 1e-15))

    mean_loss = total_loss / N

    out = Tensor(mean_loss, shape=(), requires_grad=logits.requires_grad,
                 _children=(logits,), _op="CrossEntropy")

    def _backward():
        if logits.requires_grad:
            grad_logits_data = [0.0] * c_logits.size
            for i in range(N):
                offset = i * vocab_size
                target_idx = targets[i]
                for v in range(vocab_size):
                    p = probs_data[offset + v]
                    grad_logits_data[offset + v] = (p - (1.0 if v == target_idx else 0.0)) / N

            grad_logits = Tensor(grad_logits_data, shape=logits.shape)
            logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits

    out._backward = _backward
    return out


class NanoTransformer(Module):
    """
    Complete Autoregressive Language Model:
    Token Embedding -> N x TransformerBlock -> Final RMSNorm -> LM Head Projection
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.tok_embeddings = Embedding(vocab_size, dim)
        self.layers = [
            TransformerBlock(dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(dim)
        self.lm_head = Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        tokens: List[int] | List[List[int]],
        start_pos: int = 0,
        kv_caches: Optional[List[Dict[str, Tensor]]] = None,
        return_weights: bool = False
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """
        Forward pass over token sequence.
        Returns:
            logits: (batch, seq_len, vocab_size)
            attn_weights: list of attention weight matrices per layer
        """
        # (batch, seq_len, dim)
        h = self.tok_embeddings(tokens)

        all_weights = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            h, weights = layer(h, start_pos=start_pos, kv_cache=cache, return_weights=return_weights)
            all_weights.append(weights)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, all_weights

    def generate(
        self,
        prompt: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 5,
        use_cache: bool = True,
        on_token: Optional[Callable[[int, Optional[Tensor]], None]] = None
    ) -> List[int]:
        """
        Autoregressively generates continuation tokens from a prompt.
        Utilizes KV-caching for O(N) dynamic sequence expansion.
        """
        generated = list(prompt)
        kv_caches = [{} for _ in range(self.num_layers)] if use_cache else None

        # Prefill prompt
        if use_cache:
            logits, weights = self.forward([prompt], start_pos=0, kv_caches=kv_caches, return_weights=True)
            next_token_logits = logits._data[-self.vocab_size:]
            next_token = self._sample(next_token_logits, temperature, top_k)
            generated.append(next_token)
            if on_token:
                on_token(next_token, weights[-1] if weights else None)

            # Iterative generation using cached keys and values
            for step in range(1, max_new_tokens):
                curr_pos = len(prompt) + step - 1
                logits, weights = self.forward([[next_token]], start_pos=curr_pos,
                                              kv_caches=kv_caches, return_weights=True)
                next_token_logits = logits._data[-self.vocab_size:]
                next_token = self._sample(next_token_logits, temperature, top_k)
                generated.append(next_token)
                if on_token:
                    on_token(next_token, weights[-1] if weights else None)
        else:
            for _ in range(max_new_tokens):
                logits, weights = self.forward([generated], start_pos=0, kv_caches=None, return_weights=True)
                next_token_logits = logits._data[-self.vocab_size:]
                next_token = self._sample(next_token_logits, temperature, top_k)
                generated.append(next_token)
                if on_token:
                    on_token(next_token, weights[-1] if weights else None)

        return generated

    def _sample(self, logits: List[float], temperature: float, top_k: Optional[int]) -> int:
        """Applies temperature scaling and top-k filtering before categorical sampling."""
        if temperature <= 0.0:
            # Greedy argmax
            return max(range(len(logits)), key=lambda i: logits[i])

        # Temperature scaling
        scaled = [x / temperature for x in logits]

        # Top-K filtering
        if top_k is not None and top_k < len(scaled):
            indexed = list(enumerate(scaled))
            indexed.sort(key=lambda x: x[1], reverse=True)
            cutoff = indexed[top_k - 1][1]
            scaled = [v if v >= cutoff else -float("inf") for v in scaled]

        # Numerical softmax
        max_val = max(scaled)
        exp_sum = 0.0
        exp_vals = []
        for x in scaled:
            e = 0.0 if x == -float("inf") else math.exp(x - max_val)
            exp_vals.append(e)
            exp_sum += e

        probs = [e / exp_sum for e in exp_vals]

        # Cumulative probability sampling
        r = random.random()
        cum = 0.0
        for idx, p in enumerate(probs):
            cum += p
            if r <= cum:
                return idx
        return len(probs) - 1
