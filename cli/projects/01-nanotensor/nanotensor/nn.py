"""
Neural network primitives for NanoTensor:
- Linear projection with Xavier/Kaiming initialization
- Embedding layer with full sparse gradient backpropagation
- RMSNorm (Root Mean Square Layer Normalization)
- SwiGLU (Swish-Gated Linear Unit)
- Rotary Position Embeddings (RoPE)
- Multi-Head Causal Self-Attention with KV-Cache support
- Transformer Decoder Block
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from nanotensor.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def parameters(self) -> List[Tensor]:
        params = []
        for _, val in self.__dict__.items():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """
    Fully connected linear transformation: y = x @ W + b.
    Weights initialized via Xavier uniform to ensure stable signal variance across layers.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor.uniform(
            (in_features, out_features),
            low=-limit,
            high=limit,
            requires_grad=True
        )
        if bias:
            self.bias = Tensor.zeros((1, out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Used in modern LLM architectures (LLaMA, Mistral, Qwen) for improved training stability
    and reduced compute overhead by removing mean centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        # Learnable scale parameter initialized to ones
        self.weight = Tensor.ones((1, dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x is expected to have shape (..., dim)
        c = x.contiguous()
        *batch_dims, d = c.shape
        outer_size = 1
        for bd in batch_dims:
            outer_size *= bd

        normed_data = [0.0] * c.size
        # Store inverse rms for backprop
        inv_rms_list = [0.0] * outer_size

        for b in range(outer_size):
            offset = b * d
            # Calculate sum of squares
            sum_sq = 0.0
            for i in range(d):
                val = c._data[offset + i]
                sum_sq += val * val
            rms = math.sqrt(sum_sq / d + self.eps)
            inv_rms = 1.0 / rms
            inv_rms_list[b] = inv_rms
            for i in range(d):
                normed_data[offset + i] = c._data[offset + i] * inv_rms

        normed = Tensor(normed_data, shape=c.shape, requires_grad=x.requires_grad, _children=(x,), _op="RMSNorm")

        def _backward():
            if x.requires_grad:
                grad_normed = normed.grad.contiguous()
                grad_x_data = [0.0] * c.size

                for b in range(outer_size):
                    offset = b * d
                    inv_rms = inv_rms_list[b]

                    # Inner product: sum(grad_normed_i * x_i)
                    sum_gx = 0.0
                    for i in range(d):
                        sum_gx += grad_normed._data[offset + i] * c._data[offset + i]

                    factor = sum_gx * (inv_rms ** 3) / d

                    for i in range(d):
                        # dL/dx_i = inv_rms * dL/d(norm_i) - x_i * factor
                        g = grad_normed._data[offset + i] * inv_rms - c._data[offset + i] * factor
                        grad_x_data[offset + i] = g

                grad_x = Tensor(grad_x_data, shape=c.shape)
                x.grad = grad_x if x.grad is None else x.grad + grad_x

        normed._backward = _backward
        # Scale by learnable weight
        return normed * self.weight


class Embedding(Module):
    """
    Embedding table mapping discrete token IDs to continuous dense vectors.
    Supports 1D and 2D index tensors with direct gradient accumulation.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Initialized with standard deviation 1.0
        self.weight = Tensor.randn(
            (num_embeddings, embedding_dim),
            mean=0.0,
            std=0.02,
            requires_grad=True
        )

    def forward(self, indices: List[int] | List[List[int]]) -> Tensor:
        """Looks up rows in embedding weight matrix for given indices."""
        w_contig = self.weight.contiguous()
        d = self.embedding_dim

        # Support 1D list or 2D list of indices
        if isinstance(indices[0], int):
            seq_len = len(indices)
            flat_indices = indices
            out_shape = (seq_len, d)
        else:
            batch_size = len(indices)
            seq_len = len(indices[0])
            flat_indices = [idx for row in indices for idx in row]
            out_shape = (batch_size, seq_len, d)

        out_data = []
        for idx in flat_indices:
            start = idx * d
            out_data.extend(w_contig._data[start:start + d])

        out = Tensor(out_data, shape=out_shape, requires_grad=self.weight.requires_grad,
                     _children=(self.weight,), _op="Embedding")

        def _backward():
            if self.weight.requires_grad:
                grad_w_data = [0.0] * self.weight.size
                grad_out_contig = out.grad.contiguous()

                for i, idx in enumerate(flat_indices):
                    out_offset = i * d
                    w_offset = idx * d
                    for k in range(d):
                        grad_w_data[w_offset + k] += grad_out_contig._data[out_offset + k]

                grad_w = Tensor(grad_w_data, shape=self.weight.shape)
                self.weight.grad = grad_w if self.weight.grad is None else self.weight.grad + grad_w

        out._backward = _backward
        return out


class SwiGLU(Module):
    """
    Swish-Gated Linear Unit (Shazeer, 2020):
    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, dim: int, hidden_dim: int):
        self.w_gate = Linear(dim, hidden_dim, bias=False)
        self.w_up = Linear(dim, hidden_dim, bias=False)
        self.w_down = Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x).silu()
        up = self.w_up(x)
        return self.w_down(gate * up)


def apply_rope(x: Tensor, start_pos: int = 0, theta: float = 10000.0) -> Tensor:
    """
    Applies Rotary Position Embeddings (RoPE) to a tensor of shape (batch, seq_len, num_heads, head_dim)
    or (seq_len, num_heads, head_dim) or (batch, seq_len, dim).
    Rotates coordinate pairs (x_{2i}, x_{2i+1}) by angle m * theta_i.
    """
    c = x.contiguous()
    shape = c.shape
    head_dim = shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE")

    # If shape is 3D (batch, seq_len, dim)
    if len(shape) == 3:
        batch, seq_len, dim = shape
        out_data = list(c._data)
        for b in range(batch):
            for s in range(seq_len):
                pos = start_pos + s
                base_idx = b * (seq_len * dim) + s * dim
                for i in range(0, head_dim, 2):
                    freq = 1.0 / (theta ** (i / head_dim))
                    angle = pos * freq
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)

                    idx0 = base_idx + i
                    idx1 = base_idx + i + 1
                    x0 = c._data[idx0]
                    x1 = c._data[idx1]

                    out_data[idx0] = x0 * cos_a - x1 * sin_a
                    out_data[idx1] = x0 * sin_a + x1 * cos_a

        out = Tensor(out_data, shape=shape, requires_grad=x.requires_grad, _children=(x,), _op="RoPE")

        def _backward():
            if x.requires_grad:
                grad_c = out.grad.contiguous()
                grad_x_data = list(grad_c._data)
                for b in range(batch):
                    for s in range(seq_len):
                        pos = start_pos + s
                        base_idx = b * (seq_len * dim) + s * dim
                        for i in range(0, head_dim, 2):
                            freq = 1.0 / (theta ** (i / head_dim))
                            angle = pos * freq
                            cos_a = math.cos(angle)
                            sin_a = math.sin(angle)

                            idx0 = base_idx + i
                            idx1 = base_idx + i + 1
                            g0 = grad_c._data[idx0]
                            g1 = grad_c._data[idx1]

                            # Transpose of 2D rotation matrix is rotation by -angle
                            grad_x_data[idx0] = g0 * cos_a + g1 * sin_a
                            grad_x_data[idx1] = -g0 * sin_a + g1 * cos_a

                grad_x = Tensor(grad_x_data, shape=shape)
                x.grad = grad_x if x.grad is None else x.grad + grad_x

        out._backward = _backward
        return out
    else:
        # Passthrough if not 3D
        return x


class MultiHeadAttention(Module):
    """
    Multi-Head Scaled Dot-Product Causal Self-Attention with RoPE and KV-Cache support.
    """

    def __init__(self, dim: int, num_heads: int):
        self.dim = dim
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = dim // num_heads

        self.w_q = Linear(dim, dim, bias=False)
        self.w_k = Linear(dim, dim, bias=False)
        self.w_v = Linear(dim, dim, bias=False)
        self.w_o = Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        return_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        x: (batch, seq_len, dim)
        Returns: output tensor of shape (batch, seq_len, dim), and optional attention weights.
        """
        batch, seq_len, dim = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Apply Rotary Position Embeddings
        q = apply_rope(q, start_pos=start_pos)
        k = apply_rope(k, start_pos=start_pos)

        # Handle KV cache for iterative generation
        if kv_cache is not None:
            if "k" in kv_cache and kv_cache["k"] is not None:
                # Concatenate along seq_len dimension
                old_k = kv_cache["k"]
                old_v = kv_cache["v"]
                k_data = old_k._data + k._data
                v_data = old_v._data + v._data
                total_seq = old_k.shape[1] + seq_len
                k = Tensor(k_data, shape=(batch, total_seq, dim))
                v = Tensor(v_data, shape=(batch, total_seq, dim))
            kv_cache["k"] = k
            kv_cache["v"] = v

        total_kv_len = k.shape[1]

        # Reshape to (batch, num_heads, seq_len, head_dim)
        # Using pure Python tensor reshaping and transpositions
        q_heads = q.reshape((batch, seq_len, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))
        k_heads = k.reshape((batch, total_kv_len, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))
        v_heads = v.reshape((batch, total_kv_len, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))

        # Scaled dot-product: Q @ K.T / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q_heads @ k_heads.transpose(-1, -2)) * scale

        # Apply Causal Mask if seq_len > 1
        if seq_len > 1:
            # Mask out future positions
            c_scores = scores.contiguous()
            masked_data = list(c_scores._data)
            stride_h = c_scores.strides[1]
            stride_q = c_scores.strides[2]
            stride_k = c_scores.strides[3]

            for b in range(batch):
                for h in range(self.num_heads):
                    for i in range(seq_len):
                        q_pos = start_pos + i
                        for j in range(total_kv_len):
                            if j > q_pos:
                                idx = b * c_scores.strides[0] + h * stride_h + i * stride_q + j * stride_k
                                masked_data[idx] = -1e9

            orig_scores = scores
            scores = Tensor(masked_data, shape=orig_scores.shape, requires_grad=orig_scores.requires_grad,
                            _children=(orig_scores,), _op="CausalMask")

            def _backward_mask():
                if orig_scores.requires_grad:
                    # Zero out gradients at masked positions
                    g_data = list(scores.grad.contiguous()._data)
                    for b in range(batch):
                        for h in range(self.num_heads):
                            for i in range(seq_len):
                                q_pos = start_pos + i
                                for j in range(total_kv_len):
                                    if j > q_pos:
                                        idx = b * c_scores.strides[0] + h * stride_h + i * stride_q + j * stride_k
                                        g_data[idx] = 0.0
                    g = Tensor(g_data, shape=orig_scores.shape)
                    orig_scores.grad = g if orig_scores.grad is None else orig_scores.grad + g

            scores._backward = _backward_mask

        # Softmax over last dimension
        attn_weights = scores.softmax(axis=-1)

        # Context output: Attn @ V
        context = attn_weights @ v_heads  # (batch, num_heads, seq_len, head_dim)

        # Merge heads back to (batch, seq_len, dim)
        merged = context.permute((0, 2, 1, 3)).reshape((batch, seq_len, dim))

        out = self.w_o(merged)
        weights_to_return = attn_weights if return_weights else None
        return out, weights_to_return


class TransformerBlock(Module):
    """
    Modern Pre-RMSNorm Transformer Decoder Block:
    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(self, dim: int, num_heads: int, hidden_dim: Optional[int] = None):
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        h_dim = hidden_dim or int(2 * dim * 4 / 3)  # Standard SwiGLU expansion
        self.mlp = SwiGLU(dim, h_dim)

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        return_weights: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        normed_x1 = self.norm1(x)
        attn_out, weights = self.attn(normed_x1, start_pos=start_pos, kv_cache=kv_cache, return_weights=return_weights)
        h = x + attn_out

        normed_h = self.norm2(h)
        mlp_out = self.mlp(normed_h)
        out = h + mlp_out
        return out, weights
