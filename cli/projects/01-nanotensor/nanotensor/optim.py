"""
Optimization algorithms for NanoTensor:
- AdamW (Decoupled Weight Decay, Kingma & Ba 2014, Loshchilov & Hutter 2017)
- SGD with Momentum
- Global L2 gradient norm clipping
"""

from __future__ import annotations
import math
from typing import List, Dict, Optional
from nanotensor.tensor import Tensor


def clip_grad_norm_(parameters: List[Tensor], max_norm: float) -> float:
    """
    Clips gradient norms of an iterable of parameters by scaling them in place.
    Prevents gradient explosion in recurrent and deep transformer backpropagation.
    """
    total_sum_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            c = p.grad.contiguous()
            for g in c._data:
                total_sum_sq += g * g

    total_norm = math.sqrt(total_sum_sq)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad._data = [g * scale for g in p.grad._data]

    return total_norm


class Optimizer:
    """Base optimizer class."""

    def __init__(self, parameters: List[Tensor], lr: float = 1e-3):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2017).
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # First and second moment buffers
        self.m: Dict[int, List[float]] = {}
        self.v: Dict[int, List[float]] = {}

    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        lr = self.lr
        wd = self.weight_decay
        eps = self.eps

        # Bias correction factors
        bias_corr1 = 1.0 - (b1 ** self.step_count)
        bias_corr2 = 1.0 - (b2 ** self.step_count)

        for p_idx, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            c_data = p.contiguous()._data
            g_data = p.grad.contiguous()._data
            size = p.size

            if p_idx not in self.m:
                self.m[p_idx] = [0.0] * size
                self.v[p_idx] = [0.0] * size

            m_buf = self.m[p_idx]
            v_buf = self.v[p_idx]

            for i in range(size):
                g = g_data[i]
                theta = c_data[i]

                # Update biased first moment
                m_buf[i] = b1 * m_buf[i] + (1.0 - b1) * g
                # Update biased second raw moment
                v_buf[i] = b2 * v_buf[i] + (1.0 - b2) * (g * g)

                # Compute bias-corrected moments
                m_hat = m_buf[i] / bias_corr1
                v_hat = v_buf[i] / bias_corr2

                # Decoupled weight decay + adaptive gradient step
                step_val = (m_hat / (math.sqrt(v_hat) + eps)) + (wd * theta)
                c_data[i] = theta - lr * step_val

            # Assign updated data back to parameter
            p._data = c_data
