"""
NanoTensor: Zero-Dependency Deep Learning Engine from First Principles.
"""

from nanotensor.tensor import Tensor
from nanotensor.nn import Module, Linear, RMSNorm, Embedding, SwiGLU, MultiHeadAttention, TransformerBlock, apply_rope
from nanotensor.transformer import NanoTransformer, cross_entropy_loss
from nanotensor.tokenizer import BPETokenizer
from nanotensor.optim import AdamW, clip_grad_norm_

__all__ = [
    "Tensor",
    "Module",
    "Linear",
    "RMSNorm",
    "Embedding",
    "SwiGLU",
    "MultiHeadAttention",
    "TransformerBlock",
    "apply_rope",
    "NanoTransformer",
    "cross_entropy_loss",
    "BPETokenizer",
    "AdamW",
    "clip_grad_norm_",
]
