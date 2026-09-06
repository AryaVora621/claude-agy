# NanoTensor

> **Zero-Dependency Modern Deep Learning Engine & Transformer from First Principles.**  
> Built purely in standard Python 3.10+ with reverse-mode automatic differentiation, modern LLM primitives (RoPE, RMSNorm, SwiGLU, KV-Cache), a scratch Byte-Pair Encoding (BPE) tokenizer, and live terminal attention heatmaps.

---

## Key Highlights

- **Zero External Dependencies**: Implemented entirely with Python standard library components. No PyTorch, NumPy, or C extensions required.
- **Full Reverse-Mode Autograd**:
  - Directed Acyclic Graph (DAG) with topological sorting.
  - Multi-dimensional array coordinate mapping via strides and offsets.
  - Arbitrary-rank tensor broadcasting with gradient reduction along expanded axes.
  - Finite-difference gradient checking validating analytical gradients against machine epsilon.
- **State-of-the-Art Transformer Architecture**:
  - **RoPE (Rotary Position Embeddings)**: Dynamic coordinate pair rotation preserving relative token distance.
  - **RMSNorm (Root Mean Square Layer Normalization)**: Fast, non-mean-centered normalization with learnable scale parameters.
  - **SwiGLU (Swish Gated Linear Unit)**: Shazeer-style non-linear feedforward networks.
  - **Multi-Head Causal Self-Attention**: Scaled dot-product attention with strict triangular masking.
  - **Key-Value Caching (KV-Cache)**: Transforms autoregressive generation complexity from $O(N^2)$ to $O(N)$.
- **Scratch BPE Tokenizer**:
  - 256 base byte tokens (guaranteeing zero out-of-vocabulary exceptions on any arbitrary UTF-8 input).
  - Frequency-driven pair merge training.
  - JSON serialization and deserialization.
- **Terminal Visualizer**:
  - Real-time ASCII attention matrix heatmaps with sub-block intensity shading.
  - Next-token probability distribution bar graphs.

---

## Architectural Diagram

```
                        ┌──────────────────────────────┐
                        │   Input Tokens / Raw Text    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Byte-Pair Tokenizer (BPE)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │       Embedding Layer        │
                        └──────────────┬───────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
                ▼                                             │
┌──────────────────────────────┐                              │
│         RMSNorm Layer        │                              │
└──────────────┬───────────────┘                              │
               ▼                                              │
┌──────────────────────────────┐                              │
│   RoPE Positional Rotation   │                              │
└──────────────┬───────────────┘                              │
               ▼                                              │
┌──────────────────────────────┐                              │
│   Multi-Head Self-Attention  │                              │
│     (with KV-Cache & Mask)   │                              │
└──────────────┬───────────────┘                              │
               ▼                                              │
      ( + Residual Connection ) ◄─────────────────────────────┘
               │
               ├──────────────────────────────────────────────┐
               ▼                                              │
┌──────────────────────────────┐                              │
│         RMSNorm Layer        │                              │
└──────────────┬───────────────┘                              │
               ▼                                              │
┌──────────────────────────────┐                              │
│       SwiGLU MLP Block       │                              │
│   SiLU(x @ W_gate) * (x @ W) │                              │
└──────────────┬───────────────┘                              │
               ▼                                              │
      ( + Residual Connection ) ◄─────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│        Final RMSNorm         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       LM Head Projection     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  ASCII Attention Visualizer  │
└──────────────────────────────┘
```

---

## Quickstart

### Run All Unit Tests
Verify the autograd engine, numerical gradients, neural network layers, KV-cache equivalence, and BPE tokenizer:

```bash
python3 -m unittest discover -s tests
```

### Run the Training Demo
Trains a micro-transformer on arithmetic patterns, prints loss descent, and displays terminal attention heatmaps:

```bash
python3 examples/train_toy.py
```

### Run the Interactive Attention Inspector
Inspect next-token probability distributions and view attention heads in real time:

```bash
python3 examples/chat_demo.py
```

---

## Autograd Engine Example

```python
from nanotensor import Tensor

# Create tensors with automatic gradient tracking
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = Tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)

# Forward pass: Matrix multiplication, SwiGLU activation, and reduction
y = (x @ w).silu().sum()

# Backward pass: Reverse-mode automatic differentiation
y.backward()

print("Output Value:", y)
print("Gradient dY/dX:", x.grad)
print("Gradient dY/dW:", w.grad)
```

---

## Mathematical Verification

Every mathematical operation in NanoTensor is accompanied by finite-difference numerical gradient checks:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}$$

The analytical gradients calculated by our computational graph backpropagation are verified to match numerical finite differences within a tolerance of $\Delta < 10^{-4}$.
