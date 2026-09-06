"""
Terminal visualization engine for NanoTensor:
- High-contrast ASCII/ANSI attention matrix heatmaps
- Token probability distribution bar charts
- Live generative inference visualizer with per-token attention tracking
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from nanotensor.tensor import Tensor


# Shading blocks from low to high intensity
SHADE_CHARS = [" ", "·", "░", "▒", "▓", "█"]


def render_attention_heatmap(
    weights: Tensor,
    tokens: List[str],
    head_idx: int = 0,
    title: str = "Attention Matrix"
) -> str:
    """
    Renders an ASCII heatmap of attention weights between query and key tokens.
    weights shape: (batch, num_heads, seq_len, seq_len) or (num_heads, seq_len, seq_len) or (seq_len, seq_len)
    """
    c = weights.contiguous()
    shape = c.shape

    # Normalize to 2D matrix (seq_len, seq_len)
    if len(shape) == 4:
        # (batch, num_heads, q_len, k_len)
        q_len, k_len = shape[2], shape[3]
        stride_h = c.strides[1]
        stride_q = c.strides[2]
        stride_k = c.strides[3]
        base_offset = head_idx * stride_h
        matrix = [
            [c._data[base_offset + i * stride_q + j * stride_k] for j in range(k_len)]
            for i in range(q_len)
        ]
    elif len(shape) == 3:
        q_len, k_len = shape[1], shape[2]
        stride_h = c.strides[0]
        stride_q = c.strides[1]
        stride_k = c.strides[2]
        base_offset = head_idx * stride_h
        matrix = [
            [c._data[base_offset + i * stride_q + j * stride_k] for j in range(k_len)]
            for i in range(q_len)
        ]
    else:
        q_len, k_len = shape[0], shape[1]
        matrix = [
            [c._data[i * c.strides[0] + j * c.strides[1]] for j in range(k_len)]
            for i in range(q_len)
        ]

    # Format token labels (truncate or pad to 6 characters)
    def clean_tok(t: str) -> str:
        s = t.replace("\n", "↵").replace(" ", "␣")
        return (s[:5] if len(s) > 5 else s).ljust(5)

    q_labels = [clean_tok(t) for t in tokens[:q_len]]
    k_labels = [clean_tok(t) for t in tokens[:k_len]]

    lines = []
    lines.append(f"┌─ {title} (Head {head_idx}) " + "─" * max(10, (k_len * 3) - len(title) - 10) + "┐")

    # Header row with key tokens
    header = "       " + "".join(f"{k[:2]:>3}" for k in k_labels)
    lines.append(header)
    lines.append("       " + "───" * k_len)

    # Matrix rows
    for i, row in enumerate(matrix):
        label = q_labels[i] if i < len(q_labels) else f"T{i}".ljust(5)
        row_str = f"{label} │"
        for val in row:
            # Map val in [0.0, 1.0] to shade character index
            idx = min(int(val * len(SHADE_CHARS)), len(SHADE_CHARS) - 1)
            char = SHADE_CHARS[idx]
            # Print with two spaces for aspect ratio
            row_str += f" {char}{char}"
        lines.append(row_str)

    lines.append("└" + "─" * (k_len * 3 + 8) + "┘")
    lines.append(f"Scale: [0.0] {SHADE_CHARS[0]} {SHADE_CHARS[1]} {SHADE_CHARS[2]} {SHADE_CHARS[3]} {SHADE_CHARS[4]} {SHADE_CHARS[5]} [1.0]")
    return "\n".join(lines)


def render_probability_bars(top_candidates: List[Tuple[str, float]], bar_width: int = 25) -> str:
    """
    Renders an ASCII horizontal bar chart representing next-token probability distribution.
    top_candidates: list of (token_str, probability)
    """
    lines = []
    lines.append("┌─ Next-Token Distribution ──────────────────────────────┐")
    for i, (tok, prob) in enumerate(top_candidates):
        clean_tok = tok.replace("\n", "↵").replace(" ", "␣")
        tok_display = f"'{clean_tok}'"[:10].ljust(10)
        pct = prob * 100.0
        filled_len = int(prob * bar_width)
        bar = "█" * filled_len + "░" * (bar_width - filled_len)
        lines.append(f"│ #{i + 1} {tok_display} {pct:5.1f}% │{bar}│")
    lines.append("└────────────────────────────────────────────────────────┘")
    return "\n".join(lines)
