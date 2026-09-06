#!/usr/bin/env python3
"""
Interactive NanoTensor Inspector:
Allows live prompting, displays token-by-token generation with KV-cache,
and prints interactive ASCII attention matrix heatmaps and probability distributions.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanotensor.transformer import NanoTransformer
from nanotensor.tokenizer import BPETokenizer
from nanotensor.visualizer import render_attention_heatmap, render_probability_bars


def interactive_session():
    print("=" * 65)
    print("      NanoTensor: Live Architecture & Attention Inspector")
    print("=" * 65)

    # Initialize model and tokenizer
    tokenizer = BPETokenizer()
    sample_text = (
        "Hello world! Neural networks and transformers from scratch. "
        "Attention is all you need. Autograd and backpropagation."
    )
    tokenizer.train(sample_text, target_vocab_size=tokenizer.vocab_size + 15, min_freq=2)

    model = NanoTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=48
    )

    print(f"Tokenizer loaded with {tokenizer.vocab_size} tokens.")
    print("Enter a prompt (or press Enter for default 'Hello'):")

    try:
        user_prompt = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        user_prompt = ""

    if not user_prompt:
        user_prompt = "Hello"

    print(f"\nAnalyzing Prompt: '{user_prompt}'")
    tokens = tokenizer.encode(user_prompt)
    print(f"Token IDs: {tokens}")

    # Forward pass to obtain attention maps
    logits, weights = model.forward([tokens], return_weights=True)

    # Show next token probabilities
    last_logits = logits._data[-model.vocab_size:]
    # Softmax
    import math
    max_l = max(last_logits)
    exps = [math.exp(x - max_l) for x in last_logits]
    sum_e = sum(exps)
    probs = [e / sum_e for e in exps]

    indexed_probs = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:5]
    top_candidates = [(tokenizer.decode([idx]), p) for idx, p in indexed_probs]

    print("\n" + render_probability_bars(top_candidates))

    # Render attention heatmaps for Layer 1 and Layer 2
    token_strs = [tokenizer.decode([t]) for t in tokens]
    for layer_idx, w in enumerate(weights):
        if w is not None:
            print("\n" + render_attention_heatmap(
                w,
                tokens=token_strs,
                head_idx=0,
                title=f"Layer {layer_idx + 1} Attention Matrix"
            ))


if __name__ == "__main__":
    interactive_session()
