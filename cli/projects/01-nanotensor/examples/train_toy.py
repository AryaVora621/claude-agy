#!/usr/bin/env python3
"""
NanoTensor Toy Training & Attention Visualization Demo:
Trains a micro-NanoTransformer on algorithmic arithmetic sequences,
visualizes loss descent, and renders real-time ASCII attention heatmaps.
"""

import sys
import os

# Add parent directory to path to import nanotensor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nanotensor.tensor import Tensor
from nanotensor.transformer import NanoTransformer, cross_entropy_loss
from nanotensor.tokenizer import BPETokenizer
from nanotensor.optim import AdamW, clip_grad_norm_
from nanotensor.visualizer import render_attention_heatmap, render_probability_bars


def main():
    print("=" * 70)
    print("   NanoTensor: Zero-Dependency Modern Transformer Training Demo")
    print("=" * 70)

    # 1. Prepare Toy Corpus
    corpus = [
        "1+1=2;", "1+2=3;", "1+3=4;", "2+1=3;", "2+2=4;", "2+3=5;",
        "3+1=4;", "3+2=5;", "3+3=6;", "4+1=5;", "4+2=6;", "5+1=6;",
    ]
    full_text = " ".join(corpus)

    print("\n[Step 1] Initializing and Training Byte-Pair Encoding Tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(full_text, target_vocab_size=tokenizer.vocab_size + 10, min_freq=2)
    print(f"-> Vocabulary Size: {tokenizer.vocab_size} (256 Base Bytes + Special + Learned Merges)")

    # 2. Build NanoTransformer
    print("\n[Step 2] Instantiating NanoTransformer Architecture:")
    print("-> Model Spec: 2 Layers | Dim 32 | 4 Heads | SwiGLU FFN | RMSNorm | RoPE | KV-Cache")
    model = NanoTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=48
    )

    optimizer = AdamW(model.parameters(), lr=0.015, weight_decay=0.01)

    # 3. Training Loop
    print("\n[Step 3] Training over synthetic arithmetic sequences...")
    tokenized_samples = [tokenizer.encode(item) for item in corpus]

    for epoch in range(1, 31):
        total_loss = 0.0
        for seq in tokenized_samples:
            if len(seq) < 2:
                continue

            # Input tokens x, target tokens y (shifted by 1)
            inputs = [seq[:-1]]
            targets = seq[1:]

            optimizer.zero_grad()
            logits, _ = model.forward(inputs)

            loss = cross_entropy_loss(logits, targets)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss._data[0]

        avg_loss = total_loss / len(tokenized_samples)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/30 | Mean Cross-Entropy Loss: {avg_loss:.4f}")

    # 4. Inference and Generation with KV-Cache
    print("\n[Step 4] Running Generative Inference with KV-Caching:")
    test_prompt = "2+2="
    prompt_tokens = tokenizer.encode(test_prompt)
    print(f"-> Prompt: '{test_prompt}' (Tokens: {prompt_tokens})")

    # Generate next tokens
    generated_tokens = model.generate(
        prompt=prompt_tokens,
        max_new_tokens=2,
        temperature=0.1,  # Near-greedy
        top_k=3,
        use_cache=True
    )

    decoded_result = tokenizer.decode(generated_tokens)
    print(f"-> Model Output: '{decoded_result}'")

    # 5. Extract and Render Attention Matrix
    print("\n[Step 5] Rendering Attention Matrix Heatmap for Final Layer:")
    _, weights = model.forward([generated_tokens], return_weights=True)
    if weights and weights[-1] is not None:
        token_strs = [tokenizer.decode([t]) for t in generated_tokens]
        heatmap = render_attention_heatmap(
            weights=weights[-1],
            tokens=token_strs,
            head_idx=0,
            title="Layer 2 Causal Self-Attention"
        )
        print(heatmap)

    print("\n[Done] Training and verification completed successfully!")


if __name__ == "__main__":
    main()
