"""
Byte-Pair Encoding (BPE) Tokenizer implemented from first principles.
Starts from raw byte level (0-255) to guarantee zero out-of-vocabulary issues for UTF-8 text,
learns merge rules by frequency statistics, and supports serialization.
"""

from __future__ import annotations
import json
from typing import Dict, List, Tuple, Optional


class BPETokenizer:
    """
    Byte-level BPE Tokenizer.
    Base vocabulary: 256 UTF-8 byte tokens + special tokens.
    Learns merge operations from training text corpora.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        # Special tokens
        self.special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self.special_to_id = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.id_to_special = {idx: tok for idx, tok in enumerate(self.special_tokens)}

        self.num_special = len(self.special_tokens)

        # Base byte tokens: mapped to IDs starting from num_special
        self.byte_to_id = {b: self.num_special + b for b in range(256)}
        self.id_to_byte = {self.num_special + b: b for b in range(256)}

        # Learned merges: (token_id_a, token_id_b) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}

        # Initialize base vocab bytes
        for b in range(256):
            self.vocab[self.num_special + b] = bytes([b])

    @property
    def pad_id(self) -> int:
        return self.special_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.special_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.special_to_id[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return self.num_special + 256 + len(self.merges)

    def train(self, text: str, target_vocab_size: int = 300, min_freq: int = 2) -> None:
        """
        Learns BPE merge rules from an input text corpus until target_vocab_size is reached.
        """
        raw_bytes = text.encode("utf-8")
        tokens = [self.byte_to_id[b] for b in raw_bytes]

        next_token_id = self.num_special + 256

        while self.vocab_size < target_vocab_size:
            # Count bigram frequencies
            pair_counts: Dict[Tuple[int, int], int] = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_freq:
                # Frequency below threshold, stop learning
                break

            # Register new merge rule
            self.merges[best_pair] = next_token_id

            # Construct byte representation
            bytes_a = self.vocab[best_pair[0]]
            bytes_b = self.vocab[best_pair[1]]
            self.vocab[next_token_id] = bytes_a + bytes_b

            # Apply merge across current token sequence
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(next_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            next_token_id += 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encodes UTF-8 text string into a list of token IDs."""
        raw_bytes = text.encode("utf-8")
        tokens = [self.byte_to_id[b] for b in raw_bytes]

        # Iteratively apply learned merges in order of acquisition
        for pair, merged_id in self.merges.items():
            if len(tokens) < 2:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        res = []
        if add_bos:
            res.append(self.bos_id)
        res.extend(tokens)
        if add_eos:
            res.append(self.eos_id)
        return res

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decodes a list of token IDs back into a UTF-8 string."""
        byte_chunks = []
        for tid in token_ids:
            if tid in self.id_to_special:
                if not skip_special:
                    byte_chunks.append(self.id_to_special[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                # Unknown fallback
                byte_chunks.append(b"?")

        combined = b"".join(byte_chunks)
        return combined.decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        """Serializes tokenizer merges and metadata to JSON."""
        serializable_merges = [
            {"pair": list(pair), "id": merged_id}
            for pair, merged_id in self.merges.items()
        ]
        data = {
            "special_tokens": self.special_tokens,
            "merges": serializable_merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> BPETokenizer:
        """Loads tokenizer from saved JSON file."""
        tokenizer = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        next_token_id = tokenizer.num_special + 256
        for item in data["merges"]:
            pair = tuple(item["pair"])
            merged_id = item["id"]
            tokenizer.merges[pair] = merged_id
            bytes_a = tokenizer.vocab[pair[0]]
            bytes_b = tokenizer.vocab[pair[1]]
            tokenizer.vocab[merged_id] = bytes_a + bytes_b
        return tokenizer
