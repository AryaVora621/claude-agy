"""
Unit tests for scratch Byte-Pair Encoding (BPE) tokenizer.
Verifies merge rule learning, UTF-8 lossless round-tripping, and JSON persistence.
"""

import os
import tempfile
import unittest
from nanotensor.tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = BPETokenizer()
        self.corpus = (
            "The quick brown fox jumps over the lazy dog. "
            "A fast dark creature leapt above the sleeping hound. "
            "Transformers and autograd engines built from scratch are fascinating."
        )

    def test_bpe_training_and_merges(self):
        initial_vocab_size = self.tokenizer.vocab_size
        self.tokenizer.train(self.corpus, target_vocab_size=initial_vocab_size + 20, min_freq=2)
        # Should have learned merges
        self.assertGreater(len(self.tokenizer.merges), 0)
        self.assertGreater(self.tokenizer.vocab_size, initial_vocab_size)

    def test_lossless_roundtrip(self):
        self.tokenizer.train(self.corpus, target_vocab_size=self.tokenizer.vocab_size + 15, min_freq=2)
        sample_texts = [
            "The quick brown fox",
            "autograd from scratch!",
            "12345 Hello World ✨ \n\t Unicode characters: ä ö ü 🚀",
        ]
        for text in sample_texts:
            tokens = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(tokens)
            self.assertEqual(text, decoded, f"Failed round-trip for text: {text}")

    def test_save_and_load(self):
        self.tokenizer.train(self.corpus, target_vocab_size=self.tokenizer.vocab_size + 10, min_freq=2)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            loaded = BPETokenizer.load(tmp_path)
            self.assertEqual(self.tokenizer.vocab_size, loaded.vocab_size)
            self.assertEqual(len(self.tokenizer.merges), len(loaded.merges))

            test_str = "Testing tokenizer serialization"
            self.assertEqual(self.tokenizer.encode(test_str), loaded.encode(test_str))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
