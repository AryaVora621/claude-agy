"""
Unit tests and recall verification for HNSW vector index:
Tests cosine and L2 distance metrics, multi-layer graph navigation,
recall accuracy against exact brute-force search, and persistence.
"""

import os
import random
import tempfile
import unittest
from chronodb.hnsw import HNSWIndex, cosine_distance, euclidean_distance


class TestHNSWIndex(unittest.TestCase):

    def test_hnsw_recall_vs_bruteforce(self):
        dim = 16
        num_vectors = 100
        k = 5

        index = HNSWIndex(dim=dim, metric="cosine", M=16, ef_construction=64, ef_search=32)

        random.seed(42)
        dataset = {}
        for i in range(num_vectors):
            vec = [random.gauss(0, 1) for _ in range(dim)]
            node_id = f"vec_{i}"
            dataset[node_id] = vec
            index.add(node_id, vec)

        # Run 10 queries and measure recall against exact brute-force k-NN
        recalls = []
        for q_idx in range(10):
            query = [random.gauss(0, 1) for _ in range(dim)]

            # Exact brute-force search
            all_scored = [(cosine_distance(query, v), nid) for nid, v in dataset.items()]
            all_scored.sort(key=lambda x: x[0])
            exact_top_k = set(nid for _, nid in all_scored[:k])

            # HNSW approximate search
            hnsw_results = index.search(query, k=k)
            hnsw_top_k = set(nid for nid, _ in hnsw_results)

            overlap = len(exact_top_k.intersection(hnsw_top_k))
            recall = overlap / k
            recalls.append(recall)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\nHNSW Mean Recall@{k} across {len(recalls)} queries: {avg_recall * 100:.1f}%")
        # Should achieve at least 80% recall on 100 vectors with default parameters
        self.assertGreaterEqual(avg_recall, 0.80)

    def test_hnsw_persistence(self):
        dim = 8
        index = HNSWIndex(dim=dim, metric="l2")
        index.add("v1", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        index.add("v2", [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        index.add("v3", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        with tempfile.NamedTemporaryFile(suffix=".hnsw.json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            index.save(tmp_path)
            loaded = HNSWIndex.load(tmp_path)

            results = loaded.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=2)
            self.assertEqual(len(results), 2)
            # Closest should be v1 (dist = 0), second closest v2
            self.assertEqual(results[0][0], "v1")
            self.assertEqual(results[1][0], "v2")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
