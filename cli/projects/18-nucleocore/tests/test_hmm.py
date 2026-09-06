"""
Unit tests for NucleoCore Hidden Markov Models and Profile HMM Subsystem.
"""

import unittest
from nucleocore.hmm import (
    HiddenMarkovModel,
    ProfileHMM,
    create_cpg_island_detector,
    detect_cpg_islands,
)


class TestHMM(unittest.TestCase):
    def test_viterbi_decoding_cossack_weather(self):
        # Classic Fair vs Loaded die model
        states = ["Fair", "Loaded"]
        alphabet = ["1", "2", "3", "4", "5", "6"]
        initial_probs = {"Fair": 0.5, "Loaded": 0.5}
        transition_probs = {
            "Fair": {"Fair": 0.9, "Loaded": 0.1},
            "Loaded": {"Fair": 0.1, "Loaded": 0.9}
        }
        emission_probs = {
            "Fair": {"1": 1/6, "2": 1/6, "3": 1/6, "4": 1/6, "5": 1/6, "6": 1/6},
            "Loaded": {"1": 0.1, "2": 0.1, "3": 0.1, "4": 0.1, "5": 0.1, "6": 0.5}
        }

        hmm = HiddenMarkovModel(states, alphabet, initial_probs, transition_probs, emission_probs)
        # 6s in a row should be decoded as Loaded
        obs = "666666"
        score, path = hmm.viterbi(obs)
        self.assertEqual(path, ["Loaded"] * 6)

    def test_posterior_probabilities(self):
        detector = create_cpg_island_detector()
        obs = "CGCGCGCGCG"
        posteriors = detector.posterior_probabilities(obs)
        self.assertEqual(len(posteriors), len(obs))
        for p_dict in posteriors:
            self.assertGreater(p_dict["CpG"], p_dict["NonCpG"])

    def test_profile_hmm_motif_scoring(self):
        # TATA box motif alignment
        motifs = [
            "TATAAA",
            "TATAAT",
            "TATAAA",
            "TATATA",
        ]
        phmm = ProfileHMM(motifs)
        # Exact match should have strong positive log-odds score
        score_match = phmm.score("TATAAA")
        self.assertGreater(score_match, 4.0)

        # Divergent sequence should score significantly lower
        score_bad = phmm.score("GCGCGC")
        self.assertLess(score_bad, 0.0)

    def test_detect_cpg_islands_segmentation(self):
        # Background DNA (AT rich) -> CpG island (GC rich) -> Background DNA (AT rich)
        bg1 = "ATATATATATATATATATAT" * 3  # 60 bp
        island = "CGCGCGCGCGCGCGCGCGCG" * 3 # 60 bp
        bg2 = "ATATATATATATATATATAT" * 3  # 60 bp

        seq = bg1 + island + bg2
        islands = detect_cpg_islands(seq)
        self.assertGreaterEqual(len(islands), 1)

        # Predicted island should overlap with the simulated island (indices 60 to 120)
        start, end = islands[0]
        self.assertLess(abs(start - 60), 10)
        self.assertLess(abs(end - 120), 10)


if __name__ == "__main__":
    unittest.main()
