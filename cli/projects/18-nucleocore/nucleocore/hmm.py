"""
NucleoCore: Hidden Markov Models (HMM) & Profile HMM Subsystem.
Includes log-space Viterbi decoding, Forward-Backward posterior probabilities,
Profile HMM for sequence motifs, and genomic CpG island segmenter.
"""

import math
from typing import List, Dict, Tuple, Optional, Any


NEG_INF = -1e9


def log_add(log_a: float, log_b: float) -> float:
    """Computes ln(exp(log_a) + exp(log_b)) without floating underflow."""
    if log_a <= NEG_INF:
        return log_b
    if log_b <= NEG_INF:
        return log_a
    if log_a > log_b:
        return log_a + math.log1p(math.exp(log_b - log_a))
    else:
        return log_b + math.log1p(math.exp(log_a - log_b))


class HiddenMarkovModel:
    """
    Standard discrete Hidden Markov Model with log-space inference.
    """
    def __init__(
        self,
        states: List[str],
        alphabet: List[str],
        initial_probs: Dict[str, float],
        transition_probs: Dict[str, Dict[str, float]],
        emission_probs: Dict[str, Dict[str, float]]
    ):
        self.states = states
        self.alphabet = set(alphabet)
        self.state_indices = {s: i for i, s in enumerate(states)}
        self.num_states = len(states)

        # Convert to log space
        self.log_initial: List[float] = [
            math.log(initial_probs.get(s, 1e-12)) for s in states
        ]
        self.log_trans: List[List[float]] = [
            [math.log(transition_probs.get(s1, {}).get(s2, 1e-12)) for s2 in states]
            for s1 in states
        ]
        self.log_emission: List[Dict[str, float]] = []
        for s in states:
            em_dict = {}
            for ch in alphabet:
                prob = emission_probs.get(s, {}).get(ch, 1e-12)
                em_dict[ch] = math.log(prob)
            self.log_emission.append(em_dict)

    def viterbi(self, observation: str) -> Tuple[float, List[str]]:
        """
        Computes the most probable path of hidden states for an observed sequence.
        Returns (max_log_likelihood, best_state_sequence).
        """
        t_len = len(observation)
        if t_len == 0:
            return 0.0, []

        k = self.num_states
        # V[t][s] = log probability of most likely path ending in state s at time t
        V = [[NEG_INF] * k for _ in range(t_len)]
        backpointer = [[0] * k for _ in range(t_len)]

        first_obs = observation[0]
        for s in range(k):
            em_log = self.log_emission[s].get(first_obs, NEG_INF)
            V[0][s] = self.log_initial[s] + em_log

        for t in range(1, t_len):
            obs = observation[t]
            for s in range(k):
                em_log = self.log_emission[s].get(obs, NEG_INF)
                best_prev_val = NEG_INF
                best_prev_state = 0
                for prev_s in range(k):
                    val = V[t - 1][prev_s] + self.log_trans[prev_s][s]
                    if val > best_prev_val:
                        best_prev_val = val
                        best_prev_state = prev_s
                V[t][s] = best_prev_val + em_log
                backpointer[t][s] = best_prev_state

        # Find best terminal state
        best_term_val = NEG_INF
        best_term_state = 0
        for s in range(k):
            if V[t_len - 1][s] > best_term_val:
                best_term_val = V[t_len - 1][s]
                best_term_state = s

        # Trace back
        path = [self.states[best_term_state]]
        curr_state = best_term_state
        for t in range(t_len - 1, 0, -1):
            curr_state = backpointer[t][curr_state]
            path.append(self.states[curr_state])

        path.reverse()
        return best_term_val, path

    def forward(self, observation: str) -> List[List[float]]:
        """Forward algorithm computing alpha table in log-space."""
        t_len = len(observation)
        k = self.num_states
        alpha = [[NEG_INF] * k for _ in range(t_len)]

        first_obs = observation[0]
        for s in range(k):
            alpha[0][s] = self.log_initial[s] + self.log_emission[s].get(first_obs, NEG_INF)

        for t in range(1, t_len):
            obs = observation[t]
            for s in range(k):
                em_log = self.log_emission[s].get(obs, NEG_INF)
                accum = NEG_INF
                for prev_s in range(k):
                    accum = log_add(accum, alpha[t - 1][prev_s] + self.log_trans[prev_s][s])
                alpha[t][s] = accum + em_log

        return alpha

    def posterior_probabilities(self, observation: str) -> List[Dict[str, float]]:
        """
        Computes the posterior probability of each hidden state at each position
        using the Forward-Backward algorithm.
        """
        t_len = len(observation)
        k = self.num_states
        alpha = self.forward(observation)

        # Backward algorithm (beta table)
        beta = [[NEG_INF] * k for _ in range(t_len)]
        for s in range(k):
            beta[t_len - 1][s] = 0.0  # ln(1) = 0

        for t in range(t_len - 2, -1, -1):
            next_obs = observation[t + 1]
            for s in range(k):
                accum = NEG_INF
                for next_s in range(k):
                    em_log = self.log_emission[next_s].get(next_obs, NEG_INF)
                    term = self.log_trans[s][next_s] + em_log + beta[t + 1][next_s]
                    accum = log_add(accum, term)
                beta[t][s] = accum

        # Total log-likelihood
        total_log_prob = NEG_INF
        for s in range(k):
            total_log_prob = log_add(total_log_prob, alpha[t_len - 1][s])

        posteriors: List[Dict[str, float]] = []
        for t in range(t_len):
            pos_dict = {}
            for s in range(k):
                log_p = alpha[t][s] + beta[t][s] - total_log_prob
                pos_dict[self.states[s]] = math.exp(log_p) if log_p > -30 else 0.0
            posteriors.append(pos_dict)

        return posteriors


class ProfileHMM:
    """
    Profile Hidden Markov Model for biological sequence motifs.
    Trained from an alignment of motif instances.
    """
    def __init__(self, motifs: List[str]):
        if not motifs:
            raise ValueError("Motifs list cannot be empty")
        self.length = len(motifs[0])
        self.alphabet = ["A", "C", "G", "T"]

        # Calculate position weight emission matrix with pseudocounts (+1 Laplace)
        self.match_emissions: List[Dict[str, float]] = []
        num_motifs = len(motifs)

        for col in range(self.length):
            counts = {b: 1.0 for b in self.alphabet} # Laplace pseudocount
            for seq in motifs:
                b = seq[col].upper()
                if b in counts:
                    counts[b] += 1.0
            total = sum(counts.values())
            self.match_emissions.append({b: counts[b] / total for b in self.alphabet})

        # Background null distribution (uniform 0.25)
        self.null_prob = 0.25

    def score(self, query: str) -> float:
        """
        Computes the log-odds score (in bits) of aligning query to the profile motif.
        Score = sum_i log2( P(query[i] | Match_i) / 0.25 ).
        """
        if len(query) != self.length:
            raise ValueError(f"Query length {len(query)} must match profile length {self.length}")

        bits = 0.0
        for i, b in enumerate(query.upper()):
            prob = self.match_emissions[i].get(b, 0.01)
            bits += math.log2(prob / self.null_prob)
        return bits


def create_cpg_island_detector() -> HiddenMarkovModel:
    """
    Factory creating a calibrated 2-state HMM for identifying genomic CpG islands.
    States: 'CpG' (island) and 'NonCpG' (background genomic DNA).
    """
    states = ["CpG", "NonCpG"]
    alphabet = ["A", "C", "G", "T"]

    initial_probs = {
        "CpG": 0.05,
        "NonCpG": 0.95
    }

    # Transition matrix: islands are sticky (stay in island 98% of the time)
    transition_probs = {
        "CpG": {"CpG": 0.98, "NonCpG": 0.02},
        "NonCpG": {"CpG": 0.01, "NonCpG": 0.99}
    }

    # Emission frequencies: CpG islands have elevated G+C (60%) vs background (40%)
    emission_probs = {
        "CpG": {"A": 0.15, "C": 0.35, "G": 0.35, "T": 0.15},
        "NonCpG": {"A": 0.30, "C": 0.20, "G": 0.20, "T": 0.30}
    }

    return HiddenMarkovModel(
        states=states,
        alphabet=alphabet,
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emission_probs=emission_probs
    )


def detect_cpg_islands(sequence: str) -> List[Tuple[int, int]]:
    """
    Segment genomic sequence into (start, end) coordinate intervals of predicted CpG islands.
    """
    detector = create_cpg_island_detector()
    _, states = detector.viterbi(sequence.upper())

    islands: List[Tuple[int, int]] = []
    in_island = False
    start_idx = 0

    for idx, state in enumerate(states):
        if state == "CpG" and not in_island:
            in_island = True
            start_idx = idx
        elif state == "NonCpG" and in_island:
            in_island = False
            islands.append((start_idx, idx))

    if in_island:
        islands.append((start_idx, len(states)))

    return islands
