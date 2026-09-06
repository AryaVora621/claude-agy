"""
NucleoCore: Burrows-Wheeler Transform (BWT) & FM-Index Subsystem.
Includes Suffix Array construction, checkpointed Occurrence table, LF-mapping,
exact Ferragina-Manzini backward search, and inexact search with mismatches.
"""

from typing import List, Tuple, Dict, Set, Optional


def build_suffix_array(text: str) -> List[int]:
    """
    Constructs the Suffix Array (SA) for a string text ending with sentinel '$'.
    Returns a list of start indices sorted lexicographically.
    """
    n = len(text)
    # Python's Timsort on string slices is exceptionally fast for moderate N
    suffixes = sorted(range(n), key=lambda i: text[i:])
    return suffixes


class FMIndex:
    """
    Full-text Minute (FM) Index over a reference DNA sequence.
    Provides O(m) exact pattern search and location extraction via sampled Suffix Array.
    """
    def __init__(
        self,
        reference: str,
        occ_stride: int = 32,
        sa_sample_rate: int = 16
    ):
        self.raw_reference = reference
        # Ensure sentinel '$' is appended (lexicographically smallest character)
        if not reference.endswith("$"):
            self.text = reference + "$"
        else:
            self.text = reference
        self.n = len(self.text)
        self.occ_stride = max(1, occ_stride)
        self.sa_sample_rate = max(1, sa_sample_rate)

        # 1. Build Suffix Array
        full_sa = build_suffix_array(self.text)

        # 2. Build BWT string from Suffix Array
        bwt_chars = []
        for sa_idx in full_sa:
            if sa_idx == 0:
                bwt_chars.append("$")
            else:
                bwt_chars.append(self.text[sa_idx - 1])
        self.bwt = "".join(bwt_chars)

        # 3. Alphabet and Count Array C
        # C[c] is the number of characters in text strictly smaller than c
        char_counts: Dict[str, int] = {}
        for ch in self.bwt:
            char_counts[ch] = char_counts.get(ch, 0) + 1

        self.alphabet = sorted(char_counts.keys())
        self.C: Dict[str, int] = {}
        running_total = 0
        for ch in self.alphabet:
            self.C[ch] = running_total
            running_total += char_counts[ch]

        # 4. Checkpointed Occurrence Table Occ(c, i)
        # Store counts at every occ_stride rows
        self._occ_checkpoints: List[Dict[str, int]] = []
        running_occ = {ch: 0 for ch in self.alphabet}
        for i, ch in enumerate(self.bwt):
            if i % self.occ_stride == 0:
                self._occ_checkpoints.append(dict(running_occ))
            running_occ[ch] += 1
        self._occ_checkpoints.append(dict(running_occ))

        # 5. Sampled Suffix Array
        # Store SA[i] only when i % sa_sample_rate == 0
        self._sa_samples: Dict[int, int] = {}
        for i in range(0, self.n, self.sa_sample_rate):
            self._sa_samples[i] = full_sa[i]

    def occ(self, char: str, index: int) -> int:
        """
        Returns the number of occurrences of character in BWT[0 ... index-1].
        O(1) with checkpointed lookup + short linear scan.
        """
        if char not in self.C:
            return 0
        if index <= 0:
            return 0
        if index > self.n:
            index = self.n

        ckpt_idx = index // self.occ_stride
        base_count = self._occ_checkpoints[ckpt_idx][char]
        start_idx = ckpt_idx * self.occ_stride

        # Scan residual entries up to index
        for k in range(start_idx, index):
            if self.bwt[k] == char:
                base_count += 1
        return base_count

    def lf_mapping(self, row: int) -> int:
        """Computes the Last-to-First (LF) column mapping for a row index in BWT."""
        ch = self.bwt[row]
        return self.C[ch] + self.occ(ch, row)

    def backward_search_range(self, pattern: str) -> Tuple[int, int]:
        """
        Executes Ferragina-Manzini backward search for pattern.
        Returns the half-open row interval [l, r) in the Suffix Array containing the match.
        If not found, returns (0, 0) or l >= r.
        """
        if not pattern:
            return 0, self.n

        # Start with full range
        l = 0
        r = self.n

        # Walk pattern backwards from last character to first
        for i in range(len(pattern) - 1, -1, -1):
            ch = pattern[i]
            if ch not in self.C:
                return 0, 0
            l = self.C[ch] + self.occ(ch, l)
            r = self.C[ch] + self.occ(ch, r)
            if l >= r:
                return 0, 0

        return l, r

    def count(self, pattern: str) -> int:
        """Returns the number of exact occurrences of pattern in the reference sequence."""
        l, r = self.backward_search_range(pattern)
        return max(0, r - l)

    def locate_row(self, row: int) -> int:
        """
        Resolves the reference genome offset for a given row in the Suffix Array
        using backward LF-mapping steps until a sampled SA entry is hit.
        """
        steps = 0
        curr_row = row
        while curr_row not in self._sa_samples:
            curr_row = self.lf_mapping(curr_row)
            steps += 1
        return (self._sa_samples[curr_row] + steps) % self.n

    def locate(self, pattern: str) -> List[int]:
        """
        Finds all 0-indexed start positions of exact occurrences of pattern in the reference.
        Returned positions are sorted in ascending order.
        """
        l, r = self.backward_search_range(pattern)
        if l >= r:
            return []

        positions = []
        for row in range(l, r):
            pos = self.locate_row(row)
            positions.append(pos)
        positions.sort()
        return positions

    def search_inexact(self, pattern: str, max_mismatches: int = 1) -> List[Tuple[int, int]]:
        """
        Finds occurrences of pattern in the reference with up to max_mismatches substitutions.
        Returns list of (position, mismatches) tuples.
        """
        results: Dict[int, int] = {}

        def _recurse(p_idx: int, l: int, r: int, mismatches_left: int):
            if p_idx < 0:
                # Reached beginning of pattern with valid SA interval
                for row in range(l, r):
                    pos = self.locate_row(row)
                    mm_used = max_mismatches - mismatches_left
                    if pos not in results or mm_used < results[pos]:
                        results[pos] = mm_used
                return

            curr_char = pattern[p_idx]
            # Try all characters in the alphabet
            for ch in self.alphabet:
                if ch == "$":
                    continue
                is_match = (ch == curr_char)
                if not is_match and mismatches_left == 0:
                    continue

                new_l = self.C[ch] + self.occ(ch, l)
                new_r = self.C[ch] + self.occ(ch, r)
                if new_l < new_r:
                    next_mm = mismatches_left if is_match else (mismatches_left - 1)
                    _recurse(p_idx - 1, new_l, new_r, next_mm)

        _recurse(len(pattern) - 1, 0, self.n, max_mismatches)
        sorted_res = sorted(results.items(), key=lambda x: (x[0], x[1]))
        return sorted_res

    def reconstruct_text(self) -> str:
        """
        Reconstructs the original reference text using LF-mapping from BWT.
        Demonstrates complete mathematical reversibility of the transform.
        """
        # Start at row with '$'
        dollar_row = self.bwt.index("$")
        curr_row = dollar_row
        reconstructed = []

        for _ in range(self.n - 1):
            curr_row = self.lf_mapping(curr_row)
            reconstructed.append(self.bwt[curr_row])

        # Reverse since we walked backwards
        return "".join(reversed(reconstructed))
