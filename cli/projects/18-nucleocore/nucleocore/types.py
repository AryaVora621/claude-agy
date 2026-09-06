"""
NucleoCore: Genomics Types, DNA Sequences, CIGAR Strings, and Scoring Matrices.
Pure Python standard library implementation with zero external dependencies.
"""

from typing import List, Tuple, Dict, Optional
from enum import Enum


COMPLEMENT_MAP: Dict[str, str] = {
    "A": "T", "T": "A",
    "C": "G", "G": "C",
    "a": "t", "t": "a",
    "c": "g", "g": "c",
    "N": "N", "n": "n",
    # Degenerate IUPAC bases
    "R": "Y", "Y": "R",
    "S": "S", "W": "W",
    "K": "M", "M": "K",
    "B": "V", "V": "B",
    "D": "H", "H": "D",
    "$": "$",
}


def reverse_complement(seq: str) -> str:
    """Computes the 5'-to-3' reverse complement of a DNA sequence."""
    return "".join(COMPLEMENT_MAP.get(base, "N") for base in reversed(seq))


def gc_content(seq: str) -> float:
    """Calculates the GC content fraction (0.0 to 1.0) of a nucleotide sequence."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    return gc_count / len(seq)


def kmer_spectrum(seq: str, k: int) -> List[str]:
    """Extracts all contiguous k-mers of length k from a sequence."""
    if len(seq) < k or k <= 0:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def canonical_kmer(kmer: str) -> str:
    """Returns the lexicographical minimum between a k-mer and its reverse complement."""
    rc = reverse_complement(kmer)
    return kmer if kmer <= rc else rc


class CIGAROpType(str, Enum):
    MATCH = "M"        # Alignment match (can be sequence match or mismatch)
    INSERTION = "I"    # Insertion to the reference
    DELETION = "D"     # Deletion from the reference
    SEQ_MATCH = "="    # Exact sequence match
    SEQ_MISMATCH = "X" # Sequence mismatch


class CIGARElement:
    """Represents a single CIGAR run, e.g. 10M, 2I, 3D."""
    __slots__ = ("op", "length")

    def __init__(self, op: str, length: int):
        self.op = op
        self.length = length

    def __repr__(self) -> str:
        return f"{self.length}{self.op}"


class CIGAR:
    """CIGAR alignment representation with parser, length calculator, and string formatter."""
    def __init__(self, elements: Optional[List[CIGARElement]] = None):
        self.elements: List[CIGARElement] = elements if elements is not None else []

    def add(self, op: str, length: int = 1) -> None:
        if length <= 0:
            return
        if self.elements and self.elements[-1].op == op:
            self.elements[-1].length += length
        else:
            self.elements.append(CIGARElement(op, length))

    def reference_length(self) -> int:
        """Computes the span along the reference sequence (M, D, =, X consume reference)."""
        length = 0
        for elem in self.elements:
            if elem.op in ("M", "D", "=", "X"):
                length += elem.length
        return length

    def query_length(self) -> int:
        """Computes the span along the query sequence (M, I, =, X consume query)."""
        length = 0
        for elem in self.elements:
            if elem.op in ("M", "I", "=", "X"):
                length += elem.length
        return length

    @classmethod
    def from_string(cls, cigar_str: str) -> "CIGAR":
        cigar = cls()
        num_buf = ""
        for ch in cigar_str:
            if ch.isdigit():
                num_buf += ch
            elif ch.isalpha() or ch in ("=",):
                if num_buf:
                    cigar.add(ch, int(num_buf))
                    num_buf = ""
        return cigar

    def to_string(self) -> str:
        return "".join(str(e) for e in self.elements)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"CIGAR('{self.to_string()}')"


class ScoringMatrix:
    """Nucleotide substitution scoring matrix with transition/transversion differentiation."""
    def __init__(
        self,
        match: int = 2,
        mismatch: int = -1,
        transition: Optional[int] = None,
        transversion: Optional[int] = None
    ):
        self.match = match
        self.mismatch = mismatch
        self.transition = transition if transition is not None else mismatch
        self.transversion = transversion if transversion is not None else mismatch

        # Transitions: A <-> G, C <-> T
        self._transitions = {
            ("A", "G"), ("G", "A"),
            ("C", "T"), ("T", "C")
        }

    def score(self, a: str, b: str) -> int:
        a_up = a.upper()
        b_up = b.upper()
        if a_up == b_up:
            return self.match
        pair = (a_up, b_up)
        if pair in self._transitions:
            return self.transition
        return self.transversion
