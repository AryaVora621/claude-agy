"""
NucleoCore: Pairwise Sequence Alignment Engine with Gotoh Affine Gaps.
Supports Needleman-Wunsch (Global) and Smith-Waterman (Local) alignments
with CIGAR string generation and customizable substitution scoring.
"""

from typing import Tuple, Optional, List
from enum import Enum
from nucleocore.types import ScoringMatrix, CIGAR, CIGAROpType


class AlignmentMode(str, Enum):
    GLOBAL = "GLOBAL"          # Needleman-Wunsch
    LOCAL = "LOCAL"            # Smith-Waterman
    SEMI_GLOBAL = "SEMI_GLOBAL"# Free end gaps (fitting alignment)


class AlignmentResult:
    """Encapsulates the optimal pairwise alignment between two sequences."""
    __slots__ = (
        "mode",
        "score",
        "aligned_seq1",
        "aligned_seq2",
        "cigar",
        "start1",
        "end1",
        "start2",
        "end2",
        "matches",
        "mismatches",
        "gaps",
        "identity",
    )

    def __init__(
        self,
        mode: AlignmentMode,
        score: int,
        aligned_seq1: str,
        aligned_seq2: str,
        cigar: CIGAR,
        start1: int,
        end1: int,
        start2: int,
        end2: int,
        matches: int,
        mismatches: int,
        gaps: int,
    ):
        self.mode = mode
        self.score = score
        self.aligned_seq1 = aligned_seq1
        self.aligned_seq2 = aligned_seq2
        self.cigar = cigar
        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.matches = matches
        self.mismatches = mismatches
        self.gaps = gaps
        total_len = len(aligned_seq1)
        self.identity = (matches / total_len) if total_len > 0 else 0.0

    def format_alignment(self, line_width: int = 60) -> str:
        """Formats the pairwise alignment into visual wrapped text blocks with match symbols."""
        lines = []
        lines.append(f"Alignment [{self.mode.value}] Score: {self.score} | Identity: {self.identity*100:.1f}% | CIGAR: {self.cigar.to_string()}")
        lines.append(f"Seq1: [{self.start1}:{self.end1}] | Seq2: [{self.start2}:{self.end2}]")
        lines.append("-" * 70)

        # Build match consensus line
        consensus = []
        for c1, c2 in zip(self.aligned_seq1, self.aligned_seq2):
            if c1 == "-" or c2 == "-":
                consensus.append(" ")
            elif c1.upper() == c2.upper():
                consensus.append("|")
            else:
                consensus.append(".")
        cons_str = "".join(consensus)

        length = len(self.aligned_seq1)
        for offset in range(0, length, line_width):
            chunk1 = self.aligned_seq1[offset:offset + line_width]
            chunk_m = cons_str[offset:offset + line_width]
            chunk2 = self.aligned_seq2[offset:offset + line_width]

            lines.append(f"Seq1  {offset:5d}  {chunk1}")
            lines.append(f"             {chunk_m}")
            lines.append(f"Seq2  {offset:5d}  {chunk2}")
            lines.append("")

        return "\n".join(lines)


# Traceback constants
TB_NONE = 0
TB_M_DIAG = 1
TB_M_FROM_IX = 2
TB_M_FROM_IY = 3
TB_IX_OPEN = 4
TB_IX_EXT = 5
TB_IY_OPEN = 6
TB_IY_EXT = 7

NEG_INF = -10_000_000


def align_pairwise(
    seq1: str,
    seq2: str,
    mode: AlignmentMode = AlignmentMode.GLOBAL,
    scoring: Optional[ScoringMatrix] = None,
    gap_open: int = 5,
    gap_extend: int = 2
) -> AlignmentResult:
    """
    Executes Gotoh's 3-matrix affine gap alignment in O(N * M) time.
    """
    if scoring is None:
        scoring = ScoringMatrix(match=2, mismatch=-2)

    n = len(seq1)
    m = len(seq2)

    # 3 DP matrices: M, Ix (gap in seq2), Iy (gap in seq1)
    M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Ix = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Iy = [[NEG_INF] * (m + 1) for _ in range(n + 1)]

    tb_M = [[TB_NONE] * (m + 1) for _ in range(n + 1)]
    tb_Ix = [[TB_NONE] * (m + 1) for _ in range(n + 1)]
    tb_Iy = [[TB_NONE] * (m + 1) for _ in range(n + 1)]

    # Boundary conditions
    if mode == AlignmentMode.GLOBAL:
        M[0][0] = 0
        for i in range(1, n + 1):
            Ix[i][0] = -gap_open - i * gap_extend
            tb_Ix[i][0] = TB_IX_EXT if i > 1 else TB_IX_OPEN
        for j in range(1, m + 1):
            Iy[0][j] = -gap_open - j * gap_extend
            tb_Iy[0][j] = TB_IY_EXT if j > 1 else TB_IY_OPEN
    elif mode == AlignmentMode.LOCAL:
        for i in range(n + 1):
            M[i][0] = 0
        for j in range(m + 1):
            M[0][j] = 0
    elif mode == AlignmentMode.SEMI_GLOBAL:
        # Free end gaps on boundaries
        for i in range(n + 1):
            M[i][0] = 0
        for j in range(m + 1):
            M[0][j] = 0

    max_score = NEG_INF if mode != AlignmentMode.LOCAL else 0
    max_pos = (0, 0)
    max_matrix = "M"

    for i in range(1, n + 1):
        c1 = seq1[i - 1]
        for j in range(1, m + 1):
            c2 = seq2[j - 1]
            sub_score = scoring.score(c1, c2)

            # Update Ix (gap in seq2 / vertical move)
            open_ix = M[i - 1][j] - gap_open - gap_extend
            ext_ix = Ix[i - 1][j] - gap_extend
            if open_ix >= ext_ix:
                Ix[i][j] = open_ix
                tb_Ix[i][j] = TB_IX_OPEN
            else:
                Ix[i][j] = ext_ix
                tb_Ix[i][j] = TB_IX_EXT

            # Update Iy (gap in seq1 / horizontal move)
            open_iy = M[i][j - 1] - gap_open - gap_extend
            ext_iy = Iy[i][j - 1] - gap_extend
            if open_iy >= ext_iy:
                Iy[i][j] = open_iy
                tb_Iy[i][j] = TB_IY_OPEN
            else:
                Iy[i][j] = ext_iy
                tb_Iy[i][j] = TB_IY_EXT

            # Update M (match / mismatch)
            from_m = M[i - 1][j - 1]
            from_ix = Ix[i - 1][j - 1]
            from_iy = Iy[i - 1][j - 1]

            best_prev = from_m
            best_tb = TB_M_DIAG
            if from_ix > best_prev:
                best_prev = from_ix
                best_tb = TB_M_FROM_IX
            if from_iy > best_prev:
                best_prev = from_iy
                best_tb = TB_M_FROM_IY

            curr_m = best_prev + sub_score
            if mode == AlignmentMode.LOCAL:
                if curr_m <= 0:
                    M[i][j] = 0
                    tb_M[i][j] = TB_NONE
                else:
                    M[i][j] = curr_m
                    tb_M[i][j] = best_tb
            else:
                M[i][j] = curr_m
                tb_M[i][j] = best_tb

            if mode == AlignmentMode.LOCAL:
                if M[i][j] > max_score:
                    max_score = M[i][j]
                    max_pos = (i, j)
                    max_matrix = "M"

    # Traceback resolution
    aligned_s1 = []
    aligned_s2 = []
    cigar_ops = []

    if mode == AlignmentMode.GLOBAL:
        # Determine whether M, Ix, or Iy is maximum at (n, m)
        curr_i, curr_j = n, m
        scores = [M[n][m], Ix[n][m], Iy[n][m]]
        best_val = max(scores)
        if best_val == M[n][m]:
            curr_mat = "M"
        elif best_val == Ix[n][m]:
            curr_mat = "Ix"
        else:
            curr_mat = "Iy"
        final_score = best_val
        start1, end1 = 0, n
        start2, end2 = 0, m

        while curr_i > 0 or curr_j > 0:
            if curr_mat == "M":
                tb = tb_M[curr_i][curr_j]
                aligned_s1.append(seq1[curr_i - 1])
                aligned_s2.append(seq2[curr_j - 1])
                cigar_ops.append("M")
                curr_i -= 1
                curr_j -= 1
                if tb == TB_M_FROM_IX:
                    curr_mat = "Ix"
                elif tb == TB_M_FROM_IY:
                    curr_mat = "Iy"
                else:
                    curr_mat = "M"
            elif curr_mat == "Ix":
                aligned_s1.append(seq1[curr_i - 1])
                aligned_s2.append("-")
                cigar_ops.append("D") # Deletion in seq2
                tb = tb_Ix[curr_i][curr_j]
                curr_i -= 1
                if tb == TB_IX_OPEN:
                    curr_mat = "M"
                else:
                    curr_mat = "Ix"
            elif curr_mat == "Iy":
                aligned_s1.append("-")
                aligned_s2.append(seq2[curr_j - 1])
                cigar_ops.append("I") # Insertion in seq2
                tb = tb_Iy[curr_i][curr_j]
                curr_j -= 1
                if tb == TB_IY_OPEN:
                    curr_mat = "M"
                else:
                    curr_mat = "Iy"

    elif mode == AlignmentMode.LOCAL:
        curr_i, curr_j = max_pos
        curr_mat = max_matrix
        final_score = max_score
        end1 = curr_i
        end2 = curr_j

        while curr_i > 0 and curr_j > 0 and M[curr_i][curr_j] > 0:
            if curr_mat == "M":
                tb = tb_M[curr_i][curr_j]
                if tb == TB_NONE:
                    break
                aligned_s1.append(seq1[curr_i - 1])
                aligned_s2.append(seq2[curr_j - 1])
                cigar_ops.append("M")
                curr_i -= 1
                curr_j -= 1
                if tb == TB_M_FROM_IX:
                    curr_mat = "Ix"
                elif tb == TB_M_FROM_IY:
                    curr_mat = "Iy"
                else:
                    curr_mat = "M"
            elif curr_mat == "Ix":
                aligned_s1.append(seq1[curr_i - 1])
                aligned_s2.append("-")
                cigar_ops.append("D")
                tb = tb_Ix[curr_i][curr_j]
                curr_i -= 1
                if tb == TB_IX_OPEN:
                    curr_mat = "M"
                else:
                    curr_mat = "Ix"
            elif curr_mat == "Iy":
                aligned_s1.append("-")
                aligned_s2.append(seq2[curr_j - 1])
                cigar_ops.append("I")
                tb = tb_Iy[curr_i][curr_j]
                curr_j -= 1
                if tb == TB_IY_OPEN:
                    curr_mat = "M"
                else:
                    curr_mat = "Iy"

        start1 = curr_i
        start2 = curr_j
    else: # SEMI_GLOBAL
        curr_i, curr_j = n, m
        final_score = M[n][m]
        start1, end1 = 0, n
        start2, end2 = 0, m
        curr_mat = "M"
        while curr_i > 0 and curr_j > 0:
            aligned_s1.append(seq1[curr_i - 1])
            aligned_s2.append(seq2[curr_j - 1])
            cigar_ops.append("M")
            curr_i -= 1
            curr_j -= 1

    aligned_s1.reverse()
    aligned_s2.reverse()
    cigar_ops.reverse()

    cigar = CIGAR()
    for op in cigar_ops:
        cigar.add(op, 1)

    matches = 0
    mismatches = 0
    gaps = 0
    for a, b in zip(aligned_s1, aligned_s2):
        if a == "-" or b == "-":
            gaps += 1
        elif a.upper() == b.upper():
            matches += 1
        else:
            mismatches += 1

    return AlignmentResult(
        mode=mode,
        score=final_score,
        aligned_seq1="".join(aligned_s1),
        aligned_seq2="".join(aligned_s2),
        cigar=cigar,
        start1=start1,
        end1=end1,
        start2=start2,
        end2=end2,
        matches=matches,
        mismatches=mismatches,
        gaps=gaps,
    )
