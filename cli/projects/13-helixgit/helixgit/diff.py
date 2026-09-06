"""
HelixGit: Myers O((N+M)D) Diff Engine and Unified Diff Formatter.
Implements the Eugene Myers shortest edit script (SES) graph search
and Git-compatible unified diff generation with hunk context.
"""

from typing import List, Tuple, Dict, Optional, NamedTuple
from enum import Enum


class DiffType(Enum):
    KEEP = "keep"
    INSERT = "insert"
    DELETE = "delete"


class DiffLine(NamedTuple):
    type: DiffType
    text: str
    old_lineno: Optional[int]
    new_lineno: Optional[int]


def myers_diff(a: List[str], b: List[str]) -> List[DiffLine]:
    """
    Computes optimal edit script using Eugene Myers' O((N+M)D) algorithm.
    Finds the furthest reaching path on diagonals k = x - y.
    Returns list of DiffLine objects (KEEP, INSERT, DELETE).
    """
    n = len(a)
    m = len(b)
    max_d = n + m

    if max_d == 0:
        return []

    # v[k] stores furthest reaching x on diagonal k.
    # We offset by max_d to allow negative indices in [ -max_d, max_d ].
    offset = max_d
    v = [0] * (2 * max_d + 1)
    trace: List[List[int]] = []

    x = 0
    y = 0
    # Walk initial diagonal snake
    while x < n and y < m and a[x] == b[y]:
        x += 1
        y += 1
    v[offset + 0] = x
    trace.append(list(v))

    if x >= n and y >= m:
        # Sequences are identical
        return [DiffLine(DiffType.KEEP, a[i], i + 1, i + 1) for i in range(n)]

    found_d = -1
    for d in range(1, max_d + 1):
        for k in range(-d, d + 1, 2):
            # Decide whether to move down (insertion) or right (deletion)
            if k == -d or (k != d and v[offset + k - 1] < v[offset + k + 1]):
                prev_k = k + 1
                cur_x = v[offset + prev_k]
            else:
                prev_k = k - 1
                cur_x = v[offset + prev_k] + 1

            cur_y = cur_x - k

            # Snake: follow equal elements along diagonal
            while cur_x < n and cur_y < m and a[cur_x] == b[cur_y]:
                cur_x += 1
                cur_y += 1

            v[offset + k] = cur_x

            if cur_x >= n and cur_y >= m:
                found_d = d
                break

        trace.append(list(v))
        if found_d != -1:
            break

    # Backtrack through trace to reconstruct edit script
    edits: List[DiffLine] = []
    curr_x = n
    curr_y = m

    for d in range(len(trace) - 1, 0, -1):
        k = curr_x - curr_y
        prev_v = trace[d - 1]

        if k == -d or (k != d and prev_v[offset + k - 1] < prev_v[offset + k + 1]):
            prev_k = k + 1
        else:
            prev_k = k - 1

        prev_x = prev_v[offset + prev_k]
        prev_y = prev_x - prev_k

        # Diagonal snake steps (matches / KEEP)
        while curr_x > prev_x and curr_y > prev_y:
            curr_x -= 1
            curr_y -= 1
            edits.append(DiffLine(DiffType.KEEP, a[curr_x], curr_x + 1, curr_y + 1))

        if d > 0:
            if prev_k < k:
                # Horizontal step from (prev_x, prev_y) to (prev_x + 1, prev_y): DELETE
                curr_x -= 1
                edits.append(DiffLine(DiffType.DELETE, a[curr_x], curr_x + 1, None))
            else:
                # Vertical step from (prev_x, prev_y) to (prev_x, prev_y + 1): INSERT
                curr_y -= 1
                edits.append(DiffLine(DiffType.INSERT, b[curr_y], None, curr_y + 1))

    # Remaining diagonal snake back to (0, 0)
    while curr_x > 0 and curr_y > 0:
        curr_x -= 1
        curr_y -= 1
        edits.append(DiffLine(DiffType.KEEP, a[curr_x], curr_x + 1, curr_y + 1))

    edits.reverse()
    return edits


def unified_diff(
    a_lines: List[str],
    b_lines: List[str],
    a_name: str = "a/file",
    b_name: str = "b/file",
    context: int = 3
) -> str:
    """
    Produces standard Git-compatible unified diff format with hunk headers and context lines.
    """
    edits = myers_diff(a_lines, b_lines)

    # Check if there are any changes
    has_changes = any(e.type != DiffType.KEEP for e in edits)
    if not has_changes:
        return ""

    # Group into hunks with context
    # Find change indices
    change_indices = [i for i, e in enumerate(edits) if e.type != DiffType.KEEP]
    if not change_indices:
        return ""

    hunk_ranges: List[Tuple[int, int]] = []
    cur_start = max(0, change_indices[0] - context)
    cur_end = min(len(edits), change_indices[0] + context + 1)

    for idx in change_indices[1:]:
        start_idx = max(0, idx - context)
        end_idx = min(len(edits), idx + context + 1)
        if start_idx <= cur_end:
            cur_end = end_idx
        else:
            hunk_ranges.append((cur_start, cur_end))
            cur_start = start_idx
            cur_end = end_idx
    hunk_ranges.append((cur_start, cur_end))

    out_lines: List[str] = [
        f"--- {a_name}",
        f"+++ {b_name}"
    ]

    for h_start, h_end in hunk_ranges:
        hunk_edits = edits[h_start:h_end]

        # Calculate line numbers
        old_count = sum(1 for e in hunk_edits if e.type in (DiffType.KEEP, DiffType.DELETE))
        new_count = sum(1 for e in hunk_edits if e.type in (DiffType.KEEP, DiffType.INSERT))

        old_start = next((e.old_lineno for e in hunk_edits if e.old_lineno is not None), 1)
        new_start = next((e.new_lineno for e in hunk_edits if e.new_lineno is not None), 1)

        out_lines.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@")

        for e in hunk_edits:
            if e.type == DiffType.KEEP:
                out_lines.append(f" {e.text}")
            elif e.type == DiffType.INSERT:
                out_lines.append(f"+{e.text}")
            elif e.type == DiffType.DELETE:
                out_lines.append(f"-{e.text}")

    return "\n".join(out_lines) + "\n"
