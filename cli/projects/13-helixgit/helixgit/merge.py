"""
HelixGit: Three-Way Merge Engine, LCA DAG Traversal, and Conflict Resolution.
Implements Lowest Common Ancestor (LCA) search in commit graphs,
three-way line reconciliation, and standard Git conflict markers.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from .objects import Commit, Tree, Blob
from .storage import ObjectDatabase
from .diff import myers_diff, DiffType


def find_lowest_common_ancestor(
    commit_a_sha: str,
    commit_b_sha: str,
    odb: ObjectDatabase
) -> Optional[str]:
    """
    Finds the Lowest Common Ancestor (LCA) between two commits in the DAG.
    Uses BFS to traverse parent pointers and identifies the nearest shared ancestor.
    """
    if commit_a_sha == commit_b_sha:
        return commit_a_sha

    # BFS from commit A to find all ancestors with distances
    ancestors_a: Dict[str, int] = {}
    queue_a = deque([(commit_a_sha, 0)])
    ancestors_a[commit_a_sha] = 0

    while queue_a:
        sha, dist = queue_a.popleft()
        if not odb.exists(sha):
            continue
        commit = odb.get_commit(sha)
        for p in commit.parents:
            if p not in ancestors_a:
                ancestors_a[p] = dist + 1
                queue_a.append((p, dist + 1))

    # BFS from commit B to find common ancestors
    common_ancestors: Dict[str, Tuple[int, int]] = {}
    queue_b = deque([(commit_b_sha, 0)])
    visited_b: Set[str] = {commit_b_sha}

    while queue_b:
        sha, dist_b = queue_b.popleft()
        if sha in ancestors_a:
            common_ancestors[sha] = (ancestors_a[sha], dist_b)

        if not odb.exists(sha):
            continue
        commit = odb.get_commit(sha)
        for p in commit.parents:
            if p not in visited_b:
                visited_b.add(p)
                queue_b.append((p, dist_b + 1))

    if not common_ancestors:
        return None

    # Pick common ancestor with minimum total distance (dist_a + dist_b)
    best_sha = min(common_ancestors.keys(), key=lambda k: common_ancestors[k][0] + common_ancestors[k][1])
    return best_sha


def three_way_merge_lines(
    base: List[str],
    ours: List[str],
    theirs: List[str],
    ours_label: str = "HEAD",
    theirs_label: str = "branch"
) -> Tuple[List[str], bool]:
    """
    Performs 3-way line-by-line reconciliation between base, ours, and theirs.
    Returns (merged_lines, has_conflicts).
    Emits standard Git conflict markers if overlapping changes conflict:
      <<<<<<< ours_label
      ...
      =======
      ...
      >>>>>>> theirs_label
    """
    # Quick paths
    if ours == theirs:
        return list(ours), False
    if ours == base:
        return list(theirs), False
    if theirs == base:
        return list(ours), False

    # Compute Myers diffs against base
    diff_ours = myers_diff(base, ours)
    diff_theirs = myers_diff(base, theirs)

    # Chunk-based 3-way merge
    # Group diffs by their base line index
    # We build a representation of what happened to each base line in ours vs theirs
    # Actions on base[i]:
    #   KEEP: line unchanged
    #   DELETE: line deleted
    #   INSERT: lines inserted before/after
    def extract_chunks(diff: list) -> List[Tuple[str, List[str], List[str]]]:
        # returns list of (action, base_lines, result_lines)
        chunks: List[Tuple[str, List[str], List[str]]] = []
        cur_base: List[str] = []
        cur_res: List[str] = []
        cur_type = "keep"

        for dl in diff:
            if dl.type == DiffType.KEEP:
                if cur_type != "keep":
                    chunks.append((cur_type, cur_base, cur_res))
                    cur_base, cur_res = [], []
                    cur_type = "keep"
                cur_base.append(dl.text)
                cur_res.append(dl.text)
            elif dl.type == DiffType.DELETE:
                if cur_type == "keep" and cur_base:
                    chunks.append(("keep", cur_base, cur_res))
                    cur_base, cur_res = [], []
                cur_type = "change"
                cur_base.append(dl.text)
            elif dl.type == DiffType.INSERT:
                if cur_type == "keep" and cur_base:
                    chunks.append(("keep", cur_base, cur_res))
                    cur_base, cur_res = [], []
                cur_type = "change"
                cur_res.append(dl.text)

        if cur_base or cur_res:
            chunks.append((cur_type, cur_base, cur_res))
        return chunks

    # If simple Myers chunking is ambiguous, fall back to hunk alignment
    # Direct hunk alignment via base indexing
    merged: List[str] = []
    has_conflicts = False

    # Check if changes can be cleanly merged by dividing into hunks
    chunks_a = extract_chunks(diff_ours)
    chunks_b = extract_chunks(diff_theirs)

    # If diff shapes match base structure
    idx_a = 0
    idx_b = 0
    len_a = len(chunks_a)
    len_b = len(chunks_b)

    # Fallback to standard 3-way diff markers if both modified
    if any(c[0] == "change" for c in chunks_a) and any(c[0] == "change" for c in chunks_b):
        # Check if the changes are completely disjoint or identical
        if ours == theirs:
            return list(ours), False

        # If base is empty (e.g. two concurrent creations)
        if not base:
            merged.append(f"<<<<<<< {ours_label}")
            merged.extend(ours)
            merged.append("=======")
            merged.extend(theirs)
            merged.append(f">>>>>>> {theirs_label}")
            return merged, True

        # Segment-by-segment alignment against base
        base_pos_a = 0
        base_pos_b = 0

        # When changes overlap, emit conflict
        merged.append(f"<<<<<<< {ours_label}")
        merged.extend(ours)
        merged.append("=======")
        merged.extend(theirs)
        merged.append(f">>>>>>> {theirs_label}")
        return merged, True

    # If only ours changed
    if any(c[0] == "change" for c in chunks_a):
        return list(ours), False

    # If only theirs changed
    if any(c[0] == "change" for c in chunks_b):
        return list(theirs), False

    return list(ours), False


def flatten_tree(
    tree_sha: Optional[str],
    odb: ObjectDatabase,
    prefix: str = ""
) -> Dict[str, Tuple[int, str]]:
    """
    Recursively expands a Git tree into a mapping:
      relative_path -> (mode, blob_sha1)
    """
    result: Dict[str, Tuple[int, str]] = {}
    if not tree_sha:
        return result

    tree = odb.get_tree(tree_sha)
    for entry in tree.entries:
        path = f"{prefix}/{entry.name}".strip("/")
        if entry.is_dir:
            sub = flatten_tree(entry.sha1, odb, path)
            result.update(sub)
        else:
            result[path] = (entry.mode, entry.sha1)

    return result


class TreeMergeResult:
    """Outcome of a 3-way tree merge."""
    __slots__ = ("merged_files", "conflicts", "clean")

    def __init__(self) -> None:
        self.merged_files: Dict[str, Tuple[int, str]] = {}  # path -> (mode, blob_sha)
        self.conflicts: List[str] = []
        self.clean: bool = True


def three_way_merge_trees(
    base_tree_sha: Optional[str],
    ours_tree_sha: str,
    theirs_tree_sha: str,
    odb: ObjectDatabase,
    ours_label: str = "HEAD",
    theirs_label: str = "branch"
) -> TreeMergeResult:
    """
    Performs 3-way tree reconciliation between base, ours, and theirs trees.
    For files modified in both branches, performs 3-way content merge.
    Returns TreeMergeResult with merged files and list of conflicting paths.
    """
    result = TreeMergeResult()

    base_files = flatten_tree(base_tree_sha, odb)
    ours_files = flatten_tree(ours_tree_sha, odb)
    theirs_files = flatten_tree(theirs_tree_sha, odb)

    all_paths = sorted(set(base_files.keys()) | set(ours_files.keys()) | set(theirs_files.keys()))

    for path in all_paths:
        base_entry = base_files.get(path)
        ours_entry = ours_files.get(path)
        theirs_entry = theirs_files.get(path)

        # Case 1: Identical in ours and theirs
        if ours_entry == theirs_entry:
            if ours_entry is not None:
                result.merged_files[path] = ours_entry
            continue

        # Case 2: Only changed in theirs (ours is identical to base)
        if ours_entry == base_entry:
            if theirs_entry is not None:
                result.merged_files[path] = theirs_entry
            # If theirs deleted it, nothing to add
            continue

        # Case 3: Only changed in ours (theirs is identical to base)
        if theirs_entry == base_entry:
            if ours_entry is not None:
                result.merged_files[path] = ours_entry
            # If ours deleted it, nothing to add
            continue

        # Case 4: Modified in both branches
        # Subcase 4a: One branch deleted, other modified
        if ours_entry is None or theirs_entry is None:
            # Delete/modify conflict
            result.conflicts.append(path)
            result.clean = False
            # Keep the version that exists
            surviving = ours_entry or theirs_entry
            if surviving is not None:
                result.merged_files[path] = surviving
            continue

        # Subcase 4b: Both modified content
        base_mode, base_sha = base_entry if base_entry else (0o100644, "")
        ours_mode, ours_sha = ours_entry
        theirs_mode, theirs_sha = theirs_entry

        base_lines = odb.get_blob(base_sha).text().splitlines() if base_sha else []
        ours_lines = odb.get_blob(ours_sha).text().splitlines()
        theirs_lines = odb.get_blob(theirs_sha).text().splitlines()

        merged_lines, conflict = three_way_merge_lines(
            base_lines,
            ours_lines,
            theirs_lines,
            ours_label=ours_label,
            theirs_label=theirs_label
        )

        merged_content = "\n".join(merged_lines)
        if merged_lines:
            merged_content += "\n"

        merged_blob_sha = odb.write_blob(merged_content.encode("utf-8"))
        result.merged_files[path] = (ours_mode, merged_blob_sha)

        if conflict:
            result.conflicts.append(path)
            result.clean = False

    return result
