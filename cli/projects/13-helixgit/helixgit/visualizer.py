"""
HelixGit: ANSI Terminal Commit DAG Graph Visualizer.
Implements multi-lane topological graph rendering, branch column routing,
and decorated commit history with ANSI color formatting.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
from .objects import Commit
from .repo import Repository

# ANSI Color Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

LANE_COLORS = [CYAN, YELLOW, GREEN, MAGENTA, RED, BLUE]


def get_all_refs(repo: Repository) -> Dict[str, List[str]]:
    """
    Finds all branch and tag references pointing to commit SHAs.
    Returns: sha1 -> list of ref names (e.g., 'HEAD -> main', 'v1.0')
    """
    refs_map: Dict[str, List[str]] = {}
    cur_branch, cur_sha = repo.head_ref()

    # Read branches
    for b in repo.list_branches():
        b_path = os.path.join(repo.heads_dir, b)
        if os.path.isfile(b_path):
            with open(b_path, "r") as f:
                sha = f.read().strip()
            label = f"HEAD -> {b}" if b == cur_branch else b
            refs_map.setdefault(sha, []).append(label)

    # Read tags
    if os.path.isdir(repo.tags_dir):
        for t in sorted(os.listdir(repo.tags_dir)):
            t_path = os.path.join(repo.tags_dir, t)
            if os.path.isfile(t_path):
                with open(t_path, "r") as f:
                    sha = f.read().strip()
                refs_map.setdefault(sha, []).append(f"tag: {t}")

    return refs_map


def topological_commit_sort(repo: Repository) -> List[Commit]:
    """
    Computes a topological sort of all commits reachable from any ref.
    """
    refs = get_all_refs(repo)
    start_shas = list(refs.keys())
    _, cur_sha = repo.head_ref()
    if cur_sha and cur_sha not in start_shas:
        start_shas.append(cur_sha)

    visited: Set[str] = set()
    in_degree: Dict[str, int] = {}
    children: Dict[str, List[str]] = {}
    commits: Dict[str, Commit] = {}

    # Discover all commits
    queue = list(start_shas)
    while queue:
        sha = queue.pop(0)
        if sha in visited or not repo.odb.exists(sha):
            continue
        visited.add(sha)
        c = repo.odb.get_commit(sha)
        commits[sha] = c
        for p in c.parents:
            children.setdefault(p, []).append(sha)
            queue.append(p)

    # Kahn's algorithm or reverse postorder for topological sort
    # We sort by author timestamp descending as tie-breaker
    result: List[Commit] = sorted(commits.values(), key=lambda c: c.author_time, reverse=True)
    return result


def render_commit_dag(repo: Repository, max_commits: int = 30, color: bool = True) -> str:
    """
    Renders an ASCII/ANSI multi-lane branch graph of repository history.
    """
    commits = topological_commit_sort(repo)
    if not commits:
        return "No commits in repository."

    refs_map = get_all_refs(repo)
    lines: List[str] = []

    # Lane tracking: list of active commit SHAs representing columns
    lanes: List[Optional[str]] = []

    for commit in commits[:max_commits]:
        sha = commit.sha1()
        parents = commit.parents

        # Find or allocate lane for this commit
        if sha in lanes:
            lane_idx = lanes.index(sha)
        else:
            # Find first empty slot or append
            if None in lanes:
                lane_idx = lanes.index(None)
                lanes[lane_idx] = sha
            else:
                lane_idx = len(lanes)
                lanes.append(sha)

        # Build lane glyph prefix
        lane_str = []
        for i, l_sha in enumerate(lanes):
            col_color = LANE_COLORS[i % len(LANE_COLORS)] if color else ""
            rst = RESET if color else ""
            if i == lane_idx:
                lane_str.append(f"{col_color}*{rst}")
            elif l_sha is not None:
                lane_str.append(f"{col_color}|{rst}")
            else:
                lane_str.append(" ")

        prefix = " ".join(lane_str)

        # Format commit line
        short_sha = sha[:7]
        if color:
            short_sha = f"{YELLOW}{short_sha}{RESET}"

        ref_decorations = ""
        if sha in refs_map:
            decor_list = []
            for r in refs_map[sha]:
                if "HEAD" in r and color:
                    decor_list.append(f"{CYAN}{BOLD}{r}{RESET}")
                elif "tag:" in r and color:
                    decor_list.append(f"{YELLOW}{r}{RESET}")
                elif color:
                    decor_list.append(f"{GREEN}{r}{RESET}")
                else:
                    decor_list.append(r)
            ref_decorations = f" ({', '.join(decor_list)})"

        subject = commit.message.splitlines()[0] if commit.message else ""
        author_info = f"{commit.author_name}"
        if color:
            author_info = f"{DIM}<{author_info}>{RESET}"
        else:
            author_info = f"<{author_info}>"

        lines.append(f"{prefix} {short_sha}{ref_decorations} {subject} {author_info}")

        # Update lanes for parents
        if not parents:
            # Root commit: lane ends
            lanes[lane_idx] = None
        elif len(parents) == 1:
            # Linear commit: current lane continues as parent
            lanes[lane_idx] = parents[0]
        else:
            # Merge commit: first parent takes current lane, subsequent parents get new lanes
            lanes[lane_idx] = parents[0]
            for p in parents[1:]:
                if p not in lanes:
                    if None in lanes:
                        empty_slot = lanes.index(None)
                        lanes[empty_slot] = p
                    else:
                        lanes.append(p)

        # Clean trailing None lanes
        while lanes and lanes[-1] is None:
            lanes.pop()

    return "\n".join(lines)
