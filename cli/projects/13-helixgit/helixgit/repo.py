"""
HelixGit: Repository Porcelain Engine.
Implements git init, add, commit, status, branch, checkout, merge, rebase, and log.
"""

import os
import shutil
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from .objects import Commit, Tree, Blob, Tag, TreeEntry
from .storage import ObjectDatabase
from .index import Index, IndexEntry
from .merge import find_lowest_common_ancestor, three_way_merge_trees, three_way_merge_lines


class Repository:
    """
    High-level Git repository interface managing refs, working tree, and object database.
    """

    def __init__(self, worktree_dir: str) -> None:
        self.worktree = os.path.abspath(worktree_dir)
        self.git_dir = os.path.join(self.worktree, ".git")
        self.objects_dir = os.path.join(self.git_dir, "objects")
        self.refs_dir = os.path.join(self.git_dir, "refs")
        self.heads_dir = os.path.join(self.refs_dir, "heads")
        self.tags_dir = os.path.join(self.refs_dir, "tags")
        self.index_file = os.path.join(self.git_dir, "index")
        self.head_file = os.path.join(self.git_dir, "HEAD")

        self.odb = ObjectDatabase(self.objects_dir)
        self.index = Index()
        if os.path.isfile(self.index_file):
            self.index.read(self.index_file)

    @classmethod
    def init(cls, path: str, default_branch: str = "main") -> "Repository":
        """Initializes a new Git repository at path."""
        worktree = os.path.abspath(path)
        git_dir = os.path.join(worktree, ".git")
        os.makedirs(os.path.join(git_dir, "objects"), exist_ok=True)
        os.makedirs(os.path.join(git_dir, "refs", "heads"), exist_ok=True)
        os.makedirs(os.path.join(git_dir, "refs", "tags"), exist_ok=True)

        head_file = os.path.join(git_dir, "HEAD")
        with open(head_file, "w") as f:
            f.write(f"ref: refs/heads/{default_branch}\n")

        return cls(worktree)

    def head_ref(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (branch_name, commit_sha).
        If detached HEAD, branch_name is None.
        """
        if not os.path.isfile(self.head_file):
            return None, None

        with open(self.head_file, "r") as f:
            content = f.read().strip()

        if content.startswith("ref: refs/heads/"):
            branch = content[len("ref: refs/heads/"):]
            branch_file = os.path.join(self.heads_dir, branch)
            if os.path.isfile(branch_file):
                with open(branch_file, "r") as bf:
                    return branch, bf.read().strip()
            return branch, None  # Unborn branch
        else:
            # Detached HEAD
            return None, content if content else None

    def set_head_commit(self, commit_sha: str) -> None:
        """Updates current branch ref or detached HEAD to point to commit_sha."""
        branch, _ = self.head_ref()
        if branch is not None:
            branch_file = os.path.join(self.heads_dir, branch)
            os.makedirs(os.path.dirname(branch_file), exist_ok=True)
            with open(branch_file, "w") as f:
                f.write(f"{commit_sha}\n")
        else:
            with open(self.head_file, "w") as f:
                f.write(f"{commit_sha}\n")

    def add(self, paths: List[str]) -> None:
        """Stages specified files or directories into the index."""
        for path in paths:
            norm_path = path.replace("\\", "/").strip("/")
            full_path = os.path.join(self.worktree, norm_path)

            if os.path.isdir(full_path):
                for root, _, files in os.walk(full_path):
                    if ".git" in root.split(os.sep):
                        continue
                    for fname in files:
                        sub_full = os.path.join(root, fname)
                        rel = os.path.relpath(sub_full, self.worktree)
                        self.index.stage_file(self.worktree, rel, self.odb)
            elif os.path.isfile(full_path):
                self.index.stage_file(self.worktree, norm_path, self.odb)
            elif not os.path.exists(full_path):
                # File was deleted on disk: remove from index
                self.index.remove(norm_path)

        self.index.write(self.index_file)

    def commit(
        self,
        message: str,
        author_name: str = "HelixGit User",
        author_email: str = "user@helixgit.local"
    ) -> str:
        """
        Synthesizes a Merkle tree from the index and records a new Commit object.
        Advances current branch ref to new commit SHA-1.
        """
        tree_sha = self.index.to_tree(self.odb)
        _, parent_sha = self.head_ref()
        parents = [parent_sha] if parent_sha else []

        commit_sha = self.odb.write_commit(
            tree=tree_sha,
            parents=parents,
            author_name=author_name,
            author_email=author_email,
            author_time=int(time.time()),
            author_tz="+0000",
            message=message.strip() + "\n"
        )

        self.set_head_commit(commit_sha)
        return commit_sha

    def status(self) -> Dict[str, Any]:
        """
        Computes branch name, staged changes, unstaged changes, and untracked files.
        """
        branch, head_sha = self.head_ref()
        diff_tree = self.index.diff_worktree(self.worktree)

        # Compare staged index against HEAD commit tree
        staged_added: List[str] = []
        staged_modified: List[str] = []
        staged_deleted: List[str] = []

        head_files: Dict[str, str] = {}
        if head_sha and self.odb.exists(head_sha):
            head_commit = self.odb.get_commit(head_sha)
            from .merge import flatten_tree
            head_files = {path: sha for path, (_, sha) in flatten_tree(head_commit.tree, self.odb).items()}

        index_files = {path: entry.sha1 for (path, stage), entry in self.index.entries.items() if stage == 0}

        for path, sha in index_files.items():
            if path not in head_files:
                staged_added.append(path)
            elif head_files[path] != sha:
                staged_modified.append(path)

        for path in head_files:
            if path not in index_files:
                staged_deleted.append(path)

        return {
            "branch": branch,
            "head": head_sha,
            "staged": {
                "added": sorted(staged_added),
                "modified": sorted(staged_modified),
                "deleted": sorted(staged_deleted)
            },
            "unstaged": diff_tree,
            "conflicts": sorted(list(self.index.get_conflicted_paths()))
        }

    def list_branches(self) -> List[str]:
        """Returns list of local branch names."""
        branches: List[str] = []
        if not os.path.isdir(self.heads_dir):
            return branches
        for root, _, files in os.walk(self.heads_dir):
            for fname in files:
                full_p = os.path.join(root, fname)
                rel_p = os.path.relpath(full_p, self.heads_dir).replace("\\", "/")
                branches.append(rel_p)
        return sorted(branches)

    def create_branch(self, name: str, start_point: Optional[str] = None) -> None:
        """Creates a new branch pointing to start_point or current HEAD."""
        _, head_sha = self.head_ref()
        target_sha = start_point or head_sha
        if not target_sha:
            raise RuntimeError("Cannot create branch: HEAD does not point to any commit")

        branch_path = os.path.join(self.heads_dir, name)
        if os.path.exists(branch_path):
            raise RuntimeError(f"Branch '{name}' already exists")

        os.makedirs(os.path.dirname(branch_path), exist_ok=True)
        with open(branch_path, "w") as f:
            f.write(f"{target_sha}\n")

    def checkout(self, target: str, create_branch: bool = False) -> None:
        """
        Switches to a branch or commit, updating the working directory and index.
        """
        if create_branch:
            self.create_branch(target)

        branch_file = os.path.join(self.heads_dir, target)
        if os.path.isfile(branch_file):
            # Target is a branch
            with open(branch_file, "r") as f:
                target_sha = f.read().strip()
            with open(self.head_file, "w") as f:
                f.write(f"ref: refs/heads/{target}\n")
        elif self.odb.exists(target):
            # Target is a commit SHA (detached HEAD)
            target_sha = target
            with open(self.head_file, "w") as f:
                f.write(f"{target_sha}\n")
        else:
            raise ValueError(f"Pathspec '{target}' did not match any branch or commit")

        # Checkout target tree into worktree and update index
        commit = self.odb.get_commit(target_sha)
        # Clear untracked files previously managed by index
        for (path, stage) in list(self.index.entries.keys()):
            full_p = os.path.join(self.worktree, path)
            if os.path.isfile(full_p):
                os.remove(full_p)

        self.index.from_tree(commit.tree, self.odb, self.worktree)
        self.index.write(self.index_file)

    def merge(self, other_branch: str) -> Tuple[str, bool]:
        """
        Merges other_branch into current branch.
        Handles fast-forward and 3-way merges.
        Returns (commit_sha_or_message, has_conflicts).
        """
        cur_branch, cur_sha = self.head_ref()
        if not cur_sha:
            raise RuntimeError("Cannot merge on unborn branch")

        other_branch_file = os.path.join(self.heads_dir, other_branch)
        if not os.path.isfile(other_branch_file):
            raise ValueError(f"Branch '{other_branch}' not found")

        with open(other_branch_file, "r") as f:
            other_sha = f.read().strip()

        if cur_sha == other_sha:
            return "Already up to date.", False

        lca_sha = find_lowest_common_ancestor(cur_sha, other_sha, self.odb)

        # Fast-forward check: if current HEAD is the LCA
        if lca_sha == cur_sha:
            self.checkout(other_branch)
            if cur_branch:
                # Update current branch ref to other_sha
                with open(os.path.join(self.heads_dir, cur_branch), "w") as f:
                    f.write(f"{other_sha}\n")
                with open(self.head_file, "w") as f:
                    f.write(f"ref: refs/heads/{cur_branch}\n")
            return f"Fast-forward to {other_sha[:7]}", False

        # 3-Way Merge
        cur_commit = self.odb.get_commit(cur_sha)
        other_commit = self.odb.get_commit(other_sha)
        lca_commit = self.odb.get_commit(lca_sha) if lca_sha else None
        lca_tree = lca_commit.tree if lca_commit else None

        merge_res = three_way_merge_trees(
            lca_tree,
            cur_commit.tree,
            other_commit.tree,
            self.odb,
            ours_label=cur_branch or "HEAD",
            theirs_label=other_branch
        )

        # Write merged files to worktree and index
        self.index.entries.clear()
        for path, (mode, blob_sha) in merge_res.merged_files.items():
            full_path = os.path.join(self.worktree, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            blob = self.odb.get_blob(blob_sha)
            with open(full_path, "wb") as f:
                f.write(blob.data)

            st = os.stat(full_path)
            self.index.add_entry(IndexEntry(
                path=path,
                sha1=blob_sha,
                mode=mode,
                file_size=len(blob.data),
                stage=0
            ))

        self.index.write(self.index_file)

        if not merge_res.clean:
            # Mark conflicts in index stages for conflicted paths
            for path in merge_res.conflicts:
                # Add stage 2 and 3 entries to reflect unmerged conflict state
                self.index.add_entry(IndexEntry(path, "0" * 40, stage=2))
                self.index.add_entry(IndexEntry(path, "0" * 40, stage=3))
            self.index.write(self.index_file)
            return f"Automatic merge failed. Fix conflicts in {merge_res.conflicts} and commit.", True

        # Clean 3-way merge: synthesize tree and create merge commit with 2 parents
        new_tree_sha = self.index.to_tree(self.odb)
        merge_commit_sha = self.odb.write_commit(
            tree=new_tree_sha,
            parents=[cur_sha, other_sha],
            message=f"Merge branch '{other_branch}' into {cur_branch or 'HEAD'}\n"
        )
        self.set_head_commit(merge_commit_sha)
        return merge_commit_sha, False

    def log(self, max_count: int = 20) -> List[Commit]:
        """Traverses commit history backwards starting from HEAD."""
        _, head_sha = self.head_ref()
        commits: List[Commit] = []
        if not head_sha:
            return commits

        visited: Set[str] = set()
        queue = [head_sha]

        while queue and len(commits) < max_count:
            sha = queue.pop(0)
            if sha in visited or not self.odb.exists(sha):
                continue
            visited.add(sha)
            commit = self.odb.get_commit(sha)
            commits.append(commit)
            for p in commit.parents:
                if p not in visited:
                    queue.append(p)

        return commits
