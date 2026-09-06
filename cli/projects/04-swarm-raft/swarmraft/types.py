"""
Core Raft Consensus Data Types, Enums, and RPC Messages.
Follows the Raft consensus specification (Ongaro & Ousterhout, 2014)
with extensions for fast log conflict backtracking and snapshotting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeRole(str, Enum):
    FOLLOWER = "Follower"
    CANDIDATE = "Candidate"
    LEADER = "Leader"


@dataclass
class LogEntry:
    """A replicated state machine command entry in the Raft log."""
    term: int
    index: int
    command: Any  # Replicated state machine command (e.g. {"op": "SET", "key": "k", "val": "v"})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LogEntry:
        return cls(term=d["term"], index=d["index"], command=d["command"])


@dataclass
class RequestVoteArgs:
    """
    Invoked by candidates to gather votes (§5.2).
    """
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteReply:
    """
    Response to a RequestVote RPC (§5.2).
    """
    term: int
    vote_granted: bool


@dataclass
class AppendEntriesArgs:
    """
    Invoked by leader to replicate log entries (§5.3); also used as heartbeat (§5.2).
    """
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry] = field(default_factory=list)
    leader_commit: int = 0


@dataclass
class AppendEntriesReply:
    """
    Response to an AppendEntries RPC (§5.3).
    Includes conflict optimization for fast log backtracking.
    """
    term: int
    success: bool
    conflict_term: Optional[int] = None
    conflict_first_index: Optional[int] = None
    match_index: int = 0


@dataclass
class InstallSnapshotArgs:
    """
    Invoked by leader to send chunks of a snapshot to a lagging follower (§7).
    """
    term: int
    leader_id: str
    last_included_index: int
    last_included_term: int
    data: bytes  # State machine snapshot serialized payload


@dataclass
class InstallSnapshotReply:
    """
    Response to an InstallSnapshot RPC (§7).
    """
    term: int


@dataclass
class ClientProposal:
    """A command proposed by an external client to the cluster."""
    command: Any
    client_id: str = "default_client"
    seq_num: int = 0


@dataclass
class ProposalResult:
    """Result of proposing a command to the Raft cluster."""
    success: bool
    leader_id: Optional[str] = None
    applied_index: int = 0
    result: Any = None
    error: Optional[str] = None
