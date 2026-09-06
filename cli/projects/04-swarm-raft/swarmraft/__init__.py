"""
SwarmRaft: Fault-Tolerant Distributed Consensus & Replicated State Machine.
Pure Python Standard Library Implementation (Zero External Dependencies).
"""

from swarmraft.types import (
    NodeRole,
    LogEntry,
    RequestVoteArgs,
    RequestVoteReply,
    AppendEntriesArgs,
    AppendEntriesReply,
    InstallSnapshotArgs,
    InstallSnapshotReply,
    ProposalResult,
)
from swarmraft.transport import SimulatedNetwork
from swarmraft.storage import RaftStorage, MemoryStorage, FileStorage
from swarmraft.state_machine import StateMachine, KVStateMachine
from swarmraft.node import RaftNode
from swarmraft.cluster import RaftCluster
from swarmraft.visualizer import render_cluster_ascii

__all__ = [
    "NodeRole",
    "LogEntry",
    "RequestVoteArgs",
    "RequestVoteReply",
    "AppendEntriesArgs",
    "AppendEntriesReply",
    "InstallSnapshotArgs",
    "InstallSnapshotReply",
    "ProposalResult",
    "SimulatedNetwork",
    "RaftStorage",
    "MemoryStorage",
    "FileStorage",
    "StateMachine",
    "KVStateMachine",
    "RaftNode",
    "RaftCluster",
    "render_cluster_ascii",
]
