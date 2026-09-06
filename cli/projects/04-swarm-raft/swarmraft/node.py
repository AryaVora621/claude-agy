"""
Raft Consensus Node Engine.
Complete implementation of the Raft distributed consensus protocol:
- Leader Election with randomized election timers and split-vote prevention (§5.2)
- Log Replication and Commitment via majority quorum (§5.3)
- Log Matching & Safety Invariants (§5.4)
- Fast Log Backtracking on Conflict Resolution
- Log Compaction & InstallSnapshot RPC (§7)
- Dynamic Client Proposals with Linearizability
"""

from __future__ import annotations
import random
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
from swarmraft.storage import RaftStorage, MemoryStorage
from swarmraft.state_machine import StateMachine, KVStateMachine


class RaftNode:
    """
    A single Raft consensus node.
    """

    def __init__(
        self,
        node_id: str,
        peers: List[str],
        transport: SimulatedNetwork,
        storage: Optional[RaftStorage] = None,
        state_machine: Optional[StateMachine] = None,
        election_timeout_range: Tuple[float, float] = (0.15, 0.30),
        heartbeat_interval: float = 0.05,
    ):
        self.node_id = node_id
        self.peers = [p for p in peers if p != node_id]
        self.transport = transport
        self.storage = storage or MemoryStorage()
        self.state_machine = state_machine or KVStateMachine()

        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval

        # Mutex & Condition variable for state synchronization
        self._lock = threading.RLock()
        self._running = False

        # Persistent State on all servers (§5.2, Figure 2)
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.last_included_index: int = 0
        self.last_included_term: int = 0

        # Restore from storage if present
        (
            self.current_term,
            self.voted_for,
            self.log,
            self.last_included_index,
            self.last_included_term,
            snapshot_bytes,
            persisted_commit_index
        ) = self.storage.read_state()

        if snapshot_bytes:
            self.state_machine.apply_snapshot(snapshot_bytes)

        # Volatile State on all servers (§5.2, Figure 2)
        self.commit_index: int = max(self.last_included_index, persisted_commit_index)
        self.last_applied: int = self.last_included_index
        self.role: NodeRole = NodeRole.FOLLOWER
        self.current_leader: Optional[str] = None

        # Volatile State on Leaders (reinitialized after election)
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        # Pending client proposals: index -> (threading.Event, result_container)
        self._pending_proposals: Dict[int, Tuple[threading.Event, List[Any]]] = {}

        # Replay persisted committed entries into state machine
        self._apply_committed_entries()

        # Timers
        self._last_election_reset = time.time()
        self._election_timeout = self._random_election_timeout()

        # Background worker threads
        self._timer_thread: Optional[threading.Thread] = None

        # Register RPC handlers on network
        self.transport.register_node(self.node_id)
        self.transport.register_handler(self.node_id, "RequestVote", self.handle_request_vote)
        self.transport.register_handler(self.node_id, "AppendEntries", self.handle_append_entries)
        self.transport.register_handler(self.node_id, "InstallSnapshot", self.handle_install_snapshot)

    def _random_election_timeout(self) -> float:
        return random.uniform(self.election_timeout_range[0], self.election_timeout_range[1])

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self.transport.set_node_alive(self.node_id, True)
            self._last_election_reset = time.time()
            self._timer_thread = threading.Thread(target=self._run_loop, daemon=True)
            self._timer_thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self.transport.set_node_alive(self.node_id, False)

    # ==================== Log Index Helpers ====================

    def last_log_index(self) -> int:
        """Returns the logical index of the last log entry."""
        if self.log:
            return self.log[-1].index
        return self.last_included_index

    def last_log_term(self) -> int:
        """Returns the term of the last log entry."""
        if self.log:
            return self.log[-1].term
        return self.last_included_term

    def get_term_for_index(self, index: int) -> int:
        """Returns the term of entry at logical index."""
        if index == 0:
            return 0
        if index == self.last_included_index:
            return self.last_included_term
        if index < self.last_included_index:
            return 0  # In compacted snapshot
        offset = index - self.last_included_index - 1
        if 0 <= offset < len(self.log):
            return self.log[offset].term
        return 0

    def get_entries_from(self, start_index: int) -> List[LogEntry]:
        """Returns all entries starting from logical index."""
        offset = start_index - self.last_included_index - 1
        if offset < 0:
            return []
        return [LogEntry(e.term, e.index, e.command) for e in self.log[offset:]]

    def _persist(self) -> None:
        """Saves persistent state to durable storage."""
        snapshot = self.state_machine.take_snapshot() if self.last_included_index > 0 else b""
        self.storage.save_state(
            current_term=self.current_term,
            voted_for=self.voted_for,
            log=self.log,
            last_included_index=self.last_included_index,
            last_included_term=self.last_included_term,
            snapshot=snapshot,
            commit_index=self.commit_index
        )

    # ==================== Background Loop ====================

    def _run_loop(self) -> None:
        while self._running:
            now = time.time()
            send_heartbeat = False
            start_election = False

            with self._lock:
                if not self._running:
                    break

                if self.role == NodeRole.LEADER:
                    if now - self._last_election_reset >= self.heartbeat_interval:
                        send_heartbeat = True
                        self._last_election_reset = now
                else:
                    if now - self._last_election_reset >= self._election_timeout:
                        start_election = True
                        self._last_election_reset = now
                        self._election_timeout = self._random_election_timeout()

            if send_heartbeat:
                self._broadcast_append_entries()
            elif start_election:
                self._start_election()

            # Process state machine apply loop
            self._apply_committed_entries()

            time.sleep(0.01)

    # ==================== Election Protocol ====================

    def _start_election(self) -> None:
        with self._lock:
            self.role = NodeRole.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            self.current_leader = None
            self._persist()

            current_term = self.current_term
            last_idx = self.last_log_index()
            last_t = self.last_log_term()
            votes_needed = (len(self.peers) + 1) // 2 + 1
            votes_received = 1  # Vote for self

            args = RequestVoteArgs(
                term=current_term,
                candidate_id=self.node_id,
                last_log_index=last_idx,
                last_log_term=last_t
            )

        # Broadcast RequestVote RPCs to all peers in parallel
        for peer in self.peers:
            def _send_vote(target=peer):
                nonlocal votes_received
                reply: Optional[RequestVoteReply] = self.transport.call(
                    self.node_id, target, "RequestVote", args, timeout_sec=self.election_timeout_range[0]
                )
                if not reply:
                    return

                with self._lock:
                    if self.role != NodeRole.CANDIDATE or self.current_term != current_term:
                        return

                    if reply.term > self.current_term:
                        self._step_down(reply.term)
                        return

                    if reply.vote_granted:
                        votes_received += 1
                        if votes_received >= votes_needed:
                            self._become_leader()

            threading.Thread(target=_send_vote, daemon=True).start()

    def _become_leader(self) -> None:
        self.role = NodeRole.LEADER
        self.current_leader = self.node_id
        last_idx = self.last_log_index()

        for peer in self.peers:
            self.next_index[peer] = last_idx + 1
            self.match_index[peer] = 0

        self._last_election_reset = time.time()
        # Immediately broadcast initial heartbeat
        self._broadcast_append_entries()

    def _step_down(self, new_term: int, new_leader: Optional[str] = None) -> None:
        self.role = NodeRole.FOLLOWER
        self.current_term = new_term
        self.voted_for = None
        self.current_leader = new_leader
        self._last_election_reset = time.time()
        self._election_timeout = self._random_election_timeout()
        self._persist()

    # ==================== Log Replication Protocol ====================

    def _broadcast_append_entries(self) -> None:
        with self._lock:
            if self.role != NodeRole.LEADER:
                return
            term = self.current_term
            leader_commit = self.commit_index
            peers_list = list(self.peers)

        for peer in peers_list:
            with self._lock:
                if self.role != NodeRole.LEADER or self.current_term != term:
                    return
                next_idx = self.next_index.get(peer, self.last_log_index() + 1)

                # If follower needs entries already compacted into snapshot, send snapshot!
                if next_idx <= self.last_included_index:
                    snap_args = InstallSnapshotArgs(
                        term=self.current_term,
                        leader_id=self.node_id,
                        last_included_index=self.last_included_index,
                        last_included_term=self.last_included_term,
                        data=self.state_machine.take_snapshot()
                    )
                    threading.Thread(target=self._send_snapshot_to_peer, args=(peer, snap_args), daemon=True).start()
                    continue

                prev_idx = next_idx - 1
                prev_term = self.get_term_for_index(prev_idx)
                entries = self.get_entries_from(next_idx)

                args = AppendEntriesArgs(
                    term=term,
                    leader_id=self.node_id,
                    prev_log_index=prev_idx,
                    prev_log_term=prev_term,
                    entries=entries,
                    leader_commit=leader_commit
                )

            threading.Thread(target=self._send_append_to_peer, args=(peer, args), daemon=True).start()

    def _send_append_to_peer(self, peer: str, args: AppendEntriesArgs) -> None:
        reply: Optional[AppendEntriesReply] = self.transport.call(
            self.node_id, peer, "AppendEntries", args, timeout_sec=self.heartbeat_interval * 3
        )
        if not reply:
            return

        with self._lock:
            if self.role != NodeRole.LEADER or self.current_term != args.term:
                return

            if reply.term > self.current_term:
                self._step_down(reply.term)
                return

            if reply.success:
                # Log entries replicated successfully
                self.match_index[peer] = max(self.match_index.get(peer, 0), reply.match_index)
                self.next_index[peer] = self.match_index[peer] + 1
                self._check_and_update_commit_index()
            else:
                # Fast conflict rollback (§5.3 optimization)
                if reply.conflict_term is not None and reply.conflict_first_index is not None:
                    # If leader has conflict_term, set next_index to last entry of that term
                    found_term = False
                    for e in reversed(self.log):
                        if e.term == reply.conflict_term:
                            self.next_index[peer] = e.index + 1
                            found_term = True
                            break
                    if not found_term:
                        self.next_index[peer] = max(1, reply.conflict_first_index)
                else:
                    self.next_index[peer] = max(1, self.next_index.get(peer, 1) - 1)

    def _send_snapshot_to_peer(self, peer: str, args: InstallSnapshotArgs) -> None:
        reply: Optional[InstallSnapshotReply] = self.transport.call(
            self.node_id, peer, "InstallSnapshot", args, timeout_sec=1.0
        )
        if not reply:
            return

        with self._lock:
            if self.role != NodeRole.LEADER or self.current_term != args.term:
                return

            if reply.term > self.current_term:
                self._step_down(reply.term)
                return

            self.match_index[peer] = max(self.match_index.get(peer, 0), args.last_included_index)
            self.next_index[peer] = self.match_index[peer] + 1

    def _check_and_update_commit_index(self) -> None:
        """
        If there exists an N > commit_index, a majority of match_index[i] >= N,
        and log[N].term == current_term: set commit_index = N (§5.3, §5.4.2).
        """
        match_indices = sorted(list(self.match_index.values()) + [self.last_log_index()])
        # Quorum median index
        median_idx = match_indices[len(match_indices) // 2]

        if median_idx > self.commit_index:
            if self.get_term_for_index(median_idx) == self.current_term:
                self.commit_index = median_idx
                self._persist()
                self._apply_committed_entries()

    # ==================== State Machine Execution ====================

    def _apply_committed_entries(self) -> None:
        with self._lock:
            while self.commit_index > self.last_applied:
                self.last_applied += 1
                curr_idx = self.last_applied
                offset = curr_idx - self.last_included_index - 1
                if 0 <= offset < len(self.log):
                    entry = self.log[offset]
                    res = self.state_machine.apply(entry.command)
                    # Notify pending client proposal if present
                    if curr_idx in self._pending_proposals:
                        ev, box = self._pending_proposals.pop(curr_idx)
                        box.append(res)
                        ev.set()

    # ==================== RPC Handlers ====================

    def handle_request_vote(self, args: RequestVoteArgs) -> RequestVoteReply:
        with self._lock:
            if args.term > self.current_term:
                self._step_down(args.term)

            reply = RequestVoteReply(term=self.current_term, vote_granted=False)

            if args.term < self.current_term:
                return reply

            can_vote = (self.voted_for is None or self.voted_for == args.candidate_id)

            # Election Safety Rule (§5.4.1): Candidate log must be at least as up-to-date as receiver
            my_last_term = self.last_log_term()
            my_last_index = self.last_log_index()

            log_up_to_date = (
                (args.last_log_term > my_last_term) or
                (args.last_log_term == my_last_term and args.last_log_index >= my_last_index)
            )

            if can_vote and log_up_to_date:
                self.voted_for = args.candidate_id
                self._last_election_reset = time.time()
                self._persist()
                reply.vote_granted = True

            return reply

    def handle_append_entries(self, args: AppendEntriesArgs) -> AppendEntriesReply:
        with self._lock:
            if args.term > self.current_term:
                self._step_down(args.term, args.leader_id)

            reply = AppendEntriesReply(term=self.current_term, success=False)

            if args.term < self.current_term:
                return reply

            # Legitimate heartbeat/append from leader
            self.current_leader = args.leader_id
            self._last_election_reset = time.time()
            if self.role == NodeRole.CANDIDATE:
                self.role = NodeRole.FOLLOWER

            # 1. Log Consistency Check at prev_log_index (§5.3)
            my_last_idx = self.last_log_index()
            if args.prev_log_index > my_last_idx:
                reply.conflict_first_index = my_last_idx + 1
                reply.conflict_term = None
                return reply

            if args.prev_log_index >= self.last_included_index:
                actual_term = self.get_term_for_index(args.prev_log_index)
                if actual_term != args.prev_log_term:
                    reply.conflict_term = actual_term
                    # Find first index for conflicting term
                    first_idx = args.prev_log_index
                    while first_idx > self.last_included_index and self.get_term_for_index(first_idx - 1) == actual_term:
                        first_idx -= 1
                    reply.conflict_first_index = first_idx
                    return reply

            # 2. Append new entries, resolving conflicts (§5.3 Rule 3 & 4)
            for i, new_entry in enumerate(args.entries):
                target_idx = new_entry.index
                if target_idx <= self.last_included_index:
                    continue

                offset = target_idx - self.last_included_index - 1
                if offset < len(self.log):
                    if self.log[offset].term != new_entry.term:
                        # Conflict: truncate existing entries from offset onwards
                        self.log = self.log[:offset]
                        self.log.append(new_entry)
                else:
                    self.log.append(new_entry)

            self._persist()

            # 3. Update commit_index (§5.3 Rule 5)
            if args.leader_commit > self.commit_index:
                self.commit_index = min(args.leader_commit, self.last_log_index())
                self._persist()
                self._apply_committed_entries()

            reply.success = True
            reply.match_index = self.last_log_index()
            return reply

    def handle_install_snapshot(self, args: InstallSnapshotArgs) -> InstallSnapshotReply:
        with self._lock:
            if args.term > self.current_term:
                self._step_down(args.term, args.leader_id)

            reply = InstallSnapshotReply(term=self.current_term)
            if args.term < self.current_term:
                return reply

            self.current_leader = args.leader_id
            self._last_election_reset = time.time()

            if args.last_included_index <= self.last_included_index:
                return reply

            # Discard any log entries covered by snapshot
            new_log = []
            for e in self.log:
                if e.index > args.last_included_index:
                    new_log.append(e)

            self.log = new_log
            self.last_included_index = args.last_included_index
            self.last_included_term = args.last_included_term

            # Restore state machine from snapshot bytes
            self.state_machine.apply_snapshot(args.data)
            self.commit_index = max(self.commit_index, args.last_included_index)
            self.last_applied = max(self.last_applied, args.last_included_index)

            self._persist()
            return reply

    # ==================== Log Compaction ====================

    def compact_log(self, max_log_length: int = 50) -> bool:
        """
        Compacts in-memory log entries up to commit_index if log exceeds max_log_length.
        """
        with self._lock:
            if len(self.log) <= max_log_length:
                return False

            cutoff_index = self.commit_index
            if cutoff_index <= self.last_included_index:
                return False

            cutoff_term = self.get_term_for_index(cutoff_index)
            new_log = [e for e in self.log if e.index > cutoff_index]

            self.log = new_log
            self.last_included_index = cutoff_index
            self.last_included_term = cutoff_term
            self._persist()
            return True

    # ==================== Client Proposal API ====================

    def propose(self, command: Any, timeout_sec: float = 2.0) -> ProposalResult:
        """
        Proposes a new client command to the cluster.
        If node is leader, appends to log and waits for majority quorum commit.
        If node is follower or candidate, redirects with leader_id.
        """
        ev = threading.Event()
        result_box: List[Any] = []

        with self._lock:
            if self.role != NodeRole.LEADER:
                return ProposalResult(
                    success=False,
                    leader_id=self.current_leader,
                    error=f"Not leader. Current leader is {self.current_leader}"
                )

            entry_idx = self.last_log_index() + 1
            entry_term = self.current_term
            entry = LogEntry(term=entry_term, index=entry_idx, command=command)
            self.log.append(entry)
            self._persist()

            self._pending_proposals[entry_idx] = (ev, result_box)
            # Trigger immediate append broadcast
            self._broadcast_append_entries()

        # Wait for commit and state machine apply
        committed = ev.wait(timeout=timeout_sec)
        with self._lock:
            if not committed:
                self._pending_proposals.pop(entry_idx, None)
                return ProposalResult(
                    success=False,
                    leader_id=self.current_leader,
                    error=f"Proposal at index {entry_idx} timed out waiting for quorum"
                )

            res = result_box[0] if result_box else None
            return ProposalResult(
                success=True,
                leader_id=self.node_id,
                applied_index=entry_idx,
                result=res
            )
