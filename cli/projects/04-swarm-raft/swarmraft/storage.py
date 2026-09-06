"""
Persistent Storage Engine for Raft Nodes.
Ensures durable persistence of:
- current_term
- voted_for
- log entries
- snapshots (last_included_index, last_included_term, snapshot_data)
Supports both zero-IO in-memory persistence and atomic crash-resilient file persistence.
"""

from __future__ import annotations
import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from swarmraft.types import LogEntry


class RaftStorage(ABC):
    """Abstract interface for Raft persistent state."""

    @abstractmethod
    def read_state(self) -> Tuple[int, Optional[str], List[LogEntry], int, int, bytes, int]:
        """Returns (current_term, voted_for, log, last_included_index, last_included_term, snapshot, commit_index)."""
        pass

    @abstractmethod
    def save_state(
        self,
        current_term: int,
        voted_for: Optional[str],
        log: List[LogEntry],
        last_included_index: int = 0,
        last_included_term: int = 0,
        snapshot: bytes = b"",
        commit_index: int = 0
    ) -> None:
        """Atomically saves persistent Raft state to storage."""
        pass


class MemoryStorage(RaftStorage):
    """Thread-safe in-memory persistent storage for simulation and tests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.last_included_index: int = 0
        self.last_included_term: int = 0
        self.snapshot: bytes = b""
        self.commit_index: int = 0

    def read_state(self) -> Tuple[int, Optional[str], List[LogEntry], int, int, bytes, int]:
        with self._lock:
            log_copy = [LogEntry(e.term, e.index, e.command) for e in self.log]
            return (
                self.current_term,
                self.voted_for,
                log_copy,
                self.last_included_index,
                self.last_included_term,
                bytes(self.snapshot),
                self.commit_index
            )

    def save_state(
        self,
        current_term: int,
        voted_for: Optional[str],
        log: List[LogEntry],
        last_included_index: int = 0,
        last_included_term: int = 0,
        snapshot: bytes = b"",
        commit_index: int = 0
    ) -> None:
        with self._lock:
            self.current_term = current_term
            self.voted_for = voted_for
            self.log = [LogEntry(e.term, e.index, e.command) for e in log]
            self.last_included_index = last_included_index
            self.last_included_term = last_included_term
            self.snapshot = bytes(snapshot)
            self.commit_index = commit_index


class FileStorage(RaftStorage):
    """
    Atomic crash-resilient disk persistence using temporary write & atomic rename.
    Guarantees zero corruption on power loss or node crashes.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._lock = threading.Lock()
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def read_state(self) -> Tuple[int, Optional[str], List[LogEntry], int, int, bytes, int]:
        with self._lock:
            if not os.path.exists(self.file_path):
                return 0, None, [], 0, 0, b"", 0

            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            log_entries = [LogEntry.from_dict(d) for d in data.get("log", [])]
            snapshot = bytes.fromhex(data.get("snapshot_hex", ""))

            return (
                data.get("current_term", 0),
                data.get("voted_for"),
                log_entries,
                data.get("last_included_index", 0),
                data.get("last_included_term", 0),
                snapshot,
                data.get("commit_index", 0)
            )

    def save_state(
        self,
        current_term: int,
        voted_for: Optional[str],
        log: List[LogEntry],
        last_included_index: int = 0,
        last_included_term: int = 0,
        snapshot: bytes = b"",
        commit_index: int = 0
    ) -> None:
        with self._lock:
            data = {
                "current_term": current_term,
                "voted_for": voted_for,
                "log": [e.to_dict() for e in log],
                "last_included_index": last_included_index,
                "last_included_term": last_included_term,
                "snapshot_hex": snapshot.hex(),
                "commit_index": commit_index
            }

            temp_path = f"{self.file_path}.tmp.{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace guarantees all-or-nothing disk durability
            os.replace(temp_path, self.file_path)
