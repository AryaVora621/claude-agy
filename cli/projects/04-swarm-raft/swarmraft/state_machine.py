"""
Replicated State Machine implementations.
Commands are only executed when committed by a Raft quorum.
Supports KV operations and state machine snapshotting.
"""

from __future__ import annotations
import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StateMachine(ABC):
    """Abstract replicated state machine interface."""

    @abstractmethod
    def apply(self, command: Any) -> Any:
        """Applies a committed log command and returns the execution result."""
        pass

    @abstractmethod
    def take_snapshot(self) -> bytes:
        """Serializes the entire state machine state into raw bytes."""
        pass

    @abstractmethod
    def apply_snapshot(self, data: bytes) -> None:
        """Restores state machine state from snapshot bytes."""
        pass


class KVStateMachine(StateMachine):
    """
    Linearizable In-Memory Key-Value State Machine.
    Supported Commands:
    - {"op": "SET", "key": str, "val": Any} -> returns previous value or None
    - {"op": "GET", "key": str} -> returns value or None
    - {"op": "DELETE", "key": str} -> returns deleted value or None
    - {"op": "CAS", "key": str, "expected": Any, "new": Any} -> returns bool
    - {"op": "APPEND", "key": str, "val": str} -> returns new length
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store: Dict[str, Any] = {}

    def apply(self, command: Any) -> Any:
        with self._lock:
            if not isinstance(command, dict):
                return None

            op = command.get("op", "").upper()
            key = str(command.get("key", ""))

            if op == "SET":
                old = self._store.get(key)
                self._store[key] = command.get("val")
                return old

            elif op == "GET":
                return self._store.get(key)

            elif op == "DELETE":
                return self._store.pop(key, None)

            elif op == "CAS":
                expected = command.get("expected")
                new_val = command.get("new")
                curr = self._store.get(key)
                if curr == expected:
                    self._store[key] = new_val
                    return True
                return False

            elif op == "APPEND":
                val = str(command.get("val", ""))
                curr = str(self._store.get(key, ""))
                res = curr + val
                self._store[key] = res
                return len(res)

            return None

    def get(self, key: str) -> Any:
        """Direct read from state machine without modifying state."""
        with self._lock:
            return self._store.get(key)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(list(self._store.keys()))

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._store)

    def take_snapshot(self) -> bytes:
        with self._lock:
            payload = json.dumps(self._store).encode("utf-8")
            return payload

    def apply_snapshot(self, data: bytes) -> None:
        with self._lock:
            if not data:
                self._store.clear()
                return
            self._store = json.loads(data.decode("utf-8"))
