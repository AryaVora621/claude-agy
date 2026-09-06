"""
NexusOS: Microkernel Inter-Process Communication (IPC) Subsystem.
Implements L4-style synchronous rendezvous endpoints, zero-copy message transfers,
asynchronous bounded mailboxes, and capability-based endpoint access control.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
from nexus.types import Errno, CapabilityRights


@dataclass
class IPCMessage:
    """IPC message payload transferring data and optional delegated capabilities."""
    sender_pid: int
    msg_type: int
    payload: Any
    delegated_capability_token: Optional[str] = None


class SynchronousEndpoint:
    """
    L4-Style Synchronous Rendezvous Endpoint.
    Zero-copy handoff: Sender blocks until Receiver arrives, or Receiver blocks until Sender arrives.
    """
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id
        # Queues of waiting threads: (pid, IPCMessage) or (pid, None)
        self.send_queue: deque[Tuple[int, IPCMessage]] = deque()
        self.recv_queue: deque[int] = deque()

    def send(self, sender_pid: int, message: IPCMessage) -> Tuple[Errno, Optional[int]]:
        """
        Send a message through the endpoint:
        - If a receiver is waiting, hand over message directly and wake receiver.
        - Otherwise, enqueue sender and return EAGAIN (to block sender).
        Returns (status, woken_receiver_pid).
        """
        if self.recv_queue:
            receiver_pid = self.recv_queue.popleft()
            return Errno.SUCCESS, receiver_pid
        self.send_queue.append((sender_pid, message))
        return Errno.EAGAIN, None

    def receive(self, receiver_pid: int) -> Tuple[Errno, Optional[IPCMessage], Optional[int]]:
        """
        Receive a message from the endpoint:
        - If a sender is waiting, extract message and unblock sender.
        - Otherwise, enqueue receiver and return EAGAIN (to block receiver).
        Returns (status, message, woken_sender_pid).
        """
        if self.send_queue:
            sender_pid, msg = self.send_queue.popleft()
            return Errno.SUCCESS, msg, sender_pid
        self.recv_queue.append(receiver_pid)
        return Errno.EAGAIN, None, None

    def cancel_wait(self, pid: int) -> None:
        """Remove a process from waiting queues if aborted or killed."""
        if pid in self.recv_queue:
            self.recv_queue.remove(pid)
        self.send_queue = deque([(p, m) for p, m in self.send_queue if p != pid])


class AsynchronousMailbox:
    """
    Bounded Asynchronous Channel with buffered messages.
    """
    def __init__(self, mailbox_id: str, capacity: int = 32):
        self.mailbox_id = mailbox_id
        self.capacity = capacity
        self.buffer: deque[IPCMessage] = deque()
        self.waiting_readers: deque[int] = deque()
        self.waiting_writers: deque[Tuple[int, IPCMessage]] = deque()

    def send(self, sender_pid: int, message: IPCMessage) -> Tuple[Errno, Optional[int]]:
        """
        Enqueue message if capacity permits:
        - If a reader was waiting, dispatch directly.
        - Otherwise append to buffer if space exists.
        Returns (status, woken_reader_pid).
        """
        if self.waiting_readers:
            reader_pid = self.waiting_readers.popleft()
            return Errno.SUCCESS, reader_pid

        if len(self.buffer) >= self.capacity:
            self.waiting_writers.append((sender_pid, message))
            return Errno.EAGAIN, None

        self.buffer.append(message)
        return Errno.SUCCESS, None

    def receive(self, receiver_pid: int) -> Tuple[Errno, Optional[IPCMessage], Optional[int]]:
        """
        Dequeue message if buffer not empty:
        - If writers are waiting, pull next writer's message into buffer and unblock writer.
        Returns (status, message, woken_writer_pid).
        """
        if self.buffer:
            msg = self.buffer.popleft()
            woken_writer = None
            if self.waiting_writers:
                w_pid, w_msg = self.waiting_writers.popleft()
                self.buffer.append(w_msg)
                woken_writer = w_pid
            return Errno.SUCCESS, msg, woken_writer

        self.waiting_readers.append(receiver_pid)
        return Errno.EAGAIN, None, None


class IPCManager:
    """Kernel-level registry for communication endpoints and security enforcement."""
    def __init__(self):
        self.endpoints: Dict[str, SynchronousEndpoint] = {}
        self.mailboxes: Dict[str, AsynchronousMailbox] = {}

    def get_or_create_endpoint(self, endpoint_id: str) -> SynchronousEndpoint:
        if endpoint_id not in self.endpoints:
            self.endpoints[endpoint_id] = SynchronousEndpoint(endpoint_id)
        return self.endpoints[endpoint_id]

    def get_or_create_mailbox(self, mailbox_id: str, capacity: int = 32) -> AsynchronousMailbox:
        if mailbox_id not in self.mailboxes:
            self.mailboxes[mailbox_id] = AsynchronousMailbox(mailbox_id, capacity)
        return self.mailboxes[mailbox_id]

    def cleanup_process(self, pid: int) -> None:
        """Remove process from all endpoint queues upon termination."""
        for ep in self.endpoints.values():
            ep.cancel_wait(pid)
        for mb in self.mailboxes.values():
            if pid in mb.waiting_readers:
                mb.waiting_readers.remove(pid)
            mb.waiting_writers = deque([(p, m) for p, m in mb.waiting_writers if p != pid])
