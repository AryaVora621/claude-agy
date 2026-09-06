"""
HydraNet: TCP Sliding Window Flow Control & Reassembly Buffers.
Maintains Send and Receive sliding windows, handles out-of-order reordering,
advertises receiver flow capacity, and retransmits unacknowledged segments.
"""

import time
from typing import Dict, Optional, Tuple


class SendBuffer:
    """
    Manages byte streams in-flight and awaiting transmission.
    Tracks SND.UNA (oldest unacknowledged byte) and SND.NXT (next sequence number).
    """

    def __init__(self, initial_seq: int = 0):
        self.una_seq = initial_seq
        self.next_seq = initial_seq
        self.stream = bytearray()
        self.stream_base_seq = initial_seq
        # inflight: seq_num -> (data, sent_time, retransmitted_count)
        self.inflight: Dict[int, Tuple[bytes, float, int]] = {}

    def push_data(self, data: bytes) -> None:
        """Append user application bytes into outgoing stream."""
        self.stream.extend(data)

    def get_sendable_chunk(self, max_allowed_seq: int, mss: int) -> Optional[Tuple[int, bytes]]:
        """
        Extract the next chunk available for transmission up to min(window limit, MSS).
        Returns (seq_num, chunk_bytes) or None if no data is sendable.
        """
        offset = self.next_seq - self.stream_base_seq
        if offset >= len(self.stream):
            return None  # No more pending unsent data

        bytes_avail = len(self.stream) - offset
        window_headroom = max(0, max_allowed_seq - self.next_seq)
        if window_headroom <= 0:
            return None  # Flow/Congestion window exhausted

        chunk_len = min(bytes_avail, window_headroom, mss)
        if chunk_len <= 0:
            return None

        chunk = bytes(self.stream[offset : offset + chunk_len])
        seq = self.next_seq
        self.next_seq += chunk_len
        self.inflight[seq] = (chunk, time.time(), 0)
        return seq, chunk

    def acknowledge(self, ack_seq: int) -> Tuple[int, Optional[float]]:
        """
        Process cumulative ACK advancing SND.UNA.
        Returns (bytes_newly_acked, rtt_sample_if_clean).
        """
        if ack_seq <= self.una_seq:
            return 0, None  # Duplicate or stale ACK

        bytes_acked = ack_seq - self.una_seq
        rtt_sample: Optional[float] = None
        now = time.time()

        # Remove acknowledged packets from inflight registry
        acked_keys = [s for s in self.inflight.keys() if s + len(self.inflight[s][0]) <= ack_seq]
        for k in acked_keys:
            chunk_data, sent_time, retrans_count = self.inflight[k]
            # Karn's algorithm: do not sample RTT for retransmitted segments
            if retrans_count == 0 and rtt_sample is None:
                rtt_sample = now - sent_time
            del self.inflight[k]

        # Compact buffered stream
        trim_len = ack_seq - self.stream_base_seq
        if trim_len > 0:
            self.stream = self.stream[trim_len:]
            self.stream_base_seq = ack_seq

        self.una_seq = ack_seq
        return bytes_acked, rtt_sample

    def get_oldest_unacked(self) -> Optional[Tuple[int, bytes]]:
        """Return oldest unacknowledged packet for Fast Retransmit or RTO timeout."""
        if not self.inflight:
            return None
        oldest_seq = min(self.inflight.keys())
        chunk, sent_time, retrans_count = self.inflight[oldest_seq]
        self.inflight[oldest_seq] = (chunk, time.time(), retrans_count + 1)
        return oldest_seq, chunk

    def has_unacked_data(self) -> bool:
        return bool(self.inflight) or (self.next_seq < self.stream_base_seq + len(self.stream))


class ReceiveBuffer:
    """
    Sliding window receive buffer with out-of-order segment reassembly.
    Tracks RCV.NXT and dynamically calculates Advertised Window.
    """

    def __init__(self, initial_seq: int = 0, capacity: int = 65535):
        self.rcv_nxt = initial_seq
        self.capacity = capacity
        self.ready_bytes = bytearray()
        # out_of_order: seq_num -> payload
        self.out_of_order: Dict[int, bytes] = {}

    def advertised_window(self) -> int:
        """Remaining free capacity advertised in outgoing TCP ACK headers."""
        return max(0, self.capacity - len(self.ready_bytes))

    def put_segment(self, seq_num: int, payload: bytes) -> int:
        """
        Process incoming TCP segment.
        Returns newly available contiguous bytes.
        """
        if not payload:
            return 0

        seg_end = seq_num + len(payload)

        # 1. Entirely duplicate/stale segment
        if seg_end <= self.rcv_nxt:
            return 0

        # 2. In-order arrival (seq_num matches RCV.NXT)
        if seq_num == self.rcv_nxt:
            self.ready_bytes.extend(payload)
            self.rcv_nxt += len(payload)

            # Stitch any now-contiguous segments from out_of_order pool
            while self.rcv_nxt in self.out_of_order:
                buffered = self.out_of_order.pop(self.rcv_nxt)
                self.ready_bytes.extend(buffered)
                self.rcv_nxt += len(buffered)

            return len(payload)

        # 3. Out-of-order segment ahead of expected sequence (buffer it)
        if seq_num > self.rcv_nxt:
            # Only store if within receive window bounds
            if seq_num < self.rcv_nxt + self.capacity:
                self.out_of_order[seq_num] = payload
            return 0

        # 4. Partially overlapping segment (starts before rcv_nxt, extends past it)
        overlap = self.rcv_nxt - seq_num
        new_data = payload[overlap:]
        self.ready_bytes.extend(new_data)
        self.rcv_nxt += len(new_data)

        while self.rcv_nxt in self.out_of_order:
            buffered = self.out_of_order.pop(self.rcv_nxt)
            self.ready_bytes.extend(buffered)
            self.rcv_nxt += len(buffered)

        return len(new_data)

    def read(self, max_bytes: int = 4096) -> bytes:
        """Application consumes up to max_bytes from the ready buffer."""
        if not self.ready_bytes:
            return b""
        chunk = bytes(self.ready_bytes[:max_bytes])
        self.ready_bytes = self.ready_bytes[max_bytes:]
        return chunk
