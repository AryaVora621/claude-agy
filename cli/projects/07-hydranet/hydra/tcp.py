"""
HydraNet: RFC 793 Transmission Control Protocol (TCP) State Machine & TCB.
Implements the full 11-state TCP finite state machine (FSM), 3-way handshake,
sliding window data exchange, retransmissions, and 4-way connection teardown.
"""

import time
import random
from enum import Enum, auto
from typing import Optional, Tuple, Callable, List
from hydra.types import (
    IPAddress,
    TCPHeader,
    TCPFlags,
    IPProtocol
)
from hydra.flow import SendBuffer, ReceiveBuffer
from hydra.congestion import CongestionController, RTTEstimator


class TCPState(Enum):
    CLOSED = auto()
    LISTEN = auto()
    SYN_SENT = auto()
    SYN_RCVD = auto()
    ESTABLISHED = auto()
    FIN_WAIT_1 = auto()
    FIN_WAIT_2 = auto()
    CLOSE_WAIT = auto()
    CLOSING = auto()
    LAST_ACK = auto()
    TIME_WAIT = auto()


class TCPConnection:
    """
    RFC 793 Transmission Control Block (TCB) managing an active or passive TCP socket stream.
    """

    def __init__(
        self,
        local_ip: IPAddress,
        local_port: int,
        remote_ip: IPAddress,
        remote_port: int,
        send_callback: Callable[[TCPHeader, bytes], bool],
        mss: int = 1460
    ):
        self.local_ip = local_ip
        self.local_port = local_port
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.send_callback = send_callback
        self.mss = mss

        self.state: TCPState = TCPState.CLOSED
        self.iss = random.randint(10000, 999999)  # Initial Send Sequence Number
        self.irs = 0  # Initial Receive Sequence Number

        self.send_buf = SendBuffer(self.iss)
        self.recv_buf = ReceiveBuffer(0)
        self.congestion = CongestionController(mss=mss)
        self.rtt_est = RTTEstimator()

        self.peer_advertised_window = 65535
        self.last_rto_time = time.time()
        self.time_wait_start: Optional[float] = None
        self.time_wait_timeout = 2.0  # 2MSL in simulation

        # Event flags
        self.is_connected = False
        self.is_closed = False
        self.fin_sent = False
        self.fin_acked = False

    def connect(self) -> None:
        """Active Open: Transmit SYN and transition to SYN_SENT."""
        self.state = TCPState.SYN_SENT
        self.iss = random.randint(10000, 999999)
        self.send_buf = SendBuffer(self.iss + 1)
        self._send_raw_segment(flags=int(TCPFlags.SYN), seq=self.iss, ack=0)
        self.last_rto_time = time.time()

    def listen(self) -> None:
        """Passive Open: Transition to LISTEN."""
        self.state = TCPState.LISTEN

    def send(self, data: bytes) -> int:
        """Enqueue application bytes into send buffer and flush packets."""
        if self.state not in (TCPState.ESTABLISHED, TCPState.CLOSE_WAIT):
            raise ConnectionError(f"Cannot send data in TCP state {self.state.name}")

        self.send_buf.push_data(data)
        self.flush_send_window()
        return len(data)

    def recv(self, max_bytes: int = 4096) -> bytes:
        """Read bytes available in receive buffer."""
        return self.recv_buf.read(max_bytes)

    def close(self) -> None:
        """Initiate graceful 4-way teardown."""
        if self.state == TCPState.ESTABLISHED:
            self.state = TCPState.FIN_WAIT_1
            self._send_fin()
        elif self.state == TCPState.CLOSE_WAIT:
            self.state = TCPState.LAST_ACK
            self._send_fin()
        elif self.state == TCPState.LISTEN:
            self.state = TCPState.CLOSED
            self.is_closed = True

    def flush_send_window(self) -> None:
        """Transmit any pending data allowed by Congestion and Advertised Windows."""
        if self.state not in (TCPState.ESTABLISHED, TCPState.CLOSE_WAIT, TCPState.FIN_WAIT_1):
            return

        effective_win = self.congestion.effective_window(self.peer_advertised_window)
        max_seq = self.send_buf.una_seq + effective_win

        while True:
            chunk_info = self.send_buf.get_sendable_chunk(max_seq, self.mss)
            if chunk_info is None:
                break
            seq, chunk = chunk_info
            self._send_raw_segment(
                flags=int(TCPFlags.ACK) | int(TCPFlags.PSH),
                seq=seq,
                ack=self.recv_buf.rcv_nxt,
                payload=chunk
            )

    def _send_fin(self) -> None:
        self.fin_sent = True
        fin_seq = self.send_buf.next_seq
        self.send_buf.next_seq += 1
        self._send_raw_segment(
            flags=int(TCPFlags.FIN) | int(TCPFlags.ACK),
            seq=fin_seq,
            ack=self.recv_buf.rcv_nxt
        )

    def handle_segment(self, hdr: TCPHeader, payload: bytes) -> None:
        """Process incoming TCP segment according to RFC 793 FSM."""
        now = time.time()

        # Update peer advertised window
        self.peer_advertised_window = hdr.window_size

        # 1. Handle RST Flag
        if hdr.has_flag(TCPFlags.RST):
            self.state = TCPState.CLOSED
            self.is_closed = True
            self.is_connected = False
            return

        # 2. State: LISTEN
        if self.state == TCPState.LISTEN:
            if hdr.has_flag(TCPFlags.SYN):
                self.irs = hdr.seq_num
                self.recv_buf = ReceiveBuffer(self.irs + 1)
                self.iss = random.randint(10000, 999999)
                self.send_buf = SendBuffer(self.iss + 1)
                self.state = TCPState.SYN_RCVD
                # Transmit SYN+ACK
                self._send_raw_segment(
                    flags=int(TCPFlags.SYN) | int(TCPFlags.ACK),
                    seq=self.iss,
                    ack=self.recv_buf.rcv_nxt
                )
            return

        # 3. State: SYN_SENT
        if self.state == TCPState.SYN_SENT:
            if hdr.has_flag(TCPFlags.SYN) and hdr.has_flag(TCPFlags.ACK):
                if hdr.ack_num == self.iss + 1:
                    self.irs = hdr.seq_num
                    self.recv_buf = ReceiveBuffer(self.irs + 1)
                    self.state = TCPState.ESTABLISHED
                    self.is_connected = True
                    # Transmit final ACK
                    self._send_raw_segment(flags=int(TCPFlags.ACK), seq=self.send_buf.next_seq, ack=self.recv_buf.rcv_nxt)
            return

        # 4. State: SYN_RCVD
        if self.state == TCPState.SYN_RCVD:
            if hdr.has_flag(TCPFlags.ACK) and hdr.ack_num == self.iss + 1:
                self.state = TCPState.ESTABLISHED
                self.is_connected = True
            return

        # 5. Connected States (ESTABLISHED, FIN_WAIT_1, FIN_WAIT_2, CLOSE_WAIT, etc.)
        if hdr.has_flag(TCPFlags.ACK):
            bytes_acked, rtt_sample = self.send_buf.acknowledge(hdr.ack_num)
            if bytes_acked > 0:
                self.congestion.on_ack(bytes_acked)
                if rtt_sample is not None:
                    self.rtt_est.update(rtt_sample)
                self.last_rto_time = now
            elif self.send_buf.has_unacked_data():
                # Duplicate ACK detected
                must_fast_retransmit = self.congestion.on_duplicate_ack()
                if must_fast_retransmit:
                    oldest = self.send_buf.get_oldest_unacked()
                    if oldest is not None:
                        seq, chunk = oldest
                        self._send_raw_segment(
                            flags=int(TCPFlags.ACK),
                            seq=seq,
                            ack=self.recv_buf.rcv_nxt,
                            payload=chunk
                        )

        # Process Inbound Data Payload
        if payload:
            bytes_new = self.recv_buf.put_segment(hdr.seq_num, payload)
            # Send immediate ACK for received payload
            self._send_raw_segment(
                flags=int(TCPFlags.ACK),
                seq=self.send_buf.next_seq,
                ack=self.recv_buf.rcv_nxt
            )

        # Process FIN flag
        if hdr.has_flag(TCPFlags.FIN):
            self.recv_buf.rcv_nxt = hdr.seq_num + len(payload) + 1
            # Acknowledge peer's FIN
            self._send_raw_segment(flags=int(TCPFlags.ACK), seq=self.send_buf.next_seq, ack=self.recv_buf.rcv_nxt)

            if self.state == TCPState.ESTABLISHED:
                self.state = TCPState.CLOSE_WAIT
            elif self.state == TCPState.FIN_WAIT_1:
                self.state = TCPState.CLOSING
            elif self.state == TCPState.FIN_WAIT_2:
                self.state = TCPState.TIME_WAIT
                self.time_wait_start = now

        # Handle teardown completion states
        if self.state == TCPState.FIN_WAIT_1:
            if hdr.has_flag(TCPFlags.ACK) and self.fin_sent and hdr.ack_num >= self.send_buf.next_seq:
                self.state = TCPState.FIN_WAIT_2
        elif self.state == TCPState.CLOSING:
            if hdr.has_flag(TCPFlags.ACK) and hdr.ack_num >= self.send_buf.next_seq:
                self.state = TCPState.TIME_WAIT
                self.time_wait_start = now
        elif self.state == TCPState.LAST_ACK:
            if hdr.has_flag(TCPFlags.ACK) and hdr.ack_num >= self.send_buf.next_seq:
                self.state = TCPState.CLOSED
                self.is_closed = True

        # Send any more data pending in send buffer
        self.flush_send_window()

    def tick(self) -> None:
        """Periodic timer tick for RTO retransmission and TIME_WAIT expiration."""
        now = time.time()

        # Handle TIME_WAIT
        if self.state == TCPState.TIME_WAIT:
            if self.time_wait_start and (now - self.time_wait_start >= self.time_wait_timeout):
                self.state = TCPState.CLOSED
                self.is_closed = True
            return

        # Check Retransmission Timeout (RTO)
        if self.send_buf.has_unacked_data():
            if now - self.last_rto_time >= self.rtt_est.rto:
                self.congestion.on_rto_timeout()
                self.rtt_est.backoff_rto()
                self.last_rto_time = now

                oldest = self.send_buf.get_oldest_unacked()
                if oldest is not None:
                    seq, chunk = oldest
                    self._send_raw_segment(
                        flags=int(TCPFlags.ACK),
                        seq=seq,
                        ack=self.recv_buf.rcv_nxt,
                        payload=chunk
                    )

    def _send_raw_segment(self, flags: int, seq: int, ack: int, payload: bytes = b"") -> bool:
        win = self.recv_buf.advertised_window()
        hdr = TCPHeader(
            src_port=self.local_port,
            dst_port=self.remote_port,
            seq_num=seq,
            ack_num=ack,
            flags=flags,
            window_size=win
        )
        return self.send_callback(hdr, payload)
