"""
HydraNet: TCP Congestion Control (RFC 5681 Reno) & RTT Estimator (RFC 6298).
Implements Slow Start, Congestion Avoidance, Fast Retransmit, Fast Recovery,
and Jacobson/Karels smoothed round-trip time (SRTT) calculation.
"""

from typing import Tuple


class CongestionController:
    """
    RFC 5681 TCP Reno Congestion Control Engine.
    Governs transmission rate via cwnd, ssthresh, and duplicate ACK tracking.
    """

    def __init__(self, mss: int = 1460, initial_ssthresh: int = 65535):
        self.mss = mss
        self.cwnd = float(2 * mss)  # Initial window: 2 segments
        self.ssthresh = float(initial_ssthresh)
        self.dup_ack_count = 0
        self.in_fast_recovery = False

    def on_ack(self, bytes_acked: int) -> None:
        """Process new cumulative acknowledgement advancing the window."""
        if self.in_fast_recovery:
            # Full ACK received: Exit Fast Recovery
            self.cwnd = self.ssthresh
            self.dup_ack_count = 0
            self.in_fast_recovery = False
            return

        self.dup_ack_count = 0

        if self.cwnd < self.ssthresh:
            # Phase 1: Slow Start (Exponential growth per RTT)
            self.cwnd += bytes_acked
        else:
            # Phase 2: Congestion Avoidance (Additive Increase: ~1 MSS per RTT)
            self.cwnd += (self.mss * bytes_acked) / self.cwnd

    def on_duplicate_ack(self) -> bool:
        """
        Handle duplicate ACK.
        Returns True if Fast Retransmit must be immediately executed (at 3 dup ACKs).
        """
        self.dup_ack_count += 1

        if self.dup_ack_count == 3:
            # Trigger Fast Retransmit & Enter Fast Recovery
            self.ssthresh = max(float(2 * self.mss), self.cwnd / 2.0)
            self.cwnd = self.ssthresh + (3 * self.mss)
            self.in_fast_recovery = True
            return True  # Retransmit lost segment immediately!

        if self.dup_ack_count > 3 and self.in_fast_recovery:
            # Inflate window for each additional duplicate ACK
            self.cwnd += self.mss

        return False

    def on_rto_timeout(self) -> None:
        """Handle Retransmission Timeout (severe packet loss)."""
        self.ssthresh = max(float(2 * self.mss), self.cwnd / 2.0)
        self.cwnd = float(1 * self.mss)  # Collapse to 1 MSS
        self.dup_ack_count = 0
        self.in_fast_recovery = False

    def effective_window(self, advertised_window: int) -> int:
        """Effective send window is min(Congestion Window, Receiver Advertised Window)."""
        return int(min(self.cwnd, max(1, advertised_window)))


class RTTEstimator:
    """
    RFC 6298 Round-Trip Time and Retransmission Timeout (RTO) Calculation.
    Uses Jacobson and Karels' algorithm with alpha=0.125 and beta=0.25.
    """

    def __init__(self, default_rto: float = 1.0, min_rto: float = 0.2, max_rto: float = 60.0):
        self.default_rto = default_rto
        self.min_rto = min_rto
        self.max_rto = max_rto
        self.srtt: float = 0.0
        self.rttvar: float = 0.0
        self.rto: float = default_rto
        self.has_measurement = False

    def update(self, measured_rtt: float) -> float:
        """Update RTT metrics with new round-trip sample (in seconds)."""
        if not self.has_measurement:
            self.srtt = measured_rtt
            self.rttvar = measured_rtt / 2.0
            self.rto = self.srtt + max(0.1, 4.0 * self.rttvar)
            self.has_measurement = True
        else:
            alpha = 0.125
            beta = 0.25
            diff = abs(self.srtt - measured_rtt)
            self.rttvar = (1.0 - beta) * self.rttvar + beta * diff
            self.srtt = (1.0 - alpha) * self.srtt + alpha * measured_rtt
            self.rto = self.srtt + max(0.1, 4.0 * self.rttvar)

        # Clamp RTO within bounds
        self.rto = max(self.min_rto, min(self.max_rto, self.rto))
        return self.rto

    def backoff_rto(self) -> float:
        """Exponential timer backoff on repeated retransmission timeout (Karn's algorithm)."""
        self.rto = min(self.max_rto, self.rto * 2.0)
        return self.rto
