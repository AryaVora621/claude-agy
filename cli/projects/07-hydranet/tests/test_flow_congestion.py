"""
HydraNet: Unit Tests for TCP Flow Control, Sliding Window Buffers, and Reno Congestion.
"""

import unittest
from hydra.flow import SendBuffer, ReceiveBuffer
from hydra.congestion import CongestionController, RTTEstimator


class TestFlowAndCongestion(unittest.TestCase):
    def test_send_buffer_window_slicing_and_ack(self):
        sbuf = SendBuffer(initial_seq=1000)
        sbuf.push_data(b"0123456789" * 10)  # 100 bytes

        # Allow sending up to seq 1040 with MSS 30
        c1 = sbuf.get_sendable_chunk(max_allowed_seq=1040, mss=30)
        self.assertIsNotNone(c1)
        seq1, chunk1 = c1
        self.assertEqual(seq1, 1000)
        self.assertEqual(len(chunk1), 30)

        # Second chunk up to 1040
        c2 = sbuf.get_sendable_chunk(max_allowed_seq=1040, mss=30)
        self.assertIsNotNone(c2)
        seq2, chunk2 = c2
        self.assertEqual(seq2, 1030)
        self.assertEqual(len(chunk2), 10)

        # Window exhausted (next_seq is 1040, max allowed is 1040)
        c3 = sbuf.get_sendable_chunk(max_allowed_seq=1040, mss=30)
        self.assertIsNone(c3)

        # Cumulative ACK acknowledging first 30 bytes (up to 1030)
        acked, rtt = sbuf.acknowledge(1030)
        self.assertEqual(acked, 30)
        self.assertEqual(sbuf.una_seq, 1030)
        self.assertIsNotNone(rtt)

    def test_receive_buffer_out_of_order_reassembly(self):
        rbuf = ReceiveBuffer(initial_seq=5000, capacity=10000)

        # Arrives out of order: Segment 2 (seq=5020, len=20) arrives before Segment 1 (seq=5000, len=20)
        bytes_added1 = rbuf.put_segment(5020, b"B" * 20)
        self.assertEqual(bytes_added1, 0)  # Buffered in out_of_order, not contiguous yet
        self.assertEqual(rbuf.rcv_nxt, 5000)
        self.assertEqual(len(rbuf.ready_bytes), 0)

        # Segment 1 arrives (seq=5000, len=20) - fills the gap!
        bytes_added2 = rbuf.put_segment(5000, b"A" * 20)
        self.assertEqual(bytes_added2, 20)
        # Both segments are now contiguous! RCV.NXT should advance to 5040
        self.assertEqual(rbuf.rcv_nxt, 5040)
        self.assertEqual(len(rbuf.ready_bytes), 40)

        data = rbuf.read(40)
        self.assertEqual(data, b"A" * 20 + b"B" * 20)

    def test_reno_slow_start_and_congestion_avoidance(self):
        cc = CongestionController(mss=1000, initial_ssthresh=4000)
        self.assertEqual(cc.cwnd, 2000.0)

        # Slow start: exponential increase
        cc.on_ack(1000)
        self.assertEqual(cc.cwnd, 3000.0)
        cc.on_ack(1000)
        self.assertEqual(cc.cwnd, 4000.0)

        # Reached ssthresh (4000) -> transitions to Congestion Avoidance
        # cwnd += (mss * bytes_acked) / cwnd = (1000 * 1000) / 4000 = 250
        cc.on_ack(1000)
        self.assertEqual(cc.cwnd, 4250.0)

    def test_reno_fast_retransmit_and_recovery(self):
        cc = CongestionController(mss=1000, initial_ssthresh=10000)
        cc.cwnd = 8000.0

        # Duplicate ACK 1 and 2
        self.assertFalse(cc.on_duplicate_ack())
        self.assertFalse(cc.on_duplicate_ack())

        # Duplicate ACK 3: Triggers Fast Retransmit!
        must_retrans = cc.on_duplicate_ack()
        self.assertTrue(must_retrans)
        self.assertTrue(cc.in_fast_recovery)
        # ssthresh = cwnd / 2 = 4000, cwnd = ssthresh + 3*mss = 7000
        self.assertEqual(cc.ssthresh, 4000.0)
        self.assertEqual(cc.cwnd, 7000.0)

        # Duplicate ACK 4: Inflate window
        self.assertFalse(cc.on_duplicate_ack())
        self.assertEqual(cc.cwnd, 8000.0)

        # Full ACK: Exits fast recovery, deflates window back to ssthresh
        cc.on_ack(500)
        self.assertFalse(cc.in_fast_recovery)
        self.assertEqual(cc.cwnd, 4000.0)

    def test_reno_timeout_collapse(self):
        cc = CongestionController(mss=1000, initial_ssthresh=16000)
        cc.cwnd = 12000.0

        cc.on_rto_timeout()
        self.assertEqual(cc.ssthresh, 6000.0)
        self.assertEqual(cc.cwnd, 1000.0)  # Reset to 1 MSS
        self.assertFalse(cc.in_fast_recovery)

    def test_rtt_estimator(self):
        rtt_est = RTTEstimator(default_rto=1.0)
        self.assertEqual(rtt_est.rto, 1.0)

        # First sample: 0.1s
        rto1 = rtt_est.update(0.1)
        self.assertAlmostEqual(rtt_est.srtt, 0.1)
        self.assertAlmostEqual(rtt_est.rttvar, 0.05)
        self.assertGreaterEqual(rto1, 0.2)

        # Exponential backoff
        rto_backed_off = rtt_est.backoff_rto()
        self.assertAlmostEqual(rto_backed_off, rto1 * 2.0)


if __name__ == "__main__":
    unittest.main()
