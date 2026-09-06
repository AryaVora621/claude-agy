"""
HydraNet: Unit Tests for RFC 793 TCP Finite State Machine Transitions.
"""

import unittest
from hydra.types import (
    IPAddress,
    TCPHeader,
    TCPFlags
)
from hydra.flow import SendBuffer
from hydra.tcp import TCPConnection, TCPState


class TestTCPStateMachine(unittest.TestCase):
    def test_three_way_handshake(self):
        client_outbox = []
        server_outbox = []

        def client_send(hdr: TCPHeader, payload: bytes) -> bool:
            client_outbox.append((hdr, payload))
            return True

        def server_send(hdr: TCPHeader, payload: bytes) -> bool:
            server_outbox.append((hdr, payload))
            return True

        client = TCPConnection(
            local_ip=IPAddress("10.0.0.1"),
            local_port=50000,
            remote_ip=IPAddress("10.0.0.2"),
            remote_port=80,
            send_callback=client_send
        )
        server = TCPConnection(
            local_ip=IPAddress("10.0.0.2"),
            local_port=80,
            remote_ip=IPAddress("10.0.0.1"),
            remote_port=50000,
            send_callback=server_send
        )

        # 1. Server listens
        server.listen()
        self.assertEqual(server.state, TCPState.LISTEN)

        # 2. Client initiates active open (SYN)
        client.connect()
        self.assertEqual(client.state, TCPState.SYN_SENT)
        self.assertEqual(len(client_outbox), 1)
        syn_hdr, _ = client_outbox.pop(0)
        self.assertTrue(syn_hdr.has_flag(TCPFlags.SYN))
        self.assertFalse(syn_hdr.has_flag(TCPFlags.ACK))

        # 3. Server processes SYN, transmits SYN+ACK
        server.handle_segment(syn_hdr, b"")
        self.assertEqual(server.state, TCPState.SYN_RCVD)
        self.assertEqual(len(server_outbox), 1)
        syn_ack_hdr, _ = server_outbox.pop(0)
        self.assertTrue(syn_ack_hdr.has_flag(TCPFlags.SYN))
        self.assertTrue(syn_ack_hdr.has_flag(TCPFlags.ACK))
        self.assertEqual(syn_ack_hdr.ack_num, syn_hdr.seq_num + 1)

        # 4. Client processes SYN+ACK, transitions to ESTABLISHED, transmits ACK
        client.handle_segment(syn_ack_hdr, b"")
        self.assertEqual(client.state, TCPState.ESTABLISHED)
        self.assertEqual(len(client_outbox), 1)
        ack_hdr, _ = client_outbox.pop(0)
        self.assertTrue(ack_hdr.has_flag(TCPFlags.ACK))
        self.assertFalse(ack_hdr.has_flag(TCPFlags.SYN))

        # 5. Server processes final ACK, transitions to ESTABLISHED
        server.handle_segment(ack_hdr, b"")
        self.assertEqual(server.state, TCPState.ESTABLISHED)

        # Both endpoints now fully connected!
        self.assertTrue(client.is_connected)
        self.assertTrue(server.is_connected)

    def test_data_transfer_and_teardown(self):
        c_out = []
        s_out = []

        def c_send(h: TCPHeader, p: bytes) -> bool:
            c_out.append((h, p))
            return True

        def s_send(h: TCPHeader, p: bytes) -> bool:
            s_out.append((h, p))
            return True

        client = TCPConnection(
            local_ip=IPAddress("10.0.0.1"),
            local_port=50000,
            remote_ip=IPAddress("10.0.0.2"),
            remote_port=80,
            send_callback=c_send
        )
        server = TCPConnection(
            local_ip=IPAddress("10.0.0.2"),
            local_port=80,
            remote_ip=IPAddress("10.0.0.1"),
            remote_port=50000,
            send_callback=s_send
        )

        # Pre-establish connection
        client.state = TCPState.ESTABLISHED
        server.state = TCPState.ESTABLISHED
        client.is_connected = True
        server.is_connected = True
        client.send_buf = SendBuffer(client.iss + 1)
        server.send_buf = SendBuffer(server.iss + 1)
        client.recv_buf.rcv_nxt = server.iss + 1
        server.recv_buf.rcv_nxt = client.iss + 1

        # Client sends data
        msg = b"GET /index.html HTTP/1.1\r\n\r\n"
        client.send(msg)
        self.assertEqual(len(c_out), 1)
        data_hdr, data_payload = c_out.pop(0)
        self.assertEqual(data_payload, msg)

        # Server receives data and sends ACK
        server.handle_segment(data_hdr, data_payload)
        recv_msg = server.recv(1024)
        self.assertEqual(recv_msg, msg)
        self.assertEqual(len(s_out), 1)
        ack_hdr, _ = s_out.pop(0)
        self.assertTrue(ack_hdr.has_flag(TCPFlags.ACK))

        # Client receives ACK
        client.handle_segment(ack_hdr, b"")

        # 4-Way Teardown: Client closes
        client.close()
        self.assertEqual(client.state, TCPState.FIN_WAIT_1)
        self.assertEqual(len(c_out), 1)
        fin_hdr, _ = c_out.pop(0)
        self.assertTrue(fin_hdr.has_flag(TCPFlags.FIN))

        # Server receives FIN, sends ACK, enters CLOSE_WAIT
        server.handle_segment(fin_hdr, b"")
        self.assertEqual(server.state, TCPState.CLOSE_WAIT)
        self.assertEqual(len(s_out), 1)
        s_fin_ack, _ = s_out.pop(0)

        # Client receives ACK of FIN, enters FIN_WAIT_2
        client.handle_segment(s_fin_ack, b"")
        self.assertEqual(client.state, TCPState.FIN_WAIT_2)

        # Server closes, sends FIN, enters LAST_ACK
        server.close()
        self.assertEqual(server.state, TCPState.LAST_ACK)
        self.assertEqual(len(s_out), 1)
        server_fin, _ = s_out.pop(0)

        # Client receives server FIN, sends ACK, enters TIME_WAIT
        client.handle_segment(server_fin, b"")
        self.assertEqual(client.state, TCPState.TIME_WAIT)
        self.assertEqual(len(c_out), 1)
        c_final_ack, _ = c_out.pop(0)

        # Server receives final ACK, enters CLOSED
        server.handle_segment(c_final_ack, b"")
        self.assertEqual(server.state, TCPState.CLOSED)


if __name__ == "__main__":
    unittest.main()
