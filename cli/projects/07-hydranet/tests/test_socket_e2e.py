"""
HydraNet: End-to-End Client/Server Socket Integration & Lossy Network Tests.
"""

import unittest
import time
from hydra.types import (
    MACAddress,
    IPAddress
)
from hydra.device import VirtualEthernetBus
from hydra.socket import NetworkStack, TCPSocket


class TestSocketEndToEnd(unittest.TestCase):
    def setUp(self):
        self.bus = VirtualEthernetBus("wire0", drop_rate=0.0)

        # Server Node (192.168.1.10)
        self.server_stack = NetworkStack(
            name="srv_host",
            mac=MACAddress("00:50:56:c0:00:01"),
            ip=IPAddress("192.168.1.10"),
            netmask=IPAddress("255.255.255.0")
        )
        self.bus.attach(self.server_stack.device)

        # Client Node (192.168.1.20)
        self.client_stack = NetworkStack(
            name="cli_host",
            mac=MACAddress("00:50:56:c0:00:02"),
            ip=IPAddress("192.168.1.20"),
            netmask=IPAddress("255.255.255.0")
        )
        self.bus.attach(self.client_stack.device)

    def test_client_server_echo_exchange(self):
        # 1. Setup Server
        server_sock = TCPSocket(self.server_stack)
        server_sock.bind(("192.168.1.10", 8080))
        server_sock.listen()

        # 2. Client Connects
        client_sock = TCPSocket(self.client_stack)
        client_sock.connect(("192.168.1.10", 8080), timeout=2.0)

        # 3. Server Accepts
        conn_sock, client_addr = server_sock.accept(timeout=2.0)
        self.assertEqual(client_addr[0], "192.168.1.20")

        # 4. Client sends data
        msg = b"Hello HydraNet POSIX Stack!"
        client_sock.send(msg)

        # 5. Server receives and echoes back
        received = conn_sock.recv(4096, timeout=2.0)
        self.assertEqual(received, msg)

        conn_sock.send(b"ECHO: " + received)

        # 6. Client receives echo
        echo_resp = client_sock.recv(4096, timeout=2.0)
        self.assertEqual(echo_resp, b"ECHO: Hello HydraNet POSIX Stack!")

        # 7. Teardown
        client_sock.close()
        conn_sock.close()
        server_sock.close()

    def test_multi_segment_stream(self):
        server_sock = TCPSocket(self.server_stack)
        server_sock.bind(("192.168.1.10", 9000))
        server_sock.listen()

        client_sock = TCPSocket(self.client_stack)
        client_sock.connect(("192.168.1.10", 9000), timeout=2.0)

        conn_sock, _ = server_sock.accept(timeout=2.0)

        # Send 10,000 bytes (exceeds single MSS of 1460 bytes)
        large_data = b"0123456789ABCDEF" * 625  # 10,000 bytes
        client_sock.send(large_data)

        # Server reads entire stream
        total_read = bytearray()
        start = time.time()
        while len(total_read) < len(large_data) and time.time() - start < 3.0:
            chunk = conn_sock.recv(4096, timeout=0.5)
            if chunk:
                total_read.extend(chunk)

        self.assertEqual(bytes(total_read), large_data)
        client_sock.close()
        conn_sock.close()
        server_sock.close()


if __name__ == "__main__":
    unittest.main()
