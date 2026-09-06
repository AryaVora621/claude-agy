#!/usr/bin/env python3
"""
HydraNet: Echo Client/Server and Lossy Wire Transmission Showcase.
Demonstrates:
1. Standard Berkeley Socket client/server bi-directional stream communication.
2. Transmission across an impaired Layer 2 channel with physical packet loss.
3. TCP Reno Congestion Control telemetry (cwnd expansion, fast retransmit, RTO backoff).
4. Zero-loss reliable delivery verification under hostile channel conditions.
"""

import sys
import os
import time
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra.types import (
    MACAddress,
    IPAddress
)
from hydra.device import VirtualEthernetBus
from hydra.socket import NetworkStack, TCPSocket


# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"


def run_clean_stream_demo() -> None:
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}             DEMO 1: RELIABLE DUPLEX STREAM OVER VIRTUAL BUS            {RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

    bus = VirtualEthernetBus("wire0", drop_rate=0.0)
    server_stack = NetworkStack(
        name="server_node",
        mac=MACAddress("00:50:56:a1:00:01"),
        ip=IPAddress("10.0.1.10"),
        netmask=IPAddress("255.255.255.0")
    )
    client_stack = NetworkStack(
        name="client_node",
        mac=MACAddress("00:50:56:a1:00:02"),
        ip=IPAddress("10.0.1.20"),
        netmask=IPAddress("255.255.255.0")
    )
    bus.attach(server_stack.device)
    bus.attach(client_stack.device)

    # Server Socket Setup
    srv_sock = TCPSocket(server_stack)
    srv_sock.bind(("10.0.1.10", 9000))
    srv_sock.listen()
    print(f"[*] Server listening on {srv_sock.local_ip}:{srv_sock.local_port}")

    # Client Connect
    cli_sock = TCPSocket(client_stack)
    print(f"[*] Client initiating 3-way handshake to 10.0.1.10:9000...")
    cli_sock.connect(("10.0.1.10", 9000))
    print(f"    -> Client state: {GREEN}ESTABLISHED{RESET} (Local Ephemeral Port: {cli_sock.local_port})")

    conn_sock, cli_addr = srv_sock.accept()
    print(f"    -> Server accepted inbound connection from {cli_addr[0]}:{cli_addr[1]}")

    # Ping Pong exchanges
    test_messages = [
        b"PING #1: Hello HydraNet!",
        b"PING #2: Zero-dependency user-space TCP/IP protocol stack.",
        b"PING #3: RFC 793 FSM + RFC 5681 Reno Congestion Control."
    ]

    for msg in test_messages:
        print(f"\n[>] Client Sending: '{msg.decode()}'")
        cli_sock.send(msg)

        srv_received = conn_sock.recv(1024)
        print(f"[<] Server Received: '{srv_received.decode()}'")

        echo_resp = b"ACK-ECHO: " + srv_received
        conn_sock.send(echo_resp)

        cli_received = cli_sock.recv(1024)
        print(f"[<] Client Received: '{cli_received.decode()}'")
        assert cli_received == echo_resp

    # Teardown
    print("\n[*] Performing graceful 4-way FIN teardown...")
    cli_sock.close()
    conn_sock.close()
    srv_sock.close()
    print(f"    -> Connection closed cleanly.")


def run_lossy_wire_demo() -> None:
    print(f"\n{BOLD}{MAGENTA}========================================================================{RESET}")
    print(f"{BOLD}{MAGENTA}        DEMO 2: LOSSY CHANNEL RECOVERY & TCP RENO TELEMETRY            {RESET}")
    print(f"{BOLD}{MAGENTA}========================================================================{RESET}\n")

    drop_rate = 0.08  # 8% random packet loss on physical wire
    bus = VirtualEthernetBus("wire_lossy", drop_rate=drop_rate)

    server_stack = NetworkStack(
        name="server_node",
        mac=MACAddress("00:50:56:b2:00:01"),
        ip=IPAddress("172.16.0.10"),
        netmask=IPAddress("255.255.0.0")
    )
    client_stack = NetworkStack(
        name="client_node",
        mac=MACAddress("00:50:56:b2:00:02"),
        ip=IPAddress("172.16.0.20"),
        netmask=IPAddress("255.255.0.0")
    )
    bus.attach(server_stack.device)
    bus.attach(client_stack.device)

    print(f"[*] Initialized Virtual Ethernet Bus with {RED}{drop_rate * 100:.0f}% random physical packet drop rate{RESET}")

    srv_sock = TCPSocket(server_stack)
    srv_sock.bind(("172.16.0.10", 4444))
    srv_sock.listen()

    cli_sock = TCPSocket(client_stack)
    cli_sock.connect(("172.16.0.10", 4444))
    conn_sock, _ = srv_sock.accept()

    # Generate test payload
    payload_size = 16_000  # 16 KB data transfer
    test_payload = os.urandom(payload_size)
    expected_sha256 = hashlib.sha256(test_payload).hexdigest()

    print(f"[*] Transmitting {payload_size:,} bytes payload (SHA256: {expected_sha256[:16]}...)...")

    # Send data
    t0 = time.perf_counter()
    cli_sock.send(test_payload)

    # Server receives with periodic Reno telemetry dump
    received_bytes = bytearray()
    last_dump = time.time()

    while len(received_bytes) < payload_size:
        chunk = conn_sock.recv(2048, timeout=0.1)
        if chunk:
            received_bytes.extend(chunk)

        # Print Reno Congestion Controller state on sender
        if time.time() - last_dump > 0.05:
            cc = cli_sock.conn.congestion if cli_sock.conn else None
            if cc:
                print(
                    f"    [Reno Telemetry] CWND: {cc.cwnd:6.1f} B ({cc.cwnd / cc.mss:4.1f} MSS) | "
                    f"ssthresh: {cc.ssthresh:6.1f} B | DupACKs: {cc.dup_ack_count} | "
                    f"FastRecovery: {cc.in_fast_recovery} | Wire Drops: {bus.frames_dropped}"
                )
            last_dump = time.time()

    elapsed = time.perf_counter() - t0
    received_sha256 = hashlib.sha256(received_bytes).hexdigest()

    print(f"\n{BOLD}{GREEN}[✓] Data Transfer Completed Across Lossy Wire!{RESET}")
    print(f"    -> Bytes Transferred : {len(received_bytes):,} / {payload_size:,}")
    print(f"    -> Elapsed Time      : {elapsed * 1000:.2f} ms")
    print(f"    -> Frames Dropped    : {bus.frames_dropped} frames")
    print(f"    -> Transmit SHA256   : {expected_sha256}")
    print(f"    -> Received SHA256   : {received_sha256}")
    print(f"    -> Integrity Match   : {BOLD}{GREEN}{expected_sha256 == received_sha256}{RESET}")

    cli_sock.close()
    conn_sock.close()
    srv_sock.close()


def main() -> None:
    print(f"{BOLD}{YELLOW}========================================================================{RESET}")
    print(f"{BOLD}{YELLOW}           HYDRANET: TCP/IP CLIENT/SERVER ECHO DEMONSTRATION            {RESET}")
    print(f"{BOLD}{YELLOW}========================================================================{RESET}")

    run_clean_stream_demo()
    run_lossy_wire_demo()


if __name__ == "__main__":
    main()
