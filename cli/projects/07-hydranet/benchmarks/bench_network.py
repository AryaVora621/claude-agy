#!/usr/bin/env python3
"""
HydraNet: Protocol Stack Performance Benchmarks.
Measures:
1. RFC 1071 One's Complement 16-bit Checksum Throughput (MB/s).
2. Layer 2 Ethernet Frame Packing and Unpacking Rates (frames/sec).
3. Layer 3 IPv4 Longest Prefix Match (LPM) Subnet Route Lookups (lookups/sec).
4. Layer 3 IPv4 Dynamic Fragmentation and Reassembly Throughput (datagrams/sec).
5. Layer 4 TCP Stream Throughput and Latency over Virtual Ethernet Bus.
"""

import sys
import os
import time
import struct
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra.types import (
    MACAddress,
    IPAddress,
    EthernetFrame,
    EthernetType,
    IPv4Header,
    IPProtocol,
    TCPHeader,
    TCPFlags,
    compute_checksum,
    compute_tcp_checksum
)
from hydra.device import VirtualEthernetBus, VirtualNetDevice
from hydra.arp import ARPEngine
from hydra.ip import IPEngine, RouteEntry
from hydra.socket import NetworkStack, TCPSocket


def bench_checksum(payload_size_kb: int = 64, iterations: int = 500) -> None:
    print(f"[*] Benchmarking RFC 1071 16-Bit Checksum ({payload_size_kb} KB block, {iterations} iterations)...")
    data = os.urandom(payload_size_kb * 1024)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = compute_checksum(data)
    elapsed = time.perf_counter() - t0

    total_mb = (payload_size_kb * 1024 * iterations) / (1024 * 1024)
    throughput = total_mb / elapsed
    print(f"    -> Processed {total_mb:.1f} MB in {elapsed * 1000:.2f} ms")
    print(f"    -> Checksum Throughput: {throughput:.2f} MB/s\n")


def bench_ethernet_framing(iterations: int = 50_000) -> None:
    print(f"[*] Benchmarking Ethernet II Frame Serialization ({iterations:,} frames)...")
    src = MACAddress("00:50:56:c0:00:01")
    dst = MACAddress("00:50:56:c0:00:02")
    payload = b"X" * 128

    t0 = time.perf_counter()
    for _ in range(iterations):
        frame = EthernetFrame(dst, src, int(EthernetType.IPV4), payload)
        raw = frame.pack()
        _ = EthernetFrame.unpack(raw)
    elapsed = time.perf_counter() - t0

    fps = iterations / elapsed
    total_bytes = (len(payload) + 14) * iterations
    mb_s = (total_bytes / (1024 * 1024)) / elapsed
    print(f"    -> Pack/Unpack {iterations:,} frames in {elapsed * 1000:.2f} ms")
    print(f"    -> Frame Rate: {fps:,.0f} frames/sec ({mb_s:.2f} MB/s)\n")


def bench_ipv4_lpm_routing(table_size: int = 250, lookups: int = 40_000) -> None:
    print(f"[*] Benchmarking IPv4 Longest Prefix Match (LPM) ({table_size} routes, {lookups:,} lookups)...")
    engine = IPEngine()
    dummy_dev = VirtualNetDevice("eth0", MACAddress("00:00:00:00:00:01"), IPAddress("10.0.0.1"))
    dummy_arp = ARPEngine(dummy_dev)
    engine.register_device(dummy_dev, dummy_arp)

    # Populate realistic CIDR routing table: /8, /16, /24, /30 prefixes
    prefixes = [8, 16, 24, 28, 30]
    for i in range(table_size):
        prefix_len = prefixes[i % len(prefixes)]
        mask_int = ((1 << prefix_len) - 1) << (32 - prefix_len) if prefix_len > 0 else 0
        netmask = IPAddress.from_int(mask_int)
        net_ip = IPAddress(f"10.{(i * 7) % 256}.{(i * 13) % 256}.0")
        engine.add_route(net_ip, netmask, None, dummy_dev, metric=1)

    # Default route 0.0.0.0/0
    engine.add_route(IPAddress("0.0.0.0"), IPAddress("0.0.0.0"), None, dummy_dev, metric=100)

    # Generate random test target IP addresses
    random.seed(42)
    targets = [IPAddress(f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}") for _ in range(1000)]

    t0 = time.perf_counter()
    for i in range(lookups):
        dst = targets[i % len(targets)]
        _ = engine.find_route(dst)
    elapsed = time.perf_counter() - t0

    lookups_per_sec = lookups / elapsed
    print(f"    -> Completed {lookups:,} LPM lookups in {elapsed * 1000:.2f} ms")
    print(f"    -> Route Lookup Speed: {lookups_per_sec:,.0f} lookups/sec\n")


def bench_ipv4_fragmentation_reassembly(datagram_size: int = 12_000, count: int = 500) -> None:
    print(f"[*] Benchmarking IPv4 Fragmentation and Reassembly ({datagram_size} bytes datagram, MTU=1500)...")
    bus = VirtualEthernetBus("wire_bench")
    src_dev = VirtualNetDevice("eth0", MACAddress("00:00:00:00:00:01"), IPAddress("192.168.1.1"), mtu=1500)
    dst_dev = VirtualNetDevice("eth1", MACAddress("00:00:00:00:00:02"), IPAddress("192.168.1.2"), mtu=1500)
    bus.attach(src_dev)
    bus.attach(dst_dev)

    src_arp = ARPEngine(src_dev)
    dst_arp = ARPEngine(dst_dev)
    src_ip = IPEngine()
    dst_ip = IPEngine()

    src_ip.register_device(src_dev, src_arp)
    dst_ip.register_device(dst_dev, dst_arp)
    src_ip.add_route(IPAddress("192.168.1.0"), IPAddress("255.255.255.0"), None, src_dev)
    dst_ip.add_route(IPAddress("192.168.1.0"), IPAddress("255.255.255.0"), None, dst_dev)

    # Prime ARP
    src_arp.insert_static(IPAddress("192.168.1.2"), MACAddress("00:00:00:00:00:02"))
    dst_arp.insert_static(IPAddress("192.168.1.1"), MACAddress("00:00:00:00:00:01"))

    reassembled_count = [0]
    def on_reassemble(hdr: IPv4Header, payload: bytes) -> None:
        reassembled_count[0] += 1

    dst_ip.register_protocol(IPProtocol.TCP, on_reassemble)

    payload = os.urandom(datagram_size)

    t0 = time.perf_counter()
    for _ in range(count):
        src_ip.send_packet(IPAddress("192.168.1.1"), IPAddress("192.168.1.2"), IPProtocol.TCP, payload)
        while True:
            frame = dst_dev.poll_frame()
            if frame is None:
                break
            dst_ip.handle_ip_packet(frame.payload)
    elapsed = time.perf_counter() - t0

    total_mb = (datagram_size * count) / (1024 * 1024)
    rate_mb = total_mb / elapsed
    dps = count / elapsed
    print(f"    -> Reassembled {reassembled_count[0]}/{count} datagrams ({total_mb:.2f} MB) in {elapsed * 1000:.2f} ms")
    print(f"    -> Reassembly Rate: {dps:,.1f} datagrams/sec ({rate_mb:.2f} MB/s)\n")


def bench_tcp_socket_stream(transfer_bytes: int = 50_000) -> None:
    print(f"[*] Benchmarking TCP Socket End-to-End Stream ({transfer_bytes:,} bytes)...")
    bus = VirtualEthernetBus("tcp_bench")
    srv = NetworkStack("srv", MACAddress("00:11:22:33:44:01"), IPAddress("10.0.0.1"), IPAddress("255.255.255.0"))
    cli = NetworkStack("cli", MACAddress("00:11:22:33:44:02"), IPAddress("10.0.0.2"), IPAddress("255.255.255.0"))
    bus.attach(srv.device)
    bus.attach(cli.device)

    srv_sock = TCPSocket(srv)
    srv_sock.bind(("10.0.0.1", 8080))
    srv_sock.listen()

    cli_sock = TCPSocket(cli)
    cli_sock.connect(("10.0.0.1", 8080))

    conn_sock, _ = srv_sock.accept()

    stream_data = os.urandom(transfer_bytes)

    t0 = time.perf_counter()
    cli_sock.send(stream_data)

    read_buf = bytearray()
    while len(read_buf) < transfer_bytes:
        chunk = conn_sock.recv(4096, timeout=1.0)
        if chunk:
            read_buf.extend(chunk)

    elapsed = time.perf_counter() - t0
    cli_sock.close()
    conn_sock.close()
    srv_sock.close()

    throughput_kb = (transfer_bytes / 1024) / elapsed
    print(f"    -> Transferred {len(read_buf):,} bytes in {elapsed * 1000:.2f} ms")
    print(f"    -> TCP End-to-End Goodput: {throughput_kb:,.1f} KB/s\n")


def run_all_benchmarks() -> None:
    print("==================================================================")
    print("      HYDRANET: USER-SPACE TCP/IP STACK BENCHMARK SUITE          ")
    print("==================================================================\n")
    bench_checksum()
    bench_ethernet_framing()
    bench_ipv4_lpm_routing()
    bench_ipv4_fragmentation_reassembly()
    bench_tcp_socket_stream()
    print("==================================================================")
    print("            ALL HYDRANET BENCHMARKS COMPLETED                     ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_benchmarks()
