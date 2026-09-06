#!/usr/bin/env python3
"""
NexusOS: Microkernel Performance Benchmark Suite.
Evaluates:
1. Context Switch Latency & Throughput (switches/sec)
2. Synchronous IPC Ping-Pong Round-Trip Rate (transactions/sec)
3. Copy-On-Write Fork & Page Fault Overhead (ops/sec)
4. Unix Pipe Streaming Bandwidth (MB/s)
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexus.kernel import NexusKernel
from nexus.types import PageFlags, Errno


def bench_context_switches(num_switches: int = 50000):
    print(f"1. Benchmarking Context Switch Throughput ({num_switches:,} switches)...")
    kernel = NexusKernel(num_frames=64)

    p1 = kernel.create_process("worker_1")
    p2 = kernel.create_process("worker_2")

    t0 = time.perf_counter()
    for _ in range(num_switches):
        kernel.sys_yield()
    elapsed = time.perf_counter() - t0

    rate = num_switches / elapsed
    us_per_switch = (elapsed / num_switches) * 1e6
    print(f"   -> Rate: {rate:,.0f} switches/sec")
    print(f"   -> Latency: {us_per_switch:.3f} µs/switch\n")
    return rate


def bench_ipc_ping_pong(num_round_trips: int = 25000):
    print(f"2. Benchmarking Synchronous IPC Ping-Pong ({num_round_trips:,} round-trips)...")
    kernel = NexusKernel(num_frames=64)

    client = kernel.create_process("client")
    server = kernel.create_process("server")

    ep_req = "req_channel"
    ep_res = "res_channel"

    t0 = time.perf_counter()
    for _ in range(num_round_trips):
        # Client sends request
        kernel.set_active_process(client.pid)
        kernel.sys_ipc_send(ep_req, "PING")

        # Server receives request and sends reply
        kernel.set_active_process(server.pid)
        kernel.sys_ipc_recv(ep_req)
        kernel.sys_ipc_send(ep_res, "PONG")

        # Client receives reply
        kernel.set_active_process(client.pid)
        kernel.sys_ipc_recv(ep_res)

    elapsed = time.perf_counter() - t0
    rate = num_round_trips / elapsed
    latency_us = (elapsed / num_round_trips) * 1e6

    print(f"   -> Rate: {rate:,.0f} round-trips/sec")
    print(f"   -> Latency: {latency_us:.3f} µs/round-trip\n")
    return rate


def bench_cow_fork(num_forks: int = 10000):
    print(f"3. Benchmarking Copy-On-Write (COW) Fork & Fault Overhead ({num_forks:,} cycles)...")
    kernel = NexusKernel(num_frames=128)

    parent = kernel.create_process("parent")
    kernel.set_active_process(parent.pid)

    # Map 4 pages with initial data
    vaddr = 0x00040000
    kernel.sys_mmap(vaddr, num_pages=4, flags=PageFlags.READABLE | PageFlags.WRITABLE)
    kernel.mmu.write_memory(vaddr, b"INITIAL_PARENT_DATA")

    t0 = time.perf_counter()
    for _ in range(num_forks):
        # 1. Fork child (shares frames via COW)
        kernel.set_active_process(parent.pid)
        status, child_pid = kernel.sys_fork()

        # 2. Child writes to memory (triggers COW fault)
        kernel.set_active_process(child_pid)
        kernel.mmu.write_memory(vaddr, b"CHILD_NEW_DATA")

        # 3. Child exits
        kernel.sys_exit(0)

        # 4. Parent reaps child
        kernel.set_active_process(parent.pid)
        kernel.sys_waitpid(child_pid)

    elapsed = time.perf_counter() - t0
    rate = num_forks / elapsed
    latency_us = (elapsed / num_forks) * 1e6

    print(f"   -> Rate: {rate:,.0f} fork-cow-reap cycles/sec")
    print(f"   -> Latency: {latency_us:.3f} µs/cycle\n")
    return rate


def bench_pipe_throughput(total_bytes: int = 20 * 1024 * 1024):
    mb = total_bytes / (1024 * 1024)
    print(f"4. Benchmarking Unix Pipe Streaming Bandwidth ({mb:.1f} MB)...")
    kernel = NexusKernel(num_frames=64)

    proc = kernel.create_process("pipe_tester")
    kernel.set_active_process(proc.pid)

    status, r_fd, w_fd = kernel.sys_pipe()
    chunk = b"X" * 4096

    t0 = time.perf_counter()
    transferred = 0
    while transferred < total_bytes:
        kernel.sys_write(w_fd, chunk)
        kernel.sys_read(r_fd, 4096)
        transferred += 4096

    elapsed = time.perf_counter() - t0
    bandwidth_mb_s = mb / elapsed

    print(f"   -> Throughput: {bandwidth_mb_s:,.1f} MB/s ({elapsed:.3f}s)\n")
    return bandwidth_mb_s


def main():
    print("=" * 70)
    print("           NEXUS-OS MICROKERNEL BENCHMARK MATRIX")
    print("=" * 70 + "\n")

    bench_context_switches(50000)
    bench_ipc_ping_pong(25000)
    bench_cow_fork(5000)
    bench_pipe_throughput(20 * 1024 * 1024)

    print("=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
