#!/usr/bin/env python3
"""
NexusOS: Interactive ASCII Top & Process Management Simulation.
Spawns init, shell, CPU-intensive, interactive I/O, and pipe-connected processes.
Visualizes MLFQ queue demotions, priority boosts, and memory allocation in real time.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexus.kernel import NexusKernel
from nexus.visualizer import KernelVisualizer


def run_simulation():
    print("Initializing NexusOS Microkernel Simulation...\n")
    kernel = NexusKernel(num_frames=256)

    # 1. Spawn Init Process (PID 1)
    def init_routine(k: NexusKernel, pcb):
        # Long-lived supervisor
        k.sys_yield()

    kernel.create_process("init", routine=init_routine)

    # 2. Spawn Interactive I/O Process (Yields frequently, staying in Q0)
    def interactive_routine(k: NexusKernel, pcb):
        # Emulates user keystroke listener
        k.sys_sleep(2)

    kernel.create_process("shell_io", routine=interactive_routine)

    # 3. Spawn CPU-Bound Worker (Hogs CPU, gets demoted down to Q3)
    def cpu_worker_routine(k: NexusKernel, pcb):
        # Consumes full time slices without yielding
        pass

    kernel.create_process("matrix_mul", routine=cpu_worker_routine)

    # 4. Spawn Producer-Consumer Pipe Tasks
    r_fd, w_fd = -1, -1

    def producer_routine(k: NexusKernel, pcb):
        nonlocal w_fd
        if w_fd == -1:
            _, r, w = k.sys_pipe()
            w_fd = w
        k.sys_write(w_fd, b"STREAM_DATA_CHUNK\n")
        k.sys_yield()

    kernel.create_process("log_producer", routine=producer_routine)

    print(KernelVisualizer.render_dashboard(kernel))
    print("\nExecuting 60 Kernel Ticks across MLFQ Scheduler...\n")

    for tick in range(1, 61):
        pcb, preempted = kernel.step()
        if tick in (10, 25, 40, 55):
            print(f"\n[Snapshot at Tick {tick}]")
            print(KernelVisualizer.render_dashboard(kernel))
            time.sleep(0.1)

    print("\nSimulation Complete. Final Kernel State:")
    print(KernelVisualizer.render_dashboard(kernel))


if __name__ == "__main__":
    run_simulation()
