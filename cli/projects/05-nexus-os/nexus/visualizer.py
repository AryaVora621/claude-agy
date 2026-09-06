"""
NexusOS: Real-time Terminal Visualizer & Process Manager (ASCII top/htop).
Renders kernel metrics, memory utilization bitmaps, MLFQ queue states,
TLB efficiency gauges, and per-process memory maps.
"""

from typing import List, Dict, Optional
from nexus.types import PAGE_SIZE, NUM_PHYSICAL_FRAMES, ProcessState
from nexus.kernel import NexusKernel


class KernelVisualizer:
    """ASCII Terminal Dashboard for NexusOS."""

    @staticmethod
    def render_ram_gauge(used_frames: int, total_frames: int = NUM_PHYSICAL_FRAMES, width: int = 30) -> str:
        """Render ASCII bar for physical RAM frame utilization."""
        pct = used_frames / max(1, total_frames)
        filled = int(round(pct * width))
        bar = "■" * filled + "□" * (width - filled)
        used_kb = (used_frames * PAGE_SIZE) // 1024
        total_kb = (total_frames * PAGE_SIZE) // 1024
        return f"[{bar}] {pct * 100:5.1f}% ({used_kb}KB / {total_kb}KB, {used_frames}/{total_frames} frames)"

    @classmethod
    def render_dashboard(cls, kernel: NexusKernel) -> str:
        """Generate comprehensive multi-panel kernel dashboard."""
        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║                   NEXUS-OS MICROKERNEL DASHBOARD (TOP)                      ║")
        lines.append("╚" + "═" * 78 + "╝")

        # 1. System Summary
        live_procs = [p for p in kernel.processes.values() if p.state != ProcessState.ZOMBIE]
        zombie_procs = [p for p in kernel.processes.values() if p.state == ProcessState.ZOMBIE]
        used_frames = len(kernel.frame_allocator.allocated_frames)

        tlb = kernel.mmu.tlb
        tlb_total = tlb.hits + tlb.misses
        tlb_pct = (tlb.hits / tlb_total * 100.0) if tlb_total > 0 else 0.0

        curr_proc_name = kernel.current_process.name if kernel.current_process else "IDLE"
        curr_pid = kernel.current_process.pid if kernel.current_process else "-"

        lines.append(f"  Kernel Ticks: {kernel.ticks:<8} | Live Procs: {len(live_procs):<4} | Active CPU: PID {curr_pid} ({curr_proc_name})")
        lines.append(f"  RAM Usage   : {cls.render_ram_gauge(used_frames, NUM_PHYSICAL_FRAMES, width=28)}")
        lines.append(f"  TLB Metrics : Hits={tlb.hits} Misses={tlb.misses} (Hit Rate: {tlb_pct:5.1f}%) | Page Faults: Total={kernel.mmu.page_fault_count}, COW={kernel.mmu.cow_fault_count}")
        lines.append("─" * 80)

        # 2. MLFQ Priority Queues
        lines.append("  MULTI-LEVEL FEEDBACK QUEUE (MLFQ) STATUS:")
        queue_names = ["Q0 [High/Interactive]", "Q1 [Medium]         ", "Q2 [Low]            ", "Q3 [Background/RR]  "]
        for prio, q in enumerate(kernel.scheduler.queues):
            pids = list(q)
            pid_strs = [f"PID {pid}" for pid in pids]
            contents = ", ".join(pid_strs) if pid_strs else "<empty>"
            lines.append(f"    {queue_names[prio]}: {contents}")
        lines.append(f"    Next Priority Boost in: {kernel.scheduler.boost_interval - (kernel.ticks % kernel.scheduler.boost_interval)} ticks (Total Boosts: {kernel.scheduler.boost_count})")
        lines.append("─" * 80)

        # 3. Process Table
        lines.append(f"  {'PID':<5} | {'PPID':<5} | {'Name':<14} | {'State':<12} | {'Prio':<5} | {'CPU':<5} | {'Slice':<6} | {'Memory':<8}")
        lines.append("  " + "-" * 76)

        for p in sorted(kernel.processes.values(), key=lambda x: x.pid):
            state_str = p.state.name
            mapped_pages = len(p.address_space.page_table.all_entries())
            mem_kb = f"{mapped_pages * 4} KB"
            slice_str = f"{p.time_slice_remaining} t"
            lines.append(f"  {p.pid:<5} | {p.ppid:<5} | {p.name:<14} | {state_str:<12} | {p.priority:<5} | {p.total_cpu_time:<5} | {slice_str:<6} | {mem_kb:<8}")

        lines.append("═" * 80)
        return "\n".join(lines)
