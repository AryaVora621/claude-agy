# Project 05: NexusOS - Capability-Based Microkernel & Virtual Memory Operating System

A high-performance, mathematically sound microkernel and virtual memory operating system simulator built completely from first principles in pure Python (Standard Library only - zero external dependencies).

NexusOS models hardware-software architectural primitives: multi-level hierarchical paging, hardware translation lookaside buffer (TLB), copy-on-write (COW) memory sharing, a Multi-Level Feedback Queue (MLFQ) scheduler with anti-gaming allotment tracking and starvation prevention, L4-style synchronous rendezvous IPC, capability-based security tokens, and a Virtual File System (VFS) with Unix-style unidirectional circular buffer pipes.

---

## 🏛 Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              USER SPACE                                │
│   ┌───────────────┐     ┌───────────────┐     ┌──────────────────┐    │
│   │ Process 1     │     │ Process 2     │     │ Process 3        │    │
│   │ [CSpace / FD] │     │ [CSpace / FD] │     │ [CSpace / FD]    │    │
│   └───────┬───────┘     └───────┬───────┘     └────────┬─────────┘    │
└───────────┼─────────────────────┼──────────────────────┼──────────────┘
            │ Trap (0x80)         │ Sync IPC Call        │ Unix Pipe Write
════════════╪═════════════════════╪══════════════════════╪═══════════════
┌───────────▼─────────────────────▼──────────────────────▼──────────────┐
│                    NEXUS-OS MICROKERNEL (RING 0)                       │
│                                                                        │
│  ┌──────────────────────┐  ┌────────────────────┐  ┌────────────────┐  │
│  │   MLFQ SCHEDULER     │  │  CAPABILITY & IPC  │  │   UNIX PIPES   │  │
│  │  Q0: Slice 2t / 4t   │  │  L4 Rendezvous     │  │  4KB Circular  │  │
│  │  Q1: Slice 4t / 8t   │  │  Async Mailboxes   │  │  EOF / EPIPE   │  │
│  │  Q2: Slice 8t / 16t  │  │  Token CSpace      │  │  FD Redirection│  │
│  │  Q3: Background RR   │  │                    │  │                │  │
│  └──────────┬───────────┘  └────────────────────┘  └────────────────┘  │
│             │ Preemption / Boost                                       │
│  ┌──────────▼───────────────────────────────────────────────────────┐  │
│  │              VIRTUAL MEMORY & MMU SUBSYSTEM                      │  │
│  │  • 32-bit Virtual Address Space (10-bit PDE / 10-bit PTE)        │  │
│  │  • 32-Entry LRU Hardware TLB (Hits / Misses / Invalidation)      │  │
│  │  • Copy-On-Write (COW) Engine on sys_fork()                      │  │
│  │  • Physical Frame Allocator with Reference Counting              │  │
│  │  • Backing Swap Partition with Clock / Eviction Policy           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Architectural Features

### 1. Two-Level Hierarchical Paging & TLB (`nexus/mmu.py`)
- **Address Space Layout**: 32-bit virtual addresses decomposed into 10-bit Directory Index, 10-bit Page Table Index, and 12-bit Page Offset (4096-byte pages).
- **Page Table Entries (PTE)**: Emulates architecture-grade bit flags: `PRESENT`, `READABLE`, `WRITABLE`, `USER_ACCESSIBLE`, `ACCESSED`, `DIRTY`, and `COW`.
- **Hardware TLB Simulation**: 32-entry fully associative LRU Translation Lookaside Buffer with selective and full flush on context switch.
- **Copy-On-Write (COW)**: On `sys_fork()`, parent and child share physical frames tagged with `COW` and `WRITABLE` cleared. Writes trigger a hardware `PageFaultError` which transparently allocates a new physical frame, duplicates contents, and updates the faulting page table without user-space intervention.

### 2. Multi-Level Feedback Queue (MLFQ) Scheduler (`nexus/scheduler.py`)
- **4 Priority Tiers**:
  - `Q0`: Time slice 2 ticks, allotment 4 ticks (Interactive / I/O bounded)
  - `Q1`: Time slice 4 ticks, allotment 8 ticks
  - `Q2`: Time slice 8 ticks, allotment 16 ticks
  - `Q3`: Time slice 16 ticks, round-robin (Batch / CPU bounded)
- **Anti-Gaming Allotment Tracking**: Processes that yield voluntarily retain their priority tier but have their cumulative CPU allotment deducted, preventing starvation gaming.
- **Starvation-Free Periodic Priority Boost**: Automatically boosts all live processes to `Q0` every 50 ticks to ensure low-priority batch jobs make progress and dynamically adjust if they transition to interactive behavior.

### 3. Capability-Based IPC Subsystem (`nexus/ipc.py`)
- **L4-Style Synchronous Rendezvous**: Direct thread-to-thread handoff (`send_and_wait` / `recv_and_wait`). Senders block until receivers arrive and vice versa, executing zero-copy memory transfers without intermediate kernel queuing.
- **Bounded Asynchronous Mailboxes**: Fixed-capacity ring buffers for asynchronous pub/sub message patterns.
- **CSpace Security Enforcement**: Every IPC endpoint operation requires validation of an unforgeable capability token with explicit permissions (`READ`, `WRITE`, `INVOKE`, `GRANT`).

### 4. Virtual File System & Unix Pipes (`nexus/vfs.py`)
- **Hierarchical Inodes**: Inode-based tree supporting regular files and directories (`/`, `/bin`, `/dev`, `/etc`, `/tmp`).
- **File Descriptor Tables**: Per-process mapping of integer handles (`0=stdin`, `1=stdout`, `2=stderr`, `3+`) to open file structures.
- **Unix Pipes**: Unidirectional circular byte streams with strict POSIX semantics:
  - Reading from an empty pipe blocks if writers exist, or returns `EOF` (0 bytes) if all write descriptors are closed.
  - Writing to a pipe with no active read descriptors raises `EPIPE` (broken pipe).

---

## 📊 Performance Benchmarks

Executed on standard hardware using `benchmarks/bench_kernel.py`:

| Benchmark | Operations / Volume | Throughput | Latency |
|---|---|---|---|
| **Context Switch Rate** | 50,000 switches | **5,613,904 switches/sec** | **0.178 µs / switch** |
| **Synchronous IPC Ping-Pong** | 25,000 round-trips | **453,087 round-trips/sec** | **2.207 µs / round-trip** |
| **COW Fork-Fault-Reap** | 5,000 full cycles | **75,556 cycles/sec** | **13.235 µs / cycle** |
| **Unix Pipe Streaming** | 20.0 MB transferred | **4,358.4 MB/s** | 0.005s total |

---

## 🧪 Unit & Integration Testing

The test suite covers memory translation, COW fault resolution, MLFQ scheduling, IPC synchronization, VFS inodes, and Unix pipes:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

Output:
```text
........................
----------------------------------------------------------------------
Ran 24 tests in 0.001s

OK
```

---

## 💻 Demos & Visualizers

```bash
# Real-time ASCII process manager dashboard (htop style)
python3 examples/kernel_top.py

# Step-by-step Copy-On-Write memory fork and Unix pipe pipeline
python3 examples/pipeline_demo.py
```
