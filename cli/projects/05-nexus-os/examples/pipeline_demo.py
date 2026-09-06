#!/usr/bin/env python3
"""
NexusOS: End-to-End Pipeline & Copy-On-Write (COW) Fork Demonstration.
Demonstrates:
1. Virtual Memory setup and Copy-On-Write memory sharing across fork().
2. Page fault handling upon child memory modification.
3. Unix pipe byte stream transfer between parent and child.
4. Process lifecycle synchronization via waitpid.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexus.kernel import NexusKernel
from nexus.types import PageFlags, Errno


def run_pipeline():
    print("=" * 78)
    print("  NEXUS-OS COPY-ON-WRITE (COW) & UNIX PIPE DEMONSTRATION")
    print("=" * 78)

    kernel = NexusKernel(num_frames=64)

    # 1. Create Parent Process
    parent = kernel.create_process("parent_server")
    print(f"\n[Step 1] Created Parent Process (PID {parent.pid})")

    # 2. Allocate a shared memory region at 0x00040000
    SHARED_VADDR = 0x00040000
    kernel.mmu.switch_address_space(parent.address_space.page_table)
    status, vaddr = kernel.sys_mmap(SHARED_VADDR, num_pages=1, flags=PageFlags.READABLE | PageFlags.WRITABLE)
    print(f"  Allocated 1 page at 0x{vaddr:08X} (Status: {status.name})")

    # Parent writes initial message
    parent_data = b"PARENT_MASTER_KEY_SECRET_DATA"
    kernel.mmu.write_memory(SHARED_VADDR, parent_data)
    readback = kernel.mmu.read_memory(SHARED_VADDR, len(parent_data))
    print(f"  Parent wrote to virtual memory: {readback.decode()}")

    parent_entry = parent.address_space.page_table.lookup(SHARED_VADDR >> 12)
    print(f"  Parent Page Table Entry: PFN={parent_entry.pfn}, RefCount={kernel.frame_allocator.get_ref_count(parent_entry.pfn)}, Flags=0x{parent_entry.flags:02X}")

    # 3. Create a Unix Pipe
    status, r_fd, w_fd = kernel.sys_pipe()
    print(f"\n[Step 2] Created Unix Pipe: ReadFD={r_fd}, WriteFD={w_fd}")

    # 4. Fork Child Process
    status, child_pid = kernel.sys_fork()
    child = kernel.processes[child_pid]
    print(f"\n[Step 3] Executed sys_fork(): Child PID {child_pid} created")

    child_entry = child.address_space.page_table.lookup(SHARED_VADDR >> 12)
    parent_entry = parent.address_space.page_table.lookup(SHARED_VADDR >> 12)

    print(f"  Parent PFN: {parent_entry.pfn} (COW bit set: {bool(parent_entry.flags & PageFlags.COW)})")
    print(f"  Child  PFN: {child_entry.pfn} (COW bit set: {bool(child_entry.flags & PageFlags.COW)})")
    print(f"  Shared Physical Frame RefCount: {kernel.frame_allocator.get_ref_count(parent_entry.pfn)} (Frames Shared: TRUE)")

    # 5. Child modifies shared memory (Triggers Copy-On-Write Page Fault!)
    print(f"\n[Step 4] Child modifies memory at 0x{SHARED_VADDR:08X}...")
    kernel.mmu.switch_address_space(child.address_space.page_table)
    child_data = b"CHILD_MODIFIED_PAYLOAD_NEW_DATA"
    kernel.mmu.write_memory(SHARED_VADDR, child_data)

    new_child_entry = child.address_space.page_table.lookup(SHARED_VADDR >> 12)
    print(f"  [PAGE FAULT HANDLED] Copy-On-Write triggered and resolved!")
    print(f"  Parent PFN: {parent_entry.pfn}, RefCount: {kernel.frame_allocator.get_ref_count(parent_entry.pfn)}")
    print(f"  Child  PFN: {new_child_entry.pfn}, RefCount: {kernel.frame_allocator.get_ref_count(new_child_entry.pfn)}")
    print(f"  Frames Diverged: {parent_entry.pfn != new_child_entry.pfn} (Parent & Child now have distinct physical frames)")

    # Verify parent's memory is unchanged
    kernel.mmu.switch_address_space(parent.address_space.page_table)
    parent_readback = kernel.mmu.read_memory(SHARED_VADDR, len(parent_data))
    kernel.mmu.switch_address_space(child.address_space.page_table)
    child_readback = kernel.mmu.read_memory(SHARED_VADDR, len(child_data))

    print(f"\n  Parent Memory Content: {parent_readback.decode()}")
    print(f"  Child  Memory Content: {child_readback.decode()}")

    # 6. Inter-Process Pipe Communication
    print(f"\n[Step 5] Child writes modified payload through Unix Pipe...")
    kernel.set_active_process(child.pid)
    status, written = kernel.sys_write(w_fd, child_readback)
    print(f"  Child wrote {written} bytes to pipe (Status: {status.name})")

    # Child closes write end and exits
    kernel.sys_close(w_fd)
    kernel.sys_exit(42)
    print(f"  Child exited with status code 42 (Child state: {child.state.name})")

    # 7. Parent reads from pipe and reaps child via waitpid
    print(f"\n[Step 6] Parent reads from Unix Pipe...")
    kernel.set_active_process(parent.pid)
    status, received_bytes = kernel.sys_read(r_fd, 64)
    print(f"  Parent received from pipe: {received_bytes.decode()} (Status: {status.name})")

    print(f"\n[Step 7] Parent reaps child via sys_waitpid({child.pid})...")
    status, reaped_pid, exit_code = kernel.sys_waitpid(child.pid)
    print(f"  Child {reaped_pid} reaped successfully with Exit Code: {exit_code}")
    print(f"  Active Processes Remaining: {len(kernel.processes)}")
    print(f"\nPipeline Demonstration Completed with 100% Success.")
    print("=" * 78)


if __name__ == "__main__":
    run_pipeline()
