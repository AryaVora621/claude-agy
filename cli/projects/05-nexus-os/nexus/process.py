"""
NexusOS: Process Control Block (PCB), Address Space, and Capability Space (CSpace).
Encapsulates virtual memory regions, memory cloning for Copy-On-Write fork(),
register contexts, file descriptor mappings, and capability-based security tokens.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import copy

from nexus.types import (
    PAGE_SIZE, PAGE_SHIFT, PageFlags, PrivilegeRing, ProcessState,
    CPUContext, CapabilityRights, Errno
)
from nexus.mmu import PageTable, PhysicalFrameAllocator, PageTableEntry


@dataclass
class MemoryRegion:
    """Designated segment within virtual memory space."""
    start_vaddr: int
    length: int
    flags: int
    name: str

    @property
    def end_vaddr(self) -> int:
        return self.start_vaddr + self.length

    def contains(self, vaddr: int) -> bool:
        return self.start_vaddr <= vaddr < self.end_vaddr


@dataclass
class Capability:
    """Security capability token granting specific rights to a system resource."""
    token_id: str
    resource_type: str      # 'ENDPOINT', 'FILE', 'PROCESS', 'MEMORY'
    target_id: str          # Identifier of the underlying kernel object
    rights: int             # Bitmask of CapabilityRights

    def permits(self, required_rights: int) -> bool:
        """Check if this capability contains all required access rights."""
        return (self.rights & required_rights) == required_rights


class CSpace:
    """Process Capability Space: indexed table of delegated capability tokens."""
    def __init__(self):
        self._slots: Dict[int, Capability] = {}
        self._next_slot = 1

    def insert(self, capability: Capability, slot: Optional[int] = None) -> int:
        if slot is None:
            slot = self._next_slot
            self._next_slot += 1
        self._slots[slot] = capability
        return slot

    def lookup(self, slot: int) -> Optional[Capability]:
        return self._slots.get(slot)

    def revoke(self, slot: int) -> Optional[Capability]:
        return self._slots.pop(slot, None)

    def check_access(self, slot: int, required_rights: int) -> bool:
        cap = self.lookup(slot)
        if cap is None:
            return False
        return cap.permits(required_rights)

    def clone(self) -> "CSpace":
        """Duplicate capability space on fork (inherits capabilities without GRANT right stripped)."""
        new_cspace = CSpace()
        new_cspace._next_slot = self._next_slot
        for slot, cap in self._slots.items():
            new_cspace._slots[slot] = copy.copy(cap)
        return new_cspace


class AddressSpace:
    """
    Virtual Address Space for a process.
    Organizes standard segments: Text (code), Data, Heap (brk), and Stack.
    """
    def __init__(self, page_table: Optional[PageTable] = None):
        self.page_table = page_table if page_table is not None else PageTable()
        self.regions: List[MemoryRegion] = []
        # Standard layout addresses (32-bit virtual memory)
        self.text_start = 0x00010000
        self.data_start = 0x00020000
        self.heap_start = 0x00030000
        self.brk_point = self.heap_start
        self.stack_top = 0x7FFF0000  # Grows downward
        self.stack_limit = 0x7FFE0000

    def add_region(self, start_vaddr: int, length: int, flags: int, name: str) -> MemoryRegion:
        region = MemoryRegion(start_vaddr=start_vaddr, length=length, flags=flags, name=name)
        self.regions.append(region)
        return region

    def allocate_pages_for_region(
        self,
        vaddr: int,
        num_pages: int,
        flags: int,
        frame_allocator: PhysicalFrameAllocator
    ) -> bool:
        """Allocate physical frames and map them consecutively to virtual pages."""
        allocated_pfns = []
        for i in range(num_pages):
            pfn = frame_allocator.allocate()
            if pfn is None:
                # Rollback on out-of-memory
                for rollback_pfn in allocated_pfns:
                    frame_allocator.release(rollback_pfn)
                return False
            allocated_pfns.append(pfn)
            vpn = (vaddr + (i * PAGE_SIZE)) >> PAGE_SHIFT
            self.page_table.map_page(vpn, pfn, flags)
        return True

    def clone_cow(self, frame_allocator: PhysicalFrameAllocator) -> "AddressSpace":
        """
        Create a Copy-On-Write duplicate of this address space for fork().
        Both parent and child share physical frames marked read-only with COW flag.
        """
        child_space = AddressSpace()
        child_space.text_start = self.text_start
        child_space.data_start = self.data_start
        child_space.heap_start = self.heap_start
        child_space.brk_point = self.brk_point
        child_space.stack_top = self.stack_top
        child_space.stack_limit = self.stack_limit

        for region in self.regions:
            child_space.regions.append(copy.copy(region))

        # Re-tag all mapped pages in parent and map into child with COW
        for vpn, parent_entry in self.page_table.all_entries():
            pfn = parent_entry.pfn
            frame_allocator.retain(pfn)

            # If page was writable, convert to COW (read-only until fault)
            if parent_entry.flags & PageFlags.WRITABLE:
                parent_entry.flags = (parent_entry.flags & ~PageFlags.WRITABLE) | PageFlags.COW

            child_flags = parent_entry.flags
            child_space.page_table.map_page(vpn, pfn, child_flags)

        return child_space

    def destroy(self, frame_allocator: PhysicalFrameAllocator) -> None:
        """Release all allocated physical frames when process terminates."""
        for vpn, entry in self.page_table.all_entries():
            frame_allocator.release(entry.pfn)
        self.regions.clear()


class ProcessControlBlock:
    """
    Process Control Block (PCB): Kernel's internal representation of a process.
    """
    def __init__(
        self,
        pid: int,
        name: str,
        ppid: int = 0,
        ring: PrivilegeRing = PrivilegeRing.RING_3,
        address_space: Optional[AddressSpace] = None
    ):
        self.pid = pid
        self.ppid = ppid
        self.name = name
        self.ring = ring
        self.state = ProcessState.EMBRYO

        # Scheduling properties (MLFQ)
        self.priority = 0                   # 0 is highest priority (Q0), 3 is lowest (Q3)
        self.time_slice_remaining = 2       # Depends on queue level
        self.allotment_remaining = 10       # Ticks before queue demotion
        self.total_cpu_time = 0             # Ticks on CPU
        self.sleep_ticks = 0

        # CPU register state
        self.context = CPUContext(ring=ring)

        # Virtual memory & security
        self.address_space = address_space if address_space is not None else AddressSpace()
        self.cspace = CSpace()

        # File descriptor table (maps fd int -> VFS handle / pipe)
        self.fd_table: Dict[int, Any] = {}

        # IPC & Synchronization
        self.blocked_on: Optional[str] = None
        self.ipc_message_buffer: Optional[Any] = None

        # Hierarchy & Lifecycle
        self.children: List[int] = []
        self.exit_code: Optional[int] = None

    def is_alive(self) -> bool:
        return self.state not in (ProcessState.ZOMBIE, ProcessState.EMBRYO)

    def is_ready(self) -> bool:
        return self.state == ProcessState.READY
