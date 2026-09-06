"""
NexusOS: Virtual Memory Subsystem and Memory Management Unit (MMU).
Implements multi-level page tables, TLB caching, physical frame allocation with
reference counting, Copy-On-Write (COW) mechanics, and page eviction to swap.
"""

from typing import Dict, Optional, Tuple, List, Set
from dataclasses import dataclass
from collections import deque
from nexus.types import (
    PAGE_SIZE, PAGE_SHIFT, PAGE_MASK, NUM_PHYSICAL_FRAMES,
    SWAP_CAPACITY_FRAMES, PageFlags, PrivilegeRing, Errno
)


class PageFaultError(Exception):
    """Raised when an address translation cannot be satisfied by hardware."""
    def __init__(self, vaddr: int, vpn: int, is_cow: bool, not_present: bool, permission_denied: bool):
        self.vaddr = vaddr
        self.vpn = vpn
        self.is_cow = is_cow
        self.not_present = not_present
        self.permission_denied = permission_denied
        msg = f"PageFault at 0x{vaddr:08X} (VPN={vpn})"
        if is_cow:
            msg += " [COW Write Violation]"
        elif not_present:
            msg += " [Page Not Present]"
        elif permission_denied:
            msg += " [Permission Denied]"
        super().__init__(msg)


@dataclass
class PageTableEntry:
    """Entry in a second-level page table."""
    pfn: int = 0                  # Physical Frame Number
    flags: int = 0                # Bitmask of PageFlags
    swap_slot: Optional[int] = None  # Offset in swap partition if evicted


class PageTable:
    """
    Two-Level Hierarchical Page Table (1024 Directory Entries x 1024 Table Entries).
    Maps 32-bit virtual addresses to physical frames.
    """
    def __init__(self):
        # Sparse representation: directory_idx -> (table_idx -> PageTableEntry)
        self.directory: Dict[int, Dict[int, PageTableEntry]] = {}

    @staticmethod
    def split_vpn(vpn: int) -> Tuple[int, int]:
        """Split a 20-bit VPN into 10-bit Directory Index and 10-bit Table Index."""
        pde_idx = (vpn >> 10) & 0x3FF
        pte_idx = vpn & 0x3FF
        return pde_idx, pte_idx

    def map_page(self, vpn: int, pfn: int, flags: int) -> PageTableEntry:
        """Create or update a mapping from VPN to PFN with specified flags."""
        pde_idx, pte_idx = self.split_vpn(vpn)
        if pde_idx not in self.directory:
            self.directory[pde_idx] = {}

        entry = PageTableEntry(pfn=pfn, flags=flags | PageFlags.PRESENT)
        self.directory[pde_idx][pte_idx] = entry
        return entry

    def unmap_page(self, vpn: int) -> Optional[PageTableEntry]:
        """Remove a virtual page mapping."""
        pde_idx, pte_idx = self.split_vpn(vpn)
        if pde_idx in self.directory and pte_idx in self.directory[pde_idx]:
            entry = self.directory[pde_idx].pop(pte_idx)
            if not self.directory[pde_idx]:
                del self.directory[pde_idx]
            return entry
        return None

    def lookup(self, vpn: int) -> Optional[PageTableEntry]:
        """Retrieve the PTE for a given VPN if present."""
        pde_idx, pte_idx = self.split_vpn(vpn)
        pde = self.directory.get(pde_idx)
        if pde is not None:
            return pde.get(pte_idx)
        return None

    def all_entries(self) -> List[Tuple[int, PageTableEntry]]:
        """Return list of all (vpn, entry) tuples across the page table."""
        entries = []
        for pde_idx, table in self.directory.items():
            for pte_idx, entry in table.items():
                vpn = (pde_idx << 10) | pte_idx
                entries.append((vpn, entry))
        return entries


class PhysicalFrameAllocator:
    """
    Manages physical memory (RAM) frames with reference counting for Copy-On-Write.
    Backs physical memory with an actual contiguous bytearray.
    """
    def __init__(self, num_frames: int = NUM_PHYSICAL_FRAMES):
        self.num_frames = num_frames
        self.free_frames: deque[int] = deque(range(num_frames))
        self.ref_counts: Dict[int, int] = {}
        # Backing physical RAM storage
        self.ram = bytearray(num_frames * PAGE_SIZE)
        # Clock algorithm tracking: list of in-use frames
        self.allocated_frames: List[int] = []

    def allocate(self) -> Optional[int]:
        """Allocate a free physical frame. Returns PFN or None if out of memory."""
        if not self.free_frames:
            return None
        pfn = self.free_frames.popleft()
        self.ref_counts[pfn] = 1
        self.allocated_frames.append(pfn)
        # Zero out newly allocated frame
        start = pfn * PAGE_SIZE
        self.ram[start:start + PAGE_SIZE] = b'\x00' * PAGE_SIZE
        return pfn

    def retain(self, pfn: int) -> None:
        """Increment reference count (used in COW fork)."""
        if pfn in self.ref_counts:
            self.ref_counts[pfn] += 1

    def release(self, pfn: int) -> bool:
        """
        Decrement reference count. Frees frame if ref count hits 0.
        Returns True if the frame was actually freed.
        """
        if pfn not in self.ref_counts:
            return False
        self.ref_counts[pfn] -= 1
        if self.ref_counts[pfn] <= 0:
            del self.ref_counts[pfn]
            if pfn in self.allocated_frames:
                self.allocated_frames.remove(pfn)
            self.free_frames.append(pfn)
            return True
        return False

    def get_ref_count(self, pfn: int) -> int:
        """Return the current reference count of a frame."""
        return self.ref_counts.get(pfn, 0)

    def read_bytes(self, pfn: int, offset: int, length: int) -> bytes:
        """Read bytes directly from physical frame storage."""
        start = (pfn * PAGE_SIZE) + offset
        return bytes(self.ram[start:start + length])

    def write_bytes(self, pfn: int, offset: int, data: bytes) -> None:
        """Write bytes directly into physical frame storage."""
        start = (pfn * PAGE_SIZE) + offset
        self.ram[start:start + len(data)] = data

    def copy_frame(self, src_pfn: int, dest_pfn: int) -> None:
        """Duplicate an entire 4KB frame from source to destination."""
        src_start = src_pfn * PAGE_SIZE
        dest_start = dest_pfn * PAGE_SIZE
        self.ram[dest_start:dest_start + PAGE_SIZE] = self.ram[src_start:src_start + PAGE_SIZE]


class SwapDevice:
    """
    Simulated backing swap storage for page eviction.
    """
    def __init__(self, capacity_frames: int = SWAP_CAPACITY_FRAMES):
        self.capacity = capacity_frames
        self.free_slots: deque[int] = deque(range(capacity_frames))
        self.storage: Dict[int, bytes] = {}

    def swap_out(self, data: bytes) -> Optional[int]:
        """Save a 4KB page to swap disk. Returns slot index or None if full."""
        if not self.free_slots:
            return None
        slot = self.free_slots.popleft()
        self.storage[slot] = data
        return slot

    def swap_in(self, slot: int) -> Optional[bytes]:
        """Read a 4KB page from swap disk and free the slot."""
        if slot in self.storage:
            data = self.storage.pop(slot)
            self.free_slots.append(slot)
            return data
        return None

    def free_slot(self, slot: int) -> None:
        """Free a swap slot without reading."""
        if slot in self.storage:
            del self.storage[slot]
            self.free_slots.append(slot)


class TLB:
    """
    Hardware Translation Lookaside Buffer with LRU eviction and statistics.
    """
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        # Mapping: vpn -> (pfn, flags)
        self.cache: Dict[int, Tuple[int, int]] = {}
        self.order: deque[int] = deque()
        self.hits = 0
        self.misses = 0

    def lookup(self, vpn: int) -> Optional[Tuple[int, int]]:
        """Query TLB for cached translation."""
        if vpn in self.cache:
            self.hits += 1
            self.order.remove(vpn)
            self.order.append(vpn)
            return self.cache[vpn]
        self.misses += 1
        return None

    def insert(self, vpn: int, pfn: int, flags: int) -> None:
        """Cache a newly resolved page translation."""
        if vpn in self.cache:
            self.cache[vpn] = (pfn, flags)
            self.order.remove(vpn)
            self.order.append(vpn)
            return

        if len(self.cache) >= self.capacity:
            oldest_vpn = self.order.popleft()
            del self.cache[oldest_vpn]

        self.cache[vpn] = (pfn, flags)
        self.order.append(vpn)

    def invalidate(self, vpn: Optional[int] = None) -> None:
        """Flush specific VPN or entire TLB on context switch."""
        if vpn is None:
            self.cache.clear()
            self.order.clear()
        elif vpn in self.cache:
            del self.cache[vpn]
            self.order.remove(vpn)


class MMU:
    """
    Memory Management Unit: Hardware translation, permission checking,
    COW handling, and page fault dispatching.
    """
    def __init__(self, frame_allocator: PhysicalFrameAllocator, swap_device: SwapDevice):
        self.frames = frame_allocator
        self.swap = swap_device
        self.tlb = TLB(capacity=32)
        self.current_page_table: Optional[PageTable] = None
        self.page_fault_count = 0
        self.cow_fault_count = 0

    def switch_address_space(self, page_table: PageTable) -> None:
        """Context switch: update active page table and flush TLB."""
        self.current_page_table = page_table
        self.tlb.invalidate()

    def translate(self, vaddr: int, mode: str = "r", ring: PrivilegeRing = PrivilegeRing.RING_3) -> int:
        """
        Translate a virtual address to a physical address.
        mode: 'r' (read), 'w' (write), 'x' (execute)
        Raises PageFaultError if unmapped, COW write, or permission violation.
        """
        vpn = vaddr >> PAGE_SHIFT
        offset = vaddr & PAGE_MASK

        # Check TLB first
        cached = self.tlb.lookup(vpn)
        if cached is not None:
            pfn, flags = cached
            entry = PageTableEntry(pfn=pfn, flags=flags)
        else:
            if self.current_page_table is None:
                raise PageFaultError(vaddr, vpn, False, True, False)
            entry = self.current_page_table.lookup(vpn)
            if entry is None or not (entry.flags & PageFlags.PRESENT):
                self.page_fault_count += 1
                raise PageFaultError(vaddr, vpn, False, True, False)
            self.tlb.insert(vpn, entry.pfn, entry.flags)

        # Privilege Ring Check (User ring 3 cannot access kernel-only pages)
        if ring == PrivilegeRing.RING_3 and not (entry.flags & PageFlags.USER_ACCESSIBLE):
            self.page_fault_count += 1
            raise PageFaultError(vaddr, vpn, False, False, True)

        # Mode permissions
        if mode == "w":
            # Check Copy-On-Write first
            if entry.flags & PageFlags.COW:
                self.page_fault_count += 1
                self.cow_fault_count += 1
                raise PageFaultError(vaddr, vpn, True, False, False)
            if not (entry.flags & PageFlags.WRITABLE):
                self.page_fault_count += 1
                raise PageFaultError(vaddr, vpn, False, False, True)
            entry.flags |= PageFlags.DIRTY

        entry.flags |= PageFlags.ACCESSED
        return (entry.pfn << PAGE_SHIFT) | offset

    def handle_cow_fault(self, vpn: int) -> bool:
        """
        Resolve a Copy-On-Write fault:
        If frame reference count > 1, allocate new frame, copy bytes, decrement old frame ref.
        If reference count == 1, simply upgrade page to writable and clear COW.
        """
        if self.current_page_table is None:
            return False
        entry = self.current_page_table.lookup(vpn)
        if entry is None or not (entry.flags & PageFlags.COW):
            return False

        old_pfn = entry.pfn
        ref_count = self.frames.get_ref_count(old_pfn)

        if ref_count > 1:
            new_pfn = self.frames.allocate()
            if new_pfn is None:
                return False  # OOM
            self.frames.copy_frame(old_pfn, new_pfn)
            self.frames.release(old_pfn)
            entry.pfn = new_pfn

        # Mark page writable and clear COW flag
        entry.flags = (entry.flags & ~PageFlags.COW) | PageFlags.WRITABLE | PageFlags.DIRTY
        self.tlb.invalidate(vpn)
        return True

    def read_memory(self, vaddr: int, length: int, ring: PrivilegeRing = PrivilegeRing.RING_3) -> bytes:
        """Read arbitrary virtual memory span across page boundaries."""
        result = bytearray()
        remaining = length
        curr_vaddr = vaddr

        while remaining > 0:
            paddr = self.translate(curr_vaddr, mode="r", ring=ring)
            pfn = paddr >> PAGE_SHIFT
            offset = paddr & PAGE_MASK
            chunk = min(remaining, PAGE_SIZE - offset)
            result.extend(self.frames.read_bytes(pfn, offset, chunk))
            curr_vaddr += chunk
            remaining -= chunk

        return bytes(result)

    def write_memory(self, vaddr: int, data: bytes, ring: PrivilegeRing = PrivilegeRing.RING_3) -> None:
        """Write arbitrary virtual memory span across page boundaries handling COW."""
        remaining = len(data)
        curr_vaddr = vaddr
        data_offset = 0

        while remaining > 0:
            vpn = curr_vaddr >> PAGE_SHIFT
            try:
                paddr = self.translate(curr_vaddr, mode="w", ring=ring)
            except PageFaultError as e:
                if e.is_cow:
                    if not self.handle_cow_fault(vpn):
                        raise
                    paddr = self.translate(curr_vaddr, mode="w", ring=ring)
                else:
                    raise

            pfn = paddr >> PAGE_SHIFT
            offset = paddr & PAGE_MASK
            chunk = min(remaining, PAGE_SIZE - offset)
            self.frames.write_bytes(pfn, offset, data[data_offset:data_offset + chunk])
            curr_vaddr += chunk
            data_offset += chunk
            remaining -= chunk
