"""Physical Memory, SV39 Virtual Memory MMU, and Translation Lookaside Buffers (TLB).

Provides:
1. Sparse 64-bit byte-addressable Physical Memory with little-endian byte/word accessors
2. Complete RISC-V SV39 3-level page table walk (4KB pages, 2MB megapages, 1GB gigapages)
3. Canonical virtual address verification and page fault generation
4. High-speed Translation Lookaside Buffers (ITLB and DTLB) with LRU eviction
5. SATP register decoding and address space management
6. Programmatic page mapping helpers for OS and runtime setup
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import Dict, List, Optional, Tuple

PAGE_SIZE = 4096
PAGE_OFFSET_MASK = 0xFFF
PAGE_SHIFT = 12

# SV39 PTE bitfield masks
PTE_V = 1 << 0  # Valid
PTE_R = 1 << 1  # Readable
PTE_W = 1 << 2  # Writable
PTE_X = 1 << 3  # Executable
PTE_U = 1 << 4  # User accessible
PTE_G = 1 << 5  # Global mapping
PTE_A = 1 << 6  # Accessed
PTE_D = 1 << 7  # Dirty

# SATP Modes
SATP_MODE_BARE = 0
SATP_MODE_SV39 = 8


class AccessType(Enum):
    FETCH = auto()
    LOAD = auto()
    STORE = auto()


class PrivilegeMode(Enum):
    USER = 0
    SUPERVISOR = 1
    MACHINE = 3


class PageFaultException(Exception):
    """Raised when memory translation or permission verification fails."""
    def __init__(self, vaddr: int, access_type: AccessType, message: str) -> None:
        super().__init__(message)
        self.vaddr = vaddr
        self.access_type = access_type


class InstructionPageFault(PageFaultException):
    pass


class LoadPageFault(PageFaultException):
    pass


class StorePageFault(PageFaultException):
    pass


# =====================================================================
# Physical Memory with Sparse 4KB Page Allocation
# =====================================================================

class PhysicalMemory:
    """Sparse 64-bit Physical Memory storing pages on-demand in 4KB bytearrays."""

    def __init__(self) -> None:
        self.pages: Dict[int, bytearray] = {}  # ppn -> 4096-byte array

    def _get_or_create_page(self, ppn: int) -> bytearray:
        page = self.pages.get(ppn)
        if page is None:
            page = bytearray(PAGE_SIZE)
            self.pages[ppn] = page
        return page

    def allocate_page(self, ppn: int) -> int:
        """Ensure page at physical page number ppn is allocated."""
        self._get_or_create_page(ppn)
        return ppn * PAGE_SIZE

    @staticmethod
    def _normalize_paddr(paddr: int) -> int:
        """Normalize address to handle sign-extended 32-bit addresses in RV64 bare metal."""
        paddr = paddr & 0xFFFF_FFFF_FFFF_FFFF
        if (paddr >> 32) == 0xFFFFFFFF:
            paddr = paddr & 0xFFFF_FFFF
        return paddr

    def read_u8(self, paddr: int) -> int:
        paddr = self._normalize_paddr(paddr)
        ppn = (paddr >> PAGE_SHIFT) & 0xFFF_FFFF_FFFF
        offset = paddr & PAGE_OFFSET_MASK
        page = self.pages.get(ppn)
        if page is None:
            return 0
        return page[offset]

    def read_i8(self, paddr: int) -> int:
        val = self.read_u8(paddr)
        return val - 256 if val >= 128 else val

    def write_u8(self, paddr: int, val: int) -> None:
        paddr = self._normalize_paddr(paddr)
        ppn = (paddr >> PAGE_SHIFT) & 0xFFF_FFFF_FFFF
        offset = paddr & PAGE_OFFSET_MASK
        page = self._get_or_create_page(ppn)
        page[offset] = val & 0xFF

    def read_u16(self, paddr: int) -> int:
        b0 = self.read_u8(paddr)
        b1 = self.read_u8(paddr + 1)
        return b0 | (b1 << 8)

    def read_i16(self, paddr: int) -> int:
        val = self.read_u16(paddr)
        return val - 65536 if val >= 32768 else val

    def write_u16(self, paddr: int, val: int) -> None:
        self.write_u8(paddr, val & 0xFF)
        self.write_u8(paddr + 1, (val >> 8) & 0xFF)

    def read_u32(self, paddr: int) -> int:
        b0 = self.read_u8(paddr)
        b1 = self.read_u8(paddr + 1)
        b2 = self.read_u8(paddr + 2)
        b3 = self.read_u8(paddr + 3)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

    def read_i32(self, paddr: int) -> int:
        val = self.read_u32(paddr)
        return val - (1 << 32) if val >= (1 << 31) else val

    def write_u32(self, paddr: int, val: int) -> None:
        self.write_u8(paddr, val & 0xFF)
        self.write_u8(paddr + 1, (val >> 8) & 0xFF)
        self.write_u8(paddr + 2, (val >> 16) & 0xFF)
        self.write_u8(paddr + 3, (val >> 24) & 0xFF)

    def read_u64(self, paddr: int) -> int:
        low = self.read_u32(paddr)
        high = self.read_u32(paddr + 4)
        return low | (high << 32)

    def read_i64(self, paddr: int) -> int:
        val = self.read_u64(paddr)
        return val - (1 << 64) if val >= (1 << 63) else val

    def write_u64(self, paddr: int, val: int) -> None:
        self.write_u32(paddr, val & 0xFFFF_FFFF)
        self.write_u32(paddr + 4, (val >> 32) & 0xFFFF_FFFF)

    def load_bytes(self, paddr: int, data: bytes) -> None:
        """Write raw contiguous bytes into physical memory."""
        for i, byte in enumerate(data):
            self.write_u8(paddr + i, byte)

    def write_bytes(self, paddr: int, data: bytes) -> None:
        """Alias for load_bytes to write contiguous bytes into physical memory."""
        self.load_bytes(paddr, data)

    def read_bytes(self, paddr: int, length: int) -> bytes:
        """Read raw contiguous bytes from physical memory."""
        out = bytearray(length)
        for i in range(length):
            out[i] = self.read_u8(paddr + i)
        return bytes(out)


# =====================================================================
# Translation Lookaside Buffer (TLB)
# =====================================================================

@dataclass
class TLBEntry:
    vpn: int
    ppn: int
    flags: int
    page_size_bytes: int  # 4096 (4KB), 2097152 (2MB), or 1073741824 (1GB)
    asid: int
    last_used: int = 0


class TLB:
    """Fully-associative Translation Lookaside Buffer with LRU replacement."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.entries: List[TLBEntry] = []
        self.access_counter = 0

    def lookup(self, vpn: int, asid: int) -> Optional[TLBEntry]:
        self.access_counter += 1
        for entry in self.entries:
            # Check global flag or matching ASID
            is_global = bool(entry.flags & PTE_G)
            if (is_global or entry.asid == asid):
                # Verify VPN match according to page size
                if entry.page_size_bytes == 4096:
                    if entry.vpn == vpn:
                        entry.last_used = self.access_counter
                        return entry
                elif entry.page_size_bytes == 2 * 1024 * 1024:
                    # 2MB Megapage: matches top 18 bits of VPN (VPN[2], VPN[1])
                    if (entry.vpn >> 9) == (vpn >> 9):
                        entry.last_used = self.access_counter
                        return entry
                elif entry.page_size_bytes == 1024 * 1024 * 1024:
                    # 1GB Gigapage: matches top 9 bits of VPN (VPN[2])
                    if (entry.vpn >> 18) == (vpn >> 18):
                        entry.last_used = self.access_counter
                        return entry
        return None

    def insert(self, vpn: int, ppn: int, flags: int, page_size_bytes: int, asid: int) -> None:
        self.access_counter += 1
        # Check if already present to update
        for entry in self.entries:
            if entry.vpn == vpn and entry.asid == asid:
                entry.ppn = ppn
                entry.flags = flags
                entry.page_size_bytes = page_size_bytes
                entry.last_used = self.access_counter
                return

        # Evict LRU if full
        if len(self.entries) >= self.capacity:
            lru_idx = min(range(len(self.entries)), key=lambda i: self.entries[i].last_used)
            self.entries.pop(lru_idx)

        self.entries.append(TLBEntry(
            vpn=vpn,
            ppn=ppn,
            flags=flags,
            page_size_bytes=page_size_bytes,
            asid=asid,
            last_used=self.access_counter,
        ))

    def flush(self) -> None:
        """Invalidate all TLB entries."""
        self.entries.clear()

    def flush_vaddr(self, vaddr: int, asid: Optional[int] = None) -> None:
        """Invalidate TLB entries matching virtual address and optional ASID."""
        vpn = vaddr >> PAGE_SHIFT
        self.entries = [
            e for e in self.entries
            if not (e.vpn == vpn and (asid is None or e.asid == asid))
        ]


# =====================================================================
# Memory Management Unit (SV39 MMU)
# =====================================================================

class MMU:
    """RISC-V SV39 Memory Management Unit with 3-level page table walking."""

    def __init__(self, ram: PhysicalMemory) -> None:
        self.ram = ram
        self.itlb = TLB(capacity=16)
        self.dtlb = TLB(capacity=32)

        # SATP register: [63:60] MODE, [59:44] ASID, [43:0] PPN
        self.satp: int = 0
        self.privilege_mode: PrivilegeMode = PrivilegeMode.MACHINE

        # Statistics
        self.itlb_hits: int = 0
        self.itlb_misses: int = 0
        self.dtlb_hits: int = 0
        self.dtlb_misses: int = 0
        self.page_table_walks: int = 0

    @property
    def mode(self) -> int:
        return (self.satp >> 60) & 0xF

    @property
    def asid(self) -> int:
        return (self.satp >> 44) & 0xFFFF

    @property
    def root_ppn(self) -> int:
        return self.satp & 0xFFF_FFFF_FFFF

    def is_paging_enabled(self) -> bool:
        # Machine mode bypasses translation unless MPRV is active
        if self.privilege_mode == PrivilegeMode.MACHINE:
            return False
        return self.mode == SATP_MODE_SV39

    def translate(self, vaddr: int, access_type: AccessType) -> int:
        """Translate virtual address to physical address in SV39 mode."""
        # Bare mode or translation disabled
        if not self.is_paging_enabled():
            return vaddr & 0xFFFF_FFFF_FFFF_FFFF

        # SV39 Canonical Address Check: bits 63:39 must equal bit 38
        bit38 = (vaddr >> 38) & 1
        top_bits = (vaddr >> 39) & 0x1FFFFFF
        expected_top = 0x1FFFFFF if bit38 else 0
        if top_bits != expected_top:
            self._raise_page_fault(vaddr, access_type, "Non-canonical SV39 virtual address")

        vpn = (vaddr >> PAGE_SHIFT) & 0x7FFFFFF  # 27-bit VPN
        offset = vaddr & PAGE_OFFSET_MASK

        # Check TLB
        tlb = self.itlb if access_type == AccessType.FETCH else self.dtlb
        entry = tlb.lookup(vpn, self.asid)

        if entry is not None:
            if access_type == AccessType.FETCH:
                self.itlb_hits += 1
            else:
                self.dtlb_hits += 1
            self._check_permissions(vaddr, entry.flags, access_type)
            # Compute physical address based on page size
            if entry.page_size_bytes == 4096:
                return (entry.ppn << PAGE_SHIFT) | offset
            elif entry.page_size_bytes == 2 * 1024 * 1024:
                return (entry.ppn << PAGE_SHIFT) | (vaddr & 0x1FFFFF)
            elif entry.page_size_bytes == 1024 * 1024 * 1024:
                return (entry.ppn << PAGE_SHIFT) | (vaddr & 0x3FFFFFFF)

        # TLB Miss -> Perform SV39 3-level hardware page table walk
        if access_type == AccessType.FETCH:
            self.itlb_misses += 1
        else:
            self.dtlb_misses += 1

        self.page_table_walks += 1
        paddr, flags, page_size = self._walk_page_table(vaddr, access_type)

        # Insert into TLB
        tlb.insert(vpn, paddr >> PAGE_SHIFT, flags, page_size, self.asid)
        return paddr

    def _walk_page_table(self, vaddr: int, access_type: AccessType) -> Tuple[int, int, int]:
        """Perform 3-level SV39 page walk."""
        # Split 39-bit virtual address into 9-bit indices
        vpn = [
            (vaddr >> 12) & 0x1FF,  # Level 0 (VPN[0])
            (vaddr >> 21) & 0x1FF,  # Level 1 (VPN[1])
            (vaddr >> 30) & 0x1FF,  # Level 2 (VPN[2])
        ]
        offset = vaddr & PAGE_OFFSET_MASK

        a = self.root_ppn << PAGE_SHIFT

        for level in (2, 1, 0):
            pte_addr = a + vpn[level] * 8
            pte = self.ram.read_u64(pte_addr)

            # Check valid bit
            if not (pte & PTE_V):
                self._raise_page_fault(vaddr, access_type, f"Invalid PTE at level {level}")

            # If R=0 and W=1, reserved for future use
            if not (pte & PTE_R) and (pte & PTE_W):
                self._raise_page_fault(vaddr, access_type, f"Reserved PTE state R=0, W=1 at level {level}")

            # Check if leaf PTE (R=1 or X=1)
            is_leaf = bool((pte & PTE_R) or (pte & PTE_X))

            if is_leaf:
                # Check permissions
                self._check_permissions(vaddr, pte, access_type)

                ppn2 = (pte >> 28) & 0x3FFFFFF
                ppn1 = (pte >> 19) & 0x1FF
                ppn0 = (pte >> 10) & 0x1FF

                if level == 2:
                    # 1GB Gigapage: PPN[1:0] must be zero
                    if ppn1 != 0 or ppn0 != 0:
                        self._raise_page_fault(vaddr, access_type, "Misaligned 1GB gigapage PPN")
                    paddr = (ppn2 << 30) | (vaddr & 0x3FFFFFFF)
                    return paddr, pte, 1024 * 1024 * 1024

                elif level == 1:
                    # 2MB Megapage: PPN[0] must be zero
                    if ppn0 != 0:
                        self._raise_page_fault(vaddr, access_type, "Misaligned 2MB megapage PPN")
                    paddr = (ppn2 << 30) | (ppn1 << 21) | (vaddr & 0x1FFFFF)
                    return paddr, pte, 2 * 1024 * 1024

                else:
                    # 4KB Page
                    paddr = (pte >> 10 << PAGE_SHIFT) | offset
                    return paddr, pte, 4096

            # Non-leaf PTE -> points to next level table
            if level == 0:
                self._raise_page_fault(vaddr, access_type, "Pointer PTE reached at level 0")

            a = (pte >> 10) << PAGE_SHIFT

        self._raise_page_fault(vaddr, access_type, "Page walk failed without finding leaf")

    def _check_permissions(self, vaddr: int, pte: int, access_type: AccessType) -> None:
        """Check read, write, execute, and user/supervisor privilege permissions."""
        is_user = self.privilege_mode == PrivilegeMode.USER
        pte_u = bool(pte & PTE_U)

        # User mode cannot access supervisor pages
        if is_user and not pte_u:
            self._raise_page_fault(vaddr, access_type, "User mode accessing supervisor page")

        # Supervisor mode cannot access user pages unless SUM bit is set
        if not is_user and pte_u:
            self._raise_page_fault(vaddr, access_type, "Supervisor mode accessing user page")

        if access_type == AccessType.FETCH:
            if not (pte & PTE_X):
                raise InstructionPageFault(vaddr, access_type, f"Page not executable at 0x{vaddr:x}")
        elif access_type == AccessType.LOAD:
            if not (pte & PTE_R):
                raise LoadPageFault(vaddr, access_type, f"Page not readable at 0x{vaddr:x}")
        elif access_type == AccessType.STORE:
            if not (pte & PTE_W):
                raise StorePageFault(vaddr, access_type, f"Page not writable at 0x{vaddr:x}")

    def _raise_page_fault(self, vaddr: int, access_type: AccessType, msg: str) -> None:
        if access_type == AccessType.FETCH:
            raise InstructionPageFault(vaddr, access_type, msg)
        elif access_type == AccessType.LOAD:
            raise LoadPageFault(vaddr, access_type, msg)
        else:
            raise StorePageFault(vaddr, access_type, msg)

    # -----------------------------------------------------------------
    # Virtual Memory Mapping Helpers
    # -----------------------------------------------------------------

    def map_page_4k(
        self,
        root_ppn: int,
        vaddr: int,
        paddr: int,
        flags: int = PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D,
        alloc_ppn_start: int = 0x80000,
    ) -> int:
        """Map a single 4KB virtual page to a physical address in the SV39 page table."""
        vpn2 = (vaddr >> 30) & 0x1FF
        vpn1 = (vaddr >> 21) & 0x1FF
        vpn0 = (vaddr >> 12) & 0x1FF

        next_alloc = alloc_ppn_start

        # Level 2 table
        l2_addr = (root_ppn << PAGE_SHIFT) + vpn2 * 8
        pte2 = self.ram.read_u64(l2_addr)
        if not (pte2 & PTE_V):
            next_alloc += 1
            l1_ppn = next_alloc
            self.ram.write_u64(l2_addr, (l1_ppn << 10) | PTE_V)
        else:
            l1_ppn = (pte2 >> 10) & 0xFFF_FFFF_FFFF

        # Level 1 table
        l1_addr = (l1_ppn << PAGE_SHIFT) + vpn1 * 8
        pte1 = self.ram.read_u64(l1_addr)
        if not (pte1 & PTE_V):
            next_alloc += 1
            l0_ppn = next_alloc
            self.ram.write_u64(l1_addr, (l0_ppn << 10) | PTE_V)
        else:
            l0_ppn = (pte1 >> 10) & 0xFFF_FFFF_FFFF

        # Level 0 table (Leaf PTE)
        l0_addr = (l0_ppn << PAGE_SHIFT) + vpn0 * 8
        target_ppn = (paddr >> PAGE_SHIFT) & 0xFFF_FFFF_FFFF
        leaf_pte = (target_ppn << 10) | flags | PTE_V | PTE_A | PTE_D
        self.ram.write_u64(l0_addr, leaf_pte)

        return next_alloc
