"""Unit tests for Physical Memory, SV39 Page Table Walker, and TLBs."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from siliconrisc.memory import (
    AccessType,
    InstructionPageFault,
    LoadPageFault,
    MMU,
    PhysicalMemory,
    PrivilegeMode,
    PTE_A,
    PTE_D,
    PTE_R,
    PTE_U,
    PTE_V,
    PTE_W,
    PTE_X,
    SATP_MODE_SV39,
    StorePageFault,
    TLB,
)


class TestMemoryAndMMU(unittest.TestCase):
    """Test physical RAM read/writes, endianness, TLBs, and SV39 translation."""

    def test_physical_memory_access(self) -> None:
        mem = PhysicalMemory()
        # Test 8-bit, 16-bit, 32-bit, and 64-bit access across page boundaries
        mem.write_u8(0x1000, 0xAB)
        self.assertEqual(mem.read_u8(0x1000), 0xAB)

        mem.write_u16(0x2000, 0x1234)
        self.assertEqual(mem.read_u16(0x2000), 0x1234)

        mem.write_u32(0x3000, 0x89ABCDEF)
        self.assertEqual(mem.read_u32(0x3000), 0x89ABCDEF)
        self.assertEqual(mem.read_i32(0x3000), -1985229329)

        mem.write_u64(0x4000, 0xFEDCBA9876543210)
        self.assertEqual(mem.read_u64(0x4000), 0xFEDCBA9876543210)

        # Byte array load/read
        test_bytes = b"Hello, RISC-V SiliconCore!"
        mem.load_bytes(0x5000, test_bytes)
        self.assertEqual(mem.read_bytes(0x5000, len(test_bytes)), test_bytes)

    def test_tlb_caching_and_lru(self) -> None:
        tlb = TLB(capacity=2)
        tlb.insert(vpn=0x10, ppn=0x80, flags=PTE_V | PTE_R | PTE_W, page_size_bytes=4096, asid=1)
        tlb.insert(vpn=0x20, ppn=0x90, flags=PTE_V | PTE_R | PTE_W, page_size_bytes=4096, asid=1)

        # Lookup 0x10 -> hit
        e1 = tlb.lookup(0x10, asid=1)
        self.assertIsNotNone(e1)
        self.assertEqual(e1.ppn, 0x80)

        # Insert 3rd entry -> should evict 0x20 since 0x10 was recently accessed
        tlb.insert(vpn=0x30, ppn=0xA0, flags=PTE_V | PTE_R | PTE_W, page_size_bytes=4096, asid=1)
        self.assertIsNotNone(tlb.lookup(0x10, asid=1))
        self.assertIsNone(tlb.lookup(0x20, asid=1))  # Evicted
        self.assertIsNotNone(tlb.lookup(0x30, asid=1))

    def test_sv39_virtual_memory_walk(self) -> None:
        mem = PhysicalMemory()
        mmu = MMU(mem)

        # Root page table at PPN 0x100 (Physical address 0x100000)
        root_ppn = 0x100
        # Set SATP to SV39 mode with root_ppn
        mmu.satp = (SATP_MODE_SV39 << 60) | root_ppn
        mmu.privilege_mode = PrivilegeMode.USER

        # Map virtual address 0x10000 (VPN 0x10) to physical address 0x80000000 (PPN 0x80000)
        vaddr = 0x10000
        paddr = 0x80000000
        mmu.map_page_4k(root_ppn, vaddr, paddr, flags=PTE_R | PTE_W | PTE_U)

        # Write data to physical memory
        mem.write_u64(paddr, 0xCAFEBABE_DEADBEEF)

        # Translate virtual address
        translated_paddr = mmu.translate(vaddr, AccessType.LOAD)
        self.assertEqual(translated_paddr, paddr)
        self.assertEqual(mem.read_u64(translated_paddr), 0xCAFEBABE_DEADBEEF)
        self.assertEqual(mmu.dtlb_misses, 1)

        # Second access should hit TLB
        translated_paddr_2 = mmu.translate(vaddr + 8, AccessType.LOAD)
        self.assertEqual(translated_paddr_2, paddr + 8)
        self.assertEqual(mmu.dtlb_hits, 1)

    def test_sv39_permission_faults(self) -> None:
        mem = PhysicalMemory()
        mmu = MMU(mem)
        root_ppn = 0x200
        mmu.satp = (SATP_MODE_SV39 << 60) | root_ppn
        mmu.privilege_mode = PrivilegeMode.USER

        # Read-only page
        vaddr_ro = 0x20000
        paddr_ro = 0x90000000
        mmu.map_page_4k(root_ppn, vaddr_ro, paddr_ro, flags=PTE_R | PTE_U)

        # Read should succeed
        mmu.translate(vaddr_ro, AccessType.LOAD)

        # Write should raise StorePageFault
        with self.assertRaises(StorePageFault):
            mmu.translate(vaddr_ro, AccessType.STORE)

        # Execute should raise InstructionPageFault
        with self.assertRaises(InstructionPageFault):
            mmu.translate(vaddr_ro, AccessType.FETCH)


if __name__ == "__main__":
    unittest.main()
