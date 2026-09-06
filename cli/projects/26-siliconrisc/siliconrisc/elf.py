"""ELF64 Binary Parser, Loader, and Lightweight RV64GC Assembler.

Provides:
1. Complete ELF64 Binary Parser and Loader:
   - Validates ELF magic, architecture (EM_RISCV = 243), 64-bit class, and little-endianness
   - Loads PT_LOAD segments into memory and initializes zero-filled BSS sections
   - Supports page table mapping generation with SV39 PTE permission bits
2. RV64GC 2-Pass Assembler:
   - Assembles assembly source code with labels, pseudo-instructions (li, mv, ret, j, call, etc.)
   - Emits raw instruction words or fully compliant ELF64 executable binaries
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
import struct
from typing import Any, Dict, List, Optional, Tuple

from .isa import (
    ABI_REG_NAMES,
    CSR_CYCLE,
    CSR_INSTRET,
    CSR_MCAUSE,
    CSR_MEPC,
    CSR_MIE,
    CSR_MIP,
    CSR_MISA,
    CSR_MSCRATCH,
    CSR_MSTATUS,
    CSR_MTVEC,
    CSR_MTVAL,
    CSR_SATP,
    CSR_SCAUSE,
    CSR_SEPC,
    CSR_SSTATUS,
    CSR_STVAL,
    CSR_STVEC,
    MASK_64,
    to_signed_64,
)
from .memory import MMU, AccessType, PhysicalMemory, PTE_A, PTE_D, PTE_R, PTE_U, PTE_V, PTE_W, PTE_X


# ELF Constants
ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1  # 2's complement, little endian
EV_CURRENT = 1
EM_RISCV = 243   # 0xF3
ET_EXEC = 2
ET_DYN = 3

# Segment Types
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_SHLIB = 5
PT_PHDR = 6
PT_TLS = 7

# Segment Flags
PF_X = 0x1  # Execute
PF_W = 0x2  # Write
PF_R = 0x4  # Read

# Section Types
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_HASH = 5
SHT_DYNAMIC = 6
SHT_NOTE = 7
SHT_NOBITS = 8
SHT_REL = 9
SHT_SHLIB = 10
SHT_DYNSYM = 11


@dataclass
class Elf64ProgramHeader:
    p_type: int
    p_flags: int
    p_offset: int
    p_vaddr: int
    p_paddr: int
    p_filesz: int
    p_memsz: int
    p_align: int

    @property
    def is_loadable(self) -> bool:
        return self.p_type == PT_LOAD

    @property
    def is_readable(self) -> bool:
        return bool(self.p_flags & PF_R)

    @property
    def is_writable(self) -> bool:
        return bool(self.p_flags & PF_W)

    @property
    def is_executable(self) -> bool:
        return bool(self.p_flags & PF_X)


@dataclass
class Elf64SectionHeader:
    sh_name: int
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int
    name: str = ""


@dataclass
class Elf64File:
    entry: int
    ph_headers: List[Elf64ProgramHeader] = field(default_factory=list)
    sh_headers: List[Elf64SectionHeader] = field(default_factory=list)
    raw_data: bytes = b""


class Elf64Parser:
    """Parser for 64-bit RISC-V ELF Executables."""

    @staticmethod
    def parse(data: bytes) -> Elf64File:
        if len(data) < 64:
            raise ValueError(f"File too small to be ELF64: {len(data)} bytes")

        magic = data[0:4]
        if magic != ELF_MAGIC:
            raise ValueError(f"Invalid ELF magic: {magic}")

        ei_class = data[4]
        if ei_class != ELFCLASS64:
            raise ValueError(f"Unsupported ELF class: {ei_class} (only ELF64 supported)")

        ei_data = data[5]
        if ei_data != ELFDATA2LSB:
            raise ValueError(f"Unsupported endianness: {ei_data} (only little-endian supported)")

        e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
            "<HHIQQQIHHHHHH", data, 16
        )

        if e_machine != EM_RISCV:
            raise ValueError(f"Unsupported machine architecture: 0x{e_machine:X} (expected RISC-V 0xF3)")

        elf = Elf64File(entry=e_entry, raw_data=data)

        # Parse Program Headers
        for i in range(e_phnum):
            ph_offset = e_phoff + i * e_phentsize
            if ph_offset + 56 > len(data):
                break
            p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(
                "<IIQQQQQQ", data, ph_offset
            )
            elf.ph_headers.append(
                Elf64ProgramHeader(
                    p_type=p_type,
                    p_flags=p_flags,
                    p_offset=p_offset,
                    p_vaddr=p_vaddr,
                    p_paddr=p_paddr,
                    p_filesz=p_filesz,
                    p_memsz=p_memsz,
                    p_align=p_align,
                )
            )

        # Parse Section Headers
        for i in range(e_shnum):
            sh_offset = e_shoff + i * e_shentsize
            if sh_offset + 64 > len(data):
                break
            sh_name, sh_type, sh_flags, sh_addr, sh_offset_val, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from(
                "<IIQQQQIIQQ", data, sh_offset
            )
            elf.sh_headers.append(
                Elf64SectionHeader(
                    sh_name=sh_name,
                    sh_type=sh_type,
                    sh_flags=sh_flags,
                    sh_addr=sh_addr,
                    sh_offset=sh_offset_val,
                    sh_size=sh_size,
                    sh_link=sh_link,
                    sh_info=sh_info,
                    sh_addralign=sh_addralign,
                    sh_entsize=sh_entsize,
                )
            )

        # Resolve section names using shstrtab
        if 0 <= e_shstrndx < len(elf.sh_headers):
            strtab_hdr = elf.sh_headers[e_shstrndx]
            strtab_bytes = data[strtab_hdr.sh_offset : strtab_hdr.sh_offset + strtab_hdr.sh_size]
            for sh in elf.sh_headers:
                if sh.sh_name < len(strtab_bytes):
                    null_idx = strtab_bytes.find(b"\x00", sh.sh_name)
                    if null_idx != -1:
                        sh.name = strtab_bytes[sh.sh_name : null_idx].decode("latin-1", errors="replace")

        return elf


class ElfLoader:
    """Loads an ELF64 file into physical memory and sets up SV39 page tables."""

    @staticmethod
    def load(
        elf: Elf64File,
        ram: PhysicalMemory,
        mmu: Optional[MMU] = None,
        base_load_offset: int = 0,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """Loads all PT_LOAD segments into memory.

        Returns:
            Tuple of (entry_point_pc, list of loaded segment details).
        """
        loaded_segments = []

        for ph in elf.ph_headers:
            if not ph.is_loadable:
                continue

            vaddr = ph.p_vaddr + base_load_offset
            file_data = elf.raw_data[ph.p_offset : ph.p_offset + ph.p_filesz]

            # Write file bytes into memory
            ram.write_bytes(vaddr, file_data)

            # Zero-fill BSS section if memsz > filesz
            if ph.p_memsz > ph.p_filesz:
                bss_size = ph.p_memsz - ph.p_filesz
                bss_start = vaddr + ph.p_filesz
                ram.write_bytes(bss_start, bytes(bss_size))

            # Set up MMU SV39 page tables if MMU provided and paging enabled
            if mmu is not None and mmu.is_paging_enabled():
                perm_flags = PTE_V | PTE_U | PTE_A | PTE_D
                if ph.is_readable:
                    perm_flags |= PTE_R
                if ph.is_writable:
                    perm_flags |= PTE_W
                if ph.is_executable:
                    perm_flags |= PTE_X

                # Map 4KB pages
                start_page = vaddr & ~0xFFF
                end_page = (vaddr + ph.p_memsz + 0xFFF) & ~0xFFF
                for page in range(start_page, end_page, 4096):
                    mmu.map_page_4k(mmu.root_ppn, page, page, perm_flags)

            loaded_segments.append(
                {
                    "vaddr": vaddr,
                    "memsz": ph.p_memsz,
                    "filesz": ph.p_filesz,
                    "readable": ph.is_readable,
                    "writable": ph.is_writable,
                    "executable": ph.is_executable,
                }
            )

        entry_point = elf.entry + base_load_offset
        return entry_point, loaded_segments


# =====================================================================
# RV64GC Assembler
# =====================================================================

REG_ALIAS_MAP: Dict[str, int] = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25,
    "s10": 26, "s11": 27, "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}
# Populate x0-x31
for i in range(32):
    REG_ALIAS_MAP[f"x{i}"] = i
    REG_ALIAS_MAP[f"f{i}"] = i


class Assembler:
    """Two-Pass RV64GC Assembler with pseudo-instructions and ELF generator."""

    def __init__(self, base_pc: int = 0x80000000) -> None:
        self.base_pc = base_pc
        self.labels: Dict[str, int] = {}
        self.lines: List[Tuple[int, str]] = []  # (pc, raw_assembly)

    def _parse_reg(self, token: str) -> int:
        token = token.strip().rstrip(",").lower()
        if token in REG_ALIAS_MAP:
            return REG_ALIAS_MAP[token]
        raise ValueError(f"Unknown register name: '{token}'")

    def _parse_imm(self, token: str, current_pc: int) -> int:
        token = token.strip().rstrip(",")
        if token in self.labels:
            return self.labels[token] - current_pc
        if token.startswith("0x") or token.startswith("0X"):
            return int(token, 16)
        if token.startswith("-0x") or token.startswith("-0X"):
            return -int(token[1:], 16)
        return int(token)

    def assemble(self, source: str) -> List[int]:
        """Assemble assembly source text into list of 32-bit machine instruction words."""
        # Pass 1: Normalize lines, compute instruction addresses, register labels
        raw_lines = source.splitlines()
        current_pc = self.base_pc
        parsed_instructions: List[Tuple[int, str]] = []

        for line_num, line in enumerate(raw_lines):
            line = line.split("#")[0].split(";")[0].strip()
            if not line:
                continue

            # Check for label definition
            if ":" in line:
                label_part, rest = line.split(":", 1)
                label_name = label_part.strip()
                self.labels[label_name] = current_pc
                line = rest.strip()
                if not line:
                    continue

            # Macro expansions for pseudo-instructions
            parts = line.split(None, 1)
            op = parts[0].lower()
            args = parts[1].strip() if len(parts) > 1 else ""

            if op == "li":
                # li rd, imm -> lui rd, hi ; addi rd, rd, lo or addiw rd, x0, imm
                rd_str, imm_str = [x.strip() for x in args.split(",", 1)]
                imm_val = int(imm_str, 0) if imm_str.lstrip("-").isalnum() else 0
                if -2048 <= imm_val < 2048:
                    parsed_instructions.append((current_pc, f"addi {rd_str}, zero, {imm_val}"))
                    current_pc += 4
                else:
                    # 2-instruction li
                    hi = ((imm_val + 0x800) >> 12) & 0xFFFFF
                    lo = imm_val - (hi << 12)
                    parsed_instructions.append((current_pc, f"lui {rd_str}, {hi}"))
                    current_pc += 4
                    parsed_instructions.append((current_pc, f"addi {rd_str}, {rd_str}, {lo}"))
                    current_pc += 4
                continue
            elif op == "mv":
                # mv rd, rs -> addi rd, rs, 0
                rd_str, rs_str = [x.strip() for x in args.split(",", 1)]
                parsed_instructions.append((current_pc, f"addi {rd_str}, {rs_str}, 0"))
                current_pc += 4
                continue
            elif op == "nop":
                parsed_instructions.append((current_pc, "addi zero, zero, 0"))
                current_pc += 4
                continue
            elif op == "ret":
                parsed_instructions.append((current_pc, "jalr zero, ra, 0"))
                current_pc += 4
                continue
            elif op == "j":
                parsed_instructions.append((current_pc, f"jal zero, {args}"))
                current_pc += 4
                continue

            parsed_instructions.append((current_pc, line))
            current_pc += 4

        # Pass 2: Encode machine code words
        encoded_words: List[int] = []
        for pc, line in parsed_instructions:
            parts = line.split(None, 1)
            mnemonic = parts[0].lower()
            args = [x.strip() for x in parts[1].split(",")] if len(parts) > 1 else []

            word = self._encode_instruction(pc, mnemonic, args)
            encoded_words.append(word)

        return encoded_words

    def _encode_instruction(self, pc: int, mnemonic: str, args: List[str]) -> int:
        """Encode single instruction into 32-bit RISC-V integer."""
        # 1. R-Type: funct7[31:25] | rs2[24:20] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
        R_TYPES = {
            "add":   (0x33, 0x0, 0x00),
            "sub":   (0x33, 0x0, 0x20),
            "sll":   (0x33, 0x1, 0x00),
            "slt":   (0x33, 0x2, 0x00),
            "sltu":  (0x33, 0x3, 0x00),
            "xor":   (0x33, 0x4, 0x00),
            "srl":   (0x33, 0x5, 0x00),
            "sra":   (0x33, 0x5, 0x20),
            "or":    (0x33, 0x6, 0x00),
            "and":   (0x33, 0x7, 0x00),
            "addw":  (0x3B, 0x0, 0x00),
            "subw":  (0x3B, 0x0, 0x20),
            "mul":   (0x33, 0x0, 0x01),
            "mulh":  (0x33, 0x1, 0x01),
            "mulhu": (0x33, 0x3, 0x01),
            "div":   (0x33, 0x4, 0x01),
            "divu":  (0x33, 0x5, 0x01),
            "rem":   (0x33, 0x6, 0x01),
            "remu":  (0x33, 0x7, 0x01),
        }
        if mnemonic in R_TYPES:
            opcode, funct3, funct7 = R_TYPES[mnemonic]
            rd = self._parse_reg(args[0])
            rs1 = self._parse_reg(args[1])
            rs2 = self._parse_reg(args[2])
            return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

        # 2. I-Type Arithmetic: imm[11:0] | rs1 | funct3 | rd | opcode
        I_ARITH = {
            "addi":  (0x13, 0x0),
            "slti":  (0x13, 0x2),
            "sltiu": (0x13, 0x3),
            "xori":  (0x13, 0x4),
            "ori":   (0x13, 0x6),
            "andi":  (0x13, 0x7),
            "addiw": (0x1B, 0x0),
        }
        if mnemonic in I_ARITH:
            opcode, funct3 = I_ARITH[mnemonic]
            rd = self._parse_reg(args[0])
            rs1 = self._parse_reg(args[1])
            imm = self._parse_imm(args[2], pc) & 0xFFF
            return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

        # Shifts
        I_SHIFTS = {
            "slli": (0x13, 0x1, 0x00),
            "srli": (0x13, 0x5, 0x00),
            "srai": (0x13, 0x5, 0x10),
        }
        if mnemonic in I_SHIFTS:
            opcode, funct3, f6 = I_SHIFTS[mnemonic]
            rd = self._parse_reg(args[0])
            rs1 = self._parse_reg(args[1])
            shamt = self._parse_imm(args[2], pc) & 0x3F
            return (f6 << 26) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

        # 3. Loads: lw rd, offset(rs1)
        I_LOADS = {
            "lb":  (0x03, 0x0),
            "lh":  (0x03, 0x1),
            "lw":  (0x03, 0x2),
            "ld":  (0x03, 0x3),
            "lbu": (0x03, 0x4),
            "lhu": (0x03, 0x5),
            "lwu": (0x03, 0x6),
        }
        if mnemonic in I_LOADS:
            opcode, funct3 = I_LOADS[mnemonic]
            rd = self._parse_reg(args[0])
            # parse offset(rs1)
            offset_str, rs1_str = args[1].split("(")
            rs1 = self._parse_reg(rs1_str.rstrip(")"))
            imm = self._parse_imm(offset_str, pc) & 0xFFF
            return (imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode

        # 4. Stores: sw rs2, offset(rs1)
        S_STORES = {
            "sb": (0x23, 0x0),
            "sh": (0x23, 0x1),
            "sw": (0x23, 0x2),
            "sd": (0x23, 0x3),
        }
        if mnemonic in S_STORES:
            opcode, funct3 = S_STORES[mnemonic]
            rs2 = self._parse_reg(args[0])
            offset_str, rs1_str = args[1].split("(")
            rs1 = self._parse_reg(rs1_str.rstrip(")"))
            imm = self._parse_imm(offset_str, pc) & 0xFFF
            imm_11_5 = (imm >> 5) & 0x7F
            imm_4_0 = imm & 0x1F
            return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode

        # 5. Branches: beq rs1, rs2, label
        B_BRANCHES = {
            "beq":  (0x63, 0x0),
            "bne":  (0x63, 0x1),
            "blt":  (0x63, 0x4),
            "bge":  (0x63, 0x5),
            "bltu": (0x63, 0x6),
            "bgeu": (0x63, 0x7),
        }
        if mnemonic in B_BRANCHES:
            opcode, funct3 = B_BRANCHES[mnemonic]
            rs1 = self._parse_reg(args[0])
            rs2 = self._parse_reg(args[1])
            offset = self._parse_imm(args[2], pc)
            imm12 = (offset >> 12) & 1
            imm11 = (offset >> 11) & 1
            imm10_5 = (offset >> 5) & 0x3F
            imm4_1 = (offset >> 1) & 0xF
            return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | opcode

        # 6. Upper Immediates: lui / auipc rd, imm
        if mnemonic in ("lui", "auipc"):
            opcode = 0x37 if mnemonic == "lui" else 0x17
            rd = self._parse_reg(args[0])
            imm20 = self._parse_imm(args[1], pc) & 0xFFFFF
            return (imm20 << 12) | (rd << 7) | opcode

        # 7. JAL: jal rd, label
        if mnemonic == "jal":
            rd = self._parse_reg(args[0])
            offset = self._parse_imm(args[1], pc)
            imm20 = (offset >> 20) & 1
            imm19_12 = (offset >> 12) & 0xFF
            imm11 = (offset >> 11) & 1
            imm10_1 = (offset >> 1) & 0x3FF
            return (imm20 << 31) | (imm10_1 << 21) | (imm11 << 20) | (imm19_12 << 12) | (rd << 7) | 0x6F

        # 8. JALR: jalr rd, rs1, offset
        if mnemonic == "jalr":
            rd = self._parse_reg(args[0])
            rs1 = self._parse_reg(args[1])
            offset = self._parse_imm(args[2], pc) if len(args) > 2 else 0
            return ((offset & 0xFFF) << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x67

        # 9. System / Traps
        if mnemonic == "ecall":
            return 0x00000073
        if mnemonic == "ebreak":
            return 0x00100073

        # 10. CSR Operations: csrrw rd, csr, rs1
        CSR_OPS = {
            "csrrw": 0x1,
            "csrrs": 0x2,
            "csrrc": 0x3,
        }
        if mnemonic in CSR_OPS:
            funct3 = CSR_OPS[mnemonic]
            rd = self._parse_reg(args[0])
            csr_imm = self._parse_imm(args[1], pc) & 0xFFF
            rs1 = self._parse_reg(args[2])
            return (csr_imm << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x73

        raise ValueError(f"Unsupported instruction or mnemonic: '{mnemonic}'")

    @staticmethod
    def create_elf_binary(entry_pc: int, code_words: List[int]) -> bytes:
        """Create a valid 64-bit RISC-V ELF executable binary containing the given instructions."""
        code_bytes = bytearray()
        for word in code_words:
            code_bytes.extend(struct.pack("<I", word))

        ehdr_size = 64
        phdr_size = 56
        num_phdrs = 1

        file_offset = ehdr_size + phdr_size
        filesz = len(code_bytes)
        memsz = filesz

        # Program Header (PT_LOAD, PF_R | PF_X)
        phdr = struct.pack(
            "<IIQQQQQQ",
            PT_LOAD,               # p_type
            PF_R | PF_X,           # p_flags
            file_offset,           # p_offset
            entry_pc,              # p_vaddr
            entry_pc,              # p_paddr
            filesz,                # p_filesz
            memsz,                 # p_memsz
            0x1000,                # p_align (4KB)
        )

        # ELF Header
        ident = bytearray(16)
        ident[0:4] = ELF_MAGIC
        ident[4] = ELFCLASS64
        ident[5] = ELFDATA2LSB
        ident[6] = EV_CURRENT

        ehdr = bytes(ident) + struct.pack(
            "<HHIQQQIHHHHHH",
            ET_EXEC,               # e_type
            EM_RISCV,              # e_machine
            EV_CURRENT,            # e_version
            entry_pc,              # e_entry
            ehdr_size,             # e_phoff
            0,                     # e_shoff
            0,                     # e_flags
            ehdr_size,             # e_ehsize
            phdr_size,             # e_phentsize
            num_phdrs,             # e_phnum
            0,                     # e_shentsize
            0,                     # e_shnum
            0,                     # e_shstrndx
        )

        return ehdr + phdr + bytes(code_bytes)
