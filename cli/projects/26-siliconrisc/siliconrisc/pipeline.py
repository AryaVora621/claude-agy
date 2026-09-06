"""5-Stage In-Order RISC-V Pipeline (IF, ID, EX, MEM, WB) with Data Forwarding and Hazard Detection.

Provides:
1. Classic 5-stage pipeline registers (IF/ID, ID/EX, EX/MEM, MEM/WB)
2. Forwarding Unit resolving RAW data hazards (EX->EX and MEM->EX forwarding)
3. Hazard Detection Unit injecting load-use stalls and branch misprediction flushes
4. Full 64-bit ALU and branch evaluator
5. Cycle-accurate pipeline stepping and pipeline bubble telemetry
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import Dict, List, Optional, Tuple

from .branch import BranchPredictorUnit
from .cache import MemoryHierarchy
from .isa import (
    ABI_REG_NAMES,
    DecodedInstruction,
    Disassembler,
    InstCategory,
    InstFormat,
    InstructionDecoder,
    RegisterFile,
    bits_to_float_64,
    float_to_bits_64,
    to_signed_32,
    to_signed_64,
    to_unsigned_32,
    to_unsigned_64,
)
from .memory import MMU, AccessType


# =====================================================================
# Pipeline Stage Registers
# =====================================================================

@dataclass
class IF_ID_Reg:
    pc: int = 0
    raw: int = 0x00000013  # NOP
    inst: Optional[DecodedInstruction] = None
    pred_taken: bool = False
    pred_target: Optional[int] = None
    valid: bool = False


@dataclass
class ID_EX_Reg:
    pc: int = 0
    inst: Optional[DecodedInstruction] = None
    val_rs1: int = 0
    val_rs2: int = 0
    val_fp_rs1: float = 0.0
    val_fp_rs2: float = 0.0
    imm: int = 0
    pred_taken: bool = False
    pred_target: Optional[int] = None
    valid: bool = False


@dataclass
class EX_MEM_Reg:
    pc: int = 0
    inst: Optional[DecodedInstruction] = None
    alu_result: int = 0
    fp_result: float = 0.0
    val_rs2: int = 0          # Data to store
    val_fp_rs2: float = 0.0   # FP data to store
    actual_taken: bool = False
    actual_target: int = 0
    mispredicted: bool = False
    valid: bool = False


@dataclass
class MEM_WB_Reg:
    pc: int = 0
    inst: Optional[DecodedInstruction] = None
    alu_result: int = 0
    mem_data: int = 0
    fp_result: float = 0.0
    fp_mem_data: float = 0.0
    valid: bool = False


# =====================================================================
# Forwarding and Hazard Units
# =====================================================================

class ForwardingUnit:
    """Detects Read-After-Write (RAW) data hazards and controls forwarding multiplexers."""

    @staticmethod
    def get_forward_a(id_ex: ID_EX_Reg, ex_mem: EX_MEM_Reg, mem_wb: MEM_WB_Reg) -> int:
        """Returns ForwardA control signal: 0 (No fwd), 2 (EX->EX fwd), 1 (MEM->EX fwd)."""
        if not id_ex.valid or id_ex.inst is None or not id_ex.inst.reads_rs1:
            return 0
        rs1 = id_ex.inst.rs1
        if rs1 == 0:
            return 0

        # EX Hazard: Prior instruction in EX_MEM writes rd
        if ex_mem.valid and ex_mem.inst is not None and ex_mem.inst.writes_rd and ex_mem.inst.rd == rs1:
            return 2

        # MEM Hazard: Prior instruction in MEM_WB writes rd
        if mem_wb.valid and mem_wb.inst is not None and mem_wb.inst.writes_rd and mem_wb.inst.rd == rs1:
            return 1

        return 0

    @staticmethod
    def get_forward_b(id_ex: ID_EX_Reg, ex_mem: EX_MEM_Reg, mem_wb: MEM_WB_Reg) -> int:
        """Returns ForwardB control signal: 0 (No fwd), 2 (EX->EX fwd), 1 (MEM->EX fwd)."""
        if not id_ex.valid or id_ex.inst is None or not id_ex.inst.reads_rs2:
            return 0
        rs2 = id_ex.inst.rs2
        if rs2 == 0:
            return 0

        # EX Hazard
        if ex_mem.valid and ex_mem.inst is not None and ex_mem.inst.writes_rd and ex_mem.inst.rd == rs2:
            return 2

        # MEM Hazard
        if mem_wb.valid and mem_wb.inst is not None and mem_wb.inst.writes_rd and mem_wb.inst.rd == rs2:
            return 1

        return 0


class HazardDetectionUnit:
    """Detects Load-Use hazards requiring pipeline stalls, and branch mispredictions."""

    @staticmethod
    def check_load_use(if_id: IF_ID_Reg, id_ex: ID_EX_Reg) -> bool:
        """Returns True if a 1-cycle pipeline stall is needed due to load-use hazard."""
        if not id_ex.valid or id_ex.inst is None or not id_ex.inst.is_load:
            return False
        if not if_id.valid or if_id.inst is None:
            return False

        load_rd = id_ex.inst.rd
        if load_rd == 0:
            return False

        # If instruction in ID reads load_rd as rs1 or rs2
        if if_id.inst.reads_rs1 and if_id.inst.rs1 == load_rd:
            return True
        if if_id.inst.reads_rs2 and not if_id.inst.is_store and if_id.inst.rs2 == load_rd:
            return True

        return False


# =====================================================================
# 5-Stage Pipelined Processor Engine
# =====================================================================

@dataclass
class PipelineStatus:
    cycle: int
    pc_if: int
    inst_if: str
    inst_id: str
    inst_ex: str
    inst_mem: str
    inst_wb: str
    stalled: bool
    flushed: bool
    forward_a: int
    forward_b: int
    committed_instructions: int
    ipc: float


class PipelinedCore:
    """Cycle-accurate 5-stage in-order RISC-V processor core."""

    def __init__(
        self,
        mmu: MMU,
        cache_hier: MemoryHierarchy,
        bpu: BranchPredictorUnit,
        initial_pc: int = 0x80000000,
    ) -> None:
        self.mmu = mmu
        self.cache = cache_hier
        self.bpu = bpu

        self.reg_file = RegisterFile()
        self.pc: int = initial_pc

        # Pipeline Registers
        self.if_id = IF_ID_Reg()
        self.id_ex = ID_EX_Reg()
        self.ex_mem = EX_MEM_Reg()
        self.mem_wb = MEM_WB_Reg()

        # Performance Counters
        self.cycle_count: int = 0
        self.committed_insts: int = 0
        self.stall_cycles: int = 0
        self.flush_cycles: int = 0
        self.branch_count: int = 0
        self.branch_mispredicts: int = 0

        self.halted: bool = False

    def step(self) -> PipelineStatus:
        """Advance the pipeline by one clock cycle. Executed in reverse order: WB -> MEM -> EX -> ID -> IF."""
        if self.halted:
            return self._capture_status(stalled=False, flushed=False, fwd_a=0, fwd_b=0)

        self.cycle_count += 1
        stalled = False
        flushed = False

        # -------------------------------------------------------------
        # Stage 5: Write Back (WB)
        # -------------------------------------------------------------
        if self.mem_wb.valid and self.mem_wb.inst is not None:
            inst = self.mem_wb.inst
            if inst.writes_rd and inst.rd != 0:
                wb_val = self.mem_wb.mem_data if inst.is_load else self.mem_wb.alu_result
                self.reg_file.write_x(inst.rd, wb_val)
            elif inst.writes_fp_rd:
                fp_val = self.mem_wb.fp_mem_data if inst.is_load else self.mem_wb.fp_result
                self.reg_file.write_f(inst.rd, fp_val)

            self.committed_insts += 1

            # Stop on EBREAK or ECALL termination
            if inst.category == InstCategory.SYSTEM and inst.mnemonic == "ebreak":
                self.halted = True

        # -------------------------------------------------------------
        # Stage 4: Memory Access (MEM)
        # -------------------------------------------------------------
        next_mem_wb = MEM_WB_Reg()
        if self.ex_mem.valid and self.ex_mem.inst is not None:
            inst = self.ex_mem.inst
            addr = self.ex_mem.alu_result
            next_mem_wb.pc = self.ex_mem.pc
            next_mem_wb.inst = inst
            next_mem_wb.alu_result = addr
            next_mem_wb.fp_result = self.ex_mem.fp_result
            next_mem_wb.valid = True

            if inst.is_load:
                paddr = self.mmu.translate(addr, AccessType.LOAD)
                size_map = {
                    "lb": 1, "lbu": 1,
                    "lh": 2, "lhu": 2,
                    "lw": 4, "lwu": 4,
                    "ld": 8, "fld": 8, "flw": 4,
                }
                size = size_map.get(inst.mnemonic, 8)
                val, lat = self.cache.read_data(paddr, size)

                # Sign-extension logic
                if inst.mnemonic == "lb":
                    next_mem_wb.mem_data = to_signed_64(sign_extend(val, 8))
                elif inst.mnemonic == "lh":
                    next_mem_wb.mem_data = to_signed_64(sign_extend(val, 16))
                elif inst.mnemonic == "lw":
                    next_mem_wb.mem_data = to_signed_64(sign_extend(val, 32))
                elif inst.mnemonic in ("lbu", "lhu", "lwu", "ld"):
                    next_mem_wb.mem_data = val
                elif inst.mnemonic == "fld":
                    next_mem_wb.fp_mem_data = bits_to_float_64(val)
                elif inst.mnemonic == "flw":
                    next_mem_wb.fp_mem_data = float(struct.unpack("<f", struct.pack("<I", val & 0xFFFFFFFF))[0])

            elif inst.is_store:
                paddr = self.mmu.translate(addr, AccessType.STORE)
                size_map = {"sb": 1, "sh": 2, "sw": 4, "sd": 8, "fsd": 8, "fsw": 4}
                size = size_map.get(inst.mnemonic, 8)
                store_val = self.ex_mem.val_rs2
                if inst.mnemonic == "fsd":
                    store_val = float_to_bits_64(self.ex_mem.val_fp_rs2)
                elif inst.mnemonic == "fsw":
                    store_val = struct.unpack("<I", struct.pack("<f", self.ex_mem.val_fp_rs2))[0]
                self.cache.write_data(paddr, store_val, size)

        # -------------------------------------------------------------
        # Stage 3: Execution & Branch Resolution (EX)
        # -------------------------------------------------------------
        next_ex_mem = EX_MEM_Reg()
        fwd_a = ForwardingUnit.get_forward_a(self.id_ex, self.ex_mem, self.mem_wb)
        fwd_b = ForwardingUnit.get_forward_b(self.id_ex, self.ex_mem, self.mem_wb)

        # Compute forwarded operand values
        op_a = self.id_ex.val_rs1
        if fwd_a == 2:
            op_a = self.ex_mem.alu_result
        elif fwd_a == 1:
            op_a = self.mem_wb.mem_data if self.mem_wb.inst and self.mem_wb.inst.is_load else self.mem_wb.alu_result

        op_b = self.id_ex.val_rs2
        if fwd_b == 2:
            op_b = self.ex_mem.alu_result
        elif fwd_b == 1:
            op_b = self.mem_wb.mem_data if self.mem_wb.inst and self.mem_wb.inst.is_load else self.mem_wb.alu_result

        branch_redirect_pc: Optional[int] = None

        if self.id_ex.valid and self.id_ex.inst is not None:
            inst = self.id_ex.inst
            next_ex_mem.pc = self.id_ex.pc
            next_ex_mem.inst = inst
            next_ex_mem.val_rs2 = op_b
            next_ex_mem.val_fp_rs2 = self.id_ex.val_fp_rs2
            next_ex_mem.valid = True

            # ALU & Branch Evaluation
            alu_res, fp_res, taken, target = self._execute_alu(inst, self.id_ex.pc, op_a, op_b, self.id_ex.imm)
            next_ex_mem.alu_result = alu_res
            next_ex_mem.fp_result = fp_res
            next_ex_mem.actual_taken = taken
            next_ex_mem.actual_target = target

            # Branch / Jump resolution
            if inst.is_branch or inst.is_jump:
                self.branch_count += 1
                pred_taken = self.id_ex.pred_taken
                pred_target = self.id_ex.pred_target

                mispredicted = (taken != pred_taken) or (taken and target != pred_target)
                if mispredicted:
                    self.branch_mispredicts += 1
                    flushed = True
                    self.flush_cycles += 1
                    next_ex_mem.mispredicted = True
                    branch_redirect_pc = target if taken else (self.id_ex.pc + 4)

                # Update Branch Predictor
                is_call = inst.mnemonic == "jal" and inst.rd == 1
                is_ret = inst.mnemonic == "jalr" and inst.rs1 == 1 and inst.rd == 0
                self.bpu.update(
                    pc=self.id_ex.pc,
                    predicted_taken=pred_taken,
                    actual_taken=taken,
                    actual_target=target,
                    is_conditional=inst.is_branch,
                    is_call=is_call,
                    return_pc=self.id_ex.pc + 4 if is_call else None,
                )

        # -------------------------------------------------------------
        # Hazard Detection: Load-Use Hazard Check
        # -------------------------------------------------------------
        load_use_hazard = HazardDetectionUnit.check_load_use(self.if_id, self.id_ex)

        # -------------------------------------------------------------
        # Stage 2: Instruction Decode & Register Fetch (ID)
        # -------------------------------------------------------------
        next_id_ex = ID_EX_Reg()
        if branch_redirect_pc is not None:
            # Branch misprediction flush: inject bubble into ID/EX
            next_id_ex.valid = False
        elif load_use_hazard:
            # Load-Use hazard stall: inject bubble into ID/EX, keep IF/ID and PC frozen
            stalled = True
            self.stall_cycles += 1
            next_id_ex.valid = False
        elif self.if_id.valid and self.if_id.inst is not None:
            inst = self.if_id.inst
            next_id_ex.pc = self.if_id.pc
            next_id_ex.inst = inst
            next_id_ex.val_rs1 = self.reg_file.read_x(inst.rs1)
            next_id_ex.val_rs2 = self.reg_file.read_x(inst.rs2)
            next_id_ex.val_fp_rs1 = self.reg_file.read_f(inst.rs1)
            next_id_ex.val_fp_rs2 = self.reg_file.read_f(inst.rs2)
            next_id_ex.imm = inst.imm
            next_id_ex.pred_taken = self.if_id.pred_taken
            next_id_ex.pred_target = self.if_id.pred_target
            next_id_ex.valid = True

        # -------------------------------------------------------------
        # Stage 1: Instruction Fetch (IF)
        # -------------------------------------------------------------
        next_if_id = IF_ID_Reg()
        if branch_redirect_pc is not None:
            # Redirect PC to true branch destination and fetch
            self.pc = branch_redirect_pc
            paddr = self.mmu.translate(self.pc, AccessType.FETCH)
            raw_inst, _ = self.cache.fetch_instruction(paddr)
            inst = InstructionDecoder.decode(raw_inst, self.pc)

            is_call = inst.mnemonic == "jal" and inst.rd == 1
            is_ret = inst.mnemonic == "jalr" and inst.rs1 == 1 and inst.rd == 0
            pred_taken, pred_target = self.bpu.predict(
                self.pc,
                is_conditional=inst.is_branch,
                is_call=is_call,
                is_return=is_ret,
            )

            next_if_id.pc = self.pc
            next_if_id.raw = raw_inst
            next_if_id.inst = inst
            next_if_id.pred_taken = pred_taken
            next_if_id.pred_target = pred_target
            next_if_id.valid = True

            self.pc = pred_target if (pred_taken and pred_target is not None) else (self.pc + 4)

        elif load_use_hazard:
            # Stall: preserve existing IF/ID register and do not advance PC
            next_if_id = self.if_id

        else:
            # Normal Fetch
            paddr = self.mmu.translate(self.pc, AccessType.FETCH)
            raw_inst, _ = self.cache.fetch_instruction(paddr)
            inst = InstructionDecoder.decode(raw_inst, self.pc)

            is_call = inst.mnemonic == "jal" and inst.rd == 1
            is_ret = inst.mnemonic == "jalr" and inst.rs1 == 1 and inst.rd == 0
            pred_taken, pred_target = self.bpu.predict(
                self.pc,
                is_conditional=inst.is_branch,
                is_call=is_call,
                is_return=is_ret,
            )

            next_if_id.pc = self.pc
            next_if_id.raw = raw_inst
            next_if_id.inst = inst
            next_if_id.pred_taken = pred_taken
            next_if_id.pred_target = pred_target
            next_if_id.valid = True

            self.pc = pred_target if (pred_taken and pred_target is not None) else (self.pc + 4)

        # Latch Pipeline Registers for next clock cycle
        self.mem_wb = next_mem_wb
        self.ex_mem = next_ex_mem
        self.id_ex = next_id_ex
        self.if_id = next_if_id

        return self._capture_status(stalled, flushed, fwd_a, fwd_b)

    # -----------------------------------------------------------------
    # ALU Execution Helper
    # -----------------------------------------------------------------

    def _execute_alu(
        self,
        inst: DecodedInstruction,
        pc: int,
        rs1: int,
        rs2: int,
        imm: int,
    ) -> Tuple[int, float, bool, int]:
        """Execute operation and return (alu_int_result, fp_result, branch_taken, branch_target)."""
        name = inst.mnemonic
        alu = 0
        fp = 0.0
        taken = False
        target = pc + 4

        # ALU Immediates
        if name == "addi":
            alu = to_unsigned_64(to_signed_64(rs1) + imm)
        elif name == "slti":
            alu = 1 if to_signed_64(rs1) < imm else 0
        elif name == "sltiu":
            alu = 1 if (rs1 & 0xFFFFFFFFFFFFFFFF) < (imm & 0xFFFFFFFFFFFFFFFF) else 0
        elif name == "xori":
            alu = rs1 ^ imm
        elif name == "ori":
            alu = rs1 | imm
        elif name == "andi":
            alu = rs1 & imm
        elif name == "slli":
            alu = to_unsigned_64(rs1 << (inst.shamt & 63))
        elif name == "srli":
            alu = to_unsigned_64(rs1 >> (inst.shamt & 63))
        elif name == "srai":
            alu = to_unsigned_64(to_signed_64(rs1) >> (inst.shamt & 63))

        # 32-bit Word Immediates
        elif name == "addiw":
            alu = to_signed_64(sign_extend((to_signed_32(rs1) + imm) & 0xFFFFFFFF, 32))
        elif name == "slliw":
            alu = to_signed_64(sign_extend((to_unsigned_32(rs1) << (inst.shamt & 31)) & 0xFFFFFFFF, 32))
        elif name == "srliw":
            alu = to_signed_64(sign_extend((to_unsigned_32(rs1) >> (inst.shamt & 31)) & 0xFFFFFFFF, 32))
        elif name == "sraiw":
            alu = to_signed_64(sign_extend(to_signed_32(rs1) >> (inst.shamt & 31), 32))

        # Register-Register ALU
        elif name == "add":
            alu = to_unsigned_64(to_signed_64(rs1) + to_signed_64(rs2))
        elif name == "sub":
            alu = to_unsigned_64(to_signed_64(rs1) - to_signed_64(rs2))
        elif name == "sll":
            alu = to_unsigned_64(rs1 << (rs2 & 63))
        elif name == "slt":
            alu = 1 if to_signed_64(rs1) < to_signed_64(rs2) else 0
        elif name == "sltu":
            alu = 1 if (rs1 & 0xFFFFFFFFFFFFFFFF) < (rs2 & 0xFFFFFFFFFFFFFFFF) else 0
        elif name == "xor":
            alu = rs1 ^ rs2
        elif name == "srl":
            alu = to_unsigned_64(rs1 >> (rs2 & 63))
        elif name == "sra":
            alu = to_unsigned_64(to_signed_64(rs1) >> (rs2 & 63))
        elif name == "or":
            alu = rs1 | rs2
        elif name == "and":
            alu = rs1 & rs2

        # 32-bit Word Register-Register
        elif name == "addw":
            alu = to_signed_64(sign_extend((to_signed_32(rs1) + to_signed_32(rs2)) & 0xFFFFFFFF, 32))
        elif name == "subw":
            alu = to_signed_64(sign_extend((to_signed_32(rs1) - to_signed_32(rs2)) & 0xFFFFFFFF, 32))

        # RV64M Standard Multiply / Divide
        elif name == "mul":
            alu = to_unsigned_64(to_signed_64(rs1) * to_signed_64(rs2))
        elif name == "mulh":
            prod = to_signed_64(rs1) * to_signed_64(rs2)
            alu = to_unsigned_64(prod >> 64)
        elif name == "mulhu":
            prod = (rs1 & 0xFFFFFFFFFFFFFFFF) * (rs2 & 0xFFFFFFFFFFFFFFFF)
            alu = to_unsigned_64(prod >> 64)
        elif name == "div":
            s2 = to_signed_64(rs2)
            if s2 == 0:
                alu = to_unsigned_64(-1)
            else:
                alu = to_unsigned_64(int(to_signed_64(rs1) / s2))
        elif name == "divu":
            u2 = rs2 & 0xFFFFFFFFFFFFFFFF
            if u2 == 0:
                alu = to_unsigned_64(-1)
            else:
                alu = to_unsigned_64(int((rs1 & 0xFFFFFFFFFFFFFFFF) / u2))
        elif name == "rem":
            s2 = to_signed_64(rs2)
            if s2 == 0:
                alu = rs1
            else:
                alu = to_unsigned_64(to_signed_64(rs1) % s2)
        elif name == "remu":
            u2 = rs2 & 0xFFFFFFFFFFFFFFFF
            if u2 == 0:
                alu = rs1
            else:
                alu = to_unsigned_64((rs1 & 0xFFFFFFFFFFFFFFFF) % u2)

        # Upper Immediates & Jumps
        elif name == "lui":
            alu = imm & 0xFFFFFFFFFFFFFFFF
        elif name == "auipc":
            alu = to_unsigned_64(pc + imm)
        elif name == "jal":
            alu = to_unsigned_64(pc + 4)
            taken = True
            target = pc + imm
        elif name == "jalr":
            alu = to_unsigned_64(pc + 4)
            taken = True
            target = (to_signed_64(rs1) + imm) & ~1

        # Memory address calculation for loads and stores
        elif inst.is_load or inst.is_store:
            alu = to_unsigned_64(to_signed_64(rs1) + imm)

        # Branch evaluation
        elif inst.is_branch:
            target = pc + imm
            s1, s2 = to_signed_64(rs1), to_signed_64(rs2)
            u1, u2 = rs1 & 0xFFFFFFFFFFFFFFFF, rs2 & 0xFFFFFFFFFFFFFFFF
            if name == "beq":
                taken = (rs1 == rs2)
            elif name == "bne":
                taken = (rs1 != rs2)
            elif name == "blt":
                taken = (s1 < s2)
            elif name == "bge":
                taken = (s1 >= s2)
            elif name == "bltu":
                taken = (u1 < u2)
            elif name == "bgeu":
                taken = (u1 >= u2)

        # Floating-point arithmetic
        elif name == "fadd.d":
            fp = self.reg_file.read_f(inst.rs1) + self.reg_file.read_f(inst.rs2)
        elif name == "fsub.d":
            fp = self.reg_file.read_f(inst.rs1) - self.reg_file.read_f(inst.rs2)
        elif name == "fmul.d":
            fp = self.reg_file.read_f(inst.rs1) * self.reg_file.read_f(inst.rs2)
        elif name == "fdiv.d":
            fp = self.reg_file.read_f(inst.rs1) / self.reg_file.read_f(inst.rs2)

        return alu, fp, taken, target

    def _capture_status(self, stalled: bool, flushed: bool, fwd_a: int, fwd_b: int) -> PipelineStatus:
        ipc = self.committed_insts / self.cycle_count if self.cycle_count > 0 else 0.0
        return PipelineStatus(
            cycle=self.cycle_count,
            pc_if=self.if_id.pc if self.if_id.valid else self.pc,
            inst_if=self.if_id.inst.mnemonic if (self.if_id.valid and self.if_id.inst) else "bubble",
            inst_id=self.id_ex.inst.mnemonic if (self.id_ex.valid and self.id_ex.inst) else "bubble",
            inst_ex=self.ex_mem.inst.mnemonic if (self.ex_mem.valid and self.ex_mem.inst) else "bubble",
            inst_mem=self.mem_wb.inst.mnemonic if (self.mem_wb.valid and self.mem_wb.inst) else "bubble",
            inst_wb=self.mem_wb.inst.mnemonic if (self.mem_wb.valid and self.mem_wb.inst) else "bubble",
            stalled=stalled,
            flushed=flushed,
            forward_a=fwd_a,
            forward_b=fwd_b,
            committed_instructions=self.committed_insts,
            ipc=ipc,
        )
