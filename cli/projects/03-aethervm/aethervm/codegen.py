"""
Register Allocation and Bytecode Emitter:
- Linear Scan Register Allocation mapping infinite SSA virtual registers
  to physical machine registers with parameter ABI preservation
- Automatic stack frame spilling when register pressure exceeds capacity
- De-SSA Phi-node resolution
- 3-Address register bytecode emission
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from aethervm.ir import Opcode, Value, Instruction, BasicBlock, FunctionIR, ProgramIR


NUM_PHYSICAL_REGISTERS = 16
SCRATCH_REG_A = 0   # Scratch register / Return register
SCRATCH_REG_B = 15  # Secondary scratch register for binary constants
RET_REG = 0         # Standard return register


@dataclass
class BytecodeInst:
    op: str
    r_dest: int = 0
    r_src1: int = 0
    r_src2: int = 0
    imm: Any = None
    target_pc: int = -1
    label: str = ""
    call_args: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.op == "LOAD_CONST":
            return f"LOAD_CONST   R{self.r_dest}, {self.imm}"
        elif self.op == "MOV":
            return f"MOV          R{self.r_dest}, R{self.r_src1}"
        elif self.op in ("ADD", "SUB", "MUL", "DIV", "MOD",
                         "CMP_EQ", "CMP_NE", "CMP_LT", "CMP_LE", "CMP_GT", "CMP_GE"):
            return f"{self.op:<12} R{self.r_dest}, R{self.r_src1}, R{self.r_src2}"
        elif self.op == "JMP":
            return f"JMP          PC#{self.target_pc} ({self.label})"
        elif self.op == "JMP_IF":
            return f"JMP_IF       R{self.r_src1}, PC#{self.target_pc} ({self.label})"
        elif self.op == "JMP_IF_NOT":
            return f"JMP_IF_NOT   R{self.r_src1}, PC#{self.target_pc} ({self.label})"
        elif self.op == "CALL":
            args_str = ", ".join(f"R{r}" for r in self.call_args)
            return f"CALL         {self.imm}({args_str}) -> R{self.r_dest}"
        elif self.op == "RET":
            return f"RET          R{self.r_src1}"
        elif self.op == "PRINT":
            return f"PRINT        R{self.r_src1}"
        elif self.op == "SPILL":
            return f"SPILL        [Stack#{self.imm}], R{self.r_src1}"
        elif self.op == "RELOAD":
            return f"RELOAD       R{self.r_dest}, [Stack#{self.imm}]"
        elif self.op == "HALT":
            return "HALT"
        return f"{self.op} {self.r_dest} {self.r_src1} {self.r_src2} {self.imm}"


class CompiledFunction:
    """Compiled function consisting of linear bytecode instructions and metadata."""

    def __init__(self, name: str, params: List[str], code: List[BytecodeInst], num_stack_slots: int):
        self.name = name
        self.params = params
        self.code = code
        self.num_stack_slots = num_stack_slots

    def __repr__(self) -> str:
        lines = [f"=== Function '{self.name}' (Stack Slots: {self.num_stack_slots}) ==="]
        for pc, inst in enumerate(self.code):
            lines.append(f"  [{pc:04d}] {inst}")
        return "\n".join(lines)


class LinearScanAllocator:
    """
    Computes live intervals and assigns physical registers (R1-R14) or stack spill slots.
    Preserves incoming function arguments in their caller-passed registers.
    """

    def __init__(self, num_regs: int = NUM_PHYSICAL_REGISTERS):
        self.num_regs = num_regs

    def allocate(
        self,
        linearized_instructions: List[Instruction],
        params: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, int], int]:
        intervals: Dict[str, Tuple[int, int]] = {}

        # 1. Initialize parameter intervals starting at PC 0
        for p in params:
            intervals[p] = (0, 0)

        for pc, inst in enumerate(linearized_instructions):
            if inst.dest and not inst.dest.is_const:
                name = inst.dest.name
                if name not in intervals:
                    intervals[name] = (pc, pc)
                else:
                    s, _ = intervals[name]
                    intervals[name] = (s, pc)

            for arg in inst.args:
                if not arg.is_const:
                    name = arg.name
                    if name not in intervals:
                        intervals[name] = (0, pc)
                    else:
                        s, old_e = intervals[name]
                        intervals[name] = (s, max(old_e, pc))

        # 2. Pre-assign parameters to R1, R2, ...
        reg_map: Dict[str, int] = {}
        spill_map: Dict[str, int] = {}
        spill_counter = 0

        active: List[Tuple[int, str, int]] = []  # (end_pc, var_name, phys_reg)
        for i, p in enumerate(params):
            reg = i + 1
            reg_map[p] = reg
            end_pc = intervals[p][1]
            active.append((end_pc, p, reg))

        # Allocatable register pool: R1 through R14 (excluding initial parameter registers)
        allocated_param_regs = set(reg_map.values())
        free_regs = [r for r in range(1, self.num_regs - 1) if r not in allocated_param_regs]

        # 3. Sort non-parameter variables by interval start
        non_params = [v for v in intervals if v not in params]
        sorted_vars = sorted(non_params, key=lambda v: intervals[v][0])

        for var in sorted_vars:
            start_pc, end_pc = intervals[var]

            # Expire old intervals
            active.sort(key=lambda x: x[0])
            still_active = []
            for a_end, a_var, a_reg in active:
                if a_end < start_pc:
                    free_regs.append(a_reg)
                else:
                    still_active.append((a_end, a_var, a_reg))
            active = still_active

            # Assign free register or spill
            if free_regs:
                phys_reg = free_regs.pop(0)
                reg_map[var] = phys_reg
                active.append((end_pc, var, phys_reg))
            else:
                # Spill candidate with furthest end point
                active.sort(key=lambda x: x[0], reverse=True)
                # Ensure we don't evict currently needed variables if avoidable
                furthest_end, furthest_var, furthest_reg = active[0]

                if furthest_end > end_pc and furthest_var not in params:
                    reg_map[var] = furthest_reg
                    spill_map[furthest_var] = spill_counter
                    spill_counter += 1
                    del reg_map[furthest_var]
                    active[0] = (end_pc, var, furthest_reg)
                else:
                    spill_map[var] = spill_counter
                    spill_counter += 1

        return reg_map, spill_map, spill_counter


class BytecodeEmitter:
    """
    Translates SSA Function IR into linear physical register bytecode.
    """

    def compile_function(self, fn: FunctionIR) -> CompiledFunction:
        # Step 1: De-SSA Phi-nodes
        for block in fn.blocks.values():
            phi_nodes = [i for i in block.instructions if i.op == Opcode.PHI]
            for phi in phi_nodes:
                for pred_label, incoming_val in phi.extra:
                    pred_block = fn.blocks[pred_label]
                    copy_inst = Instruction(Opcode.COPY, dest=phi.dest, args=[incoming_val])
                    insert_idx = len(pred_block.instructions)
                    if insert_idx > 0 and pred_block.instructions[-1].is_terminator():
                        insert_idx -= 1
                    pred_block.instructions.insert(insert_idx, copy_inst)

        for block in fn.blocks.values():
            block.instructions = [i for i in block.instructions if i.op != Opcode.PHI]

        # Step 2: Linearize basic blocks
        linear_instructions: List[Instruction] = []
        for block in fn.blocks.values():
            linear_instructions.extend(block.instructions)

        # Step 3: Register Allocation
        allocator = LinearScanAllocator(num_regs=NUM_PHYSICAL_REGISTERS)
        reg_map, spill_map, num_stack_slots = allocator.allocate(linear_instructions, fn.params)

        # Step 4: Emit Bytecode with jump target resolution
        bytecode: List[BytecodeInst] = []
        block_pc_map: Dict[str, int] = {}
        pending_jumps: List[Tuple[int, str]] = []

        def get_reg_for_read(v: Value, scratch_reg: int) -> Tuple[int, Optional[BytecodeInst]]:
            if v.is_const:
                return scratch_reg, BytecodeInst("LOAD_CONST", r_dest=scratch_reg, imm=v.const_val)
            name = v.name
            if name in reg_map:
                return reg_map[name], None
            elif name in spill_map:
                slot = spill_map[name]
                return scratch_reg, BytecodeInst("RELOAD", r_dest=scratch_reg, imm=slot)
            return scratch_reg, None

        def get_reg_for_write(v: Value, scratch_reg: int) -> Tuple[int, Optional[BytecodeInst]]:
            name = v.name
            if name in reg_map:
                return reg_map[name], None
            elif name in spill_map:
                slot = spill_map[name]
                return scratch_reg, BytecodeInst("SPILL", r_src1=scratch_reg, imm=slot)
            return scratch_reg, None

        for block in fn.blocks.values():
            block_pc_map[block.label] = len(bytecode)

            for inst in block.instructions:
                if inst.op == Opcode.CONST:
                    dest_r, spill = get_reg_for_write(inst.dest, SCRATCH_REG_A)
                    bytecode.append(BytecodeInst("LOAD_CONST", r_dest=dest_r, imm=inst.extra))
                    if spill:
                        bytecode.append(spill)

                elif inst.op == Opcode.COPY:
                    src_r, reload = get_reg_for_read(inst.args[0], SCRATCH_REG_A)
                    if reload:
                        bytecode.append(reload)
                    dest_r, spill = get_reg_for_write(inst.dest, SCRATCH_REG_B)
                    if dest_r != src_r:
                        bytecode.append(BytecodeInst("MOV", r_dest=dest_r, r_src1=src_r))
                    if spill:
                        bytecode.append(spill)

                elif inst.op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                                 Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
                    r1, reload1 = get_reg_for_read(inst.args[0], SCRATCH_REG_A)
                    if reload1:
                        bytecode.append(reload1)
                    r2, reload2 = get_reg_for_read(inst.args[1], SCRATCH_REG_B)
                    if reload2:
                        bytecode.append(reload2)

                    dest_r, spill = get_reg_for_write(inst.dest, SCRATCH_REG_A)
                    op_name = inst.op.upper()
                    if inst.op in (Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
                        op_name = f"CMP_{inst.op.upper()}"

                    bytecode.append(BytecodeInst(op_name, r_dest=dest_r, r_src1=r1, r_src2=r2))
                    if spill:
                        bytecode.append(spill)

                elif inst.op == Opcode.PRINT:
                    r, reload = get_reg_for_read(inst.args[0], SCRATCH_REG_A)
                    if reload:
                        bytecode.append(reload)
                    bytecode.append(BytecodeInst("PRINT", r_src1=r))

                elif inst.op == Opcode.RETURN:
                    if inst.args:
                        r, reload = get_reg_for_read(inst.args[0], SCRATCH_REG_A)
                        if reload:
                            bytecode.append(reload)
                        bytecode.append(BytecodeInst("RET", r_src1=r))
                    else:
                        bytecode.append(BytecodeInst("RET", r_src1=SCRATCH_REG_A))

                elif inst.op == Opcode.JUMP:
                    target_label = inst.extra
                    idx = len(bytecode)
                    bytecode.append(BytecodeInst("JMP", label=target_label))
                    pending_jumps.append((idx, target_label))

                elif inst.op == Opcode.BRANCH:
                    cond_r, reload = get_reg_for_read(inst.args[0], SCRATCH_REG_A)
                    if reload:
                        bytecode.append(reload)
                    then_lbl, else_lbl = inst.extra

                    idx1 = len(bytecode)
                    bytecode.append(BytecodeInst("JMP_IF", r_src1=cond_r, label=then_lbl))
                    pending_jumps.append((idx1, then_lbl))

                    idx2 = len(bytecode)
                    bytecode.append(BytecodeInst("JMP", label=else_lbl))
                    pending_jumps.append((idx2, else_lbl))

                elif inst.op == Opcode.CALL:
                    callee = inst.extra
                    call_arg_regs = []
                    # Evaluate each argument into an allocated register
                    for i, arg in enumerate(inst.args):
                        # Use argument's allocated register or load into scratch
                        r, reload = get_reg_for_read(arg, SCRATCH_REG_A if i % 2 == 0 else SCRATCH_REG_B)
                        if reload:
                            bytecode.append(reload)
                        call_arg_regs.append(r)

                    dest_r, spill = get_reg_for_write(inst.dest, SCRATCH_REG_A)
                    bytecode.append(BytecodeInst(
                        "CALL",
                        r_dest=dest_r,
                        imm=callee,
                        call_args=call_arg_regs
                    ))
                    if spill:
                        bytecode.append(spill)

        # Resolve jump targets
        for idx, lbl in pending_jumps:
            if lbl in block_pc_map:
                bytecode[idx].target_pc = block_pc_map[lbl]
            else:
                bytecode[idx].target_pc = len(bytecode)

        return CompiledFunction(fn.name, fn.params, bytecode, num_stack_slots)
