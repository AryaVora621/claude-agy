"""
AetherVM Virtual Machine:
High-throughput 3-address register bytecode virtual machine.
Features:
- 16 physical machine registers per frame
- Call frames with activation records and spilled stack slots
- Instruction profiler & step tracer
- Support for recursion, loops, conditionals, and arithmetic
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
from aethervm.codegen import BytecodeInst, CompiledFunction, NUM_PHYSICAL_REGISTERS, RET_REG


class CallFrame:
    """An execution activation record on the VM call stack."""

    def __init__(
        self,
        func: CompiledFunction,
        return_pc: int = -1,
        dest_reg: int = 0,
        caller_regs: Optional[List[Any]] = None,
    ):
        self.func = func
        self.pc = 0
        self.return_pc = return_pc
        self.dest_reg = dest_reg
        self.caller_regs = caller_regs
        self.registers: List[Any] = [0] * NUM_PHYSICAL_REGISTERS
        self.stack_slots: List[Any] = [0] * func.num_stack_slots


class VirtualMachine:
    """Register-based Bytecode Virtual Machine."""

    def __init__(self, max_instructions: int = 10_000_000, max_call_depth: int = 1000):
        self.functions: Dict[str, CompiledFunction] = {}
        self.max_instructions = max_instructions
        self.max_call_depth = max_call_depth
        self.instruction_count = 0
        self.output_log: List[str] = []

        # Profiler: opcode -> execution count
        self.opcode_counts: Dict[str, int] = {}

    def load_program(self, compiled_functions: List[CompiledFunction]) -> None:
        for fn in compiled_functions:
            self.functions[fn.name] = fn

    def run(
        self,
        entry_function: str = "main",
        args: Optional[List[Any]] = None,
        trace_hook: Optional[Callable[[CallFrame, BytecodeInst], None]] = None
    ) -> Any:
        """Executes bytecode starting at entry_function."""
        if entry_function not in self.functions:
            raise RuntimeError(f"Entry function '{entry_function}' not found")

        fn = self.functions[entry_function]
        top_frame = CallFrame(func=fn)

        # Populate initial arguments into R1, R2, ...
        if args:
            for i, val in enumerate(args):
                if i + 1 < NUM_PHYSICAL_REGISTERS:
                    top_frame.registers[i + 1] = val

        call_stack: List[CallFrame] = [top_frame]
        self.instruction_count = 0
        self.output_log = []

        while call_stack:
            frame = call_stack[-1]

            if frame.pc >= len(frame.func.code):
                # Implicit return
                res = frame.registers[RET_REG]
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1]
                    parent.registers[frame.dest_reg] = res
                    parent.pc = frame.return_pc
                continue

            inst = frame.func.code[frame.pc]
            frame.pc += 1

            self.instruction_count += 1
            if self.instruction_count > self.max_instructions:
                raise TimeoutError(f"Exceeded maximum instruction count limit ({self.max_instructions})")

            self.opcode_counts[inst.op] = self.opcode_counts.get(inst.op, 0) + 1

            if trace_hook:
                trace_hook(frame, inst)

            # Opcode execution loop
            op = inst.op

            if op == "LOAD_CONST":
                frame.registers[inst.r_dest] = inst.imm

            elif op == "MOV":
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1]

            elif op == "ADD":
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1] + frame.registers[inst.r_src2]

            elif op == "SUB":
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1] - frame.registers[inst.r_src2]

            elif op == "MUL":
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1] * frame.registers[inst.r_src2]

            elif op == "DIV":
                denom = frame.registers[inst.r_src2]
                if denom == 0:
                    raise ZeroDivisionError("Division by zero in VM")
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1] / denom

            elif op == "MOD":
                frame.registers[inst.r_dest] = frame.registers[inst.r_src1] % frame.registers[inst.r_src2]

            elif op == "CMP_EQ":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] == frame.registers[inst.r_src2])

            elif op == "CMP_NE":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] != frame.registers[inst.r_src2])

            elif op == "CMP_LT":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] < frame.registers[inst.r_src2])

            elif op == "CMP_LE":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] <= frame.registers[inst.r_src2])

            elif op == "CMP_GT":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] > frame.registers[inst.r_src2])

            elif op == "CMP_GE":
                frame.registers[inst.r_dest] = int(frame.registers[inst.r_src1] >= frame.registers[inst.r_src2])

            elif op == "JMP":
                frame.pc = inst.target_pc

            elif op == "JMP_IF":
                if bool(frame.registers[inst.r_src1]):
                    frame.pc = inst.target_pc

            elif op == "JMP_IF_NOT":
                if not bool(frame.registers[inst.r_src1]):
                    frame.pc = inst.target_pc

            elif op == "PRINT":
                val = frame.registers[inst.r_src1]
                self.output_log.append(str(val))

            elif op == "SPILL":
                slot_idx = inst.imm
                frame.stack_slots[slot_idx] = frame.registers[inst.r_src1]

            elif op == "RELOAD":
                slot_idx = inst.imm
                frame.registers[inst.r_dest] = frame.stack_slots[slot_idx]

            elif op == "CALL":
                callee_name = inst.imm
                if callee_name not in self.functions:
                    raise RuntimeError(f"Undefined function '{callee_name}' called")

                if len(call_stack) >= self.max_call_depth:
                    raise RecursionError(f"Call stack depth exceeded limit of {self.max_call_depth}")

                callee_fn = self.functions[callee_name]
                new_frame = CallFrame(
                    func=callee_fn,
                    return_pc=frame.pc,
                    dest_reg=inst.r_dest,
                    caller_regs=list(frame.registers)
                )

                # Transfer argument registers into callee's parameter registers (R1, R2, ...)
                for i, arg_reg in enumerate(inst.call_args):
                    if i + 1 < NUM_PHYSICAL_REGISTERS:
                        new_frame.registers[i + 1] = frame.registers[arg_reg]

                call_stack.append(new_frame)

            elif op == "RET":
                ret_val = frame.registers[inst.r_src1]
                call_stack.pop()
                if not call_stack:
                    return ret_val
                parent = call_stack[-1]
                parent.registers[frame.dest_reg] = ret_val
                parent.pc = frame.return_pc

            elif op == "HALT":
                return frame.registers[RET_REG]

        return None
