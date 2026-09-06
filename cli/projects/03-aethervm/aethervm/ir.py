"""
Static Single Assignment (SSA) Intermediate Representation (IR):
Defines typed instructions, phi-nodes, Basic Blocks, and Control Flow Graphs (CFG).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set, Union


class Value:
    """An operand in SSA form: either a virtual SSA register (%v) or a literal constant ($c)."""

    def __init__(self, name: str, is_const: bool = False, const_val: Union[int, float] = 0):
        self.name = name
        self.is_const = is_const
        self.const_val = const_val

    def __repr__(self) -> str:
        if self.is_const:
            return f"${self.const_val}"
        return f"%{self.name}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Value):
            return False
        return self.name == other.name and self.is_const == other.is_const and self.const_val == other.const_val

    def __hash__(self) -> int:
        return hash((self.name, self.is_const, self.const_val))


class Opcode:
    CONST = "const"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    NEG = "neg"
    NOT = "not"
    CALL = "call"
    PHI = "phi"
    COPY = "copy"
    PRINT = "print"
    RETURN = "ret"
    JUMP = "jmp"
    BRANCH = "br"


class Instruction:
    """A 3-address instruction inside an SSA basic block."""

    def __init__(
        self,
        op: str,
        dest: Optional[Value] = None,
        args: Optional[List[Value]] = None,
        extra: Optional[any] = None,
    ):
        self.op = op
        self.dest = dest
        self.args = args or []
        # extra holds labels for jumps/branches, callee name for calls, or phi incoming mappings
        self.extra = extra

    def is_terminator(self) -> bool:
        return self.op in (Opcode.JUMP, Opcode.BRANCH, Opcode.RETURN)

    def __repr__(self) -> str:
        if self.op == Opcode.CONST:
            return f"{self.dest} = const {self.extra}"
        elif self.op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                         Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
            return f"{self.dest} = {self.op} {self.args[0]}, {self.args[1]}"
        elif self.op in (Opcode.NEG, Opcode.NOT):
            return f"{self.dest} = {self.op} {self.args[0]}"
        elif self.op == Opcode.CALL:
            arg_str = ", ".join(repr(a) for a in self.args)
            return f"{self.dest} = call {self.extra}({arg_str})"
        elif self.op == Opcode.PHI:
            # extra is list of (block_label, Value)
            pairs = [f"[{lbl}: {val}]" for lbl, val in self.extra]
            return f"{self.dest} = phi {', '.join(pairs)}"
        elif self.op == Opcode.COPY:
            return f"{self.dest} = copy {self.args[0]}"
        elif self.op == Opcode.PRINT:
            return f"print {self.args[0]}"
        elif self.op == Opcode.RETURN:
            val_str = f" {self.args[0]}" if self.args else ""
            return f"ret{val_str}"
        elif self.op == Opcode.JUMP:
            return f"jmp {self.extra}"
        elif self.op == Opcode.BRANCH:
            return f"br {self.args[0]}, {self.extra[0]}, {self.extra[1]}"
        return f"{self.op} {self.dest} {self.args}"


class BasicBlock:
    """A linear sequence of instructions with a single entry and single terminator exit."""

    def __init__(self, label: str):
        self.label = label
        self.instructions: List[Instruction] = []
        self.predecessors: List[str] = []
        self.successors: List[str] = []

    def add_instruction(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def __repr__(self) -> str:
        lines = [f"{self.label}:"]
        for inst in self.instructions:
            lines.append(f"    {inst}")
        return "\n".join(lines)


class FunctionIR:
    """SSA representation of a compiled function."""

    def __init__(self, name: str, params: List[str]):
        self.name = name
        self.params = params
        self.blocks: Dict[str, BasicBlock] = {}
        self.entry_block: str = ""

    def add_block(self, block: BasicBlock) -> None:
        self.blocks[block.label] = block
        if not self.entry_block:
            self.entry_block = block.label

    def __repr__(self) -> str:
        params_str = ", ".join(f"%{p}" for p in self.params)
        lines = [f"function {self.name}({params_str}) {{"]
        for block in self.blocks.values():
            lines.append(repr(block))
        lines.append("}")
        return "\n".join(lines)


class ProgramIR:
    """Full SSA Program consisting of multiple functions."""

    def __init__(self):
        self.functions: Dict[str, FunctionIR] = {}

    def add_function(self, fn: FunctionIR) -> None:
        self.functions[fn.name] = fn

    def __repr__(self) -> str:
        return "\n\n".join(repr(fn) for fn in self.functions.values())
