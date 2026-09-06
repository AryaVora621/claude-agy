"""
WasmCore: Terminal Visualizer, Disassembler, and Interactive Debugger.
Provides rich ANSI disassembly, call frame tracing, operand stack inspection,
linear memory hexdumps, and execution timeline diagnostics.
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Callable
from .types import ValType, FuncType, Value, ExportDesc, SectionId
from .opcodes import Opcode, Instruction, MemArg
from .module import WasmModule, FunctionDef
from .memory import LinearMemory
from .instance import WasmInstance, HostFunction
from .interpreter import WasmInterpreter, CallFrame, ControlFrame

# Terminal styling ANSI codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[37m"
BG_DARK = "\033[48;5;236m"


def format_instruction(instr: Instruction, indent: int = 0) -> str:
    """Format a single instruction into readable WAT-like syntax."""
    prefix = "  " * indent
    op_name = instr.opcode.name.lower().replace("_", ".")
    operands = instr.operands

    if operands is None:
        return f"{prefix}{CYAN}{op_name}{RESET}"

    if isinstance(operands, MemArg):
        op_str = f"offset={operands.offset} align={2**operands.align}"
    elif isinstance(operands, tuple) and len(operands) == 2 and isinstance(operands[0], list):
        targets, default = operands
        targets_str = " ".join(str(t) for t in targets)
        op_str = f"[{targets_str}] default={default}"
    elif isinstance(operands, ValType):
        op_str = f"(result {operands})"
    else:
        op_str = str(operands)

    return f"{prefix}{CYAN}{op_name}{RESET} {YELLOW}{op_str}{RESET}"


class WasmDisassembler:
    """Disassembles WASM bytecode into formatted text representation."""

    @staticmethod
    def disassemble_function(
        func: FunctionDef,
        func_idx: int,
        module: Optional[WasmModule] = None,
        name: Optional[str] = None
    ) -> str:
        lines = []
        name_str = f" ${name}" if name else f" (;{func_idx};)"
        type_sig = f" (type {func_idx})"
        if module and func.type_idx < len(module.types):
            ft = module.types[func.type_idx]
            p_str = " ".join(f"(param {p})" for p in ft.params)
            r_str = " ".join(f"(result {r})" for r in ft.results)
            type_sig = f" {p_str} {r_str}".strip()

        lines.append(f"{BOLD}{BLUE}(func{name_str} {type_sig}{RESET}")

        # Locals
        for count, val_type in func.locals:
            lines.append(f"  {MAGENTA}(local {' '.join([str(val_type)] * count)}){RESET}")

        # Instructions with dynamic indentation
        indent = 1
        for instr in func.instructions:
            if instr.opcode in (Opcode.END, Opcode.ELSE):
                indent = max(1, indent - 1)

            lines.append(format_instruction(instr, indent))

            if instr.opcode in (Opcode.BLOCK, Opcode.LOOP, Opcode.IF, Opcode.ELSE):
                indent += 1

        lines.append(f"{BOLD}{BLUE}){RESET}")
        return "\n".join(lines)

    @staticmethod
    def disassemble_module(module: WasmModule) -> str:
        """Disassemble entire module header, sections, and functions."""
        out = [f"{BOLD}{WHITE}=== WasmCore Module Disassembly ==={RESET}"]

        # Types
        out.append(f"\n{BOLD}{CYAN}Types ({len(module.types)}):{RESET}")
        for idx, ft in enumerate(module.types):
            out.append(f"  [{idx}] {ft}")

        # Imports
        if module.imports:
            out.append(f"\n{BOLD}{CYAN}Imports ({len(module.imports)}):{RESET}")
            for imp in module.imports:
                out.append(f"  import \"{imp.module}\" \"{imp.name}\" ({imp.desc_type.name})")

        # Exports
        if module.exports:
            out.append(f"\n{BOLD}{CYAN}Exports ({len(module.exports)}):{RESET}")
            for exp in module.exports:
                out.append(f"  export \"{exp.name}\" -> {exp.desc_type.name} [{exp.index}]")

        # Functions
        out.append(f"\n{BOLD}{CYAN}Functions ({len(module.functions)}):{RESET}")
        for idx, fn in enumerate(module.functions):
            # Check export name
            exp_names = [e.name for e in module.exports if e.desc_type == ExportDesc.FUNC and e.index == idx]
            name = exp_names[0] if exp_names else None
            out.append(WasmDisassembler.disassemble_function(fn, idx, module, name))

        return "\n".join(out)


class MemoryViewer:
    """Formats linear memory into canonical hex and ASCII dump."""

    @staticmethod
    def hexdump(memory: LinearMemory, offset: int = 0, length: int = 128) -> str:
        total_len = len(memory.data)
        start = max(0, min(offset, total_len))
        end = min(start + length, total_len)

        lines = [f"{BOLD}{WHITE}Offset (h)  00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  ASCII{RESET}"]
        lines.append("-" * 75)

        for row_addr in range(start & ~0x0F, end, 16):
            hex_part1 = []
            hex_part2 = []
            ascii_part = []

            for col in range(16):
                curr = row_addr + col
                if curr < start or curr >= end:
                    hex_str = "  "
                    ch = " "
                else:
                    byte_val = memory.data[curr]
                    hex_str = f"{byte_val:02X}"
                    ch = chr(byte_val) if 32 <= byte_val <= 126 else "."

                if col < 8:
                    hex_part1.append(hex_str)
                else:
                    hex_part2.append(hex_str)
                ascii_part.append(ch)

            line = f"{row_addr:08X}    {' '.join(hex_part1)}  {' '.join(hex_part2)}  |{''.join(ascii_part)}|"
            lines.append(line)

        return "\n".join(lines)


class TraceStep:
    """Record of a single instruction execution step."""
    __slots__ = ("ip", "func_idx", "opcode", "operands", "stack_before", "stack_after")

    def __init__(
        self,
        ip: int,
        func_idx: int,
        opcode: Opcode,
        operands: Any,
        stack_before: List[str],
        stack_after: List[str]
    ) -> None:
        self.ip = ip
        self.func_idx = func_idx
        self.opcode = opcode
        self.operands = operands
        self.stack_before = stack_before
        self.stack_after = stack_after


class WasmDebugger:
    """
    Step-by-step Interactive WASM Debugger and Execution Tracer.
    Allows inspection of stack frames, memory, breakpoints, and single-stepping.
    """

    def __init__(self, instance: WasmInstance) -> None:
        self.instance = instance
        self.interpreter = WasmInterpreter(instance)
        self.breakpoints: set = set()
        self.history: List[TraceStep] = []

    def set_breakpoint(self, func_name_or_idx: Union[str, int], instruction_ip: int) -> None:
        """Set a breakpoint at instruction_ip within a function."""
        if isinstance(func_name_or_idx, str):
            func_idx = self.instance.get_export(func_name_or_idx)
        else:
            func_idx = func_name_or_idx
        self.breakpoints.add((func_idx, instruction_ip))

    def clear_breakpoints(self) -> None:
        self.breakpoints.clear()

    def snapshot_state(self) -> str:
        """Render a full snapshot of current interpreter state."""
        out = [f"{BOLD}{WHITE}=== WasmCore Debugger Snapshot ==={RESET}"]

        # 1. Call Stack
        out.append(f"\n{BOLD}{CYAN}Call Stack Depth: {len(self.interpreter.call_stack)}{RESET}")
        for depth, frame in enumerate(reversed(self.interpreter.call_stack)):
            out.append(f"  #{depth}: func [{frame.func_idx}] at ip={frame.ip}")
            locals_repr = ", ".join(f"l{i}:{v}" for i, v in enumerate(frame.locals))
            out.append(f"      Locals: [{locals_repr}]")

        # 2. Operand Stack
        out.append(f"\n{BOLD}{CYAN}Operand Stack ({len(self.interpreter.stack)} items):{RESET}")
        if self.interpreter.stack:
            stack_items = " | ".join(str(v) for v in self.interpreter.stack[-8:])
            out.append(f"  [top ->] {stack_items}")
        else:
            out.append("  (empty)")

        # 3. Control Stack
        out.append(f"\n{BOLD}{CYAN}Control Stack ({len(self.interpreter.control_stack)} scopes):{RESET}")
        for ctrl in self.interpreter.control_stack:
            out.append(f"  {ctrl.opcode.name} (start={ctrl.start_ip}, end={ctrl.end_ip}, stack_h={ctrl.stack_height})")

        # 4. Memory Overview
        if self.instance.memories:
            mem = self.instance.memories[0]
            out.append(f"\n{BOLD}{CYAN}Linear Memory: {mem.pages} page(s) ({len(mem.data)} bytes){RESET}")

        return "\n".join(out)
