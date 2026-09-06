"""
AetherVM Pipeline Visualizer:
Renders step-by-step transformations across the entire compiler toolchain:
Source -> AST -> Raw SSA CFG -> Optimized SSA CFG -> Bytecode -> Execution Profiling
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer
from aethervm.codegen import BytecodeEmitter, CompiledFunction
from aethervm.vm import VirtualMachine


def render_pipeline_view(source: str, entry_fn: str = "main", args: Optional[List[any]] = None) -> str:
    lines = []
    lines.append("╔════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                        AETHERVM COMPILER & VM PIPELINE                     ║")
    lines.append("║        Pratt Parsing -> SSA Translation -> Optimizations -> Bytecode VM    ║")
    lines.append("╚════════════════════════════════════════════════════════════════════════════╝")

    # 1. Source Code
    lines.append("\n[Stage 1] High-Level Source Program:")
    lines.append("┌" + "─" * 70 + "┐")
    for src_line in source.strip().split("\n"):
        lines.append(f"│  {src_line:<68}│")
    lines.append("└" + "─" * 70 + "┘")

    # 2. Tokenize & Parse AST
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse_program()
    lines.append(f"\n[Stage 2] AST Generated: {len(ast.functions)} functions, {len(ast.top_level_stmts)} top-level statements.")

    # 3. Raw SSA IR Construction
    raw_ir = SSABuilder().build_program(ast)
    lines.append("\n[Stage 3] Raw Static Single Assignment (SSA) Control Flow Graph:")
    for fn_name, fn_ir in raw_ir.functions.items():
        lines.append(f"--- Function '{fn_name}' (Raw SSA) ---")
        for blk in fn_ir.blocks.values():
            preds = ", ".join(blk.predecessors) if blk.predecessors else "none"
            lines.append(f"  {blk.label} (preds: {preds}):")
            for inst in blk.instructions:
                lines.append(f"      {inst}")

    # 4. Optimization Pipeline
    opt_ir = SSABuilder().build_program(ast)
    Optimizer().optimize_program(opt_ir)

    lines.append("\n[Stage 4] Optimized SSA Form (Constant Folding + CSE + DCE):")
    for fn_name, fn_ir in opt_ir.functions.items():
        lines.append(f"--- Function '{fn_name}' (Optimized SSA) ---")
        for blk in fn_ir.blocks.values():
            preds = ", ".join(blk.predecessors) if blk.predecessors else "none"
            lines.append(f"  {blk.label} (preds: {preds}):")
            for inst in blk.instructions:
                lines.append(f"      {inst}")

    # 5. Register Allocation & Bytecode Emission
    emitter = BytecodeEmitter()
    compiled_funcs = [emitter.compile_function(fn) for fn in opt_ir.functions.values()]

    lines.append("\n[Stage 5] Linear 3-Address Machine Bytecode (16 Physical Registers):")
    for cf in compiled_funcs:
        lines.append(repr(cf))

    # 6. Virtual Machine Execution
    vm = VirtualMachine()
    vm.load_program(compiled_funcs)
    result = vm.run(entry_function=entry_fn, args=args)

    lines.append("\n[Stage 6] Execution Outcome & Profiler:")
    lines.append(f"  -> Returned Value: {result}")
    if vm.output_log:
        lines.append(f"  -> Printed Output: {', '.join(vm.output_log)}")
    lines.append(f"  -> Instructions Executed: {vm.instruction_count}")
    lines.append("  -> Opcode Breakdown:")
    for op, cnt in sorted(vm.opcode_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / max(vm.instruction_count, 1)) * 100.0
        bar = "■" * int(pct / 5)
        lines.append(f"     {op:<12}: {cnt:>5} ({pct:5.1f}%) {bar}")

    return "\n".join(lines)
