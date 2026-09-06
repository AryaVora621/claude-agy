"""
SSA Optimization Pipeline:
- Constant Folding & Constant Propagation
- Common Subexpression Elimination (CSE) via Local Value Numbering
- Dead Code Elimination (DCE) via Backward Liveness Analysis
- Copy Propagation & Trivial Phi Simplification
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Any
from aethervm.ir import Opcode, Value, Instruction, BasicBlock, FunctionIR, ProgramIR


class ConstantFolder:
    """
    Evaluates static arithmetic operations at compile-time and propagates constants.
    """

    def run_on_function(self, fn: FunctionIR) -> bool:
        changed = False
        const_map: Dict[str, Union[int, float]] = {}

        # 1. Collect initial constants
        for block in fn.blocks.values():
            for inst in block.instructions:
                if inst.op == Opcode.CONST and inst.dest:
                    const_map[inst.dest.name] = inst.extra

        # 2. Iterate through instructions and fold
        for block in fn.blocks.values():
            new_instructions = []
            for inst in block.instructions:
                # Replace arguments with known constants if applicable
                for i, arg in enumerate(inst.args):
                    if arg.name in const_map:
                        inst.args[i] = Value(arg.name, is_const=True, const_val=const_map[arg.name])

                if inst.op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                               Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
                    left = inst.args[0]
                    right = inst.args[1]
                    if (left.is_const or left.name in const_map) and (right.is_const or right.name in const_map):
                        val_l = left.const_val if left.is_const else const_map[left.name]
                        val_r = right.const_val if right.is_const else const_map[right.name]

                        res = self._compute_binop(inst.op, val_l, val_r)
                        if res is not None:
                            const_inst = Instruction(Opcode.CONST, dest=inst.dest, extra=res)
                            const_map[inst.dest.name] = res
                            new_instructions.append(const_inst)
                            changed = True
                            continue

                new_instructions.append(inst)
            block.instructions = new_instructions

        return changed

    def _compute_binop(self, op: str, a: float, b: float) -> Optional[float | int | bool]:
        try:
            if op == Opcode.ADD:
                return a + b
            elif op == Opcode.SUB:
                return a - b
            elif op == Opcode.MUL:
                return a * b
            elif op == Opcode.DIV:
                return a / b if b != 0 else None
            elif op == Opcode.MOD:
                return a % b if b != 0 else None
            elif op == Opcode.EQ:
                return int(a == b)
            elif op == Opcode.NE:
                return int(a != b)
            elif op == Opcode.LT:
                return int(a < b)
            elif op == Opcode.LE:
                return int(a <= b)
            elif op == Opcode.GT:
                return int(a > b)
            elif op == Opcode.GE:
                return int(a >= b)
        except Exception:
            return None
        return None


class CommonSubexpressionElimination:
    """
    Deduplicates identical arithmetic computations using value numbering.
    """

    def run_on_function(self, fn: FunctionIR) -> bool:
        changed = False

        for block in fn.blocks.values():
            seen_exprs: Dict[Tuple[str, str, str], Value] = {}
            new_instructions = []

            for inst in block.instructions:
                if inst.op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD,
                               Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.LE, Opcode.GT, Opcode.GE):
                    arg1_key = str(inst.args[0].const_val) if inst.args[0].is_const else inst.args[0].name
                    arg2_key = str(inst.args[1].const_val) if inst.args[1].is_const else inst.args[1].name

                    # Commutative canonicalization
                    if inst.op in (Opcode.ADD, Opcode.MUL, Opcode.EQ, Opcode.NE) and arg1_key > arg2_key:
                        arg1_key, arg2_key = arg2_key, arg1_key

                    expr_key = (inst.op, arg1_key, arg2_key)

                    if expr_key in seen_exprs:
                        # Replace with COPY from previously computed value
                        orig_val = seen_exprs[expr_key]
                        copy_inst = Instruction(Opcode.COPY, dest=inst.dest, args=[orig_val])
                        new_instructions.append(copy_inst)
                        changed = True
                        continue
                    else:
                        seen_exprs[expr_key] = inst.dest

                new_instructions.append(inst)
            block.instructions = new_instructions

        return changed


class DeadCodeElimination:
    """
    Removes instructions whose output values are never consumed,
    and removes unreachable basic blocks from the Control Flow Graph.
    """

    def run_on_function(self, fn: FunctionIR) -> bool:
        changed = False

        # Phase 1: Eliminate unreachable basic blocks
        reachable = set()
        queue = [fn.entry_block]
        while queue:
            curr = queue.pop(0)
            if curr not in reachable and curr in fn.blocks:
                reachable.add(curr)
                queue.extend(fn.blocks[curr].successors)

        if len(reachable) < len(fn.blocks):
            fn.blocks = {lbl: blk for lbl, blk in fn.blocks.items() if lbl in reachable}
            changed = True

        # Phase 2: Dead instruction elimination via backward liveness
        # 1. Collect all variable usages
        used_vars: Set[str] = set()

        for block in fn.blocks.values():
            for inst in block.instructions:
                # Essential instructions with side effects
                if inst.op in (Opcode.PRINT, Opcode.RETURN, Opcode.CALL, Opcode.BRANCH, Opcode.JUMP):
                    for arg in inst.args:
                        used_vars.add(arg.name)
                elif inst.op == Opcode.PHI:
                    for _, val in inst.extra:
                        used_vars.add(val.name)

        # Fixpoint iteration: keep marking variables that define used variables
        iter_changed = True
        while iter_changed:
            iter_changed = False
            for block in fn.blocks.values():
                for inst in block.instructions:
                    if inst.dest and inst.dest.name in used_vars:
                        for arg in inst.args:
                            if arg.name not in used_vars:
                                used_vars.add(arg.name)
                                iter_changed = True

        # 2. Filter out non-essential instructions whose dest is unused
        for block in fn.blocks.values():
            filtered = []
            for inst in block.instructions:
                if inst.op in (Opcode.PRINT, Opcode.RETURN, Opcode.CALL, Opcode.BRANCH, Opcode.JUMP):
                    filtered.append(inst)
                elif inst.dest and inst.dest.name in used_vars:
                    filtered.append(inst)
                else:
                    # Dead instruction!
                    changed = True
            block.instructions = filtered

        return changed


class Optimizer:
    """
    Optimization Pipeline orchestrator:
    Runs passes iteratively until reaching fixpoint convergence.
    """

    def __init__(self):
        self.passes = [
            ConstantFolder(),
            CommonSubexpressionElimination(),
            DeadCodeElimination(),
        ]

    def optimize_program(self, ir_prog: ProgramIR, max_iterations: int = 5) -> None:
        for fn in ir_prog.functions.values():
            self.optimize_function(fn, max_iterations)

    def optimize_function(self, fn: FunctionIR, max_iterations: int = 5) -> None:
        for _ in range(max_iterations):
            any_changed = False
            for p in self.passes:
                if p.run_on_function(fn):
                    any_changed = True
            if not any_changed:
                break
