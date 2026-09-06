"""
SSA IR Construction from AST:
Constructs Control Flow Graphs (CFG), Basic Blocks, and introduces phi-nodes
at control flow join points using direct Braun et al. SSA construction.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set
from aethervm.ast_nodes import (
    Program, FunctionDef, Stmt, LetStmt, ExprStmt, BlockStmt,
    IfStmt, WhileStmt, ReturnStmt, PrintStmt, Expr, LiteralExpr,
    VariableExpr, BinaryExpr, UnaryExpr, CallExpr, AssignExpr
)
from aethervm.ir import (
    Opcode, Value, Instruction, BasicBlock, FunctionIR, ProgramIR
)


class SSABuilder:
    """Translates an AST Program into SSA IR."""

    def __init__(self):
        self.var_counter = 0
        self.block_counter = 0

        # Mapping: block_label -> (var_name -> Value)
        self.current_def: Dict[str, Dict[str, Value]] = {}
        # Mapping: block_label -> list of incomplete phi nodes: (var_name, phi_inst)
        self.incomplete_phis: Dict[str, List[Tuple[str, Instruction]]] = {}
        # Sealed blocks: blocks whose predecessors are all known
        self.sealed_blocks: Set[str] = set()

    def _new_val(self, prefix: str = "v") -> Value:
        self.var_counter += 1
        return Value(f"{prefix}{self.var_counter}")

    def _new_block_label(self, prefix: str = "bb") -> str:
        self.block_counter += 1
        return f"{prefix}{self.block_counter}"

    def build_program(self, ast: Program) -> ProgramIR:
        ir_prog = ProgramIR()

        for fn in ast.functions:
            fn_ir = self.build_function(fn)
            ir_prog.add_function(fn_ir)

        # Wrap top-level statements into a synthetic 'main' function if present
        if ast.top_level_stmts:
            synthetic_main = FunctionDef(name="main", params=[], body=BlockStmt(ast.top_level_stmts))
            main_ir = self.build_function(synthetic_main)
            ir_prog.add_function(main_ir)

        return ir_prog

    def build_function(self, fn: FunctionDef) -> FunctionIR:
        fn_ir = FunctionIR(name=fn.name, params=fn.params)

        entry_label = f"{fn.name}_entry"
        entry_block = BasicBlock(entry_label)
        fn_ir.add_block(entry_block)

        self.current_def[entry_label] = {}
        self.incomplete_phis[entry_label] = []

        # Ingest parameters as initial definitions in entry block
        for p in fn.params:
            p_val = Value(p)
            self.current_def[entry_label][p] = p_val

        self._seal_block(entry_label, fn_ir)

        current_block = entry_block
        current_block = self._build_statement(fn.body, current_block, fn_ir)

        # Ensure trailing return if block not terminated
        if not current_block.instructions or not current_block.instructions[-1].is_terminator():
            current_block.add_instruction(Instruction(Opcode.RETURN, None, []))

        return fn_ir

    # -------------------------------------------------------------------------
    # Braun et al. Variable Reading and Writing
    # -------------------------------------------------------------------------

    def _write_variable(self, var_name: str, block_label: str, val: Value) -> None:
        self.current_def.setdefault(block_label, {})[var_name] = val

    def _read_variable(self, var_name: str, block_label: str, fn_ir: FunctionIR) -> Value:
        if var_name in self.current_def.get(block_label, {}):
            return self.current_def[block_label][var_name]
        return self._read_variable_recursive(var_name, block_label, fn_ir)

    def _read_variable_recursive(self, var_name: str, block_label: str, fn_ir: FunctionIR) -> Value:
        val: Value
        block = fn_ir.blocks[block_label]

        if block_label not in self.sealed_blocks:
            # Incomplete phi: predecessors not all sealed yet
            val = self._new_val(f"phi_{var_name}")
            phi_inst = Instruction(Opcode.PHI, dest=val, extra=[])
            block.instructions.insert(0, phi_inst)
            self.incomplete_phis.setdefault(block_label, []).append((var_name, phi_inst))
        elif len(block.predecessors) == 1:
            val = self._read_variable(var_name, block.predecessors[0], fn_ir)
        elif len(block.predecessors) == 0:
            # Undefined variable
            val = Value(f"undef_{var_name}")
        else:
            # Block is sealed and has multiple predecessors -> create complete phi
            val = self._new_val(f"phi_{var_name}")
            phi_inst = Instruction(Opcode.PHI, dest=val, extra=[])
            block.instructions.insert(0, phi_inst)
            self._write_variable(var_name, block_label, val)
            self._add_phi_operands(var_name, phi_inst, block, fn_ir)

        self._write_variable(var_name, block_label, val)
        return val

    def _add_phi_operands(self, var_name: str, phi_inst: Instruction, block: BasicBlock, fn_ir: FunctionIR) -> None:
        incoming = []
        for pred in block.predecessors:
            op_val = self._read_variable(var_name, pred, fn_ir)
            incoming.append((pred, op_val))
        phi_inst.extra = incoming

    def _seal_block(self, block_label: str, fn_ir: FunctionIR) -> None:
        self.sealed_blocks.add(block_label)
        block = fn_ir.blocks[block_label]
        for var_name, phi_inst in self.incomplete_phis.get(block_label, []):
            self._add_phi_operands(var_name, phi_inst, block, fn_ir)

    # -------------------------------------------------------------------------
    # Statement Translation
    # -------------------------------------------------------------------------

    def _build_statement(self, stmt: Stmt, block: BasicBlock, fn_ir: FunctionIR) -> BasicBlock:
        if isinstance(stmt, BlockStmt):
            curr = block
            for s in stmt.statements:
                curr = self._build_statement(s, curr, fn_ir)
            return curr

        elif isinstance(stmt, LetStmt):
            init_val = self._build_expression(stmt.initializer, block, fn_ir)
            self._write_variable(stmt.name, block.label, init_val)
            return block

        elif isinstance(stmt, ExprStmt):
            self._build_expression(stmt.expr, block, fn_ir)
            return block

        elif isinstance(stmt, PrintStmt):
            val = self._build_expression(stmt.value, block, fn_ir)
            block.add_instruction(Instruction(Opcode.PRINT, None, [val]))
            return block

        elif isinstance(stmt, ReturnStmt):
            ret_val = self._build_expression(stmt.value, block, fn_ir) if stmt.value else None
            args = [ret_val] if ret_val else []
            block.add_instruction(Instruction(Opcode.RETURN, None, args))
            return block

        elif isinstance(stmt, IfStmt):
            cond_val = self._build_expression(stmt.condition, block, fn_ir)

            then_label = self._new_block_label("if_then")
            else_label = self._new_block_label("if_else")
            merge_label = self._new_block_label("if_merge")

            then_block = BasicBlock(then_label)
            else_block = BasicBlock(else_label)
            merge_block = BasicBlock(merge_label)

            fn_ir.add_block(then_block)
            fn_ir.add_block(else_block)
            fn_ir.add_block(merge_block)

            # Terminate current block with conditional branch
            block.add_instruction(Instruction(Opcode.BRANCH, None, [cond_val], extra=(then_label, else_label)))
            block.successors.extend([then_label, else_label])
            then_block.predecessors.append(block.label)
            else_block.predecessors.append(block.label)

            self._seal_block(then_label, fn_ir)
            self._seal_block(else_label, fn_ir)

            # Build then branch
            then_end = self._build_statement(stmt.then_branch, then_block, fn_ir)
            if not then_end.instructions or not then_end.instructions[-1].is_terminator():
                then_end.add_instruction(Instruction(Opcode.JUMP, None, extra=merge_label))
                then_end.successors.append(merge_label)
                merge_block.predecessors.append(then_end.label)

            # Build else branch
            if stmt.else_branch:
                else_end = self._build_statement(stmt.else_branch, else_block, fn_ir)
            else:
                else_end = else_block

            if not else_end.instructions or not else_end.instructions[-1].is_terminator():
                else_end.add_instruction(Instruction(Opcode.JUMP, None, extra=merge_label))
                else_end.successors.append(merge_label)
                merge_block.predecessors.append(else_end.label)

            self._seal_block(merge_label, fn_ir)
            return merge_block

        elif isinstance(stmt, WhileStmt):
            header_label = self._new_block_label("loop_header")
            body_label = self._new_block_label("loop_body")
            exit_label = self._new_block_label("loop_exit")

            header_block = BasicBlock(header_label)
            body_block = BasicBlock(body_label)
            exit_block = BasicBlock(exit_label)

            fn_ir.add_block(header_block)
            fn_ir.add_block(body_block)
            fn_ir.add_block(exit_block)

            # Jump from current block into loop header
            block.add_instruction(Instruction(Opcode.JUMP, None, extra=header_label))
            block.successors.append(header_label)
            header_block.predecessors.append(block.label)

            # Evaluate condition in header block
            cond_val = self._build_expression(stmt.condition, header_block, fn_ir)
            header_block.add_instruction(Instruction(Opcode.BRANCH, None, [cond_val], extra=(body_label, exit_label)))
            header_block.successors.extend([body_label, exit_label])
            body_block.predecessors.append(header_label)
            exit_block.predecessors.append(header_label)

            self._seal_block(body_label, fn_ir)

            # Build loop body
            body_end = self._build_statement(stmt.body, body_block, fn_ir)
            if not body_end.instructions or not body_end.instructions[-1].is_terminator():
                body_end.add_instruction(Instruction(Opcode.JUMP, None, extra=header_label))
                body_end.successors.append(header_label)
                header_block.predecessors.append(body_end.label)

            # Now all predecessors of header are known -> seal header!
            self._seal_block(header_label, fn_ir)
            self._seal_block(exit_label, fn_ir)

            return exit_block

        return block

    # -------------------------------------------------------------------------
    # Expression Translation
    # -------------------------------------------------------------------------

    def _build_expression(self, expr: Expr, block: BasicBlock, fn_ir: FunctionIR) -> Value:
        if isinstance(expr, LiteralExpr):
            val = self._new_val("c")
            block.add_instruction(Instruction(Opcode.CONST, dest=val, extra=expr.value))
            return val

        elif isinstance(expr, VariableExpr):
            return self._read_variable(expr.name, block.label, fn_ir)

        elif isinstance(expr, AssignExpr):
            val = self._build_expression(expr.value, block, fn_ir)
            self._write_variable(expr.name, block.label, val)
            return val

        elif isinstance(expr, BinaryExpr):
            left_val = self._build_expression(expr.left, block, fn_ir)
            right_val = self._build_expression(expr.right, block, fn_ir)
            dest_val = self._new_val("bin")

            op_map = {
                "+": Opcode.ADD,
                "-": Opcode.SUB,
                "*": Opcode.MUL,
                "/": Opcode.DIV,
                "%": Opcode.MOD,
                "==": Opcode.EQ,
                "!=": Opcode.NE,
                "<": Opcode.LT,
                "<=": Opcode.LE,
                ">": Opcode.GT,
                ">=": Opcode.GE,
            }
            op_code = op_map.get(expr.op, Opcode.ADD)
            block.add_instruction(Instruction(op_code, dest=dest_val, args=[left_val, right_val]))
            return dest_val

        elif isinstance(expr, UnaryExpr):
            operand_val = self._build_expression(expr.operand, block, fn_ir)
            dest_val = self._new_val("un")
            op_code = Opcode.NEG if expr.op == "-" else Opcode.NOT
            block.add_instruction(Instruction(op_code, dest=dest_val, args=[operand_val]))
            return dest_val

        elif isinstance(expr, CallExpr):
            arg_vals = [self._build_expression(a, block, fn_ir) for a in expr.args]
            dest_val = self._new_val("call")
            block.add_instruction(Instruction(Opcode.CALL, dest=dest_val, args=arg_vals, extra=expr.callee))
            return dest_val

        return self._new_val("err")
