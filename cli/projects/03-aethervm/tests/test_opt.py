"""
Unit tests for SSA Optimization Passes: Constant Folding, CSE, and DCE.
"""

import unittest
from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer, ConstantFolder, CommonSubexpressionElimination, DeadCodeElimination
from aethervm.ir import Opcode


class TestSSAOptimizations(unittest.TestCase):

    def test_constant_folding(self):
        code = """
        fn fold() {
            let a = 10;
            let b = 20;
            let c = a + b * 2;
            return c;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)
        fn_ir = ir_prog.functions["fold"]

        # Run constant folding pass
        ConstantFolder().run_on_function(fn_ir)

        entry = fn_ir.blocks["fold_entry"]
        # Find constant instruction defining c
        const_vals = [inst.extra for inst in entry.instructions if inst.op == Opcode.CONST]
        # Should fold 10 + 20 * 2 = 50
        self.assertIn(50, const_vals)

    def test_common_subexpression_elimination(self):
        code = """
        fn cse(x, y) {
            let a = x * y;
            let b = x * y;
            return a + b;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)
        fn_ir = ir_prog.functions["cse"]

        # Before CSE: two MUL instructions
        mul_count_before = sum(1 for blk in fn_ir.blocks.values() for i in blk.instructions if i.op == Opcode.MUL)
        self.assertEqual(mul_count_before, 2)

        CommonSubexpressionElimination().run_on_function(fn_ir)

        # After CSE: only one MUL instruction, second replaced with COPY
        mul_count_after = sum(1 for blk in fn_ir.blocks.values() for i in blk.instructions if i.op == Opcode.MUL)
        self.assertEqual(mul_count_after, 1)

    def test_dead_code_elimination(self):
        code = """
        fn dce(x) {
            let dead_val = x * 9999;
            let live_val = x + 1;
            return live_val;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)
        fn_ir = ir_prog.functions["dce"]

        DeadCodeElimination().run_on_function(fn_ir)

        # MUL instruction defining dead_val should be eliminated
        has_mul = any(i.op == Opcode.MUL for blk in fn_ir.blocks.values() for i in blk.instructions)
        self.assertFalse(has_mul, "Dead multiplication was not eliminated by DCE")


if __name__ == "__main__":
    unittest.main()
