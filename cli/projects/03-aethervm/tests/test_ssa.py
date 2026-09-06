"""
Unit tests for SSA Intermediate Representation and CFG construction.
"""

import unittest
from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.ir import Opcode


class TestSSAConstruction(unittest.TestCase):

    def test_basic_ssa_generation(self):
        code = """
        fn compute(a, b) {
            let x = a + b;
            let y = x * 2;
            return y;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)

        self.assertIn("compute", ir_prog.functions)
        fn_ir = ir_prog.functions["compute"]
        self.assertEqual(len(fn_ir.blocks), 1)

        entry_block = fn_ir.blocks["compute_entry"]
        opcodes = [inst.op for inst in entry_block.instructions]
        self.assertIn(Opcode.ADD, opcodes)
        self.assertIn(Opcode.MUL, opcodes)
        self.assertIn(Opcode.RETURN, opcodes)

    def test_phi_node_in_if_else(self):
        code = """
        fn max(a, b) {
            let res = 0;
            if (a > b) {
                res = a;
            } else {
                res = b;
            }
            return res;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)

        fn_ir = ir_prog.functions["max"]
        # Merge block must contain a PHI node for res
        has_phi = False
        for blk in fn_ir.blocks.values():
            for inst in blk.instructions:
                if inst.op == Opcode.PHI:
                    has_phi = True
                    self.assertEqual(len(inst.extra), 2)  # Two incoming edges
        self.assertTrue(has_phi, "Expected PHI node at if-else merge point")

    def test_phi_node_in_while_loop(self):
        code = """
        fn count_down(n) {
            let x = n;
            while (x > 0) {
                x = x - 1;
            }
            return x;
        }
        """
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).parse_program()
        ir_prog = SSABuilder().build_program(ast)

        fn_ir = ir_prog.functions["count_down"]
        has_loop_phi = False
        for blk in fn_ir.blocks.values():
            for inst in blk.instructions:
                if inst.op == Opcode.PHI:
                    has_loop_phi = True
        self.assertTrue(has_loop_phi, "Expected PHI node at while loop header")


if __name__ == "__main__":
    unittest.main()
