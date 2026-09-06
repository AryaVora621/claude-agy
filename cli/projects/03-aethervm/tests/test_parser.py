"""
Unit tests for Lexer and Pratt Parser in AetherVM.
Verifies operator precedence, AST construction, control flow, and function definitions.
"""

import unittest
from aethervm.lexer import Lexer, TokenType
from aethervm.parser import Parser
from aethervm.ast_nodes import BinaryExpr, LiteralExpr, VariableExpr, IfStmt, WhileStmt, CallExpr


class TestLexerAndParser(unittest.TestCase):

    def test_lexer_tokens(self):
        code = "fn add(a, b) { let sum = a + b * 2; return sum; }"
        lexer = Lexer(code)
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        self.assertEqual(types[0], TokenType.FN)
        self.assertEqual(types[1], TokenType.IDENTIFIER)
        self.assertEqual(tokens[1].value, "add")
        self.assertEqual(types[-1], TokenType.EOF)

    def test_pratt_precedence_multiplication_over_addition(self):
        # 2 + 3 * 4 should parse as 2 + (3 * 4)
        code = "let x = 2 + 3 * 4;"
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse_program()

        let_stmt = prog.top_level_stmts[0]
        expr = let_stmt.initializer
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.left.value, 2)
        self.assertIsInstance(expr.right, BinaryExpr)
        self.assertEqual(expr.right.op, "*")
        self.assertEqual(expr.right.left.value, 3)
        self.assertEqual(expr.right.right.value, 4)

    def test_parentheses_grouping(self):
        # (2 + 3) * 4 should parse as ((2 + 3) * 4)
        code = "let x = (2 + 3) * 4;"
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse_program()

        let_stmt = prog.top_level_stmts[0]
        expr = let_stmt.initializer
        self.assertEqual(expr.op, "*")
        self.assertEqual(expr.left.op, "+")

    def test_function_and_control_flow_parsing(self):
        code = """
        fn fib(n) {
            if (n <= 1) {
                return n;
            } else {
                return fib(n - 1) + fib(n - 2);
            }
        }
        """
        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse_program()

        self.assertEqual(len(prog.functions), 1)
        fn = prog.functions[0]
        self.assertEqual(fn.name, "fib")
        self.assertEqual(fn.params, ["n"])
        self.assertEqual(len(fn.body.statements), 1)
        self.assertIsInstance(fn.body.statements[0], IfStmt)


if __name__ == "__main__":
    unittest.main()
