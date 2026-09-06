"""
Pratt Parser (Top-Down Operator Precedence):
Parses expressions with distinct binding power and parses structured statements,
control-flow blocks, and function declarations.
"""

from __future__ import annotations
from typing import List, Optional
from aethervm.lexer import Token, TokenType
from aethervm.ast_nodes import (
    Expr, LiteralExpr, VariableExpr, BinaryExpr, UnaryExpr, CallExpr, AssignExpr,
    Stmt, LetStmt, ExprStmt, BlockStmt, IfStmt, WhileStmt, ReturnStmt, PrintStmt,
    FunctionDef, Program
)

# Precedence levels (lowest to highest)
PREC_NONE = 0
PREC_ASSIGN = 1
PREC_OR = 2
PREC_AND = 3
PREC_EQUALITY = 4
PREC_COMPARISON = 5
PREC_TERM = 6
PREC_FACTOR = 7
PREC_UNARY = 8
PREC_CALL = 9
PREC_PRIMARY = 10

TOKEN_PRECEDENCE = {
    TokenType.EQUAL: PREC_ASSIGN,
    TokenType.OR_OR: PREC_OR,
    TokenType.AND_AND: PREC_AND,
    TokenType.EQ_EQ: PREC_EQUALITY,
    TokenType.BANG_EQ: PREC_EQUALITY,
    TokenType.LESS: PREC_COMPARISON,
    TokenType.LESS_EQ: PREC_COMPARISON,
    TokenType.GREATER: PREC_COMPARISON,
    TokenType.GREATER_EQ: PREC_COMPARISON,
    TokenType.PLUS: PREC_TERM,
    TokenType.MINUS: PREC_TERM,
    TokenType.STAR: PREC_FACTOR,
    TokenType.SLASH: PREC_FACTOR,
    TokenType.PERCENT: PREC_FACTOR,
    TokenType.LPAREN: PREC_CALL,
}


class Parser:
    """Parses token stream into an Abstract Syntax Tree."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _check(self, tok_type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == tok_type

    def _match(self, *types: TokenType) -> bool:
        for t in types:
            if self._check(t):
                self._advance()
                return True
        return False

    def _consume(self, tok_type: TokenType, message: str) -> Token:
        if self._check(tok_type):
            return self._advance()
        tok = self._peek()
        raise SyntaxError(f"{message} at L{tok.line}:C{tok.column}, found '{tok.value}'")

    # -------------------------------------------------------------------------
    # Pratt Expression Parsing
    # -------------------------------------------------------------------------

    def parse_expression(self, precedence: int = PREC_NONE) -> Expr:
        """Parses expression with given minimum precedence."""
        tok = self._advance()

        # Prefix rule
        left: Expr
        if tok.type == TokenType.NUMBER:
            left = LiteralExpr(tok.value)
        elif tok.type == TokenType.IDENTIFIER:
            # Check if this is a function call
            if self._match(TokenType.LPAREN):
                args = []
                if not self._check(TokenType.RPAREN):
                    while True:
                        args.append(self.parse_expression(PREC_NONE))
                        if not self._match(TokenType.COMMA):
                            break
                self._consume(TokenType.RPAREN, "Expected ')' after function arguments")
                left = CallExpr(callee=tok.value, args=args)
            else:
                left = VariableExpr(tok.value)
        elif tok.type in (TokenType.MINUS, TokenType.BANG_EQ):
            # Unary operator
            op_str = "-" if tok.type == TokenType.MINUS else "!"
            operand = self.parse_expression(PREC_UNARY)
            left = UnaryExpr(op=op_str, operand=operand)
        elif tok.type == TokenType.LPAREN:
            left = self.parse_expression(PREC_NONE)
            self._consume(TokenType.RPAREN, "Expected ')' after grouped expression")
        else:
            raise SyntaxError(f"Unexpected token '{tok.value}' in expression at L{tok.line}:C{tok.column}")

        # Infix rule
        while precedence < TOKEN_PRECEDENCE.get(self._peek().type, PREC_NONE):
            next_tok = self._advance()

            # Assignment: x = expr
            if next_tok.type == TokenType.EQUAL:
                if not isinstance(left, VariableExpr):
                    raise SyntaxError(f"Invalid assignment target at L{next_tok.line}")
                val = self.parse_expression(PREC_ASSIGN)
                left = AssignExpr(name=left.name, value=val)
                continue

            # Binary operator
            op_str = str(next_tok.value)
            next_prec = TOKEN_PRECEDENCE[next_tok.type]
            right = self.parse_expression(next_prec)
            left = BinaryExpr(op=op_str, left=left, right=right)

        return left

    # -------------------------------------------------------------------------
    # Statement Parsing
    # -------------------------------------------------------------------------

    def parse_statement(self) -> Stmt:
        if self._match(TokenType.LET):
            return self._let_statement()
        if self._match(TokenType.IF):
            return self._if_statement()
        if self._match(TokenType.WHILE):
            return self._while_statement()
        if self._match(TokenType.RETURN):
            return self._return_statement()
        if self._match(TokenType.PRINT):
            return self._print_statement()
        if self._match(TokenType.LBRACE):
            return self._block_statement()

        return self._expression_statement()

    def _let_statement(self) -> LetStmt:
        name_tok = self._consume(TokenType.IDENTIFIER, "Expected variable name after 'let'")
        self._consume(TokenType.EQUAL, "Expected '=' after variable name")
        init_expr = self.parse_expression(PREC_NONE)
        self._consume(TokenType.SEMICOLON, "Expected ';' after let statement")
        return LetStmt(name=name_tok.value, initializer=init_expr)

    def _if_statement(self) -> IfStmt:
        self._consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.parse_expression(PREC_NONE)
        self._consume(TokenType.RPAREN, "Expected ')' after if condition")

        then_branch = self.parse_statement()
        else_branch = None
        if self._match(TokenType.ELSE):
            else_branch = self.parse_statement()

        return IfStmt(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def _while_statement(self) -> WhileStmt:
        self._consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self.parse_expression(PREC_NONE)
        self._consume(TokenType.RPAREN, "Expected ')' after while condition")
        body = self.parse_statement()
        return WhileStmt(condition=condition, body=body)

    def _return_statement(self) -> ReturnStmt:
        val = None
        if not self._check(TokenType.SEMICOLON):
            val = self.parse_expression(PREC_NONE)
        self._consume(TokenType.SEMICOLON, "Expected ';' after return statement")
        return ReturnStmt(value=val)

    def _print_statement(self) -> PrintStmt:
        self._consume(TokenType.LPAREN, "Expected '(' after 'print'")
        val = self.parse_expression(PREC_NONE)
        self._consume(TokenType.RPAREN, "Expected ')' after print argument")
        self._consume(TokenType.SEMICOLON, "Expected ';' after print statement")
        return PrintStmt(value=val)

    def _block_statement(self) -> BlockStmt:
        stmts = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            stmts.append(self.parse_statement())
        self._consume(TokenType.RBRACE, "Expected '}' after block")
        return BlockStmt(statements=stmts)

    def _expression_statement(self) -> ExprStmt:
        expr = self.parse_expression(PREC_NONE)
        self._consume(TokenType.SEMICOLON, "Expected ';' after expression")
        return ExprStmt(expr=expr)

    # -------------------------------------------------------------------------
    # Program / Top-Level Parsing
    # -------------------------------------------------------------------------

    def parse_program(self) -> Program:
        functions = []
        top_stmts = []

        while not self._is_at_end():
            if self._match(TokenType.FN):
                # Parse function declaration
                fn_name = self._consume(TokenType.IDENTIFIER, "Expected function name").value
                self._consume(TokenType.LPAREN, "Expected '(' after function name")
                params = []
                if not self._check(TokenType.RPAREN):
                    while True:
                        param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name").value
                        params.append(param_name)
                        if not self._match(TokenType.COMMA):
                            break
                self._consume(TokenType.RPAREN, "Expected ')' after parameters")
                self._consume(TokenType.LBRACE, "Expected '{' before function body")
                body = self._block_statement()
                functions.append(FunctionDef(name=fn_name, params=params, body=body))
            else:
                top_stmts.append(self.parse_statement())

        return Program(functions=functions, top_level_stmts=top_stmts)
