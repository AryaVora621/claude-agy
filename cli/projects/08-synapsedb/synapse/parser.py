"""
SynapseDB: SQL Lexer and Recursive Descent Parser.
Parses standard analytical SQL queries into an Abstract Syntax Tree (AST):
SELECT [columns / aggregates]
FROM [table]
[JOIN other_table ON condition]
[WHERE conditions]
[GROUP BY columns]
[HAVING aggregate_conditions]
[ORDER BY columns ASC/DESC]
[LIMIT n]
"""

import re
from typing import List, Optional, Any, Union, Tuple


class TokenType:
    KEYWORD = "KEYWORD"
    IDENT = "IDENT"
    NUMBER = "NUMBER"
    STRING = "STRING"
    OPERATOR = "OPERATOR"
    PUNCT = "PUNCT"
    EOF = "EOF"


class Token:
    __slots__ = ("type", "value", "pos")

    def __init__(self, type_: str, value: Any, pos: int):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "ON",
    "GROUP", "BY", "HAVING", "ORDER", "ASC", "DESC", "LIMIT",
    "AND", "OR", "NOT", "IS", "NULL", "AS",
    "SUM", "COUNT", "AVG", "MIN", "MAX"
}


class Lexer:
    """Tokenizes SQL query strings into a stream of typed tokens."""

    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.length = len(sql)

    def tokenize(self) -> List[Token]:
        tokens = []
        while self.pos < self.length:
            ch = self.sql[self.pos]

            if ch.isspace():
                self.pos += 1
                continue

            # String literals
            if ch in ("'", '"'):
                start_pos = self.pos
                quote_char = ch
                self.pos += 1
                val_chars = []
                while self.pos < self.length and self.sql[self.pos] != quote_char:
                    val_chars.append(self.sql[self.pos])
                    self.pos += 1
                if self.pos < self.length:
                    self.pos += 1  # Skip closing quote
                tokens.append(Token(TokenType.STRING, "".join(val_chars), start_pos))
                continue

            # Numbers (integers and floats)
            if ch.isdigit() or (ch == "." and self.pos + 1 < self.length and self.sql[self.pos + 1].isdigit()):
                start_pos = self.pos
                num_chars = []
                has_dot = False
                while self.pos < self.length and (self.sql[self.pos].isdigit() or self.sql[self.pos] == "."):
                    if self.sql[self.pos] == ".":
                        if has_dot:
                            break
                        has_dot = True
                    num_chars.append(self.sql[self.pos])
                    self.pos += 1
                num_str = "".join(num_chars)
                num_val = float(num_str) if has_dot else int(num_str)
                tokens.append(Token(TokenType.NUMBER, num_val, start_pos))
                continue

            # Multi-char operators: !=, <>, <=, >=
            if self.pos + 1 < self.length:
                two_ch = self.sql[self.pos : self.pos + 2]
                if two_ch in ("!=", "<>", "<=", ">="):
                    tokens.append(Token(TokenType.OPERATOR, "!=" if two_ch == "<>" else two_ch, self.pos))
                    self.pos += 2
                    continue

            # Single-char operators
            if ch in ("=", "<", ">", "+", "-", "*", "/"):
                tokens.append(Token(TokenType.OPERATOR, ch, self.pos))
                self.pos += 1
                continue

            # Punctuation: commas, parentheses, semicolons
            if ch in (",", "(", ")", ";", "."):
                tokens.append(Token(TokenType.PUNCT, ch, self.pos))
                self.pos += 1
                continue

            # Identifiers and Keywords
            if ch.isalpha() or ch == "_":
                start_pos = self.pos
                ident_chars = []
                while self.pos < self.length and (self.sql[self.pos].isalnum() or self.sql[self.pos] == "_"):
                    ident_chars.append(self.sql[self.pos])
                    self.pos += 1
                ident_str = "".join(ident_chars)
                ident_upper = ident_str.upper()
                if ident_upper in KEYWORDS:
                    tokens.append(Token(TokenType.KEYWORD, ident_upper, start_pos))
                else:
                    tokens.append(Token(TokenType.IDENT, ident_str, start_pos))
                continue

            self.pos += 1

        tokens.append(Token(TokenType.EOF, None, self.pos))
        return tokens


# AST Node Definitions
class ASTNode:
    pass


class Literal(ASTNode):
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class ColumnRef(ASTNode):
    def __init__(self, name: str, table: Optional[str] = None):
        self.name = name
        self.table = table

    def __repr__(self) -> str:
        return f"ColumnRef({(self.table + '.' + self.name) if self.table else self.name})"


class BinaryOp(ASTNode):
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryOp({self.left}, {self.op}, {self.right})"


class AggregateExpr(ASTNode):
    def __init__(self, func: str, arg: Optional[ASTNode]):
        self.func = func
        self.arg = arg

    def __repr__(self) -> str:
        return f"AggregateExpr({self.func}({self.arg}))"


class ProjectedItem(ASTNode):
    def __init__(self, expr: ASTNode, alias: Optional[str] = None):
        self.expr = expr
        self.alias = alias

    def output_name(self) -> str:
        if self.alias:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.name
        if isinstance(self.expr, AggregateExpr):
            arg_str = self.expr.arg.name if isinstance(self.expr.arg, ColumnRef) else "*"
            return f"{self.expr.func.lower()}_{arg_str}"
        return "expr"


class JoinClause(ASTNode):
    def __init__(self, join_type: str, table: str, condition: BinaryOp):
        self.join_type = join_type
        self.table = table
        self.condition = condition


class SelectStatement(ASTNode):
    def __init__(
        self,
        projections: List[ProjectedItem],
        from_table: str,
        joins: List[JoinClause],
        where: Optional[ASTNode] = None,
        group_by: Optional[List[ColumnRef]] = None,
        having: Optional[ASTNode] = None,
        order_by: Optional[List[Tuple[str, bool]]] = None,
        limit: Optional[int] = None
    ):
        self.projections = projections
        self.from_table = from_table
        self.joins = joins
        self.where = where
        self.group_by = group_by or []
        self.having = having
        self.order_by = order_by or []
        self.limit = limit


class Parser:
    """Recursive descent parser generating SelectStatement ASTs."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.idx = 0

    def current(self) -> Token:
        return self.tokens[self.idx]

    def match_keyword(self, kw: str) -> bool:
        tok = self.current()
        if tok.type == TokenType.KEYWORD and tok.value == kw:
            self.idx += 1
            return True
        return False

    def expect_keyword(self, kw: str) -> None:
        if not self.match_keyword(kw):
            raise SyntaxError(f"Expected keyword '{kw}', got {self.current()}")

    def match_punct(self, p: str) -> bool:
        tok = self.current()
        if tok.type == TokenType.PUNCT and tok.value == p:
            self.idx += 1
            return True
        return False

    def expect_punct(self, p: str) -> None:
        if not self.match_punct(p):
            raise SyntaxError(f"Expected '{p}', got {self.current()}")

    def parse_query(self) -> SelectStatement:
        self.expect_keyword("SELECT")

        # Projections
        projections = self.parse_projections()

        # FROM
        self.expect_keyword("FROM")
        from_table = self.parse_identifier()

        # JOINs
        joins = []
        while self.match_keyword("JOIN") or self.match_keyword("INNER") or self.match_keyword("LEFT"):
            # backtrack slightly to determine join type
            prev = self.tokens[self.idx - 1].value
            if prev in ("INNER", "LEFT"):
                self.expect_keyword("JOIN")
                join_type = prev
            else:
                join_type = "INNER"

            join_table = self.parse_identifier()
            self.expect_keyword("ON")
            join_cond = self.parse_expression()
            if not isinstance(join_cond, BinaryOp):
                raise SyntaxError("JOIN ON condition must be a binary comparison")
            joins.append(JoinClause(join_type, join_table, join_cond))

        # WHERE
        where_clause = None
        if self.match_keyword("WHERE"):
            where_clause = self.parse_expression()

        # GROUP BY
        group_by = []
        if self.match_keyword("GROUP"):
            self.expect_keyword("BY")
            while True:
                group_by.append(self.parse_column_ref())
                if not self.match_punct(","):
                    break

        # HAVING
        having_clause = None
        if self.match_keyword("HAVING"):
            having_clause = self.parse_expression()

        # ORDER BY
        order_by = []
        if self.match_keyword("ORDER"):
            self.expect_keyword("BY")
            while True:
                col = self.parse_identifier()
                is_desc = False
                if self.match_keyword("DESC"):
                    is_desc = True
                elif self.match_keyword("ASC"):
                    is_desc = False
                order_by.append((col, is_desc))
                if not self.match_punct(","):
                    break

        # LIMIT
        limit_val = None
        if self.match_keyword("LIMIT"):
            tok = self.current()
            if tok.type == TokenType.NUMBER:
                limit_val = int(tok.value)
                self.idx += 1
            else:
                raise SyntaxError(f"LIMIT expects an integer, got {tok}")

        return SelectStatement(
            projections=projections,
            from_table=from_table,
            joins=joins,
            where=where_clause,
            group_by=group_by,
            having=having_clause,
            order_by=order_by,
            limit=limit_val
        )

    def parse_projections(self) -> List[ProjectedItem]:
        items = []
        while True:
            # Check for SELECT *
            if self.current().type == TokenType.OPERATOR and self.current().value == "*":
                self.idx += 1
                items.append(ProjectedItem(ColumnRef("*"), alias="*"))
            else:
                expr = self.parse_expression()
                alias = None
                if self.match_keyword("AS"):
                    alias = self.parse_identifier()
                elif self.current().type == TokenType.IDENT and self.current().value not in KEYWORDS:
                    alias = self.parse_identifier()
                items.append(ProjectedItem(expr, alias=alias))

            if not self.match_punct(","):
                break
        return items

    def parse_expression(self) -> ASTNode:
        return self.parse_or()

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match_keyword("OR"):
            right = self.parse_and()
            left = BinaryOp(left, "OR", right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_comparison()
        while self.match_keyword("AND"):
            right = self.parse_comparison()
            left = BinaryOp(left, "AND", right)
        return left

    def parse_comparison(self) -> ASTNode:
        left = self.parse_term()
        tok = self.current()
        if tok.type == TokenType.OPERATOR and tok.value in ("=", "!=", "<", "<=", ">", ">="):
            op = tok.value
            self.idx += 1
            right = self.parse_term()
            return BinaryOp(left, op, right)
        return left

    def parse_term(self) -> ASTNode:
        tok = self.current()

        # Numbers
        if tok.type == TokenType.NUMBER:
            self.idx += 1
            return Literal(tok.value)

        # Strings
        if tok.type == TokenType.STRING:
            self.idx += 1
            return Literal(tok.value)

        # Aggregates
        if tok.type == TokenType.KEYWORD and tok.value in ("SUM", "COUNT", "AVG", "MIN", "MAX"):
            func_name = tok.value
            self.idx += 1
            self.expect_punct("(")
            if self.current().type == TokenType.OPERATOR and self.current().value == "*":
                self.idx += 1
                arg_expr = None
            else:
                arg_expr = self.parse_expression()
            self.expect_punct(")")
            return AggregateExpr(func_name, arg_expr)

        # Column references (with optional table.column prefix)
        if tok.type == TokenType.IDENT:
            col_name = tok.value
            self.idx += 1
            if self.match_punct("."):
                table_prefix = col_name
                field_name = self.parse_identifier()
                return ColumnRef(field_name, table=table_prefix)
            return ColumnRef(col_name)

        if self.match_punct("("):
            expr = self.parse_expression()
            self.expect_punct(")")
            return expr

        raise SyntaxError(f"Unexpected token in expression: {tok}")

    def parse_column_ref(self) -> ColumnRef:
        col = self.parse_identifier()
        if self.match_punct("."):
            table_name = col
            field_name = self.parse_identifier()
            return ColumnRef(field_name, table=table_name)
        return ColumnRef(col)

    def parse_identifier(self) -> str:
        tok = self.current()
        if tok.type in (TokenType.IDENT, TokenType.KEYWORD):
            self.idx += 1
            return str(tok.value)
        raise SyntaxError(f"Expected identifier, got {tok}")


def parse_sql(sql: str) -> SelectStatement:
    lexer = Lexer(sql)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_query()
