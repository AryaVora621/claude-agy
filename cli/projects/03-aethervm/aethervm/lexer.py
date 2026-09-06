"""
AetherVM Lexical Analyzer:
Converts raw source text into a stream of typed tokens with line/column tracking.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenType(Enum):
    # Literals & Identifiers
    NUMBER = auto()
    IDENTIFIER = auto()

    # Keywords
    FN = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    RETURN = auto()
    PRINT = auto()

    # Operators
    PLUS = auto()         # +
    MINUS = auto()        # -
    STAR = auto()         # *
    SLASH = auto()        # /
    PERCENT = auto()      # %
    EQUAL = auto()        # =
    EQ_EQ = auto()        # ==
    BANG_EQ = auto()      # !=
    LESS = auto()         # <
    LESS_EQ = auto()      # <=
    GREATER = auto()      # >
    GREATER_EQ = auto()   # >=
    AND_AND = auto()      # &&
    OR_OR = auto()        # ||

    # Delimiters
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    LBRACE = auto()       # {
    RBRACE = auto()       # }
    COMMA = auto()        # ,
    SEMICOLON = auto()    # ;

    EOF = auto()


KEYWORDS = {
    "fn": TokenType.FN,
    "let": TokenType.LET,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "return": TokenType.RETURN,
    "print": TokenType.PRINT,
}


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: any
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {repr(self.value)}, L{self.line}:C{self.column})"


class Lexer:
    """Scans source code into tokens."""

    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self) -> str:
        return self.source[self.pos] if self.pos < self.length else "\0"

    def _peek_next(self) -> str:
        return self.source[self.pos + 1] if self.pos + 1 < self.length else "\0"

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Whitespace
            if ch in " \t\r\n":
                self._advance()
                continue

            # Comments (//...)
            if ch == "/" and self._peek_next() == "/":
                while self._peek() not in ("\n", "\0"):
                    self._advance()
                continue

            start_line = self.line
            start_col = self.col

            # Numeric literals
            if ch.isdigit():
                num_str = ""
                while self._peek().isdigit():
                    num_str += self._advance()
                if self._peek() == "." and self._peek_next().isdigit():
                    num_str += self._advance()
                    while self._peek().isdigit():
                        num_str += self._advance()
                    val = float(num_str)
                else:
                    val = int(num_str)
                tokens.append(Token(TokenType.NUMBER, val, start_line, start_col))
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                ident_str = ""
                while self._peek().isalnum() or self._peek() == "_":
                    ident_str += self._advance()
                tok_type = KEYWORDS.get(ident_str, TokenType.IDENTIFIER)
                tokens.append(Token(tok_type, ident_str, start_line, start_col))
                continue

            # Two-character operators
            if ch == "=" and self._peek_next() == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.EQ_EQ, "==", start_line, start_col))
                continue
            if ch == "!" and self._peek_next() == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.BANG_EQ, "!=", start_line, start_col))
                continue
            if ch == "<" and self._peek_next() == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.LESS_EQ, "<=", start_line, start_col))
                continue
            if ch == ">" and self._peek_next() == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.GREATER_EQ, ">=", start_line, start_col))
                continue
            if ch == "&" and self._peek_next() == "&":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.AND_AND, "&&", start_line, start_col))
                continue
            if ch == "|" and self._peek_next() == "|":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.OR_OR, "||", start_line, start_col))
                continue

            # Single-character tokens
            single_map = {
                "+": TokenType.PLUS,
                "-": TokenType.MINUS,
                "*": TokenType.STAR,
                "/": TokenType.SLASH,
                "%": TokenType.PERCENT,
                "=": TokenType.EQUAL,
                "<": TokenType.LESS,
                ">": TokenType.GREATER,
                "(": TokenType.LPAREN,
                ")": TokenType.RPAREN,
                "{": TokenType.LBRACE,
                "}": TokenType.RBRACE,
                ",": TokenType.COMMA,
                ";": TokenType.SEMICOLON,
            }

            if ch in single_map:
                self._advance()
                tokens.append(Token(single_map[ch], ch, start_line, start_col))
                continue

            raise SyntaxError(f"Unexpected character '{ch}' at L{start_line}:C{start_col}")

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens
