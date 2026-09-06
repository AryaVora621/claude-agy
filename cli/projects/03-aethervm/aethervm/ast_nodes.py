"""
Abstract Syntax Tree (AST) Node Definitions for AetherVM.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union


# -----------------------------------------------------------------------------
# Expressions
# -----------------------------------------------------------------------------

class Expr:
    pass


@dataclass
class LiteralExpr(Expr):
    value: Union[int, float]

    def __repr__(self) -> str:
        return f"Literal({self.value})"


@dataclass
class VariableExpr(Expr):
    name: str

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryExpr(Expr):
    op: str
    operand: Expr

    def __repr__(self) -> str:
        return f"({self.op}{self.operand})"


@dataclass
class CallExpr(Expr):
    callee: str
    args: List[Expr]

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.callee}({args_str})"


@dataclass
class AssignExpr(Expr):
    name: str
    value: Expr

    def __repr__(self) -> str:
        return f"{self.name} = {self.value}"


# -----------------------------------------------------------------------------
# Statements
# -----------------------------------------------------------------------------

class Stmt:
    pass


@dataclass
class LetStmt(Stmt):
    name: str
    initializer: Expr

    def __repr__(self) -> str:
        return f"let {self.name} = {self.initializer};"


@dataclass
class ExprStmt(Stmt):
    expr: Expr

    def __repr__(self) -> str:
        return f"{self.expr};"


@dataclass
class BlockStmt(Stmt):
    statements: List[Stmt]

    def __repr__(self) -> str:
        stmts_str = " ".join(repr(s) for s in self.statements)
        return f"{{ {stmts_str} }}"


@dataclass
class IfStmt(Stmt):
    condition: Expr
    then_branch: Stmt
    else_branch: Optional[Stmt] = None

    def __repr__(self) -> str:
        else_str = f" else {self.else_branch}" if self.else_branch else ""
        return f"if ({self.condition}) {self.then_branch}{else_str}"


@dataclass
class WhileStmt(Stmt):
    condition: Expr
    body: Stmt

    def __repr__(self) -> str:
        return f"while ({self.condition}) {self.body}"


@dataclass
class ReturnStmt(Stmt):
    value: Optional[Expr] = None

    def __repr__(self) -> str:
        return f"return {self.value};" if self.value else "return;"


@dataclass
class PrintStmt(Stmt):
    value: Expr

    def __repr__(self) -> str:
        return f"print({self.value});"


@dataclass
class FunctionDef:
    name: str
    params: List[str]
    body: BlockStmt

    def __repr__(self) -> str:
        params_str = ", ".join(self.params)
        return f"fn {self.name}({params_str}) {self.body}"


@dataclass
class Program:
    functions: List[FunctionDef]
    top_level_stmts: List[Stmt]
