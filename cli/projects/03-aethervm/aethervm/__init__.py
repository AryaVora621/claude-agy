"""
AetherVM: Register-Based Bytecode VM with SSA Optimization Passes and Pratt Parser.
Pure Python Standard Library (Zero Dependencies).
"""

from aethervm.lexer import Lexer, Token, TokenType
from aethervm.parser import Parser
from aethervm.ast_nodes import Program, FunctionDef
from aethervm.ir import ProgramIR, FunctionIR, BasicBlock, Instruction, Opcode, Value
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer, ConstantFolder, CommonSubexpressionElimination, DeadCodeElimination
from aethervm.codegen import BytecodeEmitter, BytecodeInst, CompiledFunction
from aethervm.vm import VirtualMachine
from aethervm.visualizer import render_pipeline_view

__all__ = [
    "Lexer",
    "Token",
    "TokenType",
    "Parser",
    "Program",
    "FunctionDef",
    "ProgramIR",
    "FunctionIR",
    "BasicBlock",
    "Instruction",
    "Opcode",
    "Value",
    "SSABuilder",
    "Optimizer",
    "ConstantFolder",
    "CommonSubexpressionElimination",
    "DeadCodeElimination",
    "BytecodeEmitter",
    "BytecodeInst",
    "CompiledFunction",
    "VirtualMachine",
    "render_pipeline_view",
]
