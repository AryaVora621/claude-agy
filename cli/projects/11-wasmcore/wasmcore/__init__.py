"""
WasmCore: WebAssembly MVP Virtual Machine and Binary Toolchain.
Zero-dependency, first-principles implementation of WebAssembly in Python.
"""

from .types import (
    ValType, FuncType, Limits, TableType, MemoryType, GlobalType, ExportDesc, SectionId,
    Value, WasmTrap, WasmValidationError
)
from .leb128 import (
    encode_u32, decode_u32, encode_i32, decode_i32, encode_i64, decode_i64
)
from .opcodes import Opcode, Instruction, MemArg
from .module import (
    WasmModule, Import, Export, FunctionDef, GlobalDef, ElementSegment, DataSegment
)
from .parser import WasmParser
from .emitter import WasmModuleBuilder, FunctionBuilder
from .memory import LinearMemory, PAGE_SIZE
from .instance import WasmInstance, HostFunction, TableInstance, GlobalInstance
from .interpreter import WasmInterpreter, CallFrame, ControlFrame
from .visualizer import WasmDisassembler, MemoryViewer, WasmDebugger

__all__ = [
    "ValType",
    "FuncType",
    "Limits",
    "TableType",
    "MemoryType",
    "GlobalType",
    "ExportDesc",
    "SectionId",
    "Value",
    "WasmTrap",
    "WasmValidationError",
    "encode_u32",
    "decode_u32",
    "encode_i32",
    "decode_i32",
    "encode_i64",
    "decode_i64",
    "Opcode",
    "Instruction",
    "MemArg",
    "WasmModule",
    "Import",
    "Export",
    "FunctionDef",
    "GlobalDef",
    "ElementSegment",
    "DataSegment",
    "WasmParser",
    "WasmModuleBuilder",
    "FunctionBuilder",
    "LinearMemory",
    "PAGE_SIZE",
    "WasmInstance",
    "HostFunction",
    "TableInstance",
    "GlobalInstance",
    "WasmInterpreter",
    "CallFrame",
    "ControlFrame",
    "WasmDisassembler",
    "MemoryViewer",
    "WasmDebugger",
]
