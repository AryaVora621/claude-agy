"""
WasmCore: Module Instance and Runtime Linking Subsystem.
Manages allocated linear memory, globals, function tables, exports, and host imports.
"""

from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from .types import (
    ValType, FuncType, Limits, TableType, MemoryType, GlobalType, ExportDesc,
    Value, WasmTrap, WasmValidationError
)
from .opcodes import Opcode, Instruction
from .module import WasmModule, FunctionDef, ElementSegment, DataSegment
from .memory import LinearMemory


class TableInstance:
    """Table holding function references for indirect dispatch."""
    __slots__ = ("limits", "elements")

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.elements: List[Optional[int]] = [None] * limits.min

    def get(self, index: int) -> Optional[int]:
        if index < 0 or index >= len(self.elements):
            raise WasmTrap(f"undefined element: table index {index} out of bounds")
        return self.elements[index]

    def set(self, index: int, func_idx: int) -> None:
        if index < 0 or index >= len(self.elements):
            raise WasmTrap(f"elements segment out of range: index {index}")
        self.elements[index] = func_idx


class GlobalInstance:
    """Allocated global variable cell."""
    __slots__ = ("global_type", "value")

    def __init__(self, global_type: GlobalType, value: Value) -> None:
        self.global_type = global_type
        self.value = value

    def get(self) -> Value:
        return self.value

    def set(self, val: Value) -> None:
        if not self.global_type.mutable:
            raise WasmTrap("Cannot mutate an immutable global variable")
        if val.val_type != self.global_type.val_type:
            raise WasmTrap(f"Global type mismatch: expected {self.global_type.val_type}, got {val.val_type}")
        self.value = val


class HostFunction:
    """Native Python callback exposed to WebAssembly."""
    __slots__ = ("func_type", "callable_fn", "name")

    def __init__(self, func_type: FuncType, callable_fn: Callable[..., Any], name: Optional[str] = None) -> None:
        self.func_type = func_type
        self.callable_fn = callable_fn
        self.name = name

    def __repr__(self) -> str:
        return f"HostFunction('{self.name}', {self.func_type})"


def eval_const_expr(instructions: List[Instruction], globals_list: List[GlobalInstance]) -> Value:
    """Evaluate simple constant expression for globals, element segments, or data offsets."""
    for instr in instructions:
        if instr.opcode == Opcode.I32_CONST:
            return Value.i32(instr.operands)
        elif instr.opcode == Opcode.I64_CONST:
            return Value.i64(instr.operands)
        elif instr.opcode == Opcode.F32_CONST:
            return Value.f32(instr.operands)
        elif instr.opcode == Opcode.F64_CONST:
            return Value.f64(instr.operands)
        elif instr.opcode == Opcode.GLOBAL_GET:
            gidx = instr.operands
            return globals_list[gidx].get()
    return Value.i32(0)


class WasmInstance:
    """
    Instantiated WebAssembly Module.
    Combines parsed module definitions with allocated runtime state and linked imports.
    """

    def __init__(
        self,
        module: WasmModule,
        imports: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        self.module = module
        self.functions: List[Union[FunctionDef, HostFunction]] = []
        self.memories: List[LinearMemory] = []
        self.tables: List[TableInstance] = []
        self.globals: List[GlobalInstance] = []
        self.exports: Dict[str, Tuple[ExportDesc, Any]] = {}

        imports_map = imports or {}

        # 1. Resolve Imports
        for imp in module.imports:
            mod_dict = imports_map.get(imp.module)
            if mod_dict is None or imp.name not in mod_dict:
                raise WasmValidationError(f"Missing required import: {imp.module}::{imp.name}")
            imported_entity = mod_dict[imp.name]

            if imp.desc_type == ExportDesc.FUNC:
                expected_type = module.types[imp.desc_val]
                if isinstance(imported_entity, HostFunction):
                    if imported_entity.func_type != expected_type:
                        raise WasmValidationError(
                            f"Imported function signature mismatch for {imp.name}: "
                            f"expected {expected_type}, got {imported_entity.func_type}"
                        )
                    self.functions.append(imported_entity)
                elif callable(imported_entity):
                    # Auto-wrap raw python callable with expected signature
                    hf = HostFunction(expected_type, imported_entity, name=f"{imp.module}.{imp.name}")
                    self.functions.append(hf)
                else:
                    raise WasmValidationError(f"Invalid callable for import {imp.name}")

            elif imp.desc_type == ExportDesc.MEM:
                if not isinstance(imported_entity, LinearMemory):
                    raise WasmValidationError(f"Expected LinearMemory for import {imp.name}")
                self.memories.append(imported_entity)

            elif imp.desc_type == ExportDesc.TABLE:
                if not isinstance(imported_entity, TableInstance):
                    raise WasmValidationError(f"Expected TableInstance for import {imp.name}")
                self.tables.append(imported_entity)

            elif imp.desc_type == ExportDesc.GLOBAL:
                if not isinstance(imported_entity, GlobalInstance):
                    raise WasmValidationError(f"Expected GlobalInstance for import {imp.name}")
                self.globals.append(imported_entity)

        # 2. Add Internal Functions
        for fdef in module.functions:
            self.functions.append(fdef)

        # 3. Add Internal Tables
        for tdef in module.tables:
            self.tables.append(TableInstance(tdef.limits))

        # 4. Add Internal Memories
        for mdef in module.memories:
            self.memories.append(LinearMemory(mdef.limits))

        # 5. Add Internal Globals
        for gdef in module.globals:
            init_val = eval_const_expr(gdef.init_expr, self.globals)
            self.globals.append(GlobalInstance(gdef.global_type, init_val))

        # 6. Populate Tables with Element Segments
        for elem in module.elements:
            offset_val = eval_const_expr(elem.offset_expr, self.globals).as_i32()
            table = self.tables[elem.table_idx]
            for i, func_idx in enumerate(elem.func_indices):
                table.set(offset_val + i, func_idx)

        # 7. Initialize Memory with Data Segments
        for data in module.data_segments:
            offset_val = eval_const_expr(data.offset_expr, self.globals).as_i32()
            mem = self.memories[data.mem_idx]
            mem.write_bytes(offset_val, data.data)

        # 8. Populate Exports Map
        for exp in module.exports:
            if exp.desc_type == ExportDesc.FUNC:
                self.exports[exp.name] = (ExportDesc.FUNC, exp.index)
            elif exp.desc_type == ExportDesc.MEM:
                self.exports[exp.name] = (ExportDesc.MEM, self.memories[exp.index])
            elif exp.desc_type == ExportDesc.TABLE:
                self.exports[exp.name] = (ExportDesc.TABLE, self.tables[exp.index])
            elif exp.desc_type == ExportDesc.GLOBAL:
                self.exports[exp.name] = (ExportDesc.GLOBAL, self.globals[exp.index])

    def get_export(self, name: str) -> Any:
        """Fetch exported function index, memory, table, or global."""
        if name not in self.exports:
            raise KeyError(f"Export '{name}' not found in WASM instance")
        return self.exports[name][1]
