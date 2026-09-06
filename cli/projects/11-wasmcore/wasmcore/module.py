"""
WasmCore: Module Data Structure and AST.
Represents decoded or constructed WebAssembly modules.
"""

from typing import List, Optional, Tuple, Any, Dict
from .types import FuncType, ValType, Limits, TableType, MemoryType, GlobalType, ExportDesc
from .opcodes import Instruction


class Import:
    """Import declaration from host environment."""
    __slots__ = ("module", "name", "desc_type", "desc_val")

    def __init__(self, module: str, name: str, desc_type: ExportDesc, desc_val: Any) -> None:
        self.module = module
        self.name = name
        self.desc_type = desc_type
        self.desc_val = desc_val

    def __repr__(self) -> str:
        return f"Import({self.module}::{self.name}, {self.desc_type.name}={self.desc_val})"


class Export:
    """Exported entity accessible to host."""
    __slots__ = ("name", "desc_type", "index")

    def __init__(self, name: str, desc_type: ExportDesc, index: int) -> None:
        self.name = name
        self.desc_type = desc_type
        self.index = int(index)

    def __repr__(self) -> str:
        return f"Export({self.name} -> {self.desc_type.name} #{self.index})"


class FunctionDef:
    """Internal function definition."""
    __slots__ = ("type_idx", "locals", "instructions", "name")

    def __init__(
        self,
        type_idx: int,
        locals: Optional[List[Tuple[int, ValType]]] = None,
        instructions: Optional[List[Instruction]] = None,
        name: Optional[str] = None
    ) -> None:
        self.type_idx = int(type_idx)
        self.locals = locals or []  # List of (count, ValType)
        self.instructions = instructions or []
        self.name = name

    def __repr__(self) -> str:
        name_str = f" '{self.name}'" if self.name else ""
        return f"FunctionDef{name_str}(type={self.type_idx}, {len(self.instructions)} instrs)"


class GlobalDef:
    """Global variable declaration."""
    __slots__ = ("global_type", "init_expr")

    def __init__(self, global_type: GlobalType, init_expr: List[Instruction]) -> None:
        self.global_type = global_type
        self.init_expr = init_expr

    def __repr__(self) -> str:
        return f"GlobalDef({self.global_type}, init={self.init_expr})"


class ElementSegment:
    """Table initialization segment."""
    __slots__ = ("table_idx", "offset_expr", "func_indices")

    def __init__(self, table_idx: int, offset_expr: List[Instruction], func_indices: List[int]) -> None:
        self.table_idx = int(table_idx)
        self.offset_expr = offset_expr
        self.func_indices = func_indices

    def __repr__(self) -> str:
        return f"ElementSegment(table={self.table_idx}, indices={self.func_indices})"


class DataSegment:
    """Memory initialization segment."""
    __slots__ = ("mem_idx", "offset_expr", "data")

    def __init__(self, mem_idx: int, offset_expr: List[Instruction], data: bytes) -> None:
        self.mem_idx = int(mem_idx)
        self.offset_expr = offset_expr
        self.data = data

    def __repr__(self) -> str:
        return f"DataSegment(mem={self.mem_idx}, len={len(self.data)})"


class WasmModule:
    """Complete WebAssembly Module structure."""

    def __init__(self) -> None:
        self.types: List[FuncType] = []
        self.imports: List[Import] = []
        self.functions: List[FunctionDef] = []
        self.tables: List[TableType] = []
        self.memories: List[MemoryType] = []
        self.globals: List[GlobalDef] = []
        self.exports: List[Export] = []
        self.start_fn_idx: Optional[int] = None
        self.elements: List[ElementSegment] = []
        self.data_segments: List[DataSegment] = []
        self.custom_sections: Dict[str, bytes] = {}

    def get_imported_functions_count(self) -> int:
        return sum(1 for imp in self.imports if imp.desc_type == ExportDesc.FUNC)

    def get_imported_memories_count(self) -> int:
        return sum(1 for imp in self.imports if imp.desc_type == ExportDesc.MEM)

    def get_imported_tables_count(self) -> int:
        return sum(1 for imp in self.imports if imp.desc_type == ExportDesc.TABLE)

    def get_imported_globals_count(self) -> int:
        return sum(1 for imp in self.imports if imp.desc_type == ExportDesc.GLOBAL)

    def get_function_type(self, func_idx: int) -> FuncType:
        """Resolve function signature given an absolute function index."""
        imported_funcs = [imp for imp in self.imports if imp.desc_type == ExportDesc.FUNC]
        if func_idx < len(imported_funcs):
            type_idx = imported_funcs[func_idx].desc_val
            return self.types[type_idx]
        internal_idx = func_idx - len(imported_funcs)
        if internal_idx < len(self.functions):
            type_idx = self.functions[internal_idx].type_idx
            return self.types[type_idx]
        raise IndexError(f"Function index out of range: {func_idx}")

    def __repr__(self) -> str:
        return (
            f"WasmModule(types={len(self.types)}, imports={len(self.imports)}, "
            f"funcs={len(self.functions)}, memories={len(self.memories)}, "
            f"exports={len(self.exports)})"
        )
