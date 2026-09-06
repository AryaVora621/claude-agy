"""
WasmCore: In-Memory WebAssembly Binary Emitter & Programmatic Assembler.
Enables fluent construction and binary serialization of valid .wasm modules.
"""

import io
import struct
from typing import List, Tuple, Optional, Union, Dict, Any
from .types import (
    ValType, FuncType, Limits, TableType, MemoryType, GlobalType, ExportDesc, SectionId
)
from .leb128 import (
    encode_u32, encode_i32, encode_i64, encode_name, encode_vec
)
from .opcodes import Opcode, Instruction, MemArg
from .parser import WASM_MAGIC, WASM_VERSION


def encode_limits(limits: Limits) -> bytes:
    """Encode memory/table limits."""
    if limits.max is not None:
        return b"\x01" + encode_u32(limits.min) + encode_u32(limits.max)
    return b"\x00" + encode_u32(limits.min)


def encode_instruction(instr: Instruction) -> bytes:
    """Encode a single Instruction into raw bytecode bytes."""
    out = bytearray([instr.opcode])
    op = instr.operands

    if instr.opcode in (Opcode.BLOCK, Opcode.LOOP, Opcode.IF):
        if op is None:
            out.append(0x40)  # void block
        elif isinstance(op, ValType):
            out.append(int(op))
        else:
            out.extend(encode_i32(int(op)))

    elif instr.opcode in (Opcode.BR, Opcode.BR_IF):
        out.extend(encode_u32(int(op)))

    elif instr.opcode == Opcode.BR_TABLE:
        targets, default_target = op
        out.extend(encode_vec(targets, encode_u32))
        out.extend(encode_u32(default_target))

    elif instr.opcode == Opcode.CALL:
        out.extend(encode_u32(int(op)))

    elif instr.opcode == Opcode.CALL_INDIRECT:
        type_idx, table_idx = op
        out.extend(encode_u32(type_idx))
        out.extend(encode_u32(table_idx))

    elif instr.opcode in (Opcode.LOCAL_GET, Opcode.LOCAL_SET, Opcode.LOCAL_TEE):
        out.extend(encode_u32(int(op)))

    elif instr.opcode in (Opcode.GLOBAL_GET, Opcode.GLOBAL_SET):
        out.extend(encode_u32(int(op)))

    elif instr.opcode in (
        Opcode.I32_LOAD, Opcode.I64_LOAD, Opcode.F32_LOAD, Opcode.F64_LOAD,
        Opcode.I32_LOAD8_S, Opcode.I32_LOAD8_U, Opcode.I32_LOAD16_S, Opcode.I32_LOAD16_U,
        Opcode.I64_LOAD8_S, Opcode.I64_LOAD8_U, Opcode.I64_LOAD16_S, Opcode.I64_LOAD16_U,
        Opcode.I64_LOAD32_S, Opcode.I64_LOAD32_U,
        Opcode.I32_STORE, Opcode.I64_STORE, Opcode.F32_STORE, Opcode.F64_STORE,
        Opcode.I32_STORE8, Opcode.I32_STORE16, Opcode.I64_STORE8, Opcode.I64_STORE16,
        Opcode.I64_STORE32
    ):
        if isinstance(op, MemArg):
            out.extend(encode_u32(op.align))
            out.extend(encode_u32(op.offset))
        elif isinstance(op, (tuple, list)):
            out.extend(encode_u32(op[0]))
            out.extend(encode_u32(op[1]))
        else:
            out.extend(encode_u32(2))  # default align
            out.extend(encode_u32(0))  # offset 0

    elif instr.opcode in (Opcode.MEMORY_SIZE, Opcode.MEMORY_GROW):
        out.append(0x00)

    elif instr.opcode == Opcode.I32_CONST:
        out.extend(encode_i32(int(op)))

    elif instr.opcode == Opcode.I64_CONST:
        out.extend(encode_i64(int(op)))

    elif instr.opcode == Opcode.F32_CONST:
        out.extend(struct.pack("<f", float(op)))

    elif instr.opcode == Opcode.F64_CONST:
        out.extend(struct.pack("<d", float(op)))

    return bytes(out)


def encode_expression(instructions: List[Instruction]) -> bytes:
    """Encode a sequence of instructions terminated by END."""
    out = bytearray()
    has_end = False
    for instr in instructions:
        out.extend(encode_instruction(instr))
        if instr.opcode == Opcode.END:
            has_end = True
    if not has_end:
        out.append(Opcode.END)
    return bytes(out)


class FunctionBuilder:
    """Fluent helper for constructing function bytecode."""

    def __init__(self, func_idx: int, type_idx: int, name: Optional[str] = None) -> None:
        self.func_idx = func_idx
        self.type_idx = type_idx
        self.name = name
        self.locals: List[Tuple[int, ValType]] = []
        self.instructions: List[Instruction] = []

    def add_locals(self, count: int, val_type: ValType) -> "FunctionBuilder":
        self.locals.append((count, val_type))
        return self

    def emit(self, opcode: Opcode, operands: Any = None) -> "FunctionBuilder":
        self.instructions.append(Instruction(opcode, operands))
        return self

    # Ergonomic instruction helpers
    def i32_const(self, val: int) -> "FunctionBuilder":
        return self.emit(Opcode.I32_CONST, val)

    def i64_const(self, val: int) -> "FunctionBuilder":
        return self.emit(Opcode.I64_CONST, val)

    def f32_const(self, val: float) -> "FunctionBuilder":
        return self.emit(Opcode.F32_CONST, val)

    def f64_const(self, val: float) -> "FunctionBuilder":
        return self.emit(Opcode.F64_CONST, val)

    def local_get(self, idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.LOCAL_GET, idx)

    def local_set(self, idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.LOCAL_SET, idx)

    def local_tee(self, idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.LOCAL_TEE, idx)

    def global_get(self, idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.GLOBAL_GET, idx)

    def global_set(self, idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.GLOBAL_SET, idx)

    def i32_add(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_ADD)

    def i32_sub(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_SUB)

    def i32_mul(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_MUL)

    def i32_div_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_DIV_S)

    def i32_rem_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_REM_S)

    def i32_eq(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_EQ)

    def i32_ne(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_NE)

    def i32_lt_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_LT_S)

    def i32_gt_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_GT_S)

    def i32_le_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_LE_S)

    def i32_ge_s(self) -> "FunctionBuilder":
        return self.emit(Opcode.I32_GE_S)

    def i32_load(self, offset: int = 0, align: int = 2) -> "FunctionBuilder":
        return self.emit(Opcode.I32_LOAD, MemArg(align, offset))

    def i32_store(self, offset: int = 0, align: int = 2) -> "FunctionBuilder":
        return self.emit(Opcode.I32_STORE, MemArg(align, offset))

    def block(self, block_type: Optional[ValType] = None) -> "FunctionBuilder":
        return self.emit(Opcode.BLOCK, block_type)

    def loop(self, block_type: Optional[ValType] = None) -> "FunctionBuilder":
        return self.emit(Opcode.LOOP, block_type)

    def if_(self, block_type: Optional[ValType] = None) -> "FunctionBuilder":
        return self.emit(Opcode.IF, block_type)

    def else_(self) -> "FunctionBuilder":
        return self.emit(Opcode.ELSE)

    def end(self) -> "FunctionBuilder":
        return self.emit(Opcode.END)

    def br(self, depth: int) -> "FunctionBuilder":
        return self.emit(Opcode.BR, depth)

    def br_if(self, depth: int) -> "FunctionBuilder":
        return self.emit(Opcode.BR_IF, depth)

    def call(self, func_idx: int) -> "FunctionBuilder":
        return self.emit(Opcode.CALL, func_idx)

    def return_(self) -> "FunctionBuilder":
        return self.emit(Opcode.RETURN)

    def drop(self) -> "FunctionBuilder":
        return self.emit(Opcode.DROP)


class WasmModuleBuilder:
    """Builder for constructing and serializing WebAssembly binary modules."""

    def __init__(self) -> None:
        self.types: List[FuncType] = []
        self.imports: List[Tuple[str, str, ExportDesc, Any]] = []
        self.functions: List[FunctionBuilder] = []
        self.tables: List[TableType] = []
        self.memories: List[MemoryType] = []
        self.globals: List[Tuple[GlobalType, List[Instruction]]] = []
        self.exports: List[Tuple[str, ExportDesc, int]] = []
        self.start_fn_idx: Optional[int] = None
        self.data_segments: List[Tuple[int, List[Instruction], bytes]] = []

    def add_type(
        self,
        params: Union[List[ValType], Tuple[ValType, ...]],
        results: Union[List[ValType], Tuple[ValType, ...]]
    ) -> int:
        ft = FuncType(params, results)
        for idx, existing in enumerate(self.types):
            if existing == ft:
                return idx
        self.types.append(ft)
        return len(self.types) - 1

    def add_import_func(self, module: str, name: str, type_idx: int) -> int:
        idx = len([imp for imp in self.imports if imp[2] == ExportDesc.FUNC])
        self.imports.append((module, name, ExportDesc.FUNC, type_idx))
        return idx

    def add_function(
        self,
        type_idx: int,
        name: Optional[str] = None
    ) -> FunctionBuilder:
        imported_funcs = sum(1 for imp in self.imports if imp[2] == ExportDesc.FUNC)
        func_idx = imported_funcs + len(self.functions)
        fb = FunctionBuilder(func_idx, type_idx, name)
        self.functions.append(fb)
        return fb

    def add_memory(self, min_pages: int, max_pages: Optional[int] = None) -> int:
        idx = len(self.memories)
        self.memories.append(MemoryType(Limits(min_pages, max_pages)))
        return idx

    def add_global(
        self,
        val_type: ValType,
        mutable: bool,
        init_val: Union[int, float]
    ) -> int:
        idx = len(self.globals)
        gt = GlobalType(val_type, mutable)
        if val_type == ValType.I32:
            init_instr = [Instruction(Opcode.I32_CONST, int(init_val)), Instruction(Opcode.END)]
        elif val_type == ValType.I64:
            init_instr = [Instruction(Opcode.I64_CONST, int(init_val)), Instruction(Opcode.END)]
        elif val_type == ValType.F32:
            init_instr = [Instruction(Opcode.F32_CONST, float(init_val)), Instruction(Opcode.END)]
        else:
            init_instr = [Instruction(Opcode.F64_CONST, float(init_val)), Instruction(Opcode.END)]
        self.globals.append((gt, init_instr))
        return idx

    def add_export(self, name: str, desc: ExportDesc, index: int) -> "WasmModuleBuilder":
        self.exports.append((name, desc, index))
        return self

    def add_data(self, mem_idx: int, offset: int, data: bytes) -> "WasmModuleBuilder":
        offset_expr = [Instruction(Opcode.I32_CONST, offset), Instruction(Opcode.END)]
        self.data_segments.append((mem_idx, offset_expr, data))
        return self

    def build(self) -> bytes:
        """Serialize complete WebAssembly binary module to bytes."""
        out = bytearray()
        out.extend(WASM_MAGIC)
        out.extend(WASM_VERSION)

        def emit_section(sec_id: SectionId, payload: bytes):
            if payload:
                out.append(int(sec_id))
                out.extend(encode_u32(len(payload)))
                out.extend(payload)

        # 1. Type Section
        if self.types:
            type_payload = bytearray()
            type_payload.extend(encode_u32(len(self.types)))
            for ft in self.types:
                type_payload.append(0x60)
                type_payload.extend(encode_u32(len(ft.params)))
                for p in ft.params:
                    type_payload.append(int(p))
                type_payload.extend(encode_u32(len(ft.results)))
                for r in ft.results:
                    type_payload.append(int(r))
            emit_section(SectionId.TYPE, bytes(type_payload))

        # 2. Import Section
        if self.imports:
            imp_payload = bytearray()
            imp_payload.extend(encode_u32(len(self.imports)))
            for mod, name, kind, desc_val in self.imports:
                imp_payload.extend(encode_name(mod))
                imp_payload.extend(encode_name(name))
                imp_payload.append(int(kind))
                if kind == ExportDesc.FUNC:
                    imp_payload.extend(encode_u32(desc_val))
                elif kind == ExportDesc.MEM:
                    imp_payload.extend(encode_limits(desc_val.limits))
                elif kind == ExportDesc.GLOBAL:
                    imp_payload.append(int(desc_val.val_type))
                    imp_payload.append(1 if desc_val.mutable else 0)
            emit_section(SectionId.IMPORT, bytes(imp_payload))

        # 3. Function Section
        if self.functions:
            func_payload = bytearray()
            func_payload.extend(encode_u32(len(self.functions)))
            for fb in self.functions:
                func_payload.extend(encode_u32(fb.type_idx))
            emit_section(SectionId.FUNCTION, bytes(func_payload))

        # 4. Table Section
        if self.tables:
            tab_payload = bytearray()
            tab_payload.extend(encode_u32(len(self.tables)))
            for tab in self.tables:
                tab_payload.append(int(tab.element_type))
                tab_payload.extend(encode_limits(tab.limits))
            emit_section(SectionId.TABLE, bytes(tab_payload))

        # 5. Memory Section
        if self.memories:
            mem_payload = bytearray()
            mem_payload.extend(encode_u32(len(self.memories)))
            for mem in self.memories:
                mem_payload.extend(encode_limits(mem.limits))
            emit_section(SectionId.MEMORY, bytes(mem_payload))

        # 6. Global Section
        if self.globals:
            glob_payload = bytearray()
            glob_payload.extend(encode_u32(len(self.globals)))
            for gt, init_expr in self.globals:
                glob_payload.append(int(gt.val_type))
                glob_payload.append(1 if gt.mutable else 0)
                glob_payload.extend(encode_expression(init_expr))
            emit_section(SectionId.GLOBAL, bytes(glob_payload))

        # 7. Export Section
        if self.exports:
            exp_payload = bytearray()
            exp_payload.extend(encode_u32(len(self.exports)))
            for name, kind, idx in self.exports:
                exp_payload.extend(encode_name(name))
                exp_payload.append(int(kind))
                exp_payload.extend(encode_u32(idx))
            emit_section(SectionId.EXPORT, bytes(exp_payload))

        # 10. Code Section
        if self.functions:
            code_payload = bytearray()
            code_payload.extend(encode_u32(len(self.functions)))
            for fb in self.functions:
                body_bytes = bytearray()
                # Locals vector
                body_bytes.extend(encode_u32(len(fb.locals)))
                for count, vtype in fb.locals:
                    body_bytes.extend(encode_u32(count))
                    body_bytes.append(int(vtype))

                # Instructions
                has_end = False
                for instr in fb.instructions:
                    body_bytes.extend(encode_instruction(instr))
                    if instr.opcode == Opcode.END:
                        has_end = True
                if not has_end:
                    body_bytes.append(Opcode.END)

                # Prefix body with byte size
                code_payload.extend(encode_u32(len(body_bytes)))
                code_payload.extend(body_bytes)

            emit_section(SectionId.CODE, bytes(code_payload))

        # 11. Data Section
        if self.data_segments:
            data_payload = bytearray()
            data_payload.extend(encode_u32(len(self.data_segments)))
            for mem_idx, offset_expr, data in self.data_segments:
                data_payload.extend(encode_u32(mem_idx))
                data_payload.extend(encode_expression(offset_expr))
                data_payload.extend(encode_u32(len(data)))
                data_payload.extend(data)
            emit_section(SectionId.DATA, bytes(data_payload))

        return bytes(out)
