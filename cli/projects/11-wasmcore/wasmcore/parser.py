"""
WasmCore: WebAssembly Binary Module Parser.
Decodes standard .wasm binary streams according to the W3C WebAssembly MVP specification.
"""

import io
import struct
from typing import List, Union
from .types import (
    ValType, FuncType, Limits, TableType, MemoryType, GlobalType, ExportDesc,
    SectionId, WasmValidationError
)
from .leb128 import (
    decode_u32, decode_i32, decode_i64, decode_vec, decode_name
)
from .opcodes import Opcode, Instruction, MemArg
from .module import (
    WasmModule, Import, Export, FunctionDef, GlobalDef, ElementSegment, DataSegment
)

WASM_MAGIC = b"\x00asm"
WASM_VERSION = b"\x01\x00\x00\x00"


def parse_limits(stream: io.BytesIO) -> Limits:
    """Parse memory/table limits: flags (0=min only, 1=min and max)."""
    flags_byte = stream.read(1)
    if not flags_byte:
        raise WasmValidationError("Unexpected end of limits stream")
    flags = flags_byte[0]
    min_val = decode_u32(stream)
    max_val = None
    if flags == 1:
        max_val = decode_u32(stream)
    elif flags != 0:
        raise WasmValidationError(f"Invalid limits flags: {flags}")
    return Limits(min_val, max_val)


def parse_instructions(stream: io.BytesIO) -> List[Instruction]:
    """Parse bytecode stream into sequence of Instruction objects until end."""
    instructions = []
    while True:
        pos = stream.tell()
        byte_data = stream.read(1)
        if not byte_data:
            break
        op_byte = byte_data[0]
        try:
            opcode = Opcode(op_byte)
        except ValueError:
            raise WasmValidationError(f"Unknown opcode 0x{op_byte:02X} at offset {pos}")

        operands = None

        if opcode in (Opcode.BLOCK, Opcode.LOOP, Opcode.IF):
            # Block type: 0x40 (void), or ValType, or signed type index
            bt_byte = stream.read(1)
            if not bt_byte:
                raise WasmValidationError("Unexpected EOF reading block type")
            bt = bt_byte[0]
            if bt == 0x40:
                operands = None
            else:
                try:
                    operands = ValType(bt)
                except ValueError:
                    operands = bt
        elif opcode in (Opcode.BR, Opcode.BR_IF):
            operands = decode_u32(stream)
        elif opcode == Opcode.BR_TABLE:
            targets = decode_vec(stream, decode_u32)
            default_target = decode_u32(stream)
            operands = (targets, default_target)
        elif opcode == Opcode.CALL:
            operands = decode_u32(stream)
        elif opcode == Opcode.CALL_INDIRECT:
            type_idx = decode_u32(stream)
            table_idx = decode_u32(stream)
            operands = (type_idx, table_idx)
        elif opcode in (Opcode.LOCAL_GET, Opcode.LOCAL_SET, Opcode.LOCAL_TEE):
            operands = decode_u32(stream)
        elif opcode in (Opcode.GLOBAL_GET, Opcode.GLOBAL_SET):
            operands = decode_u32(stream)
        elif opcode in (
            Opcode.I32_LOAD, Opcode.I64_LOAD, Opcode.F32_LOAD, Opcode.F64_LOAD,
            Opcode.I32_LOAD8_S, Opcode.I32_LOAD8_U, Opcode.I32_LOAD16_S, Opcode.I32_LOAD16_U,
            Opcode.I64_LOAD8_S, Opcode.I64_LOAD8_U, Opcode.I64_LOAD16_S, Opcode.I64_LOAD16_U,
            Opcode.I64_LOAD32_S, Opcode.I64_LOAD32_U,
            Opcode.I32_STORE, Opcode.I64_STORE, Opcode.F32_STORE, Opcode.F64_STORE,
            Opcode.I32_STORE8, Opcode.I32_STORE16, Opcode.I64_STORE8, Opcode.I64_STORE16,
            Opcode.I64_STORE32
        ):
            align = decode_u32(stream)
            offset = decode_u32(stream)
            operands = MemArg(align, offset)
        elif opcode in (Opcode.MEMORY_SIZE, Opcode.MEMORY_GROW):
            res = stream.read(1)
            if not res or res[0] != 0x00:
                raise WasmValidationError("Expected 0x00 reserved byte for memory.size/grow")
            operands = 0
        elif opcode == Opcode.I32_CONST:
            operands = decode_i32(stream)
        elif opcode == Opcode.I64_CONST:
            operands = decode_i64(stream)
        elif opcode == Opcode.F32_CONST:
            raw = stream.read(4)
            if len(raw) != 4:
                raise WasmValidationError("Unexpected EOF reading f32.const")
            operands = struct.unpack("<f", raw)[0]
        elif opcode == Opcode.F64_CONST:
            raw = stream.read(8)
            if len(raw) != 8:
                raise WasmValidationError("Unexpected EOF reading f64.const")
            operands = struct.unpack("<d", raw)[0]

        instr = Instruction(opcode, operands, pos)
        instructions.append(instr)

    return instructions


def parse_expression(stream: io.BytesIO) -> List[Instruction]:
    """Parse instructions until an END (0x0B) opcode."""
    instructions = []
    while True:
        pos = stream.tell()
        byte_data = stream.read(1)
        if not byte_data:
            raise WasmValidationError("Unexpected EOF in constant expression (missing END 0x0B)")
        op_byte = byte_data[0]
        opcode = Opcode(op_byte)
        if opcode == Opcode.END:
            instructions.append(Instruction(opcode, None, pos))
            break
        # Process single const / global.get instruction in init expr
        operands = None
        if opcode == Opcode.I32_CONST:
            operands = decode_i32(stream)
        elif opcode == Opcode.I64_CONST:
            operands = decode_i64(stream)
        elif opcode == Opcode.F32_CONST:
            operands = struct.unpack("<f", stream.read(4))[0]
        elif opcode == Opcode.F64_CONST:
            operands = struct.unpack("<d", stream.read(8))[0]
        elif opcode == Opcode.GLOBAL_GET:
            operands = decode_u32(stream)
        instructions.append(Instruction(opcode, operands, pos))
    return instructions


class WasmParser:
    """Decodes binary WebAssembly modules into structured WasmModule ASTs."""

    def __init__(self, data: Union[bytes, bytearray, io.BytesIO]) -> None:
        if isinstance(data, (bytes, bytearray)):
            self.stream = io.BytesIO(data)
        else:
            self.stream = data

    def parse(self) -> WasmModule:
        # 1. Header validation
        magic = self.stream.read(4)
        if magic != WASM_MAGIC:
            raise WasmValidationError(f"Invalid magic number: {magic!r}, expected {WASM_MAGIC!r}")

        version = self.stream.read(4)
        if version != WASM_VERSION:
            raise WasmValidationError(f"Unsupported WASM version: {version!r}")

        module = WasmModule()
        func_types_indices: List[int] = []

        # 2. Iterate through sections
        while True:
            sec_id_byte = self.stream.read(1)
            if not sec_id_byte:
                break
            sec_id = sec_id_byte[0]
            sec_len = decode_u32(self.stream)
            sec_data = self.stream.read(sec_len)
            if len(sec_data) != sec_len:
                raise WasmValidationError(f"Section {sec_id} truncated: expected {sec_len} bytes")

            sec_stream = io.BytesIO(sec_data)

            if sec_id == SectionId.CUSTOM:
                # Custom section: name + payload
                name = decode_name(sec_stream)
                payload = sec_stream.read()
                module.custom_sections[name] = payload

            elif sec_id == SectionId.TYPE:
                # Type section: vector of FuncType
                def parse_func_type(s: io.BytesIO) -> FuncType:
                    form = s.read(1)[0]
                    if form != 0x60:
                        raise WasmValidationError(f"Invalid FuncType form 0x{form:02X}, expected 0x60")
                    param_count = decode_u32(s)
                    params = [ValType(s.read(1)[0]) for _ in range(param_count)]
                    result_count = decode_u32(s)
                    results = [ValType(s.read(1)[0]) for _ in range(result_count)]
                    return FuncType(params, results)

                module.types = decode_vec(sec_stream, parse_func_type)

            elif sec_id == SectionId.IMPORT:
                # Import section: vector of Import
                def parse_import(s: io.BytesIO) -> Import:
                    mod_name = decode_name(s)
                    field_name = decode_name(s)
                    kind_byte = s.read(1)[0]
                    kind = ExportDesc(kind_byte)
                    if kind == ExportDesc.FUNC:
                        desc_val = decode_u32(s)
                    elif kind == ExportDesc.TABLE:
                        elem_type = ValType(s.read(1)[0])
                        limits = parse_limits(s)
                        desc_val = TableType(elem_type, limits)
                    elif kind == ExportDesc.MEM:
                        limits = parse_limits(s)
                        desc_val = MemoryType(limits)
                    elif kind == ExportDesc.GLOBAL:
                        vtype = ValType(s.read(1)[0])
                        mut = bool(s.read(1)[0])
                        desc_val = GlobalType(vtype, mut)
                    else:
                        raise WasmValidationError(f"Unknown import kind: {kind_byte}")
                    return Import(mod_name, field_name, kind, desc_val)

                module.imports = decode_vec(sec_stream, parse_import)

            elif sec_id == SectionId.FUNCTION:
                # Function section: vector of type indices
                func_types_indices = decode_vec(sec_stream, decode_u32)

            elif sec_id == SectionId.TABLE:
                def parse_table(s: io.BytesIO) -> TableType:
                    elem_type = ValType(s.read(1)[0])
                    limits = parse_limits(s)
                    return TableType(elem_type, limits)

                module.tables = decode_vec(sec_stream, parse_table)

            elif sec_id == SectionId.MEMORY:
                def parse_mem(s: io.BytesIO) -> MemoryType:
                    limits = parse_limits(s)
                    return MemoryType(limits)

                module.memories = decode_vec(sec_stream, parse_mem)

            elif sec_id == SectionId.GLOBAL:
                def parse_global(s: io.BytesIO) -> GlobalDef:
                    vtype = ValType(s.read(1)[0])
                    mut = bool(s.read(1)[0])
                    init_expr = parse_expression(s)
                    return GlobalDef(GlobalType(vtype, mut), init_expr)

                module.globals = decode_vec(sec_stream, parse_global)

            elif sec_id == SectionId.EXPORT:
                def parse_export(s: io.BytesIO) -> Export:
                    exp_name = decode_name(s)
                    kind = ExportDesc(s.read(1)[0])
                    idx = decode_u32(s)
                    return Export(exp_name, kind, idx)

                module.exports = decode_vec(sec_stream, parse_export)

            elif sec_id == SectionId.START:
                module.start_fn_idx = decode_u32(sec_stream)

            elif sec_id == SectionId.ELEMENT:
                def parse_element(s: io.BytesIO) -> ElementSegment:
                    table_idx = decode_u32(s)
                    offset_expr = parse_expression(s)
                    func_indices = decode_vec(s, decode_u32)
                    return ElementSegment(table_idx, offset_expr, func_indices)

                module.elements = decode_vec(sec_stream, parse_element)

            elif sec_id == SectionId.CODE:
                code_count = decode_u32(sec_stream)
                if len(func_types_indices) != code_count:
                    raise WasmValidationError(
                        f"Function section count ({len(func_types_indices)}) != Code section count ({code_count})"
                    )

                for i in range(code_count):
                    body_size = decode_u32(sec_stream)
                    body_raw = sec_stream.read(body_size)
                    body_stream = io.BytesIO(body_raw)

                    # Parse locals: vec(count: u32, type: ValType)
                    local_entries = decode_vec(
                        body_stream,
                        lambda s: (decode_u32(s), ValType(s.read(1)[0]))
                    )
                    instructions = parse_instructions(body_stream)
                    type_idx = func_types_indices[i]
                    module.functions.append(FunctionDef(type_idx, local_entries, instructions))

            elif sec_id == SectionId.DATA:
                def parse_data(s: io.BytesIO) -> DataSegment:
                    mem_idx = decode_u32(s)
                    offset_expr = parse_expression(s)
                    data_len = decode_u32(s)
                    data_bytes = s.read(data_len)
                    return DataSegment(mem_idx, offset_expr, data_bytes)

                module.data_segments = decode_vec(sec_stream, parse_data)

        return module
