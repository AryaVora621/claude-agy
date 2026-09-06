"""
Unit tests for WasmCore binary emitter and parser round-trip.
"""

import unittest
from wasmcore.types import ValType, ExportDesc
from wasmcore.opcodes import Opcode
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser


class TestParserEmitter(unittest.TestCase):
    def test_emit_and_parse_simple_add(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32, ValType.I32], [ValType.I32])
        fn = builder.add_function(type_idx, name="add")
        fn.local_get(0)
        fn.local_get(1)
        fn.i32_add()
        fn.end()
        builder.add_export("add", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        self.assertTrue(wasm_bytes.startswith(b"\x00asm\x01\x00\x00\x00"))

        parser = WasmParser(wasm_bytes)
        module = parser.parse()

        self.assertEqual(len(module.types), 1)
        self.assertEqual(module.types[0].params, (ValType.I32, ValType.I32))
        self.assertEqual(module.types[0].results, (ValType.I32,))

        self.assertEqual(len(module.functions), 1)
        fdef = module.functions[0]
        self.assertEqual(fdef.type_idx, 0)
        # Check instructions: local.get 0, local.get 1, i32.add, end
        opcodes = [inst.opcode for inst in fdef.instructions]
        self.assertEqual(opcodes, [Opcode.LOCAL_GET, Opcode.LOCAL_GET, Opcode.I32_ADD, Opcode.END])

        self.assertEqual(len(module.exports), 1)
        self.assertEqual(module.exports[0].name, "add")
        self.assertEqual(module.exports[0].index, 0)

    def test_emit_and_parse_memory_and_data(self):
        builder = WasmModuleBuilder()
        builder.add_memory(min_pages=1, max_pages=4)
        builder.add_data(mem_idx=0, offset=16, data=b"Hello WebAssembly!")

        wasm_bytes = builder.build()
        module = WasmParser(wasm_bytes).parse()

        self.assertEqual(len(module.memories), 1)
        self.assertEqual(module.memories[0].limits.min, 1)
        self.assertEqual(module.memories[0].limits.max, 4)

        self.assertEqual(len(module.data_segments), 1)
        dseg = module.data_segments[0]
        self.assertEqual(dseg.mem_idx, 0)
        self.assertEqual(dseg.data, b"Hello WebAssembly!")
        self.assertEqual(dseg.offset_expr[0].opcode, Opcode.I32_CONST)
        self.assertEqual(dseg.offset_expr[0].operands, 16)


if __name__ == "__main__":
    unittest.main()
