"""
Unit tests for WasmCore visualizer, disassembler, and debugger.
"""

import unittest
from wasmcore.types import ValType, ExportDesc
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser
from wasmcore.instance import WasmInstance
from wasmcore.visualizer import WasmDisassembler, MemoryViewer, WasmDebugger


class TestWasmVisualizer(unittest.TestCase):
    def test_disassembler_output(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32, ValType.I32], [ValType.I32])
        fn = builder.add_function(type_idx, name="add")
        fn.local_get(0).local_get(1).i32_add().end()
        builder.add_export("add", ExportDesc.FUNC, fn.func_idx)

        module = WasmParser(builder.build()).parse()
        text = WasmDisassembler.disassemble_module(module)

        self.assertIn("WasmCore Module Disassembly", text)
        self.assertIn("local.get", text)
        self.assertIn("i32.add", text)
        self.assertIn("export \"add\"", text)

    def test_memory_hexdump(self):
        builder = WasmModuleBuilder()
        builder.add_memory(min_pages=1)
        builder.add_data(0, 0, b"Hello Hexdump Test 1234")

        module = WasmParser(builder.build()).parse()
        instance = WasmInstance(module)

        dump = MemoryViewer.hexdump(instance.memories[0], offset=0, length=32)
        self.assertIn("Hello Hexdump", dump)
        self.assertIn("Offset (h)", dump)

    def test_debugger_snapshot(self):
        builder = WasmModuleBuilder()
        builder.add_memory(min_pages=1)
        type_idx = builder.add_type([ValType.I32], [ValType.I32])
        fn = builder.add_function(type_idx, name="inc")
        fn.local_get(0).i32_const(1).i32_add().end()
        builder.add_export("inc", ExportDesc.FUNC, fn.func_idx)

        instance = WasmInstance(WasmParser(builder.build()).parse())
        dbg = WasmDebugger(instance)

        dbg.set_breakpoint("inc", 1)
        self.assertEqual(len(dbg.breakpoints), 1)

        snapshot = dbg.snapshot_state()
        self.assertIn("WasmCore Debugger Snapshot", snapshot)
        self.assertIn("Operand Stack", snapshot)
        self.assertIn("Linear Memory", snapshot)


if __name__ == "__main__":
    unittest.main()
