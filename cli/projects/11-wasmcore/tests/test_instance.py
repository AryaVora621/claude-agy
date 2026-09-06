"""
Unit tests for WasmCore instance creation, runtime linking, and host callbacks.
"""

import unittest
from wasmcore.types import ValType, ExportDesc, Limits, Value, WasmTrap
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser
from wasmcore.instance import WasmInstance, TableInstance, GlobalInstance, HostFunction


class TestWasmInstance(unittest.TestCase):
    def test_instantiate_with_globals_and_exports(self):
        builder = WasmModuleBuilder()
        builder.add_global(ValType.I32, mutable=True, init_val=100)
        builder.add_global(ValType.I64, mutable=False, init_val=999999)
        builder.add_export("my_glob", ExportDesc.GLOBAL, 0)
        builder.add_export("immut_glob", ExportDesc.GLOBAL, 1)

        wasm_bytes = builder.build()
        module = WasmParser(wasm_bytes).parse()
        instance = WasmInstance(module)

        g0 = instance.get_export("my_glob")
        self.assertEqual(g0.get().as_i32(), 100)
        g0.set(Value.i32(250))
        self.assertEqual(g0.get().as_i32(), 250)

        g1 = instance.get_export("immut_glob")
        self.assertEqual(g1.get().as_i64(), 999999)
        with self.assertRaises(WasmTrap):
            g1.set(Value.i64(500))

    def test_instantiate_with_memory_and_data(self):
        builder = WasmModuleBuilder()
        builder.add_memory(min_pages=1, max_pages=2)
        builder.add_data(mem_idx=0, offset=32, data=b"WasmCore Embedded Data")
        builder.add_export("memory", ExportDesc.MEM, 0)

        wasm_bytes = builder.build()
        module = WasmParser(wasm_bytes).parse()
        instance = WasmInstance(module)

        mem = instance.get_export("memory")
        read_back = mem.read_bytes(32, len(b"WasmCore Embedded Data"))
        self.assertEqual(read_back, b"WasmCore Embedded Data")

    def test_instantiate_with_imports(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32], [ValType.I32])
        imp_fn_idx = builder.add_import_func("env", "host_square", type_idx)

        # Function calling imported host function
        fn = builder.add_function(type_idx, name="call_square")
        fn.local_get(0)
        fn.call(imp_fn_idx)
        fn.end()
        builder.add_export("call_square", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        module = WasmParser(wasm_bytes).parse()

        # Link host callback
        def square_cb(x: int) -> int:
            return x * x

        instance = WasmInstance(module, imports={"env": {"host_square": square_cb}})
        self.assertEqual(len(instance.functions), 2)
        self.assertIsInstance(instance.functions[0], HostFunction)


if __name__ == "__main__":
    unittest.main()
