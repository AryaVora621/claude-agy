"""
Unit tests for WasmCore Interpreter and Execution Model.
Verifies arithmetic, loops, conditionals, recursion, memory operations, and host callbacks.
"""

import unittest
from wasmcore.types import ValType, ExportDesc, Limits, WasmTrap
from wasmcore.opcodes import Opcode, MemArg
from wasmcore.emitter import WasmModuleBuilder
from wasmcore.parser import WasmParser
from wasmcore.instance import WasmInstance
from wasmcore.interpreter import WasmInterpreter


class TestWasmInterpreter(unittest.TestCase):
    def test_arithmetic_execution(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32, ValType.I32], [ValType.I32])

        # add
        fn_add = builder.add_function(type_idx, name="add")
        fn_add.local_get(0).local_get(1).i32_add().end()
        builder.add_export("add", ExportDesc.FUNC, fn_add.func_idx)

        # sub
        fn_sub = builder.add_function(type_idx, name="sub")
        fn_sub.local_get(0).local_get(1).i32_sub().end()
        builder.add_export("sub", ExportDesc.FUNC, fn_sub.func_idx)

        # mul
        fn_mul = builder.add_function(type_idx, name="mul")
        fn_mul.local_get(0).local_get(1).i32_mul().end()
        builder.add_export("mul", ExportDesc.FUNC, fn_mul.func_idx)

        # div_s
        fn_div = builder.add_function(type_idx, name="div_s")
        fn_div.local_get(0).local_get(1).i32_div_s().end()
        builder.add_export("div_s", ExportDesc.FUNC, fn_div.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        self.assertEqual(interp.invoke("add", 15, 27), 42)
        self.assertEqual(interp.invoke("sub", 50, 8), 42)
        self.assertEqual(interp.invoke("mul", 6, 7), 42)
        self.assertEqual(interp.invoke("div_s", 84, 2), 42)

        with self.assertRaises(WasmTrap):
            interp.invoke("div_s", 10, 0)

    def test_recursive_fibonacci(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32], [ValType.I32])

        # fib(n): if n <= 1 return n else return fib(n-1) + fib(n-2)
        fn = builder.add_function(type_idx, name="fib")
        fn.local_get(0)
        fn.i32_const(1)
        fn.i32_le_s()
        fn.if_(ValType.I32)
        fn.local_get(0)
        fn.else_()
        fn.local_get(0)
        fn.i32_const(1)
        fn.i32_sub()
        fn.call(fn.func_idx)
        fn.local_get(0)
        fn.i32_const(2)
        fn.i32_sub()
        fn.call(fn.func_idx)
        fn.i32_add()
        fn.end()
        fn.end()
        builder.add_export("fib", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        self.assertEqual(interp.invoke("fib", 0), 0)
        self.assertEqual(interp.invoke("fib", 1), 1)
        self.assertEqual(interp.invoke("fib", 2), 1)
        self.assertEqual(interp.invoke("fib", 7), 13)
        self.assertEqual(interp.invoke("fib", 10), 55)

    def test_loop_counter_sum(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32], [ValType.I32])

        # sum_to_n(n):
        # i = 0, acc = 0
        # loop:
        #   if i == n break
        #   i += 1
        #   acc += i
        #   br 0
        fn = builder.add_function(type_idx, name="sum_to_n")
        fn.add_locals(1, ValType.I32)  # local 1: i
        fn.add_locals(1, ValType.I32)  # local 2: acc

        fn.block()
        fn.loop()
        # if i >= n break
        fn.local_get(1)
        fn.local_get(0)
        fn.i32_ge_s()
        fn.br_if(1)  # break out of block

        # i += 1
        fn.local_get(1)
        fn.i32_const(1)
        fn.i32_add()
        fn.local_set(1)

        # acc += i
        fn.local_get(2)
        fn.local_get(1)
        fn.i32_add()
        fn.local_set(2)

        # repeat loop
        fn.br(0)
        fn.end()  # end loop
        fn.end()  # end block

        fn.local_get(2)
        fn.end()
        builder.add_export("sum_to_n", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        # 1 + 2 + ... + 10 = 55
        self.assertEqual(interp.invoke("sum_to_n", 10), 55)
        # 1 + ... + 100 = 5050
        self.assertEqual(interp.invoke("sum_to_n", 100), 5050)

    def test_memory_load_store_in_bytecode(self):
        builder = WasmModuleBuilder()
        builder.add_memory(min_pages=1, max_pages=1)

        type_idx = builder.add_type([ValType.I32, ValType.I32], [ValType.I32])
        # write_and_read(addr, val): store val at addr, read back, return val * 2
        fn = builder.add_function(type_idx, name="mem_test")
        fn.local_get(0)
        fn.local_get(1)
        fn.i32_store(offset=0)
        # load back
        fn.local_get(0)
        fn.i32_load(offset=0)
        fn.i32_const(2)
        fn.i32_mul()
        fn.end()
        builder.add_export("mem_test", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        res = interp.invoke("mem_test", 64, 123)
        self.assertEqual(res, 246)
        # Verify directly in linear memory
        self.assertEqual(instance.memories[0].load_i32(64), 123)

    def test_host_function_linking_and_call(self):
        builder = WasmModuleBuilder()
        type_idx = builder.add_type([ValType.I32], [ValType.I32])
        imp_idx = builder.add_import_func("env", "triple", type_idx)

        fn = builder.add_function(type_idx, name="call_triple_plus_one")
        fn.local_get(0)
        fn.call(imp_idx)
        fn.i32_const(1)
        fn.i32_add()
        fn.end()
        builder.add_export("call_triple_plus_one", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        module = WasmParser(wasm_bytes).parse()

        # Host callback
        def host_triple(x: int) -> int:
            return x * 3

        instance = WasmInstance(module, imports={"env": {"triple": host_triple}})
        interp = WasmInterpreter(instance)

        self.assertEqual(interp.invoke("call_triple_plus_one", 10), 31)

    def test_bitwise_and_rotations(self):
        builder = WasmModuleBuilder()
        t_i32 = builder.add_type([ValType.I32], [ValType.I32])
        t_i64 = builder.add_type([ValType.I64], [ValType.I64])

        # clz32
        fn_clz = builder.add_function(t_i32, name="clz32")
        fn_clz.local_get(0).emit(Opcode.I32_CLZ).end()
        builder.add_export("clz32", ExportDesc.FUNC, fn_clz.func_idx)

        # popcnt32
        fn_pop = builder.add_function(t_i32, name="popcnt32")
        fn_pop.local_get(0).emit(Opcode.I32_POPCNT).end()
        builder.add_export("popcnt32", ExportDesc.FUNC, fn_pop.func_idx)

        # popcnt64
        fn_pop64 = builder.add_function(t_i64, name="popcnt64")
        fn_pop64.local_get(0).emit(Opcode.I64_POPCNT).end()
        builder.add_export("popcnt64", ExportDesc.FUNC, fn_pop64.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        self.assertEqual(interp.invoke("clz32", 0), 32)
        self.assertEqual(interp.invoke("clz32", 1), 31)
        self.assertEqual(interp.invoke("clz32", 0x80000000), 0)

        self.assertEqual(interp.invoke("popcnt32", 0b1011001), 4)
        self.assertEqual(interp.invoke("popcnt64", 0xFFFFFFFFFFFFFFFF), 64)

    def test_f64_math_and_bitcast(self):
        builder = WasmModuleBuilder()
        t_f64 = builder.add_type([ValType.F64, ValType.F64], [ValType.F64])
        fn = builder.add_function(t_f64, name="f64_hypot")
        # hypot(a, b) = sqrt(a*a + b*b)
        fn.local_get(0).local_get(0).emit(Opcode.F64_MUL)
        fn.local_get(1).local_get(1).emit(Opcode.F64_MUL)
        fn.emit(Opcode.F64_ADD)
        fn.emit(Opcode.F64_SQRT)
        fn.end()
        builder.add_export("f64_hypot", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        hypot = interp.invoke("f64_hypot", 3.0, 4.0)
        self.assertAlmostEqual(hypot, 5.0, places=6)

    def test_br_table_dispatch(self):
        builder = WasmModuleBuilder()
        t_i32 = builder.add_type([ValType.I32], [ValType.I32])
        fn = builder.add_function(t_i32, name="switch_test")

        # Nested blocks for table jump
        # block 0 (outer), block 1, block 2
        fn.block(ValType.I32)
        fn.block()
        fn.block()
        fn.block()
        fn.local_get(0)
        fn.emit(Opcode.BR_TABLE, ([0, 1, 2], 2))  # table targets
        fn.end()  # end block 3: target 0
        fn.i32_const(100)
        fn.br(2)
        fn.end()  # end block 2: target 1
        fn.i32_const(200)
        fn.br(1)
        fn.end()  # end block 1: target 2 / default
        fn.i32_const(300)
        fn.end()  # end block 0: result
        fn.end()  # end func
        builder.add_export("switch_test", ExportDesc.FUNC, fn.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        self.assertEqual(interp.invoke("switch_test", 0), 100)
        self.assertEqual(interp.invoke("switch_test", 1), 200)
        self.assertEqual(interp.invoke("switch_test", 2), 300)
        self.assertEqual(interp.invoke("switch_test", 99), 300)  # default case

    def test_traps_protection(self):
        builder = WasmModuleBuilder()
        t_void = builder.add_type([], [])
        fn_unreach = builder.add_function(t_void, name="trap_unreachable")
        fn_unreach.emit(Opcode.UNREACHABLE).end()
        builder.add_export("trap_unreachable", ExportDesc.FUNC, fn_unreach.func_idx)

        wasm_bytes = builder.build()
        instance = WasmInstance(WasmParser(wasm_bytes).parse())
        interp = WasmInterpreter(instance)

        with self.assertRaises(WasmTrap):
            interp.invoke("trap_unreachable")


if __name__ == "__main__":
    unittest.main()
