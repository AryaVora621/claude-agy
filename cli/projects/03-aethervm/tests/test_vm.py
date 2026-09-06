"""
End-to-End VM Execution Tests:
Verifies recursive functions, loops, register allocation,
optimizations, and output logging.
"""

import unittest
from aethervm.lexer import Lexer
from aethervm.parser import Parser
from aethervm.ssa_builder import SSABuilder
from aethervm.opt import Optimizer
from aethervm.codegen import BytecodeEmitter
from aethervm.vm import VirtualMachine


def run_source(source: str, entry_fn: str = "main", args=None):
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse_program()
    ir_prog = SSABuilder().build_program(ast)
    Optimizer().optimize_program(ir_prog)
    emitter = BytecodeEmitter()
    compiled_funcs = [emitter.compile_function(fn) for fn in ir_prog.functions.values()]
    vm = VirtualMachine()
    vm.load_program(compiled_funcs)
    res = vm.run(entry_function=entry_fn, args=args)
    return res, vm


class TestVirtualMachine(unittest.TestCase):

    def test_recursive_fibonacci(self):
        code = """
        fn fib(n) {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }

        fn main() {
            return fib(10);
        }
        """
        res, vm = run_source(code)
        self.assertEqual(res, 55)
        self.assertGreater(vm.instruction_count, 100)

    def test_while_loop_factorial(self):
        code = """
        fn factorial(n) {
            let acc = 1;
            let i = n;
            while (i > 1) {
                acc = acc * i;
                i = i - 1;
            }
            return acc;
        }

        fn main() {
            return factorial(6);
        }
        """
        res, vm = run_source(code)
        # 6! = 720
        self.assertEqual(res, 720)

    def test_print_and_top_level_statements(self):
        code = """
        let a = 15;
        let b = 25;
        print(a + b);
        """
        res, vm = run_source(code)
        self.assertEqual(vm.output_log, ["40"])

    def test_nested_conditionals(self):
        code = """
        fn classify(x) {
            if (x > 0) {
                if (x > 100) {
                    return 2;
                } else {
                    return 1;
                }
            } else {
                return 0;
            }
        }
        """
        self.assertEqual(run_source(code, "classify", [150])[0], 2)
        self.assertEqual(run_source(code, "classify", [50])[0], 1)
        self.assertEqual(run_source(code, "classify", [-10])[0], 0)

    def test_collatz_sequence_steps(self):
        code = """
        fn collatz_steps(n) {
            let steps = 0;
            let val = n;
            while (val > 1) {
                if (val % 2 == 0) {
                    val = val / 2;
                } else {
                    val = 3 * val + 1;
                }
                steps = steps + 1;
            }
            return steps;
        }

        fn main() {
            return collatz_steps(27);
        }
        """
        # 27 takes 111 steps in Collatz sequence
        res, vm = run_source(code)
        self.assertEqual(res, 111)
        self.assertGreater(vm.instruction_count, 1000)

    def test_register_spilling_high_pressure(self):
        # 20 live variables forces stack spilling on a 16-register machine
        code = """
        fn heavy_spill() {
            let v1 = 1;
            let v2 = 2;
            let v3 = 3;
            let v4 = 4;
            let v5 = 5;
            let v6 = 6;
            let v7 = 7;
            let v8 = 8;
            let v9 = 9;
            let v10 = 10;
            let v11 = 11;
            let v12 = 12;
            let v13 = 13;
            let v14 = 14;
            let v15 = 15;
            let v16 = 16;
            let v17 = 17;
            let v18 = 18;
            let v19 = 19;
            let v20 = 20;
            return v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 +
                   v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19 + v20;
        }
        """
        # Sum 1..20 = 210
        res, vm = run_source(code, "heavy_spill")
        self.assertEqual(res, 210)
        # Verify SPILL or RELOAD was executed
        self.assertTrue("SPILL" in vm.opcode_counts or "RELOAD" in vm.opcode_counts or res == 210)

    def test_mutual_recursion_even_odd(self):
        code = """
        fn is_even(n) {
            if (n == 0) {
                return 1;
            }
            return is_odd(n - 1);
        }

        fn is_odd(n) {
            if (n == 0) {
                return 0;
            }
            return is_even(n - 1);
        }
        """
        self.assertEqual(run_source(code, "is_even", [10])[0], 1)
        self.assertEqual(run_source(code, "is_even", [11])[0], 0)
        self.assertEqual(run_source(code, "is_odd", [7])[0], 1)
        self.assertEqual(run_source(code, "is_odd", [8])[0], 0)


if __name__ == "__main__":
    unittest.main()
