"""
WasmCore: WebAssembly MVP Stack Machine Interpreter.
Executes standard WASM instructions with structured control flow,
call frame activation, label scopes, operand stack unwinding, and trap protection.
"""

import math
import struct
from typing import List, Optional, Union, Tuple, Any, Callable
from .types import (
    ValType, FuncType, Value, WasmTrap, WasmValidationError,
    to_signed_32, to_unsigned_32, to_signed_64, to_unsigned_64,
    float_to_i32_bits, i32_bits_to_float, double_to_i64_bits, i64_bits_to_double
)
from .opcodes import Opcode, Instruction, MemArg
from .module import FunctionDef
from .instance import WasmInstance, HostFunction


class ControlFrame:
    """Structured control flow frame (block, loop, or if)."""
    __slots__ = (
        "opcode", "block_type", "stack_height",
        "start_ip", "else_ip", "end_ip"
    )

    def __init__(
        self,
        opcode: Opcode,
        block_type: Optional[ValType],
        stack_height: int,
        start_ip: int,
        else_ip: Optional[int] = None,
        end_ip: Optional[int] = None
    ) -> None:
        self.opcode = opcode
        self.block_type = block_type
        self.stack_height = stack_height
        self.start_ip = start_ip
        self.else_ip = else_ip
        self.end_ip = end_ip

    @property
    def target_ip(self) -> int:
        """Branch target IP: loops jump to start, blocks/ifs jump to end."""
        if self.opcode == Opcode.LOOP:
            return self.start_ip
        return self.end_ip if self.end_ip is not None else self.start_ip


class CallFrame:
    """Function activation record on call stack."""
    __slots__ = (
        "func", "func_idx", "locals", "ip",
        "instructions", "block_map", "stack_base", "result_count"
    )

    def __init__(
        self,
        func: FunctionDef,
        func_idx: int,
        locals_list: List[Value],
        instructions: List[Instruction],
        block_map: dict,
        stack_base: int,
        result_count: int
    ) -> None:
        self.func = func
        self.func_idx = func_idx
        self.locals = locals_list
        self.ip = 0
        self.instructions = instructions
        self.block_map = block_map
        self.stack_base = stack_base
        self.result_count = result_count


def precompute_blocks(instructions: List[Instruction]) -> dict:
    """
    Precompute pairing of block/loop/if with their corresponding else and end instructions.
    Returns map of instruction index to (else_ip, end_ip).
    """
    block_map = {}
    control_stack = []

    for ip, instr in enumerate(instructions):
        if instr.opcode in (Opcode.BLOCK, Opcode.LOOP, Opcode.IF):
            control_stack.append((ip, instr.opcode, None))
        elif instr.opcode == Opcode.ELSE:
            if not control_stack:
                raise WasmValidationError("Mismatched else instruction")
            start_ip, op, _ = control_stack[-1]
            if op != Opcode.IF:
                raise WasmValidationError("Else instruction outside of if block")
            control_stack[-1] = (start_ip, op, ip)
        elif instr.opcode == Opcode.END:
            if control_stack:
                start_ip, op, else_ip = control_stack.pop()
                block_map[start_ip] = (else_ip, ip)
                if else_ip is not None:
                    block_map[else_ip] = (None, ip)

    return block_map


# Bitwise helper utilities
def rotl32(val: int, count: int) -> int:
    val &= 0xFFFFFFFF
    count &= 31
    return ((val << count) | (val >> (32 - count))) & 0xFFFFFFFF


def rotr32(val: int, count: int) -> int:
    val &= 0xFFFFFFFF
    count &= 31
    return ((val >> count) | (val << (32 - count))) & 0xFFFFFFFF


def rotl64(val: int, count: int) -> int:
    val &= 0xFFFFFFFFFFFFFFFF
    count &= 63
    return ((val << count) | (val >> (64 - count))) & 0xFFFFFFFFFFFFFFFF


def rotr64(val: int, count: int) -> int:
    val &= 0xFFFFFFFFFFFFFFFF
    count &= 63
    return ((val >> count) | (val << (64 - count))) & 0xFFFFFFFFFFFFFFFF


def clz32(val: int) -> int:
    val &= 0xFFFFFFFF
    if val == 0:
        return 32
    return 32 - val.bit_length()


def ctz32(val: int) -> int:
    val &= 0xFFFFFFFF
    if val == 0:
        return 32
    return (val & -val).bit_length() - 1


def clz64(val: int) -> int:
    val &= 0xFFFFFFFFFFFFFFFF
    if val == 0:
        return 64
    return 64 - val.bit_length()


def ctz64(val: int) -> int:
    val &= 0xFFFFFFFFFFFFFFFF
    if val == 0:
        return 64
    return (val & -val).bit_length() - 1


class WasmInterpreter:
    """
    High-Performance WebAssembly MVP Stack Machine Interpreter.
    """

    def __init__(self, instance: WasmInstance, max_call_depth: int = 1024) -> None:
        self.instance = instance
        self.max_call_depth = max_call_depth
        self.stack: List[Value] = []
        self.call_stack: List[CallFrame] = []
        self.control_stack: List[ControlFrame] = []

        # Precompute block mappings for internal functions
        self.func_block_maps: dict = {}
        for idx, fn in enumerate(instance.functions):
            if isinstance(fn, FunctionDef):
                self.func_block_maps[idx] = precompute_blocks(fn.instructions)

    def push(self, val: Value) -> None:
        self.stack.append(val)

    def pop(self) -> Value:
        if not self.stack:
            raise WasmTrap("Operand stack underflow")
        return self.stack.pop()

    def peek(self) -> Value:
        if not self.stack:
            raise WasmTrap("Operand stack underflow")
        return self.stack[-1]

    def invoke(self, func_name_or_idx: Union[str, int], *args: Any) -> Any:
        """Entry point for executing an exported or internal function."""
        if isinstance(func_name_or_idx, str):
            func_idx = self.instance.get_export(func_name_or_idx)
        else:
            func_idx = func_name_or_idx

        fn = self.instance.functions[func_idx]
        if isinstance(fn, HostFunction):
            typed_args = []
            for i, p_type in enumerate(fn.func_type.params):
                val = args[i] if i < len(args) else 0
                typed_args.append(val if isinstance(val, Value) else Value(p_type, val).to_py())
            res = fn.callable_fn(*typed_args)
            return res

        # FunctionDef
        func_type = self.instance.module.types[fn.type_idx]
        if len(args) != len(func_type.params):
            raise WasmTrap(f"Argument count mismatch: expected {len(func_type.params)}, got {len(args)}")

        # Convert args to initial parameters
        locals_list: List[Value] = []
        for i, p_type in enumerate(func_type.params):
            arg_val = args[i]
            if isinstance(arg_val, Value):
                locals_list.append(arg_val)
            else:
                locals_list.append(Value(p_type, arg_val))

        # Initialize local variables with type default (0)
        for count, val_type in fn.locals:
            for _ in range(count):
                locals_list.append(Value.default_for_type(val_type))

        block_map = self.func_block_maps.get(func_idx, {})
        call_frame = CallFrame(
            func=fn,
            func_idx=func_idx,
            locals_list=locals_list,
            instructions=fn.instructions,
            block_map=block_map,
            stack_base=len(self.stack),
            result_count=len(func_type.results)
        )

        self.call_stack.append(call_frame)
        self._execute()

        # Gather results
        results = []
        for _ in range(len(func_type.results)):
            results.append(self.pop().to_py())
        results.reverse()

        if len(results) == 0:
            return None
        elif len(results) == 1:
            return results[0]
        return tuple(results)

    def _execute(self) -> None:
        """Main bytecode interpretation loop."""
        while self.call_stack:
            frame = self.call_stack[-1]

            if frame.ip >= len(frame.instructions):
                # End of function reached
                self.call_stack.pop()
                continue

            instr = frame.instructions[frame.ip]
            op = instr.opcode
            op_val = instr.operands

            # Advance IP by default
            frame.ip += 1

            # Control Flow
            if op == Opcode.UNREACHABLE:
                raise WasmTrap("unreachable executed")

            elif op == Opcode.NOP:
                pass

            elif op == Opcode.BLOCK:
                else_ip, end_ip = frame.block_map.get(frame.ip - 1, (None, None))
                self.control_stack.append(ControlFrame(
                    opcode=Opcode.BLOCK,
                    block_type=op_val,
                    stack_height=len(self.stack),
                    start_ip=frame.ip - 1,
                    else_ip=else_ip,
                    end_ip=end_ip
                ))

            elif op == Opcode.LOOP:
                else_ip, end_ip = frame.block_map.get(frame.ip - 1, (None, None))
                self.control_stack.append(ControlFrame(
                    opcode=Opcode.LOOP,
                    block_type=op_val,
                    stack_height=len(self.stack),
                    start_ip=frame.ip - 1,
                    else_ip=else_ip,
                    end_ip=end_ip
                ))

            elif op == Opcode.IF:
                cond = self.pop().as_i32()
                else_ip, end_ip = frame.block_map.get(frame.ip - 1, (None, None))
                if cond != 0:
                    ctrl = ControlFrame(
                        opcode=Opcode.IF,
                        block_type=op_val,
                        stack_height=len(self.stack),
                        start_ip=frame.ip - 1,
                        else_ip=else_ip,
                        end_ip=end_ip
                    )
                    self.control_stack.append(ctrl)
                else:
                    if else_ip is not None:
                        ctrl = ControlFrame(
                            opcode=Opcode.IF,
                            block_type=op_val,
                            stack_height=len(self.stack),
                            start_ip=frame.ip - 1,
                            else_ip=else_ip,
                            end_ip=end_ip
                        )
                        self.control_stack.append(ctrl)
                        frame.ip = else_ip + 1
                    elif end_ip is not None:
                        frame.ip = end_ip + 1

            elif op == Opcode.ELSE:
                # Reached end of 'then' branch; skip past 'else' block and its matching end
                if self.control_stack:
                    ctrl = self.control_stack.pop()
                    if ctrl.end_ip is not None:
                        frame.ip = ctrl.end_ip + 1

            elif op == Opcode.END:
                if self.control_stack:
                    self.control_stack.pop()
                else:
                    # Function return
                    self.call_stack.pop()

            elif op == Opcode.BR:
                depth = int(op_val)
                self._branch_to(depth, frame)

            elif op == Opcode.BR_IF:
                cond = self.pop().as_i32()
                if cond != 0:
                    depth = int(op_val)
                    self._branch_to(depth, frame)

            elif op == Opcode.BR_TABLE:
                targets, default_target = op_val
                idx = self.pop().as_i32()
                if 0 <= idx < len(targets):
                    depth = targets[idx]
                else:
                    depth = default_target
                self._branch_to(depth, frame)

            elif op == Opcode.RETURN:
                # Unwind all control frames for this call frame
                while self.control_stack and self.control_stack[-1].stack_height >= frame.stack_base:
                    self.control_stack.pop()
                self.call_stack.pop()

            elif op == Opcode.CALL:
                target_func_idx = int(op_val)
                self._invoke_internal(target_func_idx)

            elif op == Opcode.CALL_INDIRECT:
                type_idx, table_idx = op_val
                elem_idx = self.pop().as_i32()
                table = self.instance.tables[table_idx]
                func_idx = table.get(elem_idx)
                if func_idx is None:
                    raise WasmTrap(f"uninitialized element: table {table_idx} index {elem_idx}")
                # Verify type signature
                expected_type = self.instance.module.types[type_idx]
                target_fn = self.instance.functions[func_idx]
                if isinstance(target_fn, FunctionDef):
                    actual_type = self.instance.module.types[target_fn.type_idx]
                else:
                    actual_type = target_fn.func_type
                if actual_type != expected_type:
                    raise WasmTrap(f"indirect call signature mismatch: expected {expected_type}, got {actual_type}")
                self._invoke_internal(func_idx)

            # Parametric
            elif op == Opcode.DROP:
                self.pop()

            elif op == Opcode.SELECT:
                c = self.pop().as_i32()
                v2 = self.pop()
                v1 = self.pop()
                self.push(v1 if c != 0 else v2)

            # Variables
            elif op == Opcode.LOCAL_GET:
                self.push(frame.locals[int(op_val)])

            elif op == Opcode.LOCAL_SET:
                frame.locals[int(op_val)] = self.pop()

            elif op == Opcode.LOCAL_TEE:
                val = self.peek()
                frame.locals[int(op_val)] = val

            elif op == Opcode.GLOBAL_GET:
                self.push(self.instance.globals[int(op_val)].get())

            elif op == Opcode.GLOBAL_SET:
                self.instance.globals[int(op_val)].set(self.pop())

            # Memory Operations
            elif op in (
                Opcode.I32_LOAD, Opcode.I64_LOAD, Opcode.F32_LOAD, Opcode.F64_LOAD,
                Opcode.I32_LOAD8_S, Opcode.I32_LOAD8_U, Opcode.I32_LOAD16_S, Opcode.I32_LOAD16_U,
                Opcode.I64_LOAD8_S, Opcode.I64_LOAD8_U, Opcode.I64_LOAD16_S, Opcode.I64_LOAD16_U,
                Opcode.I64_LOAD32_S, Opcode.I64_LOAD32_U
            ):
                mem = self.instance.memories[0]
                base_addr = self.pop().as_i32()
                offset = op_val.offset if isinstance(op_val, MemArg) else 0
                ea = base_addr + offset

                if op == Opcode.I32_LOAD:
                    self.push(Value.i32(mem.load_i32(ea)))
                elif op == Opcode.I64_LOAD:
                    self.push(Value.i64(mem.load_i64(ea)))
                elif op == Opcode.F32_LOAD:
                    self.push(Value.f32(mem.load_f32(ea)))
                elif op == Opcode.F64_LOAD:
                    self.push(Value.f64(mem.load_f64(ea)))
                elif op == Opcode.I32_LOAD8_S:
                    self.push(Value.i32(mem.load_i8_s(ea)))
                elif op == Opcode.I32_LOAD8_U:
                    self.push(Value.i32(mem.load_i8_u(ea)))
                elif op == Opcode.I32_LOAD16_S:
                    self.push(Value.i32(mem.load_i16_s(ea)))
                elif op == Opcode.I32_LOAD16_U:
                    self.push(Value.i32(mem.load_i16_u(ea)))
                elif op == Opcode.I64_LOAD8_S:
                    self.push(Value.i64(mem.load_i8_s(ea)))
                elif op == Opcode.I64_LOAD8_U:
                    self.push(Value.i64(mem.load_i8_u(ea)))
                elif op == Opcode.I64_LOAD16_S:
                    self.push(Value.i64(mem.load_i16_s(ea)))
                elif op == Opcode.I64_LOAD16_U:
                    self.push(Value.i64(mem.load_i16_u(ea)))
                elif op == Opcode.I64_LOAD32_S:
                    self.push(Value.i64(mem.load_i32(ea)))
                elif op == Opcode.I64_LOAD32_U:
                    self.push(Value.i64(mem.load_u32(ea)))

            elif op in (
                Opcode.I32_STORE, Opcode.I64_STORE, Opcode.F32_STORE, Opcode.F64_STORE,
                Opcode.I32_STORE8, Opcode.I32_STORE16, Opcode.I64_STORE8, Opcode.I64_STORE16,
                Opcode.I64_STORE32
            ):
                mem = self.instance.memories[0]
                val = self.pop()
                base_addr = self.pop().as_i32()
                offset = op_val.offset if isinstance(op_val, MemArg) else 0
                ea = base_addr + offset

                if op == Opcode.I32_STORE:
                    mem.store_i32(ea, val.as_i32())
                elif op == Opcode.I64_STORE:
                    mem.store_i64(ea, val.as_i64())
                elif op == Opcode.F32_STORE:
                    mem.store_f32(ea, val.as_f32())
                elif op == Opcode.F64_STORE:
                    mem.store_f64(ea, val.as_f64())
                elif op == Opcode.I32_STORE8 or op == Opcode.I64_STORE8:
                    mem.store_i8(ea, val.as_i32() if op == Opcode.I32_STORE8 else val.as_i64())
                elif op == Opcode.I32_STORE16 or op == Opcode.I64_STORE16:
                    mem.store_i16(ea, val.as_i32() if op == Opcode.I32_STORE16 else val.as_i64())
                elif op == Opcode.I64_STORE32:
                    mem.store_i32(ea, val.as_i64() & 0xFFFFFFFF)

            elif op == Opcode.MEMORY_SIZE:
                self.push(Value.i32(self.instance.memories[0].size()))

            elif op == Opcode.MEMORY_GROW:
                delta = self.pop().as_i32()
                self.push(Value.i32(self.instance.memories[0].grow(delta)))

            # Constants
            elif op == Opcode.I32_CONST:
                self.push(Value.i32(op_val))

            elif op == Opcode.I64_CONST:
                self.push(Value.i64(op_val))

            elif op == Opcode.F32_CONST:
                self.push(Value.f32(op_val))

            elif op == Opcode.F64_CONST:
                self.push(Value.f64(op_val))

            # i32 Comparisons
            elif op == Opcode.I32_EQZ:
                self.push(Value.i32(1 if self.pop().as_i32() == 0 else 0))
            elif op == Opcode.I32_EQ:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l == r else 0))
            elif op == Opcode.I32_NE:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l != r else 0))
            elif op == Opcode.I32_LT_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l < r else 0))
            elif op == Opcode.I32_LT_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(1 if l < r else 0))
            elif op == Opcode.I32_GT_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l > r else 0))
            elif op == Opcode.I32_GT_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(1 if l > r else 0))
            elif op == Opcode.I32_LE_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l <= r else 0))
            elif op == Opcode.I32_LE_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(1 if l <= r else 0))
            elif op == Opcode.I32_GE_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(1 if l >= r else 0))
            elif op == Opcode.I32_GE_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(1 if l >= r else 0))

            # i32 Arithmetic
            elif op == Opcode.I32_CLZ:
                self.push(Value.i32(clz32(self.pop().as_i32())))
            elif op == Opcode.I32_CTZ:
                self.push(Value.i32(ctz32(self.pop().as_i32())))
            elif op == Opcode.I32_POPCNT:
                self.push(Value.i32(self.pop().as_u32().bit_count()))
            elif op == Opcode.I32_ADD:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(to_signed_32(l + r)))
            elif op == Opcode.I32_SUB:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(to_signed_32(l - r)))
            elif op == Opcode.I32_MUL:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(to_signed_32(l * r)))
            elif op == Opcode.I32_DIV_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                if l == -2147483648 and r == -1:
                    raise WasmTrap("integer overflow")
                res = int(math.trunc(l / r))
                self.push(Value.i32(res))
            elif op == Opcode.I32_DIV_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                self.push(Value.i32(l // r))
            elif op == Opcode.I32_REM_S:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                if r == -1:
                    self.push(Value.i32(0))
                else:
                    res = l - int(math.trunc(l / r)) * r
                    self.push(Value.i32(res))
            elif op == Opcode.I32_REM_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                self.push(Value.i32(l % r))
            elif op == Opcode.I32_AND:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(l & r))
            elif op == Opcode.I32_OR:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(l | r))
            elif op == Opcode.I32_XOR:
                r, l = self.pop().as_i32(), self.pop().as_i32()
                self.push(Value.i32(l ^ r))
            elif op == Opcode.I32_SHL:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32((l << (r & 31)) & 0xFFFFFFFF))
            elif op == Opcode.I32_SHR_S:
                r, l = self.pop().as_u32(), self.pop().as_i32()
                self.push(Value.i32(l >> (r & 31)))
            elif op == Opcode.I32_SHR_U:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(l >> (r & 31)))
            elif op == Opcode.I32_ROTL:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(rotl32(l, r)))
            elif op == Opcode.I32_ROTR:
                r, l = self.pop().as_u32(), self.pop().as_u32()
                self.push(Value.i32(rotr32(l, r)))

            # i64 Operations
            elif op == Opcode.I64_EQZ:
                self.push(Value.i32(1 if self.pop().as_i64() == 0 else 0))
            elif op == Opcode.I64_EQ:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l == r else 0))
            elif op == Opcode.I64_NE:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l != r else 0))
            elif op == Opcode.I64_LT_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l < r else 0))
            elif op == Opcode.I64_LT_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i32(1 if l < r else 0))
            elif op == Opcode.I64_GT_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l > r else 0))
            elif op == Opcode.I64_GT_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i32(1 if l > r else 0))
            elif op == Opcode.I64_LE_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l <= r else 0))
            elif op == Opcode.I64_LE_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i32(1 if l <= r else 0))
            elif op == Opcode.I64_GE_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i32(1 if l >= r else 0))
            elif op == Opcode.I64_GE_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i32(1 if l >= r else 0))

            elif op == Opcode.I64_CLZ:
                self.push(Value.i64(clz64(self.pop().as_i64())))
            elif op == Opcode.I64_CTZ:
                self.push(Value.i64(ctz64(self.pop().as_i64())))
            elif op == Opcode.I64_POPCNT:
                self.push(Value.i64(self.pop().as_u64().bit_count()))
            elif op == Opcode.I64_ADD:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(to_signed_64(l + r)))
            elif op == Opcode.I64_SUB:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(to_signed_64(l - r)))
            elif op == Opcode.I64_MUL:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(to_signed_64(l * r)))
            elif op == Opcode.I64_DIV_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                if l == -9223372036854775808 and r == -1:
                    raise WasmTrap("integer overflow")
                res = int(math.trunc(l / r))
                self.push(Value.i64(res))
            elif op == Opcode.I64_DIV_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                self.push(Value.i64(l // r))
            elif op == Opcode.I64_REM_S:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                if r == -1:
                    self.push(Value.i64(0))
                else:
                    res = l - int(math.trunc(l / r)) * r
                    self.push(Value.i64(res))
            elif op == Opcode.I64_REM_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                if r == 0:
                    raise WasmTrap("integer divide by zero")
                self.push(Value.i64(l % r))
            elif op == Opcode.I64_AND:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(l & r))
            elif op == Opcode.I64_OR:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(l | r))
            elif op == Opcode.I64_XOR:
                r, l = self.pop().as_i64(), self.pop().as_i64()
                self.push(Value.i64(l ^ r))
            elif op == Opcode.I64_SHL:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i64((l << (r & 63)) & 0xFFFFFFFFFFFFFFFF))
            elif op == Opcode.I64_SHR_S:
                r, l = self.pop().as_u64(), self.pop().as_i64()
                self.push(Value.i64(l >> (r & 63)))
            elif op == Opcode.I64_SHR_U:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i64(l >> (r & 63)))
            elif op == Opcode.I64_ROTL:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i64(rotl64(l, r)))
            elif op == Opcode.I64_ROTR:
                r, l = self.pop().as_u64(), self.pop().as_u64()
                self.push(Value.i64(rotr64(l, r)))

            # Floating Point Arithmetic (f32 and f64)
            elif op == Opcode.F32_ADD:
                r, l = self.pop().as_f32(), self.pop().as_f32()
                self.push(Value.f32(l + r))
            elif op == Opcode.F32_SUB:
                r, l = self.pop().as_f32(), self.pop().as_f32()
                self.push(Value.f32(l - r))
            elif op == Opcode.F32_MUL:
                r, l = self.pop().as_f32(), self.pop().as_f32()
                self.push(Value.f32(l * r))
            elif op == Opcode.F32_DIV:
                r, l = self.pop().as_f32(), self.pop().as_f32()
                self.push(Value.f32(l / r if r != 0 else float('inf')))
            elif op == Opcode.F32_SQRT:
                self.push(Value.f32(math.sqrt(self.pop().as_f32())))
            elif op == Opcode.F32_ABS:
                self.push(Value.f32(abs(self.pop().as_f32())))
            elif op == Opcode.F32_NEG:
                self.push(Value.f32(-self.pop().as_f32()))

            elif op == Opcode.F64_ADD:
                r, l = self.pop().as_f64(), self.pop().as_f64()
                self.push(Value.f64(l + r))
            elif op == Opcode.F64_SUB:
                r, l = self.pop().as_f64(), self.pop().as_f64()
                self.push(Value.f64(l - r))
            elif op == Opcode.F64_MUL:
                r, l = self.pop().as_f64(), self.pop().as_f64()
                self.push(Value.f64(l * r))
            elif op == Opcode.F64_DIV:
                r, l = self.pop().as_f64(), self.pop().as_f64()
                self.push(Value.f64(l / r if r != 0 else float('inf')))
            elif op == Opcode.F64_SQRT:
                self.push(Value.f64(math.sqrt(self.pop().as_f64())))
            elif op == Opcode.F64_ABS:
                self.push(Value.f64(abs(self.pop().as_f64())))
            elif op == Opcode.F64_NEG:
                self.push(Value.f64(-self.pop().as_f64()))

            # Type Conversions
            elif op == Opcode.I32_WRAP_I64:
                self.push(Value.i32(self.pop().as_i64() & 0xFFFFFFFF))
            elif op == Opcode.I64_EXTEND_I32_S:
                self.push(Value.i64(self.pop().as_i32()))
            elif op == Opcode.I64_EXTEND_I32_U:
                self.push(Value.i64(self.pop().as_u32()))
            elif op == Opcode.I32_TRUNC_F32_S:
                v = self.pop().as_f32()
                if math.isnan(v): raise WasmTrap("invalid conversion to integer")
                self.push(Value.i32(int(math.trunc(v))))
            elif op == Opcode.I32_TRUNC_F32_U:
                v = self.pop().as_f32()
                if math.isnan(v): raise WasmTrap("invalid conversion to integer")
                self.push(Value.i32(int(math.trunc(v))))
            elif op == Opcode.I32_TRUNC_F64_S:
                v = self.pop().as_f64()
                if math.isnan(v): raise WasmTrap("invalid conversion to integer")
                self.push(Value.i32(int(math.trunc(v))))
            elif op == Opcode.I32_TRUNC_F64_U:
                v = self.pop().as_f64()
                if math.isnan(v): raise WasmTrap("invalid conversion to integer")
                self.push(Value.i32(int(math.trunc(v))))
            elif op == Opcode.F32_CONVERT_I32_S:
                self.push(Value.f32(float(self.pop().as_i32())))
            elif op == Opcode.F32_CONVERT_I32_U:
                self.push(Value.f32(float(self.pop().as_u32())))
            elif op == Opcode.F64_CONVERT_I32_S:
                self.push(Value.f64(float(self.pop().as_i32())))
            elif op == Opcode.F64_CONVERT_I32_U:
                self.push(Value.f64(float(self.pop().as_u32())))
            elif op == Opcode.F32_DEMOTE_F64:
                self.push(Value.f32(self.pop().as_f64()))
            elif op == Opcode.F64_PROMOTE_F32:
                self.push(Value.f64(self.pop().as_f32()))

            # Bitcast Reinterpretations
            elif op == Opcode.I32_REINTERPRET_F32:
                self.push(Value.i32(float_to_i32_bits(self.pop().as_f32())))
            elif op == Opcode.I64_REINTERPRET_F64:
                self.push(Value.i64(double_to_i64_bits(self.pop().as_f64())))
            elif op == Opcode.F32_REINTERPRET_I32:
                self.push(Value.f32(i32_bits_to_float(self.pop().as_u32())))
            elif op == Opcode.F64_REINTERPRET_I64:
                self.push(Value.f64(i64_bits_to_double(self.pop().as_u64())))

            else:
                raise WasmTrap(f"Unsupported opcode in interpreter: {op.name} (0x{int(op):02X})")

    def _branch_to(self, depth: int, frame: CallFrame) -> None:
        """Branch to target label at depth levels up the control stack."""
        if depth >= len(self.control_stack):
            raise WasmTrap(f"Branch depth {depth} exceeds control stack size {len(self.control_stack)}")

        target_idx = len(self.control_stack) - 1 - depth
        target_ctrl = self.control_stack[target_idx]

        # Preserve block result if block type produces a value
        result_val = None
        if target_ctrl.block_type is not None and len(self.stack) > target_ctrl.stack_height:
            result_val = self.pop()

        # Unwind operand stack to target control frame height
        while len(self.stack) > target_ctrl.stack_height:
            self.pop()

        # Restore result if present
        if result_val is not None:
            self.push(result_val)

        # Remove popped control frames
        while len(self.control_stack) > target_idx:
            popped = self.control_stack.pop()
            if popped == target_ctrl and popped.opcode == Opcode.LOOP:
                # Loops stay active on backward jump
                self.control_stack.append(popped)
                break

        # Set IP to target
        if target_ctrl.opcode == Opcode.LOOP:
            frame.ip = target_ctrl.start_ip + 1
        else:
            frame.ip = (target_ctrl.end_ip + 1) if target_ctrl.end_ip is not None else frame.ip

    def _invoke_internal(self, target_func_idx: int) -> None:
        """Call internal or host function from within bytecode execution."""
        if len(self.call_stack) >= self.max_call_depth:
            raise WasmTrap(f"Call stack exhausted (depth >= {self.max_call_depth})")

        fn = self.instance.functions[target_func_idx]
        if isinstance(fn, HostFunction):
            # Pop arguments in reverse
            args = []
            for _ in range(len(fn.func_type.params)):
                args.append(self.pop().to_py())
            args.reverse()

            res = fn.callable_fn(*args)
            if len(fn.func_type.results) == 1:
                res_type = fn.func_type.results[0]
                self.push(Value(res_type, res))
            elif len(fn.func_type.results) > 1:
                for idx, r_val in enumerate(res):
                    self.push(Value(fn.func_type.results[idx], r_val))
            return

        # FunctionDef
        func_type = self.instance.module.types[fn.type_idx]
        locals_list: List[Value] = []
        # Pop params from operand stack
        for _ in range(len(func_type.params)):
            locals_list.append(self.pop())
        locals_list.reverse()

        # Add function locals
        for count, val_type in fn.locals:
            for _ in range(count):
                locals_list.append(Value.default_for_type(val_type))

        block_map = self.func_block_maps.get(target_func_idx, {})
        new_frame = CallFrame(
            func=fn,
            func_idx=target_func_idx,
            locals_list=locals_list,
            instructions=fn.instructions,
            block_map=block_map,
            stack_base=len(self.stack),
            result_count=len(func_type.results)
        )
        self.call_stack.append(new_frame)
