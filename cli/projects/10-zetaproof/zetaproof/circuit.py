"""
ZetaProof: Arithmetic Circuit DSL and Compiler.
Provides:
- Circuit builder for expressing zero-knowledge computational statements.
- Variable, LinearExpression with operator overloading (+, -, *, ==).
- Automatic flattening of multiplications into Rank-1 constraints.
- Boolean checks, equality assertions, and range checks.
- Canonical variable ordering (public variables first, followed by private witness).
- Witness generation and R1CS compilation.
"""

from typing import Dict, List, Tuple, Union, Optional, Callable, Sequence, Set
from .field import FieldElement, BN254_SCALAR_FIELD
from .r1cs import Constraint, R1CS


class Variable:
    """
    Named wire or variable in an arithmetic circuit.
    """
    __slots__ = ("id", "name", "is_public", "circuit")

    def __init__(self, var_id: int, name: str, is_public: bool, circuit: "Circuit") -> None:
        self.id = var_id
        self.name = name
        self.is_public = is_public
        self.circuit = circuit

    def __add__(self, other: Union[int, FieldElement, "Variable", "LinearExpression"]) -> "LinearExpression":
        return LinearExpression.from_variable(self) + other

    def __radd__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return LinearExpression.from_variable(self) + other

    def __sub__(self, other: Union[int, FieldElement, "Variable", "LinearExpression"]) -> "LinearExpression":
        return LinearExpression.from_variable(self) - other

    def __rsub__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return LinearExpression.from_constant(other, self.circuit) - self

    def __mul__(self, other: Union[int, FieldElement, "Variable", "LinearExpression"]) -> "LinearExpression":
        return LinearExpression.from_variable(self) * other

    def __rmul__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return LinearExpression.from_variable(self) * other

    def __neg__(self) -> "LinearExpression":
        return -LinearExpression.from_variable(self)

    def __repr__(self) -> str:
        return f"Var({self.name}, id={self.id})"


class LinearExpression:
    """
    Linear combination of variables: sum_i (c_i * v_i).
    Variable ID 0 represents the constant 1.
    """
    __slots__ = ("terms", "circuit")

    def __init__(self, terms: Dict[int, FieldElement], circuit: "Circuit") -> None:
        self.circuit = circuit
        self.terms: Dict[int, FieldElement] = {}
        for var_id, coeff in terms.items():
            if not coeff.is_zero():
                self.terms[var_id] = coeff

    @classmethod
    def from_variable(cls, var: Variable) -> "LinearExpression":
        p = var.circuit.p
        return cls({var.id: FieldElement.one(p)}, var.circuit)

    @classmethod
    def from_constant(cls, val: Union[int, FieldElement], circuit: "Circuit") -> "LinearExpression":
        fe = val if isinstance(val, FieldElement) else FieldElement(val, circuit.p)
        return cls({0: fe}, circuit)

    def is_constant(self) -> bool:
        """
        True if expression has only the constant term (var_id 0) or is empty.
        """
        return all(var_id == 0 for var_id in self.terms.keys())

    def get_constant_value(self) -> FieldElement:
        return self.terms.get(0, FieldElement.zero(self.circuit.p))

    def evaluate(self, witness: Sequence[FieldElement]) -> FieldElement:
        total = FieldElement.zero(self.circuit.p)
        for var_id, coeff in self.terms.items():
            if var_id < len(witness):
                total += coeff * witness[var_id]
        return total

    def __add__(self, other: Union[int, FieldElement, Variable, "LinearExpression"]) -> "LinearExpression":
        if isinstance(other, (int, FieldElement)):
            other = LinearExpression.from_constant(other, self.circuit)
        elif isinstance(other, Variable):
            other = LinearExpression.from_variable(other)

        new_terms = dict(self.terms)
        for var_id, coeff in other.terms.items():
            new_terms[var_id] = new_terms.get(var_id, FieldElement.zero(self.circuit.p)) + coeff
        return LinearExpression(new_terms, self.circuit)

    def __radd__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return self.__add__(other)

    def __sub__(self, other: Union[int, FieldElement, Variable, "LinearExpression"]) -> "LinearExpression":
        if isinstance(other, (int, FieldElement)):
            other = LinearExpression.from_constant(other, self.circuit)
        elif isinstance(other, Variable):
            other = LinearExpression.from_variable(other)

        new_terms = dict(self.terms)
        for var_id, coeff in other.terms.items():
            new_terms[var_id] = new_terms.get(var_id, FieldElement.zero(self.circuit.p)) - coeff
        return LinearExpression(new_terms, self.circuit)

    def __rsub__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return LinearExpression.from_constant(other, self.circuit) - self

    def __neg__(self) -> "LinearExpression":
        new_terms = {var_id: -coeff for var_id, coeff in self.terms.items()}
        return LinearExpression(new_terms, self.circuit)

    def __mul__(self, other: Union[int, FieldElement, Variable, "LinearExpression"]) -> "LinearExpression":
        if isinstance(other, (int, FieldElement)):
            fe = other if isinstance(other, FieldElement) else FieldElement(other, self.circuit.p)
            new_terms = {var_id: coeff * fe for var_id, coeff in self.terms.items()}
            return LinearExpression(new_terms, self.circuit)

        if isinstance(other, Variable):
            other = LinearExpression.from_variable(other)

        # If either is constant, scalar multiply without creating a multiplication gate
        if self.is_constant():
            return other * self.get_constant_value()
        if other.is_constant():
            return self * other.get_constant_value()

        # Both non-linear: lower to R1CS multiplication gate
        return self.circuit.multiply(self, other)

    def __rmul__(self, other: Union[int, FieldElement]) -> "LinearExpression":
        return self.__mul__(other)

    def __repr__(self) -> str:
        parts = []
        for var_id, coeff in sorted(self.terms.items()):
            name = self.circuit.vars[var_id].name if var_id < len(self.circuit.vars) else f"v{var_id}"
            if var_id == 0:
                parts.append(f"{coeff.val}")
            else:
                parts.append(f"{coeff.val}*{name}" if coeff.val != 1 else f"{name}")
        return " + ".join(parts) if parts else "0"


class Circuit:
    """
    High-level arithmetic circuit compiler.
    Tracks public/private inputs, auxiliary wires, constraints, and gate solvers.
    Guarantees canonical variable ordering:
    - Index 0: ~one
    - Indices 1..l-1: Public inputs (in declaration order)
    - Indices l..n-1: Private inputs & auxiliary wires
    """

    def __init__(self, p: int = BN254_SCALAR_FIELD) -> None:
        self.p = p
        self.vars: List[Variable] = []
        self.public_input_names: List[str] = []
        self.private_input_names: List[str] = []

        # Map name -> Variable
        self.var_map: Dict[str, Variable] = {}

        # List of raw R1CS constraints (using internal var IDs)
        self.constraints: List[Constraint] = []

        # Solvers for auxiliary wires: list of (var_id, solver_fn)
        self.solvers: List[Tuple[int, Callable[[List[FieldElement]], FieldElement]]] = []

        # Variable 0: Constant ~one
        self._one = self._alloc_var("~one", is_public=True)

    @property
    def one(self) -> LinearExpression:
        return LinearExpression.from_variable(self._one)

    def _alloc_var(self, name: str, is_public: bool) -> Variable:
        var_id = len(self.vars)
        var = Variable(var_id, name, is_public, self)
        self.vars.append(var)
        self.var_map[name] = var
        return var

    def public_input(self, name: str) -> Variable:
        """
        Declare a public input variable (known to both prover and verifier).
        """
        if name in self.var_map:
            raise ValueError(f"Variable {name} already declared")
        self.public_input_names.append(name)
        return self._alloc_var(name, is_public=True)

    def private_input(self, name: str) -> Variable:
        """
        Declare a private input / witness variable (known only to the prover).
        """
        if name in self.var_map:
            raise ValueError(f"Variable {name} already declared")
        self.private_input_names.append(name)
        return self._alloc_var(name, is_public=False)

    def multiply(self, a: LinearExpression, b: LinearExpression) -> LinearExpression:
        """
        Allocate an internal multiplication gate: c = a * b.
        Emits the constraint (a · s) * (b · s) = (c · s).
        """
        gate_num = len(self.solvers)
        out_var = self._alloc_var(f"~mul_{gate_num}", is_public=False)

        # Enforce a * b = out_var
        constraint = Constraint(
            a.terms,
            b.terms,
            {out_var.id: FieldElement.one(self.p)}
        )
        self.constraints.append(constraint)

        # Solver function evaluates wire value from known witness values
        def solver_fn(w: List[FieldElement]) -> FieldElement:
            return a.evaluate(w) * b.evaluate(w)

        self.solvers.append((out_var.id, solver_fn))
        return LinearExpression.from_variable(out_var)

    def assert_equal(
        self,
        a: Union[int, FieldElement, Variable, LinearExpression],
        b: Union[int, FieldElement, Variable, LinearExpression]
    ) -> None:
        """
        Enforce a == b by adding constraint (a - b) * 1 = 0.
        """
        if not isinstance(a, LinearExpression):
            a = LinearExpression.from_variable(a) if isinstance(a, Variable) else LinearExpression.from_constant(a, self)
        if not isinstance(b, LinearExpression):
            b = LinearExpression.from_variable(b) if isinstance(b, Variable) else LinearExpression.from_constant(b, self)

        diff = a - b
        constraint = Constraint(
            diff.terms,
            {0: FieldElement.one(self.p)},
            {}
        )
        self.constraints.append(constraint)

    def assert_zero(self, expr: Union[int, FieldElement, Variable, LinearExpression]) -> None:
        self.assert_equal(expr, 0)

    def assert_boolean(self, var: Union[Variable, LinearExpression]) -> None:
        """
        Constrain variable b to be binary: b * (1 - b) == 0.
        """
        if isinstance(var, Variable):
            expr = LinearExpression.from_variable(var)
        else:
            expr = var
        # expr * (1 - expr) == 0
        inv_expr = LinearExpression.from_constant(1, self) - expr
        prod = expr * inv_expr
        self.assert_equal(prod, 0)

    def range_check(self, expr: Union[Variable, LinearExpression], num_bits: int) -> List[Variable]:
        """
        Enforce 0 <= expr < 2^num_bits by decomposing expr into num_bits boolean variables.
        Returns the list of allocated bit variables [b_0, b_1, ..., b_{n-1}].
        """
        bits: List[Variable] = []
        bit_sum = LinearExpression.from_constant(0, self)

        for i in range(num_bits):
            b_var = self._alloc_var(f"~bit_{len(self.vars)}_{i}", is_public=False)
            self.assert_boolean(b_var)
            bits.append(b_var)
            weight = 1 << i
            bit_sum = bit_sum + (b_var * weight)

        # Solver decomposes the value of expr into bits
        if isinstance(expr, Variable):
            target_expr = LinearExpression.from_variable(expr)
        else:
            target_expr = expr

        def bit_solver_factory(bit_idx: int):
            def solver(w: List[FieldElement]) -> FieldElement:
                val = target_expr.evaluate(w).val
                bit_val = (val >> bit_idx) & 1
                return FieldElement(bit_val, self.p)
            return solver

        for i, b_var in enumerate(bits):
            self.solvers.append((b_var.id, bit_solver_factory(i)))

        self.assert_equal(bit_sum, expr)
        return bits

    def _build_canonical_permutation(self) -> Tuple[Dict[int, int], List[str]]:
        """
        Construct bijection perm: old_var_id -> canonical_var_id
        Canonical ordering guarantees:
        - Index 0: ~one
        - Indices 1..l-1: Public inputs
        - Indices l..n-1: Private inputs and auxiliary variables
        """
        remap: Dict[int, int] = {}
        canonical_names: List[str] = []

        # 1. ~one (always ID 0)
        remap[self._one.id] = 0
        canonical_names.append("~one")

        # 2. Public inputs in declaration order
        for name in self.public_input_names:
            old_id = self.var_map[name].id
            remap[old_id] = len(canonical_names)
            canonical_names.append(name)

        # 3. Private inputs in declaration order
        for name in self.private_input_names:
            old_id = self.var_map[name].id
            remap[old_id] = len(canonical_names)
            canonical_names.append(name)

        # 4. Auxiliary wires in allocation order
        named_set = set(self.public_input_names) | set(self.private_input_names) | {"~one"}
        for var in self.vars:
            if var.name not in named_set:
                remap[var.id] = len(canonical_names)
                canonical_names.append(var.name)

        return remap, canonical_names

    def solve_witness(
        self,
        public_inputs: Optional[Dict[str, Union[int, FieldElement]]] = None,
        private_inputs: Optional[Dict[str, Union[int, FieldElement]]] = None
    ) -> List[FieldElement]:
        """
        Compute full witness vector s in canonical variable order.
        """
        pub = public_inputs or {}
        priv = private_inputs or {}

        # Validate required inputs
        for name in self.public_input_names:
            if name not in pub:
                raise ValueError(f"Missing public input: '{name}'")
        for name in self.private_input_names:
            if name not in priv:
                raise ValueError(f"Missing private input: '{name}'")

        # Internal witness indexed by original internal var IDs
        internal_witness = [FieldElement.zero(self.p) for _ in range(len(self.vars))]
        internal_witness[0] = FieldElement.one(self.p)

        for name in self.public_input_names:
            v_id = self.var_map[name].id
            internal_witness[v_id] = FieldElement(pub[name], self.p)

        for name in self.private_input_names:
            v_id = self.var_map[name].id
            internal_witness[v_id] = FieldElement(priv[name], self.p)

        for out_var_id, solver_fn in self.solvers:
            internal_witness[out_var_id] = solver_fn(internal_witness)

        # Reorder into canonical layout
        remap, _ = self._build_canonical_permutation()
        canonical_witness = [FieldElement.zero(self.p) for _ in range(len(self.vars))]
        for old_id, val in enumerate(internal_witness):
            canonical_witness[remap[old_id]] = val

        return canonical_witness

    def to_r1cs(self) -> R1CS:
        """
        Compile circuit into an R1CS object with canonical variable layout.
        """
        remap, canonical_names = self._build_canonical_permutation()
        num_public = 1 + len(self.public_input_names)

        canonical_constraints: List[Constraint] = []
        for c in self.constraints:
            new_a = {remap[v]: coeff for v, coeff in c.a.items()}
            new_b = {remap[v]: coeff for v, coeff in c.b.items()}
            new_c = {remap[v]: coeff for v, coeff in c.c.items()}
            canonical_constraints.append(Constraint(new_a, new_b, new_c))

        return R1CS(
            constraints=canonical_constraints,
            num_variables=len(self.vars),
            num_public_inputs=num_public,
            variable_names=canonical_names,
            public_var_ids=list(range(num_public)),
            p=self.p
        )
