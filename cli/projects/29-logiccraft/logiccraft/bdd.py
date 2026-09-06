"""Reduced Ordered Binary Decision Diagrams (ROBDD) Engine.

Features:
1. Canonical Boolean function representation under fixed variable ordering.
2. Shannon expansion with unique table node sharing (O(1) equivalence checking).
3. If-Then-Else (ITE) operator with memoized computed table.
4. Model counting (satisfying assignments out of 2^N) and witness extraction.
5. Infix Boolean expression parser converting strings to canonical BDD roots.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union


# Terminal node constants
FALSE_ID: int = 0
TRUE_ID: int = 1


@dataclass(frozen=True)
class BDDNode:
    """Internal ROBDD node representing Shannon decomposition."""
    id: int
    var_idx: int     # Variable index in global ordering (0 <= var_idx < num_vars)
    low: int         # Negative cofactor node id (var = 0)
    high: int        # Positive cofactor node id (var = 1)


class BDDManager:
    """Manages an ROBDD graph with global unique table and ITE computed cache."""

    def __init__(self, var_names: Sequence[str] | None = None) -> None:
        self.var_names: List[str] = list(var_names) if var_names else []
        self.var_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.var_names)}

        # Nodes dictionary: id -> BDDNode
        self.nodes: Dict[int, BDDNode] = {
            FALSE_ID: BDDNode(id=FALSE_ID, var_idx=999999, low=FALSE_ID, high=FALSE_ID),
            TRUE_ID: BDDNode(id=TRUE_ID, var_idx=999999, low=TRUE_ID, high=TRUE_ID),
        }
        self.next_node_id: int = 2

        # Unique table: (var_idx, low_id, high_id) -> node_id
        self.unique_table: Dict[Tuple[int, int, int], int] = {}

        # Computed table for ITE memoization: (f_id, g_id, h_id) -> result_id
        self.computed_table: Dict[Tuple[int, int, int], int] = {}

    def get_or_add_var(self, name: str) -> int:
        """Get index of variable, registering it at the end of ordering if new."""
        if name not in self.var_to_idx:
            idx = len(self.var_names)
            self.var_names.append(name)
            self.var_to_idx[name] = idx
            return idx
        return self.var_to_idx[name]

    def var(self, name: str) -> int:
        """Create or get a BDD representing the single variable 'name'."""
        idx = self.get_or_add_var(name)
        return self.make_node(idx, FALSE_ID, TRUE_ID)

    @property
    def constant_false(self) -> int:
        return FALSE_ID

    @property
    def constant_true(self) -> int:
        return TRUE_ID

    def make_node(self, var_idx: int, low: int, high: int) -> int:
        """Get or create an ROBDD node applying reduction rules.

        Reduction Rule 1: low == high -> return low (eliminates redundant tests).
        Reduction Rule 2: If (var_idx, low, high) exists in unique table, reuse it.
        """
        if low == high:
            return low

        key = (var_idx, low, high)
        if key in self.unique_table:
            return self.unique_table[key]

        node_id = self.next_node_id
        self.next_node_id += 1
        node = BDDNode(id=node_id, var_idx=var_idx, low=low, high=high)
        self.nodes[node_id] = node
        self.unique_table[key] = node_id
        return node_id

    def ite(self, f: int, g: int, h: int) -> int:
        """Compute ITE(F, G, H) = F * G + ~F * H with dynamic programming."""
        # Terminal cases
        if f == TRUE_ID:
            return g
        if f == FALSE_ID:
            return h
        if g == TRUE_ID and h == FALSE_ID:
            return f
        if g == h:
            return g

        key = (f, g, h)
        if key in self.computed_table:
            return self.computed_table[key]

        # Determine top variable index across F, G, H
        var_f = self.nodes[f].var_idx
        var_g = self.nodes[g].var_idx
        var_h = self.nodes[h].var_idx
        top_var = min(var_f, var_g, var_h)

        # Compute positive and negative cofactors
        f_low = self.nodes[f].low if var_f == top_var else f
        f_high = self.nodes[f].high if var_f == top_var else f

        g_low = self.nodes[g].low if var_g == top_var else g
        g_high = self.nodes[g].high if var_g == top_var else g

        h_low = self.nodes[h].low if var_h == top_var else h
        h_high = self.nodes[h].high if var_h == top_var else h

        # Recursive steps
        t = self.ite(f_high, g_high, h_high)
        e = self.ite(f_low, g_low, h_low)

        res = self.make_node(top_var, e, t)
        self.computed_table[key] = res
        return res

    def not_(self, f: int) -> int:
        """Logical NOT: ~F = ITE(F, 0, 1)."""
        return self.ite(f, FALSE_ID, TRUE_ID)

    def and_(self, f: int, g: int) -> int:
        """Logical AND: F & G = ITE(F, G, 0)."""
        return self.ite(f, g, FALSE_ID)

    def or_(self, f: int, g: int) -> int:
        """Logical OR: F | G = ITE(F, 1, g)."""
        return self.ite(f, TRUE_ID, g)

    def xor_(self, f: int, g: int) -> int:
        """Logical XOR: F ^ G = ITE(F, ~G, G)."""
        return self.ite(f, self.not_(g), g)

    def nand_(self, f: int, g: int) -> int:
        """Logical NAND: ~(F & G)."""
        return self.not_(self.and_(f, g))

    def nor_(self, f: int, g: int) -> int:
        """Logical NOR: ~(F | G)."""
        return self.not_(self.or_(f, g))

    def xnor_(self, f: int, g: int) -> int:
        """Logical XNOR: ~(F ^ G)."""
        return self.not_(self.xor_(f, g))

    def equiv(self, f: int, g: int) -> bool:
        """Constant-time formal equivalence check between two Boolean functions."""
        return f == g

    def evaluate(self, root: int, assignment: Dict[str, bool]) -> bool:
        """Evaluate BDD under a variable truth assignment."""
        curr = root
        while curr not in (FALSE_ID, TRUE_ID):
            node = self.nodes[curr]
            var_name = self.var_names[node.var_idx]
            val = assignment.get(var_name, False)
            curr = node.high if val else node.low
        return curr == TRUE_ID

    def sat_count(self, root: int, total_vars: int | None = None) -> int:
        """Count exact number of satisfying assignments out of 2^N."""
        if total_vars is None:
            total_vars = len(self.var_names)

        memo: Dict[int, int] = {}

        def count_rec(node_id: int) -> int:
            if node_id == FALSE_ID:
                return 0
            if node_id == TRUE_ID:
                return 1
            if node_id in memo:
                return memo[node_id]

            node = self.nodes[node_id]
            low_node = self.nodes[node.low]
            high_node = self.nodes[node.high]

            # Account for skipped variables in ordering
            diff_low = (low_node.var_idx if node.low > 1 else total_vars) - node.var_idx - 1
            diff_high = (high_node.var_idx if node.high > 1 else total_vars) - node.var_idx - 1

            cnt_low = count_rec(node.low) * (1 << max(0, diff_low))
            cnt_high = count_rec(node.high) * (1 << max(0, diff_high))

            res = cnt_low + cnt_high
            memo[node_id] = res
            return res

        if root == FALSE_ID:
            return 0
        if root == TRUE_ID:
            return 1 << total_vars

        first_var = self.nodes[root].var_idx
        return count_rec(root) * (1 << first_var)

    def any_sat(self, root: int) -> Optional[Dict[str, bool]]:
        """Find one satisfying variable assignment (witness), or None if unsatisfiable."""
        if root == FALSE_ID:
            return None
        if root == TRUE_ID:
            return {}

        assignment: Dict[str, bool] = {}
        curr = root
        while curr not in (FALSE_ID, TRUE_ID):
            node = self.nodes[curr]
            var_name = self.var_names[node.var_idx]
            if node.high != FALSE_ID:
                assignment[var_name] = True
                curr = node.high
            else:
                assignment[var_name] = False
                curr = node.low
        return assignment

    def node_count(self, root: int) -> int:
        """Count total unique nodes reachable from root."""
        visited: Set[int] = set()

        def dfs(node_id: int) -> None:
            if node_id in visited or node_id in (FALSE_ID, TRUE_ID):
                return
            visited.add(node_id)
            node = self.nodes[node_id]
            dfs(node.low)
            dfs(node.high)

        dfs(root)
        return len(visited)

    def parse_expr(self, expr_str: str) -> int:
        """Parse Boolean infix expression into BDD root.

        Supported operators (in order of precedence):
          ~ (NOT)
          & (AND)
          ^ (XOR)
          | (OR)
        Parentheses () are supported.
        """
        tokens = self._tokenize(expr_str)
        parser = _ExprParser(self, tokens)
        return parser.parse()

    def _tokenize(self, s: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace():
                i += 1
            elif ch in ("(", ")", "~", "&", "|", "^"):
                tokens.append(ch)
                i += 1
            elif ch.isalnum() or ch in ("_", "."):
                start = i
                while i < len(s) and (s[i].isalnum() or s[i] in ("_", ".")):
                    i += 1
                tokens.append(s[start:i])
            else:
                raise ValueError(f"Unexpected character in expression: {ch}")
        return tokens


class _ExprParser:
    """Recursive descent parser for Boolean expressions."""

    def __init__(self, bdd: BDDManager, tokens: List[str]) -> None:
        self.bdd = bdd
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if expected and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self) -> int:
        res = self.expr_or()
        if self.pos < len(self.tokens):
            raise ValueError(f"Trailing tokens: {self.tokens[self.pos:]}")
        return res

    def expr_or(self) -> int:
        res = self.expr_xor()
        while self.peek() == "|":
            self.consume("|")
            rhs = self.expr_xor()
            res = self.bdd.or_(res, rhs)
        return res

    def expr_xor(self) -> int:
        res = self.expr_and()
        while self.peek() == "^":
            self.consume("^")
            rhs = self.expr_and()
            res = self.bdd.xor_(res, rhs)
        return res

    def expr_and(self) -> int:
        res = self.factor()
        while self.peek() == "&":
            self.consume("&")
            rhs = self.factor()
            res = self.bdd.and_(res, rhs)
        return res

    def factor(self) -> int:
        if self.peek() == "~":
            self.consume("~")
            return self.bdd.not_(self.factor())
        if self.peek() == "(":
            self.consume("(")
            res = self.expr_or()
            self.consume(")")
            return res
        tok = self.consume()
        if tok == "0":
            return self.bdd.constant_false
        if tok == "1":
            return self.bdd.constant_true
        return self.bdd.var(tok)
