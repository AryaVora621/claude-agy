"""
SynapseDB: Vectorized Volcano Query Execution Engine.
Processes data in columnar RecordBatches (morsel-driven chunks) using
SIMD-style selection vectors, hash joins, and streaming aggregation hash tables.
"""

from typing import Optional, List, Dict, Tuple, Any, Iterator
from synapse.types import DataType, Vector, RecordBatch, SelectionVector
from synapse.parser import ASTNode, BinaryOp, ColumnRef, Literal
from synapse.storage import ColumnarTable
from synapse.planner import (
    LogicalPlanNode,
    LogicalScan,
    LogicalFilter,
    LogicalHashJoin,
    LogicalAggregate,
    LogicalSort,
    LogicalLimit,
    LogicalProject
)


class PhysicalOperator:
    """Base class for all vectorized query execution operators."""

    def open(self) -> None:
        pass

    def next_batch(self) -> Optional[RecordBatch]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PhysicalScan(PhysicalOperator):
    """
    Vectorized Table Scan with Zone Map pruning and vectorized selection filtering.
    """

    def __init__(
        self,
        table: ColumnarTable,
        projected_columns: List[str],
        pushed_predicates: List[Tuple[str, str, Any]]
    ):
        self.table = table
        self.projected_columns = projected_columns
        self.pushed_predicates = pushed_predicates
        self._iterator: Optional[Iterator[RecordBatch]] = None

    def open(self) -> None:
        self._iterator = self.table.scan(
            projection=self.projected_columns,
            predicates=self.pushed_predicates
        )

    def next_batch(self) -> Optional[RecordBatch]:
        if self._iterator is None:
            return None

        for batch in self._iterator:
            # Apply in-batch vectorized filter for the pushed predicates
            if not self.pushed_predicates:
                return batch

            sel = SelectionVector()
            n = len(batch)
            for i in range(n):
                match = True
                for col_name, op, target_val in self.pushed_predicates:
                    val = batch.columns[col_name].get(i)
                    if val is None:
                        match = False
                        break
                    if op == "=" and not (val == target_val):
                        match = False
                        break
                    elif op == "!=" and not (val != target_val):
                        match = False
                        break
                    elif op == ">" and not (val > target_val):
                        match = False
                        break
                    elif op == ">=" and not (val >= target_val):
                        match = False
                        break
                    elif op == "<" and not (val < target_val):
                        match = False
                        break
                    elif op == "<=" and not (val <= target_val):
                        match = False
                        break
                if match:
                    sel.append(i)

            if len(sel) > 0:
                return batch.filter(sel)

        return None

    def close(self) -> None:
        self._iterator = None


class PhysicalFilter(PhysicalOperator):
    """Evaluates arbitrary boolean filter expressions across batches using SelectionVectors."""

    def __init__(self, child: PhysicalOperator, condition: ASTNode):
        self.child = child
        self.condition = condition

    def open(self) -> None:
        self.child.open()

    def next_batch(self) -> Optional[RecordBatch]:
        while True:
            batch = self.child.next_batch()
            if batch is None:
                return None

            sel = SelectionVector()
            for i in range(len(batch)):
                if self._eval_expr(self.condition, batch, i):
                    sel.append(i)

            if len(sel) > 0:
                return batch.filter(sel)

    def _eval_expr(self, expr: ASTNode, batch: RecordBatch, row_idx: int) -> Any:
        if isinstance(expr, Literal):
            return expr.value
        elif isinstance(expr, ColumnRef):
            return batch.columns[expr.name].get(row_idx)
        elif isinstance(expr, BinaryOp):
            if expr.op == "AND":
                return self._eval_expr(expr.left, batch, row_idx) and self._eval_expr(expr.right, batch, row_idx)
            elif expr.op == "OR":
                return self._eval_expr(expr.left, batch, row_idx) or self._eval_expr(expr.right, batch, row_idx)

            left_v = self._eval_expr(expr.left, batch, row_idx)
            right_v = self._eval_expr(expr.right, batch, row_idx)
            if left_v is None or right_v is None:
                return False

            if expr.op == "=":
                return left_v == right_v
            elif expr.op == "!=":
                return left_v != right_v
            elif expr.op == ">":
                return left_v > right_v
            elif expr.op == ">=":
                return left_v >= right_v
            elif expr.op == "<":
                return left_v < right_v
            elif expr.op == "<=":
                return left_v <= right_v

        return False

    def close(self) -> None:
        self.child.close()


class PhysicalHashJoin(PhysicalOperator):
    """
    In-memory Hash Join.
    Builds hash table on the right input key, streams and probes from left input.
    """

    def __init__(
        self,
        left: PhysicalOperator,
        right: PhysicalOperator,
        join_type: str,
        left_key: str,
        right_key: str
    ):
        self.left = left
        self.right = right
        self.join_type = join_type
        self.left_key = left_key
        self.right_key = right_key
        self.hash_table: Dict[Any, List[Dict[str, Any]]] = {}
        self._built = False

    def open(self) -> None:
        self.left.open()
        self.right.open()
        self._built = False
        self.hash_table.clear()

    def _build_hash_table(self) -> None:
        while True:
            batch = self.right.next_batch()
            if batch is None:
                break
            rows = batch.to_rows()
            for r in rows:
                k = r.get(self.right_key)
                if k not in self.hash_table:
                    self.hash_table[k] = []
                self.hash_table[k].append(r)
        self._built = True

    def next_batch(self) -> Optional[RecordBatch]:
        if not self._built:
            self._build_hash_table()

        while True:
            left_batch = self.left.next_batch()
            if left_batch is None:
                return None

            left_rows = left_batch.to_rows()
            joined_rows = []

            for l_row in left_rows:
                k = l_row.get(self.left_key)
                matches = self.hash_table.get(k, [])
                if matches:
                    for r_row in matches:
                        combined = dict(l_row)
                        for rk, rv in r_row.items():
                            if rk not in combined:
                                combined[rk] = rv
                            else:
                                combined[f"right_{rk}"] = rv
                        joined_rows.append(combined)
                elif self.join_type == "LEFT":
                    joined_rows.append(dict(l_row))

            if joined_rows:
                # Materialize joined RecordBatch
                cols = {}
                col_names = list(joined_rows[0].keys())
                for c in col_names:
                    vals = [row.get(c) for row in joined_rows]
                    # Infer data type
                    sample = next((v for v in vals if v is not None), 0)
                    if isinstance(sample, int) and not isinstance(sample, bool):
                        dt = DataType.INT64
                    elif isinstance(sample, float):
                        dt = DataType.FLOAT64
                    elif isinstance(sample, bool):
                        dt = DataType.BOOLEAN
                    else:
                        dt = DataType.VARCHAR
                    cols[c] = Vector(dt, vals)
                return RecordBatch(cols)

    def close(self) -> None:
        self.left.close()
        self.right.close()


class PhysicalAggregate(PhysicalOperator):
    """
    Streaming aggregation hash table.
    Accumulates SUM, COUNT, AVG, MIN, MAX across chunks and emits aggregated batches.
    """

    def __init__(
        self,
        child: PhysicalOperator,
        group_by: List[str],
        aggregates: List[Tuple[str, Optional[str], str]]  # (func, arg_col, alias)
    ):
        self.child = child
        self.group_by = group_by
        self.aggregates = aggregates
        self._emitted = False

    def open(self) -> None:
        self.child.open()
        self._emitted = False

    def next_batch(self) -> Optional[RecordBatch]:
        if self._emitted:
            return None

        # group_key -> list of accumulator state dicts
        # accumulator: {"sum": 0, "count": 0, "min": None, "max": None}
        agg_table: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}

        while True:
            batch = self.child.next_batch()
            if batch is None:
                break

            n = len(batch)
            for i in range(n):
                if self.group_by:
                    key = tuple(batch.columns[g].get(i) for g in self.group_by)
                else:
                    key = ()

                if key not in agg_table:
                    # Initialize accumulators
                    agg_table[key] = [
                        {"sum": 0.0, "count": 0, "min": None, "max": None}
                        for _ in self.aggregates
                    ]

                accs = agg_table[key]
                for agg_idx, (func, arg_col, _) in enumerate(self.aggregates):
                    acc = accs[agg_idx]
                    val = batch.columns[arg_col].get(i) if arg_col else 1

                    if func == "COUNT":
                        if arg_col is None or val is not None:
                            acc["count"] += 1
                    elif val is not None:
                        acc["count"] += 1
                        val_num = float(val)
                        acc["sum"] += val_num
                        if acc["min"] is None or val < acc["min"]:
                            acc["min"] = val
                        if acc["max"] is None or val > acc["max"]:
                            acc["max"] = val

        self._emitted = True

        if not agg_table and not self.group_by:
            # Global aggregate on empty table
            agg_table[()] = [
                {"sum": 0.0, "count": 0, "min": None, "max": None}
                for _ in self.aggregates
            ]

        # Materialize final aggregated RecordBatch
        keys_list = list(agg_table.keys())
        cols: Dict[str, Vector] = {}

        # Add group-by columns
        for g_idx, g_col in enumerate(self.group_by):
            vals = [k[g_idx] for k in keys_list]
            sample = next((v for v in vals if v is not None), None)
            if isinstance(sample, int):
                dt = DataType.INT64
            elif isinstance(sample, float):
                dt = DataType.FLOAT64
            else:
                dt = DataType.VARCHAR
            cols[g_col] = Vector(dt, vals)

        # Add aggregate output columns
        for agg_idx, (func, _, alias) in enumerate(self.aggregates):
            vals = []
            for k in keys_list:
                acc = agg_table[k][agg_idx]
                if func == "COUNT":
                    vals.append(acc["count"])
                elif func == "SUM":
                    vals.append(acc["sum"])
                elif func == "AVG":
                    vals.append(acc["sum"] / acc["count"] if acc["count"] > 0 else None)
                elif func == "MIN":
                    vals.append(acc["min"])
                elif func == "MAX":
                    vals.append(acc["max"])

            sample = next((v for v in vals if v is not None), 0)
            if isinstance(sample, int):
                dt = DataType.INT64
            elif isinstance(sample, float):
                dt = DataType.FLOAT64
            else:
                dt = DataType.VARCHAR
            cols[alias] = Vector(dt, vals)

        return RecordBatch(cols)

    def close(self) -> None:
        self.child.close()


class PhysicalSort(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, order_by: List[Tuple[str, bool]]):
        self.child = child
        self.order_by = order_by
        self._emitted = False

    def open(self) -> None:
        self.child.open()
        self._emitted = False

    def next_batch(self) -> Optional[RecordBatch]:
        if self._emitted:
            return None

        all_rows = []
        while True:
            batch = self.child.next_batch()
            if batch is None:
                break
            all_rows.extend(batch.to_rows())

        self._emitted = True
        if not all_rows:
            return None

        for col, is_desc in reversed(self.order_by):
            all_rows.sort(
                key=lambda r: (r.get(col) is None, r.get(col)),
                reverse=is_desc
            )

        cols = {}
        first_row = all_rows[0]
        for c in first_row.keys():
            vals = [r.get(c) for r in all_rows]
            sample = next((v for v in vals if v is not None), 0)
            if isinstance(sample, int) and not isinstance(sample, bool):
                dt = DataType.INT64
            elif isinstance(sample, float):
                dt = DataType.FLOAT64
            elif isinstance(sample, bool):
                dt = DataType.BOOLEAN
            else:
                dt = DataType.VARCHAR
            cols[c] = Vector(dt, vals)

        return RecordBatch(cols)

    def close(self) -> None:
        self.child.close()


class PhysicalLimit(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, limit: int):
        self.child = child
        self.limit = limit
        self.emitted_count = 0

    def open(self) -> None:
        self.child.open()
        self.emitted_count = 0

    def next_batch(self) -> Optional[RecordBatch]:
        if self.emitted_count >= self.limit:
            return None

        batch = self.child.next_batch()
        if batch is None:
            return None

        remaining = self.limit - self.emitted_count
        if len(batch) <= remaining:
            self.emitted_count += len(batch)
            return batch

        sub_batch = batch.slice(0, remaining)
        self.emitted_count += remaining
        return sub_batch

    def close(self) -> None:
        self.child.close()


class PhysicalProject(PhysicalOperator):
    def __init__(self, child: PhysicalOperator, projections: List[Tuple[str, str]]):
        self.child = child
        self.projections = projections

    def open(self) -> None:
        self.child.open()

    def next_batch(self) -> Optional[RecordBatch]:
        batch = self.child.next_batch()
        if batch is None:
            return None

        new_cols = {}
        for src, alias in self.projections:
            if src in batch.columns:
                new_cols[alias] = batch.columns[src]
        return RecordBatch(new_cols)

    def close(self) -> None:
        self.child.close()


class Engine:
    """
    High-level SQL Execution Engine.
    Coordinates table catalog, query parsing, optimization, and execution.
    """

    def __init__(self):
        self.catalog: Dict[str, ColumnarTable] = {}

    def register_table(self, table: ColumnarTable) -> None:
        self.catalog[table.name] = table

    def execute(self, sql: str) -> RecordBatch:
        """Parse, plan, optimize, and execute SQL query to completion."""
        from synapse.parser import parse_sql
        from synapse.planner import QueryPlanner

        ast = parse_sql(sql)
        planner = QueryPlanner(self.catalog)
        logical_plan = planner.plan(ast)
        physical_plan = self._build_physical_plan(logical_plan)

        physical_plan.open()
        batches = []
        try:
            while True:
                b = physical_plan.next_batch()
                if b is None:
                    break
                batches.append(b)
        finally:
            physical_plan.close()

        if not batches:
            return RecordBatch({})

        if len(batches) == 1:
            return batches[0]

        # Concatenate batches
        first = batches[0]
        merged_cols = {}
        for name in first.names:
            all_vals = []
            for b in batches:
                all_vals.extend(b.columns[name].to_list())
            merged_cols[name] = Vector(first.columns[name].data_type, all_vals)

        return RecordBatch(merged_cols)

    def _build_physical_plan(self, node: LogicalPlanNode) -> PhysicalOperator:
        if isinstance(node, LogicalScan):
            return PhysicalScan(node.table, node.projected_columns, node.pushed_predicates)
        elif isinstance(node, LogicalFilter):
            child_op = self._build_physical_plan(node.input_node)
            return PhysicalFilter(child_op, node.condition)
        elif isinstance(node, LogicalHashJoin):
            left_op = self._build_physical_plan(node.left)
            right_op = self._build_physical_plan(node.right)
            return PhysicalHashJoin(left_op, right_op, node.join_type, node.left_key, node.right_key)
        elif isinstance(node, LogicalAggregate):
            child_op = self._build_physical_plan(node.input_node)
            return PhysicalAggregate(child_op, node.group_by, node.aggregates)
        elif isinstance(node, LogicalSort):
            child_op = self._build_physical_plan(node.input_node)
            return PhysicalSort(child_op, node.order_by)
        elif isinstance(node, LogicalLimit):
            child_op = self._build_physical_plan(node.input_node)
            return PhysicalLimit(child_op, node.limit)
        elif isinstance(node, LogicalProject):
            child_op = self._build_physical_plan(node.input_node)
            return PhysicalProject(child_op, node.projections)

        raise NotImplementedError(f"Unsupported logical node: {type(node)}")
