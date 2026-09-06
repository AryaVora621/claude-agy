"""
SynapseDB: Zero-Dependency Vectorized Columnar Analytics Engine & SQL Processor.
"""

from synapse.types import DataType, Vector, RecordBatch, SelectionVector, NullBitmap
from synapse.encoding import ZoneMap, RLECodec, DictionaryCodec, BitPackingCodec
from synapse.storage import ColumnarTable, TableSchema, RowGroup
from synapse.parser import parse_sql, SelectStatement
from synapse.planner import QueryPlanner, LogicalPlanNode
from synapse.executor import Engine, PhysicalOperator

__all__ = [
    "DataType",
    "Vector",
    "RecordBatch",
    "SelectionVector",
    "NullBitmap",
    "ZoneMap",
    "RLECodec",
    "DictionaryCodec",
    "BitPackingCodec",
    "ColumnarTable",
    "TableSchema",
    "RowGroup",
    "parse_sql",
    "SelectStatement",
    "QueryPlanner",
    "LogicalPlanNode",
    "Engine",
    "PhysicalOperator",
]
