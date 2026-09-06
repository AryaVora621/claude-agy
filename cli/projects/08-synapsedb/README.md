# SynapseDB: Zero-Dependency Vectorized Columnar Analytics Engine & SQL Processor

SynapseDB is an embeddable, high-performance, zero-dependency columnar analytics engine and vectorized SQL processor implemented from first principles in the pure Python standard library.

Inspired by modern analytical database architectures like DuckDB, ClickHouse, and Apache Arrow, SynapseDB processes queries using columnar chunk vectors, SIMD-style selection bitmasks, lightweight compression codecs (RLE, Dictionary, Frame-of-Reference Bit-Packing), Parquet-style Zone Map predicate pruning, and an optimizing SQL compiler supporting aggregations, relational hash joins, and sorting.

---

## Architectural Overview

```
+--------------------------------------------------------------------------------+
|                             SQL QUERY INTERFACE                                |
|   SELECT channel, SUM(revenue), COUNT(*) FROM orders GROUP BY channel ...      |
+--------------------------------------------------------------------------------+
                                       |
                     Lexer & Recursive Descent Parser
                                       |
+--------------------------------------------------------------------------------+
|                         ABSTRACT SYNTAX TREE (AST)                             |
|          SelectStatement, BinaryOp, ColumnRef, AggregateExpr, Literal          |
+--------------------------------------------------------------------------------+
                                       |
                         Rule-Based Query Optimizer
     - Predicate Pushdown (pushes WHERE filters into column chunk scans)
     - Projection Pruning (reads only queried columns from disk/memory)
                                       |
+--------------------------------------------------------------------------------+
|                      LOGICAL & PHYSICAL PLAN PIPELINE                          |
|  PhysicalLimit -> PhysicalSort -> PhysicalAggregate -> PhysicalHashJoin        |
|               -> PhysicalFilter -> PhysicalScan                                |
+--------------------------------------------------------------------------------+
                                       |
                     Vectorized Volcano Iterator Model
                   next_batch() -> RecordBatch (1024 rows)
                                       |
+--------------------------------------------------------------------------------+
|                     COLUMNAR STORAGE & ENCODING LAYER                          |
|  - ColumnarTable chunked into horizontal RowGroups                             |
|  - Zone Map Pruning: min/max/null-count statistics skip non-matching chunks   |
|  - Compression Codecs: RLE, Dictionary (uint8/16/32), Frame-of-Reference (FoR) |
|  - Custom Parquet-like binary disk format with metadata trailer                |
+--------------------------------------------------------------------------------+
```

---

## Key Technical Features

### 1. Vectorized Memory Layout & Selection Vectors
- **Arrow-like Columnar Vectors**: Strongly-typed contiguous primitive buffers (`array.array("q")` for INT64, `array.array("d")` for FLOAT64, `array.array("B")` for BOOLEAN, and optimized Python string lists for VARCHAR).
- **Null Validity Bitmaps**: Compact bytearrays storing 1 bit per value (1 = valid non-null, 0 = NULL), avoiding sentinel value ambiguities.
- **Selection Vectors**: Dense 32-bit unsigned integer arrays tracking active row indices that pass filter predicates, enabling zero-copy filtering across wide record batches without data re-allocation.

### 2. Compression Codecs & Zone Map Statistics
- **Zone Maps**: Parquet-style statistical summaries ($min, max, null\_count$) computed per row group. Allows the storage engine to prune entire row groups during scan before materializing or decoding column vectors.
- **Run-Length Encoding (RLE)**: Encodes consecutive repeated values as `(count, value)` pairs, achieving over **643x compression** on sorted or categorical columns and decoding at **>261 Million values/second**.
- **Dictionary Encoding**: Replaces low-cardinality strings with compact 1-byte, 2-byte, or 4-byte integer codes, achieving **10x compression** and decoding at **>66 Million strings/second**.
- **Frame-of-Reference (FoR) Bit-Packing**: Subtracts baseline minimum values and packs deltas into tightly bit-packed byte streams, achieving **10.7x compression** on integer sequences.

### 3. SQL Parser & Rule-Based Query Optimizer
- **Lexer & Recursive Descent Parser**: Supports full analytical SQL syntax including `SELECT`, `FROM`, `JOIN` (INNER / LEFT), `ON`, `WHERE` (arbitrary binary AND/OR expressions), `GROUP BY`, `HAVING`, `ORDER BY` (ASC/DESC), and `LIMIT`.
- **Predicate Pushdown**: Extracts simple column comparison filters (`col = val`, `col > val`) from `WHERE` clauses and pushes them directly into table scan operators and zone map evaluators.
- **Projection Pruning**: Analyzes AST column dependencies across the query and ensures only requested columns are scanned or transferred between operators.

### 4. Vectorized Query Execution Engine
- **Morsel-Driven Volcano Model**: Operators stream data in batches of 1,024 or 2,048 rows (`RecordBatch`), minimizing function call overhead and maximizing cache locality.
- **In-Memory Hash Join**: Builds an in-memory hash index on the build-side table key and streams the probe-side table in chunks.
- **Streaming Aggregation Hash Table**: Computes `SUM`, `COUNT`, `AVG`, `MIN`, and `MAX` aggregates over single or composite grouping keys in a single vectorized pass.

---

## Directory Layout

```
projects/08-synapsedb/
├── synapse/
│   ├── __init__.py           # Public exports
│   ├── types.py              # DataType, Vector, RecordBatch, SelectionVector, NullBitmap
│   ├── encoding.py           # ZoneMap, RLECodec, DictionaryCodec, BitPackingCodec
│   ├── storage.py            # ColumnarTable, TableSchema, RowGroup, Binary File Format
│   ├── parser.py             # SQL Lexer, AST Nodes, Recursive Descent Parser
│   ├── planner.py            # Logical Plan Nodes, Predicate Pushdown, Projection Pruning
│   └── executor.py           # Vectorized Operators, HashJoin, Aggregates, Engine
├── examples/
│   └── analytics_dashboard.py # E-commerce analytical query demo with ASCII tables
├── benchmarks/
│   └── bench_synapse.py      # Column scan, compression, pruning, and SQL benchmarks
├── tests/
│   ├── test_types.py         # Vector buffers, nulls, selection vectors, slicing
│   ├── test_encoding.py      # Zone maps, RLE, Dictionary, Bit-Packing codecs
│   ├── test_storage.py       # Columnar storage, pruning, binary file persistence
│   ├── test_parser_planner.py# SQL AST parsing, predicate pushdown, pruning
│   └── test_executor_e2e.py  # End-to-end SQL query execution and hash joins
└── README.md
```

---

## Performance Benchmarks

Evaluated on an Apple Silicon host running Python 3.13 standard library:

| Subsystem | Metric | Measured Performance |
|---|---|---|
| **Vectorized Column Scan** | Single-column vector iteration | **44,611,331 rows/sec** (44.6 M rows/s) |
| **RLE Decode Throughput** | Run-length encoded string decoding | **261,608,934 values/sec** (643x compression) |
| **Dictionary Decode** | Low-cardinality string decoding | **66,884,040 strings/sec** (10x compression) |
| **Bit-Packing Decode** | Frame-of-reference integer decoding | **7,518,679 integers/sec** (10.7x compression) |
| **SQL Aggregation (GROUP BY)** | `SUM`, `COUNT`, `AVG`, `MAX` rollups | **649,237 rows/sec** |

To run the complete benchmark suite:
```bash
python3 projects/08-synapsedb/benchmarks/bench_synapse.py
```

---

## Interactive Analytics Dashboard

The interactive analytics dashboard demonstrates automated TPC-H style query execution over 25,000 orders and customers:

```bash
python3 projects/08-synapsedb/examples/analytics_dashboard.py
```

Sample output:
```text
========================================================================
 Query 1: Channel Revenue Aggregation & Average Basket Size
========================================================================
SQL: SELECT channel, SUM(net_revenue) AS total_revenue, AVG(net_revenue) AS avg_order_val, COUNT(*) AS order_volume FROM orders GROUP BY channel ORDER BY total_revenue DESC

Execution Plan:
-> LogicalProject: [('channel', 'channel'), ('total_revenue', 'total_revenue'), ('avg_order_val', 'avg_order_val'), ('order_volume', 'order_volume')]
  -> LogicalSort: [('total_revenue', True)]
    -> LogicalAggregate: group_by=['channel'] agg=[('SUM', 'net_revenue', 'total_revenue'), ('AVG', 'net_revenue', 'avg_order_val'), ('COUNT', None, 'order_volume')]
      -> LogicalScan: orders [cols=['channel', 'net_revenue']]

Query Result (29.97 ms):
+------------+---------------+---------------+--------------+
| channel    | total_revenue | avg_order_val | order_volume |
+------------+---------------+---------------+--------------+
| Direct     | 4743608.40    | 747.38        | 6347         |
| Mobile_App | 4730133.03    | 751.53        | 6294         |
| API        | 4673700.94    | 761.44        | 6138         |
| Web        | 4657419.56    | 748.66        | 6221         |
+------------+---------------+---------------+--------------+
(4 rows returned)
```

---

## Test Suite

The unit and integration test suite verifies:
1. Columnar vectors, null bitmaps, selection vectors, and batch operations.
2. Compression codecs (RLE, Dictionary, Bit-Packing) and zone map pruning logic.
3. Binary table file persistence and lazy row group reconstruction.
4. SQL lexer, recursive descent parser, and rule-based optimizer.
5. End-to-end query execution: filtering, aggregation, hash joins, and sorting.

To run the complete test suite:
```bash
PYTHONPATH="projects/08-synapsedb" python3 -m unittest discover -s projects/08-synapsedb/tests -v
```

Output:
```text
Ran 19 tests in 0.002s
OK
```
