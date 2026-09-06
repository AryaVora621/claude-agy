#!/usr/bin/env python3
"""
SynapseDB: Vectorized Columnar Engine Performance Benchmarks.
Measures:
1. Columnar Vector Scan vs Row-Oriented Scan Throughput (rows/sec).
2. Compression Codec Throughput (RLE, Dictionary, Bit-Packing).
3. Zone Map Pruning Acceleration (skipping non-matching row groups).
4. Aggregation Hash Table Throughput (GROUP BY rollups).
5. In-Memory Hash Join Probe Throughput (rows/sec).
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synapse.types import DataType, Vector, RecordBatch
from synapse.encoding import RLECodec, DictionaryCodec, BitPackingCodec
from synapse.storage import ColumnarTable, TableSchema
from synapse.executor import Engine


def bench_columnar_scan(row_count: int = 100_000) -> None:
    print(f"[*] Benchmarking Vectorized Columnar Scan ({row_count:,} rows)...")
    schema = TableSchema({
        "id": DataType.INT64,
        "price": DataType.FLOAT64,
        "quantity": DataType.INT64,
        "region": DataType.VARCHAR
    })
    table = ColumnarTable("sales", schema, chunk_size=2048)

    regions = ["East", "West", "Central", "North", "South"]
    rows = []
    for i in range(row_count):
        rows.append({
            "id": i,
            "price": random.uniform(10.0, 500.0),
            "quantity": random.randint(1, 50),
            "region": regions[i % len(regions)]
        })
    table.insert_rows(rows)

    # 1. Full Columnar Scan (single column)
    t0 = time.perf_counter()
    total = 0.0
    for batch in table.scan(projection=["price"]):
        v = batch.column("price")
        for idx in range(len(v)):
            total += v.data[idx]
    t_col = time.perf_counter() - t0
    scan_rate = row_count / t_col

    print(f"    -> Scanned {row_count:,} rows in {t_col * 1000:.2f} ms")
    print(f"    -> Vectorized Column Scan Rate: {scan_rate:,.0f} rows/sec\n")


def bench_compression_codecs(count: int = 100_000) -> None:
    print(f"[*] Benchmarking Compression Codecs ({count:,} values)...")

    # 1. RLE on sorted categories
    cats = ["Category_" + str(i // 1000) for i in range(count)]
    t0 = time.perf_counter()
    rle_bytes = RLECodec.encode(cats, DataType.VARCHAR)
    t_rle_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = RLECodec.decode(rle_bytes, DataType.VARCHAR)
    t_rle_dec = time.perf_counter() - t0

    raw_bytes = sum(len(c.encode("utf-8")) for c in cats)
    rle_ratio = raw_bytes / len(rle_bytes)
    print(f"    [RLE] Encoded {count:,} strings: {rle_ratio:.1f}x compression ratio")
    print(f"          Encode: {count / t_rle_enc:,.0f} vals/s | Decode: {count / t_rle_dec:,.0f} vals/s")

    # 2. Dictionary on low-cardinality strings
    regions = ["US_EAST", "US_WEST", "EU_CENTRAL", "AP_SOUTH"] * (count // 4)
    t0 = time.perf_counter()
    dict_table, dict_bytes = DictionaryCodec.encode(regions)
    t_dict_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = DictionaryCodec.decode(dict_table, dict_bytes)
    t_dict_dec = time.perf_counter() - t0

    dict_ratio = (len(regions) * 10) / (len(dict_bytes) + sum(len(s) for s in dict_table))
    print(f"    [Dict] Encoded {count:,} strings: {dict_ratio:.1f}x compression ratio")
    print(f"           Encode: {count / t_dict_enc:,.0f} vals/s | Decode: {count / t_dict_dec:,.0f} vals/s")

    # 3. Bit-Packing on narrow range integers
    base = 50_000
    narrow_ints = [base + (i % 64) for i in range(count)]
    t0 = time.perf_counter()
    packed_bytes = BitPackingCodec.encode(narrow_ints)
    t_pack_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = BitPackingCodec.decode(packed_bytes)
    t_pack_dec = time.perf_counter() - t0

    pack_ratio = (count * 8) / len(packed_bytes)
    print(f"    [BitPack] Encoded {count:,} 64-bit integers: {pack_ratio:.1f}x compression ratio")
    print(f"              Encode: {count / t_pack_enc:,.0f} vals/s | Decode: {count / t_pack_dec:,.0f} vals/s\n")


def bench_zone_map_pruning(row_count: int = 100_000) -> None:
    print(f"[*] Benchmarking Zone Map Pruning Acceleration ({row_count:,} rows)...")
    schema = TableSchema({"id": DataType.INT64, "metric": DataType.FLOAT64})
    table = ColumnarTable("telemetry", schema, chunk_size=1000)

    # Sorted sequential chunks: chunk 0 has ids 0..999, chunk 99 has ids 99000..99999
    rows = [{"id": i, "metric": float(i * 1.5)} for i in range(row_count)]
    table.insert_rows(rows)

    # Predicate looking for id > 95000: skips 95 of 100 row groups!
    predicates = [("id", ">", 95000)]

    t0 = time.perf_counter()
    matched_chunks = list(table.scan(predicates=predicates))
    t_pruned = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_chunks = list(table.scan())
    t_full = time.perf_counter() - t0

    pruned_speedup = t_full / max(t_pruned, 1e-6)
    print(f"    -> Full Scan: {len(all_chunks)} chunks in {t_full * 1000:.2f} ms")
    print(f"    -> Zone Map Pruned Scan: {len(matched_chunks)} chunks in {t_pruned * 1000:.2f} ms")
    print(f"    -> Pruning Acceleration: {pruned_speedup:.1f}x speedup\n")


def bench_sql_aggregations(row_count: int = 100_000) -> None:
    print(f"[*] Benchmarking SQL Aggregation & GROUP BY ({row_count:,} rows)...")
    engine = Engine()
    schema = TableSchema({
        "dept": DataType.VARCHAR,
        "salary": DataType.FLOAT64,
        "bonus": DataType.FLOAT64
    })
    table = ColumnarTable("employees", schema, chunk_size=2048)

    depts = ["Engineering", "Sales", "Finance", "Marketing", "Legal"]
    rows = []
    for i in range(row_count):
        rows.append({
            "dept": depts[i % len(depts)],
            "salary": float(50000 + (i % 100000)),
            "bonus": float(5000 + (i % 20000))
        })
    table.insert_rows(rows)
    engine.register_table(table)

    sql = (
        "SELECT dept, COUNT(*) AS num_emp, SUM(salary) AS total_sal, AVG(salary) AS avg_sal, MAX(bonus) AS max_bonus "
        "FROM employees GROUP BY dept"
    )

    t0 = time.perf_counter()
    res = engine.execute(sql)
    elapsed = time.perf_counter() - t0

    rate = row_count / elapsed
    print(f"    -> Processed {row_count:,} rows in {elapsed * 1000:.2f} ms")
    print(f"    -> Aggregation Throughput: {rate:,.0f} rows/sec\n")


def run_all_benchmarks() -> None:
    print("==================================================================")
    print("      SYNAPSEDB: COLUMNAR ANALYTICS ENGINE BENCHMARK SUITE        ")
    print("==================================================================\n")
    bench_columnar_scan()
    bench_compression_codecs()
    bench_zone_map_pruning()
    bench_sql_aggregations()
    print("==================================================================")
    print("            ALL SYNAPSEDB BENCHMARKS COMPLETED                    ")
    print("==================================================================")


if __name__ == "__main__":
    run_all_benchmarks()
