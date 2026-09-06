"""
ApexMatch: High-Frequency Exchange & Matching Engine Performance Benchmarks.
Measures order insertion, cancellation, crossing throughput, tick-to-trade latency percentiles,
and binary ITCH/OUCH wire codec speed.
"""

import time
import random
from typing import List
from apexmatch.types import Side, OrderType, price_to_int
from apexmatch.order import OrderNode
from apexmatch.order_book import OrderBook
from apexmatch.matching_engine import MatchingEngine
from apexmatch.protocol import BinaryCodec, OUCHEnterOrder, ITCHAddOrder
from apexmatch.risk import RiskGate, RiskConfig


def bench_order_book_insert_cancel():
    print("\n--- 1. OrderBook L3 In-Memory Insert & Cancel ---")
    book = OrderBook("AAPL")
    num_orders = 50_000

    base_price = price_to_int(150.00)
    orders = [
        OrderNode(
            order_id=i + 1,
            participant_id="MM1",
            symbol="AAPL",
            side=Side.BUY if i % 2 == 0 else Side.SELL,
            price=base_price + (i % 200) * (10 if i % 2 == 0 else -10),
            qty=100
        )
        for i in range(num_orders)
    ]

    # Insert benchmark
    t0 = time.perf_counter()
    for o in orders:
        book.add_order(o)
    elapsed_insert = time.perf_counter() - t0
    inserts_per_sec = num_orders / elapsed_insert
    print(f"L3 Order Insertion: {num_orders:,} orders in {elapsed_insert*1000:.2f} ms ({inserts_per_sec:,.0f} inserts/sec)")

    # Cancel benchmark
    t0 = time.perf_counter()
    for o in orders:
        book.cancel_order(o.order_id)
    elapsed_cancel = time.perf_counter() - t0
    cancels_per_sec = num_orders / elapsed_cancel
    print(f"L3 Order Cancellation: {num_orders:,} cancels in {elapsed_cancel*1000:.2f} ms ({cancels_per_sec:,.0f} cancels/sec)")


def bench_continuous_matching():
    print("\n--- 2. Continuous Matching Engine Throughput ---")
    engine = MatchingEngine("MSFT")
    num_pairs = 25_000

    base_price = price_to_int(400.00)

    # 1. Populate passive asks
    for i in range(num_pairs):
        ask = OrderNode(
            order_id=i + 1,
            participant_id=f"Maker_{i % 10}",
            symbol="MSFT",
            side=Side.SELL,
            price=base_price + (i % 50) * 100,  # 50 price levels
            qty=100
        )
        engine.process_order(ask)

    # 2. Aggressive crossing bids
    bids = [
        OrderNode(
            order_id=num_pairs + i + 1,
            participant_id=f"Taker_{i % 10}",
            symbol="MSFT",
            side=Side.BUY,
            price=base_price + 5000,  # Crosses all asks
            qty=100
        )
        for i in range(num_pairs)
    ]

    t0 = time.perf_counter()
    trade_count = 0
    for b in bids:
        _, trades, _ = engine.process_order(b)
        trade_count += len(trades)
    elapsed_match = time.perf_counter() - t0
    matches_per_sec = num_pairs / elapsed_match
    trades_per_sec = trade_count / elapsed_match

    print(f"Aggressive Matching: {num_pairs:,} incoming orders in {elapsed_match*1000:.2f} ms ({matches_per_sec:,.0f} orders/sec)")
    print(f"Executed Trades:     {trade_count:,} fills generated ({trades_per_sec:,.0f} trades/sec)")


def bench_tick_to_trade_latency():
    print("\n--- 3. Tick-to-Trade Latency Distribution ---")
    engine = MatchingEngine("NVDA")
    base_p = price_to_int(120.00)

    # Seed book with 5,000 resting orders
    for i in range(5000):
        engine.process_order(OrderNode(i + 1, "MM", "NVDA", Side.SELL, base_p + (i % 20) * 100, 100))

    trials = 10_000
    latencies_us: List[float] = []

    for i in range(trials):
        taker = OrderNode(50000 + i, "Taker", "NVDA", Side.BUY, base_p + 1000, 100)
        t_start = time.perf_counter_ns()
        engine.process_order(taker)
        t_end = time.perf_counter_ns()
        latencies_us.append((t_end - t_start) / 1000.0)

    latencies_us.sort()
    p50 = latencies_us[int(trials * 0.50)]
    p90 = latencies_us[int(trials * 0.90)]
    p99 = latencies_us[int(trials * 0.99)]
    p999 = latencies_us[int(trials * 0.999)]

    print(f"Latency Percentiles ({trials:,} executions):")
    print(f"  p50  (median):  {p50:.2f} µs")
    print(f"  p90:            {p90:.2f} µs")
    print(f"  p99:            {p99:.2f} µs")
    print(f"  p99.9:          {p999:.2f} µs")


def bench_binary_protocol():
    print("\n--- 4. Binary ITCH/OUCH Protocol Wire Codec ---")
    codec = BinaryCodec()
    count = 100_000

    ouch_msg = OUCHEnterOrder(
        order_id=987654321,
        side="B",
        qty=500,
        symbol="AAPL",
        price=1805000,
        order_type="L"
    )

    t0 = time.perf_counter()
    for _ in range(count):
        raw = codec.encode_ouch_enter(ouch_msg)
    elapsed_enc = time.perf_counter() - t0
    enc_per_sec = count / elapsed_enc
    print(f"OUCH Encode: {count:,} messages in {elapsed_enc*1000:.2f} ms ({enc_per_sec:,.0f} msgs/sec)")

    t0 = time.perf_counter()
    for _ in range(count):
        _ = codec.decode_ouch_enter(raw)
    elapsed_dec = time.perf_counter() - t0
    dec_per_sec = count / elapsed_dec
    print(f"OUCH Decode: {count:,} messages in {elapsed_dec*1000:.2f} ms ({dec_per_sec:,.0f} msgs/sec)")

    itch_msg = ITCHAddOrder(
        timestamp_ns=1700000000000,
        order_id=12345,
        side="S",
        shares=1000,
        symbol="AAPL",
        price=1805000
    )

    t0 = time.perf_counter()
    for _ in range(count):
        raw_itch = codec.encode_itch_add(itch_msg)
    elapsed_itch_enc = time.perf_counter() - t0
    print(f"ITCH Encode: {count:,} messages in {elapsed_itch_enc*1000:.2f} ms ({count / elapsed_itch_enc:,.0f} msgs/sec)")

    t0 = time.perf_counter()
    for _ in range(count):
        _ = codec.decode_itch_add(raw_itch)
    elapsed_itch_dec = time.perf_counter() - t0
    print(f"ITCH Decode: {count:,} messages in {elapsed_itch_dec*1000:.2f} ms ({count / elapsed_itch_dec:,.0f} msgs/sec)")


def bench_pre_trade_risk():
    print("\n--- 5. Pre-Trade Risk Gate Evaluation ---")
    gate = RiskGate(RiskConfig())
    book = OrderBook("AAPL")
    order = OrderNode(1, "Trader1", "AAPL", Side.BUY, price_to_int(150.00), 500)

    count = 200_000
    t0 = time.perf_counter()
    for _ in range(count):
        _ = gate.check_order(order, book)
    elapsed = time.perf_counter() - t0
    checks_sec = count / elapsed
    ns_per_check = (elapsed / count) * 1e9
    print(f"Pre-Trade Risk Checks: {count:,} checks in {elapsed*1000:.2f} ms ({checks_sec:,.0f} checks/sec, {ns_per_check:.1f} ns/check)")


def run_all_benchmarks():
    print("=" * 65)
    print("      APEXMATCH: HIGH-FREQUENCY EXCHANGE BENCHMARK SUITE")
    print("=" * 65)
    bench_order_book_insert_cancel()
    bench_continuous_matching()
    bench_tick_to_trade_latency()
    bench_binary_protocol()
    bench_pre_trade_risk()
    print("\n" + "=" * 65)
    print("      ALL APEXMATCH BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    run_all_benchmarks()
