#!/usr/bin/env python3
"""
ApexMatch: High-Frequency Trading (HFT) Exchange Simulator.
Demonstrates:
  1. Real-time L3 Limit Order Book with continuous price-time priority.
  2. Market makers providing two-sided liquidity around fair value.
  3. Institutional, retail, and iceberg orders crossing and replenishing.
  4. Binary ITCH market data broadcasting and client book reconstruction.
  5. Live ANSI TrueColor terminal order book depth ladder and execution tape.
"""

import time
import random
from typing import List
from apexmatch.types import Side, OrderType, price_to_int, int_to_price
from apexmatch.order import OrderNode
from apexmatch.matching_engine import MatchingEngine
from apexmatch.feed import MarketDataFeed, BookReconstructor
from apexmatch.visualizer import render_order_book_ladder


def run_exchange_simulation():
    symbol = "NVDA"
    engine = MatchingEngine(symbol)
    feed = MarketDataFeed(symbol)
    reconstructor = BookReconstructor(symbol)
    feed.subscribe(reconstructor.process_message)

    print("\n" + "=" * 76)
    print("        APEXMATCH: HIGH-FREQUENCY EXCHANGE & L3 ORDER BOOK LAB")
    print("      Ultra-Low Latency Deterministic Matching & Wire Protocol Engine")
    print("=" * 76)

    # Initial mid price: $125.00
    mid_price = price_to_int(125.00)
    order_id = 1

    print(f"\n[1] Seeding two-sided market maker liquidity for {symbol} around $125.00...")

    # Market Maker 1 & 2 populate depth
    for i in range(1, 9):
        # Bids below mid
        bid_p = mid_price - i * 100  # 10c increments
        bid_qty = random.randint(300, 2500)
        b_node = OrderNode(order_id, f"MM_{i%2+1}", symbol, Side.BUY, bid_p, bid_qty, timestamp_ns=time.perf_counter_ns())
        engine.process_order(b_node)
        feed.publish_order_add(b_node)
        order_id += 1

        # Asks above mid
        ask_p = mid_price + i * 100
        ask_qty = random.randint(300, 2500)
        a_node = OrderNode(order_id, f"MM_{i%2+1}", symbol, Side.SELL, ask_p, ask_qty, timestamp_ns=time.perf_counter_ns())
        engine.process_order(a_node)
        feed.publish_order_add(a_node)
        order_id += 1

    # Place an Iceberg Order on the Ask side
    iceberg = OrderNode(
        order_id=order_id,
        participant_id="Citadel",
        symbol=symbol,
        side=Side.SELL,
        price=mid_price + 100,  # Best ask
        qty=500,
        is_iceberg=True,
        visible_peak_qty=500,
        total_reserve_qty=4500,  # 5,000 total shares
        timestamp_ns=time.perf_counter_ns()
    )
    engine.process_order(iceberg)
    feed.publish_order_add(iceberg)
    order_id += 1

    print("\n[2] Initial Order Book State:")
    print(render_order_book_ladder(engine, depth=6, tape_depth=4))

    print("\n[3] Simulating high-frequency aggressive order flow & Iceberg execution...")

    # Aggressive BUY sweep: 1,200 shares
    sweep_buy = OrderNode(
        order_id=order_id,
        participant_id="JaneStreet",
        symbol=symbol,
        side=Side.BUY,
        price=mid_price + 200,
        qty=1200,
        timestamp_ns=time.perf_counter_ns()
    )
    accepted, trades, msg = engine.process_order(sweep_buy)
    order_id += 1
    for t in trades:
        feed.publish_trade(t)
    print(f" -> Aggressive Buy 1,200 shares: {len(trades)} fills generated at best asks.")

    # Aggressive SELL: 800 shares
    sweep_sell = OrderNode(
        order_id=order_id,
        participant_id="TwoSigma",
        symbol=symbol,
        side=Side.SELL,
        price=mid_price - 100,
        qty=800,
        timestamp_ns=time.perf_counter_ns()
    )
    accepted, trades, msg = engine.process_order(sweep_sell)
    order_id += 1
    for t in trades:
        feed.publish_trade(t)
    print(f" -> Aggressive Sell 800 shares: {len(trades)} fills generated at best bids.")

    # Second BUY hitting the Iceberg order again
    iceberg_hit = OrderNode(
        order_id=order_id,
        participant_id="Millennium",
        symbol=symbol,
        side=Side.BUY,
        price=mid_price + 100,
        qty=600,
        timestamp_ns=time.perf_counter_ns()
    )
    accepted, trades, msg = engine.process_order(iceberg_hit)
    order_id += 1
    for t in trades:
        feed.publish_trade(t)
    print(f" -> Second Buy hitting Iceberg order: Iceberg replenished from reserve (remaining reserve: {iceberg.total_reserve_qty:,} shares).")

    print("\n[4] Updated Order Book Depth Ladder with Live Execution Tape:")
    print(render_order_book_ladder(engine, depth=8, tape_depth=6))

    # Verify Client-side ITCH feed reconstruction
    client_bb, client_ba = reconstructor.get_top_of_book()
    print("\n[5] Binary ITCH Feed & Client Book Reconstruction Verification:")
    print(f"  Exchange Best Bid / Ask: {int_to_price(engine.book.best_bid()):.2f} / {int_to_price(engine.book.best_ask()):.2f}")
    if client_bb and client_ba:
        print(f"  Client Reconstructed:    {int_to_price(client_bb):.2f} / {int_to_price(client_ba):.2f}")
    print(f"  ITCH Messages Broadcast: {feed.msg_count} packets ({len(feed.byte_stream):,} raw binary bytes)")
    print(f"  Total Traded Volume:     {engine.total_traded_volume:,} shares")
    print(f"  Exchange VWAP:           ${engine.vwap():,.4f}")
    print(f"  Order Flow Imbalance:    {feed.ofi:+,d} shares")

    print("\n" + "=" * 76)
    print("            EXCHANGE SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_exchange_simulation()
