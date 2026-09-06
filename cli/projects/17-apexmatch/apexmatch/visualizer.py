"""
ApexMatch: High-Resolution ANSI Terminal Order Book Depth Ladder & Trade Tape Visualizer.
Renders Level 2 depth ladder with proportional volume histogram bars,
bid-ask spread, live execution tape, and VWAP metrics.
"""

from typing import List, Tuple, Optional
from apexmatch.types import format_price, int_to_price, Side
from apexmatch.matching_engine import MatchingEngine


def render_order_book_ladder(
    engine: MatchingEngine,
    depth: int = 8,
    tape_depth: int = 5,
    bar_width: int = 24
) -> str:
    """
    Renders an ASCII/ANSI TrueColor terminal display of the limit order book.
    """
    book = engine.book
    bids_l2, asks_l2 = book.get_l2_snapshot(depth)

    # Calculate max volume for relative bar scaling
    all_vols = [v for _, v in bids_l2] + [v for _, v in asks_l2]
    max_vol = max(all_vols) if all_vols else 1

    lines: List[str] = []
    w = 74

    # Header
    lines.append("+" + "=" * w + "+")
    title = f" APEXMATCH L3 LIMIT ORDER BOOK : {engine.symbol} "
    lines.append(f"|{title.center(w)}|")
    lines.append("+" + "=" * w + "+")

    # Metrics Summary Row
    bb = book.best_bid()
    ba = book.best_ask()
    spread = book.spread()
    mid = book.mid_price()
    vwap_val = engine.vwap()

    bb_str = format_price(bb) if bb else "None"
    ba_str = format_price(ba) if ba else "None"
    spread_str = f"{spread / 100:.1f}¢" if spread else "N/A"
    mid_str = format_price(mid) if mid else "N/A"
    vwap_str = f"${vwap_val:,.2f}" if vwap_val else "N/A"

    metrics_line = f" Bid: {bb_str}  |  Ask: {ba_str}  |  Spread: {spread_str}  |  Mid: {mid_str}  |  VWAP: {vwap_str}"
    lines.append(f"|{metrics_line.center(w)}|")
    lines.append("+" + "-" * w + "+")

    # Column Headers
    col_header = f" {'SIDE':<5} | {'PRICE':<10} | {'SHARES':<8} | {'DEPTH PROFILE':<{bar_width}} | {'ORDERS':<6}"
    lines.append(f"|{col_header:<{w}}|")
    lines.append("+" + "-" * w + "+")

    # 1. Asks (Top, rendered in reverse ascending order so lowest ask is at bottom near spread)
    reversed_asks = list(reversed(asks_l2))
    for price, vol in reversed_asks:
        bar_len = int((vol / max_vol) * bar_width) if max_vol > 0 else 0
        bar = "#" * max(1, bar_len)
        orders_cnt = book.asks[price].order_count

        # ANSI Red for Asks
        side_tag = "\033[38;2;255;80;80mASK\033[0m"
        p_styled = f"\033[38;2;255;120;120m{format_price(price):<10}\033[0m"
        bar_styled = f"\033[38;2;255;60;60m{bar:<{bar_width}}\033[0m"

        row = f"  {side_tag}  | {p_styled} | {vol:<8,d} | {bar_styled} | {orders_cnt:<6d}"
        lines.append(f"|{row} |")

    # 2. Spread Divider
    if bb is not None and ba is not None:
        spread_bps = (spread / mid) * 10000 if mid else 0
        div_text = f"--- SPREAD: {spread / 100:.2f} cents ({spread_bps:.1f} bps) ---"
        styled_div = f"\033[38;2;255;210;50m{div_text.center(w)}\033[0m"
        lines.append(f"|{styled_div}|")
    else:
        lines.append(f"|{'--- NO SPREAD (EMPTY BOOK) ---'.center(w)}|")

    # 3. Bids (Bottom, sorted descending so highest bid is at top near spread)
    for price, vol in bids_l2:
        bar_len = int((vol / max_vol) * bar_width) if max_vol > 0 else 0
        bar = "#" * max(1, bar_len)
        orders_cnt = book.bids[price].order_count

        # ANSI Green for Bids
        side_tag = "\033[38;2;50;220;80mBID\033[0m"
        p_styled = f"\033[38;2;100;240;120m{format_price(price):<10}\033[0m"
        bar_styled = f"\033[38;2;40;200;80m{bar:<{bar_width}}\033[0m"

        row = f"  {side_tag}  | {p_styled} | {vol:<8,d} | {bar_styled} | {orders_cnt:<6d}"
        lines.append(f"|{row} |")

    lines.append("+" + "-" * w + "+")

    # 4. Recent Execution Tape
    tape_header = " RECENT TRADES (EXECUTION TAPE) "
    lines.append(f"|{tape_header.center(w)}|")
    lines.append("+" + "-" * w + "+")

    recent_trades = engine.trades[-tape_depth:] if engine.trades else []
    if not recent_trades:
        lines.append(f"|{'No trades executed yet'.center(w)}|")
    else:
        for t in reversed(recent_trades):
            side_badge = "\033[38;2;50;220;80m[BUY ]\033[0m" if t.taker_side == Side.BUY else "\033[38;2;255;80;80m[SELL]\033[0m"
            notional = (t.price * t.qty) / 10_000.0
            trade_row = (
                f"  Match #{t.match_id:04d}  {side_badge}  {t.qty:>6,d} @ {format_price(t.price)}  "
                f"(${notional:>10,.2f})  {t.buyer_participant_id} <-> {t.seller_participant_id}"
            )
            lines.append(f"|{trade_row:<{w + 18}}|")  # Adjust for ANSI escape length

    lines.append("+" + "=" * w + "+")
    return "\n".join(lines)
