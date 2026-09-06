"""
ApexMatch: Real-Time Market Data Feed & Book Reconstructor.
Generates binary ITCH stream and reconstructs exact L2/L3 order books
from outbound tick messages.
"""

from typing import List, Dict, Tuple, Optional, Callable
from apexmatch.protocol import (
    BinaryCodec,
    ITCHAddOrder,
    ITCHOrderExecuted,
    ITCHOrderCanceled,
    ITCHTrade,
)
from apexmatch.types import Side, Trade
from apexmatch.order import OrderNode


class MarketDataFeed:
    """
    Distributes ITCH market data events to subscribers and tracks market statistics.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.subscribers: List[Callable[[bytes], None]] = []
        self.byte_stream: bytearray = bytearray()
        self.msg_count = 0

        # Real-time analytics
        self.total_volume = 0
        self.total_turnover = 0
        self.last_price: Optional[int] = None
        self.trade_count = 0

        # Order Flow Imbalance (OFI) tracking
        self.ofi = 0

    def subscribe(self, callback: Callable[[bytes], None]) -> None:
        self.subscribers.append(callback)

    def publish_order_add(self, order: OrderNode) -> bytes:
        msg = ITCHAddOrder(
            timestamp_ns=order.timestamp_ns,
            order_id=order.order_id,
            side=order.side.value,
            shares=order.qty,
            symbol=self.symbol,
            price=order.price
        )
        data = BinaryCodec.encode_itch_add(msg)
        self._emit(data)
        # Update OFI
        delta_v = order.qty if order.side == Side.BUY else -order.qty
        self.ofi += delta_v
        return data

    def publish_order_exec(self, order_id: int, executed_qty: int, match_id: int, ts_ns: int) -> bytes:
        msg = ITCHOrderExecuted(
            timestamp_ns=ts_ns,
            order_id=order_id,
            executed_shares=executed_qty,
            match_id=match_id
        )
        data = BinaryCodec.encode_itch_exec(msg)
        self._emit(data)
        return data

    def publish_order_cancel(self, order_id: int, canceled_qty: int, ts_ns: int) -> bytes:
        msg = ITCHOrderCanceled(
            timestamp_ns=ts_ns,
            order_id=order_id,
            canceled_shares=canceled_qty
        )
        data = BinaryCodec.encode_itch_cancel(msg)
        self._emit(data)
        return data

    def publish_trade(self, trade: Trade) -> bytes:
        msg = ITCHTrade(
            timestamp_ns=trade.timestamp_ns,
            match_id=trade.match_id,
            side=trade.taker_side.value,
            shares=trade.qty,
            symbol=self.symbol,
            price=trade.price
        )
        data = BinaryCodec.encode_itch_trade(msg)
        self._emit(data)

        # Update stats
        self.trade_count += 1
        self.total_volume += trade.qty
        self.total_turnover += trade.price * trade.qty
        self.last_price = trade.price
        return data

    def _emit(self, payload: bytes) -> None:
        self.byte_stream.extend(payload)
        self.msg_count += 1
        for sub in self.subscribers:
            sub(payload)

    @property
    def vwap(self) -> Optional[float]:
        if self.total_volume == 0:
            return None
        return (self.total_turnover / self.total_volume) / 10_000.0


class BookReconstructor:
    """
    Client-side engine that parses an incoming binary ITCH stream
    and reconstructs the exact Level 2 limit order book depth.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        # OrderID -> (side, price, remaining_qty)
        self.orders: Dict[int, Tuple[str, int, int]] = {}
        # Price -> total_qty
        self.bids_l2: Dict[int, int] = {}
        self.asks_l2: Dict[int, int] = {}

    def process_message(self, data: bytes) -> None:
        if not data:
            return
        msg_type = data[0:1]

        if msg_type == b"A":  # Add order
            msg = BinaryCodec.decode_itch_add(data)
            if msg.symbol == self.symbol:
                self.orders[msg.order_id] = (msg.side, msg.price, msg.shares)
                book = self.bids_l2 if msg.side == "B" else self.asks_l2
                book[msg.price] = book.get(msg.price, 0) + msg.shares

        elif msg_type == b"E":  # Execute order
            msg = BinaryCodec.decode_itch_exec(data)
            info = self.orders.get(msg.order_id)
            if info:
                side, price, rem = info
                new_rem = rem - msg.executed_shares
                book = self.bids_l2 if side == "B" else self.asks_l2
                book[price] = max(0, book.get(price, 0) - msg.executed_shares)
                if book[price] == 0:
                    del book[price]

                if new_rem <= 0:
                    del self.orders[msg.order_id]
                else:
                    self.orders[msg.order_id] = (side, price, new_rem)

        elif msg_type == b"C":  # Cancel order
            msg = BinaryCodec.decode_itch_cancel(data)
            info = self.orders.get(msg.order_id)
            if info:
                side, price, rem = info
                new_rem = rem - msg.canceled_shares
                book = self.bids_l2 if side == "B" else self.asks_l2
                book[price] = max(0, book.get(price, 0) - msg.canceled_shares)
                if book[price] == 0:
                    del book[price]

                if new_rem <= 0:
                    del self.orders[msg.order_id]
                else:
                    self.orders[msg.order_id] = (side, price, new_rem)

    def get_top_of_book(self) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(self.bids_l2.keys()) if self.bids_l2 else None
        best_ask = min(self.asks_l2.keys()) if self.asks_l2 else None
        return best_bid, best_ask
