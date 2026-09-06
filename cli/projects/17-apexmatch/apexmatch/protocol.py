"""
ApexMatch: Binary Wire Protocol Engine (ITCH & OUCH).
Implements ultra-compact, high-speed binary serialization for exchange order entry (OUCH)
and real-time market data dissemination (ITCH).
"""

import struct
from typing import NamedTuple, Tuple, Union

# Format definitions (Big-Endian Network Order)
# OUCH Inbound
FMT_OUCH_ENTER = ">cQcI8sIc"   # 28 bytes: 'O', order_id, side, qty, symbol, price, order_type
FMT_OUCH_CANCEL = ">cQI"       # 13 bytes: 'X', order_id, qty
FMT_OUCH_ACCEPTED = ">cQQcI8sI" # 33 bytes: 'A', timestamp, order_id, side, qty, symbol, price
FMT_OUCH_REJECTED = ">cQ32s"   # 41 bytes: 'J', order_id, reason

# ITCH Outbound Market Data
FMT_ITCH_SYSTEM = ">cQc"       # 10 bytes: 'S', timestamp, event_code
FMT_ITCH_ADD = ">cQQcI8sI"     # 33 bytes: 'A', timestamp, order_id, side, shares, symbol, price
FMT_ITCH_EXEC = ">cQQIQ"       # 29 bytes: 'E', timestamp, order_id, executed_shares, match_id
FMT_ITCH_CANCEL = ">cQQI"      # 21 bytes: 'C', timestamp, order_id, canceled_shares
FMT_ITCH_TRADE = ">cQQcI8sI"    # 33 bytes: 'P', timestamp, match_id, side, shares, symbol, price


class OUCHEnterOrder(NamedTuple):
    order_id: int
    side: str
    qty: int
    symbol: str
    price: int
    order_type: str


class OUCHCancelOrder(NamedTuple):
    order_id: int
    qty: int


class ITCHAddOrder(NamedTuple):
    timestamp_ns: int
    order_id: int
    side: str
    shares: int
    symbol: str
    price: int


class ITCHOrderExecuted(NamedTuple):
    timestamp_ns: int
    order_id: int
    executed_shares: int
    match_id: int


class ITCHOrderCanceled(NamedTuple):
    timestamp_ns: int
    order_id: int
    canceled_shares: int


class ITCHTrade(NamedTuple):
    timestamp_ns: int
    match_id: int
    side: str
    shares: int
    symbol: str
    price: int


class BinaryCodec:
    """
    High-speed binary codec for exchange protocol messages.
    """

    @staticmethod
    def encode_ouch_enter(order: OUCHEnterOrder) -> bytes:
        sym_bytes = order.symbol.encode("ascii").ljust(8, b" ")[:8]
        return struct.pack(
            FMT_OUCH_ENTER,
            b"O",
            order.order_id,
            order.side.encode("ascii"),
            order.qty,
            sym_bytes,
            order.price,
            order.order_type.encode("ascii")
        )

    @staticmethod
    def decode_ouch_enter(data: bytes) -> OUCHEnterOrder:
        msg_type, oid, side, qty, sym, price, otype = struct.unpack(FMT_OUCH_ENTER, data[:28])
        if msg_type != b"O":
            raise ValueError(f"Invalid OUCH enter message type: {msg_type}")
        return OUCHEnterOrder(
            order_id=oid,
            side=side.decode("ascii"),
            qty=qty,
            symbol=sym.decode("ascii").strip(),
            price=price,
            order_type=otype.decode("ascii")
        )

    @staticmethod
    def encode_ouch_cancel(cancel: OUCHCancelOrder) -> bytes:
        return struct.pack(FMT_OUCH_CANCEL, b"X", cancel.order_id, cancel.qty)

    @staticmethod
    def decode_ouch_cancel(data: bytes) -> OUCHCancelOrder:
        msg_type, oid, qty = struct.unpack(FMT_OUCH_CANCEL, data[:13])
        if msg_type != b"X":
            raise ValueError(f"Invalid OUCH cancel message type: {msg_type}")
        return OUCHCancelOrder(order_id=oid, qty=qty)

    @staticmethod
    def encode_itch_add(add: ITCHAddOrder) -> bytes:
        sym_bytes = add.symbol.encode("ascii").ljust(8, b" ")[:8]
        return struct.pack(
            FMT_ITCH_ADD,
            b"A",
            add.timestamp_ns,
            add.order_id,
            add.side.encode("ascii"),
            add.shares,
            sym_bytes,
            add.price
        )

    @staticmethod
    def decode_itch_add(data: bytes) -> ITCHAddOrder:
        msg_type, ts, oid, side, shares, sym, price = struct.unpack(FMT_ITCH_ADD, data[:34])
        return ITCHAddOrder(
            timestamp_ns=ts,
            order_id=oid,
            side=side.decode("ascii"),
            shares=shares,
            symbol=sym.decode("ascii").strip(),
            price=price
        )

    @staticmethod
    def encode_itch_exec(exec_msg: ITCHOrderExecuted) -> bytes:
        return struct.pack(
            FMT_ITCH_EXEC,
            b"E",
            exec_msg.timestamp_ns,
            exec_msg.order_id,
            exec_msg.executed_shares,
            exec_msg.match_id
        )

    @staticmethod
    def decode_itch_exec(data: bytes) -> ITCHOrderExecuted:
        msg_type, ts, oid, shares, mid = struct.unpack(FMT_ITCH_EXEC, data[:29])
        return ITCHOrderExecuted(
            timestamp_ns=ts,
            order_id=oid,
            executed_shares=shares,
            match_id=mid
        )

    @staticmethod
    def encode_itch_cancel(cancel: ITCHOrderCanceled) -> bytes:
        return struct.pack(
            FMT_ITCH_CANCEL,
            b"C",
            cancel.timestamp_ns,
            cancel.order_id,
            cancel.canceled_shares
        )

    @staticmethod
    def decode_itch_cancel(data: bytes) -> ITCHOrderCanceled:
        msg_type, ts, oid, shares = struct.unpack(FMT_ITCH_CANCEL, data[:21])
        return ITCHOrderCanceled(
            timestamp_ns=ts,
            order_id=oid,
            canceled_shares=shares
        )

    @staticmethod
    def encode_itch_trade(trade: ITCHTrade) -> bytes:
        sym_bytes = trade.symbol.encode("ascii").ljust(8, b" ")[:8]
        return struct.pack(
            FMT_ITCH_TRADE,
            b"P",
            trade.timestamp_ns,
            trade.match_id,
            trade.side.encode("ascii"),
            trade.shares,
            sym_bytes,
            trade.price
        )

    @staticmethod
    def decode_itch_trade(data: bytes) -> ITCHTrade:
        msg_type, ts, mid, side, shares, sym, price = struct.unpack(FMT_ITCH_TRADE, data[:34])
        return ITCHTrade(
            timestamp_ns=ts,
            match_id=mid,
            side=side.decode("ascii"),
            shares=shares,
            symbol=sym.decode("ascii").strip(),
            price=price
        )
