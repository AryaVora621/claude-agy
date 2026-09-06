"""
ApexMatch: Core Exchange Types, Enumerations, and Fixed-Point Arithmetic.
All prices are represented internally as 64-bit integers scaled by 10,000 (micro-cents).
"""

from enum import Enum
from typing import NamedTuple

PRICE_SCALE = 10_000  # 4 decimal places: 100.5000 -> 1,005,000


class Side(str, Enum):
    BUY = "B"
    SELL = "S"

    def opposite(self) -> "Side":
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderType(str, Enum):
    LIMIT = "L"
    MARKET = "M"
    IOC = "I"  # Immediate-or-Cancel
    FOK = "F"  # Fill-or-Kill


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good-Till-Cancel
    IOC = "IOC"  # Immediate-or-Cancel
    FOK = "FOK"  # Fill-or-Kill
    DAY = "DAY"  # Day Order


class STPMode(str, Enum):
    CANCEL_PASSIVE = "CP"    # Cancel resting order on cross
    CANCEL_AGGRESSIVE = "CA" # Reject incoming order on cross


def price_to_int(price: float) -> int:
    """Converts a floating-point dollar price to integer micro-cents."""
    return int(round(price * PRICE_SCALE))


def int_to_price(price_int: int) -> float:
    """Converts integer micro-cents to floating-point dollar price."""
    return price_int / PRICE_SCALE


def format_price(price_int: int) -> str:
    """Formats integer micro-cents into standard currency string '$XX.XXXX'."""
    dollars = price_int // PRICE_SCALE
    micro = abs(price_int % PRICE_SCALE)
    return f"${dollars}.{micro:04d}"


def calculate_notional(price_int: int, qty: int) -> int:
    """Computes total notional value in micro-cents (price_int * qty)."""
    return price_int * qty


class Trade(NamedTuple):
    match_id: int
    maker_order_id: int
    taker_order_id: int
    buyer_participant_id: str
    seller_participant_id: str
    symbol: str
    price: int
    qty: int
    timestamp_ns: int
    taker_side: Side
