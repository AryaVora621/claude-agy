"""
ApexMatch: Pre-Trade Risk Management & Compliance Engine.
Enforces price collars, maximum order sizes, notional exposure caps,
and participant position limits.
"""

from typing import Optional, Dict, Tuple
from apexmatch.types import Side, OrderType, calculate_notional, int_to_price
from apexmatch.order import OrderNode
from apexmatch.order_book import OrderBook


class RiskConfig:
    """Pre-trade risk boundaries and limits."""

    def __init__(
        self,
        max_order_qty: int = 100_000,
        max_notional_dollars: float = 1_000_000.0,
        price_collar_pct: float = 0.10,  # 10% maximum deviation from mid-price
        max_position_shares: int = 500_000
    ) -> None:
        self.max_order_qty = max_order_qty
        self.max_notional = int(round(max_notional_dollars * 10_000))
        self.price_collar_pct = price_collar_pct
        self.max_position_shares = max_position_shares


class RiskGate:
    """
    Sub-microsecond deterministic pre-trade risk evaluation.
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()
        # Participant ID -> net share position
        self.positions: Dict[str, int] = {}

    def check_order(self, order: OrderNode, book: OrderBook) -> Tuple[bool, str]:
        """
        Validates order against risk limits.
        Returns: (passed: bool, reason: str)
        """
        # 1. Quantity check
        if order.qty <= 0:
            return False, "REJECT_ZERO_OR_NEGATIVE_QUANTITY"

        total_shares = (order.qty + order.total_reserve_qty) if order.is_iceberg else order.qty
        if total_shares > self.cfg.max_order_qty:
            return False, f"REJECT_EXCEEDS_MAX_QTY: {total_shares} > {self.cfg.max_order_qty}"

        # 2. Notional value check
        if order.order_type != OrderType.MARKET:
            notional = calculate_notional(order.price, total_shares)
            if notional > self.cfg.max_notional:
                return False, f"REJECT_EXCEEDS_MAX_NOTIONAL: ${int_to_price(notional):,.2f} > ${int_to_price(self.cfg.max_notional):,.2f}"

        # 3. Price Collar Check (Limit orders only, when both book sides exist)
        if order.order_type == OrderType.LIMIT:
            mid = book.mid_price()
            if mid is not None and mid > 0:
                collar = int(round(mid * self.cfg.price_collar_pct))
                min_allowable = mid - collar
                max_allowable = mid + collar
                if order.price < min_allowable or order.price > max_allowable:
                    return False, f"REJECT_PRICE_COLLAR_VIOLATION: price {int_to_price(order.price)} outside [{int_to_price(min_allowable)}, {int_to_price(max_allowable)}]"

        # 4. Position limits
        current_pos = self.positions.get(order.participant_id, 0)
        potential_pos = current_pos + (total_shares if order.side == Side.BUY else -total_shares)
        if abs(potential_pos) > self.cfg.max_position_shares:
            return False, f"REJECT_POSITION_LIMIT_EXCEEDED: net position {potential_pos} > {self.cfg.max_position_shares}"

        return True, "PASSED"

    def record_fill(self, participant_id: str, side: Side, filled_qty: int) -> None:
        """Updates net position tracking upon trade fill."""
        delta = filled_qty if side == Side.BUY else -filled_qty
        self.positions[participant_id] = self.positions.get(participant_id, 0) + delta
