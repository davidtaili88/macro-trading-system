"""
Order execution layer.

In paper-trading mode: logs fills and updates an in-memory portfolio.
In live mode: swap _send_order for a real REST call (Alpaca, IB, etc.).

Tracks P&L, position, and drawdown so the risk manager can gate orders.
"""

import logging
import time
from risk_manager import ApprovedOrder

logger = logging.getLogger(__name__)

SIMULATED_SLIPPAGE_PCT = 0.0005   # 0.05% per fill — adjust to asset class
TAKER_FEE_PCT          = 0.0002   # 0.02% exchange taker fee


class MockExecutionHub:
    """
    Simulates order fills against a paper-trading account.

    portfolio dict keys used by the risk manager:
      position_units  — current signed position
      drawdown_pct    — peak-to-trough fraction of initial capital
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self._capital = initial_capital
        self._peak_capital = initial_capital
        self._position_units: float = 0.0
        self._last_price: float = 0.0
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> dict:
        unrealized = self._position_units * self._last_price
        equity = self._capital + unrealized
        self._peak_capital = max(self._peak_capital, equity)
        drawdown = (self._peak_capital - equity) / self._peak_capital if self._peak_capital else 0.0
        return {
            "position_units": self._position_units,
            "equity": round(equity, 2),
            "realized_pnl": round(self._realized_pnl, 2),
            "drawdown_pct": round(drawdown, 6),
        }

    def execute(self, order: ApprovedOrder) -> None:
        """
        Fill an ApprovedOrder. Records slippage and fees.
        Replace _send_order with a live API call when ready.
        """
        delta_units = order.units - self._position_units
        if delta_units == 0:
            return

        fill_price = self._simulate_fill_price(delta_units)
        cost = abs(delta_units) * fill_price
        fees = cost * TAKER_FEE_PCT

        self._realized_pnl -= fees
        self._capital -= fees
        self._position_units = order.units
        self._last_price = fill_price

        logger.info(
            "FILL: symbol=%s delta=%.2f @ %.6f | fees=$%.4f | equity=$%.2f | drawdown=%.3f%%",
            order.symbol,
            delta_units,
            fill_price,
            fees,
            self.portfolio["equity"],
            self.portfolio["drawdown_pct"] * 100,
        )

        self._send_order(order, delta_units, fill_price)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_fill_price(self, delta_units: float) -> float:
        """Apply slippage against last known price."""
        direction = 1 if delta_units > 0 else -1
        return self._last_price * (1 + direction * SIMULATED_SLIPPAGE_PCT) if self._last_price else 0.0

    def _send_order(self, order: ApprovedOrder, delta_units: float, fill_price: float) -> None:
        """
        Stub for live order routing. Replace with:
            alpaca_client.submit_order(...)
            ib_client.placeOrder(...)
        """
        logger.debug(
            "ORDER SENT [mock]: symbol=%s side=%s units=%.2f price=%.6f time=%s",
            order.symbol,
            "BUY" if delta_units > 0 else "SELL",
            abs(delta_units),
            fill_price,
            time.strftime("%H:%M:%S", time.localtime(order.timestamp)),
        )
