"""
Risk management layer — the system's circuit breakers.

Sits between the signal engine and the execution hub.
Blocks or scales orders based on:
  - Portfolio drawdown vs. hard limit
  - Bayesian model confidence (uncertainty throttle)
  - Raw position size cap
  - Minimum trade size (prevents fee bleed on tiny adjustments)
"""

import logging
from dataclasses import dataclass
from signal_engine import SignalEvent

logger = logging.getLogger(__name__)


@dataclass
class ApprovedOrder:
    timestamp: float
    symbol: str
    units: float              # final signed position size after risk scaling
    confidence: float
    regime: str


class MacroRiskManager:
    """
    Validates and scales SignalEvents before they reach the exchange.

    Parameters
    ----------
    max_drawdown_pct : float
        Fraction of initial capital at which all new orders are blocked
        and open positions are flattened (e.g. 0.05 = 5%).
    max_position_units : float
        Hard ceiling on absolute position size.
    min_trade_threshold : float
        Ignore order adjustments smaller than this to avoid fee bleed.
    confidence_floor : float
        Orders are blocked when model confidence falls below this level.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.05,
        max_position_units: float = 100.0,
        min_trade_threshold: float = 1.0,
        confidence_floor: float = 0.3,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_units = max_position_units
        self.min_trade_threshold = min_trade_threshold
        self.confidence_floor = confidence_floor

    def validate(self, signal: SignalEvent, portfolio: dict) -> ApprovedOrder | None:
        """
        Returns an ApprovedOrder if the signal passes all checks, else None.
        """
        drawdown = portfolio.get("drawdown_pct", 0.0)
        current_units = portfolio.get("position_units", 0.0)

        # 1. Hard circuit breaker: max drawdown
        if drawdown >= self.max_drawdown_pct:
            logger.warning(
                "CIRCUIT BREAKER: drawdown %.2f%% >= limit %.2f%%. Order blocked.",
                drawdown * 100, self.max_drawdown_pct * 100,
            )
            return None

        # 2. Uncertainty throttle: low-confidence signals are blocked
        if signal.confidence < self.confidence_floor:
            logger.info(
                "Low confidence (%.3f < %.3f). Signal suppressed.",
                signal.confidence, self.confidence_floor,
            )
            return None

        # 3. Scale position by confidence
        scaled_units = signal.target_position * signal.confidence * self.max_position_units

        # 4. Cap at absolute position limit
        scaled_units = max(-self.max_position_units, min(self.max_position_units, scaled_units))

        # 5. Ignore micro-adjustments that don't clear transaction costs
        delta = abs(scaled_units - current_units)
        if delta < self.min_trade_threshold:
            logger.debug("Delta %.3f below min threshold. Skipping.", delta)
            return None

        logger.info(
            "Order approved: symbol=%s units=%.2f confidence=%.3f regime=%s",
            signal.symbol, scaled_units, signal.confidence, signal.regime,
        )

        return ApprovedOrder(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            units=scaled_units,
            confidence=signal.confidence,
            regime=signal.regime,
        )
