"""
Bayesian / SSM signal engine.

This module is the black-box interface for the professor's model.
The public API (update) is stable; swap the internals for the real
dynesty / NumPyro / Kalman implementation without touching other modules.

Returns a SignalEvent that downstream modules consume.
"""

import logging
from dataclasses import dataclass
from data_handler import MarketEvent

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    timestamp: float
    symbol: str
    target_position: float    # signed units; positive = long, negative = short
    confidence: float         # [0, 1] — maps to Bayesian posterior certainty
    regime: str               # e.g. "expansion", "stagflation", "unknown"
    raw_state: dict           # pass-through of internal model state for logging


class BayesianSignalEngine:
    """
    Wraps the SSM / Bayesian inference model.

    Stub implementation uses a simple exponential moving average so the
    system can be exercised end-to-end before the real math is plugged in.
    """

    def __init__(self, ema_alpha: float = 0.1):
        self._alpha = ema_alpha
        self._ema: float | None = None
        self._variance: float = 0.0

    # ------------------------------------------------------------------
    # Public interface — do not change the signature
    # ------------------------------------------------------------------

    def update(self, event: MarketEvent) -> SignalEvent:
        """
        Consume a MarketEvent and return a SignalEvent.

        Replace the body with recursive Kalman / dynesty update logic.
        confidence should reflect the posterior uncertainty — when the
        sampler's variance is high, confidence should be low.
        """
        target, confidence, regime = self._mock_inference(event.price)

        logger.debug(
            "Signal: symbol=%s target=%.2f confidence=%.3f regime=%s",
            event.symbol, target, confidence, regime,
        )

        return SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            target_position=target,
            confidence=confidence,
            regime=regime,
            raw_state={"ema": self._ema, "variance": self._variance},
        )

    # ------------------------------------------------------------------
    # Internal stub — replace with real SSM math
    # ------------------------------------------------------------------

    def _mock_inference(self, price: float) -> tuple[float, float, str]:
        if self._ema is None:
            self._ema = price
            return 0.0, 0.0, "unknown"

        prev_ema = self._ema
        self._ema = self._alpha * price + (1 - self._alpha) * self._ema
        self._variance = self._alpha * (price - self._ema) ** 2 + (1 - self._alpha) * self._variance

        # Regime: crude threshold on 2s10s spread proxy
        regime = "expansion" if self._ema > 0 else "inversion"

        # Signal: positive spread drift → long risk assets
        drift = self._ema - prev_ema
        target = 1.0 if drift > 0 else -1.0

        # Confidence: falls as variance rises (clipped to [0, 1])
        confidence = max(0.0, 1.0 - min(self._variance * 200, 1.0))

        return target, confidence, regime
