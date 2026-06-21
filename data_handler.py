"""
Macro data ingestion layer.

Connects to live or paper data feeds and emits standardized MarketEvent objects.
Supports WebSocket streams with exponential-backoff reconnect logic.

Swap the _fetch_tick stub for a real API (Alpaca, FRED, IB) when ready.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

logger = logging.getLogger(__name__)

RECONNECT_BASE_DELAY = 1.0   # seconds
RECONNECT_MAX_DELAY  = 60.0  # seconds


@dataclass
class MarketEvent:
    timestamp: float          # Unix epoch seconds
    symbol: str               # e.g. "US2Y10Y", "EURUSD", "GC=F"
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


class MacroDataStream:
    """
    Async generator that yields MarketEvents from a live feed.

    Replace _connect and _fetch_tick with real WebSocket / REST calls.
    The reconnect loop handles dropped connections automatically.
    """

    def __init__(self, symbol: str = "US2Y10Y", poll_interval: float = 1.0):
        self.symbol = symbol
        self.poll_interval = poll_interval
        self._connected = False

    async def _connect(self) -> None:
        """Establish connection to the data provider."""
        logger.info("Connecting to data feed for %s ...", self.symbol)
        await asyncio.sleep(0.1)  # replace with real handshake
        self._connected = True
        logger.info("Connected.")

    async def _fetch_tick(self) -> MarketEvent | None:
        """
        Pull the latest tick from the provider.
        Returns None if no new data is available yet.
        """
        import random  # remove once a real feed is wired
        await asyncio.sleep(self.poll_interval)
        price = 0.45 + random.gauss(0, 0.005)   # mock 2s10s spread in %
        return MarketEvent(
            timestamp=time.time(),
            symbol=self.symbol,
            price=round(price, 6),
            bid=round(price - 0.001, 6),
            ask=round(price + 0.001, 6),
        )

    async def stream(self) -> AsyncIterator[MarketEvent]:
        """Yields MarketEvents indefinitely, reconnecting on failure."""
        delay = RECONNECT_BASE_DELAY
        while True:
            try:
                await self._connect()
                delay = RECONNECT_BASE_DELAY  # reset on success
                while True:
                    event = await self._fetch_tick()
                    if event is not None:
                        yield event
            except (ConnectionError, OSError) as exc:
                logger.warning("Feed disconnected (%s). Reconnecting in %.1fs ...", exc, delay)
                self._connected = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)
