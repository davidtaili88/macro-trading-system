"""
Event-driven macro trading system entry point.
Wires together data ingestion, signal generation, risk gating, and execution.
"""

import asyncio
import logging
from data_handler import MacroDataStream
from signal_engine import BayesianSignalEngine
from risk_manager import MacroRiskManager
from execution_hub import MockExecutionHub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


async def run():
    data_stream = MacroDataStream()
    signal_engine = BayesianSignalEngine()
    risk_manager = MacroRiskManager(max_drawdown_pct=0.05, max_position_units=100)
    execution_hub = MockExecutionHub()

    logger.info("System booted. Awaiting live macro data events...")

    async for market_event in data_stream.stream():
        try:
            raw_signal = signal_engine.update(market_event)
            approved_order = risk_manager.validate(raw_signal, execution_hub.portfolio)
            if approved_order is not None:
                execution_hub.execute(approved_order)
        except Exception as exc:
            logger.error("Event processing failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    asyncio.run(run())
