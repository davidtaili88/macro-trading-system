"""
Signal generation for the TIPS vs. nominal breakeven momentum strategy.

Three signals, corresponding to the passage:
  1. base    — always long TIPS / short nominal (no filter)
  2. oil_ma  — position on only when oil > its 55dma (lagged one day)
  3. copper_ma — same rule but with copper (the best version per passage, IR 0.81)

All signals are {+1, 0}, where 1 = hold the long TIPS / short nominal spread.
The spread return on day t = tips_ret[t] - nominal_ret[t].
"""

import pandas as pd

MA_WINDOW = 55  # days, as specified in the passage


def _ma_signal(commodity: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    """
    Return 1 when commodity close > its <window>-day simple MA, else 0.
    Signal is lagged by one day to avoid look-ahead bias.
    """
    ma = commodity.rolling(window).mean()
    raw = (commodity > ma).astype(int)
    return raw.shift(1)   # lag 1 day


def generate(prices: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame(index=prices.index)
    signals["base"]      = 1                              # always invested
    signals["oil_ma"]    = _ma_signal(prices["oil"])
    signals["copper_ma"] = _ma_signal(prices["copper"])
    return signals


def spread_returns(returns: pd.DataFrame) -> pd.Series:
    """Daily return of long TIPS / short nominal (equal-notional, unlevered)."""
    return returns["tips"] - returns["nominal"]
