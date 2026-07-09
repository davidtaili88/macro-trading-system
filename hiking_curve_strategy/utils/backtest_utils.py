"""
Generic backtest utilities — reusable across any strategy.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#Takes in the output of a strat's returns
def annualised_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised return, vol, Sharpe, max drawdown for strategy and buy-hold bond."""
    rows = []
    for col, series in [("payer_strategy", df["strat_ret"]), ("buy_hold_bonds", df["bond_ret"])]:
        # pandas Series.dropna: remove NaN entries before any arithmetic to avoid propagating nulls
        r = series.dropna()
        # pandas Series.mean: average daily return, scaled by 252 to annualise
        ann_ret = r.mean() * TRADING_DAYS
        # pandas Series.std / numpy: daily volatility annualised by multiplying by sqrt(252)
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        # numpy: np.nan used as sentinel when volatility is zero to prevent division-by-zero
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        # pandas Series: cumulative product of (1 + r) to build the equity growth curve
        cum     = (1 + r).cumprod()
        # pandas Series.cummax: running maximum of the equity curve, used as the drawdown denominator
        roll_max = cum.cummax()
        # pandas Series.min: finds the worst (most negative) drawdown trough across all dates
        max_dd  = ((cum - roll_max) / roll_max).min()
        rows.append({
            "strategy":   col,
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol":    round(ann_vol * 100, 2),
            "sharpe":     round(sharpe, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "n_days":     len(r),
        })
    # pandas DataFrame: construct from list of dicts, then set "strategy" as the row index (e.g. "payer_strategy")
    return pd.DataFrame(rows).set_index("strategy")


def rolling_sharpe(df: pd.DataFrame, window: int = TRADING_DAYS) -> pd.Series:
    """Rolling annualised Sharpe ratio for the payer strategy."""
    # pandas Series.dropna: strip NaN rows before the rolling calculation
    r = df["strat_ret"].dropna()
    # pandas Series.rolling: create a rolling window object of length `window`, then .mean() computes the rolling average
    roll_mean = r.rolling(window).mean() * TRADING_DAYS
    # pandas Series.rolling / numpy: rolling standard deviation annualised by sqrt(252)
    roll_vol  = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    # pandas Series.rename: give the resulting Series a descriptive name for plot labels
    return (roll_mean / roll_vol).rename("rolling_sharpe")


def event_time_returns(
    returns: pd.Series,
    cycles: list[dict],
    anchor: str = "first_hike",
    window: int = 120,
) -> pd.DataFrame:
    """
    Collect bond returns in event time around anchor dates (first or last hike).
    Returns a DataFrame where each column is one cycle and rows are trading-day
    offsets from -window to +window relative to the anchor.

    Useful for replicating the book's Figures 5.5 and 5.6.
    """
    # pandas DatetimeIndex: the sorted index of all trading dates used for positional lookups
    trading_days = returns.index
    # pandas Series: cumulative product of (1 + daily return) to build a price-level index
    cum_ret = (1 + returns).cumprod()
    frames = {}

    for c in cycles:
        anchor_date = c[anchor]
        # pandas DatetimeIndex.searchsorted: binary search returning integer position of anchor_date
        pos = trading_days.searchsorted(anchor_date)
        start = max(pos - window, 0)
        end   = min(pos + window + 1, len(trading_days))

        # pandas Series.iloc: integer-position slice of the cumulative return series around the anchor
        slice_ = cum_ret.iloc[start:end]
        # normalise so anchor date = 1.0
        # pandas Series.iloc: single-element access by integer position to get the anchor-day level
        anchor_val = cum_ret.iloc[pos] if pos < len(trading_days) else np.nan
        # numpy: np.isnan checks whether the anchor value is NaN before dividing
        if np.isnan(anchor_val) or anchor_val == 0:
            continue

        # pandas Series arithmetic: divide every element by the scalar anchor_val to normalise
        normalised = slice_ / anchor_val
        # numpy.arange: create integer offsets [-window, ..., -1, 0, 1, ..., +window] as the new index
        offsets = np.arange(start - pos, end - pos)
        # pandas Series: construct a Series with integer offsets as index and normalised returns as values
        frames[c["label"]] = pd.Series(normalised.values, index=offsets)

    if not frames:
        # pandas DataFrame: return an empty DataFrame when no cycles have data
        return pd.DataFrame()

    # pandas DataFrame: align multiple Series (one per cycle) on their shared integer-offset index
    return pd.DataFrame(frames)
