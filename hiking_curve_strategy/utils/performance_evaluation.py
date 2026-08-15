"""
Generic backtest utilities — reusable across any strategy.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def annualised_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Annualised return, vol, Sharpe, max drawdown for strategy and buy-hold bond."""
    rows = []
    for col, series in [("payer_strategy", df["strat_ret"]), ("buy_hold_bonds", df["bond_ret"])]:
        r = series.dropna()
        ann_ret = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = (1 + r).cumprod()
        roll_max = cum.cummax()
        max_dd  = ((cum - roll_max) / roll_max).min()
        rows.append({
            "strategy":   col,
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol":    round(ann_vol * 100, 2),
            "sharpe":     round(sharpe, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "n_days":     len(r),
        })
    return pd.DataFrame(rows).set_index("strategy")


def rolling_sharpe(df: pd.DataFrame, window: int = TRADING_DAYS) -> pd.Series:
    """Rolling annualised Sharpe ratio for the payer strategy."""
    r = df["strat_ret"].dropna()
    roll_mean = r.rolling(window).mean() * TRADING_DAYS
    roll_vol  = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    return (roll_mean / roll_vol).rename("rolling_sharpe")


def cycle_matched_sharpe(df: pd.DataFrame, cycles: list[dict]) -> dict:
    """
    Aggregate Sharpe computed only over the periods we actually hold the payer.

    For each cycle, extract the payer-active window (signal == -1) and the
    equivalent buy-hold window (same dates, bond_ret).  Pool all those windows
    together into two single return streams, then compute annualised Sharpe for
    each.  This avoids diluting the Sharpe with flat/off-signal days and gives
    a like-for-like comparison of payer vs buy-hold *in the same market windows*.
    """
    payer_days: list[pd.Series] = []
    bh_days:    list[pd.Series] = []

    for c in cycles:
        mask = (df.index >= c["first_hike"]) & (df.index <= c["last_hike"]) & (df["signal"] == -1)
        payer_days.append(df.loc[mask, "strat_ret"].dropna())
        bh_days.append(df.loc[mask, "bond_ret"].dropna())

    def _sharpe(parts: list[pd.Series]) -> float:
        r = pd.concat(parts).dropna()
        if len(r) < 5:
            return float("nan")
        ann_ret = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        return round(ann_ret / ann_vol, 3) if ann_vol > 0 else float("nan")

    return {
        "payer_sharpe":   _sharpe(payer_days),
        "bh_sharpe":      _sharpe(bh_days),
        "payer_days":     sum(len(p) for p in payer_days),
    }


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
    trading_days = returns.index
    cum_ret = (1 + returns).cumprod()
    frames = {}

    for c in cycles:
        anchor_date = c[anchor]
        pos = trading_days.searchsorted(anchor_date)
        start = max(pos - window, 0)
        end   = min(pos + window + 1, len(trading_days))

        slice_ = cum_ret.iloc[start:end]
        # normalise the window so the anchor day sits at 1.0
        anchor_val = cum_ret.iloc[pos] if pos < len(trading_days) else np.nan
        if np.isnan(anchor_val) or anchor_val == 0:
            continue

        normalised = slice_ / anchor_val
        offsets = np.arange(start - pos, end - pos)
        frames[c["label"]] = pd.Series(normalised.values, index=offsets)

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames)
