"""
Backtest engine for TIPS strategy variants.

Computes strategy returns, cumulative equity, and summary stats
(annualized return, vol, Sharpe/IR, max drawdown) for each signal.
"""

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def run(spread: pd.Series, signals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Apply each signal column to the spread return series.
    Returns dict of {signal_name -> daily stats DataFrame}.
    """
    results = {}
    for name in signals.columns:
        sig = signals[name].reindex(spread.index).fillna(0)
        strat = spread * sig
        cum = (1 + strat).cumprod()
        results[name] = pd.DataFrame({
            "signal":      sig,
            "spread_ret":  spread,
            "strat_ret":   strat,
            "cum_equity":  cum,
        })
    return results


def summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in results.items():
        r = df["strat_ret"].dropna()
        ann_ret = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        ir      = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = df["cum_equity"].dropna()
        roll_max = cum.cummax()
        max_dd  = ((cum - roll_max) / roll_max).min()
        rows.append({
            "strategy":   name,
            "ann_return": round(ann_ret * 100, 2),
            "ann_vol":    round(ann_vol * 100, 2),
            "IR":         round(ir, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "n_days":     len(r),
        })
    return pd.DataFrame(rows).set_index("strategy")
