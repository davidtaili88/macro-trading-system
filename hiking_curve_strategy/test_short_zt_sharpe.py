"""
Baseline test: continuously short ZT futures (2yr Treasury futures) across the
full available history. No signal, no cycles — just permanent short exposure.

This gives a rough sense of what the raw short-bond Sharpe looks like without
any market-timing. Useful as a floor/ceiling reference for the signal-driven
strategy in signal_market.py.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

TRADING_DAYS = 252
ROLLING_WINDOW = TRADING_DAYS  # 1-year rolling Sharpe


def fetch_zt_returns(start: str = "2002-01-01") -> pd.Series:
    """Download ZT futures prices and compute daily pct returns."""
    raw = yf.download("ZT=F", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.squeeze()
    prices = raw.dropna()
    ret = prices.pct_change().rename("zt_ret")
    return ret.dropna()


def short_zt_returns(zt_ret: pd.Series) -> pd.Series:
    """Flip the sign: continuously short ZT = negate its daily returns."""
    return (-zt_ret).rename("short_zt_ret")


def overall_sharpe(ret: pd.Series) -> dict:
    ann_ret = ret.mean() * TRADING_DAYS
    ann_vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + ret).cumprod()
    roll_max = cum.cummax()
    max_dd = ((cum - roll_max) / roll_max).min()
    return {
        "ann_return_%": round(ann_ret * 100, 2),
        "ann_vol_%":    round(ann_vol * 100, 2),
        "sharpe":       round(sharpe, 3),
        "max_dd_%":     round(max_dd * 100, 2),
        "n_days":       len(ret),
    }


def rolling_sharpe(ret: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    roll_mean = ret.rolling(window).mean() * TRADING_DAYS
    roll_vol  = ret.rolling(window).std()  * np.sqrt(TRADING_DAYS)
    return (roll_mean / roll_vol).rename("rolling_sharpe")


def plot(zt_ret: pd.Series, short_ret: pd.Series):
    rs = rolling_sharpe(short_ret)
    cum_short = (1 + short_ret).cumprod()
    cum_long  = (1 + zt_ret).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Continuously Short ZT Futures — Full History", fontsize=13, fontweight="bold")

    # ── equity curve ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(cum_short.index, cum_short.values, color="firebrick", lw=1.5, label="Short ZT (payer)")
    ax.plot(cum_long.index,  cum_long.values,  color="steelblue", lw=1.0, alpha=0.55, linestyle="--", label="Long ZT (buy-hold)")
    ax.axhline(1.0, color="gray", lw=0.6, linestyle=":")
    ax.set_ylabel("Cumulative growth")
    ax.legend(fontsize=9)
    ax.set_title("Equity Curve", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))

    # ── rolling 1-yr Sharpe ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(rs.index, rs.values, color="darkorange", lw=1.3)
    ax.axhline(0.0, color="gray", lw=0.8, linestyle="--")
    ax.axhline(rs.mean(), color="firebrick", lw=0.9, linestyle=":", label=f"mean = {rs.mean():.2f}")
    ax.fill_between(rs.index, 0, rs.values, where=(rs > 0), alpha=0.15, color="green")
    ax.fill_between(rs.index, 0, rs.values, where=(rs < 0), alpha=0.15, color="red")
    ax.set_ylabel("Rolling Sharpe (1yr)")
    ax.legend(fontsize=9)
    ax.set_title("Rolling 1-Year Sharpe — Short ZT", fontsize=10)

    # ── drawdown ──────────────────────────────────────────────────────────────
    ax = axes[2]
    roll_max = cum_short.cummax()
    drawdown = (cum_short - roll_max) / roll_max * 100
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="firebrick", lw=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown — Short ZT", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()


def main():
    print("Downloading ZT futures prices (Yahoo Finance)...")
    zt_ret   = fetch_zt_returns(start="2002-01-01")
    short_ret = short_zt_returns(zt_ret)

    print(f"  ZT data: {zt_ret.index[0].date()} to {zt_ret.index[-1].date()}  ({len(zt_ret)} days)\n")

    stats_short = overall_sharpe(short_ret)
    stats_long  = overall_sharpe(zt_ret)

    print("=" * 50)
    print("  Continuously SHORT ZT futures")
    print("=" * 50)
    for k, v in stats_short.items():
        print(f"  {k:<20} {v}")

    print()
    print("=" * 50)
    print("  Buy-and-hold LONG ZT (reference)")
    print("=" * 50)
    for k, v in stats_long.items():
        print(f"  {k:<20} {v}")

    rs = rolling_sharpe(short_ret)
    print(f"\nRolling 1-yr Sharpe (short ZT):")
    print(f"  mean   = {rs.mean():.3f}")
    print(f"  median = {rs.median():.3f}")
    print(f"  min    = {rs.min():.3f}")
    print(f"  max    = {rs.max():.3f}")
    pct_positive = (rs > 0).mean() * 100
    print(f"  % of time Sharpe > 0: {pct_positive:.1f}%")

    plot(zt_ret, short_ret)
    plt.show()


if __name__ == "__main__":
    main()
