"""
Visualization for TIPS strategy backtest results.
Produces two figures:
  1. Cumulative equity curves for all three signal variants
  2. Rolling 252-day IR for the copper_ma strategy
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curves(results: dict[str, pd.DataFrame], save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"base": "steelblue", "oil_ma": "darkorange", "copper_ma": "seagreen"}
    labels = {
        "base":      "Base (always long TIPS/short nominal)",
        "oil_ma":    "Oil > 55dma filter",
        "copper_ma": "Copper > 55dma filter",
    }
    for name, df in results.items():
        eq = df["cum_equity"].dropna()
        ax.plot(eq.index, eq, label=labels[name], color=colors.get(name), lw=1.4)

    ax.set_title("TIPS vs. Nominal: Cumulative Equity by Signal Filter", fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def rolling_ir(results: dict[str, pd.DataFrame], strategy: str = "copper_ma",
               window: int = TRADING_DAYS, save_path: str | None = None):
    df = results[strategy]
    r = df["strat_ret"].dropna()
    roll_mean = r.rolling(window).mean() * TRADING_DAYS
    roll_vol  = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    roll_ir   = roll_mean / roll_vol

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(roll_ir.index, roll_ir, color="seagreen", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Rolling {window}-day IR — {strategy}", fontsize=13)
    ax.set_ylabel("Information Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
