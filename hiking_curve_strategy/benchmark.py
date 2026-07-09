"""
Oracle benchmark for the hiking cycle 2-year payer strategy.

Uses perfect hindsight of actual FOMC hike dates to measure the best
possible performance of the strategy timing rule:
  - Enter payer 60 trading days BEFORE the first hike
  - Exit payer 30 trading days BEFORE the last hike (~penultimate hike, per book Fig 5.6)

Run across two instruments to show the carry drag problem with cash bonds:
  SHY  — iShares 1-3yr Treasury ETF. Cash bond: coupon carry bleeds against
          the short position every day, partially offsetting yield-rise gains.
  ZT=F — CME 2yr Treasury futures. Carry is baked into the futures basis and
          netted out at roll, so P&L tracks the pure duration/yield move.

This is NOT a real trading strategy — it requires knowing future FOMC decisions.
Its purpose is to set an upper-bound reference for the signal-driven version.

Only cycles with data available post-2002 are included:
  2004-2006, 2015-2018, 2022-2023.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data     import fetch_shy, fetch_zt, compute_returns
import backtest
from backtest import annualised_stats, rolling_sharpe, event_time_returns
from plot     import equity_curve, event_time_plot, rolling_sharpe_plot, cycle_breakdown


ORACLE_CYCLES: list[dict] = [
    {"label": "2004-2006", "first_hike": pd.Timestamp("2004-06-30"), "last_hike": pd.Timestamp("2006-06-29")},
    {"label": "2015-2018", "first_hike": pd.Timestamp("2015-12-16"), "last_hike": pd.Timestamp("2018-12-20")},
    {"label": "2022-2023", "first_hike": pd.Timestamp("2022-03-17"), "last_hike": pd.Timestamp("2023-07-27")},
]

ENTRY_DAYS_BEFORE_FIRST = 60
EXIT_DAYS_BEFORE_LAST   = 30


def stats_per_cycle(df: pd.DataFrame, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Per-cycle breakdown: entry/exit dates, days held, total compounded return."""
    td   = ret.index
    rows = []
    for c in cycles:
        fh_pos = td.searchsorted(c["first_hike"])
        lh_pos = td.searchsorted(c["last_hike"])
        entry  = td[max(fh_pos - ENTRY_DAYS_BEFORE_FIRST, 0)]
        exit_  = td[max(lh_pos - EXIT_DAYS_BEFORE_LAST,   0)]
        mask   = (df.index >= entry) & (df.index <= exit_) & (df["signal"] == -1)
        r      = df.loc[mask, "strat_ret"].dropna()
        cum    = (1 + r).prod() - 1
        rows.append({
            "cycle":       c["label"],
            "entry":       str(entry.date()),
            "exit":        str(exit_.date()),
            "days_held":   len(r),
            "total_ret_%": round(cum * 100, 2),
        })
    return pd.DataFrame(rows).set_index("cycle")


def _run(name: str, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Run the payer backtest for one instrument, print summary + per-cycle table, return results df."""
    df    = backtest.calc_strat_ret(ret, cycles,
                entry_days_before_first=ENTRY_DAYS_BEFORE_FIRST,
                exit_days_before_last=EXIT_DAYS_BEFORE_LAST)
    stats = backtest.annualised_stats(df)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(stats.to_string())
    print()
    print(stats_per_cycle(df, ret, cycles).to_string())
    return df


def main():
    print("Fetching price data...")
    shy_prices = fetch_shy()
    zt_prices  = fetch_zt()
    shy_ret    = compute_returns(shy_prices)
    zt_ret     = compute_returns(zt_prices)
    print(f"  SHY: {shy_prices.index[0].date()} to {shy_prices.index[-1].date()}")
    print(f"  ZT:  {zt_prices.index[0].date()}  to {zt_prices.index[-1].date()}")

    print("\nOracle timing: entry -60td before first hike, exit -30td before last hike")
    print("(Perfect hindsight — NOT tradeable. Upper bound on strategy performance.)\n")

    df_shy = _run("SHY cash bond     (carry bleeds against payer)", shy_ret, ORACLE_CYCLES)
    df_zt  = _run("ZT futures        (carry netted in basis — purer duration)", zt_ret, ORACLE_CYCLES)

    # Side-by-side comparison of payer_strategy row across both instruments
    print(f"\n{'='*55}")
    print("  Side-by-side: payer_strategy row only")
    print(f"{'='*55}")
    rows = {}
    for name, df in [("SHY_cash_bond", df_shy), ("ZT_futures", df_zt)]:
        rows[name] = annualised_stats(df).loc["payer_strategy"]
    print(pd.DataFrame(rows).T.to_string())

    # --- Plots ---
    print("\nGenerating plots...")

    # SHY
    ev_first_shy = event_time_returns(shy_ret, ORACLE_CYCLES, anchor="first_hike", window=120)
    ev_last_shy  = event_time_returns(shy_ret, ORACLE_CYCLES, anchor="last_hike",  window=120)
    equity_curve(df_shy, ORACLE_CYCLES, instrument="SHY Cash Bond")
    event_time_plot(ev_first_shy, anchor_label="first hike — SHY oracle")
    event_time_plot(ev_last_shy,  anchor_label="last hike — SHY oracle")
    cycle_breakdown(df_shy, ORACLE_CYCLES, instrument="SHY Cash Bond")
    rolling_sharpe_plot(rolling_sharpe(df_shy), cycles=ORACLE_CYCLES, df=df_shy, instrument="SHY Cash Bond")

    # ZT
    ev_first_zt = event_time_returns(zt_ret, ORACLE_CYCLES, anchor="first_hike", window=120)
    ev_last_zt  = event_time_returns(zt_ret, ORACLE_CYCLES, anchor="last_hike",  window=120)
    equity_curve(df_zt, ORACLE_CYCLES, instrument="ZT Futures")
    event_time_plot(ev_first_zt, anchor_label="first hike — ZT oracle")
    event_time_plot(ev_last_zt,  anchor_label="last hike — ZT oracle")
    cycle_breakdown(df_zt, ORACLE_CYCLES, instrument="ZT Futures")
    rolling_sharpe_plot(rolling_sharpe(df_zt), cycles=ORACLE_CYCLES, df=df_zt, instrument="ZT Futures")

    plt.show()


if __name__ == "__main__":
    main()
