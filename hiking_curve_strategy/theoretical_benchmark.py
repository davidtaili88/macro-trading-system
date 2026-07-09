"""
Theoretical oracle benchmark using DGS2-derived returns.

Verifies that the payer strategy works across the full historical record,
including cycles that SHY and ZT cannot reach (pre-2002).

Return series construction:
    r_t = -D * Δy_t     where D = 1.9, Δy_t = daily DGS2 change in decimal

This is price-only (no carry), so the strategy P&L reflects pure yield moves.
See data.fetch_dgs2_returns() for the derivation.

Also runs SHY and ZT over the overlapping post-2002 window so you can see
directly how much carry drag costs versus the theoretical benchmark.

Oracle timing (perfect hindsight, NOT tradeable):
  Entry: 60 trading days before first hike
  Exit:  30 trading days before last hike (~penultimate hike, book Fig 5.6)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# pandas: labelled time-series and DataFrame operations; pd is the standard alias
import pandas as pd
# numpy: vectorised array math; np is the standard alias
import numpy as np

# sys.path.insert: make the strategy package importable when running this file directly
sys.path.insert(0, os.path.dirname(__file__))
from data     import fetch_shy, fetch_zt, fetch_carryless_dgs2_returns, compute_returns
import backtest
from backtest import calc_strat_ret, annualised_stats, event_time_returns, rolling_sharpe
from plot     import equity_curve, event_time_plot, cycle_breakdown, rolling_sharpe_plot


# ---------------------------------------------------------------------------
# Full historical oracle cycles (from DFEDTAR/DFEDTARL ground truth)
# ---------------------------------------------------------------------------
# pd.Timestamp: converts an ISO date string into a pandas Timestamp for date arithmetic
ALL_CYCLES: list[dict] = [
    {"label": "1994-1995", "first_hike": pd.Timestamp("1994-02-04"), "last_hike": pd.Timestamp("1995-02-03")},
    {"label": "1999-2000", "first_hike": pd.Timestamp("1999-06-30"), "last_hike": pd.Timestamp("2000-05-16")},
    {"label": "2004-2006", "first_hike": pd.Timestamp("2004-06-30"), "last_hike": pd.Timestamp("2006-06-29")},
    {"label": "2015-2018", "first_hike": pd.Timestamp("2015-12-16"), "last_hike": pd.Timestamp("2018-12-20")},
    {"label": "2022-2023", "first_hike": pd.Timestamp("2022-03-17"), "last_hike": pd.Timestamp("2023-07-27")},
]

# list comprehension: filter to cycles whose first hike falls after SHY/ZT data begins
POST2002_CYCLES = [c for c in ALL_CYCLES if c["first_hike"].year >= 2003]

ENTRY_DAYS_BEFORE_FIRST = 60
EXIT_DAYS_BEFORE_LAST   = 30

# Collapse each hiking cycle into 1 row and compare stats per overall cycle. 
def stats_per_cycle(df: pd.DataFrame, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    # DatetimeIndex: the index of the return series, used for trading-day positional lookups
    td   = ret.index
    rows = []
    for c in cycles:
        # Index.searchsorted: binary search returning the integer position of the nearest date 
        # Integer position is in td, value equivalent to the hike dates in the ALL_CYCLES list
        fh_pos = td.searchsorted(c["first_hike"])
        lh_pos = td.searchsorted(c["last_hike"])
        # max(..., 0): clamp to avoid negative positions if the cycle starts before data begins
        # Find indices/integer positions of entry and exit dates
        entry  = td[max(fh_pos - ENTRY_DAYS_BEFORE_FIRST, 0)]
        exit_  = td[max(lh_pos - EXIT_DAYS_BEFORE_LAST,   0)]
        # boolean mask: selects rows within the active payer window for this cycle
        mask   = (df.index >= entry) & (df.index <= exit_) & (df["signal"] == -1)
        # .loc function returns filtered dataframe. Selects data where mask = True, and in "strat_ret"
        r      = df.loc[mask, "strat_ret"].dropna()
        # (1+r).prod() - 1: compound daily returns into scalar value of total return over the holding period
        cum    = (1 + r).prod() - 1
        #Append one dictionary per stat in each cycle
        rows.append({
            "cycle":       c["label"],
            "entry":       str(entry.date()),
            "exit":        str(exit_.date()),
            "days_held":   len(r),
            "total_ret_%": round(cum * 100, 2),
        })
    # pd.DataFrame.set_index: use the cycle label as the row label for cleaner printing
    return pd.DataFrame(rows).set_index("cycle")

# Returns strat_ret for the strategy we test. 
def _run(name: str, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    # backtest.calc_strat_ret: applies the payer signal to the return series and returns a results DataFrame
    df    = backtest.calc_strat_ret(ret, cycles,
                entry_days_before_first=ENTRY_DAYS_BEFORE_FIRST,
                exit_days_before_last=EXIT_DAYS_BEFORE_LAST)
    # backtest.annualised_stats: computes annualised return, vol, Sharpe, and max drawdown
    stats = backtest.annualised_stats(df)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    # DataFrame.to_string: renders the full summary table without truncation
    print(stats.to_string())
    print()
    print(stats_per_cycle(df, ret, cycles).to_string())
    return df


def main():
    # os.environ.get: read the API key from the shell environment first
    fred_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
    if not fred_key:
        # dotenv.load_dotenv: fall back to loading the .env file one directory up
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        print("Set FRED_API_KEY and re-run.")
        return

    print("Fetching data...")
    # data.fetch_dgs2_returns: reconstructs daily price-only returns from FRED DGS2 via r_t = -D * Δy_t
    dgs2_ret   = fetch_carryless_dgs2_returns(fred_key, start="1990-01-01")
    # data.fetch: downloads SHY adjusted close prices from Yahoo Finance
    shy_prices = fetch_shy()
    # data.fetch_zt: downloads ZT=F (2yr Treasury futures) adjusted close prices from Yahoo Finance
    zt_prices  = fetch_zt()
    # data.compute_returns: converts a price DataFrame into a daily pct_change return Series
    shy_ret    = compute_returns(shy_prices)
    zt_ret     = compute_returns(zt_prices)

    print(f"  DGS2 theoretical: {dgs2_ret.index[0].date()} to {dgs2_ret.index[-1].date()}")
    print(f"  SHY cash bond:    {shy_ret.index[0].date()} to {shy_ret.index[-1].date()}")
    print(f"  ZT futures:       {zt_ret.index[0].date()}  to {zt_ret.index[-1].date()}")

    print("\nOracle timing: entry -60td before first hike, exit -30td before last hike")
    print("(Perfect hindsight — NOT tradeable. Upper bound on strategy performance.)\n")

    # DGS2 theoretical returns — full history including pre-2002 cycles
    df_dgs2 = _run(
        "DGS2 theoretical  (price-only, D=1.921, no carry)  — ALL cycles",
        dgs2_ret, ALL_CYCLES,
    )

    # Post-2002 only for apples-to-apples comparison against SHY and ZT
    df_dgs2_p = _run(
        "DGS2 theoretical  (post-2002 cycles only, for comparison)",
        dgs2_ret, POST2002_CYCLES,
    )
    df_shy = _run(
        "SHY cash bond     (carry bleeds against payer)",
        shy_ret, POST2002_CYCLES,
    )
    df_zt = _run(
        "ZT futures        (carry netted in basis — purer duration)",
        zt_ret, POST2002_CYCLES,
    )

    # Side-by-side summary: extract just the payer_strategy row from each instrument's summary
    print(f"\n{'='*55}")
    print("  Side-by-side: post-2002 payer_strategy row only")
    print(f"{'='*55}")
    rows = {}
    for name, df, ret, cycles in [
        ("DGS2_theoretical", df_dgs2_p, dgs2_ret, POST2002_CYCLES),
        ("SHY_cash_bond",    df_shy,    shy_ret,   POST2002_CYCLES),
        ("ZT_futures",       df_zt,     zt_ret,    POST2002_CYCLES),
    ]:
        # DataFrame.loc: select the payer_strategy row by label from the summary table
        # Annualised stats converts the strategy name as the row label
        s = annualised_stats(df).loc["payer_strategy"]
        rows[name] = s
    # pd.DataFrame(rows).T: build a wide table with one row of stats per instrument
    comparison = pd.DataFrame(rows).T
    print(comparison.to_string())

    # Plots — DGS2 full history
    print("\nGenerating plots...")
    # backtest.event_time_returns: collects normalised cumulative returns in event time around each anchor date
    ev_first = event_time_returns(dgs2_ret, ALL_CYCLES, anchor="first_hike", window=120)
    ev_last  = event_time_returns(dgs2_ret, ALL_CYCLES, anchor="last_hike",  window=120)
    # plot.equity_curve: cumulative strategy vs buy-hold with cycle shading
    equity_curve(df_dgs2, ALL_CYCLES, instrument="DGS2 Theoretical")
    # plot.event_time_plot: normalised bond returns in event time, median + percentile band
    event_time_plot(ev_first, anchor_label="first hike — DGS2 theoretical")
    event_time_plot(ev_last,  anchor_label="last hike — DGS2 theoretical")
    # plot.cycle_breakdown: per-cycle bar chart of annualised return and Sharpe
    cycle_breakdown(df_dgs2, ALL_CYCLES, instrument="DGS2 Theoretical")
    # plot.rolling_sharpe_plot: 252-day rolling Sharpe with active payer windows shaded
    roll_sh = rolling_sharpe(df_dgs2)
    rolling_sharpe_plot(roll_sh, cycles=ALL_CYCLES, df=df_dgs2, instrument="DGS2 Theoretical")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
