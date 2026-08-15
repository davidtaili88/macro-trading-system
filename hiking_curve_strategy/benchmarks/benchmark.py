"""
Oracle benchmark for the hiking cycle 2-year payer strategy — DGS2.

Uses perfect hindsight of actual FOMC hike dates to measure the best possible
performance of the strategy timing rule (an UPPER BOUND — NOT tradeable, since it
requires knowing future FOMC decisions):
  - Enter payer 60 trading days BEFORE the first hike
  - Exit payer 30 trading days BEFORE the last hike (~penultimate hike, book Fig 5.6)

Runs on DGS2 (FRED constant-maturity 2yr), reconstructed into returns two ways:
    price-only:  r_t = -D * Δy_t                        (pure yield move, no carry)
    total:       r_t = price + carry_fund + carry_roll  (fetch_dgs2_full_pnl)
DGS2 reaches back to 1976, covering every historical hiking cycle. The total
line uses the SAME price+funding+roll formula as strategies/signal_trade_dgs.py,
so the oracle total-return line is directly comparable to the signal's
total-return line — the only difference is oracle vs signal timing. See
data.fetch_dgs2_full_pnl().

CYCLE CONSTANTS (single source of truth for the whole package)
-------------------------------------------------------------
FED_HIKE_CYCLES           — every hiking cycle with data (first/last hike),
                            DERIVED at import from the DFEDTAR/DFEDTARL target
                            series via signal_logic.derive_fed_hike_cycles (no
                            hand-typed dates). Filtered to first hike >= 1990.
POST2003_FED_HIKE_CYCLES  — the modern subset (first hike >= 2003): the 3-cycle
                            set strategies/*.py and resume_verification/*.py
                            import for the oracle overlay, and the window over
                            which tradable futures price data also exists.
                            Derived from FED_HIKE_CYCLES so there is ONE source —
                            change the derivation and both follow.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmarks/, utils/ on sys.path
from data         import fetch_carryless_dgs2_returns, fetch_dgs2_full_pnl
import backtest
from backtest     import annualised_stats, rolling_sharpe, event_time_returns
from plot         import equity_curve, event_time_plot, rolling_sharpe_plot, cycle_breakdown
from signal_logic import _load_signal_data, derive_fed_hike_cycles


# ── cycle constants (DERIVED from the FRED target series — single source) ────
# FED_HIKE_CYCLES: raw Fed hiking cycles (first/last hike per cycle), computed by
# signal_logic.derive_fed_hike_cycles from the actual DFEDTAR/DFEDTARL target-rate
# changes — the ground-truth oracle of what the Fed did, with NO strategy logic.
# These are RAW hike dates; the -60/-30 oracle timing offsets are applied later by
# oracle_windows(), not here. Filtered to first hike >= 1990 (the practical data
# start; drops a 1984 cycle FRED can't fully support for the signal). min_hikes=2
# DROPS the lone single-hike Mar-1997 event: one isolated hike is not a sustained
# tightening campaign, so it is not a "hiking cycle" the payer thesis can capture.
# This is a DISCRETIONARY threshold (why 2 and not 3? a judgment call on what
# counts as a cycle) and must be disclosed as such, not presented as data-derived.
#
# Shape: list[dict], one dict per cycle, each with a "label" str and two
# pd.Timestamp hike dates. The derivation currently yields (>=1990, min_hikes=2):
#     [
#         {"label": "1994-1995", "first_hike": pd.Timestamp("1994-02-04"), "last_hike": pd.Timestamp("1995-02-01")},
#         # 1997-1997 (lone Mar-1997 hike) is EXCLUDED by min_hikes=2
#         {"label": "1999-2000", "first_hike": pd.Timestamp("1999-06-30"), "last_hike": pd.Timestamp("2000-05-16")},
#         {"label": "2004-2006", "first_hike": pd.Timestamp("2004-06-30"), "last_hike": pd.Timestamp("2006-06-29")},
#         {"label": "2015-2018", "first_hike": pd.Timestamp("2015-12-16"), "last_hike": pd.Timestamp("2018-12-20")},
#         {"label": "2022-2023", "first_hike": pd.Timestamp("2022-03-17"), "last_hike": pd.Timestamp("2023-07-27")},
#     ]
# Regenerated at import from live FRED data, so this comment is illustrative only.
_FRED_KEY_FOR_CYCLES = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
_, _fed_target = _load_signal_data(_FRED_KEY_FOR_CYCLES, start="1982-10-01")
FED_HIKE_CYCLES: list[dict] = derive_fed_hike_cycles(_fed_target, start_year=1990, min_hikes=2)

# POST2003_FED_HIKE_CYCLES: the modern subset (first hike >= 2003) — the 3-cycle
# set strategies/*.py and resume_verification/*.py use for the oracle overlay, and
# the window where tradable price data also exists. Derived from FED_HIKE_CYCLES
# so there is ONE source.
POST2003_FED_HIKE_CYCLES: list[dict] = [c for c in FED_HIKE_CYCLES
                                        if c["first_hike"].year >= 2003]

ENTRY_DAYS_BEFORE_FIRST = 60
EXIT_DAYS_BEFORE_LAST   = 30


def oracle_windows(trading_days: pd.DatetimeIndex,
                   cycles: list[dict] = FED_HIKE_CYCLES) -> list[dict]:
    """The ACTUAL (entry, exit) dates the oracle trades for each cycle.

    Single source of truth for oracle timing: snaps ENTRY_DAYS_BEFORE_FIRST /
    EXIT_DAYS_BEFORE_LAST onto the real trading calendar. entry = the trading day
    ENTRY_DAYS_BEFORE_FIRST days before first_hike; exit = EXIT_DAYS_BEFORE_LAST
    days before last_hike. `trading_days` is the DatetimeIndex of the return
    series the oracle is measured on (e.g. ret.index) — passed in, not fetched,
    matching calc_strat_ret / stats_per_cycle. Positions clamp at 0 so a cycle
    starting before the data begins doesn't underflow.

    Returns one dict per cycle: {"label", "first_hike", "last_hike", "entry", "exit"}
    so callers can use the offset-adjusted entry/exit directly and never re-derive
    the searchsorted-minus-offset arithmetic themselves."""
    out = []
    for c in cycles:
        fh_pos = trading_days.searchsorted(c["first_hike"])
        lh_pos = trading_days.searchsorted(c["last_hike"])
        entry  = trading_days[max(fh_pos - ENTRY_DAYS_BEFORE_FIRST, 0)]
        exit_  = trading_days[max(lh_pos - EXIT_DAYS_BEFORE_LAST,   0)]
        out.append({**c, "entry": entry, "exit": exit_})
    return out


def stats_per_cycle(df: pd.DataFrame, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Per-cycle breakdown: entry/exit dates (after applying the -60/-30 offsets),
    days held, total compounded return."""
    rows = []
    for w in oracle_windows(ret.index, cycles):
        entry, exit_ = w["entry"], w["exit"]
        mask   = (df.index >= entry) & (df.index <= exit_) & (df["signal"] == -1)
        r      = df.loc[mask, "strat_ret"].dropna()
        cum    = (1 + r).prod() - 1
        rows.append({
            "cycle":       w["label"],
            "entry":       str(entry.date()),
            "exit":        str(exit_.date()),
            "days_held":   len(r),
            "total_ret_%": round(cum * 100, 2),
        })
    return pd.DataFrame(rows).set_index("cycle")


def _run(name: str, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Run the oracle payer backtest for one return series, print summary + per-cycle
    table, return the results df. Uses the -60/-30 oracle timing offsets."""
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
    fred_key = os.environ.get("FRED_API_KEY", "")
    if not fred_key:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".env"))
        fred_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
    if not fred_key:
        print("Set FRED_API_KEY and re-run.")
        return

    print("Fetching DGS2 data (FRED, from 1990)...")
    # price-only: pure yield-move oracle, no carry
    dgs2_ret   = fetch_carryless_dgs2_returns(fred_key, start="1990-01-01")
    # total: price + funding + roll on the SAME formula the signal strategy uses,
    # so the oracle total line is apples-to-apples with the signal total line
    # (LONG-holder perspective; calc_strat_ret's payer signal=-1 negates it).
    pnl_full   = fetch_dgs2_full_pnl(fred_key, start="1990-01-01")
    dgs2_total = pnl_full["total_ret"].rename("ret")

    print(f"  DGS2 price-only: {dgs2_ret.index[0].date()} to {dgs2_ret.index[-1].date()}")
    print(f"  DGS2 total:      {dgs2_total.index[0].date()} to {dgs2_total.index[-1].date()}")

    print("\nOracle timing: entry -60td before first hike, exit -30td before last hike")
    print("(Perfect hindsight — NOT tradeable. Upper bound on strategy performance.)\n")

    # full-history oracle, every hiking cycle DGS2 covers (back to 1990)
    df_price = _run("DGS2 price-only (D breathes, no carry)  — ALL cycles",
                    dgs2_ret, FED_HIKE_CYCLES)
    df_total = _run("DGS2 total (price + carry)              — ALL cycles",
                    dgs2_total, FED_HIKE_CYCLES)

    # --- Plots (price-only, full history) ---
    print("\nGenerating plots...")
    ev_first = event_time_returns(dgs2_ret, FED_HIKE_CYCLES, anchor="first_hike", window=120)
    ev_last  = event_time_returns(dgs2_ret, FED_HIKE_CYCLES, anchor="last_hike",  window=120)
    equity_curve(df_price, FED_HIKE_CYCLES, instrument="DGS2 Price-Only (oracle)")
    event_time_plot(ev_first, anchor_label="first hike — DGS2 oracle")
    event_time_plot(ev_last,  anchor_label="last hike — DGS2 oracle")
    cycle_breakdown(df_price, FED_HIKE_CYCLES, instrument="DGS2 Price-Only (oracle)")
    rolling_sharpe_plot(rolling_sharpe(df_price), cycles=FED_HIKE_CYCLES, df=df_price,
                        instrument="DGS2 Price-Only (oracle)")
    plt.show()


if __name__ == "__main__":
    main()
