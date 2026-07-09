"""
Signal-driven backtest for the hiking cycle 2-year payer strategy — ZT futures.

Uses the same market-driven signal as signal_market.py (FRED spread thresholds)
applied to CME 2yr Treasury futures (ZT=F) instead of the DGS2 duration proxy.

ZT advantages over DGS2:
  - Carry is netted into the roll basis rather than bleeding P&L daily
  - Directly tradable instrument
  - auto_adjust=True removes roll price gaps so pct_change() gives clean returns

ZT limitations:
  - Data only available from ~2002 (misses 1994, 1999 hiking cycles)
  - Quarterly CME roll cost deducted on each roll date when short (~0.8bp/roll)

Data sources:
  ZT=F   — CME 2yr Treasury futures (Yahoo Finance, back-adjusted)
  Signal — inherited from signal_market.detect_signal (FRED spreads)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data      import fetch_zt, compute_returns, get_zt_roll_dates
from backtest  import annualised_stats, rolling_sharpe, event_time_returns, calc_strat_ret
from plot      import equity_curve, event_time_plot, rolling_sharpe_plot, cycle_breakdown
from benchmark import ORACLE_CYCLES
from signal_trade_dgs import detect_signal, signal_to_cycles, stats_per_cycle


# ZT roll cost: 1 tick = $15.625 on a $200k notional contract.
# In return terms: 15.625 / (200_000 * ~110 price) ≈ 0.71bp. Use 0.8bp to be conservative.
ZT_ROLL_COST = 0.000080  # 0.8bp per roll, deducted on each roll date when short


def _run(name: str, ret: pd.Series, cycles: list[dict],
         roll_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Run signal-driven backtest, print summary, return df."""
    df = calc_strat_ret(
        ret, cycles,
        entry_days_before_first=0,
        exit_days_before_last=0,
        roll_dates=roll_dates,
        roll_cost=ZT_ROLL_COST if roll_dates is not None else 0.0,
    )
    stats = annualised_stats(df)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(stats.to_string())
    print()
    print(stats_per_cycle(df, ret, cycles).to_string())
    return df


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("Set FRED_API_KEY in .env and re-run.")
        return

    print("Detecting signal (FRED spreads, from 1976)...")
    signal = detect_signal(api_key, start="1976-01-01")
    cycles_all = signal_to_cycles(signal)

    if not cycles_all:
        print("No signal episodes detected — check thresholds.")
        return

    print(f"  {len(cycles_all)} episode(s):")
    for c in cycles_all:
        print(f"    {c['label']}  entry={c['first_hike'].date()}  exit={c['last_hike'].date()}")

    print("\nFetching ZT futures (Yahoo Finance, back to 2002)...")
    zt_prices  = fetch_zt(start="2002-01-01")
    ret_zt     = compute_returns(zt_prices)
    print(f"  ZT: {ret_zt.index[0].date()} to {ret_zt.index[-1].date()}")

    cycles_zt  = [c for c in cycles_all if c["first_hike"] >= ret_zt.index[0]]
    roll_dates = get_zt_roll_dates(str(ret_zt.index[0].date()), str(ret_zt.index[-1].date()))

    if not cycles_zt:
        print("No signal episodes fall within ZT data range — nothing to backtest.")
        return

    df_zt = _run(
        f"ZT futures signal-driven payer (roll cost {ZT_ROLL_COST*10_000:.1f}bp/roll)",
        ret_zt, cycles_zt, roll_dates=roll_dates,
    )

    print("\nGenerating plots...")
    ev_first = event_time_returns(ret_zt, cycles_zt, anchor="first_hike", window=120)
    ev_last  = event_time_returns(ret_zt, cycles_zt, anchor="last_hike",  window=120)
    equity_curve(df_zt, cycles_zt, instrument="ZT Futures Signal-Driven", oracle_cycles=ORACLE_CYCLES)
    event_time_plot(ev_first, anchor_label="signal on — ZT")
    event_time_plot(ev_last,  anchor_label="signal off — ZT")
    cycle_breakdown(df_zt, cycles_zt, instrument="ZT Futures Signal-Driven")
    rolling_sharpe_plot(rolling_sharpe(df_zt), cycles=cycles_zt, df=df_zt, instrument="ZT Futures Signal-Driven")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
