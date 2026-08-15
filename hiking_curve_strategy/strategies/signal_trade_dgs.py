"""
Signal-driven backtest for the hiking cycle 2-year payer strategy — DGS2.

Uses the same market-driven signal as signal.py (FRED spread thresholds)
applied to the FRED DGS2 constant-maturity yield, reconstructed into a
duration-approximated return series (data.fetch_dgs2_full_pnl).

DGS2 advantages:
  - DGS2 itself goes back to 1976, but the signal inputs (DGS3MO 1981-09,
    DFEDTAR 1982-09) bind the usable start to 1982-10 — see DATA_START below
  - Full price/carry/roll decomposition available (fetch_dgs2_full_pnl)

DGS2 limitations:
  - As a constant-maturity cash series, carry bleeds against the short position
    every day; the price-only reconstruction isolates the yield-move signal but
    a fully tradable P&L must account for that carry separately

Data sources:
  DGS2, DGS1, DFF — FRED (via data.fetch_dgs2_full_pnl)
  Signal          — inherited from signal.detect_signal (FRED spreads)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401 — registers core/, benchmark/, utils/ on sys.path
from data      import fetch_dgs2_full_pnl, fetch_carryless_dgs2_returns
from backtest  import annualised_stats, rolling_sharpe, event_time_returns, calc_strat_ret, cycle_pnl
from utils.performance_evaluation import cycle_matched_sharpe
from plot      import equity_curve, event_time_plot, rolling_sharpe_plot, cycle_breakdown, carry_decomposition_plot, pnl_components_timeseries
from benchmark import POST2003_FED_HIKE_CYCLES
from signal_logic import detect_signal, signal_to_cycles


def _run(name: str, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Run signal-driven backtest for one instrument, print summary, return df."""
    df = calc_strat_ret(ret, cycles)
    stats = annualised_stats(df)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(stats.to_string())
    per_cycle, pooled = cycle_pnl(ret, cycles)
    print("\nPer-cycle compounded payer P&L:")
    for label, pnl in per_cycle.items():
        print(f"  {label:<40}  {pnl*100:+.2f}%")
    print(f"  {'pooled (all cycles)':<40}  {pooled*100:+.2f}%")
    return df


def main():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
    if not api_key:
        print("Set FRED_API_KEY in .env and re-run.")
        return

    print("Detecting signal (FRED spreads, from 1982)...")
    signal = detect_signal(api_key, start="1982-10-01")
    cycles_all = signal_to_cycles(signal)

    if not cycles_all:
        print("No signal episodes detected — check thresholds.")
        return

    print(f"  {len(cycles_all)} episode(s):")
    for c in cycles_all:
        print(f"    {c['label']}  entry={c['first_hike'].date()}  exit={c['last_hike'].date()}")

    print("\nFetching DGS2 PnL decomposition (FRED, from 1982)...")
    pnl_full = fetch_dgs2_full_pnl(api_key, start="1982-10-01")
    print(f"  DGS2: {pnl_full.index[0].date()} to {pnl_full.index[-1].date()}")

    # price-only series: pure yield-move signal, no carry drag. Sourced from the
    # canonical carryless function (DGS2-only calendar), NOT pnl_full["price_ret"],
    # so the price-only signal never depends on DGS1/DFF availability.
    ret_price = fetch_carryless_dgs2_returns(api_key, start="1982-10-01")
    # full series: price move + funding carry + roll-down carry
    ret_total = pnl_full["total_ret"].rename("ret")

    df_price = _run("DGS2 price-only (no carry)", ret_price, cycles_all)
    df_total = _run("DGS2 total (price + carry)", ret_total, cycles_all)

    # data coverage floor: DGS3MO starts 1981-09-01, DFEDTAR starts 1982-09-27.
    # The practical first day where all signal inputs exist is late 1982, so the
    # whole analysis (signal, fetches, plot axis) is anchored here — nothing the
    # strategy uses exists before this date.
    DATA_START = pd.Timestamp("1982-10-01")

    # aggregated Sharpe: pool only payer-active windows across all cycles
    cms_price = cycle_matched_sharpe(df_price, cycles_all)
    cms_total = cycle_matched_sharpe(df_total, cycles_all)
    print(f"\nCycle-matched Sharpe (payer-active days only, pooled across all cycles):")
    print(f"  Price-only — payer: {cms_price['payer_sharpe']:.3f}   "
          f"buy-hold same window: {cms_price['bh_sharpe']:.3f}   "
          f"({cms_price['payer_days']} days)")
    print(f"  Total      — payer: {cms_total['payer_sharpe']:.3f}   "
          f"buy-hold same window: {cms_total['bh_sharpe']:.3f}   "
          f"({cms_total['payer_days']} days)")

    print("\nGenerating plots...")
    ev_first = event_time_returns(ret_price, cycles_all, anchor="first_hike", window=120)
    ev_last  = event_time_returns(ret_price, cycles_all, anchor="last_hike",  window=120)
    equity_curve(df_price, cycles_all, instrument="DGS2 Price-Only",
                 oracle_cycles=POST2003_FED_HIKE_CYCLES, data_start=DATA_START)
    equity_curve(df_total, cycles_all, instrument="DGS2 Total (with carry)",
                 oracle_cycles=POST2003_FED_HIKE_CYCLES, data_start=DATA_START)
    event_time_plot(ev_first, anchor_label="signal on — DGS2")
    event_time_plot(ev_last,  anchor_label="signal off — DGS2")
    cycle_breakdown(df_price, cycles_all, instrument="DGS2 Price-Only",
                    cycle_matched_sharpes=cms_price)
    cycle_breakdown(df_total, cycles_all, instrument="DGS2 Total (with carry)",
                    cycle_matched_sharpes=cms_total)
    rolling_sharpe_plot(rolling_sharpe(df_price), cycles=cycles_all, df=df_price,
                        instrument="DGS2 Price-Only", data_start=DATA_START)
    rolling_sharpe_plot(rolling_sharpe(df_total), cycles=cycles_all, df=df_total,
                        instrument="DGS2 Total (with carry)", data_start=DATA_START)
    carry_decomposition_plot(pnl_full, cycles_all)
    pnl_components_timeseries(pnl_full, signal)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
