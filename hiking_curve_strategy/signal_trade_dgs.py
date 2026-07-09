"""
Signal-driven backtest for the hiking cycle 2-year payer strategy.

Both entry AND exit are fully market-driven — no oracle FOMC dates used.

Entry: first day BOTH spread conditions hold inside a confirmed post-easing hold:
  - 3mo spread (DGS3MO - DFF) > 12bp  → hike priced within ~3 months
  - 1yr spread (DGS1 - DFF)   > 25bp  → market broadly pricing hikes

Exit: first day EITHER spread collapses past its exit threshold:
  - 1yr spread < -25bp  → market pricing net cuts over the next year
  - 3mo spread < -50bp  → market aggressively pricing near-term cuts
  This mirrors the book's observation that 2s rally strongly once the hiking
  cycle is priced out, and that the trough in bond returns (the best exit
  point) coincides with the market starting to price cuts — not with a
  specific FOMC date the trader couldn't have known in advance.

Signal latches: turns on at first threshold crossover, stays on until either
spread drops past its exit threshold. This prevents spurious re-triggers from
intraday spread oscillation around the entry level.

Data sources (all FRED, no subscription required):
  DFF        — daily effective fed funds rate
  DGS1       — 1yr constant-maturity Treasury yield
  DGS3MO     — 3mo constant-maturity Treasury yield
  DFEDTAR    — pre-2008 fed funds target rate
  DFEDTARL   — post-2008 fed funds target lower bound
  DGS2       — 2yr constant-maturity yield → duration-approx daily returns
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# matplotlib.pyplot: main plotting interface; plt is the standard alias
import matplotlib.pyplot as plt
# pandas: labelled time-series and DataFrame operations; pd is the standard alias
import pandas as pd
# numpy: vectorised array math; np is the standard alias
import numpy as np

# sys.path.insert: prepend this file's directory so sibling modules (data, backtest, plot) are importable
sys.path.insert(0, os.path.dirname(__file__))
from data     import fetch_carryless_dgs2_returns, fetch_dgs2_full_pnl
from backtest import annualised_stats, rolling_sharpe, event_time_returns, calc_strat_ret
from plot     import equity_curve, event_time_plot, rolling_sharpe_plot, cycle_breakdown, carry_decomposition_plot, pnl_components_timeseries
from utils.fred_utils import fetch_fred_dataframe
from benchmark import ORACLE_CYCLES


# ── tuneable thresholds ────────────────────────────────────────────────────
HOLD_MONTHS        = 6    # months of no cuts before easing cycle is "done"
THRESHOLD_1YR_BP   = 25   # bp: market must price ≥ 1 hike 1yr out
THRESHOLD_3MO_BP   = 12   # bp: entry — market must price hike within ~3 months
THRESHOLD_1YR_EXIT = -25  # bp: exit threshold for 1yr spread
THRESHOLD_3MO_EXIT = -50  # bp: exit threshold for 3mo spread (wider — noisier series)

# ──────────────────────────────────────────────────────────────────────────

#Extract cleaned versions of daily spreads of DGS1 and DGS3MO vs DFF and daily Fed targets
def _load_signal_data(fred_api_key: str, start: str = "1990-01-01") -> tuple[pd.DataFrame, pd.Series]:
    """Load FRED spread data and stitched fed funds target rate."""
    # fred_utils.fetch_fred_dataframe: downloads DFF, DGS1, DGS3MO from FRED and aligns them into one DataFrame;
    # mappings of df column name: FRED series ID
    # fill_method=None keeps only dates where all three series have real observations (inner join),
    # avoiding stale Treasury yields paired with a live DFF on holidays
    daily_yields = fetch_fred_dataframe(
        fred_api_key,
        {"dff": "DFF", "dgs1": "DGS1", "dgs3mo": "DGS3MO"},
        start,
        fill_method=None,
    )

    '''
    Following block extracts Fed target for each trading day
    '''
    # fred_utils.fetch_fred_dataframe: fetch pre-2008 fed funds target rate (DFEDTAR); exact 25bp FOMC steps
    target_pre  = fetch_fred_dataframe(fred_api_key, {"target": "DFEDTAR"},  start)["target"]
    # fred_utils.fetch_fred_dataframe: fetch post-2008 fed funds target lower bound (DFEDTARL)
    target_post = fetch_fred_dataframe(fred_api_key, {"target": "DFEDTARL"}, start)["target"]
    # pandas.concat: vertically stack the two target Series end-to-end, then sort by date
    fed_target  = pd.concat([target_pre, target_post]).sort_index()
    # pandas Series boolean indexing: drop duplicate index entries at the 2008 ZLB transition, keeping the later one
    fed_target  = fed_target[~fed_target.index.duplicated(keep="last")]
    # pandas Series.reindex: align the stitched target rate to the daily trading index,
    # forward-filling each FOMC decision across subsequent calendar days
    fed_target  = fed_target.reindex(daily_yields.index, method="ffill")

    #Construct spreads
    # pandas Series arithmetic: subtract DFF from DGS1 and multiply by 100 to convert to basis points
    daily_yields["spread_1yr_bp"] = (daily_yields["dgs1"]   - daily_yields["dff"]) * 100
    # pandas Series arithmetic: subtract DFF from DGS3MO and scale to basis points
    daily_yields["spread_3mo_bp"] = (daily_yields["dgs3mo"] - daily_yields["dff"]) * 100

    # pandas Series.diff: day-over-day change; .abs() takes absolute value;
    # < 50 masks implausible single-day jumps from holiday/thin-market FRED prints
    clean_mask = (daily_yields["spread_1yr_bp"].diff().abs() < 50) & \
                 (daily_yields["spread_3mo_bp"].diff().abs() < 50)
    # pandas DataFrame boolean indexing: drop rows where either spread moved more than 50bp in one day
    daily_spreads      = daily_yields[clean_mask]
    # pandas Series.reindex: re-align target rate to the cleaned index after rows are dropped
    fed_target = fed_target.reindex(daily_yields.index, method="ffill")

    return daily_spreads, fed_target

# Returns days with last cuts of a cycle
def _last_cut_dates(fed_target: pd.Series, hold_months: int = HOLD_MONTHS) -> list[pd.Timestamp]:
    """
    Return confirmed 'easing cycle done' dates: last cut before a hold of
    >= hold_months with no intervening hike.
    """
    # pandas Series.diff: day-over-day change in the target rate; negative = cut, positive = hike
    change  = fed_target.diff()
    # pandas Series boolean comparison: True on days where the rate fell by more than 1bp (a genuine cut)
    is_cut  = change < -0.01
    # pandas DatetimeIndex.tolist: convert index to a plain Python list for O(1) integer positional access
    dates   = fed_target.index.tolist()
    confirmed = []
    i = 0

    while i < len(dates):
        # pandas Series.iloc: positional access by integer to check whether day i is a cut
        # if the day is not a cut just move on since it can't be a last cut
        if not is_cut.iloc[i]:
            i += 1
            continue

        # At this stage the date associated with the index i must be a cut
        last_cut_idx = i
        # Consider the next day
        j = i + 1
        # pandas Series.iloc: check whether day j is a hike (change > 0.01bp) to stop the walk
        # Loop until we run out of time or day j is a hike
        while j < len(dates) and not change.iloc[j] > 0.01:
            # pandas Series.iloc: check whether day j itself is a cut to update the last-cut pointer
            if is_cut.iloc[j]:
                last_cut_idx = j
            j += 1

        hold_days_required = hold_months * 21
        hold_count    = 0
        # Working with k handles the case where tiem runs out, and returns last cut date as valid if so
        k = last_cut_idx + 1
        while k < len(dates):
            # Will only run if period of hold isn't long enough, and breaks when hike appears before 6 months
            if change.iloc[k] > 0.01:
                break
            hold_count += 1
            # Tests if hold period is actually >= 6 months. Will add to list if it is
            if hold_count >= hold_days_required:
                confirmed.append(dates[last_cut_idx])
                break
            k += 1

        # At this stage we would've confirmed a last cut date, restart algorithm to find a new one
        i = j

    return confirmed


def detect_signal(fred_api_key: str, start: str = "1993-01-01") -> pd.Series:
    """Returns a daily boolean Series — True when the payer signal is active."""
    #Extract cleaned versions of daily spreads of DGS1 and DGS3MO vs DFF and daily Fed targets. df and Series
    daily_spread, fed_target = _load_signal_data(fred_api_key, start=start)
    # Extract dates with last cuts (list of timestamps)
    last_cuts         = _last_cut_dates(fed_target)

    # in_cycle is True on date D if and only if:
    #   1. there is a confirmed last-cut date L where D >= L + HOLD_MONTHS, AND
    #   2. the Fed has not cut since L (any cut after L closes the window immediately).
    # Using confirmed last-cut dates as anchors prevents mid-pause false opens (e.g. the
    # Apr-Oct 2008 gap triggered a window under the old hold-counter approach because the
    # hold counter elapsed just before the Oct 2008 cut — but Apr 2008 is not a confirmed
    # last-cut, so it never opens a window here).
    target_change = fed_target.diff()
    is_cut        = target_change < -0.01
    in_cycle      = pd.Series(False, index=daily_spread.index)

    for cut_date in last_cuts:
        hold_end = cut_date + pd.DateOffset(months=HOLD_MONTHS)
        # Walk forward from hold_end; stop the moment any cut occurs after cut_date
        after_hold = daily_spread.index[daily_spread.index >= hold_end]
        for date in after_hold:
            if is_cut.get(date, False):
                break
            in_cycle[date] = True

    # pandas Series column access: extract the pre-computed 3-month spread as a Series
    spread_3mo = daily_spread["spread_3mo_bp"]
    # pandas Series column access: extract the pre-computed 1-year spread as a Series
    spread_1yr = daily_spread["spread_1yr_bp"]

    # 3mo exit requires this many consecutive days below threshold to fire,
    # filtering single-day holiday/thin-market spikes in the noisy 3mo series
    EXIT_3MO_CONFIRM_DAYS = 5

    latched = False
    exit_3mo_streak = 0   # consecutive days spread_3mo has been below exit threshold
    signal = pd.Series(False, index=daily_spread.index)
    for date in daily_spread.index:
        if not in_cycle[date]:
            latched = False
            exit_3mo_streak = 0
        elif not latched:
            if spread_3mo[date] > THRESHOLD_3MO_BP and spread_1yr[date] > THRESHOLD_1YR_BP:
                latched = True
                exit_3mo_streak = 0
        else:
            # 1yr exit fires immediately — cleaner series, genuine signal
            if spread_1yr[date] < THRESHOLD_1YR_EXIT:
                latched = False
                exit_3mo_streak = 0
            else:
                # 3mo exit requires EXIT_3MO_CONFIRM_DAYS consecutive days below threshold
                if spread_3mo[date] < THRESHOLD_3MO_EXIT:
                    exit_3mo_streak += 1
                    if exit_3mo_streak >= EXIT_3MO_CONFIRM_DAYS:
                        latched = False
                        exit_3mo_streak = 0
                else:
                    exit_3mo_streak = 0
        signal[date] = latched

    # pandas Series.name: assign a name so callers can identify the column in downstream DataFrames
    signal.name         = "first_hike_signal"
    return signal

# Dictionary of estimated hiking episodes and their dates
def signal_to_cycles(signal: pd.Series) -> list[dict]:
    """
    Convert a boolean signal series into cycle dicts for backtest.calc_strat_ret.

    Each contiguous True episode becomes one cycle:
      first_hike = first True day  (signal-driven entry)
      last_hike  = first False day after the episode (signal-driven exit)

    No oracle FOMC dates. Entry and exit are purely what the market spreads say.
    Episodes still active at end of data are included with last_hike = last date.
    """
    cycles     = []
    in_episode = False
    # pandas Timestamp type annotation: episode_start will hold a Timestamp or None before the first episode
    episode_start: pd.Timestamp | None = None
    episode_num = 0

    # pandas Series.items: iterate over (DatetimeIndex label, boolean value) pairs, which .items() breaks down into
    # in date order
    for date, val in signal.items():
        if val and not in_episode:
            episode_start = date
            in_episode    = True
        elif not val and in_episode:
            episode_num += 1
            cycles.append({
                "label":      f"episode_{episode_num}  {episode_start.year}–{date.year}",
                # pandas Timestamp: store the episode start/end as Timestamps for downstream date arithmetic
                "first_hike": episode_start,
                "last_hike":  date,
            })
            in_episode = False

    if in_episode and episode_start is not None:
        episode_num += 1
        cycles.append({
            "label":      f"episode_{episode_num}  {episode_start.year}–ongoing",
            "first_hike": episode_start,
            # pandas DatetimeIndex integer indexing: use the last date in the signal as the open episode's exit
            "last_hike":  signal.index[-1],
        })

    return cycles

# Takes arguments df from calc_strat_ret, daily return series, and cycles list from earlier
# Computes dataframe of stats per cycle
'''
# df: DataFrame returned by calc_strat_ret / _run. Has columns:
#       "signal"     — int position per day: -1 (short/payer), 0 (flat), +1 (long)
#       "bond_ret"   — raw daily bond return
#       "strat_ret"  — signal * bond_ret (negative of bond_ret when short)
#       "cum_equity" — running product of (1 + strat_ret), starting at 1.0
#     Index: DatetimeIndex of all trading days in the backtest.
#
# ret: daily returns Series output by fetch_carryless_dgs2_returns or compute_returns.
#      Named "ret", indexed by DatetimeIndex. NOTE: not actually used inside this
#      function — strat_ret is pulled from df instead. Kept for API symmetry with _run.
#
# cycles: list of dicts produced by signal_to_cycles. Each dict has the shape:
#       {
#           "label":      str          e.g. "episode_1  1994–1995"
#           "first_hike": pd.Timestamp — signal-on date (episode entry)
#           "last_hike":  pd.Timestamp — signal-off date (episode exit)
#       }
'''
def stats_per_cycle(df: pd.DataFrame, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Per-cycle breakdown: entry/exit dates, days held, total compounded return."""
    rows = []
    # For each day in episode
    for c in cycles:
        # pandas boolean indexing: filter to rows inside this cycle's window where position is short (-1)
        mask = (df.index >= c["first_hike"]) & (df.index <= c["last_hike"]) & (df["signal"] == -1)
        # pandas DataFrame.loc: select strat_ret for matching rows, drop any NaN values
        r    = df.loc[mask, "strat_ret"].dropna()
        # pandas Series.prod: computes (1+r1)*(1+r2)*...*(1+rN) then subtracts 1 for total compounded return
        cum  = (1 + r).prod() - 1
        rows.append({
            "cycle":       c["label"],
            # pandas Timestamp.date: strip time component for clean string display
            "entry":       str(c["first_hike"].date()),
            "exit":        str(c["last_hike"].date()),
            "days_held":   len(r),
            "total_ret_%": round(cum * 100, 2),
        })
    # pandas DataFrame: construct from list of dicts, then set "cycle" as the row index
    return pd.DataFrame(rows).set_index("cycle")

# runs backtest for strat
def _run(name: str, ret: pd.Series, cycles: list[dict]) -> pd.DataFrame:
    """Run signal-driven backtest for one instrument, print summary, return df."""
    df = calc_strat_ret(
        ret, cycles,
        entry_days_before_first=0,
        exit_days_before_last=0,
    )
    # backtest.annualised_stats: returns a pandas DataFrame of ann. return, vol, Sharpe, max drawdown
    stats = annualised_stats(df)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    # pandas DataFrame.to_string: renders the full stats table without truncation
    print(stats.to_string())
    print()
    # stats_per_cycle: returns a pandas DataFrame; .to_string() renders it without truncation
    print(stats_per_cycle(df, ret, cycles).to_string())
    return df


def main():
    from dotenv import load_dotenv
    load_dotenv()
    # os.environ.get: read FRED_API_KEY from the environment, defaulting to empty string if absent
    api_key = os.environ.get("FRED_API_KEY", "873c5477f9604271a243f7284fe594c0")
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

    print("\nFetching DGS2 PnL decomposition (FRED, back to 1976)...")
    pnl_full = fetch_dgs2_full_pnl(api_key, start="1976-01-01")
    print(f"  DGS2: {pnl_full.index[0].date()} to {pnl_full.index[-1].date()}")

    # price-only series: pure yield-move signal, no carry drag
    ret_price = pnl_full["price_ret"].rename("ret")
    # full series: price move + funding carry + roll-down carry
    ret_total = pnl_full["total_ret"].rename("ret")

    df_price = _run("DGS2 price-only (no carry)", ret_price, cycles_all)
    df_total = _run("DGS2 total (price + carry)", ret_total, cycles_all)

    print("\nGenerating plots...")
    ev_first = event_time_returns(ret_price, cycles_all, anchor="first_hike", window=120)
    ev_last  = event_time_returns(ret_price, cycles_all, anchor="last_hike",  window=120)
    equity_curve(df_price, cycles_all, instrument="DGS2 Price-Only", oracle_cycles=ORACLE_CYCLES)
    equity_curve(df_total, cycles_all, instrument="DGS2 Total (with carry)", oracle_cycles=ORACLE_CYCLES)
    event_time_plot(ev_first, anchor_label="signal on — DGS2")
    event_time_plot(ev_last,  anchor_label="signal off — DGS2")
    cycle_breakdown(df_price, cycles_all, instrument="DGS2 Price-Only")
    cycle_breakdown(df_total, cycles_all, instrument="DGS2 Total (with carry)")
    rolling_sharpe_plot(rolling_sharpe(df_price), cycles=cycles_all, df=df_price, instrument="DGS2 Price-Only")
    rolling_sharpe_plot(rolling_sharpe(df_total), cycles=cycles_all, df=df_total, instrument="DGS2 Total (with carry)")
    carry_decomposition_plot(pnl_full, cycles_all)
    pnl_components_timeseries(pnl_full, signal)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

# Start calling main if the file is run directly
if __name__ == "__main__":
    main()
