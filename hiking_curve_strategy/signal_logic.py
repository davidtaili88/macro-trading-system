"""
Market-driven signal logic for the hiking cycle 2-year payer strategy.

Both entry AND exit are fully market-driven — no oracle FOMC dates used.
Instrument-agnostic: produces cycle windows (entry/exit dates) that any
run script (signal_trade_dgs.py, signal_trade_zt.py, ...) applies to its
own return series via backtest.calc_strat_ret.

Entry: first day BOTH spread conditions hold inside a confirmed post-easing hold:
  - 3mo spread (DGS3MO - DFF) > 12bp  → hike priced within ~3 months
  - 1yr spread (DGS1 - DFF)   > 50bp  → market broadly pricing a hiking cycle
  Both spreads are smoothed with a short rolling mean (ENTRY_SMOOTH_WINDOW_DAYS)
  before comparison — a single noisy dff print (thin holiday trading) can
  otherwise flip the entry latch on for weeks.

Exit: first day ANY of the following holds:
  - false-promise exit: no hike has occurred yet since entry, AND smoothed
    1yr spread has fallen back to <= FALSE_PROMISE_THRESHOLD_1YR_BP (well
    below the entry bar). Catches cycles where the market briefly priced a
    hike that never came (e.g. 1996: spread spiked to ~80bp on entry, no
    hike ever followed, spread collapsed within weeks) without misfiring on
    normal chop around the entry threshold mid-cycle — the re-arm bar is
    set well below THRESHOLD_1YR_BP specifically so ordinary oscillation in
    an intact cycle (which routinely dips below the entry level without the
    cycle being over) doesn't trip it. Disarmed permanently once a real hike
    lands, since at that point the cycle is confirmed and the crude level
    exit below takes over.
  - 1yr spread < -25bp  → market pricing net cuts over the next year
  - 5-day rolling avg of 3mo spread (capped at ±100bp) < -50bp  → market
    aggressively pricing near-term cuts. Rolling average smooths out single-day
    holiday/thin-market spikes; the ±100bp cap prevents extreme outlier prints
    from dominating the average. Responds to sustained drift below threshold
    rather than requiring 5 unbroken consecutive days.
  This mirrors the book's observation that 2s rally strongly once the hiking
  cycle is priced out, and that the trough in bond returns (the best exit
  point) coincides with the market starting to price cuts — not with a
  specific FOMC date the trader couldn't have known in advance.

  NOTE: this fires well after the true last hike (empirically 44-111 trading
  days late vs. ORACLE_CYCLES — see check_exit.py). A hike-indexed exit
  (level/deceleration/drawdown-from-peak of spread_1yr, sampled at each FOMC
  hike) was tried and rejected: no single threshold works across cycles,
  since e.g. 2015-2018 sat at 47-63bp (a fresh cycle high two hikes prior)
  right through its true last hike, while 2004-2006 was already 80%+ off its
  peak by its true last hike. Revisit with a second signal (cumulative bp
  hiked, cycle-length prior, SEP/dot-plot data) rather than another
  single-spread-series threshold.

Signal latches: turns on at first threshold crossover, stays on until either
spread drops past its exit threshold. This prevents spurious re-triggers from
intraday spread oscillation around the entry level.

Data sources (all FRED, no subscription required):
  DFF        — daily effective fed funds rate
  DGS1       — 1yr constant-maturity Treasury yield
  DGS3MO     — 3mo constant-maturity Treasury yield
  DFEDTAR    — pre-2008 fed funds target rate
  DFEDTARL   — post-2008 fed funds target lower bound
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.fred_utils import fetch_fred_dataframe


# ── tuneable thresholds ────────────────────────────────────────────────────
HOLD_MONTHS        = 6    # months of no cuts before easing cycle is "done"
THRESHOLD_1YR_BP   = 50   # bp: market must price ≥ 1 hike 1yr out
THRESHOLD_3MO_BP   = 12   # bp: entry — market must price hike within ~3 months

# entry spreads are smoothed with a short rolling mean before being compared to
# threshold: raw dff is the *effective* (traded) fed funds rate, not the target,
# and wobbles several bp day-to-day (worse around holidays/quarter-end thin
# trading) — a single noisy print can otherwise flip the entry latch on for
# weeks (e.g. 1996-08-13, 1997-12-24 in the DGS backtest were pure dff blips,
# not real hike pricing).
ENTRY_SMOOTH_WINDOW_DAYS = 5  # trading days: rolling mean window on entry spreads

THRESHOLD_1YR_EXIT = -25  # bp: exit threshold for 1yr spread
THRESHOLD_3MO_EXIT = -50  # bp: exit threshold for 3mo spread (wider — noisier series)

# false-promise exit: if no hike has landed yet since entry and smoothed spread_1yr
# falls back to this level, treat the entry as a failed conviction and exit early.
# Set well below THRESHOLD_1YR_BP (not equal to it) so normal chop around the entry
# level mid-cycle doesn't trip it — only a near-full reversal does. Uses spread_1yr
# only (the cleaner series); spread_3mo is not part of this check.
FALSE_PROMISE_THRESHOLD_1YR_BP = 25  # bp

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
    # entry spreads smoothed over a short window so a single noisy dff print (holiday/thin
    # trading) can't flip the latch on by itself — see ENTRY_SMOOTH_WINDOW_DAYS comment above
    spread_3mo_entry = spread_3mo.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()
    spread_1yr_entry = spread_1yr.rolling(ENTRY_SMOOTH_WINDOW_DAYS, min_periods=1).mean()

    # every individual hike date — used to disarm the false-promise exit once a real
    # hike confirms the cycle
    hike_dates = set(target_change.index[target_change > 0.01].tolist())

    # 3mo exit: 5-day rolling average of the capped spread must fall below THRESHOLD_3MO_EXIT.
    # Cap clips extreme single-day prints (holiday/thin-market) before they enter the average;
    # rolling mean responds to sustained drift rather than requiring 5 unbroken consecutive days.
    EXIT_3MO_ROLL_WINDOW = 5
    EXIT_3MO_CAP_BP      = 100
    spread_3mo_roll = spread_3mo.clip(lower=-EXIT_3MO_CAP_BP, upper=EXIT_3MO_CAP_BP) \
                                 .rolling(EXIT_3MO_ROLL_WINDOW, min_periods=1).mean()

    latched = False
    hiked_since_entry = False  # whether a real hike has landed since this entry
    signal = pd.Series(False, index=daily_spread.index)
    for date in daily_spread.index:
        if not in_cycle[date]:
            latched = False
            hiked_since_entry = False
        elif not latched:
            if spread_3mo_entry[date] > THRESHOLD_3MO_BP and spread_1yr_entry[date] > THRESHOLD_1YR_BP:
                latched = True
                hiked_since_entry = False
        else:
            if date in hike_dates:
                hiked_since_entry = True

            # false-promise exit: no hike delivered yet, and conviction has collapsed
            if not hiked_since_entry and spread_1yr_entry[date] <= FALSE_PROMISE_THRESHOLD_1YR_BP:
                latched = False
            # 1yr exit fires immediately — cleaner series, genuine signal
            elif spread_1yr[date] < THRESHOLD_1YR_EXIT:
                latched = False
            # 3mo exit fires when 5-day rolling avg (capped) crosses below threshold
            elif spread_3mo_roll[date] < THRESHOLD_3MO_EXIT:
                latched = False
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
