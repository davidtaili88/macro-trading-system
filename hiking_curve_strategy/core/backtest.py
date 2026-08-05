"""
Backtest engine for the hiking cycle 2-year payer strategy.

Framework (from the book):
  - SHORT 2-year bonds (payer position) = negative of the bond return
  - Enter ~55-75 trading days before first hike (entry_days_before_first)
  - Hold continuously until ~30 trading days before last hike (exit_days_before_last)
  - After last hike: go long (receiver) or flat

The payer P&L on day t = -1 * bond_return[t]  (short = negate price return)

We expose entry/exit offsets as parameters so they can be calibrated from data.
"""

# numpy: vectorised array math; np is the standard alias
import numpy as np
# pandas: labelled time-series and DataFrame operations; pd is the standard alias
import pandas as pd

from utils.backtest_utils import annualised_stats, rolling_sharpe, event_time_returns  # noqa: F401 — re-exported for callers


TRADING_DAYS = 252

'''
# returns: daily bond return Series output by fetch_dgs2_returns or fetch_carryless_dgs2_returns.
#   Named "ret", indexed by DatetimeIndex of trading days.
#   Each value is a decimal daily return, e.g. 0.0003 = +0.03% on that day.
#
# cycles: list of dicts produced by signal_to_cycles (or ORACLE_CYCLES from benchmark.py).
#   Each dict has the shape:
#     { "label": str,  e.g. "episode_1  1994–1995"
#       "first_hike": pd.Timestamp,  — signal-on / first hike date (episode entry anchor)
#       "last_hike":  pd.Timestamp   — signal-off / last hike date (episode exit anchor) }
#
# entry_days_before_first: oracle mode only — trading days before first_hike to enter.
#   Pass None (default) for signal-driven mode: first_hike IS the entry date.
#   Pass an int (e.g. 65) for oracle/theoretical mode: shift entry that many days earlier.
#
# exit_days_before_last: oracle mode only — trading days before last_hike to exit.
#   Pass None (default) for signal-driven mode: last_hike IS the exit date.
#   Pass an int (e.g. 30) for oracle/theoretical mode: shift exit that many days earlier.
'''
def build_signal_hiking_strat(
    instrument_daily_ret: pd.Series,
    cycles: list[dict],
    entry_days_before_first: int | None = None,  # None = signal-driven (use date as-is); int = oracle offset
    exit_days_before_last: int | None = None,    # None = signal-driven (use date as-is); int = oracle offset
) -> pd.Series:
    """
    Construct a daily -1/0 position Series from a list of cycle windows.

    Signal-driven mode (entry/exit_days = None):
      first_hike and last_hike are the signal-on/off dates from signal_to_cycles —
      use them exactly. No calendar knowledge required.

    Oracle mode (entry/exit_days are ints):
      first_hike and last_hike are real FOMC dates from ORACLE_CYCLES — shift entry
      and exit by the given number of trading days to replicate the book's strategy.
    """
    # pandas Series: create a zero-filled Series with the same DatetimeIndex as instrument_daily_ret
    signal = pd.Series(0, index=instrument_daily_ret.index, dtype=float)
    # pandas DatetimeIndex: the sorted index of all trading dates, used for positional lookups
    trading_days = instrument_daily_ret.index

    for c in cycles:
        first_hike = c["first_hike"]
        last_hike  = c["last_hike"]

        if entry_days_before_first is None:
            # signal-driven: dates already encode entry/exit, use directly
            signal.loc[first_hike:last_hike] = -1
        else:
            # oracle: searchsorted to find positions, then shift by the offsets
            # pandas DatetimeIndex.searchsorted: binary search returning integer position of the anchor date
            fh_pos = trading_days.searchsorted(first_hike)
            # pandas DatetimeIndex.searchsorted: same for last hike
            lh_pos = trading_days.searchsorted(last_hike)

            exit_offset = exit_days_before_last if exit_days_before_last is not None else 0
            # clamp to 0 so we never go before the start of the return series
            entry_pos = max(fh_pos - entry_days_before_first, 0)
            exit_pos  = max(lh_pos - exit_offset, 0)

            if entry_pos >= exit_pos:
                continue

            # pandas DatetimeIndex integer indexing: convert integer positions back to actual dates
            entry_date = trading_days[entry_pos]
            exit_date  = trading_days[exit_pos]

            # pandas Series.loc: label-based slice assignment, sets all rows in the window to -1
            signal.loc[entry_date:exit_date] = -1

    return signal

'''
# bond_returns: same daily return Series as above — passed straight through to build_signal_hiking_strat
#   and also stored as the "bond_ret" column in the output DataFrame.
#
# cycles: same list of dicts as above — passed straight through to build_signal_hiking_strat.
#
# entry_days_before_first / exit_days_before_last: forwarded to build_signal_hiking_strat.
#   None (default) = signal-driven mode: use cycle dates exactly as given.
#   int            = oracle mode: shift entry/exit by that many trading days.
#
# roll_dates: DatetimeIndex of quarterly CME futures roll dates, output of get_zt_roll_dates.
#   e.g. DatetimeIndex(['2002-03-08', '2002-06-10', ...])
#   None for DGS2 (cash instruments have no futures roll).
#
# roll_cost: flat decimal return deducted on each roll date when short, e.g. 0.000080 = 0.8bp.
#   Sourced from ZT_ROLL_COST constant in signal_market.py.
#   0.0 when roll_dates is None (no roll cost for cash instruments).
'''
def calc_strat_ret(
    bond_returns: pd.Series,
    cycles: list[dict],
    entry_days_before_first: int | None = None,
    exit_days_before_last: int | None = None,
    roll_dates: pd.DatetimeIndex | None = None,
    roll_cost: float = 0.0,
) -> pd.DataFrame:
    """
    Apply the payer signal (series describing days we're short/long/neutral) to bond returns.
    Strategy return = signal * bond_return
      (signal=-1 means short bonds, so strat_ret = -bond_ret on those days)

    roll_dates: dates on which a futures roll occurs (ZT only).
    roll_cost:  flat cost deducted as a decimal return on each roll date when short
                (e.g. 0.00008 ≈ 0.8bp per roll).  Ignored when roll_dates is None.
    """
    signal = build_signal_hiking_strat(bond_returns, cycles, entry_days_before_first, exit_days_before_last)
    # pandas Series arithmetic: element-wise multiplication of signal (+1/0/-1) by daily bond returns
    strat_ret = signal * bond_returns

    if roll_dates is not None and roll_cost > 0.0:
        # deduct roll cost on roll dates where position is short (-1)
        active_rolls = roll_dates[roll_dates.isin(signal.index)]
        short_on_roll = signal.reindex(active_rolls) == -1
        roll_hit_dates = active_rolls[short_on_roll]
        strat_ret.loc[roll_hit_dates] -= roll_cost

    # cumprod for the equity curve treats a NaN daily return as a flat (0) day —
    # otherwise a single leading NaN (e.g. the first row of a pct_change() return
    # series, as with ZT futures) makes cumprod NaN from the start, which then wipes
    # out the whole plotted equity line when it is rebased by its first value.
    cum_equity = (1 + strat_ret.fillna(0)).cumprod()

    return pd.DataFrame({
        "signal":     signal,
        "bond_ret":   bond_returns,
        "strat_ret":  strat_ret,
        "cum_equity": cum_equity,
    })


# bond_returns: daily return Series (same as above)
# cycles:       list of cycle dicts (same as above)
def cycle_pnl(bond_returns: pd.Series, cycles: list[dict]) -> tuple[dict, float]:
    """
    Per-cycle and pooled compounded payer P&L on the given return series.

    Returns:
      per_cycle: dict mapping cycle label -> compounded decimal return for that episode
      pooled:    compounded decimal return across ALL payer-active days stitched together

    Uses entry/exit offsets of 0 so the signal windows are applied exactly as supplied —
    the cycle dicts already encode the precise entry/exit dates.
    """
    strat_ret_df = calc_strat_ret(bond_returns, cycles)
    per_cycle_pnl = {}
    for c in cycles:
        # pandas boolean indexing: rows inside this cycle's window where the position is short
        mask = (strat_ret_df.index >= c["first_hike"]) & (strat_ret_df.index <= c["last_hike"]) & (strat_ret_df["signal"] == -1)
        # pandas Series.prod: (1+r1)*(1+r2)*...-1 gives the compounded return for the episode
        r = strat_ret_df.loc[mask, "strat_ret"].dropna()
        per_cycle_pnl[c["label"]] = (1 + r).prod() - 1
    # pandas boolean indexing: all short days across every cycle, stitched together
    all_mask = strat_ret_df["signal"] == -1
    pooled_pnl = (1 + strat_ret_df.loc[all_mask, "strat_ret"].dropna()).prod() - 1
    return per_cycle_pnl, pooled_pnl

