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
# returns: daily bond return Series output by compute_returns or fetch_carryless_dgs2_returns.
#   Named "ret", indexed by DatetimeIndex of trading days.
#   Each value is a decimal daily return, e.g. 0.0003 = +0.03% on that day.
#
# cycles: list of dicts produced by signal_to_cycles (or ORACLE_CYCLES from benchmark.py).
#   Each dict has the shape:
#     { "label": str,  e.g. "episode_1  1994–1995"
#       "first_hike": pd.Timestamp,  — signal-on / first hike date (episode entry anchor)
#       "last_hike":  pd.Timestamp   — signal-off / last hike date (episode exit anchor) }
#
# entry_days_before_first: how many trading days before first_hike to enter the payer position.
#   Default 65 = midpoint of the 55–75 day range cited in the book.
#   When called from signal_market._run, this is 0 (enter exactly on signal date).
#
# exit_days_before_last: how many trading days before last_hike to exit.
#   Default 30 = book's Fig 5.6 trough estimate.
#   When called from signal_market._run, this is 0 (exit exactly on signal-off date).
'''
def build_signal_hiking_strat(
    returns: pd.Series,
    cycles: list[dict],
    entry_days_before_first: int = 65,   # midpoint of 55-75 range from book
    exit_days_before_last: int = 30,     # 10th-pct trough per book Fig 5.6
) -> pd.Series:
    """
    Construct a daily position series:
      +1 = long bonds (receiver)
       0 = flat
      -1 = short bonds (payer)

    For each cycle the payer window is:
      [first_hike - entry_days_before_first,  last_hike - exit_days_before_last)

    Outside any cycle window the position is flat (0).
    We use trading-day offsets via the actual price index.
    """
    # pandas Series: create a zero-filled Series with the same DatetimeIndex as returns
    signal = pd.Series(0, index=returns.index, dtype=float)
    # pandas DatetimeIndex: the sorted index of all trading dates, used for positional lookups
    trading_days = returns.index

    for c in cycles:
        first_hike = c["first_hike"]
        last_hike  = c["last_hike"]

        # find nearest index position for each anchor date
        # pandas DatetimeIndex.searchsorted: binary search returning the integer position where the date would be inserted
        fh_pos = trading_days.searchsorted(first_hike)
        # pandas DatetimeIndex.searchsorted: same binary search for the last hike date
        lh_pos = trading_days.searchsorted(last_hike)

        # shift positions back by the entry/exit day offsets; clamp to 0 so we don't go negative
        entry_pos = max(fh_pos - entry_days_before_first, 0)
        exit_pos  = max(lh_pos - exit_days_before_last, 0)

        if entry_pos >= exit_pos:
            continue

        # pandas DatetimeIndex integer indexing: convert integer positions back to actual dates
        entry_date = trading_days[entry_pos]
        exit_date  = trading_days[exit_pos]

        # pandas Series.loc: label-based slice assignment, sets all rows between the two dates to -1
        signal.loc[entry_date:exit_date] = -1

    return signal

'''
# returns: same daily return Series as above — passed straight through to build_signal_hiking_strat
#   and also stored as the "bond_ret" column in the output DataFrame.
#
# cycles: same list of dicts as above — passed straight through to build_signal_hiking_strat.
#
# entry_days_before_first / exit_days_before_last: same as above, forwarded to build_signal_hiking_strat.
#
# roll_dates: DatetimeIndex of quarterly CME futures roll dates, output of get_zt_roll_dates.
#   e.g. DatetimeIndex(['2002-03-08', '2002-06-10', ...])
#   None for DGS2/SHY (cash instruments have no futures roll).
#
# roll_cost: flat decimal return deducted on each roll date when short, e.g. 0.000080 = 0.8bp.
#   Sourced from ZT_ROLL_COST constant in signal_market.py.
#   0.0 when roll_dates is None (no roll cost for cash instruments).
'''
def calc_strat_ret(
    returns: pd.Series,
    cycles: list[dict],
    entry_days_before_first: int = 65,
    exit_days_before_last: int = 30,
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
    signal = build_signal_hiking_strat(returns, cycles, entry_days_before_first, exit_days_before_last)
    # pandas Series arithmetic: element-wise multiplication of signal (+1/0/-1) by daily bond returns
    strat_ret = signal * returns

    if roll_dates is not None and roll_cost > 0.0:
        # deduct roll cost on roll dates where position is short (-1)
        active_rolls = roll_dates[roll_dates.isin(signal.index)]
        short_on_roll = signal.reindex(active_rolls) == -1
        roll_hit_dates = active_rolls[short_on_roll]
        strat_ret.loc[roll_hit_dates] -= roll_cost

    cum_equity = (1 + strat_ret).cumprod()

    return pd.DataFrame({
        "signal":     signal,
        "bond_ret":   returns,
        "strat_ret":  strat_ret,
        "cum_equity": cum_equity,
    })
