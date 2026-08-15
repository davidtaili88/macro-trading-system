"""
Backtest engine for the hiking cycle 2-year payer strategy.

Framework (from the book):
  - SHORT 2-year bonds (payer position) = negative of the bond return
  - Enter 60 trading days before first hike (entry_days_before_first)
  - Hold continuously until 30 trading days before last hike (exit_days_before_last)
  - After last hike: go long (receiver) or flat

The payer P&L on day t = -1 * bond_return[t]  (short = negate price return)

We expose entry/exit offsets as parameters so they can be calibrated from data.
"""

import numpy as np
import pandas as pd

from utils.performance_evaluation import annualised_stats, rolling_sharpe, event_time_returns  # noqa: F401 — re-exported for callers


TRADING_DAYS = 252

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
    signal = pd.Series(0, index=instrument_daily_ret.index, dtype=float)
    trading_days = instrument_daily_ret.index

    for c in cycles:
        first_hike = c["first_hike"]
        last_hike  = c["last_hike"]

        if entry_days_before_first is None:
            # signal-driven: dates already encode entry/exit, use directly
            signal.loc[first_hike:last_hike] = -1
        else:
            # oracle: locate the hike positions, then shift entry/exit by the offsets
            fh_pos = trading_days.searchsorted(first_hike)
            lh_pos = trading_days.searchsorted(last_hike)

            exit_offset = exit_days_before_last if exit_days_before_last is not None else 0
            # clamp to 0 so we never go before the start of the return series
            entry_pos = max(fh_pos - entry_days_before_first, 0)
            exit_pos  = max(lh_pos - exit_offset, 0)

            if entry_pos >= exit_pos:
                continue

            entry_date = trading_days[entry_pos]
            exit_date  = trading_days[exit_pos]

            signal.loc[entry_date:exit_date] = -1

    return signal

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

    roll_dates: dates on which a futures roll occurs (used only by the side
                tradability experiment; None for the DGS2 constant-maturity path,
                which has no roll).
    roll_cost:  flat cost deducted as a decimal return on each roll date when short
                (e.g. 0.00008 ≈ 0.8bp per roll).  Ignored when roll_dates is None.
    """
    signal = build_signal_hiking_strat(bond_returns, cycles, entry_days_before_first, exit_days_before_last)
    strat_ret = signal * bond_returns

    if roll_dates is not None and roll_cost > 0.0:
        # deduct roll cost on roll dates where position is short (-1)
        active_rolls = roll_dates[roll_dates.isin(signal.index)]
        short_on_roll = signal.reindex(active_rolls) == -1
        roll_hit_dates = active_rolls[short_on_roll]
        strat_ret.loc[roll_hit_dates] -= roll_cost

    # cumprod for the equity curve treats a NaN daily return as a flat (0) day —
    # otherwise a single leading NaN (e.g. the first row of a pct_change() return
    # series) makes cumprod NaN from the start, which then wipes out the whole
    # plotted equity line when it is rebased by its first value.
    cum_equity = (1 + strat_ret.fillna(0)).cumprod()

    return pd.DataFrame({
        "signal":     signal,
        "bond_ret":   bond_returns,
        "strat_ret":  strat_ret,
        "cum_equity": cum_equity,
    })


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
        mask = (strat_ret_df.index >= c["first_hike"]) & (strat_ret_df.index <= c["last_hike"]) & (strat_ret_df["signal"] == -1)
        r = strat_ret_df.loc[mask, "strat_ret"].dropna()
        per_cycle_pnl[c["label"]] = (1 + r).prod() - 1
    # pool all short days across every cycle, stitched together
    all_mask = strat_ret_df["signal"] == -1
    pooled_pnl = (1 + strat_ret_df.loc[all_mask, "strat_ret"].dropna()).prod() - 1
    return per_cycle_pnl, pooled_pnl

