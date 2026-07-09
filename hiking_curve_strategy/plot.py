"""
Visualisation for the hiking cycle 2-year payer strategy.

Four plots:
  1. equity_curve      — cumulative strategy vs buy-hold bonds, with cycle shading
  2. event_time_plot   — normalised bond returns around first/last hike (replicates
                         book Figs 5.5 / 5.6): median + percentile band per cycle
  3. rolling_sharpe    — rolling 252-day Sharpe for the payer strategy
  4. cycle_breakdown   — per-cycle bar chart of annualised return and Sharpe
"""

# matplotlib.pyplot: main plotting interface; plt is the standard alias
import matplotlib.pyplot as plt
# matplotlib.dates: formatters/locators for converting datetime64 values to readable axis tick labels
import matplotlib.dates as mdates
# matplotlib.patches: geometric shapes (Rectangle, Patch, etc.) used for custom legend entries or overlays
import matplotlib.patches as mpatches
# numpy: vectorised array math; np is the standard alias
import numpy as np
# pandas: labelled time-series and DataFrame operations; pd is the standard alias
import pandas as pd

TRADING_DAYS = 252
_CYCLE_COLORS = [
    "#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"
]


def equity_curve(
    df: pd.DataFrame,
    cycles: list[dict],
    instrument: str = "2yr Payer",
    oracle_cycles: list[dict] | None = None,
    latched_series: pd.Series | None = None,
    save_path: str | None = None,
):
    """
    Cumulative equity of payer strategy vs. long bonds, with:
      - shading for active payer windows (signal-driven)
      - vertical lines for oracle first-hike dates (if oracle_cycles supplied)
      - a binary latched indicator on a twin axis (if latched_series supplied)
    """
    # two rows: top = equity curves + latched, bottom = latched indicator
    if latched_series is not None:
        fig, (ax, ax_latch) = plt.subplots(
            2, 1, figsize=(13, 6),
            gridspec_kw={"height_ratios": [4, 1]},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax_latch = None

    # buy-hold bond cumulative
    bh = (1 + df["bond_ret"]).cumprod()
    ax.plot(bh.index, bh, color="steelblue", lw=1.2, label="Buy-hold bonds (long)")

    # strategy cumulative
    ax.plot(
        df["cum_equity"].index, df["cum_equity"],
        color="firebrick", lw=1.5, label="Payer strategy"
    )

    # shade active payer windows (signal-driven)
    for i, c in enumerate(cycles):
        mask = (df.index >= c["first_hike"]) & (df["signal"] == -1)
        block = df[mask]
        if block.empty:
            continue
        ax.axvspan(
            block.index[0], block.index[-1],
            alpha=0.12, color=_CYCLE_COLORS[i % len(_CYCLE_COLORS)],
            label=f"Signal active: {c['label']}"
        )

    # oracle first-hike vertical lines — show timing gap vs signal entry
    if oracle_cycles is not None:
        for j, oc in enumerate(oracle_cycles):
            fh = oc["first_hike"]
            lh = oc["last_hike"]
            # only draw if oracle date falls within plotted range
            if fh >= df.index[0] and fh <= df.index[-1]:
                ax.axvline(
                    fh, color=_CYCLE_COLORS[j % len(_CYCLE_COLORS)],
                    lw=1.4, ls="--", alpha=0.85,
                    label=f"Oracle 1st hike: {oc['label']}",
                )
            if lh >= df.index[0] and lh <= df.index[-1]:
                ax.axvline(
                    lh, color=_CYCLE_COLORS[j % len(_CYCLE_COLORS)],
                    lw=1.0, ls=":", alpha=0.70,
                    label=f"Oracle last hike: {oc['label']}",
                )

    ax.set_title(f"{instrument} — Cumulative Equity vs. Buy-Hold", fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # latched indicator panel
    if latched_series is not None and ax_latch is not None:
        # align latched to df index in case date ranges differ
        latched_aligned = latched_series.reindex(df.index).fillna(False).astype(float)
        ax_latch.fill_between(
            latched_aligned.index, latched_aligned, 0,
            step="post", color="darkorange", alpha=0.55, label="Latched"
        )
        ax_latch.set_ylim(-0.05, 1.3)
        ax_latch.set_yticks([0, 1])
        ax_latch.set_yticklabels(["off", "on"], fontsize=8)
        ax_latch.set_ylabel("Latched", fontsize=8)
        ax_latch.legend(fontsize=7, loc="upper left")
        ax_latch.grid(alpha=0.2)
        ax_latch.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)


def event_time_plot(
    event_df: pd.DataFrame,
    anchor_label: str = "first hike",
    save_path: str | None = None,
):
    """
    Plot normalised cumulative bond returns in event time.
    event_df: output of backtest.event_time_returns()
    Each column = one cycle. Plots median + 10th/90th pct band.
    """
    # pandas DataFrame attribute: True if the DataFrame has no rows
    if event_df.empty:
        print("No event-time data to plot.")
        return

    # matplotlib.pyplot: create Figure and Axes with given dimensions
    fig, ax = plt.subplots(figsize=(12, 5))

    # plot individual cycles lightly
    for col in event_df.columns:
        # matplotlib Axes: plot each cycle column as a faint line; alpha=0.4 makes it semi-transparent
        ax.plot(event_df.index, event_df[col], lw=0.8, alpha=0.4)

    # median and percentile band
    # pandas DataFrame.median: compute row-wise median across all cycle columns (axis=1 = across columns)
    median = event_df.median(axis=1)
    # pandas DataFrame.quantile: compute the 10th percentile across columns for each row
    p10    = event_df.quantile(0.10, axis=1)
    # pandas DataFrame.quantile: compute the 90th percentile across columns for each row
    p90    = event_df.quantile(0.90, axis=1)

    # matplotlib Axes: draw the median line in bold black
    ax.plot(event_df.index, median, color="black", lw=2.0, label="Median")
    # matplotlib Axes: shade the region between p10 and p90 for the uncertainty band
    ax.fill_between(event_df.index, p10, p90, alpha=0.15, color="black", label="10th-90th pct")

    # matplotlib Axes: draw a vertical dashed red line at x=0 marking the event anchor date
    ax.axvline(0, color="red", lw=1.2, ls="--", label=f"t=0: {anchor_label}")
    # matplotlib Axes: draw a horizontal dotted grey line at y=1.0 (the normalised baseline)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")

    # matplotlib Axes: set title using an f-string built from the anchor_label argument
    ax.set_title(f"2yr Bond: Normalised Cumulative Return Around {anchor_label.title()}", fontsize=13)
    # matplotlib Axes: label the x-axis
    ax.set_xlabel("Trading days relative to anchor")
    # matplotlib Axes: label the y-axis
    ax.set_ylabel("Cumulative return (anchor = 1.0)")
    # matplotlib Axes: render the legend
    ax.legend(fontsize=9)
    # matplotlib Axes: show a light grid
    ax.grid(alpha=0.3)
    # matplotlib Figure: tighten layout
    fig.tight_layout()
    if save_path:
        # matplotlib Figure: save figure to file
        fig.savefig(save_path, dpi=150)
    # matplotlib.pyplot: display the figure
    plt.show(block=False)


def rolling_sharpe_plot(
    roll_sharpe: pd.Series,
    window: int = TRADING_DAYS,
    cycles: list[dict] | None = None,
    df: pd.DataFrame | None = None,
    instrument: str = "2yr Payer",
    save_path: str | None = None,
):
    """Rolling Sharpe ratio for the payer strategy, with optional cycle shading."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(roll_sharpe.index, roll_sharpe, color="firebrick", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(
        roll_sharpe.index, roll_sharpe, 0,
        where=(roll_sharpe > 0), alpha=0.2, color="firebrick"
    )

    # shade active payer windows for each cycle so you can see when the signal was on
    if cycles is not None and df is not None:
        for i, c in enumerate(cycles):
            mask  = (df.index >= c["first_hike"]) & (df["signal"] == -1)
            block = df[mask]
            if block.empty:
                continue
            ax.axvspan(
                block.index[0], block.index[-1],
                alpha=0.10, color=_CYCLE_COLORS[i % len(_CYCLE_COLORS)],
                label=c["label"],
            )
        ax.legend(fontsize=8, ncol=3, title="Payer active")

    ax.set_title(f"Rolling {window}-day Sharpe — {instrument}", fontsize=13)
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)


def cycle_breakdown(
    df: pd.DataFrame,
    cycles: list[dict],
    instrument: str = "2yr Payer",
    save_path: str | None = None,
):
    """Per-cycle bar chart: annualised return and Sharpe of the payer position."""
    labels, ann_rets, sharpes = [], [], []

    for c in cycles:
        # pandas boolean indexing: filter rows to this cycle's active payer period
        mask = (df.index >= c["first_hike"]) & (df["signal"] == -1)
        # pandas DataFrame.loc: select the strat_ret column for matching rows, then drop NaN values
        r = df.loc[mask, "strat_ret"].dropna()
        if len(r) < 5:
            continue
        # pandas Series.mean: average daily return, scaled to annualised percentage
        ann_ret = r.mean() * TRADING_DAYS * 100
        # pandas Series.std: daily volatility, annualised by multiplying by sqrt(252) via numpy
        ann_vol = r.std() * np.sqrt(TRADING_DAYS) * 100
        # numpy: np.nan used as a sentinel when volatility is zero to avoid division by zero
        sharpe  = (ann_ret / ann_vol) if ann_vol > 0 else np.nan
        labels.append(c["label"])
        ann_rets.append(ann_ret)
        sharpes.append(sharpe)

    if not labels:
        print("No in-sample cycles to plot.")
        return

    # numpy: create an integer array [0, 1, 2, ...] used as bar x-positions
    x = np.arange(len(labels))
    # matplotlib.pyplot: create one Figure with two side-by-side Axes (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # matplotlib Axes: draw vertical bars; slice _CYCLE_COLORS to match the number of cycles
    ax1.bar(x, ann_rets, color=_CYCLE_COLORS[: len(labels)])
    # matplotlib Axes: zero-line reference
    ax1.axhline(0, color="black", lw=0.8)
    # matplotlib Axes: set numeric tick positions on the x-axis
    ax1.set_xticks(x)
    # matplotlib Axes: replace numeric ticks with cycle label strings, rotated for readability
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    # matplotlib Axes: title for the left subplot
    ax1.set_title(f"Annualised Return (%) per Cycle — {instrument}", fontsize=12)
    # matplotlib Axes: y-axis label
    ax1.set_ylabel("Ann. Return (%)")
    # matplotlib Axes: horizontal grid lines only (axis="y"), light opacity
    ax1.grid(alpha=0.3, axis="y")

    # matplotlib Axes: draw Sharpe bars on the right subplot with matching cycle colours
    ax2.bar(x, sharpes, color=_CYCLE_COLORS[: len(labels)])
    # matplotlib Axes: zero-line reference for the Sharpe chart
    ax2.axhline(0, color="black", lw=0.8)
    # matplotlib Axes: numeric tick positions on x-axis
    ax2.set_xticks(x)
    # matplotlib Axes: replace numeric ticks with cycle label strings
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    # matplotlib Axes: title for the right subplot
    ax2.set_title(f"Sharpe Ratio per Cycle — {instrument}", fontsize=12)
    # matplotlib Axes: y-axis label
    ax2.set_ylabel("Sharpe")
    # matplotlib Axes: horizontal grid lines only
    ax2.grid(alpha=0.3, axis="y")

    # matplotlib Figure: adjust spacing between subplots to prevent overlap
    fig.tight_layout()
    if save_path:
        # matplotlib Figure: write to file
        fig.savefig(save_path, dpi=150)
    # matplotlib.pyplot: render and display both subplots
    plt.show(block=False)


def carry_decomposition_plot(
    pnl_full: pd.DataFrame,
    cycles: list[dict],
    save_path: str | None = None,
):
    """
    Per-cycle stacked bar showing how total PnL splits across three sources:
      price_ret   — pure yield-move gain (what you're trying to capture)
      carry_fund  — funding carry drag (net coupon cost)
      carry_roll  — roll-down carry drag

    All values are cumulative returns over the active payer window of each cycle,
    expressed as percentages. Bars are stacked so total height = total_ret.
    """
    labels, price_vals, fund_vals, roll_vals = [], [], [], []

    for c in cycles:
        mask = (pnl_full.index >= c["first_hike"]) & (pnl_full.index <= c["last_hike"])
        window = pnl_full[mask].dropna()
        if window.empty:
            continue

        def _cum(col: str) -> float:
            r = -window[col]
            return ((1 + r).prod() - 1) * 100

        labels.append(c["label"])
        price_vals.append(_cum("price_ret"))
        fund_vals.append(_cum("carry_fund"))
        roll_vals.append(_cum("carry_roll"))

    if not labels:
        print("No cycles to plot in carry decomposition.")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x, price_vals, label="Price move (−D·Δy)", color="#4878CF")
    ax.bar(x, fund_vals,  label="Funding carry (coupon − repo)", color="#D65F5F",
           bottom=price_vals)
    bottom_roll = [p + f for p, f in zip(price_vals, fund_vals)]
    ax.bar(x, roll_vals,  label="Roll-down carry (−D·slope/250)", color="#C4AD66",
           bottom=bottom_roll)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("DGS2 Payer — Cumulative PnL Decomposition per Cycle (%)", fontsize=13)
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)


def pnl_components_timeseries(
    pnl_full: pd.DataFrame,
    signal: pd.Series,
    save_path: str | None = None,
):
    """
    Cumulative daily PnL for all four components on one line chart, restricted
    to days when the payer signal is active (signal == -1 or True).

    Lines:
      price_ret   — pure yield-move P&L (what the signal is capturing)
      carry_fund  — funding carry drag accumulated over time
      carry_roll  — roll-down carry drag accumulated over time
      total_ret   — sum of all three (what you'd actually earn)

    All series start at 0 on the first active signal day so they are directly
    comparable on the same axis.
    """
    # align signal to pnl_full index; treat missing as not active
    # signal may be boolean (detect_signal output) or int (-1/0/1 from calc_strat_ret)
    sig = signal.reindex(pnl_full.index).fillna(0)
    active = sig.astype(bool) | (sig == -1)

    # for a short position, flip the sign of each component
    cols = ["price_ret", "carry_fund", "carry_roll", "total_ret"]
    short_pnl = pnl_full[cols][active] * -1

    if short_pnl.empty:
        print("No active signal days to plot.")
        return

    # cumulative sum in return space (additive, not compounded) for readability
    cumulative = short_pnl.cumsum() * 100   # convert to basis points / percent

    fig, ax = plt.subplots(figsize=(14, 5))

    styles = {
        "price_ret":  ("#4878CF", 2.0, "-",  "Price move (−D·Δy)"),
        "carry_fund": ("#D65F5F", 1.4, "--", "Funding carry (coupon − repo)"),
        "carry_roll": ("#C4AD66", 1.4, "--", "Roll-down carry (−D·slope/250)"),
        "total_ret":  ("black",   2.0, "-",  "Total (price + both carry)"),
    }

    for col, (color, lw, ls, label) in styles.items():
        ax.plot(cumulative.index, cumulative[col], color=color, lw=lw, ls=ls, label=label)

    ax.axhline(0, color="grey", lw=0.7, ls=":")

    ax.set_title("DGS2 Payer — Cumulative PnL Components (signal-active days only)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%, additive)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)
