"""
Visualisation for the hiking cycle 2-year payer strategy.

Four plots:
  1. equity_curve      — cumulative strategy vs buy-hold bonds, with cycle shading
  2. event_time_plot   — normalised bond returns around first/last hike (replicates
                         book Figs 5.5 / 5.6): median + percentile band per cycle
  3. rolling_sharpe    — rolling 252-day Sharpe for the payer strategy
  4. cycle_breakdown   — per-cycle bar chart of annualised return and Sharpe
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
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
    data_start: pd.Timestamp | None = None,
    raw_price: pd.Series | None = None,
    save_path: str | None = None,
):
    """
    Cumulative equity of payer strategy vs. long bonds, with:
      - data clipped to data_start (drops pre-coverage grey zone)
      - payer windows drawn in series color; outside windows the strategy
        curve goes grey to show it is tracking long bonds (counterfactual)
      - a muted buy-hold line shown only outside payer windows so the two
        curves can be compared directly when the signal is off
      - shading for each active payer window
      - vertical lines for oracle first-hike dates (if oracle_cycles supplied)
      - a binary latched indicator panel (if latched_series supplied)
      - raw instrument price on a secondary y-axis (if raw_price supplied),
        to visually check signal entry/exit timing against the actual price
        level rather than only the derived equity curve
    """
    # clip to data coverage start so pre-FRED-series grey zone is excluded
    plot_df = df[df.index >= data_start] if data_start is not None else df

    if latched_series is not None:
        fig, (ax, ax_latch) = plt.subplots(
            2, 1, figsize=(13, 6),
            gridspec_kw={"height_ratios": [4, 1]},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax_latch = None

    # ── build curves ──────────────────────────────────────────────────────
    # buy-hold: long bonds the whole period
    raw_bh = (1 + plot_df["bond_ret"]).cumprod()
    base   = raw_bh.iloc[0] if len(raw_bh) > 0 else 1.0
    bh     = raw_bh / base

    # strategy: cum_equity from calc_strat_ret is flat (=1) when signal is off
    # and compounds payer returns only during active windows — keep that as-is.
    strat = plot_df["cum_equity"] / plot_df["cum_equity"].iloc[0]

    # ── buy-hold line ─────────────────────────────────────────────────────
    ax.plot(bh.index, bh,
            color="#DC143C", lw=1.0, alpha=0.6,
            label="Buy-hold bonds (long, reference)")

    # ── strategy curve ────────────────────────────────────────────────────
    ax.plot(strat.index, strat.values,
            color="#2a78d6", lw=1.5,
            label="Payer strategy (flat when signal off)")

    # ── shade active payer windows ─────────────────────────────────────────
    for i, c in enumerate(cycles):
        mask  = (plot_df.index >= c["first_hike"]) & (plot_df["signal"] == -1)
        block = plot_df[mask]
        if block.empty:
            continue
        ax.axvspan(
            block.index[0], block.index[-1],
            alpha=0.08, color=_CYCLE_COLORS[i % len(_CYCLE_COLORS)],
            label=f"Signal active: {c['label']}",
        )

    # ── oracle vertical lines ──────────────────────────────────────────────
    if oracle_cycles is not None:
        for j, oc in enumerate(oracle_cycles):
            fh, lh = oc["first_hike"], oc["last_hike"]
            if plot_df.index[0] <= fh <= plot_df.index[-1]:
                ax.axvline(fh, color=_CYCLE_COLORS[j % len(_CYCLE_COLORS)],
                           lw=1.2, ls="--", alpha=0.8,
                           label=f"Oracle 1st hike: {oc['label']}")
            if plot_df.index[0] <= lh <= plot_df.index[-1]:
                ax.axvline(lh, color=_CYCLE_COLORS[j % len(_CYCLE_COLORS)],
                           lw=0.9, ls=":", alpha=0.65,
                           label=f"Oracle last hike: {oc['label']}")

    # ── raw instrument price on a secondary axis — lets you eyeball whether
    # entries/exits (shaded windows) line up with turns in the actual price,
    # not just the derived equity curve ─────────────────────────────────────
    if raw_price is not None:
        price_aligned = raw_price.reindex(plot_df.index)
        ax_price = ax.twinx()
        ax_price.plot(price_aligned.index, price_aligned.values,
                      color="#9b3fae", lw=1.0, alpha=0.55, ls="-",
                      label=f"{instrument} price (RHS)")
        ax_price.set_ylabel(f"{instrument} price", color="#9b3fae")
        ax_price.tick_params(axis="y", labelcolor="#9b3fae")
        price_handles, price_labels = ax_price.get_legend_handles_labels()
    else:
        price_handles, price_labels = [], []

    ax.set_title(f"{instrument} — Cumulative Equity vs. Buy-Hold", fontsize=13)
    ax.set_ylabel("Growth of $1 (rebased at plot start)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + price_handles, labels + price_labels, fontsize=7, ncol=2)
    # recessive hairline grid
    ax.grid(axis="y", color="#e1e0d9", lw=0.8, linestyle="-")
    ax.grid(axis="x", color="#e1e0d9", lw=0.8, linestyle="-")

    if latched_series is not None and ax_latch is not None:
        latched_aligned = latched_series.reindex(plot_df.index).fillna(False).astype(float)
        ax_latch.fill_between(
            latched_aligned.index, latched_aligned, 0,
            step="post", color="#eda100", alpha=0.55, label="Latched"
        )
        ax_latch.set_ylim(-0.05, 1.3)
        ax_latch.set_yticks([0, 1])
        ax_latch.set_yticklabels(["off", "on"], fontsize=8)
        ax_latch.set_ylabel("Latched", fontsize=8)
        ax_latch.legend(fontsize=7, loc="upper left")
        ax_latch.grid(color="#e1e0d9", lw=0.8)
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
    if event_df.empty:
        print("No event-time data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # plot individual cycles lightly
    for col in event_df.columns:
        ax.plot(event_df.index, event_df[col], lw=0.8, alpha=0.4)

    # median and 10th-90th percentile band across cycles
    median = event_df.median(axis=1)
    p10    = event_df.quantile(0.10, axis=1)
    p90    = event_df.quantile(0.90, axis=1)

    ax.plot(event_df.index, median, color="black", lw=2.0, label="Median")
    ax.fill_between(event_df.index, p10, p90, alpha=0.15, color="black", label="10th-90th pct")

    ax.axvline(0, color="red", lw=1.2, ls="--", label=f"t=0: {anchor_label}")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")

    ax.set_title(f"2yr Bond: Normalised Cumulative Return Around {anchor_label.title()}", fontsize=13)
    ax.set_xlabel("Trading days relative to anchor")
    ax.set_ylabel("Cumulative return (anchor = 1.0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)


def rolling_sharpe_plot(
    roll_sharpe: pd.Series,
    window: int = TRADING_DAYS,
    cycles: list[dict] | None = None,
    df: pd.DataFrame | None = None,
    instrument: str = "2yr Payer",
    data_start: pd.Timestamp | None = None,
    save_path: str | None = None,
):
    """
    Rolling Sharpe ratio for the payer strategy, with:
      - grey shading over the pre-data-coverage warm-up period
      - cycle shading for active payer windows
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # grey out the warm-up period before data coverage begins
    if data_start is not None and len(roll_sharpe) > 0:
        pre_mask = roll_sharpe.index < data_start
        if pre_mask.any():
            ax.axvspan(
                roll_sharpe.index[0], data_start,
                color="#e1e0d9", alpha=0.9, zorder=0,
                label="No data (pre-coverage)",
            )
        rs_valid = roll_sharpe[roll_sharpe.index >= data_start]
    else:
        rs_valid = roll_sharpe

    ax.plot(rs_valid.index, rs_valid, color="firebrick", lw=1.2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(
        rs_valid.index, rs_valid, 0,
        where=(rs_valid > 0), alpha=0.2, color="firebrick",
    )

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
    ax.grid(axis="y", color="#e1e0d9", lw=0.8, linestyle="-")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show(block=False)


def cycle_breakdown(
    df: pd.DataFrame,
    cycles: list[dict],
    instrument: str = "2yr Payer",
    cycle_matched_sharpes: dict | None = None,
    save_path: str | None = None,
):
    """
    Per-cycle bar chart: annualised return and Sharpe of the payer position,
    with a side-by-side buy-hold bar for the same window so you can see how
    much value the payer adds vs just holding long bonds in those same periods.

    If cycle_matched_sharpes is provided (from performance_evaluation.cycle_matched_sharpe),
    the aggregated payer vs buy-hold Sharpe is annotated above the Sharpe subplot.
    """
    labels        = []
    payer_rets    = []
    bh_rets       = []
    payer_sharpes = []
    bh_sharpes    = []

    for c in cycles:
        mask = (df.index >= c["first_hike"]) & (df["signal"] == -1)
        r_payer = df.loc[mask, "strat_ret"].dropna()
        r_bh    = df.loc[mask, "bond_ret"].dropna()
        if len(r_payer) < 5:
            continue

        def _stats(r: pd.Series) -> tuple[float, float]:
            ar  = r.mean() * TRADING_DAYS * 100
            av  = r.std()  * np.sqrt(TRADING_DAYS) * 100
            sh  = (ar / av) if av > 0 else np.nan
            return ar, sh

        pr, ps = _stats(r_payer)
        br, bs = _stats(r_bh)
        labels.append(c["label"])
        payer_rets.append(pr)
        bh_rets.append(br)
        payer_sharpes.append(ps)
        bh_sharpes.append(bs)

    if not labels:
        print("No in-sample cycles to plot.")
        return

    num_cycles     = np.arange(len(labels))
    width = 0.38                      # bar width — pair fits within each group
    gap   = 0.02                      # 2px surface gap between adjacent bars

    # palette: slot 1 (blue) for payer, slot 6 (red) for buy-hold (opposing roles)
    C_PAYER = "#2a78d6"
    C_BH    = "#e34948"

    fig, (ax_ann_returns, ax_ann_sharpe) = plt.subplots(1, 2, figsize=(13, 4))

    # ── annualised return subplot ──────────────────────────────────────────
    ax_ann_returns.bar(num_cycles - (width / 2 + gap / 2), payer_rets, width,
            color=C_PAYER, label="Payer (signal on)")
    ax_ann_returns.bar(num_cycles + (width / 2 + gap / 2), bh_rets, width,
            color=C_BH,    label="Buy-hold bonds (same window)")
    ax_ann_returns.axhline(0, color="#0b0b0b", lw=0.8)
    ax_ann_returns.set_xticks(num_cycles)
    ax_ann_returns.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_ann_returns.set_title(f"Ann. Return (%) per Cycle — {instrument}", fontsize=11)
    ax_ann_returns.set_ylabel("Ann. Return (%)")
    ax_ann_returns.legend(fontsize=8)
    ax_ann_returns.grid(axis="y", color="#e1e0d9", lw=0.8, linestyle="-")

    # ── Sharpe subplot ────────────────────────────────────────────────────
    ax_ann_sharpe.bar(num_cycles - (width / 2 + gap / 2), payer_sharpes, width,
            color=C_PAYER, label="Payer (signal on)")
    ax_ann_sharpe.bar(num_cycles + (width / 2 + gap / 2), bh_sharpes, width,
            color=C_BH,    label="Buy-hold bonds (same window)")
    ax_ann_sharpe.axhline(0, color="#0b0b0b", lw=0.8)
    ax_ann_sharpe.set_xticks(num_cycles)
    ax_ann_sharpe.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_ann_sharpe.set_title(f"Sharpe per Cycle — {instrument}", fontsize=11)
    ax_ann_sharpe.set_ylabel("Sharpe")
    ax_ann_sharpe.legend(fontsize=8)
    ax_ann_sharpe.grid(axis="y", color="#e1e0d9", lw=0.8, linestyle="-")

    # aggregated Sharpe annotation above the Sharpe subplot
    if cycle_matched_sharpes is not None:
        ps = cycle_matched_sharpes.get("payer_sharpe", float("nan"))
        bs = cycle_matched_sharpes.get("bh_sharpe",    float("nan"))
        nd = cycle_matched_sharpes.get("payer_days",   0)
        ax_ann_sharpe.set_xlabel(
            f"Pooled across all cycles ({nd} days): "
            f"Payer Sharpe = {ps:.2f}   Buy-hold Sharpe = {bs:.2f}",
            fontsize=8, color="#52514e",
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
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
    ax.bar(x, roll_vals,  label="Roll-down carry (−D·slope/252)", color="#C4AD66",
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
        "carry_roll": ("#C4AD66", 1.4, "--", "Roll-down carry (−D·slope/252)"),
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
