"""
Entry point: fetch data, generate signals, run backtest, print summary, plot.

Usage:
    python tips_strategy/run.py
"""

import warnings
warnings.filterwarnings("ignore")

from data    import fetch_shy, compute_returns
from signals import generate, spread_returns
from backtest import run, summary
from plot    import equity_curves, rolling_ir


def main():
    print("Fetching data...")
    prices  = fetch_shy()
    returns = compute_returns(prices)

    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Rows: {len(prices)}\n")

    signals = generate(prices)
    spread  = spread_returns(returns)

    results = run(spread, signals)
    stats   = summary(results)

    print("=== Strategy Summary ===")
    print(stats.to_string())
    print()
    print("Passage benchmarks: base=0.23, oil_ma=0.63, copper_ma=0.81")
    print("(divergence expected due to ETF proxy vs. exact index construction)")

    equity_curves(results)
    rolling_ir(results, strategy="copper_ma")


if __name__ == "__main__":
    main()
