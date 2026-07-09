"""
Data fetching for TIPS strategy backtest.

Proxies used (all from Yahoo Finance via yfinance):
  TIPS index  -> TIP  (iShares TIPS Bond ETF, ~7yr duration; note: passage uses ~4.95yr)
  Nominal idx -> IEF  (iShares 7-10yr Treasury ETF, ~7yr duration for rough duration match)
  Oil         -> CL=F (WTI Crude Futures continuous contract)
  Copper      -> HG=F (Copper Futures continuous contract)

The passage does not specify exact index construction. TIP/IEF are the closest
publicly available proxies; duration mismatch vs. the paper (~4.95 vs 5.7) will
shift absolute IR figures but the signal logic is identical.
"""

import yfinance as yf
import pandas as pd


TICKERS = {
    "tips": "TIP",
    "nominal": "IEF",
    "oil": "CL=F",
    "copper": "HG=F",
}

START = "2003-01-01"   # TIP ETF inception ~2003
END   = None            # through today


def fetch(start: str = START, end: str | None = END) -> pd.DataFrame:
    """Return a single DataFrame of adjusted close prices, daily frequency."""
    ticker_to_name = {v: k for k, v in TICKERS.items()}
    raw = yf.download(
        list(TICKERS.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]
    # yfinance sorts columns alphabetically; rename by ticker symbol explicitly
    raw = raw.rename(columns=ticker_to_name)
    raw = raw[list(TICKERS.keys())]   # enforce consistent column order
    raw = raw.dropna(how="all").ffill()
    return raw


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log daily returns for each series."""
    return prices.apply(lambda col: col.pct_change())
