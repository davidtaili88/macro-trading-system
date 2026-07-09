"""
Data fetching and processing for the hiking cycle 2-year payer strategy.

Three return series are available for backtesting:

  SHY   — iShares 1-3yr Treasury ETF (Yahoo Finance).
           Cash bond proxy: ~1.9yr duration, but carry bleeds against a short
           position every day, suppressing payer P&L on gradual hiking cycles.

  ZT=F  — CME 2yr Treasury futures front month (Yahoo Finance).
           Purer duration expression: carry is netted into the roll basis rather
           than bleeding directly into P&L. Best tradable instrument.

  DGS2  — FRED constant-maturity 2yr yield (via fredapi), reconstructed into
           a return series using the duration approximation:
               r_t = -D * Δy_t       (price-only, no carry)
           where D = 1.921 (fixed modified duration for a 2yr note).
           Carry is intentionally excluded to isolate the pure yield-move signal.
           Convexity (½ * C * Δy²) is negligible at the 2yr point (C ≈ 5) and
           omitted. This series goes back to 1976 — covering all historical
           hiking cycles — which SHY and ZT cannot reach.

Hiking cycle dates are NOT stored here. They are supplied by the signal engine.
"""
import fredapi
import yfinance as yf
import pandas as pd
from fredapi import Fred


TICKER_SHY = "SHY"   # iShares 1-3yr Treasury ETF — cash bond, ~1.9yr duration, carry bleeds against payer
TICKER_ZT  = "ZT=F"  # CME 2yr Treasury futures front month — purer duration bet, carry netted in basis

START = "2002-01-01"   # SHY inception ~2002; ZT=F also available from ~2002
END   = None           # through today

# str | None is union syntax: end accepts either type string or a None
def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    #yf.download returns data for an instrument's Open, High, Low, Close, Volume, indexed by date. 
    #We care about the close value here so we only return the close column as a series
    #Indexed by date since this is daily data. 
    #Auto-adjust accounts for splits (when share amounts double but prices half)
    #Auto-adjust accounts for dividends (asset price would drop as investor receives cash, we ignore this)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    #If raw is an instance of a dataframe object
    if isinstance(raw, pd.DataFrame):
        #.squeeze() converts to a series
        raw = raw.squeeze()
    #Makes the column name "price"
    raw.name = "price"
    #Converts to dataframe since we obtain a series previously. 
    return raw.dropna().to_frame()


def fetch_shy(start: str = START, end: str | None = END) -> pd.DataFrame:
    """Return adjusted close prices for SHY at daily frequency."""
    return _download(TICKER_SHY, start, end)


def fetch_zt(start: str = START, end: str | None = END) -> pd.DataFrame:
    """
    Return back-adjusted close prices for ZT (2yr Treasury futures) at daily frequency.

    auto_adjust=True removes roll gaps so pct_change() gives clean returns without
    quarterly expiry jumps. Futures carry is netted into the basis rather than bleeding
    directly into P&L, making ZT a purer duration expression than SHY for a payer trade.
    """
    return _download(TICKER_ZT, start, end)


def _modified_duration(y: pd.Series, N: float = 2.0, k: int = 2) -> pd.Series:
    """
    Compute daily modified duration for a par bond from its yield series.

    Starting from D ≡ -(1/P)(dP/dy), writing P as PV of cash flows and
    differentiating gives D_mac = PV-weighted average time to cash flows.
    For a par bond (coupon = yield) that sum collapses to a yield-only formula:

        D_mac = (1/y) * [1 - 1/(1 + y/k)^(N*k)]

    Dividing by (1 + y/k) converts Macaulay → modified duration — the
    discrete-compounding correction left over from differentiating:

        D_mod = D_mac / (1 + y/k)

    y: yield series in decimal (e.g. 0.045 for 4.5%)
    N: maturity in years (2 for DGS2)
    k: coupon frequency (2 = semi-annual)
    """
    d_mac = (1 / y) * (1 - 1 / (1 + y / k) ** (N * k))
    return d_mac / (1 + y / k)


def fetch_carryless_dgs2_returns(
    fred_api_key: str,
    start: str = "1976-01-01",
) -> pd.Series:
    """
    Reconstruct daily price-only returns for a constant-maturity 2yr Treasury
    from the FRED DGS2 yield series, using the first-order duration approximation:

        r_t = -D_mod(y_t) * Δy_t

    D_mod is computed daily from the par-bond closed form rather than fixed at
    1.921 — duration breathes with the rate level (higher yields → shorter
    duration; lower yields → longer duration).

    Carry is intentionally excluded — this isolates the pure price/yield signal.
    Convexity (½ * C * Δy²) is omitted; at C ≈ 5 it contributes < 0.03bp per
    typical daily move versus ~19bp from duration — negligible at the 2yr point.

    Returns a Series named 'ret' starting from `start`, covering back to 1976.
    """
    fred = Fred(api_key=fred_api_key)
    raw  = fred.get_series("DGS2", observation_start=start)
    raw.index = pd.to_datetime(raw.index)
    y   = raw.dropna() / 100
    dy  = y.diff()
    D   = _modified_duration(y)
    ret = (-D * dy).rename("ret")
    return ret.dropna()


def fetch_dgs2_full_pnl(
    fred_api_key: str,
    start: str = "1976-01-01",
) -> pd.DataFrame:
    """
    Daily P&L decomposition for a short constant-maturity 2yr Treasury position.

    Three components, all in decimal return space (multiply by notional for $):

        price_ret   = -D_mod * Δy
                      pure yield-move P&L; positive when yields rise (short profits)

        carry_fund  = -(2y - fed_funds) / (250 * N)
                      net coupon cost: coupon owed to bond lender minus interest
                      earned on cash collateral at fed funds rate; negative when
                      curve is upsloping (2y > fed_funds)

        carry_roll  = -D_mod * (2y - 1y) / 250
                      roll-down cost: bond ages 1/250yr per day, sliding down the
                      upsloping curve so its yield drops and price rises — a loss
                      for a short; duration-adjusted to match return space

        total_ret   = price_ret + carry_fund + carry_roll

    Units: all terms are daily % of notional (decimal). Multiply by DV01 for $.

    Sources: DGS2, DGS1, DFF from FRED.
    """
    fred = Fred(api_key=fred_api_key)

    dgs2 = fred.get_series("DGS2", observation_start=start)
    dgs1 = fred.get_series("DGS1", observation_start=start)
    dff  = fred.get_series("DFF",  observation_start=start)

    dgs2.index = pd.to_datetime(dgs2.index)
    dgs1.index = pd.to_datetime(dgs1.index)
    dff.index  = pd.to_datetime(dff.index)

    df = pd.DataFrame({"dgs2": dgs2, "dgs1": dgs1, "dff": dff}).dropna()
    y2  = df["dgs2"] / 100
    y1  = df["dgs1"] / 100
    ff  = df["dff"]  / 100

    D  = _modified_duration(y2, N=2.0, k=2)
    dy = y2.diff()

    N_YEARS    = 2.0
    TRADE_DAYS = 250

    price_ret  = -D * dy
    carry_fund = -(y2 - ff) / (TRADE_DAYS * N_YEARS)
    carry_roll = -D * (y2 - y1) / TRADE_DAYS
    total_ret  = price_ret + carry_fund + carry_roll

    return pd.DataFrame({
        "price_ret":  price_ret,
        "carry_fund": carry_fund,
        "carry_roll": carry_roll,
        "total_ret":  total_ret,
    }).dropna()


def compute_returns(prices: pd.DataFrame) -> pd.Series:
    """Daily price returns from a DataFrame with a 'price' column."""
    return prices["price"].pct_change().rename("ret")


def get_zt_roll_dates(start: str, end: str, bdays_before_fnd: int = 5) -> pd.DatetimeIndex:
    """
    CME ZT roll dates anchored to First Notice Day (FND).

    ZT FND = last business day of the month prior to the delivery month
    (delivery months: Mar/Jun/Sep/Dec → FND falls in Feb/May/Aug/Nov).

    We roll `bdays_before_fnd` business days before FND, landing inside the
    peak calendar-spread liquidity window and safely before delivery risk.
    Default of 5 business days puts the roll in the final week of the prior month.

    """
    def _last_bday(year: int, month: int) -> pd.Timestamp:
        if month == 12:
            last = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            last = pd.Timestamp(year=year, month=month + 1, day=1) - pd.Timedelta(days=1)
        while last.weekday() >= 5:
            last -= pd.Timedelta(days=1)
        return last

    def _subtract_bdays(date: pd.Timestamp, n: int) -> pd.Timestamp:
        d, count = date, 0
        while count < n:
            d -= pd.Timedelta(days=1)
            if d.weekday() < 5:
                count += 1
        return d

    # FND month = month before each delivery month
    fnd_months = {3: 2, 6: 5, 9: 8, 12: 11}

    dates = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        for del_month, fnd_month in fnd_months.items():
            fnd  = _last_bday(year, fnd_month)
            roll = _subtract_bdays(fnd, bdays_before_fnd)
            dates.append(roll)

    return pd.DatetimeIndex(sorted(dates))
