# Macro Trading System — Hiking-Cycle 2-Year Payer

A backtested, event-driven rates strategy that **shorts the 2-year Treasury (a "payer"
position) through Federal Reserve hiking cycles**, with entry and exit driven entirely
by market-observable signals — no oracle knowledge of FOMC decisions.

Built independently, mentored by a rates PM at Brevan Howard.

---

## The idea

During a Fed hiking cycle, front-end yields rise and 2-year prices fall, so a short/payer
position gains. The hard part is **timing without hindsight**: a real trader knows neither
when the first hike lands nor when the last one has passed. This strategy detects both from
the shape of the front-end curve:

- **Entry** — the market starts pricing a hiking cycle: the 3-month and 1-year spreads over
  fed funds (`DGS3MO − DFF`, `DGS1 − DFF`) both clear their thresholds inside a confirmed
  post-easing hold.
- **Exit** — the market starts pricing the cycle *out*. Several market-driven conditions,
  the primary one being a **maturity ratio** (hiking still priced ÷ hiking already
  delivered): once what's-still-priced shrinks to a small fraction of what's-been-done, the
  cycle is mostly behind us and we leave — often while the spread is still positive, well
  before a naive spread-level exit would fire.

Everything the strategy uses is knowable in real time. The Fed's actual hike dates are used
**only** to build a perfect-hindsight benchmark to measure against — never in the live rules.

## Headline result

Measured on the constant-maturity **DGS2** yield (price-only, pre-carry), pooled across the
**five hiking cycles since 1990** (1994-95, 1999-2000, 2004-06, 2015-18, 2022-23):

| Metric | Value |
|---|---|
| Capture of the perfect-hindsight benchmark | **56.2%** (17.3% strategy / 30.7% oracle) |
| Late-exit reduction vs. a naive spread-level trigger | up to **~2 months** earlier |

The perfect-hindsight benchmark enters 60 trading days before the first hike and exits 30
before the last — an untradeable upper bound. Capturing ~56% of it with a signal that uses
no future information is the core result. These figures are reproduced from live FRED data
by [`resume_verification/verify.py`](hiking_curve_strategy/resume_verification/verify.py).

## Instrument and its honest limits

DGS2 is a **constant-maturity** series: it re-strikes the exact 2-year point every day, so
it isolates the pure rate move (`return ≈ −D·Δy`) with no ageing or roll to model. That
makes it the clean series for answering *"does the signal catch the move?"* — but it
carries **no funding carry and no roll-down**, which a real position pays. Two side
experiments probe tradability under those frictions:

- [`signal_trade_dgs.py`](hiking_curve_strategy/strategies/signal_trade_dgs.py) also runs a
  full `price + funding carry + roll-down` decomposition.
- [`signal_trade_zt.py`](hiking_curve_strategy/strategies/signal_trade_zt.py) prices the
  same signal on a rollable 2-year futures contract (carry netted into the roll).
- [`signal_trade_held_note.py`](hiking_curve_strategy/strategies/signal_trade_held_note.py)
  contrasts the constant-maturity view against a single held, ageing note (locked coupon).

## Guarding against overfitting

With only ~5 cycles, tuning a threshold to P&L would be overfitting by construction. Every
discretionary parameter is instead **swept and checked for a flat plateau** — a band of
values that all behave identically, so the exact number was never load-bearing. See
[`overfitting_tests/`](hiking_curve_strategy/overfitting_tests/): each sweep runs the real
`detect_signal` across a range of the parameter and reports whether the live value sits on a
plateau or a cliff. Values that genuinely matter (e.g. the entry bar) are disclosed as such
rather than dressed up as data-derived.

## Repository layout

```
hiking_curve_strategy/
  core/
    signal_logic.py       # market-driven entry/exit signal (the heart of the strategy)
    backtest.py           # applies a signal to a return series → daily P&L
    data.py               # FRED fetch + DGS2 return reconstruction (−D·Δy, carry decomposition)
    plot.py               # equity curve, event-time, rolling-Sharpe, per-cycle charts
  strategies/
    run.py                # entry point — runs the DGS2 signal-driven backtest
    signal_trade_dgs.py   # DGS2 (primary)
    signal_trade_zt.py    # 2yr futures — side tradability experiment
    signal_trade_held_note.py  # held/ageing note — side contrast
  benchmarks/
    benchmark.py          # perfect-hindsight oracle (the upper bound)
  overfitting_tests/      # parameter sweeps + plateau/cliff verdicts
  utils/
    fred_utils.py         # FRED API helpers
    performance_evaluation.py  # Sharpe, drawdown, cycle-matched stats
  resume_verification/
    verify.py             # reproduces the headline numbers from live data
unused_mechanisms/        # signals/gates tested and rejected, with reasons (provenance)
```

## Running it

Requires Python 3, a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
(set `FRED_API_KEY` in a `.env` file or the environment), and `pandas numpy matplotlib
fredapi python-dotenv`.

```bash
cd hiking_curve_strategy

# run the signal-driven DGS2 backtest (charts + per-cycle P&L)
PYTHONPATH=. python3 strategies/run.py

# reproduce the headline capture / late-exit numbers
PYTHONPATH=. python3 resume_verification/verify.py

# check a parameter isn't overfit (plateau vs cliff)
PYTHONPATH=. python3 overfitting_tests/sweep_ratio_exit_threshold.py
```

All inputs are public FRED series (`DFF`, `DGS1`, `DGS3MO`, `DGS2`, `DFEDTAR`/`DFEDTARL`) —
no paid data required.

---

*Independent research project. Results are backtested on a constant-maturity yield proxy;
the carry-inclusive and futures tradability studies are ongoing.*
