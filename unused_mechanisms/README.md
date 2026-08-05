# unused_mechanisms/

Exploratory work for signals, exit gates, data sources, and whole strategies that were
**tested and NOT adopted** into the live hiking-cycle strategy
(`../hiking_curve_strategy/core/signal_logic.py`). Kept for provenance — every rejected
idea here has a written reason it lost — but nothing in this folder is imported by the
live algorithm or the runnable backtests in `../hiking_curve_strategy/strategies/`.

This folder sits at the project root, a sibling of `hiking_curve_strategy/`. The archived
scripts that still touch the hiking-cycle code bootstrap the sibling package onto `sys.path`
themselves, so run them **from the project root** (e.g. `python3 unused_mechanisms/derive_slope_window.py`).

The live algorithm uses only the **1yr and 3mo front-end spreads** (DGS1−DFF, DGS3MO−DFF),
the maturity ratio, a two-point median ROC gate, and a distance-to-neutral veto. Everything
below explores *other* variables/mechanisms that did not make the cut.

## `diagnostic_tests/` — rejected signal variables & gate designs

| file | mechanism explored | why not adopted |
|------|--------------------|-----------------|
| `test_2yr_vs_1yr_spread.py` | 2yr spread (DGS2−DFF) as a steadier second signal | no clean edge over the 1yr spread |
| `test_curve_slope_leading.py`, `test_curve_and_gate.py` | 10y−2y / 10y−3m term-structure slope as a late-cycle gate | doesn't separate real tops from mid-cycle false alarms |
| `test_expectations_gate.py` | term-premium-stripped expectations (Kim-Wright THREEFY1 − THREEFYTP1) | same bar the curve slope failed |
| `yield_slope_vs_spread_slope.py` | raw DGS1 yield slope vs spread slope | diagnostic; motivated but did not become a live gate |
| `compare_slope_gate.py` | OLS / Theil-Sen slope gate vs the live two-point median gate | live two-point median retained |
| `inspect_2005_vs_real_exit.py` | anatomy of the 2005 early-exit bug | diagnostic context |
| `diagnose_*`, remaining `test_*` | supporting analyses for the ratio / ROC / neutral-guard exits | diagnostic context |

## slope-gate redesign (not adopted)

- `derive_slope_window.py` — derives the window for a genuine trailing-slope Gate B
  (replacing the live two-point median gate). The redesign was **not** shipped; the live
  gate still uses the two-point median.

## IBKR ZT-carry data path (unused data source)

The live ZT backtest uses Yahoo `ZT=F`, which is back-adjusted upstream (carry removed),
so it is price-only. These scripts pull carry-preserving individual-contract histories from
Interactive Brokers instead — infrastructure for an **unresolved** data limitation, not
currently wired into any backtest:

- `fetch_zt_ibkr.py` — pull expired ZT contract histories from IBKR, manually roll to keep carry
- `probe_zt_contract_spec.py` — probe IBKR for the accepted ZT contract spec
- `ibkr_utils.py` — IBKR/TWS session helper

## `tips_strategy/` — separate strategy, shelved

A self-contained TIPS (breakeven-inflation) backtest — its own `data.py` / `signals.py` /
`backtest.py` / `plot.py` / `run.py`, independent of the hiking-cycle code. Archived here as
not-currently-pursued work. Run standalone from this folder: `python3 tips_strategy/run.py`.
