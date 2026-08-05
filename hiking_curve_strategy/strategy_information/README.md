# Learning notebook — `signal_logic.detect_signal`

A guided, hands-on walkthrough of the market-driven signal engine, combined with
fill-in-the-blank pandas/numpy drills. Built to (a) make the code legible and
(b) drill the exact array/DataFrame idioms the strategy uses. Also a good source
of visuals for a slide deck (the Stage-4 event log + marker plot especially).

## Run it

```bash
cd hiking_curve_strategy/learning_notebook
jupyter lab signal_logic_walkthrough.ipynb      # or: jupyter notebook / VS Code
```

Needs a `FRED_API_KEY` (picked up from `.env` or the environment, same as
`signal_trade_dgs.py`; a fallback key is baked in). Requires `pandas`, `numpy`,
`matplotlib`, `fredapi`, `python-dotenv`.

## How to use each stage

For every `### DRILL`:
1. **Predict** the answer to the prompt in the markdown before touching code.
2. Fill the `____` blanks in the drill cell and run it.
3. Run the **ANSWER** cell below to check yourself + see why it's written that way.
4. Run the walkthrough cell that uses the idea on real data.

The ANSWER cells re-fetch what they need, so they also run standalone.

## What each stage covers

| Stage | Source region | idioms drilled |
|-------|---------------|----------------|
| 1 | `_load_signal_data` | `.diff()/.abs()`, boolean masks, `concat`, `.duplicated()`, `reindex(method="ffill")` |
| 2 | `_last_cut_dates` | `.iloc` vs `.loc`, ±1bp thresholding |
| 3 | `detect_signal` pre-compute | `.rolling().mean()` + `min_periods`, `.clip()`, `.cumsum()`, `.diff(lag)`, double-window median, `set()` |
| 4 | `detect_signal` main loop | the **instrumented latch** — every ENTRY/EXIT logged; verified identical to the real function |
| 5 | `signal_to_cycles` / `stats_per_cycle` | `.items()`, `.prod()` compounding |
| 6 | assertion hunt | encode your mental model, break it on purpose |

Next: repeat the same method (run → check type/shape/NaN → drill → assert) on
`data.py`'s `fetch_carryless_dgs2_returns`.
