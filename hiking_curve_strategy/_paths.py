"""
Path bootstrap for the hiking_curve_strategy package.

The strategy is organised into sibling source folders (core/, benchmark/, utils/)
but every module is imported by its flat name (`from data import ...`,
`from backtest import ...`, `from benchmark import ...`). That convention predates
the folder split and is used by 15+ scripts, so rather than rewrite every import to
a dotted path, this module registers the source folders on sys.path.

Any runnable script (in strategies/, benchmarks/, overfitting_tests/,
parameter_generation/, resume_verification/) should do, before its intra-package
imports:

    import _paths  # noqa: F401  — registers core/, benchmark/, utils/ on sys.path

It is import-order safe (idempotent) and works no matter which folder the script
is run from, as long as the package root is on sys.path (PYTHONPATH=. from the
hiking_curve_strategy dir, which is how the scripts are invoked).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folders holding flat-imported modules, in priority order.
_SRC_DIRS = [
    _ROOT,                          # package root (utils/ is imported as a package)
    os.path.join(_ROOT, "core"),        # data, signal_logic, backtest, plot
    os.path.join(_ROOT, "benchmarks"),  # benchmark (ALL_CYCLES, ORACLE_CYCLES, DGS2 oracle)
]

for _d in _SRC_DIRS:
    if _d not in sys.path:
        sys.path.insert(0, _d)
