"""
overfit_test_utils — generalizable plateau/cliff statistics for parameter-sweep
overfitting tests.

Shared by every sweep_*.py driver in this folder. Given a 1-D sequence of outcome
values (typically pooled payer P&L, one per swept parameter value), it answers the
overfitting question:

    Is the live parameter value sitting on a FLAT PLATEAU (a neighbourhood of values
    that all produce essentially the same outcome -> the exact number is a don't-care,
    so there was nothing to overfit), or on a CLIFF (the outcome lurches as you nudge
    the parameter one step -> the result depends on threading a needle -> distrust it)?

The verdict is built from the LOCAL JUMPS of the sweep curve (first differences),
scored with a ROBUST z-score (median + MAD) so that one big cliff-jump cannot inflate
the very yardstick used to detect it (an ordinary stdev would mask the outlier — see
`robust_sigma`). No hand-tuned tolerance: the reference scale is the curve's own
typical roughness.

Why MAD, not stdev, and where 1.4826 comes from
-----------------------------------------------
MAD (median absolute deviation) = median(|xi - median(x)|): "the typical distance of a
point from the middle", computed with medians so outliers don't distort it. For NORMAL
data, MAD = 0.6745 * sigma (0.6745 is the ±radius around the mean capturing the central
50% of a bell curve — the sibling of "±1 sigma captures 68%"). So to express MAD on the
same scale as a standard deviation we multiply by 1/0.6745 = 1.4826. That scaling lets
the literature's "> 3.5 sigma = outlier" cutoff (Iglewicz & Hoaglin's modified z-score)
carry its intended meaning. The 1.4826 only rescales an already-clean MAD; it never
re-introduces the outlier the median discarded.

Two guards, for the two ways this can mislead on small samples
--------------------------------------------------------------
1. FLAT-FLOOR guard: if the biggest jump anywhere in the window is economically trivial
   (< flat_floor in outcome units), declare PLATEAU outright and skip scoring. This
   handles the degenerate MAD≈0 case (many identical points -> median jump 0, MAD 0 ->
   z-score explodes to infinity for any nonzero jump, paradoxically screaming "cliff").
2. BREAKDOWN guard: the median/MAD have a 50% breakdown point — if MORE than ~half the
   jumps are themselves cliffs, the "outliers" become the majority, the median moves
   into them, and the score inverts (flags the few plateau points as anomalies). When a
   large fraction of jumps are flagged we therefore report "NO STABLE REGION" rather than
   a per-point verdict: a parameter whose output lurches at most steps genuinely has no
   plateau to defend, so that IS the honest answer.
"""

import numpy as np


# Iglewicz-Hoaglin modified z-score outlier cutoff. Treated as a SOFT reference line
# (printed alongside raw numbers), not a hard gate — on ~10-30 jumps the median/MAD are
# themselves coarse, so a z of 3.2 vs 3.7 is not a meaningful distinction.
DEFAULT_Z_CUTOFF   = 3.5
# 1 / 0.6745 : converts MAD to a standard-deviation-equivalent scale for normal data.
MAD_TO_SIGMA       = 1.4826
# fraction of jumps flagged as cliffs above which we declare the sweep has no stable
# region (approaching the median/MAD 50% breakdown point; kept conservative below 0.5).
BREAKDOWN_FRACTION = 0.40


def robust_sigma(values):
    """MAD-based standard-deviation estimate: 1.4826 * median(|x - median(x)|).

    Robust to outliers because both the center (median) and the spread (median of
    absolute deviations) ignore how far the extreme points are — unlike np.std, whose
    mean-and-square step lets a single outlier inflate the result and mask itself.
    Returns 0.0 when the data has no spread (e.g. all-identical values)."""
    v = np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return MAD_TO_SIGMA * mad


def local_jumps(outcomes):
    """First differences |outcome[i] - outcome[i-1]| along the sweep.

    A cliff shows up as ONE large first difference; averaging both neighbours instead
    would smear a sharp step across two points and could hide it. Returns an array of
    length len(outcomes)-1, aligned so jumps[i] is the step INTO index i+1."""
    o = np.asarray(outcomes, dtype=float)
    return np.abs(np.diff(o))


def classify_plateau(
    outcomes,
    live_idx,
    flat_floor,
    z_cutoff=DEFAULT_Z_CUTOFF,
    window=None,
):
    """Classify whether the live value sits on a plateau or a cliff.

    outcomes:    1-D sequence of outcome values (e.g. pooled P&L %), one per swept value,
                 in sweep order.
    live_idx:    index of the live parameter value within `outcomes`.
    flat_floor:  jumps below this (in outcome units) are economically trivial. If the
                 LARGEST jump in the examined window is below it, verdict is PLATEAU and
                 scoring is skipped (handles the MAD≈0 degenerate case).
    z_cutoff:    robust-z above which a jump is flagged a cliff (soft reference line).
    window:      if given, only jumps within +/- `window` sweep steps of live_idx are
                 examined (a 'sensible range to look at' — a distant cliff at the far end
                 of an over-wide grid should not condemn the value the strategy uses, and
                 restricting the window also keeps the robust stats from being dominated
                 by a faraway regime). None => use the whole sweep.

    Returns a dict with the verdict and every raw number behind it, so the caller can
    print an auditable breakdown and overrule the soft flag by eye.
    """
    o = np.asarray(outcomes, dtype=float)
    n = len(o)
    jumps_all = local_jumps(o)  # length n-1; jumps_all[i] is the step between i and i+1

    # ---- select the window of jumps to examine ----------------------------------
    if window is None:
        lo_i, hi_i = 0, n - 1
    else:
        lo_i = max(0, live_idx - window)
        hi_i = min(n - 1, live_idx + window)
    # jumps that lie within [lo_i, hi_i]: these are indices lo_i .. hi_i-1 of jumps_all
    win_jumps = jumps_all[lo_i:hi_i] if hi_i > lo_i else np.array([])

    # jumps immediately adjacent to the live value (the step in and the step out).
    # These are the ones that decide whether *this* value is on an edge.
    adj = []
    if live_idx - 1 >= lo_i:
        adj.append(jumps_all[live_idx - 1])   # step from (live-1) -> live
    if live_idx <= hi_i - 1:
        adj.append(jumps_all[live_idx])        # step from live -> (live+1)
    adj_jumps = np.array(adj)

    result = {
        "n_points":       n,
        "window":         (o[lo_i], o[hi_i]) if window is not None else None,
        "win_lo_idx":     lo_i,
        "win_hi_idx":     hi_i,
        "max_jump":       float(win_jumps.max()) if win_jumps.size else 0.0,
        "median_jump":    float(np.median(win_jumps)) if win_jumps.size else 0.0,
        "robust_sigma":   float(robust_sigma(win_jumps)) if win_jumps.size else 0.0,
        "adj_jumps":      [float(x) for x in adj_jumps],
        "flat_floor":     flat_floor,
        "z_cutoff":       z_cutoff,
    }

    # ---- guard 1: everything is trivially flat ---------------------------------
    if result["max_jump"] < flat_floor:
        result["verdict"]      = "PLATEAU"
        result["reason"]       = "flat-floor"
        result["adj_z"]        = [0.0 for _ in adj_jumps]
        result["cliff_frac"]   = 0.0
        return result

    sigma = result["robust_sigma"]
    med   = result["median_jump"]

    # ---- robust z-scores (one-sided: a cliff is an anomalously LARGE jump) ------
    def _z(j):
        # ABSOLUTE-magnitude override, applied per-jump: a jump smaller than the
        # economic-triviality floor is never a cliff, regardless of the surrounding
        # distribution. This is essential when the sweep has a wide flat stretch, so
        # that median(jumps)=0 and MAD=0 (>half the jumps are exactly 0): without this,
        # the sigma<=0 branch below would score every tiny 0.05pp wiggle as inf and
        # falsely trip the breakdown guard. Economically: 0.05pp of pooled P&L is noise
        # no matter what its neighbours did.
        if j < flat_floor:
            return 0.0
        if sigma <= 0.0:
            # spread collapsed AND this jump is >= flat_floor: a genuine step sitting on
            # an otherwise-identical plateau. Flag it (caller sees sigma≈0 in the print).
            return float("inf")
        return (j - med) / sigma

    adj_z = [_z(j) for j in adj_jumps]
    result["adj_z"] = adj_z

    # ---- guard 2: breakdown — too many cliffs for robust stats to be trusted ----
    # The breakdown guard detects a genuinely ROUGH curve (the outcome lurches at most
    # steps). A window with sigma==0 is the OPPOSITE of rough — it is so flat that > half
    # the jumps are identical, so the z-score is not meaningful and the guard must not
    # run (otherwise the few above-floor edge jumps score inf and falsely trip it). A flat
    # window is by definition a plateau, not a no-stable-region.
    if sigma <= 0.0:
        cliff_frac = 0.0
    elif win_jumps.size:
        z_all = np.array([_z(j) for j in win_jumps])
        cliff_frac = float(np.mean(z_all > z_cutoff))
    else:
        cliff_frac = 0.0
    result["cliff_frac"] = cliff_frac

    if cliff_frac > BREAKDOWN_FRACTION:
        result["verdict"] = "NO_STABLE_REGION"
        result["reason"]  = f"{cliff_frac:.0%} of jumps are cliffs (> {z_cutoff}sigma)"
        return result

    # ---- normal case: is the live value on an edge? ----------------------------
    on_cliff = any(z > z_cutoff for z in adj_z)
    result["verdict"] = "CLIFF" if on_cliff else "PLATEAU"
    result["reason"]  = (
        f"adjacent jump z={max(adj_z):.2f} > {z_cutoff}" if on_cliff
        else f"adjacent jump z={max(adj_z, default=0.0):.2f} <= {z_cutoff}"
    )
    return result
