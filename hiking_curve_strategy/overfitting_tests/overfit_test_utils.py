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
point from the middle", computed with medians so outliers don't distort it. MAD is a
valid spread measure for ANY distribution — it assumes nothing.

The 1.4826 factor is a UNIT CONSTANT, not a normality assumption. 1.4826 = 1/0.6745,
where 0.6745 is a fixed property of the standard normal (its 0.75 quantile). Multiplying
MAD by it puts MAD on a "standard-deviation-equivalent-IF-normal" scale — like "2.54 cm
per inch". If the data is NOT normal, 1.4826*MAD is simply not equal to sigma; it is still
a perfectly good robust spread measure, just on that scale. The data's true distribution
is never assumed, transformed, or invoked by this step.

Normality enters in EXACTLY ONE place, and it is optional: the choice of the CUTOFF 3.5.
Iglewicz & Hoaglin picked 3.5 by asking "on normal data, what modified-z is a ~0.05% tail?"
We use 3.5 as a HEURISTIC SENSITIVITY DIAL — a standard, well-understood trigger for "go
look at this", exactly like a 3-sigma control-chart limit — NOT as a p-value. We never
claim "3.5 gives a 0.05% false-positive rate here". If the jumps are fatter-tailed than
normal, 3.5 simply over-flags relative to the textbook rate, which — since false flags are
free — is the SAFE direction. Crucially, the fatal direction (missing a real cliff) is
guarded by the ABSOLUTE material backstop (see classify_plateau), which assumes nothing
about the distribution — so normality never touches the safety-critical path.

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
# ABSOLUTE material backstop: a jump >= this fraction of |pooled P&L| is flagged for
# review REGARDLESS of the relative z-score. This is the distribution-free, noise-proof
# safety net that catches the cliffs the relative z misses on noisy curves. Expressed as
# a FRACTION (not a fixed pp) so it scales across the price-only and carry-inclusive
# series (carry P&L is ~3x smaller). A disclosed risk tolerance, rounded DOWN (over-
# flagging is free); ~3% ≈ half a small cycle's P&L on the price-only series.
MATERIAL_FRACTION  = 0.03


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
    pooled_pnl=None,
    material_frac=MATERIAL_FRACTION,
):
    """Classify whether the live value sits on a plateau, or should be FLAGGED FOR REVIEW.

    IMPORTANT — this is a CONSERVATIVE SCREEN, not a classifier. The costs are asymmetric:
    a false flag costs a two-minute human investigation (cheap); a MISSED cliff means
    shipping an overfit parameter (fatal). So the verdict "FLAG_FOR_REVIEW" means exactly
    that — "a human should look at this" — NOT "this is definitely a cliff". The design
    errs toward over-flagging on purpose. The ONLY error we work to eliminate is a false
    PLATEAU (a real cliff called flat); false FLAG_FOR_REVIEWs are acceptable.

    TWO TESTS, WORKING TOGETHER (a backstop, not equal partners):
      1. RELATIVE screen (robust z on jumps): flag a jump that is a large OUTLIER versus
         the curve's own roughness. Self-scaling, but FAILS on noisy curves — a real cliff
         hidden in a jumpy background is not a relative outlier, so the z can MISS it. This
         is the failure mode that motivated test 2.
      2. ABSOLUTE backstop (material bar): flag any jump >= `material_frac` * |pooled_pnl|
         REGARDLESS of the z. Distribution-free and noise-proof — it cannot be hidden by a
         jumpy background, so it catches exactly the cliffs the z misses. Because it is a
         FRACTION of pooled P&L (not a fixed pp), it scales correctly across the price-only
         and carry-inclusive series (carry P&L is ~3x smaller, so a fixed pp bar would be
         far too permissive on it).

    WHAT THIS STILL MISSES (disclosed): only cliffs BELOW the material bar — so every
    possible miss is economically immaterial BY CONSTRUCTION (anything >= material_frac of
    pooled is flagged unconditionally). Within that sub-material band, if the jump is also
    small relative to the jump-noise, cliff and noise are genuinely indistinguishable
    locally — but since it is sub-material anyway, that is an acceptable miss, not a
    dangerous one. The mitigation for that regime is keeping the sweep SMOOTH (fine grid),
    not a cleverer threshold.

    outcomes:      1-D sequence of outcome values (e.g. pooled P&L %), one per swept value,
                   in sweep order.
    live_idx:      index of the live parameter value within `outcomes`.
    flat_floor:    jumps below this (in outcome units) are economically trivial. If the
                   LARGEST jump in the examined window is below it, verdict is PLATEAU and
                   scoring is skipped (handles the MAD≈0 degenerate case).
    z_cutoff:      robust-z above which a jump is flagged (the RELATIVE screen). A heuristic
                   sensitivity dial calibrated on normal tails, NOT a p-value — we make no
                   probabilistic claim from it (see robust_sigma docstring).
    window:        if given, only jumps within +/- `window` sweep steps of live_idx are
                   examined. None => use the whole sweep.
    pooled_pnl:    the strategy's pooled P&L at the LIVE value, in the SAME units as
                   `outcomes` (e.g. percent). Enables the ABSOLUTE material backstop. If
                   None, the backstop is inactive and only the relative z-screen runs
                   (pre-backstop behaviour, kept for backward compatibility).
    material_frac: a jump >= material_frac * |pooled_pnl| is flagged regardless of the z.
                   A DISCLOSED, conservatively-low risk tolerance (default 3%), chosen from
                   a miss-rate-vs-review-rate tradeoff and rounded DOWN because over-
                   flagging is free.

    Returns a dict with the verdict and every raw number behind it, so the caller can
    print an auditable breakdown and overrule the soft flag by eye. Verdict is one of
    "PLATEAU" (confident-flat), "CLIFF" (flag for review), or "NO_STABLE_REGION" (the
    whole sweep is rough — the parameter itself is erratic and the live value likely got
    lucky). The extra key "material_review" is True when the ABSOLUTE backstop fired.
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

    # ---- absolute material backstop threshold ----------------------------------
    # A jump this large (in outcome units) is flagged for review REGARDLESS of the z.
    # None => backstop inactive (pooled_pnl not supplied): pre-backstop behaviour.
    material_bar = (material_frac * abs(pooled_pnl)) if pooled_pnl is not None else None
    # does any jump ADJACENT to the live value clear the absolute bar?
    material_review = bool(
        material_bar is not None and adj_jumps.size and float(adj_jumps.max()) >= material_bar
    )

    result = {
        "n_points":        n,
        "window":          (o[lo_i], o[hi_i]) if window is not None else None,
        "win_lo_idx":      lo_i,
        "win_hi_idx":      hi_i,
        "max_jump":        float(win_jumps.max()) if win_jumps.size else 0.0,
        "median_jump":     float(np.median(win_jumps)) if win_jumps.size else 0.0,
        "robust_sigma":    float(robust_sigma(win_jumps)) if win_jumps.size else 0.0,
        "adj_jumps":       [float(x) for x in adj_jumps],
        "flat_floor":      flat_floor,
        "z_cutoff":        z_cutoff,
        "material_bar":    material_bar,
        "material_review": material_review,
    }

    # ---- BACKSTOP: absolute material jump adjacent to live -> flag, no matter what ----
    # Checked BEFORE the flat-floor guard: a real cliff sitting on an otherwise-flat curve
    # must never be dismissed as "trivially flat". This is the noise-proof safety net and
    # the ONLY test guarding the fatal direction (a missed real cliff), so it wins.
    if material_review:
        result["verdict"]    = "FLAG_FOR_REVIEW"
        result["reason"]     = (f"adjacent jump {max(adj_jumps):.3f} >= material bar "
                                f"{material_bar:.3f} ({material_frac:.0%} of pooled) "
                                f"[absolute backstop — noise-proof, z not consulted]")
        # z not computed on this path (the backstop is distribution-free by design);
        # report the raw adjacent jumps so the caller still sees the evidence.
        result["adj_z"]      = [float("nan") for _ in adj_jumps]
        result["cliff_frac"] = 0.0
        return result

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
    result["verdict"] = "FLAG_FOR_REVIEW" if on_cliff else "PLATEAU"
    result["reason"]  = (
        f"adjacent jump z={max(adj_z):.2f} > {z_cutoff}" if on_cliff
        else f"adjacent jump z={max(adj_z, default=0.0):.2f} <= {z_cutoff}"
    )
    return result
