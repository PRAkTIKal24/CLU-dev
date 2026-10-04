#!/usr/bin/env python
"""Read every unit's artefacts and answer the PREREG's four conditions.

⛔ Reads only `build.json` / `eval.json` / `rival.json` / `preds.npz` / `rival_preds.npz`.
It re-derives every number from the saved PER-REQUEST prediction matrices, so no
subset here can have been chosen after seeing an aggregate.

⚠ Runs on whatever units exist. A missing unit is reported MISSING, never skipped
silently and never filled in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
CARDS = [10, 10, 10, 8, 4, 15]
#: ⭐ PREREG Amendment A3. The geometry of each factor, VERIFIED against
#: `factor_index.npz` rather than assumed, because it decides what "distance" means
#: and therefore whether a deletion is distinguishable from a one-step shift:
#:   hues        0.0 .. 0.9 over 10 values  -> CYCLIC (HSV hue wraps 0.9 -> 0.0)
#:   scale       0.75 .. 1.25 over 8        -> linear ordinal
#:   shape       0,1,2,3                    -> CATEGORICAL, distance is meaningless
#:   orientation -30 .. +30 deg over 15     -> linear ordinal, spans 60 deg, NOT cyclic
#: ⚠ The last one matters: a reviewer suggested orientation's far-tail mass was
#: wrap-around on a cyclic factor. It is not — the grid spans 60 degrees, not 360 —
#: so that mass is genuine error and must not be explained away as wrap.
GEOMETRY = {"floor_hue": "cyclic", "wall_hue": "cyclic", "object_hue": "cyclic",
            "scale": "ordinal", "shape": "categorical", "orientation": "ordinal"}


def grid_distance(pred: np.ndarray, target: int, factor: int) -> np.ndarray:
    """|pred - target| in GRID STEPS, respecting the factor's geometry."""
    g = GEOMETRY[FACTORS[factor]]
    d = np.abs(np.asarray(pred, np.int64) - int(target))
    if g == "cyclic":
        d = np.minimum(d, CARDS[factor] - d)
    return d


def distance_profile(pred: np.ndarray, target: int, factor: int) -> dict:
    """Mass at |d| = 0, 1, 2, >=3 — the statistic that tells FORGETTING apart from a
    ONE-STEP SHIFT.  ⛔ An exact-match rate cannot: on an ordinal factor a store that
    renders the neighbouring value scores ~0 and would 'pass' an unrecoverability
    threshold while still emitting something one grid step from the deleted value."""
    if GEOMETRY[FACTORS[factor]] == "categorical":
        return {"geometry": "categorical",
                "exact": float((np.asarray(pred) == int(target)).mean()),
                "note": ("⛔ distance is not meaningful for a categorical factor; read "
                         "exact-match only")}
    d = grid_distance(pred, target, factor)
    n = max(1, d.size)
    return {"geometry": GEOMETRY[FACTORS[factor]],
            "d0": float((d == 0).sum() / n), "d1": float((d == 1).sum() / n),
            "d2": float((d == 2).sum() / n), "d3plus": float((d >= 3).sum() / n),
            "within_1": float((d <= 1).sum() / n), "mean_abs_d": float(d.mean())}
#: PREREG §3 — fixed before the run.
DELETIONS = [(0, 5), (1, 5), (2, 5), (3, 4), (4, 2), (5, 7)]
#: PREREG §3 — the size-matched churn control for each of them.  Tag scheme merged
#: with the sibling thread's `plan_ordered.jsonl` at 11:21Z (identical specs, and its
#: ordering puts the most informative unit first), so the cell is built ONCE.
PLACEBO = {"del_f0v5": "plc_f0v5", "del_f1v5": "plc_f1v5", "del_f2v5": "plc_f2v5",
           "del_f3v4": "plc_f3v4", "del_f4v2": "plc_f4v2", "del_f5v7": "plc_f5v7"}
CONJ, CONJ_PLC = "del_conj_oh5_sh2", "plc_conj_oh5_sh2"
#: three independent placebos at ONE size (n=8234) -> a churn DISTRIBUTION, which is
#: what PREREG §7b's "within the placebo band" needs and a single placebo cannot give.
CHURN = ["plc_churn_r1", "plc_churn_r2", "plc_churn_r3"]
#: ⚠ `det_A` (the sibling's process) and `det_B` (mine) are the two INDEPENDENT
#: full-pool builds that form the determinism gate.  `det_B` is the pre-deletion BASE
#: because `det_A`'s eval was first run at reduced n during a plumbing smoke.
BASE = "det_B"
M_TIE = 0.020  # R1, fixed before the run


def tag_dir(runs: Path, tag: str) -> Path:
    return runs / tag


def load(runs: Path, tag: str) -> dict | None:
    d = tag_dir(runs, tag)
    if not (d / "build.json").exists():
        return None
    out = {"build": json.loads((d / "build.json").read_text())}
    for nm, fn in (("eval", "eval.json"), ("rival", "rival.json")):
        if (d / fn).exists():
            out[nm] = json.loads((d / fn).read_text())
    for nm, fn in (("preds", "preds.npz"), ("rpreds", "rival_preds.npz")):
        if (d / fn).exists():
            out[nm] = np.load(d / fn)
    return out


# ---------------------------------------------------------------------------
# paired BCa bootstrap, resampling REQUESTS (PREREG §7)
# ---------------------------------------------------------------------------


def bca(per_request: np.ndarray, n_boot: int = 10000, seed: int = 20260917,
        alpha: float = 0.05) -> dict:
    """BCa CI for the mean of `per_request` (one value per resampling unit).

    ⭐ The resampling unit is the REQUEST, as pre-registered — not the (request,
    deletion) pair, which would treat six correlated observations of the same
    request as independent and give a CI that is too narrow.
    """
    x = np.asarray(per_request, np.float64)
    # ⚠ dropping non-finite entries SILENTLY changes n, which is how ragged
    # participation hides.  Count it and report it instead.
    n_dropped = int((~np.isfinite(x)).sum())
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"mean": float(x.mean()) if n else float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": int(n), "note": "too few units for a CI"}
    theta = float(x.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boots = x[idx].mean(axis=1)
    # bias correction
    prop = float((boots < theta).mean())
    prop = min(max(prop, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    from math import erf, sqrt

    def ppf(p):  # inverse standard normal, Acklam-free: use erfinv via bisection
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cdf(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    z0 = ppf(prop)
    # acceleration by jackknife
    tot = x.sum()
    jack = (tot - x) / (n - 1)
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = 6.0 * (float(((jbar - jack) ** 2).sum()) ** 1.5)
    a = num / den if den != 0 else 0.0
    out, clamped = {}, False
    for name, p in (("lo", alpha / 2.0), ("hi", 1.0 - alpha / 2.0)):
        zp = ppf(p)
        den = 1.0 - a * (z0 + zp)
        # ⚠ if this clamp fires it turns a non-positive denominator into a huge
        # adjustment, i.e. the endpoint becomes the most extreme bootstrap value.
        # Conservative, but it must never be silent.
        if den <= 1e-12:
            clamped = True
        adj = z0 + (z0 + zp) / max(1e-12, den)
        out[name] = float(np.quantile(boots, min(max(cdf(adj), 0.0), 1.0)))
    return {"mean": theta, "lo": out["lo"], "hi": out["hi"], "n": int(n),
            "n_dropped_nonfinite": n_dropped, "acceleration_clamp_fired": bool(clamped),
            "z0": float(z0), "a": float(a), "n_boot": int(n_boot)}


def bca_equal_weight(M: np.ndarray, n_boot: int = 10000, seed: int = 20260917,
                     alpha: float = 0.05) -> dict:
    """BCa for **the unweighted mean of the per-deletion means**.

    ⚠ `np.nanmean(M, axis=0)` — averaging each request over however many deletions
    it survives — does NOT weight the deletions equally, because a request is
    excluded from a deletion that names it and the six deletions remove different
    fractions of the request set.  The largest deletion (`shape=2`, a quarter of the
    pool, and the one most likely to show collateral) is exactly the one most
    down-weighted.  This statistic averages **within** each deletion first and then
    **across** the six, so each deletion carries 1/6 by construction.

    `M` is `(D, n_req)` with NaN where the request names the deleted value.  The
    bootstrap resamples REQUESTS (shared across deletions, so the pairing survives).
    """
    M = np.asarray(M, np.float64)
    V = np.isfinite(M).astype(np.float64)          # (D, n)
    Z = np.where(np.isfinite(M), M, 0.0)           # (D, n)
    keep = V.sum(axis=1) > 0
    Z, V = Z[keep], V[keep]
    D, n = Z.shape
    if D == 0 or n < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_requests": int(n), "n_deletions": int(D), "note": "insufficient data"}
    theta = float(np.mean(Z.sum(1) / np.maximum(V.sum(1), 1.0)))
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), np.float64)
    step = 1000                                    # chunked: (chunk, n) counts matrix
    done = 0
    while done < n_boot:
        k = min(step, n_boot - done)
        C = rng.multinomial(n, np.full(n, 1.0 / n), size=k).astype(np.float64)  # (k, n)
        s, c = Z @ C.T, V @ C.T                    # (D, k)
        boots[done:done + k] = np.mean(s / np.maximum(c, 1.0), axis=0)
        done += k
    # jackknife over requests, closed form (no re-reduction per leave-out)
    sums, cnts = Z.sum(1), V.sum(1)                # (D,)
    num = sums[:, None] - Z                        # (D, n)
    den = np.maximum(cnts[:, None] - V, 1.0)
    jack = np.mean(num / den, axis=0)              # (n,)
    from math import erf, sqrt

    def ppf(p):
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def cdf(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    prop = float((boots < theta).mean())
    prop = min(max(prop, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    z0 = ppf(prop)
    jbar = jack.mean()
    dd = jbar - jack
    den_a = 6.0 * (float((dd ** 2).sum()) ** 1.5)
    a = float((dd ** 3).sum()) / den_a if den_a != 0 else 0.0
    out, clamped = {}, False
    for name, p in (("lo", alpha / 2.0), ("hi", 1.0 - alpha / 2.0)):
        zp = ppf(p)
        dn = 1.0 - a * (z0 + zp)
        if dn <= 1e-12:
            clamped = True
        adj = z0 + (z0 + zp) / max(1e-12, dn)
        out[name] = float(np.quantile(boots, min(max(cdf(adj), 0.0), 1.0)))
    return {"mean": theta, "lo": out["lo"], "hi": out["hi"],
            "per_deletion_means": (Z.sum(1) / np.maximum(V.sum(1), 1.0)).tolist(),
            "n_requests": int(n), "n_deletions": int(D),
            "requests_per_deletion": V.sum(1).astype(int).tolist(),
            "acceleration_clamp_fired": bool(clamped),
            "z0": float(z0), "a": float(a), "n_boot": int(n_boot)}


def verdict(mean: float, lo: float, hi: float, m: float = M_TIE) -> str:
    """R1, verbatim: win / tie / loss."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "UNDETERMINED"
    if abs(mean) <= m:
        return "tie"
    if lo > 0.0:
        return "win"
    if hi < 0.0 and mean < -m:
        return "loss"
    return "tie"  # CI includes 0


# ---------------------------------------------------------------------------


def _kappa(ret: float, null: float) -> float:
    """Fraction of the AVAILABLE persistence achieved: (ret - null) / (1 - null).

    ⭐ The headroom-normalised excess — Cohen's kappa for pre/post correctness
    agreement with independence at the post marginal as the chance model.  Free of
    BOTH artefacts: the level effect that makes raw retention favour a strong arm,
    and the ceiling effect that makes ratio-to-null favour a WEAK one.
    ⚠ Undefined when `null == 1` (no headroom); returns nan, which is itself the
    finding rather than a hole in the table.
    """
    import math
    if not (null < 1.0):
        return float("nan")
    return (ret - null) / (1.0 - null)


def per_request_agg(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean-over-factors correctness, one scalar per request (PREREG §7)."""
    return (np.asarray(preds) == np.asarray(labels)).mean(axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="dev", help="dev (DEV) or b (the scored held-out set)")
    args = ap.parse_args(argv)
    runs = Path(args.runs)
    S = args.set
    R = {}
    # ⭐ the extra replicates that turn every deletion size's churn band from a single
    # draw into >= 3 measured draws (LOG 11:28Z).
    EXTRA_PLC = ["plc_n2142_r2", "plc_n2142_r3", "plc_n4095_r2", "plc_n4095_r3",
                 "plc_n852_r2", "plc_n852_r3"]
    for t in [BASE, "det_A", "det_C", CONJ, CONJ_PLC,
              "del_seq_f0v5_f1v5", "del_union_f0v5_f1v5"] + CHURN + EXTRA_PLC + \
             [f"del_f{f}v{v}" for f, v in DELETIONS] + list(PLACEBO.values()):
        R[t] = load(runs, t)

    rep: dict = {"schema": "chlu-hub-deletion/analysis/1", "set": S, "m_tie": M_TIE,
                 "missing": [t for t, v in R.items() if v is None]}

    # ---- §4a DETERMINISM (blocking) ---------------------------------------
    dig = {t: (R[t]["build"]["digest"] if R[t] else None) for t in ("det_A", "det_B", "det_C")}
    rep["determinism"] = {
        "digests": {t: (d or {}).get("store_digest") for t, d in dig.items()},
        "param_hashes": {t: (d or {}).get("param_hash") for t, d in dig.items()},
        "book_hashes": {t: (d or {}).get("book_hash") for t, d in dig.items()},
        "live_sha256": {t: (R[t]["build"]["live_sha256"] if R[t] else None)
                        for t in ("det_A", "det_B", "det_C")},
        "batching": {t: (R[t]["build"]["build_inputs"] if R[t] else None)
                     for t in ("det_B", "det_C")},
    }
    # ⭐ The gate is `det_A` vs `det_B`: two full-pool builds from an IDENTICAL live
    # set, made by two INDEPENDENT PROCESSES (det_A by the sibling thread 11:03-11:13Z,
    # det_B by mine) — which is exactly the two-build comparison the falsifier asks
    # for.  `det_C` is a third build, run ONLY if A and B disagree, to arbitrate.
    a, b, c = dig["det_A"], dig["det_B"], dig["det_C"]
    eq = lambda x, y: bool(x and y and x["store_digest"] == y["store_digest"])
    rep["determinism"]["A_equals_B"] = eq(a, b)
    rep["determinism"]["B_equals_C"] = eq(b, c)
    rep["determinism"]["live_sets_identical"] = bool(
        R["det_A"] and R["det_B"]
        and R["det_A"]["build"]["live_sha256"] == R["det_B"]["build"]["live_sha256"])
    # ⛔ PASS requires the live sets to have been identical too: two builds that agree
    # on different inputs would be a bug, not determinism.
    rep["determinism"]["PASS"] = bool(rep["determinism"]["A_equals_B"]
                                      and rep["determinism"]["live_sets_identical"])
    rep["determinism"]["gate_pair"] = ["det_A", "det_B"]

    # ---- §4b PROVENANCE + §4c PATH-INDEPENDENCE ---------------------------
    prov = {}
    for f, v in DELETIONS:
        t = f"del_f{f}v{v}"
        if R[t]:
            prov[t] = {"n_live": R[t]["build"]["n_live"], "n_deleted": R[t]["build"]["n_deleted"],
                       "digest": R[t]["build"]["digest"]["store_digest"],
                       "live_sha256": R[t]["build"]["live_sha256"],
                       "rebuild_wall_s": R[t]["build"]["timing_s"]["total"]}
    rep["provenance"] = prov
    sq, un = R.get("del_seq_f0v5_f1v5"), R.get("del_union_f0v5_f1v5")
    if sq and un:
        rep["path_independence"] = {
            "sequence_digest": sq["build"]["digest"]["store_digest"],
            "union_digest": un["build"]["digest"]["store_digest"],
            "live_equal": sq["build"]["live_sha256"] == un["build"]["live_sha256"],
            "PASS": sq["build"]["digest"]["store_digest"] == un["build"]["digest"]["store_digest"],
        }

    # ---- §5 pre-deletion control · §6 unrecoverable ------------------------
    base = R[BASE]
    cond = {}
    if base and "preds" in base:
        for f, v in DELETIONS:
            t, tgt = f"del_f{f}v{v}", f"tgt_f{f}v{v}"
            row = {"factor": FACTORS[f], "value": int(v), "chance": 1.0 / CARDS[f]}
            kb = f"{tgt}_clu_preds"
            if kb in base["preds"].files:
                lb = base["preds"][f"{tgt}_labels"]
                pb = base["preds"][kb]
                n_t = int(lb.shape[0])
                acc_pre = float((pb[:, f] == lb[:, f]).mean())
                row["pre_deletion_acc_on_targeted"] = acc_pre
                row["n_targeted"] = n_t
                # ---- PREREG Amendment A2.4: the bar, two clauses ----------------
                # 1. above chance, one-sided exact binomial at alpha = 0.05
                k_succ = int((pb[:, f] == lb[:, f]).sum())
                p0 = 1.0 / CARDS[f]
                # survival P[X >= k] under Binomial(n, p0), via the regularised
                # incomplete beta identity -> no scipy dependency
                from math import lgamma, log, exp

                def _binom_sf(k, n, p):
                    if k <= 0:
                        return 1.0
                    tot = 0.0
                    for i in range(int(k), int(n) + 1):
                        lt = (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                              + i * log(p) + (n - i) * log(1.0 - p))
                        tot += exp(lt)
                        if lt < -60 and i > k + 5:
                            break
                    return min(1.0, tot)

                pval = _binom_sf(k_succ, n_t, p0)
                row["chance"] = p0
                row["binomial_p_one_sided"] = float(pval)
                row["clause1_above_chance"] = bool(pval < 0.05)
                # 2. ⭐ above the ARM'S OWN NULL: the laundering floor on the SAME rows
                #    and the SAME codes.  "the store supplied this factor" MEANS
                #    "better than the same read with the store's contents removed".
                kf = f"{tgt}_store_removed_preds"
                if kf in base["preds"].files:
                    floor_acc = float((base["preds"][kf][:, f] == lb[:, f]).mean())
                    row["pre_deletion_laundering_floor"] = floor_acc
                    row["clause2_above_own_null"] = bool(acc_pre > floor_acc)
                else:
                    row["pre_deletion_laundering_floor"] = None
                    row["clause2_above_own_null"] = None
                row["pre_deletion_control_PASS"] = bool(
                    row["clause1_above_chance"] and bool(row.get("clause2_above_own_null")))
                # A2.6 — the confusion row, so NOT MEASURABLE is never a bare hole
                row["pre_deletion_renders_instead"] = {
                    int(u): int(c) for u, c in
                    zip(*np.unique(pb[:, f], return_counts=True))}
                # ⭐ A3 — the distance profile, and the MARGINAL null.
                row["pre_distance_profile"] = distance_profile(pb[:, f], v, f)
                # ⚠ the right null for a single-value targeted set is not chance but
                # the store's OWN marginal rate of rendering v on a diverse set: if it
                # simply over-renders v, `acc_pre` is BIAS, not steering.
                if f"{S}_clu_preds" in base["preds"].files:
                    marg = float((base["preds"][f"{S}_clu_preds"][:, f] == v).mean())
                    base_rate = float((base["preds"][f"{S}_labels"][:, f] == v).mean())
                    row["marginal_render_rate_of_v"] = marg
                    row["base_rate_of_v_in_dev"] = base_rate
                    row["render_bias"] = marg / max(1e-9, base_rate)
                    row["steer_over_marginal"] = acc_pre / max(1e-9, marg)
            if R[t] and "preds" in R[t] and kb in R[t]["preds"].files:
                la = R[t]["preds"][f"{tgt}_labels"]
                pa = R[t]["preds"][kb]
                row["post_deletion_acc_on_targeted"] = float((pa[:, f] == la[:, f]).mean())
                row["bar_6a"] = 1.0 / CARDS[f] + 0.05
                row["cond_6a_unrecoverable_PASS"] = bool(
                    row["post_deletion_acc_on_targeted"] <= row["bar_6a"])
                # ⚠ 6b on the TARGETED set would be arithmetically identical to 6a
                # (every targeted label IS v), so it is evaluated over the WHOLE dev
                # set instead: does `v` appear in ANY render, for any request?
                # ⭐ That is strictly STRONGER than the targeted reading, not weaker.
                if f"{S}_clu_preds" in R[t]["preds"].files:
                    row["post_frac_rendering_v_anywhere"] = float(
                        (R[t]["preds"][f"{S}_clu_preds"][:, f] == v).mean())
                    row["pre_frac_rendering_v_anywhere"] = float(
                        (base["preds"][f"{S}_clu_preds"][:, f] == v).mean())
                    row["bar_6b"] = 1.0 / CARDS[f] + 0.05
                    row["cond_6b_no_render_PASS"] = bool(
                        row["post_frac_rendering_v_anywhere"] <= row["bar_6b"])
                # A2.6 — what it renders INSTEAD of v, after the deletion
                row["post_deletion_renders_instead"] = {
                    int(u): int(c) for u, c in
                    zip(*np.unique(pa[:, f], return_counts=True))}
                # ⭐⭐ A3 — THE DISCRIMINATING STATISTIC. Three pre-registered outcomes:
                #   mass at d0 collapses -> d1, rest ~unchanged  = structural deletion
                #   mass STAYS at d0                              = re-composed from parts
                #   mass spreads to d2+                           = damage, not deletion
                # ⛔ An exact-match threshold cannot tell the first from a one-step shift.
                row["post_distance_profile"] = distance_profile(pa[:, f], v, f)
                pre_p, post_p = row.get("pre_distance_profile", {}), row["post_distance_profile"]
                if pre_p.get("geometry") != "categorical":
                    d0_pre, d0_post = pre_p.get("d0"), post_p.get("d0")
                    d2p_pre = (pre_p.get("d2", 0.0) or 0.0) + (pre_p.get("d3plus", 0.0) or 0.0)
                    d2p_post = (post_p.get("d2", 0.0) or 0.0) + (post_p.get("d3plus", 0.0) or 0.0)
                    row["A3_verdict"] = (
                        "RE-COMPOSED (mass stayed at d0)" if d0_post > 0.5 * d0_pre else
                        "DAMAGE (mass spread to d2+)" if d2p_post > d2p_pre + 0.10 else
                        "STRUCTURAL DELETION (d0 -> d1, tail ~unchanged)")
                    row["A3_d0_pre"], row["A3_d0_post"] = d0_pre, d0_post
                    row["A3_tail_pre"], row["A3_tail_post"] = d2p_pre, d2p_post
                # ⭐⭐ A3.7 — THE CHURN CONTROL FOR THE DISTANCE PROFILE.
                # A rebuild moves the profile on its own, so the decisive quantity is
                # d0(deletion) - d0(size-matched PLACEBO), ⛔ not d0(deletion) - d0(base).
                # Same defect, same fix, as for collateral: without the placebo this
                # would be reporting the rebuild rather than the deletion. The placebo
                # scores the IDENTICAL targeted set, so it is free.
                pt = PLACEBO.get(t)
                if pt and R.get(pt) and "preds" in R[pt] and kb in R[pt]["preds"].files:
                    pp = R[pt]["preds"][kb]
                    row["placebo_distance_profile"] = distance_profile(pp[:, f], v, f)
                    row["placebo_acc_on_targeted"] = float((pp[:, f] == la[:, f]).mean())
                    if GEOMETRY[FACTORS[f]] != "categorical":
                        d0_plc = row["placebo_distance_profile"]["d0"]
                        row["A3_d0_placebo"] = d0_plc
                        row["A3_d0_del_minus_placebo"] = d0_post - d0_plc
                        tail_plc = (row["placebo_distance_profile"]["d2"]
                                    + row["placebo_distance_profile"]["d3plus"])
                        row["A3_tail_placebo"] = tail_plc
                        row["A3_tail_del_minus_placebo"] = d2p_post - tail_plc
                        # the churn-corrected three-way verdict (⭐ the reported one)
                        row["A3_verdict_churn_corrected"] = (
                            "RE-COMPOSED (d0 no lower than the placebo's)"
                            if d0_post >= d0_plc - 0.05 else
                            "DAMAGE (tail grew beyond the placebo's)"
                            if (d2p_post - tail_plc) > 0.10 else
                            "STRUCTURAL DELETION (d0 fell below the churn floor)")
            # the rival, same question
            if R[t] and "rpreds" in R[t] and f"{tgt}_knn_proto_preds" in R[t]["rpreds"].files:
                la = R[t]["rpreds"][f"{tgt}_labels"]
                pa = R[t]["rpreds"][f"{tgt}_knn_proto_preds"]
                row["knn_post_deletion_acc_on_targeted"] = float((pa[:, f] == la[:, f]).mean())
            if base and "rpreds" in base and f"{tgt}_knn_proto_preds" in base["rpreds"].files:
                lb = base["rpreds"][f"{tgt}_labels"]
                pb = base["rpreds"][f"{tgt}_knn_proto_preds"]
                row["knn_pre_deletion_acc_on_targeted"] = float((pb[:, f] == lb[:, f]).mean())
            cond[t] = row
    rep["conditions"] = cond

    # ---- §7 COLLATERAL — the headline --------------------------------------
    # ⭐ SECONDARY, declared 11:20Z before any deletion build had finished (LOG):
    # the CHURN-CORRECTED delta.  A rebuild moves the numbers for two reasons — the
    # rows that went, and the fact that `jax.random.choice(key, n, ...)` re-randomises
    # the whole batch sequence when `n` changes.  The size-matched placebo measures
    # the second alone, so subtracting it isolates the SEMANTICS of the deletion:
    #   delta_churn_corrected = [C_clu(del) - C_clu(plc)] - [C_knn(del) - C_knn(plc)]
    # ⛔ Reported BESIDE the PREREG §7 primary, never instead of it.
    coll = {}
    if base and "preds" in base and "rpreds" in base and f"{S}_clu_preds" in base["preds"].files:
        lab = base["preds"][f"{S}_labels"]
        base_clu = per_request_agg(base["preds"][f"{S}_clu_preds"], lab)
        base_knn = per_request_agg(base["rpreds"][f"{S}_knn_proto_preds"], lab)
        n_req = lab.shape[0]
        acc_delta = np.full((len(DELETIONS), n_req), np.nan)
        acc_clu_d = np.full((len(DELETIONS), n_req), np.nan)
        acc_knn_d = np.full((len(DELETIONS), n_req), np.nan)
        for di, (f, v) in enumerate(DELETIONS):
            t = f"del_f{f}v{v}"
            if not (R[t] and "preds" in R[t] and "rpreds" in R[t]):
                continue
            if f"{S}_clu_preds" not in R[t]["preds"].files:
                continue
            keep = lab[:, f] != v            # ⭐ requests that do NOT name v
            a_clu = per_request_agg(R[t]["preds"][f"{S}_clu_preds"], lab)
            a_knn = per_request_agg(R[t]["rpreds"][f"{S}_knn_proto_preds"], lab)
            d_clu = np.where(keep, a_clu - base_clu, np.nan)
            d_knn = np.where(keep, a_knn - base_knn, np.nan)
            acc_clu_d[di], acc_knn_d[di], acc_delta[di] = d_clu, d_knn, d_clu - d_knn
            ent = {"factor": FACTORS[f], "value": int(v), "n_non_v": int(keep.sum()),
                   "C_clu": float(np.nanmean(d_clu)), "C_knn_proto": float(np.nanmean(d_knn)),
                   "delta": float(np.nanmean(d_clu - d_knn)),
                   "clu_before": float(base_clu[keep].mean()),
                   "clu_after": float(a_clu[keep].mean()),
                   "knn_before": float(base_knn[keep].mean()),
                   "knn_after": float(a_knn[keep].mean())}
            # relative change, because a floor effect must not read as a win (LOG 11:17Z)
            ent["C_clu_relative"] = ent["C_clu"] / max(1e-9, ent["clu_before"])
            ent["C_knn_relative"] = ent["C_knn_proto"] / max(1e-9, ent["knn_before"])
            ent["delta_relative"] = ent["C_clu_relative"] - ent["C_knn_relative"]
            # the placebo / churn floor for the same size
            p = PLACEBO.get(t)
            if p and R.get(p) and "preds" in R[p] and f"{S}_clu_preds" in R[p]["preds"].files:
                pc = per_request_agg(R[p]["preds"][f"{S}_clu_preds"], lab)
                pk = per_request_agg(R[p]["rpreds"][f"{S}_knn_proto_preds"], lab)
                ent["placebo"] = {
                    "tag": p, "n_deleted": R[p]["build"]["n_deleted"],
                    "C_clu": float((pc - base_clu)[keep].mean()),
                    "C_knn_proto": float((pk - base_knn)[keep].mean()),
                    "delta": float(((pc - base_clu) - (pk - base_knn))[keep].mean()),
                }
                # ⭐ CHURN-CORRECTED (the declared secondary): subtract the placebo, so
                # what remains is the SEMANTICS of the deletion rather than the fact
                # that a rebuild re-randomises everything.  ⛔ Beside the primary.
                ent["delta_churn_corrected"] = float(
                    ((d_clu - (pc - base_clu)) - (d_knn - (pk - base_knn)))[keep].mean())
            coll[t] = ent
        # pooled: mean over deletions per request, then bootstrap over requests
        with np.errstate(invalid="ignore"):
            per_req_delta = np.nanmean(acc_delta, axis=0)
            per_req_clu = np.nanmean(acc_clu_d, axis=0)
            per_req_knn = np.nanmean(acc_knn_d, axis=0)
        rep["collateral_pooled"] = {
            "delta": bca(per_req_delta), "C_clu": bca(per_req_clu),
            "C_knn_proto": bca(per_req_knn),
            "n_deletions_included": int(np.isfinite(acc_delta).any(axis=1).sum()),
            "note": ("⚠ request-averaged: a request excluded from a deletion that names "
                     "it makes the six deletions UNEQUALLY weighted. See "
                     "`collateral_pooled_equal_weight`, which is the reported primary."),
        }
        d = rep["collateral_pooled"]["delta"]
        rep["collateral_pooled"]["VERDICT_R1"] = verdict(d["mean"], d["lo"], d["hi"])
        # ⭐ PRIMARY (PREREG Amendment A2): each deletion carries exactly 1/6.
        ew = bca_equal_weight(acc_delta)
        rep["collateral_pooled_equal_weight"] = {
            "delta": ew, "C_clu": bca_equal_weight(acc_clu_d),
            "C_knn_proto": bca_equal_weight(acc_knn_d),
            "VERDICT_R1": verdict(ew["mean"], ew["lo"], ew["hi"]),
            "realised_weights_of_request_averaged_version": (
                (np.isfinite(acc_delta).sum(1) / max(1, np.isfinite(acc_delta).sum())).tolist()),
        }
        # ⭐ CO-PRIMARY (A2): churn-corrected — semantic deletion minus a size-matched
        # random deletion, which is the only version that charges the CLU's fresh
        # 1200-step optimisation to churn rather than to the deletion.
        cc = np.full_like(acc_delta, np.nan)
        for di, (f, v) in enumerate(DELETIONS):
            t, p = f"del_f{f}v{v}", PLACEBO.get(f"del_f{f}v{v}")
            if not (R.get(t) and R.get(p) and "preds" in R[p]
                    and f"{S}_clu_preds" in R[p]["preds"].files
                    and np.isfinite(acc_delta[di]).any()):
                continue
            keep = lab[:, f] != v
            pc = per_request_agg(R[p]["preds"][f"{S}_clu_preds"], lab) - base_clu
            pk = per_request_agg(R[p]["rpreds"][f"{S}_knn_proto_preds"], lab) - base_knn
            cc[di] = np.where(keep, (acc_clu_d[di] - pc) - (acc_knn_d[di] - pk), np.nan)
        if np.isfinite(cc).any():
            ccb = bca_equal_weight(cc)
            rep["collateral_churn_corrected"] = {
                "delta": ccb, "VERDICT_R1": verdict(ccb["mean"], ccb["lo"], ccb["hi"]),
                "what": ("[C_clu(del) - C_clu(placebo)] - [C_knn(del) - C_knn(placebo)], "
                         "size-matched. Answers: does deleting a MEANING cost more than "
                         "deleting the same number of random rows?"),
            }

        # ---- ⭐ RETENTION, the unlearning literature's estimand ----------------
        # Of the (request, factor) cells an arm got RIGHT before the deletion, what
        # fraction does it still get right after?  ⭐ Each arm is scored only on its
        # OWN competence set, so the 13-point level gap between the arms cancels BY
        # CONSTRUCTION and the statistic is bounded in [0, 1].  This is the estimand
        # that is immune to the "the CLU had less to lose" objection, which neither
        # the raw nor the relative view fully answers.
        ret = {}
        bc_cell = (base["preds"][f"{S}_clu_preds"] == lab)
        bk_cell = (base["rpreds"][f"{S}_knn_proto_preds"] == lab)
        r_clu, r_knn = [], []
        for f, v in DELETIONS:
            t = f"del_f{f}v{v}"
            if not (R.get(t) and "preds" in R[t] and f"{S}_clu_preds" in R[t]["preds"].files):
                continue
            keep = lab[:, f] != v
            ac = (R[t]["preds"][f"{S}_clu_preds"] == lab)[keep]
            ak = (R[t]["rpreds"][f"{S}_knn_proto_preds"] == lab)[keep]
            bc, bk = bc_cell[keep], bk_cell[keep]
            rc = float(ac[bc].mean()) if bc.any() else float("nan")
            rk = float(ak[bk].mean()) if bk.any() else float("nan")
            # ⭐⭐ LOST / GAINED — what the net aggregate HIDES.
            # A small net delta can mean "the store barely moved" or "it churned
            # enormously and the churn happened to be symmetric".  Measured on
            # del_f2v5 those are 1-in-5 vs 1-in-90 for the two arms, so the
            # distinction is not hypothetical.  This is the per-cell analogue of the
            # ARI partition result: the layout moved, and so did the answers.
            lost_c = float((bc & ~ac).mean()); gain_c = float((~bc & ac).mean())
            lost_k = float((bk & ~ak).mean()); gain_k = float((~bk & ak).mean())
            # ⚠ THE INDEPENDENCE NULL.  Retention is NOT bounded below by anything
            # near 1 for a weak predictor: if pre- and post-correctness were
            # independent, retention would equal the POST accuracy.  For factors where
            # an arm is weak (the CLU is at 0.211 on scale, 0.184 on orientation),
            # "right before" is substantially luck, so a retention well below 1 is
            # expected and the gap against a near-deterministic rival overstates the
            # loss.  ⛔ Retention is never reported without this null beside it.
            ret[t] = {"factor": FACTORS[f],
                      "retention_clu": rc, "retention_knn_proto": rk,
                      "delta_retention": rc - rk,
                      "independence_null_clu": float(ac.mean()),
                      "independence_null_knn_proto": float(ak.mean()),
                      # ⛔⛔ ratio-to-null (`ret/null`) was here and is REMOVED: it is
                      # ceiling-limited at 1/null, so a WEAK arm has room for a large
                      # ratio while a strong arm is capped near 1 — it systematically
                      # flatters the weaker arm, which here is the CLU.  Measured on the
                      # aggregate it REVERSES the conclusion: by ratio the CLU looks
                      # better (1.397 vs 1.355), by retention it is far worse (0.825 vs
                      # 0.993).  ⭐ Replaced by KAPPA = (ret - null) / (1 - null): the
                      # fraction of the AVAILABLE persistence achieved, free of both the
                      # level effect retention has and the ceiling effect the ratio has.
                      "kappa_clu": _kappa(rc, float(ac.mean())),
                      "kappa_knn_proto": _kappa(rk, float(ak.mean())),
                      "delta_kappa": (_kappa(rc, float(ac.mean()))
                                      - _kappa(rk, float(ak.mean())))
                      if np.isfinite(_kappa(rk, float(ak.mean()))) else float("nan"),
                      # ⚠ kappa is undefined when post accuracy is exactly 1.0 (no
                      # headroom).  For knn_proto on the hues that is not a gap in the
                      # table, it IS the finding: perfect and perfectly stable.
                      "kappa_degenerate_clu": bool(float(ac.mean()) > 0.99),
                      "kappa_degenerate_knn": bool(float(ak.mean()) > 0.99),
                      "clu_lost": lost_c, "clu_gained": gain_c,
                      "clu_answer_churn": lost_c + gain_c,
                      "knn_lost": lost_k, "knn_gained": gain_k,
                      "knn_answer_churn": lost_k + gain_k,
                      "n_cells_clu": int(bc.sum()), "n_cells_knn": int(bk.sum())}
            r_clu.append(rc)
            r_knn.append(rk)
        if r_clu:
            rep["retention"] = {
                "per_deletion": ret,
                "mean_retention_clu": float(np.nanmean(r_clu)),
                "mean_retention_knn_proto": float(np.nanmean(r_knn)),
                "mean_delta_retention": float(np.nanmean(np.array(r_clu) - np.array(r_knn))),
                "what": ("fraction of PRE-DELETION-CORRECT (request, factor) cells still "
                         "correct after; each arm scored on its own competence set, so "
                         "the level gap cancels."),
            }

        # ---- ⚠ THE TWO BIASES IN THE PRIMARY, measured, both reported ----------
        rep["primary_estimand_biases"] = {
            "level_gap_favours_clu": {
                "clu_before": float(base_clu.mean()), "knn_before": float(base_knn.mean()),
                "gap": float(base_clu.mean() - base_knn.mean()),
                "why": ("if damage is roughly proportional to level, the arm that starts "
                        "LOWER loses fewer raw accuracy points, so Delta reads positive "
                        "from the level gap alone. This favours the CLU."),
            },
            "rebuild_churn_favours_knn_proto": {
                "measured_by": "the size-matched placebo delta per deletion",
                "why": ("the CLU's 'after' is a fresh 1200-step optimisation on a "
                        "re-derived channel partition; knn_proto has NO optimiser and its "
                        "table is a closed-form mean. The uncorrected primary therefore "
                        "charges the CLU for rebuild churn as if it were deletion damage. "
                        "This favours knn_proto."),
            },
            "resolution": ("⛔ Neither bias is argued about: the churn term IS "
                           "`collateral_churn_corrected` and the level gap IS the numbers "
                           "above. Both are reported beside the primary, and `retention` "
                           "is the estimand in which the level gap cancels."),
        }
    rep["collateral"] = coll

    # ---- the conjunction (the gs charter's re-chartered target) -------------
    t = CONJ
    if base and R.get(t) and "preds" in R[t]:
        tgt = "tgt_conj_f2v5_f4v2"
        if f"{tgt}_clu_preds" in R[t]["preds"].files:
            lb = base["preds"][f"{tgt}_labels"]
            pb, pa = base["preds"][f"{tgt}_clu_preds"], R[t]["preds"][f"{tgt}_clu_preds"]
            both_pre = ((pb[:, 2] == 5) & (pb[:, 4] == 2)).mean()
            both_post = ((pa[:, 2] == 5) & (pa[:, 4] == 2)).mean()
            rep["conjunction"] = {
                "deleted": "object_hue=5 AND shape=2",
                "n_targeted": int(lb.shape[0]),
                "pre_frac_render_BOTH": float(both_pre),
                "post_frac_render_BOTH": float(both_post),
                "pre_acc_object_hue": float((pb[:, 2] == lb[:, 2]).mean()),
                "post_acc_object_hue": float((pa[:, 2] == lb[:, 2]).mean()),
                "pre_acc_shape": float((pb[:, 4] == lb[:, 4]).mean()),
                "post_acc_shape": float((pa[:, 4] == lb[:, 4]).mean()),
                "note": ("⭐ the compositional question: after every (object_hue=5 AND "
                         "shape=2) row is gone, can the store still COMPOSE one from "
                         "the object_hue=5 rows and the shape=2 rows it still holds?"),
            }

    # ---- headline aggregates, for the report -------------------------------
    if base and "eval" in base and "rival" in base:
        rep["pre_deletion_levels"] = {
            "clu": base["eval"]["sets"][S]["reads"]["clu"]["aggregate"],
            "store_removed_floor": base["eval"]["sets"][S]["reads"]["store_removed"]["aggregate"],
            "feature_addressed": base["eval"]["sets"][S]["reads"]["feature_addr"]["aggregate"],
            "knn_proto": base["rival"]["sets"][S]["aggregate"],
            "clu_store_bytes": 524288, "knn_proto_bytes": 475136,
        }
    # ---- ⭐ THE CHURN BANDS, one per deletion SIZE, all MEASURED ---------------
    # ⚠ A band built at n=8234 is the wrong ruler for a 3,207-row deletion: churn
    # scales with how much the live set changed.  As first planned, five of the six
    # deletions had a "band" of ONE draw — a point with no spread.  Every placebo in
    # the cell is therefore grouped by its realised `n_deleted` (3 % tolerance, which
    # absorbs the three hue placebos' 62-row / 1.92 % spread) so each deletion is read
    # against replicates at ITS OWN size.  ⛔ No band here is modelled or extrapolated.
    if base and "preds" in base and f"{S}_clu_preds" in base["preds"].files:
        lab0 = base["preds"][f"{S}_labels"]
        bc0 = per_request_agg(base["preds"][f"{S}_clu_preds"], lab0)
        bk0 = per_request_agg(base["rpreds"][f"{S}_knn_proto_preds"], lab0)
        plc = []
        for t, u in R.items():
            if not (t.startswith("plc") and u and "preds" in u and "rpreds" in u):
                continue
            if f"{S}_clu_preds" not in u["preds"].files:
                continue
            pc = per_request_agg(u["preds"][f"{S}_clu_preds"], lab0)
            pk = per_request_agg(u["rpreds"][f"{S}_knn_proto_preds"], lab0)
            plc.append({"tag": t, "n_deleted": int(u["build"]["n_deleted"]),
                        "C_clu": float((pc - bc0).mean()),
                        "C_knn_proto": float((pk - bk0).mean()),
                        "delta": float(((pc - bc0) - (pk - bk0)).mean())})
        groups = []
        for e_ in sorted(plc, key=lambda x: x["n_deleted"]):
            for g in groups:
                if abs(e_["n_deleted"] - g["mean_n"]) / max(1.0, g["mean_n"]) <= 0.03:
                    g["members"].append(e_)
                    g["mean_n"] = float(np.mean([m["n_deleted"] for m in g["members"]]))
                    break
            else:
                groups.append({"members": [e_], "mean_n": float(e_["n_deleted"])})
        for g in groups:
            for k in ("C_clu", "C_knn_proto", "delta"):
                vals = [m[k] for m in g["members"]]
                g[k] = {"min": float(min(vals)), "max": float(max(vals)),
                        "mean": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
            g["n_replicates"] = len(g["members"])
            g["size_spread_frac"] = float(
                (max(m["n_deleted"] for m in g["members"])
                 - min(m["n_deleted"] for m in g["members"])) / max(1.0, g["mean_n"]))
        rep["churn_bands_by_size"] = groups
        # attach the matching band to each deletion, and answer the band question
        for t, ent in coll.items():
            nd = R[t]["build"]["n_deleted"] if R.get(t) else None
            if nd is None:
                continue
            best = min(groups, key=lambda g: abs(g["mean_n"] - nd)) if groups else None
            if not best or abs(best["mean_n"] - nd) / max(1.0, nd) > 0.05:
                ent["band"] = {"status": "NO MATCHED BAND", "n_deleted": nd}
                continue
            ent["band"] = {
                "matched_mean_n": best["mean_n"], "n_replicates": best["n_replicates"],
                "C_clu_band": best["C_clu"], "delta_band": best["delta"],
                # ⭐ the claim about the CLU ALONE: did deleting do anything beyond rebuilding?
                "C_clu_inside_band": bool(best["C_clu"]["min"] <= ent["C_clu"]
                                          <= best["C_clu"]["max"]),
                "delta_inside_band": bool(best["delta"]["min"] <= ent["delta"]
                                          <= best["delta"]["max"]),
                "note": ("⛔ `C_clu inside band` and `Delta > 0` are DIFFERENT claims and can "
                         "disagree: the CLU can be damaged more than random removal would "
                         "damage it while still being damaged less than knn_proto. Without "
                         "the band a raw C_clu is reporting the rebuild, not the deletion."),
            }

    # ---- (legacy) the fixed n=8234 replicate set, kept for continuity ----------
    # PREREG §7b needs a band, and one placebo cannot give one.  ⭐ This is the
    # honest substitute for the charter's "within pre-deletion seed spread", which is
    # unavailable because §1c forbids spending seeds 1 and 2 on this box.
    if base and "preds" in base and f"{S}_clu_preds" in base["preds"].files:
        lab = base["preds"][f"{S}_labels"]
        bc = per_request_agg(base["preds"][f"{S}_clu_preds"], lab)
        bk = per_request_agg(base["rpreds"][f"{S}_knn_proto_preds"], lab)
        band = []
        for p in CHURN:
            if R.get(p) and "preds" in R[p] and f"{S}_clu_preds" in R[p]["preds"].files:
                pc = per_request_agg(R[p]["preds"][f"{S}_clu_preds"], lab)
                pk = per_request_agg(R[p]["rpreds"][f"{S}_knn_proto_preds"], lab)
                band.append({"tag": p, "C_clu": float((pc - bc).mean()),
                             "C_knn_proto": float((pk - bk).mean()),
                             "delta": float(((pc - bc) - (pk - bk)).mean())})
        if band:
            rep["churn_band"] = {
                "replicates": band, "n_deleted_each": 8234,
                "C_clu": {"min": min(x["C_clu"] for x in band),
                          "max": max(x["C_clu"] for x in band),
                          "mean": float(np.mean([x["C_clu"] for x in band]))},
                "delta": {"min": min(x["delta"] for x in band),
                          "max": max(x["delta"] for x in band),
                          "mean": float(np.mean([x["delta"] for x in band]))},
                "note": ("⭐ pure rebuild churn at the largest deletion size. A semantic "
                         "deletion whose collateral sits INSIDE this band has not been "
                         "shown to cost anything beyond re-randomising the build."),
            }

    # ---- ⭐⭐ HOW FAR THE CHANNEL PARTITION MOVED — underneath every accuracy ----
    # PREREG A3.12.  AUDIT-01 predicted this and `del_f2v5` realised it: `n_corr =
    # 16384` means the correlation matrix is fitted on the first 16,384 rows of the
    # SORTED live vector, so removing rows shifts that window and `ward` can return a
    # DIFFERENT partition.  Measured on del_f2v5: well sizes [303,305,573,501,366] ->
    # [672,234,234,546,362], ARI 0.5656, best agreement under relabelling 0.8027.
    # ⛔ So a post-delete store is NOT a perturbation of the pre-delete store, and every
    # per-factor difference confounds "v was deleted" with "a fifth of the
    # representation was re-partitioned".  That is the mechanism working as specified —
    # a canonical function of the live set may return a different layout — but it means
    # the UNCORRECTED collateral is close to uninterpretable on its own.
    # ⭐ Free: straight out of modes.npz, no eval needed.
    from itertools import permutations
    try:
        from chlu.core.gen_store import adjusted_rand_index
    except Exception:
        adjusted_rand_index = None
    base_lab_path = tag_dir(runs, BASE) / "modes.npz"
    if adjusted_rand_index is not None and base_lab_path.exists():
        lref = np.load(base_lab_path)["labels"]
        part = {}
        for t, u in R.items():
            if not u:
                continue
            mp = tag_dir(runs, t) / "modes.npz"
            if not mp.exists():
                continue
            lt = np.load(mp)["labels"]
            if lt.shape != lref.shape:
                continue
            k = int(max(lref.max(), lt.max())) + 1
            best = max(float((np.array(p)[lt] == lref).mean())
                       for p in permutations(range(k))) if k <= 6 else None
            part[t] = {
                "n_deleted": int(u["build"]["n_deleted"]),
                "well_sizes": u["build"]["digest"]["well_sizes"],
                "ARI_vs_undeleted": float(adjusted_rand_index(lt, lref)),
                "best_agreement_under_relabelling": best,
                "n_channels_moved": (int(round((1.0 - best) * lt.size))
                                     if best is not None else None),
            }
        rep["partition_move"] = {
            "reference": BASE, "per_unit": part,
            "note": ("⭐ ARI = 1.0 between the three undeleted builds, so the metric is "
                     "sound. The decisive read is ARI(deletion) vs ARI(size-matched "
                     "placebo): comparable => partition churn is generic to rebuilding "
                     "at that size and the churn-corrected collateral is the only "
                     "meaningful number; placebo much higher => the STRUCTURED deletion "
                     "destabilises the layout in a way random removal does not, which "
                     "is a finding no accuracy number would have revealed."),
        }

    # ---- ⭐⭐ THE MECHANISM CHECK — no probe, no decoder, no accuracy ----------
    # PREREG A3.8. `fit_address_book` fits the deleted value's conditional row on ZERO
    # live rows, so Laplace smoothing gives (0+1)/(0+1*M) = 1/M for EVERY mode, i.e.
    # the row must be EXACTLY log(1/64) = -4.1588830833596715 with within-well range
    # exactly 0.0.  Calibrated on det_B, where every value's row has a within-well
    # log-range of ~5.4 and none is uniform.
    # ⭐ This is decisive in a way an accuracy is not:
    #   - if the row is NOT exactly uniform, the delete did not do what the mechanism
    #     says, whatever the accuracies show;
    #   - if it IS exactly uniform and the value still comes back, then the addressing
    #     term is PROVABLY gone and the value re-composed anyway — the strongest form
    #     of the re-composition finding.
    mech = {}
    for f, v in DELETIONS + [(2, 5)]:
        t = CONJ if (f, v) == (2, 5) and CONJ in mech else f"del_f{f}v{v}"
        if t in mech or not R.get(t):
            continue
        mz = tag_dir(runs, t) / "modes.npz"
        if not mz.exists():
            continue
        z = np.load(mz)
        key = f"cond_{f}"
        if key not in z.files:
            continue
        c = z[key]                       # (V, K, M)
        M = int(c.shape[2])
        ref32 = np.float32(np.log(1.0 / M))
        row = c[v]                       # (K, M)
        # ⚠ DEFINITION, stated because two readings differ by up to 1.3 nats and a
        # reader comparing runs would otherwise see a spurious change.  The reported
        # range is the MEAN OF THE PER-WELL RANGES, not the range pooled across wells:
        # the argmax the address book drives is taken WITHIN a well, so a cross-well
        # range mixes quantities that never compete.  (Pooled would read cond_0 6.58
        # where per-well reads 5.55.)  ⭐ The deleted row gives exactly 0 under either,
        # so A3.8's verdict does not depend on the choice.
        def _per_well_range(a):          # a: (K, M)
            return float(np.mean([a[k].max() - a[k].min() for k in range(a.shape[0])]))
        others = [_per_well_range(c[u]) for u in range(c.shape[0]) if u != v]
        mech[t] = {
            "factor": FACTORS[f], "deleted_value": int(v), "M": M,
            "expected_uniform_logprob": float(ref32),
            "deleted_row_exactly_uniform": bool(np.all(row == ref32)),
            "deleted_row_within_well_range": _per_well_range(row),
            "range_definition": "mean of the per-well ranges (argmax is taken within a well)",
            "deleted_row_max_abs_dev_from_uniform": float(
                np.max(np.abs(row.astype(np.float64) - float(np.log(1.0 / M))))),
            "surviving_rows_mean_range_per_well": float(np.mean(others)) if others else None,
            "surviving_rows_min_range_per_well": float(min(others)) if others else None,
            "note": ("⭐ probe-free and decoder-free. Pre-deletion every conditional row "
                     "has a within-well log-range ~5.4; a row fitted on zero live rows "
                     "must be exactly log(1/M) with range exactly 0."),
        }
    if mech:
        rep["mechanism_check"] = mech

    # ---- ⛔ CODE PROVENANCE: one fingerprint across the cell, or it is visible ---
    fps = {t: (R[t]["build"].get("code", {}) or {}).get("combined") for t in R if R[t]}
    seen = {}
    for t, fp in fps.items():
        seen.setdefault(fp, []).append(t)
    rep["code_provenance"] = {
        "per_unit": fps,
        "distinct_fingerprints": len(seen),
        "groups": {str(k): v for k, v in seen.items()},
        "ALL_UNITS_SAME_CODE": len(seen) == 1,
        "note": ("⚠ units built before 11:26Z carry `null` — the fingerprint was added "
                 "then, after the sibling thread pointed out that exp_gen_deletion.py "
                 "was edited at 11:20:14Z, AFTER det_A was built. det_C is built under "
                 "the current code so the delta is MEASURED inert, not asserted."),
    }

    rep["cost"] = {
        "rebuild_wall_s": {t: (R[t]["build"]["timing_s"]["total"] if R[t] else None)
                           for t in R if R[t]},
    }

    Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: rep[k] for k in ("determinism", "path_independence",
                                          "collateral_pooled", "conjunction", "missing")
                      if k in rep}, indent=2, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
