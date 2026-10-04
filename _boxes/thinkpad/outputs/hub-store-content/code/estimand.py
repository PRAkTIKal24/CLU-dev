"""hub/store-content — the MULTI-CORNER PAIRED ESTIMAND (w3-hub-common §3).

⛔ "A single held-out cell may not decide a target."  Wave 2 measured that the corner
explains **90.9 %** of the paired variance and the store seed 9.1 %, so a single
72-item cell cannot distinguish "won this cell" from "ties in expectation".

The verdict object is therefore, for a metric ``m`` and a pair (arm, control):

    Delta_c = m(arm, corner c) - m(control, corner c),  for each of the 49 corners

reported as the paired mean, its paired standard error, a paired 95 % interval, the
sign count (won / tied / lost) and a two-sided Wilcoxon signed-rank p over the 49
differences.  ⭐ The margin is declared in the PREREG **before** the run and is passed
in here; this module never chooses one.
"""
from __future__ import annotations

import numpy as np


def per_corner_metrics(pred: np.ndarray, target: np.ndarray, corner: np.ndarray,
                       factor_names, seen_tuples=None) -> dict:
    """Per-corner accuracy, per-factor accuracy and novelty counts.

    Args:
        pred: (n, F) tuples the external judge read back.
        target: (n, F) tuples the request asked for.
        corner: (n,) corner id per request.
        seen_tuples: set of tuples in the store's fit pool; enables the rows 3-4
            novelty metric (distinct valid NOVEL tuples).
    """
    pred = np.asarray(pred, np.int64)
    target = np.asarray(target, np.int64)
    corner = np.asarray(corner)
    ids = list(dict.fromkeys(corner.tolist()))
    hit = pred == target
    out = {}
    for cid in ids:
        m = corner == cid
        pf = {n: float(hit[m, i].mean()) for i, n in enumerate(factor_names)}
        rec = {"n": int(m.sum()), "aggregate": float(np.mean(list(pf.values()))),
               "exact_match": float(hit[m].all(axis=1).mean()), "per_factor": pf}
        if seen_tuples is not None:
            valid = hit[m].all(axis=1)
            produced = {tuple(int(x) for x in t) for t in pred[m][valid]}
            novel = produced - seen_tuples
            rec.update({"distinct_valid_tuples": len(produced),
                        "distinct_valid_novel_tuples": len(novel),
                        "replayed_seen_tuples": len(produced & seen_tuples),
                        "n_valid": int(valid.sum())})
        out[str(cid)] = rec
    return out


def _wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p, normal approximation with tie correction.

    ⛔ Implemented here rather than imported: scipy is not in this environment, and a
    verdict that depends on a package that is not installed is not a verdict.  The
    normal approximation is appropriate at n = 49 non-zero differences; with fewer
    than 6 it returns NaN rather than a number nobody should read.
    """
    d = np.asarray(d, np.float64)
    nz = d[d != 0.0]
    n = nz.size
    if n < 6:
        return float("nan")
    order = np.argsort(np.abs(nz), kind="stable")
    a = np.abs(nz)[order]
    ranks = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    signs = np.sign(nz)[order]
    w = float((ranks * (signs > 0)).sum())
    mu = n * (n + 1) / 4.0
    # tie correction on the |d| ties
    _, counts = np.unique(a, return_counts=True)
    tie = float((counts ** 3 - counts).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0:
        return float("nan")
    z = (w - mu) / np.sqrt(var)
    # two-sided normal tail without scipy
    from math import erfc, sqrt
    return float(erfc(abs(z) / sqrt(2.0)))


def paired_verdict(arm_corners: dict, ctl_corners: dict, key: str,
                   margin: float, arm_name: str, ctl_name: str,
                   factor: str | None = None) -> dict:
    """The paired verdict for one metric over the corners both arms answered."""
    ids = [c for c in arm_corners if c in ctl_corners]

    def get(rec):
        return float(rec["per_factor"][factor]) if factor else float(rec[key])

    d = np.array([get(arm_corners[c]) - get(ctl_corners[c]) for c in ids], np.float64)
    a = np.array([get(arm_corners[c]) for c in ids], np.float64)
    b = np.array([get(ctl_corners[c]) for c in ids], np.float64)
    n = d.size
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    lo, hi = (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)) if n > 1 else (
        float("nan"), float("nan"))
    beats = bool(n > 1 and d.mean() >= margin and lo > 0.0)
    return {
        "metric": factor or key, "arm": arm_name, "control": ctl_name,
        "n_corners": int(n),
        "arm_mean": float(a.mean()), "control_mean": float(b.mean()),
        "paired_mean_delta": float(d.mean()), "paired_se": se,
        "ci95": [lo, hi],
        "won": int((d > 0).sum()), "tied": int((d == 0).sum()), "lost": int((d < 0).sum()),
        "wilcoxon_p_two_sided": _wilcoxon_p(d),
        "declared_margin": float(margin),
        "verdict": ("beats the control beyond the declared margin" if beats else
                    ("⛔ A CLU mechanism that ties basis+head is decoration and is "
                     "reported as such" if ctl_name == "basis_head" and d.mean() >= 0
                     else "does NOT beat the control at the declared margin")),
        "per_corner_delta": [float(x) for x in d],
        "corner_ids": ids,
    }
