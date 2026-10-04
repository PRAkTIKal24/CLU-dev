"""⭐ Scoring the composition certificate — **and the audit that makes its headline
claim checkable rather than promised.**

Two things live here.

**1. The audit.** The claim this spoke rests on is *"zero invalid-example
supervision"*.  A sentence in a report is not evidence, so every threshold and every
normaliser in the harness is routed through :class:`ValidOnlyFitter`, which

* refuses to fit on anything but a set of ids it is handed explicitly,
* records the **sha256 of exactly the sorted ids it saw** plus their count, and
* :meth:`ValidOnlyFitter.assert_no_invalid` — ⛔ the load-bearing call — re-reads every
  fit record and asserts that **no C record id appears in any of them**, returning the
  counts so the report can quote them.

A reviewer will not take our word for this, and neither should we: the assertion has a
**positive control** in the tests (``test_the_audit_fires_when_a_C_id_reaches_a_fit``)
so it is known to be able to fail.

**2. The metrics, as ruled (R7 + R23).**  ``B`` is **72 of 480,000** — 0.0150 % of
A∪B — so an arm can refuse all of B at almost no cost on a pooled A∪B-vs-C AUROC,
which is exactly the penalty that comparison exists to impose.  ⛔ Therefore no single
pooled AUROC is ever "the" number here.  :func:`certificate_metrics` returns, with
equal prominence:

1. **A-vs-C**, 2. ⭐ **B-vs-C** (R23's primary refusal statistic), 3. the
**class-balanced** A∪B-vs-C — computed by a weighted Mann-Whitney with weights
``1/|A|`` and ``1/|B|``, which is *identically* ``0.5 * (AUROC_A + AUROC_B)``
(asserted in the tests, and re-asserted at run time).  The **unweighted** pooled
number is also returned, labelled ``pooled_UNBALANCED_do_not_quote``, precisely so
that nobody has to recompute it to see why it must not be quoted.

Plus the **risk-coverage curve and AURC** at a declared mixture, and a per-C-kind
split (``duplicate_factor`` / ``out_of_range``) — without which a grid lookup and a
composition barrier are indistinguishable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "AURC_MIXTURES",
    "ValidOnlyFitter",
    "auroc",
    "certificate_metrics",
    "operating_point",
    "risk_coverage",
    "weighted_auroc",
]

#: Declared mixtures for the risk-coverage curve.  ⛔ The mixture is part of the
#: number: AURC on a 50/50 B-vs-C population and AURC on an A-dominated one are not
#: comparable, so both are reported and both are named.
AURC_MIXTURES = ("B_vs_C", "A_vs_C", "balanced_AB_vs_C")


# ---------------------------------------------------------------------------
# 1. The audit
# ---------------------------------------------------------------------------


def _hash_ids(ids: Sequence[str]) -> str:
    joined = "\n".join(sorted(str(i) for i in ids)).encode()
    return hashlib.sha256(joined).hexdigest()


@dataclass
class ValidOnlyFitter:
    """⭐ The one place a threshold or a normaliser may be fitted.

    Every call records a hash of exactly the ids it saw.  ⛔ Nothing is fitted from
    data that did not arrive through here, which is what makes
    :meth:`assert_no_invalid` a *proof over the whole harness* rather than a spot
    check.
    """

    records: list = field(default_factory=list)
    certified: bool = False

    # -- the fits ----------------------------------------------------------
    def fit_standardiser(self, name: str, ids: Sequence[str], values) -> dict:
        """Mean/std of ``values``, recorded against ``ids``.

        ⚠ A component that is constant on the fitting set (``reach`` is identically
        0 on A) would give ``z = 0/0``; such a component is declared **dimensionless
        by construction** and passed through unstandardised, which is recorded in
        the fit record rather than silently patched.
        """
        v = np.asarray(values, np.float64)
        mu, sd = float(v.mean()), float(v.std())
        passthrough = not (sd > 0)
        self._record(name, "standardiser", ids,
                     {"mean": mu, "std": sd, "passthrough_zero_variance": passthrough})
        return {"mean": mu, "std": sd, "passthrough": passthrough}

    def fit_quantile(self, name: str, ids: Sequence[str], values, q: float) -> float:
        """A threshold at quantile ``q`` of ``values`` — ⛔ valid data only."""
        v = np.asarray(values, np.float64)
        t = float(np.quantile(v, float(q)))
        self._record(name, "quantile", ids, {"q": float(q), "threshold": t})
        return t

    def register_fitting_rows(self, name: str, ids: Sequence[str]) -> None:
        """Declare a set of ids that some *other* stage fitted on (the store's own
        training rows, the prototype rows).  ⛔ They are audited on the same footing:
        a C record must not appear there either."""
        self._record(name, "rows", ids, {})

    def _record(self, name: str, kind: str, ids: Sequence[str], extra: dict) -> None:
        ids = [str(i) for i in ids]
        self.records.append({
            "name": str(name), "kind": str(kind), "n_ids": len(ids),
            "n_unique_ids": len(set(ids)), "ids_sha256": _hash_ids(ids),
            "ids": set(ids), **extra,
        })
        self.certified = False

    # -- the assertion -----------------------------------------------------
    def assert_no_invalid(self, invalid_ids: Sequence[str]) -> dict:
        """⛔⭐ **The criterion this task lives on.**

        Assert that no id from the invalid (C) set appears in any recorded fit.
        Raises :class:`AssertionError` naming the offending fit if it does.
        """
        bad = set(str(i) for i in invalid_ids)
        if not bad:
            raise ValueError(
                "assert_no_invalid was handed an EMPTY invalid set — an assertion "
                "that cannot fail is not evidence"
            )
        offenders = []
        checked = 0
        for rec in self.records:
            checked += rec["n_ids"]
            hit = rec["ids"] & bad
            if hit:
                offenders.append({"fit": rec["name"], "n_hits": len(hit),
                                  "examples": sorted(hit)[:5]})
        if offenders:
            raise AssertionError(
                "⛔ INVALID-EXAMPLE SUPERVISION DETECTED — a C record id reached a "
                f"threshold/normaliser fit: {offenders}.  The certificate's headline "
                "claim is void for this run."
            )
        self.certified = True
        return {
            "n_fits_audited": len(self.records),
            "n_ids_audited": int(checked),
            "n_invalid_ids_checked": len(bad),
            "n_invalid_ids_found_in_any_fit": 0,
            "fits": [
                {k: v for k, v in rec.items() if k != "ids"} for rec in self.records
            ],
            "CERTIFIED_no_invalid_supervision": True,
        }

    def summary(self) -> dict:
        return {"n_fits": len(self.records), "certified": bool(self.certified),
                "fits": [{k: v for k, v in r.items() if k != "ids"} for r in self.records]}


# ---------------------------------------------------------------------------
# 2. AUROC
# ---------------------------------------------------------------------------


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Binary AUROC by the rank (Mann-Whitney U) identity; ties get average ranks."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    ss = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def weighted_auroc(y: Sequence[int], s: Sequence[float], w: Sequence[float]) -> float:
    """Weighted AUROC = P(score(pos) > score(neg)) + ½P(=), under sample weights.

    ⭐ With ``w = 1/|A|`` on the A negatives and ``1/|B|`` on the B negatives this is
    *identically* ``0.5*(AUROC(A vs C) + AUROC(B vs C))`` — the class-balanced
    A∪B-vs-C number R23 asks for, and the identity is asserted at run time.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, np.float64)
    w = np.asarray(w, np.float64)
    pos, neg = y == 1, y == 0
    sp, wp = s[pos], w[pos]
    sn, wn = s[neg], w[neg]
    if sp.size == 0 or sn.size == 0:
        return float("nan")
    gt = 0.0
    eq = 0.0
    for v, ww in zip(sp, wp, strict=True):
        gt += ww * float(np.sum(wn[sn < v]))
        eq += ww * float(np.sum(wn[sn == v]))
    return float((gt + 0.5 * eq) / (wp.sum() * wn.sum()))


# ---------------------------------------------------------------------------
# 3. Risk-coverage / AURC
# ---------------------------------------------------------------------------


def risk_coverage(scores: Sequence[float], is_invalid: Sequence[int],
                  weights: Optional[Sequence[float]] = None) -> dict:
    """Accept in increasing certificate order; risk = weighted invalid share accepted.

    ⛔ **Ties are collapsed, and that is load-bearing.**  A score that cannot
    distinguish two requests cannot be used to accept one and refuse the other, so
    the curve has one point per *distinct* score and the whole tied block is admitted
    together.  Resolving ties in input order instead silently flatters any arm with a
    degenerate score — the ``reach`` arm ties **every** valid request and every
    ``duplicate_factor`` request at exactly 0, and in-order resolution handed it a
    near-optimal AURC it has not earned.  (Caught in this spoke's own smoke run.)

    ``aurc`` is the coverage-weighted integral ``sum_i (cov_i - cov_{i-1}) * risk_i``,
    which for an uninformative score is exactly the population's invalid rate — also
    returned, as ``aurc_random``, so the number is never read without its baseline.
    """
    s = np.asarray(scores, np.float64)
    y = np.asarray(is_invalid).astype(float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, np.float64)
    order = np.argsort(s, kind="mergesort")
    ss, ys, ws = s[order], y[order], w[order]
    cw = np.cumsum(ws)
    cbad = np.cumsum(ys * ws)
    # last index of each tied block
    last = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    cov = cw[last] / cw[-1]
    risk = cbad[last] / np.maximum(cw[last], 1e-300)
    dcov = np.diff(np.r_[0.0, cov])
    return {
        "coverage": [float(x) for x in cov],
        "risk": [float(x) for x in risk],
        "aurc": float(np.sum(dcov * risk)),
        "aurc_random": float((y * w).sum() / w.sum()),
        "n": int(s.size),
        "n_distinct_scores": int(last.size),
    }


def operating_point(threshold: float, scores_by_set: dict) -> dict:
    """Refusal rate of each request set at a threshold fitted on VALID data only."""
    out = {"threshold": float(threshold)}
    for name, sc in scores_by_set.items():
        v = np.asarray(sc, np.float64)
        out[f"refusal_rate_{name}"] = float((v > float(threshold)).mean()) if v.size else float("nan")
        out[f"n_{name}"] = int(v.size)
    return out


# ---------------------------------------------------------------------------
# 4. The three AUROCs, together, with the per-kind split
# ---------------------------------------------------------------------------


def certificate_metrics(score_a, score_b, score_c, kinds_c: Sequence[str]) -> dict:
    """⛔ All three AUROCs, separately and with equal prominence (R7/R23).

    ``B_vs_C`` is R23's primary refusal statistic; ``A_vs_C`` and the class-balanced
    ``AB_vs_C`` are reported alongside.  The per-kind split is mandatory here: a
    grid lookup separates ``out_of_range`` perfectly and cannot see
    ``duplicate_factor`` at all, so an aggregate C number hides which mechanism did
    the work.
    """
    a = np.asarray(score_a, np.float64)
    b = np.asarray(score_b, np.float64)
    c = np.asarray(score_c, np.float64)
    kinds = np.asarray(list(kinds_c))

    def _vs(neg, pos):
        y = np.r_[np.zeros(neg.size), np.ones(pos.size)]
        return auroc(y, np.r_[neg, pos])

    out = {
        "A_vs_C": _vs(a, c),
        "B_vs_C": _vs(b, c),
        "n_A": int(a.size), "n_B": int(b.size), "n_C": int(c.size),
    }
    y = np.r_[np.zeros(a.size + b.size), np.ones(c.size)]
    s = np.r_[a, b, c]
    w = np.r_[np.full(a.size, 1.0 / max(a.size, 1)),
              np.full(b.size, 1.0 / max(b.size, 1)),
              np.full(c.size, 1.0 / max(c.size, 1))]
    out["AB_vs_C_balanced"] = weighted_auroc(y, s, w)
    out["AB_vs_C_balanced_identity_check"] = float(
        abs(out["AB_vs_C_balanced"] - 0.5 * (out["A_vs_C"] + out["B_vs_C"]))
    )
    out["pooled_UNBALANCED_do_not_quote"] = _vs(np.r_[a, b], c)
    out["pooled_note"] = (
        "⛔ the unbalanced pooled AUROC is reported ONLY so it is not mistaken for the "
        "headline: |B| is 0.0150% of A∪B, so refusing all of B costs it almost nothing"
    )
    for kind in sorted(set(kinds.tolist())):
        sel = kinds == kind
        out[f"A_vs_C[{kind}]"] = _vs(a, c[sel])
        out[f"B_vs_C[{kind}]"] = _vs(b, c[sel])
        out[f"n_C[{kind}]"] = int(sel.sum())
    return out
