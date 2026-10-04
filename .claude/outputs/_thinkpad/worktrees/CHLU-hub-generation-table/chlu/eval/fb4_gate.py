"""⭐ **FB4 — "the instrument is invalid"** (`PREREG-Bprime.md` §6, falsifier FB4).

    *"The +0 B substitute is at ceiling for every family **including full
    attention** ⇒ the protocol measures the task, not the memory."*

B′ (the audit paper) applies **one** protocol — matched-byte launder + two-sided
byte ledger + a **+0 B** substitute audit + same-keys null — to CLU and to six
rival memory families. Before that protocol is spent on six families it has to be
shown to measure the **memory** and not the **task**. That is this module.

It contains exactly three things, and none of them runs a store:

1. :func:`attention_read` — ⭐ **the attention arm**: a full-attention reader over
   *the launder's own* ``(key, payload)`` table. ``softmax(q·kᵀ/(τ√d))``,
   value-weighted, **no learned parameters beyond a scalar temperature** fitted on
   the family's own train split. It is a reader of the *same bytes*, so its ledger
   is the table's ledger **+ 4 B** (:data:`ATTENTION_TEMPERATURE_BYTES`).
   ⚠ It is a **table** reader and **never sees a trajectory** — it is *not*
   ``AttentionPsi`` and inherits none of that quarantine.
2. :func:`saturation` / :func:`family_saturated` — the **pre-registered** D0.1
   rule, with its two constants frozen (``0.95`` and the 2-SE attention leg).
3. :func:`fb4_verdict` — ``FIRES`` / ``PARTIAL`` / ``CLEARS`` computed from those,
   so the gate is **computed, not argued** (the C2W2 precedent).

**The rule** (Head-ratified 2026-07-31, non-tunable after seeing data). With
``M(f)`` the metric's exact maximum (``1.0`` for ``decode``/``acc``/``r2``,
``0.0`` for ``neg_mae``), ``blank(f)`` the blank-store control, ``sub(f)`` the
best **+0 B** substitute and ``attn(f)`` the attention arm::

    S(f) = (sub(f) - blank(f)) / (M(f) - blank(f))
    f is SUBSTITUTE-SATURATED  iff  S(f) >= 0.95  and  sub(f) >= attn(f) - 2 SE

    FIRES   iff all four families saturate  => the protocol measures the task
    PARTIAL iff 1..3 saturate               => those families are struck from
                                               B'-s cross-family audit as
                                               protocol-invalid; the wave
                                               proceeds on the survivors
    CLEARS  iff none saturate (or only the EXPECTED `manifold`, whose `echo`
                               substitute scores 1.0000 at +0 B by construction)

The normalisation against each family's own floor exists because the four gym
metrics have incommensurable scales (``decode`` ∈[0,1] · ``neg_mae`` ≤0 ·
``acc`` ∈[0,1] · ``r2`` ≤1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: One float32 temperature — the attention arm's ENTIRE parameter budget.
ATTENTION_TEMPERATURE_BYTES = 4

#: The metric's exact maximum, per gym primary metric (the ``M(f)`` of the rule).
METRIC_MAX: Dict[str, float] = {"decode": 1.0, "acc": 1.0, "r2": 1.0, "neg_mae": 0.0}

#: The pre-registered saturation threshold. ⛔ Not tunable after seeing data.
SATURATION_THRESHOLD = 0.95

#: The pre-registered attention leg's width, in SEs of the paired difference.
ATTENTION_SE_LEGS = 2.0

#: The family whose saturation is the PREDICTED outcome, not news: ``echo_launder``
#: scores 1.0000 at +0 B by construction (intervention §8.3 in its purest form).
EXPECTED_SATURATED = ("manifold",)


# --------------------------------------------------------------------------
# 1. the attention arm
# --------------------------------------------------------------------------
def attention_logits(keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """``q·kᵀ/√d`` — scaled dot-product scores, ``(n_queries, n_keys)``.

    Deliberately the *textbook* form and not a distance: this arm exists to be
    "the strongest metric-native reader anyone would reach for", so it must be
    the thing they would actually reach for. (It differs from the launder's
    ``argmin‖q−k‖`` only through the ``‖k‖²/2`` term, which is a real difference
    whenever the stored keys have unequal norms.)
    """
    q = np.asarray(queries, dtype=float)
    k = np.asarray(keys, dtype=float)
    if q.ndim != 2 or k.ndim != 2 or q.shape[1] != k.shape[1]:
        raise ValueError(f"attention needs (n,d)/(K,d); got {q.shape} / {k.shape}")
    return (q @ k.T) / float(np.sqrt(max(k.shape[1], 1)))


def attention_weights(keys: np.ndarray, queries: np.ndarray, *,
                      temperature: float = 1.0) -> np.ndarray:
    """``softmax(q·kᵀ/(τ√d))`` over the table's rows, ``(n_queries, n_keys)``."""
    z = attention_logits(keys, queries) / max(float(temperature), 1e-12)
    z = z - np.max(z, axis=1, keepdims=True)
    w = np.exp(z)
    return np.asarray(w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-30))


def attention_read(keys: np.ndarray, values: np.ndarray, queries: np.ndarray, *,
                   temperature: float = 1.0, kind: str = "value",
                   pairs: Optional[np.ndarray] = None) -> np.ndarray:
    """⭐ The attention arm's prediction for one gym family.

    ``kind`` follows :class:`~chlu.experiments.memory_gym.QuerySet.kind`:

    ``"value"`` / ``"coord"``
        the value-weighted read ``Σ_i w_i · values[i]`` — a convex combination of
        the table's own stored values, ``(n_queries, m)``.
    ``"index"``
        the question is *which row*, so the read is the arg-max of the attention
        logits; when ``pairs`` is given (the C2W2 D4 fix, which every other arm
        also gets) the arg-max is taken over the query's **own two candidates**.
        ⚠ For this ``kind`` the arm is **temperature-independent** — a positive
        scalar cannot reorder two logits — and that is reported, not hidden.
    """
    ky = np.asarray(keys, dtype=float)
    q = np.asarray(queries, dtype=float)
    if kind == "index":
        z = attention_logits(ky, q) / max(float(temperature), 1e-12)
        if pairs is None:
            return np.asarray(np.argmax(z, axis=1))
        from chlu.experiments.memory_gym import restrict_to_pair

        return restrict_to_pair(z, np.asarray(pairs, dtype=int))
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    w = attention_weights(ky, q, temperature=temperature)
    return np.asarray(w @ vals)


#: The declared temperature grid (log-spaced, 41 points). Fixed in PREREG §1.1(d).
TEMPERATURE_GRID: Tuple[float, ...] = tuple(np.logspace(-2.0, 2.0, 41))


def fit_attention_temperature(keys: np.ndarray, values: np.ndarray,
                              fit_queries: np.ndarray,
                              score_fn: Callable[[np.ndarray], float], *,
                              kind: str = "value",
                              pairs: Optional[np.ndarray] = None,
                              grid: Sequence[float] = TEMPERATURE_GRID
                              ) -> Dict[str, Any]:
    """Fit the arm's **only** parameter on the family's own *train* split.

    Grid search maximising ``score_fn`` (the family's own primary metric, through
    the family's own scorer) on an independent draw of the same query law. The
    fitted ``tau`` costs :data:`ATTENTION_TEMPERATURE_BYTES` = 4 B and nothing
    else — which is what makes the arm's ledger commensurable with the table's
    (FB4 is *undecidable as specified* if it is not).

    Returns ``tau``, the ``curve`` (one score per grid point) and
    ``degenerate``: ``True`` when the score is constant over the whole grid, i.e.
    the temperature is **not identifiable** for this family (the ``index`` case).
    """
    curve = []
    for t in grid:
        pred = attention_read(keys, values, fit_queries, temperature=float(t),
                              kind=kind, pairs=pairs)
        curve.append(float(score_fn(pred)))
    arr = np.asarray(curve, dtype=float)
    best = int(np.nanargmax(arr))
    spread = float(np.nanmax(arr) - np.nanmin(arr))
    return {"tau": float(grid[best]), "fit_score": float(arr[best]),
            "curve": [float(x) for x in arr], "grid": [float(x) for x in grid],
            "degenerate": bool(spread <= 1e-12),
            "bytes": int(ATTENTION_TEMPERATURE_BYTES),
            "n_fit_queries": int(np.asarray(fit_queries).shape[0])}


# --------------------------------------------------------------------------
# 2. the pre-registered rule
# --------------------------------------------------------------------------
def saturation(sub: float, blank: float, metric_max: float) -> float:
    """``S = (sub − blank) / (M − blank)`` — the family's floor-normalised ceiling.

    ``M`` is the metric's **exact** maximum, never an empirical one: normalising
    against a measured best would make the rule self-referential.
    """
    denom = float(metric_max) - float(blank)
    if abs(denom) < 1e-12:
        return float("nan")
    return (float(sub) - float(blank)) / denom


def _sd(a: np.ndarray) -> float:
    a = np.asarray([x for x in np.asarray(a, dtype=float).ravel() if np.isfinite(x)])
    return float(np.std(a, ddof=1)) if a.size > 1 else 0.0


@dataclass
class FamilyVerdict:
    """One family's row of the FB4 table — every quantity the rule consumes."""

    family: str
    metric: str
    metric_max: float
    blank: float
    sub: float
    attn: float
    S: float
    se_paired: float
    saturated: bool
    sub_name: str = ""
    n_seeds: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["detail"] = dict(self.detail)
        return d


def family_saturated(family: str, metric: str, *,
                     sub_seeds: Sequence[float], attn_seeds: Sequence[float],
                     blank_seeds: Sequence[float], sub_name: str = "",
                     metric_max: Optional[float] = None,
                     threshold: float = SATURATION_THRESHOLD,
                     detail: Optional[dict] = None) -> FamilyVerdict:
    """Apply the pre-registered rule to one family's three seeds.

    ``S`` is computed from the 3-seed **means** (declared aggregation order), and
    the attention leg uses the SE of the **paired** per-seed difference
    ``sub_s − attn_s`` (both arms are read off the same store at the same seed).
    Per-seed ``S_s`` and the unpaired SEs are reported in ``detail``.
    """
    M = float(METRIC_MAX[metric] if metric_max is None else metric_max)
    sub_a = np.asarray(sub_seeds, dtype=float)
    attn_a = np.asarray(attn_seeds, dtype=float)
    blank_a = np.asarray(blank_seeds, dtype=float)
    n = int(min(sub_a.size, attn_a.size, blank_a.size))
    sub_m, attn_m, blank_m = (float(np.mean(sub_a)), float(np.mean(attn_a)),
                              float(np.mean(blank_a)))
    S = saturation(sub_m, blank_m, M)
    diff = sub_a[:n] - attn_a[:n]
    se_paired = _sd(diff) / np.sqrt(max(n, 1))
    leg_ceiling = bool(np.isfinite(S) and S >= float(threshold))
    leg_attn = bool(sub_m >= attn_m - ATTENTION_SE_LEGS * se_paired)
    det = dict(detail or {})
    det.update({
        "sub_seeds": [float(x) for x in sub_a],
        "attn_seeds": [float(x) for x in attn_a],
        "blank_seeds": [float(x) for x in blank_a],
        "S_per_seed": [float(saturation(s, b, M))
                       for s, b in zip(sub_a[:n], blank_a[:n], strict=False)],
        "se_sub": _sd(sub_a) / np.sqrt(max(sub_a.size, 1)),
        "se_attn": _sd(attn_a) / np.sqrt(max(attn_a.size, 1)),
        "se_paired_diff": float(se_paired),
        "leg_ceiling(S>=%.2f)" % threshold: leg_ceiling,
        "leg_attention(sub>=attn-2SE)": leg_attn,
        "threshold": float(threshold),
    })
    return FamilyVerdict(
        family=family, metric=metric, metric_max=M, blank=blank_m, sub=sub_m,
        attn=attn_m, S=float(S), se_paired=float(se_paired),
        saturated=bool(leg_ceiling and leg_attn), sub_name=str(sub_name),
        n_seeds=n, detail=det,
    )


def fb4_verdict(rows: Sequence[FamilyVerdict]) -> Dict[str, Any]:
    """``FIRES`` / ``PARTIAL = {...}`` / ``CLEARS`` — the gate, computed.

    ⛔ ``FIRES`` **only** if every family measured saturates (the rule is written
    for the four gym families; a partial family list is reported as such and can
    never fire the gate).
    """
    fams = [r.family for r in rows]
    sat = [r.family for r in rows if r.saturated]
    survivors = [r.family for r in rows if not r.saturated]
    if len(rows) >= 4 and len(sat) == len(rows):
        verdict = "FIRES"
        note = ("⛔ every family is substitute-saturated ⇒ the B′ protocol measures "
                "the TASK, not the memory. STOP: report to the Hub the same hour; "
                "the wave pauses for a Head/Advisor protocol ruling before "
                "`bprime-rivals` is built (§A11 task 1).")
    elif sat:
        unexpected = [f for f in sat if f not in EXPECTED_SATURATED]
        verdict = "PARTIAL = {" + ", ".join(sorted(sat)) + "}"
        note = ("◐ each saturated family is STRUCK from B′'s cross-family audit as "
                "protocol-invalid; the wave proceeds on the survivors. "
                + ("`manifold` alone is the PREDICTED outcome, not news (echo = "
                   "1.0000 at +0 B by construction)." if not unexpected else
                   "Unexpectedly saturated (beyond the predicted `manifold`): "
                   + ", ".join(sorted(unexpected)) + "."))
        if not unexpected:
            verdict = "CLEARS (only the expected `manifold` saturates)"
    else:
        verdict = "CLEARS"
        note = ("✅ no family is substitute-saturated ⇒ the protocol is validated "
                "and `bprime-rivals` is released against all of them.")
    return {"verdict": verdict, "saturated": sorted(sat),
            "surviving_families": sorted(survivors), "families_measured": fams,
            "rule": ("S(f) = (sub - blank)/(M - blank); saturated iff S >= "
                     f"{SATURATION_THRESHOLD} AND sub >= attn - "
                     f"{ATTENTION_SE_LEGS:.0f} SE (paired, 3 seeds, ddof=1). "
                     "FIRES iff ALL FOUR saturate."),
            "note": note,
            "rows": [r.as_dict() for r in rows]}


__all__ = [
    "ATTENTION_TEMPERATURE_BYTES", "METRIC_MAX", "SATURATION_THRESHOLD",
    "ATTENTION_SE_LEGS", "EXPECTED_SATURATED", "TEMPERATURE_GRID",
    "attention_logits", "attention_weights", "attention_read",
    "fit_attention_temperature", "saturation", "FamilyVerdict",
    "family_saturated", "fb4_verdict",
]
