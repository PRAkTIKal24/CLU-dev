"""⭐ The ``fit`` / ``val`` row split for the generative cell — **stratified**, and the
assertion that proves it balanced.

⛔ **Read this before changing anything below.**

**The defect this module repairs (inherited, measured, not hypothesised).**
``exp_gen_baselines.select_baseline_rows`` — and its two verbatim copies in
``exp_gen_cell`` and ``exp_gen_atom_mean`` — took ``fit = pool[:n_fit]`` and
``val = pool[n_fit : n_fit + n_val]`` of a **SORTED** row-index array.  Both Shapes3D
and dSprites enumerate every factor combination exactly once in C order, so a dataset
row index *is* a lexicographic factor-index tuple and a **sorted tail is a contiguous
corner of the factor grid**.  At production scale (``pool`` = 32,768 rows,
``n_val`` = 512) the realised consequence on Shapes3D was:

===============  =====  ===========================================
row set          n      ``n_values_realised`` per factor
===============  =====  ===========================================
``fit``          32256  floor 10 · wall 10 · object 10 · scale 8 · shape 4 · orient 15
``val`` (broken)   512  **floor 1** · **wall 2** · object 10 · scale 8 · shape 4 · orient 15
===============  =====  ===========================================

— feature variance **0.0924** against ``fit``'s **0.2229**, and the two halves of
``val`` chose a *different* winning tuning trial in **5 of 6** (seed, budget) cells
(``gen-atom-mean-control`` §7).  ⛔ Every configuration selected on that ``val`` was a
coin-flip wearing a choice's clothes.

**The rule (declared in ``gen-val-split-fix/PREREG.md`` §3 before it was coded).**
*Proportional marginal stratification by a randomised greedy quota filter.*  ``pool``
itself is **untouched** — the store, the atoms and the held-out disjointness are
exactly what they were — only the *partition* of it changes:

1. per-factor proportional **target** ``t[f][v] = n_val · count_pool[f][v] / n_pool``;
2. take ``n_val`` rows one at a time, each time the row whose six buckets are
   collectively furthest **below** target, ties broken by a seeded jitter;
3. ``fit = sort(pool \\ val)[:n_fit]``, ``val = sort(val)``.

⛔ The first draft of this used a *cap*-and-relax filter instead, and it was not good
enough: a cap is one-sided, so a row rejected late because one of its six buckets was
full left the other five underfilled — measured floor-hue counts of **42..53** on seed 1
(ratio 1.262), which is no tighter than plain random sampling. The deficit-priority rule
is two-sided and lands inside ±1 of target. ``row_split_balance`` measures the realised
counts and ``assert_row_split_balanced`` gates them.

⚠ **``fit`` moves too, and that is not a side effect to be apologised for.** With
``n_fit + n_val == pool.size`` (the shipped 32,256 + 512 = 32,768) ``fit`` is exactly
``pool`` minus ``val``, so repairing ``val`` swaps 1.56 % of ``fit``'s rows.  Everything
fitted on ``fit`` — the kNN k-means store, the z-score normaliser, LACE's whitening and
classifier, ``FactorPrototypes`` — therefore moves.  ⛔ There is no version of this
repair in which the published numbers do not move; a "fix" that left them bit-identical
would be a fix that did not happen.

⛔ **``mode="head_tail"`` reproduces the broken split bit-for-bit.**  It exists so a
historical run can be reproduced deliberately, it must be asked for by name, and it
shouts in the log when it is used.  ⛔ It is never the default.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    "RowSplitError",
    "assert_row_split_balanced",
    "head_tail_fit_val",
    "row_split_balance",
    "split_fit_val",
    "stratified_fit_val",
]


class RowSplitError(AssertionError):
    """Raised when the realised ``fit``/``val`` split is not balanced."""


# ---------------------------------------------------------------------------
# The two constructions
# ---------------------------------------------------------------------------


def head_tail_fit_val(pool: np.ndarray, n_fit: int, n_val: int) -> tuple:
    """⛔ THE BROKEN SPLIT, reproduced verbatim.  Head and tail of a SORTED array.

    Kept only so a historical configuration can be reproduced on purpose.
    """
    pool = np.asarray(pool, np.int64)
    n_fit = int(min(int(n_fit), pool.size))
    n_val = int(min(int(n_val), max(pool.size - n_fit, 0)))
    return pool[:n_fit], pool[n_fit : n_fit + n_val]


def stratified_fit_val(
    pool: np.ndarray,
    labels: np.ndarray,
    n_fit: int,
    n_val: int,
    seed: int,
    *,
    tiebreak_jitter: float = 1.0,
    seed_mult: int = 6_637,
    seed_offset: int = 20_260_914,
) -> tuple:
    """``(fit, val, info)`` — ``val`` stratified on **every column of** ``labels``.

    ⭐ **Deficit-priority greedy.**  Target counts are the *pool's own* marginals scaled
    to ``n_val`` (``target[f][v] = n_val · count_pool[f][v] / n_pool``); at each step the
    row taken is the one whose six factor buckets are collectively furthest **below**
    target, with a seeded jitter breaking the (very many) ties.  ⛔ Deliberately not a
    cap-and-relax filter: a cap is a one-sided constraint, and a row rejected late
    because *one* of its six buckets is full leaves the other five underfilled — that
    construction measured a floor-hue count range of 42..53 on seed 1, which is no
    better than plain random sampling.

    Args:
        pool: (N,) dataset row indices.  ⛔ Returned partitioned, never resampled.
        labels: (N, F) **factor value indices** for those rows, in ``pool`` order.
        n_fit, n_val: the shipped 32,256 / 512.  ⛔ Never a caller's own budget.
        seed: the run seed; the jitter is ``default_rng(seed*seed_mult + seed_offset)``.
            ⚠ Deliberately a NumPy Generator and ⛔ **not** a lengthened
            ``jax.random.split`` — ``split(key, n)`` is prefix-stable here, so arm keys
            derived by lengthening a split silently share a prefix.

    Returns:
        ``(fit_rows, val_rows, info)`` with both row arrays **sorted ascending** (the
        downstream feature gather uses ``searchsorted`` against the cache).
    """
    pool = np.asarray(pool, np.int64)
    labels = np.asarray(labels, np.int64)
    if labels.ndim != 2 or labels.shape[0] != pool.size:
        raise ValueError(
            f"labels must be (n_pool, n_factors); got {labels.shape} for "
            f"{pool.size} pool rows"
        )
    n_fit = int(min(int(n_fit), pool.size))
    n_val = int(min(int(n_val), max(pool.size - n_fit, 0)))
    if n_val == 0:
        raise ValueError(
            f"no validation rows left: n_fit={n_fit} exhausts the {pool.size}-row pool"
        )

    n_pool, n_fac = labels.shape
    cards = [int(labels[:, f].max()) + 1 for f in range(n_fac)]
    rng = np.random.default_rng(int(seed) * int(seed_mult) + int(seed_offset))
    # ⭐ PROPORTIONAL, not uniform: the target is "val has the pool's marginals", which
    # is what exchangeability with ``fit`` means.  A uniform target would *impose* a
    # distribution the pool does not have.
    target = [
        n_val * np.bincount(labels[:, f], minlength=cards[f]) / max(n_pool, 1)
        for f in range(n_fac)
    ]
    need = [t.copy() for t in target]
    jitter = rng.random(n_pool) * float(tiebreak_jitter)
    taken = np.zeros(n_pool, bool)
    cols = [labels[:, f] for f in range(n_fac)]

    for _ in range(n_val):
        score = jitter.copy()
        for f in range(n_fac):
            score += need[f][cols[f]]
        score[taken] = -np.inf
        i = int(np.argmax(score))
        taken[i] = True
        for f in range(n_fac):
            need[f][labels[i, f]] -= 1.0

    val = np.sort(pool[taken])
    fit = np.sort(pool[~taken][:n_fit])
    info = {
        "mode": "stratified",
        "rule": (
            "deficit-priority greedy on the pool's own per-factor marginals "
            "(target[f][v] = n_val * count_pool[f][v] / n_pool), seeded tie-break"
        ),
        "n_pool": int(n_pool),
        "n_fit": int(fit.size),
        "n_val": int(val.size),
        "tiebreak_jitter": float(tiebreak_jitter),
        "rng": f"np.random.default_rng({int(seed)}*{int(seed_mult)}+{int(seed_offset)})",
        "fit_is_pool_minus_val": bool(fit.size + val.size == n_pool),
    }
    return fit, val, info


def split_fit_val(
    pool: np.ndarray,
    labels: np.ndarray,
    n_fit: int,
    n_val: int,
    seed: int,
    *,
    mode: str = "stratified",
    tiebreak_jitter: float = 1.0,
    seed_mult: int = 6_637,
    seed_offset: int = 20_260_914,
    log: Optional[Callable] = None,
) -> tuple:
    """Dispatch on ``mode``.  ⛔ ``head_tail`` is the broken split and says so."""
    mode = str(mode)
    if mode == "head_tail":
        fit, val = head_tail_fit_val(pool, n_fit, n_val)
        if log is not None:
            log(
                "  ⛔⛔ row_split.mode='head_tail': this is the BROKEN split "
                "(val = the sorted TAIL of the factor grid; 1 floor_hue value against "
                "fit's 10). Every configuration selected under it is a coin flip. "
                "Historical reproduction ONLY."
            )
        return fit, val, {"mode": "head_tail", "rule": "pool[:n_fit] / pool[n_fit:]",
                          "n_fit": int(fit.size), "n_val": int(val.size),
                          "BROKEN": True}
    if mode != "stratified":
        raise ValueError(f"unknown row-split mode {mode!r} (have 'stratified', 'head_tail')")
    return stratified_fit_val(
        pool, labels, n_fit, n_val, seed,
        tiebreak_jitter=tiebreak_jitter, seed_mult=seed_mult, seed_offset=seed_offset,
    )


# ---------------------------------------------------------------------------
# The balance report + its gate
# ---------------------------------------------------------------------------


def row_split_balance(
    labels_fit: np.ndarray,
    labels_val: np.ndarray,
    names: Sequence[str],
    cardinalities: Optional[Sequence[int]] = None,
) -> dict:
    """⭐ ``n_values_realised`` per factor **on every row**, fit vs val (R31).

    ⛔ Reported for **every** factor, not only the ones that were visibly broken: the
    phase turns on ``scale`` and ``orientation``, which the head/tail split already
    realised in full — their failure mode is *per-value count* imbalance, so the
    max/min count ratio is carried beside the realised count.
    """
    lf = np.asarray(labels_fit, np.int64)
    lv = np.asarray(labels_val, np.int64)
    n_fac = lf.shape[1]
    if cardinalities is None:
        cardinalities = [
            int(max(lf[:, f].max(), lv[:, f].max())) + 1 for f in range(n_fac)
        ]
    rows = {}
    for f in range(n_fac):
        card = int(cardinalities[f])
        cf = np.bincount(lf[:, f], minlength=card)
        cv = np.bincount(lv[:, f], minlength=card)
        nz = cv[cv > 0]
        rows[str(names[f])] = {
            "cardinality": card,
            "fit_n_values_realised": int((cf > 0).sum()),
            "val_n_values_realised": int((cv > 0).sum()),
            "val_counts": [int(x) for x in cv],
            "val_count_min": int(nz.min()) if nz.size else 0,
            "val_count_max": int(cv.max()),
            "val_count_ratio": (float(cv.max()) / float(nz.min())) if nz.size else float("inf"),
            "val_covers_fit": bool((cv > 0).sum() == (cf > 0).sum()),
        }
    return {
        "n_fit": int(lf.shape[0]),
        "n_val": int(lv.shape[0]),
        "factors": rows,
        "min_val_n_values_realised": int(
            min(r["val_n_values_realised"] for r in rows.values())
        ),
        "max_val_count_ratio": float(max(r["val_count_ratio"] for r in rows.values())),
    }


def assert_row_split_balanced(
    balance: dict,
    *,
    require_all_values: bool = True,
    max_count_ratio: float = 1.25,
) -> dict:
    """⛔ Raise :class:`RowSplitError` if ``val`` is not balanced.  Returns ``balance``.

    ⭐ Deliberately capable of failing: run it on the head/tail split and it raises on
    ``floor_hue`` (1 realised value of 10).  ``tests/test_gen_rowsplit.py`` is that
    positive control.
    """
    bad = []
    for name, r in balance["factors"].items():
        if require_all_values and r["val_n_values_realised"] < r["cardinality"]:
            bad.append(
                f"{name}: val realises {r['val_n_values_realised']} of "
                f"{r['cardinality']} values (fit realises {r['fit_n_values_realised']})"
            )
        elif r["val_count_ratio"] > float(max_count_ratio):
            bad.append(
                f"{name}: val per-value counts range "
                f"{r['val_count_min']}..{r['val_count_max']} "
                f"(ratio {r['val_count_ratio']:.3f} > {float(max_count_ratio):.3f})"
            )
    balance = dict(balance)
    balance["balanced"] = not bad
    balance["violations"] = bad
    if bad:
        raise RowSplitError(
            "⛔ the fit/val split is NOT balanced — a configuration selected on this "
            "`val` is not identified:\n  - " + "\n  - ".join(bad)
        )
    return balance
