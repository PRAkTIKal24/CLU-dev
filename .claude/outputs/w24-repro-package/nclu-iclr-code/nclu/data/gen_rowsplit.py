"""gen rowsplit."""

from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np


class RowSplitError(AssertionError):
    """Raised when the realised ``fit``/``val`` split is not balanced."""


def head_tail_fit_val(pool: np.ndarray, n_fit: int, n_val: int) -> tuple:
    pool = np.asarray(pool, np.int64)
    n_fit = int(min(int(n_fit), pool.size))
    n_val = int(min(int(n_val), max(pool.size - n_fit, 0)))
    return (pool[:n_fit], pool[n_fit : n_fit + n_val])


def stratified_fit_val(
    pool: np.ndarray,
    labels: np.ndarray,
    n_fit: int,
    n_val: int,
    seed: int,
    *,
    tiebreak_jitter: float = 1.0,
    seed_mult: int = 6637,
    seed_offset: int = 20260914,
) -> tuple:
    pool = np.asarray(pool, np.int64)
    labels = np.asarray(labels, np.int64)
    if labels.ndim != 2 or labels.shape[0] != pool.size:
        raise ValueError(
            f"labels must be (n_pool, n_factors); got {labels.shape} for {pool.size} pool rows"
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
    target = [
        n_val * np.bincount(labels[:, f], minlength=cards[f]) / max(n_pool, 1) for f in range(n_fac)
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
        "rule": "deficit-priority greedy on the pool's own per-factor marginals (target[f][v] = n_val * count_pool[f][v] / n_pool), seeded tie-break",
        "n_pool": int(n_pool),
        "n_fit": int(fit.size),
        "n_val": int(val.size),
        "tiebreak_jitter": float(tiebreak_jitter),
        "rng": f"np.random.default_rng({int(seed)}*{int(seed_mult)}+{int(seed_offset)})",
        "fit_is_pool_minus_val": bool(fit.size + val.size == n_pool),
    }
    return (fit, val, info)


def split_fit_val(
    pool: np.ndarray,
    labels: np.ndarray,
    n_fit: int,
    n_val: int,
    seed: int,
    *,
    mode: str = "stratified",
    tiebreak_jitter: float = 1.0,
    seed_mult: int = 6637,
    seed_offset: int = 20260914,
    log: Optional[Callable] = None,
) -> tuple:
    mode = str(mode)
    if mode == "head_tail":
        fit, val = head_tail_fit_val(pool, n_fit, n_val)
        if log is not None:
            log(
                "  ⛔⛔ row_split.mode='head_tail': this is the BROKEN split (val = the sorted TAIL of the factor grid; 1 floor_hue value against fit's 10). Every configuration selected under it is a coin flip. Historical reproduction ONLY."
            )
        return (
            fit,
            val,
            {
                "mode": "head_tail",
                "rule": "pool[:n_fit] / pool[n_fit:]",
                "n_fit": int(fit.size),
                "n_val": int(val.size),
                "BROKEN": True,
            },
        )
    if mode != "stratified":
        raise ValueError(f"unknown row-split mode {mode!r} (have 'stratified', 'head_tail')")
    return stratified_fit_val(
        pool,
        labels,
        n_fit,
        n_val,
        seed,
        tiebreak_jitter=tiebreak_jitter,
        seed_mult=seed_mult,
        seed_offset=seed_offset,
    )


def row_split_balance(
    labels_fit: np.ndarray,
    labels_val: np.ndarray,
    names: Sequence[str],
    cardinalities: Optional[Sequence[int]] = None,
) -> dict:
    lf = np.asarray(labels_fit, np.int64)
    lv = np.asarray(labels_val, np.int64)
    n_fac = lf.shape[1]
    if cardinalities is None:
        cardinalities = [int(max(lf[:, f].max(), lv[:, f].max())) + 1 for f in range(n_fac)]
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
            "val_count_ratio": float(cv.max()) / float(nz.min()) if nz.size else float("inf"),
            "val_covers_fit": bool((cv > 0).sum() == (cf > 0).sum()),
        }
    return {
        "n_fit": int(lf.shape[0]),
        "n_val": int(lv.shape[0]),
        "factors": rows,
        "min_val_n_values_realised": int(min((r["val_n_values_realised"] for r in rows.values()))),
        "max_val_count_ratio": float(max((r["val_count_ratio"] for r in rows.values()))),
    }


def assert_row_split_balanced(
    balance: dict, *, require_all_values: bool = True, max_count_ratio: float = 1.25
) -> dict:
    bad = []
    for name, r in balance["factors"].items():
        if require_all_values and r["val_n_values_realised"] < r["cardinality"]:
            bad.append(
                f"{name}: val realises {r['val_n_values_realised']} of {r['cardinality']} values (fit realises {r['fit_n_values_realised']})"
            )
        elif r["val_count_ratio"] > float(max_count_ratio):
            bad.append(
                f"{name}: val per-value counts range {r['val_count_min']}..{r['val_count_max']} (ratio {r['val_count_ratio']:.3f} > {float(max_count_ratio):.3f})"
            )
    balance = dict(balance)
    balance["balanced"] = not bad
    balance["violations"] = bad
    if bad:
        raise RowSplitError(
            "⛔ the fit/val split is NOT balanced — a configuration selected on this `val` is not identified:\n  - "
            + "\n  - ".join(bad)
        )
    return balance
