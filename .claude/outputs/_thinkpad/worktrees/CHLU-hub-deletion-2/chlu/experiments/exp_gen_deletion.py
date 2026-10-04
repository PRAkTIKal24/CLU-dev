"""⭐ ``hub/deletion`` — **exact structural deletion** on the generative store.

The mechanism is ported verbatim from *Retention You Can Predict, Scope and
Delete* (§2.3 / App. F / App. G.8) and ⛔ is not reinvented here:

    Deletion is exact when the store layout is a **canonical function of the live
    item set and the fixed geometry** (history-independent placement).  So

        delete(S, P) := rebuild( live \\ rows(P) )

    and the post-delete store is **byte-identical** to a store that never held the
    deleted rows.  Cost: **O(n) rebuild per delete**, accepted openly.

⚠ The caveat carries over verbatim: *store-level guarantee only; the frozen encoder
is a separate channel; ⛔ no certified (eps, delta) unlearning is claimed.*

⛔ :func:`chlu.core.gen_store.delete_store` is **NOT** this operator — its own
docstring says it is the laundering control (zero / mean / scramble the modes).
It is used here as exactly that and never as the delete.

**The three layers, kept apart on purpose**

1. :func:`canonical_build` — the canonical rebuild.  Everything it consumes is
   either the sorted live row vector or a pinned constant; ⛔ nothing carries over
   from a previous build.  This is what makes the provenance assertion provable.
2. :func:`delete_rows` / :func:`row_predicate` — the delete is a **predicate over
   factor tuples**, so a factor value (``f == v``) and a conjunction
   (``f1 == v1 AND f2 == v2``) are the same code.  Rivals take the identical
   predicate, which is why the comparison is a comparison.
3. ⚠ Zeroing a mode's column is **NOT canonical** (the other modes were fitted
   with it present).  Reported as **scoped**, with its measured residue, ⛔ never
   as exact.
"""

from __future__ import annotations

import hashlib
import time
from typing import Callable, Optional, Sequence

import jax
import numpy as np

from ..core.gen_bridge import param_hash
from ..core.gen_dream import fit_address_book
from ..core.gen_store import (
    address_codes,
    channel_correlation,
    dead_channels,
    grouping_labels,
    init_store,
    train_store,
    well_channels_from_labels,
)

__all__ = [
    "canonical_build",
    "delete_rows",
    "factor_tuples",
    "row_predicate",
    "store_digest",
]


# ---------------------------------------------------------------------------
# 1. The digest — what "byte-identical" is asserted on
# ---------------------------------------------------------------------------


def store_digest(store, labels_vec=None, book=None) -> dict:
    """sha256 over **everything the deployed read consumes**, not just the modes.

    ⚠ :func:`chlu.core.gen_bridge.param_hash` covers the mode arrays only.  Two
    other things would have slipped past it, and both are part of the layout the
    mechanism claims is canonical:

    * ``GenStore.well_channels`` is an ``eqx.field(static=True)`` **tuple**, not an
      array leaf.  A store with a different channel partition but coincidentally
      equal mode bytes is a **different store**.
    * ⭐ The **address book** is what turns a request into codes.  A byte claim that
      covered the modes but not the book would be a claim about a component, not
      about the thing that answers a request — and the book is fitted on the live
      rows' *labels*, so it is precisely where a deleted value could survive.

    ⛔ Strictly stronger than the hash that already existed; nothing is loosened.
    """
    h = hashlib.sha256()
    h.update(b"chlu-gen-deletion/store-digest/2\n")
    for w in store.well_channels:
        h.update(np.asarray(w, np.int64).tobytes())
        h.update(b"|")
    h.update(b"\n")
    ph = param_hash(store)
    h.update(ph.encode())
    out = {
        "param_hash": ph,
        "well_sizes": [len(w) for w in store.well_channels],
        "n_modes_per_well": list(store.n_modes_per_well),
    }
    if book is not None:
        bh = hashlib.sha256()
        bh.update(np.ascontiguousarray(np.asarray(book.log_prior, np.float32)).tobytes())
        for lc in book.log_cond:
            a = np.ascontiguousarray(np.asarray(lc, np.float32))
            bh.update(str(a.shape).encode())
            bh.update(a.tobytes())
        out["book_hash"] = bh.hexdigest()
        h.update(out["book_hash"].encode())
    out["store_digest"] = h.hexdigest()
    if labels_vec is not None:
        out["labels_sha256"] = hashlib.sha256(
            np.asarray(labels_vec, np.int64).tobytes()
        ).hexdigest()
    return out


# ---------------------------------------------------------------------------
# 2. The delete predicate — over factor tuples, identical for every arm
# ---------------------------------------------------------------------------


def factor_tuples(grid, rows: np.ndarray) -> np.ndarray:
    """``(n, F)`` integer factor tuple per row, from the dataset grid."""
    from ..eval.gen_probe import factor_labels

    return np.asarray(factor_labels(grid, np.asarray(rows, np.int64)), np.int64)


def row_predicate(tuples: np.ndarray, spec: Sequence[tuple]) -> np.ndarray:
    """Boolean mask: ``True`` where the row matches **every** ``(factor, value)``.

    ``spec = [(0, 3)]`` is the factor-value delete *floor_hue == 3*.
    ``spec = [(2, 0), (4, 1)]`` is the conjunction *object_hue == 0 AND shape == 1*.
    ⭐ One predicate, so the rivals cannot be given a different deletion by accident.
    """
    tuples = np.asarray(tuples, np.int64)
    mask = np.ones(tuples.shape[0], bool)
    for f, v in spec:
        mask &= tuples[:, int(f)] == int(v)
    return mask


def delete_rows(live: np.ndarray, tuples: np.ndarray, spec: Sequence[tuple]) -> dict:
    """``live \\ rows(spec)``, kept **sorted** so the survivor set is canonical."""
    live = np.asarray(live, np.int64)
    mask = row_predicate(tuples, spec)
    kept = np.sort(live[~mask])
    return {
        "live": kept,
        "n_deleted": int(mask.sum()),
        "n_live": int(kept.size),
        "deleted_rows": np.sort(live[mask]),
    }


# ---------------------------------------------------------------------------
# 3. The canonical rebuild
# ---------------------------------------------------------------------------


def canonical_build(
    X_live: np.ndarray,
    cfg,
    seed: int,
    k: int,
    grouping: str,
    feat_shape,
    labels_live: Optional[np.ndarray] = None,
    cardinalities: Optional[Sequence[int]] = None,
    log: Optional[Callable] = None,
) -> dict:
    """Build a store as a **pure function of** ``(X_live, cfg, seed, k, grouping)``.

    ⛔ No state from any previous build enters — not an optimiser state, not a
    grouping, not a key.  The PRNG keys are derived from ``seed`` alone, exactly as
    :func:`chlu.experiments.exp_gen_store.run_arm` derives them, so a build here on
    the undeleted pool reproduces the shipped arm.

    ⚠ ``X_live`` MUST be gathered from the live rows **in sorted order** — the row
    order is part of the canonical input, and a permuted pool is a different store.
    """
    t0 = time.perf_counter()
    Xc = np.asarray(X_live[: int(cfg.n_corr)], np.float32)
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels_vec, ginfo = grouping_labels(
        corr,
        int(k),
        grouping,
        int(seed),
        cfg.grouping_method,
        dead=dead,
        min_well_frac=float(cfg.min_well_frac),
    )
    wc = well_channels_from_labels(labels_vec, int(k))
    t_group = time.perf_counter() - t0

    key = jax.random.PRNGKey(
        int(seed) * 10_007 + 20_260_911 + (0 if grouping == "correlation" else 1)
    )
    key, ik, tk = jax.random.split(key, 3)
    t1 = time.perf_counter()
    store = init_store(X_live, wc, int(cfg.n_modes), feat_shape, ik, seed=int(seed))
    t_init = time.perf_counter() - t1
    t2 = time.perf_counter()
    store, hist, key = train_store(store, X_live, cfg, tk, log=log)
    t_train = time.perf_counter() - t2

    # ---- the ADDRESSING HEAD, recanonicalised from the same live set ----------
    # ⭐ This is the half of the rebuild that carries the deleted value's *name*.
    # `fit_address_book` (gen_dream.py:313-318, smoothing a = 1.0) counts
    # P(mode_k = m | factor_f = v) over the live rows, so after `delete(v)` the row
    # for `v` is fitted on ZERO rows and gives `(0 + 1)/(0 + 1*M) = 1/M` — the
    # identical float64 scalar for every mode, i.e. **exactly uniform, bit-for-bit**.
    # No NaN, no division by zero.
    #
    # ⚠ PRECISION (this comment previously said the term "contributes nothing", which
    # is wrong and the wrong version would have reached the report): in
    # `address_requests` the score accumulates `lc[v] - lp` per asked factor, so for
    # the deleted `v` the bracket is `(log(1/M) - log P(m))`.  That is **constant in
    # `m`, so it cannot move an argmax by itself** — but it is NOT identical to
    # omitting the factor (the `R[:, f] = -1` path), because it still leaves the
    # score one extra `-log P(m)` lighter.
    # ⚠ And a request naming ONLY the deleted value scores `-log M` everywhere, so
    # `argmax` returns mode 0 in every well: a degenerate constant read.  The
    # deployed eval always sends the full 6-factor tuple, so the surviving five terms
    # dominate — which is exactly why the read still returns a real image, and why
    # ⛔ whether that image nevertheless shows `v` is the measured open question and
    # never an assumption.
    codes = book = None
    t3 = time.perf_counter()
    if labels_live is not None and cardinalities is not None:
        flat = np.asarray(X_live, np.float32).reshape((X_live.shape[0], -1))
        codes = np.asarray(address_codes(store, flat), np.int64)
        book = fit_address_book(codes, np.asarray(labels_live, np.int64),
                                [int(c) for c in cardinalities])
    t_book = time.perf_counter() - t3

    return {
        "store": store,
        "book": book,
        "codes": codes,
        "well_channels": wc,
        "labels": labels_vec,
        "grouping_info": ginfo,
        "history": hist,
        "digest": store_digest(store, labels_vec, book),
        "timing_s": {
            "grouping": t_group,
            "init": t_init,
            "train": t_train,
            "book": t_book,
            "total": time.perf_counter() - t0,
        },
        "build_inputs": {
            "n_live": int(X_live.shape[0]),
            "train_batch": int(min(int(cfg.train_batch), int(X_live.shape[0]))),
            "train_steps": int(cfg.train_steps),
            "n_corr": int(cfg.n_corr),
            "n_modes": int(cfg.n_modes),
            "k": int(k),
            "grouping": str(grouping),
            "seed": int(seed),
            "tau": [float(cfg.tau_start), float(cfg.tau_end)],
            "lr": float(cfg.train_lr),
            "commitment_beta": float(cfg.commitment_beta),
        },
    }
