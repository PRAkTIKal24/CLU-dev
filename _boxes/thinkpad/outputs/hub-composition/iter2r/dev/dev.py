# ⭐ INHERITED VERBATIM from hub/generation-table (predecessor, iteration 1):
#   .claude/outputs/hub-generation-table/dev/dev.py
# ⛔ NOT rebuilt (LAUNCH-hub-composition-2: the instrument is inherited and reused).
# ⛔ Not one line below is modified. hub/composition-2 adds its columns in its own modules.
"""hub/generation-table — DEV scoring harness.

⛔ Never touches ``eval_b``/``eval_a``.  Scores a Montero-SHAPED corner cut from
inside the train split, held out from the DEV store and the DEV address book.

Every call is a resumable unit: results are appended to a JSONL ledger keyed by a
deterministic hash of the configuration, and a repeat is served from the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

import ctx

OUT = Path(os.environ["CHLU_OUT"])
LEDGER = OUT / "dev" / "dev_ledger.jsonl"


def _key(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def ledger_lookup(k: str):
    if not LEDGER.exists():
        return None
    with open(LEDGER, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("key") == k:
                return r
    return None


def ledger_append(rec: dict):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------------------

class HeldOutOverlap(RuntimeError):
    """⛔ A DEV corner that touches eval_a / eval_b. Refused, never trimmed."""


DEV_CORNER = dict(hue_lo=0.2, scale_hi=True, shape_value=0.0, orient_hi=True)


def dev_setup(c, corner=None, log=ctx.log):
    """Rows, features and labels for a DEV run.  ⭐ The corner is removed from the pool."""
    from chlu.eval.gen_probe import factor_labels
    corner = dict(DEV_CORNER if corner is None else corner)
    v, grid = np.asarray(c["values"]), c["grid"]
    dev_rows = ctx.montero_like_corner(v, grid, **corner)
    assert dev_rows.size == 72, f"DEV corner is {dev_rows.size} rows, expected 72"
    # ⛔ B and A must never be inside the DEV pool or the DEV corner
    B = np.asarray(c["rows"]["eval_b"], np.int64)
    A = np.asarray(c["rows"]["eval_a"], np.int64)
    nb_ = int(np.intersect1d(dev_rows, B).size)
    na_ = int(np.intersect1d(dev_rows, A).size)
    if nb_ or na_:
        # ⭐ THE GUARD. A DEV corner that touches a held-out evaluation set is REFUSED,
        # ⛔ never trimmed: a trimmed corner is not a Montero-shaped cell any more.
        raise HeldOutOverlap(
            f"⛔ DEV corner {corner} intersects the held-out sets: eval_b {nb_} rows, "
            f"eval_a {na_} rows — REFUSED")
    pool = np.setdiff1d(c["pool"], dev_rows)
    return {
        "corner": corner, "dev_rows": dev_rows, "pool": pool,
        "n_removed_from_pool": int(c["pool"].size - pool.size),
        "lab_dev": factor_labels(grid, dev_rows),
    }


def build_store(X_store, feat_shape, K: int, M: int, seed: int, cfg, *,
                grouping_method="auto", train=True, log=ctx.log, train_steps=None):
    """A store at an arbitrary (K, M).  ⭐ bytes = M·D·4, independent of K."""
    from chlu.core.gen_store import (channel_correlation, correlation_grouping,
                                     dead_channels, init_store, store_bytes,
                                     train_store, well_channels_from_labels)
    gs = cfg.gen_store
    Xc = np.asarray(X_store, np.float32)[: int(gs.n_corr)]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    labels, ginfo = correlation_grouping(corr, int(K), grouping_method, dead=dead,
                                         min_well_frac=float(gs.min_well_frac),
                                         return_info=True)
    wc = well_channels_from_labels(labels, int(K))
    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911)
    key, ik, tk = jax.random.split(key, 3)
    st = init_store(X_store, wc, int(M), feat_shape, ik, seed=int(seed))
    hist = None
    if train:
        import copy
        g2 = copy.copy(gs)
        if train_steps is not None:
            g2.train_steps = int(train_steps)
        st, hist, key = train_store(st, X_store, g2, tk, log=None)
    return st, {"grouping": ginfo, "well_sizes": [len(w) for w in wc],
                "store_bytes": store_bytes(st),
                "loss_final": (float(hist[-1]) if hist else None)}


def score_features(c, feats, lab, *, batch=256):
    """arm features -> decoder -> external classifier -> (aggregate, distinct)."""
    from chlu.experiments.exp_gen_cell import distinct_counts, render, score_images
    imgs = render(c["decoder"], np.asarray(feats, np.float32), c["feat_shape"], int(batch))
    sc = score_images(c["external"], imgs, lab, c["names"], int(batch))
    di = distinct_counts(c["external"], np.asarray(feats, np.float32), imgs, int(batch))
    return {"aggregate": float(sc["aggregate"]), "per_factor": sc["per_factor"],
            "exact_match": float(sc["exact_match"]),
            "distinct_factor_tuples": int(di["factor_tuples"]),
            "distinct_feature_vectors": int(di["feature_vectors"]),
            "n": int(di["n"])}
