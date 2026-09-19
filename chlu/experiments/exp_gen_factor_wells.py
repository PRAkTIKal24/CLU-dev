"""hub/factor-wells -- the multi-corner runner for the seven-row scoreboard.

ONE path, identical for every arm::

    request -> request design Phi -> linear head -> store span -> THE frozen decoder
            -> image -> THE external classifier -> factor tuple

DIAL DECLARATION -- *how the store is ORGANISED* (the partition).  Every arm here shares
the same encoder, the same frozen feature cache, the same fit pool, the same mode count,
the same 524,288 B store ledger, the same 30-bit read channel, the same read verb, the
same head design, the same ridge, the same decoder and the same judge.  **The label
vector is the only thing that moves.**  Its laundering control
(``gen_store.random_grouping``) permutes that label vector holding the well-size
histogram exactly, so an arm's margin over its floor is attributable to which coordinate
sits in which well and to nothing else.

THE ESTIMAND -- w3-hub-common section 3 is binding: a single held-out cell decides
nothing.  Every verdict here is the **multi-corner paired estimand**: 49 ``nway-*``
corners for row 4 and 60 tiled ``pair-*`` sub-corners for row 3, paired corner by corner
against the control, with the margin declared in the PREREG before the first scored
number.

THE LIKE-FOR-LIKE -- ``basis_head``, the wave-2 object: a factored dictionary over frozen
encoder features plus a request-conditioned linear regression head, at matched bytes, on
the **shipped ward grouping**.  ``knn_proto`` is the second row.  A CLU mechanism that
ties basis+head is decoration and is reported as such, in those words --
``gen_scoreboard.verdict_vs_control`` writes that sentence itself.

Runnable::

    python -m chlu.experiments.exp_gen_factor_wells --stage partition --out DIR
    python -m chlu.experiments.exp_gen_factor_wells --stage fit  --arm ARM --out DIR
    python -m chlu.experiments.exp_gen_factor_wells --stage run  --arm ARM --out DIR
    python -m chlu.experiments.exp_gen_factor_wells --stage table --out DIR

Every stage is **resumable**: it writes one artifact per (stage, arm) and skips a stage
whose artifact is already on disk and complete.  Every stage takes ``--max-seconds`` and
honours it between corners, so a kill costs one corner and not the run.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 0. THE PINS -- the store fit is pinned COMPLETELY (w3-hub-common section 3)
# ---------------------------------------------------------------------------

PINS = {
    "dataset": "shapes3d",
    "pretext": "recon",
    "seed": 0,
    # "val" = the selection condition (charter A.2); "test" = the scored corners.
    "condition": "test",
    # -- the partition ------------------------------------------------------
    "K": 5,
    "n_modes": 64,
    "D": 2048,
    "feat_shape": (128, 4, 4),
    "store_bytes": 524_288,
    "bits_per_read": 30.0,
    "relevance_n_bins": 8,
    "relevance_quantile_subsample": 65_536,
    "merge_rule": "min_relevance_mass",
    "dead_channel_eps": 1e-8,
    "n_corr": 16_384,
    "grouping_method": "auto",
    "min_well_frac": 0.10,
    # -- the fit pool -------------------------------------------------------
    # ONE pinned RNG stream draws one ordered pool; the store fit takes its first
    # n_store_train rows and the head takes the whole pool.  Two declared sizes, one
    # stream, so the pool is reproducible from (seed, n_pool) alone.
    "pool_rng_seed": 20_260_919,
    "n_pool": 131_072,
    "n_store_train": 32_768,     # the shipped gen_store value, inherited verbatim
    # -- the store fit (gen_store defaults, inherited verbatim and pinned here) --
    "train_steps": 1200,
    "train_batch": 128,
    "train_lr": 0.003,
    "commitment_beta": 0.25,
    "tau_start": 1.0,
    "tau_end": 0.1,
    "store_key_seed": 0,
    # Lloyd budget: the shipped path is k-means++ SEEDING ONLY (gen_store._kmeanspp_init)
    # followed by adam on the soft-occupancy objective.  There are no Lloyd sweeps and
    # the budget is pinned at zero so that is on the record rather than inferred.
    "lloyd_iters": 0,
    # -- the head -----------------------------------------------------------
    "head_ridge": 1.0,
    "head_pair_block": True,
    # -- rendering ----------------------------------------------------------
    "eval_batch": 256,
    # -- row 3's tiling -----------------------------------------------------
    "pair_tile_hue_band": 2,
    "pair_tile_scale_corner": 3,
    "pair_tile_orientation_corner": 3,
}

#: Every arm on the table.  ``basis_head`` FIRST -- it is the like-for-like.
ARMS = (
    "basis_head",              # the wave-2 object on the shipped ward grouping
    "factor_aligned",          # wells carry a factor, modes are its values (argmax)
    "factor_aligned_balanced",  # THE arm: the same, rebalanced to equal wells
    "basis_head_k1",           # the strict unfactored reading; DECLARED not K-matched
    "random_dmatched_ward",    # laundering floor for basis_head
    "random_dmatched_factor",  # laundering floor for factor_aligned
    "random_dmatched_balanced",  # laundering floor for factor_aligned_balanced
    "landscape_removed",       # charter section C.2 -- atoms zeroed, must collapse
    "landscape_shuffled",      # charter section C.2 -- atoms kept, addressing broken
    "lookup_argmax",           # charter section C.1/C.4 -- one atom per well
    "lookup_convex",           # charter section C.1 -- the mixture, simplex imposed
    "knn_proto",               # w3-hub-common section 1 -- the second row
    "factor_aligned_nopair",   # the ablation the target file demands BEFORE any scored run
    "basis_head_nopair",       # its matched control
    "balanced_nopair",         # the ablation on the balanced arm
)

#: Which partition each arm deploys.  ``None`` means the arm does not deploy a store.
ARM_PARTITION = {
    "basis_head": "ward",
    "factor_aligned": "factor",
    "factor_aligned_balanced": "balanced",
    "basis_head_k1": "k1",
    "random_dmatched_ward": "random_ward",
    "random_dmatched_factor": "random_factor",
    "random_dmatched_balanced": "random_balanced",
    "landscape_removed": "balanced",
    "landscape_shuffled": "balanced",
    "lookup_argmax": "balanced",
    "lookup_convex": "balanced",
    "knn_proto": None,
    "factor_aligned_nopair": "factor",
    "basis_head_nopair": "ward",
    "balanced_nopair": "balanced",
}

ROW_IDS = (
    "r1_seen", "r2_unseen_valid", "r3_comp_2way", "r4_comp_nway",
    "r5_comp_refusal", "r6_invalid_refusal", "r7_deletion",
)


def _now() -> float:
    return time.time()


def _jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------------------
# 1. The frozen instrument and the value-holdout condition
# ---------------------------------------------------------------------------


def data_root() -> Path:
    return Path(os.environ["CHLU_GEN_DATA_ROOT"]) / PINS["dataset"]


def bridge_seed_dir(seed: int) -> Path:
    return (Path(os.environ["CHLU_GEN_BRIDGE_ROOT"])
            / f"{PINS['dataset']}_{PINS['pretext']}" / f"seed_{int(seed)}")


def classifier_dir(seed: int) -> Path:
    # the external classifier lives beside the frozen bridge, in the same outputs tree
    return (Path(os.environ["CHLU_GEN_BRIDGE_ROOT"]).parent.parent
            / "gen-external-classifier" / f"seed_{int(seed)}")


def store_plan_path() -> Path:
    return (Path(os.environ["CHLU_GEN_STORE_RUNS"])
            / f"{PINS['dataset']}_{PINS['pretext']}" / "store_plan.json")


def build_grid():
    """The FactorGrid, from the condition's own factor index -- never regenerated."""
    from ..data.gen_splits import FactorGrid

    z = np.load(data_root() / "value_holdout" / "factor_index.npz", allow_pickle=True)
    names = tuple(str(x) for x in z["factor_names"])
    values = tuple(np.asarray(z["values__" + n], np.float64) for n in names)
    return FactorGrid(names=names, values=values, dataset=PINS["dataset"]), z


def load_condition() -> dict:
    """train/test rows, the cells, and the query sets of the value-holdout condition."""
    vh = data_root() / "value_holdout"
    grid, z = build_grid()
    with open(vh / "value_cells.json", encoding="utf-8") as fh:
        cells = json.load(fh)
    qs = {}
    for name in ("row1_seen", "row2_unseen", "row5_held_out", "c_invalid",
                 "a7_redundant"):
        p = vh / f"{name}.json"
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                qs[name] = json.load(fh)
    return {
        "grid": grid,
        "train_rows": np.asarray(z["train_rows"], np.int64),
        "test_rows": np.asarray(z["test_rows"], np.int64),
        "cells": cells,
        "query_sets": qs,
        "scope": json.loads(str(z["meta"]))["scope"],
    }


def features_mmap(seed: int):
    return np.load(bridge_seed_dir(seed) / "features.npy", mmap_mode="r")


def gather(F, rows: np.ndarray, chunk: int = 16_384) -> np.ndarray:
    """rows -> (n, D) float32.  Chunked so a big pool never doubles in memory."""
    r = np.asarray(rows, np.int64)
    out = np.empty((r.size, int(np.prod(F.shape[1:]))), np.float32)
    for s in range(0, r.size, chunk):
        e = min(s + chunk, r.size)
        out[s:e] = np.asarray(F[r[s:e]], np.float32).reshape((e - s, -1))
    return out


def fit_pool(train_rows: np.ndarray, n_pool: int, rng_seed: int,
             exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """The pinned fit pool: ONE ordered draw from the value-holdout train index.

    The store fit takes this pool's first ``n_store_train`` entries and the head takes
    all of it, so both are reproducible from ``(rng_seed, n_pool, exclude)`` alone.
    Under the ``val`` condition ``exclude`` is the val corners' rows, so the store has
    never held a row of the set every mechanism choice is made on.
    """
    rng = np.random.default_rng(int(rng_seed))
    tr = np.asarray(train_rows, np.int64)
    if exclude is not None and np.asarray(exclude).size:
        tr = np.setdiff1d(tr, np.asarray(exclude, np.int64))
    n = int(min(int(n_pool), tr.size))
    return tr[rng.permutation(tr.size)[:n]]


# ---------------------------------------------------------------------------
# 2. The corners -- the multi-corner estimand
# ---------------------------------------------------------------------------


def _cell_mask_rows(grid, values: dict) -> np.ndarray:
    """All dataset rows realising a cell's value constraints."""
    cards = tuple(int(c) for c in grid.cardinalities)
    total = int(np.prod(cards))
    L = np.stack(np.unravel_index(np.arange(total), cards), axis=-1)
    m = np.ones(total, bool)
    for f, name in enumerate(grid.names):
        if name in values:
            want = [int(np.argmin(np.abs(grid.values[f] - float(v))))
                    for v in values[name]]
            m &= np.isin(L[:, f], want)
    return np.flatnonzero(m)


#: How many VAL corners to carve out of the unused n-way slots.  The condition's own
#: builder had capacity for 120 structurally-disjoint Montero slots and emitted 49; the
#: rest were never emitted, so their rows sit in the value-holdout TRAIN index.  Taking
#: 24 of them and removing them from the fit pool gives a selection set with the SAME
#: geometry as the scored corners that has never been scored.
N_VAL_CORNERS = 24
VAL_SLOT_OFFSET = 49


def _montero_slots(cond) -> list:
    """The condition's own n-way slot enumeration, in its own odometer order."""
    from ..data.gen_value_holdout import ValueHoldoutConfig, build_nway_cells
    from ..data.gen_splits import FactorCell

    grid = cond["grid"]
    cells = cond["cells"]
    cfg = ValueHoldoutConfig(**{k: v for k, v in cells["config"].items()
                                if k in {f.name for f in
                                         __import__("dataclasses").fields(ValueHoldoutConfig)}})
    pair = [FactorCell(cell_id=c["cell_id"],
                       values={k: tuple(float(x) for x in v)
                               for k, v in c["values"].items()},
                       meta=c["meta"])
            for c in cells["cells"] if c["meta"]["family"] == "pair"]
    import dataclasses as _dc
    cfg = _dc.replace(cfg, n_nway_cells=VAL_SLOT_OFFSET + N_VAL_CORNERS)
    return build_nway_cells(grid, cfg, pair_cells=pair)


def val_corners(cond) -> list:
    """The SELECTION set: n-way slots the condition never emitted (charter section A.2).

    Every mechanism choice on this hub -- which partition, the balanced variant, the
    pair-interaction ablation -- is made HERE and nowhere else.  These corners live in
    the value-holdout train index, so they are removed from the fit pool by
    :func:`fit_pool` whenever the condition is ``val``; the store has then never held a
    row of them and they are a real hold-out that costs no scored run.
    """
    grid = cond["grid"]
    slots = _montero_slots(cond)[VAL_SLOT_OFFSET:VAL_SLOT_OFFSET + N_VAL_CORNERS]
    out = []
    for c in slots:
        rows = _cell_mask_rows(grid, dict(c.values))
        out.append({"corner_id": f"val-{c.cell_id}", "family": "val_nway",
                    "rows": rows, "meta": dict(c.meta)})
    return out


def val_corner_rows(cond) -> np.ndarray:
    return np.unique(np.concatenate([c["rows"] for c in val_corners(cond)]))


def nway_corners(cond) -> list:
    """Row 4: the 49 Montero-shaped corners the condition already built."""
    grid = cond["grid"]
    out = []
    for c in cond["cells"]["cells"]:
        if c["meta"]["family"] != "nway":
            continue
        rows = _cell_mask_rows(grid, c["values"])
        if rows.size != int(c["n_rows_realised"]):
            raise AssertionError(
                f"{c['cell_id']}: realised {rows.size} rows, the condition recorded "
                f"{c['n_rows_realised']} -- the corner is not the one that was built"
            )
        out.append({"corner_id": c["cell_id"], "family": "nway", "rows": rows,
                    "meta": c["meta"]})
    return out


def pair_subcorners(cond) -> list:
    """Row 3: each ``pair-*`` cell TILED into Montero-shaped sub-corners.

    The 3 pair cells are only 3 corners and w3-hub-common section 3 demands at least 49.
    The tiling uses the **same widths the n-way builder uses** -- a band of 2 consecutive
    floor/wall hues, a corner of 3 consecutive scales, a corner of 3 consecutive
    orientations -- with ``object_hue`` and ``shape`` pinned by the pair itself.  The
    sub-corners are pairwise disjoint by the n-way builder's own argument: they are
    enumerated over a product of pairwise-disjoint coordinate options, so any two differ
    in at least one coordinate whose value sets do not intersect.
    """
    grid = cond["grid"]
    bw = int(PINS["pair_tile_hue_band"])
    sw = int(PINS["pair_tile_scale_corner"])
    rw = int(PINS["pair_tile_orientation_corner"])
    fh = grid.values[grid.index_of("floor_hue")]
    sc = grid.values[grid.index_of("scale")]
    ro = grid.values[grid.index_of("orientation")]
    floor_windows = [np.arange(i, i + bw) for i in range(0, fh.size - bw + 1, bw)]
    corners = []
    for c in cond["cells"]["cells"]:
        if c["meta"]["family"] != "pair":
            continue
        for fi, win in enumerate(floor_windows):
            for scn in ("high", "low"):
                for rcn in ("high", "low"):
                    vals = dict(c["values"])
                    band = tuple(float(x) for x in fh[win])
                    vals["floor_hue"] = band
                    vals["wall_hue"] = band
                    vals["scale"] = tuple(
                        float(x) for x in (sc[-sw:] if scn == "high" else sc[:sw]))
                    vals["orientation"] = tuple(
                        float(x) for x in (ro[-rw:] if rcn == "high" else ro[:rw]))
                    rows = _cell_mask_rows(grid, vals)
                    if rows.size == 0:
                        continue
                    corners.append({
                        "corner_id": f"{c['cell_id']}/f{fi}-{scn[0]}{rcn[0]}",
                        "family": "pair",
                        "rows": rows,
                        "meta": {"parent": c["cell_id"], "floor_band_start_index": int(win[0]),
                                 "scale_corner": scn, "orientation_corner": rcn,
                                 "pair": c["meta"]["pair"], "tiled_by": "hub/factor-wells",
                                 "widths": {"hue_band": bw, "scale_corner": sw,
                                            "orientation_corner": rw}},
                    })
    return corners


def assert_corners_disjoint_and_heldout(corners, test_rows, label: str) -> dict:
    """Structural proof, run before any number: disjoint, and inside the hold-out."""
    seen = np.concatenate([c["rows"] for c in corners]) if corners else np.zeros(0, np.int64)
    uniq = np.unique(seen)
    if uniq.size != seen.size:
        raise AssertionError(
            f"{label}: the corners overlap ({seen.size - uniq.size} shared rows)")
    held = np.setdiff1d(uniq, np.asarray(test_rows, np.int64))
    if held.size:
        raise AssertionError(
            f"{label}: {held.size} corner rows are NOT in the hold-out "
            f"(e.g. {held[:8].tolist()}) -- the corner is partly in the fit pool")
    return {"n_corners": len(corners), "n_rows": int(seen.size),
            "pairwise_disjoint": True, "all_rows_held_out": True,
            "rows_per_corner": sorted({int(c["rows"].size) for c in corners})}


# ---------------------------------------------------------------------------
# 3. The partitions -- MY DIAL
# ---------------------------------------------------------------------------


def build_partitions(out_dir: Path, log: Callable = print) -> dict:
    """Every arm's label vector, plus the label-path audit.  Resumable."""
    from ..core.gen_factor_partition import (
        assert_relevance_is_train_only, channel_factor_relevance,
        factor_aligned_balanced_labels, factor_aligned_labels, label_path_audit,
        well_rank_headroom,
    )
    from ..core.gen_store import (
        adjusted_rand_index, block_diagonality, channel_correlation, correlation_grouping,
        dead_channels, random_grouping,
    )

    art = out_dir / "partitions.json"
    npz = out_dir / "partitions.npz"
    if art.exists() and npz.exists():
        log("  partitions: already on disk, skipping")
        with open(art, encoding="utf-8") as fh:
            return {"info": json.load(fh), "labels": dict(np.load(npz))}

    cond = load_condition()
    grid = cond["grid"]
    seed = int(PINS["seed"])
    F = features_mmap(seed)
    excl = val_corner_rows(cond) if PINS["condition"] == "val" else None
    pool = fit_pool(cond["train_rows"], PINS["n_pool"], PINS["pool_rng_seed"],
                    exclude=excl)
    held = (np.union1d(cond["test_rows"], excl) if excl is not None
            else cond["test_rows"])
    disj = assert_relevance_is_train_only(pool, held)
    log(f"  fit pool {pool.size} rows; disjoint from the hold-out: {disj['intersection']} shared")

    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    dead = dead_channels(X[: int(PINS["n_corr"])], eps=float(PINS["dead_channel_eps"]))
    log(f"  dead channels: {int(dead.sum())}")

    t = _now()
    rel = channel_factor_relevance(
        X, Y, cards, n_bins=int(PINS["relevance_n_bins"]), seed=seed)
    R = rel["mi_bits"]
    log(f"  channel-factor relevance: {R.shape} in {_now() - t:.1f}s; "
        f"mean top-factor MI {float(R.max(axis=1).mean()):.4f} bits")

    lab_factor, info_factor = factor_aligned_labels(
        R, int(PINS["K"]), dead=dead, merge_rule=str(PINS["merge_rule"]),
        factor_names=list(grid.names))
    corr = channel_correlation(X[: int(PINS["n_corr"])])
    lab_ward, info_ward = correlation_grouping(
        corr, int(PINS["K"]), method=str(PINS["grouping_method"]), dead=dead,
        min_well_frac=float(PINS["min_well_frac"]), return_info=True)
    lab_bal, info_bal = factor_aligned_balanced_labels(
        R, int(PINS["K"]), dead=dead, merge_rule=str(PINS["merge_rule"]),
        factor_names=list(grid.names))
    lab_rw = random_grouping(lab_ward, seed)
    lab_rf = random_grouping(lab_factor, seed)
    lab_rb = random_grouping(lab_bal, seed)
    lab_k1 = np.zeros(int(PINS["D"]), np.int64)

    labels = {"factor": lab_factor, "balanced": lab_bal, "ward": lab_ward,
              "random_ward": lab_rw, "random_factor": lab_rf,
              "random_balanced": lab_rb, "k1": lab_k1}
    info = {
        "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PINS.items()},
        "fit_pool": {"n": int(pool.size), "rng_seed": int(PINS["pool_rng_seed"]),
                     "sha_first8": [int(x) for x in pool[:8]]},
        "condition": PINS["condition"],
        "disjointness": disj,
        "relevance": {k: v for k, v in rel.items() if k != "mi_bits"},
        "relevance_top_factor_share": {
            str(grid.names[f]): float((np.argmax(R, axis=1) == f).mean())
            for f in range(len(cards))},
        "factor_aligned": info_factor,
        "factor_aligned_balanced": info_bal,
        "ward": info_ward,
        "label_path_audit": label_path_audit(rel, info_factor, disj),
        "agreement_ARI": {
            "factor_vs_ward": float(adjusted_rand_index(lab_factor, lab_ward)),
            "balanced_vs_ward": float(adjusted_rand_index(lab_bal, lab_ward)),
            "balanced_vs_factor": float(adjusted_rand_index(lab_bal, lab_factor)),
        },
        "block_diagonality": {
            k: block_diagonality(corr, v) for k, v in labels.items() if k != "k1"
        },
        "rank_headroom": {
            k: well_rank_headroom(
                np.bincount(v, minlength=int(v.max()) + 1).tolist(), int(PINS["n_modes"]))
            for k, v in labels.items()},
        "well_sizes": {k: np.bincount(v).tolist() for k, v in labels.items()},
    }
    _jdump(info, art)
    np.savez_compressed(npz, **{k: v for k, v in labels.items()},
                        mi_bits=R, pool=pool, dead=dead)
    log(f"  partitions written -> {art}")
    return {"info": info, "labels": labels}


# ---------------------------------------------------------------------------
# 4. The arms -- store fit + head fit, pinned
# ---------------------------------------------------------------------------


def train_store_traced_tau(store, X: np.ndarray, key, log: Optional[Callable] = None):
    """``gen_store.train_store``'s objective, optimiser, anneal and RNG stream exactly --
    with ``tau`` passed as a **traced** scalar instead of a Python float.

    MEASURED, not guessed: the shipped ``train_store`` hands ``tau`` to an
    ``eqx.filter_jit``-ed step as a Python float, which ``filter_jit`` treats as STATIC.
    ``tau`` moves on every one of the 1200 steps, so the step function is recompiled
    1200 times.  Measured on this box: **1.33 s/step static vs 0.035 s/step traced, a
    38x difference** -- 27 minutes per arm against 42 seconds, and this hub fits 14 arms.

    Everything that touches a number is unchanged: the same loss (soft occupancy in GRAM
    form plus the commitment pull), the same ``optax.adam(lr)``, the same geometric
    anneal ``tau_start -> tau_end``, the same ``jax.random.split`` /
    ``jax.random.choice(..., replace=False)`` minibatch stream from the same key.  The
    only difference is where ``tau`` enters the compiled graph, and it is identical for
    EVERY arm on this hub, so it is a constant of the comparison.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from ..core.gen_store import _well_views

    flat = jnp.asarray(np.asarray(X, np.float32).reshape((X.shape[0], -1)))
    n = int(flat.shape[0])
    batch = min(int(PINS["train_batch"]), n)
    steps = int(PINS["train_steps"])
    beta = float(PINS["commitment_beta"])
    t0, t1 = float(PINS["tau_start"]), float(PINS["tau_end"])
    opt = optax.adam(float(PINS["train_lr"]))
    opt_state = opt.init(eqx.filter(store, eqx.is_inexact_array))

    def loss_fn(st, xb, tau):
        views = _well_views(st, xb)
        soft = 0.0
        commit = 0.0
        for z, m in zip(views, st.modes, strict=True):
            d2 = (jnp.sum(z * z, axis=-1)[:, None]
                  - 2.0 * (z @ m.T)
                  + jnp.sum(m * m, axis=-1)[None, :])
            pi = jax.nn.softmax(-d2 / (2.0 * tau ** 2), axis=-1)
            soft = soft + jnp.mean(jnp.sum((z - pi @ m) ** 2, axis=-1))
            commit = commit + jnp.mean(jnp.min(d2, axis=-1))
        return (soft + beta * commit) / len(st.modes)

    grad_fn = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(st, s, xb, tau):
        loss, grads = grad_fn(st, xb, tau)
        updates, s = opt.update(grads, s, st)
        return eqx.apply_updates(st, updates), s, loss

    history = []
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n, (batch,), replace=False)
        tau = jnp.float32(t0 * (t1 / t0) ** (i / max(1, steps - 1)))
        store, opt_state, loss = step(store, opt_state, flat[idx], tau)
        history.append(float(loss))
        if log is not None and (i == 0 or (i + 1) % max(1, steps // 4) == 0):
            log(f"    step {i + 1}/{steps}  loss={history[-1]:.5f}  tau={float(tau):.3f}")
    return store, history, key


def _store_for_labels(labels: np.ndarray, X_store: np.ndarray, log: Callable):
    """init_store (k-means++ seeding) + the pinned adam fit.  Both fully pinned."""
    import jax
    from ..core.gen_store import init_store, well_channels_from_labels

    k = int(labels.max()) + 1
    wc = well_channels_from_labels(labels, k)
    key = jax.random.PRNGKey(int(PINS["store_key_seed"]))
    st = init_store(X_store, wc, int(PINS["n_modes"]), tuple(PINS["feat_shape"]),
                    key, int(PINS["seed"]))
    st, hist, _ = train_store_traced_tau(st, X_store, key, log=log)
    return st, hist


def build_arm(arm: str, out_dir: Path, log: Callable = print) -> dict:
    """Fit one arm's store and head.  Resumable: skips if the artifact exists."""
    import equinox as eqx
    from ..core.gen_request_head import (
        HeadSpec, accumulate_gram, assert_head_request_only, head_bytes,
        request_design, solve_head,
    )
    from ..core.gen_store import (
        delete_store, reconstruction_mse, store_bytes,
    )

    arm_dir = out_dir / "arms" / arm
    done = arm_dir / "arm.pkl"
    if done.exists():
        log(f"  arm {arm}: already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cond = load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    parts = build_partitions(out_dir, log=log)
    pool = np.asarray(np.load(out_dir / "partitions.npz")["pool"], np.int64)
    F = features_mmap(int(PINS["seed"]))

    pair_block = not arm.endswith("_nopair")
    spec = HeadSpec(cards, pair_block=pair_block, ridge=float(PINS["head_ridge"]))

    t0 = _now()
    if arm == "knn_proto":
        from ..core.gen_request import FactorPrototypes
        X = gather(F, pool)
        protos = FactorPrototypes.fit(grid, X, pool)
        led = protos.byte_ledger()
        obj = {
            "arm": arm, "kind": "knn_proto", "protos": protos,
            "byte_ledger": {
                "deployed_bytes": int(led["bytes"]),
                "budget_bytes": int(PINS["store_bytes"]),
                "what": "per-(factor,value) mean train features + the global mean",
                "budget_note": (
                    "DECLARED: the prototype table is "
                    f"{int(led['bytes']):,} B against a {int(PINS['store_bytes']):,} B "
                    "budget -- it is not byte-matched, it is the second-row rival, and "
                    "the shipped knn_proto configuration was SELECTED at the retired "
                    "688,128 B budget (~31% advantage), which is the conservative "
                    "direction and is declared here rather than in a footnote."),
            },
            "head_bytes": {"head_bytes": 0, "note": "the prototype table IS its head"},
            "fit_seconds": _now() - t0,
            "prov": {"n_rows_fit": int(protos.n_rows_fit)},
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        with open(done, "wb") as fh:
            pickle.dump(obj, fh)
        _jdump({k: v for k, v in obj.items() if k != "protos"}, arm_dir / "arm.json")
        return obj

    part = ARM_PARTITION[arm]
    labels = np.asarray(parts["labels"][part], np.int64)
    X_store = gather(F, pool[: int(PINS["n_store_train"])])
    log(f"  arm {arm}: partition={part} sizes={np.bincount(labels).tolist()}")
    store, hist = _store_for_labels(labels, X_store, log=log)
    mse = reconstruction_mse(store, X_store)
    led = store_bytes(store)
    if int(led["store_bytes"]) != int(PINS["store_bytes"]):
        raise AssertionError(
            f"{arm}: store is {led['store_bytes']} B, the pinned ledger is "
            f"{PINS['store_bytes']} B -- the byte ledger has moved")
    del X_store

    if arm == "landscape_shuffled":
        pass          # the shuffle is applied AFTER the head is solved, below
    if arm == "landscape_removed":
        # charter section C.2: the laundering control. Atoms zeroed; the ledger is
        # UNCHANGED (the bytes are still deployed), so the row stays byte-matched and
        # the collapse is attributable to the atoms and not to a smaller object.
        store = delete_store(store, mode="zero", seed=int(PINS["seed"]))

    # -- the head: one Gram pass over the whole pool, shared shape across arms ----
    A_p = out_dir / f"gram_{'pair' if pair_block else 'nopair'}.npz"
    if A_p.exists():
        z = np.load(A_p)
        A, Mx, n_head = z["A"], z["Mx"], int(z["n"])
        log(f"  gram: reused from {A_p.name}")
    else:
        # ONE pass over the pinned pool. A and Mx are independent of the PARTITION, so
        # one accumulation serves every arm of the same head shape -- which is also the
        # proof that every arm's head was fitted from the same rows with the same design.
        A = np.zeros((spec.width, spec.width), np.float64)
        Mx = np.zeros((spec.width, int(PINS["D"])), np.float64)
        n_head = 0
        for s in range(0, pool.size, 8192):
            e = min(s + 8192, pool.size)
            rows = pool[s:e]
            Phi, _ = request_design(grid.values_of(rows), grid.values, spec)
            Z = gather(F, rows).astype(np.float64)
            P = Phi.astype(np.float64)
            A += P.T @ P
            Mx += P.T @ Z
            n_head += int(P.shape[0])
        np.savez_compressed(A_p, A=A, Mx=Mx, n=n_head)
        log(f"  gram: accumulated over {n_head} rows -> {A_p.name}")

    heads = solve_head(A, Mx, store, float(PINS["head_ridge"]))
    if arm == "landscape_shuffled":
        # charter section C.2, the INFORMATIVE half: the atoms are the same atoms and the
        # ledger is byte-identical; only which mode index names which atom is permuted,
        # AFTER the head was solved for the unpermuted store. The reachable span is
        # unchanged, so this control isolates the ADDRESSING and not the capacity.
        rng = np.random.default_rng(int(PINS["seed"]) * 7919 + 20_260_919)
        store = type(store)(
            store.well_channels, int(PINS["n_modes"]), store.feat_shape,
            __import__("jax").random.PRNGKey(0),
            init=[np.asarray(G)[rng.permutation(np.asarray(G).shape[0])]
                  for G in store.modes])
    hb = head_bytes(heads)
    Phi0, _ = request_design(grid.values_of(pool[:64]), grid.values, spec)
    req_only = assert_head_request_only(heads, store, Phi0, rng_seed=int(PINS["seed"]))

    obj = {
        "arm": arm, "kind": "span", "partition": part, "labels": labels,
        "store": store, "heads": heads, "spec": spec,
        "read_mode": ("argmax" if arm == "lookup_argmax"
                      else "convex" if arm == "lookup_convex" else "linear"),
        "byte_ledger": {
            "deployed_bytes": int(led["store_bytes"]),
            "budget_bytes": int(PINS["store_bytes"]),
            "what": f"{led['n_wells']} wells x {PINS['n_modes']} modes, float32",
        },
        "head_bytes": hb,
        "store_ledger": led,
        "recon_mse": mse,
        "train_history_tail": [float(x) for x in hist[-5:]],
        "request_only": req_only,
        "fit_seconds": _now() - t0,
        "prov": {"n_store_train": int(PINS["n_store_train"]), "n_head_rows": n_head,
                 "well_sizes": np.bincount(labels).tolist()},
    }
    arm_dir.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items()
            if k not in ("store", "heads", "spec", "labels", "protos")},
           arm_dir / "arm.json")
    log(f"  arm {arm}: fitted in {obj['fit_seconds']:.1f}s; store "
        f"{obj['byte_ledger']['deployed_bytes']:,} B + head {hb['head_bytes']:,} B")
    return obj


# ---------------------------------------------------------------------------
# 5. The read, the render and the judge
# ---------------------------------------------------------------------------


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """(N, F) requested VALUES -> (N, D) features + the out-of-range mask.

    No feature, no row index and no label enters: a request is a factor assignment.
    """
    from ..core.gen_request_head import head_coefficients, head_read, request_design

    if obj["kind"] == "knn_proto":
        reqs = []
        oor = np.zeros(values.shape[0], bool)
        for i in range(values.shape[0]):
            terms = []
            for f in range(values.shape[1]):
                v = values[i, f]
                if np.isnan(v):
                    continue
                terms.append({"factor_index": f, "value": float(v)})
                gv = grid.values[f]
                if v < gv[0] or v > gv[-1]:
                    oor[i] = True
            reqs.append(terms)
        return obj["protos"].embed_many(reqs), oor

    Phi, oor = request_design(values, grid.values, obj["spec"])
    mode = obj["read_mode"]
    if mode == "linear":
        return head_read(obj["heads"], obj["store"], Phi), oor
    # the two matched-bytes lookups over the SAME atoms (charter section C.1/C.4)
    store = obj["store"]
    C = head_coefficients(obj["heads"], Phi)
    out = np.zeros((Phi.shape[0], int(store.dim)), np.float32)
    for c, w, G in zip(C, store.well_channels, store.modes):
        Gm = np.asarray(G, np.float32)
        if mode == "argmax":
            out[:, np.asarray(w, np.int64)] = Gm[np.argmax(c, axis=1)]
        else:  # convex: the mixture, simplex imposed -- section 4.5's rejected object
            p = np.maximum(c, 0.0)
            s = p.sum(axis=1, keepdims=True)
            p = np.where(s > 0, p / np.where(s > 0, s, 1.0), 1.0 / p.shape[1])
            out[:, np.asarray(w, np.int64)] = p @ Gm
    return out, oor


class Judge:
    """THE frozen decoder + THE hash-asserted external classifier, loaded once."""

    def __init__(self, seed: int, log: Callable = print):
        from ..core.gen_bridge import param_hash
        from ..utils.checkpoints import load_model

        wroot = Path(os.environ["CHLU_GEN_DECLIFT_DIR"])
        self.decoder = load_model(wroot / f"decoder_v1_s{int(seed)}.pkl")
        with open(wroot / "winner_hashes.json", encoding="utf-8") as fh:
            wh = json.load(fh)
        exp = wh[f"seed_{int(seed)}"]["param_hash"]
        got = param_hash(self.decoder)
        if got != exp:
            raise AssertionError(
                f"decoder hash {got[:16]} != the winner manifest's {exp[:16]} -- the "
                "decoder on disk is not the one that was gated")
        cdir = classifier_dir(seed)
        self.external = load_model(cdir / "external_classifier.pkl")
        with open(cdir / "classifier_manifest.json", encoding="utf-8") as fh:
            cman = json.load(fh)
        cexp = cman["classifier"]["param_hash"]
        cgot = param_hash(self.external)
        if cgot != cexp:
            raise AssertionError(
                f"external classifier hash {cgot[:16]} != its manifest's {cexp[:16]}")
        self.info = {
            "decoder_param_hash": got, "decoder_recipe": wh[f"seed_{int(seed)}"]["recipe"]["name"],
            "classifier_param_hash": cgot,
            "classifier_gate_balacc_on_real":
                ((cman.get("gate") or {}).get("external") or {}).get("balanced_accuracy"),
            "note": "ONE instrument for every arm, loaded once and hash-asserted",
        }
        log(f"  judge: decoder {got[:12]} ({self.info['decoder_recipe']}) + "
            f"classifier {cgot[:12]}")

    def read_back(self, feats: np.ndarray, batch: int = 256) -> np.ndarray:
        from ..eval.gen_probe import decode_features, predict_factors
        Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(PINS["feat_shape"]))
        out = np.empty((Z.shape[0], 6), np.int64)
        for s in range(0, Z.shape[0], batch):
            e = min(s + batch, Z.shape[0])
            imgs = decode_features(self.decoder, Z[s:e], batch)
            out[s:e] = np.asarray(predict_factors(self.external, imgs, batch), np.int64)
        return out


# ---------------------------------------------------------------------------
# 6. Rows, outcomes and the paired estimand
# ---------------------------------------------------------------------------


def _request_values_from_rows(grid, rows: np.ndarray) -> np.ndarray:
    return grid.values_of(np.asarray(rows, np.int64))


def _records_to_values(records: Sequence[dict], grid) -> tuple:
    """A query-set record list -> (N, F) requested values + a duplicate-factor mask.

    A duplicated factor cannot be expressed as one value; the SECOND term is kept and
    the row is flagged, so the head sees a well-formed request and the arm's refusal
    rule is the thing that has to notice, not the encoder.
    """
    n, Fn = len(records), grid.n_factors
    vals = np.full((n, Fn), np.nan, np.float64)
    dup = np.zeros(n, bool)
    for i, rec in enumerate(records):
        seen = set()
        for term in rec["request"]:
            k = int(term["factor_index"])
            if k in seen:
                dup[i] = True
            seen.add(k)
            vals[i, k] = float(term["value"])
    return vals, dup


def value_presence_bitmap(grid, train_rows: np.ndarray) -> list:
    """Which (factor, value) the store's fit pool actually holds.

    w3-hub-common section 7 measured this exact object: an 8-byte value-presence bitmap
    plus a free bracket check accepts rows 1 and 2 and refuses rows 5 and 6 at 1.0000,
    against a 524,288 B budget.  It is built here, reported as the free thing it is, and
    it is the SAME acceptance rule for every arm -- so the refusal rows compare arms at
    an identical threshold and nobody's refusal number is quoted as evidence.
    """
    fi = grid.factor_indices_of(np.asarray(train_rows, np.int64))
    return [np.isin(np.arange(int(c)), np.unique(fi[:, f]))
            for f, c in enumerate(grid.cardinalities)]


def accept_rule(values: np.ndarray, oor: np.ndarray, dup: np.ndarray,
                bitmap: list, grid) -> np.ndarray:
    """THE acceptance rule, identical for every arm, at one threshold.

    Accept iff no factor is duplicated, no term is outside its factor's realised range,
    and every ON-GRID term names a value the fit pool holds.  An off-grid term strictly
    inside the range is accepted if BOTH bracketing grid values are held -- the "in
    basin" condition row 2 means.

    This is DECLARED FREE.  w3-hub-common section 7: refusal died on measurement and no
    hub quotes a row-5 or row-6 number as evidence for the store.  It is here because
    the scoreboard requires the pair on every row and because rows 5-7 are the leading
    indicator `hub/refuse-delete` asked for -- not because it is a result.
    """
    n = values.shape[0]
    ok = ~(np.asarray(dup, bool) | np.asarray(oor, bool))
    for f in range(values.shape[1]):
        gv = grid.values[f]
        bm = bitmap[f]
        v = values[:, f]
        asked = ~np.isnan(v)
        idx = np.zeros(n, np.int64)
        exact = np.zeros(n, bool)
        vv = np.where(asked, v, gv[0])
        hi = np.clip(np.searchsorted(gv, vv, side="left"), 0, gv.size - 1)
        lo = np.clip(hi - 1, 0, gv.size - 1)
        exact = np.isclose(gv[hi], vv, rtol=0.0, atol=1e-12)
        idx = hi
        held = np.where(exact, bm[idx], bm[hi] & bm[lo])
        ok &= (~asked) | held
    return ok


def judge_valid(values: np.ndarray, tuples: np.ndarray, grid) -> np.ndarray:
    """Does the judge's read-back satisfy the request?

    On-grid term: the judged index must equal it exactly.  Off-grid term strictly inside
    the range: the judged index must be one of the two bracketing indices -- the only
    thing "a hue BETWEEN two stored values" can mean to a grid-valued judge.  All six
    factors must hold; a partial match is not a satisfied request.
    """
    n = values.shape[0]
    ok = np.ones(n, bool)
    for f in range(values.shape[1]):
        gv = grid.values[f]
        v = values[:, f]
        asked = ~np.isnan(v)
        vv = np.where(asked, v, gv[0])
        hi = np.clip(np.searchsorted(gv, vv, side="left"), 0, gv.size - 1)
        lo = np.clip(hi - 1, 0, gv.size - 1)
        exact = np.isclose(gv[hi], vv, rtol=0.0, atol=1e-12)
        t = tuples[:, f]
        hit = np.where(exact, t == hi, (t == hi) | (t == lo))
        ok &= (~asked) | hit
    return ok


def seen_tuple_mask(grid, train_rows: np.ndarray) -> np.ndarray:
    """A boolean over the whole factor grid: which tuples the fit pool's split holds."""
    total = int(np.prod([int(c) for c in grid.cardinalities]))
    m = np.zeros(total, bool)
    m[np.asarray(train_rows, np.int64)] = True
    return m


def per_corner_scores(corners, values_all, tuples_all, valid_all, accept_all,
                      offsets, grid, seen_mask) -> list:
    """The estimand's unit: one row per corner, for the paired comparison."""
    cards = tuple(int(c) for c in grid.cardinalities)
    out = []
    for c, (s, e) in zip(corners, offsets):
        v = valid_all[s:e]
        t = tuples_all[s:e]
        keep = t[v]
        if keep.size:
            lin = np.ravel_multi_index(tuple(keep.T), cards)
            uniq = np.unique(lin)
            novel = uniq[~seen_mask[uniq]]
        else:
            uniq = np.zeros(0, np.int64)
            novel = np.zeros(0, np.int64)
        out.append({
            "corner_id": c["corner_id"], "family": c["family"],
            "n_requests": int(e - s),
            "n_accepted": int(accept_all[s:e].sum()),
            "n_valid": int(v.sum()),
            "judge_accuracy": float(v.mean()) if e > s else 0.0,
            "distinct_valid_tuples": int(uniq.size),
            "distinct_valid_novel_tuples": int(novel.size),
        })
    return out


def paired_estimand(arm_rows: list, ctl_rows: list, metric: str, margin: float,
                    arm_name: str, ctl_name: str) -> dict:
    """The multi-corner paired estimand.  Never a single cell.

    The unit is the corner; the statistic is the mean paired difference; the margin was
    declared in the PREREG before the first scored number.  A two-sided sign test and a
    paired-bootstrap interval are reported beside it, because with 49-60 corners the
    normal approximation is the thing a referee will not take on trust.
    """
    if [r["corner_id"] for r in arm_rows] != [r["corner_id"] for r in ctl_rows]:
        raise AssertionError("the two arms were not scored on the same corners, in order")
    a = np.array([float(r[metric]) for r in arm_rows])
    b = np.array([float(r[metric]) for r in ctl_rows])
    d = a - b
    n = d.size
    rng = np.random.default_rng(20_260_919)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10_000)])
    nz = d[d != 0]
    if nz.size:
        from math import comb
        k = int((nz > 0).sum())
        m = nz.size
        tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / (2 ** m)
        p = float(min(1.0, 2.0 * tail))
    else:
        p = 1.0
    return {
        "metric": metric, "arm": arm_name, "control": ctl_name,
        "n_corners": int(n),
        "arm_mean": float(a.mean()), "control_mean": float(b.mean()),
        "mean_paired_delta": float(d.mean()),
        "median_paired_delta": float(np.median(d)),
        "bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "corners_arm_wins": int((d > 0).sum()),
        "corners_tied": int((d == 0).sum()),
        "corners_control_wins": int((d < 0).sum()),
        "sign_test_p_two_sided": p,
        "margin_declared_in_advance": float(margin),
        "beats_beyond_margin": bool(d.mean() > float(margin)
                                    and float(np.quantile(boot, 0.025)) > float(margin)),
        "note": (
            "the unit is the CORNER, paired; the verdict is this estimand and never a "
            "single cell (w3-hub-common section 3: the corner explains 90.9% of the "
            "paired variance)"),
    }


# ---------------------------------------------------------------------------
# 7. The run stage
# ---------------------------------------------------------------------------


def run_arm(arm: str, out_dir: Path, max_seconds: float = 3600.0,
            log: Callable = print) -> dict:
    """Every row for one arm, corner by corner, resumable and wall-clock capped."""
    art = out_dir / "arms" / arm / "rows.npz"
    meta_p = out_dir / "arms" / arm / "rows.json"
    cond = load_condition()
    grid = cond["grid"]
    obj = build_arm(arm, out_dir, log=log)
    judge = _JUDGE[0] if _JUDGE else None
    if judge is None:
        judge = Judge(int(PINS["seed"]), log=log)
        _JUDGE.append(judge)

    val_mode = PINS["condition"] == "val"
    excl = val_corner_rows(cond) if val_mode else None
    fitted_rows = (np.setdiff1d(cond["train_rows"], excl) if val_mode
                   else cond["train_rows"])
    bitmap = value_presence_bitmap(grid, fitted_rows)
    if val_mode:
        # the SELECTION condition: only the val corners, and no query-set rows -- rows
        # 1/2/5/6 are scored on the test condition and are not selection material.
        n4, n3 = val_corners(cond), []
    else:
        n4 = nway_corners(cond)
        n3 = pair_subcorners(cond)
    corners = n3 + n4

    partial_p = out_dir / "arms" / arm / "partial.npz"
    done_corners: dict = {}
    if partial_p.exists():
        z = np.load(partial_p, allow_pickle=True)
        done_corners = dict(z["done"].item())
        log(f"  arm {arm}: resuming with {len(done_corners)} corners already scored")

    t0 = _now()
    for c in corners:
        if c["corner_id"] in done_corners:
            continue
        if _now() - t0 > float(max_seconds):
            np.savez_compressed(partial_p, done=np.array(done_corners, dtype=object))
            log(f"  arm {arm}: wall-clock cap hit after {len(done_corners)} corners "
                "-- state saved, rerun to continue")
            return {"incomplete": True, "n_done": len(done_corners)}
        vals = _request_values_from_rows(grid, c["rows"])
        feats, oor = arm_features(obj, vals, grid)
        dup = np.zeros(vals.shape[0], bool)
        acc = accept_rule(vals, oor, dup, bitmap, grid)
        tup = judge.read_back(feats, int(PINS["eval_batch"]))
        val = judge_valid(vals, tup, grid) & acc
        tup_out = np.where(acc[:, None], tup, -1)
        done_corners[c["corner_id"]] = {
            "accept": acc, "valid": val, "tuples": tup_out,
            "n": int(vals.shape[0]), "family": c["family"],
        }
    np.savez_compressed(partial_p, done=np.array(done_corners, dtype=object))

    if val_mode:
        np.savez_compressed(art, corners=np.array(done_corners, dtype=object),
                            qrows=np.array({}, dtype=object))
        _jdump({"arm": arm, "condition": "val", "n_corners": len(done_corners),
                "byte_ledger": obj["byte_ledger"], "head_bytes": obj["head_bytes"],
                "seconds": _now() - t0}, meta_p)
        log(f"  arm {arm}: VAL corners done in {_now() - t0:.1f}s")
        return {"incomplete": False, "n_done": len(done_corners)}

    # -- the query-set rows (1, 2, 5, 6) ------------------------------------
    qrows = {}
    for row_id, qname in (("r1_seen", "row1_seen"), ("r2_unseen_valid", "row2_unseen"),
                          ("r5_comp_refusal", "row5_held_out"),
                          ("r6_invalid_refusal", "c_invalid")):
        recs = cond["query_sets"][qname]["records"]
        vals, dup = _records_to_values(recs, grid)
        feats, oor = arm_features(obj, np.nan_to_num(vals, nan=np.nan), grid)
        acc = accept_rule(vals, oor, dup, bitmap, grid)
        tup = judge.read_back(feats, int(PINS["eval_batch"]))
        val = judge_valid(vals, tup, grid) & acc
        qrows[row_id] = {"accept": acc, "valid": val,
                         "tuples": np.where(acc[:, None], tup, -1),
                         "ids": [r.get("id") for r in recs]}
        log(f"  arm {arm}: {row_id} accept={acc.mean():.4f} valid={val.mean():.4f}")

    # -- row 7: deletion, with NO retraining ---------------------------------
    f, v = DELETED_FACTOR, DELETED_VALUE_INDEX
    dval = float(grid.values[f][v])
    recs = cond["query_sets"]["row1_seen"]["records"]
    rv, rdup = _records_to_values(recs, grid)
    hits = np.isclose(rv[:, f], dval)
    del_obj = delete_value_from_head(obj, f, v)
    bm_post = [b.copy() for b in bitmap]
    bm_post[f][v] = False
    survivors = [c for c in n3 if not np.isclose(
        float(c["meta"]["pair"]["object_hue"]), dval)]

    def _measure(o, vals, bm, dup=None):
        feats, oor = arm_features(o, vals, grid)
        d = np.zeros(vals.shape[0], bool) if dup is None else dup
        acc = accept_rule(vals, oor, d, bm, grid)
        tup = judge.read_back(feats, int(PINS["eval_batch"]))
        val = judge_valid(vals, tup, grid) & acc
        return acc, val, np.where(acc[:, None], tup, -1)

    pre_acc, pre_val, pre_tup = (qrows["r1_seen"]["accept"][hits],
                                 qrows["r1_seen"]["valid"][hits],
                                 qrows["r1_seen"]["tuples"][hits])
    post_acc, post_val, post_tup = _measure(
        del_obj, rv[hits], bm_post, rdup[hits])
    surv = {}
    for c in survivors:
        vals = _request_values_from_rows(grid, c["rows"])
        a, vl, t = _measure(del_obj, vals, bm_post)
        surv[c["corner_id"]] = {"accept": a, "valid": vl, "tuples": t,
                                "n": int(vals.shape[0]), "family": c["family"]}
    # the fit fingerprint: the store's own modes, before and after. EQUAL means no retrain.
    # An arm that deploys no store (knn_proto) fingerprints its prototype table instead --
    # it is still the object that must not have been refitted between the two halves.
    def _fp(o):
        if o["kind"] == "span":
            return tuple(float(np.asarray(m).sum()) for m in o["store"].modes)
        return tuple(float(np.asarray(P).sum()) for P in o["protos"].protos)
    fp, fp_post = _fp(obj), _fp(del_obj)
    deletion = {
        "deleted_factor": str(grid.names[f]), "deleted_value": dval,
        "deleted_note": del_obj.get("deleted"),
        "pre": {"accept": pre_acc, "valid": pre_val, "tuples": pre_tup},
        "post": {"accept": post_acc, "valid": post_val, "tuples": post_tup},
        "survivors": surv,
        "fit_fingerprint": [fp, fp_post],
        "n_deleted_requests": int(hits.sum()),
        "n_survivor_corners": len(survivors),
    }
    log(f"  arm {arm}: row7 delete {grid.names[f]}={dval} -- "
        f"accept on its requests {pre_acc.mean():.4f} -> {post_acc.mean():.4f}, "
        f"{len(survivors)} survivor corners, modes untouched={fp == fp_post}")

    payload = {"corners": np.array(done_corners, dtype=object),
               "qrows": np.array(qrows, dtype=object),
               "deletion": np.array(deletion, dtype=object)}
    np.savez_compressed(art, **payload)
    _jdump({"arm": arm, "n_corners": len(done_corners),
            "byte_ledger": obj["byte_ledger"], "head_bytes": obj["head_bytes"],
            "seconds": _now() - t0}, meta_p)
    log(f"  arm {arm}: all rows done in {_now() - t0:.1f}s")
    return {"incomplete": False, "n_done": len(done_corners)}


_JUDGE: list = []

# ---------------------------------------------------------------------------
# 8. Row 7 -- deletion, and what it does and does NOT satisfy
# ---------------------------------------------------------------------------

#: The value deleted for row 7, declared in code.  It is a value the store HOLDS (so the
#: deletion is a real removal, not a no-op) and it is ``pair-000``'s hue, so the survivor
#: set is exactly the sub-corners of the other two pair cells.
DELETED_FACTOR, DELETED_VALUE_INDEX = 2, 0          # object_hue = 0.0


def delete_value_from_head(obj: dict, f: int, v: int) -> dict:
    """Delete one (factor, value) from the ADDRESS BOOK.  No retraining.

    The head's main-effect column for ``(f, v)`` and every pair-interaction column
    containing it are zeroed, and the value's bit is cleared from the presence bitmap.
    The store's modes are **untouched**, so the fit fingerprint is unchanged and the
    deployed byte ledger is unchanged -- which is what "no retraining" means mechanically.

    DECLARED HONESTLY, because it is the leading indicator `hub/refuse-delete` asked for:
    this is an operation on the **interface**, not on atoms.  Charter section B.4 requires
    deletion to be an operation on atoms, and under a SHARED-BASIS read it cannot be:
    every well's modes are fitted over the whole pool and no atom belongs to one value.
    So row 7 here measures address-book deletion and is reported as exactly that.  A hub
    that needs erasure rather than gating does not get it from this architecture, and
    that is a finding, not a gap.
    """
    import copy

    if obj["kind"] != "span":
        return obj
    lay = obj["spec"].layout
    cols = [lay["main_offsets"][f] + v]
    for (a, b), o in zip(lay["pairs"], lay["pair_offsets"]):
        ca, cb = lay["cardinalities"][a], lay["cardinalities"][b]
        if a == f:
            cols += [o + v * cb + j for j in range(cb)]
        elif b == f:
            cols += [o + i * cb + v for i in range(ca)]
    heads = [np.array(W, np.float32, copy=True) for W in obj["heads"]]
    for W in heads:
        W[np.asarray(cols, np.int64), :] = 0.0
    out = dict(obj)
    out["heads"] = heads
    out["deleted"] = {"factor_index": int(f), "value_index": int(v),
                      "n_head_columns_zeroed": len(cols),
                      "store_modes_touched": False}
    return out


# ---------------------------------------------------------------------------
# 9. The table -- the seven rows and the multi-corner paired estimand
# ---------------------------------------------------------------------------

#: THE MARGIN, declared in advance (w3-hub-common section 3 and the target file).
#: Rows 3 and 4 score DISTINCT VALID NOVEL TUPLES per corner; the estimand is the mean
#: paired per-corner difference against ``basis_head``.  The margin is **+1.0 novel tuple
#: per corner**: one whole extra distinct valid combination per corner, on corners of 72
#: (row 4) and 36 (row 3) requests.  It is set before the first scored number and it is
#: NOT moved afterwards (charter section A.3: amendments are appended, never edited).
DECORATION_MARGIN_NOVELTY = 1.0
#: Rows 1-2 score judge accuracy; the margin there is +0.02 absolute.  Reported, not claimed.
DECORATION_MARGIN_ACCURACY = 0.02


def _cat(d: dict, keys, field: str) -> np.ndarray:
    return np.concatenate([np.asarray(d[k][field]) for k in keys]) if keys else np.zeros(0)


def build_table(out_dir: Path, primary_arm: str = "factor_aligned_balanced",
                log: Callable = print) -> dict:
    """Every arm, every row, plus the estimand that actually decides rows 3 and 4."""
    from ..experiments.gen_scoreboard import (
        ScoreboardConfig, build_scoreboard, grid_syntax_outcome, make_outcome,
        refuse_everything_outcome, accept_everything_outcome, write_scoreboard,
    )

    cond = load_condition()
    grid = cond["grid"]
    seen_mask = seen_tuple_mask(grid, cond["train_rows"])
    n4 = nway_corners(cond)
    n3 = pair_subcorners(cond)
    id3 = [c["corner_id"] for c in n3]
    id4 = [c["corner_id"] for c in n4]

    have = [a for a in ARMS if (out_dir / "arms" / a / "rows.npz").exists()]
    log(f"  arms with rows on disk: {have}")
    outcomes: dict = {}
    per_corner: dict = {}
    deletion_blk: dict = {}
    extras: dict = {}

    for arm in have:
        z = np.load(out_dir / "arms" / arm / "rows.npz", allow_pickle=True)
        cor = z["corners"].item()
        q = z["qrows"].item()
        dele = z["deletion"].item() if "deletion" in z.files else None
        with open(out_dir / "arms" / arm / "arm.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        led = meta["byte_ledger"]
        setting = {"arm": arm, "read": "signed linear span over the store's modes",
                   "head": "request-conditioned ridge, pair block",
                   "threshold_rule": "value-presence bitmap + bracket check"}
        rows = {}
        for row_id, ids in (("r3_comp_2way", id3), ("r4_comp_nway", id4)):
            rows[row_id] = make_outcome(
                accept=_cat(cor, ids, "accept").astype(bool),
                valid=_cat(cor, ids, "valid").astype(bool),
                tuples=np.concatenate([np.asarray(cor[k]["tuples"]) for k in ids]),
                threshold=0.0, setting=setting, byte_ledger=led)
        for row_id in ("r1_seen", "r2_unseen_valid", "r5_comp_refusal",
                       "r6_invalid_refusal"):
            rows[row_id] = make_outcome(
                accept=np.asarray(q[row_id]["accept"], bool),
                valid=np.asarray(q[row_id]["valid"], bool),
                tuples=np.asarray(q[row_id]["tuples"]),
                threshold=0.0, setting=setting, byte_ledger=led,
                request_ids=q[row_id]["ids"])
        if dele is not None:
            sids = sorted(dele["survivors"])
            deletion_blk[arm] = {
                "survivor": make_outcome(
                    accept=_cat(dele["survivors"], sids, "accept").astype(bool),
                    valid=_cat(dele["survivors"], sids, "valid").astype(bool),
                    tuples=np.concatenate(
                        [np.asarray(dele["survivors"][k]["tuples"]) for k in sids]),
                    threshold=0.0, setting=setting, byte_ledger=led),
                "deleted": make_outcome(
                    accept=np.asarray(dele["post"]["accept"], bool),
                    valid=np.asarray(dele["post"]["valid"], bool),
                    tuples=np.asarray(dele["post"]["tuples"]),
                    threshold=0.0, setting=setting, byte_ledger=led),
                "pre": {"accept": np.asarray(dele["pre"]["accept"], bool)},
                "fit_fingerprint": (tuple(dele["fit_fingerprint"][0]),
                                    tuple(dele["fit_fingerprint"][1])),
            }
            rows["r7_deletion"] = rows["r4_comp_nway"]     # placeholder; scored via `deletion`
            extras[arm] = {"deletion_detail": {
                k: v for k, v in dele.items()
                if k in ("deleted_factor", "deleted_value", "deleted_note",
                         "n_deleted_requests", "n_survivor_corners")},
                "pre_accept_on_deleted_requests": float(
                    np.asarray(dele["pre"]["accept"]).mean()),
                "post_accept_on_deleted_requests": float(
                    np.asarray(dele["post"]["accept"]).mean()),
                "modes_untouched": bool(
                    tuple(dele["fit_fingerprint"][0]) == tuple(dele["fit_fingerprint"][1])),
            }
        outcomes[arm] = rows
        allc = {**{k: cor[k] for k in id3}, **{k: cor[k] for k in id4}}
        per_corner[arm] = {
            "r3": _per_corner(n3, cor, grid, seen_mask),
            "r4": _per_corner(n4, cor, grid, seen_mask),
        }
        extras.setdefault(arm, {})["head_bytes"] = meta["head_bytes"]["head_bytes"]

    # the no-model laundering rule, on the refusal rows it exists for
    lr = {}
    for row_id, qname in (("r1_seen", "row1_seen"), ("r2_unseen_valid", "row2_unseen"),
                          ("r5_comp_refusal", "row5_held_out"),
                          ("r6_invalid_refusal", "c_invalid")):
        lr[row_id] = grid_syntax_outcome(cond["query_sets"][qname]["records"], grid)
    n3n = sum(int(c["rows"].size) for c in n3)
    n4n = sum(int(c["rows"].size) for c in n4)
    for row_id, n in (("r3_comp_2way", n3n), ("r4_comp_nway", n4n),
                      ("r7_deletion", n4n)):
        lr[row_id] = make_outcome(
            accept=np.zeros(n, bool), valid=np.zeros(n, bool),
            tuples=-np.ones((n, 6), np.int64), threshold=0.0,
            setting=lr["r5_comp_refusal"]["setting"],
            byte_ledger=lr["r5_comp_refusal"]["byte_ledger"])
    outcomes["grid_syntax_rule"] = lr

    cfg = ScoreboardConfig(decoration_margin=float(DECORATION_MARGIN_NOVELTY))
    seen = [tuple(int(x) for x in r) for r in
            grid.factor_indices_of(cond["train_rows"])]
    table = build_scoreboard(
        primary_arm, outcomes, seen_tuples=seen, cfg=cfg,
        deletion=deletion_blk,
        provenance={"pins": {k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in PINS.items()},
                    "corners": {"r3": len(n3), "r4": len(n4)},
                    "hub": "hub/factor-wells"})

    # -- THE ESTIMAND -------------------------------------------------------
    estimand = {}
    for row, key in (("r3_comp_2way", "r3"), ("r4_comp_nway", "r4")):
        for ctl in [a for a in have if a != primary_arm]:
            estimand.setdefault(row, {})[ctl] = paired_estimand(
                per_corner[primary_arm][key], per_corner[ctl][key],
                "distinct_valid_novel_tuples", DECORATION_MARGIN_NOVELTY,
                primary_arm, ctl)
    out = {
        "scoreboard": table,
        "multi_corner_estimand": estimand,
        "per_corner": per_corner,
        "extras": extras,
        "margins_declared_in_advance": {
            "novelty_rows_3_4": DECORATION_MARGIN_NOVELTY,
            "accuracy_rows_1_2": DECORATION_MARGIN_ACCURACY,
            "unit": "distinct valid NOVEL tuples per corner, mean paired difference",
        },
    }
    _jdump(out, out_dir / "table.json")
    write_scoreboard(table, out_dir / "scoreboard.json")
    log(f"  table written -> {out_dir / 'table.json'}")
    return out


def _per_corner(corners, cor: dict, grid, seen_mask) -> list:
    cards = tuple(int(c) for c in grid.cardinalities)
    out = []
    for c in corners:
        r = cor[c["corner_id"]]
        v = np.asarray(r["valid"], bool)
        t = np.asarray(r["tuples"])
        keep = t[v]
        if keep.size:
            lin = np.unique(np.ravel_multi_index(tuple(keep.T), cards))
            novel = lin[~seen_mask[lin]]
        else:
            lin = np.zeros(0, np.int64)
            novel = np.zeros(0, np.int64)
        out.append({
            "corner_id": c["corner_id"], "family": c["family"],
            "n_requests": int(r["n"]),
            "n_accepted": int(np.asarray(r["accept"]).sum()),
            "n_valid": int(v.sum()),
            "judge_accuracy": float(v.mean()) if v.size else 0.0,
            "distinct_valid_tuples": int(lin.size),
            "distinct_valid_novel_tuples": int(novel.size),
        })
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("partition", "fit", "run", "table", "corners"))
    p.add_argument("--arm", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seconds", type=float, default=3600.0)
    p.add_argument("--condition", default="test", choices=("val", "test"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    PINS["condition"] = args.condition
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = _now()

    def log(msg):
        print(f"[{_now() - t0:7.1f}s] {msg}", flush=True)

    if args.stage == "partition":
        build_partitions(out, log=log)
    elif args.stage == "corners":
        cond = load_condition()
        n4 = nway_corners(cond)
        n3 = pair_subcorners(cond)
        info = {
            "row4_nway": assert_corners_disjoint_and_heldout(n4, cond["test_rows"], "nway"),
            "row3_pair_tiled": assert_corners_disjoint_and_heldout(
                n3, cond["test_rows"], "pair-tiled"),
        }
        _jdump(info, out / "corners.json")
        log(json.dumps(info, indent=1))
    elif args.stage == "fit":
        build_arm(args.arm, out, log=log)
    elif args.stage == "run":
        run_arm(args.arm, out, max_seconds=args.max_seconds, log=log)
    elif args.stage == "table":
        build_table(out, primary_arm=(args.arm or "factor_aligned_balanced"), log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
