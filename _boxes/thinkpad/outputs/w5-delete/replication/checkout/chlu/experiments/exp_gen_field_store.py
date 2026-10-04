"""hub/field-store -- BUILD ONLY.  Build the field store, smoke it, prove the byte property.

⛔⛔ **SCORED BUDGET: ZERO.**  Nothing in this module touches the held-out test corners or
the shipped query sets that the seven-row scoreboard is scored on.  Every number it
produces is measured on the **selection (``val``) condition** the charter sanctions in
section A.2, on corners whose rows are removed from the fit pool of *every* arm here, so
the comparison is like-for-like and costs no scored run.  It claims nothing.

What it measures, and against what
----------------------------------
Two objects, one pool, one judge, one set of corners:

``aligned_w3``  the wave-3 aligned store -- the factor-aligned partition, ``gen_store``'s
                k-means++/adam modes and ``gen_request_head``'s request-conditioned ridge
                head with the pair-interaction block.  **This is the incumbent** and the
                acceptance bar: the field store must land WITHIN its spread, not beyond it.
``field``       the field store of ``chlu.core.gen_field``.

plus the charter section C.2 controls on each (``landscape_removed``, ``landscape_shuffled``).

The rows
--------
``r1_seen``  512 requests drawn from the fit pool itself -- seen combinations.
``r3_2way``  60 sub-corners over three ``(object_hue, shape)`` conjunctions that **every
             row carrying them** is removed from the fit pool for, so the conjunction is
             genuinely unheld -- the ``val`` analogue of the scored ``pair-*`` cells.
``r4_nway``  the 24 ``val`` Montero slots the condition enumerated but never emitted --
             exactly ``exp_gen_factor_wells.val_corners``, the charter's own selection set.

Runnable::

    python -m chlu.experiments.exp_gen_field_store --stage corners --out DIR
    python -m chlu.experiments.exp_gen_field_store --stage fit  --arm ARM --seed S --out DIR
    python -m chlu.experiments.exp_gen_field_store --stage run  --arm ARM --seed S --out DIR
    python -m chlu.experiments.exp_gen_field_store --stage delete --seed S --out DIR
    python -m chlu.experiments.exp_gen_field_store --stage report --out DIR

Every stage is resumable: it writes one artifact per (stage, arm, seed) and skips a stage
whose artifact is already complete.  Every stage takes ``--max-seconds`` and honours it
between corners, so a kill costs one corner and not the run.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .exp_gen_factor_wells import (
    PINS as W3PINS,
    Judge,
    _cell_mask_rows,
    _jdump,
    accept_rule,
    build_partitions,
    features_mmap,
    gather,
    judge_valid,
    load_condition,
    paired_estimand,
    val_corners,
    val_corner_rows,
    value_presence_bitmap,
)

# ---------------------------------------------------------------------------
# 0. The pins.  The store fit is inherited from hub/factor-wells verbatim; the
#    field store's own dials are declared here and nowhere else.
# ---------------------------------------------------------------------------

PINS = dict(W3PINS)
PINS.update({
    # the SELECTION condition, always.  This hub has no scored run.
    "condition": "val",
    # WARNING, measured and binding: the frozen feature cache is FULL SCOPE for bridge
    # seed 0 only -- `features.npy` is 480,000 rows at seed 0 and **9,288 rows** at seeds
    # 1 and 2.  Extending it would mean writing inside the read-only frozen instrument and
    # re-encoding 960,000 images, neither of which a hub may do.  So the bridge-seed spread
    # is a DECLARED NOT-RUN (never a null), and the spread reported here is the spread of
    # the OBJECT at a fixed instrument: three STORE-FIT seeds, which vary the fit pool
    # draw, the incumbent's k-means++/adam stream and the field store's atom selection,
    # identically for both arms.
    "bridge_seed": 0,
    "seeds": (0, 1, 2),
    "seed_meaning": "store-fit seed at bridge seed 0",
    "pool_seed_stride": 7919,
    "atom_seed_stride": 1_000_003,
    # -- the val row-3 conjunctions ----------------------------------------
    # (object_hue index, shape index).  ⛔ Disjoint from the scored pair cells
    # ((0,1), (3,2), (7,3)), from the held-out values (object_hue 0.5 -> index 5,
    # shape 0.0 -> index 0), from row1_seen's own pair (index 4, shape 3), AND from
    # every (object_hue, shape) combination the 24 val n-way slots realise -- which is
    # {1,2,8,9} x {1,2,3} -- so the row-3 and row-4 corner sets are pairwise disjoint.
    "val_pairs": ((3, 1), (4, 2), (6, 3)),
    "n_row1_requests": 512,
    "row1_rng_seed": 20_260_919,
    # -- the field store ----------------------------------------------------
    "field_n_atoms": 64,          # 64 items x 2048 float32 = 524,288 B, the w3 ledger
    "field_eta": 1.0,
    "field_ridge_frac": 1e-3,
    # ⭐ SELECTED ON VAL by --stage sweep (no judge, no scored run): sigma_mult 0.5 and
    # coupling_eta 0.0 minimise val reconstruction error. The coupling gain HURT
    # (0.3566 -> 0.3617) because the surviving PMI on this pool is the value hold-out's
    # own depletion structure, not a property of shapes3d. The coupling table stays --
    # its job is the HOLE and the unseen-pair prior, and both are measured separately.
    "field_sigma_mult": 1.0,
    # ⭐ atoms chosen by greedy SPAN selection against the address manifold, with the
    # basin-coverage constraint.  Val rel-err 0.2048/0.3297/0.2410 on rows 1/3/4 against
    # a plain Latin design's 0.2509/0.3382/0.2542.
    "field_atom_select": "span",
    # ⛔ MARGINAL, not backfitted: the joint estimate was tried and measured WORSE on
    # every row (val address rel-err 0.2556/0.3537/0.2810 after one sweep, 0.2837/0.4061/
    # 0.2823 after four, against 0.1610/0.3015/0.1910). The cost is a SHAPE deficit on the
    # two-way corners, and it is reported rather than papered over.
    "field_term_map": "marginal",
    "field_coupling_prior_strength": 32.0,
    # ⭐ rank 8 SELECTED ON VAL: val rel-err 0.2652 (r=4) / 0.2524 (r=8) / 0.2461 (r=16)
    # / 0.2406 (full), against the incumbent head's 0.2798 -- r=8 is the smallest rank
    # that beats the head, and its table is 1,025,344 B inside the head's 1,766,400 B.
    "field_coupling_rank": 8,
    # SELECTED ON VAL. One-step de-contamination of the marginal term coordinates, applied
    # only where the fit pool is unbalanced. It buys the SHAPE read-back on the two-way
    # corners (0.453 -> 0.568 at 0.5, -> 0.629 at 1.0) and costs object_hue (0.998 -> 0.969
    # -> 0.893) and row-1 exact-tuple accuracy (0.584 -> 0.578 -> 0.522). 0.5 is where the
    # row-3 gain is bought without paying for it on row 1.
    "field_decontaminate": 0.5,
    "field_coupling_shrink": 2.0,
    "field_coupling_eta": 0.0,
    # CALIBRATED ON VAL. The gap rule is a distance cut, not a discriminator: at tau=1.5
    # it refuses 0.195 of SEEN requests and 0.292 of unseen two-way ones -- it does not
    # separate them, because on this encoder the within-value spread is comparable to the
    # between-value gap. At tau=4.0 it costs 0.0039 on seen and 0.0000 on unseen two-way,
    # and the two refusal mechanisms that DO discriminate are untouched by it: the empty
    # basin (1.0000 on a deleted value at every tau tried) and the hole.
    "field_gap_tau": 4.0,
    "field_basin_live_tau": 2.0,
    "field_atom_seed": 20_260_919,
    # -- the deletion targets, declared in code ----------------------------
    # object_hue = 0.0 is w4-hub-common section 4.10's eligible target.
    "delete_factor": 2, "delete_value": 0,
    # THE RED CYLINDER: object_hue = 0.0 AND shape = 1.0.  Deleted as a HOLE, so no
    # shape content is evicted and section 4.10's shape=cube exclusion is not engaged.
    "hole_pair": ((2, 0), (4, 1)),
})

ARMS = (
    "field",                       # THE object
    "aligned_w3",                  # the incumbent, the acceptance bar
    "field_landscape_removed",     # charter C.2 -- atoms zeroed, must collapse
    "field_landscape_shuffled",    # charter C.2 -- atoms kept, coordinates scrambled
    "field_landscape_index_permuted",  # atom INDICES permuted: a declared no-op, see below
    "aligned_landscape_removed",   # the same control on the incumbent
)

ROWS = ("r1_seen", "r3_2way", "r4_nway")

#: The band declared BEFORE the first smoke number, in the charter's own unit and at the
#: wave-3 hub's own margin: a mean paired per-corner difference of 1.0 distinct valid novel
#: tuple.  "Within the spread" means |delta| <= this, on rows 3 and 4, against aligned_w3.
#: ⛔ Not moved once a number has been seen against it (charter section A.3).
WITHIN_BAND_NOVELTY = 1.0
#: Row 1 scores judge accuracy; the band is the wave-3 hub's own +-0.02 absolute.
WITHIN_BAND_ACCURACY = 0.02


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# 1. The corners and the pool -- one construction, shared by every arm
# ---------------------------------------------------------------------------


def pair_rows_all(cond, oh: int, sh: int) -> np.ndarray:
    """EVERY dataset row carrying the conjunction ``object_hue[oh] AND shape[sh]``.

    All of them leave the fit pool, not just the corner's own rows -- otherwise the
    conjunction is still held elsewhere in the pool and the corner is not a two-way
    composition at all.
    """
    grid = cond["grid"]
    return _cell_mask_rows(grid, {
        "object_hue": (float(grid.values[2][int(oh)]),),
        "shape": (float(grid.values[4][int(sh)]),),
    })


def excluded_rows(cond) -> dict:
    """Everything removed from the fit pool of EVERY arm on this hub."""
    n4 = val_corner_rows(cond)
    pr = {f"{oh}x{sh}": pair_rows_all(cond, oh, sh) for oh, sh in PINS["val_pairs"]}
    allp = np.unique(np.concatenate(list(pr.values()))) if pr else np.zeros(0, np.int64)
    return {"val_nway_rows": n4, "val_pair_rows": allp,
            "per_pair": pr, "all": np.unique(np.concatenate([n4, allp]))}


def field_fit_pool(cond, store_seed: int = 0) -> np.ndarray:
    """The ONE fit pool for a store seed.  Identical rows and order for BOTH arms.

    The store seed moves the draw, so the three reported seeds are three genuinely
    different fit pools -- which is what makes their spread the object's spread and not
    just an optimiser's jitter.
    """
    from .exp_gen_factor_wells import fit_pool

    rng_seed = int(PINS["pool_rng_seed"]) + int(PINS["pool_seed_stride"]) * int(store_seed)
    return fit_pool(cond["train_rows"], PINS["n_pool"], rng_seed,
                    exclude=excluded_rows(cond)["all"])


def row3_corners(cond) -> list:
    """The val two-way corners: the scored row-3 tiling, on unheld conjunctions."""
    grid = cond["grid"]
    bw = int(PINS["pair_tile_hue_band"])
    sw = int(PINS["pair_tile_scale_corner"])
    rw = int(PINS["pair_tile_orientation_corner"])
    fh = grid.values[grid.index_of("floor_hue")]
    sc = grid.values[grid.index_of("scale")]
    ro = grid.values[grid.index_of("orientation")]
    windows = [np.arange(i, i + bw) for i in range(0, fh.size - bw + 1, bw)]
    out = []
    for oh, sh in PINS["val_pairs"]:
        for fi, win in enumerate(windows):
            for scn in ("high", "low"):
                for rcn in ("high", "low"):
                    band = tuple(float(x) for x in fh[win])
                    vals = {
                        "object_hue": (float(grid.values[2][int(oh)]),),
                        "shape": (float(grid.values[4][int(sh)]),),
                        "floor_hue": band, "wall_hue": band,
                        "scale": tuple(float(x) for x in
                                       (sc[-sw:] if scn == "high" else sc[:sw])),
                        "orientation": tuple(float(x) for x in
                                             (ro[-rw:] if rcn == "high" else ro[:rw])),
                    }
                    rows = _cell_mask_rows(grid, vals)
                    if rows.size == 0:
                        continue
                    out.append({
                        "corner_id": f"valpair-{oh}x{sh}/f{fi}-{scn[0]}{rcn[0]}",
                        "family": "val_pair", "rows": rows,
                        "meta": {"pair": {"object_hue_index": int(oh),
                                          "shape_index": int(sh)},
                                 "floor_band_start_index": int(win[0]),
                                 "scale_corner": scn, "orientation_corner": rcn},
                    })
    return out


def row4_corners(cond) -> list:
    return val_corners(cond)


def row1_requests(cond, pool: np.ndarray) -> np.ndarray:
    """``n`` rows drawn from the fit pool itself -- the seen row, by construction."""
    rng = np.random.default_rng(int(PINS["row1_rng_seed"]))
    n = int(min(int(PINS["n_row1_requests"]), pool.size))
    return np.sort(pool[rng.permutation(pool.size)[:n]])


def assert_smoke_is_not_scored(cond, corners3, corners4, r1rows, pool) -> dict:
    """⛔ The structural proof that this hub spends no scored run.

    Every smoke corner's rows are (a) outside the fit pool of every arm, and (b) in the
    value-holdout TRAIN index -- i.e. they are selection material, never the scored
    ``test`` corners.  Row 1's requests are IN the pool, which is what "seen" means.
    """
    tr = np.asarray(cond["train_rows"], np.int64)
    te = np.asarray(cond["test_rows"], np.int64)
    P = np.asarray(pool, np.int64)
    rep = {}
    for name, cs in (("r3_2way", corners3), ("r4_nway", corners4)):
        rows = np.unique(np.concatenate([c["rows"] for c in cs]))
        rep[name] = {
            "n_corners": len(cs),
            "n_rows": int(rows.size),
            "rows_in_fit_pool": int(np.intersect1d(rows, P).size),
            "rows_in_scored_test_index": int(np.intersect1d(rows, te).size),
            "rows_in_train_index": int(np.intersect1d(rows, tr).size),
            "rows_per_corner": sorted({int(c["rows"].size) for c in cs}),
        }
        if rep[name]["rows_in_fit_pool"]:
            raise AssertionError(
                f"{name}: {rep[name]['rows_in_fit_pool']} corner rows are still in the "
                "fit pool -- the corner is not held out from the store")
    rep["r1_seen"] = {"n_requests": int(r1rows.size),
                      "rows_in_fit_pool": int(np.intersect1d(r1rows, P).size)}
    if rep["r1_seen"]["rows_in_fit_pool"] != int(r1rows.size):
        raise AssertionError("row 1's requests are not all in the fit pool -- not 'seen'")
    seen = np.concatenate([c["rows"] for c in corners3 + corners4])
    if np.unique(seen).size != seen.size:
        raise AssertionError("the smoke corners overlap")
    rep["pairwise_disjoint"] = True
    rep["statement"] = (
        "every smoke corner is held out of the fit pool of every arm and lives in the "
        "value-holdout TRAIN index; the scored test corners and the shipped query sets "
        "are not touched by this module, so this hub spends 0 of its 0 scored runs")
    return rep


# ---------------------------------------------------------------------------
# 2. The arms
# ---------------------------------------------------------------------------


def _host_factors(part_info: dict, K: int) -> list:
    prov = part_info["factor_aligned"]["provenance"]
    return [list(map(int, prov[str(k)]["source_factor_indices"])) for k in range(K)]


def field_spec(store_seed: int = 0) -> "object":
    from ..core.gen_field import FieldSpec

    return FieldSpec(
        n_atoms=int(PINS["field_n_atoms"]), eta=float(PINS["field_eta"]),
        ridge_frac=float(PINS["field_ridge_frac"]),
        sigma_mult=float(PINS["field_sigma_mult"]),
        coupling_prior_strength=float(PINS["field_coupling_prior_strength"]),
        coupling_rank=int(PINS["field_coupling_rank"]),
        coupling_shrink=float(PINS["field_coupling_shrink"]),
        coupling_eta=float(PINS["field_coupling_eta"]),
        atom_select=str(PINS["field_atom_select"]),
        decontaminate=float(PINS["field_decontaminate"]),
        term_map=str(PINS["field_term_map"]),
        gap_tau=float(PINS["field_gap_tau"]),
        basin_live_tau=float(PINS["field_basin_live_tau"]),
        atom_seed=int(PINS["field_atom_seed"])
        + int(PINS["atom_seed_stride"]) * int(store_seed))


def seed_dir(out: Path, seed: int) -> Path:
    return out / f"seed_{int(seed)}"


def build_field_arm(out: Path, seed: int, log: Callable = print) -> dict:
    """Write the field store for one seed.  Resumable."""
    from ..core.gen_field import (
        assert_read_request_only, build_field, field_byte_ledger,
    )

    d = seed_dir(out, seed) / "arms" / "field"
    done = d / "arm.pkl"
    if done.exists():
        log(f"  field(seed {seed}): already built, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cond = load_condition()
    grid = cond["grid"]
    parts = build_partitions(out / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    K = int(labels.max()) + 1
    hosts = _host_factors(parts["info"], K)
    pool = field_fit_pool(cond, seed)
    F = features_mmap(int(PINS["bridge_seed"]))
    t0 = _now()
    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    cards = [int(c) for c in grid.cardinalities]
    log(f"  field(seed {seed}): pool {pool.size} rows, hosts {hosts}")
    field = build_field(X, Y, pool, labels, hosts, cards, field_spec(seed), log=log)
    del X
    led = field_byte_ledger(field, int(PINS["store_bytes"]))
    if not led["matches_budget"]:
        raise AssertionError(
            f"the field store deploys {led['deployed_bytes']} B against the pinned "
            f"{PINS['store_bytes']} B ledger -- the byte ledger has moved")
    vals = grid.values_of(pool[:64])
    ro = assert_read_request_only(field, vals, grid.values)
    obj = {"arm": "field", "kind": "field", "seed": int(seed), "field": field,
           "byte_ledger": led, "request_only": ro,
           "manifest": field.manifest(), "fit_seconds": _now() - t0}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items() if k != "field"}, d / "arm.json")
    log(f"  field(seed {seed}): built in {obj['fit_seconds']:.1f}s; atoms "
        f"{led['deployed_bytes']:,} B + interface {led['interface_bytes']:,} B "
        f"(image {led['total_image_bytes']:,} B)")
    return obj


def build_aligned_arm(out: Path, seed: int, log: Callable = print) -> dict:
    """Fit the wave-3 aligned store + head on the SAME pool.  Resumable."""
    from ..core.gen_request_head import (
        HeadSpec, assert_head_request_only, head_bytes, request_design, solve_head,
    )
    from ..core.gen_store import reconstruction_mse, store_bytes
    from .exp_gen_factor_wells import _store_for_labels

    d = seed_dir(out, seed) / "arms" / "aligned_w3"
    done = d / "arm.pkl"
    if done.exists():
        log(f"  aligned_w3(seed {seed}): already fitted, loading")
        with open(done, "rb") as fh:
            return pickle.load(fh)

    cond = load_condition()
    grid = cond["grid"]
    cards = [int(c) for c in grid.cardinalities]
    parts = build_partitions(out / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    pool = field_fit_pool(cond, seed)
    F = features_mmap(int(PINS["bridge_seed"]))
    spec = HeadSpec(cards, pair_block=bool(PINS["head_pair_block"]),
                    ridge=float(PINS["head_ridge"]))
    # the incumbent's own fit stream follows the same store seed, so the two arms' spreads
    # are produced by the same perturbation and are comparable.
    W3PINS["store_key_seed"] = int(seed)
    t0 = _now()
    X_store = gather(F, pool[: int(PINS["n_store_train"])])
    store, hist = _store_for_labels(labels, X_store, log)
    mse = reconstruction_mse(store, X_store)
    led = store_bytes(store)
    if int(led["store_bytes"]) != int(PINS["store_bytes"]):
        raise AssertionError(f"aligned_w3: store is {led['store_bytes']} B")
    del X_store

    gp = seed_dir(out, seed) / "gram.npz"
    if gp.exists():
        z = np.load(gp)
        A, Mx, n_head = z["A"], z["Mx"], int(z["n"])
        log(f"  gram(seed {seed}): reused")
    else:
        A = np.zeros((spec.width, spec.width), np.float64)
        Mx = np.zeros((spec.width, int(PINS["D"])), np.float64)
        n_head = 0
        for s in range(0, pool.size, 8192):
            e = min(s + 8192, pool.size)
            rows = pool[s:e]
            Phi, _ = request_design(grid.values_of(rows), grid.values, spec)
            P = Phi.astype(np.float64)
            A += P.T @ P
            Mx += P.T @ gather(F, rows).astype(np.float64)
            n_head += int(P.shape[0])
        gp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gp, A=A, Mx=Mx, n=n_head)
        log(f"  gram(seed {seed}): accumulated over {n_head} rows")
    heads = solve_head(A, Mx, store, float(PINS["head_ridge"]))
    hb = head_bytes(heads)
    Phi0, _ = request_design(grid.values_of(pool[:64]), grid.values, spec)
    ro = assert_head_request_only(heads, store, Phi0, rng_seed=int(seed))
    obj = {"arm": "aligned_w3", "kind": "span", "seed": int(seed), "store": store,
           "heads": heads, "spec": spec, "labels": labels, "read_mode": "linear",
           "byte_ledger": {"deployed_bytes": int(led["store_bytes"]),
                           "budget_bytes": int(PINS["store_bytes"]),
                           "what": f"{led['n_wells']} wells x {PINS['n_modes']} modes, float32"},
           "head_bytes": hb, "store_ledger": led, "recon_mse": mse,
           "train_history_tail": [float(x) for x in hist[-5:]],
           "request_only": ro, "fit_seconds": _now() - t0}
    d.mkdir(parents=True, exist_ok=True)
    with open(done, "wb") as fh:
        pickle.dump(obj, fh)
    _jdump({k: v for k, v in obj.items()
            if k not in ("store", "heads", "spec", "labels")}, d / "arm.json")
    log(f"  aligned_w3(seed {seed}): fitted in {obj['fit_seconds']:.1f}s; store "
        f"{led['store_bytes']:,} B + head {hb['head_bytes']:,} B")
    return obj


def build_arm(arm: str, out: Path, seed: int, log: Callable = print) -> dict:
    """Any arm, including the charter section C.2 controls, which are derived in place."""
    if arm == "field":
        return build_field_arm(out, seed, log=log)
    if arm == "aligned_w3":
        return build_aligned_arm(out, seed, log=log)
    if arm.startswith("field_landscape_"):
        base = build_field_arm(out, seed, log=log)
        f = base["field"].copy()
        at = f.view("atoms")
        rng = np.random.default_rng(int(seed) * 7919 + 20_260_919)
        if arm.endswith("removed"):
            at[:] = 0.0
        elif arm.endswith("index_permuted"):
            # ⚠ MEASURED AND REPORTED AS A NO-OP, on purpose.  Wave 3's landscape_shuffled
            # permutes MODE INDICES after the head is solved, which breaks the head's
            # coefficient->mode correspondence.  This store has no such correspondence: the
            # read weights atoms by DISTANCE, so a permutation of atom indices is invisible
            # to it and the arm reproduces the field arm exactly.  That is not a broken
            # control, it is a positive structural statement -- the store is ADDRESSED, not
            # INDEXED (charter section B.2) -- and it is kept so the statement is testable.
            perm = rng.permutation(f.A)
            for k in range(f.K):
                blk = f.atoms_of_well(k)
                blk[:] = blk[perm]
        else:
            # THE informative control: the same atoms, the same bytes, the same multiset of
            # numbers in every well -- but each well's CHANNEL axis is permuted, so an
            # atom's coordinates no longer line up with the address's.  The content is kept
            # and the addressing is destroyed, which is what charter section C.2 asks for.
            for k in range(f.K):
                blk = f.atoms_of_well(k)
                blk[:] = blk[:, rng.permutation(blk.shape[1])]
        return {**base, "arm": arm, "field": f}
    if arm == "aligned_landscape_removed":
        from ..core.gen_store import delete_store
        base = build_aligned_arm(out, seed, log=log)
        return {**base, "arm": arm,
                "store": delete_store(base["store"], mode="zero", seed=int(seed))}
    raise ValueError(f"unknown arm {arm!r}")


def arm_features(obj: dict, values: np.ndarray, grid) -> tuple:
    """``(N, F) requested VALUES -> (N, D) features`` + the out-of-range mask."""
    from ..core.gen_field import field_read
    from ..core.gen_request_head import head_read, request_design

    if obj["kind"] == "field":
        from ..core.gen_field import field_address
        addr = field_address(obj["field"], values, grid.values)
        return field_read(obj["field"], values, grid.values), addr["out_of_range"]
    Phi, oor = request_design(values, grid.values, obj["spec"])
    return head_read(obj["heads"], obj["store"], Phi), oor


# ---------------------------------------------------------------------------
# 3. The run stage
# ---------------------------------------------------------------------------

_JUDGE: dict = {}


def _judge(seed: int, log: Callable) -> Judge:
    if int(seed) not in _JUDGE:
        _JUDGE[int(seed)] = Judge(int(seed), log=log)
    return _JUDGE[int(seed)]


def run_arm(arm: str, out: Path, seed: int, max_seconds: float = 3600.0,
            log: Callable = print) -> dict:
    """Rows 1, 3 and 4 for one arm at one seed.  Resumable, wall-clock capped."""
    d = seed_dir(out, seed) / "arms" / arm
    art = d / "rows.npz"
    if art.exists():
        log(f"  {arm}(seed {seed}): rows already on disk")
        return {"incomplete": False}
    cond = load_condition()
    grid = cond["grid"]
    obj = build_arm(arm, out, seed, log=log)
    J = _judge(int(PINS["bridge_seed"]), log)
    pool = field_fit_pool(cond, seed)
    fitted = np.setdiff1d(np.asarray(cond["train_rows"], np.int64),
                          excluded_rows(cond)["all"])
    bitmap = value_presence_bitmap(grid, fitted)
    c3, c4 = row3_corners(cond), row4_corners(cond)
    r1 = row1_requests(cond, pool)
    corners = ([{"corner_id": "r1_seen", "family": "r1", "rows": r1}]
               + c3 + c4)

    part = d / "partial.npz"
    done: dict = {}
    if part.exists():
        done = dict(np.load(part, allow_pickle=True)["done"].item())
        log(f"  {arm}(seed {seed}): resuming with {len(done)} corners done")
    t0 = _now()
    for c in corners:
        if c["corner_id"] in done:
            continue
        if _now() - t0 > float(max_seconds):
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(part, done=np.array(done, dtype=object))
            log(f"  {arm}(seed {seed}): wall-clock cap after {len(done)} corners -- "
                "state saved, rerun to continue")
            return {"incomplete": True, "n_done": len(done)}
        vals = grid.values_of(np.asarray(c["rows"], np.int64))
        feats, oor = arm_features(obj, vals, grid)
        acc = accept_rule(vals, oor, np.zeros(vals.shape[0], bool), bitmap, grid)
        tup = J.read_back(feats, int(PINS["eval_batch"]))
        val = judge_valid(vals, tup, grid) & acc
        done[c["corner_id"]] = {"accept": acc, "valid": val,
                                "tuples": np.where(acc[:, None], tup, -1),
                                "n": int(vals.shape[0]), "family": c["family"]}
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(part, done=np.array(done, dtype=object))
    np.savez_compressed(art, corners=np.array(done, dtype=object))
    meta = {"arm": arm, "seed": int(seed), "n_corners": len(done),
            "byte_ledger": obj["byte_ledger"], "seconds": _now() - t0}
    if "head_bytes" in obj:
        meta["head_bytes"] = obj["head_bytes"]
    _jdump(meta, d / "rows.json")
    log(f"  {arm}(seed {seed}): {len(done)} corners in {_now() - t0:.1f}s")
    return {"incomplete": False, "n_done": len(done)}


#: Which factor every corner on this scoreboard holds CONSTANT.  Not an assumption: it is
#: asserted per corner below, and ``corners.json``'s ``corner_geometry`` records it for the
#: scored corner sets as well as for this hub's val ones.
GATE_FACTOR = 4          # shape


def per_corner(corners, cor: dict, grid, seen_mask) -> list:
    """One row per corner, with the metric DECOMPOSED along the scoreboard's own geometry.

    Every corner holds ``shape`` constant and varies the three hues over 2 values, ``scale``
    over 3 and ``orientation`` over 3 (``corners.json:corner_geometry``).  So ``shape`` can
    only act as a per-corner GATE -- it cannot generate distinct tuples -- while the count
    of distinct tuples is driven by the factors that do vary.

    ``..._novel_tuples`` is the scoreboard metric, unchanged.
    ``..._novel_tuples_gate_held`` counts the same thing with the gate factor supplied by
    the CORNER rather than by the judge: the judged non-gate factors must all be right, and
    the tuple is completed with the corner's own constant ``shape``.  ⭐ It is computed for
    EVERY corner and conditions on NOTHING the arm produced, which is what makes it a fair
    decomposition rather than a comparison restricted to the corners an arm got right.
    """
    cards = tuple(int(c) for c in grid.cardinalities)
    keepf = [f for f in range(len(cards)) if f != GATE_FACTOR]
    out = []
    for c in corners:
        r = cor[c["corner_id"]]
        v = np.asarray(r["valid"], bool)
        t = np.asarray(r["tuples"])
        acc = np.asarray(r["accept"], bool)
        true = grid.factor_indices_of(np.asarray(c["rows"], np.int64))
        gate_vals = np.unique(true[:, GATE_FACTOR])
        if gate_vals.size != 1:
            raise AssertionError(
                f"{c['corner_id']}: factor {GATE_FACTOR} takes {gate_vals.size} values in "
                "this corner, so it is not a gate here and the decomposition does not apply")
        keep = t[v]
        if keep.size:
            lin = np.unique(np.ravel_multi_index(tuple(keep.T), cards))
            novel = lin[~seen_mask[lin]]
        else:
            lin = np.zeros(0, np.int64)
            novel = np.zeros(0, np.int64)
        ok5 = acc & np.all(t[:, keepf] == true[:, keepf], axis=1)
        if ok5.any():
            g = np.array(t[ok5])
            g[:, GATE_FACTOR] = int(gate_vals[0])
            lin5 = np.unique(np.ravel_multi_index(tuple(g.T), cards))
            novel5 = lin5[~seen_mask[lin5]]
        else:
            lin5 = np.zeros(0, np.int64)
            novel5 = np.zeros(0, np.int64)
        out.append({"corner_id": c["corner_id"], "n_requests": int(r["n"]),
                    "n_accepted": int(acc.sum()),
                    "n_valid": int(v.sum()),
                    "judge_accuracy": float(v.mean()) if v.size else 0.0,
                    "gate_accuracy": float((t[:, GATE_FACTOR]
                                            == true[:, GATE_FACTOR]).mean()),
                    "distinct_valid_tuples": int(lin.size),
                    "distinct_valid_novel_tuples": int(novel.size),
                    "distinct_valid_novel_tuples_gate_held": int(novel5.size)})
    return out


# ---------------------------------------------------------------------------
# 4. ⭐⭐ The deletion stage -- the acceptance property, asserted byte-wise
# ---------------------------------------------------------------------------


def _byte_check(before, after, ranges, label: str, log: Callable) -> dict:
    """Run the acceptance assertion and RECORD it, pass or fail.

    The assertion raises by design -- an approximate version of this property is worth
    nothing -- but a raise in the middle of the stage would leave the other deletions
    unmeasured.  So a failure is captured, marked, and re-stated in the report, and the
    stage's ``ALL_BYTE_PROPERTIES_HOLD`` flag is what a reader checks.
    """
    from ..core.gen_field import assert_changed_bytes_within, byte_diff

    try:
        rep = assert_changed_bytes_within(before, after, ranges, label)
        rep["holds"] = True
    except AssertionError as exc:
        d = byte_diff(before, after)
        rep = {"label": label, "holds": False, "error": str(exc),
               "bytes_changed": int(d["n_changed"]),
               "per_segment": d["per_segment"],
               "total_image_bytes": int(d["total_bytes"]),
               "permitted_byte_span": int(sum(int(e) - int(s) for s, e in ranges))}
        log(f"  !! {label}: THE BYTE PROPERTY DOES NOT HOLD -- {exc}")
    return rep


def deletion_proof(out: Path, seed: int, log: Callable = print) -> dict:
    """Three deletions on the field store, each with its changed-byte count PRINTED.

    1. ``evict``  -- a value deletion (``object_hue = 0.0``): every item holding the value
       is zeroed in every well.  Permitted bytes: those items'.
    2. ``hole``   -- the RED CYLINDER conjunction (``object_hue = 0.0 AND shape = cylinder``):
       ONE byte of ``coupling_hole``.  Permitted bytes: that byte.
    3. ``decay``  -- a per-item lifetime set to zero without a refit.

    Each is asserted with :func:`gen_field.assert_changed_bytes_within`, which RAISES if a
    single byte outside the permitted ranges moved, and with
    :func:`gen_field.assert_no_refit`, which requires every fitted segment to be
    byte-identical across the operation.
    """
    from ..core.gen_field import (
        assert_no_refit, byte_diff, coupling_index, decay_items, evict_term, field_read,
        gap_read, punch_hole, set_lifetime, term_index,
    )

    cond = load_condition()
    grid = cond["grid"]
    base = build_field_arm(out, seed, log=log)
    f0 = base["field"]
    cards = [int(c) for c in grid.cardinalities]
    pool = field_fit_pool(cond, seed)
    rng = np.random.default_rng(20_260_919)
    probe_rows = pool[rng.permutation(pool.size)[:256]]
    probe_vals = grid.values_of(probe_rows)
    fac, valx = int(PINS["delete_factor"]), int(PINS["delete_value"])
    (hf, hv), (hg, hu) = PINS["hole_pair"]
    rep = {"seed": int(seed), "image_bytes": int(f0.buf.size),
           "manifest": f0.manifest()}

    # ---- 1. EVICT a value ------------------------------------------------
    field = f0.copy()
    before = field.copy()
    pre_read = field_read(field, probe_vals, grid.values)
    pre_gap = gap_read(field, probe_vals, grid.values)
    ev = evict_term(field, fac, valx, log=log)
    acc = _byte_check(before, field, ev["permitted_ranges"],
                      f"evict factor {fac} = value {valx}", log)
    nr = assert_no_refit(before, field)
    post_gap = gap_read(field, probe_vals, grid.values)
    post_read = field_read(field, probe_vals, grid.values)
    # which probe requests ASK for the deleted value
    asks = np.isclose(probe_vals[:, fac], float(grid.values[fac][valx]))
    rep["evict"] = {
        **{k: v for k, v in ev.items() if k != "permitted_ranges"},
        "byte_assertion": {k: v for k, v in acc.items() if k != "outside_examples"},
        "no_refit": nr,
        "refused_before_on_deleted_requests": float(pre_gap["refuse"][asks].mean())
        if asks.any() else None,
        "refused_after_on_deleted_requests": float(post_gap["refuse"][asks].mean())
        if asks.any() else None,
        "refused_after_on_surviving_requests": float(post_gap["refuse"][~asks].mean()),
        "read_changed_on_surviving_requests_frac": float(
            (np.abs(post_read[~asks] - pre_read[~asks]) > 0).any(axis=1).mean()),
        "read_rel_change_on_surviving_requests": float(
            np.linalg.norm(post_read[~asks] - pre_read[~asks])
            / max(np.linalg.norm(pre_read[~asks]), 1e-12)),
        "live_atoms_in_deleted_basin_after": int(
            post_gap["term_live_atoms"][term_index(cards, fac, valx)]),
    }
    log(f"  EVICT object_hue index {valx}: {acc['bytes_changed']:,} bytes changed of "
        f"{acc['total_image_bytes']:,}; OUTSIDE the evicted atoms: "
        f"{acc['bytes_changed_outside_permitted']}")

    # ---- 2. THE HOLE -----------------------------------------------------
    field2 = f0.copy()
    before2 = field2.copy()
    pre2 = field_read(field2, probe_vals, grid.values)
    pre_g2 = gap_read(field2, probe_vals, grid.values)
    ho = punch_hole(field2, hf, hv, hg, hu, log=log)
    acc2 = _byte_check(before2, field2, ho["permitted_ranges"],
                       "hole (the red cylinder)", log)
    nr2 = assert_no_refit(before2, field2)
    post2 = field_read(field2, probe_vals, grid.values)
    post_g2 = gap_read(field2, probe_vals, grid.values)
    # WARNING, and it is why these probes are CONSTRUCTED rather than filtered: the red
    # cylinder is held out of the split entirely, so no pool row realises it and a mask
    # over pool rows would have tested the hole on ZERO requests.  Four groups, same 256
    # backgrounds, with the two factors set: the conjunction, each marginal alone, neither.
    def _set(vals, pairs):
        v = np.array(vals, np.float64, copy=True)
        for ax, ix in pairs:
            v[:, ax] = float(grid.values[ax][ix])
        return v

    alt_f = (hv + 1) % len(grid.values[hf])
    alt_g = (hu + 1) % len(grid.values[hg])
    P_both = _set(probe_vals, [(hf, hv), (hg, hu)])
    P_a = _set(probe_vals, [(hf, hv), (hg, alt_g)])
    P_b = _set(probe_vals, [(hf, alt_f), (hg, hu)])
    P_none = _set(probe_vals, [(hf, alt_f), (hg, alt_g)])
    g_both_pre = gap_read(f0.copy(), P_both, grid.values)
    g_both = gap_read(field2, P_both, grid.values)
    g_a = gap_read(field2, P_a, grid.values)
    g_b = gap_read(field2, P_b, grid.values)
    g_none = gap_read(field2, P_none, grid.values)
    # ⭐ the trap test: an UNSEEN pair must still be answered.  The val row-3 pairs are
    # conjunctions the pool never held; none of them is a hole, so none may be refused.
    c3 = row3_corners(cond)
    unseen_vals = np.concatenate(
        [grid.values_of(np.asarray(c["rows"], np.int64))[:8] for c in c3[:12]])
    ug = gap_read(field2, unseen_vals, grid.values)
    ci = coupling_index(cards, hf, hv, hg, hu)
    rep["hole"] = {
        **{k: v for k, v in ho.items() if k != "permitted_ranges"},
        "byte_assertion": {k: v for k, v in acc2.items() if k != "outside_examples"},
        "no_refit": nr2,
        "n_probe_requests_per_group": int(P_both.shape[0]),
        "hole_flag_before_on_conjunction": float(g_both_pre["hole"].mean()),
        "hole_flag_after_on_conjunction": float(g_both["hole"].mean()),
        "hole_flag_after_on_marginal_A_only": float(g_a["hole"].mean()),
        "hole_flag_after_on_marginal_B_only": float(g_b["hole"].mean()),
        "hole_flag_after_on_neither": float(g_none["hole"].mean()),
        "refused_after_on_marginal_A_only": float(g_a["refuse"].mean()),
        "refused_after_on_marginal_B_only": float(g_b["refuse"].mean()),
        "rendered_features_bit_identical": bool(np.array_equal(pre2, post2)),
        "UNSEEN_pairs_refused_after_the_hole": float(ug["refuse"].mean()),
        "UNSEEN_pairs_hole_flagged_after_the_hole": float(ug["hole"].mean()),
        "n_unseen_pair_requests_checked": int(unseen_vals.shape[0]),
        "coupling_value_at_the_hole": float(field2.view("coupling")[ci]),
        "coupling_evidence_weight_at_the_hole": float(field2.view("coupling_w")[ci]),
        "refusal_changed_on_the_general_probe_set": float(
            (pre_g2["refuse"] != post_g2["refuse"]).mean()),
        "n_general_probes": int(probe_vals.shape[0]),
    }
    log(f"  HOLE (red cylinder): {acc2['bytes_changed']} byte(s) changed of "
        f"{acc2['total_image_bytes']:,}; OUTSIDE the hole: "
        f"{acc2['bytes_changed_outside_permitted']}; unseen-pair refusal rate "
        f"{rep['hole']['UNSEEN_pairs_refused_after_the_hole']:.4f}")

    # ---- 3. DECAY --------------------------------------------------------
    field3 = f0.copy()
    before3 = field3.copy()
    ids = np.arange(0, f0.A, 8)
    half = decay_items(field3, ids, 0.5, log=log)
    acc3a = _byte_check(before3, field3, half["permitted_ranges"],
                        "decay to half", log)
    mid = field3.copy()
    zero = decay_items(field3, ids, 0.0, log=log)
    acc3b = _byte_check(mid, field3, zero["permitted_ranges"],
                        "decay to zero", log)
    nr3 = assert_no_refit(before3, field3)
    rep["decay"] = {
        "half": {**{k: v for k, v in half.items() if k != "permitted_ranges"},
                 "byte_assertion": {k: v for k, v in acc3a.items()
                                    if k != "outside_examples"}},
        "to_zero": {**{k: v for k, v in zero.items() if k != "permitted_ranges"},
                    "byte_assertion": {k: v for k, v in acc3b.items()
                                       if k != "outside_examples"}},
        "no_refit": nr3,
    }
    log(f"  DECAY: half -> {acc3a['bytes_changed']} bytes changed "
        f"({acc3a['bytes_changed_outside_permitted']} outside); to zero -> "
        f"{acc3b['bytes_changed']} bytes changed "
        f"({acc3b['bytes_changed_outside_permitted']} outside)")

    # ---- 3b. a per-item LIFETIME set without a refit ---------------------
    field5 = f0.copy()
    before5 = field5.copy()
    lf = set_lifetime(field5, [3, 11, 29], 0.25, log=log)
    acc5 = _byte_check(before5, field5, lf["permitted_ranges"], "set_lifetime", log)
    rep["set_lifetime"] = {**{k: v for k, v in lf.items() if k != "permitted_ranges"},
                           "byte_assertion": {k: v for k, v in acc5.items()
                                              if k != "outside_examples"},
                           "no_refit": assert_no_refit(before5, field5)}
    log(f"  LIFETIME: {acc5['bytes_changed']} bytes changed "
        f"({acc5.get('bytes_changed_outside_permitted', 'n/a')} outside)")

    # ---- 4. the negative control: a REFIT is visible as a byte storm ----
    field4 = f0.copy()
    before4 = field4.copy()
    field4.view("term_mu")[0] += np.float32(1e-3)
    d4 = byte_diff(before4, field4)
    rep["negative_control_refit"] = {
        "what": ("a single 1e-3 nudge to ONE row of the address book -- the smallest "
                 "possible refit -- to show the instrument is not blind"),
        "bytes_changed": int(d4["n_changed"]), "per_segment": d4["per_segment"],
        "would_fail_assert_no_refit": True,
    }
    log(f"  NEGATIVE CONTROL: nudging one address-book row moves "
        f"{d4['n_changed']:,} bytes -- the instrument is not blind")
    rep["ALL_BYTE_PROPERTIES_HOLD"] = bool(
        rep["evict"]["byte_assertion"].get("holds")
        and rep["hole"]["byte_assertion"].get("holds")
        and rep["decay"]["half"]["byte_assertion"].get("holds")
        and rep["decay"]["to_zero"]["byte_assertion"].get("holds")
        and rep["set_lifetime"]["byte_assertion"].get("holds"))
    log(f"  ALL BYTE PROPERTIES HOLD: {rep['ALL_BYTE_PROPERTIES_HOLD']}")
    _jdump(rep, seed_dir(out, seed) / "deletion.json")
    return rep


# ---------------------------------------------------------------------------
# 4b. The dial sweep -- SELECTION, on the val corners only, with no judge
# ---------------------------------------------------------------------------


def sweep(out: Path, seed: int = 0, log: Callable = print) -> dict:
    """Choose the field store's dials on ``val``, by feature reconstruction error.

    Charter section A.2: selection happens on ``val`` and nowhere else.  This stage never
    calls the judge -- it scores the read against the TRUE encoder features of the rows
    that realise each val corner, which is a cheap proxy and costs no scored run.  It is
    run BEFORE the smoke, and the dials it picks are then pinned and not touched again.
    """
    from ..core.gen_field import FieldSpec, build_field, field_read

    cond = load_condition()
    grid = cond["grid"]
    parts = build_partitions(out / "partitions", log=log)
    labels = np.asarray(parts["labels"]["factor"], np.int64)
    hosts = _host_factors(parts["info"], int(labels.max()) + 1)
    cards = [int(c) for c in grid.cardinalities]
    pool = field_fit_pool(cond, seed)
    F = features_mmap(int(PINS["bridge_seed"]))
    X = gather(F, pool)
    Y = grid.factor_indices_of(pool)
    c3, c4 = row3_corners(cond), row4_corners(cond)
    ev_rows = np.concatenate([c["rows"] for c in c3 + c4])
    ev_vals = grid.values_of(ev_rows)
    ev_true = gather(F, ev_rows)
    nrm = np.linalg.norm(ev_true, axis=1)

    def score(xhat):
        e = np.linalg.norm(xhat - ev_true, axis=1) / np.maximum(nrm, 1e-9)
        cs = ((xhat * ev_true).sum(1)
              / np.maximum(np.linalg.norm(xhat, axis=1) * nrm, 1e-9))
        return {"rel_err": float(e.mean()), "cosine": float(cs.mean())}

    rows = []
    for ce in (0.0, 1.0):
        for sm in (0.5, 1.0, 2.0, 4.0):
            for rf in (1e-3, 1e-1):
                sp = FieldSpec(n_atoms=int(PINS["field_n_atoms"]), eta=1.0,
                               ridge_frac=rf, sigma_mult=sm, coupling_eta=ce,
                               coupling_prior_strength=float(
                                   PINS["field_coupling_prior_strength"]),
                               coupling_rank=int(PINS["field_coupling_rank"]),
                               atom_select=str(PINS["field_atom_select"]),
        decontaminate=float(PINS["field_decontaminate"]),
        term_map=str(PINS["field_term_map"]),
                               coupling_shrink=float(PINS["field_coupling_shrink"]),
                               atom_seed=int(PINS["field_atom_seed"])
        + int(PINS["atom_seed_stride"]) * int(store_seed))
                fld = build_field(X, Y, pool, labels, hosts, cards, sp,
                                  log=lambda *_: None)
                r = {"coupling_eta": ce, "sigma_mult": sm, "ridge_frac": rf,
                     "eta": 1.0, **score(field_read(fld, ev_vals, grid.values))}
                # eta = 0 is the address book ALONE: the landscape does nothing.  It is
                # reported so the atoms' contribution is a measured number, not a claim.
                fld.meta["spec"]["eta"] = 0.0
                r["rel_err_address_book_only"] = score(
                    field_read(fld, ev_vals, grid.values))["rel_err"]
                rows.append(r)
                log(f"  coupling_eta={ce} sigma_mult={sm} ridge={rf}: "
                    f"rel_err={r['rel_err']:.4f} cos={r['cosine']:.4f} "
                    f"(address book alone {r['rel_err_address_book_only']:.4f})")
    del X
    best = min(rows, key=lambda r: r["rel_err"])
    inc = None
    p = seed_dir(out, seed) / "arms" / "aligned_w3" / "arm.pkl"
    if p.exists():
        from ..core.gen_request_head import head_read, request_design
        with open(p, "rb") as fh:
            a = pickle.load(fh)
        Phi, _ = request_design(ev_vals, grid.values, a["spec"])
        inc = score(head_read(a["heads"], a["store"], Phi))
        log(f"  INCUMBENT aligned_w3: rel_err={inc['rel_err']:.4f} "
            f"cos={inc['cosine']:.4f}")
    rep = {"grid": rows, "best": best, "incumbent_aligned_w3": inc,
           "n_eval_rows": int(ev_rows.size), "seed": int(seed),
           "note": ("selection on the val corners by feature reconstruction error; no "
                    "judge, no scored run, charter section A.2")}
    _jdump(rep, out / f"sweep_seed{int(seed)}.json")
    return rep


# ---------------------------------------------------------------------------
# 4c. The content ceiling -- a DECLARED NON-ARM
# ---------------------------------------------------------------------------


def content_ceiling(out: Path, seed: int = 0, condition: str = "val",
                    log: Callable = print) -> dict:
    """⛔ **NOT AN ARM.**  It reads a FEATURE, so it can never be quoted as a mechanism.

    The question it answers, and it is the only way to answer it cheaply: how much of the
    factor structure do the 64 written items CARRY, independently of whether the address
    can reach it?  Project the TRUE encoder feature of each corner row onto each well's
    atom span and judge the result per factor.  That is the best any read of these atoms
    could do.  A factor the ceiling already misses is a CONTENT problem and no better
    address will fix it; a factor the ceiling reads and the deployed read misses is an
    ADDRESS problem.

    (The construction is the one `hub/read` calls ``oracle_span``; it is reported here in
    the same declared NON-ARM status.)

    ⚠⚠ ``condition="test"`` runs this on the SCORED corners.  ``hub/read`` measured the same
    ceiling on val and on the scored set and got **orientation 0.550 vs 0.8804** — it
    concluded from val that the atoms could not carry orientation and that was WRONG.  The
    val Montero slots are a different region of the grid; `hub/placement` independently saw
    a val margin shrink 8.6x on the scored set.  ⛔ So the val ceiling in this hub's report
    bounds the OBJECT, not the scored rows, and the scored-corner ceiling is the first thing
    wave 5 should measure.  ⛔ It is a number on the held-out set and therefore a SCORED RUN
    (charter section A.2): it needs a PREREG file and a row in
    ``/c/chlu/SCORED-LEDGER.md`` BEFORE it runs.  ``hub/field-store`` has a budget of ZERO
    and did not run it; the guard below is why.
    """
    if str(condition) == "test" and not os.environ.get("CHLU_FIELD_SPEND_SCORED_RUN"):
        raise SystemExit(
            "REFUSED: --condition test puts a number on the held-out set, which is a "
            "SCORED RUN under charter section A.2. hub/field-store's budget is ZERO and "
            "it did not spend it. To run this deliberately from a hub that HAS budget: "
            "pre-register it, append the row to /c/chlu/SCORED-LEDGER.md, then set "
            "CHLU_FIELD_SPEND_SCORED_RUN=1.")
    cond = load_condition()
    grid = cond["grid"]
    names = list(grid.names)
    base = build_field_arm(out, seed, log=log)
    field = base["field"]
    J = _judge(int(PINS["bridge_seed"]), log)
    F = features_mmap(int(PINS["bridge_seed"]))
    pool = field_fit_pool(cond, seed)
    if str(condition) == "test":
        from .exp_gen_factor_wells import nway_corners, pair_subcorners
        c3, c4 = pair_subcorners(cond), nway_corners(cond)
    else:
        c3, c4 = row3_corners(cond), row4_corners(cond)
    sets = {"r1_seen": row1_requests(cond, pool),
            "r3_2way": np.concatenate([c["rows"] for c in c3])[::3],
            "r4_nway": np.concatenate([c["rows"] for c in c4])[::2]}
    rep = {"seed": int(seed), "condition": str(condition),
           "declared": "NON-ARM: this reads a feature (charter B.3)"}
    for name, rows in sets.items():
        Xt = gather(F, rows)
        proj = np.zeros_like(Xt)
        for k in range(field.K):
            ch = np.asarray(field.meta["well_channels"][k], np.int64)
            A = np.asarray(field.atoms_of_well(k), np.float64)
            live = np.asarray(field.view("atom_alive"), np.uint8) > 0
            A = A[live]
            G = A @ A.T
            G[np.arange(G.shape[0]), np.arange(G.shape[0])] +=                 1e-6 * np.trace(G) / max(1, G.shape[0])
            M = np.asarray(Xt[:, ch], np.float64)
            c = np.linalg.solve(G, (M @ A.T).T).T
            proj[:, ch] = (c @ A).astype(np.float32)
        tup = J.read_back(proj, int(PINS["eval_batch"]))
        raw = J.read_back(Xt, int(PINS["eval_batch"]))
        true = grid.factor_indices_of(rows)
        rep[name] = {
            "n": int(rows.size),
            "oracle_span_per_factor": {names[f]: float((tup[:, f] == true[:, f]).mean())
                                       for f in range(6)},
            "true_feature_per_factor": {names[f]: float((raw[:, f] == true[:, f]).mean())
                                        for f in range(6)},
            "rel_err_of_projection": float(
                (np.linalg.norm(proj - Xt, axis=1)
                 / np.maximum(np.linalg.norm(Xt, axis=1), 1e-9)).mean()),
        }
        log(f"  ceiling {name}: " + "  ".join(
            f"{names[f]}={rep[name]['oracle_span_per_factor'][names[f]]:.3f}"
            f"/{rep[name]['true_feature_per_factor'][names[f]]:.3f}" for f in range(6)))
    _jdump(rep, seed_dir(out, seed) / f"content_ceiling_{condition}.json")
    return rep


# ---------------------------------------------------------------------------
# 5. The report stage
# ---------------------------------------------------------------------------


def build_report(out: Path, log: Callable = print) -> dict:
    cond = load_condition()
    grid = cond["grid"]
    excl = excluded_rows(cond)
    fitted = np.setdiff1d(np.asarray(cond["train_rows"], np.int64), excl["all"])
    total = int(np.prod([int(c) for c in grid.cardinalities]))
    seen_mask = np.zeros(total, bool)
    seen_mask[fitted] = True
    c3, c4 = row3_corners(cond), row4_corners(cond)

    tables: dict = {}
    for seed in PINS["seeds"]:
        for arm in ARMS:
            p = seed_dir(out, seed) / "arms" / arm / "rows.npz"
            if not p.exists():
                continue
            cor = np.load(p, allow_pickle=True)["corners"].item()
            r1 = cor.get("r1_seen")
            tables.setdefault(arm, {})[int(seed)] = {
                "r1_seen": {"judge_accuracy": float(np.asarray(r1["valid"]).mean()),
                            "accept": float(np.asarray(r1["accept"]).mean()),
                            "n": int(r1["n"])} if r1 is not None else None,
                "r3": per_corner(c3, cor, grid, seen_mask),
                "r4": per_corner(c4, cor, grid, seen_mask),
            }

    def _agg(arm, row, metric):
        vals = []
        for seed in PINS["seeds"]:
            t = tables.get(arm, {}).get(int(seed))
            if t is None:
                continue
            if row == "r1_seen":
                vals.append(t["r1_seen"]["judge_accuracy"])
            else:
                vals.append(float(np.mean([r[metric] for r in t[row]])))
        return vals

    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for row, metric in (("r1_seen", "judge_accuracy"),
                            ("r3", "distinct_valid_novel_tuples"),
                            ("r4", "distinct_valid_novel_tuples")):
            v = _agg(arm, row, metric)
            summary[arm][row] = {
                "per_seed": v, "mean": float(np.mean(v)) if v else None,
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                "metric": "judge accuracy" if row == "r1_seen"
                else "distinct valid NOVEL tuples per corner",
            }
        for row, metric in (("r3", "judge_accuracy"), ("r4", "judge_accuracy")):
            v = _agg(arm, row, metric)
            summary[arm][row + "_accuracy"] = {
                "per_seed": v, "mean": float(np.mean(v)) if v else None,
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}

    # the paired per-corner estimand, field vs the incumbent, per seed
    estimand = {}
    gate_held = {}
    for seed in PINS["seeds"]:
        a = tables.get("field", {}).get(int(seed))
        b = tables.get("aligned_w3", {}).get(int(seed))
        if a is None or b is None:
            continue
        for row in ("r3", "r4"):
            estimand.setdefault(row, {})[int(seed)] = paired_estimand(
                a[row], b[row], "distinct_valid_novel_tuples",
                WITHIN_BAND_NOVELTY, "field", "aligned_w3")
            # ⭐ THE DECOMPOSITION, unconditioned: the same paired estimand with the gate
            # factor supplied by the corner instead of by the judge.  Every corner is in it.
            gate_held.setdefault(row, {})[int(seed)] = paired_estimand(
                a[row], b[row], "distinct_valid_novel_tuples_gate_held",
                WITHIN_BAND_NOVELTY, "field", "aligned_w3")

    def _spread(row, key):
        a = summary.get("field", {}).get(key, {})
        b = summary.get("aligned_w3", {}).get(key, {})
        if a.get("mean") is None or b.get("mean") is None or not b.get("sd"):
            return {}
        z = (a["mean"] - b["mean"]) / b["sd"]
        return {"field_mean": a["mean"], "field_sd": a["sd"],
                "incumbent_mean": b["mean"], "incumbent_seed_sd": b["sd"],
                "gap_in_incumbent_sds": float(z),
                "INSIDE_INCUMBENT_SEED_SPREAD": bool(abs(z) <= 1.0),
                "direction": ("field ahead" if z > 0 else "field behind")}

    verdict = {}
    for row in ("r3", "r4"):
        ds = [estimand.get(row, {}).get(int(s), {}).get("mean_paired_delta")
              for s in PINS["seeds"]]
        ds = [d for d in ds if d is not None]
        inside = (bool(abs(float(np.mean(ds))) <= WITHIN_BAND_NOVELTY) if ds else None)
        verdict[row] = {
            "mean_paired_delta_per_seed": ds,
            "mean_over_seeds": float(np.mean(ds)) if ds else None,
            "per_seed_all_inside_band": (bool(np.all(np.abs(ds) <= WITHIN_BAND_NOVELTY))
                                         if ds else None),
            "band_declared_in_advance": WITHIN_BAND_NOVELTY,
            "WITHIN_THE_DECLARED_BAND": inside,
            "vs_incumbent_seed_spread": _spread(row, row),
        }
    d1 = [x - y for x, y in zip(summary.get("field", {}).get("r1_seen", {}).get("per_seed", []),
                                summary.get("aligned_w3", {}).get("r1_seen", {}).get("per_seed", []))]
    verdict["r1_seen"] = {
        "accuracy_delta_per_seed": d1,
        "mean_over_seeds": float(np.mean(d1)) if d1 else None,
        "band_declared_in_advance": WITHIN_BAND_ACCURACY,
        "WITHIN_THE_DECLARED_BAND": (bool(abs(float(np.mean(d1))) <= WITHIN_BAND_ACCURACY)
                                     if d1 else None),
        "vs_incumbent_seed_spread": _spread("r1_seen", "r1_seen"),
    }

    gate_acc = {}
    for arm in ARMS:
        for row in ("r3", "r4"):
            vals = [float(np.mean([r["gate_accuracy"] for r in t[row]]))
                    for sd in PINS["seeds"]
                    if (t := tables.get(arm, {}).get(int(sd))) is not None]
            if vals:
                gate_acc.setdefault(arm, {})[row] = {
                    "per_seed": vals, "mean": float(np.mean(vals))}
    rep = {"summary": summary, "paired_estimand": estimand,
           "paired_estimand_gate_held": gate_held,
           "gate_factor": {"index": GATE_FACTOR, "name": str(grid.names[GATE_FACTOR]),
                           "why": ("every corner on this scoreboard holds it constant, so "
                                   "it gates validity and cannot generate distinct tuples"),
                           "accuracy": gate_acc},
           "verdict": verdict, "tables": tables,
           "bands_declared_in_advance": {"novelty_rows_3_4": WITHIN_BAND_NOVELTY,
                                         "accuracy_row_1": WITHIN_BAND_ACCURACY},
           "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PINS.items()},
           "hub": "hub/field-store", "scored_runs_spent": 0}
    _jdump(rep, out / "report.json")
    log(json.dumps(verdict, indent=1, default=str))
    return rep


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("corners", "sweep", "fit", "run", "ceiling",
                            "delete", "report"))
    p.add_argument("--arm", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seconds", type=float, default=3600.0)
    p.add_argument("--condition", default="val", choices=("val", "test"),
                   help=("val is the selection condition and the only one this hub used; "
                         "test is a SCORED RUN and is guarded"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = _now()

    def log(msg):
        print(f"[{_now() - t0:7.1f}s] {msg}", flush=True)

    if args.stage == "corners":
        cond = load_condition()
        pool = field_fit_pool(cond, args.seed)
        c3, c4 = row3_corners(cond), row4_corners(cond)
        r1 = row1_requests(cond, pool)
        info = assert_smoke_is_not_scored(cond, c3, c4, r1, pool)
        ex = excluded_rows(cond)
        from .exp_gen_factor_wells import nway_corners, pair_subcorners

        def _geometry(corners):
            P = np.array([[len(np.unique(
                cond["grid"].factor_indices_of(np.asarray(c["rows"], np.int64))[:, f]))
                for f in range(6)] for c in corners])
            return {str(cond["grid"].names[f]): {
                "distinct_values_per_corner": sorted(set(int(x) for x in P[:, f])),
                "constant_in_n_corners": int((P[:, f] == 1).sum()),
                "n_corners": int(P.shape[0])} for f in range(6)}
        # ⭐ The FIGURE OF MERIT's geometry, recorded so nobody has to re-derive it: every
        # corner on this scoreboard holds SHAPE constant and varies the three hues over 2
        # values, scale over 3 and orientation over 3.  Every arm measured here reads the
        # hues at 1.0000, so distinct-tuple COUNT is driven by scale x orientation and
        # shape acts as a per-corner GATE.  Structural: it reads the condition's own cell
        # definitions and no held-out number, so it costs no scored run.
        info["corner_geometry"] = {
            "my_val_row3": _geometry(c3), "my_val_row4": _geometry(c4),
            "SCORED_row3": _geometry(pair_subcorners(cond)),
            "SCORED_row4": _geometry(nway_corners(cond)),
            "note": ("structural only -- the condition's cell definitions, not a number on "
                     "the held-out set"),
        }
        info["pool"] = {"n": int(pool.size),
                        "n_excluded_from_pool": int(ex["all"].size),
                        "n_val_nway_rows": int(ex["val_nway_rows"].size),
                        "n_val_pair_rows": int(ex["val_pair_rows"].size)}
        _jdump(info, out / "corners.json")
        log(json.dumps(info, indent=1, default=str))
    elif args.stage == "sweep":
        sweep(out, args.seed, log=log)
    elif args.stage == "fit":
        build_arm(args.arm, out, args.seed, log=log)
    elif args.stage == "run":
        run_arm(args.arm, out, args.seed, max_seconds=args.max_seconds, log=log)
    elif args.stage == "ceiling":
        content_ceiling(out, args.seed, condition=args.condition, log=log)
    elif args.stage == "delete":
        deletion_proof(out, args.seed, log=log)
    elif args.stage == "report":
        build_report(out, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
