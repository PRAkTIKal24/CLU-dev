"""hub/store-content — the content ablation.

⛔ SELECTION RUN by default (``--set val``): every number it produces is measured on
the store pool's own held-back validation rows, which no scored set touches.  The
scored sets are reached only with ``--set corners`` and only after a PREREG file
exists; the script refuses otherwise.

WHAT IT MEASURES, per content spec, per charter row::

    request -> address book -> per-well argmax atom -> penultimate slice
            -> THE frozen decoder -> THE external classifier -> per-factor accuracy

⭐ The per-factor breakdown is emitted on every row, because the aggregate can move
for reasons that have nothing to do with this hub's hypothesis and `scale` /
`orientation` moving IS the hypothesis.

RESUMABLE: one JSON per (spec, set) under ``runs/``; an existing file is skipped
unless ``--force``.  MEMORY: the pool matrix is the only large array (32,768 x D
float32 = 537 MB at D=4096) and the h5 is never opened.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import SPECS, LEGAL, KILLED_BY_DECODER, BUDGET_BYTES, open_sources  # noqa
from arm import build_arm, read_arm, penult_of, check_P_reproduces_cache, ROWS  # noqa
from instrument import load_instrument, render, judge, per_factor_accuracy, FACTORS  # noqa
from estimand import per_corner_metrics  # noqa

from chlu.data import shapes3d as ds                                    # noqa
from chlu.data.gen_value_holdout import (                               # noqa
    ValueHoldoutConfig, build_value_holdout_split, assert_value_absent_from_train)
from chlu.data.gen_rowsplit import stratified_fit_val                   # noqa

OUT = Path(os.environ.get("CHLU_HUB_OUTPUTS", "C:/chlu/outputs")) / "hub-store-content"

#: pinned, from the frozen gen_store config
N_STORE_TRAIN = 32_768
#: pinned, from the frozen gen_cell config
N_FIT, N_VAL = 32_256, 512
#: pinned RNG stream of the store pool draw - gen_store.select_store_rows' own
POOL_SEED_MULT, POOL_SEED_OFF = 7919, 20_260_910
#: ⭐ Declared subsample of each pair corner (they hold 12,000 rows each). 1,176 x 3
#: = 3,528, exactly the size of the 49-corner n-way set.
PAIR_CAP = 1_176


def build_world(seed: int, log=print) -> dict:
    """Grid, value-holdout split, store pool, fit/val cut.  Fully pinned."""
    t0 = time.time()
    labels = ds.load_labels(None)
    grid = ds.build_grid(labels)
    vcfg = ValueHoldoutConfig()
    split, cells = build_value_holdout_split(grid, vcfg)
    # the hold-out property itself, checked on the train index
    n_ck = assert_value_absent_from_train(split, vcfg.held_out_hue_factor,
                                          vcfg.held_out_hue_value)
    assert_value_absent_from_train(split, vcfg.held_out_shape_factor,
                                   vcfg.held_out_shape_value)

    rng = np.random.default_rng(int(seed) * POOL_SEED_MULT + POOL_SEED_OFF)
    pool = np.sort(rng.permutation(np.asarray(split.train_rows, np.int64))[:N_STORE_TRAIN])
    fit_rows, val_rows, sinfo = stratified_fit_val(
        pool, grid.factor_indices_of(pool), N_FIT, N_VAL, int(seed))
    if int(val_rows.size) != N_VAL:
        raise AssertionError(f"n_val requested {N_VAL}, realised {val_rows.size}")

    fams = {}
    for c in cells:
        fams.setdefault(c.meta.get("family"), []).append(c)
    log(f"  world: train {split.n_train:,} test {split.n_test:,} | pool {pool.size:,} "
        f"| fit {fit_rows.size:,} val {val_rows.size} | cells "
        f"{ {k: len(v) for k, v in fams.items()} } | {time.time()-t0:.1f}s "
        f"(holdout asserted on {n_ck:,} train rows)")
    return {"grid": grid, "split": split, "cells": cells, "families": fams,
            "pool": pool, "fit_rows": fit_rows, "val_rows": val_rows,
            "split_info": sinfo, "vcfg": vcfg}


def seen_tuples(grid, rows) -> set:
    """The factor tuples the store's fit pool contains - rows 3/4's novelty baseline.

    ⛔ Built from the FIT rows, which are exactly the rows the store was written from;
    an arm that replays one of these has produced nothing novel.
    """
    fi = grid.factor_indices_of(np.asarray(rows, np.int64))
    return {tuple(int(x) for x in t) for t in fi}


def request_matrix(grid, rows: np.ndarray) -> np.ndarray:
    """(n,) rows -> (n, F) factor-index request matrix.  A complete valid request."""
    return np.asarray(grid.factor_indices_of(np.asarray(rows, np.int64)), np.int64)


def eval_rows(world, which: str) -> tuple:
    """Return ``(rows, corner_ids, label)`` for an evaluation set.

    ``corner_ids`` is what makes the multi-corner paired estimand possible: every
    request carries the cell it came from, so the verdict is a paired statistic over
    corners and never a single pooled number (w3-hub-common §3).
    """
    if which == "val":
        r = np.asarray(world["val_rows"], np.int64)
        # `val` is one pooled set; it is a SELECTION set and never a verdict, so it
        # carries a single synthetic corner id and no paired statistic is computed.
        return r, np.array(["val"] * r.size, object), "val (store pool, held back)"
    if which in ("row1", "row2", "row5", "c_invalid"):
        return _query_rows(world, which)
    fam = {"corners": "nway", "pairs": "pair"}.get(which)
    if fam is None:
        raise ValueError(which)
    # ⭐ The 3 pair corners hold 12,000 rows EACH (36,000 total, 10x the n-way set).
    # Scoring all of them for 8 arms x 4 charter rows is ~3 h of render+judge for a
    # SUPPORTING breakdown. A seeded subsample of 1,176 per pair corner gives 3,528
    # requests - exactly the n-way set's size, so the two rows are directly
    # comparable - and it is DECLARED here rather than buried.
    cap = PAIR_CAP if fam == "pair" else 0
    rows, cids = [], []
    rng = np.random.default_rng(20_260_919)
    for c in world["families"][fam]:
        rr = c.rows(world["grid"]).astype(np.int64)
        if cap and rr.size > cap:
            rr = np.sort(rng.choice(rr, cap, replace=False))
        rows.append(rr)
        cids.append(np.array([c.cell_id] * rr.size, object))
    label = (f"the {len(world['families'][fam])} "
             f"{'n-way' if fam == 'nway' else 'pair'} corners"
             + (f" (seeded subsample of {cap}/corner, declared)" if cap else ""))
    return np.concatenate(rows), np.concatenate(cids), label


def _query_rows(world, which: str):
    """Rows 1, 2, 5 and the invalid set, as (rows, corner_ids, label).

    ⛔ Only records whose request is fully ON-GRID have a dataset row, so row 2 (an
    OFF-GRID hue by definition) and the out-of-range half of the invalid set have no
    row to read a per-factor target from. Those sets are scored in the seven-row
    table, not here, and this helper refuses rather than inventing a target.
    """
    import json as _json
    from pathlib import Path as _P
    VH = _P(os.environ.get("CHLU_GEN_DATA_ROOT", "C:/chlu/data/gen")) / "shapes3d" / "value_holdout"
    fn = {"row1": "row1_seen.json", "row2": "row2_unseen.json",
          "row5": "row5_held_out.json", "c_invalid": "c_invalid.json"}[which]
    d = _json.load(open(VH / fn))
    rows = [r["row"] for r in d["records"] if isinstance(r.get("row"), int)]
    if len(rows) != len(d["records"]):
        raise ValueError(
            f"{which}: {len(d['records']) - len(rows)} record(s) carry no dataset row "
            "(off-grid by construction) - this set has no per-factor target and is "
            "scored in the seven-row table instead")
    r = np.asarray(rows, np.int64)
    return r, np.array([which] * r.size, object), f"{which} ({fn})"


def run_spec(name: str, world, inst, src, eset: str, seed: int, log=print) -> dict:
    spec = SPECS[name]
    t0 = time.time()
    grid = world["grid"]
    pool = world["fit_rows"]                     # ⛔ the store fits on `fit`, never `val`
    lab_pool = request_matrix(grid, pool)
    arm = build_arm(spec, pool, lab_pool, list(grid.cardinalities), seed=seed,
                    src=src, log=log)

    prov = dict(arm.provenance)
    if name == "P":
        prov["P_reproduces_frozen_cache"] = check_P_reproduces_cache(arm, pool, src)
        log(f"  [P] reproduces the frozen cache: {prov['P_reproduces_frozen_cache']}")

    rows, corner_ids, set_label = eval_rows(world, eset)
    R = request_matrix(grid, rows)
    target = R                                    # the request IS the target tuple
    seen = seen_tuples(grid, world["fit_rows"]) if eset != "val" else None
    out = {"spec": spec.assert_matched(), "set": eset, "set_label": set_label,
           "n_requests": int(rows.size), "seed": int(seed),
           "grouping": arm.grouping, "byte_ledger": arm.ledger,
           "provenance": prov, "rows": {}}

    for row in ROWS:
        r = read_arm(arm, R, row=row, seed=seed)
        imgs = render(inst["decoder"], penult_of(arm, r["x"]))
        pred = judge(inst["classifier"], imgs)
        acc = per_factor_accuracy(pred, target)
        extra = {}
        if "atom_index" in r:
            ai = r["atom_index"]
            extra["n_distinct_index_tuples"] = int(np.unique(ai, axis=0).shape[0])
        if "atom_index_shared" in r:
            extra["n_distinct_rows_returned"] = int(np.unique(r["atom_index_shared"]).size)
        if "confidence" in r:
            extra["mean_confidence"] = float(np.mean(r["confidence"]))
        out["rows"][row] = {**acc, **extra, "launch_mode": r["launch_mode"],
                            "per_corner": per_corner_metrics(pred, target, corner_ids,
                                                             FACTORS, seen)}
        log(f"    {name:14s} {row:18s} agg {acc['aggregate']:.4f}  exact {acc['exact_match']:.4f}"
            f"  scale {acc['per_factor']['scale']:.4f}  orient {acc['per_factor']['orientation']:.4f}"
            f"  shape {acc['per_factor']['shape']:.4f}")
        del imgs
    out["wall_seconds"] = float(time.time() - t0)
    return out


def run_control(name: str, world, inst, src, eset: str, seed: int, log=print) -> dict:
    """``basis_head`` (the like-for-like) or ``knn_proto`` (the second row).

    ⛔ Both are fitted on the SAME `fit` rows, in the SAME penultimate feature space,
    and judged by the SAME external classifier on the SAME requests as every content
    arm.  A control that differs in any of those is not a control.
    """
    from content import SPECS as _S
    from controls import BasisHead, KnnProto

    # ⭐ "basis_head@WIDEN" runs the like-for-like READ on this hub's CONTENT, which
    # is the only way to vary the dial with the read held fixed.  Bare "basis_head"
    # is the wave-2 object in the penultimate space and is the required control.
    base, _, cname = name.partition("@")
    t0 = time.time()
    grid = world["grid"]
    pool = world["fit_rows"]
    spec = _S[cname or "P"]
    Zp, scales = build_Z_for(spec, pool, seed, src)
    R_fit = request_matrix(grid, pool)
    rows, corner_ids, set_label = eval_rows(world, eset)
    R = request_matrix(grid, rows)
    seen = seen_tuples(grid, pool) if eset != "val" else None

    out = {"arm": name, "base": base, "content": spec.name, "set": eset,
           "set_label": set_label, "spec": spec.assert_matched(),
           "n_requests": int(rows.size), "seed": int(seed), "rows": {}}
    if base == "basis_head":
        obj = BasisHead(Zp, R_fit, list(grid.cardinalities), spec.n_atoms,
                        seed=seed, log=log)
        x = obj.read(R)
        out["grouping"] = obj.grouping
        out["byte_ledger"] = obj.ledger
        out["head_bytes"] = obj.head_bytes
        out["coefficients"] = obj.coefficient_report(R)
        out["provenance"] = {"ridge_proj": 1e-2, "ridge_head": 1e-1,
                             "atom_selection": "kmeanspp", "M": spec.n_atoms,
                             "D": spec.D, "seed": int(seed),
                             "read": "signed linear combination over the basis"}
    elif base == "knn_proto":
        obj = KnnProto(grid, Zp, pool, log=log)
        x = obj.read(R)
        out["byte_ledger"] = obj.ledger
        out["provenance"] = {"rule": "mu_global + sum_terms (mu_kv - mu_global)",
                             "D": spec.D, "seed": int(seed)}
    else:
        raise ValueError(name)

    penult = (np.asarray(x, np.float32)[:, :spec.n_penult]
              * np.float32(scales["p"])).reshape((-1, 128, 4, 4))
    if spec.n_penult != 2048:
        raise AssertionError(f"{name}: penultimate slice is {spec.n_penult}, not 2048")
    imgs = render(inst["decoder"], penult)
    pred = judge(inst["classifier"], imgs)
    acc = per_factor_accuracy(pred, R)
    out["rows"]["full"] = {**acc, "launch_mode": name,
                           "per_corner": per_corner_metrics(pred, R, corner_ids,
                                                            FACTORS, seen)}
    log(f"    {name:14s} {'full':18s} agg {acc['aggregate']:.4f}  exact {acc['exact_match']:.4f}"
        f"  scale {acc['per_factor']['scale']:.4f}  orient {acc['per_factor']['orientation']:.4f}"
        f"  shape {acc['per_factor']['shape']:.4f}")
    out["wall_seconds"] = float(time.time() - t0)
    return out


def build_Z_for(spec, pool, seed, src):
    """Content for a control: scales measured on the control's OWN atom rows, so the
    standardisation convention is identical to the arms'."""
    from content import build_Z
    from chlu.core.gen_landscape import select_atom_rows
    Z0, _ = build_Z(spec, pool, seed=seed, src=src)
    a = select_atom_rows(Z0, spec.n_atoms, "kmeanspp", int(seed))
    _, scales = build_Z(spec, np.asarray(pool)[a], seed=seed, src=src)
    Z, _ = build_Z(spec, pool, seed=seed, src=src, scales=scales)
    return Z, scales


CONTROLS = ("basis_head", "knn_proto")


def is_control(name: str) -> bool:
    return name.partition("@")[0] in CONTROLS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=",".join(LEGAL))
    ap.add_argument("--set", dest="eset", default="val",
                    choices=("val", "pairs", "corners", "row1", "row2", "row5", "c_invalid"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prereg", default="", help="required for a scored set")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    if a.eset in ("corners", "pairs", "row1", "row2", "row5"):
        if not a.prereg or not Path(a.prereg).exists():
            print("REFUSING: a scored set needs an existing PREREG file (--prereg)")
            return 3

    rdir = OUT / "runs" / (a.tag or f"{a.eset}_seed{a.seed}")
    rdir.mkdir(parents=True, exist_ok=True)
    print(f"hub/store-content - content ablation - set={a.eset} seed={a.seed}")
    inst = load_instrument(a.seed)
    world = build_world(a.seed)
    src = open_sources(a.seed)

    for name in [s.strip() for s in a.specs.split(",") if s.strip()]:
        p = rdir / f"{name}.json"
        if p.exists() and not a.force:
            print(f"  [{name}] already done -> {p.name}")
            continue
        if name in KILLED_BY_DECODER:
            print(f"  [{name}] KILLED_BY_DECODER - diagnostic only, skipped")
            continue
        if is_control(name):
            res = run_control(name, world, inst, src, a.eset, a.seed)
        else:
            res = run_spec(name, world, inst, src, a.eset, a.seed)
        json.dump(res, open(p, "w"), indent=1)
        print(f"  [{name}] -> {p.name}  ({res['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
