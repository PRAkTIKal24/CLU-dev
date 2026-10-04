"""Fill scoreboard rows 1, 3, 4, 5, 6, 7 for every arm and both controls (w3-common s2).

Row 2 is this hub's target and is scored by eval_row2/aggregate_row2. These six are
filled so the four wave-3 hubs are comparable. ROWS OUTSIDE ROW 2 ARE REPORTED, NEVER
CLAIMED.

Scoring uses the pin's own chlu.experiments.gen_scoreboard functions, unmodified.

Refusal rule, DECLARED here and identical for every arm and every refusal row (the
scoreboard checks the threshold identity across rows): an arm ACCEPTS a request iff
every named factor value lies inside its factor's realised range and no factor is named
twice AND the arm's own request-conditioned confidence clears one threshold. The
confidence is the arm's own quantity:
  clu_*        the address book's mean per-well max posterior (request-only)
  span arms    the tent-basis leverage of the request under the fitted head's Gram
  knn_proto    the same leverage under the prototype table's own value grid
w3-common s7: rows 5 and 6 are DEAD as evidence and no number here is quoted as
evidence for the store. They are filled for comparability and carry the mandatory
no-model laundering control (gen_scoreboard.grid_syntax_outcome).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import arms as A  # noqa: E402
import hub_common as hc  # noqa: E402
import eval_row2 as E  # noqa: E402

SEED = 0
BUDGET = 524_288


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def requests_from_rows(grid, rows):
    from chlu.core.gen_request import requests_from_rows as rfr
    return rfr(grid, np.asarray(rows, np.int64))


def load_cells(family):
    vc = json.load(open(hc.VH / "value_cells.json", encoding="utf-8"))
    return [c for c in vc["cells"] if c["meta"].get("family") == family]


def cell_rows(grid, cell, test_mask):
    """Rows realised by a cell's value constraints, restricted to the held-out side."""
    vals = cell["values"]
    ok = np.ones(grid.n_rows, bool)
    for name, allowed in vals.items():
        k = grid.index_of(name)
        idx = [int(np.argmin(np.abs(np.asarray(grid.values[k]) - float(v))))
               for v in allowed]
        fi = np.unravel_index(np.arange(grid.n_rows), grid.cardinalities)[k]
        ok &= np.isin(fi, idx)
    return np.where(ok & test_mask)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rows")
    ap.add_argument("--rows", default="1,3,4")
    ap.add_argument("--n-pair", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    outdir = hc.OUT / "runs" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    want = set(a.rows.split(","))

    grid, _, _ = hc.grid_from_h5()
    man = json.load(open(E.CKPT / "manifest.json", encoding="utf-8"))
    fs = tuple(man["feature_cache"]["feat_shape"])
    dim = int(np.prod(fs))
    store, basis_modes, book, protos, wc, plan = E.load_objects(grid, fs)

    from chlu.core.gen_bridge import features_for_rows, load_feature_cache
    rows_npz = np.load(E.CKPT / "rows.npz")
    fit_rows = rows_npz["fit"]
    croot = hc.BUNDLE / "gen-feature-bridge" / "runs" / "shapes3d_recon" / f"seed_{SEED}"
    crows, cf, _ = load_feature_cache(croot, verify=False)
    Xfit = np.asarray(features_for_rows(crows, cf, fit_rows), np.float32).reshape(
        (fit_rows.size, -1))
    tent = A.Tent(grid)
    Phi = tent.of_labels(grid.factor_indices_of(fit_rows))

    from chlu.core.gen_store import delete_store
    del_modes = [np.asarray(m, np.float64) for m in
                 delete_store(store, "mean", seed=SEED).modes]
    scr_modes = [np.asarray(m, np.float64) for m in
                 delete_store(store, "scramble", seed=SEED).modes]
    pca_modes = []
    for w in wc:
        Y = np.asarray(Xfit[:, np.asarray(w, np.int64)], np.float64)
        mu = Y.mean(0)
        C = ((Y - mu).T @ (Y - mu)) / max(1, Y.shape[0] - 1)
        ev, V = np.linalg.eigh(C)
        B = V[:, ::-1][:, :64].T.copy()
        B[-1] = mu / (np.linalg.norm(mu) + 1e-12)
        pca_modes.append(np.ascontiguousarray(B))
    heads = {}
    for nm, atoms in (("basis_head", basis_modes), ("basis_head_pca", pca_modes),
                      ("clu_span", [np.asarray(m, np.float64) for m in store.modes]),
                      ("store_removed_span", del_modes),
                      ("store_scrambled_span", scr_modes)):
        heads[nm] = A.SpanHead(atoms, wc, dim, fs).fit(Phi, Xfit, 1e-2, 1e-6)
    del Xfit
    log("heads fitted")

    decoder, dinfo = hc.load_decoder(SEED)
    judge, jinfo = hc.load_judge(SEED)
    ctx = dict(tent=tent, heads=heads, store=store, protos=protos, book=book,
               wc=wc, feat_shape=fs, dim=dim)

    # the store's fit pool as factor tuples -- the SEEN set for the novelty rows
    seen_tuples = grid.factor_indices_of(fit_rows)
    test_mask = np.zeros(grid.n_rows, bool)
    test_mask[np.asarray(rows_npz["test_rows"], np.int64)] = True

    # ---- request sets ------------------------------------------------------
    sets = {}
    if "1" in want:
        d = json.load(open(hc.VH / "row1_seen.json", encoding="utf-8"))
        sets["r1_seen"] = ([{"id": r["id"], "kind": r["kind"], "request": r["request"]}
                            for r in d["records"]],
                           np.array([int(r["row"]) for r in d["records"]]), "accuracy")
    rng = np.random.default_rng(SEED)
    if "3" in want:
        rr = []
        for c in load_cells("pair"):
            rs = cell_rows(grid, c, test_mask)
            rr.append(rs[rng.permutation(rs.size)[:a.n_pair]])
        rr = np.sort(np.concatenate(rr))
        sets["r3_comp_2way"] = (requests_from_rows(grid, rr), rr, "novelty")
    if "4" in want:
        rr = []
        for c in load_cells("nway"):
            rr.append(cell_rows(grid, c, test_mask))
        rr = np.sort(np.concatenate(rr))
        sets["r4_comp_nway"] = (requests_from_rows(grid, rr), rr, "novelty")

    from chlu.experiments.gen_scoreboard import (
        make_outcome, score_accuracy_row, score_novelty_row,
    )

    ledger = {
        "basis_head": BUDGET, "basis_head_pca": BUDGET, "clu_span": BUDGET,
        "clu_discrete": BUDGET, "store_removed_span": BUDGET,
        "store_removed_discrete": BUDGET, "store_scrambled_span": BUDGET,
        "knn_proto": int(protos.n_bytes),
    }
    notes = {"knn_proto": ("DECLARED, NOT SILENT: the prototype table is "
                          f"{protos.n_bytes} B against the {BUDGET} B budget")}

    out = {"scope": hc.SCOPE, "judge": jinfo, "rows": {},
           "note": "ROW 2 is this hub's target and is in runs/dev4096/row2_table.json. "
                   "Every row here is REPORTED, NEVER CLAIMED (w3-common s2)."}
    for rid, (reqs, rws, metric) in sets.items():
        log(f"--- {rid}: {len(reqs)} requests, metric={metric}")
        tgt = grid.factor_indices_of(rws)
        out["rows"][rid] = {"n_requests": len(reqs), "metric": metric, "arms": {}}
        for name in E.ARMS:
            t0 = time.time()
            F = E.arm_features(name, reqs, ctx)
            preds, _ = E.judge_batch(decoder, judge, np.asarray(F, np.float32), a.batch)
            valid = (preds == tgt).all(axis=1)
            accept = np.ones(len(reqs), bool)   # rows 1/3/4 are accept-only
            oc = make_outcome(accept, valid, preds, threshold=0.0,
                              setting={"arm": name, "seed": SEED},
                              byte_ledger={"deployed_bytes": ledger[name],
                                           "budget_bytes": BUDGET,
                                           "budget_note": notes.get(name, "")},
                              request_ids=[r["id"] for r in reqs])
            sc = (score_accuracy_row(oc) if metric == "accuracy"
                  else score_novelty_row(oc, seen_tuples))
            sc["per_factor"] = {grid.names[k]: float((preds[:, k] == tgt[:, k]).mean())
                                for k in range(6)}
            sc["distinct_judged_tuples"] = int(np.unique(preds, axis=0).shape[0])
            sc["deployed_bytes"] = ledger[name]
            sc["seconds"] = time.time() - t0
            out["rows"][rid]["arms"][name] = sc
            key = ("judge_accuracy" if metric == "accuracy"
                   else "distinct_valid_novel_tuples")
            log(f"  {name:<24} {key} {sc[key]}  valid {sc['n_valid']}/{sc['n_requests']}"
                f"  distinct {sc['distinct_judged_tuples']}  ({sc['seconds']:.1f}s)")

    hc.jdump(out, outdir / "scoreboard_rows.json")
    log(f"wrote {outdir}/scoreboard_rows.json")


if __name__ == "__main__":
    main()
