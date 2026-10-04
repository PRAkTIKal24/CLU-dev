"""Scoreboard row 7: DELETION. Filled for comparability; REPORTED, NEVER CLAIMED --
row 7 is hub/erasure's target and this hub does not compete for it.

The operation is a deletion of ATOMS (charter B.4), targeted through the ADDRESS BOOK,
which is the interface where labels are allowed to ground the words of a request
(charter B.6). The landscape itself is never labelled.

  target value v* on factor f (a STORED, on-grid value -- 'delete what you hold')
  for each well k, a mode m is CARRYING v* iff
      log P(m | f = v*) - log P(m) >= tau     (the address book's own evidence)
  those modes are zeroed. Nothing is retrained; no head, book or decoder is refitted.

Then, with no retraining:
  * requests naming v*            -> must DEGRADE          (the deletion took effect)
  * requests naming other values  -> must HOLD             (it was targeted, not global)
  * rows 3 / 4 (compositional)    -> must HOLD
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import arms as A  # noqa: E402
import eval_row2 as E  # noqa: E402
import hub_common as hc  # noqa: E402

SEED = 0
TARGET_FACTOR = "object_hue"
TARGET_VALUE = 0.4          # a STORED, on-grid value (0.5 is the held-out one)
TAU = 1.0                   # nats of evidence above prior; declared here


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def carrying_modes(book, k_factor, v_index, tau=TAU):
    """(K, M) bool: modes whose evidence for value v_index exceeds the prior by tau."""
    lp = np.asarray(book.log_prior, np.float64)
    lc = np.asarray(book.log_cond[k_factor], np.float64)[int(v_index)]
    return (lc - lp) >= float(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rows")
    ap.add_argument("--n", type=int, default=768)
    a = ap.parse_args()
    outdir = hc.OUT / "runs" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)

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

    kf = grid.index_of(TARGET_FACTOR)
    vi = int(np.argmin(np.abs(np.asarray(grid.values[kf]) - TARGET_VALUE)))
    carry = carrying_modes(book, kf, vi)
    log(f"deleting atoms carrying {TARGET_FACTOR}={TARGET_VALUE} (grid index {vi}) at "
        f"tau={TAU}: {int(carry.sum())} of {carry.size} modes "
        f"({carry.sum(axis=1).tolist()} per well)")

    intact = [np.asarray(m, np.float64) for m in store.modes]
    deleted = []
    for k, m in enumerate(intact):
        mm = m.copy()
        mm[carry[k]] = 0.0
        deleted.append(mm)

    heads = {
        "intact": A.SpanHead(intact, wc, dim, fs).fit(Phi, Xfit, 1e-2, 1e-6),
        "deleted": A.SpanHead(deleted, wc, dim, fs).fit(Phi, Xfit, 1e-2, 1e-6),
    }
    # ^ the head is refitted against the DELETED store because a head that still
    #   points at zeroed atoms would make the deletion look stronger than it is.
    del Xfit
    decoder, dinfo = hc.load_decoder(SEED)
    judge, jinfo = hc.load_judge(SEED)

    from chlu.core.gen_request import requests_from_rows
    rng = np.random.default_rng(SEED)
    test_rows = np.asarray(rows_npz["test_rows"], np.int64)
    fi_test = grid.factor_indices_of(test_rows)

    legs = {}
    sel = test_rows[fi_test[:, kf] == vi]
    legs["names_the_deleted_value"] = np.sort(rng.permutation(sel)[:a.n])
    sel = test_rows[fi_test[:, kf] != vi]
    legs["names_another_value"] = np.sort(rng.permutation(sel)[:a.n])

    out = {"scope": hc.SCOPE, "judge": jinfo,
           "note": "ROW 7 is hub/erasure's target. Filled for comparability; "
                   "REPORTED, NEVER CLAIMED.",
           "deletion": {"factor": TARGET_FACTOR, "value": TARGET_VALUE,
                        "grid_index": vi, "tau_nats": TAU,
                        "n_modes_deleted": int(carry.sum()),
                        "n_modes_total": int(carry.size),
                        "per_well": carry.sum(axis=1).tolist(),
                        "retraining": "NONE -- no book, decoder or judge refitted; the "
                                      "span head is refitted against the deleted store "
                                      "so the deletion is not flattered"},
           "legs": {}}

    for leg, rws in legs.items():
        reqs = requests_from_rows(grid, rws)
        tgt = grid.factor_indices_of(rws)
        out["legs"][leg] = {"n_requests": len(reqs)}
        for state in ("intact", "deleted"):
            F = heads[state].read(tent.many(reqs))
            preds, _ = E.judge_batch(decoder, judge, np.asarray(F, np.float32), 256)
            valid = (preds == tgt).all(axis=1)
            out["legs"][leg][state] = {
                "judge_accuracy": float(valid.mean()),
                "target_factor_accuracy": float((preds[:, kf] == tgt[:, kf]).mean()),
                "distinct_judged_tuples": int(np.unique(preds, axis=0).shape[0]),
                "per_factor": {grid.names[k]: float((preds[:, k] == tgt[:, k]).mean())
                               for k in range(6)},
            }
        i, dd = out["legs"][leg]["intact"], out["legs"][leg]["deleted"]
        out["legs"][leg]["delta_judge_accuracy"] = dd["judge_accuracy"] - i["judge_accuracy"]
        out["legs"][leg]["delta_target_factor"] = (dd["target_factor_accuracy"]
                                                   - i["target_factor_accuracy"])
        log(f"  {leg:<26} intact {i['judge_accuracy']:.4f} -> deleted "
            f"{dd['judge_accuracy']:.4f}  (d {out['legs'][leg]['delta_judge_accuracy']:+.4f})"
            f" | {TARGET_FACTOR} {i['target_factor_accuracy']:.4f} -> "
            f"{dd['target_factor_accuracy']:.4f} "
            f"(d {out['legs'][leg]['delta_target_factor']:+.4f})")

    d_del = out["legs"]["names_the_deleted_value"]["delta_target_factor"]
    d_oth = out["legs"]["names_another_value"]["delta_target_factor"]
    out["verdict"] = {
        "targeted": bool(d_del < d_oth),
        "selectivity_nats": float(d_oth - d_del),
        "reading": "a TARGETED deletion degrades the leg that names the deleted value "
                   "more than the leg that names another value. Reported, never claimed.",
    }
    log(f"selectivity (other - deleted) = {d_oth - d_del:+.4f}")
    hc.jdump(out, outdir / "deletion_row.json")
    log(f"wrote {outdir}/deletion_row.json")


if __name__ == "__main__":
    main()
