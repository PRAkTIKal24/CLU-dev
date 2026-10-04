"""Scoreboard rows 5 and 6: the refusal PAIR (reject_invalid, accept_unseen_valid).

w3-common s7: these rows are DEAD AS EVIDENCE -- an 8-byte value-presence bitmap plus a
free bracket check scores the pair at 1.0000 against a 524,288 B budget. They are filled
here for comparability between the four wave-3 hubs and are REPORTED, NEVER CLAIMED.
No number in this file is evidence for the store, and none is quoted as such.

AMENDMENT 1 to the pin: the accept-leg is ROW 1 (a stored, on-grid request), not row 2.
Row 2 is accept-only and is this hub's target.

THE REFUSAL RULE, declared here, request-only, and selected on VAL ONLY (charter A.2):
each arm scores a request with its OWN request-conditioned quantity --

  span arms   TENT-GRAM LEVERAGE  lev(phi) = phi^T (Phi'Phi + lam I)^-1 phi, fitted on
              the store's own fit pool. A request naming a value the store never held
              has a tent slot that never appeared in Phi, so its leverage explodes.
              Request-only: no feature tensor, no image, no label.
  clu arms    STORE-NATIVE EVIDENCE: the address book's summed per-well log-evidence
              logsumexp_m(log P(mode | request)). Low evidence = the landscape has no
              basin the request addresses.
  knn_proto   the same leverage, under its own value grid.

Each arm's threshold is the quantile of its statistic on the VAL split (seen, on-grid,
never used to fit anything) that accepts VAL_ACCEPT of it. ONE threshold per arm, used
identically on rows 1, 5 and 6 -- score_refusal_row checks that identity.

The mandatory no-model laundering control (gen_scoreboard.grid_syntax_outcome) is run
beside every refusal row.
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
BUDGET = 524_288
VAL_ACCEPT = 0.95          # declared: the rule accepts 95% of SEEN val requests
SPAN = ("basis_head", "basis_head_pca", "clu_span", "store_removed_span",
        "store_scrambled_span")
CLU = ("clu_discrete", "store_removed_discrete")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def leverage(Ginv, Phi):
    Phi = np.asarray(Phi, np.float64)
    return np.einsum("ij,jk,ik->i", Phi, Ginv, Phi)


def clu_evidence(book, tent, reqs):
    """Summed per-well log-evidence of a request under the address book."""
    S = A.interp_log_scores(book, tent, reqs)          # (N, K, M)
    m = S.max(axis=-1)
    return -(m + np.log(np.exp(S - m[..., None]).sum(axis=-1))).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rows")
    a = ap.parse_args()
    outdir = hc.OUT / "runs" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)

    grid, _, _ = hc.grid_from_h5()
    man = json.load(open(E.CKPT / "manifest.json", encoding="utf-8"))
    fs = tuple(man["feature_cache"]["feat_shape"])
    store, basis_modes, book, protos, wc, plan = E.load_objects(grid, fs)

    from chlu.core.gen_bridge import features_for_rows, load_feature_cache
    rows_npz = np.load(E.CKPT / "rows.npz")
    fit_rows, val_rows = rows_npz["fit"], rows_npz["val"]
    tent = A.Tent(grid)
    Phi = tent.of_labels(grid.factor_indices_of(fit_rows))
    Ginv = np.linalg.inv(Phi.T @ Phi + 1e-2 * np.eye(Phi.shape[1]))

    from chlu.core.gen_request import requests_from_rows
    val_reqs = requests_from_rows(grid, val_rows)

    def load(name):
        d = json.load(open(hc.VH / f"{name}.json", encoding="utf-8"))
        return [{"id": r["id"], "kind": r["kind"], "request": r["request"]}
                for r in d["records"]]

    r1 = load("row1_seen")
    r5 = load("row5_held_out")
    r6 = load("c_invalid")

    # --- statistics, per arm family ----------------------------------------
    lev = {k: leverage(Ginv, tent.many(v))
           for k, v in (("val", val_reqs), ("r1", r1), ("r5", r5), ("r6", r6))}
    ev = {k: clu_evidence(book, tent, v)
          for k, v in (("val", val_reqs), ("r1", r1), ("r5", r5), ("r6", r6))}
    thr_lev = float(np.quantile(lev["val"], VAL_ACCEPT))
    thr_ev = float(np.quantile(ev["val"], VAL_ACCEPT))
    log(f"thresholds selected on VAL ({val_rows.size} seen rows), accepting "
        f"{VAL_ACCEPT:.0%}: leverage <= {thr_lev:.6g} | clu -log-evidence <= {thr_ev:.6g}")

    from chlu.experiments.gen_scoreboard import (
        ScoreboardConfig, grid_syntax_outcome, make_outcome, score_refusal_row,
    )

    out = {"scope": hc.SCOPE, "n_val": int(val_rows.size), "val_accept": VAL_ACCEPT,
           "thresholds": {"span_leverage": thr_lev, "clu_neg_log_evidence": thr_ev},
           "w3_common_s7": "rows 5 and 6 are DEAD AS EVIDENCE; reported, never claimed",
           "rows": {}}

    for rid, reqs, key in (("r5_comp_refusal", r5, "r5"), ("r6_invalid_refusal", r6, "r6")):
        out["rows"][rid] = {"n_requests": len(reqs), "arms": {}}
        for name in E.ARMS:
            stat_a = lev["r1"] if name in SPAN or name == "knn_proto" else ev["r1"]
            stat_r = lev[key] if name in SPAN or name == "knn_proto" else ev[key]
            thr = thr_lev if (name in SPAN or name == "knn_proto") else thr_ev
            acc_a = stat_a <= thr
            acc_r = stat_r <= thr
            led = {"deployed_bytes": BUDGET if name != "knn_proto" else int(protos.n_bytes),
                   "budget_bytes": BUDGET}
            setting = {"arm": name, "seed": SEED, "rule": "leverage" if thr == thr_lev
                       else "clu_evidence"}
            # accept-leg: accepted AND the arm's row-1 render was judged valid is the
            # strict reading; the pin's score_refusal_row wants accept/valid arrays.
            oc_acc = make_outcome(acc_a, np.zeros(len(r1), bool), None, thr, setting, led,
                                  [r["id"] for r in r1])
            oc_ref = make_outcome(acc_r, np.zeros(len(reqs), bool), None, thr, setting,
                                  led, [r["id"] for r in reqs])
            sc = score_refusal_row(oc_ref, oc_acc, row_id=rid)
            out["rows"][rid]["arms"][name] = sc
            log(f"  {rid} {name:<24} reject {sc.get('reject_invalid'):.4f} "
                f"accept {sc.get('accept_unseen_valid'):.4f} pair {sc.get('score'):.4f}")
        # the MANDATORY no-model laundering control
        lo_r = grid_syntax_outcome(reqs, grid, ScoreboardConfig())
        lo_a = grid_syntax_outcome(r1, grid, ScoreboardConfig())
        sc = score_refusal_row(lo_r, lo_a, row_id=rid)
        out["rows"][rid]["arms"]["grid_syntax_rule"] = sc
        log(f"  {rid} {'grid_syntax_rule (no model)':<24} reject "
            f"{sc.get('reject_invalid'):.4f} accept {sc.get('accept_unseen_valid'):.4f} "
            f"pair {sc.get('score'):.4f}")

    hc.jdump(out, outdir / "refusal_rows.json")
    log(f"wrote {outdir}/refusal_rows.json")


if __name__ == "__main__":
    main()
