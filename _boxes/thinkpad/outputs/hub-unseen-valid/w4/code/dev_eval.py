"""DEV evaluation of READS on the second judge.  ⛔ Selection set only, never scored.

Every read below runs on the SAME fitted object (same atoms, same head, same bytes),
so the only thing that differs is the read verb.  The selection set is the row-2
request block at indices >= ``--select-from``, which the scored block (0 .. 4095)
never touches; the pinned builder's draw order makes the two blocks disjoint and
deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import w4_common as C  # noqa: E402
import valley as V  # noqa: E402
import chlu.experiments.exp_gen_factor_wells as W  # noqa: E402
from j_readout_choice import hue_logits  # noqa: E402


def make_anchors(J, rows, hues, batch):
    """The ORACLE ANCHORS, decoded ONCE and reused by every read."""
    i_lo = np.round(hues[:, 0] / 0.1).astype(int)
    i_hi = np.round(hues[:, 1] / 0.1).astype(int)
    F = W.features_mmap(J.seed)
    n = np.arange(len(i_lo))
    L_lo = hue_logits(J.judge, W.gather(F, rows[:, 0]), J.k_hue, batch)
    L_hi = hue_logits(J.judge, W.gather(F, rows[:, 1]), J.k_hue, batch)
    return {"n": n, "i_lo": i_lo, "i_hi": i_hi,
            "a_lo": L_lo[n, i_hi] - L_lo[n, i_lo],
            "a_hi": L_hi[n, i_hi] - L_hi[n, i_lo]}


def second_judge_e(J, feats, A, batch):
    """e = |p - 0.5| on the second judge, oracle-referenced.  0 perfect, 0.5 lookup."""
    L = hue_logits(J.judge, feats, J.k_hue, batch)
    r = L[A["n"], A["i_hi"]] - L[A["n"], A["i_lo"]]
    p = (r - A["a_lo"]) / (A["a_hi"] - A["a_lo"])
    return np.abs(p - 0.5), p


def per_corner(cids, e):
    return {c: float(e[cids == c].mean()) for c in sorted(set(cids))}


def paired(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    boot = np.array([d[np.random.default_rng(20_260_919 + i).integers(0, d.size, d.size)].mean()
                     for i in range(2000)])
    return {"n_corners": len(ks), "mean_delta": float(d.mean()),
            "sem": float(d.std(ddof=1) / np.sqrt(d.size)),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "corners_better": int((d < 0).sum()), "corners_worse": int((d > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="val")
    ap.add_argument("--arm", default="factor_aligned_balanced")
    ap.add_argument("--n-build", type=int, default=8192)
    ap.add_argument("--select-from", type=int, default=4096)
    ap.add_argument("--n-select", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--rhos", default="0.5,1.0,2.0")
    ap.add_argument("--lams", default="0.0,0.02,0.1,0.5")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tag", default="dev")
    a = ap.parse_args()

    W.PINS["condition"] = a.condition
    out = C.OUT / "runs" / f"dev_eval_{a.tag}_{a.condition}_{a.arm}.json"
    t0 = time.time()

    row2, grid, train_rows = C.build_row2(a.n_build, seed=0)
    recs = row2["records"][a.select_from:a.select_from + a.n_select]
    rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
    cids = C.corner_ids(recs, grid, vals, hues)
    print(f"[dev] selection block [{a.select_from}, {a.select_from+len(recs)}): "
          f"{len(recs)} requests, {len(set(cids))} corners")

    obj = W.build_arm(a.arm, C.OUT / "objects" / a.condition, log=print)
    proto = W.build_arm("knn_proto", C.OUT / "objects" / a.condition, log=print)
    J = C.SecondJudge(a.seed)

    from chlu.core.gen_request_head import head_read, head_coefficients, request_design
    Phi, oor = request_design(vals, grid.values, obj["spec"])
    store, heads = obj["store"], obj["heads"]

    feats = {}
    feats["linear_span"] = head_read(heads, store, Phi)          # the pinned read
    feats["table_interp_knn_proto"] = W.arm_features(proto, vals, grid)[0]
    # the matched-bytes lookup over the SAME atoms (charter C.1/C.4)
    Cc = head_coefficients(heads, Phi)
    look = np.zeros((Phi.shape[0], int(store.dim)), np.float32)
    for c, w, G in zip(Cc, store.well_channels, store.modes):
        look[:, np.asarray(w, np.int64)] = np.asarray(G, np.float32)[np.argmax(c, 1)]
    feats["lookup_argmax"] = look

    z0 = feats["linear_span"]
    for rho in [float(x) for x in a.rhos.split(",")]:
        for lam in [float(x) for x in a.lams.split(",")]:
            t = time.time()
            X, res = V.valley_read(heads, store, Phi, z0, rho, steps=a.steps, lam=lam)
            feats[f"valley_r{rho:g}_l{lam:g}"] = X
            print(f"[dev]   valley rho={rho:g} lam={lam:g} settled in "
                  f"{time.time()-t:.0f}s (residual |dz|={res:.3e})")

    res = {"tag": a.tag, "condition": a.condition, "arm": a.arm, "seed": a.seed,
           "n_select": len(recs), "n_corners": len(set(cids)),
           "select_block": [a.select_from, a.select_from + len(recs)],
           "overlap": V.overlap_report(store, 1.0), "reads": {}}
    A = make_anchors(J, rows, hues, a.batch)
    pc = {}
    for name, X in feats.items():
        e, p = second_judge_e(J, X, A, a.batch)
        pc[name] = per_corner(cids, e)
        res["reads"][name] = {"mean_e": float(e.mean()), "sd_e": float(e.std()),
                              "mean_p": float(p.mean()), "sd_p": float(p.std()),
                              "frac_interior_20_80": float(((p > .2) & (p < .8)).mean())}
        print(f"[dev] {name:28s} e={e.mean():.4f}  p={p.mean():+.3f}+/-{p.std():.3f}  "
              f"interior={((p>.2)&(p<.8)).mean():.3f}")
    res["paired_vs_table_interp"] = {
        k: paired(pc[k], pc["table_interp_knn_proto"]) for k in pc if k != "table_interp_knn_proto"}
    res["paired_vs_linear_span"] = {
        k: paired(pc[k], pc["linear_span"]) for k in pc if k != "linear_span"}
    res["seconds"] = time.time() - t0
    C.jdump(res, out)
    print(f"[dev] wrote {out} in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
