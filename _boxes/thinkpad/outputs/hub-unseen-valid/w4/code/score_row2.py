"""THE SCORED ROW-2 RUN.  ⛔ Runs only from a PREREG file, only on cache generation 2.

Every arm, rival and control is read off the SAME fitted object where it can be, on the
SAME requests, in the SAME corners, judged by the SAME second judge, on three seeds.
The verdict is the multi-corner PAIRED estimand against the declared margin, and never
a single cell.
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
from dev_eval import make_anchors, second_judge_e, per_corner  # noqa: E402


def paired_estimand(arm_pc: dict, ctl_pc: dict, margin: float, arm: str, ctl: str) -> dict:
    """Lower e is better, so the arm wins when ctl - arm > margin."""
    ks = sorted(set(arm_pc) & set(ctl_pc))
    d = np.array([ctl_pc[k] - arm_pc[k] for k in ks])
    rng = np.random.default_rng(20_260_919)
    boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(10_000)])
    nz = d[d != 0]
    if nz.size:
        from math import comb
        k = int((nz > 0).sum()); m = nz.size
        p = float(min(1.0, 2.0 * sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m))
    else:
        p = 1.0
    lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
    return {"arm": arm, "control": ctl, "n_corners": len(ks),
            "arm_mean_e": float(np.mean([arm_pc[k] for k in ks])),
            "control_mean_e": float(np.mean([ctl_pc[k] for k in ks])),
            "mean_paired_delta": float(d.mean()), "median_paired_delta": float(np.median(d)),
            "bootstrap_ci95": [lo, hi], "sign_test_p_two_sided": p,
            "corners_arm_wins": int((d > 0).sum()), "corners_tied": int((d == 0).sum()),
            "corners_control_wins": int((d < 0).sum()),
            "margin_declared_in_advance": float(margin),
            "beats_beyond_margin": bool(d.mean() > margin and lo > margin)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--arm", default="factor_aligned_balanced")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    a = ap.parse_args()

    assert Path(a.prereg).exists(), f"PREREG {a.prereg} does not exist -- refusing to score"
    W.PINS["condition"] = "test"
    prov = C.use_cache_generation2(tuple(int(s) for s in a.seeds.split(",")))
    print(f"[score] cache generation 2 asserted: {json.dumps(prov['encoder_proof'])}")

    from chlu.core.gen_request_head import head_coefficients, head_read, request_design
    t0 = time.time()
    per_seed = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = C.OUT / "runs" / f"score_{a.tag}_s{seed}.json"
        if out.exists():
            per_seed[seed] = json.load(open(out, encoding="utf-8"))
            print(f"[score] seed {seed}: reusing {out.name} (resumable)")
            continue
        if time.time() - t0 > a.max_seconds:
            print(f"[score] wall-clock cap hit before seed {seed}; NOT RUN")
            break
        W.PINS["seed"] = seed
        row2, grid, train_rows = C.build_row2(a.n, seed=0)
        recs = row2["records"][:a.n]
        fr = json.load(open(W.data_root() / "value_holdout" / "row2_unseen.json",
                            encoding="utf-8"))["records"]
        assert all(json.dumps(recs[i], sort_keys=True) == json.dumps(fr[i], sort_keys=True)
                   for i in range(len(fr))), "row 2 is not a superset of the frozen file"
        rows, hues, vals = C.bracket_rows(recs, grid, train_rows)
        cids = C.corner_ids(recs, grid, vals, hues)
        assert len(set(cids)) >= 49, f"only {len(set(cids))} corners"

        odir = C.OUT / "objects" / f"test_s{seed}"
        obj = W.build_arm(a.arm, odir, log=print)
        proto = W.build_arm("knn_proto", odir, log=print)
        rem = W.build_arm("landscape_removed", odir, log=print)
        shf = W.build_arm("landscape_shuffled", odir, log=print)
        base = W.build_arm("basis_head", odir, log=print)

        Phi, _ = request_design(vals, grid.values, obj["spec"])
        store, heads = obj["store"], obj["heads"]
        z0 = head_read(heads, store, Phi)
        feats = {"linear_span_ARM": z0,
                 "table_interp_knn_proto_RIVAL": W.arm_features(proto, vals, grid)[0],
                 "correlation_split_linear_head_ABLATION": W.arm_features(base, vals, grid)[0],
                 "landscape_removed_CONTROL": W.arm_features(rem, vals, grid)[0],
                 "landscape_shuffled_CONTROL": W.arm_features(shf, vals, grid)[0]}
        Cc = head_coefficients(heads, Phi)
        look = np.zeros((Phi.shape[0], int(store.dim)), np.float32)
        for c, w, G in zip(Cc, store.well_channels, store.modes):
            look[:, np.asarray(w, np.int64)] = np.asarray(G, np.float32)[np.argmax(c, 1)]
        feats["lookup_argmax_FLOOR"] = look

        # the continuous-input MLP rival, at BOTH declared parameter bases
        import cont_mlp as M
        pool = np.asarray(np.load(odir / "partitions.npz")["pool"], np.int64)
        Fm = W.features_mmap(seed)
        arm_params = (int(obj["byte_ledger"]["deployed_bytes"])
                      + int(obj["head_bytes"]["head_bytes"])) // 4
        mlp_info = {}
        for label, target in (("cont_mlp_ARMMATCHED_RIVAL", arm_params),
                              ("cont_mlp_TABLEMATCHED_RIVAL", 118_784)):
            mdl, inf = M.fit(grid, pool, Fm, target, seed=seed, steps=4000)
            feats[label] = M.predict(mdl, vals, grid)
            mlp_info[label] = inf

        # SPOKE A's OWN fitted rivals, exported through their API on these same requests
        spa = C.OUT / "runs" / f"spokea_rivals_s{seed}_n{a.n}.npz"
        spokea = {}
        if spa.exists():
            z = np.load(spa, allow_pickle=True)
            assert np.allclose(z["values"], vals), (
                "spoke A's rival features were exported for DIFFERENT requests")
            for nm in ("table", "cont_mlp"):
                if f"feats__{nm}" in z:
                    feats[f"spokeA_{nm}_RIVAL"] = np.asarray(z[f"feats__{nm}"], np.float32)
                    spokea[nm] = {"n_oor": int(z[f"oor__{nm}"].sum())}
            for nm in ("mlp", "transformer", "lstm"):
                if f"refused__{nm}" in z:
                    spokea[nm] = {"DASH": str(z[f"refused__{nm}"])}
        else:
            spokea = {"MISSING": "spoke A rival export absent -- rows get a dash"}
        zz, resid = V.valley_read(heads, store, Phi, z0, a.rho, lam=a.lam)
        feats["valley_ARM"] = zz

        J = C.SecondJudge(seed)
        A = make_anchors(J, rows, hues, a.batch)
        pc, summ = {}, {}
        for name, X in feats.items():
            e, p = second_judge_e(J, X, A, a.batch)
            pc[name] = per_corner(cids, e)
            summ[name] = {"mean_e": float(e.mean()), "sd_e": float(e.std()),
                          "mean_p": float(p.mean())}
            print(f"[score] seed {seed} {name:40s} e={e.mean():.4f}")
        res = {"seed": seed, "n_requests": len(recs), "n_corners": len(set(cids)),
               "cache": prov, "rho": a.rho, "lam": a.lam, "settle_residual": resid,
               "reads": summ, "per_corner": pc,
               "diagnostics_C4": V.settle_diagnostics(store, Phi, heads, z0, zz, None, None),
               "mlp_info": mlp_info, "spokeA_rivals": spokea,
               "byte_ledger": obj["byte_ledger"], "head_bytes": obj["head_bytes"],
               "request_only": obj["request_only"]}
        C.jdump(res, out)
        per_seed[seed] = res

    if per_seed:
        agg = {"tag": a.tag, "prereg": a.prereg, "seeds": sorted(per_seed),
               "margin": a.margin, "estimands": {}}
        ctls = ["spokeA_cont_mlp_RIVAL", "spokeA_table_RIVAL",
                "cont_mlp_ARMMATCHED_RIVAL", "cont_mlp_TABLEMATCHED_RIVAL",
                "table_interp_knn_proto_RIVAL", "lookup_argmax_FLOOR",
                "correlation_split_linear_head_ABLATION", "valley_ARM",
                "landscape_removed_CONTROL", "landscape_shuffled_CONTROL"]
        one = next(iter(per_seed.values()))["per_corner"]
        for ctl in [c for c in ctls if c in one]:
            per = [paired_estimand(r["per_corner"]["linear_span_ARM"], r["per_corner"][ctl],
                                   a.margin, "linear_span_ARM", ctl) for r in per_seed.values()]
            agg["estimands"][ctl] = {
                "per_seed": per,
                "mean_delta_across_seeds": float(np.mean([x["mean_paired_delta"] for x in per])),
                "sd_delta_across_seeds": float(np.std([x["mean_paired_delta"] for x in per])),
                "all_seeds_beat_margin": bool(all(x["beats_beyond_margin"] for x in per))}
        C.jdump(agg, C.OUT / "runs" / f"score_{a.tag}_AGG.json")
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                          for k, v in agg["estimands"].items()}, indent=1))


if __name__ == "__main__":
    main()
