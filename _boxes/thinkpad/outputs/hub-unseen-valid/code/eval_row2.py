"""Score scoreboard ROW 2 for every arm and both controls, on the 63-corner paired
estimand registered in PREREG-unseen-valid-0 s3/s4.

Resumable: each arm's judged outputs are checkpointed to runs/<tag>/<arm>.npz and an
arm already on disk is skipped. Memory is capped by decoding/judging in batches.
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

CKPT = hc.OUT / "objects"
DELTA = hc.DELTA
SEED = 0
# PREREG-0 AMENDMENT 4: scale QUARTILES -- terciles realise only 48 corners (< 49)
SCALE_TERCILE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
ARMS = ("basis_head", "basis_head_pca", "knn_proto", "clu_span", "clu_discrete",
        "store_removed_span", "store_removed_discrete", "store_scrambled_span")
SPAN_ARMS = ("basis_head", "basis_head_pca", "clu_span", "store_removed_span",
             "store_scrambled_span")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_requests(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = d["records"]
    import re
    meta = []
    for r in recs:
        t = {x["factor"]: float(x["value"]) for x in r["request"]}
        m = re.search(r"stored values ([\d.]+) and ([\d.]+)", r["reason"])
        lo, hi = float(m.group(1)), float(m.group(2))
        meta.append({"id": r["id"], "h_req": t["object_hue"], "h_lo": lo, "h_hi": hi,
                     "shape": t["shape"], "scale": t["scale"]})
    return recs, meta, d


def corner_keys(meta, grid):
    sv = np.asarray(grid.values[3], np.float64)
    out = []
    for m in meta:
        si = int(np.argmin(np.abs(sv - m["scale"])))
        out.append(f"br{m['h_lo']:.1f}-{m['h_hi']:.1f}|sh{int(m['shape'])}|st{SCALE_TERCILE[si]}")
    return np.array(out)


def load_objects(grid, feat_shape):
    import jax
    from chlu.core.gen_dream import AddressBook
    from chlu.core.gen_store import GenStore, well_channels_from_labels

    plan = json.load(open(hc.BUNDLE / "gen-store-grouping" / "runs" / "shapes3d_recon"
                          / "store_plan.json", encoding="utf-8"))
    wc = well_channels_from_labels(np.asarray(plan["labels"], np.int64), int(plan["K"]))
    key = jax.random.PRNGKey(SEED)
    st = pickle.load(open(CKPT / "store.pkl", "rb"))
    ba = pickle.load(open(CKPT / "basis.pkl", "rb"))
    bk = pickle.load(open(CKPT / "book.pkl", "rb"))
    pr = pickle.load(open(CKPT / "protos.pkl", "rb"))
    store = GenStore(wc, 64, feat_shape, key, init=st["modes"])
    book = AddressBook(bk["log_prior"], bk["log_cond"])
    from chlu.core.gen_request import FactorPrototypes
    protos = FactorPrototypes(
        names=tuple(grid.names), values=tuple(grid.values),
        protos=tuple(pr["protos"]), mu_global=pr["mu_global"],
        n_rows_fit=int(pr["n_rows_fit"]), n_per_value=tuple(pr["n_per_value"]))
    return store, ba["modes"], book, protos, wc, plan


def arm_features(name, reqs, ctx):
    """request list -> (N,) + feat_shape features. No feature tensor enters here."""
    tent, heads, store, protos, book, wc, fs, dim = (
        ctx["tent"], ctx["heads"], ctx["store"], ctx["protos"], ctx["book"],
        ctx["wc"], ctx["feat_shape"], ctx["dim"])
    if name == "knn_proto":
        return protos.embed_many(reqs).reshape((len(reqs),) + fs)
    if name in SPAN_ARMS:
        return heads[name].read(tent.many(reqs))
    if name in ("clu_discrete", "store_removed_discrete"):
        P = A.posterior_from_scores(A.interp_log_scores(book, tent, reqs))
        codes = A.codes_from_posterior(P)
        from chlu.core.gen_store import delete_store, read_from_codes
        s = store if name == "clu_discrete" else delete_store(store, "mean", seed=SEED)
        return np.asarray(read_from_codes(s, codes), np.float32)
    raise ValueError(name)


def judge_batch(decoder, judge, feats, batch=256):
    """features -> THE frozen decoder -> images -> THE external classifier."""
    from chlu.eval.gen_probe import decode_features, predict_factors
    import equinox as eqx, jax, jax.numpy as jnp
    from chlu.core.gen_bridge import model_dtype
    dt = model_dtype(judge)
    fwd = eqx.filter_jit(jax.vmap(judge))
    preds, hp = [], []
    for i in range(0, feats.shape[0], batch):
        img = decode_features(decoder, feats[i:i + batch], batch)
        lg = fwd(jnp.asarray(img, dt))
        preds.append(np.stack([np.asarray(jnp.argmax(l, -1)) for l in lg], 1))
        hp.append(np.asarray(jax.nn.softmax(lg[hc.HUE_F], -1), np.float64))
    return np.concatenate(preds).astype(np.int32), np.concatenate(hp)


def score(meta, preds, hp, hue_vals, grid, reqs, oracle=None):
    h_hat = hc.circ_mean_hue(hp, hue_vals)
    h_lin = hc.lin_mean_hue(hp, hue_vals)
    h_req = np.array([m["h_req"] for m in meta])
    h_lo = np.array([m["h_lo"] for m in meta])
    h_hi = np.array([m["h_hi"] for m in meta])
    err = hc.cdist(h_hat, h_req)
    sn = hc.snap_metrics(h_hat, h_lo, h_hi)
    # the other five factors, top-1 by the same judge
    other = [k for k in range(6) if k != hc.HUE_F]
    tgt = np.zeros((len(reqs), 6), np.int64)
    for n, r in enumerate(reqs):
        for t in r["request"]:
            k = int(t["factor_index"])
            if k == hc.HUE_F:
                continue
            tgt[n, k] = int(np.argmin(np.abs(np.asarray(grid.values[k]) - float(t["value"]))))
    ok5 = np.ones(len(reqs), bool)
    per_factor = {}
    for k in other:
        c = preds[:, k] == tgt[:, k]
        per_factor[grid.names[k]] = float(c.mean())
        ok5 &= c
    out = {
        "h_hat": h_hat, "h_lin": h_lin, "err": err, "t": sn["t"],
        "d_req": sn["d_req"], "d_near": sn["d_near"], "snap": sn["snap"],
        "ok5": ok5, "hit": err <= 0.5 * DELTA,
        "row2acc": ok5 & (err <= 0.5 * DELTA),
        "per_factor_other5": per_factor, "preds": preds,
    }
    if oracle is not None:
        # the INSTRUMENT-ANCHORED metric (j5): distance to the ORACLE's readout on the
        # SAME request. 0 at a perfect render, 0.5 D at a perfect lookup -- 2.67x the
        # dynamic range of the registered metric, whose floor is 0.313 D.
        out["err_ref"] = hc.cdist(h_hat, oracle["h_oracle"])
        out["err_ref_lookup"] = 0.5 * (
            hc.cdist(oracle["h_lookup_lo"], oracle["h_oracle"])
            + hc.cdist(oracle["h_lookup_hi"], oracle["h_oracle"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default=str(hc.VH / "row2_unseen.json"))
    ap.add_argument("--tag", default="dev512")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--oracle", default=str(hc.OUT / "runs" / "oracle_n4096.npz"))
    a = ap.parse_args()
    t_start = time.time()

    outdir = hc.OUT / "runs" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)

    grid, _, _ = hc.grid_from_h5()
    hue_vals = np.asarray(grid.values[hc.HUE_F], np.float64)
    reqs, meta, payload = load_requests(a.requests)
    oracle = None
    if Path(a.oracle).exists():
        z = np.load(a.oracle)
        if z["h_oracle"].size == len(reqs):
            oracle = {k: z[k] for k in ("h_oracle", "h_lookup_lo", "h_lookup_hi")}
            log(f"oracle readouts loaded from {a.oracle} -- both metrics will be scored")
        else:
            log(f"oracle cache has {z['h_oracle'].size} rows for {len(reqs)} requests; "
                "the instrument-anchored metric is SKIPPED")
    ck = corner_keys(meta, grid)
    log(f"{len(reqs)} row-2 requests; {len(set(ck))} corners realised")

    man = json.load(open(CKPT / "manifest.json", encoding="utf-8"))
    fs = tuple(man["feature_cache"]["feat_shape"])
    dim = int(np.prod(fs))
    store, basis_modes, book, protos, wc, plan = load_objects(grid, fs)

    # heads: fitted ONCE on the train features, offline. Read time sees phi(request).
    from chlu.core.gen_bridge import features_for_rows, load_feature_cache
    rows = np.load(CKPT / "rows.npz")
    croot = hc.BUNDLE / "gen-feature-bridge" / "runs" / "shapes3d_recon" / f"seed_{SEED}"
    crows, cf, _ = load_feature_cache(croot, verify=False)
    Xfit = np.asarray(features_for_rows(crows, cf, rows["fit"]), np.float32).reshape(
        (rows["fit"].size, -1))
    tent = A.Tent(grid)
    Phi = tent.of_labels(grid.factor_indices_of(rows["fit"]))
    from chlu.core.gen_store import delete_store
    del_modes = [np.asarray(m, np.float64) for m in delete_store(store, "mean", seed=SEED).modes]
    scr_modes = [np.asarray(m, np.float64)
                 for m in delete_store(store, "scramble", seed=SEED).modes]
    # the PCA basis: the OPTIMAL rank-64 subspace per well for a SIGNED linear span,
    # and therefore the sharpest fitted-basis control available at these bytes.
    pca_modes = []
    for w in wc:
        Y = np.asarray(Xfit[:, np.asarray(w, np.int64)], np.float64)
        mu = Y.mean(0)
        C = ((Y - mu).T @ (Y - mu)) / max(1, Y.shape[0] - 1)
        ev, V = np.linalg.eigh(C)
        B = V[:, ::-1][:, :64].T.copy()          # (64, d_k), orthonormal
        B[-1] = mu / (np.linalg.norm(mu) + 1e-12)  # keep the mean direction in the span
        pca_modes.append(np.ascontiguousarray(B))
    heads = {}
    for nm, atoms in (("basis_head", basis_modes),
                      ("basis_head_pca", pca_modes),
                      ("clu_span", [np.asarray(m, np.float64) for m in store.modes]),
                      ("store_removed_span", del_modes),
                      ("store_scrambled_span", scr_modes)):
        heads[nm] = A.SpanHead(atoms, wc, dim, fs).fit(Phi, Xfit, 1e-2, 1e-6)
    log("heads fitted (two-sided ridge, phi=1e-2, atom=1e-6); "
        f"basis variants: kmeans++ and PCA, both 64 x d_k = "
        f"{sum(len(w) for w in wc)*64*4} B")
    coeff_rep = {nm: heads[nm].coeff_report(tent.many(reqs))
                 for nm in ("basis_head", "basis_head_pca", "clu_span")}
    del Xfit

    decoder, dinfo = hc.load_decoder(SEED)
    judge, jinfo = hc.load_judge(SEED)
    ctx = dict(tent=tent, heads=heads, store=store, protos=protos, book=book,
               wc=wc, feat_shape=fs, dim=dim)

    results = {}
    for name in a.arms.split(","):
        f = outdir / f"{name}.npz"
        if f.exists():
            log(f"{name}: on disk, skipped")
            results[name] = dict(np.load(f, allow_pickle=True))
            continue
        if time.time() - t_start > a.max_seconds:
            log(f"{name}: WALL CLOCK CAP reached, stopping (resumable)")
            break
        t0 = time.time()
        F = arm_features(name, reqs, ctx)
        preds, hp = judge_batch(decoder, judge, np.asarray(F, np.float32), a.batch)
        s = score(meta, preds, hp, hue_vals, grid, reqs, oracle)
        # distinct-output counts: never an accuracy without them
        s["n_distinct_feature_vectors"] = int(np.unique(
            np.asarray(F, np.float32).reshape(len(reqs), -1), axis=0).shape[0])
        s["n_distinct_judged_tuples"] = int(np.unique(preds, axis=0).shape[0])
        s["n_distinct_hue_readings"] = int(np.unique(np.round(s["h_hat"], 6)).size)
        s["seconds"] = time.time() - t0
        np.savez_compressed(f, **{k: v for k, v in s.items()
                                  if not isinstance(v, dict)},
                            per_factor_other5=json.dumps(s["per_factor_other5"]))
        results[name] = s
        ref = (f"  HUEMAE_ref {s['err_ref'].mean()/DELTA:.4f}D" if "err_ref" in s else "")
        log(f"{name}: HUEMAE {s['err'].mean()/DELTA:.4f}D{ref}  "
            f"ROW2ACC {s['row2acc'].mean():.4f}  SNAPFRAC {s['snap'].mean():.4f}  "
            f"tuples {s['n_distinct_judged_tuples']}  ({s['seconds']:.1f}s)")

    hc.jdump({"tag": a.tag, "scope": hc.SCOPE, "n_requests": len(reqs),
              "n_corners": len(set(ck)), "corner_keys": ck.tolist(),
              "judge": jinfo, "decoder_param_hash": dinfo["param_hash"],
              "coefficient_report": coeff_rep,
              "store_pin": man["pin"], "arms_run": list(results.keys())},
             outdir / "run_manifest.json")
    log(f"wrote {outdir}/run_manifest.json")


if __name__ == "__main__":
    main()
