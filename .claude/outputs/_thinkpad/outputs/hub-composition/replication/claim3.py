"""CLAIM 3 -- decoder ceiling, store round-trip, and the norm-matched residual control.

Seed 0, the 512 `val` rows, features read ONLY from the frozen cache.
Everything downstream of the loaded artifacts is written here; no hub script imported.
"""
import json
import os
import sys
import time

import numpy as np

from chlu.config import get_default_config, gen_row_split_config
from chlu.core import gen_bridge, gen_store, gen_compose
from chlu.data import gen_rowsplit
from chlu.eval.gen_probe import factor_labels, decode_features, predict_factors
from chlu.experiments import exp_gen_store, exp_gen_compose, exp_gen_cell, exp_gen_dream

SEED = 0
DATASET, PRETEXT = "shapes3d", "recon"
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def exact_count(decoder, clf, F, feat_shape, true_lab, batch, tag):
    """decode -> classify -> count rows whose SIX predicted factors ALL match."""
    Z = np.asarray(F, np.float32).reshape((-1,) + tuple(feat_shape))
    imgs = decode_features(decoder, Z, batch)
    pred = predict_factors(clf, imgs, batch)
    tl = np.asarray(true_lab, pred.dtype)
    ok = np.all(pred == tl, axis=1)
    per_factor = (pred == tl).mean(axis=0)
    log("  %-16s exact %4d / %d   per-factor %s"
        % (tag, int(ok.sum()), ok.size, " ".join("%.4f" % v for v in per_factor)))
    return {"exact": int(ok.sum()), "n": int(ok.size),
            "per_factor": [float(v) for v in per_factor],
            "distinct_pred_tuples": int(np.unique(pred, axis=0).shape[0])}


def rel_err(A, X):
    A = np.asarray(A, np.float64)
    Xd = np.asarray(X, np.float64)
    num = np.linalg.norm(A - Xd, axis=1)
    den = np.linalg.norm(Xd, axis=1)
    r = num / den
    return {"mean_per_row": float(r.mean()), "median_per_row": float(np.median(r)),
            "min": float(r.min()), "max": float(r.max()),
            "global_frobenius": float(np.linalg.norm(A - Xd) / np.linalg.norm(Xd))}


def setup():
    """Load every frozen artifact and return the pieces. Re-used by claim3_sweep.py."""
    cfg = get_default_config()
    cfg.gen_compose.claude_root = os.environ["CHLU_CLAUDE_ROOT"]
    cfg.gen_compose.decoder_source = "winner"
    cfg.gen_compose.expected_decoder_recipe = "v1_capacity_schedule"
    # Refuse to let the store be re-trained: the claim is about the LOADED store.
    cfg.gen_cell.allow_store_training = False

    res = {"seed": SEED, "dataset": DATASET, "pretext": PRETEXT}

    bdir = exp_gen_store.bridge_dir(cfg, DATASET, PRETEXT, SEED)
    log("bridge dir", bdir)
    res["bridge_dir"] = str(bdir)

    mod = gen_bridge.dataset_module(DATASET)
    split, values = mod.build_split(cfg.gen_data.data_root)
    grid = split.grid
    log("split built; train", split.train_rows.size, "test", split.test_rows.size)

    rows = exp_gen_store.select_store_rows(split, cfg, SEED)
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(cfg)
    fit_rows, val_rows, split_info = gen_rowsplit.split_fit_val(
        pool, grid.factor_indices_of(pool), int(cfg.gen_cell.n_fit),
        int(cfg.gen_cell.n_val), SEED, mode=str(rs.mode),
        tiebreak_jitter=float(rs.tiebreak_jitter), seed_mult=int(rs.seed_mult),
        seed_offset=int(rs.seed_offset), log=None,
    )
    log("pool %d  fit %d  val %d" % (pool.size, fit_rows.size, val_rows.size))
    res["n_pool"] = int(pool.size)
    res["n_fit"] = int(fit_rows.size)
    res["n_val"] = int(val_rows.size)
    res["val_rows_checksum_sum"] = int(np.asarray(val_rows, np.int64).sum())
    res["val_rows_first10"] = [int(x) for x in val_rows[:10]]
    res["split_info_mode"] = str(split_info.get("mode"))

    # ---- features: READ-ONLY from the frozen cache -------------------------
    crows, feats, cman = gen_bridge.load_feature_cache(bdir, verify=False)
    D = int(cman["D"])
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    log("cache rows %s D=%d feat_shape=%s scope=%s"
        % (cman["n_rows"], D, feat_shape, cman.get("scope")))
    X = np.asarray(gen_bridge.features_for_rows(crows, feats, val_rows), np.float32)
    X = X.reshape((-1, D))
    X_store = np.asarray(gen_bridge.features_for_rows(crows, feats, pool), np.float32)
    X_store = X_store.reshape((-1, D))
    log("X", X.shape, "X_store", X_store.shape)
    res["cache"] = {"n_rows": int(cman["n_rows"]), "D": D,
                    "feat_shape": list(feat_shape), "scope": cman.get("scope"),
                    "encoder_param_hash": cman.get("encoder_param_hash")}

    true_lab = factor_labels(grid, val_rows)
    res["n_factors"] = int(true_lab.shape[1])
    res["factor_names"] = [str(n) for n in grid.names]

    # ---- decoder + external classifier -------------------------------------
    decoder, dinfo, bman = exp_gen_compose.resolve_decoder(cfg, bdir, SEED, log=log)
    clf, cinfo = exp_gen_cell.load_external_classifier(cfg, SEED, log=log)
    res["decoder"] = {k: dinfo.get(k) for k in
                      ("decoder_param_hash", "decoder_recipe", "decoder_path",
                       "decoder_source", "param_count")}
    res["classifier"] = {k: cinfo.get(k) for k in ("param_hash", "param_count", "path")}

    # ---- store: LOADED, never trained --------------------------------------
    plan = gen_store.load_store_plan(exp_gen_cell.store_plan_path(cfg, DATASET, PRETEXT))
    K = int(plan["K"])
    cfg.gen_dream.grouping_source = str(cfg.gen_cell.grouping_source)
    labels_vec, ginfo = exp_gen_dream.resolve_grouping(
        plan, X_store, cfg, SEED, bman["encoder_param_hash"], log=log)
    wc = gen_store.well_channels_from_labels(labels_vec, K)
    store, sprov = exp_gen_cell.resolve_store(cfg, DATASET, PRETEXT, SEED,
                                              X_store, wc, plan, feat_shape, log=log)
    if sprov.get("trained_here") is not False:
        raise AssertionError("store was TRAINED, not loaded -- refusing")
    res["store"] = {"provenance": sprov, "K": K, "n_modes": int(plan["n_modes"]),
                    "grouping_source": ginfo.get("grouping_source"),
                    "reproduces_plan": ginfo.get("reproduces_the_plan_bit_identically")}

    batch = int(cfg.gen_cell.eval_batch)
    return {"cfg": cfg, "res": res, "decoder": decoder, "clf": clf, "store": store,
            "X": X, "true_lab": true_lab, "feat_shape": feat_shape, "D": D,
            "batch": batch, "val_rows": val_rows, "grid": grid}


def main():
    S = setup()
    cfg, res, decoder, clf, store = S["cfg"], S["res"], S["decoder"], S["clf"], S["store"]
    X, true_lab, feat_shape, D = S["X"], S["true_lab"], S["feat_shape"], S["D"]
    batch = S["batch"]

    # ---- (a) DECODER CEILING ------------------------------------------------
    res["a_decoder_ceiling"] = exact_count(decoder, clf, X, feat_shape, true_lab,
                                           batch, "a/ceiling X")

    # ---- (b) STORE ROUND-TRIP ----------------------------------------------
    code = gen_store.address_codes(store, X)
    rt = np.asarray(gen_compose.read_compose(store, code), np.float32).reshape((-1, D))
    ncode = int(np.unique(np.asarray(code), axis=0).shape[0])
    log("codes", np.asarray(code).shape, "distinct codes", ncode)
    res["b_roundtrip"] = exact_count(decoder, clf, rt, feat_shape, true_lab,
                                     batch, "b/roundtrip rt")
    res["b_roundtrip"]["rel_err"] = rel_err(rt, X)
    res["b_roundtrip"]["n_distinct_codes"] = ncode
    log("  rel-err", res["b_roundtrip"]["rel_err"])

    e = rt - X
    en = np.linalg.norm(e.astype(np.float64), axis=1)
    xn = np.linalg.norm(X.astype(np.float64), axis=1)

    # ---- (c) CONTROLS -------------------------------------------------------
    controls = {}
    for gseed in (0, 1, 2):
        rng = np.random.default_rng(gseed)
        g = rng.standard_normal(X.shape)
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * en[:, None]
        Xg = (X.astype(np.float64) + g).astype(np.float32)
        k = "gauss_rng%d" % gseed
        controls[k] = exact_count(decoder, clf, Xg, feat_shape, true_lab, batch, "c/" + k)
        controls[k]["rel_err"] = rel_err(Xg, X)
        controls[k]["max_norm_mismatch"] = float(
            np.abs(np.linalg.norm(Xg.astype(np.float64) - X.astype(np.float64), axis=1)
                   - en).max())
        if gseed == 0:
            controls[k]["frac_nonneg"] = float((Xg >= 0).mean())

    Xm = (X.astype(np.float64) - e.astype(np.float64)).astype(np.float32)
    controls["minus_e"] = exact_count(decoder, clf, Xm, feat_shape, true_lab, batch,
                                      "c/minus_e")
    controls["minus_e"]["rel_err"] = rel_err(Xm, X)
    controls["minus_e"]["frac_nonneg"] = float((Xm >= 0).mean())

    for pseed in (0, 1, 2):
        rng = np.random.default_rng(1000 + pseed)
        perm = rng.permutation(X.shape[0])
        k = "perm_rng%d" % pseed
        Xp = (X.astype(np.float64) + e[perm].astype(np.float64)).astype(np.float32)
        controls[k] = exact_count(decoder, clf, Xp, feat_shape, true_lab, batch, "c/" + k)
        controls[k]["rel_err"] = rel_err(Xp, X)
        controls[k]["n_fixed_points"] = int((perm == np.arange(X.shape[0])).sum())
        if pseed == 0:
            controls[k]["frac_nonneg"] = float((Xp >= 0).mean())
    res["c_controls"] = controls

    cos = np.einsum("ij,ij->i", e.astype(np.float64), X.astype(np.float64)) / (en * xn)
    res["residual_stats"] = {
        "mean_e_norm": float(en.mean()), "mean_X_norm": float(xn.mean()),
        "mean_per_row_rel": float((en / xn).mean()),
        "cos_e_X_mean": float(cos.mean()),
        "frac_X_nonneg": float((X >= 0).mean()),
        "frac_rt_nonneg": float((rt >= 0).mean()),
        "mean_X": float(X.mean()), "mean_rt": float(rt.mean()),
        "std_X": float(X.std()),
    }
    return res


if __name__ == "__main__":
    r = main()
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    if dest:
        with open(dest, "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        log("wrote", dest)
