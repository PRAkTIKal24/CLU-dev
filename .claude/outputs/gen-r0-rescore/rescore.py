"""gen-r0-rescore — re-express R0's results under R20's corrected metrics.

SCRATCH ANALYSIS SCRIPT (protocol: not production code; lives under .claude/outputs/).

⛔ Nothing is trained here that did not already exist:
  * encoder  — loaded frozen from the bridge, param hash asserted
  * decoders — loaded frozen (bridge baseline + gen-decoder-lift V1 winner), hashes asserted
  * probe    — loaded frozen from the bridge, param hash asserted
  * the store — NOT checkpointed by gen-store-grouping, so it is RE-MATERIALISED from the
    identical seed/key chain and its bit-identity against the shipped store_manifest.json is
    ASSERTED (relative_mse, read.aggregate, read.shape, grouping labels) before any new number
    is believed.  A failed assertion aborts the seed.
  * the per-well leakage probes — re-fitted on the SAME rows with the SAME key chain and their
    old (degenerate) accuracies asserted against the manifest, then re-read under the corrected
    balanced-accuracy definition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from chlu.config import get_default_config
from chlu.core.gen_bridge import (
    assert_decoder_hash,
    dataset_module,
    load_feature_cache,
    model_dtype,
    param_hash,
)
from chlu.core.gen_store import (
    channel_correlation,
    dead_channels,
    grouping_labels,
    init_store,
    leakage_from_transfer,
    load_store_plan,
    reconstruction_mse,
    store_read,
    train_store,
    train_well_probe,
    well_channels_from_labels,
    well_probe_accuracy,
)
from chlu.eval.gen_probe import decode_features, factor_labels
from chlu.experiments.exp_gen_store import (
    _assert_cache_reencodes,
    _decoded_probe,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from chlu.utils.checkpoints import load_model

DATASET, PRETEXT = "shapes3d", "recon"
OUT = Path(os.environ["RESCORE_OUT"])
STORE_RUNS = Path(os.environ["STORE_RUNS"])     # gen-store-grouping/runs/shapes3d_recon
WINNER = Path(os.environ["WINNER_DIR"])          # gen-decoder-lift/winner


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- metrics
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Binary AUROC by the rank (Mann-Whitney U) identity, average ranks for ties."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, np.float64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def balanced_accuracy_binary(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tpr = float((yhat[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    tnr = float((yhat[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return {"balanced_accuracy": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr,
            "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
            "raw_accuracy": float((y == yhat).mean())}


def balanced_accuracy_multiclass(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Mean per-class recall over the classes REALISED in ``y``. Chance = 1/n_realised."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    classes = np.unique(y)
    rec = [float((yhat[y == c] == c).mean()) for c in classes]
    return {"balanced_accuracy": float(np.mean(rec)), "n_classes": int(classes.size),
            "chance": 1.0 / max(1, classes.size), "per_class_recall": rec,
            "raw_accuracy": float((y == yhat).mean()), "n": int(y.size)}


def probe_shape_readout(probe, images, shape_idx, batch=256):
    """Frozen probe -> (argmax shape class, softmax probabilities over shape classes)."""
    dtype = model_dtype(probe)
    fwd = eqx.filter_jit(jax.vmap(probe))
    preds, probs = [], []
    for i in range(0, int(images.shape[0]), batch):
        logits = fwd(jnp.asarray(images[i : i + batch], dtype))
        lg = np.asarray(logits[shape_idx], np.float64)
        preds.append(lg.argmax(-1))
        e = np.exp(lg - lg.max(-1, keepdims=True))
        probs.append(e / e.sum(-1, keepdims=True))
    return np.concatenate(preds), np.concatenate(probs)


def probe_all_factors(probe, images, batch=256):
    dtype = model_dtype(probe)
    fwd = eqx.filter_jit(jax.vmap(probe))
    out = []
    for i in range(0, int(images.shape[0]), batch):
        logits = fwd(jnp.asarray(images[i : i + batch], dtype))
        out.append(np.stack([np.asarray(jnp.argmax(lg, -1)) for lg in logits], axis=1))
    return np.concatenate(out).astype(np.int64)


def corner_discrimination(decoder, probe, f_pos, f_neg, shape_idx, cyl_class, batch):
    """⭐ Kill rule (i), corrected: cylinder-vs-not WITHIN the colour corner.
    positives = held-out cylinders; negatives = matched-seen non-cylinders, same corner."""
    d_pos = decode_features(decoder, np.asarray(f_pos, np.float32), batch)
    d_neg = decode_features(decoder, np.asarray(f_neg, np.float32), batch)
    p_pos, pr_pos = probe_shape_readout(probe, d_pos, shape_idx, batch)
    p_neg, pr_neg = probe_shape_readout(probe, d_neg, shape_idx, batch)
    y = np.r_[np.ones(p_pos.size), np.zeros(p_neg.size)]
    score = np.r_[pr_pos[:, cyl_class], pr_neg[:, cyl_class]]
    yhat = np.r_[p_pos == cyl_class, p_neg == cyl_class]
    out = balanced_accuracy_binary(y, yhat)
    out["auroc"] = auroc(y, score)
    out["pos_called_cylinder"] = int((p_pos == cyl_class).sum())
    out["neg_called_cylinder"] = int((p_neg == cyl_class).sum())
    out["mean_p_cyl_pos"] = float(pr_pos[:, cyl_class].mean())
    out["mean_p_cyl_neg"] = float(pr_neg[:, cyl_class].mean())
    out["pos_pred_hist"] = np.bincount(p_pos, minlength=pr_pos.shape[1]).tolist()
    out["neg_pred_hist"] = np.bincount(p_neg, minlength=pr_neg.shape[1]).tolist()
    out["decoded_mean_pos"] = float(d_pos.mean())
    return out


def well_probe_balanced(probe, norm, Z, Y, names) -> dict:
    """Balanced accuracy per factor for one well's linear probe on set ``Z``."""
    mu, sd = norm
    dt = probe.linears[0].weight.dtype
    Zj = (jnp.asarray(np.asarray(Z, np.float32), dt) - jnp.asarray(mu, dt)) / jnp.asarray(sd, dt)
    logits = eqx.filter_jit(jax.vmap(probe))(Zj)
    preds = np.stack([np.asarray(jnp.argmax(lg, -1)) for lg in logits], axis=1)
    Y = np.asarray(Y, np.int64)
    per = {str(nm): balanced_accuracy_multiclass(Y[:, j], preds[:, j])
           for j, nm in enumerate(names)}
    return {"n": int(Y.shape[0]),
            "per_factor": {k: v["balanced_accuracy"] for k, v in per.items()},
            "per_factor_detail": per,
            "aggregate": float(np.mean([v["balanced_accuracy"] for v in per.values()]))}


# --------------------------------------------------------------------------- one seed
def run_seed(seed: int) -> dict:
    t0 = time.time()
    cfg_all = get_default_config()
    cfg = cfg_all.gen_store
    bdir = bridge_dir(cfg_all, DATASET, PRETEXT, seed)
    shipped = json.load(open(STORE_RUNS / f"seed_{seed}" / "store_manifest.json"))
    plan = load_store_plan(STORE_RUNS / "store_plan.json")
    log(f"== seed {seed} · bridge {bdir} · K pinned {plan['K']}")

    mod = dataset_module(DATASET)
    split, values = mod.build_split(cfg_all.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    shape_idx = names.index("shape")
    cyl_class = int(np.where(np.asarray(grid.values[shape_idx]) == 1.0)[0][0])
    rows = select_store_rows(split, cfg_all, seed)

    bman = json.load(open(bdir / "bridge_manifest.json"))
    encoder = load_model(bdir / "encoder.pkl")
    dec_old = load_model(bdir / "decoder.pkl")
    probe = load_model(bdir / "probe.pkl")
    assert_decoder_hash(dec_old, bman["decoder_param_hash"])
    assert param_hash(encoder) == bman["encoder_param_hash"]
    assert param_hash(probe) == bman["probe_param_hash"]
    assert bman["decoder_param_hash"] == shipped["decoder_param_hash"]

    wh = json.load(open(WINNER / "winner_hashes.json"))[f"seed_{seed}"]
    dec_new = load_model(WINNER / f"decoder_v1_s{seed}.pkl")
    got = param_hash(dec_new)
    assert got == wh["param_hash"], f"V1 decoder hash {got} != {wh['param_hash']}"
    assert wh["encoder_param_hash_it_was_fit_against"] == bman["encoder_param_hash"]
    log(f"  frozen: enc {bman['encoder_param_hash'][:16]} · dec_old {bman['decoder_param_hash'][:16]}"
        f" · dec_new {wh['param_hash'][:16]} · probe {bman['probe_param_hash'][:16]}")

    crows, feats, cman = load_feature_cache(bdir, verify=True)
    assert cman["features_sha256"] == shipped["feature_cache_sha256"]
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    reenc = _assert_cache_reencodes(DATASET, cfg_all, encoder, crows, feats, cman, log=log)
    log(f"  cache scope={cman['scope']} rows={cman['n_rows']} reencode max|d|={reenc['max_abs_delta']}")

    ms_rows = matched_seen_rows(DATASET, split, values, rows["store_train"])
    assert int(ms_rows.size) == int(shipped["rows"]["matched_seen"]), (
        f"matched-seen count {ms_rows.size} != shipped {shipped['rows']['matched_seen']}")

    def gather(r):
        return gather_features(DATASET, cfg_all, encoder, crows, feats, cman, r, log=log)

    X_store = gather(rows["store_train"])
    X_a = gather(rows["eval_a"])
    X_b = gather(rows["eval_b"])
    X_ms = gather(ms_rows)
    lab_a = factor_labels(grid, rows["eval_a"])
    lab_b = factor_labels(grid, rows["eval_b"])
    lab_ms = factor_labels(grid, ms_rows)
    lab_store = factor_labels(grid, rows["store_train"])
    X_probe, lab_probe = X_store[: cfg.n_probe_train], lab_store[: cfg.n_probe_train]
    log(f"  rows: store={rows['store_train'].size} a={rows['eval_a'].size} b={rows['eval_b'].size} ms={ms_rows.size}")

    # corner predicate (same one matched_seen_rows uses) on eval_a, for the robustness denominator
    va = grid.values_of(rows["eval_a"])
    in_corner_a = ((va[:, 0] >= 0.8) & (va[:, 1] >= 0.8) & (va[:, 2] >= 0.8)
                   & (va[:, 3] >= 1.1) & (va[:, 5] > 20))
    outside = ~in_corner_a
    log(f"  eval_a: {int(in_corner_a.sum())}/{in_corner_a.size} rows inside the Montero colour corner")

    # the union set: held-out cylinders + matched-seen other shapes, SAME corner
    X_union = np.concatenate([X_b, X_ms], 0)
    lab_union = np.concatenate([lab_b, lab_ms], 0)

    Xc = X_store[: cfg.n_corr]
    dead = dead_channels(Xc)
    corr = channel_correlation(Xc)
    k = int(plan["K"])
    assert int(dead.sum()) == int(shipped["n_dead_channels"])

    # ⭐ the no-store ceiling, re-scored through BOTH decoders
    ceil_old = _decoded_probe(dec_old, probe, X_a.reshape((-1,) + feat_shape), lab_a, names, cfg.eval_batch)
    ceil_new = _decoded_probe(dec_new, probe, X_a.reshape((-1,) + feat_shape), lab_a, names, cfg.eval_batch)
    d_ceil = abs(ceil_old["aggregate"] - shipped["ceiling_no_store"]["aggregate"])
    assert d_ceil == 0.0, f"ceiling re-materialisation drifted by {d_ceil}"
    log(f"  ceiling old {ceil_old['aggregate']:.6f} (shipped {shipped['ceiling_no_store']['aggregate']:.6f}) "
        f"-> new {ceil_new['aggregate']:.6f}")

    # ⭐ kill rule (i), corrected, on REAL features (the decoder gate layer)
    gate = {
        "baseline_decoder": corner_discrimination(dec_old, probe, X_b.reshape((-1,) + feat_shape),
                                                  X_ms.reshape((-1,) + feat_shape), shape_idx,
                                                  cyl_class, cfg.eval_batch),
        "v1_decoder": corner_discrimination(dec_new, probe, X_b.reshape((-1,) + feat_shape),
                                            X_ms.reshape((-1,) + feat_shape), shape_idx,
                                            cyl_class, cfg.eval_batch),
    }
    for kk, vv in gate.items():
        log(f"  ⭐ GATE[{kk}] balacc={vv['balanced_accuracy']:.4f} auroc={vv['auroc']:.4f} "
            f"tpr={vv['tpr']:.4f} tnr={vv['tnr']:.4f}")

    arms = {}
    for grouping in ("correlation", "random"):
        ship = shipped["arms"][grouping]
        labels_vec, ginfo = grouping_labels(corr, k, grouping, seed, cfg.grouping_method,
                                            dead=dead, min_well_frac=cfg.min_well_frac)
        assert [int(x) for x in labels_vec] == list(ship["labels"]), \
            f"{grouping}: re-derived grouping labels differ from the shipped ones"
        wc = well_channels_from_labels(labels_vec, k)
        assert [len(w) for w in wc] == list(ship["well_sizes"])
        log(f"  [{grouping}] labels reproduced exactly; wells {[len(w) for w in wc]}")

        key = jax.random.PRNGKey(seed * 10_007 + 20_260_911 + (0 if grouping == "correlation" else 1))
        key, ik, tk = jax.random.split(key, 3)
        store = init_store(X_store, wc, cfg.n_modes, feat_shape, ik, seed=seed)
        log(f"  [{grouping}] re-materialising the store ({cfg.train_steps} steps) …")
        store, hist, key = train_store(store, X_store, cfg, tk, log=None)

        rec = reconstruction_mse(store, X_a, cfg.eval_batch)
        d = abs(rec["relative_mse"] - ship["recon"]["relative_mse"])
        assert d == 0.0, f"{grouping}: store re-materialisation drifted, |d rel_mse| = {d}"
        read_a = store_read(store, X_a, cfg.eval_batch)
        read_old = _decoded_probe(dec_old, probe, read_a, lab_a, names, cfg.eval_batch)
        for fld in ("aggregate",):
            dd = abs(read_old[fld] - ship["read"][fld])
            assert dd == 0.0, f"{grouping}: shipped read {fld} not reproduced (|d| = {dd})"
        assert read_old["per_factor"]["shape"] == ship["read"]["per_factor"]["shape"]
        log(f"  [{grouping}] ⭐ BIT-IDENTICAL re-materialisation: rel_mse {rec['relative_mse']!r}, "
            f"read agg {read_old['aggregate']!r}")

        # ⭐ correction 3: the same store read, the NEW decoder
        read_new = _decoded_probe(dec_new, probe, read_a, lab_a, names, cfg.eval_batch)
        log(f"  [{grouping}] READ old agg {read_old['aggregate']:.4f} shape {read_old['per_factor']['shape']:.4f}"
            f"  ->  new agg {read_new['aggregate']:.4f} shape {read_new['per_factor']['shape']:.4f}")

        # ⭐ correction 1 on the STORE READ (both decoders)
        read_b = store_read(store, X_b, cfg.eval_batch)
        read_ms = store_read(store, X_ms, cfg.eval_batch)
        corner = {
            "baseline_decoder": corner_discrimination(dec_old, probe, read_b, read_ms, shape_idx,
                                                      cyl_class, cfg.eval_batch),
            "v1_decoder": corner_discrimination(dec_new, probe, read_b, read_ms, shape_idx,
                                                cyl_class, cfg.eval_batch),
        }
        for kk, vv in corner.items():
            log(f"  [{grouping}] corner-store[{kk}] balacc={vv['balanced_accuracy']:.4f} auroc={vv['auroc']:.4f}")

        # ---- correction 2: the leakage x-axis, old (reproduced) and corrected ----
        per_well = []
        for j, w in enumerate(wc):
            cols = np.asarray(w, np.int64)
            key, pk = jax.random.split(key)
            wp, norm, phist = train_well_probe(X_probe[:, cols], lab_probe, cards, cfg, pk)
            acc_ho = well_probe_accuracy(wp, norm, X_b[:, cols], lab_b, names)
            acc_ms = well_probe_accuracy(wp, norm, X_ms[:, cols], lab_ms, names)
            old = leakage_from_transfer(acc_ho, acc_ms, names)
            ship_w = ship["leakage"]["per_well"][j]
            assert acc_ho["aggregate"] == ship_w["acc_heldout"]["aggregate"], (
                f"{grouping} well {j}: old held-out accuracy not reproduced")
            assert acc_ms["aggregate"] == ship_w["acc_matched_seen"]["aggregate"]

            b_union = well_probe_balanced(wp, norm, X_union[:, cols], lab_union, names)
            b_seen = well_probe_balanced(wp, norm, X_a[:, cols], lab_a, names)
            b_seen_out = well_probe_balanced(wp, norm, X_a[outside][:, cols], lab_a[outside], names)
            new = {
                "leakage_per_factor": {str(n): 1.0 - b_union["per_factor"][str(n)] / b_seen["per_factor"][str(n)]
                                       for n in names},
                "leakage_aggregate": 1.0 - b_union["aggregate"] / b_seen["aggregate"],
                "leakage_per_factor_outside": {
                    str(n): 1.0 - b_union["per_factor"][str(n)] / b_seen_out["per_factor"][str(n)] for n in names},
                "leakage_aggregate_outside": 1.0 - b_union["aggregate"] / b_seen_out["aggregate"],
            }
            per_well.append({"well": j, "d_k": int(cols.size),
                             "old": {"acc_heldout": acc_ho, "acc_matched_seen": acc_ms, **old},
                             "balacc_union": b_union, "balacc_seen": b_seen,
                             "balacc_seen_outside": b_seen_out, "corrected": new,
                             "probe_loss": [float(phist[0]), float(phist[-1])]})
            log(f"    [{grouping}] well {j}: OLD leak agg {old['leakage_aggregate']:+.4f} "
                f"shape {old['leakage_per_factor']['shape']:+.4f}  |  NEW leak agg "
                f"{new['leakage_aggregate']:+.4f} shape {new['leakage_per_factor']['shape']:+.4f}")

        pooled_union = {str(n): float(np.mean([p["balacc_union"]["per_factor"][str(n)] for p in per_well]))
                        for n in names}
        pooled_seen = {str(n): float(np.mean([p["balacc_seen"]["per_factor"][str(n)] for p in per_well]))
                       for n in names}
        pooled = {
            "balacc_union_per_factor": pooled_union, "balacc_seen_per_factor": pooled_seen,
            "balacc_union_aggregate": float(np.mean(list(pooled_union.values()))),
            "balacc_seen_aggregate": float(np.mean(list(pooled_seen.values()))),
            "leakage_per_factor": {n: 1.0 - pooled_union[n] / pooled_seen[n] for n in pooled_union},
        }
        pooled["leakage_aggregate"] = 1.0 - pooled["balacc_union_aggregate"] / pooled["balacc_seen_aggregate"]
        pooled["mean_of_per_well_leakage"] = float(np.mean([p["corrected"]["leakage_aggregate"] for p in per_well]))
        cells = [p["corrected"]["leakage_per_factor"][str(n)] for p in per_well for n in names]
        old_cells = [p["old"]["leakage_per_factor"][str(n)] for p in per_well for n in names]
        pooled["cell_min"], pooled["cell_max"] = float(np.min(cells)), float(np.max(cells))
        pooled["cell_min_old"], pooled["cell_max_old"] = float(np.nanmin(old_cells)), float(np.nanmax(old_cells))
        pooled["n_cells"] = len(cells)
        log(f"  [{grouping}] ⭐ POOLED corrected leakage {pooled['leakage_aggregate']:+.4f} "
            f"(old {ship['leakage']['pooled']['leakage_aggregate']:+.4f}); cells in "
            f"[{pooled['cell_min']:+.4f}, {pooled['cell_max']:+.4f}] "
            f"(old [{pooled['cell_min_old']:+.4f}, {pooled['cell_max_old']:+.4f}])")

        arms[grouping] = {
            "well_sizes": [len(w) for w in wc],
            "recon": rec,
            "read_old_decoder": read_old,
            "read_new_decoder": read_new,
            "corner_store_read": corner,
            "leakage": {"per_well": per_well, "pooled": pooled},
        }

    out = {
        "schema": "chlu-gen-r0-rescore/1",
        "seed": seed, "K": k, "dataset": DATASET, "pretext": PRETEXT,
        "base_commit": os.environ.get("BASE_COMMIT", ""),
        "hashes": {"encoder": bman["encoder_param_hash"], "decoder_baseline": bman["decoder_param_hash"],
                   "decoder_v1": wh["param_hash"], "probe": bman["probe_param_hash"],
                   "feature_cache_sha256": cman["features_sha256"], "cache_scope": cman["scope"],
                   "cache_n_rows": int(cman["n_rows"]), "cache_reencode": reenc},
        "rows": {k2: int(v) for k2, v in
                 dict(store_train=rows["store_train"].size, eval_a=rows["eval_a"].size,
                      eval_b=rows["eval_b"].size, matched_seen=ms_rows.size,
                      union=X_union.shape[0], eval_a_outside_corner=int(outside.sum())).items()},
        "cyl_class_index": cyl_class,
        "ceiling_no_store": {"old_decoder": ceil_old, "new_decoder": ceil_new,
                             "shipped_aggregate": shipped["ceiling_no_store"]["aggregate"]},
        "gate_real_features": gate,
        "arms": arms,
        "shipped_reference": {g: {"read": shipped["arms"][g]["read"],
                                  "leakage_pooled": shipped["arms"][g]["leakage"]["pooled"]}
                              for g in ("correlation", "random")},
        "wall_clock_s": time.time() - t0,
    }
    p = OUT / f"rescore_seed_{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2, sort_keys=True, default=float)
    log(f"== seed {seed} done in {out['wall_clock_s']:.0f}s -> {p}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    run_seed(a.seed)
