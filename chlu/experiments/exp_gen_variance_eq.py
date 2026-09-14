"""Experiment: ⭐ **does k-means drop ``scale`` because it is LOW-VARIANCE?**

⭐ **DIAL DECLARATION — *feature scaling before the store*.**
- **Dial:** feature scaling before the store.  ⛔ The encoder, the decoder, the
  **partition** and ``K`` are all UNCHANGED.  The only thing any arm changes is the
  **metric the per-well k-means codebook sees**.
- **Laundering control:** ⭐ **``knn_proto`` must be BIT-IDENTICAL.**  It never sees
  the transform, so if its number moves **something changed that was not meant to**
  — the run stops and says so.
- **Falsifies the claim:** ⭐ **(a)** the ``scale`` + ``orientation`` probe **on the
  store code** rises above the unwhitened baseline **beyond seed spread**; **(b)**
  the **distinct configuration count rises**.  ⛔ If neither moves, the variance
  hypothesis is **refuted** and that is the result.
- **Does NOT falsify:** a whitened arm that trades hue retention for
  ``scale``/``orientation`` retention at a **flat total**.  ⭐ That is the
  pre-registered CAPACITY reading (PREREG §8) — a first-class result that sends the
  repair to ``M``, ⛔ never to the metric.

⛔ **What the path actually is** (R40.B, binding): a **product quantiser + count
table** — correlation-partitioned wells × per-well k-means codebook; address book =
per-factor-value mode histograms, naive-Bayes, ``argmax`` per well; read = codebook
copy-back.  ⛔ No landscape, no particle, no settling, no inter-well coupling.

**The four legs, in the ruled order:**

1. ⛔ **TEST THE PREMISE BEFORE ACTING ON IT** — where in each well's variance
   spectrum does each factor's discriminative signal sit?  ⛔ If ``scale`` and
   ``orientation`` do **not** sit in low-variance directions the hypothesis is dead,
   this experiment says so and **STOPS**; it does not whiten anyway to see what
   happens.
2. **Equalise, then re-fit and re-probe.** ⭐ **ORDER IS RULED (R43.1): the grouping
   is fitted on UNWHITENED features FIRST, then the transform.**  Primary arm =
   per-well PCA whitening, secondary = diagonal standardisation, control = the
   literal identity through the identical code path.
3. ⭐ **The six-factor graded profile, before and after, with a TOTAL-RETAINED
   line.**  ⛔ Not just ``scale`` and ``orientation``: the question is **trade vs
   reduction.**
4. ⭐⭐ **The per-well capacity bound.**  ``K·log₂M = 30 bits`` vs
   ``log₂(480,000) = 18.87`` ⇒ the code is **sufficient in aggregate** and ⛔ "30
   bits" may not be written as the cause; **per well** ``M = 64`` is already exceeded
   by two hues alone (100).

⚠ **Three environment facts** (paid for already by R31/R34/`gen-loss-attribution`):
a worktree has no ``.claude/``, so ``$CHLU_GEN_DATA_ROOT``, ``$CHLU_CLAUDE_ROOT``
**and** ``$CHLU_GEN_BRIDGE_ROOT`` must all be set, or the run dies at ``bridge_dir``.

Runnable::

    python -m chlu.experiments.exp_gen_variance_eq --quick   # ⚠ plumbing smoke only
    python -m chlu.experiments.exp_gen_variance_eq
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, gen_variance_eq_config, get_default_config  # noqa: E402
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_compose import compose_codes, request_posterior  # noqa: E402
from ..core.gen_dream import (  # noqa: E402
    WINNING_DECODER_RECIPE,
    assert_dream_decoder,
    fit_address_book,
    requests_from_labels,
)
from ..core.gen_request import FactorPrototypes  # noqa: E402
from ..core.gen_store import (  # noqa: E402
    address_codes,
    block_diagonality,
    channel_correlation,
    init_store,
    read_from_codes,
    store_bytes,
    train_store,
    well_channels_from_labels,
    well_relevance,
)
from ..core.gen_whiten import (  # noqa: E402
    TRANSFORMS,
    WellTransform,
    assert_deployment_equivalence,
    assert_grouping_is_unwhitened,
    capacity_reading,
    per_well_capacity,
    premise_verdict,
    transform_byte_ledger,
    unwhiten_store,
    well_variance_spectrum,
)
from ..eval.gen_attribution import (  # noqa: E402
    CONTROL_POINT,
    chance_rates,
    majority_baseline,
    mi_table,
    n_values_realised,
    onehot_codes,
    per_factor_rows,
    probe_point,
    random_codes,
    retention,
    summarize_across_seeds,
)
from ..eval.gen_probe import decode_features, factor_labels, probe_accuracy  # noqa: E402
from ..experiments.exp_gen_attribution import load_grouping, load_attrib_store  # noqa: E402
from ..experiments.exp_gen_compose import _claude_root, _write_json  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: The row sets the probe is SCORED on.  ⭐ ``SPREAD`` is the headline (the loss is
#: global, R34); the corner is reported separately and ⛔ never pooled into it.
EVAL_SETS = ("SPREAD", "CORNER_B", "CORNER_MS")
#: ⭐ The reference row: `gen-read-compose`'s frozen store, so this spoke's
#: re-measured ladder can be checked against `gen-loss-attribution`'s.
REFERENCE_ARM = "shipped"


def _cfg(config: CHLUConfig):
    return gen_variance_eq_config(config)


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only.  ⛔ A ``--quick`` number is NEVER reported."""
    c = _cfg(config)
    c.seeds = list(c.quick_seeds)
    c.n_store_train = int(c.quick_n_store_train)
    c.n_spread_train = int(c.quick_n_spread_train)
    c.n_spread_test = int(c.quick_n_spread_test)
    c.probe_steps = int(c.quick_probe_steps)
    c.spectrum_rows = int(c.quick_spectrum_rows)
    c.max_matched_seen = int(c.quick_max_matched_seen)
    c.round_trip_rows = int(c.quick_round_trip_rows)
    c.decode_rows = int(c.quick_decode_rows)
    c.block_diagonality_rows = int(c.quick_block_diagonality_rows)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    c.stop_if_premise_dead = bool(c.quick_stop_if_premise_dead)
    config.gen_store.train_steps = int(c.quick_store_train_steps)
    return config


def veq_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-variance-equalisation"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


def _sha(a: np.ndarray) -> str:
    """sha256 of an array's bytes — the laundering control's fingerprint."""
    x = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(x.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Rows — ⛔ identical draw to gen-loss-attribution's, so the probes compare
# ---------------------------------------------------------------------------


def select_rows(config: CHLUConfig, dataset: str, split, values, seed: int) -> dict:
    """``store_train`` (the rows the codebook is fitted on) + the probe's row sets.

    ⭐ ``SPREAD_train`` / ``SPREAD`` are drawn with **the same RNG stream and the same
    pool** as ``exp_gen_attribution.select_rows``, so a retention measured here is
    comparable to the one measured there.  ⛔ Nothing is ever fitted on the corner.
    """
    c = _cfg(config)
    if int(c.n_store_train):
        config.gen_store.n_store_train = int(c.n_store_train)
    rows = select_store_rows(split, config, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), rows["store_train"])
    rng = np.random.default_rng(int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset))
    n = min(int(c.n_spread_train) + int(c.n_spread_test), pool.size)
    take = rng.permutation(pool)[:n]
    n_tr = min(int(c.n_spread_train), n)
    ms = matched_seen_rows(dataset, split, values, rows["store_train"])
    if int(c.max_matched_seen) and ms.size > int(c.max_matched_seen):
        ms = np.sort(rng.permutation(ms)[: int(c.max_matched_seen)])
    return {
        "store_train": rows["store_train"],
        "SPREAD_train": np.sort(take[:n_tr]),
        "SPREAD": np.sort(take[n_tr:]),
        "CORNER_B": np.asarray(rows["eval_b"], np.int64),
        "CORNER_MS": np.asarray(ms, np.int64),
        "bridge_train": rows["bridge_train"],
    }


# ---------------------------------------------------------------------------
# 2. ⭐ THE LAUNDERING CONTROL — knn_proto never sees the transform
# ---------------------------------------------------------------------------


def knn_proto_fingerprint(grid, X: dict, rows: dict, min_per_value: int = 1) -> dict:
    """⭐ A fingerprint of everything ``knn_proto`` is computed FROM.

    ``knn_proto`` is R8's per-factor **prototype** embedding, decoded directly; it
    reads the frozen features and ⛔ never the store, so the transform is out of its
    scope **by construction**.  The realistic way that guarantee breaks is an
    **in-place mutation of the shared feature array**, which a source diff cannot
    see — so this takes the prototype table's own embedding of the eval rows plus a
    sha256 of every feature array, before and after the whitening pipeline runs.

    ⚠ **Declared limit:** the *shipped* ``knn_proto`` balanced accuracy is a
    decoder + classifier cell result this spoke does not re-run.  What is proved
    here is that its **inputs and its embedding are byte-identical**.
    """
    protos = FactorPrototypes.fit(
        grid, X["store_train"], rows["store_train"], min_per_value=int(min_per_value)
    )
    emb = {
        k: np.asarray(protos.embed_many(
            [{"request": [{"factor_index": int(j), "value": float(grid.values[j][v])}
                          for j, v in enumerate(row)]}
             for row in factor_labels(grid, rows[k])]
        ), np.float32)
        for k in ("SPREAD", "CORNER_B")
    }
    return {
        "prototype_table_bytes": int(protos.n_bytes),
        "n_prototypes": int(protos.n_prototypes),
        "n_rows_fit": int(protos.n_rows_fit),
        "embedding_sha256": {k: _sha(v) for k, v in emb.items()},
        "embedding": emb,
        "feature_sha256": {k: _sha(v) for k, v in X.items()},
    }


def assert_knn_proto_identical(before: dict, after: dict) -> dict:
    """⛔ Stop the run if ``knn_proto``'s inputs or embedding moved by one bit."""
    deltas = {}
    for k, e in before["embedding"].items():
        d = float(np.abs(e.astype(np.float64) - after["embedding"][k].astype(np.float64)).max())
        deltas[k] = d
    feat_ok = before["feature_sha256"] == after["feature_sha256"]
    emb_ok = before["embedding_sha256"] == after["embedding_sha256"]
    worst = max(deltas.values()) if deltas else 0.0
    if not (feat_ok and emb_ok and worst == 0.0):
        raise AssertionError(
            "⛔ knn_proto MOVED — the transform escaped its scope and the comparison "
            f"is VOID. feature sha equal={feat_ok} embedding sha equal={emb_ok} "
            f"max |Δ|={worst}"
        )
    return {
        "feature_sha256_identical": bool(feat_ok),
        "embedding_sha256_identical": bool(emb_ok),
        "max_abs_delta": float(worst),
        "per_set_max_abs_delta": deltas,
        "verdict": "BIT-IDENTICAL",
        "scope_note": (
            "proves the INPUTS and the prototype EMBEDDING are byte-identical; the "
            "shipped knn_proto balanced accuracy is a decoder+classifier cell result "
            "this spoke does not re-run"
        ),
    }


# ---------------------------------------------------------------------------
# 3. One arm: fit the transform, re-fit the codebook, read it back
# ---------------------------------------------------------------------------


def _hard_read_mse(store, tr, Z: np.ndarray, X: np.ndarray) -> dict:
    """Hard-read error of ``store`` **in the ORIGINAL feature space** (every arm)."""
    codes = address_codes(store, Z)
    r = np.asarray(read_from_codes(store, codes), np.float32)
    r = tr.inverse(r.reshape((r.shape[0], -1)))
    Xf = np.asarray(X, np.float32).reshape((np.asarray(X).shape[0], -1))
    err = float(np.mean((r - Xf) ** 2))
    var = float(np.mean((Xf - Xf.mean(axis=0, keepdims=True)) ** 2))
    return {"mse": err, "feature_variance": var,
            "relative_mse": err / var if var > 0 else float("nan"),
            "n": int(Xf.shape[0])}


def run_arm(
    config: CHLUConfig,
    arm: str,
    X: dict,
    Y: dict,
    rows: dict,
    well_channels,
    feat_shape,
    cards,
    names,
    seed: int,
    log: Callable = print,
) -> dict:
    """⭐ One metric, one codebook, one read — everything else held fixed.

    ⛔ The key, the steps, the rows, ``K``, ``M`` and the **partition** are identical
    across arms by construction: the only difference is the space
    :func:`chlu.core.gen_store.init_store` / :func:`chlu.core.gen_store.train_store`
    see.
    """
    c = _cfg(config)
    t0 = time.time()
    tr = WellTransform.fit(
        X["store_train"], well_channels, kind=str(arm), eps=float(c.whiten_eps),
        scale_match=bool(c.scale_match_to_raw),
    )
    assert_grouping_is_unwhitened(tr.well_channels, well_channels, f"arm {arm}")
    rt_rows = X["SPREAD"][: int(c.round_trip_rows)]
    rt = tr.round_trip(rt_rows)
    if np.isfinite(rt["max_rel_error"]) and rt["max_rel_error"] > float(c.round_trip_tol):
        raise AssertionError(
            f"arm {arm}: round trip inverse(forward(x)) != x (max rel error "
            f"{rt['max_rel_error']:.3e} > {c.round_trip_tol}) — ⛔ a non-invertible "
            "transform cannot hand the decoder its usual space"
        )
    led = transform_byte_ledger(arm, tr.well_sizes)
    log(f"    [{arm}] round-trip max|Δ| {rt['max_abs_error']:.3e} "
        f"(rel {rt['max_rel_error']:.3e}) · deployed {led['deployed_bytes']:,} B "
        f"({led['deployed_vs_budget']:.2f}× the store budget) · fit-time-only 0 B")

    # ⭐ The ONLY thing that differs between arms.
    Zt = tr.forward(X["store_train"])
    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911)
    key, ik, tk = jax.random.split(key, 3)
    store0 = init_store(Zt, well_channels, int(config.gen_store.n_modes), feat_shape,
                        ik, seed=int(seed))
    store, hist, key = train_store(store0, Zt, config.gen_store, tk, log=None)
    # ⭐ Under-training is a live alternative explanation for any arm that loses, so
    # the k-means++ INIT's hard read is measured beside the trained one rather than
    # inferred from a loss trace whose tau moves during the run.
    _sp = tr.forward(X["SPREAD"])
    _init_mse = _hard_read_mse(store0, tr, _sp, X["SPREAD"])
    log(f"    [{arm}] codebook re-fitted: {config.gen_store.train_steps} steps, "
        f"loss {hist[0]:.5f} -> {hist[-1]:.5f}")

    # ⭐ The fit-time-only deployment, proved rather than claimed.
    store_unw = unwhiten_store(store, tr)
    codes = {}
    for k, v in X.items():
        codes[k] = address_codes(store, tr.forward(v))
    dep = assert_deployment_equivalence(
        store, store_unw, tr, codes["SPREAD"][: int(c.round_trip_rows)]
    )

    # ⭐ The read, mapped BACK into the decoder's usual space before anything scores it.
    reads = {}
    for k in EVAL_SETS:
        r = np.asarray(read_from_codes(store, codes[k]), np.float32)
        reads[k] = tr.inverse(r.reshape((r.shape[0], -1)))
    mse = {}
    for k in EVAL_SETS:
        Xk = np.asarray(X[k], np.float32).reshape((X[k].shape[0], -1))
        err = float(np.mean((reads[k] - Xk) ** 2))
        var = float(np.mean((Xk - Xk.mean(axis=0, keepdims=True)) ** 2))
        mse[k] = {"mse": err, "feature_variance": var,
                  "relative_mse": err / var if var > 0 else float("nan"),
                  "n": int(Xk.shape[0]),
                  "note": "⛔ measured in the ORIGINAL feature space, every arm"}
    mse["SPREAD_at_kmeanspp_init"] = _init_mse
    mse["training_improved_the_hard_read"] = bool(
        mse["SPREAD"]["mse"] < _init_mse["mse"])

    book = fit_address_book(codes["store_train"], Y["store_train"], cards,
                            float(c.address_smoothing))
    req = {k: requests_from_labels(Y[k], "full", 0, int(seed)) for k in EVAL_SETS}
    comp = {k: compose_codes(request_posterior(book, req[k])) for k in EVAL_SETS}
    distinct = {
        "request_side": {k: {"n_distinct_code_configs": int(comp[k]["n_distinct_code_configs"]),
                             "n_requests": int(req[k].shape[0]),
                             "mean_confidence": float(comp[k]["mean_confidence"])}
                         for k in EVAL_SETS},
        "feature_side": {k: {"n_distinct_codes": int(np.unique(codes[k], axis=0).shape[0]),
                             "n_rows": int(codes[k].shape[0])}
                         for k in EVAL_SETS} if bool(c.distinct_on_spread) else {},
    }
    log(f"    [{arm}] distinct-of-{req['CORNER_B'].shape[0]} on CORNER_B "
        f"(request side) = {comp['CORNER_B']['n_distinct_code_configs']} · "
        f"distinct codes on SPREAD = {distinct['feature_side']['SPREAD']['n_distinct_codes']}"
        if bool(c.distinct_on_spread) else "")

    mi = {k: mi_table(codes[k], Y[k], cards, names) for k in EVAL_SETS}
    rel = well_relevance(codes["SPREAD"], np.asarray(Y["SPREAD"], np.int64), list(cards))
    cap = per_well_capacity(
        np.asarray(rel["mi_bits"]), np.asarray(rel["factor_entropy_bits"]),
        cards, names, int(config.gen_store.n_modes),
        float(c.capacity_carry_threshold),
    )
    return {
        "arm": str(arm),
        "transform": {
            "kind": str(arm),
            "well_sizes": tr.well_sizes,
            "eps": float(tr.eps),
            "n_rows_fit": int(tr.n_rows_fit),
            "scale_match_to_raw": bool(tr.scale_match),
            "per_well_scalar": [float(x) for x in tr.scales],
            "round_trip": rt,
            "byte_ledger": led,
            "deployment_equivalence": dep,
            "fit_time_only": True,
            "deployment_declaration": (
                "⭐ FIT-TIME-ONLY on the request-side (cell) path: the M centres are "
                "stored UN-whitened (unwhiten_store) and the read is a plain gather "
                "⇒ 0 deployed transform bytes, ASSERTED by deployment_equivalence. "
                "⛔ DEPLOYED on any feature-side encode (address_codes): "
                f"{led['deployed_bytes']:,} B = {led['deployed_vs_budget']:.2f}× the "
                "524,288 B store budget."
            ),
        },
        "store_bytes": store_bytes(store),
        "train_loss_first": float(hist[0]),
        "train_loss_last": float(hist[-1]),
        "reconstruction_mse": mse,
        "distinct": distinct,
        "mi": mi,
        "capacity": cap,
        "codes": codes,
        "reads": reads,
        "store": store,
        "store_unwhitened": store_unw,
        "wall_clock_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 4. One seed
# ---------------------------------------------------------------------------


def run_seed(config: CHLUConfig, dataset: str, pretext: str, seed: int,
             out_dir: Path, log: Callable = print) -> dict:
    c = _cfg(config)
    t0 = time.time()
    bdir = bridge_dir(config, dataset, pretext, int(seed))
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(c.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(f"  bridge {bdir} · cache scope={cman.get('scope')!r} rows={crows.size}")

    from ..core.gen_bridge import dataset_module

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_rows(config, dataset, split, values, int(seed))
    config.gen_store.encode_chunk = int(c.encode_chunk)
    reencode = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)

    X, Y, NVR = {}, {}, {}
    for name in ("store_train", "SPREAD_train") + EVAL_SETS:
        r = rows[name]
        log(f"    gathering {name}: {r.size} rows")
        X[name] = gather_features(dataset, config, encoder, crows, feats, cman, r, log=log)
        Y[name] = factor_labels(grid, r)
        NVR[name] = n_values_realised(Y[name], names, cards)

    # ⛔ The partition: loaded from gen-commitment-collapse, fitted on UNWHITENED
    # features, and asserted channel-for-channel against the SHIPPED store.
    labels_vec = load_grouping(config, dataset, pretext, int(seed))
    shipped, sinfo = load_attrib_store(config, dataset, pretext, int(seed), labels_vec, log)
    wc = well_channels_from_labels(labels_vec, int(shipped.n_wells))
    order = assert_grouping_is_unwhitened(wc, shipped.well_channels, "shipped store")
    config.gen_store.n_modes = int(shipped.n_modes)
    feat_shape = tuple(shipped.feat_shape)
    log(f"  ⭐ grouping ORDER asserted: {order['grouping_fitted_on']} · "
        f"well sizes {order['well_sizes']}")

    # ⭐ THE LAUNDERING CONTROL, taken BEFORE anything is transformed.
    proto_before = (knn_proto_fingerprint(grid, X, rows)
                    if bool(c.assert_knn_proto_identical) else None)

    # ---- ⛔ LEG 1: TEST THE PREMISE BEFORE ACTING ON IT ---------------------
    ns = int(c.spectrum_rows) or X["store_train"].shape[0]
    ns = min(ns, X["store_train"].shape[0])
    Xs, Ys = X["store_train"][:ns], Y["store_train"][:ns]
    spectra = [well_variance_spectrum(Xs, w, Ys, names, cards) for w in wc]
    premise = premise_verdict(
        spectra, names, list(c.premise_probe_factors), list(c.premise_reference_factors),
        float(c.premise_rank_ratio), float(c.premise_share_ratio),
    )
    for nm in names:
        best = premise["headline_well_per_factor"][nm]
        row = spectra[best]["per_factor"][nm]
        log(f"    [leg1] {nm:>12s} well{best} VE={row['variance_explained']:.4f} "
            f"rank_frac_median={row['rank_frac_median']:.4f} "
            f"top10%={row['frac_signal_in_top10pct']:.3f} "
            f"n_val={row['n_values_realised']}")
    log(f"  ⭐ LEG 1 PREMISE: {premise['verdict']} "
        f"(direction {premise['direction_test_passed']}, share {premise['share_test_passed']})")

    rec = {
        "schema": "chlu-gen-variance-eq/1",
        "seed": int(seed),
        "dataset": dataset,
        "pretext": pretext,
        "dial": "feature scaling before the store — ⛔ encoder/decoder/partition/K unchanged",
        "factor_names": names,
        "cardinalities": [int(x) for x in cards],
        "bridge_dir": str(bdir),
        "encoder_param_hash": enc_hash,
        "cache": {"scope": cman.get("scope"), "n_rows": int(crows.size),
                  "D": int(cman["D"]), "reencode_check": reencode},
        "shipped_store": sinfo,
        "grouping_order": order,
        "rows": {k: int(np.asarray(v).size) for k, v in rows.items()},
        "n_values_realised": NVR,
        "leg1_spectra": [{"well": i, **{k: v for k, v in s.items()}} for i, s in enumerate(spectra)],
        "leg1_premise": premise,
        "wall_clock_s": float(time.time() - t0),
    }
    if premise["stop_after_leg1"] and bool(c.stop_if_premise_dead):
        rec["stopped"] = (
            "⛔ PREMISE DEAD (PREREG §3): scale/orientation do NOT sit in low-variance "
            "directions. The variance hypothesis is REFUTED at Leg 1 and nothing was "
            "whitened — ⛔ deliberately, rather than whitening anyway to see what happens."
        )
        rec["complete"] = True
        log("  ⛔ " + rec["stopped"])
        _write_json(rec, Path(out_dir) / f"veq_s{int(seed)}.json")
        return rec

    # ---- the cheap witness for WHY the order is ruled ----------------------
    nb = min(int(c.block_diagonality_rows), X["store_train"].shape[0])
    if nb > 8:
        tr_pca = WellTransform.fit(X["store_train"], wc, "pca_whiten", float(c.whiten_eps),
                                   scale_match=bool(c.scale_match_to_raw))
        bd_before = block_diagonality(channel_correlation(X["store_train"][:nb]), labels_vec)
        bd_after = block_diagonality(
            channel_correlation(tr_pca.forward(X["store_train"][:nb])), labels_vec)
        rec["order_witness"] = {
            "n_rows": int(nb),
            "block_diagonality_unwhitened": bd_before,
            "block_diagonality_whitened": bd_after,
            "why": (
                "⭐ The correlation structure the grouping was fitted ON is destroyed "
                "by per-well whitening (within-well correlations become the identity). "
                "⛔ That is exactly why R43.1 rules the grouping is fitted FIRST: "
                "whitening first would move the partition, which is the one part "
                "gen-loss-attribution EXONERATED."
            ),
        }
        del tr_pca

    # ---- ⭐ THE ONE PROBE — one key, the encoder point shared by every arm ---
    key = jax.random.PRNGKey(int(c.probe_seed_offset) + int(seed))

    def _fit(name, Ztr, Zev):
        return probe_point(name, Ztr, Y["SPREAD_train"],
                           {k: (Zev[k], Y[k]) for k in EVAL_SETS},
                           cards, names, c, key, log=log)

    points = {"encoder": _fit("encoder", X["SPREAD_train"], {k: X[k] for k in EVAL_SETS})}

    # ---- the reference row: the SHIPPED store, re-measured here -------------
    M = int(shipped.n_modes)
    arms_out = {}
    ship_codes = {k: address_codes(shipped, np.asarray(v, np.float32)) for k, v in X.items()}
    ship_book = fit_address_book(ship_codes["store_train"], Y["store_train"], cards,
                                 float(c.address_smoothing))
    ship_req = {k: requests_from_labels(Y[k], "full", 0, int(seed)) for k in EVAL_SETS}
    ship_comp = {k: compose_codes(request_posterior(ship_book, ship_req[k])) for k in EVAL_SETS}
    ship_rel = well_relevance(ship_codes["SPREAD"], np.asarray(Y["SPREAD"], np.int64), list(cards))
    arms_out[REFERENCE_ARM] = {
        "arm": REFERENCE_ARM,
        "role": ("⭐ REFERENCE, not an arm: gen-read-compose's frozen store, loaded and "
                 "hash-asserted, so this spoke's re-measured ladder can be checked "
                 "against gen-loss-attribution's."),
        "store_bytes": store_bytes(shipped),
        "distinct": {
            "request_side": {k: {"n_distinct_code_configs":
                                 int(ship_comp[k]["n_distinct_code_configs"]),
                                 "n_requests": int(ship_req[k].shape[0]),
                                 "mean_confidence": float(ship_comp[k]["mean_confidence"])}
                             for k in EVAL_SETS},
            "feature_side": {k: {"n_distinct_codes": int(np.unique(ship_codes[k], axis=0).shape[0]),
                                 "n_rows": int(ship_codes[k].shape[0])} for k in EVAL_SETS},
        },
        "mi": {k: mi_table(ship_codes[k], Y[k], cards, names) for k in EVAL_SETS},
        "capacity": per_well_capacity(
            np.asarray(ship_rel["mi_bits"]), np.asarray(ship_rel["factor_entropy_bits"]),
            cards, names, M, float(c.capacity_carry_threshold)),
        "codes": ship_codes,
        "reads": {k: np.asarray(read_from_codes(shipped, ship_codes[k]), np.float32).reshape(
            (ship_codes[k].shape[0], -1)) for k in EVAL_SETS},
        "transform": {"kind": "none (the shipped, unwhitened store)",
                      "byte_ledger": transform_byte_ledger("identity", [len(w) for w in wc])},
    }
    for arm in list(c.arms):
        if str(arm) not in TRANSFORMS:
            raise ValueError(f"unknown arm {arm!r} — have {TRANSFORMS}")
        log(f"  arm {arm}")
        arms_out[str(arm)] = run_arm(config, str(arm), X, Y, rows, wc, feat_shape,
                                     cards, names, int(seed), log=log)

    # ---- the probes on every arm's code, ⛔ the SAME probe ------------------
    for arm, a in arms_out.items():
        points[f"code/{arm}"] = _fit(
            f"code/{arm}", onehot_codes(a["codes"]["SPREAD_train"], M),
            {k: onehot_codes(a["codes"][k], M) for k in EVAL_SETS})
    rnd = {k: random_codes(v.shape[0], int(shipped.n_wells), M,
                           int(c.random_code_seed) + 977 * int(seed) + 13 * i)
           for i, (k, v) in enumerate(
               {kk: X[kk] for kk in ("SPREAD_train",) + EVAL_SETS}.items())}
    points[CONTROL_POINT] = _fit(
        CONTROL_POINT, onehot_codes(rnd["SPREAD_train"], M),
        {k: onehot_codes(rnd[k], M) for k in EVAL_SETS})

    # ---- ⭐ LEG 3: the six-factor graded profile + the TOTAL-RETAINED line --
    chance = chance_rates(cards, names)
    majority = {k: majority_baseline(Y["SPREAD_train"], Y[k], names) for k in EVAL_SETS}
    tables, totals = {}, {}
    for es in EVAL_SETS:
        enc = points["encoder"]["eval"][es]["per_factor"]
        crn = points[CONTROL_POINT]["eval"][es]["per_factor"]
        tables[es] = {}
        totals[es] = {}
        for arm in arms_out:
            cod = points[f"code/{arm}"]["eval"][es]["per_factor"]
            ret = {nm: retention(cod[nm], enc[nm], chance[nm]) for nm in names}
            tables[es][arm] = per_factor_rows(
                points["encoder"]["eval"][es], NVR[es], names,
                acc_encoder={nm: enc[nm] for nm in names},
                acc_code={nm: cod[nm] for nm in names},
                acc_code_random={nm: crn[nm] for nm in names},
                chance={nm: chance[nm] for nm in names},
                majority={nm: majority[es][nm] for nm in names},
                retention_code=ret,
                mi_max_frac={nm: arms_out[arm]["mi"][es]["per_factor"][nm][
                    "max_fraction_of_entropy"] for nm in names},
            )
            live = [nm for nm in names if NVR[es][nm]["n_values_realised"] > 1]
            totals[es][arm] = {
                "total_retained": float(np.nansum([ret[nm] for nm in live])),
                "mean_retained": float(np.nanmean([ret[nm] for nm in live])),
                "n_factors_counted": len(live),
                "factors_counted": live,
                "note": ("⛔ degenerate factors (1 realised value, R31) are EXCLUDED "
                         "from the total; the count travels with it"),
            }
    rec["points"] = points
    rec["chance"] = chance
    rec["majority"] = majority
    rec["tables"] = tables
    rec["totals"] = totals
    rec["arms"] = {a: {k: v for k, v in d.items()
                       if k not in ("codes", "reads", "store", "store_unwhitened")}
                   for a, d in arms_out.items()}

    # ---- ⭐ the decoder (⛔ the standing trap), on the INVERTED read ---------
    if bool(c.decode_reads):
        rec["decoded"] = _decode_arms(config, arms_out, Y, names, int(seed), log=log)

    # ---- ⭐ the laundering control, re-taken AFTER everything ---------------
    if proto_before is not None:
        proto_after = knn_proto_fingerprint(grid, X, rows)
        rec["knn_proto_laundering"] = assert_knn_proto_identical(proto_before, proto_after)
        rec["knn_proto_laundering"]["prototype_table_bytes"] = int(
            proto_before["prototype_table_bytes"])
        log("  ⭐ knn_proto laundering control: "
            f"{rec['knn_proto_laundering']['verdict']} "
            f"(max |Δ| = {rec['knn_proto_laundering']['max_abs_delta']})")

    rec["wall_clock_s"] = float(time.time() - t0)
    rec["complete"] = True
    _write_json(rec, Path(out_dir) / f"veq_s{int(seed)}.json")
    return rec


def _decode_arms(config: CHLUConfig, arms_out: dict, Y: dict, names, seed: int,
                 log: Callable = print) -> dict:
    """⭐ Render each arm's read through ``gen-decoder-lift``'s **winner** and read it
    with the frozen image probe.

    ⛔ **The standing decoder trap:** ``GenBridgeConfig.dec_recipe`` still defaults to
    ``"baseline"`` — the recipe that FAILED the gate (0.7191, 0/3 seeds) — so the
    recipe name **and** the per-seed ``param_hash`` are both asserted before anything
    is rendered.  ⭐ Every arm's read has already been mapped **back** through the
    inverse, so the decoder sees its usual space.
    """
    c = _cfg(config)
    dpath = _abs(config, c.winner_decoder_pattern.format(seed=int(seed)))
    hpath = _abs(config, c.winner_hashes_path)
    if not dpath.exists() or not hpath.exists():
        return {"skipped": f"⛔ no winner decoder at {dpath} — the decoded column is "
                           "DECLARED ABSENT, never silently replaced by the baseline "
                           "recipe"}
    with open(hpath) as fh:
        hashes = json.load(fh)
    decoder = load_model(dpath)
    guard = assert_dream_decoder(decoder, hashes[f"seed_{int(seed)}"]["param_hash"],
                                str(c.dec_recipe), WINNING_DECODER_RECIPE)
    bdir = bridge_dir(config, c.dataset, c.pretext, int(seed))
    probe = load_model(bdir / "probe.pkl")
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("the image probe is not the one the bridge froze")
    n = int(c.decode_rows) or None
    out = {"guard": guard, "probe_param_hash": bman["probe_param_hash"], "arms": {}}
    for arm, a in arms_out.items():
        r = a["reads"]["SPREAD"]
        r = r[:n] if n else r
        imgs = decode_features(decoder, np.asarray(r, np.float32).reshape(
            (r.shape[0],) + tuple(decoder.feat_shape)), int(c.decode_batch))
        acc = probe_accuracy(probe, imgs, np.asarray(Y["SPREAD"][: r.shape[0]], np.int32),
                             names, int(c.decode_batch))
        out["arms"][arm] = acc
        log(f"    [decoded/{arm}] aggregate={acc['aggregate']:.4f} "
            + " ".join(f"{k}={v:.3f}" for k, v in acc["per_factor"].items()))
    return out


# ---------------------------------------------------------------------------
# 5. Across seeds
# ---------------------------------------------------------------------------


def summarize(records: list, config: CHLUConfig) -> dict:
    """3-seed roll-up + ⭐ the two pre-registered falsifiers, answered yes/no."""
    c = _cfg(config)
    names = records[0]["factor_names"]
    dead = [r for r in records if r["leg1_premise"]["stop_after_leg1"]]
    out = {
        "n_seeds": len(records), "seeds": [r["seed"] for r in records],
        "factor_names": names,
        "leg1": {
            "verdict_per_seed": [r["leg1_premise"]["verdict"] for r in records],
            "unanimous": len({r["leg1_premise"]["verdict"] for r in records}) == 1,
            "per_factor": {
                nm: {
                    "variance_explained": summarize_across_seeds([
                        r["leg1_spectra"][r["leg1_premise"]["headline_well_per_factor"][nm]]
                        ["per_factor"][nm]["variance_explained"] for r in records]),
                    "rank_frac_median": summarize_across_seeds([
                        r["leg1_spectra"][r["leg1_premise"]["headline_well_per_factor"][nm]]
                        ["per_factor"][nm]["rank_frac_median"] for r in records]),
                    "frac_signal_in_top10pct": summarize_across_seeds([
                        r["leg1_spectra"][r["leg1_premise"]["headline_well_per_factor"][nm]]
                        ["per_factor"][nm]["frac_signal_in_top10pct"] for r in records]),
                    "var_frac_at_median_rank": summarize_across_seeds([
                        r["leg1_spectra"][r["leg1_premise"]["headline_well_per_factor"][nm]]
                        ["per_factor"][nm]["var_frac_at_median_rank"] for r in records]),
                    "headline_well_per_seed": [
                        r["leg1_premise"]["headline_well_per_factor"][nm] for r in records],
                    "n_values_realised": sorted({
                        int(r["leg1_spectra"][r["leg1_premise"]["headline_well_per_factor"][nm]]
                            ["per_factor"][nm]["n_values_realised"]) for r in records}),
                } for nm in names
            },
        },
    }
    if dead:
        out["stopped"] = dead[0].get("stopped", "premise dead")
        return out

    arms = list(records[0]["tables"]["SPREAD"].keys())
    out["arms"] = arms
    out["by_eval_set"] = {}
    for es in EVAL_SETS:
        out["by_eval_set"][es] = {}
        for arm in arms:
            per_factor = {}
            for nm in names:
                rws = [r["tables"][es][arm][nm] for r in records]
                per_factor[nm] = {
                    "n_values_realised": sorted({int(x["n_values_realised"]) for x in rws}),
                    "degenerate": bool(all(x["degenerate"] for x in rws)),
                    "chance": float(rws[0]["chance"]),
                    "encoder": summarize_across_seeds([x["acc_encoder"] for x in rws]),
                    "code": summarize_across_seeds([x["acc_code"] for x in rws]),
                    "code_random": summarize_across_seeds([x["acc_code_random"] for x in rws]),
                    "retention_code": summarize_across_seeds([x["retention_code"] for x in rws]),
                    "mi_max_fraction_of_entropy": summarize_across_seeds(
                        [x["mi_max_frac"] for x in rws]),
                }
            out["by_eval_set"][es][arm] = {
                "per_factor": per_factor,
                "total_retained": summarize_across_seeds(
                    [r["totals"][es][arm]["total_retained"] for r in records]),
                "mean_retained": summarize_across_seeds(
                    [r["totals"][es][arm]["mean_retained"] for r in records]),
                "distinct_request_side": summarize_across_seeds(
                    [r["arms"][arm]["distinct"]["request_side"][es]["n_distinct_code_configs"]
                     for r in records]),
                "distinct_feature_side": summarize_across_seeds(
                    [r["arms"][arm]["distinct"]["feature_side"][es]["n_distinct_codes"]
                     for r in records]),
                "n_wells_over_capacity": summarize_across_seeds(
                    [r["arms"][arm]["capacity"]["n_wells_over_capacity"] for r in records]),
            }

    base, prim = "identity", "pca_whiten"
    out["falsifiers"] = _falsifiers(out, records, names, base, prim, c)
    out["byte_ledger"] = {
        arm: records[0]["arms"][arm]["transform"]["byte_ledger"] for arm in arms
    }
    out["round_trip"] = {
        arm: summarize_across_seeds(
            [rr["arms"][arm]["transform"]["round_trip"]["max_abs_error"] for rr in records])
        for arm in arms
        if all("round_trip" in rr["arms"][arm]["transform"] for rr in records)
    }
    return out


def _falsifiers(out: dict, records: list, names, base: str, prim: str, c) -> dict:
    """⭐ PREREG §7, applied mechanically and answered in the pre-registered words."""
    es = "SPREAD"
    if base not in out["by_eval_set"][es] or prim not in out["by_eval_set"][es]:
        return {"skipped": f"need both {base!r} and {prim!r} arms"}
    A, B = out["by_eval_set"][es][base], out["by_eval_set"][es][prim]
    per = {}
    for nm in list(c.premise_probe_factors):
        a, b = A["per_factor"][nm]["retention_code"], B["per_factor"][nm]["retention_code"]
        spread = max(float(a["std"]), float(b["std"]))
        per[nm] = {
            "retention_unwhitened": a, "retention_whitened": b,
            "delta": float(b["mean"] - a["mean"]), "seed_spread_used": spread,
            "rose_beyond_spread": bool(float(b["mean"] - a["mean"]) > spread),
        }
    a_rose = all(v["rose_beyond_spread"] for v in per.values())
    da, db = A["distinct_request_side"], B["distinct_request_side"]
    dspread = max(float(da["std"]), float(db["std"]))
    b_rose = bool(float(db["mean"] - da["mean"]) > dspread)
    fa, fb = A["distinct_feature_side"], B["distinct_feature_side"]
    total = capacity_reading(float(A["total_retained"]["mean"]), float(B["total_retained"]["mean"]),
                             max(float(A["total_retained"]["std"]),
                                 float(B["total_retained"]["std"])))
    return {
        "falsifier_a_probe_rises": {
            "answer": "YES" if a_rose else "NO",
            "rule": ("PREREG §7(a): retention_code(pca_whiten) − retention_code(identity) "
                     "exceeds the larger of the two arms' 3-seed population sds for BOTH "
                     "scale and orientation"),
            "per_factor": per,
        },
        "falsifier_b_distinct_rises": {
            "answer": "YES" if b_rose else "NO",
            "primary": "request-side distinct code configurations on CORNER_B",
            "unwhitened": da, "whitened": db, "seed_spread_used": dspread,
            "delta": float(db["mean"] - da["mean"]),
            "secondary_feature_side_on_SPREAD": {"unwhitened": fa, "whitened": fb,
                                                 "delta": float(fb["mean"] - fa["mean"]),
                                                 "note": ("⚠ already near-injective on the "
                                                          "grid, so little headroom exists "
                                                          "here by construction")},
        },
        "pre_registered_pass": {
            "answer": "YES" if (a_rose and b_rose) else "NO",
            "rule": ("⭐ the pass the task demands is BOTH. ⛔ One without the other is "
                     "NOT the pre-registered pass, and the two measure different things."),
        },
        "total_retained_reading": total,
        "totals": {"unwhitened": A["total_retained"], "whitened": B["total_retained"]},
    }


def make_plot(summary: dict, path: Path) -> Path:
    """Leg 1's spectrum row beside Leg 3's before/after retention ladder."""
    names = list(summary["factor_names"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    v = [summary["leg1"]["per_factor"][nm]["variance_explained"]["mean"] for nm in names]
    r = [summary["leg1"]["per_factor"][nm]["rank_frac_median"]["mean"] for nm in names]
    ax.scatter(v, r)
    for nm, x, y in zip(names, v, r, strict=True):
        ax.annotate(nm, (x, y), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("variance explained by the factor (best well)")
    ax.set_ylabel("median signal rank / d_k")
    ax.set_title("Leg 1 — where in the spectrum the signal sits")
    if "by_eval_set" in summary:
        es = summary["by_eval_set"]["SPREAD"]
        arms = [a for a in summary["arms"]]
        w = 0.8 / max(1, len(arms))
        for i, arm in enumerate(arms):
            vals = [es[arm]["per_factor"][nm]["retention_code"]["mean"] for nm in names]
            errs = [es[arm]["per_factor"][nm]["retention_code"]["std"] for nm in names]
            axes[1].bar(np.arange(len(names)) + i * w, vals, w, yerr=errs, label=arm, capsize=2)
        axes[1].set_xticks(np.arange(len(names)) + 0.4)
        axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        axes[1].set_ylabel("retention of above-chance signal")
        axes[1].set_title("Leg 3 — the graded profile, before and after")
        axes[1].legend(fontsize=7)
        tot = [es[a]["total_retained"]["mean"] for a in arms]
        terr = [es[a]["total_retained"]["std"] for a in arms]
        axes[2].bar(np.arange(len(arms)), tot, 0.6, yerr=terr, capsize=3)
        axes[2].set_xticks(np.arange(len(arms)))
        axes[2].set_xticklabels(arms, rotation=20, ha="right", fontsize=7)
        axes[2].set_ylabel("TOTAL retained (6 factors)")
        axes[2].set_title("trade vs reduction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_experiment_gen_variance_eq(config: Optional[CHLUConfig] = None,
                                   out_dir: Optional[str] = None,
                                   log: Callable = print) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else veq_out_root(config)
    runs = root / "runs" / f"{c.dataset}_{c.pretext}"
    runs.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in list(c.seeds):
        path = runs / f"veq_s{int(seed)}.json"
        if path.exists() and bool(c.skip_if_exists):
            with open(path) as fh:
                rec = json.load(fh)
            if rec.get("complete"):
                log(f"  seed {seed}: artefact HIT {path.name} — skipped")
                records.append(rec)
                continue
        log(f"  seed {seed}: running")
        records.append(run_seed(config, c.dataset, c.pretext, int(seed), runs, log=log))
    summary = summarize(records, config)
    _write_json(summary, root / "summary.json")
    if bool(c.make_plots):
        make_plot(summary, root / "variance_eq.png")
    log("⭐ LEG 1 premise: " + " / ".join(summary["leg1"]["verdict_per_seed"]))
    if "falsifiers" in summary:
        f = summary["falsifiers"]
        log(f"⭐ falsifier (a) probe rises: {f['falsifier_a_probe_rises']['answer']} · "
            f"(b) distinct rises: {f['falsifier_b_distinct_rises']['answer']} · "
            f"pre-registered pass: {f['pre_registered_pass']['answer']}")
        log("⭐ " + f["total_retained_reading"]["reading"])
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS) + [None])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=None, choices=list(TRANSFORMS) + [None])
    p.add_argument("--out", default=None)
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--no-scale-match", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)
    config = get_default_config()
    c = _cfg(config)
    if args.dataset:
        c.dataset = args.dataset
    if args.pretext:
        c.pretext = args.pretext
    if args.seeds:
        c.seeds = list(args.seeds)
    if args.arms:
        c.arms = list(args.arms)
    if args.no_decode:
        c.decode_reads = False
    if args.no_scale_match:
        c.scale_match_to_raw = False
    if args.quick:
        apply_quick(config)
    run_experiment_gen_variance_eq(config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
