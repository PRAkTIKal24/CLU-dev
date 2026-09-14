"""Experiment: ⭐⭐ **spend codewords where the distortion is** — D6, at CONSTANT
TOTAL BYTES.

⭐ **DIAL DECLARATION — *codeword allocation across wells, at constant total bytes*.**
- **Dial:** how many codewords each well gets.  ⛔ The encoder, the decoder, the
  **partition**, ``K``, the store recipe, the row draw and the probe are all
  UNCHANGED.
- **Laundering control:** ⭐ **uniform ``M = 64`` at the same total bytes — the
  shipped arm**, re-fitted through the identical code path.  ⛔ Any gain must be over
  it.  A second control rides along: a **random-code probe at each arm's own
  ``M_k`` widths**, because the one-hot probe width ``Σ_k M_k`` is arm-dependent.
- **Falsifies:** the scaling law's **actionability**.
  ⛔ **This is a CONTROL ARM for the dynamical read, NOT a gate on it.**

**The law under test** (`gen-variance-equalisation` §4): ``ε ≈ M^(−2/PR)`` with
per-well ``PR = 3.12 … 13.63`` measured.  Equal-distortion allocation sets
``M_k ∝ ε^(−PR_k/2)``.  ⭐ Constant total bytes ⇒ **no byte-ledger renegotiation**,
which is the whole appeal.

⛔ **It CANNOT save the high-``PR`` wells.**  At ε = 0.10 a ``PR = 13.63`` well needs
~6.5 **million** codewords (:func:`chlu.core.gen_alloc.infeasibility_table`, printed
beside every result).  ⛔ A partial fix is not a fix.

**Three allocations, one budget:** ``uniform`` (control) · ``equal_distortion`` (the
rule D6 names) · ``mse_optimal`` (the byte-constrained minimiser of the read error —
⭐ if it ties ``uniform``, re-allocation is not a lever).

⚠ **Three environment facts** (paid for by R31/R34/`gen-loss-attribution`): a
worktree has no ``.claude/``, so ``$CHLU_GEN_DATA_ROOT``, ``$CHLU_CLAUDE_ROOT``
**and** ``$CHLU_GEN_BRIDGE_ROOT`` must all be set, or the run dies at ``bridge_dir``.

Runnable::

    python -m chlu.experiments.exp_gen_codebook_alloc --quick   # ⚠ plumbing smoke only
    python -m chlu.experiments.exp_gen_codebook_alloc
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, gen_codebook_alloc_config, get_default_config  # noqa: E402
from ..core.gen_alloc import (  # noqa: E402
    ALLOCATIONS,
    allocation_ledger,
    assert_bytes_not_exceeded,
    build_allocation,
    infeasibility_table,
    law_fit,
    mask_posterior_to_allocation,
    mode_usage_per_well,
    onehot_codes_per_well,
    per_well_distortion,
    random_codes_per_well,
    well_spectra,
)
from ..core.gen_bridge import PRETEXTS, load_feature_cache, param_hash  # noqa: E402
from ..core.gen_compose import compose_codes, request_posterior  # noqa: E402
from ..core.gen_dream import (  # noqa: E402
    WINNING_DECODER_RECIPE,
    assert_dream_decoder,
    fit_address_book,
    requests_from_labels,
)
from ..core.gen_store import (  # noqa: E402
    address_codes,
    init_store,
    read_from_codes,
    store_bytes,
    train_store,
    well_channels_from_labels,
    well_relevance,
)
from ..core.gen_whiten import assert_grouping_is_unwhitened, per_well_capacity  # noqa: E402
from ..eval.gen_attribution import (  # noqa: E402
    chance_rates,
    majority_baseline,
    mi_table,
    n_values_realised,
    per_factor_rows,
    probe_point,
    retention,
    summarize_across_seeds,
)
from ..eval.gen_probe import decode_features, factor_labels, probe_accuracy  # noqa: E402
from ..experiments.exp_gen_attribution import load_attrib_store, load_grouping  # noqa: E402
from ..experiments.exp_gen_compose import _claude_root, _write_json  # noqa: E402
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    bridge_dir,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..experiments.exp_gen_variance_eq import (  # noqa: E402
    assert_knn_proto_identical,
    knn_proto_fingerprint,
)
from ..utils.checkpoints import load_model  # noqa: E402

#: The row sets the probe is SCORED on.  ⭐ ``SPREAD`` is the headline; the corner is
#: reported separately and ⛔ never pooled into it.
EVAL_SETS = ("SPREAD", "CORNER_B", "CORNER_MS")
#: ⭐ The reference row: `gen-read-compose`'s frozen store, so this spoke's
#: re-measured ladder can be checked against `gen-loss-attribution`'s.
REFERENCE_ARM = "shipped"
#: ⭐ The laundering control the task names.  ⛔ Any gain must be over THIS arm.
CONTROL_ARM = "uniform"


def _cfg(config: CHLUConfig):
    return gen_codebook_alloc_config(config)


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
    c.decode_rows = int(c.quick_decode_rows)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    config.gen_store.train_steps = int(c.quick_store_train_steps)
    return config


def alloc_out_root(config: CHLUConfig) -> Path:
    c = _cfg(config)
    if c.out_root:
        return Path(c.out_root).expanduser()
    return _claude_root(config) / "outputs" / "gen-codebook-allocation"


def _abs(config: CHLUConfig, p: str) -> Path:
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else _claude_root(config) / q


# ---------------------------------------------------------------------------
# 1. Rows — ⛔ identical draw to gen-loss-attribution's, so the probes compare
# ---------------------------------------------------------------------------


def select_rows(config: CHLUConfig, dataset: str, split, values, seed: int) -> dict:
    """``store_train`` + the probe's row sets, with `exp_gen_attribution`'s own RNG
    stream — ⛔ so a retention measured here is comparable to the one measured there.
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
# 2. One arm: one allocation, one codebook, one read
# ---------------------------------------------------------------------------


def run_arm(
    config: CHLUConfig,
    arm: str,
    alloc: np.ndarray,
    led: dict,
    X: dict,
    Y: dict,
    well_channels,
    feat_shape,
    cards,
    names,
    seed: int,
    log: Callable = print,
) -> dict:
    """⭐ One allocation, everything else held fixed.

    ⛔ The key, the steps, the rows, ``K`` and the **partition** are identical across
    arms by construction: the only difference is ``M_k``.
    """
    c = _cfg(config)
    t0 = time.time()
    M = [int(x) for x in alloc]
    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911)
    key, ik, tk = jax.random.split(key, 3)
    store0 = init_store(X["store_train"], well_channels, M, feat_shape, ik, seed=int(seed))
    store, hist, key = train_store(store0, X["store_train"], config.gen_store, tk, log=None)
    if list(store.n_modes_per_well) != M:
        raise AssertionError(
            f"⛔ arm {arm}: the fitted store holds {store.n_modes_per_well} modes per "
            f"well, not the allocated {M} — the allocation did not reach the store")
    sb = store_bytes(store)
    if int(sb["store_bytes"]) != int(led["store_bytes"]):
        raise AssertionError(
            f"⛔ arm {arm}: the fitted store is {sb['store_bytes']:,} B but the ledger "
            f"says {led['store_bytes']:,} B — the byte gate was applied to a different "
            "object than the one that was measured")

    codes = {k: address_codes(store, np.asarray(v, np.float32)) for k, v in X.items()}
    reads = {k: np.asarray(read_from_codes(store, codes[k]), np.float32).reshape(
        (codes[k].shape[0], -1)) for k in EVAL_SETS}

    # ⭐ Under-training is a live alternative explanation for any arm that loses, so
    # the k-means++ INIT's hard read is measured beside the trained one.
    mse = {}
    for k in EVAL_SETS:
        Xk = np.asarray(X[k], np.float32).reshape((np.asarray(X[k]).shape[0], -1))
        err = float(np.mean((reads[k] - Xk) ** 2))
        var = float(np.mean((Xk - Xk.mean(axis=0, keepdims=True)) ** 2))
        mse[k] = {"mse": err, "feature_variance": var,
                  "relative_mse": err / var if var > 0 else float("nan"),
                  "n": int(Xk.shape[0])}
    c0 = address_codes(store0, np.asarray(X["SPREAD"], np.float32))
    r0 = np.asarray(read_from_codes(store0, c0), np.float32).reshape((c0.shape[0], -1))
    X0 = np.asarray(X["SPREAD"], np.float32).reshape((c0.shape[0], -1))
    mse["SPREAD_at_kmeanspp_init"] = {"mse": float(np.mean((r0 - X0) ** 2))}
    mse["training_improved_the_hard_read"] = bool(
        mse["SPREAD"]["mse"] < mse["SPREAD_at_kmeanspp_init"]["mse"])

    # ⭐⭐ THE MEASURED PER-WELL DISTORTION — the quantity the law predicts.
    dist = {k: per_well_distortion(store, X[k], codes[k]) for k in ("store_train", "SPREAD")}

    # ---- the request side: distinct configurations on the 72 held-out requests
    # ⛔ fit_address_book builds a RECTANGULAR (K, max M_k) table with Laplace
    # smoothing, so a short well gets non-zero mass on modes it does not own and
    # argmax could emit an OUT-OF-RANGE code. The posterior is masked first.
    book = fit_address_book(codes["store_train"], Y["store_train"], cards,
                            float(c.address_smoothing))
    req = {k: requests_from_labels(Y[k], "full", 0, int(seed)) for k in EVAL_SETS}
    post = {k: mask_posterior_to_allocation(request_posterior(book, req[k]), M)
            for k in EVAL_SETS}
    comp = {k: compose_codes(post[k]) for k in EVAL_SETS}
    for k in EVAL_SETS:
        hi = comp[k]["code"].max(axis=0) if comp[k]["code"].size else np.zeros(len(M))
        if np.any(np.asarray(hi) >= np.asarray(M)):
            raise AssertionError(
                f"⛔ arm {arm}/{k}: composed code {list(hi)} exceeds the allocation {M} "
                "— the address-book mask failed and read_from_codes would gather a "
                "wrapped index")
    distinct = {
        "request_side": {k: {"n_distinct_code_configs": int(comp[k]["n_distinct_code_configs"]),
                             "n_requests": int(req[k].shape[0]),
                             "mean_confidence": float(comp[k]["mean_confidence"])}
                         for k in EVAL_SETS},
        "feature_side": {k: {"n_distinct_codes": int(np.unique(codes[k], axis=0).shape[0]),
                             "n_rows": int(codes[k].shape[0])}
                         for k in EVAL_SETS} if bool(c.distinct_on_spread) else {},
    }
    rel = well_relevance(codes["SPREAD"], np.asarray(Y["SPREAD"], np.int64), list(cards))
    cap = per_well_capacity(
        np.asarray(rel["mi_bits"]), np.asarray(rel["factor_entropy_bits"]),
        cards, names, M, float(c.capacity_carry_threshold))
    usage = mode_usage_per_well(codes["SPREAD"], M)
    log(f"    [{arm}] M={M} · {sb['store_bytes']:,} B "
        f"({led['bytes_delta']:+,} B vs uniform) · {led['bits_per_read']:.2f} bits/read · "
        f"measured eps={dist['SPREAD']['variance_weighted_relative_distortion']:.4f} "
        f"(predicted {led['predicted_variance_weighted_eps']:.4f}) · "
        f"distinct-of-{req['CORNER_B'].shape[0]}="
        f"{comp['CORNER_B']['n_distinct_code_configs']} · "
        f"used {usage['total_used_modes']}/{usage['total_allocated_modes']} modes")
    return {
        "arm": str(arm),
        "allocation": led,
        "store_bytes": sb,
        "train_loss_first": float(hist[0]),
        "train_loss_last": float(hist[-1]),
        "reconstruction_mse": mse,
        "distortion": dist,
        "distinct": distinct,
        "mi": {k: mi_table(codes[k], Y[k], cards, names) for k in EVAL_SETS},
        "capacity": cap,
        "mode_usage": usage,
        "codes": codes,
        "reads": reads,
        "store": store,
        "wall_clock_s": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 3. One seed
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

    # ⛔ The partition: loaded from gen-commitment-collapse and asserted
    # channel-for-channel against the SHIPPED store.  ⛔ Never re-derived here.
    labels_vec = load_grouping(config, dataset, pretext, int(seed))
    shipped, sinfo = load_attrib_store(config, dataset, pretext, int(seed), labels_vec, log)
    wc = well_channels_from_labels(labels_vec, int(shipped.n_wells))
    order = assert_grouping_is_unwhitened(wc, shipped.well_channels, "shipped store")
    feat_shape = tuple(shipped.feat_shape)
    M_uniform = int(c.m_uniform)
    if int(shipped.n_modes) != M_uniform:
        raise AssertionError(
            f"⛔ the shipped store holds M={shipped.n_modes}, but the byte budget is "
            f"being taken at M={M_uniform} — the control would not be the shipped arm")
    log(f"  ⭐ grouping ORDER asserted: {order['grouping_fitted_on']} · "
        f"well sizes {order['well_sizes']}")

    # ⭐ THE LAUNDERING CONTROL, taken BEFORE anything is fitted.
    proto_before = (knn_proto_fingerprint(grid, X, rows)
                    if bool(c.assert_knn_proto_identical) else None)

    # ---- ⭐⭐ THE MEASURED INPUT: per-well effective dimension, LABEL-FREE ----
    ns = min(int(c.spectrum_rows) or X["store_train"].shape[0], X["store_train"].shape[0])
    # ⛔ A PREFIX of store_train is NOT a subsample (veq §12 L1): the Shapes3D grid is
    # a lexicographic bijection and the rows are sorted.  Seeded permutation only.
    sel = np.sort(np.random.default_rng(
        int(seed) * int(c.row_seed_mult) + int(c.row_seed_offset) + 101
    ).permutation(X["store_train"].shape[0])[:ns])
    spectra = well_spectra(X["store_train"][sel], wc)
    PR = [float(s["participation_ratio"]) for s in spectra]
    DK = [int(s["d_k"]) for s in spectra]
    VAR = [float(s["total_variance"]) for s in spectra]
    log(f"  ⭐ label-free PR = {[round(p, 3) for p in PR]} · d_k = {DK} "
        f"(n={ns} rows, ⛔ a seeded permutation)")

    # ---- ⭐ THE ALLOCATIONS, and the byte gate ------------------------------
    allocs, ledgers = {}, []
    for arm in list(c.arms):
        if str(arm) not in ALLOCATIONS:
            raise ValueError(f"unknown allocation {arm!r} — have {ALLOCATIONS}")
        a = build_allocation(str(arm), PR, DK, VAR, M_uniform, int(c.m_min), int(c.m_max))
        allocs[str(arm)] = a
        ledgers.append(allocation_ledger(str(arm), a, DK, PR, VAR, M_uniform))
    gate = assert_bytes_not_exceeded(ledgers, CONTROL_ARM) if bool(
        c.assert_bytes_within_budget) else {"skipped": "byte gate DISABLED"}
    log(f"  ⭐ BYTE GATE: {gate.get('verdict', gate)} · reference "
        f"{gate.get('reference_bytes', 0):,} B · max shortfall "
        f"{100.0 * gate.get('max_shortfall_frac', float('nan')):.3f} %")
    by_name = {x["allocation"]: x for x in ledgers}

    rec = {
        "schema": "chlu-gen-codebook-alloc/1",
        "seed": int(seed),
        "dataset": dataset,
        "pretext": pretext,
        "dial": ("codeword allocation across wells at CONSTANT TOTAL BYTES — ⛔ "
                 "encoder/decoder/partition/K/store-recipe/probe unchanged"),
        "control_arm": CONTROL_ARM,
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
        "spectrum": {"n_rows": int(ns), "per_well": spectra,
                     "note": "⛔ LABEL-FREE; a seeded permutation of store_train"},
        "allocations": {x["allocation"]: x for x in ledgers},
        "byte_gate": gate,
        "infeasibility": infeasibility_table(PR, DK),
        "wall_clock_s": float(time.time() - t0),
    }

    # ---- ⛔ THE ONE PROBE — one key, the encoder point shared by every arm ---
    key = jax.random.PRNGKey(int(c.probe_seed_offset) + int(seed))

    def _fit(name, Ztr, Zev):
        return probe_point(name, Ztr, Y["SPREAD_train"],
                           {k: (Zev[k], Y[k]) for k in EVAL_SETS},
                           cards, names, c, key, log=log)

    points = {"encoder": _fit("encoder", X["SPREAD_train"], {k: X[k] for k in EVAL_SETS})}

    # ---- the reference row: the SHIPPED store, re-measured here -------------
    arms_out = {}
    ship_codes = {k: address_codes(shipped, np.asarray(v, np.float32)) for k, v in X.items()}
    ship_M = [int(x) for x in shipped.n_modes_per_well]
    ship_book = fit_address_book(ship_codes["store_train"], Y["store_train"], cards,
                                 float(c.address_smoothing))
    ship_req = {k: requests_from_labels(Y[k], "full", 0, int(seed)) for k in EVAL_SETS}
    ship_comp = {k: compose_codes(request_posterior(ship_book, ship_req[k])) for k in EVAL_SETS}
    ship_rel = well_relevance(ship_codes["SPREAD"], np.asarray(Y["SPREAD"], np.int64),
                              list(cards))
    arms_out[REFERENCE_ARM] = {
        "arm": REFERENCE_ARM,
        "role": ("⭐ REFERENCE, not an arm: gen-read-compose's frozen store, loaded and "
                 "hash-asserted, so this spoke's re-measured ladder can be checked "
                 "against gen-loss-attribution's."),
        "allocation": allocation_ledger(REFERENCE_ARM, ship_M, DK, PR, VAR, M_uniform),
        "store_bytes": store_bytes(shipped),
        "distortion": {k: per_well_distortion(shipped, X[k], ship_codes[k])
                       for k in ("store_train", "SPREAD")},
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
            cards, names, ship_M, float(c.capacity_carry_threshold)),
        "mode_usage": mode_usage_per_well(ship_codes["SPREAD"], ship_M),
        "codes": ship_codes,
        "reads": {k: np.asarray(read_from_codes(shipped, ship_codes[k]), np.float32).reshape(
            (ship_codes[k].shape[0], -1)) for k in EVAL_SETS},
    }
    for arm in list(c.arms):
        log(f"  arm {arm}")
        arms_out[str(arm)] = run_arm(config, str(arm), allocs[str(arm)], by_name[str(arm)],
                                     X, Y, wc, feat_shape, cards, names, int(seed), log=log)

    # ---- the probes on every arm's code, ⛔ the SAME probe ------------------
    # ⚠ The one-hot width is `sum_k M_k` and is therefore ARM-DEPENDENT. That is a
    # probe-capacity confound in BOTH directions, which is why each arm also gets a
    # random-code control at its OWN widths rather than one shared uniform control.
    for arm, a in arms_out.items():
        Ma = [int(x) for x in a["allocation"]["n_modes_per_well"]]
        points[f"code/{arm}"] = _fit(
            f"code/{arm}", onehot_codes_per_well(a["codes"]["SPREAD_train"], Ma),
            {k: onehot_codes_per_well(a["codes"][k], Ma) for k in EVAL_SETS})
        if bool(c.per_arm_random_control):
            rnd = {k: random_codes_per_well(
                int(np.asarray(X[k]).shape[0]), Ma,
                int(c.random_code_seed) + 977 * int(seed) + 13 * i)
                for i, k in enumerate(("SPREAD_train",) + EVAL_SETS)}
            points[f"random/{arm}"] = _fit(
                f"random/{arm}", onehot_codes_per_well(rnd["SPREAD_train"], Ma),
                {k: onehot_codes_per_well(rnd[k], Ma) for k in EVAL_SETS})

    # ---- ⭐ THE SIX-FACTOR LADDER + the TOTAL-RETAINED line ------------------
    chance = chance_rates(cards, names)
    majority = {k: majority_baseline(Y["SPREAD_train"], Y[k], names) for k in EVAL_SETS}
    tables, totals = {}, {}
    for es in EVAL_SETS:
        enc = points["encoder"]["eval"][es]["per_factor"]
        tables[es], totals[es] = {}, {}
        for arm in arms_out:
            cod = points[f"code/{arm}"]["eval"][es]["per_factor"]
            crn = (points[f"random/{arm}"]["eval"][es]["per_factor"]
                   if f"random/{arm}" in points else {nm: float("nan") for nm in names})
            ret = {nm: retention(cod[nm], enc[nm], chance[nm]) for nm in names}
            rret = {nm: retention(crn[nm], enc[nm], chance[nm]) for nm in names}
            tables[es][arm] = per_factor_rows(
                points["encoder"]["eval"][es], NVR[es], names,
                acc_encoder={nm: enc[nm] for nm in names},
                acc_code={nm: cod[nm] for nm in names},
                acc_code_random={nm: crn[nm] for nm in names},
                chance={nm: chance[nm] for nm in names},
                majority={nm: majority[es][nm] for nm in names},
                retention_code=ret,
                retention_code_random=rret,
                mi_max_frac={nm: arms_out[arm]["mi"][es]["per_factor"][nm][
                    "max_fraction_of_entropy"] for nm in names},
            )
            live = [nm for nm in names if NVR[es][nm]["n_values_realised"] > 1]
            totals[es][arm] = {
                "total_retained": float(np.nansum([ret[nm] for nm in live])),
                "total_retained_random_control": float(np.nansum([rret[nm] for nm in live])),
                "mean_retained": float(np.nanmean([ret[nm] for nm in live])),
                "n_factors_counted": len(live),
                "factors_counted": live,
                "onehot_width": int(points[f"code/{arm}"]["input_dim"]),
                "note": ("⛔ degenerate factors (1 realised value, R31) are EXCLUDED "
                         "from the total; the count travels with it"),
            }
    rec["points"] = points
    rec["chance"] = chance
    rec["majority"] = majority
    rec["tables"] = tables
    rec["totals"] = totals

    # ---- ⭐⭐ THE LAW'S FIT — acceptance 5, "report the fit, not just the win"
    pred, meas, pts = [], [], []
    for arm, a in arms_out.items():
        Ma = a["allocation"]["n_modes_per_well"]
        ep = a["allocation"]["predicted_eps_per_well"]
        for w in a["distortion"]["SPREAD"]["per_well"]:
            j = int(w["well"])
            pred.append(float(ep[j]))
            meas.append(float(w["relative_distortion"]))
            pts.append({"arm": arm, "well": j, "n_modes": int(Ma[j]),
                        "participation_ratio": float(PR[j]), "d_k": int(DK[j]),
                        "predicted_eps": float(ep[j]),
                        "measured_eps": float(w["relative_distortion"]),
                        "ratio": float(w["relative_distortion"] / ep[j]) if ep[j] > 0
                        else float("nan")})
    rec["law"] = {"points": pts, "fit": law_fit(pred, meas),
                  "note": ("⭐ pooled over every arm x well on the SPREAD rows; the "
                           "aggregate law was already confirmed by "
                           "gen-variance-equalisation, the PER-WELL law is the "
                           "untested part")}
    rec["arms"] = {a: {k: v for k, v in d.items() if k not in ("codes", "reads", "store")}
                   for a, d in arms_out.items()}

    # ---- ⭐ the decoder (⛔ the standing trap) ------------------------------
    if bool(c.decode_reads):
        rec["decoded"] = _decode_arms(config, arms_out, Y, names, int(seed), log=log)

    # ---- ⭐ the laundering control, re-taken AFTER everything ---------------
    if proto_before is not None:
        rec["knn_proto_laundering"] = assert_knn_proto_identical(
            proto_before, knn_proto_fingerprint(grid, X, rows))
        log("  ⭐ knn_proto laundering control: "
            f"{rec['knn_proto_laundering']['verdict']} "
            f"(max |Δ| = {rec['knn_proto_laundering']['max_abs_delta']})")

    rec["wall_clock_s"] = float(time.time() - t0)
    rec["complete"] = True
    _write_json(rec, Path(out_dir) / f"alloc_s{int(seed)}.json")
    return rec


def _decode_arms(config: CHLUConfig, arms_out: dict, Y: dict, names, seed: int,
                 log: Callable = print) -> dict:
    """⭐ Render each arm's read through `gen-decoder-lift`'s **winner**.

    ⛔ **The standing decoder trap:** ``GenBridgeConfig.dec_recipe`` still defaults to
    ``"baseline"`` — the recipe that FAILED the gate — so the recipe name **and** the
    per-seed ``param_hash`` are both asserted before anything is rendered.
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
# 4. Across seeds
# ---------------------------------------------------------------------------


def summarize(records: list, config: CHLUConfig) -> dict:
    """3-seed roll-up + ⭐ the pre-registered pass, answered in its own words."""
    names = records[0]["factor_names"]
    arms = list(records[0]["arms"].keys())
    out = {
        "n_seeds": len(records), "seeds": [r["seed"] for r in records],
        "factor_names": names, "arms": arms, "control_arm": CONTROL_ARM,
        "dial": records[0]["dial"],
        "n_corner_b_requests": int(records[0]["rows"]["CORNER_B"]),
    }
    # ---- the allocation + byte ledger, per seed (⛔ never averaged) ---------
    out["allocation"] = {
        arm: {
            "n_modes_per_well_per_seed": [r["allocations"][arm]["n_modes_per_well"]
                                          for r in records if arm in r["allocations"]],
            "store_bytes_per_seed": [r["allocations"][arm]["store_bytes"]
                                     for r in records if arm in r["allocations"]],
            "bytes_delta_vs_uniform_per_seed": [r["allocations"][arm]["bytes_delta"]
                                                for r in records if arm in r["allocations"]],
            "bits_per_read": summarize_across_seeds(
                [r["allocations"][arm]["bits_per_read"] for r in records
                 if arm in r["allocations"]]),
            "onehot_width_per_seed": [r["allocations"][arm]["onehot_width"]
                                      for r in records if arm in r["allocations"]],
            "predicted_variance_weighted_eps": summarize_across_seeds(
                [r["allocations"][arm]["predicted_variance_weighted_eps"] for r in records
                 if arm in r["allocations"]]),
        }
        for arm in records[0]["allocations"]
    }
    out["byte_gate"] = {
        "per_seed": [r["byte_gate"] for r in records],
        "max_shortfall_frac": max(float(r["byte_gate"].get("max_shortfall_frac", 0.0))
                                  for r in records),
        "any_arm_over_budget": any(
            not v["within_budget"] for r in records
            for v in r["byte_gate"].get("per_arm", {}).values()),
    }
    out["participation_ratio_per_seed"] = [
        [w["participation_ratio"] for w in r["spectrum"]["per_well"]] for r in records]

    # ---- the ladder ---------------------------------------------------------
    out["by_eval_set"] = {}
    for es in EVAL_SETS:
        out["by_eval_set"][es] = {}
        for arm in arms:
            rows = {}
            for nm in names:
                rows[nm] = {
                    "n_values_realised": summarize_across_seeds(
                        [r["tables"][es][arm][nm]["n_values_realised"] for r in records]),
                    "accuracy_code": summarize_across_seeds(
                        [r["tables"][es][arm][nm]["acc_code"] for r in records]),
                    "accuracy_encoder": summarize_across_seeds(
                        [r["tables"][es][arm][nm]["acc_encoder"] for r in records]),
                    "retention_code": summarize_across_seeds(
                        [r["tables"][es][arm][nm]["retention_code"] for r in records]),
                    "retention_code_random": summarize_across_seeds(
                        [r["tables"][es][arm][nm]["retention_code_random"] for r in records]),
                    "chance": float(records[0]["tables"][es][arm][nm]["chance"]),
                }
            out["by_eval_set"][es][arm] = {
                "per_factor": rows,
                "total_retained": summarize_across_seeds(
                    [r["totals"][es][arm]["total_retained"] for r in records]),
                "total_retained_random_control": summarize_across_seeds(
                    [r["totals"][es][arm]["total_retained_random_control"] for r in records]),
                "onehot_width_per_seed": [r["totals"][es][arm]["onehot_width"]
                                          for r in records],
                "distinct_request_side": summarize_across_seeds(
                    [r["arms"][arm]["distinct"]["request_side"][es]["n_distinct_code_configs"]
                     for r in records]),
                "distinct_feature_side": summarize_across_seeds(
                    [r["arms"][arm]["distinct"]["feature_side"][es]["n_distinct_codes"]
                     for r in records if r["arms"][arm]["distinct"].get("feature_side")]),
            }

    # ---- ⭐⭐ the law --------------------------------------------------------
    pred = [p["predicted_eps"] for r in records for p in r["law"]["points"]]
    meas = [p["measured_eps"] for r in records for p in r["law"]["points"]]
    out["law"] = {
        "pooled_fit": law_fit(pred, meas),
        "n_points": len(pred),
        "per_arm_fit": {
            arm: law_fit([p["predicted_eps"] for r in records
                          for p in r["law"]["points"] if p["arm"] == arm],
                         [p["measured_eps"] for r in records
                          for p in r["law"]["points"] if p["arm"] == arm])
            for arm in arms},
        "measured_variance_weighted_eps": {
            arm: summarize_across_seeds(
                [r["arms"][arm]["distortion"]["SPREAD"][
                    "variance_weighted_relative_distortion"] for r in records])
            for arm in arms},
        "points": [p for r in records for p in r["law"]["points"]],
    }
    out["law"]["within_well_M_exponent"] = within_well_m_exponent(records)
    out["infeasibility"] = records[0]["infeasibility"]

    # ---- ⭐ THE PRE-REGISTERED PASS ----------------------------------------
    out["verdict"] = {
        arm: _verdict(out, arm) for arm in arms
        if arm not in (CONTROL_ARM, REFERENCE_ARM)}
    if "decoded" in records[0]:
        out["decoded"] = {
            arm: {
                "aggregate": summarize_across_seeds(
                    [r["decoded"]["arms"][arm]["aggregate"] for r in records
                     if "arms" in r.get("decoded", {})]),
                "per_factor": {nm: summarize_across_seeds(
                    [r["decoded"]["arms"][arm]["per_factor"][nm] for r in records
                     if "arms" in r.get("decoded", {})]) for nm in names},
            } for arm in arms
            if all(arm in r.get("decoded", {}).get("arms", {}) for r in records)}
    out["knn_proto_laundering"] = [r.get("knn_proto_laundering") for r in records]
    return out


def within_well_m_exponent(records: list) -> dict:
    """⚠ **POST-HOC** (`PREREG_ADDENDUM_1` §2), and labelled as such.

    The three allocations give every well a small **``M`` sweep** for free.  Fit
    ``log ε_measured`` against ``log M`` within each (seed, well) and compare the
    fitted exponent with the law's own ``−2/PR``.  ⭐ This is the law's *exponent*,
    which the pooled predicted-vs-measured fit cannot isolate.

    ⛔ Limitations, on the row: for several wells two of the three ``M`` values are
    nearly equal, so the slope is effectively a two-point contrast over the measured
    range ``M ∈ [4, 186]``; it may **not** be extrapolated beyond it.
    """
    rows = []
    for r in records:
        for w in range(len(r["spectrum"]["per_well"])):
            sel = [p for p in r["law"]["points"] if p["well"] == w and p["arm"] != "shipped"]
            M = np.asarray([p["n_modes"] for p in sel], np.float64)
            e = np.asarray([p["measured_eps"] for p in sel], np.float64)
            if np.unique(M).size < 2 or not np.all(e > 0):
                continue
            pr = float(sel[0]["participation_ratio"])
            fitted = float(np.polyfit(np.log(M), np.log(e), 1)[0])
            law = -2.0 / pr
            rows.append({
                "seed": int(r["seed"]), "well": int(w), "participation_ratio": pr,
                "n_modes_swept": sorted(int(x) for x in M),
                "m_range": [int(M.min()), int(M.max())],
                "measured_eps": [float(x) for x in e[np.argsort(M)]],
                "fitted_exponent": fitted, "law_exponent": law,
                "ratio_fitted_over_law": float(fitted / law) if law else float("nan"),
            })
    f = np.asarray([x["fitted_exponent"] for x in rows])
    lw = np.asarray([x["law_exponent"] for x in rows])
    return {
        "status": "⚠ POST-HOC — not pre-registered (PREREG_ADDENDUM_1 §2)",
        "per_well": rows,
        "n_points": len(rows),
        "fitted_exponent": {"mean": float(f.mean()), "std": float(f.std()),
                            "min": float(f.min()), "max": float(f.max())},
        "law_exponent": {"mean": float(lw.mean()), "min": float(lw.min()),
                         "max": float(lw.max())},
        "ratio_fitted_over_law": {"mean": float((f / lw).mean()),
                                  "median": float(np.median(f / lw)),
                                  "min": float((f / lw).min()),
                                  "max": float((f / lw).max())},
        "reading": (
            "⭐ The law says the exponent VARIES with PR; if the fitted exponents are "
            "near-constant across wells while -2/PR spans a 4x range, the law's LEVEL "
            "may be right while its EXPONENT is not."
        ),
    }


def _verdict(out: dict, arm: str) -> dict:
    """⭐ The pre-registered pass, applied mechanically and answered in its words.

    **PASS** = TOTAL-RETAINED rises above ``uniform``'s beyond seed spread **AND**
    distinct-of-72 does not fall, at ``bytes <= uniform``.
    **TRADE** (⛔ not a pass) = TOTAL falls but ``scale``+``orientation`` rise.
    **NULL** (⛔ not a pass) = within seed spread on every row ⇒ ⭐ uniform ``M`` is
    already optimal and re-allocation is not a lever.
    **SPLIT** (⛔ not a pass; PREREG_ADDENDUM_1) = TOTAL flat **but** distinct-of-72
    rises beyond spread.  ⛔ The pre-registered pass requires TOTAL to **rise**, so
    the §4 answer is still NO — but calling this a LOSS, as the first implementation
    did, is factually false: nothing fell.
    **LOSS** = ``Δ TOTAL < −seed spread``, and nothing weaker.
    """
    es = "SPREAD"
    A = out["by_eval_set"][es][CONTROL_ARM]
    B = out["by_eval_set"][es][arm]
    ta, tb = A["total_retained"], B["total_retained"]
    tspread = max(float(ta["std"]), float(tb["std"]))
    d_total = float(tb["mean"] - ta["mean"])
    CA = out["by_eval_set"]["CORNER_B"][CONTROL_ARM]["distinct_request_side"]
    CB = out["by_eval_set"]["CORNER_B"][arm]["distinct_request_side"]
    dspread = max(float(CA["std"]), float(CB["std"]))
    d_distinct = float(CB["mean"] - CA["mean"])
    probe_rise = {}
    for nm in ("scale", "orientation"):
        if nm not in A["per_factor"]:
            continue
        a, b = A["per_factor"][nm]["retention_code"], B["per_factor"][nm]["retention_code"]
        sp = max(float(a["std"]), float(b["std"]))
        probe_rise[nm] = {"control": a, "arm": b, "delta": float(b["mean"] - a["mean"]),
                          "seed_spread_used": sp,
                          "rose_beyond_spread": bool(float(b["mean"] - a["mean"]) > sp)}
    total_rose = bool(d_total > tspread)
    total_fell = bool(d_total < -tspread)
    total_flat = not (total_rose or total_fell)
    distinct_rose = bool(d_distinct > dspread)
    distinct_fell = bool(d_distinct < -dspread)
    within_budget = all(
        int(x) <= 0 for x in out["allocation"][arm]["bytes_delta_vs_uniform_per_seed"])
    passed = bool(total_rose and not distinct_fell and within_budget)
    if passed:
        label = "PASS"
        reading = ("⭐ PASS: TOTAL-RETAINED rose beyond seed spread at bytes <= the "
                   "uniform arm's, and distinct-of-72 did not fall.")
    elif total_fell and probe_rise and all(
            v["rose_beyond_spread"] for v in probe_rise.values()):
        label = "TRADE"
        reading = ("⭐ TRADE, ⛔ NOT A PASS: TOTAL-RETAINED fell but scale AND "
                   "orientation rose beyond seed spread — a redistribution, reported "
                   "as one and ⛔ never promoted to primary.")
    elif total_flat and distinct_rose:
        label = "SPLIT"
        reading = ("⭐ SPLIT, ⛔ NOT A PASS (PREREG_ADDENDUM_1): TOTAL-RETAINED is "
                   "WITHIN seed spread of the uniform arm while distinct-of-72 ROSE "
                   "beyond spread. ⛔ The pre-registered pass requires TOTAL to RISE, "
                   "so the answer is NO — but nothing fell, so this is ⛔ not a loss. "
                   "⚠ A 72-row statistic on 3 seeds: POST-HOC, replicate before use.")
    elif total_flat and not distinct_fell:
        label = "NULL"
        reading = ("⭐ NULL, ⛔ NOT A PASS: every row is within seed spread of the "
                   "uniform arm ⇒ uniform M is ALREADY at the byte-constrained "
                   "optimum and re-allocation is NOT a lever. ⭐ A first-class "
                   "negative result for D6.")
    else:
        label = "LOSS"
        reading = ("⛔ LOSS: the allocated arm is worse than the uniform control "
                   "beyond seed spread.")
    return {
        "arm": arm,
        "pre_registered_pass": "YES" if passed else "NO",
        "reading_label": label,
        "reading": reading,
        "total_retained": {"control": ta, "arm": tb, "delta": d_total,
                           "seed_spread_used": tspread, "rose_beyond_spread": total_rose},
        "distinct_of_corner_b": {"control": CA, "arm": CB, "delta": d_distinct,
                                 "seed_spread_used": dspread,
                                 "rose_beyond_spread": distinct_rose,
                                 "fell_beyond_spread": distinct_fell},
        "scale_orientation": probe_rise,
        "bytes_within_uniform_every_seed": bool(within_budget),
        "bytes_delta_per_seed": out["allocation"][arm]["bytes_delta_vs_uniform_per_seed"],
    }


def make_plot(summary: dict, path: Path) -> Path:
    """The ladder, the TOTAL line, and ⭐ the law's log-log fit."""
    names = list(summary["factor_names"])
    arms = list(summary["arms"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    es = summary["by_eval_set"]["SPREAD"]
    w = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        vals = [es[arm]["per_factor"][nm]["retention_code"]["mean"] for nm in names]
        errs = [es[arm]["per_factor"][nm]["retention_code"]["std"] for nm in names]
        axes[0].bar(np.arange(len(names)) + i * w, vals, w, yerr=errs, label=arm, capsize=2)
    axes[0].set_xticks(np.arange(len(names)) + 0.4)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("retention of above-chance signal")
    axes[0].set_title("the six-factor ladder, by allocation")
    axes[0].legend(fontsize=7)
    tot = [es[a]["total_retained"]["mean"] for a in arms]
    terr = [es[a]["total_retained"]["std"] for a in arms]
    axes[1].bar(np.arange(len(arms)), tot, 0.6, yerr=terr, capsize=3)
    axes[1].set_xticks(np.arange(len(arms)))
    axes[1].set_xticklabels(arms, rotation=20, ha="right", fontsize=7)
    axes[1].set_ylabel("TOTAL retained (6 factors)")
    axes[1].set_title("at CONSTANT TOTAL BYTES")
    pts = summary["law"]["points"]
    for arm in arms:
        p = [q["predicted_eps"] for q in pts if q["arm"] == arm]
        m = [q["measured_eps"] for q in pts if q["arm"] == arm]
        axes[2].scatter(p, m, s=16, label=arm)
    lim = [1e-3, 1.2]
    axes[2].plot(lim, lim, "k--", lw=0.8)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"predicted $\varepsilon = M^{-2/PR}$")
    axes[2].set_ylabel("measured relative distortion")
    fit = summary["law"]["pooled_fit"]
    axes[2].set_title(f"the law: slope {fit.get('slope', float('nan')):.2f}, "
                      f"r {fit.get('pearson_r_log', float('nan')):.2f}")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_experiment_gen_codebook_alloc(config: Optional[CHLUConfig] = None,
                                      out_dir: Optional[str] = None,
                                      log: Callable = print) -> dict:
    config = config or get_default_config()
    c = _cfg(config)
    root = Path(out_dir) if out_dir else alloc_out_root(config)
    runs = root / "runs" / f"{c.dataset}_{c.pretext}"
    runs.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in list(c.seeds):
        path = runs / f"alloc_s{int(seed)}.json"
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
        make_plot(summary, root / "codebook_alloc.png")
    for arm, v in summary["verdict"].items():
        log(f"⭐ {arm}: pre-registered pass = {v['pre_registered_pass']} — {v['reading']}")
    f = summary["law"]["pooled_fit"]
    log(f"⭐ the law, pooled ({f.get('n_points')} points): slope "
        f"{f.get('slope', float('nan')):.3f} · offset "
        f"{f.get('multiplicative_offset', float('nan')):.3f} · r "
        f"{f.get('pearson_r_log', float('nan')):.3f}")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS) + [None])
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=None, choices=list(ALLOCATIONS) + [None])
    p.add_argument("--out", default=None)
    p.add_argument("--no-decode", action="store_true")
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
    if args.quick:
        apply_quick(config)
    run_experiment_gen_codebook_alloc(config, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
