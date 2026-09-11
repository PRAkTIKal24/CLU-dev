"""Experiment: group the cached channels into wells, learn the modes, and **prove
the read cannot bypass the store** (R0 of the factorisation ladder).

⭐ DIAL DECLARATION — *isolation*.
- **Laundering control:** ⭐ random grouping at the same ``K`` and the same well-size
  histogram, run as a **full arm on every number**, never a footnote.
- **Falsifies:** random grouping matches correlation clustering within seed spread;
  **or** deleting the store leaves the read output unchanged.
- **Does NOT falsify:** a low absolute read accuracy that is low for *both*
  groupings — that is a hard dataset or a weak feature cache, and the *contrast*
  is what is under test.

⛔ **The failure this exists to catch.**  The programme's standing result is that the
block routed around the memory: deleting the entire store changed the loss by under
0.001 bits.  So here **no encoder->decoder bypass exists by construction**
(:func:`chlu.core.gen_store.read_from_codes` does not take the features as an
argument) *and* the deletion is **measured**:
:func:`chlu.core.gen_store.delete_store` empties the store and the read is re-run.
⛔ A near-zero delta is a **failed task**, reported as such, not repaired and re-run.

⚠ **Read the compositional term separately.**  ``gen-feature-bridge`` showed the
aggregate is dominated by four factors sitting at ~1.0 that carry no compositional
information, and that the ``scale`` deficit is a *photometric* rendering artefact of
the Montero corner.  Every table below therefore carries ``shape`` as its own
column, and the leakage transfer is normalised against the **matched-seen** control
(R16), never against random-A.

⛔ **Not this spoke's job:** no B arm comparison, no certificate, no C-as-queries, no
kill-rule verdict.  ``eval_b`` is touched **only** by the per-well leakage probes,
which never go near the decoder.

⛔ **Standing convention:** wells are never named semantically.  Integer indices only.

Runnable:
    chlu exp-gen-store [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_store \\
        --dataset shapes3d --pretext recon --stage store --arm clu --seed 0 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import (  # noqa: E402
    PRETEXTS,
    assert_decoder_hash,
    assert_no_holdout_leak,
    dataset_module,
    encode_images,
    features_for_rows,
    load_feature_cache,
    param_hash,
    stream_images,
)
from ..core.gen_store import (  # noqa: E402
    GROUPINGS,
    address_codes,
    assert_no_bypass,
    block_diagonality,
    c_well_mapping,
    channel_correlation,
    choose_k,
    dead_channels,
    dci_scores,
    delete_store,
    grouping_labels,
    init_store,
    leakage_from_transfer,
    mig_score,
    mode_usage,
    reconstruction_mse,
    save_store_plan,
    store_bytes,
    store_read,
    train_store,
    train_well_probe,
    well_channels_from_labels,
    well_probe_accuracy,
    well_relevance,
)
from ..eval.gen_probe import factor_labels, probe_accuracy  # noqa: E402
from ..eval.gen_probe import decode_features  # noqa: E402
from ..experiments.exp_gen_bridge import select_rows  # noqa: E402
from ..utils.checkpoints import load_model  # noqa: E402

STAGES = ("store", "aggregate")
#: Kept so the CSF3 launcher's other --stage values do not silently no-op here.
NOT_MINE = ("encoder", "decoder", "cell")
#: ⭐ The compositional term (R16).  The aggregate is reported alongside it, never
#: instead of it.
COMPOSITIONAL_FACTOR = "shape"


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's numbers are NEVER reported."""
    g = config.gen_store
    g.seeds = list(g.quick_seeds)
    g.n_corr = g.quick_n_corr
    g.n_store_train = g.quick_n_store_train
    g.train_steps = g.quick_train_steps
    g.n_probe_train = g.quick_n_probe_train
    g.probe_steps = g.quick_probe_steps
    g.n_modes = g.quick_n_modes
    g.k_max = g.quick_k_max
    return config


def bridge_root(config: CHLUConfig) -> Path:
    """Where ``gen-feature-bridge`` wrote its per-seed runs.

    ⚠ A worktree has no ``.claude/``, so ``$CHLU_GEN_BRIDGE_ROOT`` must be set when
    running from one (same trap as ``$CHLU_GEN_DATA_ROOT``)."""
    r = config.gen_store.bridge_root or os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude" / "outputs" / "gen-feature-bridge" / "runs"


def bridge_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    d = bridge_root(config) / f"{dataset}_{pretext}" / f"seed_{seed}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this spoke consumes the FROZEN cache and "
            "⛔ never rebuilds it (set gen_store.bridge_root or $CHLU_GEN_BRIDGE_ROOT)"
        )
    return d


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def select_store_rows(split, config: CHLUConfig, seed: int) -> dict:
    """Rows for the store, reusing the bridge's own ``eval_a``/``eval_b`` verbatim.

    ⛔ ``eval_a`` is excluded from the store's training pool, so the read metric is
    measured on rows no stage of this pipeline ever fitted.  The held-out rows are
    excluded structurally (the pool is ``split.train_rows``) *and* re-asserted.
    """
    bridge_rows = select_rows(split, config.gen_bridge, int(seed))
    pool = np.setdiff1d(np.asarray(split.train_rows, np.int64), bridge_rows["eval_a"])
    rng = np.random.default_rng(int(seed) * 7919 + 20_260_910)
    n = min(int(config.gen_store.n_store_train), pool.size)
    return {
        "store_train": np.sort(rng.permutation(pool)[:n]),
        "eval_a": bridge_rows["eval_a"],
        "eval_b": bridge_rows["eval_b"],
        "bridge_train": bridge_rows["train"],
    }


def matched_seen_rows(dataset: str, split, values: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """⭐ R16's permanent companion: the photometrically MATCHED SEEN control.

    Same hue corner as the Montero held-out cell, ``shape != cylinder`` (so the
    combinations are seen), minus every row used for fitting.  ⛔ Normalising the
    per-well transfer against random-A instead would score the corner's rendering
    artefact as leakage — ``gen-feature-bridge`` measured that the ``scale`` deficit
    on B is statistically indistinguishable from the same deficit on matched-seen
    data.
    """
    if dataset != "shapes3d":
        raise NotImplementedError(
            "the matched-seen control is defined by the Shapes3D Montero corner; "
            f"⛔ do not invent one for {dataset!r} — return to the Hub"
        )
    v = np.asarray(values)
    corner = (
        (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8)
        & (v[:, 3] >= 1.1) & (v[:, 5] > 20)
    )
    matched = np.where(corner & (v[:, 4] != 1.0))[0]
    matched = np.setdiff1d(matched, np.asarray(exclude, np.int64))
    if np.intersect1d(matched, split.test_rows).size:
        raise AssertionError("the matched-seen control overlaps the held-out split")
    return matched


# ---------------------------------------------------------------------------
# One (grouping, seed) arm
# ---------------------------------------------------------------------------



def gather_features(
    dataset: str,
    config: CHLUConfig,
    encoder,
    cache_rows: np.ndarray,
    features,
    cman: dict,
    rows: np.ndarray,
    log: Callable = print,
) -> np.ndarray:
    """Cached features for ``rows``, **extending coverage through the FROZEN encoder**
    for rows the cache does not hold.

    ⚠ Measured constraint, not a choice: ``gen-feature-bridge`` wrote seed 0 at
    ``cache_scope="full"`` (480,000 rows) but seeds 1 and 2 at ``"used"`` (9,288
    rows = its own train + eval_A + eval_B).  This spoke needs more train rows than
    that, and it needs the matched-seen control rows, which are in **no** ``used``
    cache.

    ⛔ This is **not** rebuilding the bridge.  The encoder is loaded from the
    bridge's own checkpoint and its parameter hash is asserted against the cache
    manifest, so an on-the-fly row is produced by *the same frozen map* that wrote
    the cache.  Newly encoded rows are quantised to the cache's own dtype so a
    cached row and an encoded row are numerically indistinguishable — and that is
    **verified**, not asserted: :func:`_assert_cache_reencodes` re-encodes a sample
    of genuinely cached rows and requires bit-identity.
    """
    rows = np.asarray(rows, np.int64)
    d = int(cman["D"])
    out = np.zeros((rows.size, d), np.float32)
    pos = np.searchsorted(cache_rows, rows)
    safe = np.minimum(pos, cache_rows.size - 1)
    hit = (pos < cache_rows.size) & (cache_rows[safe] == rows)
    if hit.any():
        out[hit] = features_for_rows(cache_rows, features, rows[hit]).reshape((-1, d))
    n_miss = int((~hit).sum())
    if n_miss:
        log(f"    extending cache coverage: {n_miss} row(s) encoded through the frozen encoder")
        missing = rows[~hit]
        slot = np.flatnonzero(~hit)
        # ⚠ chunked: 32k Shapes3D images as float32 is ~1.6 GB resident, and the
        # h5 block scan is re-walked per chunk, so this trades a second pass for
        # not holding the whole split in RAM.
        chunk = int(config.gen_store.encode_chunk)
        for i in range(0, missing.size, chunk):
            sel = slice(i, min(i + chunk, missing.size))
            imgs = stream_images(
                dataset, missing[sel], config.gen_data.data_root,
                int(config.gen_bridge.cache_block),
            )
            f = encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
            out[slot[sel]] = (
                f.astype(np.dtype(cman["dtype"])).astype(np.float32).reshape((-1, d))
            )
            del imgs, f
    return out


def _assert_cache_reencodes(dataset, config, encoder, cache_rows, features, cman, n=64, log=print):
    """⭐ Verify that re-encoding CACHED rows reproduces the cache bit-for-bit.

    Without this, "extending the cache through the frozen encoder" is a claim.  With
    it, it is a measurement.  Any mismatch means the loaded encoder is not the one
    that wrote the cache and the run must stop.
    """
    take = np.asarray(cache_rows[: int(n)], np.int64)
    cached = features_for_rows(cache_rows, features, take).reshape((take.size, -1))
    imgs = stream_images(dataset, take, config.gen_data.data_root, int(config.gen_bridge.cache_block))
    fresh = (
        encode_images(encoder, imgs, int(config.gen_bridge.eval_batch))
        .astype(np.dtype(cman["dtype"]))
        .astype(np.float32)
        .reshape((take.size, -1))
    )
    max_abs = float(np.max(np.abs(cached - fresh)))
    if max_abs != 0.0:
        raise AssertionError(
            f"re-encoding {take.size} cached rows through the loaded encoder did not "
            f"reproduce the cache (max |delta| = {max_abs}) — this is NOT the encoder "
            "that wrote the cache"
        )
    log(f"  ⭐ cache re-encode check: {take.size} rows bit-identical through the frozen encoder")
    return {"n_checked": int(take.size), "max_abs_delta": max_abs}


def _decoded_probe(decoder, probe, feats, labels, names, batch) -> dict:
    """Decode features through THE frozen decoder and read them with THE probe."""
    dec = decode_features(decoder, np.asarray(feats, np.float32), int(batch))
    out = probe_accuracy(probe, dec, labels, names, int(batch))
    out["decoded_mean"] = float(dec.mean())
    return out


def run_arm(
    dataset: str,
    pretext: str,
    seed: int,
    grouping: str,
    k: int,
    corr: np.ndarray,
    ctx: dict,
    config: CHLUConfig,
    out_dir: Path,
    log: Callable = print,
) -> dict:
    """Fit and score one (grouping, seed) arm.  ⭐ Identical code path for both
    groupings — the ONLY difference is the label vector."""
    cfg = config.gen_store
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ctx["names"]

    labels_vec, ginfo = grouping_labels(
        corr, int(k), grouping, int(seed), cfg.grouping_method,
        dead=ctx["dead"], min_well_frac=float(cfg.min_well_frac),
    )
    wc = well_channels_from_labels(labels_vec, int(k))
    sizes = [len(w) for w in wc]
    bd = block_diagonality(corr, labels_vec)
    log(
        f"  [{grouping}] method={ginfo['method_used']} (requested {ginfo['method_requested']}) "
        f"well sizes {sizes}  in-block |corr|^2 = {bd['in_block_fraction']:.4f}"
    )

    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_911 + (0 if grouping == "correlation" else 1))
    key, ik, tk = jax.random.split(key, 3)
    store = init_store(ctx["X_store"], wc, int(cfg.n_modes), ctx["feat_shape"], ik, seed=int(seed))
    log(f"  [{grouping}] training the modes ({cfg.train_steps} steps, store-only objective) …")
    store, hist, key = train_store(store, ctx["X_store"], cfg, tk, log=log)

    # ---- ⛔ the structural no-bypass proof, BEFORE any read number is believed --
    bypass = assert_no_bypass(store, ctx["X_eval_a"][:64], rng_seed=int(seed))
    log(f"  [{grouping}] ⭐ no-bypass: pure codebook gather, {bypass['bits_per_read']:.0f} bits/read")

    # ---- the read metric ---------------------------------------------------
    read_a = store_read(store, ctx["X_eval_a"], int(cfg.eval_batch))
    intact = _decoded_probe(ctx["decoder"], ctx["probe"], read_a, ctx["lab_a"], names, cfg.eval_batch)
    recon = reconstruction_mse(store, ctx["X_eval_a"], int(cfg.eval_batch))
    codes = address_codes(store, np.asarray(ctx["X_store"], np.float32).reshape((ctx["X_store"].shape[0], -1)))
    usage = mode_usage(codes, int(cfg.n_modes))
    log(
        f"  [{grouping}] READ: aggregate={intact['aggregate']:.4f} "
        f"{COMPOSITIONAL_FACTOR}={intact['per_factor'][COMPOSITIONAL_FACTOR]:.4f} "
        f"rel_mse={recon['relative_mse']:.4f} modes_used={usage['mean_used_modes']:.1f}"
    )

    # ---- ⭐ DELETE THE STORE ------------------------------------------------
    deletions = {}
    for dmode in list(cfg.delete_modes):
        dele = delete_store(store, dmode, seed=int(seed))
        dread = store_read(dele, ctx["X_eval_a"], int(cfg.eval_batch))
        dacc = _decoded_probe(ctx["decoder"], ctx["probe"], dread, ctx["lab_a"], names, cfg.eval_batch)
        drec = reconstruction_mse(dele, ctx["X_eval_a"], int(cfg.eval_batch))
        deletions[dmode] = {
            "read": dacc,
            "recon": drec,
            "delta_aggregate": float(intact["aggregate"] - dacc["aggregate"]),
            "delta_compositional": float(
                intact["per_factor"][COMPOSITIONAL_FACTOR]
                - dacc["per_factor"][COMPOSITIONAL_FACTOR]
            ),
            "delta_read_output_l2": float(
                np.sqrt(np.mean((np.asarray(read_a) - np.asarray(dread)) ** 2))
            ),
            "delta_relative_mse": float(drec["relative_mse"] - recon["relative_mse"]),
        }
        log(
            f"  [{grouping}] ⭐ DELETE({dmode}): aggregate {intact['aggregate']:.4f} -> "
            f"{dacc['aggregate']:.4f} (Δ={deletions[dmode]['delta_aggregate']:+.4f})  "
            f"{COMPOSITIONAL_FACTOR} Δ={deletions[dmode]['delta_compositional']:+.4f}  "
            f"read-output RMS Δ={deletions[dmode]['delta_read_output_l2']:.4f}"
        )

    # ---- ⭐ leakage: the factorisation ladder's x-axis ----------------------
    leak = run_leakage(store, wc, ctx, config, key, grouping, log)

    # ---- companions --------------------------------------------------------
    rel = well_relevance(codes, ctx["lab_store"], ctx["cardinalities"])
    comp = {
        "block_diagonality": bd,
        "mig": mig_score(codes, ctx["lab_store"], ctx["cardinalities"]),
        "dci": dci_scores(codes, ctx["lab_store"], ctx["cardinalities"]),
        "well_relevance": rel,
    }
    cmap = c_well_mapping(ctx["c_records"], rel, names, float(cfg.c_top_share))
    log(
        f"  [{grouping}] MIG={comp['mig']['mig']:.4f} DCI-D={comp['dci']['disentanglement']:.4f} "
        f"pooled leakage={leak['pooled']['leakage_aggregate']:.4f} "
        f"C intra-well={cmap['intra_well_conflict_fraction']:.3f}"
    )

    return {
        "grouping": grouping,
        "seed": int(seed),
        "K": int(k),
        "well_sizes": sizes,
        "grouping_info": ginfo,
        "labels": [int(x) for x in labels_vec],
        "store_bytes": store_bytes(store),
        "no_bypass": bypass,
        "training": {
            "steps": int(cfg.train_steps),
            "loss_first": float(hist[0]) if hist else None,
            "loss_final": float(hist[-1]) if hist else None,
        },
        "read": intact,
        "recon": recon,
        "mode_usage": usage,
        "deletion": deletions,
        "leakage": leak,
        "companions": comp,
        "c_well_mapping": cmap,
    }


def run_leakage(store, wc, ctx, config: CHLUConfig, key, grouping: str, log) -> dict:
    """⭐ ``leakage = 1 - per-well probe transfer``, normalised against matched-seen.

    Per well: train per-factor linear heads on **that well's sub-vector only**, on
    SEEN rows; evaluate on the held-out combinations and on the matched-seen
    control.  ⛔ The probes never touch the decoder and render no verdict — they are
    the ladder's x-axis, not a cell measurement.
    """
    cfg = config.gen_store
    names = ctx["names"]
    per_well = []
    for j, w in enumerate(wc):
        cols = np.asarray(w, np.int64)
        key, pk = jax.random.split(key)
        probe, norm, hist = train_well_probe(
            ctx["X_probe"][:, cols], ctx["lab_probe"], ctx["cardinalities"], cfg, pk
        )
        acc_seen = well_probe_accuracy(probe, norm, ctx["X_probe"][:, cols], ctx["lab_probe"], names)
        acc_ho = well_probe_accuracy(probe, norm, ctx["X_b"][:, cols], ctx["lab_b"], names)
        acc_ms = well_probe_accuracy(probe, norm, ctx["X_ms"][:, cols], ctx["lab_ms"], names)
        tr = leakage_from_transfer(acc_ho, acc_ms, names)
        per_well.append(
            {
                "well": int(j),
                "d_k": int(cols.size),
                "acc_fit_seen": acc_seen,
                "acc_heldout": acc_ho,
                "acc_matched_seen": acc_ms,
                **tr,
                "probe_loss_first": float(hist[0]),
                "probe_loss_final": float(hist[-1]),
            }
        )
    pooled_ho = {
        "per_factor": {
            str(n): float(np.mean([p["acc_heldout"]["per_factor"][str(n)] for p in per_well]))
            for n in names
        },
        "aggregate": float(np.mean([p["acc_heldout"]["aggregate"] for p in per_well])),
    }
    pooled_ms = {
        "per_factor": {
            str(n): float(np.mean([p["acc_matched_seen"]["per_factor"][str(n)] for p in per_well]))
            for n in names
        },
        "aggregate": float(np.mean([p["acc_matched_seen"]["aggregate"] for p in per_well])),
    }
    pooled = leakage_from_transfer(pooled_ho, pooled_ms, names)
    pooled["pooled_acc_heldout"] = pooled_ho
    pooled["pooled_acc_matched_seen"] = pooled_ms
    pooled["mean_of_per_well_leakage"] = float(
        np.mean([p["leakage_aggregate"] for p in per_well])
    )
    pooled["mean_of_per_well_leakage_compositional"] = float(
        np.mean([p["leakage_per_factor"][COMPOSITIONAL_FACTOR] for p in per_well])
    )
    return {"per_well": per_well, "pooled": pooled}


# ---------------------------------------------------------------------------
# One seed: shared context + both groupings
# ---------------------------------------------------------------------------


def run_seed(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    k_pin: Optional[int] = None,
    log: Callable = print,
) -> dict:
    cfg = config.gen_store
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bdir = bridge_dir(config, dataset, pretext, seed)
    log(f"  frozen bridge: {bdir}")

    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    rows = select_store_rows(split, config, seed)

    # ⛔ R3.3, re-run on OUR training rows (the bridge's assertion does not cover them)
    leakcheck = assert_no_holdout_leak(split, rows["store_train"])
    log(
        f"  ⭐ no-leak: {leakcheck['n_checked']} store-training rows checked, "
        f"{leakcheck['n_holdout_excluded']}/{leakcheck['n_holdout_total']} held-out "
        f"excluded, {leakcheck['n_holdout_leaked']} leaked"
    )

    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    encoder = load_model(bdir / "encoder.pkl")
    decoder = load_model(bdir / "decoder.pkl")
    probe = load_model(bdir / "probe.pkl")
    assert_decoder_hash(decoder, bman["decoder_param_hash"])  # ⛔ THE shared decoder
    enc_hash = param_hash(encoder)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest hash")
    crows, feats, cman = load_feature_cache(bdir, verify=bool(cfg.verify_cache_checksum))
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("the feature cache was written by a different encoder")
    log(
        f"  feature cache verified: {cman['n_rows']} rows (scope={cman['scope']}), "
        f"D={cman['D']}, {cman['dtype']}"
    )
    reenc = _assert_cache_reencodes(dataset, config, encoder, crows, feats, cman, log=log)

    ms_rows = matched_seen_rows(dataset, split, values, rows["store_train"])
    probe_rows = rows["store_train"][: int(cfg.n_probe_train)]
    corr_rows = rows["store_train"][: int(cfg.n_corr)]
    log(
        f"  rows: store_train={rows['store_train'].size} eval_a={rows['eval_a'].size} "
        f"eval_b={rows['eval_b'].size} matched_seen={ms_rows.size} "
        f"probe_train={probe_rows.size} corr={corr_rows.size}"
    )

    def gather(r):
        return gather_features(dataset, config, encoder, crows, feats, cman, r, log=log)

    ctx = {
        "names": names,
        "cardinalities": cards,
        "feat_shape": feat_shape,
        "decoder": decoder,
        "probe": probe,
        "X_store": gather(rows["store_train"]),
        "X_eval_a": gather(rows["eval_a"]),
        "X_b": gather(rows["eval_b"]),
        "X_ms": gather(ms_rows),
        "lab_a": factor_labels(grid, rows["eval_a"]),
        "lab_b": factor_labels(grid, rows["eval_b"]),
        "lab_ms": factor_labels(grid, ms_rows),
        "lab_store": factor_labels(grid, rows["store_train"]),
        "c_records": load_c_records(dataset, config, log),
    }
    ctx["X_probe"] = ctx["X_store"][: int(cfg.n_probe_train)]
    ctx["lab_probe"] = ctx["lab_store"][: int(cfg.n_probe_train)]

    log(f"  channel correlation over {corr_rows.size} train rows (D={ctx['X_store'].shape[1]}) …")
    Xc = ctx["X_store"][: int(cfg.n_corr)]
    ctx["dead"] = dead_channels(Xc)
    corr = channel_correlation(Xc)
    log(
        f"  ⚠ {int(ctx['dead'].sum())}/{ctx['dead'].size} coordinates are DEAD "
        "(constant post-ReLU); they are clustered around and placed deterministically"
    )

    if k_pin is not None:
        k = int(k_pin)
        kinfo = {"k": k, "rule": "PINNED from the store plan (⛔ never re-derived per arm)"}
    elif cfg.k_wells is not None:
        k = int(cfg.k_wells)
        kinfo = {"k": k, "rule": "pinned by gen_store.k_wells"}
    else:
        kinfo = choose_k(corr, int(cfg.k_min), int(cfg.k_max), dead=ctx["dead"])
        k = int(kinfo["k"])
    log(f"  ⭐ K = {k}  ({kinfo['rule']})")

    # ⭐ THE ceiling companion: the SAME probe reading the SAME real cached features
    # through the SAME decoder with NO store at all.  Every read number below is a
    # fraction of this, never an absolute.
    ceiling = _decoded_probe(decoder, probe, ctx["X_eval_a"].reshape((-1,) + feat_shape),
                             ctx["lab_a"], names, cfg.eval_batch)
    log(
        f"  ceiling (real cached features, no store): aggregate={ceiling['aggregate']:.4f} "
        f"{COMPOSITIONAL_FACTOR}={ceiling['per_factor'][COMPOSITIONAL_FACTOR]:.4f}"
    )

    arms = {}
    for grouping in list(cfg.groupings):
        arms[grouping] = run_arm(
            dataset, pretext, seed, grouping, k, corr, ctx, config, out_dir, log
        )

    manifest = {
        "schema": "chlu-gen-store/1",
        "dataset": dataset,
        "pretext": pretext,
        "seed": int(seed),
        "bridge_dir": str(bdir),
        "decoder_param_hash": bman["decoder_param_hash"],
        "probe_param_hash": bman["probe_param_hash"],
        "encoder_param_hash": bman["encoder_param_hash"],
        "feature_cache_sha256": cman["features_sha256"],
        "feature_cache_scope": cman["scope"],
        "feature_cache_n_rows": int(cman["n_rows"]),
        "cache_reencode_check": reenc,
        "K": int(k),
        "k_choice": kinfo,
        "n_modes": int(cfg.n_modes),
        "D": int(np.prod(feat_shape)),
        "feat_shape": [int(x) for x in feat_shape],
        "rows": {
            "store_train": int(rows["store_train"].size),
            "eval_a": int(rows["eval_a"].size),
            "eval_b": int(rows["eval_b"].size),
            "matched_seen": int(ms_rows.size),
            "probe_train": int(probe_rows.size),
            "corr": int(corr_rows.size),
        },
        "no_leak": leakcheck,
        "n_dead_channels": int(ctx["dead"].sum()),
        "ceiling_no_store": ceiling,
        "arms": arms,
        "contrast": contrast(arms, ceiling),
    }
    _write_json(manifest, out_dir / "store_manifest.json")
    if cfg.make_plots:
        _plot_seed(manifest, corr, out_dir / "grouping.png")
    return manifest


def load_c_records(dataset: str, config: CHLUConfig, log: Callable = print) -> list:
    """The FACTOR-SPACE C set, byte-identical for every arm (R2).  ⛔ Never rewritten."""
    from ..data.gen_splits import default_gen_data_root

    root = Path(config.gen_data.data_root) if config.gen_data.data_root else default_gen_data_root()
    p = root / dataset / "c_invalid.json"
    if not p.exists():
        log(f"  ⚠ no C set at {p} — the well-space C mapping is SKIPPED, not faked")
        return []
    with open(p) as fh:
        d = json.load(fh)
    log(f"  C set: {len(d['records'])} factor-space records (⛔ diagnostic mapping only)")
    return d["records"]


def contrast(arms: dict, ceiling: dict) -> dict:
    """⭐ correlation minus random, at the same ``K`` — the number under test."""
    if not all(g in arms for g in GROUPINGS):
        return {}
    c, r = arms["correlation"], arms["random"]
    out = {
        "read_aggregate": {
            "correlation": c["read"]["aggregate"],
            "random": r["read"]["aggregate"],
            "margin": float(c["read"]["aggregate"] - r["read"]["aggregate"]),
            "ceiling": ceiling["aggregate"],
        },
        "read_compositional": {
            "factor": COMPOSITIONAL_FACTOR,
            "correlation": c["read"]["per_factor"][COMPOSITIONAL_FACTOR],
            "random": r["read"]["per_factor"][COMPOSITIONAL_FACTOR],
            "margin": float(
                c["read"]["per_factor"][COMPOSITIONAL_FACTOR]
                - r["read"]["per_factor"][COMPOSITIONAL_FACTOR]
            ),
            "ceiling": ceiling["per_factor"][COMPOSITIONAL_FACTOR],
        },
        "relative_mse": {
            "correlation": c["recon"]["relative_mse"],
            "random": r["recon"]["relative_mse"],
            "relative_improvement": float(
                1.0 - c["recon"]["relative_mse"] / r["recon"]["relative_mse"]
            ),
        },
        "leakage_pooled": {
            "correlation": c["leakage"]["pooled"]["leakage_aggregate"],
            "random": r["leakage"]["pooled"]["leakage_aggregate"],
            "margin": float(
                r["leakage"]["pooled"]["leakage_aggregate"]
                - c["leakage"]["pooled"]["leakage_aggregate"]
            ),
        },
        "leakage_compositional": {
            "factor": COMPOSITIONAL_FACTOR,
            "correlation": c["leakage"]["pooled"]["leakage_per_factor"][COMPOSITIONAL_FACTOR],
            "random": r["leakage"]["pooled"]["leakage_per_factor"][COMPOSITIONAL_FACTOR],
            "margin": float(
                r["leakage"]["pooled"]["leakage_per_factor"][COMPOSITIONAL_FACTOR]
                - c["leakage"]["pooled"]["leakage_per_factor"][COMPOSITIONAL_FACTOR]
            ),
        },
        "block_diagonality": {
            "correlation": c["companions"]["block_diagonality"]["in_block_fraction"],
            "random": r["companions"]["block_diagonality"]["in_block_fraction"],
        },
        "mig": {"correlation": c["companions"]["mig"]["mig"], "random": r["companions"]["mig"]["mig"]},
        "dci_disentanglement": {
            "correlation": c["companions"]["dci"]["disentanglement"],
            "random": r["companions"]["dci"]["disentanglement"],
        },
        "c_intra_well": {
            "correlation": c["c_well_mapping"]["intra_well_conflict_fraction"],
            "random": r["c_well_mapping"]["intra_well_conflict_fraction"],
        },
    }
    for dmode in c["deletion"]:
        out[f"delete_{dmode}_delta_aggregate"] = {
            "correlation": c["deletion"][dmode]["delta_aggregate"],
            "random": r["deletion"][dmode]["delta_aggregate"],
        }
        out[f"delete_{dmode}_delta_compositional"] = {
            "correlation": c["deletion"][dmode]["delta_compositional"],
            "random": r["deletion"][dmode]["delta_compositional"],
        }
    return out


# ---------------------------------------------------------------------------
# Aggregate over seeds
# ---------------------------------------------------------------------------


def _ms(vals) -> dict:
    a = np.asarray([v for v in vals], np.float64)
    return {
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "values": [float(x) for x in a],
    }


def aggregate_runs(run_dirs, log: Callable = print) -> dict:
    mans = []
    for d in run_dirs:
        with open(Path(d) / "store_manifest.json") as fh:
            mans.append(json.load(fh))
    if not mans:
        raise ValueError("no per-seed runs to aggregate")
    names = list(mans[0]["ceiling_no_store"]["per_factor"].keys())
    ks = sorted({m["K"] for m in mans})
    if len(ks) != 1:
        raise AssertionError(f"seeds disagree on K: {ks} — K is pinned, never per-arm")
    out = {
        "schema": "chlu-gen-store-aggregate/1",
        "dataset": mans[0]["dataset"],
        "pretext": mans[0]["pretext"],
        "seeds": [m["seed"] for m in mans],
        "K": int(ks[0]),
        "n_modes": mans[0]["n_modes"],
        "D": mans[0]["D"],
        "store_bytes": mans[0]["arms"]["correlation"]["store_bytes"],
        "ceiling_no_store": {
            "aggregate": _ms([m["ceiling_no_store"]["aggregate"] for m in mans]),
            "per_factor": {
                n: _ms([m["ceiling_no_store"]["per_factor"][n] for m in mans]) for n in names
            },
        },
        "no_leak": {
            "n_holdout_leaked": [m["no_leak"]["n_holdout_leaked"] for m in mans],
            "n_holdout_excluded": [m["no_leak"]["n_holdout_excluded"] for m in mans],
        },
        "arms": {},
    }
    for g in GROUPINGS:
        if g not in mans[0]["arms"]:
            continue
        A = [m["arms"][g] for m in mans]
        entry = {
            "read_aggregate": _ms([a["read"]["aggregate"] for a in A]),
            "read_per_factor": {
                n: _ms([a["read"]["per_factor"][n] for a in A]) for n in names
            },
            "relative_mse": _ms([a["recon"]["relative_mse"] for a in A]),
            "mean_used_modes": _ms([a["mode_usage"]["mean_used_modes"] for a in A]),
            "leakage_pooled": _ms([a["leakage"]["pooled"]["leakage_aggregate"] for a in A]),
            "leakage_compositional": _ms(
                [a["leakage"]["pooled"]["leakage_per_factor"][COMPOSITIONAL_FACTOR] for a in A]
            ),
            "leakage_mean_of_wells": _ms(
                [a["leakage"]["pooled"]["mean_of_per_well_leakage"] for a in A]
            ),
            "block_diagonality": _ms(
                [a["companions"]["block_diagonality"]["in_block_fraction"] for a in A]
            ),
            "mig": _ms([a["companions"]["mig"]["mig"] for a in A]),
            "dci_disentanglement": _ms([a["companions"]["dci"]["disentanglement"] for a in A]),
            "dci_completeness": _ms([a["companions"]["dci"]["completeness"] for a in A]),
            "dci_informativeness": _ms([a["companions"]["dci"]["informativeness"] for a in A]),
            "c_intra_well_conflict_fraction": _ms(
                [a["c_well_mapping"]["intra_well_conflict_fraction"] for a in A]
            ),
            "well_sizes": A[0]["well_sizes"],
            "store_bytes": A[0]["store_bytes"]["store_bytes"],
            "bits_per_read": A[0]["store_bytes"]["bits_per_read"],
            "deletion": {},
        }
        for dmode in A[0]["deletion"]:
            entry["deletion"][dmode] = {
                "delta_aggregate": _ms([a["deletion"][dmode]["delta_aggregate"] for a in A]),
                "delta_compositional": _ms(
                    [a["deletion"][dmode]["delta_compositional"] for a in A]
                ),
                "delta_read_output_rms": _ms(
                    [a["deletion"][dmode]["delta_read_output_l2"] for a in A]
                ),
                "read_aggregate_after": _ms([a["deletion"][dmode]["read"]["aggregate"] for a in A]),
            }
        out["arms"][g] = entry

    if all(g in out["arms"] for g in GROUPINGS):
        c, r = out["arms"]["correlation"], out["arms"]["random"]
        per_seed_margin = [
            m["contrast"]["read_aggregate"]["margin"] for m in mans
        ]
        per_seed_margin_comp = [
            m["contrast"]["read_compositional"]["margin"] for m in mans
        ]
        out["contrast"] = {
            "read_aggregate_margin": _ms(per_seed_margin),
            "read_compositional_margin": _ms(per_seed_margin_comp),
            "relative_mse_improvement": _ms(
                [m["contrast"]["relative_mse"]["relative_improvement"] for m in mans]
            ),
            "leakage_pooled_margin": _ms(
                [m["contrast"]["leakage_pooled"]["margin"] for m in mans]
            ),
            "leakage_compositional_margin": _ms(
                [m["contrast"]["leakage_compositional"]["margin"] for m in mans]
            ),
            "mig_margin": _ms([
                m["contrast"]["mig"]["correlation"] - m["contrast"]["mig"]["random"] for m in mans
            ]),
            "c_intra_well_margin": _ms([
                m["contrast"]["c_intra_well"]["correlation"]
                - m["contrast"]["c_intra_well"]["random"]
                for m in mans
            ]),
            "note": (
                "⭐ margin = correlation - random at the SAME K and the SAME well-size "
                "histogram; leakage margins are random - correlation (lower leakage is "
                "the predicted direction). Spread is the population std over seeds."
            ),
        }
        # ⛔ the falsifier, evaluated mechanically rather than in prose
        sp = max(c["read_aggregate"]["std"], r["read_aggregate"]["std"])
        out["falsifier_grouping_is_arbitrary"] = {
            "margin": out["contrast"]["read_aggregate_margin"]["mean"],
            "max_seed_spread": float(sp),
            "margin_exceeds_seed_spread": bool(
                abs(out["contrast"]["read_aggregate_margin"]["mean"]) > sp
            ),
        }
        dm = c["deletion"].get("zero", {}).get("delta_aggregate", {}).get("mean", float("nan"))
        out["falsifier_store_is_bypassed"] = {
            "delete_zero_delta_aggregate": float(dm),
            "threshold_prereg": 0.30,
            "near_zero_threshold": 0.01,
            "FAILED_TASK": bool(abs(dm) < 0.01),
        }
        log(
            f"⭐ CONTRAST (seeds {out['seeds']}, K={out['K']}): "
            f"read aggregate margin = {out['contrast']['read_aggregate_margin']['mean']:+.4f} "
            f"± {out['contrast']['read_aggregate_margin']['std']:.4f}; "
            f"{COMPOSITIONAL_FACTOR} margin = "
            f"{out['contrast']['read_compositional_margin']['mean']:+.4f} "
            f"± {out['contrast']['read_compositional_margin']['std']:.4f}"
        )
        log(
            f"⭐ DELETE-THE-STORE (zero): Δaggregate = {dm:+.4f} "
            f"({'⛔ FAILED TASK — the read bypasses the store' if abs(dm) < 0.01 else 'the read uses the store'})"
        )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_gen_store(
    config: Optional[CHLUConfig] = None,
    dataset: Optional[str] = None,
    pretext: Optional[str] = None,
    out_dir: Optional[str] = None,
    seeds=None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    """Config-load -> data -> init -> train-or-load -> evaluate -> plot, over seeds."""
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_store
    dataset = dataset or cfg.dataset
    pretext = pretext or cfg.pretext
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}")
    seeds = list(seeds) if seeds is not None else list(cfg.seeds)
    root = Path(out_dir or Path("results") / "gen_store") / f"{dataset}_{pretext}"
    root.mkdir(parents=True, exist_ok=True)

    run_dirs, k_pin = [], (int(cfg.k_wells) if cfg.k_wells is not None else None)
    plan_path = root / "store_plan.json"
    if plan_path.exists():
        from ..core.gen_store import load_store_plan

        k_pin = int(load_store_plan(plan_path)["K"])
        log(f"[gen-store] K pinned from {plan_path}: {k_pin}")
    for i, seed in enumerate(seeds):
        d = root / f"seed_{seed}"
        log(f"[gen-store] {dataset}/{pretext} seed={seed} -> {d}")
        if (d / "store_manifest.json").exists():
            log("  store artifacts present — reusing (train-or-load)")
            with open(d / "store_manifest.json") as fh:
                man = json.load(fh)
        else:
            man = run_seed(dataset, pretext, seed, d, config, k_pin=k_pin, log=log)
        if k_pin is None:
            k_pin = int(man["K"])
        run_dirs.append(d)
        # ⭐ the reusable artifact: pinned at the K-choice seed, consumed verbatim
        # by every later factorisation-ladder rung.
        if i == 0 and not plan_path.exists():
            save_store_plan(
                plan_path,
                dataset=dataset,
                pretext=pretext,
                k=man["K"],
                n_modes=man["n_modes"],
                dim=man["D"],
                feat_shape=man["feat_shape"],
                labels=man["arms"]["correlation"]["labels"],
                k_choice=man["k_choice"],
                grouping_method=man["arms"]["correlation"]["grouping_info"]["method_used"],
                store_bytes_dict=man["arms"]["correlation"]["store_bytes"],
                extra={
                    "rung": "R0",
                    "rung_description": (
                        "fixed correlation-clustered grouping, plain reconstruction "
                        "encoder, label-free"
                    ),
                    "k_choice_seed": int(seed),
                    "grouping_info": man["arms"]["correlation"]["grouping_info"],
                    "n_dead_channels": man["n_dead_channels"],
                    "encoder_param_hash": man["encoder_param_hash"],
                    "feature_cache_sha256": man["feature_cache_sha256"],
                },
            )
            log(f"  ⭐ store plan pinned -> {plan_path}")

    agg = aggregate_runs(run_dirs, log=log)
    agg["quick"] = bool(quick)
    if quick:
        agg["WARNING"] = "quick run — plumbing only, NOT a reported number"
    _write_json(agg, root / "store_results.json")
    _write_json(
        {
            "schema": "chlu-gen-store-bytes/1",
            "dataset": dataset,
            "pretext": pretext,
            **agg["store_bytes"],
            "note": (
                "⭐ gen-baselines byte-matches its kNN arm to `store_bytes`. "
                "The addressing head is NOT included; it is reported in "
                "store_results.json under arms.*.bits_per_read."
            ),
        },
        root / "store_bytes.json",
    )
    if config.gen_store.make_plots:
        _plot_aggregate(agg, root / "grouping_contrast.png")
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_seed(man: dict, corr: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, g in zip(axes, GROUPINGS, strict=False):
        if g not in man["arms"]:
            continue
        lab = np.asarray(man["arms"][g]["labels"])
        order = np.argsort(lab, kind="stable")
        ax.imshow(np.abs(corr[np.ix_(order, order)]), cmap="magma", vmin=0, vmax=1)
        ax.set_title(
            f"{g} grouping, K={man['K']}\nin-block |corr|² = "
            f"{man['arms'][g]['companions']['block_diagonality']['in_block_fraction']:.3f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "channel correlation, coordinates ordered by well index (wells are integers only)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_aggregate(agg: dict, path: Path) -> None:
    if not all(g in agg["arms"] for g in GROUPINGS):
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = ["aggregate", COMPOSITIONAL_FACTOR]
    x = np.arange(len(labels))
    for i, g in enumerate(GROUPINGS):
        a = agg["arms"][g]
        m = [a["read_aggregate"]["mean"], a["read_per_factor"][COMPOSITIONAL_FACTOR]["mean"]]
        e = [a["read_aggregate"]["std"], a["read_per_factor"][COMPOSITIONAL_FACTOR]["std"]]
        axes[0].bar(x + (i - 0.5) * 0.35, m, 0.35, yerr=e, capsize=3, label=g)
    axes[0].axhline(
        agg["ceiling_no_store"]["aggregate"]["mean"], ls="--", lw=1, color="k",
        label="no-store ceiling (aggregate)",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("probe top-1 on the store read")
    axes[0].set_title("read metric — same K, same well sizes", fontsize=9)
    axes[0].legend(fontsize=7)

    for i, g in enumerate(GROUPINGS):
        a = agg["arms"][g]
        axes[1].bar(i, a["leakage_pooled"]["mean"], 0.6, yerr=a["leakage_pooled"]["std"], capsize=3)
        axes[1].bar(
            i + 2.5, a["leakage_compositional"]["mean"], 0.6,
            yerr=a["leakage_compositional"]["std"], capsize=3,
        )
    axes[1].set_xticks([0, 1, 2.5, 3.5])
    axes[1].set_xticklabels(list(GROUPINGS) + list(GROUPINGS), rotation=20, fontsize=7)
    axes[1].set_title(
        f"leakage = 1 − transfer (pooled | {COMPOSITIONAL_FACTOR})\n"
        "normalised against MATCHED-SEEN (R16)", fontsize=9,
    )

    for i, g in enumerate(GROUPINGS):
        d = agg["arms"][g]["deletion"].get("zero")
        if d is None:
            continue
        axes[2].bar(i, d["delta_aggregate"]["mean"], 0.5, yerr=d["delta_aggregate"]["std"],
                    capsize=3, label=None)
    axes[2].axhline(0.01, ls=":", color="r", lw=1.4, label="⛔ near-zero = bypass")
    axes[2].set_xticks(range(len(GROUPINGS)))
    axes[2].set_xticklabels(list(GROUPINGS), fontsize=8)
    axes[2].set_ylabel("Δ aggregate when the store is DELETED")
    axes[2].set_title("delete-the-store delta", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI CONTRACT (binding): --dataset --pretext --stage --arm --seed --out
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chlu.experiments.exp_gen_store",
        description="channel grouping + per-well modes + the store-only read (R0)",
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument(
        "--stage", default="store", choices=list(STAGES) + list(NOT_MINE),
        help="store | aggregate (encoder/decoder -> exp_gen_bridge, cell -> exp_gen_cell)",
    )
    p.add_argument(
        "--arm", default="clu",
        help="recorded in the manifest ONLY; the groupings run inside one job so "
             "correlation and random share a byte-identical context",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="artifacts directory")
    p.add_argument("--quick", action="store_true", help="plumbing smoke; NOT a result")
    p.add_argument("--bridge-root", default=None, help="overrides $CHLU_GEN_BRIDGE_ROOT")
    p.add_argument("--data-root", default=None, help="overrides $CHLU_GEN_DATA_ROOT")
    p.add_argument("--k-wells", type=int, default=None, help="pin K (default: eigengap rule)")
    p.add_argument("--n-modes", type=int, default=None)
    p.add_argument("--n-store-train", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--groupings", nargs="+", default=None, choices=list(GROUPINGS))
    p.add_argument("--runs", nargs="+", default=None, help="per-seed dirs for --stage aggregate")
    p.add_argument("--no-plots", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    if args.quick:
        config = apply_quick(config)
    g = config.gen_store
    if args.data_root:
        config.gen_data.data_root = args.data_root
    if args.bridge_root:
        g.bridge_root = args.bridge_root
    for attr, val in (
        ("k_wells", args.k_wells),
        ("n_modes", args.n_modes),
        ("n_store_train", args.n_store_train),
        ("train_steps", args.train_steps),
    ):
        if val is not None:
            setattr(g, attr, int(val))
    if args.groupings:
        g.groupings = list(args.groupings)
    if args.no_plots:
        g.make_plots = False
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(
            f"[gen-store] stage={args.stage} is not this module's "
            "(encoder/decoder -> exp_gen_bridge, cell -> exp_gen_cell). Refusing to no-op.",
            file=sys.stderr,
        )
        return 2
    config = _config_from_args(args)
    cfg = config.gen_store
    dataset = args.dataset or cfg.dataset
    pretext = args.pretext or cfg.pretext
    out = Path(args.out or (Path("results") / "gen_store" / f"{dataset}_{pretext}"))
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[gen-store] dataset={dataset} pretext={pretext} stage={args.stage} "
        f"arm={args.arm} seed={args.seed} out={out}"
    )
    print(
        "[gen-store] ⭐ both groupings run in this job; ⛔ random grouping is the "
        "laundering control and is a full arm on every number."
    )
    if args.stage == "store":
        run_seed(dataset, pretext, args.seed, out, config,
                 k_pin=(int(cfg.k_wells) if cfg.k_wells is not None else None))
    if args.stage == "aggregate":
        if not args.runs:
            print("[gen-store] --stage aggregate needs --runs", file=sys.stderr)
            return 2
        agg = aggregate_runs(args.runs)
        _write_json(agg, out / "store_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
