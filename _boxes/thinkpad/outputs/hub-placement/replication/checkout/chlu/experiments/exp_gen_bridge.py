"""Experiment: the frozen encoder/decoder **bridge** and the ceiling that gates the phase.

⭐ DIAL DECLARATION — *none: instrument, plus one gating measurement.*  There is
no store, no well, no arm and no CLU number in this file.  Nothing it produces is
evidence for the primitive.

What it builds, once, for every downstream arm to share:

1. a small conv **encoder** fit on the Montero *train split only* with a declared
   pretext (``recon`` = the mandatory primary, R1; ``cls`` = the diagnostic arm),
   then frozen and hashed;
2. the **feature cache** of penultimate maps, written once with a checksum;
3. ⭐ **THE decoder** — one renderer, fit on real ``(feature, image)`` pairs from
   the train split, frozen, hashed, and shared by every arm forever.  ⛔ No arm
   substitutes its own; ``chlu.core.gen_bridge.assert_decoder_hash`` is the check;
4. the **probe** — a per-factor classifier on real images (R1's measurement
   exemption), whose accuracy on *real* held-out images is reported so the
   instrument is auditable before anything is built on it;
5. **the gate**: ``ceiling_A`` / ``ceiling_B`` from decoding real cached features,
   and the verdict ``PROCEED`` iff ``ceiling_B >= 0.8 * ceiling_A`` (R3.1).
   ⛔ A ``STOP`` is a legitimate result; it is not a licence to tune the decoder.

The run is **arm-independent by construction**: every draw is threaded from
``seed`` alone, so the decoder frozen at a seed is bit-identical whichever
``--arm`` the launcher passes.  That is asserted in ``tests/test_gen_bridge.py``.

Runnable:
    chlu exp-gen-bridge [--dataset shapes3d] [--pretext recon] [--quick]
    uv run python -m chlu.experiments.exp_gen_bridge \\
        --dataset shapes3d --pretext recon --stage encoder --arm clu --seed 0 --out DIR

⛔ The CSF3 launcher (``scripts/csf3/job_gpu_gen.sh``) defaults ``STAGE=encoder``
to the module ``chlu.experiments.exp_gen_encoder``; submit the bridge stages with
``PAYLOAD_MODULE=chlu.experiments.exp_gen_bridge`` (the launcher's own override).
"""

from __future__ import annotations

import argparse
import json
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
    assert_no_holdout_leak,
    build_feature_cache,
    encode_images,
    features_for_rows,
    load_feature_cache,
    param_count,
    param_hash,
    resolve_decoder_cfg,
    stream_images,
    train_decoder,
    train_encoder,
    train_probe,
    whitening_feasibility,
)
from ..eval.gen_probe import (  # noqa: E402
    binary_auroc,  # noqa: F401  (re-exported for the dSprites tests)
    decode_features,
    discrimination_metrics,
    discrimination_verdict,
    factor_labels,
    factor_probs,
    gate_verdict,
    measure_ceiling,
    multiclass_balanced_accuracy,
    predict_factors,
    probe_accuracy,
    summarize_per_factor,
    summarize_seeds,
)
from ..utils.checkpoints import load_model, save_model  # noqa: E402

STAGES = ("encoder", "decoder", "all", "aggregate", "discrimination")
#: The HEADLINE hold-out condition.  Anything else is a **secondary full
#: configuration** (R17/R20): its encoder, cache, decoder and probe are re-fitted
#: with its own cells excluded and land in their own artifacts directory.
#: ⛔ Artifacts frozen under one condition have SEEN the other's held-out rows
#: and must never be scored on them.
DEFAULT_CONDITION = "recomb2element"
#: Kept only so the CSF3 launcher's --stage {store,cell} do not silently no-op here.
NOT_MINE = ("store", "cell")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — a quick run's ceiling is NEVER a reported number."""
    gb = config.gen_bridge
    gb.enc_steps = gb.quick_enc_steps
    gb.dec_steps = gb.quick_dec_steps
    gb.probe_steps = gb.quick_probe_steps
    gb.n_train = gb.quick_n_train
    gb.n_eval_a = gb.quick_n_eval_a
    gb.cache_scope = gb.quick_cache_scope
    gb.seeds = list(gb.quick_seeds)
    return config


def _data_root(config: CHLUConfig) -> Optional[str]:
    """Gen-data root: ``gen_data.data_root`` else ``$CHLU_GEN_DATA_ROOT`` else default.

    ⚠ ``CHLU_GEN_DATA_ROOT`` is distinct from the industrial ``CHLU_DATA_ROOT``,
    and a worktree has no ``.claude/``, so it must be set when running from one.
    """
    return config.gen_data.data_root


def artifacts_tag(dataset: str, pretext: str, condition: str) -> str:
    """``<dataset>_<pretext>`` for the headline condition, suffixed otherwise.

    ⛔ Back-compatible by construction: every Montero path already in the ledger
    stays valid, and two conditions can never overwrite each other's frozen
    artifacts (the load-bearing edit in `gen-wide-split` §6).
    """
    tag = f"{dataset}_{pretext}"
    return tag if condition == DEFAULT_CONDITION else f"{tag}_{condition}"


def condition_of(config: CHLUConfig) -> str:
    """The hold-out condition this run is fitted against (``gen_data.condition``)."""
    return str(getattr(config.gen_data, "condition", DEFAULT_CONDITION))


def load_split_and_cells(dataset: str, config: CHLUConfig, log: Callable = print):
    """Load the pinned split for ``gen_data.condition`` (⛔ never re-derive one here).

    Returns ``(split, cells)``; ``cells`` is ``None`` for the headline Montero
    condition and the list of :class:`~chlu.data.gen_splits.FactorCell` for a
    multi-cell secondary condition (``recomb2element_wide``, R17).
    """
    from ..data.gen_splits import assert_holdout_disjoint

    mod = _dataset_mod(dataset)
    condition = condition_of(config)
    cells = None
    if condition == DEFAULT_CONDITION:
        # ⚠ The two loaders' arities differ and always have:
        # ``shapes3d.build_split`` -> (split, labels); ``dsprites.build_split`` ->
        # (split, latent_values, latent_classes).  ⛔ The old ``split, _ = …``
        # unpack raised ``ValueError: too many values to unpack`` on dSprites —
        # i.e. **the dSprites bridge path had never been run end to end.**
        split = mod.build_split(_data_root(config))[0]
    elif condition == getattr(mod, "WIDE_CONDITION", None):
        gd = config.gen_data
        split, _, cells = mod.build_wide_split(
            _data_root(config),
            n_cells=int(gd.wide_n_cells),
            seed=int(gd.wide_seed),
            include_montero=bool(gd.wide_include_montero),
        )
    else:
        raise ValueError(
            f"unknown condition {condition!r} for dataset {dataset!r} "
            f"(have {DEFAULT_CONDITION!r}"
            + (
                f", {mod.WIDE_CONDITION!r})"
                if hasattr(mod, "WIDE_CONDITION")
                else ")"
            )
        )
    n_checked = assert_holdout_disjoint(split)
    log(
        f"  split {dataset}/{condition}: n_total={split.n_total} "
        f"n_train={split.n_train} n_test={split.n_test} "
        f"(hold-out assertion over {n_checked} train rows"
        + (f", {len(cells)} cells)" if cells is not None else ")")
    )
    return split, cells


def load_split(dataset: str, config: CHLUConfig, log: Callable = print):
    """Back-compatible wrapper: the split alone."""
    return load_split_and_cells(dataset, config, log)[0]


def per_cell_no_leak(split, cells, used_rows: np.ndarray) -> list:
    """R3.3 **per cell**: no fitted row may sit in any held-out cell, with counts.

    ⛔ A pooled assertion over an OR-predicate can pass while one disjunct is
    wrong; this re-derives every cell's rows from the grid and checks each.
    """
    from ..data.gen_splits import HoldoutViolation

    used = np.unique(np.asarray(used_rows, dtype=np.int64))
    out = []
    for cell in cells:
        rows = cell.rows(split.grid)
        leaked = np.intersect1d(rows, used, assume_unique=False)
        if leaked.size:
            raise HoldoutViolation(
                f"cell {cell.cell_id!r}: {leaked.size} fitted row(s) are held out "
                f"(first: {int(leaked[0])})"
            )
        out.append(
            {
                "cell_id": cell.cell_id,
                "n_rows": int(rows.size),
                "n_excluded": int(rows.size - leaked.size),
                "n_leaked": int(leaked.size),
                "contrast": float(cell.meta.get("object_bg_hue_contrast", float("nan"))),
                "shape_value": cell.meta.get("shape_value"),
            }
        )
    return out


def _dataset_mod(dataset: str):
    from ..core.gen_bridge import dataset_module

    return dataset_module(dataset)


def select_rows(split, cfg, seed: int) -> dict:
    """Deterministic, arm-independent row selection.

    ``train`` fits everything; ``eval_a`` is drawn from the SAME (seen) split but
    is held back from every fitting stage, so ceiling_A is measured on novel
    *images* of seen *combinations* — exactly matched to B (novel images, unseen
    combinations).  Every held-out row is in ``eval_b``, never subsampled.
    """
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(split.train_rows)
    n_train = min(int(cfg.n_train), perm.size)
    n_a = min(int(cfg.n_eval_a), perm.size - n_train)
    return {
        "train": np.sort(perm[:n_train]),
        "eval_a": np.sort(perm[n_train : n_train + n_a]),
        "eval_b": np.asarray(split.test_rows, np.int64),
    }


def cache_rows_for(split, rows: dict, scope: str) -> np.ndarray:
    """Rows the feature cache covers: the whole grid (``full``) or just what is used."""
    if scope == "full":
        return np.arange(split.n_total, dtype=np.int64)
    if scope == "used":
        return np.unique(
            np.concatenate([rows["train"], rows["eval_a"], rows["eval_b"]])
        )
    raise ValueError(f"cache_scope must be 'full' or 'used', got {scope!r}")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def run_encoder_stage(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    log: Callable = print,
) -> dict:
    """Fit + freeze the encoder, then write the feature cache and its manifest."""
    cfg = config.gen_bridge
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split, cells = load_split_and_cells(dataset, config, log)
    rows = select_rows(split, cfg, seed)
    grid = split.grid

    leak = assert_no_holdout_leak(split, rows["train"])
    log(
        f"  ⭐ R3.3 no-leak: {leak['n_checked']} training rows checked, "
        f"{leak['n_holdout_excluded']}/{leak['n_holdout_total']} held-out images "
        f"excluded, {leak['n_holdout_leaked']} leaked"
    )
    cell_leak = per_cell_no_leak(split, cells, rows["train"]) if cells else None
    if cell_leak is not None:
        log(
            f"  ⭐ R3.3 per cell: {len(cell_leak)} cells, "
            f"{sum(c['n_excluded'] for c in cell_leak)} rows excluded, "
            f"{sum(c['n_leaked'] for c in cell_leak)} leaked"
        )

    log(f"  streaming {rows['train'].size} train images …")
    imgs = stream_images(dataset, rows["train"], _data_root(config), int(cfg.cache_block))
    labels = factor_labels(grid, rows["train"])

    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_909)
    key, ek = jax.random.split(key)
    log(f"  fitting encoder (pretext={pretext}, {cfg.enc_steps} steps) …")
    encoder, aux, hist = train_encoder(
        imgs,
        pretext,
        cfg,
        ek,
        labels=labels,
        cardinalities=grid.cardinalities,
        log=log,
    )
    enc_hash = param_hash(encoder)
    save_model(
        encoder,
        out_dir / "encoder.pkl",
        {"pretext": pretext, "seed": int(seed), "param_hash": enc_hash},
    )
    # the pretext head/decoder is DISCARDED (it is not the shared renderer); its
    # hash is kept only so the pretext is auditable.
    aux_hash = param_hash(aux)

    scope = str(cfg.cache_scope)
    crows = cache_rows_for(split, rows, scope)
    log(f"  caching penultimate features: scope={scope}, {crows.size} rows …")
    cache = build_feature_cache(
        encoder, dataset, crows, out_dir, cfg, _data_root(config), scope=scope, log=log
    )

    manifest = {
        "schema": "chlu-gen-bridge-encoder/1",
        "dataset": dataset,
        "condition": condition_of(config),
        "pretext": pretext,
        "seed": int(seed),
        "encoder": {
            "param_hash": enc_hash,
            "param_count": param_count(encoder),
            "feat_shape": [int(x) for x in encoder.feat_shape],
            "D": int(encoder.feature_dim),
            "channels": list(cfg.enc_channels),
            "kernel": int(cfg.enc_kernel),
            "stride": int(cfg.enc_stride),
            "groups": int(cfg.enc_groups),
            "steps": int(cfg.enc_steps),
            "batch": int(cfg.enc_batch),
            "lr": float(cfg.enc_lr),
            "loss_first": float(hist[0]) if hist else None,
            "loss_final": float(hist[-1]) if hist else None,
        },
        "pretext_head": {"param_hash": aux_hash, "discarded": True},
        "rows": {k: int(v.size) for k, v in rows.items()},
        "no_leak": leak,
        "no_leak_per_cell": cell_leak,
        "cells": (
            [c.to_dict(grid) for c in cells] if cells is not None else None
        ),
        "feature_cache": cache,
        "whitening": whitening_feasibility(encoder.feature_dim, int(rows["train"].size)),
    }
    _write_json(manifest, out_dir / "encoder_manifest.json")
    log(
        f"  ⭐ D = {encoder.feature_dim} ({encoder.feat_shape}) — "
        f"{manifest['whitening']['note']}"
    )
    return manifest


def run_decoder_stage(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    encoder_dir: Optional[Path] = None,
    log: Callable = print,
) -> dict:
    """Fit + freeze THE shared decoder and the probe; measure the ceiling and gate."""
    cfg = config.gen_bridge
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = Path(encoder_dir) if encoder_dir is not None else out_dir
    split, cells = load_split_and_cells(dataset, config, log)
    rows = select_rows(split, cfg, seed)
    grid = split.grid
    names = grid.names

    leak = assert_no_holdout_leak(split, rows["train"])
    log(
        f"  ⭐ R3.3 no-leak: {leak['n_checked']} decoder-training rows checked, "
        f"{leak['n_holdout_excluded']}/{leak['n_holdout_total']} held-out images "
        f"excluded, {leak['n_holdout_leaked']} leaked"
    )
    cell_leak = per_cell_no_leak(split, cells, rows["train"]) if cells else None
    if cell_leak is not None:
        log(
            f"  ⭐ R3.3 per cell: {len(cell_leak)} cells, "
            f"{sum(c['n_excluded'] for c in cell_leak)} rows excluded, "
            f"{sum(c['n_leaked'] for c in cell_leak)} leaked"
        )

    encoder = load_model(encoder_dir / "encoder.pkl")
    with open(encoder_dir / "encoder_manifest.json") as fh:
        enc_manifest = json.load(fh)
    enc_hash = param_hash(encoder)
    if enc_hash != enc_manifest["encoder"]["param_hash"]:
        raise AssertionError("encoder checkpoint does not match its manifest hash")
    cache_rows, features, cache_manifest = load_feature_cache(
        encoder_dir, verify=bool(cfg.verify_cache_checksum)
    )
    if cache_manifest["encoder_param_hash"] != enc_hash:
        raise AssertionError("feature cache was written by a different encoder")

    # ONE contiguous pass for train + eval_A + eval_B: the HDF5 files are gzip
    # chunked (15000, 4, 4, 1), so every scattered read costs a full scan.
    all_rows = np.concatenate([rows["train"], rows["eval_a"], rows["eval_b"]])
    log(f"  streaming {all_rows.size} images (train + eval_A + eval_B, one pass) …")
    all_imgs = stream_images(
        dataset, all_rows, _data_root(config), int(cfg.cache_block)
    )
    n_tr, n_a = rows["train"].size, rows["eval_a"].size
    imgs = all_imgs[:n_tr]
    real_imgs = {"A": all_imgs[n_tr : n_tr + n_a], "B": all_imgs[n_tr + n_a :]}
    labels = factor_labels(grid, rows["train"])

    key = jax.random.PRNGKey(int(seed) * 10_007 + 20_260_910)
    key, dk, pk = jax.random.split(key, 3)
    log(f"  fitting THE shared decoder ({cfg.dec_steps} steps, encoder frozen) …")
    decoder, dec_hist = train_decoder(encoder, imgs, cfg, dk, log=log)
    dec_hash = param_hash(decoder)
    save_model(
        decoder,
        out_dir / "decoder.pkl",
        {"seed": int(seed), "param_hash": dec_hash, "encoder_param_hash": enc_hash},
    )

    log(f"  fitting the probe on REAL images ({cfg.probe_steps} steps) …")
    probe, probe_hist = train_probe(imgs, labels, grid.cardinalities, cfg, pk, log=log)
    probe_hash = param_hash(probe)
    save_model(probe, out_dir / "probe.pkl", {"seed": int(seed), "param_hash": probe_hash})
    del imgs

    # ---- the measurement ------------------------------------------------
    result = {"per_split": {}}
    for tag, key_ in (("A", "eval_a"), ("B", "eval_b")):
        r = rows[key_]
        real = real_imgs[tag]
        lab = factor_labels(grid, r)
        feats = features_for_rows(cache_rows, features, r)
        ceiling = measure_ceiling(
            decoder, probe, feats, lab, names, int(cfg.eval_batch)
        )
        real_acc = probe_accuracy(probe, real, lab, names, int(cfg.eval_batch))
        recon_mse = float(
            np.mean(
                (decode_features(decoder, feats, int(cfg.eval_batch)) - real) ** 2
            )
        )
        result["per_split"][tag] = {
            "n": int(r.size),
            "ceiling": ceiling,
            "probe_on_real": real_acc,
            "decode_mse": recon_mse,
        }
        log(
            f"  {tag}: n={r.size}  ceiling={ceiling['aggregate']:.4f}  "
            f"probe_on_real={real_acc['aggregate']:.4f}  decode_mse={recon_mse:.5f}"
        )

    ca = result["per_split"]["A"]["ceiling"]["aggregate"]
    cb = result["per_split"]["B"]["ceiling"]["aggregate"]
    gate = gate_verdict(ca, cb, float(cfg.ceiling_floor))
    result["gate"] = gate
    log(
        f"  ⭐ seed {seed}: ceiling_A={ca:.4f} ceiling_B={cb:.4f} "
        f"ratio={gate['ratio']:.4f} -> {gate['verdict']}"
    )

    manifest = {
        "schema": "chlu-gen-bridge/1",
        "dataset": dataset,
        "condition": condition_of(config),
        "pretext": pretext,
        "seed": int(seed),
        "encoder_dir": str(encoder_dir),
        "encoder_param_hash": enc_hash,
        # ⭐ downstream arms assert against THIS hash before scoring anything.
        "decoder_param_hash": dec_hash,
        "decoder_param_count": param_count(decoder),
        "probe_param_hash": probe_hash,
        "probe_param_count": param_count(probe),
        "feature_cache": cache_manifest,
        "no_leak": leak,
        "no_leak_per_cell": cell_leak,
        "rows": {k: int(v.size) for k, v in rows.items()},
        "D": int(encoder.feature_dim),
        "whitening": enc_manifest.get("whitening"),
        "training": {
            "dec_recipe": resolve_decoder_cfg(cfg).as_dict(),
            "dec_steps": int(cfg.dec_steps),
            "dec_loss_first": float(dec_hist[0]) if dec_hist else None,
            "dec_loss_final": float(dec_hist[-1]) if dec_hist else None,
            "probe_steps": int(cfg.probe_steps),
            "probe_loss_first": float(probe_hist[0]) if probe_hist else None,
            "probe_loss_final": float(probe_hist[-1]) if probe_hist else None,
        },
        "result": result,
    }
    _write_json(manifest, out_dir / "bridge_manifest.json")
    _write_json(
        {
            "decoder_param_hash": dec_hash,
            "encoder_param_hash": enc_hash,
            "probe_param_hash": probe_hash,
            "dataset": dataset,
            "pretext": pretext,
            "seed": int(seed),
            "note": (
                "THE shared frozen decoder. Every arm renders through this one. "
                "Assert with chlu.core.gen_bridge.assert_decoder_hash before scoring."
            ),
        },
        out_dir / "shared_decoder_hash.json",
    )

    if cfg.make_plots:
        _plot_ceiling_grid(real_imgs, decoder, cache_rows, features, rows, cfg,
                           out_dir / "ceiling_grid.png")
        _plot_ceiling_bars(result, names, float(cfg.ceiling_floor),
                           out_dir / "ceiling_bars.png")
    return manifest


# ---------------------------------------------------------------------------
# ⭐ dSPRITES — the SECOND dataset.  ADDITIVE ONLY.
#
# ⛔ Everything below is reached only when ``dataset == "dsprites"``.  The
# Shapes3D path above is untouched and is asserted bit-identical before/after by
# ``tests/test_gen_bridge_dsprites.py::test_shapes3d_path_is_bit_identical``.
#
# Two things the dSprites path needs that Shapes3D does not:
#   1. ⭐ the WINNING decoder recipe, pinned and asserted.  ``dec_recipe`` still
#      ships as ``"baseline"`` — the recipe that FAILED the Shapes3D gate — so a
#      dSprites run that inherits it looks normal in the logs and is **void**.
#   2. ⭐ R20's in-corner DISCRIMINATION read.  The dSprites held-out cell pins
#      ``latent_classes[:, shp] == 1`` (ellipse), i.e. it is a ONE-CLASS cell
#      exactly like Montero's ``shape == cylinder`` — so a probe that answers
#      "ellipse" to everything in that corner scores 1.000 on plain accuracy.
#      ⛔ The gate therefore reads BALANCED ACCURACY (R23.1) of held-out ellipses
#      against **matched-seen** other shapes in the same corner, with AUROC as a
#      mandatory companion and never as the gate.
# ---------------------------------------------------------------------------


def apply_dsprites_profile(config: CHLUConfig, quick: bool = False) -> CHLUConfig:
    """Pin the dSprites profile onto ``config.gen_bridge`` (⛔ dSprites only).

    Applies ``gen_bridge_dsprites.dec_recipe`` (⭐ R19.1's ``v1_capacity_schedule``)
    and the chunk-aligned HDF5 read block.  In ``quick`` mode the recipe falls
    back to ``quick_dec_recipe`` so a smoke run does not silently inherit the
    winner's 1,800-step schedule — ⚠ a quick run is void as a result either way.
    """
    dcfg = config.gen_bridge_dsprites
    gb = config.gen_bridge
    gb.dec_recipe = str(dcfg.quick_dec_recipe if quick else dcfg.dec_recipe)
    gb.cache_block = int(dcfg.cache_block)
    return config


def dsprites_populations(
    split,
    latent_values: np.ndarray,
    latent_classes: np.ndarray,
    fitted_rows: np.ndarray,
    dcfg,
    seed: int,
    quick: bool = False,
) -> dict:
    """⭐ R20's three populations for the dSprites Montero cell.

    * ``positives``   — the held-out ``shape``-class-1 (ellipse) rows.
    * ``negatives``   — **matched-seen**: the cell's whole conjunction with the
      ``shape`` clause relaxed to a different, SEEN shape, minus every fitted row
      and every held-out row.  ⭐ R16 made this a permanent companion; it is not
      optional here.
    * ``seen``        — the seen normaliser: the cell's *position* corner with its
      own shape (ellipse) but **outside the cell**, minus every fitted row.  This
      is ``gen-bridge-wide``'s definition transposed — it keeps the cell's context
      clause and relaxes exactly the two factors the cell also constrains
      (``scale``, ``orientation``).  Scored against the **same** negatives.

    ⛔ Nothing here re-derives the split: the held-out rows come from ``split``.
    Caps are seeded subsamples and the **realised** counts are what get reported.
    """
    from ..data import dsprites as ds

    target = int(ds.HELDOUT_SHAPE_CLASS)
    corner = ds.montero_corner_mask(latent_values)
    pos_corner = ds.montero_position_corner_mask(latent_values)
    is_target = latent_classes[:, ds.shp] == target

    test = np.asarray(split.test_rows, np.int64)
    fitted = np.unique(np.asarray(fitted_rows, np.int64))
    in_test = np.zeros(latent_values.shape[0], bool)
    in_test[test] = True
    in_fit = np.zeros(latent_values.shape[0], bool)
    in_fit[fitted] = True

    neg_all = np.where(corner & ~is_target & ~in_test & ~in_fit)[0]
    seen_all = np.where(pos_corner & is_target & ~in_test & ~in_fit)[0]

    cap_p = int(dcfg.quick_max_positives if quick else dcfg.max_positives)
    cap_n = int(dcfg.quick_max_negatives if quick else dcfg.max_negatives)
    cap_s = int(dcfg.quick_max_seen if quick else dcfg.max_seen)

    def take(pool, cap, salt):
        pool = np.asarray(pool, np.int64)
        if cap <= 0 or pool.size <= cap:
            return np.sort(pool)
        rng = np.random.default_rng(int(seed) * 104_729 + salt)
        return np.sort(rng.permutation(pool)[:cap])

    out = {
        "target": target,
        "positives": take(test, cap_p, 1),
        "negatives": take(neg_all, cap_n, 2),
        "seen": take(seen_all, cap_s, 3),
        "pool_sizes": {
            "corner_total": int(corner.sum()),
            "positives_pool": int(test.size),
            "negatives_pool": int(neg_all.size),
            "seen_pool": int(seen_all.size),
        },
    }
    # ⛔ The assertions that make the number mean what it says.
    for name in ("negatives", "seen"):
        r = out[name]
        if np.intersect1d(r, test).size:
            raise AssertionError(f"dSprites {name} overlap the held-out split")
        if np.intersect1d(r, fitted).size:
            raise AssertionError(f"dSprites {name} overlap the fitted rows")
    if np.intersect1d(out["positives"], fitted).size:
        raise AssertionError("dSprites positives overlap the fitted rows (R3.3 leak)")
    if out["negatives"].size < int(dcfg.min_negatives):
        raise AssertionError(
            f"only {out['negatives'].size} matched-seen negatives "
            f"(< {int(dcfg.min_negatives)}): a balanced accuracy over that is not "
            "a measurement"
        )
    if np.intersect1d(out["negatives"], out["seen"]).size:
        raise AssertionError("negatives and seen-normaliser positives overlap")
    return out


def features_cache_or_encoder(
    dataset: str,
    config: CHLUConfig,
    encoder,
    cache_rows: np.ndarray,
    features,
    cache_manifest: dict,
    rows: np.ndarray,
    log: Callable = print,
) -> np.ndarray:
    """Cached features for ``rows``; the **FROZEN** encoder for rows outside scope.

    ⚠ Measured constraint, not a choice (the ``gen-decoder-lift`` / ``exp_gen_store``
    precedent): a ``cache_scope="used"`` cache holds train + eval_A + eval_B and
    therefore **no** matched-seen row.  ⛔ This does **not** rebuild the bridge —
    the encoder is the bridge's own checkpoint, its parameter hash is asserted
    against the cache manifest by the caller, and freshly encoded rows are
    quantised to the cache's own dtype, so a side-buffer row is byte-for-byte
    what ``build_feature_cache`` would have written.
    """
    rows = np.asarray(rows, np.int64)
    if rows.size == 0:
        return np.zeros((0,) + tuple(encoder.feat_shape), np.float32)
    cache_rows = np.asarray(cache_rows, np.int64)
    if cache_rows.size == 0:
        hit = np.zeros(rows.size, bool)
        pos = np.zeros(rows.size, np.int64)
    else:
        pos = np.searchsorted(cache_rows, rows)
        safe = np.minimum(pos, cache_rows.size - 1)
        hit = (pos < cache_rows.size) & (cache_rows[safe] == rows)
    out = np.zeros((rows.size,) + tuple(encoder.feat_shape), np.float32)
    if hit.any():
        out[hit] = np.asarray(features[pos[hit]], np.float32)
    miss = ~hit
    if miss.any():
        dtype = np.dtype(cache_manifest.get("dtype", config.gen_bridge.cache_dtype))
        want = rows[miss]
        log(f"    side buffer: {want.size} row(s) outside the cache scope")
        imgs = stream_images(
            dataset, want, _data_root(config), int(config.gen_bridge.cache_block)
        )
        out[miss] = encode_images(
            encoder, imgs, int(config.gen_bridge.eval_batch)
        ).astype(dtype).astype(np.float32)
    return out


def run_discrimination_stage(
    dataset: str,
    pretext: str,
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    encoder_dir: Optional[Path] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    """⭐ THE GATE on the compositional term, R20's in-corner metric, UNCHANGED.

    Loads the frozen encoder / decoder / probe this seed already produced (⛔
    fits nothing), asserts the decoder was built by the pinned recipe, and reads
    balanced accuracy + AUROC of ``shape`` for held-out-vs-matched-seen and for
    the seen normaliser vs the SAME negatives.

    ⛔ ``FAIL`` is a legitimate outcome and returns to the Hub.  It is not a
    licence to tune.
    """
    if dataset != "dsprites":
        raise NotImplementedError(
            "the in-corner discrimination stage is wired for the dSprites Montero "
            f"cell; ⛔ do not invent one for {dataset!r} — the Shapes3D read lives "
            "in exp_gen_classifier / gen-bridge-wide.  Return to the Hub."
        )
    from ..core.gen_bridge import assert_decoder_hash
    from ..data import dsprites as ds

    cfg = config.gen_bridge
    dcfg = config.gen_bridge_dsprites
    out_dir = Path(out_dir)
    encoder_dir = Path(encoder_dir) if encoder_dir is not None else out_dir

    with open(out_dir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    got_recipe = str(bman["training"]["dec_recipe"]["name"])
    if got_recipe != str(dcfg.expected_decoder_recipe):
        msg = (
            f"⛔ decoder recipe {got_recipe!r} != expected "
            f"{str(dcfg.expected_decoder_recipe)!r}: this run is VOID "
            "(R19.1/R20 — 'baseline' is the recipe that FAILED the gate)"
        )
        # ⚠ A quick run deliberately uses ``quick_dec_recipe`` so a smoke test does
        # not inherit the winner's 1,800-step schedule.  Its numbers are plumbing
        # only and are never reported, so the assertion warns instead of raising —
        # ⛔ but ONLY under ``quick``.
        if not quick:
            raise AssertionError(msg)
        log(f"  ⚠ QUICK: {msg} — plumbing only, NOT a reported gate")

    encoder = load_model(encoder_dir / "encoder.pkl")
    decoder = load_model(out_dir / "decoder.pkl")
    probe = load_model(out_dir / "probe.pkl")
    enc_hash = param_hash(encoder)
    dec_hash = assert_decoder_hash(decoder, bman["decoder_param_hash"])
    probe_hash = param_hash(probe)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match the bridge manifest")
    if probe_hash != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest")

    split, _ = load_split_and_cells(dataset, config, log)
    rows = select_rows(split, cfg, seed)
    grid = split.grid
    names = list(grid.names)
    factor_index = names.index(str(dcfg.gate_factor))

    latent_values, latent_classes = ds.load_latents(_data_root(config))
    pops = dsprites_populations(
        split, latent_values, latent_classes, rows["train"], dcfg, seed, quick=quick
    )
    target = pops["target"]
    log(
        f"  populations: positives={pops['positives'].size} "
        f"negatives={pops['negatives'].size} seen={pops['seen'].size} "
        f"(pools {pops['pool_sizes']})"
    )

    cache_rows, features, cman = load_feature_cache(
        encoder_dir, verify=bool(cfg.verify_cache_checksum)
    )
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("feature cache was written by a different encoder")

    # ⭐ The side buffer is proved equal to the cache on rows the cache DOES hold,
    # before it is trusted on rows it does not.
    probe_rows = pops["positives"][: min(32, pops["positives"].size)]
    side = features_cache_or_encoder(
        dataset, config, encoder, np.zeros(0, np.int64), features, cman,
        probe_rows, log=lambda *_: None,
    )
    cached = features_for_rows(cache_rows, features, probe_rows)
    side_identical = bool(np.array_equal(side, cached))
    log(f"  side-buffer == cache on {probe_rows.size} cached rows: {side_identical}")

    eb = int(dcfg.eval_batch)
    out_pop = {}
    for name in ("positives", "negatives", "seen"):
        r = pops[name]
        feats = features_cache_or_encoder(
            dataset, config, encoder, cache_rows, features, cman, r, log=log
        )
        dec = decode_features(decoder, feats, eb)
        out_pop[name] = {
            "n": int(r.size),
            "probs": factor_probs(probe, dec, factor_index, eb),
            "preds": predict_factors(probe, dec, eb),
            "decoded_mse": None,
        }
        if bool(dcfg.real_instrument):
            real = stream_images(dataset, r, _data_root(config), int(cfg.cache_block))
            out_pop[name]["decoded_mse"] = float(np.mean((dec - real) ** 2))
            out_pop[name]["real_probs"] = factor_probs(probe, real, factor_index, eb)
            del real
        del feats, dec

    held = discrimination_metrics(
        out_pop["positives"]["probs"], out_pop["negatives"]["probs"],
        target, int(dcfg.min_negatives),
    )
    seen = discrimination_metrics(
        out_pop["seen"]["probs"], out_pop["negatives"]["probs"],
        target, int(dcfg.min_negatives),
    )
    verdict = discrimination_verdict(
        held["balanced_accuracy"], seen["balanced_accuracy"],
        float(dcfg.disc_floor), float(dcfg.disc_abs_floor),
    )
    result = {
        "schema": "chlu-gen-bridge-dsprites-discrimination/1",
        "dataset": dataset,
        "pretext": pretext,
        "seed": int(seed),
        "quick": bool(quick),
        "gate_factor": str(dcfg.gate_factor),
        "target_class": target,
        "encoder_param_hash": enc_hash,
        "decoder_param_hash": dec_hash,
        "decoder_recipe": bman["training"]["dec_recipe"],
        "decoder_param_count": bman["decoder_param_count"],
        "probe_param_hash": probe_hash,
        "feature_cache": {k: cman[k] for k in ("scope", "n_rows", "dtype", "features_sha256")},
        "side_buffer_identical_to_cache": side_identical,
        "populations": {k: int(pops[k].size) for k in ("positives", "negatives", "seen")},
        "pool_sizes": pops["pool_sizes"],
        "held_out": held,
        "seen_normaliser": seen,
        "verdict": verdict,
        "decoded_mse": {k: out_pop[k]["decoded_mse"] for k in out_pop},
        "no_leak": assert_no_holdout_leak(split, rows["train"]),
    }
    # ⭐ the 3-way companion (a DIFFERENT quantity, in its own row — R20)
    for name in ("positives", "negatives", "seen"):
        lab = factor_labels(grid, pops[name])[:, factor_index]
        result.setdefault("multiclass", {})[name] = multiclass_balanced_accuracy(
            lab, out_pop[name]["preds"][:, factor_index], int(grid.cardinalities[factor_index])
        )
    # ⭐ the instrument: the same metric on REAL images.  ⛔ If this is not ~1.0,
    # nothing decoded below it is interpretable.
    if bool(dcfg.real_instrument):
        result["real_images"] = discrimination_metrics(
            out_pop["positives"]["real_probs"], out_pop["negatives"]["real_probs"],
            target, int(dcfg.min_negatives),
        )
    log(
        f"  ⭐ seed {seed}: balacc held-out={held['balanced_accuracy']:.4f} "
        f"seen={seen['balanced_accuracy']:.4f} ratio={verdict['ratio']:.4f} "
        f"AUROC={held['auroc']:.4f} -> {verdict['verdict']}"
    )
    _write_json(result, out_dir / "discrimination.json")
    return result


def aggregate_discrimination(run_dirs, log: Callable = print) -> dict:
    """Aggregate per-seed ``discrimination.json`` into the reported gate verdict."""
    from ..eval.gen_probe import balanced_accuracy  # noqa: F401  (documented sibling)

    res = []
    for d in run_dirs:
        path = Path(d) / "discrimination.json"
        if not path.exists():
            log(f"  ⚠ {path} missing — seed reported NOT RUN, never as a zero")
            continue
        with open(path) as fh:
            res.append(json.load(fh))
    if not res:
        raise ValueError("no per-seed discrimination results to aggregate")

    def stat(getter):
        v = np.asarray([float(getter(r)) for r in res], np.float64)
        return {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "values": [float(x) for x in v],
        }

    out = {
        "schema": "chlu-gen-bridge-dsprites-aggregate/1",
        "dataset": res[0]["dataset"],
        "pretext": res[0]["pretext"],
        "gate_factor": res[0]["gate_factor"],
        "seeds": [r["seed"] for r in res],
        "decoder_recipe": res[0]["decoder_recipe"]["name"],
        "decoder_param_hashes": sorted({r["decoder_param_hash"] for r in res}),
        "balacc_held_out": stat(lambda r: r["held_out"]["balanced_accuracy"]),
        "balacc_seen": stat(lambda r: r["seen_normaliser"]["balanced_accuracy"]),
        "auroc_held_out": stat(lambda r: r["held_out"]["auroc"]),
        "auroc_seen": stat(lambda r: r["seen_normaliser"]["auroc"]),
        "tpr_held_out": stat(lambda r: r["held_out"]["tpr"]),
        "tnr_held_out": stat(lambda r: r["held_out"]["tnr"]),
        "ratio": stat(lambda r: r["verdict"]["ratio"]),
        "per_seed_verdicts": [r["verdict"]["verdict"] for r in res],
        "populations": [r["populations"] for r in res],
    }
    if all("real_images" in r for r in res):
        out["balacc_real_images"] = stat(
            lambda r: r["real_images"]["balanced_accuracy"]
        )
    out["gate"] = discrimination_verdict(
        out["balacc_held_out"]["mean"],
        out["balacc_seen"]["mean"],
        float(res[0]["verdict"]["floor"]),
        float(res[0]["verdict"]["abs_floor"]),
    )
    log(
        f"⭐ dSPRITES GATE (seeds {out['seeds']}): balacc held-out="
        f"{out['balacc_held_out']['mean']:.4f}±{out['balacc_held_out']['std']:.4f} "
        f"seen={out['balacc_seen']['mean']:.4f}±{out['balacc_seen']['std']:.4f} "
        f"ratio={out['gate']['ratio']:.4f} (floor {out['gate']['floor']}, abs "
        f"{out['gate']['abs_floor']}) -> {out['gate']['verdict']} "
        f"[per seed {out['per_seed_verdicts']}]"
    )
    return out


def aggregate_runs(run_dirs, floor: float, log: Callable = print) -> dict:
    """Aggregate per-seed ``bridge_manifest.json`` into the reported gate verdict."""
    mans = []
    for d in run_dirs:
        with open(Path(d) / "bridge_manifest.json") as fh:
            mans.append(json.load(fh))
    if not mans:
        raise ValueError("no per-seed runs to aggregate")
    names = list(mans[0]["result"]["per_split"]["A"]["ceiling"]["per_factor"].keys())
    dec_hashes = sorted({m["decoder_param_hash"] for m in mans})
    out = {
        "schema": "chlu-gen-bridge-aggregate/1",
        "dataset": mans[0]["dataset"],
        "condition": mans[0].get("condition", DEFAULT_CONDITION),
        "pretext": mans[0]["pretext"],
        "seeds": [m["seed"] for m in mans],
        "D": mans[0]["D"],
        "whitening": mans[0].get("whitening"),
        "decoder_param_hashes": dec_hashes,
        "no_leak": {
            "n_holdout_excluded": [m["no_leak"]["n_holdout_excluded"] for m in mans],
            "n_holdout_leaked": [m["no_leak"]["n_holdout_leaked"] for m in mans],
            "n_checked": [m["no_leak"]["n_checked"] for m in mans],
        },
    }
    for tag in ("A", "B"):
        cs = [m["result"]["per_split"][tag]["ceiling"] for m in mans]
        rs = [m["result"]["per_split"][tag]["probe_on_real"] for m in mans]
        out[f"ceiling_{tag}"] = summarize_seeds(cs, "aggregate")
        out[f"ceiling_{tag}_per_factor"] = summarize_per_factor(cs, names)
        out[f"probe_on_real_{tag}"] = summarize_seeds(rs, "aggregate")
        out[f"probe_on_real_{tag}_per_factor"] = summarize_per_factor(rs, names)
        out[f"n_{tag}"] = mans[0]["result"]["per_split"][tag]["n"]
    ratios = np.asarray(
        [m["result"]["gate"]["ratio"] for m in mans], np.float64
    )
    out["ratio"] = {
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "values": [float(r) for r in ratios],
    }
    out["gate"] = gate_verdict(
        out["ceiling_A"]["mean"], out["ceiling_B"]["mean"], floor
    )
    out["per_seed_verdicts"] = [m["result"]["gate"]["verdict"] for m in mans]
    # the instrument control: is the B corner simply harder for the PROBE?
    out["probe_ratio_B_over_A"] = float(
        out["probe_on_real_B"]["mean"] / out["probe_on_real_A"]["mean"]
        if out["probe_on_real_A"]["mean"] > 0
        else float("nan")
    )
    log(
        f"⭐ GATE ({out['dataset']}/{out['pretext']}, seeds {out['seeds']}): "
        f"ceiling_A={out['ceiling_A']['mean']:.4f}±{out['ceiling_A']['std']:.4f} "
        f"ceiling_B={out['ceiling_B']['mean']:.4f}±{out['ceiling_B']['std']:.4f} "
        f"ratio={out['gate']['ratio']:.4f} floor={floor} -> {out['gate']['verdict']}"
    )
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_experiment_gen_bridge(
    config: Optional[CHLUConfig] = None,
    dataset: Optional[str] = None,
    pretext: Optional[str] = None,
    out_dir: Optional[str] = None,
    seeds=None,
    quick: bool = False,
    discrimination: Optional[bool] = None,
    log: Callable = print,
) -> dict:
    """Config-load -> data -> init -> train-or-load -> evaluate -> plot, over seeds."""
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    cfg = config.gen_bridge
    dataset = dataset or cfg.dataset
    # ⭐ dSprites ONLY (additive).  ⛔ The Shapes3D path never reaches this line,
    # so every Shapes3D default — including ``dec_recipe="baseline"`` — is exactly
    # as shipped.  On dSprites the pinned recipe is R19.1's winner, asserted per
    # seed by the discrimination stage against the frozen decoder's manifest.
    if dataset == "dsprites":
        config = apply_dsprites_profile(config, quick=quick)
        cfg = config.gen_bridge
        log(f"[gen-bridge] ⭐ dSprites profile: dec_recipe={cfg.dec_recipe!r} "
            f"cache_block={cfg.cache_block}")
    # ⭐ Default ON for dSprites: the gate on that dataset IS the in-corner
    # discrimination read (R20/R23.1), so it is not an opt-in extra.  ⛔ Always
    # OFF for Shapes3D — that read belongs to gen-bridge-wide / exp_gen_classifier.
    if discrimination is None:
        discrimination = dataset == "dsprites"
    pretext = pretext or cfg.pretext
    if pretext not in PRETEXTS:
        raise ValueError(f"pretext must be one of {PRETEXTS}")
    seeds = list(seeds) if seeds is not None else list(cfg.seeds)
    condition = condition_of(config)
    # ⛔ Back-compatible by construction: the headline condition keeps its path,
    # a secondary condition gets its own, so the two configurations can never
    # overwrite each other's frozen artifacts.
    root = Path(out_dir or Path("results") / "gen_bridge") / artifacts_tag(
        dataset, pretext, condition
    )
    root.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for seed in seeds:
        d = root / f"seed_{seed}"
        log(f"[gen-bridge] {dataset}/{pretext}/{condition} seed={seed} -> {d}")
        if (d / "encoder_manifest.json").exists():
            log("  encoder artifacts present — reusing (train-or-load)")
        else:
            run_encoder_stage(dataset, pretext, seed, d, config, log=log)
        if (d / "bridge_manifest.json").exists():
            log("  bridge artifacts present — reusing (train-or-load)")
        else:
            run_decoder_stage(dataset, pretext, seed, d, config, encoder_dir=d, log=log)
        # ⭐ dSprites: the gate reads the COMPOSITIONAL term, so the in-corner
        # discrimination read is part of the per-seed unit of work and is written
        # incrementally (skip-if-exists) — a run that dies at seed 2 still leaves
        # seeds 0 and 1 fully reported.
        if discrimination and dataset == "dsprites":
            if (d / "discrimination.json").exists():
                log("  discrimination artifacts present — reusing")
            else:
                run_discrimination_stage(
                    dataset, pretext, seed, d, config,
                    encoder_dir=d, quick=quick, log=log,
                )
        run_dirs.append(d)

    agg = aggregate_runs(run_dirs, float(cfg.ceiling_floor), log=log)
    agg["quick"] = bool(quick)
    if quick:
        agg["WARNING"] = "quick run — plumbing only, NOT a reported ceiling"
    _write_json(agg, root / "ceiling.json")
    _write_json(agg["gate"], root / "gate.json")
    if discrimination and dataset == "dsprites":
        dagg = aggregate_discrimination(run_dirs, log=log)
        dagg["quick"] = bool(quick)
        if quick:
            dagg["WARNING"] = "quick run — plumbing only, NOT a reported gate"
        _write_json(dagg, root / "discrimination.json")
        _write_json(dagg["gate"], root / "discrimination_gate.json")
        agg["discrimination"] = dagg
    return agg


# ---------------------------------------------------------------------------
# Plots / IO
# ---------------------------------------------------------------------------


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


def _show(ax, img):
    a = np.transpose(np.asarray(img), (1, 2, 0))
    ax.imshow(a[:, :, 0], cmap="gray", vmin=0, vmax=1) if a.shape[2] == 1 else ax.imshow(
        np.clip(a, 0, 1)
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_ceiling_grid(real_imgs, decoder, cache_rows, features, rows, cfg, path, n=8):
    fig, axes = plt.subplots(4, n, figsize=(1.5 * n, 6.4))
    for col_block, (tag, key_) in enumerate((("A", "eval_a"), ("B", "eval_b"))):
        r = rows[key_][:n]
        feats = features_for_rows(cache_rows, features, r)
        dec = decode_features(decoder, feats, int(cfg.eval_batch))
        for j in range(min(n, r.size)):
            _show(axes[2 * col_block, j], real_imgs[tag][j])
            _show(axes[2 * col_block + 1, j], dec[j])
        axes[2 * col_block, 0].set_ylabel(f"real {tag}", fontsize=9)
        axes[2 * col_block + 1, 0].set_ylabel(f"decoded {tag}", fontsize=9)
    fig.suptitle(
        "real image vs. real cached features decoded through THE shared frozen decoder\n"
        "(A = seen combinations, unseen images · B = held-out combinations)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_ceiling_bars(result, names, floor, path):
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 3.5, 4.0))
    x = np.arange(len(names))
    a = [result["per_split"]["A"]["ceiling"]["per_factor"][str(n)] for n in names]
    b = [result["per_split"]["B"]["ceiling"]["per_factor"][str(n)] for n in names]
    ax.bar(x - 0.2, a, 0.4, label="ceiling_A (seen combos)", color="#3b6ea5")
    ax.bar(x + 0.2, b, 0.4, label="ceiling_B (held-out combos)", color="#c0603c")
    agg_a = result["per_split"]["A"]["ceiling"]["aggregate"]
    ax.axhline(agg_a, ls="--", lw=1, color="#3b6ea5", label=f"ceiling_A = {agg_a:.3f}")
    ax.axhline(
        floor * agg_a, ls=":", lw=1.4, color="k", label=f"gate floor = {floor}·ceiling_A"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in names], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("probe top-1 accuracy on decoded images")
    ax.set_title("decoder ceiling, per factor — the gate is read on the aggregate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI CONTRACT (binding, from gen-csf3-launcher):
#   --dataset --pretext --stage --arm --seed --out
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chlu.experiments.exp_gen_bridge",
        description="frozen encoder / feature cache / ONE shared frozen decoder + the ceiling gate",
    )
    p.add_argument("--dataset", default=None, choices=["shapes3d", "dsprites"])
    p.add_argument("--pretext", default=None, choices=list(PRETEXTS))
    p.add_argument(
        "--condition", default=None,
        help="hold-out condition (default: gen_data.condition = recomb2element, "
             "Montero's headline). ⛔ A non-default condition is a SEPARATE full "
             "configuration: encoder, cache, decoder and probe are re-fitted with "
             "its cells excluded, into their own artifacts directory.",
    )
    p.add_argument(
        "--stage", default="all", choices=list(STAGES) + list(NOT_MINE),
        help="encoder | decoder | all | aggregate | discrimination "
             "(⭐ discrimination = the dSprites in-corner gate, R20; "
             "store/cell belong to other spokes)",
    )
    p.add_argument(
        "--arm", default="clu",
        help="recorded in the manifest ONLY — the bridge is arm-independent by "
             "construction (every draw is threaded from --seed)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="artifacts directory")
    # extras (passed through EXTRA_ARGS on the cluster)
    p.add_argument("--quick", action="store_true", help="plumbing smoke; NOT a result")
    p.add_argument("--encoder-dir", default=None,
                   help="where the encoder stage wrote its artifacts "
                        "(default: <out>/../../encoder/<arm>_s<seed>/artifacts, else <out>)")
    p.add_argument("--data-root", default=None, help="overrides $CHLU_GEN_DATA_ROOT")
    p.add_argument("--cache-scope", default=None, choices=["full", "used"])
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--enc-steps", type=int, default=None)
    p.add_argument("--dec-steps", type=int, default=None)
    p.add_argument("--probe-steps", type=int, default=None)
    p.add_argument("--runs", nargs="+", default=None,
                   help="per-seed run dirs for --stage aggregate")
    p.add_argument("--no-plots", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    if args.quick:
        config = apply_quick(config)
    gb = config.gen_bridge
    if args.condition:
        config.gen_data.condition = args.condition
    if args.data_root:
        config.gen_data.data_root = args.data_root
    if args.cache_scope:
        gb.cache_scope = args.cache_scope
    for attr, val in (
        ("n_train", args.n_train),
        ("enc_steps", args.enc_steps),
        ("dec_steps", args.dec_steps),
        ("probe_steps", args.probe_steps),
    ):
        if val is not None:
            setattr(gb, attr, int(val))
    if args.no_plots:
        gb.make_plots = False
    return config


def _default_encoder_dir(out: Path, arm: str, seed: int) -> Path:
    """The launcher's sibling layout: ``<ws>/<stage>/<arm>_s<seed>/artifacts``."""
    sibling = out.parent.parent.parent / "encoder" / f"{arm}_s{seed}" / "artifacts"
    return sibling if (sibling / "encoder.pkl").exists() else out


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.stage in NOT_MINE:
        print(
            f"[gen-bridge] stage={args.stage} is not this module's "
            "(store -> exp_gen_store, cell -> exp_gen_cell). Refusing to no-op.",
            file=sys.stderr,
        )
        return 2
    config = _config_from_args(args)
    cfg = config.gen_bridge
    dataset = args.dataset or cfg.dataset
    pretext = args.pretext or cfg.pretext
    if dataset == "dsprites":
        config = apply_dsprites_profile(config, quick=bool(args.quick))
        cfg = config.gen_bridge
        print(f"[gen-bridge] ⭐ dSprites profile: dec_recipe={cfg.dec_recipe!r} "
              f"cache_block={cfg.cache_block}")
    condition = condition_of(config)
    tag = artifacts_tag(dataset, pretext, condition)
    out = Path(args.out or (Path("results") / "gen_bridge" / tag / f"seed_{args.seed}"))
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[gen-bridge] dataset={dataset} condition={condition} pretext={pretext} "
        f"stage={args.stage} arm={args.arm} seed={args.seed} out={out}"
    )
    print("[gen-bridge] ⛔ arm is metadata only: the bridge is arm-independent.")

    if args.stage in ("encoder", "all"):
        run_encoder_stage(dataset, pretext, args.seed, out, config)
    if args.stage in ("decoder", "all"):
        enc_dir = (
            Path(args.encoder_dir)
            if args.encoder_dir
            else _default_encoder_dir(out, args.arm, args.seed)
        )
        run_decoder_stage(
            dataset, pretext, args.seed, out, config, encoder_dir=enc_dir
        )
    if args.stage == "discrimination":
        enc_dir = (
            Path(args.encoder_dir)
            if args.encoder_dir
            else _default_encoder_dir(out, args.arm, args.seed)
        )
        run_discrimination_stage(
            dataset, pretext, args.seed, out, config,
            encoder_dir=enc_dir, quick=bool(args.quick),
        )
    if args.stage == "aggregate":
        if not args.runs:
            print("[gen-bridge] --stage aggregate needs --runs", file=sys.stderr)
            return 2
        agg = aggregate_runs(args.runs, float(cfg.ceiling_floor))
        _write_json(agg, out / "ceiling.json")
        _write_json(agg["gate"], out / "gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
