"""Experiment: ⭐ **the external classifier** — the instrument that reads IMAGES.

⭐ DIAL DECLARATION — *the instrument*.  ⛔ **No arm, no CLU number, no kill-rule
verdict appears in this module.**  It builds a measuring device, gates it, and
reports what it says about the frozen decoder's renders.

- **Laundering control:** independence of the decoder, **demonstrated** rather than
  asserted — the two instruments are compared where they can actually disagree
  (leg 3, on the decoder's own renders), not on architecture.
- **Falsifies:** the phase's whole ceiling stack.  ⭐ If the external classifier
  scores the decoder's renders materially below what the linear probe reports,
  every ceiling number in phase 1 is optimistic — **a finding, not a failure**.
- **Does NOT falsify:** the external classifier scoring below the probe on REAL
  images.  That is an instrument defect and the **gate** catches it (leg 2).

**The three legs.**

1. **Build** — :class:`~chlu.eval.gen_classifier.ExternalFactorCNN` over raw images,
   fitted on **Shapes3D train-split rows only**, disjoint from everything the bridge
   fitted and from the whole photometric corner.  ⛔ It never sees a decoder render
   and ⛔ never sees a held-out combination; both are asserted, and the render
   assertion is a **bit-identity check against the HDF5 file**, so it can fire.
2. ⭐ **The gate** — in-corner ``shape`` balanced accuracy on **REAL** held-out
   images must clear ``gate_floor`` on **every** seed.  ⛔ On FAIL the run stops
   before leg 3 and returns to the Hub; ⛔ it does not tune toward a pass.
3. ⭐⭐ **The comparison that can kill the phase's ceilings** — external classifier
   vs the bridge's own linear probe, on **the same rendered arrays**, per seed, with
   the gap.  ⛔ This module reports the gap; it does **not** adjudicate which
   instrument is right.

⛔ **THE DECODER TRAP, CHECKED FIRST.**  ``GenBridgeConfig.dec_recipe`` defaults to
``"baseline"`` — the decoder that FAILED the gate (0.7191, 0/3 seeds).  The winner
is ``v1_capacity_schedule`` (0.9414, 3/3).  A run through the baseline looks
entirely normal in the logs and is **void**, so this module loads
``gen-decoder-lift``'s winner and asserts **both** its parameter hash and its recipe
name before recording a number.

⭐ **Incremental by construction:** every seed writes its own manifest and its own
checkpoint, a seed whose manifest already exists is **skipped**, and the aggregate
writes ``classifier_gate.json`` **last** — a downstream spoke gates on that exact
file and its ``"gate"`` key.

Runnable::

    python -m chlu.experiments.exp_gen_classifier --stage all --out DIR [--quick]
    python -m chlu.experiments.exp_gen_classifier --stage fit --seed 0 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CHLUConfig, get_default_config  # noqa: E402
from ..core.gen_bridge import param_count, param_hash  # noqa: E402
from ..core.gen_dream import assert_dream_decoder, load_winner_hashes  # noqa: E402
from ..eval.gen_classifier import (  # noqa: E402
    assert_no_render_leak,
    classifier_rows,
    gate_verdict,
    instrument_gap,
    no_leak_report,
    train_external_classifier,
)
from ..eval.gen_probe import (  # noqa: E402
    decode_features,
    discrimination_metrics,
    factor_labels,
    factor_probs,
    probe_accuracy,
)
from ..experiments.exp_gen_store import (  # noqa: E402
    _assert_cache_reencodes,
    gather_features,
    matched_seen_rows,
    select_store_rows,
)
from ..utils.checkpoints import load_model, save_model  # noqa: E402

STAGES = ("fit", "aggregate", "all")
#: The two instruments, in the order every table uses them.
INSTRUMENTS = ("external", "probe")
#: Evaluation sets.  ``MS`` = R16's permanent matched-seen companion.
EVAL_SETS = ("A", "B", "MS")


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------


def apply_quick(config: CHLUConfig) -> CHLUConfig:
    """⚠ Plumbing smoke only — ⛔ a quick number is never reported."""
    c = config.gen_classifier
    c.seeds = list(c.quick_seeds)
    c.n_clf_train = int(c.quick_n_clf_train)
    c.clf_steps = int(c.quick_clf_steps)
    c.clf_checkpoint_every = int(c.quick_clf_checkpoint_every)
    c.render_check_n = int(c.quick_render_check_n)
    c.verify_cache_checksum = bool(c.quick_verify_cache_checksum)
    return config


def _claude_root(config: CHLUConfig) -> Path:
    r = config.gen_classifier.claude_root or os.environ.get("CHLU_CLAUDE_ROOT")
    if r:
        return Path(r).expanduser()
    return Path(__file__).resolve().parents[2] / ".claude"


def bridge_dir(config: CHLUConfig, dataset: str, pretext: str, seed: int) -> Path:
    """⚠ A worktree has no ``.claude/``, so ``$CHLU_GEN_BRIDGE_ROOT`` must be set
    when running from one (same trap as ``$CHLU_GEN_DATA_ROOT``)."""
    r = config.gen_classifier.bridge_root or os.environ.get("CHLU_GEN_BRIDGE_ROOT")
    root = Path(r).expanduser() if r else _claude_root(config) / "outputs" / "gen-feature-bridge" / "runs"
    d = root / f"{dataset}_{pretext}" / f"seed_{int(seed)}"
    if not (d / "bridge_manifest.json").exists():
        raise FileNotFoundError(
            f"no frozen bridge at {d} — this spoke consumes the FROZEN bridge and "
            "⛔ never rebuilds it (set gen_classifier.bridge_root or $CHLU_GEN_BRIDGE_ROOT)"
        )
    return d


def _write_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
        fh.write("\n")
    return path


def _ms(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], np.float64)
    return {
        "mean": float(v.mean()) if v.size else float("nan"),
        "std": float(v.std()) if v.size else float("nan"),
        "n": int(v.size),
        "per_seed": [float(x) for x in v],
    }


# ---------------------------------------------------------------------------
# ⛔ The decoder guard — resolved and asserted BEFORE anything is measured
# ---------------------------------------------------------------------------


def resolve_decoder(config: CHLUConfig, bdir: Path, seed: int, log: Callable = print):
    """Load the frozen decoder for this seed and **prove** it is the gate winner."""
    cfg = config.gen_classifier
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    source = str(cfg.decoder_source)
    if source == "winner":
        root = (
            Path(cfg.winner_root).expanduser()
            if cfg.winner_root
            else _claude_root(config) / "outputs" / "gen-decoder-lift" / "winner"
        )
        rec = load_winner_hashes(root / "winner_hashes.json")[f"seed_{int(seed)}"]
        path = root / cfg.winner_decoder_pattern.format(seed=int(seed))
        dec = load_model(path)
        info = assert_dream_decoder(
            dec, rec["param_hash"], rec["recipe"]["name"], cfg.expected_decoder_recipe
        )
        enc_expect = rec.get("encoder_param_hash_it_was_fit_against")
        if enc_expect and enc_expect != bman["encoder_param_hash"]:
            raise AssertionError(
                "the decoder-lift winner for this seed was fitted against encoder "
                f"{enc_expect}, but the bridge here is {bman['encoder_param_hash']} "
                "— ⛔ these are different feature spaces"
            )
    elif source == "bridge":
        recipe = (bman.get("training") or {}).get("dec_recipe") or {}
        name = recipe.get("name") if isinstance(recipe, dict) else recipe
        if name is None:
            raise AssertionError(
                f"{bdir}/bridge_manifest.json records no decoder recipe, so this decoder "
                "cannot be proven to be the gate winner. ⛔ Refusing: a run through the "
                "superseded 'baseline' recipe is void."
            )
        dec = load_model(bdir / "decoder.pkl")
        info = assert_dream_decoder(
            dec, bman["decoder_param_hash"], name, cfg.expected_decoder_recipe
        )
        path = bdir / "decoder.pkl"
    else:
        raise ValueError(f"decoder_source must be 'winner' or 'bridge', got {source!r}")
    info.update({
        "decoder_path": str(path),
        "decoder_source": source,
        "param_count": int(param_count(dec)),
    })
    log(
        f"  ⭐ decoder: {source} recipe={info['decoder_recipe']} "
        f"hash={info['decoder_param_hash'][:16]}… (hash + recipe asserted)"
    )
    return dec, info, bman


# ---------------------------------------------------------------------------
# Scoring — ⛔ the SAME functions read both instruments
# ---------------------------------------------------------------------------


def read_both(external, probe, images: np.ndarray, labels, names, batch: int) -> dict:
    """Both instruments over **one** image array.

    ⛔ The array is rendered (or streamed) once and handed to both readers, so a gap
    between them can never be a difference in what they looked at.
    """
    return {
        "external": probe_accuracy(external, images, labels, names, int(batch)),
        "probe": probe_accuracy(probe, images, labels, names, int(batch)),
    }


def corner_read(model, pos_images, neg_images, factor_index: int, target: int,
                batch: int, min_negatives: int) -> dict:
    """⭐ The in-corner discrimination statistic (R20/R23) for ONE instrument.

    Positives = the held-out cell (all of class ``target``); negatives = matched-seen
    items in the same corner with a different, SEEN class.  Balanced accuracy is the
    statistic, AUROC the mandatory companion, chance 0.5.
    """
    pp = factor_probs(model, pos_images, int(factor_index), int(batch))
    pn = factor_probs(model, neg_images, int(factor_index), int(batch))
    return discrimination_metrics(pp, pn, int(target), int(min_negatives))


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def run_seed(
    seed: int,
    out_dir: Path,
    config: CHLUConfig,
    log: Callable = print,
) -> dict:
    """⭐ Legs 1-3 for one seed, written incrementally.  A finished seed is skipped."""
    cfg = config.gen_classifier
    dataset, pretext = str(cfg.dataset), str(cfg.pretext)
    sdir = Path(out_dir) / f"seed_{int(seed)}"
    sdir.mkdir(parents=True, exist_ok=True)
    man_path = sdir / "classifier_manifest.json"
    if man_path.exists():
        log(f"  ⏭ seed {seed}: {man_path.name} exists — skipping (incremental)")
        with open(man_path) as fh:
            return json.load(fh)

    t0 = time.time()
    from ..data import shapes3d as s3d

    root = config.gen_data.data_root
    split, values = s3d.build_split(root)
    log(
        f"  split {dataset}: n_total={split.n_total} n_train={split.train_rows.size} "
        f"n_test={split.test_rows.size}"
    )

    # ---- leg 1: rows ------------------------------------------------------
    extra = None
    wide_n = 0
    if bool(cfg.exclude_wide_heldout):
        wsplit, _, _ = s3d.build_wide_split(root)
        extra = np.asarray(wsplit.test_rows, np.int64)
        wide_n = int(extra.size)
    rows = classifier_rows(split, values, cfg, config.gen_bridge, int(seed), extra)
    store_rows = select_store_rows(split, config, int(seed))
    ms_rows = matched_seen_rows(dataset, split, values, store_rows["store_train"])
    log(
        f"  rows: clf_train={rows['clf_train'].size} (pool {rows['pool_size']}, "
        f"{rows['n_excluded']} excluded incl. {rows['corner'].size} corner + "
        f"{wide_n} wide held-out) · eval_a={rows['eval_a'].size} "
        f"eval_b={rows['eval_b'].size} matched_seen={ms_rows.size}"
    )

    # ---- the frozen bridge ------------------------------------------------
    from ..core.gen_bridge import load_feature_cache

    bdir = bridge_dir(config, dataset, pretext, int(seed))
    encoder = load_model(bdir / "encoder.pkl")
    probe = load_model(bdir / "probe.pkl")
    with open(bdir / "bridge_manifest.json") as fh:
        bman = json.load(fh)
    enc_hash, probe_hash = param_hash(encoder), param_hash(probe)
    if enc_hash != bman["encoder_param_hash"]:
        raise AssertionError("encoder checkpoint does not match its manifest hash")
    if probe_hash != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match its manifest hash")
    cache_rows, features, cman = load_feature_cache(bdir, verify=bool(cfg.verify_cache_checksum))
    if cman["encoder_param_hash"] != enc_hash:
        raise AssertionError("feature cache was written by a different encoder")
    log(
        f"  ⭐ FROZEN: encoder {enc_hash[:16]}… probe {probe_hash[:16]}… "
        f"cache scope={cman['scope']} n={cman['n_rows']}"
    )
    # ⭐ ENFORCED. ⛔ No coverage gate in this module, and `gather_features` (leg 2)
    # extends the cache for any row it does not hold — the hazard is live here.
    reencode = _assert_cache_reencodes(
        dataset, config, encoder, cache_rows, features, cman, log=log,
        will_extend=True,
        reason="exp_gen_classifier has no cache-coverage gate; gather_features may extend",
    )
    decoder, dec_info, _ = resolve_decoder(config, bdir, int(seed), log)

    grid = split.grid
    names = list(grid.names)
    fidx = names.index(str(cfg.gate_factor))

    # ---- ONE contiguous image pass ---------------------------------------
    from ..core.gen_bridge import stream_images

    order = [rows["clf_train"], rows["eval_a"], rows["eval_b"], ms_rows]
    allr = np.concatenate(order)
    log(f"  streaming {allr.size} images (clf_train + A + B + MS, one pass) …")
    t_s = time.time()
    imgs = stream_images(dataset, allr, root, int(config.gen_bridge.cache_block))
    log(f"  streamed in {time.time() - t_s:.0f}s")
    o = np.cumsum([0] + [int(r.size) for r in order])
    train_imgs = imgs[o[0]:o[1]]
    real = {"A": imgs[o[1]:o[2]], "B": imgs[o[2]:o[3]], "MS": imgs[o[3]:o[4]]}

    # ---- leg 1 assertions (all three, and they can fire) ------------------
    render = assert_no_render_leak(
        train_imgs, rows["clf_train"], dataset, root,
        int(cfg.render_check_n), int(config.gen_bridge.cache_block), int(seed),
    )
    leak = no_leak_report(split, rows["clf_train"], rows, render)
    log(
        f"  ⭐ no-leak: {leak['holdout']['n_checked']} training rows checked, "
        f"{leak['holdout']['n_holdout_excluded']}/{leak['holdout']['n_holdout_total']} "
        f"held-out-combination images excluded, {leak['holdout']['n_holdout_leaked']} leaked; "
        f"{render['n_render_rows']} decoder renders in training "
        f"({render['n_checked']} rows bit-identical to the h5); "
        f"lineage overlap with bridge_train/eval_a/corner = "
        f"{leak['lineage']['n_overlap_bridge_train']}/"
        f"{leak['lineage']['n_overlap_eval_a']}/{leak['lineage']['n_overlap_corner']}"
    )

    # ---- build: fit the instrument ---------------------------------------
    ckpt = sdir / "external_classifier.pkl"
    labels_train = factor_labels(grid, rows["clf_train"])
    key = jax.random.PRNGKey(int(seed) * int(cfg.row_seed_mult) + int(cfg.row_seed_offset))
    key, ck = jax.random.split(key)
    log(f"  fitting the EXTERNAL classifier on REAL images ({cfg.clf_steps} steps) …")

    def _ckpt(step, model, loss):
        save_model(model, ckpt, {"seed": int(seed), "step": int(step), "loss": float(loss)})

    external, hist, key = train_external_classifier(
        train_imgs, labels_train, grid.cardinalities, cfg, ck, log=log, on_epoch=_ckpt
    )
    clf_hash = param_hash(external)
    save_model(external, ckpt, {"seed": int(seed), "param_hash": clf_hash, "step": int(cfg.clf_steps)})
    del train_imgs
    log(f"  external classifier {clf_hash[:16]}… ({param_count(external)} params)")

    # ---- ⭐ LEG 2: THE GATE, on REAL images -------------------------------
    lab = {
        "A": factor_labels(grid, rows["eval_a"]),
        "B": factor_labels(grid, rows["eval_b"]),
        "MS": factor_labels(grid, ms_rows),
    }
    targets = np.unique(lab["B"][:, fidx])
    if targets.size != 1:
        raise AssertionError(
            f"the held-out cell realises {targets.size} {cfg.gate_factor} classes; the "
            "in-corner binary read assumes exactly one (R20)"
        )
    target = int(targets[0])
    real_reads = {
        tag: read_both(external, probe, real[tag], lab[tag], names, int(cfg.eval_batch))
        for tag in EVAL_SETS
    }
    real_corner = {
        inst: corner_read(m, real["B"], real["MS"], fidx, target,
                          int(cfg.eval_batch), int(cfg.gate_min_negatives))
        for inst, m in (("external", external), ("probe", probe))
    }
    gate_value = float(real_corner["external"]["balanced_accuracy"])
    gate_pass = bool(gate_value >= float(cfg.gate_floor))
    log(
        f"  ⭐ GATE seed {seed}: external in-corner {cfg.gate_factor} balacc on REAL = "
        f"{gate_value:.4f} (floor {cfg.gate_floor}) -> {'PASS' if gate_pass else '⛔ FAIL'} "
        f"| probe {real_corner['probe']['balanced_accuracy']:.4f} "
        f"| n_pos={real_corner['external']['n_pos']} n_neg={real_corner['external']['n_neg']}"
    )

    manifest = {
        "seed": int(seed),
        "dataset": dataset,
        "pretext": pretext,
        "condition": "recomb2element",
        "elapsed_seconds": float(time.time() - t0),
        "rows": {
            "n_clf_train": int(rows["clf_train"].size),
            "clf_train_row_seed": int(seed) * int(cfg.row_seed_mult) + int(cfg.row_seed_offset),
            "pool_size": int(rows["pool_size"]),
            "n_excluded_from_pool": int(rows["n_excluded"]),
            "n_corner_rows_excluded": int(rows["corner"].size),
            "n_wide_heldout_excluded": wide_n,
            "n_eval_a": int(rows["eval_a"].size),
            "n_eval_b": int(rows["eval_b"].size),
            "n_matched_seen": int(ms_rows.size),
        },
        "no_leak": leak,
        "frozen": {
            "bridge_dir": str(bdir),
            "encoder_param_hash": enc_hash,
            "probe_param_hash": probe_hash,
            "cache": {k: cman[k] for k in ("scope", "n_rows", "dtype", "features_sha256", "D")},
            "cache_reencode": reencode,
            **dec_info,
        },
        "classifier": {
            "param_hash": clf_hash,
            "param_count": int(param_count(external)),
            "channels": [int(c) for c in cfg.clf_channels],
            "kernel": int(cfg.clf_kernel),
            "hidden": int(cfg.clf_hidden),
            "steps": int(cfg.clf_steps),
            "batch": int(cfg.clf_batch),
            "lr": float(cfg.clf_lr),
            "loss_first": float(hist[0]) if hist else None,
            "loss_final": float(hist[-1]) if hist else None,
            "checkpoint": str(ckpt),
        },
        "gate": {
            "factor": str(cfg.gate_factor),
            "floor": float(cfg.gate_floor),
            "balanced_accuracy": gate_value,
            "pass": gate_pass,
            "target_class": target,
            "external": real_corner["external"],
            "probe_companion": real_corner["probe"],
        },
        "real_reads": real_reads,
    }

    if not gate_pass:
        manifest["leg3"] = {
            "status": "NOT RUN",
            "reason": "⛔ the gate FAILED on this seed; leg 3 is stop-and-return (task leg 2)",
        }
        _write_json(manifest, man_path)
        log(f"  ⛔ seed {seed}: gate FAILED — leg 3 NOT RUN, returning to the Hub")
        return manifest

    # ---- ⭐⭐ LEG 3: the same renders, both instruments --------------------
    del real
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    render_reads, render_corner_imgs = {}, {}
    for tag, r in (("A", rows["eval_a"]), ("B", rows["eval_b"]), ("MS", ms_rows)):
        f = gather_features(dataset, config, encoder, cache_rows, features, cman, r, log=log)
        Z = np.asarray(f, np.float32).reshape((-1,) + feat_shape)
        dec_imgs = decode_features(decoder, Z, int(cfg.eval_batch))
        render_reads[tag] = read_both(external, probe, dec_imgs, lab[tag], names, int(cfg.eval_batch))
        render_reads[tag]["decoded_mean"] = float(dec_imgs.mean())
        if tag in ("B", "MS"):
            render_corner_imgs[tag] = dec_imgs
        else:
            del dec_imgs
        del f, Z
    render_corner = {
        inst: corner_read(m, render_corner_imgs["B"], render_corner_imgs["MS"], fidx,
                          target, int(cfg.eval_batch), int(cfg.gate_min_negatives))
        for inst, m in (("external", external), ("probe", probe))
    }
    del render_corner_imgs
    ext_b = float(render_corner["external"]["balanced_accuracy"])
    prb_b = float(render_corner["probe"]["balanced_accuracy"])
    ref = list(cfg.probe_reference_per_seed)
    ref_seed = float(ref[int(seed)]) if int(seed) < len(ref) else float("nan")
    log(
        f"  ⭐⭐ LEG 3 seed {seed}: on the SAME renders — external {ext_b:.4f} vs "
        f"probe {prb_b:.4f} (gap {ext_b - prb_b:+.4f}); R22 reference for this seed "
        f"{ref_seed:.4f}"
    )
    manifest["leg3"] = {
        "status": "complete",
        "render_reads": render_reads,
        "corner": render_corner,
        "external_balacc": ext_b,
        "probe_balacc": prb_b,
        "delta": ext_b - prb_b,
        "probe_reference_this_seed": ref_seed,
        "probe_reproduces_reference": (
            bool(abs(prb_b - ref_seed) < 1e-6) if np.isfinite(ref_seed) else None
        ),
    }
    manifest["elapsed_seconds"] = float(time.time() - t0)
    _write_json(manifest, man_path)
    log(f"  ✓ seed {seed} in {manifest['elapsed_seconds']:.0f}s -> {man_path}")
    return manifest


# ---------------------------------------------------------------------------
# Aggregate — ⭐ ``classifier_gate.json`` is written LAST
# ---------------------------------------------------------------------------


def aggregate_runs(out_dir: Path, config: CHLUConfig, log: Callable = print) -> dict:
    """Pool the per-seed manifests, then write the gate file a downstream spoke reads."""
    cfg = config.gen_classifier
    out_dir = Path(out_dir)
    mans, missing = [], []
    for s in cfg.seeds:
        p = out_dir / f"seed_{int(s)}" / "classifier_manifest.json"
        if p.exists():
            with open(p) as fh:
                mans.append(json.load(fh))
        else:
            missing.append(int(s))
    if not mans:
        raise FileNotFoundError(f"no per-seed manifests under {out_dir}")
    if missing:
        log(f"  ⚠ seeds {missing} have no manifest — reported NOT RUN, ⛔ never as a zero")

    gate = gate_verdict(
        {str(m["seed"]): m["gate"]["balanced_accuracy"] for m in mans},
        float(cfg.gate_floor),
        str(cfg.gate_factor),
    )
    gate["seeds_not_run"] = missing
    gate["n_seeds_requested"] = len(list(cfg.seeds))
    gate["probe_companion_on_real"] = _ms(
        [m["gate"]["probe_companion"]["balanced_accuracy"] for m in mans]
    )
    gate["decoder"] = {
        "recipe": mans[0]["frozen"]["decoder_recipe"],
        "source": mans[0]["frozen"]["decoder_source"],
        "param_hash_per_seed": {
            str(m["seed"]): m["frozen"]["decoder_param_hash"] for m in mans
        },
    }
    gate["no_leak"] = {
        "n_decoder_renders_in_training": 0,
        "n_holdout_images_in_training": int(
            sum(m["no_leak"]["holdout"]["n_holdout_leaked"] for m in mans)
        ),
        "per_seed_holdout_excluded": {
            str(m["seed"]): m["no_leak"]["holdout"]["n_holdout_excluded"] for m in mans
        },
        "per_seed_render_rows_checked": {
            str(m["seed"]): m["no_leak"]["render"]["n_checked"] for m in mans
        },
    }

    done = [m for m in mans if m.get("leg3", {}).get("status") == "complete"]
    summary = {
        "dataset": cfg.dataset,
        "pretext": cfg.pretext,
        "seeds_complete": [int(m["seed"]) for m in mans],
        "seeds_not_run": missing,
        "gate": gate,
        "n_clf_train": _ms([m["rows"]["n_clf_train"] for m in mans]),
        "real_corner": {
            inst: _ms([
                m["gate"]["external" if inst == "external" else "probe_companion"][
                    "balanced_accuracy"
                ]
                for m in mans
            ])
            for inst in INSTRUMENTS
        },
    }
    if done:
        ext = {str(m["seed"]): m["leg3"]["external_balacc"] for m in done}
        prb = {str(m["seed"]): m["leg3"]["probe_balacc"] for m in done}
        summary["leg3"] = {
            "status": "complete",
            "n_seeds": len(done),
            "in_corner_balacc": {"external": _ms(ext.values()), "probe": _ms(prb.values())},
            "gap": instrument_gap(
                ext, prb, float(cfg.gap_abs_threshold), float(cfg.gap_spread_mult)
            ),
            "declared_reference_probe_balacc": float(cfg.probe_reference_balacc),
            "declared_reference_wide_11cell": float(cfg.wide_reference_balacc),
            "aggregate_on_A_renders": {
                inst: _ms([m["leg3"]["render_reads"]["A"][inst]["aggregate"] for m in done])
                for inst in INSTRUMENTS
            },
            f"{cfg.gate_factor}_on_A_renders": {
                inst: _ms([
                    m["leg3"]["render_reads"]["A"][inst]["per_factor"][str(cfg.gate_factor)]
                    for m in done
                ])
                for inst in INSTRUMENTS
            },
        }
    else:
        summary["leg3"] = {
            "status": "NOT RUN",
            "reason": "⛔ no seed cleared the gate; leg 3 is stop-and-return",
        }

    _write_json(summary, out_dir / "summary.json")
    if bool(cfg.make_plots):
        try:
            _plot(summary, mans, out_dir / "classifier_gate.png", cfg)
        except Exception as exc:  # ⛔ a plot must never lose a measured number
            log(f"  ⚠ plot skipped: {exc}")
    # ⭐ LAST: a downstream spoke gates on this exact file and this exact key.
    _write_json(gate, out_dir / "classifier_gate.json")
    log(
        f"  ⭐ gate = {gate['gate']} ({gate['n_seeds']}/{gate['n_seeds_requested']} seeds, "
        f"min {gate['min']:.4f}, floor {gate['floor']}) -> {out_dir / 'classifier_gate.json'}"
    )
    return summary


def _plot(summary: dict, mans: list, path: Path, cfg) -> None:
    """Two panels: the gate on REAL images, and the two instruments on the SAME renders."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    seeds = [int(m["seed"]) for m in mans]
    x = np.arange(len(seeds))

    ax = axes[0]
    ax.bar(x - 0.18, [m["gate"]["balanced_accuracy"] for m in mans], 0.34, label="external")
    ax.bar(x + 0.18, [m["gate"]["probe_companion"]["balanced_accuracy"] for m in mans],
           0.34, label="probe")
    ax.axhline(float(cfg.gate_floor), ls="--", c="k", lw=1,
               label=f"gate floor {cfg.gate_floor}")
    ax.set_xticks(x, [f"s{s}" for s in seeds])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel(f"in-corner {cfg.gate_factor} balanced accuracy")
    ax.set_title("leg 2 — REAL held-out images (the gate)")
    ax.legend(fontsize=7)

    ax = axes[1]
    l3 = summary.get("leg3", {})
    if l3.get("status") == "complete":
        done = [m for m in mans if m.get("leg3", {}).get("status") == "complete"]
        xd = np.arange(len(done))
        ax.bar(xd - 0.18, [m["leg3"]["external_balacc"] for m in done], 0.34, label="external")
        ax.bar(xd + 0.18, [m["leg3"]["probe_balacc"] for m in done], 0.34, label="probe")
        ax.axhline(float(cfg.probe_reference_balacc), ls=":", c="crimson", lw=1.2,
                   label=f"R22 reference {cfg.probe_reference_balacc}")
        ax.set_xticks(xd, [f"s{int(m['seed'])}" for m in done])
        ax.set_ylim(0.4, 1.02)
        ax.set_title("leg 3 — the SAME decoder renders")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "leg 3 NOT RUN\n(gate did not pass)", ha="center", va="center")
        ax.set_axis_off()
    fig.suptitle("gen-external-classifier — ⛔ an instrument, not an arm", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_experiment_gen_classifier(
    out_dir: str | Path,
    config: Optional[CHLUConfig] = None,
    stage: str = "all",
    seed: Optional[int] = None,
    quick: bool = False,
    log: Callable = print,
) -> dict:
    """config-load-with-overrides → data → frozen models → fit-or-load → evaluate → plot."""
    config = config or get_default_config()
    if quick:
        config = apply_quick(config)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed)] if seed is not None else [int(s) for s in config.gen_classifier.seeds]
    if stage in ("fit", "all"):
        for s in seeds:
            log(f"[gen-classifier] seed {s}")
            run_seed(s, out_dir, config, log)
    if stage in ("aggregate", "all"):
        return aggregate_runs(out_dir, config, log)
    return {"stage": stage, "seeds": seeds}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exp-gen-classifier",
        description="⭐ the EXTERNAL factor classifier — an instrument, ⛔ not an arm",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="all", choices=list(STAGES))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--pretext", default=None)
    p.add_argument("--steps", type=int, default=None, help="clf_steps override")
    p.add_argument("--n-train", type=int, default=None, help="n_clf_train override")
    p.add_argument("--bridge-root", default=None)
    p.add_argument("--winner-root", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def _config_from_args(args) -> CHLUConfig:
    config = get_default_config()
    c = config.gen_classifier
    for field_name, value in (
        ("dataset", args.dataset),
        ("pretext", args.pretext),
        ("clf_steps", args.steps),
        ("n_clf_train", args.n_train),
        ("bridge_root", args.bridge_root),
        ("winner_root", args.winner_root),
    ):
        if value is not None:
            setattr(c, field_name, value)
    if args.no_plots:
        c.make_plots = False
    return config


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)

    def log(*a):
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    run_experiment_gen_classifier(
        args.out, config, stage=args.stage, seed=args.seed, quick=args.quick, log=log
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
