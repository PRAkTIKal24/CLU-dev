"""The shared scoring context every w4 rival is measured in - on GENERATION 2.

One context, built once per seed, handed to every rival: the same stratified
fit/val split, the same labels, the same frozen decoder, the same frozen judge
(the external classifier) and its probe companion. A rival scored on a context of
its own is not on this table.

The generation rule is enforced here rather than trusted: :func:`build_context`
loads the feature cache from the generation-2 root and asserts
``manifest["generation"] == 2``. There is no code path in this module that reads a
generation-1 cache, so a w4 rival number cannot be a generation-1 number by
accident.

Everything structural - the split, the row selection, the request packing, the
decoder/judge resolution - is imported from the frozen modules as shipped. This
file adds no science; it only re-points the feature source at generation 2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from ...config import CHLUConfig, gen_row_split_config
from ...core.gen_bridge import dataset_module, param_hash
from ...core.gen_compose import COMPOSITIONAL_FACTOR
from ...core.gen_request import (
    FactorPrototypes,
    dense_request_arrays,
    requests_from_rows,
)
from ...data.gen_rowsplit import split_fit_val
from ...eval.gen_probe import (
    decode_features,
    discrimination_metrics,
    factor_labels,
    factor_probs,
    predict_factors,
    probe_accuracy,
)
from ...experiments.exp_gen_compose import resolve_decoder
from ...experiments.exp_gen_store import matched_seen_rows, select_store_rows
from ...utils.checkpoints import load_model

#: The request sets the gate statistic needs: the 72 held-out rows and their
#: photometrically matched-seen control.
REQUEST_SETS = ("B", "MS")


# ---------------------------------------------------------------------------
# Generation-2 cache access
# ---------------------------------------------------------------------------


def load_gen2_cache(gen2_dir: str | Path, verify: bool = False):
    """``(rows, features_memmap, manifest)`` from a GENERATION 2 cache.

    Asserts the generation, the scope and the schema. A generation-1 manifest
    has no ``generation`` key at all and is refused here, which is the point.
    """
    from ...data.gen_splits import file_sha256

    d = Path(gen2_dir)
    p = d / "feature_cache.json"
    if not p.exists() and (d / "features.npy").exists():
        from ...paths import data_manifest
        m = data_manifest()["features"]
        seed = int(d.name.split("_")[-1])
        man = {"generation": 2, "scope": m["scope"], "features_file": "features.npy",
               "rows_file": None, "features_sha256": m["seeds"][str(seed)]["sha256"],
               "encoder_param_hash": m["seeds"][str(seed)]["encoder_param_hash"],
               "n_rows": m["n_rows"], "feat_shape": m["feat_shape"], "dtype": m["dtype"],
               "device": {"devices": ["not recorded"]}}
        if verify:
            from ...data.gen_splits import file_sha256 as _sha
            assert _sha(d / "features.npy") == man["features_sha256"]
        return (np.arange(int(m["n_rows"]), dtype=np.int64),
                np.load(d / "features.npy", mmap_mode="r"), man)
    if not p.exists():
        raise FileNotFoundError(
            f"no generation-2 cache at {d} - build it with "
            "`exp_w4_rivals --stage cache2` before scoring any rival")
    with open(p) as fh:
        man = json.load(fh)
    if int(man.get("generation", -1)) != 2:
        raise AssertionError(
            f"{p} declares generation={man.get('generation')!r}, not 2. This module "
            "scores on generation 2 only; mixing generations in either direction is "
            "forbidden.")
    if man.get("scope") != "full":
        raise AssertionError(f"{p} has scope={man.get('scope')!r}, expected 'full'")
    feat_path = d / man["features_file"]
    if verify:
        got = file_sha256(feat_path)
        if got != man["features_sha256"]:
            raise AssertionError(
                f"generation-2 cache checksum mismatch at {feat_path}: "
                f"{got} != {man['features_sha256']}")
    rp = d / man["rows_file"]
    if rp.exists():
        rows = np.load(rp)
    elif man.get("scope") == "full":
        rows = np.arange(int(man["n_rows"]), dtype=np.int64)
    else:
        raise FileNotFoundError(rp)
    feats = np.load(feat_path, mmap_mode="r")
    return rows, feats, man


def gather_gen2(cache_rows: np.ndarray, features, rows: np.ndarray) -> np.ndarray:
    """Cached features for ``rows``, as ``(N, D)``.

    A full-scope generation-2 cache covers every row the grid indexes, so there is
    no side buffer and no on-the-fly re-encode. A miss is an error, not a fallback:
    a silently re-encoded row is exactly the mixed-device hazard generation 2 was
    built to remove.

    ⛔ The ``(N, D)`` shape is part of the contract, not a convenience. The shipped
    ``exp_gen_store.gather_features`` returns ``(rows.size, cman["D"])``, and every
    arm written against it assumes rank 2. The cache itself is stored at
    ``(N,) + feat_shape`` = ``(N, 128, 4, 4)``, so returning the raw gather makes a
    context that LOOKS right and is not: arms that flatten internally (the
    prototype table, LACE's whitening, the generic learners) keep working, while
    ``exp_gen_cell.train_expert`` infers its input width from ``X.shape[1]`` and
    silently builds a 128-wide expert for a 2048-dimensional input. Measured: that
    is exactly how the PoE-EBM arm died here, and it died only AFTER every other
    arm on the same context had succeeded.
    """
    rows = np.asarray(rows, np.int64)
    cache_rows = np.asarray(cache_rows, np.int64)
    pos = np.searchsorted(cache_rows, rows)
    safe = np.minimum(pos, cache_rows.size - 1)
    miss = (pos >= cache_rows.size) | (cache_rows[safe] != rows)
    if miss.any():
        raise KeyError(
            f"{int(miss.sum())} row(s) are not in the generation-2 cache "
            f"(first: {int(rows[miss][0])}). A full-scope cache should have every "
            "row; refusing to re-encode on the fly.")
    out = np.asarray(features[pos], np.float32)
    return out.reshape((out.shape[0], -1))


# ---------------------------------------------------------------------------
# Scoring - the same functions read every rival and both instruments
# ---------------------------------------------------------------------------


def render(decoder, feats: np.ndarray, feat_shape, batch: int) -> np.ndarray:
    """``(N, D)`` rival output -> ``(N, C, H, W)`` images, through THE decoder."""
    Z = np.asarray(feats, np.float32).reshape((-1,) + tuple(feat_shape))
    return decode_features(decoder, Z, int(batch))


def in_corner(model, img_pos, img_neg, lab_pos, shape_idx: int, batch: int) -> dict:
    """The gate statistic: discrimination within the corner, on IMAGES."""
    pp = factor_probs(model, img_pos, int(shape_idx), int(batch))
    pn = factor_probs(model, img_neg, int(shape_idx), int(batch))
    targets = np.unique(np.asarray(lab_pos, np.int64)[:, int(shape_idx)])
    if targets.size != 1:
        raise AssertionError(
            f"the held-out cell realises {targets.size} {COMPOSITIONAL_FACTOR} classes; "
            "the in-corner binary read assumes exactly one")
    return discrimination_metrics(pp, pn, int(targets[0]))


def score_images(model, images, labels, names, batch: int) -> dict:
    out = probe_accuracy(model, images, labels, names, int(batch))
    out["compositional_factor"] = COMPOSITIONAL_FACTOR
    return out


def distinct_counts(model, feats: np.ndarray, images: np.ndarray, batch: int) -> dict:
    """The column an accuracy is never reported without.

    ``factor_tuples`` is the cross-arm comparable one: ``feature_vectors`` and
    ``images`` are trivially N for any continuous-output rival.
    """
    F = np.asarray(feats, np.float32).reshape((int(feats.shape[0]), -1))
    G = np.asarray(images, np.float32).reshape((int(images.shape[0]), -1))
    preds = predict_factors(model, images, int(batch))
    return {
        "n": int(F.shape[0]),
        "feature_vectors": int(np.unique(F, axis=0).shape[0]),
        "images": int(np.unique(G, axis=0).shape[0]),
        "factor_tuples": int(np.unique(np.asarray(preds, np.int64), axis=0).shape[0]),
        "note": ("feature_vectors/images are trivially N for a continuous rival; "
                 "factor_tuples is the comparable diversity count"),
    }


def score_rival(ctx: dict, feats_by_set: dict, batch: int | None = None) -> dict:
    """Score one rival's features on BOTH instruments, plus its distinct counts.

    ``feats_by_set`` maps each of :data:`REQUEST_SETS` to that rival's ``(N, D)``
    output. The judge is the external classifier; the probe runs beside it as the
    companion, because a probe-vs-external gap must never be readable as a
    difference in what the two looked at.
    """
    batch = int(batch or ctx["eval_batch"])
    imgs = {k: render(ctx["decoder"], feats_by_set[k], ctx["feat_shape"], batch)
            for k in REQUEST_SETS}
    out = {}
    for iname, model in (("external", ctx["external"]), ("probe", ctx["probe"])):
        out[iname] = {
            "in_corner": in_corner(model, imgs["B"], imgs["MS"], ctx["labT_B"],
                                   ctx["shape_idx"], batch),
            "composition_B": score_images(model, imgs["B"], ctx["labT_B"],
                                          ctx["names"], batch),
        }
    out["distinct_counts_B"] = distinct_counts(ctx["external"], feats_by_set["B"],
                                               imgs["B"], batch)
    out["generation"] = 2
    return out


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------


def build_context(config: CHLUConfig, dataset: str, pretext: str, seed: int,
                  gen2_dir: str | Path, bridge_dir: str | Path,
                  log: Callable = print) -> dict:
    """Everything a w4 rival needs, on generation 2, for one seed."""
    cfg = config.gen_cell
    bdir = Path(bridge_dir)

    # ---- the two guards, first: the decoder and the judge -------------------
    from ...paths import out_root
    claude_root = Path(config.gen_cell.claude_root or out_root())
    config.gen_compose.claude_root = str(claude_root)
    config.gen_compose.decoder_source = str(cfg.decoder_source)
    config.gen_compose.winner_root = cfg.winner_root
    config.gen_compose.winner_decoder_pattern = str(cfg.winner_decoder_pattern)
    config.gen_compose.expected_decoder_recipe = str(cfg.expected_decoder_recipe)
    decoder, dec_info, bman = resolve_decoder(config, bdir, seed, log=log)

    external, clf_info = _load_external(config, claude_root, seed, log)
    probe = load_model(bdir / "probe.pkl")
    if param_hash(probe) != bman["probe_param_hash"]:
        raise AssertionError("probe checkpoint does not match the bridge manifest hash")

    # ---- rows: the SAME stratified cut the scored cell makes -----------------
    mod = dataset_module(dataset)
    split, values = mod.build_split(config.gen_data.data_root)
    grid = split.grid
    names, cards = list(grid.names), list(grid.cardinalities)
    shape_idx = names.index(COMPOSITIONAL_FACTOR)
    rows = select_store_rows(split, config, seed)
    pool = np.asarray(rows["store_train"], np.int64)
    rs = gen_row_split_config(config)
    fit_rows, val_rows, split_info = split_fit_val(
        pool, grid.factor_indices_of(pool), int(cfg.n_fit), int(cfg.n_val), int(seed),
        mode=str(rs.mode), tiebreak_jitter=float(rs.tiebreak_jitter),
        seed_mult=int(rs.seed_mult), seed_offset=int(rs.seed_offset), log=log,
    )
    if int(val_rows.size) != int(cfg.n_val):
        raise AssertionError(
            f"SILENT TRUNCATION: n_val requested {cfg.n_val}, realised {val_rows.size}")
    row_sets = {
        "B": rows["eval_b"],
        "MS": matched_seen_rows(dataset, split, values, rows["store_train"]),
        "fit": fit_rows,
        "val": val_rows,
    }

    # ---- GENERATION 2 features ----------------------------------------------
    crows, feats, cman = load_gen2_cache(gen2_dir)
    if cman["encoder_param_hash"] != bman["encoder_param_hash"]:
        raise AssertionError(
            "the generation-2 cache was written by a different encoder than this "
            "bridge - these are different feature spaces")
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    X = {k: gather_gen2(crows, feats, v) for k, v in row_sets.items()}
    X_store = gather_gen2(crows, feats, pool)
    lab = {k: factor_labels(grid, v) for k, v in row_sets.items()}
    log(f"  generation 2 cache: {cman['n_rows']} rows, scope={cman['scope']}, "
        f"sha256={cman['features_sha256'][:16]}... device="
        f"{cman['device']['devices']}")

    # ---- requests -----------------------------------------------------------
    req_dicts = {k: requests_from_rows(grid, row_sets[k]) for k in REQUEST_SETS}
    req = {k: dense_request_arrays(grid, req_dicts[k]) for k in REQUEST_SETS}
    req_fit = dense_request_arrays(grid, requests_from_rows(grid, row_sets["fit"]))
    req_val = dense_request_arrays(grid, requests_from_rows(grid, row_sets["val"]))

    # ---- the prototype table (the incumbent), fitted on the FIT rows only ----
    protos = FactorPrototypes.fit(grid, X["fit"], row_sets["fit"])

    # ---- normalised continuous factor targets (for LACE and the cont-MLP) ----
    cont = {}
    for setname in ("fit", "val"):
        cols = []
        for k in range(grid.n_factors):
            vals_k = np.asarray(grid.values[k], np.float64)
            span = max(float(vals_k[-1] - vals_k[0]), 1e-12)
            cols.append((vals_k[lab[setname][:, k]] - vals_k[0]) / span)
        cont[setname] = np.stack(cols, axis=1).astype(np.float32)

    ctx = {
        "seed": int(seed), "dataset": dataset, "pretext": pretext,
        "grid": grid, "split": split, "values": values,
        "names": names, "cardinalities": cards, "shape_idx": shape_idx,
        "feat_shape": feat_shape, "D": int(np.prod(feat_shape)),
        "eval_batch": int(cfg.eval_batch),
        "decoder": decoder, "probe": probe, "external": external,
        "decoder_info": dec_info, "classifier_info": clf_info,
        "row_sets": row_sets, "row_split": split_info, "pool": pool,
        "X_fit": X["fit"], "X_val": X["val"], "X_B": X["B"], "X_MS": X["MS"],
        "X_store": X_store,
        "lab_fit": lab["fit"], "lab_val": lab["val"],
        "labT_B": lab["B"], "labT_MS": lab["MS"],
        "cont_fit": cont["fit"], "cont_val": cont["val"],
        "req_B": req["B"], "req_MS": req["MS"],
        "req_fit": req_fit, "req_val": req_val,
        "protos": protos,
        # Generation-2 features for an arbitrary row set, for the stages that need
        # rows the fixed sets do not cover (the deletion comparator's forget/retain
        # partitions). Still a pure cache gather: it cannot re-encode.
        "_gather": (lambda r, _cr=crows, _f=feats: gather_gen2(_cr, _f, r)),
        "cache_manifest": cman,
        "bridge_manifest": {k: bman.get(k) for k in
                            ("encoder_param_hash", "probe_param_hash")},
        "generation": 2,
    }
    log(f"  rows: store_train={pool.size} B={row_sets['B'].size} "
        f"MS={row_sets['MS'].size} fit={fit_rows.size} val={val_rows.size}")
    return ctx


def _load_external(config: CHLUConfig, claude_root: Path, seed: int, log: Callable):
    """The gated external classifier - THE judge, loaded and hash-asserted."""
    cfg = config.gen_cell
    from ...paths import weights_dir
    root = Path(cfg.classifier_root).expanduser() if cfg.classifier_root else (weights_dir() / "classifier")
    ckpt = root / cfg.classifier_checkpoint_pattern.format(seed=int(seed))
    man_p = root / cfg.classifier_manifest_pattern.format(seed=int(seed))
    if not (ckpt.exists() or ckpt.with_suffix(".eqx").exists()):
        raise FileNotFoundError(
            f"no external classifier at {ckpt} - this seed's instrument does not "
            "exist and its rivals are reported NOT RUN, never as zeros")
    with open(man_p) as fh:
        man = json.load(fh)
    model = load_model(ckpt)
    h = param_hash(model)
    expect = str((man.get("classifier") or {}).get("param_hash"))
    if expect and h != expect:
        raise AssertionError(
            f"external classifier hash {h[:16]} does not match its own manifest "
            f"{expect[:16]} - the instrument on disk is not the one that was gated")
    info = {"path": str(ckpt), "param_hash": h,
            "param_count": (man.get("classifier") or {}).get("param_count"),
            "gate_balacc_on_real": ((man.get("gate") or {}).get("external") or {}).get(
                "balanced_accuracy"),
            "note": "ONE judge for every rival, loaded and hash-asserted"}
    log(f"  judge: external classifier {h[:16]} ({info['param_count']} params)")
    return model, info
