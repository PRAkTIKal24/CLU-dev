"""hub/generation-table — DEV context builder.

⛔ NOT a scored harness. Nothing here ever touches ``rows['eval_b']`` (the Montero
held-out cell) or ``rows['eval_a']``.  Every number produced from this module is a
DEV number, used for mechanism selection only (charter §A.2 "selection on val only" —
the cell's own ``val`` is not compositional, so a compositional DEV corner is cut
from inside the store pool instead, and DECLARED).

Resumable: the expensive context is cached to ``$CHLU_OUT/dev/cache``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_OUT"])
CACHE = OUT / "dev" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _t():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{_t()}] {msg}", flush=True)


def load_config(seed: int):
    from chlu.config import get_default_config
    c = get_default_config()
    c.gen_cell.claude_root = os.environ["CHLU_CLAUDE_ROOT"]
    c.gen_compose.claude_root = os.environ["CHLU_CLAUDE_ROOT"]
    return c


def build_context(seed: int = 0, n_store_train: int | None = None):
    """Everything the cell builds, minus every held-out evaluation."""
    from chlu.core.gen_bridge import dataset_module, load_feature_cache, param_hash
    from chlu.experiments.exp_gen_bridge import condition_of
    from chlu.experiments.exp_gen_cell import _cell_root, load_external_classifier
    from chlu.experiments.exp_gen_compose import resolve_decoder
    from chlu.experiments.exp_gen_store import bridge_dir, gather_features, select_store_rows
    from chlu.eval.gen_probe import factor_labels
    from chlu.utils.checkpoints import load_model

    cfg = load_config(seed)
    cfg.gen_compose.decoder_source = cfg.gen_cell.decoder_source
    cfg.gen_compose.winner_root = cfg.gen_cell.winner_root
    cfg.gen_compose.winner_decoder_pattern = cfg.gen_cell.winner_decoder_pattern
    cfg.gen_compose.expected_decoder_recipe = cfg.gen_cell.expected_decoder_recipe

    dataset, pretext = "shapes3d", "recon"
    bdir = bridge_dir(cfg, dataset, pretext, seed)
    log(f"bridge {bdir}")

    decoder, dec_info, bman = resolve_decoder(cfg, bdir, seed, log=log)
    external, clf_info = load_external_classifier(cfg, seed, log=log)
    encoder = load_model(bdir / "encoder.pkl")
    assert param_hash(encoder) == bman["encoder_param_hash"], "encoder hash mismatch"
    probe = load_model(bdir / "probe.pkl")
    assert param_hash(probe) == bman["probe_param_hash"], "probe hash mismatch"

    mod = dataset_module(dataset)
    split, values = mod.build_split(cfg.gen_data.data_root)
    grid = split.grid
    names = list(grid.names)
    cards = [int(c) for c in grid.cardinalities]

    rows = select_store_rows(split, cfg, int(seed))
    pool = np.asarray(rows["store_train"], np.int64)
    if n_store_train is not None:
        pool = pool[: int(n_store_train)]

    crows, feats_cache, cman = load_feature_cache(bdir, verify=False)
    feat_shape = tuple(int(x) for x in cman["feat_shape"])
    assert cman["encoder_param_hash"] == bman["encoder_param_hash"]

    def gather(r):
        return gather_features(dataset, cfg, encoder, crows, feats_cache, cman,
                               np.asarray(r, np.int64), log=log)

    return {
        "cfg": cfg, "seed": int(seed), "dataset": dataset, "pretext": pretext,
        "bdir": bdir, "decoder": decoder, "dec_info": dec_info, "bman": bman,
        "external": external, "clf_info": clf_info, "encoder": encoder, "probe": probe,
        "split": split, "values": values, "grid": grid, "names": names, "cards": cards,
        "rows": rows, "pool": pool, "crows": crows, "feats_cache": feats_cache,
        "cman": cman, "feat_shape": feat_shape, "gather": gather,
        "factor_labels": factor_labels, "condition": condition_of(cfg),
    }


# ---------------------------------------------------------------------------
# The DEV corner — a Montero-SHAPED cell, cut from inside the train split
# ---------------------------------------------------------------------------

def montero_like_corner(values: np.ndarray, grid, *, hue_lo: float, scale_hi: bool,
                        shape_value: float, orient_hi: bool) -> np.ndarray:
    """Rows of a corner with Montero's own SHAPE: a 2-value band on each of the three
    hues, a 3-value corner of scale, exactly ONE shape, a 2-value corner of orientation.

    ⭐ Montero's own cell is (hue>=0.8, scale>=1.1, shape==cylinder, orientation>20).
    This builds the same object at a different location so it can be held out from a
    DEV store without touching the real one.
    """
    v = np.asarray(values)
    hv = np.asarray(grid.values[0], float)
    step = float(np.min(np.diff(hv)))
    band = (v[:, 0] >= hue_lo - 1e-9) & (v[:, 0] <= hue_lo + step + 1e-9)
    band &= (v[:, 1] >= hue_lo - 1e-9) & (v[:, 1] <= hue_lo + step + 1e-9)
    band &= (v[:, 2] >= hue_lo - 1e-9) & (v[:, 2] <= hue_lo + step + 1e-9)
    sv = np.asarray(grid.values[3], float)
    band &= (v[:, 3] >= 1.1 - 1e-9) if scale_hi else (v[:, 3] <= sv[2] + 1e-9)
    band &= (np.abs(v[:, 4] - shape_value) < 1e-9)
    band &= (v[:, 5] > 20.0) if orient_hi else (v[:, 5] < -20.0)
    return np.where(band)[0].astype(np.int64)
