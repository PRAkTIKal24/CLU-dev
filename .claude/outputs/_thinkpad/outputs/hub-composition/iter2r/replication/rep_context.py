"""Independent replication spoke — context builder.

⛔ Written from the pinned library only. No hub script was read.
Stage 1: build the frozen context (decoder, classifier, split, rows, cached
features, grouping at K=32) and persist it so the scoring stage is cheap.
"""
import json, os, sys, time, hashlib
from pathlib import Path
import numpy as np

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

import chlu
log(f"chlu.__file__ = {chlu.__file__}")
import jax, numpy
log(f"python {sys.version.split()[0]}  numpy {numpy.__version__}  jax {jax.__version__}")

from chlu.config import get_default_config
from chlu.core.gen_bridge import dataset_module, load_feature_cache
from chlu.core.gen_store import (channel_correlation, dead_channels, grouping_labels,
                                 well_channels_from_labels, block_diagonality)
from chlu.eval.gen_probe import factor_labels
from chlu.experiments.exp_gen_store import bridge_dir, select_store_rows, gather_features
from chlu.experiments.exp_gen_compose import resolve_decoder
from chlu.experiments.exp_gen_cell import load_external_classifier
from chlu.utils.checkpoints import load_model
from chlu.core.gen_bridge import param_hash

REP = Path(os.environ["REP"])
SEED = 0
K = 32
M = 26

cfg = get_default_config()
cr = os.environ["CHLU_CLAUDE_ROOT"]
cfg.gen_cell.claude_root = cr
cfg.gen_compose.claude_root = cr
for f in ("decoder_source", "winner_root", "winner_decoder_pattern",
          "expected_decoder_recipe"):
    setattr(cfg.gen_compose, f, getattr(cfg.gen_cell, f))
log(f"claude_root = {cr}")

bdir = bridge_dir(cfg, "shapes3d", "recon", SEED)
log(f"bridge dir = {bdir}")

decoder, dec_info, bman = resolve_decoder(cfg, bdir, SEED, log=log)
clf, clf_info = load_external_classifier(cfg, SEED, log=log)
probe = load_model(bdir / "probe.pkl")
log(f"bridge probe hash {param_hash(probe)[:16]}… (manifest {bman['probe_param_hash'][:16]}…)")

split, values = dataset_module("shapes3d").build_split(cfg.gen_data.data_root)
grid = split.grid
names = list(grid.names); cards = [int(c) for c in grid.cardinalities]
log(f"factors {names} cards {cards}")

rows = select_store_rows(split, cfg, SEED)
store_train = np.asarray(rows["store_train"], np.int64)
eval_b = np.asarray(rows["eval_b"], np.int64)
log(f"rows: store_train={store_train.size} eval_a={rows['eval_a'].size} eval_b={eval_b.size}")

crows, feats, cman = load_feature_cache(bdir, verify=False)
feat_shape = tuple(int(x) for x in cman["feat_shape"])
log(f"cache: n_rows={cman['n_rows']} scope={cman['scope']} D={cman['D']} "
    f"dtype={cman['dtype']} feat_shape={feat_shape}")
log(f"cache encoder hash {cman['encoder_param_hash'][:16]}…  bridge manifest "
    f"{bman['encoder_param_hash'][:16]}…  match={cman['encoder_param_hash']==bman['encoder_param_hash']}")

# ---- ⛔ THE NO-ENCODE GATE: every row I need must already be cached ----------
need = np.union1d(store_train, eval_b)
pos = np.searchsorted(crows, need)
safe = np.minimum(pos, crows.size - 1)
hit = (pos < crows.size) & (crows[safe] == need)
shortfall = int((~hit).sum())
log(f"⭐ coverage gate: need {need.size} rows, shortfall = {shortfall}")
if shortfall:
    raise SystemExit(f"⛔ SHORTFALL {shortfall} rows uncached — STOPPING, never encoding.")

# encoder=None is safe: gather_features only touches it on a miss, and there are none.
X_store = gather_features("shapes3d", cfg, None, crows, feats, cman, store_train, log=log)
X_b = gather_features("shapes3d", cfg, None, crows, feats, cman, eval_b, log=log)
log(f"features: store {X_store.shape} B {X_b.shape}")

lab_store = factor_labels(grid, store_train)
lab_b = factor_labels(grid, eval_b)

# ---- the grouping: the shipped correlation path, at K = 32 ------------------
Xf = np.asarray(X_store, np.float32).reshape((X_store.shape[0], -1))
dead = dead_channels(Xf)
corr = channel_correlation(Xf)
labels_vec, ginfo = grouping_labels(corr, K, "correlation", SEED,
                                    cfg.gen_store.grouping_method, dead=dead,
                                    min_well_frac=float(cfg.gen_store.min_well_frac))
wc = well_channels_from_labels(labels_vec, K)
sizes = [len(w) for w in wc]
bd = block_diagonality(corr, labels_vec)
log(f"grouping: method_used={ginfo['method_used']} dead={int(dead.sum())} "
    f"well sizes {sizes} in-block |corr|^2={bd['in_block_fraction']:.4f}")

np.savez_compressed(
    REP / "context.npz",
    store_train=store_train, eval_b=eval_b,
    X_store=Xf, X_b=np.asarray(X_b, np.float32).reshape((X_b.shape[0], -1)),
    lab_store=lab_store, lab_b=lab_b,
    labels_vec=np.asarray(labels_vec, np.int64), dead=dead,
)
meta = {
    "chlu_file": chlu.__file__,
    "python": sys.version, "numpy": numpy.__version__, "jax": jax.__version__,
    "bridge_dir": str(bdir),
    "decoder": dec_info, "classifier": clf_info,
    "bridge_probe_param_hash": param_hash(probe),
    "bridge_manifest_probe_hash": bman["probe_param_hash"],
    "bridge_manifest_encoder_hash": bman["encoder_param_hash"],
    "cache": {k: cman[k] for k in ("n_rows", "scope", "D", "dtype", "encoder_param_hash")},
    "cache_feat_shape": list(feat_shape),
    "coverage_shortfall": shortfall,
    "n_store_train": int(store_train.size), "n_eval_b": int(eval_b.size),
    "names": names, "cardinalities": cards,
    "K": K, "M": M,
    "grouping_info": ginfo, "well_sizes": sizes,
    "block_diagonality": bd,
}
(REP / "context_meta.json").write_text(json.dumps(meta, indent=2, default=str))
log("context written -> context.npz / context_meta.json")
