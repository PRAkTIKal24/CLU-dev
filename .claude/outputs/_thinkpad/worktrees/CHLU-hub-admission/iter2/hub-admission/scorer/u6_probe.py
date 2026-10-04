"""U6 PROBE -- scale + loader reconnaissance for the certified-access figure.

⛔ Reads only.  No re-encoding: features come from ctx["X_store"] (the frozen cache).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ["CHLU_HUB_OUTPUTS"]) / "hub-admission"
sys.path.insert(0, str(OUT / "scorer"))
sys.path.insert(0, os.path.join(os.environ["CHLU_HUB_OUTPUTS"], "hub-refusal", "scorer"))

import admission_lib as AL          # noqa: E402
import refusal_lib as R             # noqa: E402

T0 = time.time()
def log(m=""):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

SEED, BUDGET = 0, "A"
ctx = R.build_context(SEED)
built = R.build_land(ctx, budget=BUDGET, log=lambda *a, **k: None)
land, store, grid = built["land"], built["store"], ctx["grid"]
log(f"land K={store.n_wells} N={store.n_atoms} dim={store.dim} feat_shape={ctx['feat_shape']}")
log(f"sigma={np.asarray(land.sigma)}  cert_radius={AL.certified_radius(land)}")
log(f"well sizes = {[len(w) for w in store.well_channels]}")

pool = np.asarray(ctx["pool"], np.int64)
val_rows = np.asarray(ctx["val_rows"], np.int64)
pos = {int(r): i for i, r in enumerate(pool)}
log(f"pool={pool.size} val_rows={val_rows.size} all val in pool: "
    f"{all(int(r) in pos for r in val_rows)}")

idx = np.asarray([pos[int(r)] for r in val_rows[:64]], np.int64)
X0 = np.asarray(ctx["X_store"], np.float32)[idx]
log(f"X0 {X0.shape} dtype={X0.dtype}")

rng = np.random.default_rng(20260917)
U = rng.standard_normal(X0.shape).astype(np.float32)
U /= np.linalg.norm(U, axis=1, keepdims=True)

sig = np.asarray(land.sigma, np.float64)
log("t-grid reconnaissance (mean over 64 val rows):")
for t in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]:
    d = AL.per_well_nearest(land, X0 + t * U)
    E = AL.per_well_energy(land, X0 + t * U)
    ds = d / sig[None, :]
    log(f"  t={t:7.1f}  min_k d/sig={ds.min(1).mean():8.4f}  max_k d/sig={ds.max(1).mean():8.4f}"
        f"  max_k E={E.max(1).mean():12.3f}  refuse={float((E>0).any(1).mean()):.3f}"
        f"  certified={float((d >= AL.certified_radius(land)[None,:]).any(1).mean()):.3f}")

# ---- loaders -------------------------------------------------------------
log("")
log("loading the shipped decoder + external classifier ...")
from chlu.experiments.exp_gen_baselines import load_shared_decoder   # noqa: E402
from chlu.experiments.exp_gen_cell import load_external_classifier   # noqa: E402
from chlu.experiments.exp_gen_store import bridge_dir                # noqa: E402
from chlu.eval.gen_probe import decode_features, factor_labels, probe_accuracy  # noqa: E402

config = ctx["config"]
bdir = bridge_dir(config, "shapes3d", "recon", SEED)
bman = json.loads((bdir / "bridge_manifest.json").read_text())
dec, dinfo = load_shared_decoder(config, bdir, bman, SEED, log)
log(f"decoder info: {dinfo}")
clf, cinfo = load_external_classifier(config, SEED, log=log)
log(f"clf info: {cinfo}")

lab = factor_labels(grid, val_rows[:64])
imgs = decode_features(dec, X0.reshape((-1,) + tuple(ctx["feat_shape"])), 256)
log(f"decoded {imgs.shape}")
acc = probe_accuracy(clf, imgs, lab, list(grid.names), 256)
log(f"t=0 accuracy: aggregate={acc['aggregate']:.4f} per_factor={acc['per_factor']}")

# chance level
cards = list(grid.cardinalities)
log(f"cards={cards} chance(mean over factors)={float(np.mean([1.0/c for c in cards])):.4f}")

# ---- settle timing -------------------------------------------------------
import jax                                                            # noqa: E402
from chlu.config import gen_dynamical_read_config                     # noqa: E402
from chlu.core.gen_landscape import SettleCfg, settle_chain           # noqa: E402

c = gen_dynamical_read_config(config)
log(f"dynread cfg: step_size={c.step_size} sig_start={c.sigma_scale_start} "
    f"sig_end={c.sigma_scale_end} temp={c.temp_start}->{c.temp_end} metropolis={c.metropolis} "
    f"request_chunk={c.request_chunk}")
cfg50 = SettleCfg(steps=50, step_size=float(c.step_size),
                  sigma_scale_start=float(c.sigma_scale_start),
                  sigma_scale_end=float(c.sigma_scale_end),
                  temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                  metropolis=bool(c.metropolis))
t1 = time.time()
o = settle_chain(land, X0 + 20.0 * U, None, cfg50, jax.random.PRNGKey(0))
dt = time.time() - t1
log(f"settle_chain n=64 steps=50 -> {dt:.1f}s  ({dt/50:.3f}s/step)  "
    f"displacement_total={float(np.asarray(o['trace']['displacement']).sum(0).mean()):.3f}")
log("PROBE DONE")
