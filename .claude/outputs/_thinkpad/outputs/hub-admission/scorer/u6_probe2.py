"""U6 PROBE 2 -- settle_chain cost scaling, and how far the CLU path actually moves."""
from __future__ import annotations

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
land, store = built["land"], built["store"]
sig = np.asarray(land.sigma, np.float64)

pool = np.asarray(ctx["pool"], np.int64)
val_rows = np.asarray(ctx["val_rows"], np.int64)
pos = {int(r): i for i, r in enumerate(pool)}
Xs = np.asarray(ctx["X_store"], np.float32)

import jax                                                            # noqa: E402
from chlu.config import gen_dynamical_read_config                     # noqa: E402
from chlu.core.gen_landscape import SettleCfg, settle_chain           # noqa: E402

c = gen_dynamical_read_config(ctx["config"])
log("t=0 min_k d/sig on 256 val rows:")
idx = np.asarray([pos[int(r)] for r in val_rows[:256]], np.int64)
X0 = Xs[idx]
d0 = AL.per_well_nearest(land, X0) / sig[None, :]
log(f"  min_k d/sig: mean={d0.min(1).mean():.4f} min={d0.min(1).min():.4f} "
    f"max={d0.min(1).max():.4f}  E_max mean={AL.per_well_energy(land, X0).max(1).mean():.3f}")

rng = np.random.default_rng(20260917)
U = rng.standard_normal(X0.shape).astype(np.float32)
U /= np.linalg.norm(U, axis=1, keepdims=True)

for t in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
    ds = (AL.per_well_nearest(land, X0 + t * U) / sig[None, :]).min(1)
    E = AL.per_well_energy(land, X0 + t * U).max(1)
    log(f"  t={t:6.1f} min_k d/sig mean={ds.mean():7.4f}  E_max mean={E.mean():10.3f} "
        f"refuse={float((E>0).mean()):.3f}")


def mk(steps):
    return SettleCfg(steps=steps, step_size=float(c.step_size),
                     sigma_scale_start=float(c.sigma_scale_start),
                     sigma_scale_end=float(c.sigma_scale_end),
                     temp_start=float(c.temp_start), temp_end=float(c.temp_end),
                     metropolis=bool(c.metropolis))


for n in (256, 1024):
    Xn = np.repeat(X0[:64], n // 64, axis=0)
    Un = np.repeat(U[:64], n // 64, axis=0)
    t1 = time.time(); settle_chain(land, Xn + 30.0 * Un, None, mk(5), jax.random.PRNGKey(0))
    warm = time.time() - t1
    t1 = time.time(); settle_chain(land, Xn + 30.0 * Un, None, mk(20), jax.random.PRNGKey(0))
    dt = time.time() - t1
    log(f"n={n:5d}  warm(5 steps)={warm:.1f}s  20 steps={dt:.1f}s -> {dt/20:.4f}s/step "
        f"=> 1000 steps ~ {dt/20*1000/60:.1f} min")

# how far does the CLU path move a point launched at t=30 ?
o = settle_chain(land, X0[:64] + 30.0 * U[:64], None, mk(200), jax.random.PRNGKey(0))
xs = np.asarray(o["x"], np.float32)
mv = np.linalg.norm(xs - (X0[:64] + 30.0 * U[:64]), axis=1)
log(f"200-step CLU path: |x_settled - x_launch| mean={mv.mean():.3f} "
    f"(launch offset was 30.0); min_k d/sig after = "
    f"{(AL.per_well_nearest(land, xs)/sig[None,:]).min(1).mean():.4f} "
    f"before = {(AL.per_well_nearest(land, X0[:64]+30.0*U[:64])/sig[None,:]).min(1).mean():.4f}")
log("PROBE2 DONE")
