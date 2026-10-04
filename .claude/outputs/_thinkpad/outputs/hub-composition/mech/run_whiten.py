"""⭐⭐ THE REMEDY: QUANTISE IN A PER-CHANNEL STANDARDISED METRIC.

Diagnosis (LOG 19:50Z): the quantiser's residual is ~39x more damaging per unit norm than
isotropic noise, because factor identity lives in LOW-VARIANCE feature directions and every
variance-optimal compressor discards them.  Remedy: change the METRIC, not the budget.

⭐ sigma is per-CHANNEL, fitted on the store's own TRAIN features. ⛔ LABEL-FREE — no label
enters the write, the read or the landscape (§B.6 intact).
⭐ Byte cost: D x 4 = 8,192 B (1.4 % of the 598,528 B total budget), DECLARED in every row.
⛔ Measures the ROUND-TRIP BOUND, which caps every request-only read. ⛔ Not an arm.
⛔ `val` / `MS` only. ⛔ B never enters.  Resumable: one JSON per unit.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np, jax
from pathlib import Path
from chlu.core.gen_store import (address_codes, channel_correlation, dead_channels,
                                 grouping_labels, init_store, store_bytes, train_store,
                                 well_channels_from_labels)
from chlu.core.gen_compose import read_compose
from mech_driver import build_context, score

TOTAL_BUDGET, HEAD_PER_KM = 598_528, 232
WHITEN_BYTES = 2048 * 4
KS = tuple(int(x) for x in os.environ.get("WH_KS", "5,32,64").split(","))
MODES = tuple(os.environ.get("WH_MODES", "plain,whiten").split(","))
TRAIN = os.environ.get("WH_TRAIN", "0") == "1"
led = Path(os.environ['OUT']) / "mech" / ("whiten_trained" if TRAIN else "whiten_init")
led.mkdir(parents=True, exist_ok=True)

ctx = build_context(0)
gs = ctx["cfg"].gen_store
Xtr = np.asarray(ctx["X_store"], np.float32)
Xc = Xtr[: int(gs.n_corr)]
dead = dead_channels(Xc); corr = channel_correlation(Xc)
sigma = Xtr.std(axis=0)
sigma = np.where(sigma > 1e-6, sigma, 1.0).astype(np.float32)   # ⛔ dead channels untouched
print(f"sigma: min={sigma.min():.4g} max={sigma.max():.4g} "
      f"ratio={sigma.max()/max(sigma.min(),1e-12):.1f}", flush=True)

for K in KS:
    M = TOTAL_BUDGET // (8192 + HEAD_PER_KM * K)
    labels, _ = grouping_labels(corr, K, "correlation", 0, gs.grouping_method,
                               dead=dead, min_well_frac=float(gs.min_well_frac))
    wc = well_channels_from_labels(labels, K)
    for mode in MODES:
        unit = led / f"K{K}_M{M}_{mode}.json"
        if unit.exists():
            print(f"[skip] {unit.name}", flush=True); continue
        t0 = time.time()
        s = sigma if mode == "whiten" else np.ones_like(sigma)
        Xq = Xtr / s
        k1, k2 = jax.random.split(jax.random.PRNGKey(20_260_916 + K))
        store = init_store(Xq, wc, int(M), ctx["feat_shape"], k1, seed=0)
        if TRAIN:
            store, _, _ = train_store(store, Xq, gs, k2, log=None)
        sb = store_bytes(store)
        extra = WHITEN_BYTES if mode == "whiten" else 0
        total = int(sb["store_bytes"]) + HEAD_PER_KM * K * M + extra
        rec = {"K": K, "M": M, "mode": mode, "trained": TRAIN,
               "store_bytes": int(sb["store_bytes"]), "head_bytes": HEAD_PER_KM * K * M,
               "whitening_bytes": extra, "total_bytes": total,
               "within_budget": bool(total <= TOTAL_BUDGET),
               "sigma_ratio": float(sigma.max() / max(sigma.min(), 1e-12))}
        for which in ("val", "MS"):
            rows, X, lab = ctx["sets"][which]
            Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
            code = address_codes(store, Xf / s)
            rt = np.asarray(read_compose(store, code), np.float32).reshape(Xf.shape) * s
            rep = score(ctx, rt, lab, None)
            err = np.linalg.norm(rt - Xf, axis=1) / np.maximum(np.linalg.norm(Xf, axis=1), 1e-9)
            rec[which] = {"valid": rep["distinct_valid"], "n": rep["n"],
                          "exact": rep["exact_match_frac"], "distinct": rep["distinct"],
                          "relerr_unwhitened": float(err.mean())}
        rec["wall_s"] = time.time() - t0
        json.dump(rec, open(unit, "w"), indent=2)
        v = rec["val"]
        print(f"K={K:4d} M={M:4d} {mode:7s} total={total:7d} "
              f"{'OK ' if rec['within_budget'] else 'OVER'} | val VALID={v['valid']:4d}/{v['n']} "
              f"exact={v['exact']:.4f} relerr={v['relerr_unwhitened']:.4f} | "
              f"MS VALID={rec['MS']['valid']:3d} exact={rec['MS']['exact']:.4f} ({rec['wall_s']:.0f}s)",
              flush=True)
print("done")
