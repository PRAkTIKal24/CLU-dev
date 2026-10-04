"""⭐⭐ THE PARTITION SWEEP — round-trip fidelity vs K at CONSTANT DECLARED BYTES.

§B permits the PARTITION and the CODEBOOK to change.  store bytes = M·D·4 is independent
of K; bits per read = K·log₂M is linear in K.  ⛔ But the addressing head is 232·K·M B and
grows with BOTH, so the honest budget is TOTAL = store + head, held at the shipped
524,288 + 74,240 = 598,528 B.  M(K) = floor(598,528 / (8,192 + 232·K)).

⛔ The grouping is LABEL-FREE (`grouping_labels(corr, K, "correlation", …)` — channel
correlation only), so §B.6 is not touched: no label enters the landscape, the read or the
write.  ⛔ This measures the ROUND-TRIP BOUND (true feature in, code, read back), which
caps every request-only read; it is a bound, ⛔ not an arm.

⛔ `val` / `MS` only. ⛔ B never enters.  Resumable: one JSON per unit.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np, jax
from pathlib import Path
from chlu.core.gen_store import (address_codes, channel_correlation, dead_channels,
                                 grouping_labels, init_store, store_bytes, train_store,
                                 well_channels_from_labels, reconstruction_mse)
from chlu.core.gen_compose import read_compose
from mech_driver import build_context, score

TOTAL_BUDGET = 598_528          # ⭐ the shipped store (524,288) + shipped head (74,240)
HEAD_PER_KM = 232               # (K·M + Σ_f V_f·K·M)·4 with Σ_f V_f = 57  ->  232·K·M
KS = tuple(int(x) for x in os.environ.get("PS_KS", "5,16,32,64,128,256,512").split(","))
TRAIN = os.environ.get("PS_TRAIN", "0") == "1"
tag = "trained" if TRAIN else "init"
led = Path(os.environ['OUT']) / "mech" / f"partition_{tag}"
led.mkdir(parents=True, exist_ok=True)

ctx = build_context(0)
cfg = ctx["cfg"]
gs = cfg.gen_store
D = int(ctx["X_store"].shape[1])
Xc = np.asarray(ctx["X_store"], np.float32)[: int(gs.n_corr)]
dead = dead_channels(Xc)
corr = channel_correlation(Xc)
print(f"D={D} n_dead={int(np.asarray(dead).sum())}", flush=True)

for K in KS:
    M = TOTAL_BUDGET // (8192 + HEAD_PER_KM * K)
    if M < 2:
        print(f"[skip] K={K}: M={M} < 2"); continue
    unit = led / f"K{K}_M{M}.json"
    if unit.exists():
        print(f"[skip] {unit.name}", flush=True); continue
    t0 = time.time()
    labels, ginfo = grouping_labels(corr, K, "correlation", 0, gs.grouping_method,
                                    dead=dead, min_well_frac=float(gs.min_well_frac))
    wc = well_channels_from_labels(labels, K)
    sizes = [len(w) for w in wc]
    key = jax.random.PRNGKey(20_260_916 + K)
    k1, k2 = jax.random.split(key)
    store = init_store(ctx["X_store"], wc, int(M), ctx["feat_shape"], k1, seed=0)
    hist = None
    if TRAIN:
        store, hist, _ = train_store(store, ctx["X_store"], gs, k2, log=None)
    sb = store_bytes(store)
    head = HEAD_PER_KM * K * M
    total = int(sb["store_bytes"]) + head
    rec = {"K": K, "M": M, "well_sizes_minmax": [int(min(sizes)), int(max(sizes))],
           "store_bytes": int(sb["store_bytes"]), "head_bytes": int(head),
           "total_bytes": total, "total_budget": TOTAL_BUDGET,
           "within_budget": bool(total <= TOTAL_BUDGET),
           "bits_per_read": float(K * np.log2(M)), "trained": TRAIN,
           "train_steps": int(gs.train_steps) if TRAIN else 0}
    for which in ("val", "MS"):
        rows, X, lab = ctx["sets"][which]
        Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
        code = address_codes(store, Xf)
        rt = np.asarray(read_compose(store, code), np.float32).reshape(Xf.shape)
        rep = score(ctx, rt, lab, None)
        err = np.linalg.norm(rt - Xf, axis=1) / np.maximum(np.linalg.norm(Xf, axis=1), 1e-9)
        rec[which] = {"valid": rep["distinct_valid"], "n": rep["n"],
                      "exact": rep["exact_match_frac"], "distinct": rep["distinct"],
                      "relerr": float(err.mean()),
                      "n_distinct_codes": int(np.unique(np.asarray(code), axis=0).shape[0])}
    rec["wall_s"] = time.time() - t0
    json.dump(rec, open(unit, "w"), indent=2)
    v = rec["val"]
    print(f"K={K:4d} M={M:4d} bits={rec['bits_per_read']:7.1f} store={rec['store_bytes']:7d} "
          f"head={head:8d} total={total:8d} {'OK ' if rec['within_budget'] else '⛔OVER'} | "
          f"val VALID={v['valid']:4d}/{v['n']} exact={v['exact']:.4f} relerr={v['relerr']:.4f} "
          f"| MS VALID={rec['MS']['valid']:3d} exact={rec['MS']['exact']:.4f} ({rec['wall_s']:.0f}s)",
          flush=True)
print("done")
