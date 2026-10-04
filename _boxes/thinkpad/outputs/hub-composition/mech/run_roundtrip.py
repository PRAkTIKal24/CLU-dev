"""⭐⭐ THE STORE ROUND-TRIP CEILING — the bound that governs EVERY request-only read.

Hand the store the TRUE feature, let it address itself, read it back, decode, classify.
⛔ Not an arm (it takes features, §B.3 bars it forever) — a BOUND. Whatever a request-only
read could ever address to, it must read OUT of this store, so no such read can beat this.

⛔ `val` / `MS` only. ⛔ B never enters.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.environ['OUT'], 'mech'))
import numpy as np
from chlu.core.gen_store import address_codes
from chlu.core.gen_compose import read_compose
from mech_driver import build_context, score

ctx = build_context(0)
store = ctx["store"]
res = {}
print(f"store: n_wells={len(store.well_channels) if hasattr(store,'well_channels') else '?'} "
      f"dim={getattr(store,'dim','?')}", flush=True)
for which in ("val", "MS"):
    rows, X, lab = ctx["sets"][which]
    Xf = np.asarray(X, np.float32).reshape((X.shape[0], -1))
    code = address_codes(store, Xf)                    # ⭐ the store addresses ITSELF
    rt = np.asarray(read_compose(store, code), np.float32)
    rep = score(ctx, rt, lab, None)
    err = np.linalg.norm(rt.reshape(Xf.shape) - Xf, axis=1) / np.maximum(
        np.linalg.norm(Xf, axis=1), 1e-9)
    ceil = score(ctx, X, lab, None)
    res[which] = {"roundtrip": rep, "ceiling": ceil, "relerr_mean": float(err.mean()),
                  "n_distinct_codes": int(np.unique(np.asarray(code), axis=0).shape[0])}
    print(f"ROUNDTRIP/{which:3s} n={rep['n']:4d}  store round-trip VALID={rep['distinct_valid']:4d} "
          f"(exact={rep['exact_match_frac']:.4f}, relerr={err.mean():.4f}, "
          f"distinct_codes={res[which]['n_distinct_codes']})   vs DECODER CEILING "
          f"VALID={ceil['distinct_valid']:4d} (exact={ceil['exact_match_frac']:.4f})", flush=True)
json.dump(res, open(os.path.join(os.environ['OUT'], 'mech', 'roundtrip.json'), 'w'), indent=2)
print("done")
