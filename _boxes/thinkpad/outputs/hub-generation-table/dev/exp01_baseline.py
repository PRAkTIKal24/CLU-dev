"""DEV experiment 01 — reproduce the shipped CLU read on the DEV corner, with its
controls, and measure the achievability of the 0.5278 best-constant bound.

⛔ DEV numbers.  Nothing here touches eval_a / eval_b.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx, dev

t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
c = ctx.build_context(SEED)
s = dev.dev_setup(c)
ctx.log(f"DEV corner {s['corner']} · {s['dev_rows'].size} rows · removed {s['n_removed_from_pool']} from pool "
        f"· pool now {s['pool'].size}")

X_store = c["gather"](s["pool"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_dev = s["lab_dev"]
ctx.log(f"features gathered in {time.time()-t0:.1f}s · X_store {X_store.shape} X_dev {X_dev.shape}")

# --- the DEV corner's own best-constant bound (structural) ------------------
bound = float(np.mean([1.0 / np.unique(lab_dev[:, j]).size for j in range(6)]))
ctx.log(f"DEV corner best-constant structural bound = {bound:.4f}")

res = {"seed": SEED, "corner": s["corner"], "best_constant_bound": bound, "arms": {}}

# --- the shipped store ------------------------------------------------------
t = time.time()
store, sinfo = dev.build_store(X_store, c["feat_shape"], 5, 64, SEED, c["cfg"])
ctx.log(f"store K=5 M=64 trained in {time.time()-t:.1f}s · bytes={sinfo['store_bytes']['store_bytes']} "
        f"· bits/read={sinfo['store_bytes']['bits_per_read']} · wells={sinfo['well_sizes']}")
res["store_info"] = {"bytes": sinfo["store_bytes"]["store_bytes"],
                     "bits_per_read": sinfo["store_bytes"]["bits_per_read"],
                     "well_sizes": sinfo["well_sizes"]}

from chlu.core.gen_store import address_codes, delete_store
from chlu.core.gen_dream import fit_address_book, requests_from_labels
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior, store_removed_read

codes_train = address_codes(store, np.asarray(X_store, np.float32).reshape((X_store.shape[0], -1)))
book = fit_address_book(codes_train, lab_store, c["cards"], float(c["cfg"].gen_cell.address_smoothing))
req = requests_from_labels(lab_dev, "full", 0, SEED)
post = request_posterior(book, req)
cd = compose_codes(post)
ctx.log(f"request codes: {cd['n_distinct_code_configs']} distinct code configs of {req.shape[0]} "
        f"· mean confidence {cd['mean_confidence']:.4f}")
res["request_addressing"] = {"n_distinct_code_configs": int(cd["n_distinct_code_configs"]),
                             "mean_confidence": float(cd["mean_confidence"])}

# clu
feats = np.asarray(read_compose(store, cd["code"]), np.float32)
res["arms"]["clu"] = dev.score_features(c, feats, lab_dev)
ctx.log(f"  [clu] agg={res['arms']['clu']['aggregate']:.4f} distinct_tuples="
        f"{res['arms']['clu']['distinct_factor_tuples']}/72")

# store_removed
feats_sr = np.asarray(store_removed_read(store, cd["code"], "mean", SEED), np.float32)
res["arms"]["store_removed"] = dev.score_features(c, feats_sr, lab_dev)
ctx.log(f"  [store_removed] agg={res['arms']['store_removed']['aggregate']:.4f} distinct_tuples="
        f"{res['arms']['store_removed']['distinct_factor_tuples']}/72")

# --- ORACLE CEILING: decode the REAL dev features --------------------------
res["arms"]["decoder_ceiling"] = dev.score_features(c, X_dev, lab_dev)
ctx.log(f"  [decoder_ceiling] agg={res['arms']['decoder_ceiling']['aggregate']:.4f} distinct_tuples="
        f"{res['arms']['decoder_ceiling']['distinct_factor_tuples']}/72")

# --- best-constant ACHIEVABILITY: emit ONE real dev feature 72 times --------
best = None
for i in range(0, 72, 1):
    const = np.repeat(X_dev[i:i+1], 72, axis=0)
    r = dev.score_features(c, const, lab_dev)
    if best is None or r["aggregate"] > best["aggregate"]:
        best = r; best["row"] = int(i)
res["arms"]["best_constant_realised"] = best
ctx.log(f"  [best_constant_realised] agg={best['aggregate']:.4f} distinct_tuples="
        f"{best['distinct_factor_tuples']}/72 (row {best['row']}) · structural bound {bound:.4f}")

res["wall_clock_s"] = float(time.time() - t0)
p = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", f"exp01_seed{SEED}.json")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"wrote {p} in {res['wall_clock_s']:.0f}s")
