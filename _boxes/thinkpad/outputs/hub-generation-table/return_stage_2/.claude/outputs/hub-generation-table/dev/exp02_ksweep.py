"""DEV experiment 02 — ⭐ SEPARATE the two failures, and sweep K at CONSTANT BYTES.

For each K (M fixed at 64, so store_bytes = M·D·4 = 524,288 B for every K):

  quant_ceiling : codes taken from the FEATURES by nearest mode (⛔ a DIAGNOSTIC, not an
                  arm — it bypasses the request, so it can never be reported as a result).
                  This is the BEST any addresser could do through this codebook.
  clu           : codes from the REQUEST through the shipped naive-Bayes address book.
  store_removed : the laundering control.

  clu  <  quant_ceiling  ⇒ the gap is ADDRESSING.
  quant_ceiling low      ⇒ the gap is the CODEBOOK.

⛔ DEV numbers.  Nothing here touches eval_a / eval_b.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx, dev

t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["5", "10", "20", "40", "64"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", f"exp02_seed{SEED}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_dev = s["lab_dev"]
ctx.log(f"setup {time.time()-t0:.1f}s · pool {s['pool'].size} · dev {s['dev_rows'].size}")

from chlu.core.gen_store import address_codes, read_from_codes
from chlu.core.gen_dream import fit_address_book, requests_from_labels
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior, store_removed_read

res = {}
if os.path.exists(OUTP):
    res = json.load(open(OUTP, encoding="utf-8"))
res.setdefault("seed", SEED)
res.setdefault("corner", s["corner"])
res.setdefault("K", {})

req = requests_from_labels(lab_dev, "full", 0, SEED)
Xf = np.asarray(X_store, np.float32).reshape((X_store.shape[0], -1))
Xdf = np.asarray(X_dev, np.float32).reshape((X_dev.shape[0], -1))

for K in KS:
    if str(K) in res["K"]:
        ctx.log(f"K={K} already in ledger — skipped")
        continue
    tk = time.time()
    store, sinfo = dev.build_store(X_store, c["feat_shape"], K, 64, SEED, c["cfg"])
    row = {"bytes": sinfo["store_bytes"]["store_bytes"],
           "bits_per_read": sinfo["store_bytes"]["bits_per_read"],
           "well_sizes": sinfo["well_sizes"],
           "grouping_method_used": sinfo["grouping"]["method_used"],
           "train_s": round(time.time() - tk, 1)}
    # quant ceiling (DIAGNOSTIC — bypasses the request)
    oc = address_codes(store, Xdf)
    row["quant_ceiling"] = dev.score_features(c, np.asarray(read_from_codes(store, oc), np.float32), lab_dev)
    # the arm
    ct = address_codes(store, Xf)
    book = fit_address_book(ct, lab_store, c["cards"], float(c["cfg"].gen_cell.address_smoothing))
    cd = compose_codes(request_posterior(book, req))
    row["n_distinct_code_configs"] = int(cd["n_distinct_code_configs"])
    row["mean_confidence"] = float(cd["mean_confidence"])
    row["clu"] = dev.score_features(c, np.asarray(read_compose(store, cd["code"]), np.float32), lab_dev)
    row["store_removed"] = dev.score_features(
        c, np.asarray(store_removed_read(store, cd["code"], "mean", SEED), np.float32), lab_dev)
    # ⭐ code agreement: how often does the request find the feature's own code?
    row["code_agreement"] = float(np.mean(cd["code"] == oc))
    res["K"][str(K)] = row
    ctx.log(f"K={K:3d} bits={row['bits_per_read']:6.1f} bytes={row['bytes']} | "
            f"quant_ceiling={row['quant_ceiling']['aggregate']:.4f} ({row['quant_ceiling']['distinct_factor_tuples']}/72) | "
            f"clu={row['clu']['aggregate']:.4f} ({row['clu']['distinct_factor_tuples']}/72, "
            f"{row['n_distinct_code_configs']} codes) | "
            f"store_removed={row['store_removed']['aggregate']:.4f} ({row['store_removed']['distinct_factor_tuples']}/72) | "
            f"code_agree={row['code_agreement']:.3f} | {row['train_s']}s")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

ctx.log(f"done in {time.time()-t0:.0f}s -> {OUTP}")
