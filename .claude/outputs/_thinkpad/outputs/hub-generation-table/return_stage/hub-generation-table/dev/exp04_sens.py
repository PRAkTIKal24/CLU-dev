"""DEV-04 — is the codebook spending its bits where the DECODER looks?

The shipped store is fitted to minimise FEATURE-space distortion.  The table reads a
classifier's accuracy on the DECODED IMAGE.  This measures ``diag(JᵀJ)`` of the frozen
decoder (Hutchinson), reports how far it is from the feature variance the unweighted fit
chases, and then refits the codebook with that weighting — ⭐ at **identical store bytes**,
because only the ASSIGNMENT changes, never what is stored.

⛔ DEV numbers.  eval_a / eval_b are never touched.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctx  # noqa: E402
import dev  # noqa: E402
import fast  # noqa: E402

T0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["16", "32"])]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp04_seed{SEED}.json")
CACHE = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev", "cache")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
X_store = c["gather"](s["pool"])
X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
Zd = np.asarray(X_dev, np.float32).reshape((-1, D))
corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)

# --- decoder sensitivity, cached ------------------------------------------
from chlu.core.gen_sensitivity import decoder_sensitivity, sensitivity_report  # noqa: E402

wpath = os.path.join(CACHE, f"dec_sens_s{SEED}.npz")
if os.path.exists(wpath):
    w = np.load(wpath)["w"]
    ctx.log("decoder sensitivity loaded from cache")
else:
    with fast.timer("decoder_sensitivity", ctx.log):
        sens = decoder_sensitivity(c["decoder"], Zs[:512], c["feat_shape"],
                                   n_probe=8, batch=32, seed=SEED)
    w = sens["w"]
    np.savez_compressed(wpath, w=w)
rep = sensitivity_report(w, Zs[:4096])
ctx.log("sensitivity report: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                             for k, v in rep.items() if k != "note_corr"}))

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("sensitivity_report", {k: v for k, v in rep.items()})
res.setdefault("K", {})

from chlu.core.gen_address import as_address_book, fit_discriminative_address_book  # noqa: E402
from chlu.core.gen_compose import (address_book_bytes, compose_codes, read_compose,  # noqa: E402
                                   request_posterior, store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import read_from_codes, store_bytes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)


def arm_rows(pq, wts, tag):
    """Every row for one fitted codebook: reference, NB head, disc head, control."""
    store = pq.to_gen_store()
    sb = store_bytes(store)
    out = {"bytes": int(sb["store_bytes"]), "bits_per_read": float(sb["bits_per_read"])}
    cdv = fast.weighted_codes(pq, Zd, wts)
    ct = fast.weighted_codes(pq, Zs, wts)
    rec = np.asarray(read_from_codes(store, cdv), np.float32).reshape((-1, D))
    out["recon_mse_dev_unweighted"] = float(np.mean((rec - Zd) ** 2))
    out["recon_mse_dev_weighted"] = float(
        np.mean(np.asarray(w, np.float64)[None] * (rec - Zd) ** 2))
    out["nearest_mode_reference"] = dev.score_features(c, rec, lab_dev)
    nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
    cd_nb = compose_codes(request_posterior(nb, req))
    out["clu_nb"] = dev.score_features(
        c, np.asarray(read_compose(store, cd_nb["code"]), np.float32), lab_dev)
    out["nb_codes"] = int(cd_nb["n_distinct_code_configs"])
    dbk = fit_discriminative_address_book(ct, lab_store, c["cards"], l2=1e-3,
                                          steps=150, lr=0.5)
    cd_d = compose_codes(request_posterior(as_address_book(dbk), req))
    out["clu_disc"] = dev.score_features(
        c, np.asarray(read_compose(store, cd_d["code"]), np.float32), lab_dev)
    out["disc_codes"] = int(cd_d["n_distinct_code_configs"])
    out["store_removed"] = dev.score_features(
        c, np.asarray(store_removed_read(store, cd_d["code"], "mean", SEED), np.float32),
        lab_dev)
    out["head_bytes"] = int(address_book_bytes(nb)["bytes"])
    ctx.log(f"  [{tag}] mse={out['recon_mse_dev_unweighted']:.4f} "
            f"ref={out['nearest_mode_reference']['aggregate']:.4f}"
            f"({out['nearest_mode_reference']['distinct_factor_tuples']}) "
            f"nb={out['clu_nb']['aggregate']:.4f}({out['clu_nb']['distinct_factor_tuples']}) "
            f"disc={out['clu_disc']['aggregate']:.4f}({out['clu_disc']['distinct_factor_tuples']}) "
            f"sr={out['store_removed']['aggregate']:.4f}")
    return out


for K in KS:
    res["K"].setdefault(str(K), {})
    labels = fast.balanced_labels(order_ward, dead, K, D)
    for tag, wts in (("unweighted", None), ("decoder_weighted", w)):
        if tag in res["K"][str(K)]:
            ctx.log(f"K={K} {tag} already in ledger — skipped")
            continue
        t = time.time()
        pq = fast.kmeans_pq_weighted(Zs, labels, K, 64, SEED, wts, iters=25)
        row = arm_rows(pq, wts, f"K={K} {tag}")
        row["fit_seconds"] = round(time.time() - t, 1)
        res["K"][str(K)][tag] = row
        with open(OUTP, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)

res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
