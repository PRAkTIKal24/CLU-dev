"""DEV-03 — the K sweep, cached and instrumented.

At every K (M = 64 fixed ⇒ store_bytes = M·D·4 = 524,288 B at EVERY K):

  quant_ceiling  codes from the TRUE features (⛔ DIAGNOSTIC, bypasses the request)
  clu_nb         codes from the REQUEST, shipped naive-Bayes head
  clu_disc       codes from the REQUEST, discriminatively-fitted head (SAME bytes)
  store_removed  the laundering control
  recon_mse      the hard read's feature-space error (cheap, no decoder)

⭐ Every scored read goes through the SHIPPED `gen_store.read_from_codes`, and
`assert_no_bypass` is run on the converted store, so nothing here is a second read path.
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
KS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                       ["5", "10", "20", "40", "64", "128"])]
PART = sys.argv[3] if len(sys.argv) > 3 else "balanced"
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp03_seed{SEED}_{PART}.json")

c = ctx.build_context(SEED)
s = dev.dev_setup(c)
with fast.timer("gather", ctx.log):
    X_store = c["gather"](s["pool"])
    X_dev = c["gather"](s["dev_rows"])
lab_store = c["factor_labels"](c["grid"], s["pool"])
lab_dev = s["lab_dev"]
D = int(np.prod(c["feat_shape"]))
ctx.log(f"pool {s['pool'].size} · dev {s['dev_rows'].size} · D={D}")

corr, dead, order_ward, order_avg = fast.channel_geometry(
    X_store, c["cfg"], f"s{SEED}_pool{s['pool'].size}", log=ctx.log)
ctx.log(f"dead channels {int(dead.sum())} / {D}")

from chlu.core.gen_address import as_address_book, fit_discriminative_address_book  # noqa: E402
from chlu.core.gen_compose import compose_codes, read_compose, request_posterior, store_removed_read  # noqa: E402
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_store import assert_no_bypass, read_from_codes  # noqa: E402

req = requests_from_labels(lab_dev, "full", 0, SEED)
Zs = np.asarray(X_store, np.float32).reshape((-1, D))
Zd = np.asarray(X_dev, np.float32).reshape((-1, D))

res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("partition", PART)
res.setdefault("corner", s["corner"])
res.setdefault("K", {})


def score(feats):
    return dev.score_features(c, feats, lab_dev)


for K in KS:
    if str(K) in res["K"]:
        ctx.log(f"K={K} already in ledger — skipped")
        continue
    tk = time.time()
    with fast.timer(f"partition", ctx.log):
        if PART == "balanced":
            labels = fast.balanced_labels(order_ward, dead, K, D)
            pinfo = {"partition": "balanced_correlation(ward leaf order)"}
        else:
            labels, gi = fast.ward_labels(corr, dead, K)
            pinfo = {"partition": "shipped correlation_grouping", "method_used": gi["method_used"]}
    sizes = np.bincount(labels, minlength=K)
    row = {"well_sizes": [int(x) for x in sizes], **pinfo}

    if PART != "balanced":
        # unequal wells: fall back to the shipped trainer via dev.build_store
        from chlu.core.gen_store import init_store, train_store, well_channels_from_labels
        import jax
        wc = well_channels_from_labels(labels, K)
        key = jax.random.PRNGKey(int(SEED) * 10_007 + 20_260_911)
        key, ik, tkk = jax.random.split(key, 3)
        with fast.timer("init+train(shipped)", ctx.log):
            st = init_store(X_store, wc, 64, c["feat_shape"], ik, seed=int(SEED))
            st, hist, key = train_store(st, X_store, c["cfg"].gen_store, tkk, log=None)
        store = st
        codes_dev_oracle = None
    else:
        with fast.timer("kmeans_pq", ctx.log):
            pq = fast.kmeans_pq(Zs, labels, K, 64, SEED, iters=25, log=None)
        with fast.timer("to_gen_store", ctx.log):
            store = pq.to_gen_store()
        codes_dev_oracle = None

    from chlu.core.gen_store import address_codes, store_bytes
    sb = store_bytes(store)
    row["bytes"] = int(sb["store_bytes"])
    row["bits_per_read"] = float(sb["bits_per_read"])
    with fast.timer("address_codes(train)", ctx.log):
        ct = address_codes(store, Zs)
    with fast.timer("address_codes(dev)", ctx.log):
        cdv = address_codes(store, Zd)

    # recon mse of the hard read on the DEV rows (cheap, no decoder)
    rec = np.asarray(read_from_codes(store, cdv), np.float32).reshape((-1, D))
    row["recon_mse_dev"] = float(np.mean((rec - Zd) ** 2))
    row["feature_var_dev"] = float(np.var(Zd))

    with fast.timer("score:quant_ceiling", ctx.log):
        row["quant_ceiling"] = score(rec)

    with fast.timer("fit_nb", ctx.log):
        nb = fit_address_book(ct, lab_store, c["cards"], 1.0)
    cd_nb = compose_codes(request_posterior(nb, req))
    row["nb_codes"] = int(cd_nb["n_distinct_code_configs"])
    row["nb_code_agreement"] = float(np.mean(cd_nb["code"] == cdv))
    with fast.timer("score:clu_nb", ctx.log):
        row["clu_nb"] = score(np.asarray(read_compose(store, cd_nb["code"]), np.float32))

    with fast.timer("fit_disc", ctx.log):
        dbk = fit_discriminative_address_book(ct, lab_store, c["cards"], l2=1e-3,
                                              steps=150, lr=0.5)
    cd_d = compose_codes(request_posterior(as_address_book(dbk), req))
    row["disc_codes"] = int(cd_d["n_distinct_code_configs"])
    row["disc_code_agreement"] = float(np.mean(cd_d["code"] == cdv))
    with fast.timer("score:clu_disc", ctx.log):
        row["clu_disc"] = score(np.asarray(read_compose(store, cd_d["code"]), np.float32))

    with fast.timer("score:store_removed", ctx.log):
        row["store_removed"] = score(
            np.asarray(store_removed_read(store, cd_d["code"], "mean", SEED), np.float32))

    with fast.timer("assert_no_bypass", ctx.log):
        bp = assert_no_bypass(store, Zd[:64], rng_seed=int(SEED))
    row["no_bypass"] = {"bits_per_read": float(bp["bits_per_read"]),
                        "pure_codebook_gather": bool(bp["read_is_pure_codebook_gather"]),
                        "max_abs_delta": float(bp["max_abs_delta_vs_independent_gather"])}

    from chlu.core.gen_compose import address_book_bytes
    row["head_bytes_nb"] = int(address_book_bytes(nb)["bytes"])
    row["head_bytes_disc"] = int(address_book_bytes(as_address_book(dbk))["bytes"])
    row["total_bytes_disc"] = row["bytes"] + row["head_bytes_disc"]
    row["seconds"] = round(time.time() - tk, 1)

    res["K"][str(K)] = row
    ctx.log(
        f"K={K:4d} bits={row['bits_per_read']:8.1f} store={row['bytes']} head={row['head_bytes_disc']} "
        f"| mse={row['recon_mse_dev']:.4f} "
        f"| ceil={row['quant_ceiling']['aggregate']:.4f}({row['quant_ceiling']['distinct_factor_tuples']}) "
        f"| nb={row['clu_nb']['aggregate']:.4f}({row['clu_nb']['distinct_factor_tuples']},{row['nb_codes']}c) "
        f"| disc={row['clu_disc']['aggregate']:.4f}({row['clu_disc']['distinct_factor_tuples']},{row['disc_codes']}c) "
        f"| sr={row['store_removed']['aggregate']:.4f}({row['store_removed']['distinct_factor_tuples']}) "
        f"| {row['seconds']}s")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

res["timings"] = fast.times()
res["total_s"] = round(time.time() - T0, 1)
with open(OUTP, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
ctx.log(f"done in {res['total_s']}s -> {OUTP}")
ctx.log("timings: " + json.dumps({k: round(v, 1) for k, v in fast.times().items()}))
