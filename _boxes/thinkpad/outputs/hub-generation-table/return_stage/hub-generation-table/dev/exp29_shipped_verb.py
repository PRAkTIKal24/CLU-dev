"""DEV-29 — ⭐⭐ THE SHIPPED STORE, THE SHIPPED BYTES, ONLY THE READ VERB CHANGED.

This is the experiment my top recommendation rests on, and it is deliberately the most
conservative one in the whole run:

  * the store is the SHIPPED one — `K = 5` ward-grouped wells, `M = 64` modes, fitted by
    the SHIPPED `gen_store.train_store` (the soft/commitment Adam objective, ⛔ not my
    Lloyd variant) — **524,288 B**;
  * the head is the SHIPPED SIZE — the additive block alone is `58 · K·M · 4` =
    `58 · 320 · 4` = **74,240 B**, ⭐ the shipped addressing head **to the byte**;
  * ⇒ `524,288 + 74,240 = 598,528 B`, **byte-identical in every column to the shipped
    `clu` arm**;
  * the ONLY change is `read_from_codes(store, argmax)`  →
    `clu_wide_vector(store, real-valued request-addressed weights)`.

Rows per corner, all on the SAME fitted store:
  clu_argmax        the shipped arm (naive-Bayes head, argmax)      598,528 B
  linear_additive   the linear read, additive head only             598,528 B  ⭐ byte-identical
  linear_pairs16    + rank-16 interactions over 2 pair terms        629,248 B  ⚠ +5 %, declared
  store_removed     §C.2, same coefficients, modes -> well mean
  atoms_medoid      §C.1, the real row nearest each learned centre
  knn_proto         the ceiling arm                                 475,136 B

⛔ DEV corners only.  eval_a / eval_b are never touched.  Nothing is selected here.
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
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp29_seed{SEED}.json")

CORNERS = [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
           for h in (0.0, 0.2, 0.4, 0.6) for sh in (0.0, 1.0, 2.0, 3.0)
           for sc in (True, False) for o in (True, False)]

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]

from chlu.core.gen_compose import (address_book_bytes, clu_wide_vector,  # noqa: E402
                                   compose_codes, deployed_byte_ledger, read_compose,
                                   request_posterior, store_removed_read)
from chlu.core.gen_dream import fit_address_book, requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_request import FactorPrototypes, requests_from_rows  # noqa: E402
from chlu.core.gen_store import address_codes, delete_store, store_bytes  # noqa: E402

PAIRS = default_pairs(c["names"], PAIR_NAMES)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("corners", {})

done = 0
for ci, corner in enumerate(CORNERS):
    if done >= NMAX:
        break
    ck = f"c{ci}"
    if ck in res["corners"]:
        done += 1
        continue
    try:
        s = dev.dev_setup(c, corner)
    except (dev.HeldOutOverlap, AssertionError):
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    req = requests_from_labels(lab_dev, "full", 0, SEED)

    # ⭐ THE SHIPPED STORE: ward grouping at the plan's K=5, the shipped Adam trainer.
    store, sinfo = dev.build_store(X_store, c["feat_shape"], 5, 64, SEED, c["cfg"])
    sb = store_bytes(store)
    row = {"corner": corner, "well_sizes": sinfo["well_sizes"],
           "store_bytes": int(sb["store_bytes"]),
           "bits_per_read_argmax": float(sb["bits_per_read"])}

    # --- the shipped arm ---------------------------------------------------
    ct = address_codes(store, Zs)
    nb = fit_address_book(ct, lab_store, cards, 1.0)
    led = deployed_byte_ledger(store, nb)
    row["shipped_head_bytes"] = int(led["addressing_head"]["bytes"])
    cd = compose_codes(request_posterior(nb, req))
    row["clu_argmax"] = dev.score_features(
        c, np.asarray(read_compose(store, cd["code"]), np.float32), lab_dev)
    row["clu_argmax"]["total_bytes"] = int(sb["store_bytes"] + row["shipped_head_bytes"])
    row["store_removed_argmax"] = dev.score_features(
        c, np.asarray(store_removed_read(store, cd["code"], "mean", SEED), np.float32),
        lab_dev)

    # --- the SAME store, the linear read -----------------------------------
    for tag, pairs, rank in (("linear_additive", [], 0), ("linear_pairs16", PAIRS, 16)):
        head = fit_factored_head(store, Zs, lab_store, cards, pairs, rank, 10.0)
        hb = head.byte_ledger()["bytes"]
        a = head.coefficients(req)
        row[tag] = dev.score_features(
            c, np.asarray(clu_wide_vector(store, a), np.float32), lab_dev)
        row[tag]["head_bytes"] = int(hb)
        row[tag]["total_bytes"] = int(sb["store_bytes"] + hb)
        row[tag]["byte_identical_to_shipped_clu"] = bool(
            int(hb) == int(row["shipped_head_bytes"]))
        if tag == "linear_additive":
            sr = delete_store(store, "mean", SEED)
            row["store_removed_linear"] = dev.score_features(
                c, np.asarray(clu_wide_vector(sr, a), np.float32), lab_dev)
            # §C.1: the real row nearest each learned mode, same bytes
            wc = [np.asarray(w, np.int64) for w in store.well_channels]
            med = []
            for k, w in enumerate(wc):
                Mk = np.asarray(store.modes[k], np.float32)
                V = Zs[:, w]
                mm = np.einsum("md,md->m", Mk, Mk)
                cross = V @ Mk.T
                med.append(np.stack([V[int(np.argmin(mm[m] - 2.0 * cross[:, m]))]
                                     for m in range(Mk.shape[0])]))
            import equinox as eqx
            import jax.numpy as jnp
            mstore = eqx.tree_at(lambda t: t.modes, store,
                                 [jnp.asarray(x, jnp.float32) for x in med])
            mhead = fit_factored_head(mstore, Zs, lab_store, cards, [], 0, 10.0)
            row["atoms_medoid_linear"] = dev.score_features(
                c, np.asarray(clu_wide_vector(mstore, mhead.coefficients(req)), np.float32),
                lab_dev)

    protos = FactorPrototypes.fit(c["grid"], X_store, s["pool"])
    row["knn_proto"] = dev.score_features(
        c, np.asarray(protos.embed_many(requests_from_rows(c["grid"], s["dev_rows"])),
                      np.float32), lab_dev)
    row["knn_proto"]["deployed_bytes"] = int(protos.byte_ledger()["bytes"])
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ck] = row
    done += 1
    ctx.log(f"[c{ci}] argmax={row['clu_argmax']['aggregate']:.4f}"
            f"({row['clu_argmax']['distinct_factor_tuples']}) "
            f"linear={row['linear_additive']['aggregate']:.4f}"
            f"({row['linear_additive']['distinct_factor_tuples']}) "
            f"linear+p16={row['linear_pairs16']['aggregate']:.4f}"
            f"({row['linear_pairs16']['distinct_factor_tuples']}) "
            f"| sr_lin={row['store_removed_linear']['aggregate']:.4f} "
            f"medoid={row['atoms_medoid_linear']['aggregate']:.4f} "
            f"knn={row['knn_proto']['aggregate']:.4f} "
            f"| store={row['store_bytes']} head={row['linear_additive']['head_bytes']} "
            f"identical={row['linear_additive']['byte_identical_to_shipped_clu']} "
            f"[{row['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

rows = list(res["corners"].values())
if rows:
    n = len(rows)
    KEYS = ("clu_argmax", "linear_additive", "linear_pairs16", "store_removed_argmax",
            "store_removed_linear", "atoms_medoid_linear", "knn_proto")
    g = lambda k: np.array([r[k]["aggregate"] for r in rows])
    t_ = lambda k: np.array([r[k]["distinct_factor_tuples"] for r in rows])
    summ = {"n_corners": n,
            "store_bytes": rows[0]["store_bytes"],
            "shipped_head_bytes": rows[0]["shipped_head_bytes"],
            "linear_head_bytes": rows[0]["linear_additive"]["head_bytes"],
            "byte_identical": rows[0]["linear_additive"]["byte_identical_to_shipped_clu"]}
    for k in KEYS:
        summ[k] = {"mean": float(g(k).mean()), "sd": float(g(k).std(ddof=1)),
                   "tuples": float(t_(k).mean())}
    for a_, b_ in (("linear_additive", "clu_argmax"), ("linear_pairs16", "clu_argmax"),
                   ("linear_additive", "atoms_medoid_linear"),
                   ("linear_additive", "store_removed_linear"),
                   ("linear_additive", "knn_proto"), ("linear_pairs16", "knn_proto")):
        v = g(a_) - g(b_)
        se = v.std(ddof=1) / np.sqrt(n)
        summ[f"{a_}_minus_{b_}"] = {"mean": float(v.mean()), "se": float(se),
                                    "ci95": [float(v.mean() - 1.96 * se),
                                             float(v.mean() + 1.96 * se)],
                                    "wins": int((v > 0).sum()), "t": float(v.mean() / se)}
    res["summary"] = summ
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    ctx.log(f"[SUMMARY n={n}] store={summ['store_bytes']} shipped_head={summ['shipped_head_bytes']} "
            f"linear_head={summ['linear_head_bytes']} byte_identical={summ['byte_identical']}")
    for k in KEYS:
        ctx.log(f"   {k:<24} {summ[k]['mean']:.4f} ± {summ[k]['sd']:.4f}  "
                f"({summ[k]['tuples']:.1f} tuples)")
    for k, v in summ.items():
        if k.endswith("_minus_clu_argmax") or "_minus_" in k and isinstance(v, dict) and "ci95" in v:
            ctx.log(f"   {k:<44} {v['mean']:+.4f}  SE {v['se']:.4f}  "
                    f"95%CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  wins {v['wins']}/{n}  t={v['t']:+.2f}")
ctx.log(f"done in {time.time() - T0:.0f}s -> {OUTP}")
