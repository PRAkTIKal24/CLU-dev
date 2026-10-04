"""DEV-27 — ⭐ the §C.1 floor, made as HARD as I can make it.

DEV-24's atom control picks `M` train rows **uniformly at random** per seed.  That is the
easy version of "store real rows", and my §C.1 margin over it is only
`+0.0180 [+0.0005, +0.0356]` at n = 49 — marginal enough that the choice of atom selector
could decide it.  ⛔ Reporting a §C.1 pass against the easiest possible floor would be
exactly the kind of unearned control the charter exists to forbid.

So this measures three harder atom selectors, all at IDENTICAL bytes and identical head
configuration:

  atoms_random    M uniformly random train rows                 (DEV-24's control)
  atoms_kpp       M rows chosen by k-means++ ON THE ROWS         ⭐ maximally spread real rows
  atoms_medoid    for each of the M k-means clusters, the real row NEAREST its centroid
                  ⭐ the strongest "lookup table" there is: it is the learned partition of
                  the data, represented by actual stored rows instead of fitted centres
  modes           the arm's learned k-means centroids

⭐ `atoms_medoid` is the sharpest possible §C.1 test: it differs from the arm in exactly
ONE thing — whether the dictionary entry is a **fitted centre** or the **real row closest
to it**.  If the medoid dictionary matches the arm, the fitting buys nothing and the store
is a lookup table.

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
NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 24
K, M, RANK, RIDGE = 32, 26, 16, 10.0
PAIR_NAMES = ["floor_huexscale", "object_huexscale"]
OUTP = os.path.join(os.environ["CHLU_OUT"].replace("/c/", "C:/", 1), "dev",
                    f"exp27_seed{SEED}.json")

CORNERS = [dict(hue_lo=h, scale_hi=sc, shape_value=sh, orient_hi=o)
           for h in (0.0, 0.2, 0.4, 0.6) for sh in (0.0, 1.0, 2.0, 3.0)
           for sc in (True, False) for o in (True, False)]

c = ctx.build_context(SEED)
D = int(np.prod(c["feat_shape"]))
cards = c["cards"]
F = len(cards)

from chlu.core.gen_compose import clu_wide_vector  # noqa: E402
from chlu.core.gen_dream import requests_from_labels  # noqa: E402
from chlu.core.gen_factored_read import default_pairs, fit_factored_head  # noqa: E402
from chlu.core.gen_store import store_bytes  # noqa: E402

PAIRS = default_pairs(c["names"], PAIR_NAMES)
res = json.load(open(OUTP, encoding="utf-8")) if os.path.exists(OUTP) else {}
res.setdefault("seed", SEED)
res.setdefault("config", {"K": K, "M": M, "rank": RANK, "ridge": RIDGE, "pairs": PAIR_NAMES})
res.setdefault("corners", {})
KINDS = ("modes", "atoms_random", "atoms_kpp", "atoms_medoid")

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
    except (dev.HeldOutOverlap, AssertionError) as exc:
        continue
    t0 = time.time()
    X_store = c["gather"](s["pool"])
    lab_store = c["factor_labels"](c["grid"], s["pool"])
    lab_dev = s["lab_dev"]
    Zs = np.asarray(X_store, np.float32).reshape((-1, D))
    corr, dead, ow, oa = fast.channel_geometry(
        X_store, c["cfg"], f"s{SEED}_c{ci}_n{s['pool'].size}", log=None)
    labels = fast.balanced_labels(ow, dead, K, D)
    perm = np.stack([np.flatnonzero(labels == k) for k in range(K)])
    req = requests_from_labels(lab_dev, "full", 0, SEED)

    pq = fast.kmeans_pq(Zs, labels, K, M, SEED, iters=25)
    modes = pq.modes.astype(np.float32)                       # (K, M, d)
    bases = {"modes": modes}

    rng = np.random.default_rng(int(SEED) * 7919 + 20_260_916 + 17 * ci)
    ar = rng.choice(Zs.shape[0], M, replace=False)
    bases["atoms_random"] = np.stack([Zs[ar][:, perm[k]] for k in range(K)]).astype(np.float32)

    # ⭐ k-means++ ON THE ROWS: M maximally-spread real rows, shared across wells
    from chlu.core.gen_store import _kmeanspp_init
    sub = rng.choice(Zs.shape[0], min(8192, Zs.shape[0]), replace=False)
    seedsM = _kmeanspp_init(Zs[sub], M, np.random.default_rng(int(SEED) * 13 + ci))
    d2 = ((Zs[sub][:, None, :] - seedsM[None]) ** 2).sum(-1) if False else None
    # nearest real row (in the full D space) to each k-means++ seed
    idx_kpp = []
    for m in range(M):
        dd = ((Zs[sub] - seedsM[m]) ** 2).sum(axis=1)
        idx_kpp.append(int(sub[int(np.argmin(dd))]))
    idx_kpp = np.array(sorted(set(idx_kpp)), np.int64)
    while idx_kpp.size < M:
        extra = rng.choice(Zs.shape[0], M - idx_kpp.size, replace=False)
        idx_kpp = np.array(sorted(set(idx_kpp.tolist()) | set(extra.tolist())), np.int64)[:M]
    bases["atoms_kpp"] = np.stack([Zs[idx_kpp][:, perm[k]] for k in range(K)]).astype(np.float32)

    # ⭐⭐ MEDOIDS: per well, the real row whose well-slice is nearest that well's centroid
    med = np.empty_like(modes)
    for k in range(K):
        V = Zs[:, perm[k]]
        mm = np.einsum("md,md->m", modes[k], modes[k])
        cross = V @ modes[k].T
        for m in range(M):
            med[k, m] = V[int(np.argmin(mm[m] - 2.0 * cross[:, m]))]
    bases["atoms_medoid"] = med.astype(np.float32)

    row = {"corner": corner}
    for kind, bas in bases.items():
        st = fast.EqualPQ(perm, bas, (D,)).to_gen_store()
        head = fit_factored_head(st, Zs, lab_store, cards, PAIRS, RANK, RIDGE)
        feats = np.asarray(clu_wide_vector(st, head.coefficients(req)), np.float32)
        row[kind] = dev.score_features(c, feats, lab_dev)
        if kind == "modes":
            row["total_bytes"] = int(store_bytes(st)["store_bytes"] + head.byte_ledger()["bytes"])
    row["seconds"] = round(time.time() - t0, 1)
    res["corners"][ck] = row
    done += 1
    ctx.log(f"[c{ci}] " + "  ".join(
        f"{k}={row[k]['aggregate']:.4f}({row[k]['distinct_factor_tuples']})" for k in KINDS)
        + f"  [{row['seconds']}s]")
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)

rows = list(res["corners"].values())
if rows:
    n = len(rows)
    g = lambda k: np.array([r[k]["aggregate"] for r in rows])
    t_ = lambda k: np.array([r[k]["distinct_factor_tuples"] for r in rows])
    summ = {"n_corners": n}
    for k in KINDS:
        summ[k] = {"mean": float(g(k).mean()), "sd": float(g(k).std(ddof=1)),
                   "tuples": float(t_(k).mean())}
    for k in KINDS[1:]:
        v = g("modes") - g(k)
        se = v.std(ddof=1) / np.sqrt(n)
        summ[f"delta_vs_{k}"] = {"mean": float(v.mean()), "se": float(se),
                                 "ci95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
                                 "wins": int((v > 0).sum()), "t": float(v.mean() / se)}
    res["summary"] = summ
    with open(OUTP, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    ctx.log(f"[SUMMARY n={n}] " + "  ".join(
        f"{k}={summ[k]['mean']:.4f}({summ[k]['tuples']:.1f})" for k in KINDS))
    for k in KINDS[1:]:
        d_ = summ[f"delta_vs_{k}"]
        ctx.log(f"   arm - {k:<14} {d_['mean']:+.4f}  SE {d_['se']:.4f}  "
                f"95%CI [{d_['ci95'][0]:+.4f},{d_['ci95'][1]:+.4f}]  "
                f"wins {d_['wins']}/{n}  t={d_['t']:+.2f}")
ctx.log(f"done in {time.time() - T0:.0f}s -> {OUTP}")
