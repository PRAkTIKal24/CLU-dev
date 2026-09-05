"""CAMELS mandatory companion rows + the LOCAL (same-basin) exemplar arms.

Companion rows (PREREG §4):
  mean_train  per-basin training-period mean flow
  mean_test   per-basin TEST-period mean flow  (NSE = 0 by definition)
  doy_clim    per-basin day-of-year climatology from the training period
              (store = 531 x 366 float32 = 777,384 B -> INSIDE the byte budget)
  persistence q(t-1)  -- OUT OF PROTOCOL, a DIFFERENT TASK, labelled as such

LOCAL exemplar arms: same-basin-only store, statics dropped (constant in-basin).
"""
import argparse
import itertools
import json

import numpy as np

from camels_tripwire import (BUDGET_B, KS, OUTD, Arm, Camels, nse_per_basin,
                             windows)


def companions(C, out):
    te = C.te
    b447 = set(json.load(open("/Users/user/Desktop/CHLU/.claude/scratch/"
                              "c3-trackb-tripwire/ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    rows = []
    doy = np.array([int(str(d)[5:7]) * 100 + int(str(d)[8:10])
                    for d in C.dates])
    uniq = np.unique(doy)
    dmap = {v: i for i, v in enumerate(uniq)}
    clim = np.full((C.nb, len(uniq)), np.nan)
    for i in range(C.nb):
        for v in uniq:
            m = (doy == v)
            q = C.Q[i, m & np.isin(np.arange(len(doy)), C.tr)]
            q = q[q >= 0]
            if len(q):
                clim[i, dmap[v]] = q.mean()
        col = clim[i]
        col[np.isnan(col)] = np.nanmean(col)

    for name in ("mean_train", "mean_test", "doy_clim", "persistence"):
        v = np.full(C.nb, np.nan)
        for i in range(C.nb):
            obs = C.Q[i, te]
            if name == "mean_train":
                sim = np.full(len(te), C.qmu[i])
            elif name == "mean_test":
                o = obs[obs >= 0]
                sim = np.full(len(te), o.mean())
            elif name == "doy_clim":
                sim = clim[i][[dmap[d] for d in doy[te]]]
            else:
                sim = C.Q[i, te - 1]
                bad = sim < 0
                sim = sim.copy()
                sim[bad] = C.qmu[i]
            v[i] = nse_per_basin(obs, sim)
        nb = dict(mean_train=531 * 4, mean_test=531 * 4,
                  doy_clim=int(531 * len(uniq) * 4), persistence=4)[name]
        rows.append(dict(venue="camels", arm=name, store="companion",
                         bytes=nb, in_budget=bool(nb <= BUDGET_B),
                         in_protocol=(name != "persistence"),
                         median_nse_447=float(np.nanmedian(v[m447])),
                         mean_nse_447=float(np.nanmean(v[m447])),
                         n_le0_447=int(np.sum(v[m447] <= 0)),
                         median_nse_531=float(np.nanmedian(v)),
                         mean_nse_531=float(np.nanmean(v))))
        np.save(OUTD / f"perbasin_companion_{name}.npy", v)
    with open(out, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for r in rows:
        print(r["arm"], round(r["median_nse_447"], 4), r["bytes"], flush=True)


def local_arms(C, W, scaling, Ls, seeds, out, sub=1):
    F = C.scaled_forcing(scaling)
    te = C.te[C.te >= W - 1]
    if sub > 1:
        te = te[::sub]
    D = W * 5
    b447 = set(json.load(open("/Users/user/Desktop/CHLU/.claude/scratch/"
                              "c3-trackb-tripwire/ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    keys = list(itertools.product(Ls, seeds, KS, ("dist", "unif"),
                                  ("raw", "std")))
    res = {k: np.full(C.nb, np.nan) for k in keys}
    for i in range(C.nb):
        wv = windows(F[i], W)
        tr = C.tr[(C.tr >= W - 1) & (C.Q[i, C.tr] >= 0)]
        Qy = np.ascontiguousarray(wv[te - (W - 1)])
        obs = C.Q[i, te]
        for L in Ls:
            for sd in seeds:
                rng = np.random.default_rng(1000 * sd + i)
                sel = tr if L >= len(tr) else rng.choice(tr, L, replace=False)
                k_ = np.ascontiguousarray(wv[sel - (W - 1)])
                qq = C.Q[i, sel]
                qz = (qq - C.qmu[i]) / C.qsd[i]
                a = Arm(k_, qq, qz, "loc")
                pr = a.predict(Qy)
                for (k, wt, tg), p in pr.items():
                    if tg == "std":
                        p = p * C.qsd[i] + C.qmu[i]
                    res[(L, sd, k, wt, tg)][i] = nse_per_basin(obs, p)
        if i % 100 == 0:
            print("  local basin", i, flush=True)
    with open(out, "a") as fh:
        for (L, sd, k, wt, tg), v in res.items():
            nb = int(C.nb * min(L, 3287) * (D + 1) * 4)
            fh.write(json.dumps(dict(
                venue="camels", store="local", window=W, scaling=scaling,
                dim=D, L_per_basin=L, seed=sd, k=k, weight=wt, target=tg,
                sub=sub, bytes_total=nb, in_budget=bool(nb <= BUDGET_B),
                budget_multiple=round(nb / BUDGET_B, 2),
                median_nse_447=float(np.nanmedian(v[m447])),
                mean_nse_447=float(np.nanmean(v[m447])),
                n_le0_447=int(np.sum(v[m447] <= 0)),
                median_nse_531=float(np.nanmedian(v)),
                mean_nse_531=float(np.nanmean(v)))) + "\n")
    np.savez_compressed(OUTD / f"perbasin_local_{W}_{scaling}.npz",
                        **{f"{a}|{b}|{c}|{d}|{e}": v
                           for (a, b, c, d, e), v in res.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--scaling", default="perbasin")
    ap.add_argument("--Ls", default="5,250,1000,3287")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--out", default=str(OUTD / "arms.jsonl"))
    a = ap.parse_args()
    C = Camels("daymet")
    if a.mode == "companions":
        companions(C, a.out)
    else:
        local_arms(C, a.window, a.scaling,
                   [int(x) for x in a.Ls.split(",")],
                   [int(x) for x in a.seeds.split(",")], a.out, a.sub)
