"""DECLARED POST-HOC, UNREGISTERED extra (C2W10 precedent: the 'unregistered
but stronger' SAM-kNN L=1000 row).  Not an exemplar store, therefore NOT part
of the registered criterion-4 arithmetic -- reported so that nobody can claim
the classical side was hobbled.

  per-basin ridge on the W-day forcing window   (531 x (5W+1) x 4 bytes)
  pooled  ridge on the W-day window + 27 statics (one model for all basins)
Targets: raw mm/day and per-basin standardised (train-period stats only).
"""
import argparse
import json

import numpy as np

from camels_tripwire import BUDGET_B, Camels, OUTD, nse_per_basin, windows

SCR = "/Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire/"


def fit(X, y, lam):
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--scaling", default="perbasin")
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--out", default=str(OUTD / "arms_ridge.jsonl"))
    a = ap.parse_args()
    C = Camels("daymet")
    F = C.scaled_forcing(a.scaling)
    S = C.statics(a.scaling)
    W = a.window
    b447 = set(json.load(open(SCR + "ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    te = C.te[C.te >= W - 1]

    per = {t: np.full(C.nb, np.nan) for t in ("raw", "std")}
    XtX = None
    rows_pool, y_pool_raw, y_pool_std = [], [], []
    for i in range(C.nb):
        wv = windows(F[i], W)
        tr = C.tr[(C.tr >= W - 1) & (C.Q[i, C.tr] >= 0)]
        Xtr = np.concatenate([wv[tr - (W - 1)],
                              np.ones((len(tr), 1), np.float32)], 1)
        ytr = C.Q[i, tr]
        ztr = (ytr - C.qmu[i]) / C.qsd[i]
        Xte = np.concatenate([wv[te - (W - 1)],
                              np.ones((len(te), 1), np.float32)], 1)
        obs = C.Q[i, te]
        for t, yy in (("raw", ytr), ("std", ztr)):
            w = fit(Xtr.astype(np.float64), yy, a.lam)
            p = Xte.astype(np.float64) @ w
            if t == "std":
                p = p * C.qsd[i] + C.qmu[i]
            per[t][i] = nse_per_basin(obs, p)
        # pooled design (subsample for tractability)
        sub = tr[::7]
        rows_pool.append(np.concatenate(
            [wv[sub - (W - 1)], np.repeat(S[i][None], len(sub), 0),
             np.ones((len(sub), 1))], 1))
        y_pool_raw.append(C.Q[i, sub])
        y_pool_std.append((C.Q[i, sub] - C.qmu[i]) / C.qsd[i])
        if i % 150 == 0:
            print("  ridge", i, flush=True)
    Xp = np.concatenate(rows_pool).astype(np.float64)
    pool = {}
    for t, yv in (("raw", np.concatenate(y_pool_raw)),
                  ("std", np.concatenate(y_pool_std))):
        w = fit(Xp, yv, a.lam)
        v = np.full(C.nb, np.nan)
        for i in range(C.nb):
            wv = windows(F[i], W)
            Xte = np.concatenate([wv[te - (W - 1)],
                                  np.repeat(S[i][None], len(te), 0),
                                  np.ones((len(te), 1))], 1)
            p = Xte @ w
            if t == "std":
                p = p * C.qsd[i] + C.qmu[i]
            v[i] = nse_per_basin(C.Q[i, te], p)
        pool[t] = v

    with open(a.out, "a") as fh:
        for t in ("raw", "std"):
            nb = int(C.nb * (5 * W + 1) * 4)
            fh.write(json.dumps(dict(
                venue="camels", arm="ridge_per_basin", window=W,
                scaling=a.scaling, target=t, lam=a.lam, bytes=nb,
                in_budget=bool(nb <= BUDGET_B), registered=False,
                median_nse_447=float(np.nanmedian(per[t][m447])),
                mean_nse_447=float(np.nanmean(per[t][m447])),
                median_nse_531=float(np.nanmedian(per[t])))) + "\n")
            nb2 = int((5 * W + 27 + 1) * 4)
            fh.write(json.dumps(dict(
                venue="camels", arm="ridge_pooled", window=W,
                scaling=a.scaling, target=t, lam=a.lam, bytes=nb2,
                in_budget=True, registered=False,
                median_nse_447=float(np.nanmedian(pool[t][m447])),
                mean_nse_447=float(np.nanmean(pool[t][m447])),
                median_nse_531=float(np.nanmedian(pool[t])))) + "\n")
            print("per_basin", W, t, round(float(np.nanmedian(per[t][m447])), 4),
                  nb, "| pooled", round(float(np.nanmedian(pool[t][m447])), 4),
                  nb2, flush=True)


if __name__ == "__main__":
    main()
