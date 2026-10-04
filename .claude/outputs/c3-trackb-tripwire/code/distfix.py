"""Re-run the DECISIVE in-budget CAMELS configs with a corrected
distance-weighted kNN.

Defect found after the ladder had run: Arm.predict computed the squared
distance as ||k||^2 - 2 q.k, omitting the query term ||q||^2.  Ranking (hence
every uniform-weight row and every k=1 row) is UNAFFECTED, because ||q||^2 is
constant within a query; only the distance-WEIGHTED rows used a wrong weight.
This script recomputes the affected rows correctly so the report can state
whether the consumed max moves.  Writes arms_distfix.jsonl.
"""
import argparse
import itertools
import json

import numpy as np

import camels_tripwire as CT
from camels_tripwire import KS, OUTD, Camels, nse_per_basin, windows

SCR = "/Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire/"


def predict_fixed(self, Qy):
    L = self.K.shape[0]
    kmax = min(max(KS), L)
    CH = max(128, int(2.0e7 // max(L, 1)))
    out_idx = np.empty((len(Qy), kmax), np.int64)
    out_d = np.empty((len(Qy), kmax), np.float64)
    for s0 in range(0, len(Qy), CH):
        q = Qy[s0:s0 + CH]
        d2c = (-2.0) * (q @ self.K.T).astype(np.float64)
        d2c += self.knorm[None, :]
        d2c += (q.astype(np.float64) ** 2).sum(1)[:, None]   # <-- the fix
        ii = np.argpartition(d2c, kmax - 1, axis=1)[:, :kmax]
        dd = np.take_along_axis(d2c, ii, 1)
        o = np.argsort(dd, axis=1, kind="stable")
        out_idx[s0:s0 + len(q)] = np.take_along_axis(ii, o, 1)
        out_d[s0:s0 + len(q)] = np.take_along_axis(dd, o, 1)
    out_d = np.sqrt(np.maximum(out_d, 0))
    out = {}
    for k in KS:
        if k > L:
            continue
        ii, dd = out_idx[:, :k], out_d[:, :k]
        w = 1.0 / np.maximum(dd, 1e-9)
        w = w / w.sum(1, keepdims=True)
        for tgt, arr in (("raw", self.q_raw), ("std", self.q_std)):
            v = arr[ii]
            out[(k, "dist", tgt)] = (w * v).sum(1)
            out[(k, "unif", tgt)] = v.mean(1)
    return out


CT.Arm.predict = predict_fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--scaling", required=True)
    ap.add_argument("--Ls", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kmeans", type=int, default=1)
    a = ap.parse_args()
    C = Camels("daymet")
    F = C.scaled_forcing(a.scaling)
    S = C.statics(a.scaling)
    Ls = [int(x) for x in a.Ls.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else []
    stores = CT.build_regional_stores(C, F, S, a.window, Ls, seeds,
                                      bool(a.kmeans))
    res = CT.evaluate(C, F, S, a.window, stores, sub=1)
    b447 = set(json.load(open(SCR + "ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    D = a.window * 5 + S.shape[1]
    with open(OUTD / "arms_distfix.jsonl", "a") as fh:
        for s in stores:
            L = s.K.shape[0]
            for key, v in res[s.tag].items():
                fh.write(json.dumps(dict(
                    venue="camels", window=a.window, scaling=a.scaling, dim=D,
                    store="regional", tag=s.tag, L=L,
                    bytes=int(L * (D + 1) * 4),
                    in_budget=bool(L * (D + 1) * 4 <= CT.BUDGET_B),
                    k=key[0], weight=key[1], target=key[2], sub=1,
                    fixed_distance=True,
                    median_nse_447=float(np.nanmedian(v[m447])),
                    mean_nse_447=float(np.nanmean(v[m447])),
                    median_nse_531=float(np.nanmedian(v)))) + "\n")
    best = max((max(np.nanmedian(res[s.tag][k][m447]) for k in res[s.tag])
                for s in stores))
    print("best (fixed) median NSE 447:", round(float(best), 4))


if __name__ == "__main__":
    main()
