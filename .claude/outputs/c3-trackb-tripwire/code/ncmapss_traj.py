"""N-CMAPSS DS02 — the classical SIMILARITY-BASED RUL arm (the published
competitive classical method on the C-MAPSS family, i.e. the venue's highest
criterion-4 hazard), at the same byte budget.

Library = per-cycle health-index trajectories of the 6 training units.
Query   = the trailing S-cycle segment of a test unit's own health index.
Match   = Euclidean over the segment (and a DTW variant), read off the RUL of
          the matched library point.  k nearest library points are averaged.
"""
import argparse
import json

import h5py
import numpy as np

H5 = "/Users/user/Desktop/CHLU/.claude/data/c3-ncmapss/N-CMAPSS_DS02-006.h5"
OUT = ("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire/"
       "ncmapss_arms.jsonl")
BUDGET_B = 1_966_080
DS = 10


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_s(y, p):
    d = p - y
    a = np.where(d < 0, 1 / 13.0, 1 / 10.0)
    return float(np.sum(np.exp(a * np.abs(d))))


def dtw_dist(a, b, band=5):
    """Sakoe-Chiba banded DTW between (S,d) sequences."""
    S = len(a)
    inf = np.inf
    D = np.full((S + 1, S + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, S + 1):
        lo, hi = max(1, i - band), min(S, i + band)
        for j in range(lo, hi + 1):
            c = np.sum((a[i - 1] - b[j - 1]) ** 2)
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[S, S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--dtw", type=int, default=0)
    a = ap.parse_args()
    with h5py.File(H5, "r") as f:
        Wd = np.array(f["W_dev"])[::DS]
        Xd = np.array(f["X_s_dev"])[::DS]
        Ad = np.array(f["A_dev"])[::DS]
        Yd = np.array(f["Y_dev"])[::DS].ravel()
        Wt = np.array(f["W_test"])[::DS]
        Xt = np.array(f["X_s_test"])[::DS]
        At = np.array(f["A_test"])[::DS]
        Yt = np.array(f["Y_test"])[::DS].ravel()

    cd, ud = Ad[:, 1], Ad[:, 0]
    ct, ut = At[:, 1], At[:, 0]

    def design(W):
        return np.concatenate([np.ones((len(W), 1)), W, W ** 2], 1)
    hm = cd <= 10
    B = np.linalg.lstsq(design(Wd[hm]), Xd[hm], rcond=None)[0]
    Rd = Xd - design(Wd) @ B
    Rt = Xt - design(Wt) @ B
    sd = Rd.std(0) + 1e-12
    Rd, Rt = Rd / sd, Rt / sd

    def cycle_means(R, u, c, Y):
        out = {}
        for uu in np.unique(u):
            m = u == uu
            cs = np.unique(c[m]).astype(int)
            H = np.stack([R[m & (c == cc)].mean(0) for cc in cs])
            yy = np.array([Y[m & (c == cc)][0] for cc in cs])
            out[int(uu)] = (cs, H, yy)
        return out

    dev = cycle_means(Rd, ud, cd, Yd)
    tst = cycle_means(Rt, ut, ct, Yt)

    # 1-D health index = projection on the mean degradation direction
    allH = np.concatenate([v[1] for v in dev.values()])
    allY = np.concatenate([v[2] for v in dev.values()])
    w = np.linalg.lstsq(np.concatenate(
        [allH, np.ones((len(allH), 1))], 1), -allY.astype(float),
        rcond=None)[0][:-1]
    w = w / (np.linalg.norm(w) + 1e-12)

    for rep in ("hi1d", "hi14"):
        S = a.S
        lib_seg, lib_y = [], []
        for uu, (cs, H, yy) in dev.items():
            Z = H @ w if rep == "hi1d" else H
            Z = Z.reshape(len(cs), -1)
            for i in range(S - 1, len(cs)):
                lib_seg.append(Z[i - S + 1:i + 1].ravel())
                lib_y.append(yy[i])
        Lseg = np.array(lib_seg, np.float32)
        Ly = np.array(lib_y, float)
        d = Lseg.shape[1]
        nbytes = int(len(Lseg) * (d + 1) * 4)

        pred_cycle = {}
        for uu, (cs, H, yy) in tst.items():
            Z = H @ w if rep == "hi1d" else H
            Z = Z.reshape(len(cs), -1)
            for i, cc in enumerate(cs):
                if i < S - 1:
                    seg = np.repeat(Z[:1], S - 1 - i, 0)
                    seg = np.concatenate([seg, Z[:i + 1]])
                else:
                    seg = Z[i - S + 1:i + 1]
                q = seg.ravel()
                if a.dtw and rep == "hi1d":
                    dist = np.array([dtw_dist(seg, Lseg[j].reshape(S, -1))
                                     for j in range(len(Lseg))])
                else:
                    dist = ((Lseg - q) ** 2).sum(1)
                order = np.argsort(dist)
                for k in (1, 3, 5, 10, 25):
                    pred_cycle.setdefault(k, {})[(int(uu), int(cc))] = \
                        Ly[order[:k]].mean()
        for k in (1, 3, 5, 10, 25):
            p = np.array([pred_cycle[k][(int(u_), int(c_))]
                          for u_, c_ in zip(ut, ct)])
            with open(OUT, "a") as fh:
                fh.write(json.dumps(dict(
                    venue="ncmapss", arm="traj_similarity", rep=rep,
                    metric="dtw" if a.dtw and rep == "hi1d" else "euclid",
                    S=S, k=k, n_library=int(len(Lseg)), dim=int(d),
                    bytes=nbytes, in_budget=bool(nbytes <= BUDGET_B),
                    matched_inputs=True,
                    rmse=rmse(Yt, p), s=nasa_s(Yt, p),
                    s_1e5=nasa_s(Yt, p) / 1e5)) + "\n")
            print(rep, k, len(Lseg), nbytes, round(rmse(Yt, p), 3),
                  round(nasa_s(Yt, p) / 1e5, 3), flush=True)


if __name__ == "__main__":
    main()
