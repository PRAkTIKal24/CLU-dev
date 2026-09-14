"""CAMELS-US criterion-4 tripwire: matched-bytes exemplar store vs the published
LSTM/EA-LSTM ensembles.  Implements PREREG.md exactly.

⛔ No CLU code, no chlu/ import, no training.  Pure numpy/sklearn.
Outputs JSONL rows to .claude/outputs/c3-trackb-tripwire/arms.jsonl
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATA = Path("/Users/user/Desktop/CHLU/.claude/data/c3-camels/staged")
OUTD = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire")
OUTD.mkdir(parents=True, exist_ok=True)

TRAIN = ("1999-10-01", "2008-09-30")
TEST = ("1989-10-01", "1999-09-30")
KS = (1, 3, 5, 10, 25)
BUDGET_B = 1_966_080


def nse_per_basin(obs, sim):
    """Kratzert calc_nse convention: drop obs<0 (the -999 flag) first."""
    m = obs >= 0
    obs, sim = obs[m], sim[m]
    if obs.size < 2:
        return np.nan
    den = np.sum((obs - obs.mean()) ** 2)
    if den == 0:
        return np.nan
    return float(1.0 - np.sum((sim - obs) ** 2) / den)


class Camels:
    def __init__(self, product="maurer_extended"):
        f = np.load(DATA / f"forcing_{product}.npz", allow_pickle=True)
        d = np.load(DATA / "discharge.npz", allow_pickle=True)
        a = np.load(DATA / "attributes.npz", allow_pickle=True)
        self.X = f["X"].astype(np.float32)           # (531, T, 5)
        self.Q = d["Q"].astype(np.float64)           # (531, T)
        self.basins = list(f["basins"])
        self.dates = np.array(f["dates"], dtype="datetime64[D]")
        self.attr_names = list(a["names"])
        A = a["values"].astype(np.float64)
        assert list(a["basins"]) == self.basins
        self.A_raw = A
        # statics z-scored across basins (used by every non-raw scaling)
        self.A_z = (A - A.mean(0)) / (A.std(0) + 1e-12)
        self.tr = np.where((self.dates >= np.datetime64(TRAIN[0])) &
                           (self.dates <= np.datetime64(TRAIN[1])))[0]
        self.te = np.where((self.dates >= np.datetime64(TEST[0])) &
                           (self.dates <= np.datetime64(TEST[1])))[0]
        self.nb = len(self.basins)
        # per-basin train discharge stats (valid days only)
        self.qmu = np.zeros(self.nb)
        self.qsd = np.zeros(self.nb)
        for i in range(self.nb):
            q = self.Q[i, self.tr]
            q = q[q >= 0]
            self.qmu[i], self.qsd[i] = q.mean(), q.std() + 1e-12

    def scaled_forcing(self, scaling):
        X = self.X.astype(np.float32)
        if scaling == "raw":
            return X
        if scaling == "perbasin":
            mu = X[:, self.tr].mean(1, keepdims=True)
            sd = X[:, self.tr].std(1, keepdims=True) + 1e-6
            return ((X - mu) / sd).astype(np.float32)
        if scaling == "global":
            mu = X[:, self.tr].mean((0, 1))[None, None]
            sd = X[:, self.tr].std((0, 1))[None, None] + 1e-6
            return ((X - mu) / sd).astype(np.float32)
        raise ValueError(scaling)

    def statics(self, scaling, use_statics=True):
        if not use_statics:
            return np.zeros((self.nb, 0))
        return self.A_raw if scaling == "raw" else self.A_z


def windows(F_b, W):
    """(T,5) -> (T-W+1, W*5); row j corresponds to end-day index j+W-1."""
    sw = sliding_window_view(F_b, (W, F_b.shape[1]))[:, 0]
    return np.ascontiguousarray(sw.reshape(sw.shape[0], -1))


class Arm:
    """One store: keys (L,D) float32, targets raw + standardised."""

    def __init__(self, keys, q_raw, q_std, tag):
        self.K = np.ascontiguousarray(keys.astype(np.float32))
        self.q_raw = q_raw.astype(np.float64)
        self.q_std = q_std.astype(np.float64)
        self.knorm = (self.K.astype(np.float64) ** 2).sum(1)
        self.tag = tag

    def predict(self, Qy):
        """Qy (n,D) -> dict[(k,weight,target)] = predictions (n,).

        Query-chunked: arithmetic is bit-identical to the unchunked form, the
        chunking only bounds the (n x L) float64 distance buffer.
        """
        L = self.K.shape[0]
        kmax = min(max(KS), L)
        CH = max(128, int(2.0e7 // max(L, 1)))
        idx = np.empty((len(Qy), kmax), np.int64)
        dsel = np.empty((len(Qy), kmax), np.float64)
        for s0 in range(0, len(Qy), CH):
            q = Qy[s0:s0 + CH]
            d2c = (-2.0) * (q @ self.K.T).astype(np.float64)
            d2c += self.knorm[None, :]
            ii = np.argpartition(d2c, kmax - 1, axis=1)[:, :kmax]
            dd = np.take_along_axis(d2c, ii, 1)
            o = np.argsort(dd, axis=1, kind="stable")
            idx[s0:s0 + len(q)] = np.take_along_axis(ii, o, 1)
            dsel[s0:s0 + len(q)] = np.take_along_axis(dd, o, 1)
        out = {}
        dsel = np.sqrt(np.maximum(dsel - dsel.min() * 0 , 0))
        for k in KS:
            if k > self.K.shape[0]:
                continue
            ii, dd = idx[:, :k], dsel[:, :k]
            w = 1.0 / np.maximum(dd, 1e-9)
            w = w / w.sum(1, keepdims=True)
            for tgt, arr in (("raw", self.q_raw), ("std", self.q_std)):
                v = arr[ii]
                out[(k, "dist", tgt)] = (w * v).sum(1)
                out[(k, "unif", tgt)] = v.mean(1)
        return out


def build_regional_stores(C, F, S, W, Ls, seeds, kmeans, sub_pool=50000,
                          rng_pool=0):
    """Sample training (basin, day) pairs and materialise stores."""
    tr = C.tr
    valid = []
    for i in range(C.nb):
        ok = tr[(tr >= W - 1) & (C.Q[i, tr] >= 0)]
        valid.append(ok)
    counts = np.array([len(v) for v in valid])
    total = counts.sum()
    stores = []

    def gather(pairs):
        """pairs: (n,2) basin_idx, day_idx -> keys, q"""
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        keys = np.empty((len(pairs), W * 5 + S.shape[1]), np.float32)
        qq = np.empty(len(pairs))
        qz = np.empty(len(pairs))
        p = 0
        for i in np.unique(pairs[:, 0]):
            sel = pairs[pairs[:, 0] == i]
            wv = windows(F[i], W)
            rows = wv[sel[:, 1] - (W - 1)]
            n = len(sel)
            keys[p:p + n, :W * 5] = rows
            if S.shape[1]:
                keys[p:p + n, W * 5:] = S[i]
            qq[p:p + n] = C.Q[i, sel[:, 1]]
            qz[p:p + n] = (qq[p:p + n] - C.qmu[i]) / C.qsd[i]
            p += n
        return keys, qq, qz

    for L in Ls:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            bi = rng.choice(C.nb, size=L, p=counts / total)
            dj = np.array([valid[b][rng.integers(len(valid[b]))] for b in bi])
            keys, qq, qz = gather(np.stack([bi, dj], 1))
            stores.append(Arm(keys, qq, qz, f"L{L}_rand{sd}"))
        if kmeans and L < sub_pool:
            from sklearn.cluster import MiniBatchKMeans
            rng = np.random.default_rng(rng_pool)
            bi = rng.choice(C.nb, size=sub_pool, p=counts / total)
            dj = np.array([valid[b][rng.integers(len(valid[b]))] for b in bi])
            keys, qq, qz = gather(np.stack([bi, dj], 1))
            km = MiniBatchKMeans(n_clusters=L, random_state=0, n_init=1,
                                 batch_size=4096, max_iter=100)
            lab = km.fit_predict(keys)
            cen = km.cluster_centers_.astype(np.float32)
            tq = np.zeros(L)
            tz = np.zeros(L)
            for c in range(L):
                m = lab == c
                if m.any():
                    tq[c], tz[c] = qq[m].mean(), qz[m].mean()
            stores.append(Arm(cen, tq, tz, f"L{L}_kmeans"))
    return stores


def evaluate(C, F, S, W, stores, sub=1, shuffle_seed=None):
    """Return per-store dict of per-basin NSE arrays (531,)."""
    te = C.te[C.te >= W - 1]
    if sub > 1:
        te = te[::sub]
    res = {s.tag: {kk: np.full(C.nb, np.nan) for kk in
                   itertools.product(KS, ("dist", "unif"), ("raw", "std"))}
           for s in stores}
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for i in range(C.nb):
        wv = windows(F[i], W)
        Qy = np.empty((len(te), W * 5 + S.shape[1]), np.float32)
        Qy[:, :W * 5] = wv[te - (W - 1)]
        if rng is not None:
            block = Qy[:, :W * 5].reshape(len(te), W, 5)
            for r in range(len(te)):
                block[r] = block[r][rng.permutation(W)]
            Qy[:, :W * 5] = block.reshape(len(te), -1)
        if S.shape[1]:
            Qy[:, W * 5:] = S[i]
        obs = C.Q[i, te]
        for s in stores:
            preds = s.predict(Qy)
            for key, p in preds.items():
                if key[2] == "std":
                    p = p * C.qsd[i] + C.qmu[i]
                res[s.tag][key][i] = nse_per_basin(obs, p)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--scaling", required=True)
    ap.add_argument("--product", default="daymet")
    ap.add_argument("--Ls", default="265")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kmeans", type=int, default=1)
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--shuffle-seed", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(OUTD / "arms.jsonl"))
    a = ap.parse_args()

    t0 = time.time()
    C = Camels(a.product)
    F = C.scaled_forcing(a.scaling)
    S = C.statics(a.scaling)
    Ls = [int(x) for x in a.Ls.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else []
    stores = build_regional_stores(C, F, S, a.window, Ls, seeds, bool(a.kmeans))
    print(f"built {len(stores)} stores in {time.time()-t0:.0f}s", flush=True)
    res = evaluate(C, F, S, a.window, stores, sub=a.sub,
                   shuffle_seed=None if a.shuffle_seed < 0 else a.shuffle_seed)
    b447 = set(json.load(open("/Users/user/Desktop/CHLU/.claude/scratch/"
                              "c3-trackb-tripwire/ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    D = a.window * 5 + S.shape[1]
    with open(a.out, "a") as fh:
        for s in stores:
            L = s.K.shape[0]
            for key, v in res[s.tag].items():
                row = dict(venue="camels", product=a.product,
                           window=a.window, scaling=a.scaling, dim=D,
                           store="regional", tag=s.tag, L=L,
                           bytes=int(L * (D + 1) * 4),
                           in_budget=bool(L * (D + 1) * 4 <= BUDGET_B),
                           k=key[0], weight=key[1], target=key[2],
                           sub=a.sub, shuffle=a.shuffle_seed,
                           note=a.tag,
                           median_nse_447=float(np.nanmedian(v[m447])),
                           mean_nse_447=float(np.nanmean(v[m447])),
                           n_le0_447=int(np.sum(v[m447] <= 0)),
                           median_nse_531=float(np.nanmedian(v)),
                           mean_nse_531=float(np.nanmean(v)))
                fh.write(json.dumps(row) + "\n")
        np.savez_compressed(
            OUTD / f"perbasin_{a.product}_{a.window}_{a.scaling}"
                   f"_{a.tag or 'main'}.npz",
            **{f"{s.tag}|{k[0]}|{k[1]}|{k[2]}": res[s.tag][k]
               for s in stores for k in res[s.tag]})
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
