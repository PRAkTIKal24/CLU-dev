"""Deep dive on ONE CAMELS store config: the venue's own NSE decomposition
(alpha, beta, FHV, FMS, FLV -- definitions copied verbatim from the reference
implementation's papercode/metrics.py), regime-resolved NSE slices, and the
Metro shuffled-order null.  Compared per-basin against the published
LSTM/EA-LSTM/benchmark values in all_metrics.p.
"""
import argparse
import json
import pickle

import numpy as np

from camels_tripwire import (Arm, Camels, OUTD, build_regional_stores,
                             nse_per_basin, windows)

SCR = "/Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire/"


def alpha_nse(o, s):
    return float(np.std(s) / np.std(o))


def beta_nse(o, s):
    return float((np.mean(s) - np.mean(o)) / np.std(o))


def fhv(o, s, h=0.02):
    o = -np.sort(-o.copy()); s = -np.sort(-s.copy())
    n = int(np.round(h * len(o)))
    return float(np.sum(s[:n] - o[:n]) / (np.sum(o[:n]) + 1e-6) * 100)


def flv(o, s, l=0.7):
    o = o.copy(); s = s.copy()
    s[s == 0] = 1e-6; o[o == 0] = 1e-6
    o = -np.sort(-o); s = -np.sort(-s)
    i = int(np.round(l * len(o)))
    o, s = np.log(o[i:] + 1e-6), np.log(s[i:] + 1e-6)
    qsl, qol = np.sum(s - s.min()), np.sum(o - o.min())
    return float(-1 * (qsl - qol) / (qol + 1e-6) * 100)


def fms(o, s, m1=0.2, m2=0.7):
    o = o.copy(); s = s.copy()
    s[s == 0] = 1e-6; o[o == 0] = 1e-6
    o = -np.sort(-o); s = -np.sort(-s)
    a = np.log(s[int(np.round(m1 * len(s)))] + 1e-6)
    b = np.log(s[int(np.round(m2 * len(s)))] + 1e-6)
    c = np.log(o[int(np.round(m1 * len(o)))] + 1e-6)
    d = np.log(o[int(np.round(m2 * len(o)))] + 1e-6)
    return float(((a - b) - (c - d)) / (c - d + 1e-6) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--scaling", default="perbasin")
    ap.add_argument("--L", type=int, default=2000)
    ap.add_argument("--sel", default="kmeans")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--target", default="std")
    ap.add_argument("--weight", default="dist")
    ap.add_argument("--shuffle-seed", type=int, default=-1)
    ap.add_argument("--shuffle-keys", type=int, default=0)
    ap.add_argument("--tag", default="best")
    a = ap.parse_args()

    C = Camels("daymet")
    F = C.scaled_forcing(a.scaling)
    S = C.statics(a.scaling)
    seeds = [] if a.sel == "kmeans" else [int(a.sel)]
    stores = build_regional_stores(C, F, S, a.window, [a.L], seeds,
                                   a.sel == "kmeans")
    st = stores[0]
    print("store", st.tag, st.K.shape, flush=True)
    if a.shuffle_seed >= 0 and a.shuffle_keys:
        # registered null: independent per-row permutation of the W time slots
        # applied to the STORED KEYS as well as the queries
        rk = np.random.default_rng(a.shuffle_seed + 777)
        blk = st.K[:, :a.window * 5].reshape(-1, a.window, 5)
        for r in range(blk.shape[0]):
            blk[r] = blk[r][rk.permutation(a.window)]
        st.K[:, :a.window * 5] = blk.reshape(blk.shape[0], -1)
        st.knorm = (st.K.astype(np.float64) ** 2).sum(1)
        print("store keys shuffled", flush=True)

    te = C.te[C.te >= a.window - 1]
    rng = (np.random.default_rng(a.shuffle_seed)
           if a.shuffle_seed >= 0 else None)
    doy_month = np.array([int(str(d)[5:7]) for d in C.dates])
    frac_snow = C.A_raw[:, C.attr_names.index("frac_snow")]

    keys = ["nse", "alpha", "beta", "fhv", "fms", "flv",
            "nse_high", "nse_mid", "nse_low", "nse_snowseason"]
    res = {k: np.full(C.nb, np.nan) for k in keys}
    for i in range(C.nb):
        wv = windows(F[i], a.window)
        Qy = np.empty((len(te), a.window * 5 + S.shape[1]), np.float32)
        Qy[:, :a.window * 5] = wv[te - (a.window - 1)]
        if rng is not None:
            blk = Qy[:, :a.window * 5].reshape(len(te), a.window, 5)
            for r in range(len(te)):
                blk[r] = blk[r][rng.permutation(a.window)]
            Qy[:, :a.window * 5] = blk.reshape(len(te), -1)
        if S.shape[1]:
            Qy[:, a.window * 5:] = S[i]
        p = st.predict(Qy)[(a.k, a.weight, a.target)]
        if a.target == "std":
            p = p * C.qsd[i] + C.qmu[i]
        obs = C.Q[i, te]
        m = obs >= 0
        o, s = obs[m], p[m]
        if len(o) < 10:
            continue
        res["nse"][i] = nse_per_basin(obs, p)
        res["alpha"][i] = alpha_nse(o, s)
        res["beta"][i] = beta_nse(o, s)
        res["fhv"][i] = fhv(o, s)
        res["fms"][i] = fms(o, s)
        res["flv"][i] = flv(o, s)
        thr_hi = np.quantile(o, 0.98)
        thr_lo = np.quantile(o, 0.30)
        for nm, sel in (("nse_high", o >= thr_hi), ("nse_low", o <= thr_lo),
                        ("nse_mid", (o > thr_lo) & (o < thr_hi))):
            if sel.sum() > 10 and np.std(o[sel]) > 0:
                res[nm][i] = 1 - np.sum((s[sel] - o[sel]) ** 2) / \
                    np.sum((o[sel] - o[sel].mean()) ** 2)
        mm = np.isin(doy_month[te][m], (3, 4, 5, 6))
        if frac_snow[i] > 0.3 and mm.sum() > 10:
            res["nse_snowseason"][i] = 1 - np.sum((s[mm] - o[mm]) ** 2) / \
                np.sum((o[mm] - o[mm].mean()) ** 2)
        if i % 150 == 0:
            print("  ", i, flush=True)

    b447 = set(json.load(open(SCR + "ref/basins_447_derived.json")))
    m447 = np.array([b in b447 for b in C.basins])
    am = pickle.load(open(SCR + "ref/all_metrics.p", "rb"))
    out = dict(config=vars(a), n447=int(m447.sum()))
    for k in keys:
        v = res[k][m447]
        out[k] = dict(median=float(np.nanmedian(v)),
                      mean=float(np.nanmean(v)),
                      n=int(np.sum(~np.isnan(v))))
    # published comparators on the same 447 basins
    pub = {}
    for met, key in (("NSE", "nse"), ("alpha_nse", "alpha"),
                     ("beta_nse", "beta"), ("FHV", "fhv"), ("FLV", "flv"),
                     ("FMS", "fms")):
        pub[key] = {}
        for mod in ("lstm_NSE", "ealstm_NSE"):
            d = am[met][mod]["ensemble"]
            pub[key][mod] = float(np.nanmedian(
                [d[b] for b in C.basins if b in b447 and b in d]))
        for mod, d in am[met]["benchmarks"].items():
            vals = [d[b] for b in C.basins if b in b447 and b in d]
            if vals:
                pub[key][mod] = float(np.nanmedian(vals))
    out["published_median_447"] = pub
    fn = OUTD / f"deepdive_{a.tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    np.savez_compressed(OUTD / f"deepdive_{a.tag}.npz", **res)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
