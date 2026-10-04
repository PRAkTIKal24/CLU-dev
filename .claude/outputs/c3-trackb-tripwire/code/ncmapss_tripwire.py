"""N-CMAPSS DS02: the missing criterion-2 trivial baselines + the criterion-4
matched-bytes exemplar tripwire.

Source of record: NASA PCoE '17. Turbofan Engine Degradation Simulation-2',
member data_set/N-CMAPSS_DS02-006.h5 (sha256 in the report).
⛔ CAFE EMBARGO: nothing here touches any CAFE artefact or preprocessing.

Reference (primary, open access): Arias Chao, Kulkarni, Goebel, Fink,
"Fusing Physics-based and Deep Learning Models for Prognostics",
arXiv:2003.00732v2 (2020-10-27) = Reliab. Eng. Syst. Saf. 217:107961 (2022),
Table 5, same DS02 test units {11,14,15}:
    FNN data-driven  RMSE 7.89 +- 0.12   s*1e5 1.39 +- 0.04
    FNN hybrid       RMSE 4.22 +- 0.10   s*1e5 0.44 +- 0.01
    CNN data-driven  RMSE 4.95 +- 0.15   s*1e5 0.56 +- 0.03
    CNN hybrid       RMSE 4.14 +- 0.09   s*1e5 0.44 +- 0.02
Their metric: per-sample over the 0.1 Hz down-sampled test set (m* = 0.12M).
"""
import argparse
import json
import time

import h5py
import numpy as np

H5 = "/Users/user/Desktop/CHLU/.claude/data/c3-ncmapss/N-CMAPSS_DS02-006.h5"
OUT = ("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire/"
       "ncmapss_arms.jsonl")
BUDGET_B = 1_966_080
KS = (1, 3, 5, 10, 25)
DS = 10  # 1 Hz file -> 0.1 Hz, the rate the reference reports on


def load():
    with h5py.File(H5, "r") as f:
        d = {}
        for k in ("W", "X_s", "A", "Y"):
            d[k + "_dev"] = np.array(f[k + "_dev"])[::DS]
            d[k + "_test"] = np.array(f[k + "_test"])[::DS]
        d["W_var"] = [x.decode() for x in np.array(f["W_var"]).ravel()]
        d["X_s_var"] = [x.decode() for x in np.array(f["X_s_var"]).ravel()]
        d["A_var"] = [x.decode() for x in np.array(f["A_var"]).ravel()]
    return d


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nasa_s(y, p):
    """s = sum exp(alpha |delta|), delta = yhat - y;
    alpha = 1/13 when RUL is UNDER-estimated (yhat < y), 1/10 otherwise."""
    d = p - y
    a = np.where(d < 0, 1 / 13.0, 1 / 10.0)
    return float(np.sum(np.exp(a * np.abs(d))))


def row(**kw):
    with open(OUT, "a") as fh:
        fh.write(json.dumps(kw) + "\n")
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in kw.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    t0 = time.time()
    d = load()
    Wd, Xd, Ad, Yd = d["W_dev"], d["X_s_dev"], d["A_dev"], d["Y_dev"].ravel()
    Wt, Xt, At, Yt = (d["W_test"], d["X_s_test"], d["A_test"],
                      d["Y_test"].ravel())
    ud, ut = Ad[:, 0], At[:, 0]
    cd, ct = Ad[:, 1], At[:, 1]
    print("dev", Xd.shape, "test", Xt.shape, "units dev",
          np.unique(ud), "test", np.unique(ut), flush=True)
    print("EOL dev", {int(u): int(cd[ud == u].max()) for u in np.unique(ud)},
          "test", {int(u): int(ct[ut == u].max()) for u in np.unique(ut)},
          flush=True)

    # ---------- criterion-2 trivial baselines (the missing rows) ----------
    p = np.full(len(Yt), Yd.mean())
    row(venue="ncmapss", arm="mean_RUL", inputs="none", bytes=4,
        in_budget=True, rmse=rmse(Yt, p), s=nasa_s(Yt, p),
        s_1e5=nasa_s(Yt, p) / 1e5, n_test=int(len(Yt)))

    Ades = np.stack([np.ones_like(cd), cd], 1)
    coef = np.linalg.lstsq(Ades, Yd, rcond=None)[0]
    p = coef[0] + coef[1] * ct
    row(venue="ncmapss", arm="affine_cycle_index", inputs="cycle", bytes=8,
        in_budget=True, coef=[float(coef[0]), float(coef[1])],
        rmse=rmse(Yt, p), s=nasa_s(Yt, p), s_1e5=nasa_s(Yt, p) / 1e5)
    pc = np.clip(p, 0, None)
    row(venue="ncmapss", arm="affine_cycle_index_clip0", inputs="cycle",
        bytes=8, in_budget=True, rmse=rmse(Yt, pc), s=nasa_s(Yt, pc),
        s_1e5=nasa_s(Yt, pc) / 1e5)

    # per-unit "known EOL mean" oracle-ish reference (declared, not a baseline)
    p = np.array([np.mean([75, 89, 82, 63, 71, 66])] * len(Yt)) - ct
    row(venue="ncmapss", arm="mean_EOL_minus_cycle", inputs="cycle",
        bytes=8, in_budget=True, rmse=rmse(Yt, p), s=nasa_s(Yt, p),
        s_1e5=nasa_s(Yt, p) / 1e5,
        note="train-EOL mean 74.33 minus cycle index; the 'RUL is defined "
             "piecewise-linear in cycle' construction, stated explicitly")

    if a.stage == "crit2":
        print("crit2 done in %.0fs" % (time.time() - t0))
        return

    # ---------- criterion-4: exemplar stores ----------
    # condition-residual features (the classical health index):
    #   fit X_s ~ [1, W, W^2] on the HEALTHY dev cycles (cycle <= 10), pooled
    hm = cd <= 10
    def design(W):
        return np.concatenate([np.ones((len(W), 1)), W, W ** 2], 1)
    B = np.linalg.lstsq(design(Wd[hm]), Xd[hm], rcond=None)[0]
    Rd = Xd - design(Wd) @ B
    Rt = Xt - design(Wt) @ B

    feats = {
        "w_xs": (np.concatenate([Wd, Xd], 1), np.concatenate([Wt, Xt], 1)),
        "resid": (Rd, Rt),
        "w_xs_resid": (np.concatenate([Wd, Xd, Rd], 1),
                       np.concatenate([Wt, Xt, Rt], 1)),
        "resid_cycle": (np.concatenate([Rd, cd[:, None]], 1),
                        np.concatenate([Rt, ct[:, None]], 1)),
    }
    rng = np.random.default_rng(0)
    for fname, (Fd, Ft) in feats.items():
        for scal in ("raw", "std"):
            if scal == "std":
                mu, sd = Fd.mean(0), Fd.std(0) + 1e-12
                Kd, Kt = (Fd - mu) / sd, (Ft - mu) / sd
            else:
                Kd, Kt = Fd, Ft
            Kd = Kd.astype(np.float32)
            Kt = Kt.astype(np.float32)
            D = Kd.shape[1]
            bpe = (D + 1) * 4
            atb = BUDGET_B // bpe
            for L in sorted({250, 500, 1000, 2000, 5000, atb}):
                if L > len(Kd):
                    continue
                sel = rng.choice(len(Kd), L, replace=False)
                K, q = Kd[sel], Yd[sel].astype(np.float64)
                kn = (K.astype(np.float64) ** 2).sum(1)
                preds = {k: np.empty(len(Kt)) for k in KS}
                CH = max(1000, min(20000, int(4e8 / (8 * L))))
                for s in range(0, len(Kt), CH):
                    Qy = Kt[s:s + CH]
                    d2 = kn[None, :] - 2.0 * (Qy @ K.T).astype(np.float64)
                    km = min(max(KS), L)
                    idx = np.argpartition(d2, km - 1, 1)[:, :km]
                    dd = np.take_along_axis(d2, idx, 1)
                    o = np.argsort(dd, 1, kind="stable")
                    idx = np.take_along_axis(idx, o, 1)
                    for k in KS:
                        if k > L:
                            continue
                        preds[k][s:s + len(Qy)] = q[idx[:, :k]].mean(1)
                for k in KS:
                    if k > L:
                        continue
                    p = preds[k]
                    row(venue="ncmapss", arm="knn", feats=fname, scaling=scal,
                        dim=D, L=L, k=k, bytes=int(L * bpe),
                        in_budget=bool(L * bpe <= BUDGET_B),
                        at_budget_L=int(atb),
                        matched_inputs=bool(fname == "w_xs"),
                        rmse=rmse(Yt, p), s=nasa_s(Yt, p),
                        s_1e5=nasa_s(Yt, p) / 1e5)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
