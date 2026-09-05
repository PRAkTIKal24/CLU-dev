"""STEP 4 -- M2: criterion-4 tripwire on Metro.

Arms: k-NN over past windows (the input-metric attack) at the INSECTS byte budgets,
vs tuned strong baselines (GBDT, GRU, online ridge).  Prequential test-then-train.
Usage: python m2.py <arm> [args...]
"""
import io, json, os, pickle, sys, time
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
d = np.load(os.path.join(OUT, "pairs.npz"))
X0 = d["X"].astype(np.float64)
Y = d["y"].astype(np.float64)
N, D = X0.shape
SHUF = os.environ.get("METRO_SHUFFLE", "")
# ---- 24-h LABEL EMBARGO (protocol correction, 2026-08-11) --------------------------------
# At the moment pair t is forecast, the current time is its ORIGIN (target - 24 h).  Only pairs
# whose TARGET has already occurred by then are legally available.  A(t) = index of the last such
# pair.  Without this, plain test-then-train hands every learner up to 23 h of future traffic.
TGT = d["tgt"]
A = np.searchsorted(TGT, TGT - 24, side="right") - 1
EMB = os.environ.get("METRO_EMBARGO", "1") == "1"
if not EMB:
    A = np.arange(N) - 1                      # legacy (leaky) protocol, kept for the delta table
if SHUF:
    A = np.arange(N) - 24                     # order is meaningless after shuffling, but the store
                                              # keeps the same 24-pair staleness for comparability
if SHUF:
    rng = np.random.default_rng(int(SHUF))
    perm = rng.permutation(N)
    X0, Y = X0[perm], Y[perm]


def causal_std(X):
    """Per-feature z-score using statistics from pairs strictly before t (prequential, causal)."""
    n, d = X.shape
    cs = np.cumsum(X, 0); cs2 = np.cumsum(X * X, 0)
    k = np.arange(n).reshape(-1, 1).astype(np.float64)     # count of strictly-earlier rows
    s1 = np.vstack([np.zeros((1, d)), cs[:-1]])
    s2 = np.vstack([np.zeros((1, d)), cs2[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(k > 0, s1 / np.maximum(k, 1), 0.0)
        var = np.where(k > 1, s2 / np.maximum(k, 1) - mu ** 2, 1.0)
    sd = np.sqrt(np.maximum(var, 1e-12))
    sd[sd < 1e-6] = 1.0
    # DECLARED (harness fix, 2026-08-11): clip to +-10 SD.  Without it the running variance of a
    # near-constant early feature (snow_1h, holiday) makes z-scores reach 1e6 at t<500 and the GRU
    # emitted 7.7e6 twice (RMSE 57288).  Causal, arm-agnostic, applied to EVERY standardised arm.
    return np.clip((X - mu) / sd, -10.0, 10.0)


def mets(p, y=Y):
    e = np.asarray(p) - y
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                bias=float(e.mean()))


# ---------------------------------------------------------------- k-NN over past windows
def knn_window(L, k=5, std=False, block=256):
    """Distance-weighted k-NN over a sliding window of the last L (feature, target) pairs.
    Exact prequential: the store at step t is rows [t-L, t)."""
    X = causal_std(X0) if std else X0
    pred = np.empty(N)
    sq = (X * X).sum(1)
    for b0 in range(0, N, block):
        b1 = min(b0 + block, N)
        lo = max(0, int(A[b0]) - L + 1)
        S = X[lo:b1]                                   # candidate superset
        Q = X[b0:b1]
        d2 = sq[b0:b1][:, None] + sq[lo:b1][None, :] - 2.0 * (Q @ S.T)
        np.maximum(d2, 0.0, out=d2)
        gi = np.arange(lo, b1)[None, :]
        ai = A[b0:b1][:, None]
        mask = (gi <= ai) & (gi >= ai - L + 1)          # embargoed window of the last L legal pairs
        d2 = np.where(mask, d2, np.inf)
        kk = min(k, d2.shape[1])
        idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        dd = np.take_along_axis(d2, idx, 1)
        yy = Y[lo:b1][idx]
        good = np.isfinite(dd)
        w = np.where(good, 1.0 / np.maximum(np.sqrt(np.maximum(dd, 0.0)), 1e-9), 0.0)
        ws = w.sum(1)
        out = np.where(ws > 0, (w * yy).sum(1) / np.maximum(ws, 1e-30), np.nan)
        pred[b0:b1] = out
    warm = np.isnan(pred)
    if warm.any():                                      # step 0 only: no past -> global-so-far mean
        pred[warm] = Y[0]
    return pred, L * 132


# ------------------------------------------------- SAM-kNN-style dual memory (OUR adaptation)
def knn_sam(Lmax, k=5, std=True, reeval=500, probe=200, seed=0):
    """STM = most recent m pairs, m selected every `reeval` steps from candidate sizes by
    lowest MAE over the last `probe` queries.  LTM = reservoir subsample of evicted pairs
    filling the residual byte budget.  Predicting memory (STM / LTM / union) also selected
    by lowest recent MAE.  Declared as OUR regression adaptation of Losing et al. 2016."""
    X = causal_std(X0) if std else X0
    rng = np.random.default_rng(seed)
    cands = [c for c in (50, 100, 250, 500, 1000, 2500, 5000, 10000, Lmax) if c <= Lmax]
    cands = sorted(set(cands))
    m = min(500, Lmax)
    pred = np.empty(N)
    ltm_idx = np.zeros(0, dtype=np.int64)
    n_seen_evicted = 0
    mode = "STM"
    mode_counts = {"STM": 0, "LTM": 0, "UNION": 0}
    m_trace = []

    def knn_pred_from(store_idx, q):
        if len(store_idx) == 0:
            return np.nan
        S = X[store_idx]
        d2 = ((S - q) ** 2).sum(1)
        kk = min(k, len(store_idx))
        idx = np.argpartition(d2, kk - 1)[:kk]
        w = 1.0 / np.maximum(np.sqrt(np.maximum(d2[idx], 0.0)), 1e-9)
        return float((w * Y[store_idx[idx]]).sum() / w.sum())

    for t in range(N):
        at = int(A[t])
        if at < 0:
            pred[t] = Y[0]
        else:
            stm = np.arange(max(0, at - m + 1), at + 1)
            ltm = ltm_idx
            if mode == "STM":
                p = knn_pred_from(stm, X[t])
            elif mode == "LTM":
                p = knn_pred_from(ltm, X[t]) if len(ltm) else knn_pred_from(stm, X[t])
            else:
                p = knn_pred_from(np.concatenate([stm, ltm]), X[t])
            pred[t] = p if np.isfinite(p) else Y[max(at, 0)]
        mode_counts[mode] += 1
        # LTM: reservoir over pairs that have fallen out of the largest STM candidate
        ev = at - m
        if ev >= 0:
            budget_ltm = max(0, Lmax - m)
            n_seen_evicted += 1
            if len(ltm_idx) < budget_ltm:
                ltm_idx = np.append(ltm_idx, ev)
            elif budget_ltm > 0:
                r = rng.integers(0, n_seen_evicted)
                if r < budget_ltm:
                    ltm_idx[r] = ev
        if t > 0 and t % reeval == 0 and t >= probe + 30:
            qs = np.arange(t - probe, t)
            best, bestmae = m, np.inf
            for c in cands:
                errs = []
                for q in qs:
                    aq = int(A[q])
                    if aq < 0: continue
                    s = np.arange(max(0, aq - c + 1), aq + 1)
                    p = knn_pred_from(s, X[q])
                    if np.isfinite(p):
                        errs.append(abs(p - Y[q]))
                if errs:
                    e = float(np.mean(errs))
                    if e < bestmae:
                        bestmae, best = e, c
            m = best
            m_trace.append((t, m))
            # which memory predicts
            scores = {}
            for name in ("STM", "LTM", "UNION"):
                errs = []
                for q in qs:
                    aq = int(A[q])
                    if aq < 0: continue
                    s = np.arange(max(0, aq - m + 1), aq + 1)
                    src = s if name == "STM" else (ltm_idx[ltm_idx <= aq] if name == "LTM"
                                                   else np.concatenate([s, ltm_idx[ltm_idx <= aq]]))
                    p = knn_pred_from(src, X[q])
                    if np.isfinite(p):
                        errs.append(abs(p - Y[q]))
                scores[name] = float(np.mean(errs)) if errs else np.inf
            mode = min(scores, key=scores.get)
    return pred, Lmax * 132, dict(m_trace=m_trace[-20:], mean_m=float(np.mean([x[1] for x in m_trace])),
                                  mode_counts=mode_counts)


# ---------------------------------------------------------------- strong baselines
def gbdt(refit=720, window=None, cat=False, seed=0, **kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xf = X0
    if cat:
        wm = np.load(os.path.join(OUT, "wmain_onehot.npy"))
        if SHUF:
            wm = wm[perm]
        Xf = np.hstack([X0, wm])
    pred = np.empty(N)
    nbytes = 0
    # cold start: causal running mean until the first refit
    pred[:refit] = np.concatenate([[Y[0]], np.cumsum(Y)[:refit - 1] / np.arange(1, refit)])
    for t in range(refit, N, refit):
        hi_fit = int(A[t]) + 1
        lo = 0 if window is None else max(0, hi_fit - window)
        mdl = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **kw)
        mdl.fit(Xf[lo:hi_fit], Y[lo:hi_fit])
        nbytes = len(pickle.dumps(mdl, protocol=5))
        hi = min(t + refit, N)
        pred[t:hi] = mdl.predict(Xf[t:hi])           # model frozen between refits (exact, causal)
    return pred, nbytes


def rls(lam=1.0, forget=1.0):
    Xs = causal_std(X0)
    Az = np.hstack([Xs, np.ones((N, 1))])
    d = Az.shape[1]
    P = np.eye(d) / lam
    w = np.zeros(d)
    pred = np.empty(N)
    nxt = 0
    for t in range(N):
        while nxt <= int(A[t]):                      # absorb every pair whose target has occurred
            xu = Az[nxt]
            Px = P @ xu
            gain = Px / (forget + xu @ Px)
            w = w + gain * (Y[nxt] - xu @ w)
            P = (P - np.outer(gain, Px)) / forget
            nxt += 1
        pred[t] = Az[t] @ w
    return pred, (d * d + d) * 8


def gru(hidden=64, refit=720, buf=8760, epochs=2, bs=128, lr=1e-3, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed); torch.set_num_threads(4)
    Xs = causal_std(X0).astype(np.float32)
    seq = Xs[:, :24][:, ::-1].copy()            # chronological order of the 24 recent lags
    stat = Xs[:, 24:].copy()                    # 8 static features
    ys = (Y / 1000.0).astype(np.float32)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.g = nn.GRU(1, hidden, batch_first=True)
            s.h = nn.Sequential(nn.Linear(hidden + 8, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(s, a, b):
            o, _ = s.g(a.unsqueeze(-1))
            return s.h(torch.cat([o[:, -1], b], 1)).squeeze(-1)

    net = M()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.SmoothL1Loss()
    pred = np.zeros(N)
    tseq = torch.from_numpy(seq); tstat = torch.from_numpy(stat); tY = torch.from_numpy(ys)
    trained = False
    for t in range(N):
        if t >= refit and t % refit == 0:
            hi_fit = int(A[t]) + 1
            lo = max(0, hi_fit - buf)
            idx = np.arange(lo, hi_fit)
            net.train()
            for _ in range(epochs):
                np.random.shuffle(idx)
                for b in range(0, len(idx), bs):
                    j = torch.from_numpy(idx[b:b + bs])
                    opt.zero_grad()
                    l = lossf(net(tseq[j], tstat[j]), tY[j])
                    l.backward(); opt.step()
            trained = True
        if trained:
            net.eval()
            with torch.no_grad():
                pred[t] = float(net(tseq[t:t + 1], tstat[t:t + 1])[0]) * 1000.0
        else:
            pred[t] = Y[:t].mean() if t else Y[0]
    nb = sum(p.numel() for p in net.parameters()) * 4
    return pred, nb


def mlp(refit=1440, seed=0, hidden=(256, 128)):
    from sklearn.neural_network import MLPRegressor
    Xs = causal_std(X0)
    pred = np.empty(N); nbytes = 0
    pred[:refit] = np.concatenate([[Y[0]], np.cumsum(Y)[:refit - 1] / np.arange(1, refit)])
    for t in range(refit, N, refit):
        m = MLPRegressor(hidden_layer_sizes=hidden, random_state=seed, max_iter=400,
                         early_stopping=False, learning_rate_init=1e-3)
        m.fit(Xs[:int(A[t]) + 1], Y[:int(A[t]) + 1] / 1000.0)
        nbytes = len(pickle.dumps(m, protocol=5))
        hi = min(t + refit, N)
        pred[t:hi] = m.predict(Xs[t:hi]) * 1000.0
    return pred, nbytes


def ridge_batch(refit=720, alpha=1.0):
    from sklearn.linear_model import Ridge
    Xs = causal_std(X0)
    pred = np.empty(N); nbytes = 0
    pred[:refit] = np.concatenate([[Y[0]], np.cumsum(Y)[:refit - 1] / np.arange(1, refit)])
    for t in range(refit, N, refit):
        m = Ridge(alpha=alpha).fit(Xs[:int(A[t]) + 1], Y[:int(A[t]) + 1])
        nbytes = len(pickle.dumps(m, protocol=5))
        hi = min(t + refit, N)
        pred[t:hi] = m.predict(Xs[t:hi])
    return pred, nbytes


ARMS = {}
if __name__ == "__main__":
    arm = sys.argv[1]
    tag = arm + ("_shuf" + SHUF if SHUF else "")
    t0 = time.time()
    extra = {}
    if arm.startswith("knnwin"):
        _, L, sc = arm.split("_")
        p, nb = knn_window(int(L), std=(sc == "std"))
    elif arm.startswith("knnsam"):
        _, L, sc = arm.split("_")
        p, nb, extra = knn_sam(int(L), std=(sc == "std"))
    elif arm == "gbdt":
        p, nb = gbdt()
    elif arm == "gbdt_recent":
        p, nb = gbdt(window=8760)
    elif arm == "gbdt_cat":
        p, nb = gbdt(cat=True)
    elif arm == "gbdt_fast":
        p, nb = gbdt(refit=2000)
    elif arm == "gbdt_tuned":
        p, nb = gbdt(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=10)
    elif arm == "gbdt_tuned_recent":
        p, nb = gbdt(window=8760, max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                     min_samples_leaf=10)
    elif arm == "gbdt_tuned_r180":
        p, nb = gbdt(refit=180, max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                     min_samples_leaf=10)
    elif arm == "rls":
        p, nb = rls()
    elif arm == "rls_ff":
        p, nb = rls(forget=0.999)
    elif arm == "gru":
        p, nb = gru()
    elif arm == "gru_big":
        p, nb = gru(hidden=128, epochs=6, buf=10 ** 9)
    elif arm == "mlp":
        p, nb = mlp()
    elif arm == "ridge_batch":
        p, nb = ridge_batch()
    elif arm.startswith("knnk"):                    # k-ladder, anti-hobbling on the launder side
        _, L, kk, sc = arm.split("_")
        p, nb = knn_window(int(L), k=int(kk), std=(sc == "std"))
    else:
        raise SystemExit("unknown arm " + arm)
    r = mets(p); r["state_bytes"] = int(nb); r["wall_s"] = round(time.time() - t0, 1)
    r["arm"] = tag; r.update(extra)
    os.makedirs(os.path.join(OUT, "res"), exist_ok=True)
    with open(os.path.join(OUT, "res", tag + ".json"), "w") as f:
        json.dump(r, f, indent=2)
    np.savez_compressed(os.path.join(OUT, "res", tag + ".npz"), pred=p.astype(np.float32))
    print(tag, json.dumps({k: v for k, v in r.items() if k != "m_trace"}))
