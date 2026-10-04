"""PART B (full) -- the Metro drift map, OURS not the literature's.

Three published objects:
  1. Webb-style drift-magnitude matrices (TV between windowed distributions) at day and week scale.
  2. Candidate regimes + the explicit revisit-schedule index table.
  3. Band diagnostics + the registered exclusions (the INSECTS b=4 lesson applied BEFORE use).
"""
import json, os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
g = np.load(os.path.join(OUT, "grid.npz"))
p = np.load(os.path.join(OUT, "pairs.npz"))
t0 = datetime.fromtimestamp(g["t0"][0])
X, Y, tgt = p["X"].astype(np.float64), p["y"].astype(np.float64), p["tgt"]
N = len(Y)
times = [t0 + timedelta(hours=int(h)) for h in tgt]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
temp, rain, snow, clouds = X[:, 27], X[:, 28], X[:, 29], X[:, 30]

NB = 10
def qbin(v, nb=NB):
    e = np.quantile(v, np.linspace(0, 1, nb + 1)); e[0] -= 1e-9; e[-1] += 1e-9
    return np.clip(np.digitize(v, e[1:-1]), 0, nb - 1)
ybin, tbin, cbin = qbin(Y), qbin(temp), qbin(clouds)

def sigs(keys, minn, joint):
    uk = sorted(set(keys.tolist())); S, M = [], []
    for k in uk:
        m = keys == k
        if m.sum() < minn: continue
        h = np.bincount(ybin[m], minlength=NB).astype(float); h /= h.sum()
        if joint:
            h2 = np.bincount(tbin[m], minlength=NB).astype(float); h2 /= h2.sum()
            h3 = np.bincount(cbin[m], minlength=NB).astype(float); h3 /= h3.sum()
            h = np.concatenate([h, h2, h3]) / 3.0
        S.append(h); M.append(dict(key=int(k), n=int(m.sum()), idx=np.where(m)[0]))
    return np.array(S), M

def tvmat(S, nblocks=1):
    return (nblocks / 2.0) * np.abs(S[:, None, :] - S[None, :, :]).sum(-1)

def pick_K(S, lo=2, hi=8):
    sil = {}
    for K in range(lo, hi + 1):
        km = KMeans(K, n_init=10, random_state=0).fit(S)
        sil[K] = float(silhouette_score(S, km.labels_))
    return sil, max(sil, key=sil.get)

def relabel_by_mean(lab, K, idxs):
    means = [np.mean(np.concatenate([idxs[i] for i in np.where(lab == c)[0]])) for c in range(K)]
    return lab, means

def build(keys, minn, joint, K=None, name=""):
    S, M = sigs(keys, minn, joint)
    TV = tvmat(S, 3 if joint else 1)
    sil, Kb = pick_K(S)
    Kuse = K or Kb
    km = KMeans(Kuse, n_init=10, random_state=0).fit(S)
    lab = km.labels_
    order = np.argsort([Y[np.concatenate([M[i]["idx"] for i in np.where(lab == c)[0]])].mean()
                        for c in range(Kuse)])
    rm = {int(c): int(r) for r, c in enumerate(order)}
    lab = np.array([rm[int(c)] for c in lab])
    return dict(S=S, M=M, TV=TV, sil=sil, Kbest=Kb, K=Kuse, lab=lab, name=name, keys=keys, minn=minn)


def describe(B, unit_days):
    M, lab, K = B["M"], B["lab"], B["K"]
    regs = []
    for c in range(K):
        ii = np.where(lab == c)[0]
        idx = np.concatenate([M[i]["idx"] for i in ii])
        dows, months, years = defaultdict(int), defaultdict(int), defaultdict(int)
        for i in ii:
            d0 = datetime.fromordinal(M[i]["key"] if unit_days else M[i]["key"] * 7)
            for off in range(1 if unit_days else 7):
                dt = d0 + timedelta(days=off)
                dows[DOW[dt.weekday()]] += 1; months[dt.month] += 1; years[dt.year] += 1
        tot = sum(dows.values())
        regs.append(dict(regime=int(c), n_windows=len(ii), n_pairs=int(len(idx)),
                         mean_y=round(float(Y[idx].mean()), 1), sd_y=round(float(Y[idx].std()), 1),
                         mean_temp_K=round(float(temp[idx].mean()), 2),
                         mean_clouds=round(float(clouds[idx].mean()), 1),
                         dow_pct={k: round(100 * v / tot, 1) for k, v in
                                  sorted(dows.items(), key=lambda x: -x[1])},
                         month_pct={str(k): round(100 * v / tot, 1) for k, v in sorted(months.items())},
                         year_counts={str(k): v for k, v in sorted(years.items())}))
    # revisit schedule
    keysS = [M[i]["key"] for i in range(len(M))]
    step = 1
    runs, seen = [], defaultdict(int)
    cl, cs, prev = lab[0], 0, keysS[0]
    for i in range(1, len(lab)):
        contig = (keysS[i] == prev + step)
        if lab[i] != cl or not contig:
            runs.append((int(cl), cs, i - 1)); cl, cs = lab[i], i
        prev = keysS[i]
    runs.append((int(cl), cs, len(lab) - 1))
    rev = []
    for r, a, b in runs:
        seen[r] += 1
        idx = np.concatenate([M[i]["idx"] for i in range(a, b + 1)])
        d0 = datetime.fromordinal(keysS[a] if unit_days else keysS[a] * 7)
        d1 = datetime.fromordinal(keysS[b] if unit_days else keysS[b] * 7)
        rev.append(dict(regime=r, visit=seen[r], first=str(d0.date()), last=str(d1.date()),
                        n_windows=b - a + 1, n_pairs=int(len(idx)),
                        pair_first=int(idx.min()), pair_last=int(idx.max())))
    return regs, rev, dict(seen)


maps = {}
maps["D_primary"] = build(np.array([t.toordinal() for t in times]), 12, False, None, "day/y-only")
maps["D_fine4"] = build(np.array([t.toordinal() for t in times]), 12, False, 4, "day/y-only K=4")
maps["D_joint"] = build(np.array([t.toordinal() for t in times]), 12, True, None, "day/joint(y,temp,clouds)")
maps["W_season"] = build(np.array([t.toordinal() // 7 for t in times]), 60, True, None, "week/joint")

out = {}
for k, B in maps.items():
    regs, rev, cnt = describe(B, unit_days=k.startswith("D"))
    iu = np.triu_indices(len(B["S"]), 1)
    out[k] = dict(name=B["name"], n_windows=len(B["S"]), min_pairs_per_window=B["minn"],
                  silhouette={str(a): round(b, 4) for a, b in B["sil"].items()},
                  K_silhouette=B["Kbest"], K_used=B["K"],
                  mean_pairwise_TV=round(float(B["TV"][iu].mean()), 4),
                  TV_p5_p50_p95=[round(float(np.percentile(B["TV"][iu], q)), 4) for q in (5, 50, 95)],
                  regimes=regs, n_visits_per_regime=cnt, revisit_schedule=rev)
    lab_pair = np.full(N, -1, np.int64)
    for i in range(len(B["M"])):
        lab_pair[B["M"][i]["idx"]] = B["lab"][i]
    np.save(os.path.join(OUT, f"pair_regime_{k}.npy"), lab_pair)
    out[k]["n_pairs_unassigned"] = int((lab_pair < 0).sum())
    print(k, "K_sil", B["Kbest"], "K_used", B["K"], "windows", len(B["S"]),
          "visits", cnt, "unassigned", out[k]["n_pairs_unassigned"])

np.savez_compressed(os.path.join(OUT, "driftmaps.npz"),
                    **{f"TV_{k}": v["TV"] for k, v in maps.items()},
                    **{f"lab_{k}": v["lab"] for k, v in maps.items()},
                    **{f"key_{k}": np.array([m["key"] for m in v["M"]]) for k, v in maps.items()})
out["_method"] = ("Webb, Hyde, Cao, Nguyen & Petitjean (2016), Characterizing concept drift, "
                  "DMKD 30:964-994: drift magnitude = total variation distance between the "
                  "distributions in two windows. Estimator: 10 global-quantile bins of the target "
                  "(and of temp and clouds_all for the 'joint' maps), per-window normalised "
                  "histogram, TV between every window pair. Windows = calendar days (>=12 scored "
                  "pairs) or calendar weeks (>=60). Regimes = KMeans on the window signature, K by "
                  "silhouette. Calendar attributes (day-of-week, month, year) are attached AFTER "
                  "clustering, for LABELLING only, and were never inputs to the discovery. "
                  "*** THIS DRIFT ANNOTATION IS OURS, NOT THE LITERATURE'S. *** Metro Interstate "
                  "has no published drift annotation (verified by c2w10-benchmark-scout).")
with open(os.path.join(OUT, "driftmaps.json"), "w") as f:
    json.dump(out, f, indent=2)
print("wrote driftmaps.json")
