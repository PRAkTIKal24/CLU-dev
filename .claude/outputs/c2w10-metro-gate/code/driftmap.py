"""PART B -- the drift map.  OURS, not the literature's.

Webb-style drift magnitude (total variation between windowed distributions, Webb et al.
DMKD 30:964-994, 2016) at two scales:
  D-map  window = one calendar day   (where day-type / holiday regimes live)
  W-map  window = one calendar week  (where seasonal / era regimes live)
Discovery uses ONLY the stream (y and the weather covariates).  The calendar is used
afterwards, for LABELLING the discovered regimes, never for discovering them.
"""
import json, os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
g = np.load(os.path.join(OUT, "grid.npz"))
p = np.load(os.path.join(OUT, "pairs.npz"))
t0 = datetime.fromtimestamp(g["t0"][0])
X, Y, tgt = p["X"].astype(np.float64), p["y"].astype(np.float64), p["tgt"]
N = len(Y)
times = [t0 + timedelta(hours=int(h)) for h in tgt]
day_key = np.array([t.toordinal() for t in times])
NBINS = 10
qe = np.quantile(Y, np.linspace(0, 1, NBINS + 1))
qe[0] -= 1e-6; qe[-1] += 1e-6
ybin = np.clip(np.digitize(Y, qe[1:-1]), 0, NBINS - 1)
# weather covariates for the signature (all observed at the origin)
temp, rain, snow, clouds = X[:, 27], X[:, 28], X[:, 29], X[:, 30]


def window_sigs(keys, minn):
    uk = sorted(set(keys.tolist()))
    sigs, meta = [], []
    for k in uk:
        m = keys == k
        if m.sum() < minn:
            continue
        h = np.bincount(ybin[m], minlength=NBINS).astype(np.float64)
        h /= h.sum()
        sigs.append(h)
        meta.append(dict(key=int(k), n=int(m.sum()), idx=np.where(m)[0],
                         mean_y=float(Y[m].mean()), sd_y=float(Y[m].std()),
                         mean_temp=float(temp[m].mean()), sum_rain=float(rain[m].sum()),
                         sum_snow=float(snow[m].sum()), mean_clouds=float(clouds[m].mean())))
    return np.array(sigs), meta


def tvmat(S):
    return 0.5 * np.abs(S[:, None, :] - S[None, :, :]).sum(-1)


# ---------------- D-map (primary) ----------------
Sd, Md = window_sigs(day_key, 12)
TVd = tvmat(Sd)
print("D-map: %d days retained (of %d), mean pairwise TV %.4f" %
      (len(Sd), len(set(day_key.tolist())), TVd[np.triu_indices(len(Sd), 1)].mean()))

# regime discovery: k-means on the y-histogram signature, K chosen by silhouette over 2..8
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sil = {}
for K in range(2, 9):
    km = KMeans(K, n_init=10, random_state=0).fit(Sd)
    sil[K] = float(silhouette_score(Sd, km.labels_))
Kbest = max(sil, key=sil.get)
km = KMeans(Kbest, n_init=10, random_state=0).fit(Sd)
lab = km.labels_
print("silhouette:", {k: round(v, 4) for k, v in sil.items()}, "-> K =", Kbest)

# order regimes by mean y so labels are stable/interpretable
order = np.argsort([np.mean([Md[i]["mean_y"] for i in range(len(Md)) if lab[i] == c])
                    for c in range(Kbest)])
remap = {int(c): int(r) for r, c in enumerate(order)}
lab = np.array([remap[int(c)] for c in lab])

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
regimes = []
for c in range(Kbest):
    ii = np.where(lab == c)[0]
    idx = np.concatenate([Md[i]["idx"] for i in ii])
    dows = defaultdict(int); months = defaultdict(int)
    for i in ii:
        dt = datetime.fromordinal(Md[i]["key"])
        dows[DOW[dt.weekday()]] += 1
        months[dt.month] += 1
    regimes.append(dict(
        regime=int(c), n_days=len(ii), n_pairs=int(len(idx)),
        mean_y=float(Y[idx].mean()), sd_y=float(Y[idx].std()),
        mean_temp_K=float(temp[idx].mean()),
        dow_composition={k: v for k, v in sorted(dows.items(), key=lambda x: -x[1])},
        month_composition={str(k): v for k, v in sorted(months.items())},
        first_visit=str(datetime.fromordinal(Md[ii[0]]["key"]).date()),
        last_visit=str(datetime.fromordinal(Md[ii[-1]]["key"]).date()),
    ))

# revisit schedule: contiguous runs of days with the same regime label
days_sorted = [Md[i]["key"] for i in range(len(Md))]
runs = []
cur_lab, cur_start, prev_day = lab[0], 0, days_sorted[0]
for i in range(1, len(lab)):
    contiguous = (days_sorted[i] == prev_day + 1)
    if lab[i] != cur_lab or not contiguous:
        runs.append((int(cur_lab), cur_start, i - 1))
        cur_lab, cur_start = lab[i], i
    prev_day = days_sorted[i]
runs.append((int(cur_lab), cur_start, len(lab) - 1))
revisit = []
seen = defaultdict(int)
for r, a, b in runs:
    seen[r] += 1
    idx = np.concatenate([Md[i]["idx"] for i in range(a, b + 1)])
    revisit.append(dict(regime=r, visit_number=seen[r],
                        first_day=str(datetime.fromordinal(days_sorted[a]).date()),
                        last_day=str(datetime.fromordinal(days_sorted[b]).date()),
                        n_days=b - a + 1, n_pairs=int(len(idx)),
                        pair_idx_first=int(idx.min()), pair_idx_last=int(idx.max())))
print("D-map: %d regime visits, per-regime visit counts %s" % (len(revisit), dict(seen)))

# ---------------- W-map (secondary, seasonal/era) ----------------
week_key = np.array([t.toordinal() // 7 for t in times])
Sw, Mw = window_sigs(week_key, 60)
TVw = tvmat(Sw)
print("W-map: %d weeks retained, mean pairwise TV %.4f" %
      (len(Sw), TVw[np.triu_indices(len(Sw), 1)].mean()))

np.savez_compressed(os.path.join(OUT, "driftmap.npz"),
                    TVd=TVd, labd=lab, dayk=np.array(days_sorted),
                    TVw=TVw, weekk=np.array([m["key"] for m in Mw]),
                    Sd=Sd, Sw=Sw)
pair_regime = np.full(N, -1, dtype=np.int64)
for i in range(len(Md)):
    pair_regime[Md[i]["idx"]] = lab[i]
np.save(os.path.join(OUT, "pair_regime.npy"), pair_regime)

with open(os.path.join(OUT, "driftmap.json"), "w") as f:
    json.dump(dict(method=("Webb et al. 2016 drift magnitude = total variation distance between "
                           "windowed distributions of the 10-quantile-binned target; windows are "
                           "calendar days (primary) and calendar weeks (secondary); regimes by "
                           "KMeans on the window signature, K by silhouette; calendar labels applied "
                           "POST HOC, never used for discovery. THIS ANNOTATION IS OURS, NOT THE "
                           "LITERATURE'S."),
                   n_days_retained=len(Sd), n_days_total=len(set(day_key.tolist())),
                   n_pairs_unassigned=int((pair_regime < 0).sum()),
                   silhouette=sil, K=int(Kbest),
                   mean_pairwise_TV_days=float(TVd[np.triu_indices(len(Sd), 1)].mean()),
                   mean_pairwise_TV_weeks=float(TVw[np.triu_indices(len(Sw), 1)].mean()),
                   regimes=regimes, revisit_schedule=revisit), f, indent=2)
print("wrote driftmap.json ; unassigned pairs:", int((pair_regime < 0).sum()))
