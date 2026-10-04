"""STEP 2 -- build the frozen prequential pair stream declared in PREREG §1.

Emits pairs.npz: X (n,32) float32, y (n,) float32, tgt_grid (n,) int32, and the
sha256 of the emitted array bytes (the contract for the reproduction gate).
"""
import hashlib, json, os
import numpy as np
from datetime import datetime, timedelta

OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
MAXGAP = 3
LAGS_RECENT = 24          # y_i .. y_{i-23}
WEEKLY = (143, 144, 145)  # y_{i-143}, y_{i-144}, y_{i-145}
DEPTH = 146               # deepest lag needed at the origin
HORIZON = 24

d = np.load(os.path.join(OUT, "grid.npz"))
vol = d["vol"].copy(); temp = d["temp"].copy(); rain = d["rain"].copy()
snow = d["snow"].copy(); clouds = d["clouds"].copy(); holiday = d["holiday"].copy()
pres = d["present"].copy(); t0 = datetime.fromtimestamp(d["t0"][0])
n = len(pres)

# --- declared gap-fill: runs of <= MAXGAP missing hours ---
runs = []; run = 0; start = 0
for i, p in enumerate(pres):
    if not p:
        if run == 0: start = i
        run += 1
    elif run:
        runs.append((start, run)); run = 0
if run: runs.append((start, run))

valid = pres.copy()
n_filled = 0
for s, L in runs:
    if L <= MAXGAP and s > 0 and s + L < n:
        a, b = s - 1, s + L                       # bracketing observed hours
        w = (np.arange(1, L + 1) / (L + 1))
        vol[s:s + L] = vol[a] * (1 - w) + vol[b] * w   # linear interp on traffic
        for arr in (temp, rain, snow, clouds, holiday):
            arr[s:s + L] = arr[a]                      # forward-fill on weather
        valid[s:s + L] = True
        n_filled += L

# --- scored pairs ---
c = np.concatenate([[0], np.cumsum(valid.astype(np.int64))])
origins = []
for i in range(DEPTH, n - HORIZON):
    if c[i + 1] - c[i - DEPTH] == DEPTH + 1 and pres[i + HORIZON]:
        origins.append(i)
origins = np.asarray(origins, dtype=np.int64)
m = len(origins)

X = np.empty((m, 32), dtype=np.float32)
for k in range(LAGS_RECENT):
    X[:, k] = vol[origins - k]
for k, w in enumerate(WEEKLY):
    X[:, LAGS_RECENT + k] = vol[origins - w]
X[:, 27] = temp[origins]; X[:, 28] = rain[origins]
X[:, 29] = snow[origins]; X[:, 30] = clouds[origins]
X[:, 31] = holiday[origins]
y = vol[origins + HORIZON].astype(np.float32)
tgt = (origins + HORIZON).astype(np.int64)
assert np.isfinite(X).all() and np.isfinite(y).all()

FEATNAMES = ([f"y_lag{k}" for k in range(LAGS_RECENT)] +
             [f"y_lag{w}" for w in WEEKLY] +
             ["temp", "rain_1h", "snow_1h", "clouds_all", "holiday"])

h = hashlib.sha256()
h.update(X.tobytes()); h.update(y.tobytes()); h.update(tgt.astype(np.int64).tobytes())
stream_sha = h.hexdigest()

np.savez(os.path.join(OUT, "pairs.npz"), X=X, y=y, tgt=tgt, origins=origins)
meta = dict(n_pairs=int(m), n_features=32, feature_names=FEATNAMES,
            bytes_per_exemplar=132, horizon_hours=HORIZON, maxgap_filled=MAXGAP,
            n_hours_filled=int(n_filled), n_valid_hours=int(valid.sum()),
            stream_sha256=stream_sha,
            t_first_target=str(t0 + timedelta(hours=int(tgt[0]))),
            t_last_target=str(t0 + timedelta(hours=int(tgt[-1]))),
            y_mean=float(y.mean()), y_std=float(y.std()),
            budget_L={"665000B_0.634MiB": 665000 // 132, "133000B": 133000 // 132,
                      "1966080B_CLU_d12": 1966080 // 132})
with open(os.path.join(OUT, "pairs_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2)[:1400])
