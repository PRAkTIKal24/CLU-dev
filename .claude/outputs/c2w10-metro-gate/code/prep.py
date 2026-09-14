"""c2w10-metro-gate :: STEP 1 -- freeze, hash, canonicalise, and report STRUCTURE ONLY.

No scored quantity (no MAE/RMSE/accuracy) is computed here.  Structural facts only,
so that PREREG.md can be written with knowledge of shape but not of outcome.

Outputs: facts.json, grid.npz  (in .claude/scratch/c2w10-metro/)
"""
import csv, gzip, hashlib, json, os, sys
from datetime import datetime, timedelta
import numpy as np

DATA = "/Users/user/Desktop/CHLU/.claude/data/c2w10-metro"
OUT = "/Users/user/Desktop/CHLU/.claude/scratch/c2w10-metro"
CSVP = os.path.join(DATA, "Metro_Interstate_Traffic_Volume.csv")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows = []
with open(CSVP, newline="") as f:
    r = csv.DictReader(f)
    header = r.fieldnames
    for d in r:
        rows.append(d)
n_raw = len(rows)

ts = [datetime.strptime(d["date_time"], "%Y-%m-%d %H:%M:%S") for d in rows]
assert all(t.minute == 0 and t.second == 0 for t in ts), "non-hourly timestamp present"

# monotone?
n_nonmono = sum(1 for a, b in zip(ts, ts[1:]) if b < a)

# duplicates
seen = {}
dup_count = 0
dup_ts = set()
for k, t in enumerate(ts):
    if t in seen:
        dup_count += 1
        dup_ts.add(t)
    else:
        seen[t] = k
# DECLARED RULE: keep the FIRST row for a duplicated timestamp.
keep_idx = sorted(seen.values())

t0, t1 = min(ts), max(ts)
n_hours = int((t1 - t0).total_seconds() // 3600) + 1

# canonical hourly grid
vol = np.full(n_hours, np.nan, dtype=np.float64)
temp = np.full(n_hours, np.nan, dtype=np.float64)
rain = np.full(n_hours, np.nan, dtype=np.float64)
snow = np.full(n_hours, np.nan, dtype=np.float64)
clouds = np.full(n_hours, np.nan, dtype=np.float64)
holiday = np.zeros(n_hours, dtype=np.float64)
present = np.zeros(n_hours, dtype=bool)
wmain = np.full(n_hours, -1, dtype=np.int32)
wmain_vocab = {}

for k in keep_idx:
    d = rows[k]
    g = int((ts[k] - t0).total_seconds() // 3600)
    present[g] = True
    vol[g] = float(d["traffic_volume"])
    temp[g] = float(d["temp"])
    rain[g] = float(d["rain_1h"])
    snow[g] = float(d["snow_1h"])
    clouds[g] = float(d["clouds_all"])
    holiday[g] = 0.0 if d["holiday"] == "None" else 1.0
    wm = d["weather_main"]
    if wm not in wmain_vocab:
        wmain_vocab[wm] = len(wmain_vocab)
    wmain[g] = wmain_vocab[wm]

n_present = int(present.sum())
n_missing = n_hours - n_present

# gap run-lengths
gaps = []
run = 0
for p in present:
    if not p:
        run += 1
    elif run:
        gaps.append(run)
        run = 0
if run:
    gaps.append(run)
gaps_sorted = sorted(gaps, reverse=True)

# holiday quirk check
hol_hours = int(holiday.sum())

facts = dict(
    source_zip=dict(path=os.path.join(DATA, "metro.zip"), sha256=sha256(os.path.join(DATA, "metro.zip")),
                    bytes=os.path.getsize(os.path.join(DATA, "metro.zip")),
                    url="https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"),
    stream_csv_gz=dict(path=os.path.join(DATA, "Metro_Interstate_Traffic_Volume.csv.gz"),
                       sha256=sha256(os.path.join(DATA, "Metro_Interstate_Traffic_Volume.csv.gz")),
                       bytes=os.path.getsize(os.path.join(DATA, "Metro_Interstate_Traffic_Volume.csv.gz"))),
    stream_csv=dict(path=CSVP, sha256=sha256(CSVP), bytes=os.path.getsize(CSVP)),
    header=header,
    n_raw_records=n_raw,
    n_nonmonotone_steps=n_nonmono,
    n_duplicate_rows=dup_count,
    n_duplicated_timestamps=len(dup_ts),
    dedupe_rule="keep FIRST row per date_time (declared)",
    n_unique_timestamps=len(seen),
    t_first=str(t0), t_last=str(t1),
    grid_hours=n_hours,
    n_present=n_present,
    n_missing_hours=n_missing,
    missing_frac=n_missing / n_hours,
    n_gap_runs=len(gaps),
    longest_gap_hours=gaps_sorted[:10],
    weather_main_vocab=wmain_vocab,
    holiday_flagged_hours=hol_hours,
    ranges=dict(
        traffic_volume=[float(np.nanmin(vol)), float(np.nanmax(vol)), float(np.nanmean(vol)), float(np.nanstd(vol))],
        temp_K=[float(np.nanmin(temp)), float(np.nanmax(temp))],
        rain_1h=[float(np.nanmin(rain)), float(np.nanmax(rain))],
        snow_1h=[float(np.nanmin(snow)), float(np.nanmax(snow))],
        clouds_all=[float(np.nanmin(clouds)), float(np.nanmax(clouds))],
    ),
    n_temp_zero_kelvin=int(np.nansum(temp == 0.0)),
    n_rain_gt_1000=int(np.nansum(rain > 1000.0)),
)

np.savez_compressed(os.path.join(OUT, "grid.npz"), vol=vol, temp=temp, rain=rain, snow=snow,
                    clouds=clouds, holiday=holiday, present=present, wmain=wmain,
                    t0=np.array([t0.timestamp()]))
with open(os.path.join(OUT, "facts.json"), "w") as f:
    json.dump(facts, f, indent=2)
print(json.dumps(facts, indent=2))
