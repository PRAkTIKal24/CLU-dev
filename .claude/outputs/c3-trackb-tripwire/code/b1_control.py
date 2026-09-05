"""B1b — the CAMELS loader positive control (must pass before any tripwire
number is quoted).

Two legs:
  (i)  our loader's observed discharge in mm/day vs CAMELS' own OBS_RUN column
       in the shipped SAC-SMA + Snow-17 model-output files  (checks the area
       normalisation, the date alignment and the missing-value flag)
  (ii) median NSE of the shipped calibrated SAC-SMA over the 447-basin common
       set in the published validation window, vs Kratzert 2019 Table 3
       (0.603) and vs their per-basin values in all_metrics.p
"""
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/user/Desktop/CHLU/.claude/data/c3-camels")
OUTD = Path("/Users/user/Desktop/CHLU/.claude/outputs/c3-trackb-tripwire")
SCR = Path("/Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire")
TEST = (np.datetime64("1989-10-01"), np.datetime64("1999-09-30"))


def calc_nse(obs, sim):
    m = obs >= 0
    obs, sim = obs[m], sim[m]
    den = np.sum((obs - obs.mean()) ** 2)
    if den == 0 or obs.size < 2:
        return np.nan
    return float(1 - np.sum((sim - obs) ** 2) / den)


def main():
    st = np.load(ROOT / "staged/discharge.npz", allow_pickle=True)
    Q = st["Q"].astype(np.float64)
    basins = list(st["basins"])
    dates = np.array(st["dates"], dtype="datetime64[D]")
    bidx = {b: i for i, b in enumerate(basins)}
    b447 = set(json.load(open(SCR / "ref/basins_447_derived.json")))

    zf = zipfile.ZipFile(ROOT / "basin_timeseries_v1p2_modelOutput_daymet.zip")
    pat = re.compile(r"model_output.*?/(\d{8})_(\d+)_model_output\.txt$")
    files = defaultdict(dict)
    for n in zf.namelist():
        m = pat.search(n)
        if m:
            files[m.group(1)][m.group(2)] = n
    print("basins with model output:", len(files),
          "seeds/basin:", sorted({len(v) for v in files.values()}))

    common = [b for b in basins if b in b447 and b in files]
    print("447-set basins with model output:", len(common))

    obs_delta = []
    nse_seed = defaultdict(list)
    nse_ens = {}
    for j, b in enumerate(common):
        seeds = sorted(files[b])
        mods = {}
        obs_ref = None
        for s in seeds:
            df = pd.read_csv(io.BytesIO(zf.read(files[b][s])), sep=r"\s+")
            d = pd.to_datetime(dict(year=df.YR, month=df.MNTH, day=df.DY))
            df.index = d
            df = df[(df.index >= pd.Timestamp(str(TEST[0]))) &
                    (df.index <= pd.Timestamp(str(TEST[1])))]
            mods[s] = df.MOD_RUN.values.astype(np.float64)
            obs_ref = df.OBS_RUN.values.astype(np.float64)
            didx = df.index.values.astype("datetime64[D]")
        i = bidx[b]
        ours = Q[i][np.isin(dates, didx)]
        m = (obs_ref >= 0) & (ours >= 0)
        if m.sum():
            obs_delta.append(float(np.max(np.abs(ours[m] - obs_ref[m]))))
        for s in seeds:
            nse_seed[s].append(calc_nse(obs_ref, mods[s]))
        nse_ens[b] = calc_nse(obs_ref,
                              np.mean([mods[s] for s in seeds], axis=0))
        if j % 100 == 0:
            print("  ", j, b, flush=True)

    res = dict(
        n_basins=len(common),
        obs_max_abs_delta_p50=float(np.median(obs_delta)),
        obs_max_abs_delta_p95=float(np.percentile(obs_delta, 95)),
        obs_max_abs_delta_max=float(np.max(obs_delta)),
        per_seed_median_nse={s: float(np.nanmedian(v))
                             for s, v in nse_seed.items()},
        seed_ensemble_median_nse=float(np.nanmedian(list(nse_ens.values()))),
        seed_ensemble_mean_nse=float(np.nanmean(list(nse_ens.values()))),
        seed_ensemble_n_le0=int(np.sum(np.array(list(nse_ens.values())) <= 0)),
        published_table3_sacsma_median=0.603,
        published_table3_sacsma_mean=0.564,
        published_table3_sacsma_n_le0=13)

    # per-basin comparison with Kratzert's own SAC_SMA values
    import pickle
    am = pickle.load(open(SCR / "ref/all_metrics.p", "rb"))["NSE"]["benchmarks"]
    ref = am["SAC_SMA"]
    d = np.array([nse_ens[b] - ref[b] for b in common if b in ref])
    res.update(per_basin_vs_kratzert=dict(
        n=int(len(d)), median_abs=float(np.median(np.abs(d))),
        p90_abs=float(np.percentile(np.abs(d), 90)),
        frac_within_0p01=float(np.mean(np.abs(d) <= 0.01)),
        max_abs=float(np.max(np.abs(d)))))
    json.dump(res, open(OUTD / "B1_loader_control.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
