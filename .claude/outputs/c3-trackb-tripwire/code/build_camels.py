"""Stage CAMELS-US into one consolidated, sha256-frozen array set.

Conventions taken VERBATIM from the reference implementation of the strong
reference (kratzert/ealstm_regional_modeling, Apache-2.0):
  forcings  basin_mean_forcing/<product>/**/<basin>_lump_*_forcing_leap.txt
            columns prcp(mm/day), srad(W/m2), tmax(C), tmin(C), vp(Pa)
  area      line 3 of the forcing header
  discharge usgs_streamflow/**/<basin>_streamflow_qc.txt,
            q_mm_day = 28316846.592 * QObs_cfs * 86400 / (area * 1e6)
  missing   QObs < 0 (the -999 flag) -> dropped from BOTH arrays before NSE
Writes .claude/data/c3-camels/staged/*.npz  (gitignored).
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/user/Desktop/CHLU/.claude/data/c3-camels")
OUT = ROOT / "staged"
OUT.mkdir(exist_ok=True)
FORCING_ZIP = ROOT / "basin_timeseries_v1p2_metForcing_obsFlow.zip"

INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'dom_land_cover_frac',
    'dom_land_cover', 'high_prec_timing', 'low_prec_timing', 'huc', 'q_mean',
    'runoff_ratio', 'stream_elas', 'slope_fdc', 'baseflow_index', 'hfd_mean',
    'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur',
    'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99',
    'organic_frac', 'water_frac', 'other_frac']

DATES = pd.date_range("1980-01-01", "2014-12-31", freq="D")


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_attributes(basins):
    frames = []
    for f in sorted(ROOT.glob("camels_*.txt")):
        df = pd.read_csv(f, sep=";", header=0, dtype={"gauge_id": str})
        df = df.set_index("gauge_id")
        frames.append(df)
    df = pd.concat(frames, axis=1)
    df.index = [str(i).zfill(8) for i in df.index]
    df = df.loc[basins]
    df = df.drop(columns=[c for c in ["huc_02", "gauge_lat", "gauge_lon"]
                          if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c in INVALID_ATTR])
    return df


def main():
    basins = [l.strip() for l in
              open("/Users/user/Desktop/CHLU/.claude/scratch/c3-trackb-tripwire/"
                   "ref/basin_list_531.txt") if l.strip()]
    assert len(basins) == 531, len(basins)

    attrs = build_attributes(basins)
    print("attributes:", attrs.shape, list(attrs.columns))
    assert attrs.shape[0] == 531
    np.savez_compressed(OUT / "attributes.npz",
                        names=np.array(list(attrs.columns)),
                        basins=np.array(basins),
                        values=attrs.values.astype(np.float64))

    zf = zipfile.ZipFile(FORCING_ZIP)
    names = zf.namelist()
    print("zip members:", len(names))

    # NOTE (measured, 2026-08-13): the shipped `maurer` product in CAMELS v1.2
    # has tmax == tmin on >99 % of days in all 20 basins sampled (it carries the
    # daily MEAN duplicated), and 3/531 of its files have a malformed header
    # (02108000, 05120500, 09492400 lose the 'Year Mnth Day Hr' column names).
    # It is therefore a hobbled input; Daymet (the scorecard's own choice) is
    # the primary and `maurer` is a declared NOT-RUN.
    for product in ("daymet", ):
        pat = re.compile(rf"v1p2/basin_mean_forcing/{product}/.*_forcing_leap\.txt$")
        fmap = {}
        for n in names:
            if pat.search(n):
                fmap[Path(n).name[:8]] = n
        print(product, "forcing files:", len(fmap))
        missing = [b for b in basins if b not in fmap]
        assert not missing, missing[:5]

        X = np.full((531, len(DATES), 5), np.nan, dtype=np.float32)
        areas = np.zeros(531, dtype=np.float64)
        for i, b in enumerate(basins):
            raw = zf.read(fmap[b]).decode("utf-8")
            lines = raw.split("\n")
            areas[i] = float(lines[2].strip())
            hrow = next(j for j, l in enumerate(lines)
                        if l.lower().startswith("year"))
            df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=hrow)
            df.columns = [c.lower() for c in df.columns]
            idx = pd.to_datetime(dict(year=df["year"], month=df["mnth"],
                                      day=df["day"]))
            df.index = idx
            df = df.reindex(DATES)
            X[i] = df[["prcp(mm/day)", "srad(w/m2)", "tmax(c)", "tmin(c)",
                       "vp(pa)"]].values.astype(np.float32)
            if i % 100 == 0:
                print("  ", product, i, b, flush=True)
        np.savez_compressed(OUT / f"forcing_{product}.npz", X=X,
                            areas=areas, basins=np.array(basins),
                            dates=np.array([str(d.date()) for d in DATES]))
        print(product, "nan frac", float(np.isnan(X).mean()))

    # discharge (area from maurer_extended headers, as the reference does)
    areas = np.load(OUT / "forcing_daymet.npz")["areas"]
    qmap = {}
    for n in names:
        if n.endswith("_streamflow_qc.txt") and "v1p2/usgs_streamflow/" in n:
            qmap[Path(n).name[:8]] = n
    print("streamflow files:", len(qmap))
    Q = np.full((531, len(DATES)), np.nan, dtype=np.float64)
    for i, b in enumerate(basins):
        df = pd.read_csv(io.BytesIO(zf.read(qmap[b])), sep=r"\s+", header=None,
                         names=["basin", "Year", "Mnth", "Day", "QObs", "flag"])
        idx = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day))
        df.index = idx
        q = df.QObs.reindex(DATES).values.astype(np.float64)
        Q[i] = 28316846.592 * q * 86400 / (areas[i] * 1e6)
        if i % 100 == 0:
            print("   q", i, b, flush=True)
    np.savez_compressed(OUT / "discharge.npz", Q=Q.astype(np.float32),
                        areas=areas, basins=np.array(basins),
                        dates=np.array([str(d.date()) for d in DATES]))
    frac_missing = float(np.mean(~(Q >= 0)))
    print("discharge missing/neg frac:", frac_missing)

    manifest = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(OUT.glob("*.npz"))}
    manifest["_source_zip"] = dict(
        name=FORCING_ZIP.name, bytes=FORCING_ZIP.stat().st_size,
        sha256=sha256(FORCING_ZIP))
    json.dump(manifest, open(OUT / "MANIFEST.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    sys.exit(main())
