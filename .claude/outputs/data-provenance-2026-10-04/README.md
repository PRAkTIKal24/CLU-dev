# Dataset provenance, recorded before the local datasets were deleted (2026-10-04)
Checksums: `data-sha256.txt` (87 files) and sizes: `data-sizes.txt`. Paths are relative to `.claude/data/`. Re-download, then verify against these.

| dir | size | source (as found in the files and records; not invented) |
|---|---|---|
| `c3-camels/` | 7.2 GB | CAMELS (US), attributes v2.0, described in HESS, doi:10.5194/hess-2017-169 (the dataset's own `readme.txt`, head kept in `c3-camels-readme-head.txt`). The exact download URL is not recorded in the code. |
| `c3-ncmapss/` | 2.3 GB | NASA turbofan degradation data. URLs found in the records: `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip` (N-CMAPSS) and `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (C-MAPSS) |
| `gen/` | 331 MB | Shapes3D: `https://storage.googleapis.com/3d-shapes/3dshapes.h5` (`chlu/data/shapes3d.py`); dSprites: the DeepMind dsprites-dataset GitHub (`chlu/data/dsprites.py`) |
| `hepa/` | 172 MB | not recorded in code; see the `hepa-*` records |
| `c2w10-streams/`, `c2w10-metro/` | 160 MB | generated/derived streams from the C2 campaign; regenerable from code |
