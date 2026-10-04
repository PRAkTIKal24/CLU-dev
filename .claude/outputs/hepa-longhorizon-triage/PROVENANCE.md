# PROVENANCE — hepa-longhorizon-triage data staging
Written by **results-analyst** (task `hepa-lh-audit`), **2026-09-07**. All fetches performed on
2026-09-07 (machine clock; `date` = Mon Sep  7 12:20–12:25 CEST 2026) from this machine over HTTPS.

⛔ **The staged data under `.claude/data/hepa/` is never committed and never backed up. THIS FILE IS
THE BACKUP.** Every row below re-creates its dataset from scratch.

## 0 · Code provenance
| Item | Value |
|---|---|
| CHLU repo HEAD at audit time | `7fcef50` (matches the PREREG freeze point) |
| HEPA clone | `.claude/hepa-src`, `https://github.com/Forgis-Labs/HEPA` |
| HEPA commit | `85f55f39a72a677d74c3371707baea67600020d6` ("Anonymise repo for double-blind review", 2026-07-27) |
| HEPA licence | **CC BY-NC-SA 4.0** (`LICENSE` = "Attribution-NonCommercial-ShareAlike 4.0 International"). Non-commercial + share-alike; it lives under `.claude/` only and must never enter a tracked file. |
| Protocol file used | `hepa/utils/config.py` (clone-actual), per PREREG §1 |
| Python env | isolated venv `.claude/scratch/hepa-lh-audit/venv`, CPython **3.11.13**, `torch 2.14.0`, `numpy 2.x`, `pandas`, `scikit-learn 1.9.0`, `openpyxl`. (Torch is needed only because `hepa/__init__.py` imports the model; the loaders themselves use numpy+pandas.) The CHLU `.venv` was NOT modified. |
| Data root | `HEPA_DATA_DIR=/Users/user/Desktop/CHLU/.claude/data/hepa` |

## 1 · Source table (9 datasets / 11 rows)
| Dataset | Source URL (exact, as fetched) | Fetched | Licence | Notes |
|---|---|---|---|---|
| C-MAPSS FD001–FD004 | `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (NASA PCoE S3 mirror; the `download_data.py` URL is a landing page, not a file) | 2026-09-07 | **NOT STATED** (NASA Prognostics Data Repository; no licence file in the archive) | outer zip sha256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`, 12 429 152 B; contains `CMAPSSData.zip` → the 12 staged `.txt` files |
| PSM | `https://raw.githubusercontent.com/eBay/RANSynCoders/main/data/{train,test,test_label}.csv` | 2026-09-07 | **BSD License** (repo `LICENSE`; GitHub API reports `NOASSERTION`) | matches the loader's CSV branch exactly |
| MBA | `https://raw.githubusercontent.com/imperial-qore/TranAD/master/data/MBA/{train,test,labels}.xlsx` | 2026-09-07 | **BSD-3-Clause** (GitHub API) | ⚠ `train.xlsx` and `test.xlsx` are **byte-identical** (same sha256) — see report §MBA |
| GECCO 2018 | `https://zenodo.org/api/records/3884398/files/1_gecco2018_water_quality.csv/content` (DOI **10.5281/zenodo.3884398**) — the `spotseven.de` challenge URL in `download_data.py` returns **404** (link rot) | 2026-09-07 | **CC BY 4.0** (Zenodo record metadata) | staged as `GECCO/gecco2018.csv`; 139 566 data rows, 1 726 `EVENT=TRUE` rows — matches the loader's 996+653+77 positives exactly |
| BATADAL | `https://www.batadal.net/data/BATADAL_dataset0{3,4}.csv` | 2026-09-07 | **CC BY 4.0** (footer of `batadal.net/data.html`) | site live |
| ETTm1 | `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv` | 2026-09-07 | **CC BY-ND 4.0** (repo `LICENSE`: "Attribution-NoDerivatives 4.0 International") | ⚠ NoDerivatives — do not redistribute a modified copy |
| Weather (Jena) | `https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv` | 2026-09-07 | HF dataset card: **CC BY 4.0**; the underlying MPI-BGC Jena data itself is **NOT STATED** | ⚠ **SOURCE SUBSTITUTION**: `download_data.py` names `bgc-jena.mpg.de/wetter/`, which serves yearly raw `mpi_roof` archives, NOT the single `weather.csv` the loader expects. I used the canonical Informer/Autoformer `weather.csv` (21 numeric channels + `date`, 10-min, 52 696 rows) mirrored by the Tsinghua THUML Time-Series-Library dataset. Building `weather.csv` from the Jena raw archives would have been a loader reimplementation, which the task forbids. |
| Beijing-AQ | `https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip` (UCI id 501) | 2026-09-07 | **CC BY 4.0** (UCI dataset page) | outer zip → `PRSA2017_Data_20130301-20170228.zip` → 12 station CSVs staged under `BeijingAQ/PRSA_Data_20130301-20170228/` (the loader's `inner` path) |

**Out of scope, not downloaded, not audited, not mentioned elsewhere:** SMAP, TEP, VIX (PREREG §6).
**NOT_ACQUIRED list: EMPTY.** All 9 datasets / 11 rows were obtained and all 11 load through HEPA's own loaders.

## 2 · Staged files — bytes + sha256
Paths are relative to `.claude/data/hepa/`.

| File | Bytes | sha256 |
|---|---:|---|
| `BATADAL/BATADAL_dataset03.csv` | 2,940,389 | `8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b` |
| `BATADAL/BATADAL_dataset04.csv` | 1,225,392 | `4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv` | 2,835,916 | `a94fbcfc71708b6ffdc033360163d91efb7f675e496f6b5860929aa96b96351b` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv` | 2,722,295 | `a2e21402a6d95fb7043f87959e6e3dcbecb957aac356a6f8028a85b047c772e7` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Dingling_20130301-20170228.csv` | 2,675,856 | `3d8dff81f18a80d9f455876a785642bffe1e9b9924ec1e6b7ce8cca790432eeb` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv` | 2,636,684 | `9c0020214425e5ee2a64d42f97a1f632e1d8f6b53ad50edd85a686cc7999f27e` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Guanyuan_20130301-20170228.csv` | 2,695,860 | `59390fee3b032597b387491f5230e7edb90a8558e30217ebd6d9befbe669fbf9` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Gucheng_20130301-20170228.csv` | 2,654,625 | `10ae48801ea55869fee56a48b885fb1127859c130551888d71673f34216ba951` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Huairou_20130301-20170228.csv` | 2,641,027 | `812dcbe848b3ba2c2d25731064797bff803c46eb0c602156a1fa044df136dc26` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Nongzhanguan_20130301-20170228.csv` | 2,839,705 | `572cf87b65ef543cd55d3a711b28474def224a46b0ec3fc8add8aaf13b67a487` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Shunyi_20130301-20170228.csv` | 2,620,654 | `520b59e4ead3b0cfe7f1ed673e30275de8441e999db6f413d5172d2277b8c15a` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Tiantan_20130301-20170228.csv` | 2,655,061 | `0743e34799cf1ee36195b73321b91baa5625276532e7263c588f68cdef3b4daa` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Wanliu_20130301-20170228.csv` | 2,659,544 | `0b1a6d754a07f0def2f55434ddfa3870dcd0063b7f766a6f66b6c11bb05a377a` |
| `BeijingAQ/PRSA_Data_20130301-20170228/PRSA_Data_Wanshouxigong_20130301-20170228.csv` | 2,871,076 | `ab1d6a356d6997a45f6d657ad47de1e05a4a08d4a7015b38a329a90665ab74d0` |
| `CMAPSS/RUL_FD001.txt` | 429 | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |
| `CMAPSS/RUL_FD002.txt` | 1,110 | `c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b` |
| `CMAPSS/RUL_FD003.txt` | 428 | `df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb` |
| `CMAPSS/RUL_FD004.txt` | 1,084 | `196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1` |
| `CMAPSS/test_FD001.txt` | 2,228,855 | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `CMAPSS/test_FD002.txt` | 5,734,587 | `de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02` |
| `CMAPSS/test_FD003.txt` | 2,826,651 | `299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790` |
| `CMAPSS/test_FD004.txt` | 6,957,759 | `1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5` |
| `CMAPSS/train_FD001.txt` | 3,515,356 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `CMAPSS/train_FD002.txt` | 9,082,480 | `dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f` |
| `CMAPSS/train_FD003.txt` | 4,213,862 | `2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2` |
| `CMAPSS/train_FD004.txt` | 10,350,705 | `27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218` |
| `ETT/ETTm1.csv` | 10,360,719 | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` |
| `GECCO/gecco2018.csv` | 10,784,171 | `c41221f3045696b75b79374f4c41c63f6c988e1df03b1dedcdbf910d44ea71cd` |
| `MBA/labels.xlsx` | 11,616 | `59a92961c16a05aa83fd146c77d7ce7a8f773ad8657834454e6467326fa4191c` |
| `MBA/test.xlsx` | 161,971 | `bf2de57b5736ba59c2844a0725c7cc7b631cd8ef527571226c908260defed00d` |
| `MBA/train.xlsx` | 161,971 | `bf2de57b5736ba59c2844a0725c7cc7b631cd8ef527571226c908260defed00d` |
| `PSM/test.csv` | 42,523,490 | `566bc50117d13f58793d830c14975c55f627cab63953265d7d6ce2dbcf2b0f80` |
| `PSM/test_label.csv` | 966,273 | `e5b6b0c716a9b3d52e3a1b06761527811f75a218790c37b8a5e806741aeb0676` |
| `PSM/train.csv` | 63,628,087 | `816e0e2a5356b3b09d356925cd9a0fe1d327e3b350e9336d26ee3e650c72e112` |
| `Weather/weather.csv` | 7,235,425 | `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63` |
(`.claude/data/hepa/cmapss/READY` is a marker file written for the concurrent `hepa-lh-tripwire`
spoke — see §7 of the task file — and is deliberately excluded from the hash table.)

## 3 · Reproduce the audit
```bash
git clone https://github.com/Forgis-Labs/HEPA .claude/hepa-src && \
  git -C .claude/hepa-src checkout 85f55f39a72a677d74c3371707baea67600020d6
# stage the files listed in §1 under .claude/data/hepa/<DIR>/ (dirs: CMAPSS PSM MBA GECCO BATADAL ETT Weather BeijingAQ)
uv venv --python 3.11 .claude/scratch/hepa-lh-audit/venv
VIRTUAL_ENV=.claude/scratch/hepa-lh-audit/venv uv pip install "torch>=2.0" numpy pandas scikit-learn openpyxl
HEPA_DATA_DIR=$PWD/.claude/data/hepa PYTHONPATH=$PWD/.claude/hepa-src \
  .claude/scratch/hepa-lh-audit/venv/bin/python .claude/scratch/hepa-lh-audit/audit.py
python3 .claude/scratch/hepa-lh-audit/finalize.py   # -> .claude/outputs/hepa-longhorizon-triage/audit.json
```
Seed 42 throughout (loader engine-id permutation AND the τ window sample). The audit is deterministic:
a rerun reproduces `audit.json` byte-for-byte apart from `runtime_s`.
