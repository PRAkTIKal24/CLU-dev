<div align="center">

<img src="assets/banner.png" alt="CAFE banner" width="100%"/>

# CAFE

### CAFE: Classification, Anomaly Detection, Forecasting of Events

**Classification · Anomaly Detection · Event Prediction**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)
[![Datasets: 50+](https://img.shields.io/badge/datasets-50+-orange.svg)](#datasets)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-brightgreen.svg)](#leaderboard)

</div>

---

**CAFE** is the companion evaluation suite for the [HEPA paper](https://arxiv.org/abs/2506.XXXXX).
It provides a single, reproducible harness to load, standardize, and evaluate any time-series foundation model across **three downstream tasks**, **50+ datasets**, and **6 domains** — with one command.

> **Design principles:** standardized splits · task-appropriate metrics · one-line model registration · JSON-first results · live leaderboard

---

## Overview

```
                 ┌─────────────────────────────────────────────────┐
                 │              cafe-bench pipeline                  │
                 │                                                  │
  raw data ──► Dataset.load() ──► Model.encode() ──► Evaluator ──► results/
                 │                      │                           │
                 │           (linear probe /                        │
                 │            kNN / CoxPH default)           leaderboard.json
                 └─────────────────────────────────────────────────┘
```

| Task | Primary Metric | # Datasets | Domains |
|---|---|---|---|
| [Classification](#1--physical-understanding) | Macro-F1 | 36 | healthcare, wearable, speech, industrial, neuro |
| [Anomaly Detection](#2--anomaly-detection) | VUS-PR | 16 | IT, ICS, healthcare, aerospace, climate |
| [Event Prediction](#3--event-prediction) | h-AUROC | 6 | aerospace, healthcare |

---

## Install

```bash
git clone https://github.com/forgislabs/cafe-bench
cd cafe-bench
pip install -e .

# optional: MOMENT baseline
pip install momentfm

# optional: UniTS baseline (install from source)
pip install git+https://github.com/mims-harvard/UniTS.git
```

---

## Quick Start

```bash
# see all registered datasets
cafe-bench ls
cafe-bench ls --task anomaly

# evaluate HEPA on a single dataset
cafe-bench run uea_epilepsy hepa \
  --checkpoint checkpoints/hepa.pt \
  --data-root data/

# sweep MOMENT across all classification datasets
cafe-bench run-all --task classification moment \
  --data-root data/

# sweep UniTS (supervised checkpoint) across all tasks
cafe-bench run-all units_ft --data-root data/

# print the ranked leaderboard
cafe-bench leaderboard

# export leaderboard to markdown
cafe-bench leaderboard --export-md
```

---

## Leaderboard

Results are ranked by the primary metric for each task.
Submit your model results via a pull request (see [Contributing](#contributing)).

### Classification — Macro-F1 ↑

| Rank | Model | UEA Avg (30) | PTB-XL | UCI-HAR | Sleep-EDF | CWRU | TEP |
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **HEPA** | — | — | — | — | — | — |
| 🥈 | MOMENT | — | — | — | — | — | — |
| 🥉 | UniTS | — | — | — | — | — | — |
| | PatchTST (supervised) | — | — | — | — | — | — |
| | iTransformer (supervised) | — | — | — | — | — | — |

### Anomaly Detection — VUS-PR ↑

| Rank | Model | SMAP | MSL | SMD | SWaT | PSM | Daphnet | MITDB | NAB | Yahoo-S5 | KPI |
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **HEPA** | — | — | — | — | — | — | — | — | — | — |
| 🥈 | MOMENT | — | — | — | — | — | — | — | — | — | — |
| 🥉 | UniTS | — | — | — | — | — | — | — | — | — | — |
| | DeepSVDD | — | — | — | — | — | — | — | — | — | — |
| | LSTM-AE | — | — | — | — | — | — | — | — | — | — |
| | IsolationForest | — | — | — | — | — | — | — | — | — | — |

### Event Prediction — h-AUROC ↑

| Rank | Model | FD001 | FD002 | FD003 | FD004 | PhysioNet-2012 | PhysioNet-2019 |
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **HEPA** | 0.918 | 0.661 | 0.960 | 0.627 | — | — |
| 🥈 | MOMENT | — | — | — | — | — | — |
| 🥉 | UniTS | — | — | — | — | — | — |
| | DeepHit | — | — | — | — | — | — |
| | CoxPH | — | — | — | — | — | — |

*— = not yet evaluated. Run the model and submit a PR.*

---

## Datasets

### 1 · Classification


**UCR Univariate Archive** (128 datasets) — the most cited time series classification benchmark. Univariate, standardized train/test splits, auto-downloaded via `aeon`.

<details>
<summary>Show all 128 UCR datasets</summary>

| Key | Dataset | Domain | Classes |
|-----|---------|--------|:-------:|
| `ucr_acsf1` | ACSF1 | industrial | 10 |
| `ucr_adiac` | Adiac | image | 37 |
| `ucr_arrowhead` | ArrowHead | image | 3 |
| `ucr_beef` | Beef | spectro | 5 |
| `ucr_beetlefly` | BeetleFly | image | 2 |
| `ucr_birdchicken` | BirdChicken | image | 2 |
| `ucr_bme` | BME | simulated | 3 |
| `ucr_car` | Car | sensor | 4 |
| `ucr_cbf` | CBF | simulated | 3 |
| `ucr_chinatown` | Chinatown | traffic | 2 |
| `ucr_chlorineconcentration` | ChlorineConcentration | water | 3 |
| `ucr_cincecgtorso` | CinCECGTorso | healthcare | 4 |
| `ucr_coffee` | Coffee | spectro | 2 |
| `ucr_computers` | Computers | device | 2 |
| `ucr_cricketx` | CricketX | wearable | 12 |
| `ucr_crickety` | CricketY | wearable | 12 |
| `ucr_cricketz` | CricketZ | wearable | 12 |
| `ucr_crop` | Crop | agriculture | 24 |
| `ucr_earthquakes` | Earthquakes | geophysical | 2 |
| `ucr_ecg200` | ECG200 | healthcare | 2 |
| `ucr_ecg5000` | ECG5000 | healthcare | 5 |
| `ucr_ecgfivedays` | ECGFiveDays | healthcare | 2 |
| `ucr_electricdevices` | ElectricDevices | device | 7 |
| `ucr_ethanollevel` | EthanolLevel | spectro | 4 |
| `ucr_faceall` | FaceAll | image | 14 |
| `ucr_facefour` | FaceFour | image | 4 |
| `ucr_facesucr` | FacesUCR | image | 14 |
| `ucr_fiftywords` | FiftyWords | image | 50 |
| `ucr_fish` | Fish | image | 7 |
| `ucr_forda` | FordA | industrial | 2 |
| `ucr_fordb` | FordB | industrial | 2 |
| `ucr_gunpoint` | GunPoint | motion | 2 |
| `ucr_ham` | Ham | spectro | 2 |
| `ucr_haptics` | Haptics | wearable | 5 |
| `ucr_herring` | Herring | image | 2 |
| `ucr_inlineskate` | InlineSkate | wearable | 7 |
| `ucr_italypowerdemand` | ItalyPowerDemand | energy | 2 |
| `ucr_lightning2` | Lightning2 | weather | 2 |
| `ucr_lightning7` | Lightning7 | weather | 7 |
| `ucr_mallat` | Mallat | simulated | 8 |
| `ucr_meat` | Meat | spectro | 3 |
| `ucr_medicalimages` | MedicalImages | image | 10 |
| `ucr_melbournepedestrian` | MelbournePedestrian | traffic | 10 |
| `ucr_mixedshapesregulartrain` | MixedShapesRegularTrain | image | 5 |
| `ucr_motestrain` | MoteStrain | sensor | 2 |
| `ucr_noninvasivefatalecgthorax1` | NonInvasiveFatalECGThorax1 | healthcare | 42 |
| `ucr_noninvasivefatalecgthorax2` | NonInvasiveFatalECGThorax2 | healthcare | 42 |
| `ucr_oliveoil` | OliveOil | spectro | 4 |
| `ucr_osuleaf` | OSULeaf | image | 6 |
| `ucr_phoneme` | Phoneme | speech | 6 |
| `ucr_plane` | Plane | image | 7 |
| `ucr_powercons` | PowerCons | energy | 2 |
| `ucr_refrigerationdevices` | RefrigerationDevices | device | 3 |
| `ucr_rock` | Rock | geophysical | 4 |
| `ucr_screentype` | ScreenType | device | 3 |
| `ucr_shapeletSim` | ShapeletSim | simulated | 2 |
| `ucr_shapesall` | ShapesAll | image | 60 |
| `ucr_smallkitchenappliances` | SmallKitchenAppliances | device | 3 |
| `ucr_smoothsubspace` | SmoothSubspace | simulated | 3 |
| `ucr_sonyaiborobotsurface1` | SonyAIBORobotSurface1 | robot | 2 |
| `ucr_sonyaiborobotsurface2` | SonyAIBORobotSurface2 | robot | 2 |
| `ucr_starlightcurves` | StarLightCurves | astronomy | 3 |
| `ucr_strawberry` | Strawberry | spectro | 2 |
| `ucr_swedishleaf` | SwedishLeaf | image | 15 |
| `ucr_symbols` | Symbols | image | 6 |
| `ucr_syntheticcontrol` | SyntheticControl | simulated | 6 |
| `ucr_toesegmentation1` | ToeSegmentation1 | wearable | 2 |
| `ucr_toesegmentation2` | ToeSegmentation2 | wearable | 2 |
| `ucr_trace` | Trace | industrial | 4 |
| `ucr_twoleadecg` | TwoLeadECG | healthcare | 2 |
| `ucr_twopatterns` | TwoPatterns | simulated | 4 |
| `ucr_wafer` | Wafer | industrial | 2 |
| `ucr_wine` | Wine | spectro | 2 |
| `ucr_wordsynonyms` | WordSynonyms | image | 25 |
| `ucr_worms` | Worms | biology | 5 |
| `ucr_wormstwocclass` | WormsTwoClass | biology | 2 |
| `ucr_yoga` | Yoga | image | 2 |
| *(+ 50 gesture/wearable variants)* | AllGestureWiimote*, GestureMidAir*, etc. | wearable | varies |

</details>

**UEA Multivariate Archive** (30 datasets) — the standard for multivariate time series classification.

<details>
<summary>Show all 30 UEA datasets</summary>

| Key | Dataset | Domain | Classes | Channels |
|-----|---------|--------|:-------:|:--------:|
| `uea_articularywordrecognition` | ArticularyWordRecognition | speech | 25 | 9 |
| `uea_atrialfibrillation` | AtrialFibrillation | healthcare | 3 | 2 |
| `uea_basicmotions` | BasicMotions | wearable | 4 | 6 |
| `uea_charactertrajectories` | CharacterTrajectories | motion | 20 | 3 |
| `uea_cricket` | Cricket | wearable | 12 | 6 |
| `uea_duckduckgeese` | DuckDuckGeese | audio | 5 | 1345 |
| `uea_eigenworms` | EigenWorms | biology | 5 | 6 |
| `uea_epilepsy` | Epilepsy | healthcare | 4 | 3 |
| `uea_ethanolconcentration` | EthanolConcentration | industrial | 4 | 3 |
| `uea_ering` | ERing | wearable | 6 | 4 |
| `uea_facedetection` | FaceDetection | neuro | 2 | 144 |
| `uea_fingermovements` | FingerMovements | neuro | 2 | 28 |
| `uea_handmovementdirection` | HandMovementDirection | neuro | 4 | 10 |
| `uea_handwriting` | Handwriting | motion | 26 | 3 |
| `uea_heartbeat` | Heartbeat | healthcare | 2 | 61 |
| `uea_insectwingbeat` | InsectWingbeat | audio | 10 | 1 |
| `uea_japanesevowels` | JapaneseVowels | speech | 9 | 12 |
| `uea_libras` | Libras | motion | 15 | 2 |
| `uea_lsst` | LSST | astronomy | 14 | 6 |
| `uea_motorimagery` | MotorImagery | neuro | 2 | 64 |
| `uea_natops` | NATOPS | wearable | 6 | 24 |
| `uea_pendigits` | PenDigits | motion | 10 | 2 |
| `uea_pems-sf` | PEMS-SF | traffic | 7 | 267 |
| `uea_phoneme` | Phoneme | speech | 39 | 11 |
| `uea_racketsports` | RacketSports | wearable | 4 | 6 |
| `uea_selfregulationscp1` | SelfRegulationSCP1 | neuro | 2 | 6 |
| `uea_selfregulationscp2` | SelfRegulationSCP2 | neuro | 2 | 7 |
| `uea_spokenarabicdigits` | SpokenArabicDigits | speech | 10 | 13 |
| `uea_standwalkjump` | StandWalkJump | wearable | 3 | 4 |
| `uea_uwavegesturelibrary` | UWaveGestureLibrary | wearable | 8 | 3 |

</details>

**Additional benchmarks:**

| Key | Dataset | Domain | Classes | Channels | Notes |
|-----|---------|--------|:-------:|:--------:|-------|
| `ptbxl` | PTB-XL | healthcare | 5 | 12 | Most cited ECG benchmark; 21,837 records |
| `uci_har` | UCI-HAR | wearable | 6 | 9 | 10,299 instances; 30 subjects |
| `sleep_edf` | Sleep-EDF | healthcare | 5 | 2 | 30-second EEG/EOG epochs |
| `cwru` | CWRU Bearing | industrial | 10 | 1 | 3 fault locations × 3 severities |
| `tep` | Tennessee Eastman | industrial | 21 | 52 | Gold standard process fault classification |

---

### 2 · Anomaly Detection

Primary metric: **VUS-PR** (Volume Under the PR Surface) — the [TSB-AD 2024](https://github.com/TheDatumOrg/TSB-AD) standard. Robust to point-adjust gaming.

**TSB-AD-M** (NeurIPS 2024 benchmark, active leaderboard):

| Key | Dataset | Domain | Channels | Anomaly % |
|-----|---------|--------|:--------:|:---------:|
| `tsb_ad_smap` | SMAP | aerospace | 25 | 13.1% |
| `tsb_ad_msl` | MSL | aerospace | 55 | 10.7% |
| `tsb_ad_smd` | SMD | IT | 38 | 4.2% |
| `tsb_ad_swat` | SWaT | ICS | 51 | 11.9% |
| `tsb_ad_psm` | PSM | IT | 25 | 27.8% |
| `tsb_ad_exathlon` | Exathlon | IT | 19 | varies |
| `tsb_ad_daphnet` | Daphnet | wearable | 9 | 19.0% |
| `tsb_ad_mitdb` | MITDB | healthcare | 2 | 0.4% |
| `tsb_ad_catsv2` | CATSv2 | industrial | 1 | varies |
| `tsb_ad_tao` | TAO | climate | 1 | varies |
| `tsb_ad_ghl` | GHL | ICS | 1 | varies |
| `tsb_ad_genesis` | Genesis | industrial | 18 | varies |
| `tsb_ad_opp` | OPP | wearable | 77 | varies |

**Additional anomaly benchmarks:**

| Key | Dataset | Domain | # Series | Notes |
|-----|---------|--------|:--------:|-------|
| `nab` | NAB | mixed | 58 | Most widely used univariate streaming benchmark |
| `yahoo_s5` | Yahoo S5 | web | 367 | Real + synthetic; requires Webscope license |
| `kpi` | KPI | IT | 58 | Alibaba production microservice KPIs |

---

### 3 · Event Prediction

Primary metric: **h-AUROC** — AUROC integrated across all prediction horizons Δt ∈ {1, …, H}.

**Prognostics (RUL):**

| Key | Dataset | Domain | Channels | Horizon | Notes |
|-----|---------|--------|:--------:|:-------:|-------|
| `cmapss_fd001` | C-MAPSS FD001 | aerospace | 14 | 125 cycles | 1 op condition, 1 fault mode |
| `cmapss_fd002` | C-MAPSS FD002 | aerospace | 14 | 125 cycles | 6 op conditions, 1 fault mode |
| `cmapss_fd003` | C-MAPSS FD003 | aerospace | 14 | 125 cycles | 1 op condition, 2 fault modes |
| `cmapss_fd004` | C-MAPSS FD004 | aerospace | 14 | 125 cycles | 6 op conditions, 2 fault modes |

**Clinical event prediction:**

| Key | Dataset | Domain | Channels | Horizon | Notes |
|-----|---------|--------|:--------:|:-------:|-------|
| `physionet2012` | PhysioNet 2012 | healthcare | 37 | 48 h | ICU 28-day mortality; 12,000 stays |
| `physionet2019` | PhysioNet 2019 | healthcare | 40 | 72 h | Sepsis onset (Sepsis-3, 6h lead) |

---

## Metrics

| Task | Metric | Why |
|------|--------|-----|
| Classification | **Macro-F1** | Class-balanced; standard for imbalanced multiclass |
| Anomaly Detection | **VUS-PR** | Integrates PR-AUC over buffer sizes; immune to point-adjust inflation |
| Event Prediction | **h-AUROC** | Evaluates discrimination at every horizon; penalizes late predictions |

---

## Baselines

### Model registry keys

| Key | Model | Checkpoint | Classification | Anomaly | Event |
|-----|-------|-----------|:---:|:---:|:---:|
| `hepa` | **HEPA** (ours) | local `.pt` | ✓ | ✓ | ✓ |
| `moment` | MOMENT-1-large | `AutonLab/MOMENT-1-large` | ✓ | ✓ | ✓ |
| `units` | UniTS | `mims-harvard/UniTS-supervised-m` | ✓ | ✓ | ✓ |
| `units_ft` | UniTS (supervised) | `mims-harvard/UniTS-supervised-m` | ✓ | ✓ | ✓ |

### Task coverage

| Model | Type | Classification | Anomaly | Event |
|-------|------|:---:|:---:|:---:|
| **HEPA** | Foundation encoder (ours) | ✓ | ✓ | ✓ |
| MOMENT | Foundation encoder | ✓ | ✓ | ✓ |
| UniTS | Foundation encoder | ✓ | ✓ | ✓ |
| PatchTST | Supervised Transformer | ✓ | — | — |
| iTransformer | Supervised Transformer | ✓ | — | — |
| DeepSVDD | Anomaly detection | — | ✓ | — |
| LSTM-AE | Anomaly detection | — | ✓ | — |
| IsolationForest | Anomaly detection | — | ✓ | — |
| DeepHit | Survival model | — | — | ✓ |
| CoxPH | Survival model | — | — | ✓ |

> **Note:** Forecasting-only models (Chronos, Moirai, TimesFM) are excluded — they have no classification head or reconstruction path and cannot be fairly compared on these tasks.

---

## Data Setup

### Step 1 — Auto-downloaded datasets

Run the master download script to fetch all datasets that don't require credentials:

```bash
python scripts/download_all.py --data-root data/
```

This downloads and prepares: **NAB**, **UCI-HAR**, **PTB-XL**, **TSB-AD**, **KPI**, **Sleep-EDF**, **C-MAPSS**.

UCR/UEA datasets are fetched automatically by `aeon` on first use — no action needed.

### Step 2 — HuggingFace datasets

CWRU (bearing faults) and PhysioNet clinical datasets are mirrored on HuggingFace:

```bash
python scripts/download_from_hf.py --data-root data/ --datasets cwru physionet2012 physionet2019
```

### Step 3 — Datasets requiring a free account

| Dataset | Where to download | Prepare step |
|---------|------------------|-------------|
| **PTB-XL** | [physionet.org/content/ptb-xl](https://physionet.org/content/ptb-xl/) | Place files under `data/ptbxl/` — no prepare script needed |
| **PhysioNet 2012** | [physionet.org/content/challenge-2012](https://physionet.org/content/challenge-2012/1.0.0/) | `python scripts/prepare_physionet2012.py --data-root data/` |
| **PhysioNet 2019** | [physionet.org/content/challenge-2019](https://physionet.org/content/challenge-2019/1.0.0/) | `python scripts/prepare_physionet2019.py --data-root data/` |

### Step 4 — Licensed datasets

| Dataset | Access | How to obtain |
|---------|--------|--------------|
| **Yahoo S5** | Webscope license (free for research) | [webscope.sandbox.yahoo.com](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s) — place CSVs under `data/yahoo_s5/` |

### Step 5 — TEP (optional, larger version)

The Tennessee Eastman Process dataset auto-downloads a public version via `prepare_tep.py`. For the full Rieth 2017 variant (500 runs per fault, Harvard Dataverse):

```bash
python scripts/prepare_tep.py --data-root data/
```

### Expected directory layout

```
data/
  Multivariate_ts/        ← UEA (auto-populated by aeon)
  UCRArchive_2018/        ← UCR (auto-populated by aeon)
  cmapss/
    train_FD001.txt  test_FD001.txt  RUL_FD001.txt  ...
  tsb_ad/
    SMAP/  MSL/  SMD/  SWaT/  PSM/  ...
  nab/
    data/  labels/
  physionet2012/
    physionet2012_prepared.npz
  physionet2019/
    physionet2019_prepared.npz
  ptbxl/
    ptbxl_database.csv  scp_statements.csv  records100/  ...
  uci_har/
    UCI HAR Dataset/
  sleep_edf/
    sleep_edf_prepared.npz
  cwru/
    cwru_prepared.npz
  tep/
    tep_prepared.npz
  kpi/
    train.csv  test.csv
  yahoo_s5/
    *.csv
```

---

## Adding Your Model

Implement `encode()` and register — everything else is handled automatically:

```python
# my_model.py
import numpy as np
from cafe_bench.models.base import BaseModel
from cafe_bench.registry import register_model

class MyModel(BaseModel):
    name = "my_model"

    def encode(self, X: np.ndarray) -> np.ndarray:
        # X: (N, T, C)  →  return (N, D) embeddings
        ...

register_model("my_model", MyModel)
```

The default probes are inherited:
- **Classification** → `sklearn.LogisticRegression` on frozen embeddings
- **Anomaly** → kNN distance in embedding space
- **Event** → `lifelines.CoxPHFitter` on frozen embeddings

Override any probe if your model has a task-specific head:

```python
class MyModel(BaseModel):
    name = "my_model"

    def encode(self, X):       ...   # required
    def classify(self, ...):   ...   # optional override
    def anomaly_score(self, ...): ... # optional override
    def event_predict(self, ...): ... # optional override
```

Then run:

```bash
cafe-bench run uea_epilepsy my_model --data-root data/
```

---

## Repository Structure

```
cafe-bench/
├── cafe_bench/
│   ├── datasets/
│   │   ├── base.py                   BaseDataset, batch dataclasses
│   │   ├── classification/
│   │   │   ├── uea.py                All 30 UEA datasets (auto-registered)
│   │   │   └── extras.py             PTB-XL, UCI-HAR, Sleep-EDF, CWRU, TEP
│   │   ├── anomaly/
│   │   │   ├── tsb_ad.py             TSB-AD-M (13 datasets)
│   │   │   └── others.py             NAB, Yahoo-S5, KPI
│   │   └── event/
│   │       ├── cmapss.py             C-MAPSS FD001–FD004
│   │       └── physionet.py          PhysioNet 2012 & 2019
│   ├── models/
│   │   ├── base.py                   BaseModel + default probes
│   │   ├── hepa_model.py             HEPA frozen encoder
│   │   └── baselines/
│   │       ├── moment_model.py       MOMENT (embedding + reconstruction)
│   │       └── units_model.py        UniTS (classify + detect_anomaly)
│   ├── evaluators/
│   │   ├── classification.py         accuracy, macro-F1, weighted-F1
│   │   ├── anomaly.py                VUS-PR, AUROC, AUPRC
│   │   └── event.py                  h-AUROC (per-horizon + integrated)
│   ├── pipeline.py                   dataset → model → evaluator → JSON
│   ├── leaderboard.py                aggregate JSONs → ranked table
│   ├── registry.py                   dataset & model registration
│   └── __main__.py                   CLI (cafe-bench)
├── scripts/
│   ├── download_all.py               master download script (no-credentials datasets)
│   ├── download_from_hf.py           HuggingFace downloader (CWRU, PhysioNet)
│   ├── download_data.py              per-dataset download helper
│   ├── prepare_cwru.py               CWRU bearing fault preprocessing
│   ├── prepare_sleep_edf.py          Sleep-EDF EDF → npz conversion
│   ├── prepare_tep.py                TEP download + preprocessing
│   ├── prepare_physionet2012.py      PhysioNet 2012 → npz conversion
│   └── prepare_physionet2019.py      PhysioNet 2019 → npz conversion
├── results/                          auto-populated by runs
└── pyproject.toml
```

---

## Contributing

We welcome model submissions and dataset additions.

**To submit your model results:**
1. Fork the repo
2. Add your model wrapper under `cafe_bench/models/`
3. Run the full benchmark: `cafe-bench run-all your_model --data-root data/`
4. Open a PR with your `results/your_model/*.json` files

**To add a dataset:**
1. Implement `BaseDataset` (see [Adding a Dataset](docs/adding_dataset.md))
2. Register it with `register_dataset(key, instance)`
3. Open a PR

---

## Citation

If you use CAFE in your research, please cite:

```bibtex
@article{petersen2025hepa,
  title   = {HEPA: Self-Supervised Physical Encoding Enables
             System Understanding, Anomaly Detection, and Event Prediction},
  author  = {Petersen, Jonas and Lombardi, Gian-Alessandro and
             Maggioni, Riccardo and Mazzoleni, Camilla and
             Martelli, Federico and Petersen, Philipp},
  journal = {arXiv preprint arXiv:2506.XXXXX},
  year    = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">
<sub>Built by <a href="https://forgislabs.ai">Forgis Labs</a></sub>
</div>
