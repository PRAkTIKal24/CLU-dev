# Task: f2-eval-harness — the dataset-agnostic industrial-eval harness (F2+F3 build)

- **Agent:** `experiment-engineer` · **Base:** `main` · **Branch:** `agent/experiment-engineer/f2-eval-harness` · **Output:** `.claude/outputs/f2-eval-harness.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md`, **`.claude/outputs/scout-industrial-datasets.md` (the dataset decisions + BINDING eval rules — authoritative)**, roadmap "Wave-1 scout results" block, handover §6/§7 ops notes.

**Goal:** the evaluation backbone the V3/ICLR chapters run on. Loaders + metrics + baselines + splits, dataset-agnostic so FactoryWave (Forgis bonus) can drop in later. NO CLU modeling in this task — pure harness.

## Binding rules (from the scout report; violating these is a task failure)
- **VUS-PR primary** metric; range-aware/event metrics + AUROC/AUPR secondary; **point-adjust F1 is FORBIDDEN** everywhere in the harness (do not even implement it, so nobody can quietly use it).
- **Wrap the Apache-2.0 TSB-AD harness** for metrics — do not reimplement VUS-PR.
- **Statistical baselines mandatory:** PCA-recon, IsolationForest, LOF, KNN (sklearn / TSB-AD implementations) wired into every eval run by default.
- **Leakage-safe splits:** split by physical unit/run (bearing, robot run, simulation seed), never by window. Split logic must be explicit + tested.

## Build
1. **Loader interface** (e.g. `chlu/data/industrial/`): a small `IndustrialDataset` protocol (channels, sampling rate, unit-ids for splits, labels [anomaly spans / fault class], metadata) + implementations:
   - **voraus-AD** (100 Hz variant first; ~1.1 GB download; CC BY-NC-SA — note license in the module docstring),
   - **SKAB** (tiny; GPL-3.0 — LOAD from user-downloaded path, do NOT vendor/redistribute files in-repo; license note in docstring),
   - **TEP-Rieth** (Harvard Dataverse CC0, DOI 10.7910/DVN/6C3JR1),
   - **SMD negative control via TSB-AD's curated version**.
   Downloads: implement fetch-or-point-at-path with checksums where feasible; document manual steps where auth/size demands.
2. **Metrics module:** TSB-AD wrapper (VUS-PR + secondaries) + per-dataset reporting (never a single grand mean); results to `results/*.npz` + a markdown table emitter (paper-ready).
3. **Baseline runners:** the four statistical baselines end-to-end on each loader (windowing config explicit); this doubles as the loader integration test.
4. **Split utilities:** unit-level train/test splitting + cross-operating-condition option (Paderborn-style, for later datasets).
5. **Stretch (only if smooth):** MIMII acoustic loader skeleton (Zenodo, CC BY-SA; ~100 GB — metadata/subset handling only, no full download on this laptop; the full run happens on CSF3).

## Verify
Run all four statistical baselines on **SKAB** (tiny) end-to-end and report the actual VUS-PR numbers vs the repo leaderboard's F1 context; loader smoke tests for voraus-AD/TEP on a subset (document download sizes/times). pytest for split logic + metric wrapper (compare one VUS-PR value against TSB-AD's own reference output on its sample data).

**Scope guards:** no CLU models here; no new metrics inventions; keep loaders lazy/streaming where files are big. License hygiene per above. Record every download URL + checksum in the output file.
