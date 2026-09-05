# Task: voraus-baseline-floors — establish the numbers CLU must beat (w15, analyst, CSF3-ready)

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/voraus-baseline-floors.md`
- **Read first:** protocol (**§5 flag-provenance + pre-registration**) · `.claude/outputs/f2-eval-harness.md` (the binding eval rules + the SKAB/SMD floors already measured: SKAB VUS-PR knn 0.873 / lof 0.868 / pca 0.580 / iforest 0.522; SMD negative-control pca 0.465) · `chlu/eval/harness.py`, `chlu/eval/baselines.py` · `chlu/data/industrial/{voraus_ad,tep_rieth}.py` · `scripts/csf3/README.md` (§5 smoke checklist; run under `~/scratch`).
- **Repo:** read-only on `chlu/`; writes analysis + `.claude/outputs/`.
- **This is CSF3-ready NOW** — it needs **no CLU scorer**, only the statistical baselines that already exist. It is a legitimate first-day CSF3 run and it produces numbers the flagship needs regardless.

## Why
Before any CLU-vs-baseline claim can mean anything, we need the **baseline floors on the flagship datasets**, measured under our own harness (VUS-PR primary, **point-adjust forbidden**, unit-level splits, statistical baselines mandatory). voraus-AD is the G7b headline set; TEP is the scale/CC0 set. We have floors for SKAB and SMD; we have **none for voraus or TEP**.

## Items
1. **Pre-register** (`PREREG.md`) the expected baseline ranking and rough VUS-PR band from the SKAB/SMD priors and the voraus-AD paper's own reported numbers, before running. State what would surprise you.
2. **Baseline floors on voraus-AD** (100 Hz, the sha-pinned variant): all four statistical baselines (PCA/IForest/LOF/KNN) through `evaluate_dataset`, **unit = one pick-and-place episode**, episode-level labels (the loader exposes the 12 anomaly categories). Report **per-category** VUS-PR where the harness supports it — voraus's collisions/load-changes/etc. are exactly the "geometric vs control" structure the flagship will exploit, so a per-category baseline map is high-value.
3. **Baseline floors on TEP-Rieth** (the scale set): same protocol, report the floors and the compute cost (this is the set that will stress the CSF3 budget later).
4. **Sanity vs the literature:** confirm our VUS-PR numbers are in the voraus-AD paper's ballpark for comparable methods; if wildly off, the loader/splits have a bug — **flag it, do not paper over it** (this is exactly the kind of thing that must be caught before the flagship, not after).
5. **Report the memory/time envelope** per dataset (voraus is ~2.5 GB RAM loaded, 2122 episodes) so the CSF3 job for the CLU run can be sized correctly (`-G`, `-t`, host-core count).
6. **The negative-control discipline:** if any dataset is meant as a negative control (SMD was), state its expected-low floor and confirm it.

## Environment
- Run on CSF3 under `~/scratch` (the env smoke in README §5 must pass first). voraus re-downloads on a **compute node** (login nodes have no internet) via `VorausAD(download=True)`, **or** rsync the already-verified local parquet (`.claude/scratch/f2-eval-harness/data/voraus_ad/…100hz.parquet`) up with `push_repo.sh`-style transfer. Record which, and the sha.
- If voraus fits in laptop RAM for a subset, a `--limit`ed laptop pre-smoke is fine to de-risk before the CSF3 submit — say so.

## Acceptance
Pre-registered expectations; voraus-AD + TEP baseline floors under the binding harness rules (VUS-PR, no point-adjust, unit-level, per-category where available); a literature sanity check with any discrepancy flagged; the memory/time envelope for sizing the CLU run; PREREG + flag-provenance per §5. **These floors become the reference line for every CLU-vs-baseline claim in the ICLR long — get them right and get them honest.**
