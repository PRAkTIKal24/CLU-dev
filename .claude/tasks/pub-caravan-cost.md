# Task: pub-caravan-cost — what the ungauged venue would actually cost to run, on hardware that actually exists

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/pub-caravan-cost.md`
- **Read first:** protocol **§5** · `.claude/outputs/camels-caravan-refs.md` (**the frozen reference set + the re-train pricing**) · `.claude/data/c3-camels/staged/MANIFEST.json` · `.claude/outputs/c3-trackb-tripwire/code/build_camels.py` (the CAMELS loader you are pricing a delta against) · `.claude/sample_script_csf3.sh`
- **Repo:** read-only on `chlu/`. **Zero worktrees, zero branches, zero tracked-code edits.**
- **Wave:** ICLR-long W1, spoke 3 of 3. Runs **in parallel with** `camels-pub-tripwire`; the two are file-disjoint.

## ⭐ DIAL DECLARATION (protocol §7 — missing from v1, Hub error, supplied 2026-09-07)
- **Dial:** *none — feasibility / cost recon.* ⛔ No CLU number of any kind.
- **Laundering control:** n/a (no performance measurement). Every estimate is stated against the hardware
  it assumes; an estimate with no named hardware is not an estimate.
- **Falsifies:** a measured reference-LSTM cost differing from your extrapolation by >2×.
- **Does NOT falsify:** the venue being expensive. Cost is an input to the Head's decision, not a verdict.

## ⛔ PRECONDITION — mechanical
Start **only** when `.claude/outputs/camels-caravan-refs.md` exists **and** contains the literal string `## REFERENCE SET (FROZEN)`.

## The one job
Produce the **cost table** that tells the Head whether the ungauged venue is affordable: data volume, staging effort, loader delta, and **GPU-hours for a full cell** — priced on hardware we can actually reach today.

## ⚠ Read this before you estimate anything — the cluster assumption is broken
The Hub probed this on 2026-09-07 from the primary workstation. DNS resolves normally (`github.com` fine), and:
- `csf3.itservices.manchester.ac.uk` → **NXDOMAIN. The hostname no longer exists.**
- `login01-csf3-test.itservices.manchester.ac.uk` → **NXDOMAIN.**
- `csf4.itservices.manchester.ac.uk` → resolves to **10.99.203.215** (RFC-1918 private), and SSH to it **times out** from this network.

⇒ **CSF3 as configured is unreachable and appears retired; CSF4 exists in DNS but needs VPN or on-campus network.** Every "cluster credits are effectively unlimited, jobs cap at 2 GPUs for 4 days" assumption is **unverified until you re-establish access.**

**Item 0, and it blocks the rest:** establish what compute is *actually* available — is CSF4 the successor, is there a new hostname, does VPN restore access, what are its real queue limits, GPU types, walltime cap and per-job GPU cap. **Re-price everything on what you find.** If you cannot reach any cluster, say so plainly and price the venue on (a) the workstation and (b) a named commercial GPU rate, and mark the cluster column **NOT ESTABLISHED** — ⛔ never carry the old CSF3 limits forward as if verified.

## Items
1. **Item 0 above.** Report reachability as a fact with the command and its output, not as an impression.
2. **Data volume and staging effort**, for CAMELS-PUB and — only if the refs spoke reported `caravan_protocol: USABLE` — for Caravan: download size, staged size, wall-clock to stage, licence, checksum discipline. CAMELS is **already staged** (7.2 GB raw, 93 MB staged, sha256'd) — price only the **delta**.
3. **Loader delta** over the already-specified CAMELS loader (`build_camels.py`, 157 lines) for each option: what has to change for a **basin-held-out k-fold** split versus the existing temporal split, in files-touched and rough effort. For Caravan, price a new loader honestly.
4. **Measured cost per training run for a reference LSTM at this scale.** Measured, on hardware you can reach, at the reference's published size — not extrapolated from a paper's hardware section. If the refs spoke says a re-train is required, this measurement is what prices it. ⛔ **Do not launch a full reference re-train — measure a short calibrated run and extrapolate, and report the extrapolation factor.** A full re-train is a separate funding decision that comes back to the Hub.
5. **GPU-hours for a full cell:** `k folds × ≥3 seeds × ~5 arms`, **plus an ensemble top-up matching the reference's published seed count** (an ensemble reference must be met with an ensemble, or the comparison is rigged in our favour). Show the arithmetic.
6. **A cost table** with a row per option (CAMELS-PUB / Caravan) and columns: staging hours · loader effort · GPU-hours per cell · wall-clock at the real queue limits · what it displaces.

## Acceptance
- Item 0 answered with evidence; every downstream estimate stated against the hardware it assumes.
- Every number traces to a file or a command transcript on disk under `.claude/outputs/pub-caravan-cost/`.
- The extrapolation factor from the calibrated run to the full run is stated explicitly, with its assumptions.
- **Anything not run or not reachable is reported as NOT RUN / NOT ESTABLISHED — never as a zero.**

## What would prove this spoke wrong
- The reference LSTM turning out to be far cheaper or far dearer than the published hardware section implies — a 5× miss here changes the recommendation, so state the confidence on the extrapolation.
- Cluster access being restored trivially (a VPN toggle), which would make a pessimistic commercial-rate pricing misleading. Chase that before pricing pessimistically.

## Constraints
- ⛔ No sub-agents. ⛔ No CLU arm, no CLU number. ⛔ No model training beyond the short calibrated reference run in item 4.
- Under ~3 pages. The cost table is the deliverable; prose is support.

---

## ⭐ AMENDMENTS FOLDED IN AT SCOUT REVIEW — 2026-09-07. Spawn THIS file.

**B1 — Caravan is OUT of scope. `caravan_protocol: NONE`.** It has no split, no metric convention and no
published baselines, so it cannot carry a SOTA claim and there is nothing to cost. Do **not** price a
Caravan loader or its 39.3 GB download. Record one line: not a venue, not costed, and why.
⇒ **This task is now CAMELS-US PUB only.** Its title is legacy; ignore the "caravan" in the slug.

**B2 — The re-train question is ANSWERED: no reference re-train is required.** Two references ship
per-basin artefacts, so the frontier and the 2019 baseline are both reproducible without training:
- Li 2025 (the 0.785833 frontier): per-basin **daily simulations**, Zenodo 10.5281/zenodo.16895228.
- Kratzert 2019 (0.69 / 0.64 / 0.58): per-basin **metrics** CSVs — already downloaded, checksummed and
  **recomputed by the Hub** (0.6931 / 0.6447 / 0.5766, all 531 rows) in `.claude/scratch/hub-refs-verify/`.
⇒ **Item 4's calibrated reference-LSTM run is now optional, not blocking.** Run it only to price *our own*
future arms; do not treat it as needed to obtain a reference number. This removes the largest cost item.

**B3 — Price the cell against the published training cost, which is now sourced.** Li et al. 2025, verbatim:
*"training LSTM and δHBV for temporal testing under a single meteorological forcing dataset takes
approximately **5** and **21** hours, respectively, using a single **NVIDIA Tesla V100** GPU."*
Their full PUB row is (3 LSTM forcings + 3 δHBV + 1 LSTM-multi) × 10 rounds × 3 seeds = **210 trainings**
≈ **2,490 V100-GPU-hours ≈ 104 GPU-days**. ⚠ Those hours are quoted for the *temporal* test; PUB trains on
9/10 of basins over 1980–1999, so treat as same-order, not exact. Use this to sanity-check your own
extrapolation — if yours disagrees by >2×, say so and explain which you trust.
Kratzert 2019's setup is fully specified (hidden 256, 30 epochs, batch 2000, seq_len 270, dropout 0.4,
lr 1e-3, ≈0.30 M params) but ⛔ **its wall-clock and hardware are NOT-FOUND** — report as NOT-FOUND.

**B4 — Loader delta is now concrete.** The split to implement is **Li 2025 PUB**: 531 basins, 10 random
basin groups × 10 rounds, every basin held out exactly once, test concatenated across rounds before
metrics, **Daymet only**, unclipped NSE, median over 531. Price the delta over `build_camels.py` for
**basin-held-out k-fold** — note the existing loader already reads the 531-basin list and already
restricts to Daymet, so the delta is the fold machinery and the concatenation, not the ingest.

**B5 — Add one staging row the original task did not have:** Li 2025's Zenodo artefacts
(`py999_PUB_simulations.npy` 531 MB + `py999_OBS.npy` 43.4 MB + `py999_ids.npy`). Small, one-off,
CC BY 4.0, and required by the tripwire spoke's positive control. Cost the download and record checksums.
⚠ Coordinate: **the tripwire spoke owns that download.** If it has already landed it, checksum and reuse —
⛔ do not download it twice and do not write into that spoke's directories.

---

## ⭐ HEAD RULING — 2026-09-07: FEASIBILITY TESTING IS **LOCAL ONLY**. This supersedes the cluster section.

**CSF3 is the Head's personal allocation and its queue times are high.** Feasibility work therefore runs
**on this workstation, and nowhere else.**

**C1 — ⛔ Item 0 is CANCELLED. Do not probe, ssh to, or submit to any cluster.** Do not attempt to
establish access, do not test VPN, do not touch the Head's account. Delete the reachability question from
your plan. The earlier "CSF3 unreachable / NXDOMAIN" note in this file is **superseded and is not yours to
chase** — it is recorded for the Head only.

**C2 — Your measurement hardware, which is the only hardware you may quote as measured:**
Apple **M1 Pro** · **8 CPU cores** (6 performance + 2 efficiency) · **14-core GPU (Metal, no CUDA)** ·
**16 GB unified RAM** · **82 GB free disk** (460 GB volume, 81% used). CAMELS raw already occupies 7.2 GB
of that.

**C3 — ⛔ Do NOT benchmark a reference LSTM locally and call it a cost estimate.** There is no CUDA GPU
here; an M1 Pro timing is not a V100 proxy and dressing one up as one would be a fabricated number.
Item 4 is **cancelled as a measurement**. Replace it with two clearly separated things:
- **MEASURED (local, real):** the wall-clock, peak RSS and disk footprint of the **tripwire/store arms** —
  that is what actually runs here, and the tripwire spoke is running it now. Cost it honestly.
- **PROJECTED (arithmetic, labelled as such):** GPU-hours for any future trained arm, computed from
  **Li et al. 2025's published V100 figures** (LSTM ≈ 5 h, δHBV ≈ 21 h per single-forcing training) times
  the cell size. ⭐ **Label every such row `PROJECTION (V100-h, from published timings)`.** ⛔ Never present a
  projection as a measurement, and never blend the two in one column.

**C4 — Report GPU-hours, not calendar time.** Queue latency on CSF3 is high and is the **Head's** variable,
not yours. Give the resource number; ⛔ do not estimate wall-clock-to-result or "days to finish."

**C5 — Memory is the binding local constraint, not compute.** 16 GB total. State the peak RSS of every
arm you cost, and name the largest cell that fits in RAM. If something does not fit, report it as
**NOT RUN — exceeds local RAM**, never as a failure and never silently subsampled.

**C6 — Disk.** 82 GB free. CAMELS is staged (7.2 GB raw / 93 MB derived). The only new download is Li 2025's
Zenodo bundle (~575 MB). ⚠ Caravan's 39.3 GB is moot — it is not a venue and is not being staged.
