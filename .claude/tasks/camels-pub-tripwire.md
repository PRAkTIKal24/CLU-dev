# Task: camels-pub-tripwire — the matched-byte tripwire on the CAMELS-US ungauged (PUB) split

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/camels-pub-tripwire.md` + artifacts under `.claude/outputs/camels-pub-tripwire/`
- **Read first:** protocol **§5 (pre-registration + flag-provenance)** · `.claude/outputs/camels-caravan-refs.md` (**the frozen reference set — your thresholds come from there, not from memory**) · `.claude/outputs/c3-trackb-tripwire/PREREG.md` §1–§2 (the budget and the ladder convention you inherit) · `.claude/outputs/c3-trackb-tripwire/code/{build_camels.py,camels_tripwire.py,camels_local.py,analyse.py}`
- **Repo:** read-only on `chlu/`. **Zero worktrees, zero branches, zero tracked-code edits.** CPU-only, hours not days.
- **Wave:** ICLR-long W1, spoke 2 of 3.

## ⭐ DIAL DECLARATION (protocol §7 — was missing from v1 of this file, Hub error, supplied 2026-09-07)
- **Dial:** *none — venue admissibility / instrument.* ⛔ No CLU cell, store, byte dividend or performance
  number is computed or implied. You are measuring the **classical** competition only.
- **Laundering control:** the shuffled-order null leg (§5) plus the companion rows — a store arm that does
  no better than day-of-year climatology is measuring the calendar, not the catchment.
- **Falsifies (fires the tripwire):** best in-budget store arm within 0.02 abs / 2% rel of **any** reference.
- **Does NOT falsify:** the store losing badly to the LSTM references. That is the metric-native-ceiling
  theorem, is expected, and is not news. ⛔ Nor does a *fired* tripwire falsify the CLU — it falsifies the
  **venue**.

## ⛔ PRECONDITION — mechanical, do not start without it
Start **only** when `.claude/outputs/camels-caravan-refs.md` exists **and** contains the literal string `## REFERENCE SET (FROZEN)`. If the file is absent or the string is missing, **stop and report to the Hub**. Do not reconstruct the reference set yourself and do not proceed on remembered numbers.

## The one job
Run the matched-byte tripwire on the **CAMELS-US ungauged/PUB** split and return a fired / did-not-fire verdict with the margin against **every** reference in the frozen set.

⛔ **No CLU arm, no CLU number, no store-of-ours number, no model training.** This measures the *classical* competition only.

## Why the gauged verdict does not carry
Gauged cleared at 0.256 vs 0.758. The PUB task's classical competition is **donor-basin regionalization — a similarity lookup over static catchment attributes**, which is structurally the same object as the kNN-over-statics arm below. This is the arm most likely to fire. **Build the thing that can kill the venue before you build anything else.**

## Order of work — this order is binding

### 1. PREREG first, before any tripwire number exists
Write `.claude/outputs/camels-pub-tripwire/PREREG.md` containing:
- The **frozen reference set copied verbatim** from the refs spoke's output, with the primary threshold named.
- The full arm list (§3 below), the L-ladder, and the byte budget.
- The **kill rule verbatim** (§5).
- **Per-item numeric point predictions with intervals** — one per arm family, plus a prediction for `E` (the best in-budget arm) and for `margin_abs` against the primary reference. State what would surprise you.
- **§0 Disclosure:** anything computed before the prereg was filed, including anything you learned from the gauged run. Repo HEAD and `git status --porcelain` at filing.

### 2. Loader positive control — before any tripwire number is quoted
Recompute a **published median NSE on the PUB protocol** through our own loader and require it within a **pre-registered tolerance** (state the tolerance in the prereg, before you run it). If the control misses tolerance, **the loader is wrong — stop and report; do not proceed to tripwire numbers.**
- ⚠ **Known trap, already defused in `build_camels.py`:** the shipped **maurer** forcing product is defective — `tmax == tmin` on >99% of days. Use **daymet only**, keep the `tmax > tmin` assertion, and additionally **assert that the assertion fires on maurer** — an untested guard is not a guard.
- Staged data already exists and is checksummed: `.claude/data/c3-camels/staged/{forcing_daymet,discharge,attributes}.npz` (531 basins) with `MANIFEST.json`. Verify the sha256s before use.

### 3. Store arms — maximum consumed over all of them
- **kNN over static catchment attributes** ⭐ *the donor-basin analogue, the dangerous arm*
- kNN over forcing-window descriptors
- **both combined**
- targets **raw** and **standardised**
- `k ∈ {1, 3, 5, 10, 25}`
- **distance-weighted** and **uniform**
- exemplar selection: **random** and **k-means-condensed**

### 4. The ladder and the budget
- Inherit budget **B = 1,966,080 B** (the program's Track-A budget, as used in the gauged run).
- Additionally, if the frozen reference set gives the PUB reference model's **parameter count**, report a second ladder point at a **parameter-matched** budget. ⚠ **If the two budgets differ materially, report both and escalate the choice to the Hub — do not silently pick one.**
- Full **L-ladder to the budget and beyond**. **In-budget arms decide the verdict**; over-budget arms are reported as an **unbounded upper bound** and are never admissible.
- **Seeded arms (k-means init, random selection) get ≥ 3 seeds**, reported with spread.

### 5. Companion rows
Per-basin **day-of-year climatology** · **spatial-nearest-neighbour donor** · **regional mean flow** · **any published regionalization baseline you can source** from the refs spoke's output. Plus a **shuffled-order null leg**.

### 6. The verdict
Write `VERDICT.json` with, for **every** reference in the frozen set: `S`, `margin_abs`, `margin_rel`, `fires` (bool). Plus the overall `dies` bool.

## KILL RULE — pre-registered, not negotiable
> **The venue DIES if the best in-budget store arm comes within 0.02 absolute or 2% relative of the best strong reference.** Evaluated and reported against **EVERY** reference in the frozen set, not just the strongest.

⚠ **A fired tripwire falsifies the VENUE, not the CLU.** If it fires: **stop, write up what you have, and escalate to the Head and the Main Advisor.** Do not soften it, do not hunt for a framing that saves the venue, and do not escalate to another spoke.

## Acceptance
- PREREG filed **before** the first tripwire number, with §0 disclosure.
- Loader positive control **passed within pre-registered tolerance**, reported with the published number it reproduces.
- Every arm in §3 run across the full ladder; maximum consumed; `VERDICT.json` covering every reference.
- Every quoted number traces to a file on disk under `.claude/outputs/camels-pub-tripwire/`. ⛔ No number that exists only in the report text.
- **Anything not run is reported as NOT RUN — never as a zero, never as a failure, never silently dropped.**

## What would prove this spoke wrong
- The kNN-over-statics arm scoring far **below** the published regionalization baselines — that would mean our store arm is **hobbled**, not that the venue is safe. A weak dangerous-arm result must be treated as a suspected bug and chased, not banked as good news.
- The loader control passing on gauged but the PUB fold assignment being reconstructed rather than published — margins then aren't against the real protocol. Say so if it happens.

## File ownership
- **Owns:** `.claude/outputs/camels-pub-tripwire/**`, `.claude/scratch/camels-pub-tripwire/**`.
- **Read-only:** `.claude/data/c3-camels/**`, `.claude/outputs/c3-trackb-tripwire/**` (copy code into your own scratch; ⛔ do not edit it in place), `.claude/outputs/camels-caravan-refs.md`.
- ⛔ No sub-agents. ⛔ Caravan is **not** in this spoke's scope even if the refs spoke says a protocol exists — that is a separate spoke the Hub will scope.

---

## ⭐ AMENDMENTS FOLDED IN AT SCOUT REVIEW — 2026-09-07. Spawn THIS file, not a remembered version.

The reference audit landed and the Hub verified it independently. Six things changed. Read all six
before you write the prereg.

**A1 — The protocol is PINNED. There is no canonical PUB protocol; three incompatible ones are live.**
- **Verdict protocol (binding): Li et al. 2025 PUB.** 531 basins, 10 random basin groups × 10 rounds
  (every basin held out exactly once), test results **concatenated across rounds before metrics**,
  per-basin NSE then **median over 531**, **unclipped** (−∞, 1].
- **Forcing: Daymet only** — our staged cache is Daymet-only, and Li 2025 publishes a Daymet-only column,
  so the comparison is forcing-matched rather than forcing-mismatched. This is why this protocol was chosen.
- **Secondary reported leg (not verdict-binding): Kratzert et al. 2019**, 12-fold,
  `KFold(n_splits=12, shuffle=True, random_state=seed)` over the published basin-list order, seeds 300–309,
  test 1995-10-01 → 2010-09-30. It is exactly reconstructible, so run it if time allows and report it.
- ⛔ Do **not** invent a fold assignment and call it the protocol. No reference publishes its fold file;
  say so in the report, and state that your folds are a faithful reconstruction, not the published partition.

**A2 — ⛔ THE LEAK RULE. This is the one that will silently invalidate everything.**
PUB is a **spatial** holdout. Li's train period (1980–1999) *overlaps* its test period (1995–1999) — that
is legitimate **because the basins are disjoint**, not the dates. Therefore: **the store's exemplars must be
drawn ONLY from training-fold basins.** A held-out basin's own discharge history must never enter the store,
at any lag, through any window descriptor, in any arm. **Write an explicit assertion that fails loudly** if a
test-fold basin id appears in the exemplar set, and report that the assertion ran.

**A3 — The loader positive control now has an exact target, and it tests the loader properly.**
Download Li 2025's per-basin daily simulations: Zenodo **10.5281/zenodo.16895228**, CC BY 4.0 —
`py999_PUB_simulations.npy` (531 MB, shape **6 × 3 × 531 × 6939** = 6 model/forcing configs × 3 seeds ×
531 basins × days), `py999_OBS.npy` (43.4 MB), `py999_ids.npy`. Record sha256 for each.
Then compute NSE using **our own discharge array and our own metric code** against **their** published
simulations, and reproduce:

| target | published |
|---|---|
| LSTM + δHBV, Daymet | **0.742270** |
| LSTM, Daymet | **0.702636** |
| δHBV, Daymet | **0.706809** |

This is a real loader control: it exercises our discharge loading, our date/hydrological-year alignment,
our basin ordering **and** our NSE implementation. If it misses the pre-registered tolerance, **the loader
is wrong — stop and report.** ⚠ Cross-check `py999_ids.npy` against `ref/basin_list_531.txt`; if the orders
differ, align by id and say so.

**A4 — Use the reference's own 35 static attributes, not the gauged run's 27.**
Feng/Li's static set is listed verbatim in the refs output §8 V5 (from `hess-27-2357-2023-t01.xlsx`,
sha256 `1a5eb605d433aebebf4d925f16b68eec3e12d11895d9b1d8796fabc8c034f6b0`, Hub-parsed). The donor-basin arm
must use **those 35**. ⛔ Running it on the gauged run's 27 hobbles the arm relative to the method it is
impersonating, and a hobbled dangerous arm is the failure mode that makes a venue look safe when it is not.

**A5 — The binding threshold moved, twice. Do not use a remembered number.**
- The frontier is **0.785833**, not 0.69 (Hub-confirmed from the publisher's machine-readable Table D2).
- The kill rule bites on the **weakest** reference, not the frontier. `feng2023_MPR+mHM = 0.53` was
  **struck from the set** — it is an in-sample/temporal-test number, not PUB (refs output §8 V5).
- Live floor rows: `qu2026_CAMELSattributes = 0.553` (⚠ secondary) · `kratzert2019_NWM = 0.5766`
  (✅ confirmed, Hub-recomputed: median 0.5766, mean 0.3060, 53 basins ≤ 0, all 531 rows).
- ⇒ **the store must stay below ≈ 0.533 to clear.** Take the reference set from the **amended**
  `.claude/outputs/camels-caravan-refs.md` **including its §8 addendum**, and copy it into your prereg verbatim.

**A6 — One bounded literature item, because the binding row is still secondary.**
Retrieve **Qu et al. 2026, arXiv:2601.01558** and confirm or refute its `CAMELS attributes = 0.553` and
`ALPHAEARTH = 0.612` PUB rows against the primary. ⭐ Read it properly: its abstract states it studies
*"how selecting appropriate donor basins influences prediction in ungauged regions… similarity based on the
embeddings helps identify basins with comparable environmental and hydrological behavior"* — **that is our
dangerous arm, published.** It is both the binding threshold and the closest published analogue to what you
are building. Box this at one paper; ⛔ do not open a general literature search.

**A7 — Caravan is dead. `caravan_protocol: NONE`, confirmed by three independent routes.** Its dataset
paper has no split, no metric convention and no baseline numbers. There is no Caravan spoke and no Caravan
leg in this one.

**Already-verified artefacts you may consume without re-deriving** (`.claude/scratch/hub-refs-verify/`,
sha256s in the refs output §8): `pub_lstm.csv` median **0.6931** · `benchmark_sacsma_ensemble.csv` **0.6447**
· `benchmark_nwm_retrospective.csv` **0.5766** · `basin_list.txt` 531 unique · `li2025_tableD2.xlsx` parsed.

**A8 — Local-only, and memory is the ceiling (Head ruling 2026-09-07).** This spoke runs on the workstation:
M1 Pro, 8 cores, **16 GB RAM**, 82 GB free disk, no CUDA. ⛔ Do not submit anything to a cluster or attempt
cluster access. The gauged tripwire already ran on this machine, so the harness is known to fit — but PUB
adds 10 folds on top of the arm ladder. **Report peak RSS per arm.** If an arm does not fit in RAM, report
it as **NOT RUN — exceeds local RAM** and say which arm; ⛔ never quietly subsample a basin set or a ladder
point to make it fit, because a silently shrunk dangerous arm is how a venue gets declared safe when it
is not.
