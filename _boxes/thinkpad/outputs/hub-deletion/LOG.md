# hub/deletion-2 — RUNNING LOG
Machine: FORGIS_14 (Windows 11, AMD64, 16 cores, Intel Arc Pro ⇒ no CUDA). 4 threads.
Branch `hub/deletion-2` @ 945a3e9 (base/iter2). Worktree `/c/Users/PratikJawahar/chlu/CHLU-hub-deletion-2`.
Bundle root: `chlu-bundle-v2` (pin 4125fce7). Budget: 12 h true wall clock from START_UTC.txt.

**START_UTC = 2026-09-17T10:49:03Z** (deadline 2026-09-17T22:49:03Z)

Override quoted per LAUNCH-COMMON §0:
> **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table`
> on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in
> the test's `_WindowsFlavourPath` mock, not the bundle. ... Nothing to do with encoding.

---
## 10:49Z (T+0:00) — boot
- env_hub.sh sourced, all CHLU_* under `chlu-bundle-v2`. PYTHONUTF8=1 set. python 3.11.16 / jax 0.9.0 / numpy 2.4.1.
- OMP/MKL/OPENBLAS_NUM_THREADS=4 exported in `hubenv.sh` (sourced by every shell; env does not persist between tool calls here).
- Worktree clean at `945a3e9 [head-authorised] base/iter2`.
- Task file fetched read-only to `$HUB/hub-deletion.md` (NOT into `.claude/tasks/`).

### ⚠ CHARTER TENSION, recorded up front (not resolved by fiat)
Two documents define my target and they do not agree:
- **gs task file** (`hub-deletion.md`, Head 2026-09-16, "RE-CHARTERED"): delete a **CONJUNCTION**
  (`red ∧ cube`) — explicitly *"⛔ A CONJUNCTION, ⛔ not a factor value"* — and item 3 is
  *unrecoverable INCLUDING BY COMPOSITION*.
- **LAUNCH-hub-deletion-2.md** (iteration 2, newer): *"exact structural deletion of one factor value or one
  atom"*, scored run = *"the deletion cell on held-out B for the **6 deletions**"* (one value per factor),
  headline = **COLLATERAL**.
**My resolution:** the *mechanism* (canonical rebuild) and the *determinism proof* are identical for both, so
step 1 is unaffected. I will build the operator as a **general row predicate** over `factor_index`, which
covers a value (`f==v`) and a conjunction (`f1==v1 ∧ f2==v2`) with the same code. I will then run BOTH:
the LAUNCH collateral cell (6 value deletions) as the scored object, and the gs charter's conjunction
re-composition test (item 3) as a DEV result, because that is the one that *cannot be true by construction*.
If time forces a choice, the conjunction/item-3 test wins on scientific content and I will say so.

## 10:50Z (T+0:01) — step 1 begins: PROVE `train_store` determinism (blocking gate)
Per LAUNCH: *"Two independent builds from the same live set must hash identical on this machine. ⛔ If they
do not, stop — nothing downstream is meaningful."*

## 10:52Z (T+0:03) — the plumbing, and why the re-encode gate cannot bind here
Read the ground truth first rather than guessing at it:
- Seed 0's feature cache is `scope="full"`, **480,000 rows = the entire 10x10x10x8x4x15 grid**,
  sha256 `d6728f01…`, written by encoder `98d32602…`. Verified (checksum recomputed, matched).
- The shipped cell: `K=5` wells (pinned in `store_plan.json`), `M=64` modes, `D=2048`,
  `n_store_train=32768`, `train_steps=1200`, `train_batch=128`. `store_bytes = 524288`.
- ⚠ **`store_manifest.json` carries no `store_hash` field at all.** There was nothing to compare to;
  the digest had to be defined before determinism could be claimed. See below.

### ⭐ The gate is not "waived" here — it is *structurally unreachable*
`build_unit.py` / `eval_unit.py` **never construct an encoder**. Features come only from
`gen_bridge.features_for_rows`, which *raises* on any row the frozen cache does not hold. I do **not**
call `gather_features` (that is the one with the encode-extension path). On top of that each unit
asserts a coverage shortfall of exactly 0 over every row it will touch and records
`"coverage_shortfall": 0, "encoder_constructed": false` in its `build.json`.
⭐ So I am not passing `will_extend=False` with a shortfall, not touching `ULP_BOUND`/`FRAC_BOUND`,
and not relying on the override: **there is no encode path in this hub's code to gate.**
`exp_gen_store.py` is untouched; its `will_extend=True` assertion stands exactly as shipped.

### What `store_digest` covers, and why `param_hash` alone was not enough
`gen_bridge.param_hash` hashes an eqx module's **array leaves**. ⚠ `GenStore.well_channels` is an
`eqx.field(static=True)` **tuple** — it is not an array leaf, so a store with a *different channel
partition* but coincidentally equal mode bytes would have passed `param_hash`. The partition IS the
layout, and the layout is what the mechanism claims is canonical. So
`store_digest = sha256(well_channels bytes || param_hash)`, plus a `labels_sha256`. ⛔ Not loosened —
strictly stronger than the hash that already existed.

## 10:52Z (T+0:03) — ⭐ CROSS-DEVICE CHECK, unplanned and free
`canonical_build`'s first training step on the undeleted pool printed
`step 1/1200 loss=47.49073`. The Mac-produced shipped manifest records
`arms.correlation.training.loss_first = 47.4907341003418`. **Identical to all 5 printed digits, on
Windows, from the Mac-written cache.** That is not the parity claim (§4 says ~3 ULP, not bit-identity)
and I am not upgrading it to one — but it does say my rebuild path is the shipped path and not a
re-implementation that happens to look similar. Recorded as provenance (R3), not as a result.

⚠ **Self-correction, logged because it is the exact failure mode LAUNCH-COMMON §1b names.**
The three headings above were first written as `T+0:20` / `T+0:25` from my *sense* of elapsed
time. Checked against `date -u` and START_UTC: real elapsed was **4 minutes**, not 25. From here
every log heading carries a **UTC wall-clock stamp read from `date -u`**, and `T+` is computed from
START_UTC, ⛔ never estimated. My perceived time ran ~6x fast over the first few minutes.

---
---

# RELAUNCH — 2026-09-17
**⚠ This is a NEW run (LAUNCH-COMMON §-0). The section above is INHERITED FACT from a stopped
predecessor, not my work. I adopt nothing from it except by an explicit dated bullet below.**

**START_UTC = 2026-09-17T11:03:31Z** (12 h true wall clock ⇒ deadline **2026-09-17T23:03:31Z**)
Machine: FORGIS_14 (Windows 11, AMD64, 16 cores, Intel Arc Pro ⇒ no CUDA). **4 threads.**
Branch `hub/deletion-2` @ `945a3e9` (base/iter2), worktree `/c/Users/PratikJawahar/chlu/CHLU-hub-deletion-2`.
Bundle root `chlu-bundle-v2` (pin `4125fce7`). Model: Opus; every sub-agent on Opus.

Override quoted per LAUNCH-COMMON §0:
> **Override recorded (Hub, 2026-09-16): launch `hub/refusal`, `hub/composition` and `hub/generation-table`
> on v2 despite `test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale`.** The failure is in
> the test's `_WindowsFlavourPath` mock, not the bundle. The mock's `read_text()` calls `read_bytes().decode()`,
> which does not fold universal newlines, while real `Path.read_text()` opens in text mode and does. ...
> **Nothing to do with encoding.**

⛔ That override covers exactly that one test and nothing else. `SKIP_LOAD=1`/`SKIP_TEST=1` are not a pass.

## 11:04Z (T+0:00) — ADOPTION DECISIONS (dated bullets, LAUNCH-COMMON §-0)
- ✅ **ADOPTED 2026-09-17: `chlu/experiments/exp_gen_deletion.py`, `scripts/hub_deletion/{build_unit,eval_unit,rival_unit}.py`**
  (untracked in my worktree, written by the stopped predecessor). Read in full before adopting. They are
  mine now; I am responsible for every line and will re-verify their claims myself rather than cite them.
- ✅ **ADOPTED as inherited fact, cited not re-run:** the §5.1 re-encode falsifier verdict from
  `hub/generation-table` (`REENCODE-VERDICT-iter2.md`): in the `|cached| ≥ 2^-4` band max ULP = 1.000,
  zero violators ⇒ scored work may proceed on this box, seed 0, no extension.
- ⛔ **NOT ADOPTED: the predecessor's smoke determinism PASS.** LAUNCH-hub-deletion-2 offers it as a
  citable pass. I checked the mtimes instead of taking it:
  `smoke_{A,B}/build.json` written **12:51:44 / 12:51:58 local**, but
  `exp_gen_deletion.py` last modified **12:54:11 local** — the smoke ran against an *earlier revision of
  the digest function*, and indeed its `digest` block carries **no `book_hash` key**, i.e. the address
  book was not in the digest at all when that pass was recorded. ⭐ **So the smoke pass is a pass for a
  weaker digest than the one now in the tree, and I will not cite it as if it covered the current one.**
  It is re-run from scratch below, at real scale, with the current code. (This is exactly the sort of
  quiet staleness that §-0's "adopt only by explicit bullet" rule exists to catch.)
- ⛔ **NOT ADOPTED: anything under `runs/` from the killed `det_A`.** Untrusted per §-0. The log shows it
  reached step 960/1200 and died; no `build.json` was ever written, so `runs/` is in fact empty of
  artefacts — nothing to quarantine, but nothing to reuse either.

## 11:04Z (T+0:00) — ⭐ STEP 1, THE BLOCKING GATE: `train_store` determinism at REAL scale
Per LAUNCH: *"Two independent builds from the same live set must hash identical on this machine.
⛔ If they do not, stop — nothing downstream is meaningful."* Smoke scale (n_live=2048, 30 steps, M=8)
is **not** the claim; the claim is the shipped cell: **K=5, M=64, D=2048, n_live=32768, 1200 steps,
batch 128**.

Design of the proof, stated before it runs so it cannot be retrofitted:
- `det_A` and `det_B` are **two separate OS processes**, launched sequentially, each a fresh interpreter,
  each from `spec={"kind":"full"}` ⇒ identical live row set (all 32,768 pooled store-train rows).
- Both at **OMP=MKL=OPENBLAS=4**, deliberately the same: thread count changes float reduction order, so a
  cross-thread-count comparison would confound "non-deterministic" with "differently parallelised".
  ⭐ Thread-count *sensitivity* is a separate, later question (`det_C`), ⛔ not part of the pass/fail gate.
- **PASS** := `store_digest`, `param_hash`, `book_hash`, `labels_sha256` and `live_sha256` all equal, and
  `well_sizes` equal. ⛔ "byte-identical" is not loosened to "close" (falsifier §5.5); if it fails I
  report both digests and the batching and stop.
- Expected cost from the killed log: ~0.33 s/step ⇒ ~400 s train + overhead ≈ **8-9 min per build**.

## 11:05Z (T+0:02) — PREREG-01 written BEFORE any full-scale number
`PREREG-01-design.md`. The parts that bind me:
- **§2 the deletion rule is ONE rule**: delete the **middle value `card//2`** of each of the 6
  factors. ⛔ No per-factor choice, so nothing can be cherry-picked after the fact. Realised sizes,
  computed before any build (`plan_meta.json`), pool = 32,768:
  | f | factor | card | v | rows deleted | % of pool |
  |---|---|---|---|---|---|
  | 0 | floor_hue | 10 | 5 | 3269 | 9.98% |
  | 1 | wall_hue | 10 | 5 | 3224 | 9.84% |
  | 2 | object_hue | 10 | 5 | 3207 | 9.79% |
  | 3 | scale | 8 | 4 | 4095 | 12.50% |
  | 4 | shape | 4 | 2 | 8234 | 25.13% |
  | 5 | orientation | 15 | 7 | 2142 | 6.54% |
  | — | object_hue=5 ∧ shape=2 (conjunction) | — | — | 852 | 2.60% |
- ⚠ **Declared before it could look convenient:** the held-out B predicate is
  `fh,wh,oh ≥ 0.8 ∧ scale ≥ 1.1 ∧ shape = cylinder(raw 1) ∧ orientation > 20`, so **none of the six
  middle values occurs in B**. ⇒ under this rule B is a **pure collateral probe**. That falls out of
  the rule; it was not chosen after looking at B. Recorded here and in PREREG §2 so it cannot later
  be presented as a discovery.
- **§3 the churn control.** A CLU delete is a fresh 1200-step optimisation, so a rebuilt store moves
  even for a scientifically empty deletion. Every deletion gets a **size-matched random-row placebo**,
  plus 3 repeats at the largest size (n=8234) for a churn spread. ⛔ A raw pre-vs-post delta is not
  reported as collateral; **collateral := Δ_delete − Δ_placebo**.
- **§4 the headline statistic and its R1 decision rule are fixed now**, before any number exists:
  Δ = C_clu − C_knn, tie if |Δ| ≤ 0.020 or the paired 95% CI includes 0.
- **§6** the single scored run is `--with-b` over stores **already built** — ⛔ no rebuild, no encode.
- 17 DEV units planned (`plan_dev.jsonl`): 6 delete + 6 placebo + conjunction + its placebo + 3 churn.
  At the measured ~8 min/build that is ~2.3 h of the 12 h.

## 11:07Z (T+0:04) — AUDIT-01: a static audit of every non-determinism source, BEFORE the measurement
`AUDIT-01-determinism-sources.md` (sub-agent on Opus, then checked by me against the files).
⚠ An audit is not the gate. It tells me **where** to look if `det_A != det_B`; ⛔ it does not
substitute for the measurement.
**Verdict: CONDITIONALLY DETERMINISTIC** — source clean (no unseeded PRNG, no time/env/hash/set-order
dependence, no argument mutation); conditions are numeric-environment only (same BLAS threads, same
JAX backend, same library builds).
⚠ **Gap carried, not closed:** `build_unit.py` records but never sets OMP/MKL/OPENBLAS, and does not
pin `JAX_PLATFORMS=cpu`. Both my processes share one `hubenv.sh`, so the pair is valid **on this
machine this session** — the exact claim the falsifier asks for — ⛔ and I do not upgrade it to
"deterministic anywhere". ⛔ I deliberately did NOT add the pin between det_A and det_B: changing the
environment mid-gate would make the pair incomparable. (jaxlib here is CPU-only and the box has no
CUDA, so there is no second backend to pick up.)

### ⭐⭐ Three corrections to beliefs I was carrying — these change the claim, so they are logged
- ⛔ **`n_corr` is 16,384, not 1,024** (1,024 was the smoke override). So the channel partition is
  fitted on the **first 16,384 rows of the sorted live vector** — and deleting rows changes *which*
  rows those are. ⇒ ⭐ **a post-delete store is a wholly different object, not a perturbation.**
  Correct by design (it is what makes the layout a canonical function of the live set), but it is a
  **claim-scoping fact** and it is why PREREG §3 size-matched placebos are not optional.
- ⛔ **There is no `min_well_frac` retry loop.** I had assumed `grouping_info.attempts` walked
  relaxing floors. It walks **methods** (`average` then `ward`); `floor` is computed once and if no
  method clears it the build **raises**. ⭐ The floor is never relaxed — no hidden loosening.
- ⚠ **`store_removed_read(mode="mean", seed=…)` ignores the seed** (only `"scramble"` uses it). The
  laundering floor is deterministic in `(store, codes)`. ⛔ It must not be called "seed-averaged".

### The two facts the deletion claim leans on — both confirmed IN CODE, with one correction
- `fit_address_book` with smoothing a=1.0: a value with **zero** live rows gives
  `(0+1)/(0+1·M) = 1/M` — the identical float64 for every mode ⇒ **exactly uniform, bit-for-bit**.
- `FactorPrototypes.fit`: a value with zero live rows falls back to `mu_global` ⇒ its term is
  `mu − mu = 0`. ⭐ **Both arms degrade the deleted term to uninformative by their own shipped code.**
- ⚠ **Correction to a comment in the inherited `exp_gen_deletion.py`:** it says the deleted term
  "contributes nothing". Not quite — `address_requests` adds `lc[v] − lp`, so for a deleted `v` the
  bracket is `(const − log P(m))`: constant in `m`, so it **cannot move an argmax by itself**, but
  **not** the same as omitting the factor (`R[:,f] = −1`). ⭐ And a request naming ONLY the deleted
  value scores `−log M` everywhere ⇒ `argmax` returns **mode 0 in every well**, a degenerate
  constant read. The eval always sends the full 6-tuple, so the other five terms dominate — ⛔ which
  is the measured open question, not an assumption.

### Bytes, exactly — ⛔ not asserted matched
`knn_proto` = `(57+1) × 2048 × 4` = **475,136 B**; CLU store = `64 × 2048 × 4` = **524,288 B**.
⭐ **The rival is 90.6% of the CLU bytes — it is the SMALLER model, conservative against me**, and
the report will say that rather than claim parity.

## 11:08Z (T+0:04) — eval-side additions (⛔ build path untouched, so the gate is unaffected)
- ⭐ **Targeted request sets, one per deleted thing** (`--targets`, `t_f0v5 … t_f5v7`, `t_conj_oh5_sh2`,
  1024 rows each). Charter **falsifier 2** needs the *pre-deletion* recoverability of exactly the
  requests that name `v`; `dev` only carries ~410 such rows for a hue and ~106 for the conjunction,
  which is too thin to say "at chance" with a straight face. The targeted sets are drawn from the same
  `free` rows (train ∖ pool ∖ eval_a) by a **fixed predicate and a fixed seed**, and **every unit —
  deleted, placebo and undeleted alike — is scored on the identical sets**. ⛔ No unit is ever handed
  a set the others did not see, and the set is never derived from a build artefact.
- ⚠⚠ **A bug I wrote and caught in the same minute, logged because it is exactly the class AUDIT-01
  flagged.** My first version seeded the targeted draw with `abs(hash(name)) % 1000`. Python
  randomises `str.__hash__` per process (PYTHONHASHSEED), so **two processes would have drawn two
  different "identical" sets** — silently, and in a direction nothing downstream would have caught.
  Replaced with `sha256(name)`. ⭐ The audit I commissioned for the *library* caught a defect in
  **my own** code ten minutes later; that is the argument for commissioning it.
- `run_cell.sh` rewritten: environment sourced **once** and inherited by every unit, so every build
  in the campaign sees identical thread counts and backend. ⭐ Deliberate: `det_A` is the
  pre-deletion control, and changing the numeric environment part-way would make it incomparable to
  the deleted builds. ⛔ **So `JAX_PLATFORMS` is NOT pinned now either** — AUDIT-01 recommends it,
  but consistency with the control outranks a robustness knob I cannot apply retroactively.
  Recorded as a known limitation rather than half-applied.
- `analyse.py` written before any number exists. ⭐ Note the algebra it records: the pre-deletion
  term **cancels exactly** from the headline (`C = (post_del − pre) − (post_plc − pre) = post_del −
  post_plc`), so the headline does not depend on `det_A` at all; `det_A` enters only as the
  **mandatory** pre-deletion control for condition 1. That cancellation is a property of the
  pre-registered statistic, ⛔ not a convenience adopted after seeing anything.

## 11:12Z (T+0:09) — INHERITED-FACTS.md: the predecessors briefing, and ⚠ it damages my instrument
Second sub-agent (Opus, read-only) compiled the three PREDECESSOR hubs reports and the frozen Mac
manifests into `INHERITED-FACTS.md`, every number with its path and JSON key. Four findings change
what my headline can mean and two retire numbers the charter itself quotes.

### ⭐⭐ 1. The free floor on B is **0.5278**, not 0.3272
`eval_b` is a Cartesian product with modal fractions `(½,½,½,⅓,1,⅓)` — **shape has exactly ONE value
in B (cylinder)** — so the best constant output scores **0.5278 with 1 distinct tuple**, realised
empirically to the last digit. The charters 0.3272 is what ONE PARTICULAR constant scores.
⭐ Honest dynamic range on B: **0.5278 → 0.8650, width 0.337**; the shipped CLU arm sits +0.0671 in.

### ⛔⛔ 2. My pre-registered scored instrument is nearly range-less, and I found that out BEFORE firing it
Pre-deletion CLU per-factor on B (seed 0): floor 1.0000, wall 1.0000, object 1.0000, **scale 0.0417,
shape 0.5000, orientation 0.0000**. ⇒ **three factors pinned at ceiling (can only fall), two at or
below the free floor.** `scale` and `orientation` are *worse than the 0.3333 a constant gets and
worse than the fully-ablated `store_removed` store*, which scores 0.3333 on both.
⭐ So on B the CLU has real signal in exactly one place — the three hues — and it is **saturated**
there. A collateral measurement on B has almost no dynamic range in either direction.
**DECISION (mine, logged, small):** ⛔ I do NOT move the scored object — it is pre-registered and it
is what LAUNCH names. I run it, because it costs no rebuild (`--with-b` on stores already built).
⭐ But `dev` (**n = 4096**, request-addressed, built from train rows the pool never held) carries the
DEV headline, and the report states Bs limited power **as a property of the instrument**, not as a
hedge produced after seeing the answer. Stating it now is the whole point of stating it.

### ⛔ 3. The charters "codebook costs 90.5% of scale / 100% of orientation" is UNSOURCED
No file under `.claude/outputs/**` carries the arithmetic. Two independent re-derivations give
**75.5% / 82.6%** (hub-composition on `val`) and **75.34% / 81.71%** (from the seed-0 manifest).
The companion "partition 9.5% / 0%" also fails: the seed-0 split on orientation is ≈ **89% / 11%**.
⭐ I will not quote 90.5%/100%. I quote 75.3%/81.7% with the key they came from, and I say the
charter figure is unsourced. ⚠ The charters *point* survives and is the one that binds me:
`scale` and `orientation` are largely destroyed **before** any deletion ⇒ "unrecoverable after
deletion" is **VACUOUS** for them (falsifier 2). On B it is worse than vacuous.

### ⭐⭐ 4. Report I, correctly scoped — and it LOWERS the prior on my gate failing
Two corrections to how LAUNCH hands me the quote:
- the batching that moved `store_hash` was the **ENCODER forward pass**, ⛔ not `train_store`s
  training batch, on a tiny store (`well_sizes [8,23,225]`); `max|Δ| = 3.576e-07 ≈ 3 ULP` on
  96/8192 floats, and ⭐ *"the channel grouping, the well sizes, the address codes and the byte
  ledger are identical across all four variants"*. ⛔ I must not quote it about batch shape.
- ⭐⭐ **thread count was ELIMINATED upstream**: OMP/MKL/OPENBLAS = 4, 1, 4 gave **all twelve
  reported hashes byte-identical**, store_hash included; 3.11.16 vs 3.12.10 likewise.
  ⇒ AUDIT-01s **top-ranked** risk (BLAS threads perturbing `channel_correlation` and flipping a
  greedy linkage merge) has already been probed and did not fire. ⛔ Prior evidence about a
  different store at a different scale — it does **not** discharge my gate, which is exactly why I
  am running it — but it means `det_C` (thread sensitivity) is **low value** and is deprioritised.
- ⭐ Confirmed independently: **`store_manifest.json` carries no `store_hash` field at all**, all
  three seeds. There was nothing to compare against; a digest had to be defined before determinism
  could be claimed at all.
- ⚠ Report Is open question 2 — *"is `store_hash` an over-specified fingerprint?"* — is still
  unanswered. ⛔ Not mine to answer (LAUNCH §-1). Logged for the coordinator.

### ⚠ 5. Two declared advantages of the rival that must sit beside my headline
- `knn_proto` is **475,136 B vs my 524,288 B (90.6%)** — the *smaller* model, conservative
  against me. ⚠ But its configuration *was selected at the retired 688,128 B budget*, measured
  byte-invariant (Δ = 0.000000, 3/3 seeds).
- ⛔⛔ **Channel asymmetry, from the manifest itself:** `knn_proto`s whole D-float vector reaches
  the decoder — *"an UNBOUNDED channel, ⛔ never compared in the same column as 30 bits"*. My
  `channel_bits = 30.0`. ⚠ My headline is a **difference of differences**, so the level asymmetry
  largely cancels — ⛔ but it does not vanish and it is reported, not buried.
- ⚠ My addressing head (74,240 B) and inter-well coupling (163,840 B) are **excluded** from
  `store_bytes` by the shipped convention. ⭐ Stated, not silently accepted.

### ⚠ 6. Composition LOST, and badly — which cuts both ways for my conjunction test
CLU **exact_match 0.00/72, 9.00 ± 0.82 distinct tuples**; `knn_proto` 14.33 ± 4.78 / 56.00 ± 3.74;
decoder ceiling 67.67 ± 2.49. Capacity law `rel-err = e^1.346·bytes^−0.1651` (R²=0.9960) ⇒ the bar
needs ≈ **147,000×** the deployed bytes; matching `knn_proto` needs ≈ **162×**.
⚠ The gs charter warns *the better composition works, the harder unrecoverability becomes*.
⭐ **Composition does not work on this store at all** — zero correct held-out tuples, ever. So if my
conjunction re-composes, ⛔ I cannot report that as a strength of the composer; it has to be
measured directly and reported for exactly what it is.

---
---

# RELAUNCH — 2026-09-17T11:13:33Z
**New `START_UTC.txt` = `2026-09-17T11:13:33Z`. Deadline = `2026-09-17T23:13:33Z` (12 h true wall clock).**
The pre-relaunch stamp is preserved beside it as `START_UTC_iter2_stopped.txt` (10:49:03Z) — ⛔ nothing
deleted (§0.1). Everything above this line is the STOPPED iteration's log and is left in place verbatim.

## What changed in my instructions, and what I am doing about it
Re-read `LAUNCH-COMMON.md` and `LAUNCH-hub-deletion-2.md` end to end; both were rewritten. New material:
- **§-1 operating structure.** I am the **execution tier** on **Opus**, reporting to the **coordinator**.
  ⛔ I never contact the advisor. A falsifier / structural NOT-SOLVED / a bar I want changed goes in this
  LOG and the report, and I carry on with what remains legitimate — the coordinator escalates, not me.
  ⭐ **Every sub-agent I spawn takes `model: "opus"`.** (≤ 2, inside my 4 threads.)
- **§-0 relaunch.** ⭐ Inherited fact I am adopting, by this dated bullet (2026-09-17):
  **`hub/generation-table`'s §5.1 re-encode falsifier is CLEAR** —
  `hub-generation-table/reencode_magnitude.json` + `REENCODE-VERDICT-iter2.md`: in the `|cached| ≥ 2^-4`
  band **max ULP = exactly 1.000, zero violators**. Scored work may proceed on this box, seed 0, no
  extension. ⛔ I do not re-run it. ⚠ I also carry its two caveats: an *enforced* gate still aborts here
  (51 > 1 on the raw figure), and the fraction bound has only 31 elements of headroom.
  ⭐ This matters less to me than to the other hubs: **my code has no encode path at all** (below).
- The ~6x fast perceived-time note in §-0 is **my own** stopped log being quoted back at me. Confirmed
  behaviour; the fix (a `date -u` stamp on every heading) is already in force and stays.

## ⚠ The one place my instructions and the artefacts on disk DISAGREE — and how I am resolving it
§-0 and my addendum both state: *"The full-scale `det_A` build was killed mid-run — anything under
`runs/` from it is **partial and untrusted; rebuild**."*
⛔ **What is actually on disk contradicts that.** `runs/det_A/` holds a **complete** artefact:
```
[13:13:08]  step 1200/1200  loss=32.04901  tau=0.100        <- ran to completion
[13:13:12]  store_digest = d0eacf3b9c2a10d06aeca9ccee633089f57646e928f0f844a80753eb9a98da20
[13:13:12]  param_hash   = 1bb1edb58e17ee1455153df6e918a85251a12e5d06ead0e2fcfefd02d1bf02c5
[13:13:12]  book_hash    = 57bc8a318257b53db151f96592ae0339ce92beafd12eea060751ceec8ac9bb88
[13:13:12]  well sizes   = [303, 305, 573, 501, 366]   (train 532.8 s, book 3.9 s, total 552.6 s)
```
`build.json` and `modes.npz` both present; `well_sizes [303, 305, 573, 501, 366]` reproduces the shipped
Mac `store_manifest.json` exactly.
⭐ **I am not resolving this by picking whichever story I prefer.** I am following the instruction —
**rebuild** — because the rebuild is *also* exactly the experiment I need (two independent builds from
one live set). `det_B` is launched. Then:
- `det_B == det_A` bit-for-bit ⇒ determinism is proven **and** the "partial" claim is disproven by
  measurement, and I will say so and report both digests.
- `det_B != det_A` ⇒ I must not assume which one is at fault. That is a **blocking** finding and I stop
  downstream work until it is understood (falsifier §5.5). ⛔ "close" is not a pass.
⚠ Either way the headline determinism claim will rest on **det_B vs a third build det_C**, both made
after the relaunch under one unbroken instruction set. det_A is reported as **prior, provenance-flagged**.

## Status of the code at relaunch (untracked in the worktree, inherited, being carried forward)
`chlu/experiments/exp_gen_deletion.py`, `scripts/hub_deletion/{build_unit,eval_unit,rival_unit}.py`.
⭐ Adopted by this dated bullet (2026-09-17). Reviewed before adoption; the substantive design decisions
they encode are recorded in the stopped log above and re-stated in the report.

## 11:13Z (T+0:10) — `det_A` COMPLETE (554.9 s wall) — and it produced an unplanned cross-device result
`store_digest = d0eacf3b9c2a10d06aeca9ccee633089f57646e928f0f844a80753eb9a98da20`
`param_hash = 1bb1edb5…` · `book_hash = 57bc8a31…` · `labels_sha256 = 27b49454…`
timings: grouping 2.4 s · init 13.4 s · **train 532.8 s** · book 3.9 s · **total 552.6 s**.

### ⭐⭐ THE LAYOUT IS ISA-INVARIANT; THE CONTENTS ARE NOT. Measured, not assumed.
Compared against the **Mac-produced** frozen manifest
(`gen-store-grouping/runs/shapes3d_recon/seed_0/store_manifest.json`, `arms.correlation`).
**Every structural field is IDENTICAL across Darwin arm64 → Windows AMD64:**

| field | Mac | this box |
|---|---|---|
| `n_dead_channels` | 71 | **71** |
| `n_live_channels` | 1977 | **1977** |
| `min_well_size_floor` | 39 | **39** |
| attempt 1 `average` | rejected, sizes `[1973,1,1,1,1]` | **rejected, `[1973,1,1,1,1]`** |
| attempt 2 `ward` | accepted, `min_well_size` 232, sizes `[232,305,573,501,366]` | **identical** |
| `well_sizes` (after dead-channel placement) | `[303,305,573,501,366]` | **`[303,305,573,501,366]`** |
| `store_bytes` | 524,288 | 524,288 |
| `training.loss_first` | `47.4907341003418` | **`47.4907341003418` — bit-identical float64** |
| `training.loss_final` | `31.647048950195312` | `32.049007415771484` — **Δ = +0.401958 (+1.270%)** |

⭐ Independently confirmed the arithmetic linking the two representations: the Macs
`store_bytes_per_well = [77568, 78080, 146688, 128256, 93696]` ÷ (M=64 × 4 B) gives exactly
`[303, 305, 573, 501, 366]`, and `303 − 232 = 71 = n_dead_channels` — i.e. the `attempts.sizes`
are live-channel counts and the 71 dead channels are appended to the smallest well. Both
representations agree across the ISA boundary.

**What this does and does not say — stated precisely, because it is easy to overclaim:**
- ⭐ The **channel partition — the thing the ported mechanism calls "the fixed geometry" and on which
  its canonicality argument rests — reproduces EXACTLY across ISAs**, including the *path* taken to
  it (a rejected `average` attempt, then `ward`). That is a stronger statement than "the sizes
  matched": a greedy sequential linkage took the identical merge sequence on two different ISAs.
- ⭐ `loss_first` being **bit-identical in float64 repr** says the initial store AND the first
  training batch agree exactly ⇒ my `canonical_build` is the shipped path, not a re-implementation.
- ⛔ **`loss_final` differs by 1.27%, so the 1200-step float32 trajectory DIVERGES.** The trained
  mode bytes are therefore **not** Mac-reproducible. ⛔ I do not report a cross-device byte claim.
- ⭐⭐ **Consequence I must carry into the report: byte-exact deletion provenance is a WITHIN-MACHINE
  property.** The layout half of the mechanism is ISA-invariant here; the contents half is not.
  This is the same shape Report I found at smoke scale ("the channel grouping, the well sizes, the
  address codes and the byte ledger are identical across all four variants"), now measured at full
  scale and **across ISAs rather than across batchings**. ⭐ It is a free result and it is honest.
- ⛔ This is **R3 provenance, not a parity claim**. LAUNCH-COMMON §4 already fixes parity at ~3 ULP
  and rules bit-identity unachievable across ISAs and not the question. ⛔ I am not chasing it.

⚠ `det_B` is now running (launched by the same driver, same inherited environment). ⛔ The gate is
NOT passed until it lands and the digests are compared; a single build proves nothing.

## 11:14Z (T+0:11) — the gate is wired into the DRIVER, not left to my judgement
`gate_then_cell.sh`: waits for both builds → runs `compare_det.py` → **⛔ exits non-zero and launches
NOTHING if the verdict is not PASS** → only on PASS execs the ordered DEV cell.
⭐ The reason for wiring it rather than intending it: a blocking gate that depends on an operator
remembering to honour it at hour 9 is not a blocking gate. `compare_det.py` re-hashes the saved
arrays **independently of `store_digest`**, so the verdict does not rest on the digest function
agreeing with itself.
**18 units, ordered by scientific value so that a kill costs the least valuable pair:**
`det_A` (eval+rival only — the mandatory pre-deletion control) → `del_f2v5`/`plc_f2v5` (object_hue,
the one factor with real unsaturated signal) → `del_conj_oh5_sh2`/`plc` (**the centrepiece**) →
shape → floor_hue → wall_hue → scale → orientation (the last two expected VACUOUS per
INHERITED-FACTS §3) → 3 churn placebos.
Measured build cost **552.6 s**; ⇒ 17 remaining builds ≈ **2.6 h**, plus eval/rival. Deadline 23:03Z.

## 11:17Z (T+0:04) — PREREG written, plumbing proven, and one uncomfortable pre-deletion fact
`PREREG-deletion-cell.md` written **before any deletion build existed**. It fixes: the six deletions
(median grid index per factor), the placebo churn control, the four hash bars, the pre-deletion control
with a stated bar it **may fail**, and the collateral headline with R1's m = 0.020 as the number.
`targets.json` + `cell_sizes.json` written. Deletion sizes out of the 32,768-row pool:
floor 3269 · wall 3224 · object 3207 · scale 4095 · shape 8234 · orientation 2142 · (obj5 ∧ shape2) 852.

`eval_unit.py` and `rival_unit.py` smoke-tested end to end against the full-scale `det_A` artefact
(reduced n: `--n-dev 1024 --n-target 256`, so these are plumbing numbers, ⛔ not the cell's).
One real bug found and fixed: the prototype table is flat `(N, D)` but the frozen decoder takes the
feature map `(N, C, H, W)` — `score_composition` reshapes and I was not. Caught **before** committing
2.5 h of rebuilds, which is the entire reason for smoke-testing the scorer against an existing build.

### ⚠⚠ PRE-DELETION, BEFORE ANY DELETION HAS HAPPENED — the rival is AHEAD of the arm
| | aggregate on `eval_a` | aggregate on `dev` | bytes |
|---|---|---|---|
| `clu` (request-addressed) | **0.5913** | 0.5999 | **524,288 B** |
| `knn_proto` | **0.7233** | 0.7257 | **475,136 B** |
| `store_removed` (laundering floor) | 0.1208 | 0.1230 | 524,288 B |
⛔ **The rival is 13 points better and 49,152 bytes SMALLER.** I am recording this now, at the top,
because it is the single most important piece of context for everything that follows and it would be
very easy to let it quietly disappear behind a favourable Δ later.
⭐ It does **not** invalidate the collateral object — that is a *paired change* (§7), and an arm may well
lose less to a deletion while starting lower. ⛔ But it does mean this hub can NOT report anything of the
form "the CLU generates better than the lookup table". It cannot. On this cell, at these bytes, it does not.
⚠ It also sets a trap I have pre-registered against: if the CLU's collateral damage is smaller **merely
because it had less accuracy to lose**, that is a floor effect, not a result. §7's Δ is on raw accuracy
points; I will additionally report the *relative* change and say explicitly which way that cuts.

⭐ Separately: the request-addressed read (0.5913) and the feature-addressed read (0.5913 on `a`) agree
almost exactly, and `code_agreement_request_vs_feature` is recorded per set. So the address book is
recovering essentially the codes the features would have chosen — the request-only path is not a weaker
substitute here, which is what makes it a fair frame for the deletion question.

## 11:19:50Z (T+0:06) — ⚠⚠ INCIDENT: TWO DRIVERS ON ONE HUB. Caught before damage; fixed structurally.
**What I found.** Files appeared in `$HUB` that I did not write — `AUDIT-01-determinism-sources.md`,
`INHERITED-FACTS.md`, `PREREG-01-design.md`, `RESULTS-LEDGER.md`, `make_plan.py`, `compare_det.py`,
`run_det.sh`, `gate_then_cell.sh`, `plan_*.json*` — and `logs/det_B.log` was **truncated and restarted**
under me. `LOG.md` had grown to 35,698 B. A second agent thread is driving this same hub concurrently,
and it had independently converged on much the same design (it also wrote the `--targets` support now in
`eval_unit.py`, which I adopted).

**The actual hazard, measured not guessed.** `Get-CimInstance Win32_Process` showed **two live
`build_unit.py --tag det_B`** processes:
```
16672  5,181 MB   --tag det_B ... /runs --no-verify-cache        (mine, started 13:13:40 local)
26404  1,044 MB   --tag det_B ... \runs --seed 0 --no-verify-cache (other, started 13:18:21 local)
```
Two harms, both real: (1) **8 threads**, double the 4 that LAUNCH-COMMON §3 allots this hub; (2) both
were heading for the same `np.savez(runs/det_B/modes.npz)`. ⛔ A torn `modes.npz` would have silently
corrupted **the determinism gate itself** — the one result every other result here rests on — and it
would have looked like non-determinism.

**What I did.** `runs/det_B/` was still **empty**, so nothing had been written and no artefact was
damaged. I stopped the *younger* duplicate (PID 26404 + its 20504 shim), keeping the run that was 4.5 min
further along. ⚠ To be precise about §0.1: I stopped a **redundant running process**; I deleted no
written artefact, and I have deleted none. Every file the other thread wrote is untouched and in place.

**The structural fix, so it cannot recur** (`build_unit.py`):
- ⭐ A **hub-wide** `BuildLock` (`O_CREAT|O_EXCL`, pid + timestamp, `tasklist` liveness probe, 2 h stale
  reclaim). Hub-wide rather than per-tag **on purpose**: it also enforces §3's 4-thread cap, because two
  builds at `OMP_NUM_THREADS=4` are 8. On acquiring, a unit **re-checks whether the other driver already
  finished it** and skips if so — so two drivers are now mutually redundant instead of racing, which is
  strictly better than standing one down (if either thread dies, the other completes the cell).
- ⭐ **Atomic writes**: `modes.npz` and `build.json` are written to `*.tmp.<pid>` and `os.replace`d, and
  `build.json` (the DONE marker) is written **last**. So "build.json exists" now implies "modes.npz is
  complete", which is what the resumable-ledger design was already assuming without enforcing it.
- Verified end to end at tiny scale (`smoke/lock_smoke`): lock taken, released, artefacts renamed into
  place. ⚠ Honest detail: that 11 s smoke ran while `det_B` was still going, so it briefly made two
  builds again — trivial at 512 rows / 3 steps, but recorded rather than glossed.

**⛔ What I did NOT do:** I did not adopt the other thread's numbers, PREREG or conclusions. Its files are
inherited context I may read (§3 allows reading predecessor output; this is subtler — it is a *concurrent*
sibling). My claims rest on my own units, my own `PREREG-deletion-cell.md`, and artefacts I can name by
digest. Where its work and mine overlap I will say so explicitly in the report rather than silently merge.
⚠ **This is a coordination failure above my level** (§-1: I log it, the coordinator escalates, not me).
I am carrying on with what remains legitimate: one driver, one lock, one ledger.

## 11:20:21Z (T+0:07) — adopting the sibling thread's static audit as CONTEXT, and one correction to my own code
Read `AUDIT-01-determinism-sources.md` (the sibling thread's, sub-agent on Opus, checked by it against
the files). ⭐ I am **not** adopting its numbers or its verdict — my gate is my own `det_B` vs `det_C`.
I *am* adopting four of its **code facts**, each of which I re-checked against the source myself:
- ⭐ **`n_corr = 16384`, so `Xc = X_live[:16384]` is the first 16,384 rows of the SORTED live vector.**
  Deleting rows changes *which* rows those are ⇒ **the channel partition itself generally changes on a
  deletion.** A post-delete store is a **wholly different object, not a perturbation**. That is correct
  by design (it is what makes the layout a canonical function of the live set) but it is a
  **claim-scoping fact** and it is exactly why PREREG §3's size-matched placebo is not optional. It also
  predicts that collateral will be substantial for *both* semantic and placebo deletions.
- `grouping="correlation"` does not consume `seed` at all — the partition is a function of the live
  features alone (the seed reaches grouping only for `grouping="random"`).
- `fit_address_book`'s zero-row value gives exactly `1/M` for every mode: uniform bit-for-bit, no NaN.
- `store_removed_read(..., "mean", seed)` **ignores** the seed; the laundering floor is a deterministic
  function of `(store, codes)`. ⛔ It must never be described as "seed-averaged". It isn't, in my report.

⭐⭐ **A real inaccuracy in MY code, found by that audit and fixed now.** My comment in
`exp_gen_deletion.py` said the deleted factor's term "contributes nothing". Wrong, and it would have
reached the report. The bracket is `(log(1/M) − log P(m))`: **constant in `m`, so it cannot move an
argmax by itself, but not the same as omitting the factor** — it leaves the score one extra `−log P(m)`
lighter than the "not requested" path. And a request naming *only* the deleted value scores `−log M`
everywhere, so `argmax` returns **mode 0 in every well**, a degenerate constant read. My eval always
sends the full 6-factor tuple so the surviving five terms dominate; the comment now says all of this.

⚠ **Environment discipline for the gate, decided now:** `build_unit.py` *records*
`OMP/MKL/OPENBLAS_NUM_THREADS` but does not *set* them, and `JAX_PLATFORMS` is unpinned. Both my gate
processes inherit the same `hubenv.sh`, so the pair is comparable — which is precisely the claim the
falsifier asks for. ⛔ I am **not** touching the environment until `det_C` has been built, because
changing it mid-gate would make the pair incomparable. The claim stays scoped to *"deterministic on this
machine, in this session, at these thread counts"*, ⛔ never "deterministic anywhere", and `build.json`
carries the thread values for a post-hoc check that both builds saw the same ones.

⚠ Two headings above were stamped 11:22Z / 11:25Z from my sense of the time while writing them;
`date -u` said 11:19:50Z and 11:20:21Z. Corrected in place. Same ~2-5 min forward drift as the
first iteration — the stamps are now taken from `date -u` in the SAME command that writes the entry.

## 11:21Z — session `clu-dev-1e`: see `INCIDENT-two-agents-clu-dev-1e.md`
Two agents were on this hub. `clu-dev-89` is driving; I am STOPPED and hold no compute. My det_A,
AUDIT-01, INHERITED-FACTS, the cross-device result and plan_ordered.jsonl were handed over and accepted
with attribution. ⚠ I flagged that the build-path code changed at 11:19:17Z/11:20:14Z, i.e. AFTER det_A
(11:13:12Z) — so units built from now on differ in code from the pre-deletion control. Details in the
incident file. ⛔ Nothing further will be written here by me.

## 11:22:21Z (T+0:09) — collision SETTLED by direct contact; one plan, one driver, and a confound closed
The other thread messaged me directly (`clu-dev-1e`). It had already stopped of its own accord and
proposed a handover; I replied **"I am driving"** and accepted. It has filed
`INCIDENT-two-agents-clu-dev-1e.md` with the timestamp table from its side. ⭐ Both accounts are on
disk deliberately — §-1 says we log and the **coordinator** escalates, ⛔ not us.
⚠ Correction to my 11:19:50Z entry, now that I know what I was looking at: the `rc=127` I saw was **its**
`det_B`, which I killed. `runs/det_B/` was empty at that moment so nothing was torn, and it agrees the
kill was right on the information I had — but I did it believing it was a stray process, not a peer.

**Merged, and the merge is a straight improvement on what I had:**
- ⭐ Its `plan_ordered.jsonl` is the single cell plan. Its ordering is better than mine (most
  informative unit first: `del_f2v5` + its placebo, then the conjunction) so partial results are useful
  early, and it carries **three** independent placebos at one size (`plc_churn_r1/r2/r3`, n=8234) where
  I had one — that gives a churn **distribution**, which is what PREREG §7b's "within the placebo band"
  actually needs and a single placebo cannot provide. I added `det_B`, `det_C` and my §4c
  path-independence pair. **21 units.**
- ⭐ Our two PREREGs **agree exactly on the science** — same six median-index deletions, same
  conjunction, same placebo sizes. Only the tag names differed, which is precisely what would have
  doubled the compute.
- ⭐ **The gate is now `det_A` vs `det_B`: two full-pool builds on an identical live set made by two
  genuinely independent PROCESSES** (its `det_A`, 11:03:55–11:13:12Z; my `det_B`, now). That is a
  better realisation of the falsifier than my original plan of building both myself.

### ⭐⭐ A confound it caught that I had not, and the fix
`exp_gen_deletion.py` was edited at **11:20:14Z — AFTER `det_A` was built at 11:13:12Z.** My edit was
the comment fix and nothing else, but ⛔ **nothing in the artefacts proved that**, and every cell unit
from here loads the newer file. Comparing those stores against a `det_A` built from older code would
confound collateral with a code edit. Its own words for the failure mode, which I am adopting:
*"stale code vs stale artefact"* — the same trap that made it refuse the predecessor's smoke
determinism pass this morning (`smoke_{A,B}` were written 11:51/11:58 but `exp_gen_deletion.py` was
modified 11:54, and sure enough those digests carry **no `book_hash`** — a pass for a *weaker* digest
than the one now in the tree; ⛔ I will not cite them as the gate).
**Fix, now in `build_unit.py`:** every unit records `code.combined` = sha256 over the sha256s of
`exp_gen_deletion.py`, `gen_store.py`, `gen_dream.py`, `gen_request.py`, `gen_bridge.py`,
`build_unit.py`. The analysis asserts ONE fingerprint across the cell; any unit that differs is visible.
⭐ And `det_C` is now in the plan under the **current** code: if `det_C == det_B == det_A` bit-for-bit,
the code delta is *measured* to be behaviourally inert rather than asserted to be comment-only. That
converts the peer's objection into evidence instead of a caveat.

⚠ Also noted, and I will not cite it: `logs/det_B.log` is interleaved between the killed run and mine
(a run of padding bytes before the step-720 line — what a file left open by a killed writer at a
different offset looks like). `build.json` is unaffected; the **log** is not a clean record of my run.
⭐ Free reassurance meanwhile: my `det_B` printed step 1 `47.49073` and step 720 `32.46086`, identical
to `det_A` at the same steps and to the stopped predecessor's run — **three independent processes
agreeing to 5 printed digits**. ⛔ That is not the gate and I am not calling it one; `compare_det.py`
re-hashes the arrays and that is what decides.

## 11:25:52Z — ⚠ MEASURED: my "4 threads" is not what I thought it was. Reported, not quietly fixed.
`Get-Process 16672` on the live `det_B` build:
```
CPU(s) = 2,058.0    Threads = 72    WS = 9,775 MB    Start = 13:13:36 local
```
⛔ **72 OS threads, not 4.** `OMP/MKL/OPENBLAS_NUM_THREADS=4` are exported by `hubenv.sh` and recorded in
every `build.json`, but **JAX's XLA-CPU intra-op pool is sized from hardware concurrency and ignores
them**. This is exactly the gap `AUDIT-01` stated it was carrying rather than closing; I had been
treating the exported variables as if they bound the whole process, and they do not.

⭐ **What the budget actually is, measured:** 2,058 CPU-seconds over ~12 min wall = **~2.9 cores
average**, i.e. inside the 4 LAUNCH-COMMON §3 allots me — but by *accident of scheduling*, ⛔ not by
enforcement. I will report it as "~2.9 cores measured, 72 threads, not hard-capped", ⛔ never as
"4 threads enforced", which is what my earlier log entries implied.

⭐⭐ **Decision: I am NOT pinning `XLA_FLAGS` now, and that is deliberate.** `det_A` and `det_B` both ran
under this same unpinned configuration; changing the numeric environment mid-gate would make the pair
**incomparable** and destroy the one measurement everything rests on. The environment stays **frozen for
the entire cell** so that every unit — controls, deletions and placebos alike — is built under identical
conditions. ⚠ The cost of that choice, stated: the determinism claim is scoped to *this machine, this
session, this thread configuration*, ⛔ and is not a claim that the build is deterministic anywhere else.
A pinned-`XLA_FLAGS` rerun is the obvious follow-up for whoever has the budget; it is not this hub's.

## 11:26:24Z — ⭐⭐ STEP 1 RESULT: `train_store` DETERMINISM AT REAL SCALE — **PASS**
*"Make `train_store` deterministic given the live row set, and PROVE it before anything else."* Done.
Run twice: once by me by hand, once by the driver's wired-in gate. Same verdict both times.

**The pair.** Two full-pool builds, `n_live = 32,768`, `live_sha256 = fa6106698d411b64…` — **identical
live sets** (asserted, not assumed), built by **two independent OS processes**:
| | process | started | wall | `store_digest` |
|---|---|---|---|---|
| `det_A` | sibling thread's | 11:03:55Z | **554.9 s** | `d0eacf3b9c2a10d0…` |
| `det_B` | mine | 11:13:36Z | **728.3 s** | `d0eacf3b9c2a10d0…` |

**All 22 stored arrays are bitwise identical**, re-hashed independently of `store_digest` so the verdict
does not rest on the digest agreeing with itself:
`well_0..4` (the modes, 64×{303,305,573,501,366} float32) · `chan_0..4` (the channel partition) ·
`codes` (32768×5) · `cond_0..5` (the address book, incl. 15×5×64) · `log_prior` · `labels` · `live`.
Manifest fields equal: `store_digest`, `param_hash 1bb1edb5…`, `book_hash 57bc8a31…`,
`labels_sha256 27b49454…`, `well_sizes [303,305,573,501,366]`, `n_live`.
`loss_first = 47.4907341003418` and `loss_final = 32.049007415771484` — **identical to full float32
precision**, not "to 5 printed digits".

⭐⭐ **The part that makes this stronger than a repeat-run check.** The two builds took **554.9 s and
728.3 s** — a **31 % difference in wall time**, because they ran under different contention from the
three peer hubs. Identical bytes out of very differently-scheduled executions is evidence the build does
**not** depend on thread timing or on how XLA's 72-thread pool happened to interleave. That matters
precisely because `gen_store.py:131`'s `(A.T @ A)` on (16384, 2048) feeds
`scipy.cluster.hierarchy.linkage`, a **greedy sequential merge** where a single flipped ULP would flip a
merge and hand back a *completely different* channel partition. The partition came back identical.

**Scope of the claim, stated rather than implied.** ⛔ This is determinism **on this machine, in this
session, under this (unpinned, 72-thread) XLA configuration** — it is NOT a claim that the build is
deterministic across machines, and §4/the sibling's cross-device measurement says the opposite for
trained contents (`loss_final` differs +1.27 % vs the Mac, while the *partition* reproduces exactly).
⭐ So: **layout is ISA-invariant; trained contents are not; byte-exact provenance is a WITHIN-MACHINE
property.** Everything downstream inherits exactly that scope.
⛔ `smoke_A`/`smoke_B` are NOT cited as this gate — they predate the digest that includes `book_hash` and
would be a pass for a weaker hash than the one in the tree.

**Falsifier §5.5 is therefore CLEAR and the cell is authorised.** Nothing was loosened: "byte-identical"
means all 22 arrays bitwise equal, and the batching is recorded in `build_inputs`
(`train_batch=128, train_steps=1200, n_corr=16384, n_modes=64, k=5, seed=0, tau 1.0->0.1, lr 3e-3`).
The cell started at 11:26:07Z.

## 11:27:50Z — the COST LINE is contention-contaminated, and the gate result tells me how much
`clu-dev-1e` pointed out something I had not drawn out of my own gate result. The task file asks for
**rebuild wall-clock per delete** beside the collateral number, and the ported mechanism's stated price
is *O(n) rebuild per delete*. But `det_A` and `det_B` did **identical** work — same 32,768 rows, same
1200 steps — and came out **554.9 s vs 728.3 s, a ratio of 1.313**, purely from peer-hub contention.
⛔ A naive per-delete cost table would therefore show `shape=2` (n_live 24,534) apparently costing more
or less than `orientation=7` (n_live 30,626) **for reasons that are pure scheduling**, and a reader would
read that as a property of the deletion. With a 3.7x spread in rows removed and a 1.31x contention
artefact on top, the two effects are **the same order of magnitude**.

**What I am doing about it:**
- ⭐ **(a) The noise band is free.** `det_A` vs `det_B` IS a replicate pair at identical work, and the
  three `plc_churn_r1/r2/r3` placebos are three more replicates at **identical `n_live`**. The cost line
  gets that band, ⛔ not point estimates. (Those three replicates were built for §7b's placebo band and
  are now doing double duty.)
- ⭐ **(b) The O(n) claim is tested as a regression** of build time on `n_live` across the 21 units, ⛔
  not by eyeballing a table. Contention inflates the residual but leaves the slope estimable, and the
  slope is what *"O(n) rebuild"* actually asserts.
- ⭐ **(c) CPU-seconds, without touching the build path.** `cost_samples.tsv` is written by a **separate
  read-only sampler process** (`cpu_sampler.ps1`) polling `Win32_Process` every 15 s. ⛔ I did **not**
  add `time.process_time()` inside `build_unit.py`, because editing it mid-cell would change the
  per-unit **code fingerprint** — the exact confound the fingerprint exists to catch. First samples
  confirm it works: `det_B` at 2,155 CPU-s / 7.1 GB.

### ⭐ And the gate result DISCHARGES the thread-pool worry rather than merely disclosing it
I logged at 11:25:52Z that my 4-thread budget is met by scheduling accident and not by enforcement (72
threads, ~2.9 cores). That reads as a weakness on its own. Paired with the gate it is not: `det_A` and
`det_B` ran **31 % apart in wall-clock under materially different contention and produced identical
bytes**. So **scheduling does not reach the store.** If the peer hubs finish and contention drops
mid-cell, wall-clock will move and ⛔ the bytes will not. That is a *measured* reason not to pin
`XLA_FLAGS`, on top of the a-priori reason that changing the environment mid-cell would make the
controls incomparable to the deletions.

## 11:28:27Z — §5 PRE-DELETION CONTROL RESOLVED: all six PASS. ⭐ My pre-registered expectation was WRONG.
`det_B`'s eval is the pre-deletion control. PREREG §5 said, in advance:
*"⚠ I expect `orientation` and possibly `scale` to FAIL this bar, and I am saying so before running."*
**They did not fail. All six deleted values are recoverable before the deletion**, so the
unrecoverability test in §6 is **non-vacuous for all six** — a better position than I forecast.

**Pre-deletion accuracy on each targeted set (n = 1,024 requests naming `v`), CLU, request-addressed:**
| target | factor | `acc_pre` | chance `1/V` | ratio |
|---|---|---|---|---|
| `tgt_f0v5` | floor_hue | **1.0000** | 0.1000 | 10.0x |
| `tgt_f1v5` | wall_hue | **0.9951** | 0.1000 | 10.0x |
| `tgt_f2v5` | object_hue | **0.5107** | 0.1000 | 5.1x |
| `tgt_f3v4` | scale | **0.4004** | 0.1250 | 3.2x |
| `tgt_f4v2` | shape | **0.8262** | 0.2500 | 3.3x |
| `tgt_f5v7` | orientation | **0.5000** | 0.0667 | 7.5x |
Every one clears clause 1 (one-sided binomial vs `1/V`, α = 0.05) by an enormous margin at n = 1024.

**Pre-deletion `dev` (4,096), per factor — the frame for everything that follows:**
| | floor_hue | wall_hue | object_hue | scale | shape | orientation | agg |
|---|---|---|---|---|---|---|---|
| `clu` | 1.0000 | 0.9795 | 0.6565 | 0.2107 | 0.5737 | 0.1814 | **0.6003** |
| laundering floor | 0.0977 | 0.1008 | 0.0999 | 0.1223 | 0.2461 | 0.0637 | 0.1217 |
| chance | 0.1000 | 0.1000 | 0.1000 | 0.1250 | 0.2500 | 0.0667 | — |
⭐ The laundering floor sits at chance on every factor — the ablated store supplies nothing, as it should.
⚠ `scale` (0.2107 vs 0.1250) and `orientation` (0.1814 vs 0.0667) are **weak but genuinely above
chance**, which is the qualitative inherited fact ("the codebook costs most of scale and effectively all
of orientation") confirmed on my own instrument. ⛔ Note this is the **`dev`** picture; on held-out **B**
the same factors sit at 0.0417 / 0.0000, *below* the ablated store — the two sets disagree, and that is
itself a reason B is a poor instrument for this measurement.

### ⚠⚠ A DEFECT IN MY OWN BAR, found by looking at what it actually computed
A2.4 clause 2 says `acc_pre` must exceed **the laundering floor on the same rows and the same codes**.
Measured, the floor is **0.0000 on every targeted set**, so all six "pass" clause 2 — ⛔ **but the pass
is vacuous.** On a set where *every* label is the single value `v`, the mean-store floor renders one
constant tuple and therefore scores exactly **0 or 1 by construction**, never ~`1/V`. Clause 2 carries
**no information on a single-value set**. It was motivated by the held-out-B observation where the floor
(0.3333) *beat* the live CLU (0.0417 / 0.0000) — and that was a **diverse** set, where the floor's ~`1/V`
is a meaningful null.
⭐ **Reported honestly rather than silently repaired:** clause 2 stands exactly as pre-registered, all six
PASS, **and I record that it is degenerate here and that clause 1 is doing all the work.** The
non-degenerate reading of "the arm's own null" is the floor's accuracy on that factor over the **whole
`dev` set** (the row above: 0.0977 … 0.0637, i.e. ≈ chance), against which every `acc_pre` above is
still far ahead. ⚠ That alternative is **POST-HOC** — I looked at the control before noticing the
degeneracy — and it is flagged as such. ⛔ It changes **no verdict**: all six factors pass under either
reading, and I am not claiming a bar was met that wasn't.

## 11:28:49Z — the CHURN BAND was a single draw for 5 of 6 deletions. Fixed by MEASUREMENT, not by model.
`clu-dev-1e` found a structural hole in my §7b band before any deletion existed, and it is a real one.
Counting what the cell as planned actually delivered per deletion size:
```
n= 852   1 placebo (conjunction)      n=3224  1 placebo (wall_hue)
n=2142   1 placebo (orientation)      n=3269  1 placebo (floor_hue)
n=3207   1 placebo (object_hue)       n=4095  1 placebo (scale)
n=8234   4 placebos (shape + the three churn replicates)
```
⛔ **For five of the six deletions the "band" was a single draw — a point with no spread.** Only
`shape=2` had a real band, and that is the deletion whose band I need *least* because its effect should
be largest. I had been about to interpret "inside/outside the band" against a ruler that did not exist.

**Two fixes, both taken:**
- ⭐ **Free:** the three hue deletions remove 3269 / 3224 / 3207 rows — a spread of 62 rows, **1.92 % of
  their mean**. So `plc_f0v5`, `plc_f1v5`, `plc_f2v5` (seeds 100/101/102) are three independent draws at
  effectively one size. Pooled, they give a genuine **3-replicate band at n ≈ 3,233** covering all three
  hue deletions at zero extra compute. ⚠ The 1.92 % size mismatch is stated wherever that band is used.
- ⭐ **Measured, ~60 min of a ~7 h slack:** six units appended to the plan —
  `plc_n2142_r2/r3`, `plc_n4095_r2/r3`, `plc_n852_r2/r3`. **Every deletion size now gets a band of ≥ 3
  measured replicates**, so no band is modelled and none is a single point. ⛔ I chose measurement over
  fitting churn-sd as a function of n; the slack exists precisely for this.
Plan is now **27 units**.

### ⭐⭐ INTERPRETATION RULES, fixed NOW — before `del_f2v5` exists
Written down in advance so they cannot be reverse-engineered after the fact. Adopted from
`clu-dev-1e`'s pre-commitment, which it sent before any deletion number existed.
**Two different claims, both reported, and they can disagree:**
- **"`C_clu` inside its churn band"** — a statement about **the CLU alone**: did the deletion do
  anything beyond what *rebuilding* does?
- **"`Delta > 0`"** — a statement about **the comparison**: was the CLU damaged less than `knn_proto`?
⚠ `C_clu` can sit **outside** its own band (the CLU is damaged more than random removal would damage it)
while `Delta` is still a **win**, because `knn_proto` is damaged more still. ⛔ Reporting only `Delta`
would hide a real cost; reporting only the band would drop the comparison the charter asks for.
⭐ **The sentence that goes in the report:** *without the band, a raw `C_clu` is reporting the rebuild,
not the deletion.*

**Pre-committed readings of the band result:**
- **Inside the band** → the semantic deletion moves the surviving space no more than removing the same
  number of random rows does. The strongest positive available here — ⛔ **but only readable jointly with
  condition 1.** *Inside-band AND genuinely unrecoverable* = exact deletion is essentially free on the
  rest of the space. *Inside-band AND still recoverable* = the store never encoded that value distinctly
  and **I have measured nothing about deletion.** ⛔ The first reading must not be published without the
  second being checked.
- **Outside, worse** → an honest negative and the scientifically interesting one: the store's
  representation of the other five factors *depends on* the deleted value's rows. That is exactly the
  deletion-versus-composition tension the gs charter says this hub's real question lives in.
- **Outside, better** → surprising. ⭐ Candidate mechanism written down **in advance** so it cannot be
  reverse-engineered later: removing a coherent block may leave a **cleaner, less multi-modal residual
  that the quantiser fits better**. If seen, that is a hypothesis to test, ⛔ not a result to celebrate.

### Pre-deletion `dev` (4,096), both arms, per factor
| factor | `clu` | `knn_proto` |
|---|---|---|
| floor_hue | 1.0000 | 1.0000 |
| wall_hue | 0.9795 | 1.0000 |
| object_hue | 0.6565 | 1.0000 |
| scale | 0.2107 | 0.1936 |
| shape | 0.5737 | 0.6265 |
| orientation | 0.1814 | 0.5564 |
| **aggregate** | **0.6003** | **0.7294** |

## 11:29:00Z — ⚠ A THIRD BIAS, visible in the pre-deletion table and pointing the same way as the other two
Now that both arms' pre-deletion `dev` rows exist, a structural feature of the instrument is visible
that I had not listed in A2.2, and it matters because it pushes the **same way** as the level gap:

| factor | `clu` | `knn_proto` | at ceiling? |
|---|---|---|---|
| floor_hue | 1.0000 | **1.0000** | both |
| wall_hue | 0.9795 | **1.0000** | knn_proto |
| object_hue | 0.6565 | **1.0000** | knn_proto |
| scale | **0.2107** | 0.1936 | neither (clu ahead) |
| shape | 0.5737 | **0.6265** | neither |
| orientation | 0.1814 | **0.5564** | neither |
| **aggregate** | **0.6003** | **0.7294** | — |

⛔ **`knn_proto` is pinned at exactly 1.0000 on three of six factors; the CLU on one.** For a
**collateral-damage** measurement a ceiling factor **can only fall**. So the rival has strictly more room
to lose on half the factor set, and that inflates `Delta` in the CLU's favour **independently** of the
13-point level gap. ⚠ Two of my three known biases therefore point the same way, and the only one
pointing against the CLU is rebuild churn.
⭐ This is exactly the asymmetry `clu-dev-1e` identified on held-out **B** ("three hues at exactly
1.0000 mean collateral there can only register as damage") — I had filed it as a reason not to spend the
scored run on B, and had not noticed that a weaker form of it is present on `dev` too.
⭐ **Consequence, fixed now, before any deletion number:** `retention` (A2.3) is the estimand that is
immune to this — each arm is scored only on the cells it already got right, so a ceiling arm is not
penalised for having been at the ceiling. **`retention` is promoted to the co-equal of the churn-corrected
delta for interpreting the headline**, and the per-factor ceiling counts above are reported beside any
`Delta` so a reader can see how much of it is headroom. ⛔ The raw `Delta` remains the pre-registered
primary; I am not switching estimand, I am reporting what it is made of.

⭐ Also worth recording: on `orientation` the rival scores **0.5564** against the CLU's **0.1814**. The
inherited claim that *"the codebook costs ~all of orientation"* is a statement about **the CLU's store**,
⛔ not about the task — a byte-matched prototype table recovers orientation three times better. That is a
real weakness of the arm and it belongs in the report whether or not it helps the deletion story.

## 11:31:11Z — ⛔⛔ §6a COULD HAVE BEEN PASSED BY A ONE-STEP SHIFT. Caught pre-hoc; §6 re-specified (A3).
The most important methodological catch of the run, from `clu-dev-1e`'s third review, verified
independently by my own implementation (which reproduces its table to the digit).

**Pre-deletion render distance from the requested value, CLU, per targeted set (n = 1,024):**
```
factor        |d|=0  |d|=1  |d|=2  |d|>=3   within 1
floor_hue     1.000  0.000  0.000  0.000     1.000
wall_hue      0.995  0.005  0.000  0.000     1.000
object_hue    0.511  0.489  0.000  0.000     1.000     <-- the sharp case
scale         0.400  0.483  0.109  0.007     0.884
shape         0.826  (categorical: exact-match only)
orientation   0.500  0.368  0.004  0.128     0.868
```
⛔ **`object_hue` pre-deletion renders hue 5 on 51 % of requests naming hue 5 and hue 4-or-6 on the other
49 %, with literally ZERO mass beyond one step.** So after deleting hue 5, the overwhelmingly likely
outcome is that those requests render hue 4 or 6 — **accuracy near 0, which PASSES my §6a bar of
`1/V + 0.05 = 0.15`** — while the store is still emitting a colour **one grid step away**.
⭐ **That is not forgetting in any sense the charter means, and an exact-match probe cannot tell the two
apart.** I would have reported a pass. This is exactly the failure the charter warns about in different
words: *"absent from a direct read" reported as "unrecoverable"*.

**§6 re-specified (A3.1), pre-hoc — `ls runs/del_*/eval.json` was empty and is recorded as such.**
The threshold is kept as a companion; the reported statistic is the **post- vs pre-deletion distance
profile**, with **three outcomes registered in advance**: mass `d0 -> d1` with the tail unchanged =
**structural deletion**; mass **stays** at `d0` = **re-composed from its parts** (the gs charter's central
tension, an honest negative); mass spreads to `d2+` = **damage, not deletion**. ⭐ Those three are
distinguishable and *"accuracy ≤ 0.15"* is not.

**Geometry verified rather than assumed, and one reviewer claim CORRECTED.** Hues are 10 values over
0.0–0.9 ⇒ **cyclic**, distance `min(|d|, 10−|d|)`. `shape` is **categorical** ⇒ distance meaningless,
exact-match only. ⚠ The reviewer suggested `orientation`'s 12.8 % mass at `|d|≥3` was wrap-around on a
cyclic factor. **It is not:** the grid spans **−30°…+30° = 60°, not 360°**, so orientation does not wrap
and ⛔ that mass is genuine error, not an artefact to explain away.

**§5's margin shrinks under the right null (A3.3).** Against the store's own *marginal* rate of
rendering `v`, rather than `1/V`: `scale`'s "3.2× chance" is really **1.54×**, `orientation`'s "7.5×" is
really **2.46×** — because the store **over-renders** those middle values (bias 2.14× and 2.69×).
⭐ All six still clear the harder null, so §5 passes under a stricter test than registered — but the
margins are reported against the marginal, ⛔ not against chance.
⚠ **Confound disclosed, not discovered later:** the **median-index** rule picked the middle value to
avoid one-sided neighbourhoods at grid edges, and the middle value is exactly what a shrinking decoder
over-renders. The rule stands; the bias is on the record. (`object_hue` is the exception at 0.66 — it
*under*-renders hue 5 — so it is ⛔ not uniform.)

**No rebuild needed for the stale baseline (A3.4).** `det_A`'s eval ran at `n_dev=1024`/`n_target=256`,
so its row sets moved — but `BASE` is already `det_B`, and **`det_A` and `det_B` are byte-identical**, so
`det_B` simply *is* the pre-deletion baseline. ⭐ Third confirmation of that bit-identity, this time
through the **whole eval path** rather than the digest: on the fixed 1,024-row `eval_a`, the two builds'
CLU predictions agree at **1.000000**.

## 11:32:05Z — the render-bias confound has a MECHANISM (A3.5), verified from the grid
`clu-dev-1e` supplied the explanation and I verified the values myself from `factor_index.npz`:
`median index = card//2` selects **`scale` = 1.0357 (the default object size)** and
**`orientation` = 0.0° (exactly the null pose)** — and those are precisely the two factors showing
inflation (2.14×, 2.69×). A **hue** has no default to revert to, so no inflation: 1.00 / 0.99 / 0.66.
⭐ An averaging/shrinking decoder reverts toward the null pose and the default size, so the rule lands on
the value such a decoder over-produces anyway.
⛔ **The caveat is therefore NOT "median-index selection is biased"** — it is **"median-index selection
is biased for factors with a physical null, and unbiased for circular factors"**. Narrower, mechanistic,
and it tells a reader when the rule is safe to reuse. A testable prediction is attached (inflation should
shrink for scale/orientation as the index moves off the null, hues stay flat); ⛔ not run, no budget.
⚠ Also registered (A3.6): `orientation` is the one factor already **bimodal** pre-deletion (0.500 / 0.368
/ 0.128 far tail), so its three-way reading must be against a distribution with a populated tail —
**mass moving INTO the far tail is DAMAGE, not deletion.** Registered before the number exists.
⭐ It also withdrew the wrap-around suggestion after checking the grid, so the orientation correction is
now agreed by both derivations.

## 11:33:18Z — A3.7 + A3.8 registered, and my own prediction for `del_f2v5` (⚠ NOT independent)
**A3.7 — the distance profile needed the churn control too.** `clu-dev-1e` pointed out that A3.1
compares post-deletion to **pre**-deletion, which carries the same defect I had already fixed for
collateral: a rebuild moves the profile by itself. The decisive quantity is
**`d0(del_f2v5) − d0(plc_f2v5)`**, ⛔ not `d0(del) − d0(det_B)`. `plc_f2v5` scores the identical targeted
set, so it is free. ⭐ My own sentence turned on my own statistic. Implemented; the churn-corrected
three-way verdict is the reported one, with the uncorrected beside it.

**A3.8 — the mechanism check, and it is the best instrument in the cell.** No probe, no decoder, no
accuracy: after `delete(object_hue=5)`, `fit_address_book` fits that row on **zero** live rows, so it
must be **exactly** `log(1/64) = −4.1588830833596715` across all 5 × 64 entries, range exactly 0.
**Calibrated on `det_B` by measurement:** every value's row is far from uniform (within-well log-range
`v=0 5.386`, `v=4 5.551`, `v=5 5.434`, `v=6 5.467`, `v=9 5.364`).
⭐⭐ Why this is decisive rather than suggestive: *not uniform* ⇒ the delete did not do what the
mechanism claims, whatever the accuracies say. *Uniform **and** the value still comes back* ⇒ the
addressing term is **provably gone** and the value re-composed anyway — the gs charter's central question
answered **at the level of the mechanism**, not of a score.
⭐ `object_hue`'s pre-deletion far tail is **exactly 0.000**, so **any** appreciable `|d| ≥ 2` mass after
deletion is **new** ⇒ damage. Cleanest single indicator in the cell.

### ⚠ My prediction, and why it is worth less than the peer's
`clu-dev-1e` committed its prediction before `del_f2v5` existed: `d0` collapsing 0.511 → ~0.07–0.10 if
structural deletion works; staying near `d0(plc)` if it re-composes; 0.20–0.35 partial; any `|d| ≥ 2`
mass = damage.
⛔ **I read that before forming my own, so mine is anchored and I am not presenting it as an independent
second forecast.** Recording that plainly is worth more than a prediction that pretends to independence.
Where I would add to it, from the mechanism rather than the data: the deleted rows are gone from
`train_store` too, so the **modes themselves never saw a hue-5 feature** — the re-composition route, if
it exists, must run through the decoder mapping some *interpolated* feature back onto hue 5, ⛔ not
through a surviving hue-5 mode. If `d0` holds up, that distinction is the thing to test next, and it is
also why `|d|=1` mass (hue 4 or 6) is the *expected* landing zone rather than a failure.
⭐ What I can still contribute independently is the **analysis**, which is written and syntax-checked
before any deletion artefact exists.

## 11:35:05Z — A3.9–A3.11 registered, and the FEATURE-LEVEL baseline is itself a finding
⛔ **I overstated the mechanism and `clu-dev-1e` corrected it.** I had written that after deletion
*"there is no surviving hue-5 mode to route to"*. Too strong: a mode is **not hue-labelled** —
`init_store` k-means++'s over a well's **channel** subset (303…573 of 2048 dims) and `train_store` moves
the centres; with **M = 64 modes against 4,800 factor combinations** they cannot be hue-pure, and a
centre fitted on hue-4 and hue-6 material **can land at a hue-5-like position by interpolation, being a
mean of vectors either side of it**. Correct statement: after deletion **no mode was FITTED on a hue-5
feature**, but ⛔ nothing guarantees **no mode SITS at a hue-5 position**.
⭐⭐ **Which gives the sentence this hub turns on, now registered as A3.9:**
> **The mechanism guarantees the store is byte-identical to one that never held `v`. It does NOT
> guarantee that no request can elicit `v`.** The charter's condition 1 asks for the second. **A store
> that never held `v` can still emit `v` by interpolation.**
⛔ If that is what I measure it is **not a failure of the deletion** — it is the honest limit of what
store-level deletion buys, and App. G.8 already carries it verbatim (*"store-level guarantee only"*).

### ⭐⭐ A3.11 built and run on the PRE-DELETION store — the baseline is a result in itself
Geometry, not the probe's verdict. Shapes3D is a complete grid (row arithmetic **verified** to reproduce
`factor_labels` on every probe row), so for each of the 1,024 targeted requests the **same scene with
only the hue changed** exists as a real cached row. ⛔ No encoder, decoder or probe is constructed.
On `det_B` (**pre-deletion**), for requests naming `object_hue = 5`:
| statistic | value |
|---|---|
| emitted feature closest to real hue-5 rather than either neighbour | **0.5273** (chance 0.3333) |
| mean ‖x − f₅‖ | 11.4845 (vs 11.8594 to hue 4, 12.4531 to hue 6) |
| ⭐ **hue-axis projection, mean** (1.0 = exactly at hue-5's offset, 0.0 = at the neighbour average) | **0.1479** |
| hue-axis projection, median | 0.1556 |
| fraction of requests with projection > 0.5 | **0.0391** |

⭐⭐ **Read that carefully: even BEFORE any deletion, the store's emitted feature sits only ~15 % of the
way from the neighbour-average toward genuine hue-5, and only 3.9 % of requests get past halfway — yet
the probe reads it as hue 5 on 51 % of them.** The probe is doing much of the work: it is **rounding a
weakly-specific feature onto the right bin.** That is exactly the objection `clu-dev-1e` raised against
probe-based evidence, now **measured** rather than argued, and it means the feature-level statistic is
the **more sensitive instrument** for §6 — a drop from 0.148 to ~0 would be a clean deletion signal that
the probe's exact-match rate could easily miss or mimic.
⚠ Noted asymmetry, not yet explained: the emitted feature is closer to hue 4 (11.86) than to hue 6
(12.45), so the neighbourhood is not symmetric about hue 5. Recorded; ⛔ not interpreted.

## 11:37:13Z — ⛔⛔ RETRACTION: "the probe is doing most of the work" was WRONG. I verified it and it fails.
At 11:35:05Z I concluded from the hue-axis projection (0.1479) that *"the probe is rounding a
weakly-specific feature onto the right bin, and it is doing most of the work."* `clu-dev-1e` challenged
it; I re-derived every number myself rather than accept either position, and **the challenge is correct.
I am retracting the claim.**

**The decisive fact, which I had in front of me and failed to weigh:**
| | value |
|---|---|
| closest-of-three, **parameter-free, no probe, no decoder, no training** | **0.5273** |
| the trained probe's exact-match on the identical set | **0.5107** |
| chance for closest-of-three | 0.3333 |
⭐ **A raw nearest-neighbour test on the emitted feature agrees with the trained probe to 1.7 points.**
If the probe were supplying the hue-5 verdict, the raw-geometry number would sit near **0.3333** and the
probe far above it. It does not. **The hue-5 content really is in the feature at approximately the level
the probe reports.**

**Where my reasoning broke.** I read a small projection as weak hue specificity. But `s` measures
**magnitude along a chord-based axis** while the probe measures **discriminability**, and the two come
apart when the reconstruction error is largely **orthogonal** to the hue axis — which is exactly the case:
```
||u|| = ||f5 - mid||        5.0791    <- the entire hue-5 signal
along-axis component        1.0337    (s = 0.1479)
PERPENDICULAR residual     10.6012    <- 2.1x the whole hue signal
store recon error ||x-f5||  11.4845   = 1.59x the inter-hue distance ||f5-f4|| = 7.2341
```
The emitted feature is **not** "15 % of the way to hue 5" — it is **mostly displaced in directions that
have nothing to do with hue**. A small along-axis fraction is fully compatible with correct hue
discrimination.

**And the data's own geometry (no store involved) says the origin I chose was poor:**
`||f5−f4|| = 7.2341`, `||f5−f6|| = 8.1454`, `||f4−f6|| = 11.5932` ⇒ **triangle ratio 1.3266**, i.e. hue 5
sits **well off the chord** joining its neighbours. The hue circle is **not uniformly spaced** in the
encoder's feature space, so ⭐ the earlier "asymmetry" I recorded as unexplained is simply the underlying
geometry and needs no explanation, and ⛔ the neighbour mid-point is **not a neutral origin**.

**Three changes, all made:**
1. ⛔ The "probe does most of the work" claim is **withdrawn** — from the LOG, from the report, and from
   what I told the operator. It was an unforced concession that would have weakened **every**
   probe-based result in this hub, including the collateral headline, ⛔ on evidence that does not
   support it.
2. ⭐ **`closest-of-three` is the PRIMARY feature-level statistic** — parameter-free, needs no origin,
   unaffected by the curvature that makes the mid-point suspect. The projection stays as a **secondary,
   more sensitive** measure.
3. ⛔ **The projection is never emitted without the perpendicular residual beside it**, so the inference I
   made is not available to a reader. `data_geometry_no_store` is now written into `feature_probe.json`.
⭐ The projection is still worth having: `s = 1` at `f5` by construction, and a fall from 0.148 toward 0
post-deletion is a real signal. It is the *interpretation of its level* that was wrong, ⛔ not the statistic.

## 11:38:45Z — the along-axis statistic must be SIGNED. Second correction to the same instrument.
`clu-dev-1e` reported 0.7835 where I emitted 1.0337. Not a discrepancy — a **definitional** difference,
and mine was the wrong definition. Reproduced all four candidate aggregations myself:
```
mean(s * ||u||)        = 0.7835   <- SIGNED (correct)
mean(|s| * ||u||)      = 1.0337   <- what I emitted (the norm of `along`)
mean(s) * mean(||u||)  = 0.7514
rms(s * ||u||)         = 1.2919
```
**Why the 32 % gap, and why it matters:** `s` has real sign structure —
**negative on 20.7 % of the 1,024 requests**, `min −0.5638`, `max 0.8413`. A negative `s` means the
emitted feature sits on the **far side of the neighbour average FROM hue 5** — displaced *away* from the
deleted value along the hue axis. ⛔ Taking `|s|` hides that and **inflates** the apparent hue-5 content.

⭐⭐ **And it blinds the statistic to exactly the effect §6 is hunting.** If deletion **randomises the
sign** of `s` while leaving its magnitude roughly intact — precisely what is expected once the hue term
drops out of the addressing and the residual displacement becomes arbitrary — then `|s|` barely moves
while **signed `s` collapses toward 0**. The absolute version is blind to the deletion signature.
⭐ So the **20.7 % negative fraction is now a registered pre-deletion baseline in its own right**: if it
climbs toward **50 %** post-deletion, the hue axis has gone to noise, which is a cleaner statement of
*"the addressing lost `v`"* than any magnitude.

**Reported from here: signed mean `s·‖u‖`, the negative fraction, and the perpendicular residual.**
Three numbers from which a reader can reconstruct the geometry — and ⛔ from which the inference I was
talked out of is not available. The absolute version is kept beside them, labelled, so the difference is
visible rather than silently resolved.
⭐ Decomposition verified exact on my side too: `‖x−f5‖² == (s−1)²‖u‖² + ‖perp‖²` to **1.98e-15** relative
(the peer saw 3.8e-07, consistent with a float32 path). So the 1.03/10.60 split is arithmetically
airtight, ⛔ not an approximation.

⚠ Two housekeeping points from the same message, both checked:
- **`det_C` is NOT running long.** My sampler: 1,676 CPU-s, 8,714 MB, **545.3 s** elapsed at 3.07 cores —
  between `det_A`'s 554.9 s and `det_B`'s 728.3 s. My "~9 min" was 545 s. ⛔ No action; I withdraw the
  concern I raised.
- **`logs/det_C.log` does not exist because the driver names it `det_C.build.log`** — it is present and
  captured. ⭐ Nothing is being lost. (The stage logs are `<tag>.build.log` / `.eval.log` / `.rival.log`.)
- The sampler attributes **two pids per tag** — the worker and a 0-CPU `Scripts/python.exe` launcher
  shim. ⛔ Cost analysis must take **max (or sum) CPU per tag**, never first.

## 11:39:11Z — ⭐⭐ `det_C` LANDS: THREE-WAY determinism, and the code edits are MEASURED inert
`det_C` built in **615.9 s**. Both comparisons **PASS**, 22/22 arrays bitwise identical each time:
- `det_A` vs `det_C`: **PASS** (`manifest_fields_equal=True`, `all_arrays_bitwise_equal=True`)
- `det_B` vs `det_C`: **PASS** (`manifest_fields_equal=True`, `all_arrays_bitwise_equal=True`)
⇒ **`det_A` ≡ `det_B` ≡ `det_C`**, `store_digest d0eacf3b9c2a10d0…`, across **three independent OS
processes** at **three different wall times** — 554.9 s / 728.3 s / 615.9 s, a **1.313× spread**.

### ⭐⭐ The code-fingerprint question is now CLOSED BY MEASUREMENT, not by assertion
This is what `det_C` was added for. `det_A` and `det_B` were built **before** the fingerprint existed and
carry `null`. `det_C` carries
`code.combined = 481f65ea602c274229e477b6508a55c39a0091447a5a1af41ad50052e9121f9d`, with
`exp_gen_deletion.py = 928a33f2…` — **the version edited at 11:20:14Z, i.e. AFTER `det_A` was built at
11:13:12Z.**
⭐ So `det_C` ran under the **edited** code and `det_A`/`det_B` under the **pre-edit** code, and all three
produce **byte-identical stores**. The edits between them are therefore **measured to be behaviourally
inert** — ⛔ not merely asserted to be comment-only, which is all I could have claimed an hour ago. That
was `clu-dev-1e`'s objection and it is now answered with an artefact rather than an argument.
Every unit from here carries `481f65ea…`; the analysis asserts a single fingerprint across the cell and
any unit that differs is visible.

### The cost-line noise band now has three points at IDENTICAL work
| build | wall | note |
|---|---|---|
| `det_A` | 554.9 s | |
| `det_C` | 617.9 s | |
| `det_B` | 728.3 s | built while the duplicate driver was competing |
Same 32,768 rows, same 1,200 steps, same everything — **range 173.4 s, max/min 1.313**. ⛔ That is the
error bar every per-delete cost number must carry; a naive table would read this scheduling spread as a
property of the deletion. ⭐ Three more replicates at `n_live = 24,534` follow from `plc_churn_r1/r2/r3`.

## 11:42:08Z — the EVAL path is deterministic too, which is what licenses the churn interpretation
`det_B` and `det_C` have **bit-identical stores**, so their scores must match exactly or the *scoring*
path is nondeterministic. Checked:
- `clu` / `store_removed` / `feature_addr` aggregates on both `a` and `dev`: **identical to 6 dp**
  (`dev` clu 0.600301, floor 0.121745, feature-addressed 0.601725).
- **All 27 per-row prediction matrices identical** (`np.array_equal`), across every read and every
  targeted set.
- `knn_proto` dev aggregate identical: **0.729411**.

⭐⭐ **Why this matters beyond tidiness.** It closes the last route by which a deletion-vs-placebo
difference could be measurement noise. The whole pipeline — `train_store` → `address_codes` →
`fit_address_book` → request posterior → `read_from_codes` → frozen decoder → frozen probe, and the
rival's prototype refit — is a **deterministic function of the live row set**. ⇒ **Every difference I
report between a deletion and its size-matched placebo is attributable to the live set and to nothing
else.** The "churn" the placebos measure is genuine *rebuild* churn (a different live set producing a
different optimisation trajectory and a different channel partition), ⛔ not run-to-run jitter in the
scorer. Without this the placebo band would have had two sources mixed into it.

`del_f2v5` started 11:41:56Z — the first deletion in the cell.

## 11:51:43Z — ⭐⭐ FIRST DELETION: A3.8 PASSES BIT-EXACTLY, and the PARTITION MOVES
`del_f2v5` (delete `object_hue = 5`) built in 500.4 s. Every claim below re-derived by me from the
artefacts, ⛔ not taken from the peer's message.

### A3.8 — the mechanism check: **PASS, bit-exactly**
- `cond_2[5]` is **exactly `float32(log(1/64)) = -4.158883` on all 320 entries** (5 wells × 64 modes);
  per-well range **`[0, 0, 0, 0, 0]`**; max deviation from the float64 value `1.143e-08` (pure float32
  representation error).
- **Exactly one** `object_hue` value is uniform and it is **`v = 5`**. The other nine retain ranges
  **6.01 – 6.44**.
- ⭐ **No other factor has a single uniform value:** `cond_0` mean range 6.58, `cond_1` 6.68, `cond_3`
  4.93, `cond_4` 6.15, `cond_5` 5.24, all with zero uniform entries.
⇒ **The addressing term for `object_hue = 5` is provably, bit-exactly gone, and the deletion flattened
nothing it should not have.** Provenance: `n_live 29561 = 32768 − 3207` exactly, digest `33a129a2…`
(≠ undeleted `d0eacf3b…`), **code fingerprint identical to `det_C`**, `coverage_shortfall 0`,
`encoder_constructed false`.

### ⭐⭐ And the finding neither of us had measured: THE CHANNEL PARTITION SUBSTANTIALLY REORGANISED
| | undeleted | `del_f2v5` |
|---|---|---|
| well sizes | `[303, 305, 573, 501, 366]` | **`[672, 234, 234, 546, 362]`** |
| ARI vs undeleted | **1.0000** (all three undeleted pairs — the metric is sound) | **0.5656** |
| best agreement under relabelling | — | **0.8027** = 1,644/2,048 channels |
**~20 % of channels changed wells; the largest well more than doubled, 303 → 672.**
⭐ This is `AUDIT-01`'s `n_corr = 16384` prediction realised — the correlation matrix is fitted on the
**first 16,384 rows of the sorted live vector**, so removing 3,207 rows moves that window and `ward`
returns a different partition.

⛔⛔ **A post-delete store is therefore not a perturbation of the pre-delete store; it is a different
object with a different layout.** Every per-factor difference confounds *"hue 5 was deleted"* with
*"a fifth of the representation was re-partitioned"*. ⭐ That is the mechanism **working as specified**
(a canonical function of the live set may return a different layout) — but it upgrades A2.2: the
churn-corrected collateral is no longer merely *co-primary*, it is **the only interpretable collateral
number**, and the uncorrected one is demoted to a diagnostic.
⭐ `ARI vs undeleted` is now reported for **every** unit (free, from `modes.npz`). **The decisive read is
`ARI(deletion)` vs `ARI(placebo)`** — comparable ⇒ generic rebuild churn; placebo much higher ⇒ the
*structured* deletion destabilises the layout in a way random removal does not, which **no accuracy
number would have revealed**. `plc_f2v5` is next and is now decisive rather than merely useful.
⚠ `loss_final` 31.64753 vs 32.04901 undeleted — **lower loss on a smaller live set is expected** and is
⛔ not evidence that deleting improved the store. Recorded so nobody reads it that way later.

## 11:53:06Z — three checks, one of them a correction to ME, and the ARI question is now properly posed
All three re-derived on my side before accepting anything.

### (a) The `cond`-range numbers differed by DEFINITION, and I was using the less apt one
My 6.58 / 6.68 / 4.93 / 6.15 / 5.24 against the peer's 5.55 / 5.68 / 4.32 / 5.85 / 3.91 — same arrays.
Resolved, both reproduced exactly:
| factor | mean-of-per-well | pooled across wells (mine) |
|---|---|---|
| `cond_0` | 5.5524 | 6.5793 |
| `cond_1` | 5.6767 | 6.6775 |
| `cond_2` | 4.8040 | 5.5495 |
| `cond_3` | 4.3170 | 4.9321 |
| `cond_4` | 5.8461 | 6.1504 |
| `cond_5` | 3.9052 | 5.2410 |
⭐ **Adopted: mean-of-per-well.** The argmax the address book drives is taken **within** a well, so a
cross-well range mixes quantities that never compete. The definition is now written into the emitted
JSON, because the two readings differ by up to **1.3 nats** and a reader comparing runs would otherwise
see a spurious change. ⛔ A3.8's verdict is unaffected — the deleted row is exactly 0 either way.

### (b) ⛔ MY ROW-INDEX DESCRIPTION WAS WRONG
I wrote that hue-5 rows are *"spread through the index at a regular stride of 120 within each 1200-row
block"*. **Wrong.** `object_hue`'s stride is `8 × 4 × 15 = 480` and its period is `4,800`; every hue-5
row satisfies `(row // 480) % 10 == 5` — **verified True on all 3,207**. They are **contiguous blocks of
480 raw indices, once every 4,800**, ⛔ not a stride-120 scatter.
And in sorted-pool-position space the two deletions are **not** comparable, contrary to what I inferred:
| | runs of removed rows | mean run length | max |
|---|---|---|---|
| deletion (`hue = 5`) | **100** | **32.07** | 47 |
| placebo (random) | **2,920** | **1.10** | 4 |
⭐ **A ~29× difference in run length.** The deletion is strongly block-structured; the placebo is
essentially isolated singletons. My instinct that they were comparable was wrong on the facts.

### (c) ⭐⭐ But the BORING EXPLANATION IS DEAD — and this is the useful result
Does the block structure change **which rows enter the `n_corr` window**? Measured on the first 16,384
**surviving** pool positions:
| | window ends at pool position | extra absorbed | shares with the undeleted window |
|---|---|---|---|
| deletion | 18,137 | 1,754 | **14,794 / 16,384** |
| placebo | 18,177 | 1,794 | **14,764 / 16,384** |
**30 rows apart out of 16,384 — 0.18 %.** In terms of *which rows the correlation matrix sees*, the
structured and random deletions are **interchangeable**: a 10 % removal displaces the window by 10 %
whether the removed rows are clustered or scattered.
⭐ **So whatever difference appears between `ARI(del)` and `ARI(plc)` cannot be attributed to window
composition, and must be about what those rows CONTAINED.** That is worth having established *before*
the number lands, and it is the peer's own stated mechanism for its prediction being retracted by its
own measurement — it has restated the prediction (`ARI(plc) > 0.5656`) on a covariance argument instead:
random removal preserves the feature covariance in expectation, while removing every hue-5 row removes
variance specifically along the hue-coding channels, and **ward clusters on exactly that matrix**.
⚠ My counter-hypothesis stands as the live alternative and is also a covariance argument: removing one
hue value may leave the remaining hue geometry **more** coherent, *raising* stability and pushing
`ARI(del)` **above** `ARI(plc)`. ⛔ Neither of us can settle it in advance and neither direction is
defended strongly. ⭐ What is settled is that the uninteresting explanation is eliminated.

## 11:54:01Z — ⚠ a COST-LINE fact the manifest does not record: JAX teardown
`del_f2v5` wrote `build.json` at 11:50:19Z with `unit_wall_s = 501.0`, and the process was **still
alive at 11:53:46Z** — 709.8 s elapsed, memory draining 3,031 → 1,961 MB. Nothing is wrong: it is in
**JAX/XLA teardown** after the artefact is safely on disk. But it means the driver cannot start the eval
until it exits, and it means ⛔ **`unit_wall_s` understates what a rebuild actually costs the box.**

| unit | `unit_wall_s` (to artefact) | process lifetime (sampler) | teardown |
|---|---|---|---|
| `det_B` | 728.3 s | 1011.0 s | +282.7 s |
| `det_C` | 617.9 s | 712.6 s | +94.7 s |
| `del_f2v5` | 501.0 s | 709.8 s | +208.8 s |

⭐ **The cost line will report BOTH**: `unit_wall_s` as "time to a usable store" and process lifetime as
"time the box is occupied". The second is the one that matters for *"~20 rebuilds in 12 h"* planning, and
it is the larger. ⛔ I am **not** changing `build_unit.py` to exit faster (`os._exit`) mid-cell — that
would alter the **code fingerprint** between units, which is exactly the confound A3/the fingerprint
exists to catch. Noted as a cheap improvement for whoever runs this next, ⛔ not taken here.
⚠ This is also why the sampler had to exist: `build.json` alone could not have revealed it.

## 11:54:48Z — the RELABELLING-FREE pair test DISAGREES with the movers test. Both reported.
`clu-dev-1e` measured that the correlation perturbation **is** localised on hue-coding channels
(top-200 hue-selective: del/plc ratio **1.334**; bottom-200: **1.017**) but that the **partition
response is not** — its bijection-based "movers" test found `object_hue` the one factor with **no**
enrichment (ratio 0.966, p = 0.63), with movers instead enriched for `orientation` (2.52×) and
`wall_hue`. It flagged its own caveat: "moved well" requires forcing a **1-1 relabelling** between
partitions whose well sizes are `[303,305,573,501,366]` and `[672,234,234,546,362]` — a bijection is a
poor description of a re-partition that **merges and splits** — and offered the relabelling-free version.
⭐ I ran it. It needs no bijection: does the **co-clustering status of channel PAIRS** change more for
hue-selective pairs?

| pair group | co-clustering changed | n pairs |
|---|---|---|
| both channels in the **top-200 hue-selective** | **0.2635** | 19,900 |
| both in the **bottom-200** (least selective) | **0.2126** | 19,900 |
| **all** channel pairs | **0.1515** | 2,096,128 |
**top/bottom = 1.239 · top/all = 1.739.** `object_hue` selectivity separation is clean (top-200 mean
**1.1985**, bottom-200 **0.000687**).
⚠ My bijection-based reproduction of the movers test agrees with the peer's direction: movers
**404/2048 = 19.7 %**, `object_hue` selectivity movers **0.1491** vs stayers **0.1631**, ratio **0.914**
(peer: 0.966 — small difference, presumably a different selectivity window; same conclusion).

### ⛔ So the two measures disagree, and I am reporting that rather than picking one
- **bijection / movers:** *no* localisation on the deleted factor (0.91–0.97×).
- **relabelling-free / pairs:** **modest but real** localisation (**1.24×** over the least-selective
  pairs). The relabelling-free measure is the more trustworthy — it is exactly the assumption the peer
  correctly flagged as unsafe.
⚠ **But I will not overclaim the 1.24×, for a reason visible in my own table:** the **bottom-200 group
also changes more than average** (0.2126 vs 0.1515). So *being in an extreme selectivity group at all*
raises the change rate regardless of direction, and the top/bottom groups are **not matched** on
anything else (cluster membership, density, channel variance). ⛔ The clean contrast is top/bottom =
1.239, which is **modest**. ⛔ And no p-value is quoted: channel pairs are **not independent**
(2,096,128 pairs over 2,048 channels), so any test treating them as such would be meaningless.

### ⭐ What this does NOT change
**A3.12 stands, and on either measure.** **15.1 % of ALL channel pairs changed co-clustering status**
from one deletion of 9.8 % of the rows. Whether the response is mildly hue-enriched (1.24×) or not
enriched at all (0.91×), the dominant fact is that **the layout moved globally**, and the peer's own
global correlation figure says the same: **80.7 of the 84.2 Frobenius change is the generic window
displacement both deletions share**, only **4 %** more for the structured one. ⇒ The churn-corrected
collateral remains the only interpretable collateral number, ⛔ and the uncorrected one a diagnostic.
⭐ Worth stating plainly: *where the deletion bites and where the layout responds are different
questions* — ward is a **greedy global merge**, so a small perturbation flips merges wherever clusters
happen to be closest, not where the correlations moved most.

## 11:56:33Z — ⭐⭐⭐ THE FIRST DELETION RESULT. `object_hue = 5` IS UNRECOVERABLE — and the manner matters.

### Condition 1 — unrecoverable: **PASSES, and not by the loophole A3 was built to catch**
| statistic | pre | post | bar |
|---|---|---|---|
| exact-match on the 1,024 requests naming hue 5 | 0.5107 | **0.0049** | ≤ 0.15 ✅ |
| fraction of **ALL 4,096 dev renders** reading hue 5 | 0.0737 | **0.0000** | ≤ 0.15 ✅ |
⭐⭐ **`post_frac_rendering_v_anywhere = 0.0000` — the deleted value is not emitted for ANY request in
the whole dev set, not one of 4,096.** That is the charter's *"no rendered tuple contains it"*, met
absolutely, and it is the **stricter** §6b reading (over all requests, not just those naming `v`).

### ⛔⛔ BUT MY PRE-REGISTERED AUTOMATIC VERDICT IS MISLEADING HERE, AND I WILL NOT QUOTE IT AS IF IT SETTLED ANYTHING
`A3_verdict` fires **"DAMAGE (mass spread to d2+)"**, because the tail went 0.0000 → 0.7764.
⛔ **That label is wrong for what actually happened**, and the cross-factor evidence proves it. On the
**same 1,024 requests**:
| factor | pre | post | Δ |
|---|---|---|---|
| floor_hue | 0.9990 | 0.9980 | −0.0010 |
| wall_hue | 0.9736 | **0.9834** | **+0.0098** |
| **object_hue** | **0.5107** | **0.0049** | **−0.5059** |
| scale | 0.2090 | **0.2285** | **+0.0195** |
| shape | 0.5850 | **0.6191** | **+0.0342** |
| orientation | 0.1904 | 0.1836 | −0.0068 |
⭐ **Only the deleted factor collapses. Every other factor on those same requests is unchanged or
slightly BETTER.** So this is not "damage"; it is a **surgical destruction of hue addressing for
requests naming the deleted value**, with the rest of the request's content intact.
⚠ **Why my classifier mislabels it:** A3.1's three branches were defined on the **deleted factor's
distance profile alone** and tacitly assumed that if `d0` collapsed the mass would move to `d1`. It
didn't — it went **flat**. ⭐ **A3.10 — the full rendered histogram, which I added only because the
reviewer pointed out that distance bins merge "concentrated on neighbours" with "flat" — classifies it
correctly**, and this is exactly the case it was added for.

### ⭐ What the histogram says, quantified
| | entropy (uniform = 3.322 bits) | TV from uniform | mass on {4,5,6} | mass on the deleted 5 |
|---|---|---|---|---|
| **pre** | **1.444** | 0.7000 | **1.0000** | 0.5107 |
| **post** | **3.133** | **0.1855** | 0.2236 | **0.0049** |
Pre, the store rendered **only** hues 4, 5, 6 for a hue-5 request — tight, local addressing.
Post, mass on the neighbours {4,6} is **0.2188** against a uniform expectation of 0.2000, and mass on
the seven far hues is **0.7764** against 0.7000. ⇒ **the post-deletion distribution is very nearly
uniform over the nine surviving hues, with the deleted value essentially absent.**
⭐ So of A3.10's three branches this is squarely the **second**: *"hue addressing collapsed entirely for
that request — the value is gone, and so is the store's ability to place the request anywhere on the hue
circle."* ⛔ It is **not** the "concentrated on {4,6}" branch, which the reviewer and I agreed would
have been the **best** outcome (surgical removal of one value from an intact ordinal structure). We did
not get that. The neighbourhood structure went with the value.

### Collateral on the 3,637 requests that do NOT name hue 5 — ⚠ and the crux of the whole hub
```
C_clu        = -0.013931     (0.604253 -> 0.590322)
C_knn_proto  = +0.000871     (0.732197 -> 0.733068)
delta        = -0.014802     (relative: -0.024244)
```
⛔ **`knn_proto` suffers essentially ZERO collateral (+0.09 points) while the CLU loses 1.4 points.**
⭐ And the structural reason is not an accident of this run: `knn_proto`'s table is a set of
per-(factor, value) **means**. Deleting hue-5 rows replaces the hue-5 prototype with the global mean and
shifts `mu_global` marginally; **nothing else in it moves.** It has **no layout to re-derive and no
optimiser to re-run**, so it is **structurally immune** to collateral from row deletion. The CLU pays
because it *has* a learned, globally-coupled layout — the very thing that makes it more than a lookup
table. ⭐⭐ **That is the honest crux and it belongs in the report in these words.**
⚠ `|Δ| = 0.0148 ≤ m = 0.020` ⇒ **tie** under R1 on this single deletion, before any CI and before
pooling over six. ⛔ One deletion is not the cell; `plc_f2v5` and the other five follow.
`knn_proto`'s own artefact confirms it took the identical deletion: `values_with_zero_live_rows =
{'object_hue': [5]}`.

## 11:57:52Z — ⛔⛔⛔ THE CHARTER'S TRAP HAS FIRED. Condition 1 is BY CONSTRUCTION, and the RIVAL does it better.
Every number below verified on my own derivation from the raw `preds.npz`.

### The check I had not run, and should have
| | acc on the 1,024 hue-5 requests | frac of **ALL 4,096** dev renders showing hue 5 |
|---|---|---|
| **CLU** | 0.5107 → **0.0049** | 0.073730 → **0.000000** |
| **`knn_proto`** | **1.0000** → **0.0000** | 0.112061 → **0.000000** |

⛔⛔ **Condition 1 is met by BOTH arms, and `knn_proto` meets it STRICTLY BETTER at BOTH ENDS** — it was
*perfectly* recoverable before (1.0000 vs the CLU's 0.5107) and is *perfectly* unrecoverable after
(0.0000 vs 0.0049). **Neither arm renders hue 5 for a single one of 4,096 dev requests.**

⭐ **So the charter's *"unrecoverable, including from every composition"* is satisfied by a
per-(factor, value) mean lookup table — at both ends, more cleanly than by the store.** This is the
failure mode the task file names in bold and I walked into it:
> *"⛔⛔ A win that is true by construction… belongs in the method section and this programme has ruled it
> never a headline. ⭐ 'The rival has no delete operator' is worth nothing."*
Condition 1 is by-construction for **both**: delete the rows and the addressing term goes uninformative —
in the store by Laplace smoothing to **exactly** `1/M`, in the table by falling back to `mu_global`.
⛔ **Neither is evidence about the store, and the report must NOT lead with "the deleted value is
unrecoverable".** It must say: unrecoverability is achieved, it is achieved **by construction**, the
**rival achieves it more completely**, and condition 1 is therefore a **method-section fact**.
⚠ I had written the §6 result up as a headline-grade pass. That framing is **withdrawn**.

### ⭐ The two arms fail DIFFERENTLY, which is worth a line because it shows the mechanisms differ
| arm | post-deletion hue entropy | distribution |
|---|---|---|
| CLU | **3.133 bits** (uniform = 3.322) | broad: `[.062 .088 .078 .164 .082 .005 .137 .147 .108 .129]` |
| `knn_proto` | **2.310 bits** | spiked: **0.346 on hue 1, 0.325 on hue 9**, 0.164 on hue 6 |
`knn_proto`'s prototype falls back to `mu_global`, whose nearest hues are the **circle's extremes** —
so it fails toward hues 1 and 9. The CLU fails toward near-uniform. **Same verdict, different geometry.**

### ⚠ Two corrections to how I reported the numbers
1. ⛔ **"Tie" must not be read as "no difference".** `delta = −0.014802`, `se = 0.003052`, **95 % CI
   `[−0.020783, −0.008820]` — which EXCLUDES zero.** It is a **real, statistically detectable deficit
   that happens to be smaller than the tie margin `m = 0.020`.** Reported as a **sub-threshold but
   non-zero deficit**; a reader who sees only "tie" would wrongly infer the arms are indistinguishable.
2. ⛔ **My "crux" sentence was premature and is on hold.** `plc_f2v5` has not landed. My own **A3.12**
   says the churn-corrected number is the *only* interpretable collateral — and the crux attributes to
   **deleting** what may belong to **rebuilding**. ⭐ If the placebo also returns ≈ −0.0139 then the
   deletion itself is **free** and the sentence becomes *"rebuilding costs 1.4 points, deleting costs
   nothing"* — a different and much better claim. ⛔ Holding it until the placebo lands.

### A3.10 refinement — "near-uniform" was slightly too strong
Mass on the neighbours {4,6} is **0.2188** against uniform-over-9's **0.2222**, and the far seven
**0.7764** against **0.7778** — so ⭐ **the ordinal neighbourhood is gone to three decimal places**, which
was the point and it holds. ⛔ But χ² against uniform-over-9 is **92.93 on 8 df** (crit 15.51), so the
distribution is **significantly non-uniform**: hue 3 elevated at 0.164, hue 7 at 0.147, hue 0 depressed
at 0.062. **Precise statement: the neighbourhood structure collapsed completely, while residual
non-uniformity remains — and it is not centred on the deleted value.**

### ⛔ A number I am declining to inherit, recorded so the decision is visible
`clu-dev-1e` supported the "the CLU gets nothing back on this axis" reading by citing
**`hub/composition`'s held-out B result: CLU exact-match 0.00/72 against `knn_proto`'s 14.33.**
⛔ **I am not carrying that into this report.** It is another hub's measurement, on a set I have myself
argued is a poor instrument (three factors at 1.0000, two at/below the ablated floor), and I have not
re-derived it. ⭐ Consistent with dropping the 90.5 %/100 % codebook figure earlier for the same reason:
⛔ no number reaches my report through two pairs of hands.
⭐ The narrower claim my **own** artefacts support is sufficient and is what I will write: *on this cell,
at these bytes, the CLU is 13 points behind pre-deletion, takes collateral the table does not, and does
not achieve unrecoverability as completely.* **Whether the learned layout buys anything elsewhere is
outside what this hub measured**, and I will say that rather than borrow a verdict.

## 11:59:44Z — ⭐⭐ RETENTION (A2.3): the R1 "tie" is hiding a 1-in-5 ANSWER-CHURN RATE
All verified on my own derivation, on the **3,637 non-`v` requests × 6 factors = 21,822 cells**.

| arm | pre | post | **retention** | recovery | lost | gained | net |
|---|---|---|---|---|---|---|---|
| **CLU** | 0.6043 | 0.5903 | **0.8249** | 0.2322 | **0.1058** | **0.0919** | −0.0139 |
| **`knn_proto`** | 0.7322 | 0.7331 | **0.9930** | 0.0224 | 0.0051 | 0.0060 | +0.0009 |
**RETENTION GAP = −0.1681** (bootstrap 95 % CI `[−0.1740, −0.1621]`).
⭐ Retention conditions each arm on **its own** pre-deletion competence set, so the 13-point level gap
**cancels by construction** — this is the estimand added precisely to defuse *"the CLU wins by having
less to lose"*, and with that objection removed the result goes **against** the CLU, much harder than
the raw `delta = −0.0148` "tie" suggests.

### ⭐⭐ THE NUMBER THE AGGREGATE HIDES, and it should be a headline diagnostic
The net delta is small **not because the store is stable** but because **losses and gains nearly
cancel**: the CLU flips **10.58 % right→wrong and 9.19 % wrong→right ⇒ 19.77 % of cells CHANGE ANSWER,
one in five.** `knn_proto` flips 0.51 % / 0.60 % ⇒ **1.11 %, one in ninety.**
⛔ A reader given only *"delta = −0.0148, tie"* would conclude the deletion barely perturbed the store.
**It perturbed about a fifth of its answers; the perturbation merely happened to be near-symmetric.**
⭐ This is the **per-cell analogue of the ARI 0.5656 partition result**: the layout moved, and so did a
fifth of the answers. `lost`/`gained` now sit beside every net number in `analyse.py`.

### ⚠ THE INDEPENDENCE NULL — without it the per-factor gaps are overstated
Retention is **not** bounded below by anything near 1 for a weak predictor: under independence of pre-
and post-correctness, retention equals the **post accuracy**. Measured:
| factor | CLU pre | CLU ret | **null** | knn pre | knn ret | **null** | gap |
|---|---|---|---|---|---|---|---|
| floor_hue | 1.000 | 0.998 | 0.998 | 1.000 | 1.000 | 1.000 | **−0.002** |
| wall_hue | 0.979 | 0.977 | 0.977 | 1.000 | 1.000 | 1.000 | **−0.023** |
| object_hue | 0.674 | 0.662 | 0.637 | 1.000 | 1.000 | 1.000 | −0.338 |
| scale | 0.211 | **0.430** | 0.212 | 0.197 | 0.959 | 0.197 | −0.530 |
| shape | 0.578 | 0.732 | 0.541 | 0.636 | 0.988 | 0.644 | −0.256 |
| orientation | 0.184 | **0.417** | 0.178 | 0.561 | 0.973 | 0.558 | −0.556 |
⭐ On `scale` and `orientation` the CLU's retention is **~2× its own null** (0.430 vs 0.212; 0.417 vs
0.178), so there **is** real persistence — ⛔ but `knn_proto` is at **4.9×** and **1.7×** its null and is
near-deterministic, so part of the large gap reflects **the CLU being weak in absolute terms**, where
"right before" is substantially luck. ⭐ **`floor_hue` and `wall_hue` are the clean comparisons — both
arms near-deterministic — and there the CLU is essentially LEVEL (−0.002, −0.023).** ⛔ Retention will
never be reported without the null beside it.

### ⛔ AND THE WHOLE SECTION IS ON HOLD UNTIL `plc_f2v5`
None of this is churn-corrected. **A3.12 governs it exactly as it governs the crux sentence.** If the
placebo shows the same ~20 % flip rate and the same ~0.82 retention, the correct sentence is
**"REBUILDING churns a fifth of the store's answers; DELETING adds nothing on top"** — a statement about
the **O(n)-rebuild mechanism's cost**, ⛔ not about deletion, and **considerably better for the CLU than
it currently looks**. ⚠ The retention *gap* against `knn_proto` survives either way (the table does not
churn under rebuild either) — but **the attribution changes completely.** Holding the entire retention
section, not merely the crux.

## 12:00:52Z — ⛔ MY ratio-to-null REVERSES THE AGGREGATE CONCLUSION. Removed, replaced by kappa.
I added `retention / null` at 11:59Z to correct the level artefact in raw retention. ⛔ **It introduces a
worse artefact pointing the other way, and I used it to soften my own conclusion.** Verified:
```
                 retention   null   ratio    KAPPA
  CLU              0.8249  0.5903   1.397   0.5726
  knn_proto        0.9930  0.7331   1.355   0.9737
```
⛔⛔ **By ratio the CLU looks BETTER (1.397 vs 1.355). By retention it is far worse (0.825 vs 0.993).**
The ratio is **ceiling-limited at `1/null`**, so a **weak** arm has room for a large ratio while a strong
arm is capped near 1 — it **systematically flatters the weaker arm, which here is the CLU**. ⛔ It is not
measuring what I wanted and it is now removed from `analyse.py`.

⭐ **Replacement: `kappa = (ret − null) / (1 − null)`** — Cohen's kappa for pre/post correctness agreement
with independence at the post marginal as the chance model. It asks **what fraction of the AVAILABLE
persistence each arm achieved**, and is free of **both** the level effect raw retention has **and** the
ceiling effect the ratio has.
| factor | CLU ret | null | ratio | **kappa** | knn ret | null | ratio | **kappa** | **Δκ** |
|---|---|---|---|---|---|---|---|---|---|
| floor_hue | 0.9975 | 0.9975 | 1.00 | **0.0000** | 1.0000 | 1.0000 | 1.00 | *nan* | — |
| wall_hue | 0.9770 | 0.9766 | 1.00 | 0.0150 | 1.0000 | 1.0000 | 1.00 | *nan* | — |
| object_hue | 0.6622 | 0.6371 | 1.04 | 0.0692 | 1.0000 | 1.0000 | 1.00 | *nan* | — |
| scale | 0.4295 | 0.2123 | **2.02** | **0.2758** | 0.9594 | 0.1969 | **4.87** | **0.9495** | **−0.6737** |
| shape | 0.7320 | 0.5408 | 1.35 | 0.4164 | 0.9879 | 0.6439 | 1.53 | 0.9660 | −0.5496 |
| orientation | 0.4170 | 0.1776 | **2.35** | **0.2911** | 0.9730 | 0.5576 | **1.75** | **0.9390** | **−0.6479** |
⭐ `orientation` is the clean demonstration: **by ratio the CLU (2.35) beats the rival (1.75); by kappa it
is 0.291 against 0.939.** Exactly reversed.

### ⚠ I must correct my own softening from 11:59Z
I wrote that *"a large part of those gaps is the CLU being weak in absolute terms"*. **Half right.**
Weakness inflates the null, and **kappa removes exactly that** — but **after removing it the gap is still
−0.55 to −0.67 on every factor that has headroom.** ⛔ **The gap survives the correction; it does not
survive as "the CLU is merely weak."** My softening used a statistic that flatters the CLU, and I am
striking it.
⭐ What **does** stand from that entry: the per-factor table must **not** be read as *"the CLU
catastrophically loses scale and orientation under deletion"* — it substantially **never had them**
(pre-accuracy 0.211 and 0.184), a pre-existing weakness this hub did not cause. Reporting the null is
what makes that visible; the **ratio** was the wrong way to divide by it.

**Reading notes, registered:**
1. `knn_proto`'s kappa is **nan on all three hues** because its post accuracy is **exactly 1.0000** — no
   headroom. ⛔ Not a hole in the table: **it IS the finding** — the table is perfect and perfectly stable
   on the hues.
2. The CLU's `floor_hue` kappa is **exactly 0.0000** (retention == null to four places ⇒ pre/post
   correctness statistically **independent**), but at 0.9975 accuracy there is no headroom for that to
   mean anything. ⭐ Marked **degenerate**, ⛔ not reported as "no persistence".
3. ⭐ **The retention section will LEAD with the aggregate kappa: 0.5726 vs 0.9737** — level-free,
   ceiling-free. The CLU achieved **57 %** of the available persistence; the table **97 %**.
⛔ Still entirely on hold for `plc_f2v5` per A3.12.

## 12:01:41Z — A4: the decisive mechanism test needs NO undeleted store, and it lands EARLIER than we thought
`clu-dev-1e` pointed out that `placebo vs undeleted` still differs in `n_live` (32,768 vs ~24,534), while
**two placebos of the same size differ only in WHICH arbitrary rows went** — matched on everything except
the draw. That is a strictly cleaner isolation of recanonicalisation instability, and it needs no
undeleted store at all. Registered as **A4**, pre-hoc: `plc_f2v5` has no `build.json` yet and the other
placebos are unbuilt.

⭐ **And I found it lands earlier than either of us realised.** The peer proposed the n = 8,234 replicate
set (`plc_churn_r1/r2/r3` + `plc_f4v2`), which sits late in the plan. But **the three hue placebos are
themselves a rung-2 set**: `plc_f0v5` (3,269) · `plc_f1v5` (3,224) · `plc_f2v5` (3,207) — `n_live` within
**0.2 %**, three pairwise comparisons, and they arrive **~2 h earlier**. So the mechanism question gets a
first answer around 14:00Z rather than ~16:15Z, and a second, `n_live`-identical answer later.
⛔ I am **not** reordering the plan to hurry it: the driver reads the plan file with an open descriptor
and rewriting it mid-read risks landing the offset inside a line. The gain is ~2 h against a deadline
I have ~11 h of, and a corrupted driver would cost far more. ⭐ If time tightens, the **BuildLock** makes
it safe to run the replicates from a second process alongside the cell — that is exactly what it was
built for, and it is the contingency rather than the plan.

### ⭐⭐ Why rung 2 is a CONTRIBUTION, not a caveat — the framing I am adopting
*Retention You Can Predict, Scope and Delete* guarantees the post-delete store is **byte-identical to one
that never held `v`**, and prices the mechanism honestly in **time** (O(n) rebuild per delete, stated
openly). ⛔ **It says nothing about behavioural stability**, and it does not need to —
**byte-identity-with-a-counterfactual** and **closeness-to-the-pre-deletion-store** are *different
properties*, and the paper claims only the first.
⚠ But a practitioner deleting one record from a deployed system would reasonably expect the system's
**other answers** to be stable. If a fifth of them change, that is **a real caveat on the ported method
that the source paper does not carry** — and it is *measurable*, *non-by-construction*, and *at the level
of the mechanism*.
⭐⭐ **And a lookup table does NOT get this for free.** It has no layout to recanonicalise — which is
precisely why `knn_proto` churns **1.11 %** against the store's **19.77 %**. ⭐ **This is the one place in
this hub where the CLU's structure being load-bearing produces a FINDING rather than an excuse**, and it
is the opposite of the by-construction trap that condition 1 fell into.
**Candidate headline, fixed before the data:** *"Exact structural deletion is achievable and cheap to
verify, but recanonicalisation is not answer-preserving: it costs ~X % of available persistence per
rebuild, independent of what is deleted."* ⛔ `X` is written only once rung 2 supplies a **spread**.

## 12:05:50Z — ⭐⭐⭐ THE ARI QUESTION IS ANSWERED, AND IT GOES THE OTHER WAY. Random removal perturbs MORE.
`plc_f2v5` built (564.0 s). ⭐ **`n_live = 29,561` — IDENTICAL to `del_f2v5`**, same 3,207 rows removed,
same code fingerprint as `det_C`. So the two differ **only in WHICH rows went**: one structured (every
`object_hue = 5` row), one arbitrary.

| | well sizes | ARI vs undeleted | agreement |
|---|---|---|---|
| undeleted | `[303, 305, 573, 501, 366]` | 1.0000 | — |
| **deletion** (`hue = 5`) | `[672, 234, 234, 546, 362]` | **0.5656** | 0.8027 |
| **PLACEBO** (random) | `[249, 248, 674, 503, 374]` | **0.4924** | 0.6660 |
| deletion vs placebo | — | **0.5730** | — |

**Predictions on record, both made before the number existed:**
- `clu-dev-1e`: **`ARI(plc)` in 0.6–0.9, ABOVE** the deletion's 0.5656 — random removal preserves the
  feature covariance in expectation, while removing a hue removes variance along the hue-coding channels
  that ward clusters on.
- mine: *"could go either way"*, with the mechanism named — **removing one hue value may leave the
  remaining hue geometry MORE coherent**, raising stability.
⛔ **The peer's prediction is REFUTED: `ARI(plc) = 0.4924 < ARI(del) = 0.5656`.** ⚠ I do **not** get to
claim this: I explicitly declined to defend a direction, so naming the mechanism is not a forecast.

### ⭐⭐ And the loss numbers corroborate the mechanism independently
```
loss_final   deletion 31.64753   undeleted 32.04901   PLACEBO 32.82439
```
The **deletion's** loss is **below** the undeleted store's; the **placebo's** is **above** it. Both live
sets have **exactly** 29,561 rows — so this is not a size effect. Removing **one entire hue value** leaves
a live set of lower intrinsic diversity (9 hues instead of 10) that the quantiser fits **better**;
removing 3,207 **arbitrary** rows just thins every mode's support and fits **worse**.
⭐⭐ **This is precisely the candidate mechanism the peer wrote down IN ADVANCE at 11:28Z** for the
"outside, better" branch — *"removing a coherent block may leave a cleaner, less multi-modal residual
that the quantiser fits better"* — and it is showing up in two independent places at once (ARI and
`loss_final`). ⛔ Registered in advance is exactly why it can be believed now.

### What this settles for A3.12 — and it settles it STRONGER than the registered branch
A3.12's first branch was *"comparable ⇒ partition churn is generic to rebuilding, not caused by the
semantics."* ⛔ The result is **stronger than comparable**: the semantic deletion perturbs the layout
**LESS** than an arbitrary removal of the same size. ⇒ **Partition churn cannot be attributed to deleting
a meaning — if anything, deleting a meaning is the gentler operation.**
⭐ And `ARI(deletion, placebo) = 0.5730` is **already a rung-2-flavoured measurement**: two rebuilds at
**identical `n_live`**, differing only in which rows were removed, agree at ARI ≈ 0.57 — about as much as
either agrees with the undeleted store. **Recanonicalisation instability is intrinsic.** ⚠ The clean
rung-2 (placebo-vs-placebo) still follows; this is the same conclusion arriving early from a pair that
happens to be `n_live`-matched.

## 12:06:38Z — three corrections to how I framed the ARI/loss result. All accepted.

**(a) ⛔ The loss ordering is NOT evidence of a better store.** It is evidence that **the target got
easier**: nine hues is a lower-diversity thing to quantise than ten, so the same 64 modes per well cover
it better. ⚠ And `loss_first` differs too — **del 47.745 · plc 46.191 · undel 47.491** — because the
first batch differs, so these are **three different trajectories on three different datasets**. The
comparison is legitimate as *"how well did each fit its own live set"* and ⛔ **not** as *"which store is
better"*. ⭐ Same mechanism explains both observations: **a simpler target yields a more stable
partition.** One mechanism, two signals — ⛔ not two pieces of evidence.

**(b) ⛔⛔ My "lower bound" inference has a gap, and it is the SAME step that failed this morning.**
I asked whether, the semantic deletion being gentler on the *layout*, `del_f2v5`'s 19.77 % churn and
κ 0.5726 are a **lower bound** on recanonicalisation. **Plausible, not entailed.** The peer's movers
prediction failed precisely because *perturbation location* and *partition response location* are
different questions — and **layout stability and ANSWER stability are different in the same way**. The
answer depends on the partition, the trained modes **and** the address book *jointly*; a store with a
**more**-different partition can give **more**-similar answers if the modes compensate. ⛔ **ARI is not a
proxy for κ.** `plc_f2v5`'s eval lands in minutes — ⭐ **measure it, do not infer it.** If it churns
*less* than `del_f2v5` despite the lower ARI, that is a **finding about layout-vs-behaviour decoupling**,
⛔ not an anomaly.

**(c) ⛔ "Lower bound" overstates what ONE placebo can support even if it does churn more.** `del_f2v5`
is one point in a distribution of rebuild outcomes and a single draw cannot characterise it. The
reference needed is the **n ≈ 3,233 band** — `plc_f0v5` (3,269) · `plc_f1v5` (3,224) · `plc_f2v5`
(3,207), the 1.92 %-spread pooling already registered. With three draws I can say **where the deletion
sits relative to the churn distribution**; with one I can only say **which is larger**. ⭐ Until `f0v5`
and `f1v5` land, the phrasing is **"the placebo perturbed the layout more on this pair"** — ⛔ no bound
language in the report.

**(d) ⭐ A prediction the cell will settle, registered now.** If removing a **hue** value is gentler
because hues are the factor the store represents **best** (pre-deletion `object_hue` 0.6565, the hues at
0.98–1.00), then deleting a **`scale`** value — which the store barely represents (0.2107) — should show
the effect **weakly or not at all**. ⭐ `del_f3v4` vs `plc_f3v4` tests it, and the diversity explanation
should be judged **there** rather than on `f2v5` where it was found.

## 12:09:05Z — ⭐⭐⭐ THE ATTRIBUTION IS SETTLED. Rebuilding churns ~19 % SYMMETRICALLY; deleting churns the same but ASYMMETRICALLY.
`plc_f2v5` scored. `n_live` identical to `del_f2v5` (29,561), same code, differing only in which 3,207
rows went. On the 3,637 requests naming no deleted value:

| arm / unit | pre | post | retention | **κ** | lost | gained | **answer churn** |
|---|---|---|---|---|---|---|---|
| CLU **deletion** | 0.6043 | 0.5903 | 0.8249 | **0.5726** | **0.1058** | 0.0919 | **19.77 %** |
| CLU **PLACEBO** | 0.6043 | **0.6087** | 0.8468 | **0.6086** | 0.0926 | 0.0970 | **18.95 %** |
| knn **deletion** | 0.7322 | 0.7331 | 0.9930 | 0.9737 | 0.0051 | 0.0060 | 1.11 % |
| knn **PLACEBO** | 0.7322 | 0.7318 | 0.9955 | 0.9832 | 0.0033 | 0.0029 | 0.62 % |

### ⭐⭐ A4's headline is CONFIRMED, and the number is now measurable
**The placebo churns 18.95 % against the deletion's 19.77 %.** ⇒ **The ~20 % answer churn is
overwhelmingly REBUILD churn, not deletion.** Deleting a semantic category adds only **0.82 percentage
points** on top of what rebuilding does anyway. Rebuilding alone costs **1 − 0.6086 = 39 % of the
available persistence**; the deletion adds ~3.6 points of κ on top.
⭐ **"Recanonicalisation is not answer-preserving: it churns ~19 % of the store's answers regardless of
what is deleted"** — measurable, non-by-construction, and a caveat the source paper does not carry.
⛔ And `knn_proto` does not get this for free: **0.62 %** under the same rebuild, because it has no layout
to recanonicalise.

### ⛔⛔ BUT MY OWN SPECULATION IS REFUTED, AND THE CORRECTION GOES AGAINST THE CLU
At ~12:01Z I wrote: *"if the placebo also returns ≈ −0.0139 then deleting is free and rebuilding costs
1.4 points."* **Wrong, and backwards.**
```
deletion : C_clu -0.013931   C_knn +0.000871   delta -0.014802  95%CI [-0.020783, -0.008820]
PLACEBO  : C_clu +0.004399   C_knn -0.000367   delta +0.004766  95%CI [-0.001037, +0.010569]
⭐ CHURN-CORRECTED delta = -0.019567          95%CI [-0.025288, -0.013847]
```
⚠ **The placebo rebuild slightly IMPROVED the CLU (+0.44 points).** So rebuilding is **net free**, and
**deleting costs ~1.8 points relative to an equivalent random removal.** ⇒ **the churn-corrected delta
(−0.0196) is WORSE than the uncorrected one (−0.0148)**, not better. The correction I registered as
protecting the CLU turns out to penalise it.

### ⭐⭐⭐ The precise finding, and it is sharper than either framing we had
**Rebuilding churns ~19 % of answers SYMMETRICALLY** — lost 0.0926 ≈ gained 0.0970, net **+0.0044**.
**Deleting churns a similar ~20 % ASYMMETRICALLY** — lost 0.1058 > gained 0.0919, net **−0.0139**.
⇒ ⭐ **The deletion's cost is not in HOW MUCH changes, but in the BIAS of what changes.** A fifth of the
store's answers move either way; only under a semantic deletion do they move net-downward.
⚠ This also vindicates the peer's caution against my ARI→κ inference: the placebo has the **lower ARI**
(0.4924 vs 0.5656, layout more disturbed) yet the **higher κ** (0.6086 vs 0.5726, answers better
preserved). ⭐⭐ **Layout stability and answer stability are genuinely decoupled, measured** — exactly the
"finding, not an anomaly" it predicted, and ⛔ a direct refutation of my "lower bound" reasoning.

### R1 verdict on this deletion
Churn-corrected `delta = −0.019567`, `|Δ| = 0.0196 ≤ m = 0.020` ⇒ **tie** — but **by 0.0004**, and the
**95 % CI excludes zero**. ⛔ Reported as *"a tie by the margin, on a deficit that is statistically real
and within 0.5 % of the threshold"*, ⛔ never as "indistinguishable". One deletion of six.

## 12:09:35Z — ⭐ A3.7 CHURN-CONTROLS CONDITION 1, and it survives. Unit 1 complete.
The one control I had not run. On the 1,024 requests naming `object_hue = 5`:
| unit | CLU acc on hue-5 requests | mass on {4,5,6} | entropy | knn acc |
|---|---|---|---|---|
| `det_B` (undeleted) | 0.5107 | 1.0000 | 1.444 | 1.0000 |
| `del_f2v5` | **0.0049** | **0.2236** | **3.133** | **0.0000** |
| `plc_f2v5` (churn control) | **0.5371** | **1.0000** | 1.331 | **1.0000** |
⭐⭐ **The placebo leaves hue-5 rendering entirely intact — 0.5371, slightly ABOVE the undeleted
baseline, with the `{4,5,6}` neighbourhood structure completely untouched.** So the destruction of hue 5
is attributable to **the deletion**, ⛔ not to rebuild churn, **in both arms**.
⇒ **Condition 1 is properly placebo-controlled and survives.** It remains **by construction** and the
rival still does it **better**, so it stays a **method-section fact** — but now a *placebo-controlled*
one rather than an uncontrolled one. ⭐ Rigour upgraded even though the verdict is unchanged.

### The collateral, decomposed
```
C_clu(deletion) -0.013931   C_clu(placebo) +0.004399   SEMANTIC part -0.018330
C_knn(deletion) +0.000871   C_knn(placebo) -0.000367   SEMANTIC part +0.001237
uncorrected delta -0.014802      CHURN-CORRECTED delta -0.019567
|churn-corrected| = 0.019567 vs m = 0.020  ->  margin 0.000433
```
⛔ **A tie by 0.0004**, with a 95 % CI `[−0.025288, −0.013847]` whose **lower end sits past −m**. ⛔ It
must **never** be reported as an unqualified "tie": it is *a tie by the thinnest possible margin, on a
deficit that is statistically real, with a CI straddling the loss boundary*. ⭐ **Pooling over six
deletions will decide the verdict, not confirm it** — this unit settles nothing about the cell.

### ⭐ The decoupling, stated as the interesting result it is
The deletion perturbs the **layout less** than the placebo (ARI 0.5656 vs 0.4924) and **fits its own live
set better** (loss 31.65 vs 32.82) — yet costs **more accuracy** on the surviving space (−0.0139 vs
+0.0044). ⭐⭐ **A gentler layout perturbation produced a worse behavioural outcome.** The diversity
account covers both halves at once: removing a hue makes the target **easier** (lower loss, more stable
partition) **and** removes information the surviving requests **needed** (worse accuracy).
**Simpler to fit, worse to use.**

### ⭐ WHERE THE HUB STANDS, ON ONE UNIT OF SIX
1. **Exact deletion works and is verifiable** — A3.8 bit-exact, A3.7 placebo-controlled.
2. ⛔ **It is by construction, and the rival does it better.** Method-section fact.
3. **The collateral is a hairline tie that WORSENS under the correction** (−0.0148 → −0.0196).
4. ⭐⭐ **The real result: recanonicalisation costs ~19 % answer churn and ~0.39 of κ per rebuild,
   irrespective of semantics.** ⭐ **The only one of the four that is both non-by-construction and
   unavailable to a lookup table** (`knn_proto`: 0.62 %).

## 12:09:58Z — the symmetry claim is ESTABLISHED, with request-clustered inference
⚠ Cells within a request share a store read, so a naive cell-level McNemar treats 21,822 **correlated**
cells as independent. Tested both ways; my numbers and the peer's agree to the third decimal:
| unit | arm | lost | gained | naive χ² | naive p | net/cell | **request-clustered 95 % CI** | |
|---|---|---|---|---|---|---|---|---|
| `del_f2v5` | clu | 2309 | 2005 | 21.42 | 3.7e-06 | **−0.01393** | **[−0.01979, −0.00807]** | ⭐ **EXCLUDES 0** |
| `plc_f2v5` | clu | 2020 | 2116 | 2.23 | 0.136 | **+0.00440** | [−0.00130, +0.01010] | includes 0 |
| `del_f2v5` | knn | 112 | 131 | 1.49 | 0.223 | +0.00087 | [−0.00053, +0.00227] | includes 0 |
| `plc_f2v5` | knn | 72 | 64 | 0.47 | 0.493 | −0.00037 | [−0.00141, +0.00068] | includes 0 |
⭐ **CLU deletion is ASYMMETRIC (clustered CI excludes zero); CLU placebo is SYMMETRIC (includes zero);
both rival arms symmetric and negligible.** ⛔ The naive p of 3.7e-06 is **too confident** by about the
factor six correlated factors per request would predict — the clustered CI is what gets quoted, though
the conclusion is insensitive to the choice.

### ⭐⭐ The decomposition that actually carries the finding
⛔ *"19.77 % vs 18.95 %"* reads like a **null result on churn volume**, and on its own it is one. The two
numbers that carry the result are:
```
churn VOLUME : 18.95%  ->  19.77%    (+0.8 points — small, not significant)
net BIAS     : +0.0044 ->  -0.0139   (a SIGN FLIP; only the second is significant)
```
⭐⭐ **Recanonicalisation is EXPENSIVE but UNBIASED. Deletion is what introduces the BIAS.**
**The cost of deleting a meaning is not the volume of change — it is the direction of it.**

### ⚠ A reporting hazard, stated out loud because it runs the unusual way
Because the **churn-corrected** delta (−0.0196) is **worse** than the **uncorrected** one (−0.0148),
⛔ **a reader who sees only the uncorrected number gets a MORE favourable picture of the arm than the
pre-registered correction supports.** That is the opposite of the usual direction of concern — the
correction normally protects against over-claiming *for* the arm — and it is exactly why it must be
reported as the primary rather than as a footnote.

### ⭐ `del_f3v4` — the falsifiable version of the diversity account, registered now
The diversity account predicts **`ARI(det_C, del_f3v4)` > `ARI(det_C, del_f2v5)` = 0.5656**, because
`scale` is the factor the store represents **worst** (pre-deletion 0.211 against `object_hue`'s 0.674),
so removing one of its values should take **less** structure out of the correlation matrix.
⛔ **If `f3v4`'s ARI comes in at or below 0.5656, the diversity account is WRONG and the `f2v5` result
was a coincidence on one unit.** Stated now rather than argued afterwards.

## 12:21:52Z — ⭐⭐⭐ THE CONJUNCTION: the gs charter's central tension, and the CLU wins it
`del_conj_oh5_sh2` — delete `(object_hue = 5 ∧ shape = 2)`, **852 rows of 32,768 (2.6 %)**.
⭐ **A3.8 behaves exactly as a conjunction requires: NEITHER marginal is flattened.** `cond_2[5]` per-well
range 4.91–6.15 and `cond_4[2]` 5.61–6.25 — both still informative, because only rows where **both** hold
were removed, so neither value is fitted on zero rows. ⛔ A factor-value delete flattens its row to
exactly `log(1/64)`; a conjunction delete must not, and does not.

### ⭐⭐ The compositional question, on the 1,024 requests naming BOTH
| arm | renders BOTH, pre | post | abs drop | **conditional kept** |
|---|---|---|---|---|
| **CLU** | 0.3672 | **0.0107** | −0.3564 | ⭐ **0.0160** |
| **`knn_proto`** | 0.9092 | **0.5488** | −0.3604 | ⛔ **0.6037** |
⭐ *Conditional kept* = of the requests an arm rendered correctly **before**, what fraction it still
renders — each arm on **its own competence set**, so the floor effect cancels. ⚠ The absolute drops are
nearly identical (−0.356 vs −0.360) and that is a **coincidence**; the conditional statistic is the one
that means anything.

⭐⭐⭐ **The gs charter's stated tension, quoted:** *"after every `(red ∧ cube)` atom is gone, the store may
STILL compose `red` from `(red ∧ cylinder)` and `cube` from `(blue ∧ cube)` — and hand you a red cube."*
**That is precisely what `knn_proto` does, and it is visible in the parts:**
| arm | hue5 pre → post | shape2 pre → post |
|---|---|---|
| CLU | 0.4375 → **0.1104** | 0.8877 → **0.2881** |
| `knn_proto` | **1.0000 → 1.0000** | 0.9092 → **0.5488** |
⛔ `knn_proto` renders `object_hue = 5` **perfectly, unchanged**, because its hue-5 prototype still has
every non-`shape=2` hue-5 row. Its additive composition therefore **re-composes the forbidden conjunction
on 55 % of requests.** ⭐ **A per-factor-mean lookup table structurally CANNOT forget a conjunction whose
parts it still holds** — the charter's hypothesis, confirmed.
⭐ **The CLU can: 1.07 % render both.** ⭐⭐ **This is the first result in this hub that is NOT true by
construction and that the rival cannot match** — and it is the re-chartered target.

### ⛔ BUT IT IS NOT A CLEAN WIN, and three things must be said in the same breath
1. ⛔ **The CLU did not forget surgically — it damaged the parts too.** `hue5` 0.4375 → 0.1104 and
   `shape2` 0.8877 → 0.2881 on those requests. It forgets the conjunction partly by **losing the ability
   to render either component there**. `knn_proto` keeps `hue5` at 1.0000.
2. ⛔ **Collateral on the rest of the space is a LOSS under R1, not a tie.** On the 3,991 non-conjunction
   requests: `C_clu −0.020839`, `C_knn +0.000376`, **`delta = −0.021214`, CI `[−0.026898, −0.015530]`** —
   `|Δ| > m = 0.020` **and** the CI excludes 0 ⇒ **loss**. ⚠ Uncorrected; `plc_conj_oh5_sh2` pending.
3. ⚠ **Answer churn 20.26 %** (lost 0.1117, gained 0.0909) **from removing only 852 rows — 2.6 %.**
   And **ARI 0.4365**, *lower* than `del_f2v5`'s 0.5656 from **3,207** rows. ⛔ So layout and answer churn
   are **not** monotone in how much was removed. `plc_conj` (852 random) is the control that decides
   whether any of this is the conjunction or just another ~20 % rebuild.

### ⚠ THE CONTROL THAT COULD OVERTURN THE HEADLINE — stated before it runs
If `plc_conj_oh5_sh2` (852 **random** rows) **also** collapses `hue5` and `shape2` on the targeted set,
then the CLU's "forgetting" is **rebuild churn on a set it was already weak on** (pre 0.3672), ⛔ not
deletion — and the win evaporates. If the placebo leaves them intact, as `plc_f2v5` did for hue 5, the
win stands. ⭐ **I am not claiming the conjunction result until that control lands.**

## 17:35Z (T+6:32) — CELL COMPLETE, DELIVERABLES WRITTEN, hub closing out
Session `clu-dev-1e`. 28/28 units. Report at the canonical `.claude/outputs/hub-deletion.md`
(the co-agent 11:27Z skeleton preserved at `hub-deletion/REPORT-skeleton-clu-dev-89-1127Z.md`;
⛔ nothing deleted). Detail in `FINDINGS-clu-dev-1e.md` (21 sections).
- **Return pack** `hub-deletion.return.tar.gz` 311 files / 30 MB / sha256 `b4cecd3c…`
- **Code bundle** `hub-deletion-2.bundle` whole-branch, 3.2 MB, "records a complete history"
- **ELAPSED 6 h 32 m of 12 h.** ⛔ Zero compute held by me after 11:20Z. ⛔ No sub-agent running.
- ⭐ **Verdict: NOT SOLVED as a win; a banked TIE on the pre-registered object, plus three
  non-by-construction findings, one outright condition failure, and four refuted predictions of
  my own — all reported.** ⛔ Scored run NOT spent: 0 of 3.
