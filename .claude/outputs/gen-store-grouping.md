# gen-store-grouping — experiment-engineer report

**Task + acceptance criterion:** group the cached feature channels into `K` wells, learn the modes, train an
addressing/read path with a **store-only** objective, and show the correlation-clustered grouping beats random
grouping at the same `K` — with **the delete-the-store delta reported as a number**.

**Status: done.** ⭐ Both falsifiers were tested and **neither fired**. Three of five pre-registered predictions
**held** (two by large margins), **two were falsified** and are reported as findings.

> ## ⛔ DOWNSTREAM RECONCILIATION LIST — 5 items, needs a Hub owner (protocol §5 corollary)
> 1. ⭐ **R16's `shape` gate ratio compares a ONE-class number to a THREE-class number.** Montero's held-out cell
>    is `shape == cylinder`, so `shape` on B has **exactly 1 realised class** (majority rate **1.000**), while the
>    matched-seen control has **3** (majority 0.354). `0.5000 / 0.8756 = 0.571` is a *detection rate over a
>    discrimination rate*. **This does not overturn R16** — "can the decoder render the held-out shape" is a fair
>    question — but the **ratio** is not a like-for-like normalisation and the number should not be read as an
>    accuracy drop. **Owner needed before `gen-cell-run`.** (§6)
> 2. ⭐ **The factorisation ladder's x-axis, as literally specified, is degenerate on this cell.**
>    `leakage = 1 − acc_heldout/acc_matched_seen` has a denominator that goes to ~0 on some (well, factor) cells
>    and a `shape` numerator whose chance level is 1.0. The pooled number still separates the arms cleanly and in
>    the predicted direction, but its per-factor decomposition contains values like `−12.9`. **A redefinition is
>    needed before R1…R5 are run, or the rungs will not be comparable.** Concrete proposal in §7. **Owner needed.**
> 3. **`gen-baselines` byte-match target = `524,288 B` exactly** (machine-readable at
>    `runs/shapes3d_recon/store_bytes.json`). It cannot start without this; it now has it. (§5)
> 4. **`store_plan.json` is the ladder contract.** Later rungs must `load_store_plan()` it verbatim; a rung that
>    re-derives `K` is refused by `assert_plan_matched`. Path in §9. (§8)
> 5. ⚠ **The bridge's feature caches are NOT uniform**: seed 0 is `scope="full"` (480,000 rows), seeds 1 and 2 are
>    `scope="used"` (**9,288 rows**) and contain **none** of the matched-seen control rows. Every future gen spoke
>    hits this. My handling (and its bit-identity proof) is in §10. (§10)

---

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** *isolation* — whether a factored store can be addressed and read without the read path routing around it.
- **Laundering control:** ⭐ **random grouping at the same `K` and the same well-size histogram**, run as a **full
  arm on every number below**, never a footnote.
- **Falsifies:** (a) random grouping matches correlation clustering within seed spread; (b) deleting the store
  leaves the read output unchanged. ⛔ **Neither fired** — evidence in §3 and §4.
- **Does NOT falsify:** a low absolute read accuracy that is low for *both* groupings. ⚠ This **did** arise: the
  absolute read is 0.60 against a 0.95 no-store ceiling, because the store is a **30-bit** bottleneck by design.
  The *contrast* is what is under test.

## FLAG-PROVENANCE TABLE (mandatory, protocol §5)
| | value |
|---|---|
| branch / commits | `agent/experiment-engineer/gen-store-grouping` — `834ebc3`, `09eb608`, `c207741`, `19576ee`, `f10b3ee` |
| base | `main @ fcdca43` |
| dataset / pretext | `shapes3d` / `recon` (R1 mandatory primary). ⛔ No `cls`, no dSprites, no B arm, no C queries, no kill-rule verdict |
| seeds | **{0, 1, 2}**; spread = population std over seeds (ddof=0) |
| feature cache | FROZEN `gen-feature-bridge`; encoder hash `98d326028316ae2c…` (seed 0); cache sha256 `d6728f01d102b0d0…` (seed 0) |
| decoder / probe | THE shared frozen decoder `47c0f83196cd220d…`, hash-asserted via `assert_decoder_hash`; probe `c03bfb8916d00932…` |
| `gen_store.k_wells` | `None` → eigengap rule → **K = 5** (pinned into `store_plan.json` for all seeds/arms) |
| `k_min` / `k_max` | 4 / 16 (pre-registered candidate range) |
| `grouping_method` | `auto` → **`ward`** used (`average` rejected: min well size 1 < floor 39). See §2 |
| `min_well_frac` | 0.10 |
| `n_modes` | **64** → `store_bytes = 64·2048·4 = 524,288 B`, `bits_per_read = 30` |
| `n_store_train` / `n_corr` / `n_probe_train` | 32,768 / 16,384 / 8,192 |
| `train_steps` / `train_batch` / `train_lr` | 1200 / 128 / 3e-3 |
| `commitment_beta` / `tau_start` → `tau_end` | 0.25 / 1.0 → 0.1 (geometric anneal) |
| probe (leakage) | `probe_steps` 400, `probe_batch` 512, `probe_lr` 3e-3 |
| `delete_modes` | `zero`, `mean`, `scramble` |
| eval set | the bridge's `eval_a` (**1024** rows, held back from every fitting stage, mine included) |
| JAX / env | JAX **0.9.0**, **main venv** (`/Users/user/Desktop/CHLU/.venv`, protocol §4 — ⛔ no `uv sync` in the worktree) |
| wall clock | 15,131 s for seeds 1–2 + ~3,000 s seed 0, CPU only |
| reproducibility | ⭐ **bit-exact**: an independent re-run of seed 0/correlation reproduced the shipped `relative_mse` `0.28266923706753433` with `abs_delta = 0.0` |

---

## 1. Preconditions, checked mechanically first

| Check | Result |
|---|---|
| `git cat-file -e main:chlu/core/gen_bridge.py` | ✅ exit 0 |
| feature-cache manifest exists | ✅ `bridge_manifest.json` + `feature_cache.json`, all 3 seeds |
| gate verdict | ✅ `PROCEED` (R15) — precondition **MET** |
| R18 | ✅ this spoke continues; the store is upstream of the decoder, so `gen-decoder-lift` cannot invalidate it |
| worktree | fresh `../CHLU-gen-store`; ⛔ none of the parked worktrees touched |

⚠ **Concurrent spokes observed**: `CHLU-gen-declift` and `CHLU-gen-wide` are live worktrees on `main @ fcdca43`.
I share **only** `chlu/config.py` and `chlu/cli/experiment_cmd.py` with them, and my edits to both are **strictly
append-only** (+94 and +76 lines; the branch diff is 2,910 insertions and **zero deletions**), so a merge conflict there is textual, not semantic.

I read `AGENT_PROTOCOL.md`, `gen-clu-phase1-rulings.md`, `gen-feature-bridge.md`, `factored_store.py`,
`multiwell_read.py`, `gen_splits.py`, and — per R13 — **`handover_context.md` §7 Known Issues only**.

---

## 2. ⛔ FINDING: the textbook correlation-clustering default DOES NOT WORK on this cache

Everything here is from **train activations only**, and all of it was measured **before** any read, leakage or
deletion number existed. It is recorded in `PREREG.md` amendment **D6-b** with the same timestamps.

| observation | value |
|---|---|
| `D` | 2048 |
| **DEAD coordinates** (constant post-ReLU) | **71** of 2048 (93 at the 1,024-row smoke) |
| mean off-diagonal `|corr|`, live coordinates | 0.143 |
| `fcluster(average, maxclust)` on all 2048, `k ∈ {4,6,8,12,16}` | ⛔ **1 cluster every time** |
| `fcluster(complete, maxclust)` on all 2048 | ⛔ **1 cluster every time** |
| average linkage on the **1,983 live** coordinates, `k=7` | `[1972, 4, 3, 1, 1, 1, 1]` — chaining |
| **shipped run**: `average` attempt | min well size **1** → **rejected** (floor 39) |
| **shipped run**: `ward` attempt | min well size **232** → accepted |

**Cause.** The dead coordinates correlate with nothing, so every distance involving them is *exactly* 1.0. That
creates a tie plateau at the top of the dendrogram, and `maxclust` cannot cut a plateau. Removing them exposes the
second, independent problem: on a dense low-contrast correlation matrix average linkage **chains**.

**What I did** (mechanical, outcome-independent, both pre-registered before measurement):
1. dead coordinates are **clustered around** and then placed deterministically into the currently-smallest well
   (a constant coordinate is reconstructed exactly by any mode, so its placement carries no information);
2. `grouping_method="auto"` walks `("average", "ward")` — **the pre-registered default first** — and accepts the
   first method whose smallest well holds ≥10% of an equal share.

⛔ The acceptance test is a property of the **partition**; no held-out quantity, no read metric and no arm outcome
enters it, `ward` was already named in `PREREG.md` D6 before any of this was measured, and the method actually used
is recorded per-arm in `store_manifest.json.arms.*.grouping_info` (including the rejected attempt and its sizes).

### The realised grouping
`K = 5` by the eigengap rule (`k=5` gap **0.0514**, runner-up `k=6` 0.0452 — ⚠ not a wide margin, reported as such).
Well sizes **[303, 305, 573, 501, 366]**. ⛔ Wells are integer indices only, here and everywhere.

---

## 3. ⭐ THE RESULT — correlation vs random at the same `K`, 3 seeds

**Read metric (declared up front):** probe top-1 on images obtained by decoding the **store read** of cached
A-split features through THE shared frozen decoder. `shape` — the only genuinely compositional factor (R16) — is
reported **separately** from the aggregate, per the task amendment.

| | aggregate | **`shape`** | feature rel-MSE |
|---|---|---|---|
| **no-store ceiling** (real cached features, same decoder+probe) | 0.9507 ± 0.0129 | 0.9076 ± 0.0122 | — |
| ⭐ **correlation grouping** | **0.5997 ± 0.0019** | **0.5785 ± 0.0124** | **0.2789 ± 0.0065** |
| ⭐ **random grouping** (same `K`, same well sizes) | 0.4492 ± 0.0008 | 0.3770 ± 0.0156 | 0.4301 ± 0.0156 |
| **margin (corr − rand)** | **+0.1504 ± 0.0023** | **+0.2015 ± 0.0064** | **35.1 % ± 2.0 % lower** |
| per-seed margins | +0.1481 / +0.1496 / +0.1536 | +0.2100 / +0.1943 / +0.2002 | 32.5 / 35.7 / 37.2 % |

⭐ **The margin is 81× the largest seed spread** (`falsifier_grouping_is_arbitrary.margin_exceeds_seed_spread =
true`; margin 0.1504 vs max spread 0.00186). **Falsifier (a) did not fire.**

The no-store ceiling reproduces `gen-feature-bridge`'s `ceiling_A` **exactly** (0.9507 ± 0.0129 vs its 0.9507 ±
0.0129) — an independent provenance check that I am reading the same instrument.

**Per-factor read (correlation | random):** floor_hue 0.995 | 0.838 · wall_hue 0.987 | 0.908 · object_hue 0.623 |
0.336 · **shape 0.579 | 0.377** · scale 0.211 | 0.154 · orientation 0.204 | 0.082.

⚠ **Honest reading of the absolute level.** 0.60 against a 0.95 ceiling is *low*, and it is low by construction:
the store is a **30-bit** channel (`K·log₂M = 5·6`) reconstructing a 2048-dim vector, against a factor grid that
needs 19.4 bits. It is low for **both** groupings, which the dial declaration named in advance as a
non-falsifier. ⛔ This is **not** a cell result and no arm comparison rides on it.

### Is the store just k-means? ⭐ No — and the run is bit-reproducible
Post-hoc (not pre-registered), seed 0, correlation arm:

| | feature rel-MSE on `eval_a` |
|---|---|
| k-means++ **initialisation**, untrained | 0.4410 |
| after 1200 Adam steps | **0.2827** (**−35.9 %**) |
| shipped manifest value | 0.28266923706753433 — **reproduced with `abs_delta = 0.0`** |

⚠ Note the k-means++ init (0.4410) sits almost exactly at the **trained random** store (0.4301): learning the modes
buys about as much as the grouping does, and the two effects are roughly additive here.

⚠ ⛔ **Do not read the training loss as a curve.** `tau` is annealed during the run, so `loss[0]` and `loss[-1]`
are values of *different* functions and the trace can rise while the store strictly improves (measured on synthetic
data: rel-MSE 0.0568 → 0.0391 while the reported loss rose 0.283 → 0.398). This is pinned in the module docstring
and asserted by `test_training_improves_the_hard_read_over_the_kmeanspp_init`.

---

## 4. ⭐⛔ THE DELETE-THE-STORE DELTA — the number this spoke lives or dies on

**Structural first.** `chlu.core.gen_store.read_from_codes(store, codes)` — the function that produces the read
output — **does not take the features as an argument at all**. `assert_no_bypass` checks this three ways, and ran
on **every arm of every seed**:

| check | result (all 6 arms) |
|---|---|
| signature is exactly `(store, codes)` | ✅ |
| output equals an **independent** NumPy codebook gather written without reference to the features | ✅ `max |Δ| = 0.0` |
| the channel is **live**: replacing the features moves the codes, and moving the codes moves the output | ✅ codes changed; output moved by 3.86 (corr) / 4.48 (rand) RMS |
| width of the only encoder→decoder channel | **30.0 bits** |

**Then measured.** Deleting the store and re-running the read, on the same 1024 `eval_a` rows:

| deletion | arm | aggregate before → after | **Δ aggregate** | **Δ `shape`** | read-output RMS Δ |
|---|---|---|---|---|---|
| ⭐ **`zero`** (headline) | correlation | 0.5997 → 0.1245 | **+0.4752 ± 0.0029** | **+0.3301 ± 0.0120** | 0.6795 |
| ⭐ **`zero`** | random | 0.4492 → 0.1245 | **+0.3247 ± 0.0011** | +0.1286 ± 0.0167 | 0.6587 |
| **`mean`** (fairest) | correlation | 0.5997 → 0.1254 | **+0.4743 ± 0.0023** | +0.3265 ± 0.0096 | 0.4153 |
| **`mean`** | random | 0.4492 → 0.1243 | +0.3249 ± 0.0025 | +0.1266 ± 0.0150 | 0.3787 |
| `scramble` | correlation | 0.5997 → 0.1702 | +0.4295 ± 0.0118 | +0.2894 ± 0.0274 | 0.7031 |
| `scramble` | random | 0.4492 → 0.1520 | +0.2972 ± 0.0064 | +0.1087 ± 0.0204 | 0.6944 |

⭐ **`delete_zero_delta_aggregate = +0.4752`. Pre-registered threshold ≥ 0.30. `FAILED_TASK = false`.**
The programme's standing failure mode — *"deleting the entire store changed the loss by under 0.001 bits"* — is
**+475× above the 0.01 near-zero line here**. ⛔ **Falsifier (b) did not fire.**

⭐ **The `mean` deletion discharges the one caveat I pre-registered against the `zero` number.** I warned in
`PREREG.md` P2-b that part of a `zero` delta could be "the decoder handed an off-manifold all-zeros input" rather
than "the store removed". The `mean` deletion leaves the read output an on-manifold, correctly-scaled feature
vector carrying **zero** information, and gives **+0.4743 — statistically identical to `zero` (+0.4752)**. So the
delta is the *store's contribution*, not a decoder artefact. That was the right control to have registered.

⚠ **Two honest observations.**
1. The post-deletion accuracy is **0.1245**, against my pre-registered point prediction of **0.117** (the eval
   set's mean majority-class rate). The read collapses to a constant image, as predicted.
2. ⭐ **The deletion delta is itself larger for the clustered grouping (+0.475 vs +0.325).** That is the same
   contrast as §3 seen from the other side — a better-organised store simply has more to lose — and it is *not* an
   independent result. I report it as a consequence, not as a second win.

**⛔ A negative control that could have fooled us, and did not.** My first `shuffle` deletion permuted each
codebook's **rows**. It measured **Δ = 0.0000 exactly** — because nearest-neighbour addressing re-finds the same
vector, so a row permutation is a *no-op*, not a deletion. Had I shipped only that variant I would have reported a
false bypass. It is now pinned by `test_row_permutation_deletion_is_a_no_op` and named in the docstring.

---

## 5. ⭐ Store byte count — machine-readable (item 3)

`runs/shapes3d_recon/store_bytes.json`:
```json
{"store_bytes": 524288, "n_wells": 5, "n_modes": 64, "dim": 2048,
 "dtype": "float32", "bytes_formula": "n_modes * sum_k d_k * 4 = n_modes * D * 4",
 "bits_per_read": 30.0}
```
⭐ **`store_bytes = 524,288 B` exactly, and it is independent of `K`** (the wells *partition* the 2048
coordinates, so `Σ_k d_k = D` whatever `K` is). ⛔ The addressing head is **not** in this count and is reported
separately as `bits_per_read`; a byte-matched baseline matches `store_bytes`.

---

## 6. ⭐ Leakage — the factorisation ladder's x-axis (R0's x-coordinate)

`leakage = 1 − transfer`, `transfer(k,f) = acc_heldout(k,f) / acc_matched_seen(k,f)`, per-factor linear probes
trained on **that well's sub-vector only**, on seen rows, normalised against the **matched-seen** control (R16 —
⛔ never random-A). ⛔ These probes never touch the decoder and render no verdict.

| | pooled leakage | `shape` leakage | block-diagonality | MIG | DCI-D | DCI-C | DCI-I |
|---|---|---|---|---|---|---|---|
| **correlation** | **−0.0444 ± 0.0921** | −0.2435 ± 0.5142 | **0.4787 ± 0.0106** | **0.2026 ± 0.0355** | 0.3094 ± 0.0078 | **0.1968 ± 0.0124** | **0.5838 ± 0.0179** |
| **random** | +0.2257 ± 0.0443 | +0.7946 ± 0.1599 | 0.2223 ± 0.0095 | 0.0204 ± 0.0047 | 0.3208 ± 0.0082 | 0.0223 ± 0.0013 | 0.2992 ± 0.0052 |
| **margin (rand − corr)** | **+0.2701 ± 0.0518** | +1.0381 ± 0.4426 | 2.15× | +0.1823 ± 0.0376 | −0.011 | +0.175 | +0.285 |

**The direction predicted in P3 held, and by 3.4× the registered point estimate** (predicted 0.08, threshold 0.03,
measured 0.270). The registered counter-hypothesis (random projections transferring better) is **rejected**.

### ⛔ BUT: the metric as literally specified is degenerate on this cell — reconciliation item 2
Post-hoc audit of the class supports (`posthoc.json`):

| factor | realised classes on **B** | majority rate on **B** | classes on **matched-seen** | majority on matched-seen |
|---|---|---|---|---|
| floor_hue / wall_hue / object_hue | 2 / 2 / 2 | 0.50 / 0.50 / 0.50 | 2 / 2 / 2 | 0.51 / 0.505 / 0.51 |
| scale | 3 | 0.333 | 3 | 0.343 |
| ⛔ **shape** | **1** | ⛔ **1.000** | **3** | 0.354 |
| orientation | 3 | 0.333 | 3 | 0.343 |

⭐ **Montero's held-out cell IS `shape == cylinder`, so `shape` on B has exactly one class and its chance level is
1.000.** `acc_heldout(shape)` is therefore a *detection rate*, not a discrimination accuracy, and dividing it by a
3-way accuracy is not a like-for-like normalisation. Chance-correcting it (`κ = (acc−chance)/(1−chance)`) makes it
**undefined** (÷0). The other factors are comparable in class count but several (well, factor) denominators sit
near 0, producing per-factor raw leakages as extreme as **−12.9**.

**The pooled contrast is still real** (it separates the arms 3/3 seeds, consistently, and the relevance matrix in
§7 independently confirms the structure) — but ⛔ **it must not be published as a calibrated x-axis, and later
rungs computed this way will not be comparable.** Concrete proposals, for a Hub owner:
- **(i)** restrict the matched-seen normaliser to a **single** non-cylinder shape so both sides are 1-class on
  `shape`; or **(ii)** drop `shape` from the pooled leakage and report it as an explicitly-labelled detection rate;
  or **(iii)** ⭐ **use the R17 widened split** — which is already **MANDATORY** — so the held-out set contains more
  than one shape. This is an **independent, mechanical argument for R17** arrived at from a different direction.
- ⛔ Whichever is chosen must be fixed **before** R1…R5 run, and R0 re-scored under it. R0's *store* is unaffected;
  only the x-coordinate would be recomputed, which is cheap (the probes are ~2 min/arm).

⚠ **This also touches R16** (reconciliation item 1): the `shape` gate ratio `0.5000/0.8756 = 0.571` divides the
same 1-class detection rate by the same 3-class discrimination. The finding that the decoder struggles on held-out
shapes stands; the **ratio** is the part that needs a decision.

---

## 7. Well-space C mapping (R2 addendum, item 4) — ⛔ a reported diagnostic, never a query set

The C set is loaded **verbatim** from `.claude/data/gen/shapes3d/c_invalid.json` (72 factor-space records) and is
never rewritten, re-sampled or re-expressed in well space as queries. What is reported is where each record's
`violated_factor` lands on the `I(code_k ; factor_f)` relevance matrix.

| | pre-registered `intra_well_conflict_fraction` (top-1 share ≥ 0.5) | mean top-1 well share | uniform baseline |
|---|---|---|---|
| correlation | 0.2222 ± 0.1571 (per seed 0.000 / 0.333 / 0.333) | **0.434** | 0.200 |
| random | **0.0000 ± 0.0000** | 0.266 | 0.200 |

⛔ **P5's point prediction (0.6 vs 0.3) is FALSIFIED.** The direction holds but the binary ≥0.5 threshold was set
too high for `K=5`. The **continuous** statistic is the better one and it separates cleanly: the clustered grouping
concentrates a violated factor on one well at **2.17× uniform**, random at **1.33×**.

**Why, visible directly in the relevance matrix** (seed 0, bits; ⛔ integer well indices only):

| | floor_hue | wall_hue | object_hue | scale | shape | orientation |
|---|---|---|---|---|---|---|
| **correlation** well 0 | 0.99 | **3.05** | 0.11 | 0.11 | 0.24 | 0.93 |
| well 1 | 0.65 | **3.24** | 0.11 | 0.09 | 0.19 | 1.23 |
| well 2 | 0.53 | 2.43 | 1.10 | 0.20 | 0.43 | 0.31 |
| well 3 | 1.69 | 0.10 | **2.23** | 0.35 | 0.76 | 0.03 |
| well 4 | **3.31** | 0.16 | 1.13 | 0.42 | 0.31 | 0.04 |
| **random** wells 0–4 | 1.87–2.30 | 2.20–2.39 | 0.37–0.77 | 0.06–0.10 | 0.11–0.19 | 0.07–0.24 |
| `H(f)` | 3.32 | 3.32 | 3.32 | 3.00 | 2.00 | 3.91 |

⭐ **The clustered wells differentiate; the random wells are five copies of the same profile.** Well 4 carries 3.31
of floor_hue's 3.32 bits; wells 0/1 carry wall_hue; well 3 carries object_hue. Under random grouping every well
carries ~2 bits of each hue and nothing distinguishes them. **This is the clearest single piece of evidence that
the grouping carries structure, and it is independent of the read metric and of the leakage metric.**

---

## 8. ⭐ The reusable store plan (the ladder contract)

`runs/shapes3d_recon/store_plan.json`, schema `chlu-gen-store-plan/1`:
`K=5` · `n_modes=64` · `D=2048` · `feat_shape=[128,4,4]` · `grouping_method="ward"` · `store_bytes=524288` ·
`well_sizes=[303,305,573,501,366]` · `n_dead_channels=71` · `rung="R0"` · full 2048-entry `labels` vector ·
`k_choice` (rule + every eigengap) · `encoder_param_hash` · `feature_cache_sha256`.

⛔ Later rungs call `load_store_plan()`; `assert_plan_matched()` **refuses** a rung that disagrees on `K`,
`n_modes`, `D` or `store_bytes` (`test_a_later_rung_that_redefines_k_is_refused`). Only `labels` may change.

---

## 9. Pre-registration scorecard (`PREREG.md`, mtime **09:54:44**, ~25 min before the first harness existed; amended
D6-b at 10:09 and P2-b at ~10:35, both still before any reported number)

| # | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| P1a | rel-MSE ≥5 % lower (point 10 %) | 5 % | **35.1 % ± 2.0 %** | ⭐ **HELD**, 3.5× the point |
| P1b | read aggregate margin ≥ +0.02 (point +0.04) | +0.02 | **+0.1504 ± 0.0023** | ⭐ **HELD**, 3.8× the point |
| P1c | `shape` margin ≥ +0.02 (point +0.05) | +0.02 | **+0.2015 ± 0.0064** | ⭐ **HELD**, 4.0× the point |
| P2 | delete-`zero` Δagg ≥ 0.30 (point 0.60); Δ`shape` ≥ 0.10 | 0.30 / 0.10 | **+0.4752 / +0.3301** | ✅ **HELD** (point 0.60 over-predicted) |
| P2 | post-deletion aggregate ≈ 0.117 | — | **0.1245** | ✅ close |
| P2-b | `mean` deletion Δ ≥ 0.15, and *smaller* than `zero` | 0.15 | **+0.4743**, ⚠ **equal to** `zero` | ✅ threshold held; ⛔ **"smaller than zero" FALSIFIED** — and that is the *better* outcome (§4) |
| P3 | pooled leakage margin ≥ 0.03 (point 0.08), clustered lower | 0.03 | **+0.2701 ± 0.0518** | ⭐ **HELD**; counter-hypothesis rejected — ⚠ but see §6 |
| P4a | block-diagonality ≥ **3×** random | 3× | **2.15×** (0.4787 vs 0.2223) | ⛔ **FALSIFIED as a ratio**; direction holds |
| P4b | MIG/DCI-D margin +0.01…+0.05 | — | MIG **+0.1823**; DCI-D **−0.011** | ⚠ **split**: MIG far exceeds; DCI-**D** does not separate at all (DCI-C +0.175, DCI-I +0.285 do) |
| P5 | C intra-well 0.6 vs 0.3 | — | **0.222 vs 0.000** | ⛔ **point FALSIFIED**; direction holds (§7) |

⚠ **DCI-D is worth the Hub's attention:** *disentanglement* is per-well-normalised, so five wells that each
carry the same mixed profile score the same as five specialised ones. **DCI-D did not distinguish the arms
(0.309 vs 0.321) while completeness and informativeness did, by +0.175 and +0.285.** ⛔ Do not quote DCI-D alone
as a factorisation measure in this setting.

---

## 10. ⚠ Environment findings the next gen spoke needs

1. ⭐ **The bridge's caches are not uniform.** Seed 0 is `scope="full"` (480,000 rows); **seeds 1 and 2 are
   `scope="used"` (9,288 rows = the bridge's own train + eval_A + eval_B)** and hold **no** matched-seen control
   rows. The first 3-seed run died on seed 1 with
   `KeyError: '32199 requested row(s) are not in the feature cache'`.
   **My fix, and why it is not a rebuild:** rows the cache lacks are encoded on the fly through **the bridge's own
   checkpointed encoder**, whose `param_hash` is asserted against both the bridge manifest and the cache manifest,
   and quantised to the cache's own `float16`. ⭐ **That this is the same map is verified, not asserted:**
   `_assert_cache_reencodes` re-encodes 64 genuinely cached rows and requires bit-identity — measured
   `max_abs_delta = 0.0` on every seed. ⛔ No cache was rewritten and the frozen cache is untouched on disk.
2. **§7.23 (x64 at module import) did not bite**, because the store's modes are explicitly `float32` and every
   entry point casts its input to its own parameter dtype (the `gen_bridge` exemplar). All 31 new tests pass in the
   **full suite**, not just in isolation.
3. **§7.12** (`uv run chlu` broken by the hidden `.pth`) is why every command here is
   `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`.
4. **CSF3 is unreachable from this session** (no VPN/DNS). Everything ran locally on CPU. ⛔ **NOT RUN** on the
   cluster; the submit line I would have used:
   ```
   PAYLOAD_MODULE=chlu.experiments.exp_gen_store ARMS=clu GEN_GIT_COMMIT=$(git rev-parse HEAD) \
   sbatch scripts/csf3/job_gpu_gen.sh --dataset shapes3d --pretext recon --stage store --arm clu --seed 0
   ```
   ⛔ No job id exists; I report no unrun job as a pass.

## 11. Suite

| | count |
|---|---|
| **before** (`main @ fcdca43`, detached worktree, main venv) | **2010 passed, 0 failed** (6:41:25) |
| **after** (`agent/experiment-engineer/gen-store-grouping`) | **2041 passed, 0 failed** (1:03:30) |
| delta | **+31 = exactly the 31 new tests in `tests/test_gen_store.py`** |

⚠ The `before` number matches R15's reported post-merge **2010** exactly, so the baseline is the state the Hub
recorded. Both runs used the **main venv** (JAX 0.9.0) per protocol §4; ⛔ no `uv sync` was run in any worktree, so
the w6 resolver hazard does not apply. The `before` run was executed in a throwaway detached worktree at
`fcdca43` (`../CHLU-gs-base`, since removed) — ⛔ none of the parked worktrees was used.

⭐ **All 31 new tests pass in the FULL suite, not only in isolation** — the §7.23 hazard that broke 9 of
`gen-feature-bridge`'s tests did not recur (§10.2).

---

## Git footprint
- **Branch:** `agent/experiment-engineer/gen-store-grouping` (worktree `/Users/user/Desktop/CHLU-gen-store`),
  base `main @ fcdca43`.
- **Commits** (`main..HEAD`, re-verified from the MAIN repo before any worktree removal — see §12):
  - `834ebc3` add the factored channel store: grouping, modes, no-bypass read
  - `09eb608` wire GenStoreConfig through the three config sites
  - `c207741` add exp-gen-store: both groupings, the deletion delta, leakage
  - `19576ee` tests: the no-bypass positive control and the plan contract
  - `f10b3ee` extend cache coverage through the frozen encoder, verified
- **Files touched:** `chlu/core/gen_store.py` (new), `chlu/experiments/exp_gen_store.py` (new),
  `tests/test_gen_store.py` (new), `chlu/config.py` (**+94, append-only**),
  `chlu/cli/experiment_cmd.py` (**+76, append-only**).
  ⭐ The whole branch diff is **2,910 insertions / 0 deletions** — not one existing line was modified. ⛔ Nothing under `scripts/csf3/`, `chlu/eval/` or
  `chlu/core/gen_bridge.py` was touched.
- **Rebased onto local `main`** (not `origin/main`, §7.21/protocol §3.5): no-op, base unmoved.
- **Unresolved conflicts:** none.
- **Artifacts** (all under `.claude/`): `outputs/gen-store-grouping/PREREG.md`, `runs/shapes3d_recon/`
  (`store_results.json`, `store_plan.json`, `store_bytes.json`, `seed_*/store_manifest.json`, `grouping.png`,
  `grouping_contrast.png`), `posthoc.json`, `init_vs_trained.json`, `run3.log`, `run3b.log`,
  `suite_before.log`, `suite_after.log`.

## Open questions / follow-ups / risks
1. ⛔ **Reconciliation items 1 and 2 (§6) need a Hub owner before `gen-cell-run` and before ladder rungs R1–R5.**
2. ⚠ **`K = 5` won its eigengap narrowly** (0.0514 vs 0.0452 at `k=6`). The rule was pre-registered and I did not
   revisit it, but the ladder is now pinned to `K=5` by `store_plan.json`. If the Hub wants `K=6` that is a
   *decision*, not a re-derivation, and it must be taken **now** — before any later rung runs.
3. ⚠ **I did not reuse `FactoredStore`/`multiwell_read`'s physics.** `FactoredStore` is an
   `AtomDictionaryPotential` landscape settled by damped Verlet, bound to `CatTestConfig` and to a synthetic
   set-code family with per-well **anchors** and an item→well incidence — neither of which exists for a channel
   partition of a frozen feature cache; supplying anchors would have meant inventing the answer. What I did reuse
   is `soft_occupancy` (the soft-assignment verb, and the "⛔ no argmax in the differentiable path" discipline) and
   the `n_bytes` accounting. ⭐ **If the Hub wants R0 to be a *physics* store rather than a factored codebook, that
   is a scope change and should be a decision, not a silent substitution.** I flag it rather than assume.
4. ⚠ The store is **not checkpointed** to `.pkl` — the manifests are the artifact. Re-running reproduces it
   bit-exactly (verified, §3), so this is cheap to add if a later rung wants to load rather than refit.
5. ⚠ `scale` (0.211) and `orientation` (0.204) are read very poorly by *both* arms. Consistent with the 30-bit
   budget and with the bridge's photometric finding; ⛔ not investigated, not this spoke's job.

## Proposed handover updates (for the Hub)
- **§3 config:** new `GenStoreConfig` block (dataclass + `load_config` + `save_config`, append-only) and the new
  `chlu exp-gen-store` subcommand. Notable defaults: `k_wells=None` (eigengap rule over `[4,16]`),
  `grouping_method="auto"`, `n_modes=64`, `groupings=["correlation","random"]` (⛔ the control is never optional),
  `delete_modes=["zero","mean","scramble"]`, `encode_chunk=16384`.
- **§7 NEW known issue — the bridge's caches are not uniform.** Seed 0 `scope="full"` (480k rows), seeds 1/2
  `scope="used"` (9,288 rows), and no `used` cache holds the matched-seen control rows. Any downstream spoke
  needing more than the bridge's own rows must extend coverage through the frozen encoder (pattern + bit-identity
  check shipped at `exp_gen_store.gather_features` / `_assert_cache_reencodes`).
- **§7 NEW known issue — `fcluster(..., 'maxclust')` cannot cut a tie plateau.** Dead post-ReLU coordinates (71 of
  2048 here) sit at distance exactly 1.0 from everything, and average/complete linkage then return **1 cluster at
  every `k`**. Cluster around dead coordinates and place them afterwards.
- **§7 NEW discipline note — a permuted codebook is NOT a deletion.** Permuting a codebook's rows leaves a
  nearest-neighbour read **bit-identical** (measured Δ = 0.0000). Any future "delete the memory" control must
  destroy content, not reorder it.
- **§7 NEW discipline note — annealed-temperature training losses are not curves.** Compare the hard read's
  reconstruction against the initialisation, never `loss[-1]` against `loss[0]`.
- **Rulings ledger:** R0 of the factorisation ladder is **done and pinned**; the ladder's x-axis definition is
  **open** (§6) and blocks R1–R5.

---

# ⟲ POST-MERGE ADDENDUM (2026-09-11) — written after the Hub merged this branch

⚠ **Read this before reusing any number above.** My branch was merged by the Hub at **`e4f3d71`**; `main` is now
**`9a4457d`**. The worktree `../CHLU-gen-store` was removed as part of that cleanup. **Nothing was lost** — verified
from the main repo: the branch ref still resolves to `f10b3ee`, all five commits are ancestors of `main`, and
`gen_store.py` / `exp_gen_store.py` / `test_gen_store.py` / `experiment_cmd.py` are **byte-identical** to my branch
(`config.py` differs only because `gen-decoder-lift`, `gen-wide-split` and `gen-bridge-wide` appended their own
blocks; my `GenStoreConfig` block is byte-identical and all three wiring sites are present, and it round-trips on
the merged tree with `store_bytes = 524288`, `bits_per_read = 30.0`).

**Suite:** ⛔ my §11 figure (2010 → 2041) was measured at base `fcdca43` and **no longer describes `main`.** The
Hub's own run on the merged tree is the authoritative number: **2078 passed, 0 failed**, with exact accounting
`2010 + 18 + 19 + **31 (mine)** = 2078`. My +31 is intact and not double-counted.

## What the Head ruled on my two reconciliation items (R20) — both accepted, both superseded my wording
1. **Item 1 (detection vs discrimination) — RULED.** The Head's normaliser is stronger than my proposal: positives =
   held-out cylinders, negatives = **matched-seen non-cylinders in the same colours**, scored by binary **AUROC /
   balanced accuracy within the corner**. ⭐ This is what kill rule (i) reads until the widened split makes it moot.
2. **Item 2 (the ladder x-axis) — RULED, and my definition is SUPERSEDED.** Leakage is now the transfer of a
   **balanced** metric, never a ratio of raw accuracies:
   `leakage = 1 − BA(per-well shape probe on held-out ∪ matched-seen-other-shapes, same corner) ÷ BA(same, seen corners)`.
   ⛔ **The pooled leakage column in §6 above is therefore obsolete** and must not be carried into the ladder.
   ⭐ The finding that produced the rule stands; the numbers it produced do not.

## ⛔ One thing I could not have known, and it invalidates part of §3
⭐ **`gen-decoder-lift` succeeded**: `shape` on B went 0.5926 → **0.9398**, so **R16's STOP is lifted** and the
decoder I rendered through is **superseded**. My read numbers (§3) and deletion deltas (§4) were all measured
through the *old* decoder and **must be re-scored against the decoder-lift winner.**
- ⭐ **What survives the re-score untouched:** the store itself (it trains on cached *features*, never on the
  decoder — R18 exactly as written), `K = 5`, the grouping, `store_bytes = 524,288 B`, the 30-bit channel, and the
  **structural** no-bypass result, which is a property of `read_from_codes`'s signature and cannot move.
- ⚠ **What will move:** every absolute read accuracy and every deletion delta in §3–§4. The *direction* and the
  laundering-control contrast are very unlikely to invert (the margin is 81× the seed spread), but ⛔ I am not
  asserting the re-scored margin — that is a measurement, and it has not been made.
- The Hub has folded this into a single cheap re-score spoke; nothing retrains.

⭐ **R20 promoted the structural result to a design constant:** the headline is *not* `+0.4752`, it is that **a
bypass is inexpressible** — the read never sees the features and the only encoder→decoder channel is **30 bits**
(`K·log₂M = 5·6`), now a declared design constant. ⛔ Any later change to `K` or `n_modes` is a declared change to
the claim, not an implementation detail.
