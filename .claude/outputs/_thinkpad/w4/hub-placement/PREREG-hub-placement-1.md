# PREREG — hub/placement — scored run 1/3

**Written 2026-09-19, BEFORE the first held-out number of this hub exists.** Charter §A.1/§A.3:
the bars below are fixed on writing; amendments are appended and dated, never edited.

**Hub:** `hub/placement` · **branch** `w4/placement` · **base** `base/iter4 = 9dfed5e`
**Target name for the scored-run count:** `placement` · **this is run 1 of 3**
**Out:** `/c/chlu/outputs/hub-placement/scored-1` · **LOG:** `/c/chlu/outputs/hub-placement/LOG.md`

---

## 1. THE DIAL, and the declaration that it is the only thing moving

**Dial: WHERE THE WELLS ARE** — which of the 2048 encoder channels sits in which of the `K = 5`
wells. The deployed object is one vector `labels: (2048,) -> {0..4}`, consumed by
`gen_store.well_channels_from_labels`.

Held **identical** across every arm in this run, and asserted in code:

| held fixed | value | where it is asserted |
|---|---|---|
| store byte ledger | `n_modes * D * 4 = 64 * 2048 * 4 = 524,288 B` | `build_arm` raises if it moves |
| head bytes | `1,766,400 B` (singles + all pairwise + bias) | `head_bytes`, printed per arm |
| wells | `K = 5` | `build_partitions` raises if an extra partition is not exactly K=5 |
| modes per well | `64` | `PINS["n_modes"]` |
| read channel width | `K * log2 M = 30 bits` | `PINS["bits_per_read"]` |
| read verb | linear span over the well's modes | `arm_features`, `read_mode="linear"` |
| ridge | `1.0` | `PINS["head_ridge"]` |
| fit pool | one pinned draw, `n_pool = 131,072`, `n_store_train = 32,768` | `fit_pool`, `pool_rng_seed = 20260919` |
| store fit | k-means++ seeding + 1200 adam steps, `tau 1.0 -> 0.1` | `train_store_traced_tau` |
| decoder | `decoder_v1_s{seed}.pkl`, hash-asserted against `winner_hashes.json` | `Judge.__init__` |
| external classifier | hash-asserted against `classifier_manifest.json` | `Judge.__init__` |
| Gram / head rows | one accumulation per head shape, shared by every arm | `build_arm`, `gram_pair.npz` |

**⭐ The byte ledger is mathematically invariant under the dial.** `n_modes * D * 4` does not
depend on the partition, so a placement change cannot buy itself bytes, atoms, read width or head
shape. This is why placement is a clean dial and it is why no byte argument can explain a result
here.

**Labels (w4-common §5, declared in every caption):** labels enter **once**, through
`I(bin(channel_d); factor_f)` on the store's **own fit pool**, and at the address book. No
held-out row, no cell, no arm outcome, no learned rotation. `assert_relevance_is_train_only`
certifies pool ∩ hold-out = 0 rows. ⛔ Zero-supervision structure is **not** claimed.

## 2. THE ARM, fully specified

`place_norm_rank` — **rank-normalised relevance, same merge, same balanced assignment.**

1. `R[d,f] = I(bin(x_d); f)` in bits, 8 quantile bins, on the fit pool (`channel_factor_relevance`).
2. **The one change vs the baseline:** replace each column of `R` by the channel's **per-factor
   rank**, `rank_f(d) / (n_live - 1)` in `[0,1]` (`normalise_relevance(..., "rank")`).
3. Merge the 6 factors to `K = 5` groups by the **shipped** `min_relevance_mass` rule, computed on
   **raw** `R` — the merge is held at the baseline's own value so no candidate can win by merging
   differently.
4. `S[d,w] = max_{f in group w} R_rank[d,f]`.
5. Assign with the **shipped** capacity-constrained rule (`balanced_assign`, lifted verbatim from
   `factor_aligned_balanced_labels`): channels ranked by margin (best minus second-best)
   descending, each taking its best well with capacity `floor/ceil(D/K)`; dead channels fill what
   is left, lowest-occupancy well first. Deterministic, train-only, no optimiser, no seed.

Well sizes `[410, 410, 410, 409, 409]` — **identical to the baseline's**, so the laundering floor is
exactly size-matched and "equal wells" cannot be the explanation.

**⭐ The method is what is pre-registered, not the vector.** Steps 1–5 are re-run on each bridge
seed's own encoder, so the label vector differs per seed. That is the point of the seed axis.

## 3. THE COMPARISON ROWS — and which half of the target each one decides

| row | arm | what it decides |
|---|---|---|
| **A** | `factor_aligned_balanced` — the **label-MI partition** | ⭐ **the ABLATION half of the target.** w4-common §1: a win over an earlier CLU configuration is an ABLATION and is labelled so in every table. |
| **B** | `basis_head` — wave-2 object, correlation split + linear head | the like-for-like w4-common §1 demands on every row |
| **C** | `ward_balanced` — correlation order, **equal wells**, no label | the CONFOUND: separates *alignment* from *balance* |
| **D** | `place_rand_norm_rank` — my arm's own size-matched laundering floor | charter §C: the arm's margin over its own floor |
| **E** | `random_dmatched_balanced` — the baseline's laundering floor | the baseline's floor, so both floors are on the table |
| **F** | `knn_proto` — prototype table | w4-common §1 second row / charter §C.1 matched-bytes lookup |
| **G** | `place_norm_rank_removed` / `_shuffled` | charter §C.2 — atoms zeroed / addressing broken; must collapse |
| **H** | `place_norm_rank_lookupargmax` / `_lookupconvex` | charter §C.1/§C.4 — is the settled read a lookup table? |

**⛔ The RESULT half — row 4 against the full rival set (MLP · transformer · LSTM ·
continuous-input MLP · prototype table · repaired LACE · PoE-EBM) — is NOT decided by this run**
and is registered separately in §7 below, because at the time of writing
`spoke-a-rivals/RIVALS-READY.json` is `status: partial` with **every** rival `false`.

## 4. THE ESTIMAND — declared before the first number

- **Unit: the CORNER, paired.** w4-common §3: a single cell never decides a target.
- **Row 4:** the condition's **49 `nway-*` Montero corners**, 72 requests each.
- **Row 3:** the **60 tiled `pair-*` sub-corners**, 72 requests each. (≥ 49, as §3 requires.)
- **Metric (headline, row 4):** **distinct VALID NOVEL tuples per corner** — judged by THE external
  classifier on THE frozen decoder's image, `valid` conjoined with the accept rule, `novel` =
  not in the fitted split's seen-tuple mask.
- **Secondary, reported always:** judge accuracy per corner; distinct valid tuples per corner.
- **Statistic:** mean paired difference, with a two-sided **sign test** and a **paired bootstrap**
  CI95 (10,000 resamples, `paired_estimand`).
- **Seeds:** bridge seeds **0, 1, 2** on cache **GENERATION 2** (w4-common §2). Three seeds before
  any number is a result. The generation travels with every number
  (`feature_cache_manifest`); the run **refuses to start** on generation 1.

### ⭐⭐ THE BARS (fixed now; charter §A.3 forbids moving them once a result is seen)

| # | claim | bar | direction |
|---|---|---|---|
| **1** | `place_norm_rank` beats the **label-MI partition** on row 4 | mean paired Δ ≥ **+1.0 novel tuple/corner** AND paired-bootstrap CI95 lower bound **> +1.0**, on **all three seeds** | arm higher |
| **2** | the gain is **placement**, not equal wells | Δ vs `ward_balanced` ≥ **+1.0 novel tuple/corner** | arm higher |
| **3** | the arm beats its **own size-matched floor** | Δ vs `place_rand_norm_rank` ≥ **+1.0 novel tuple/corner**, beyond seed spread | arm higher |
| **4** | charter §C.2 — the landscape is load-bearing | `_removed` and `_shuffled` each **collapse**: ≤ **25 %** of the arm's novel/corner | controls lower |
| **5** | charter §C.1/§C.4 — it is not a lookup | `place_norm_rank` ≥ the better of `_lookupargmax` / `_lookupconvex` / `knn_proto` on row 4; if it TIES, §C.3 is invoked explicitly and the tie is reported as a tie | arm ≥ |
| **6** | row 3 is not paid for | Δ vs the label-MI partition on the 60 pair sub-corners ≥ **0.0** (a hold, not a win) | arm ≥ |

**⛔ The margin of +1.0 novel tuple/corner is the same currency and the same number wave 3's
factor-wells PREREG declared, chosen so this hub's bar cannot be accused of being softer.**

### What FALSIFIES this hub (task file's own list, answered in advance)
- Bar 1 fails on any seed ⇒ **the label-MI partition is at or near the reachable optimum for this
  encoder.** Reported as such; that closes the placement question and is a real finding, not a null.
- Bar 1 passes but §7's rival leg loses ⇒ ⛔ **not a result.** Stated in those words.
- Bar 2 fails ⇒ the gain was equal wells, which the baseline already has ⇒ **not a placement result.**
- Bar 4 fails ⇒ the store is decorative and the arm is reported as decorative.

## 5. Expected direction, and the evidence it rests on (all of it SELECTION, none of it scored)

**Expected: bar 1 passes.** On the val condition (24 val corners never in the store's pool,
generation 1), `place_norm_rank` scored **37.417** novel tuples/corner against the label-MI
partition's **25.458** — paired Δ **+11.958**, CI95 **[8.125, 16.000]**, 21 wins / 1 tie / 2 losses,
sign test **p = 6.6e-05**; judge accuracy 0.5197 vs 0.3536 (Δ +0.1661). The judge-scored swap loss
predicted this ordering from **fit-pool pairs alone** before any corner was run
(`norm_rank 0.2457` vs `mi_balanced 0.1516`).

**Why it should work, mechanically — and this is the pre-registered explanation, so a different
mechanism showing up counts against it.** Raw MI compares `I(x_d; wall_hue)` with `I(x_d; scale)` as
if bits were commensurable across factors. Measured on this encoder they are not: **41 % of
channels top-rank `floor_hue` and 46 % `wall_hue`, against 0.6 % `shape` and 0.2 % `scale`**
(relevance mass 587 bits for `wall_hue` against 0.67 for `scale`). Consequently **the label-MI
partition spends one of its five wells on `scale` and leaves `shape` unowned, while every
rank-normalised candidate does the opposite** — and the swap transfer curve measured that `scale`
transfers at only **0.172** at well scale while `shape` transfers at **0.711**. The predicted
mechanism is therefore: *the gain comes from not spending a well on the one factor this encoder
cannot move.*

## 6. PLACEMENT versus THE ENCODER CEILING — registered as a required report, not an outcome

The task file requires: *"report what fraction of any gain is placement and what is the ceiling."*
Declared in advance, so it cannot be chosen after the fact:
- **FLOOR** = the size-matched random partition's swap-loss `exact`.
- **REACHED** = the arm's.
- **CEILING** = the **relaxed oracle**: every well takes the best 410 channels for its own factor
  with disjointness **dropped**. Every real partition is feasible for the relaxed problem, so this
  is a valid **upper bound** on any partition and is reported as a bound, never as an arm.
- **JUDGE CEILING** = no swap at all.
- **Reported fraction** = `(REACHED − BASELINE) / (CEILING − BASELINE)`, with the per-factor
  transfer curve beside it, because the per-factor numbers are what actually answer the question.

## 7. The rival leg — registered now, run only if the gate opens

**⛔ MECHANICAL PRECONDITION (w4-common §2):** `spoke-a-rivals/RIVALS-READY.json` must have the
needed rivals `true`. At writing: `status: partial`, `table` has seed 1 only, and `mlp`,
`transformer`, `lstm`, `cont_mlp`, `poe_ebm`, `lace_repaired` have **no seeds**.
**⚠ DECLARED LIKE-FOR-LIKE HAZARD, recorded before any comparison is attempted:** Spoke A scores
rivals on the **CELL harness** (`composition_B`, one 72-request cell from `eval_b`, metrics
`external/composition_B/aggregate`, `per_factor`, `distinct_counts_B/factor_tuples`). This hub's
row 4 is the **49-corner Montero estimand** on the **value-holdout** condition. ⛔ These are two
different instruments on two different splits and a number may not be moved between them — exactly
the trap charter §A.4 declared-property 4 names for the `lace` row. Therefore:
- If the gate opens in time, the rival leg is scored by **putting this hub's arm through the cell
  harness's own request set and metric**, not by importing the rival numbers into this hub's table.
- If it does not, the rival leg is reported **NOT RUN — gate-blocked**, with the gate's own
  timestamp and field values quoted. ⛔ **A declared NOT-RUN is never a null, and a dash is never a
  win for us** (w4-common §1).

## 8. Cost, caps and resumability (charter: cap both on every launch)
Local CPU only, 3 threads. Per-arm wall-clock cap **900 s**, honoured between corners; a kill costs
one corner, not the run. Every stage skips a complete artifact. Expected ~22 arm-runs
(7 core arms × 3 seeds + 4 charter controls on seed 0) at ~3 min each ≈ 70 min. ⛔ No GPU: this run's
best case is an ABLATION win, and charter §D forbids a GPU for anything but a headline number.

## 9. Files this run must produce
`scored-1/seed{0,1,2}/` — `candidates.npz` + `extra_partitions.npz` (the deployed vector),
`partitions.json`, `arms/*/arm.json`, `arms/*/rows.npz`, `run.log`;
`scored-1/table.json` (the paired estimand for every bar); `same_partition_proof.json`;
one row appended to `/c/chlu/SCORED-LEDGER.md` **before** the run starts.

---

## AMENDMENT 1 — 2026-09-19T18:25Z (appended, never edited; charter §A.3)

**§6's "CEILING" is downgraded to a "RELAXED REFERENCE", and the correction is made BEFORE the
number is reported.** §6 called the disjointness-relaxed configuration a *"valid upper bound on any
partition"*. That is wrong, and the first measured value exposed it: the relaxed well for
`floor_hue` transfers its factor at **1.000** yet scores `exact = 0.2383`, because `exact` also pays
for leakage onto the other five factors while the relevance ordering targets **transfer only**. A
partition that transfers slightly less and leaks much less can therefore score a higher `exact` than
the relaxed configuration.

**Corrected wording, binding from here:**
- The relaxed configuration bounds **`control`** (owned-factor transfer), **not `exact`**. It is
  reported as a **RELAXED REFERENCE** and never as a proven ceiling.
- If the arm's `exact` **exceeds** the relaxed reference, that is **not** a contradiction and is
  **not** evidence of a bug: it means the arm bought its `exact` by *leaking less* rather than by
  *transferring more*. That reading is registered now, before the number is seen.
- **The encoder-ceiling statement this hub will actually quote** is the per-factor one from the
  transfer curve, with its own scope stated: *among relevance-ranked channel sets, no set of this
  encoder moves `scale` past 0.172 at well scale (400 channels) or 0.422 at any size measured, while
  `shape` reaches 0.711 at well scale.* That is a measured statement about the encoder, caveated to
  the family of sets actually searched.
- ⛔ **No bar in §4 is touched by this amendment.** §6 was registered as *"a required report, not an
  outcome"*, and it stays one. Nothing here changes what counts as a pass.
