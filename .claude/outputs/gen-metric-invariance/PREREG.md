# PREREG — gen-metric-invariance
**Written 2026-09-13 00:40 UTC-local, BEFORE any measurement of this spoke.** Branch
`agent/experiment-engineer/gen-metric-invariance`, base `main @ 1bcfcb9` (⚠ the task file names `3faef27`; local
`main` has since fast-forwarded to `1bcfcb9` = the `gen-baselines` merge, a strict superset — declared, not chosen).

## 0. What is being pre-registered
R26 ruling 2 promotes **DCI-C** as the ladder's x-axis on one criterion only:

> **The axis must give the same value for the same features under different partitions.**

This file fixes, in advance: (i) the operational form of that criterion, (ii) the numeric tolerances, (iii) the
composition rule that turns leg results into a PASS/FAIL verdict, and (iv) my predicted outcome for each of the
three candidate axes. ⛔ Nothing below may be edited after the harness runs.

## 1. ⚠ WHAT IS *NOT* BLIND — declared up front
Two of the numbers this check consumes are **already published** and I have read them:
- Montero DCI-C per seed (`gen-store-grouping` manifests): corr `0.1903 / 0.1859 / 0.2142`, rand `0.0241 / 0.0211 / 0.0216`
  ⇒ margin mean **+0.1745**.
- Wide DCI-C (R25 / `gen-r0-wide-refit` §3.3): margin **+0.1299 ± 0.0036**.
- Leakage margins: Montero **+0.1337 ± 0.0592** (R22), wide **−0.0660 ± 0.0144** (R25).

⇒ **Leg 3's outcome on the `correlation`/`random` pair is NOT a blind prediction for DCI-C or for raw leakage.**
It is included because it is the criterion the Advisor named, and because it is the *negative control's* failure mode.
The genuinely blind content of this spoke is: **leg 1, leg 2, leg 2b, the four NEW partitions, the chance-corrected
companion, and the pipeline noise floor.** The leg-3 tolerance below is derived from program convention (the
program-wide "3× seed spread" rule, R23.2 / `underpowered_spread_multiple = 3.0`), ⛔ not from DCI-C's values.

## 2. The three candidate axes, scored by the identical check
| tag | axis | role |
|---|---|---|
| `dci_c` | DCI completeness, `chlu.core.gen_store.dci_scores(...)["completeness"]` | the candidate (R26.2) |
| `leak_cc` | chance-corrected balanced-accuracy transfer leakage, `1 − ccBA(num)/ccBA(den)`, `ccBA = (BA − 1/c)/(1 − 1/c)` | the companion (R26.2) |
| `leak_raw` | raw balanced-accuracy transfer leakage, `1 − BA(num)/BA(den)` (R23.2, as shipped) | ⭐ **THE NEGATIVE CONTROL — it must FAIL** |

Secondary, reported but not gating: `mig`, `dci_d`, `dci_i`.
Every axis is reduced to the **margin** `m = value(correlation) − value(random)`, per seed, per configuration.

## 3. The partitions (item 1) — the features never move
One seed's cached features are loaded once and held fixed; only the **partition of the encoder channels into the
`K = 5` pinned wells** varies. `K`, `n_modes = 64`, `D = 2048`, `store_bytes = 524288` are ladder constants
(`assert_plan_matched`).

| tag | construction |
|---|---|
| `correlation` | the shipped clustered grouping (`grouping_labels(..., "correlation")`) |
| `random` | the shipped laundering control (`random_grouping(ref, seed)`) — same well-size histogram |
| `random_alt` | a second independent draw, `random_grouping(ref, seed + 1000)` |
| `well_permuted` | ⭐ **the same SET partition as `correlation`**, with the well *indices* permuted by a fixed derangement |
| `contiguous` | channels in index order cut into blocks matching `correlation`'s well-size histogram |
| `balanced_random` | random assignment with equal well sizes (a different histogram at the same `K`) |

## 4. The legs

### Leg 1a — RELABELLING INVARIANCE (metric level, exact)
Hold the fitted store and its `codes (N, K)` matrix fixed; permute the **well index order** of the codes matrix (and,
separately, the **mode index order** within each well). The set partition, the features and the information content
are unchanged. **Tolerance: `max |Δ axis| ≤ 1e-12` (absolute).** This is the literal reading of the criterion.

### Leg 1b — PIPELINE RELABELLING FLOOR (reported, not gating)
Re-fit the store under `well_permuted` (identical set partition, different well order ⇒ a different `_kmeanspp_init`
RNG consumption order). The resulting `|Δ axis|` defines **σ_pipe**, the empirical noise floor of "the same value".
⛔ Not a pass/fail leg — it calibrates legs 2 and 3.

### Leg 2 — CLASS-COUNT INVARIANCE (item 3 — the disease the old axis died of)
Same features, same partition, same fitted store. Only the **realised class count of the evaluation** changes: rows
are restricted so that `orientation` realises 6 of 15 values and `scale` realises 6 of 8 (the wide corners' realised
counts, measured in `gen-r0-wide-refit` §4.2). The restriction operator is applied to **each axis's own evaluation
set(s)**; for the ratio axes it is applied to numerator **and** denominator together (the *matched* form).
- **Statistic:** `S = |m_restricted − m_full| / |m_full|`.
- **PASS iff** `sign(m_restricted) == sign(m_full)` **and** `S ≤ 0.50` on the per-configuration seed means,
  **and** the sign holds in ≥ 5 of 6 (configuration × seed) cells.

### Leg 2b — THE MISMATCHED-CLASS-COUNT STRESS (ratio axes only, by definition)
Numerator on the restricted (6-class) rows, denominator on the unrestricted (15-class) rows — **the shipped
construction**. Statistic `S' = |m_mismatched − m_matched| / |m_matched|`; **PASS iff** sign preserved and `S' ≤ 0.50`.
⭐ **`dci_c` has no leg-2b because it is not a ratio of two evaluation sets.** That structural fact is the claim, and
it is tested separately as prediction **P-A** below, not waived.

### Leg 3 — CONFIGURATION INVARIANCE (item 2)
Montero (`recomb2element`) and wide (`recomb2element_wide`), 3 seeds each, same partitions, same code path.
- **PASS iff** `sign(m)` agrees in **6/6** (configuration × seed) cells **and**
  `|mean m_wide| / |mean m_montero| ∈ [1/3, 3]` (the program's 3× convention).
- ⛔ A single inverted seed is reported per seed and is a FAIL of the 6/6 clause, never averaged away.

### Verdict composition (fixed now)
An axis **PASSES the invariance check iff it passes leg 1a, leg 2, leg 3, and leg 2b where leg 2b is defined.**
Any single leg failure ⇒ **FAIL**.

### ⛔ The check's own validity gate (item 5)
`leak_raw` is scored **first**. **If `leak_raw` PASSES, the check is wrong: I stop and report that**, and no DCI-C
verdict is issued.

## 5. Predictions (committed before measurement)

| id | prediction | derivation |
|---|---|---|
| **P1** | `leak_raw` **FAILS** overall — leg 3 fails on sign (Montero `+`, wide `−`) | measured, R22/R25 (⚠ not blind) |
| **P2** | `leak_raw` leg 2b `S' ≥ 1.0` and the sign flips | the 6-vs-15-class ratio artefact diagnosed in `gen-r0-wide-refit` §4.2 |
| **P3** | `leak_raw` **PASSES** leg 2 (the *matched* form) | if the artefact is the mismatch, matching removes it; this is what makes leg 2b the discriminating leg |
| **P4** | `dci_c` passes leg 1a **exactly** (`Δ = 0.0`, not merely `< 1e-12`) | permuting wells permutes the rows of the `K × F` importance matrix `R`; DCI-C reads only the column entropies `H(P_c[:, f])`, which are row-permutation-invariant |
| **P5** | `dci_c` passes leg 2 with `S ≤ 0.25` | DCI-C's two normalisers are `log K` (a ladder constant) and the **realised** `H_f`; neither is a declared class count |
| **P6** | `dci_c` passes leg 3: 6/6 signs positive, ratio `0.1299 / 0.1745 = 0.74 ∈ [1/3, 3]` | ⚠ not blind (§1) |
| **P7** | ⭐ **`leak_cc` FAILS leg 2b**, with `S'` reduced by a factor between **1.5× and 4×** relative to `leak_raw` | the chance correction removes the *floor* (`1/c`) but not the *difficulty* term: a 6-way problem is easier than a 15-way one above chance too, so the ratio keeps a class-count dependence. ⛔ The Advisor offered it as a route, not a promise |
| **P8** | **P-A:** `dci_c` is **bit-identical** when handed the full-grid `cardinalities` vs the realised ones on the same restricted rows | `_mi_bits` gains only empty bins (zero contribution); `_entropy_bits` is computed from realised counts |
| **P9** | **P-B (provenance):** my harness reproduces Montero seed-0 DCI-C `0.1903` / `0.0241` to `≤ 1e-6` and the wide margin to within its published `±0.0036` | identical rows, identical key chain |
| **P10** | **P-C:** `well_permuted` equals `correlation` exactly at metric level and within `σ_pipe` at pipeline level | it is the same set partition |
| **P11** | **P-D:** `contiguous` and `balanced_random` DCI-C sit **at or below** `random`'s, i.e. within `random ± 3 σ_seed` or lower | they carry no channel-correlation information; if either *beats* `random`, DCI-C is reading well-size structure and that is a **finding** |
| **P12** | `σ_pipe` (leg 1b) `≤ 0.02` in DCI-C units | the store fit is a 1200-step Adam run from a k-means++ seeding on 32,768 fixed rows |

## 6. What would make this spoke wrong
- The check passes `leak_raw` ⇒ the check has no teeth (P1 falsified) ⇒ **stop**.
- The check fails everything, including `dci_c` ⇒ report FAIL and return the ladder to Head + Advisor. ⭐ A legitimate outcome.
- ⛔ No third candidate metric is introduced under any outcome. That is metric-shopping.
- ⛔ No tolerance in §4 is edited after §5's numbers are measured.

## 7. Flags fixed in advance
`gen_data.condition ∈ {recomb2element, recomb2element_wide}` · `gen_bridge.dec_recipe = v1_capacity_schedule` for the
wide stack / `baseline` is the Montero bridge's own recorded recipe (⛔ the decoder is never used by this spoke — no
axis here decodes; declared for completeness) · `gen_store.k_wells` pinned from `store_plan.json` (`K = 5`) ·
`n_modes = 64` · `n_store_train = 32768` · `n_corr = 16384` · `train_steps = 1200` · `n_probe_train = 8192` ·
`probe_steps = 400` · seeds `{0, 1, 2}` · `jax_enable_x64` default.
