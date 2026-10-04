# PREREG — gen-arms-a7-wiring (filed 2026-09-13, ⛔ before any full-run number existed)

**Timing, stated honestly.** Written after the `--quick` plumbing smoke (n_fit 2,048, 24 kNN
trials, **1**-epoch LACE classifier) and ~3 minutes into leg H seed 0, i.e. **before any
number from either reported leg was on disk** (the kNN arm's A7 line lands ≈8 min in, LACE's
≈25 min in). ⚠ The smoke numbers are disclosed below wherever they informed a prediction;
⛔ they are not results and are never quoted as such.

## What is predicted

### P1 — THE REGRESSION (the one that would prove the spoke wrong)
`out_of_range` reproduces **bit-identically** in leg H (historical 688,128 B budget, the
published configuration), because A7 leaves that kind's control at the ordinary valid pool
and because the control sets sit outside `REQUEST_SETS` and draw from an independent PRNG
stream. Targets, read from `.claude/outputs/gen-baselines/runs/.../baselines_manifest.json`:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `knn` = `knn_proto` | 0.6628086419753086 | 0.8753858024691358 | 0.8753858024691358 |
| `lace/ode` | 0.5135030864197531 | 0.3788580246913580 | 0.2989969135802469 |
| `lace/ld_published` | 0.2881944444444444 | 0.5216049382716049 | 0.5050154320987654 |
| `lace/ld_corrected` | 0.6130401234567901 | 0.4544753086419753 | 0.3406635802469136 |

**Falsifier: any difference at all, in any digit.** A non-identical `out_of_range` means the
wiring moved a rival, and the whole controlled column is then unquotable.

### P2 — LACE's `duplicate_factor`, A7-controlled: **0.90, interval [0.70, 1.00]**
⚠ **Two competing hypotheses, both filed, because they predict opposite things.**
- **H2a (mine, predicted):** LACE detects a *genuine* conflict. A `duplicate_factor` request
  names one factor at two different values; LACE's categorical head is one-hot over a closed
  vocabulary, so one of the two terms **must** be unsatisfied and the composed energy rises.
  A7 repeats a factor at the **same** value, so both terms are satisfied. ⇒ the controlled
  number stays high and the published 0.9991 is mostly mechanism.
- **H2b (the alternative):** the 0.9991 was mostly the **term count**. A7 has the same 7
  terms, so the number collapses to ≈0.5. The `gen-baselines-consolidate` prereg's
  "LACE/ode dup **0.95** [0.85–1.00]" is closer to H2a than to H2b.
- ⚠ **Disclosure:** the 1-epoch smoke gave 0.53–0.57, which looks like H2b — but a 1-epoch
  classifier cannot detect a conflict it has not learned to score, so the smoke is
  **uninformative** about the 50-epoch arm and did not move this prediction.

### P3 — kNN's `duplicate_factor`, A7-controlled: **RISES** to 0.55, interval [0.35, 0.80]
Derivation, in two competing parts:
1. **Term count** removes a giveaway that was *against* kNN here (its contaminated dup number
   is 0.39, i.e. **below chance**), so the controlled number should move **up**.
2. ⚠ **A confound I am filing before measuring it:** A7 is built from **train** rows by ruling
   ("held-out rows are never spent on a control"), so the A7 pool is **seen**-valid while B is
   **unseen**-valid. kNN's score is distance-to-store, so seen requests score *lower* and the
   controlled AUROC rises for a reason that has nothing to do with term count.
⇒ **The decomposition, pre-registered:** `A7` vs `A7D` (the same records, redundant term
removed — identical rows, identical seen-ness) isolates the **term-count** effect exactly;
`A7D` vs `B` isolates the **seen/unseen** effect. **Prediction: the seen/unseen leg is the
LARGER of the two for kNN**, and the A7-controlled kNN dup number must therefore be reported
with that decomposition attached or not at all.
⚠ Smoke disclosure: 0.2855 → 0.3839 (up, as predicted in sign).

### P4 — arithmetic identities
- pooled-controlled (companion) `= ½(dup + oor)` **exactly** at 36/36, to machine precision.
- `A7D` embeddings are **bit-identical** to the plain request embeddings of the same rows
  (pinned by a unit test), so any `A7` vs `A7D` gap is the doubled redundant term and nothing
  else.

### P5 — the byte flip does NOT move `knn_proto`'s composition
`prototype_direct_read(embeddings)` never consults the byte-sized store, so leg D (524,288 B)
must return `knn_proto` in-corner balanced accuracy **bit-identical** to leg H's. ⛔ If it
differs, `gen-baselines` §3's "the composition cell is byte-budget-invariant" is wrong and the
Head's mandatory-re-run trigger is **not** a no-op there.

### P6 — R27.3's fallback
The gate fires on seeds **0** (`1e-3`, 0.2382) and **2** (`ledoit_wolf`, 0.1438) and not on
seed 1 (`none`, 0.0567). The fallback leg is `shrinkage="none"` at the same sigma/form on both.
**Prediction: the fallback leg is WEAKER on validation (it already is, by −0.019 / −0.010) but
I predict it is NOT uniformly weaker on the reported metrics** — because no valid-only
statistic selects for refusal power (ρ = −0.588 for LACE). ⛔ Whichever is stronger is quoted;
⛔ neither is dropped.

## What would prove this spoke wrong
1. P1 fails ⇒ a rival moved because of my wiring. Stop, report, do not quote the column.
2. A rival's number falls at the **deployed** budget *and* I present that as a finding rather
   than as the mandated correction of an over-resourced competitor (R26.3).
3. The A7-controlled dup number is quoted **without** the seen/unseen decomposition of P3.
