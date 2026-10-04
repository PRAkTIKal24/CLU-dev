# w12r spoke/registry-repair — FINDINGS (report to the Hub)

**Agent:** `doc-curator`. **Branch:** `w12r/registry-repair`. **Base:** `base/iter8 = aa91785`.
**Scored runs: ⛔ NONE. Compute: ⛔ NONE. GPU: ⛔ NONE. Background jobs / PIDs launched: ⛔ NONE.
Box-wide kill: ⛔ never issued, never possible.** ⭐ The three `w12-*` wave-12 tmux sessions, their
worktrees, their output dirs and their processes were **never touched, listed or signalled.**
**Window:** `START_UTC.txt` **2026-09-21T12:38:57Z** → `END_UTC.txt`.
⛔ **`$CHLU_CLAUDE_ROOT` was NOT opened at any point.** ⛔ **`/c/chlu/evidence/` was NOT written** —
md5 re-checked at the end of the session and all four source files are byte-identical to arrival:
`negative_results.md` **c3597411aa834304be27110dab115673** ·
`claims_matrix.md` **39be4b5f1d8f29c5058c5a0bef935243** ·
`outputs/w10-registry-fold/negative_results.md` **3b3e34f001cf7e9a1bfbd3078b254850** ·
`outputs/w10-registry-fold/claims_matrix.md` **47833db0756f2226632335d70ae49ba5**.
⛔ **The curator re-derived no statistic.** Every number below is quoted from an agent's own artefact
with the path that holds it. **The one exception is `N324`, which this pass measured itself, and it
says so in the entry.**

---

## 0. WHAT SHIPPED

| file | what it is | size |
|---|---|---|
| `negative_results.md` | the registry **+ the `⌖` w12r REPAIR block**: the `N89`/`N90` renumbering with redirects, the W5/W6 CROSSWALK, and entries **`N316`–`N325`** | 4739 → **5268 lines** |
| `claims_matrix.md` | the matrix **+ a new `§I8.4` / `§I8.5`**, title bumped to **v2.19 (PROPOSED)**: rows **`CM-GP-1`, `CM-GP-2`, `CM-R12-1` … `CM-R12-6`** | 1039 → **1236 lines** |
| `LOG.md`, `FINDINGS.md`, `START_UTC.txt`, `END_UTC.txt` | the working record and the clock | — |

### ⭐ BASE CHOICE, decided mechanically and not assumed

The live `/c/chlu/evidence/` pair is **3794 / 643** lines; the `w10-registry-fold` pair is **4739 /
1039**. Checked before writing a byte: `head -3794 w10.../negative_results.md | diff -` against the
live copy is **EMPTY**, and `head -643 w10.../claims_matrix.md | diff -` differs on **LINE 1 ONLY**
(the v2.18 title bump, prior title preserved verbatim as its tail). ⇒ **the w10 copies are a strict
superset, so this pass bases on them.** ⛔ Basing on the live copies would have silently dropped the
21 `N-I8-*` entries and 13 `CM-I8-*` rows.

### ⭐⭐ ADDITIVITY, MECHANICALLY VERIFIED, NOT ASSERTED

- `negative_results.md`: of the base copy's **4739** lines, **exactly 2** are not present verbatim in
  the new file — **the two renumbered label lines**, and both bodies are carried **verbatim** as the
  tail of the new label (checked with `str.endswith`: both **True**). Everything else is additive,
  including six in-place supersession/annotation notes, each of which **adds** lines and edits none.
- `claims_matrix.md`: **lines 2–1039 are byte-identical** to the base copy (list comparison: **True**).
  Line 1 is the only edited line, and the **entire prior title line is preserved verbatim as its tail**
  (`str.endswith`: **True**, 14,098 → 16,270 chars).

---

## 1. ⛔⛔ DEFECT 1 — `N89` / `N90` WERE ALLOCATED TWICE. RENUMBERED, WITH A REDIRECT AT EVERY OLD NUMBER.

### The decision, and it is precedent rather than invention

| number | entry | status on arrival |
|---|---|---|
| `N89` | *the doctrine flagship: with a real learned `φ` the designed store still never beats kNN-in-φ* | tier A, ⭐ **HUB-CONFIRMED** |
| `N89` | *the "a decoder-only fix will NOT clear the 0.8 floor" prior is FALSIFIED — it reached 0.94* | tier C, `⟲ 2026-09-11` gen pivot, **never version-confirmed** |
| `N90` | *the feedforward-NN floor dominates CLU-gated retry in every cell* | tier A, ⭐ **HUB-CONFIRMED** |
| `N90` | *the designed loss lost to the null control, badly* | tier C, same addendum, **never version-confirmed** |

⭐⭐ **EXECUTED: `N89`(gen-pivot) → `N316`, `N90`(gen-pivot) → `N317`, content VERBATIM.**
This is the rule C2W11 Part C applied to the identical `N302`/`N303` collision — *"executed the Hub's
duplicate-N ruling (`N302 ✯ → N313`, `N303 ✯ → N314`, **content verbatim**)"*, `claims_matrix.md`
line 1, v2.17 block. **The later, unconfirmed pair moves. The HUB-CONFIRMED pair does not.**

### ⭐ A REDIRECT IS PRINTED AT FIVE PLACES, so no citation breaks silently

1. under `### N89` — a disambiguation note naming `N316`;
2. under `### N90` — the same, naming `N317`;
3. at the head of `## ⟲ 2026-09-11 — GEN PIVOT, PHASE 1` — a full REDIRECT block with the table;
4. the registry's **file header**, so a reader loading top-down meets it first;
5. the canonical **REDIRECT TABLE** in `⌖.1`, mirrored as matrix row **`CM-R12-6`**.
⭐ **A grep for `N89` or `N90` lands on a redirect from either direction.**

### ⛔ THE FALSIFIER THIS SPOKE WAS GIVEN, CHECKED RATHER THAN ASSERTED

*"The numbering cannot be repaired without breaking existing citations ⇒ say so and propose the
scheme."* ⭐ **It does not fire.**
1. **No HUB-CONFIRMED number moved.**
2. **Cross-file sweep run BEFORE the move:** `N89` / `N90` appear in `claims_matrix.md` only as
   main-series citations. ⛔ **There was no matrix citation of the gen-pivot pair to dangle, because
   there was no matrix row at all** — which is defect 2, below.
3. ⚠ **ONE residual, DECLARED and not silently fixed:** `N316`'s body reads *"see N90's companion
   note and R20"*. Left **verbatim** under the annotate-in-place discipline; it means `N317`, the
   entry directly below it. Stated in the redirect block, in `⌖.1`, and in `CM-R12-6`.
4. **Two prior statements superseded IN PLACE, not edited:** the iter8 fold's *"NUMBERING PROTOCOL,
   AND WHY IT IS NOT `N316`"* paragraph, and `N-I8-21`(a)'s *"FLAGGED, NOT FIXED"*. ⭐ Both decisions
   were **right when made**, and both supersession notes say so.

---

## 2. ⭐ DEFECT 2 — THE GEN-PIVOT NEGATIVES NOW HAVE ROWS: `CM-GP-1`, `CM-GP-2`

⛔ **What they are, and the row says it first:** *"Neither entry is a CLU negative. Both are negatives
about our own instrument and our own priors."*

- **`CM-GP-1`** — approved wording for both, with the numbers: `N316` shipped decoder **0.5926 ±
  0.0866** → V1 `capacity_schedule` **0.9398 ± 0.0131**, **3/3 seeds** clear of the 0.8 floor,
  reconstruction MSE already **0.0027–0.0036** so pixel error was never what failed. `N317` designed
  `object_weighted` loss **0.5231 ± 0.2923**, ⛔ **worse than the 0.5926 baseline it was meant to
  improve**, against the null control's **0.9398 ± 0.0131**.
  ⛔ **NEVER-QUOTE carried into the row:** the ratio-vs-matched-seen **0.9964** / **0.9656** — the
  held-out cell is one-class, so it divides a **detection** rate by a **discrimination** rate.
  ⭐ **The 0.59 → 0.94 LIFT survives that correction; the ratio-to-ceiling does not.**
  ⚠ And `N317`'s per-seed spread **0.140 / 0.916 / 0.583** is **larger than the effect** — printed
  with the effect or not at all.
  **Source:** `.claude/outputs/gen-decoder-lift.md` §5, as quoted verbatim in the registry entries.
- **`CM-GP-2`** — the standing rule, because it is the part with blast radius: ⭐⭐ ***before reaching
  for a designed objective, exhaust capacity and schedule. A null control of "make it bigger and
  train it longer" is mandatory in any variant search — here it won.***

---

## 3. ⭐ DEFECT 3 — NO w5/w6 ENTRIES. REPAIRED IN TWO PARTS, AND ONE PART IS A REFUSAL.

**The complaint** (`outputs/w10-physics-section-evidence/LOG.md:24-31`): *"`negative_results.md`
(3,794 lines, N1–N313) … contains not a single wave-5 or wave-6 ICLR-long entry —
`grep -in "gym-1|gym-2|illegal"` returns **0 hits**."*

### ⭐ RE-MEASURED ON ARRIVAL, NOT ASSUMED

After `w10 spoke/registry-fold` the same grep returns **6 hits**, all inside `N-I8-16`.
⇒ **the CONTENT half was already closed for the two gym deaths** (the physics spoke read the registry
*before* that fold landed). ⛔ **The DISCOVERABILITY half was not:** every hit sits in the `N-I8-n`
namespace, which a main-series citation search does not reach.

### Part A — the **W5/W6 CROSSWALK** (`⌖.2`), 10 rows. **Now 13 hits.**

Each row carries **the number the finding ALREADY holds** and **the path**: `N-I8-16` (GYM-1's
tripwire **0.8520 on 41,657 of 997,896 bytes**, lookup **0.9797** vs **0.3700 / 0.4173**, **2.3×**,
proved over **25.3 M configurations**; the **T1 exact-key-lookup trap** that scored **0.0000** and
would have certified a dead gym sound; GYM-2's **TW-B 0.9825 vs prereg 0.60** and **41.85 %** of paths
bent against 1-D's **0 of 25,261,855**; ⭐⭐ **the ILLEGAL-rule law**; the law **satisfied** by CMAZE-1
and ⭐ **`MiniGrid-DoorKey-*` rejected at 1.0000 composition-blind by construction**, `ENV.md:117`) ·
`N-I8-6` (refusal closed, **8 bytes**, **(1.0000, 1.0000)**, **59,392×**) · `N-I8-7` (lifetimes,
**+0.0562 / +0.0337 / +0.0112**) · `N-I8-13` (dynamics worse than decoration) · `N-I8-14` (⛔ **the
Advisor's Mexican-hat / interpolation law does not exist** — *"interpolation"* **0** occurrences,
*"Mexican hat"* **1**, and ⭐ **the nearest real object INVERTS it**) · `N-I8-15` (**51 / 81 / 59 ULP**
against a **1-ULP** bound).

⛔⛔ **A REFUSAL, DECLARED RATHER THAN QUIETLY TAKEN.** The obvious repair — allocate main-series
`N318`–`N326` as **aliases** of `N-I8-6/7/13/14/15/16` — **was rejected.** It manufactures exactly the
two-numbers-one-finding state that **`N315`**, this registry's own self-filed entry, registers as a
hazard: *"the second-order damage was a CORRECT citation struck as if it were wrong."*
⭐ **The alias table is nevertheless written out in full, so if the Hub prefers aliases only the
numbers need pinning.** **This is the one place this spoke did not do what the literal wording of the
defect asked, and it is on the record rather than silent.**

### Part B — three w6 negatives with **NO entry anywhere**, now registered

⭐ **These are genuinely missing content, not a namespace problem** — not in `N-I8-n`, not in
`N1–N317`, not in the matrix. Found by `w10 spoke/physics-section-evidence`, escalated at
`outputs/w10-advisor-report.md` §5.

- **`N318`** (tier A) ⚠⚠ **the paper's SHARPEST physics number has its WEAKEST trail.** `1 + γ/(2−γ) =
  1.1385` only at **γ = 0.243303**, which is **not on the recorded scored grid** `[0.002, 0.005, 0.01,
  0.02, 0.05, 0.1, 0.2, 0.35, 0.5]` (`test/s0/results.json → pins.gammas`), and the per-cell slope
  rows are **in no machine-readable artefact** — `grep -rIl "1.1368\|1.1367"` under `w6-potentials/`
  hits **only** `LOG.md`, `REPORT.txt`, `RESULTS-LEDGER.md`.
  ⭐ **Unaffected and must travel with it:** the **law** D17 reproduces exactly, `> +1` is verified
  over the whole band, B1 **0.0433 / 0.0435 / 0.0431** against a 0.10 bar on a **0.0349** instrument
  floor, B2 **49/49 × 3**. ⛔ **This is a REPORTING gap, not evidence the number is wrong.**
  (`outputs/w10-physics-section-evidence/EVIDENCE.md:166,320,532-558`.)
- **`N319`** (tier B) ⛔ **D18 is INCONCLUSIVE, and the shorts are the outlier-free side.** Derived
  **−1.0050** at γ = 0.009917 vs measured **−1.1090** (which the same expression reaches at
  γ = 0.184104). ⭐ The spoke computed the γ ≈ 0.2 candidate explanation (**−1.1204** vs **−1.1090**,
  error **0.011** not **0.1368**, ~12× closer) **and refused to claim it**, on three grounds: γ*_exact
  ≈ **0.1185** puts γ ≈ 0.184 on the wrong side of the branch; **no secant over the recorded grid
  reaches −1.109** (**−1.01309** over [0.002, 0.1], **−1.02356** over [0.002, 0.2]); and the shorts'
  own slopes are **−1.006** and **−1.0020 ± 0.0003**, so **w6's −1.1090 is the outlier, not the
  derivation.** ⛔ **Never write D18 up as a confirmation.** (`EVIDENCE.md:566-596`.)
- **`N320`** (tier B) ⚠⚠ **a TAUTOLOGICAL field ships in the artefacts and reads as a perfect pass.**
  `d14_confinement.max_abs_residual` is **exactly `0.0` on 60 cells, all three seeds**, because it
  compares `hess_eig(α)` with `hess_eig(0) + 2α` — **the same algebra on both sides.**
  ⭐ The hub caught it, said so, and replaced it: the SCORED B6 is `hessian_fd_check`, **3.519e-11 /
  1.202e-10 / 2.699e-11** against a **1e-9** bar. ⛔⛔ **Never quote `d14_confinement`; quote
  `hessian_fd_check`.** ⚠ Separately: D14's **`2α` lift is CONFIRMED** (max |lift − 2α| =
  **1.976e-14**) but ⛔ **the `τ_max = Γ/(2α)` CAP is UNTESTED.** (`EVIDENCE.md:142,312,590-600`.)

---

## 4. ⭐ THE NEGATIVES FOLDED IN — all ten accounted for, none dropped

| # | the task's item | number | new? |
|---|---|---|---|
| 1 | encoder-loop is a WASH; the **DECOMPOSITION** is the finding | `N-I8-4` | already registered |
| 1b | **GPU corroboration, 10× steps / 4× data, swap loss +0.0100 / +0.0129 / +0.0075, 3/3 seeds** | ⭐ **`N321`** | **NEW** |
| 2 | `K ≥ 36` collapse; composition dies above K ≈ 5 at the old pin | `N-I8-5` **+ `N325`** | **QUALIFIED** |
| 3 | the TensorFaces / MLC-VAE **STOP** | `N-I8-18` | already registered |
| 4 | MiniGrid composition-blind; DoorKey **1.0000** blind | `N-I8-16` | already registered |
| 5 | the Advisor's Mexican-hat error | `N-I8-14` | already registered |
| 6 | w9 — composition does **not** transfer to a route task | `N-I8-2` | already registered |
| 6b | the mechanism — plan **1.0000** vs latent **0.13** | `N-I8-3` | already registered |
| 7 | CMAZE-2's rule B untested; **1242 of 1449** routes use one key | `N-I8-17` | already registered |
| 8 | **the w9 refuser does not work, and WHY** | ⭐ **`N323`** | **NEW** |
| 9 | **BAR C.2 fails on all 3 GPU seeds** | ⭐ **`N322`** | **NEW** |
| 10 | **3 pre-existing failing tests on `base/iter8`, UNOWNED** | ⭐ **`N324`** | **NEW** |

### `N321` (tier A) — ⭐⭐ the decomposition, corroborated, and a programme-wide qualifier

⛔⛔ **The sentence, and no looser paraphrase is admissible: the net is a WASH BECAUSE TWO REAL EFFECTS
CANCEL** — unfreezing **hurts**, the swap loss **repairs**. CPU: A′ **5/6 negative** at p ≤ .0013,
A″ **4/6 positive** at p ≤ .0003 (`_thinkpad/w8/w8-encoder-loop/LOG.md:798-846`).
⭐⭐ **GPU, BAR E, probe as a margin over random init** (`outputs/w8-BAR-D-fold.md` §3): A″ =
**+0.0100 / +0.0129 / +0.0075** — **positive on all three seeds**, on a different machine, at
`joint_steps=4000, dec_refit_steps=1800, warm_steps=4000, n_train=32768` against the CPU leg's
`400 / 400 / 1500 / 8192`. ⛔ **A′ is NOT consistent** (+0.067 / −0.007 / +0.111) and is **not** a
result. ⛔⛔ **Separate table, never merged rows:** *"a run at one budget and a run at another do not
share a table … it can corroborate or contradict the DIRECTION; it cannot add rows to it."*
⚠⚠ **AND THE MOST USEFUL THING IN THE ENTRY IS A DEFECT IN THE CONTROL:** `alone`'s probe margin is
**negative on two of three seeds** (**−0.0505**, **−0.0862**) — the trained encoder probes worse than
an **untrained** one — with a random-init floor constant within a seed (**0.9714 / 0.9681**).
⭐ **The w9 harness found the same failure mode independently, in a different codebase** (seed 1 at BCE
**0.080** / AUC **0.61** vs **0.005 / 0.92**; fixed across all six combinations, worst case
**0.101→0.010** and **0.441→0.754**, **at no extra cost**). ⛔⛔ **Two waves, two codebases: until
`alone` is stabilised, EVERY A′ COLUMN IN THE PROGRAMME measures the CONTROL's variance.**
**Cost, measured:** A100-40 spot, 3/3 seeds complete, **no pre-emptions**, wall clock **2160 / 2152 /
2159 s**, **$1.02 / $1.02 / $1.25 = $3.30** + **$0.57** staging = ⭐ **$3.87 all-in** (the **$26.46**
pre-estimate assumed the full 6 h; the jobs took ~0.6 h).
⛔ **BAR D NOT SCORED — a DECLARED NOT-RUN with two real blockers:** it is a scored run needing a new
PREREG (the pins differ) **and** a `/c/chlu/SCORED-LEDGER.md` row from a deliberately non-inheritable
guard (`encoder-loop` has **2** runs left); and the ledger is on the ThinkPad while the caches are
**28 GB** on the Mac. ⭐ **The Hub did NOT fabricate a ledger to get past the guard.**

### `N322` (tier A) — ⛔⛔ the equal-seconds control was under-funded, in the treatment's favour

`alone_wc` **297.6 / 312.8 / 318.2 s** against `loop_swap` **338.8 / 340.6 / 336.8 s** ⇒ shortfalls
**−41.2 / −27.8 / −18.6 s**; the PREREG requires `alone_wc ≥ loop_swap` and ⛔ **it is short every
time.** ⭐ **Cause from the job log:** `alone_wc` sized its steps from **`alone`'s** measured
seconds/step (*"matching loop_swap's 338.8s at alone's 0.0531s/step → 6376 steps"*) and then ran
faster than that estimate ⇒ ⛔ **6–12 % LESS compute than the treatment.**
⭐ **BAR C.1, equal COMPUTE, PASSES 3/3** — `gradient_steps` **4000**, `images_seen` **256,000**,
identical across arms. ⇒ **the two bars are different and must never be conflated.**
⭐ **The fix is one line and costs nothing: size a seconds-matched control from its OWN measured rate.**

### `N323` (tier A) — ⛔⛔ the refuser fails, and the failure is a property of the TASK

Separation **+0.047 [0.000, 0.115]** (`cmaze1`) and **+0.020 [−0.005, 0.050]** (`cmaze1-easy`); class
weighting moves the operating point (seed 1 refused 100 % of everything), **never the separation**.
⭐⭐ **A ridge probe on the same observation at t = 0 is at CHANCE — AUC 0.48–0.55 across all six
seed × tier runs**; by-budget 0.44–0.63, by-legality 0.54–0.85, **the budget scalar alone 0.38–0.46**
⇒ ⭐ **no shortcut exists either.** (`_thinkpad/w9/w9-harness/LOG.md:222-236`.)
⭐⭐ **RECORDED AS THE TASK'S PROPERTY, NOT A WIRING BUG: deciding solvability-within-budget is close
to solving the maze.** ⛔⛔ **Binding consequence: the supervised refuser is the ONLY comparator for
DISPLAY rows, so every wave-9 DISPLAY refusal comparison is NOT YET INFORMATIVE** — written into
`HARNESS-READY.json` as a caveat, not left in prose. ⭐ **What survives as a CHECKED property of the
code:** `assert_unseen` **0** impossible instances in any non-refuser arm, all seeds, both tiers;
`certify_impossible` **0** failures over **2700** impossible instances.
⚠ **Mandatory companion `N-I8-19`** (refusal is novelty detection: **0.9781** vs **0.9801**;
false-refusal **0.8182** vs prereg **0.20**) — ⭐ `N323` is **why** that row has no baseline.

### `N324` (tier B) — ⚠⚠ UNOWNED: `base/iter8` at `aa91785` ships three red tests

⭐ **VERIFIED AT THIS PASS, NOT ASSERTED.**
`python -m pytest tests/test_gen_value_holdout.py -q --no-header -p no:cacheprovider`
⇒ ⛔ **`3 failed, 28 passed in 63.11s`**, in the `w12r/registry-repair` worktree at `aa91785`,
`git status` clean before and after (the two emitted coverage files were deleted).
⛔ **A unit-test invocation, NOT a scored run and NOT an experiment; no SCORED-LEDGER row.**

⭐ **One root cause: `AMENDMENT 1` replaced the accept-leg `row2_unseen` with `row1_seen` in the
module, and the test file was never moved with it.**
1. `test_rows_5_and_6_are_both_controlled_by_row_2` (`tests/test_gen_value_holdout.py:300`) asserts
   `controls == row2_unseen`; ⛔ the module ships **`row1_seen`** (`chlu/data/gen_value_holdout.py:1433-1434`,
   docstring table `:81-85`).
2. `test_the_detector_fires_when_a_structural_giveaway_is_injected` (`:313`) copies **5** artefacts to
   `tmp_path`; AMENDMENT 1 added a **sixth** ⇒ ⛔ `FileNotFoundError: … query set 'row1_seen' is
   registered at …/row1_seen.json but the file is missing -- the audit cannot be run`
   (`chlu/data/gen_splits.py:1006`).
3. `test_the_grid_reach_diagnostic_separates_row_5_from_row_2` (`:338`) reads `reach["row5_vs_row2"]`
   and `reach["row6_out_of_range_vs_row2"]`; ⛔ **neither key exists** — the report emits
   `row5_vs_row1`, **`row5_vs_row2_SUPERSEDED`**, `row6_out_of_range_vs_row1`,
   `row6_duplicate_factor_vs_a7` (`chlu/data/gen_value_holdout.py:1679-1690`).

⛔⛔ **THE REPAIR IS IN THE TEST FILE, NOT THE MODULE.** The module is the corrected side and keeps
`row5_vs_row2_SUPERSEDED` deliberately: *"it is retained because it is the FINDING — 1.0000 on
`n_offgrid_terms` is why row 2 could not be row 5's accept-leg"* (`:1670-1674`). "Fixing" the module
would reinstate the defective accept-leg AMENDMENT 1 exists to remove.
⛔ **UNOWNED, DECLARED OPEN.** Three assertion edits and one filename in a copy list. **No run.**
⚠ **Why it matters:** a red suite on the baseline branch means the next wave cannot tell a **new**
breakage from an inherited one.

### `N325` (tier A) — ⛔⛔ the `K ≥ 36` verdict SURVIVES; the PHRASING does not

⭐ **The finding stands** (covariance costs `Σ_k d_k²`; the pin forces K ≥ 36; **0.0648** at K=40/A=7
against **0.8918** at K=5, which costs **7.48×** the pin). ⛔ **The unconditional form is not
supported:** the hub's own 18:32Z `ward_balanced` sweep lists **five** float64 configurations that fit
the pin (K = 36 / 38 / 40·A=9 / 42 / 48; spans **144 / 266 / 360 / 504 / 864**; **523,680 / 523,744 /
518,080 / 522,704 / 521,984 B**) and **composition was measured on exactly one of them** — and that
one (**K = 40, A = 7, span 280**) sits **below the sweep's own K = 40 row** and below the composing
object's realised span of **308**. ⛔ **Three of the five carry MORE deployed span than the object
that composes at 0.8918.** ⛔ **Dtype-conditional too:** a **float32** point (**K = 21, A = 94,
517,952 B = 0.988×, span 336**) was **excluded by argument and never measured**.
⭐ **Two qualifiers close it at ZERO cost** — written out verbatim as approved wording in `N325` and
`CM-R12-4`. ⛔ **The runs that would close it properly are NAMED, not requested:** five val legs on
`single-design`, whose scored budget is **3/3 intact**. **This pass spends none and recommends none.**
(`outputs/w10-referee/REFEREE-v0.md` §MF16; `_thinkpad/w8/w8-single-design/LOG.md:99,593-620`.)

---

## 5. ⛔ DECLARED OPEN ITEMS — every one with its reason. **Nothing was silently dropped.**

| # | open item | reason it is open | what would close it |
|---|---|---|---|
| 1 | **`N324` — three red tests on `base/iter8`** | ⛔ **UNOWNED.** `tests/` on `base/iter8` is not this spoke's; editing it is outside the task file | three assertion edits + one filename. **No run** |
| 2 | **`N318` / `N319` — the `+1.1385` and D18 artefact trails** | the closing action is a **re-emission on the w6 branch**, which this spoke does not own | emit per-cell `(γ_window, slope_measured, slope_derived)` rows to JSON. ⭐ **Not a measurement — no scored run** |
| 3 | **`N325` — is `K ≥ 36` a threshold or a point?** | ⛔ **requires runs**, and `single-design`'s 3/3 budget is not this spoke's to spend | composition on 4 unmeasured float64 pin-fitting configs + the float32 K=21 point = **five val legs** |
| 4 | **`N321` — BAR D** | ⛔ **a scored run**: needs a new PREREG (pins differ) **and** a SCORED-LEDGER row; caches are 28 GB on another machine | `rivals` + `score` + cross-arm report. `encoder-loop` has **2** runs left |
| 5 | **Defect 3 repaired by CROSSWALK, not alias numbers** | ⛔ **a deliberate refusal** on `N315`'s authority — two numbers for one finding is the registered hazard | ⭐ the alias table is written; the Hub pins numbers if it disagrees |
| 6 | **`N-I8-21`(b) is now on its THIRD instance** (BAR D void · the `alone` control · the equal-seconds sizing) | ⭐ promoting a wave-scoped pattern to a STANDING rule is the Hub's call, not the curator's | a Hub ruling; **no run** |
| 7 | **All `⌖` tiers are CURATOR-PROPOSED** (`N318` A · `N319` B · `N320` B · `N321` A · `N322` A · `N323` A · `N324` B · `N325` A) | ⛔ the curator never self-confirms | Hub confirmation |
| 8 | **`v2.19` is PROPOSED**; v2.18 / v2.17 / v2.16 / v2.15 stay PROPOSED | ⛔ this pass does not confirm another fold's version | each version's own Hub |
| 9 | **`§I8.3` items 3–6 (the iter8 fold's own open flags)** | ⛔ outside this task's three defects | restated as still open in `§I8.5` item 5 rather than dropped |
| 10 | **`N316`'s body says "see N90's companion note"** | ⛔ annotate-in-place discipline — the body is carried **verbatim** | nothing; it is declared in three places and means `N317` |

---

## 6. ⛔ WHAT WOULD PROVE THIS SPOKE WRONG — checked against the result

The falsifier: *"the numbering cannot be repaired without breaking existing citations ⇒ say so and
propose the scheme; do not renumber silently and leave dangling references."*

- ⭐ **It does not fire** — §1's four-point check, run **before** the move, not after.
- ⭐ **Nothing was renumbered silently:** five redirect sites in the registry plus matrix row
  `CM-R12-6`.
- ⭐ **No dangling reference was created:** the pair that moved had **no matrix row at all** — the very
  gap `CM-GP-1` / `CM-GP-2` close — and no HUB-CONFIRMED number moved.
- ⛔ **The one place this spoke did NOT do what the defect's literal wording asked** is the alias
  refusal in §3, and it is written into the registry, the matrix and this report rather than
  paraphrased away.

**The acceptance criterion, checked:** *"every entry carries a number and a path · every renumbered
entry carries a redirect line · a verdict recorded without the measurement that decided it is not a
registry entry."*
- ⭐ **`N316`–`N325` (10 entries) and `CM-GP-1`–`CM-R12-6` (8 rows) each carry at least one number and
  the file path that holds it**; the 10 crosswalk rows each carry the number the finding already has
  plus its path.
- ⭐ **Both renumbered entries carry a redirect**, at five sites.
- ⚠ **`N324` is the only entry whose deciding measurement this pass produced itself** (`3 failed, 28
  passed`), and the entry names the exact command, the commit, and the fact that `git status` was
  clean before and after.

---

## 7. COST

⛔ **Zero scored runs. Zero compute. Zero GPU. No cluster leg. No background jobs. No PIDs launched.
No box-wide kill possible or issued.** ⭐ **Wave 12's three `w12-*` tmux sessions, worktrees, output
dirs and processes were never touched, never listed, never signalled.**
One session, reading and writing documents, plus **one** unit-test invocation (63.11 s, local CPU)
named in `N324`. Deliverables are in `/c/chlu/outputs/w12r-registry-repair/`, committed on branch
`w12r/registry-repair` under `w12r_registry_repair/`, and bundled to
`w12r-registry-repair.bundle` — **that branch only**, verified, `list-heads` showing exactly one ref.
