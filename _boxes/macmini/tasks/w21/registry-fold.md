# w21 `spoke/registry-fold` — PASS 5 BRIEF (batmac, `doc-curator`)
**ZERO scored runs. ZERO compute. ⛔ Never write to any ledger. ⛔ Never edit a draft.**

## THE ONE JOB
Fold **wave 20 and wave 21** into the negative-results registry and the claims matrix. `registry-fold` pass 4 **declared a fifth pass required** for w20; w21 has since landed too.

## ⛔⛔ THE STANDING RULE THAT CHANGED (Head, 2026-09-24, ALL WAVES)
**Pre-registrations and pre-declarations are BANNED.** ⛔ Write no PREREG. ⛔ Declare no bar, label or prediction before a number.
⭐⭐ **REPORT EVERYTHING, OMIT NOTHING.** Anything previously withheld **because a pre-registration did not cover it** is now **reportable and must be folded with its status**. ⚠ Sweep for these markers and fold each with the number, the status, and why it was withheld:
`DECLARED NOT-RUN` · `VOID`/`VOIDED` · `UNMEASURED` · `never a bar`/`not a bar` · `not reported` · `informational only` · `out of brief` · `withdrawn` · `printed, never the label`.

## WHAT TO FOLD — the evidence is staged in `~/clu/evidence/outputs/`
**w20 `shapes-data-matched`** (3/3 spent, target exhausted). The headline results and every declared item:
- ⭐⭐ Data-matched margin **+7.03** (seeds +7.49/+8.60/+5.02), **+6.54** vs best-of-all-fits; the draft's recomputed margin is **+39.7…+49.1** ⇒ **~6× shrink, no collapse** (lowest CI lower bound +3.73).
- ⭐⭐ **M1's CAUSE WAS MISDIAGNOSED: it is TRAINING SCHEDULE, not data.** MLP on the SAME 16,384 rows at 20,480 steps = **38.6**; at 1,280 steps (the draft's comparator) = **5.5**. 8× more data at matched steps buys ≈ **0.5**. The store is equally data-insensitive (47.28@16k → 48.01@131k). ⭐ This is a **new registry entry in its own right.**
- P1 convergence: mlp FAIL/FAIL/PASS, cont_mlp FAIL×3, lstm & transformer PASS×3; **mlp seed 0 still +1.47 at 30× ⇒ NOT CONVERGED**. All families selected the **edge** of both search grids (lr 3e-3, 40 epochs), grids not widened.
- ⭐⭐ **M3 MEASURED AND REPLICATED:** under the Head's refit definition, deleting `red ∧ sphere` costs survivors **−6.13/−6.43/−6.30** (bar |Δ|≤1.0) with the **20 red non-sphere corners at −32.1**; a **random-1 % control moves −0.02/−0.11/−0.01** ⇒ **content, not instability**. (a)/(b) are exact (≤1e-8, statistic dev **0**). ⭐ The w15 "composition unchanged" row held **only because the book was carried**.
- ⚠ **The ALS book has NO compact exact statistic: 0.97 GB = the fit data itself**; refit costs 2,367–5,913 s. *"Adaptation reads 0 pool rows"* is true but ⛔ **is not a size claim**.
- ⚠ `red ∧ sphere` **not significant on 3 of 12 legs** (CIs straddle 0); rivals lose **<1** where the store loses **6**.
- Ladder margins **+5.13 (n=2), +10.3…+12.6 (n=3–6)**; rebuilt K\*=32 reproduces w12b on **15/15**.
- ⛔ **Existing ladder K=5 arm: n=4/n=5 NOT CLAIMABLE** (strong-form `random_atoms` 1.24 and 2.03 over the 1.0 bar; two controls never run).
- **Defects found and fixed:** generation-1 dev smoke; VOID val `seen_mask` pass; ⭐ **per-seed rung dead-mask ⇒ seed-1/2 rung stores VOIDED and rebuilt**; ladder threshold-width check; one orphan scorer.
- **Declared not-run (all 8), incl.** LACE/PoE@131k on rungs (cost measured), VAL lstm, TENX-on-ladder beyond spot checks, 30× only for the P1 failures, retrain-without at TENX/30×, and ⚠ **`learned_1x` (536k / ~3.2M params) NOT PRINTED** on the read-path matching rule though kept in artefacts.

**w21** — four units:
- `words-full-test`: **WORDS-FULL TEST = LOSS**, Δ̄ −0.0175, σ 0.0140, 0/3 seeds ahead; ⚠ **σ-sensitivity: COMPARABLE under two other defensible spreads**; VAL repro **bit-identical** to w19.
- `maze-t-clu-improve`: **WIN** (BOTH TEST 0.929 vs 0.685, 3/3 seeds) — ⛔ but **~0.18 of the 0.24 margin is the CODING**: rivals given the same relational columns reach **0.862**; CLU keeps **+0.066**, ahead on only 2/3; ⛔ **store contribution ZERO on WORDS-FULL VAL** (rival+rel 0.941 vs CLU 0.937). **68 candidates tried, 57 gate-passers** (its own ledger says 69 — correction noted, rows not edited).
- `maze-t-seed4`: ⛔⛔ **the win row does NOT replicate** — 6-seed mean margin **+0.046** (was +0.099 on 3); ⛔ **seed 5, the EASIEST seed, CLU LOSES by 0.185**, traced to **0.000 on two fully-supported class-A tuples** where 8 arms score 1.000. Easy {0,2,3,5} / hard {1,4} by **measured ceiling**, cut 0.641; sensitivity printed. ⚠ Seed 3's +0.262 inflated by a **non-converged matched MLP**. **Rollout validity 0.000 on all 6 seeds.**
- ⭐⭐ **`CLU-REL-A10` store-gate FAILURE on seed 4 VAL: a charter-C control TIES the CLU at 0.8994** ⇒ the store is **not load-bearing** there.
- ⭐ **The matched-float rule itself is now a measured defect:** a 954-dim-input rival gets **4 hidden units** and **did not converge at either budget** on seed 3 (loss **0.180@8k → 0.205@80k**, worse with 10× training).

## ACCEPTANCE
- New `N…` entries for each finding above, **additive**: prove it with difflib (inserted lines, **0 replaced, 0 deleted**), duplicate-free `### N` headings.
- `⟲ PASS 5` annotations on the entries these supersede — ⛔ **never edit an original entry.**
- Every decimal and every 4+-digit token in new text **verified verbatim against its source file**.
- `FINDINGS.md`, updated `negative_results.md` and `claims_matrix.md`, branch committed WHOLE and bundled, `git bundle verify` passing.

## OUTPUT
`~/clu/outputs/w21-registry-fold/`.
