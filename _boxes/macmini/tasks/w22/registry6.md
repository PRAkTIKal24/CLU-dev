# w22 `spoke/registry-fold` PASS 6 — BRIEF (batmac, doc-curator)
**ZERO compute, ZERO scored runs. ⛔ No ledger write. ⛔ No `.tex` edit, and ⛔ do NOT take the `.tex` lock.**
⛔ **PRE-REGISTRATIONS ARE BANNED** (Head, 2026-09-24). ⭐ **REPORT EVERYTHING, OMIT NOTHING** — anything previously withheld only because a PREREG did not cover it is now reportable, with its status.

## WHY THIS PASS EXISTS
Pass 5 ended **15:32Z**. `rel-newseeds` ended **16:16Z** and `ladder-rivals-32k` **22:03Z** — ⛔ **both AFTER it, and neither is folded.** Pass 5's own registry mentions neither.

## FOLD THESE THREE
### 1. `w21/rel-newseeds` — the improved CLU on all six seeds
⭐ `CLU-REL-A10` BOTH TEST six-seed mean **0.925** vs A1's **0.677**; it **fixes A1's two weak seeds** (seed 1 **0.566 → 0.894**, seed 5 **0.682 → 0.939**) and reaches the **ceiling on all six**. ⭐ Margins do **not** rest on failed rivals: no WORSE@80k on seeds 3–5, best rival unflagged.
⛔⛔ **Three measurements say the STORE is not the cause — fold all three:**
- **`+rel` erases the margin:** CLU − best `+rel` = −0.007 / +0.024 / +0.165 / **+0.006 / +0.000 / +0.000**, six-seed mean **+0.031**; share of margin recovered by handing rivals the columns **0.97 / 1.00 / n.a.**
- ⛔ **seed 4 TEST: `control: random_atoms` = CLU = 0.9004**, matching **exactly on all 16 held-out tuples** (the VAL tie repeated on TEST).
- ⛔ **seed 5: CLU 0.939 == best matched rival 0.939** — no margin.
⭐ **The claim: colour-relational request coding — an INPUT REPRESENTATION carried through the closed-form head. ⛔ NOT a store/memory result.**

### 2. `w21/ladder-rivals-32k` — MF-2 answered
⭐ **MF-2 is real on paper and IMMATERIAL measured.** K\*=32 minus strongest rival, mean over rungs: **A** (32,256 rows / 5,040 steps) **+17.53** · **B** (131,072 rows) **+9.97** · **C** (32,768 rows = the store's own content rows) **+10.28** · **C step-matched +10.75**. Rivals on the store's exact rows land within **±1.5** of the 131k set, **in both directions**. Every CI95 lower bound in B/C/C-sm is **above 0** (lowest **+1.98**).
⭐⭐ **Independently confirms w20: at matched steps the POOL barely moves these rivals; the STEP BUDGET does.** ⭐ Pool identity proven **15/15** legs (`n_fit_rows = n_store_train = 32768`, rung pool identical to the w12 partitions).
⚠⚠ **FOLD THIS AS ITS OWN ENTRY — it bears on every single-fit margin in the paper:** its **X1** check held rows, steps, width, params, trainer and schedule identical, leaving **only the PRNG draw** — and per-seed deltas reach **±4.1 novel/corner with CIs excluding 0 on 10/15 legs**. ⇒ **a single 2,560-step rival fit has RNG-draw variance up to ±4.1**, against a headline margin of ≈+7.

### 3. `w22/figures3` — a figure defect the Head found, now measured and fixed
⛔ CMAZE-T panels drew the resulting **state** while the judge scores the **fact vector**, so facts that do not change the state were invisible. **33,412** arm-pairs rendered identically with opposite marks (of 92,628 opposite-mark pairs); **3,740** differed only in the legality bit and **16,283** only in `pocket-next`. ⭐ **Fixed to 0** by a fact-strip. ⛔ **No number was ever affected** — scoring always used all four facts. ⚠ Also folded: the task figure had `c1` **blue** while the panels drew `c1` **red**; now unified and verified **pixel-identical (7/7 discs)**.

## ACCEPTANCE
Additive, proven with difflib (**inserted only; 0 replaced, 0 deleted**), duplicate-free `### N` headings, every decimal and 4+-digit token verified verbatim against its source file, `⟲ PASS 6` annotations on superseded entries (⛔ never edit an original). `FINDINGS.md`, updated `negative_results.md` + `claims_matrix.md`, branch committed WHOLE and bundled, `git bundle verify` passing.

## OUTPUT
`~/clu/outputs/w22-registry-fold6/`. Extend the pass-5 files in `~/clu/outputs/w21-registry-fold/`.
