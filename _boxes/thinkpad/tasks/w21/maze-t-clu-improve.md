# w21 `hub/maze-t-clu-improve` — BRIEF (autonomous hub)
**Target `maze-t-symmetric`, runs 7-9/9. Base = `w19/maze-t-symmetric` tip 0455961. Local CPU.**

## THE ONE QUESTION
**Widen the CLU's CMAZE-T composition margin under identical inputs, by changing ONLY the CLU.**

w19 established the problem: under symmetric inputs the CLU does not beat a matched rival. On words-plus-observation the CLU reads **0.725** against the best matched rival's **0.626** on TEST but missed its bar; on WORDS-FULL a transformer handed the CLU's own 318-wide design reads **0.742 vs 0.725**. The mechanism: **the pair block carries the margin, and a network given the same pair indicators does as well.** Your job is to find a CLU that does better than that - on the same input, against the same rivals.

## ⛔⛔ FROZEN - never edited, never re-run, never "improved"
Banks · hold-outs · the exact judge · the w9 state encoder · the codec · the rival architectures and their training recipes (8,000 / 80,000 steps) · the read-path-float **matching rule** · the **per-condition input** (words-plus-observation = observation ⊕ 318 design for every arm; words-full = 318 design for every arm).
⭐ Rivals are re-matched to your improved CLU's float count **by the standing rule** and retrained by the frozen recipe. **That is the only rival change permitted, and it is automatic.** ⛔ You do not get to pick a weaker rival.

## OPEN - anything inside the CLU that consumes the same input
Design features (the observation's **dynamics scalars** as design columns, and their interactions with the door / tint / direction words) · pair-block **hierarchical shrinkage** (heavier ridge on pair columns, or mains-then-residual fit) · triple indicators · colour coding · K · atoms · latent · ridge · address-book rank · item vs field store · read verb.

**The Advisor's ranked starting points:**
1. ⭐ **Consume the dynamics scalars** - the CLU ignores the observation today.
2. ⭐ **Pair-block shrinkage** - this targets the measured failure: on unseen 2-way, main-only reads **0.557** while the full design reads **0.410**. The full design is being *hurt* by its pair block on that row.
3. K / atoms / rank re-sweep.

## THE RULE - val-first, and the gate is strict
- Search on the **words-plus-observation** condition, on **VAL** (free).
- ⭐⭐ A candidate is **KEPT only if its VAL n-way beats current A1 on the mean AND on no seed falls below A1.**
  A1 = **0.725 mean**, per seed **0.802 / 0.569 / 0.803**.
- ⛔ **Every candidate you try is recorded in `LOG.md` with its VAL numbers, kept or not.** The rejected ones are the result too.
- The best candidate is **FROZEN in a PREREG (full config + a hash) BEFORE any TEST number exists.**
- Then and only then: **3 TEST seeds = runs 7, 8, 9** on words-plus-observation, with the **full arm table, all four controls, the learned blind arms, the floors, the deletion rows, per-step exactness and rollout**.
- ⭐ Also print the frozen config on **WORDS-FULL VAL (free)**, so the two input classes travel together.
- **3 seeds everywhere.** No exceptions.

## ⭐ THE OUTCOME THAT IS STILL A RESULT
⛔ **If no candidate clears the gate, that is a REAL FINDING and you spend ZERO scored runs.** Report it with every candidate's VAL numbers and what each changed. ⛔ Do **not** relax the gate, do not re-tune it, and do not spend a scored run on a candidate that failed VAL. A negative here is worth more than a rescued number.

## THE LABEL RULE - the Head's, replaces the PREREG margin bars
WIN (ahead on the mean and a majority of seeds, store gate holds) · COMPARABLE (within one seed's spread) · LOSS (behind). No numeric margin bar. Name the mechanism.

## ACCEPTANCE
Either **(a)** a frozen improved config with its PREREG, 3 TEST seeds, the full arm table and every row above; or **(b)** a declared "no candidate cleared the VAL gate", every candidate logged, **0 scored runs spent**. `FINDINGS.md`, `TABLES.md`, machine JSON, branch committed WHOLE and bundled, `git bundle verify` passing.
⭐ The Hub compiles `w21-advisor-report.md` from your `FINDINGS.md` - write that file as the primary source, not a summary.

## BUDGET AND THE LEDGER
Runs **7, 8, 9 of 9** on `maze-t-symmetric` - and **only** if a candidate cleared the VAL gate. ⛔ **Ledger row to `/c/chlu/SCORED-LEDGER.md` BEFORE each run, APPEND ONLY (`>>`)** - a peer agent is appending rows 4-6 to that same file concurrently; ⛔ never rewrite it.
⛔ Runs 4-6 belong to `spoke/words-full-test`. **Do not touch them.**

## OUTPUT
`/c/chlu/outputs/w21-maze-t-clu-improve/` - the only place you write outside your worktree.
