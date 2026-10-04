# w21 `spoke/rel-newseeds` — BRIEF
**NEW TARGET `maze-t-rel-newseeds`, 3 scored runs. Base = `w21/maze-t-seed4` tip bd2468f. Local CPU.**

## THE ONE JOB
**Score the IMPROVED CLU (`CLU-REL-A10`) on words-plus-observation TEST, seeds 3, 4 and 5** — the three new seeds — **so the maze tables are consistent across all six seeds for both configs.**

Right now: A1 has BOTH TEST on **all six** seeds; `CLU-REL-A10` has BOTH TEST on seeds **0–2 only**, plus **free VAL** on 3–5. ⛔ The tables cannot be read across configs until this gap is closed. **That is the whole job.**

## ⛔⛔ THE STANDING RULE (Head, 2026-09-24, ALL WAVES)
**PRE-REGISTRATIONS AND PRE-DECLARATIONS ARE BANNED.** ⛔ Write no PREREG. ⛔ Declare no bar, label, threshold or prediction before you have the number. You will find `PREREG-*.md` all over `/c/chlu/outputs` from earlier waves — ⛔ **do not copy that pattern.**
⭐ **PRODUCE AND PRINT EVERYTHING** — every arm, seed and row, including the bad ones. **The Head decides afterwards what is quotable.**
⭐ **Provenance continues:** config, seed, `param_sha256` + `file_sha256` for every object.
⚠ **The ledger row is a SPEND record, not a prediction — keep it.** Before each scored run, **append only (`>>`)**, stating seed and condition, ⛔ with no expected outcome.

## WHAT IS FROZEN — ⛔ nothing here is re-tuned, re-selected or re-fitted
- The CLU is **`CLU-REL-A10` EXACTLY as the peer froze it**: `peer_w21_clu_improve_e7fdb9b/CLU-REL-A10.json`, sha256 **`0d4ec5b9…`**, built by the vendored `peer_w21_clu_improve_e7fdb9b/w21_clu.py`. ⛔ **Assert that hash before any number.** ⛔ No re-sweep, no re-selection, no new candidate.
- Banks, hold-outs, the exact judge, the w9 state encoder, the codec, the rival architectures and their recipes (8k / 80k steps), the read-path-float matching rule, and the input (observation ⊕ 318 design to every arm).
- ⭐ Rivals are **re-matched to CLU-REL-A10's read-path float count (9,200)** by the standing rule and retrained by the frozen recipe — automatic, and the only rival change permitted.

## THE THREE SCORED RUNS
**Runs 1/3, 2/3, 3/3 = seeds 3, 4, 5** of words-plus-observation on **TEST**, each with the full table:
every rival at **8k and 80k**, the **10×-read-floats diagnostic**, all four controls with the **derangement asserted and printed**, the learned blind arms, the floors, **deletion rows**, **per-step exactness**, **rollout**. ⭐ **Persist every object with both hashes.**

## ⭐⭐ PRINT THESE FOUR THINGS — they are what the wave has learned to look for
1. ⭐⭐ **The `+rel` DIAGNOSTIC — the most important column in your table.** Append the CLU's relational columns to every rival's input at matched floats under the frozen recipe. The peer measured that this lifts the best rival from 0.685 to **0.862** on seeds 0–2, i.e. **most of the margin is the CODING, not the store.** ⛔ Print it on every new seed. It is a diagnostic, never a label.
2. ⭐⭐ **The per-seed CEILING row** — best score any **trained** arm reaches, and **name the arm**. ⛔ Exclude planner columns (ORACLE = 1.000 by construction); print them beside.
3. ⭐⭐ **The four charter-C controls, per seed, against the CLU.** ⚠ On seed 4's **VAL** leg a control **TIED CLU-REL-A10 at 0.8994**, so the store was **not load-bearing** there. ⛔ Check this on TEST and **print it whether it repeats or not.**
4. ⚠ **Rival convergence.** The matched rival is narrow; on seed 3 A1's matched MLP **did not converge at either budget** (loss 0.180@8k → **0.205@80k**). Print final losses for every rival and **flag any non-convergence** — a margin over a rival that failed to train is not a margin.

## ACCEPTANCE
- `test/BOTH_test_seed{3,4,5}.json` with every arm and row; `TABLES.md` (every cell traceable to a JSON); `FINDINGS.md`; persisted objects with hashes; a **zero-fit replay** proving hashes match and numbers reproduce.
- A six-seed table for `CLU-REL-A10` beside A1's, **same rows, same arms, same ceiling definition**.
- Branch committed WHOLE and bundled; `git bundle verify` passes.
- ⛔ **No PREREG file anywhere in your output.**

## ⛔ WHAT WOULD MAKE THIS WORTHLESS
Re-tuning the config, re-selecting a candidate, or quietly dropping the `+rel` diagnostic. ⭐ The point is a **like-for-like** extension of an already-frozen object.

## OUTPUT
`/c/chlu/outputs/w21-rel-newseeds/`.
