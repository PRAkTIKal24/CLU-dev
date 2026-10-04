# HUB CHARTER — `hub/refusal`

- **Agent:** `autonomous-hub` · **Branch:** `hub/refusal` · **Worktree:** `../CHLU-hub-refusal`
- **Base:** ⭐ **the PIN `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`**
- **Boot:** `.claude/autonomous-hub-charter.md` → `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7 → this file. ⛔ Nothing else.
- **Report:** `.claude/outputs/hub-refusal.md` · ⭐ **running log: `.claude/outputs/hub-refusal/LOG.md`, updated as you go**
- **Limits:** ≤ 10 h · ⛔ **≤ 3 scored runs, all reported** · ⭐ CPU default · ⛔ **GPU only for a headline**

---

## ⭐⭐ YOUR TARGET
**A refusal win: ONE landscape-derived criterion that refuses BOTH kinds of invalid request while accepting
unseen-valid — with ZERO invalid supervision.**
- **Refuse:** `out_of_range` **AND** `duplicate_factor` (A7-controlled). ⭐ **One criterion covering both kinds** —
  that is the claim; each rival covers one.
- **Accept:** unseen-valid. ⭐ **A refuser that refuses everything scores zero** — the two halves are one number.
- **Beat:** kNN **0.8457** (C-vs-B) and **0.9583** (A7-controlled `out_of_range`), LACE **0.8097** — **on both kinds**.
- **Report:** the **(reject-invalid, accept-unseen-valid) pair at ONE threshold**, plus AUROC, **3 seeds**.

## ⛔⛔ THE THRESHOLD RULE — this is the claim, not an administrative detail
⭐ The criterion is a function of **the read's own settled energy and its convergence** — ⛔ nothing else.
⛔ **"No support entry" is NOT a refusal.** An empty-table lookup is a missing row, not a judgement.
⛔⛔ **THE SINGLE THRESHOLD IS CHOSEN ON VALID DATA ONLY** — on the repaired stratified `val` split, e.g. a declared
quantile of settled energy on seen-valid rows — **and written into your PREREG BEFORE any invalid (C-set) row is
scored.**
⛔ **Touching ANY invalid example to choose or adjust it is invalid-example supervision and VOIDS the
zero-supervision claim for that run.** ⭐ *That sentence is the entire result. If it is not true, there is no win
here worth having.*

## ⭐ WHY THIS IS THE CHEAPEST UNCLAIMED TARGET
⭐ **It is scored off artefacts that already exist** — the energy traces, convergence reports, settled states and
decoded tuples persisted by `gen-read-cell-arm`, under
`.claude/outputs/gen-read-cell-arm/runs/`. ⛔ **You should not need a new scored generation run to answer this.**
⚠ **The rivals are already re-scored on the repaired split** (`gen-val-split-fix/refusal_scores.json`, complete,
both byte legs × 3 seeds). ⛔ **`gen-arms-a7-wiring/refusal_scores.json` is SUPERSEDED — never quote it.**

## ⭐ §E.4 — THE MECHANISM FACTS YOU INHERIT (measured, ⛔ do not re-derive)
- ⭐⭐ **The launch beats the settled state 0.6852 → 0.2114 on the generation metric — the dynamics HURT there.**
  ⚠ **But refusal is a different question:** you are asked whether the *energy* separates valid from invalid, ⛔ not
  whether descending it improves a reconstruction. ⭐ **A landscape can be a bad generator and a good detector.**
- ⚠ **Convergence is part of your criterion and it is not clean:** still-descending was **0.72–0.79** at rung A and
  **0.15–0.60** at rung B, and **1.00** on both request-off rows. ⛔ A chain still descending at its budget is
  **NOT CONVERGED** and yields no quotable number — ⭐ **but "did not converge" is itself a signal you may use**,
  provided you declare it in the PREREG before scoring.
- ⭐ **Steering is real:** request-off collapses to 1 tuple of 72, so the request energy carries the distinctness.
- ⭐ **λ comes from the coercivity certificate (≈ 0.054)** — ⛔ never 1.0. The launch is the posterior **argmax**;
  dreaming's thresholded clamp is dead on atoms (0.00 of 5 wells).

## ⛔ Falsifiers
1. **Beats kNN and LACE on BOTH kinds**, beyond seed spread, at one threshold, 3 seeds.
2. ⭐ **Accept-unseen-valid holds** — ⛔ a high rejection rate bought by refusing unseen-valid is not a win.
3. ⭐ **Zero invalid supervision, demonstrable** — the threshold's provenance is in the PREREG, timestamped before
   any C-set row was scored.
4. ⭐ **The store is load-bearing** — `store_removed` destroys the criterion. ⛔ If an energy computed without the
   store refuses just as well, the landscape is decorative.

## ⭐ What would make this hub's answer worthless
- ⛔⛔ **The threshold was tuned, even slightly, using an invalid example.** The claim is *zero invalid
  supervision*; that makes it false and the run unusable.
- ⛔ **Two criteria, one per kind.** ⭐ **One criterion covering both** is the entire novelty — each rival already
  covers one.
- ⛔ **Quoting the superseded `gen-arms-a7-wiring` refusal scores.**
- ⛔ **Reporting AUROC alone.** The (reject-invalid, accept-unseen-valid) pair at one threshold is the result;
  AUROC is a companion.
