# Task: gen-baselines-consolidate — the report that was never written, on runs that already exist

- **Agent:** `results-analyst` · **Output:** ⭐ **`.claude/outputs/gen-baselines.md`** (the missing deliverable) + artefacts under `.claude/outputs/gen-baselines-consolidate/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all, **§5 output format**) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R26 rulings 1 and 3, and the byte-budget consequence** · `.claude/tasks/gen-baselines.md` (the original task the runs were made against) · `.claude/outputs/gen-baselines/PREREG.md` **including amendment A5** · the run artefacts under `.claude/outputs/gen-baselines/runs/` and `seed{0,1,2}.log`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3faef27`. **Repo: READ-ONLY on `chlu/`.** ⛔ No branch, no worktree, no production code. Scratch analysis scripts under your own output dir are fine.
- ⛔ **PRECONDITION — mechanical:** A7 must exist. `git cat-file -e main:tests/test_gen_c_set_a7.py` succeeds **and**
  the A7 artefact is on disk. If not, **stop and report** — the C re-score is the point.
- **Env:** main venv. ⛔ CSF3 unreachable — local CPU only.

## ⭐ DIAL DECLARATION
- **Dial:** *none — competitor arms, consolidated.* ⛔ No CLU number is produced here.
- **Laundering control:** N/A — **these arms ARE the control.**
- **Falsifies:** N/A for a dial.

## What happened, so you know what you are walking into
The `gen-baselines` spoke **built both arms and ran all three seeds** (exit 0 at 15:20 / 16:27 / 17:10 on 2026-09-12,
8 commits on its branch) and then **ended without writing its report or its post-change suite run.** ⭐ **The runs are
on disk and the numbers exist. The deliverable does not.** ⛔ You are not re-running anything.

## The one job
Write `.claude/outputs/gen-baselines.md` from the existing artefacts, with the C numbers **re-scored under A7** and
the byte position **declared**.

## Three corrections the runs predate
1. ⭐ **The C numbers are contaminated.** All 36 `duplicate_factor` records carry **7 terms** against A/B's **6**, so a
   term counter alone scores **AUROC 1.0000** on that half and **0.7500 pooled**. Every arm inherits it.
   ⭐ **Re-score C against A7** (valid requests carrying a redundant term — 7 terms, still satisfiable, built from
   valid data alone). Report **both** columns, contaminated and controlled, ⛔ clearly labelled, never merged.
   ⚠ Expect the naive arms to collapse: the certificate's in-house control fell **0.8372 → 0.5630** under A7.
2. ⭐ **The kNN arm is OVER-resourced and it is declared, not fixed** (Head ruling). It was byte-matched to
   **688,128 B** (store + the 163,840 B coupling field), relayed mid-run as amendment A5. With conditional dreaming
   dropped (R26 ruling 3) the coupling leaves the deployed store, so the true budget is **524,288 B** — the kNN arm
   carries a **~31 % byte advantage**.
   ⭐ **Declare it IN THE TABLE ROW, not in a footnote.** Precedent: over-budget is reported as an unbounded upper bound.
   ⛔ **TRIGGER:** if the kNN arm **beats or ties** CLU on any headline cell, a matched re-run at 524,288 B becomes
   **mandatory before that cell is quoted** — because then *"does it win at parity?"* is the whole result. Say so
   explicitly in your report if the trigger fires.
3. **LACE:** report both Langevin temperatures and its best; carry the declared concessions (dense factor labels, a
   validation split for the OR/NOT constants) **as stated asymmetries, not caveats** — ⭐ they make LACE the stronger
   baseline, which is the direction we want.

## Acceptance
1. `.claude/outputs/gen-baselines.md` exists in protocol §5 format, with a **flag-provenance table** (commit, seeds,
   every non-default flag) and every number traced to a file. ⛔ No number that lives only in the report.
2. C reported in **both** forms — contaminated and A7-controlled — per kind and pooled, ⛔ never in one column.
3. The byte position declared in the table row; the trigger evaluated explicitly.
4. Both LACE temperatures; the tuning-trial budget per arm.
5. The spoke's own `PREREG.md` scored, **including its falsified predictions.**
6. ⭐ **Anything the artefacts cannot support is reported as NOT RUN** — in particular the missing post-change suite
   run. ⛔ Never a zero, never a silent gap.
7. ⛔ **No kill-rule verdict** — `gen-cell-run`'s job.

## What would prove this spoke wrong
- ⭐ **You quoted a C number without saying which set it was scored against.** That is the exact error that made the
  certificate's result look equivocal until it was controlled.
- You presented the kNN arm as byte-matched. It is not; it is over-budget, and that is the honest, harder-for-us
  position.
- You inferred a number the logs do not contain rather than reporting NOT RUN.

## ⟲ AMENDMENT 2026-09-13 — R27, binding
- ⛔ **Kill rule (ii) reads the BYTE-MATCHED `knn_proto` arm (unbounded channel), as pre-registered.** ⛔ Do not
  substitute the channel-matched number — that would be an amendment made after seeing a baseline result, in the
  kinder direction. ⭐ If (ii) fires it kills the **composition** claim only; refusal and structural deletion were
  never conditioned on it and do not inherit its verdict.
- ⛔ **Kill rule (iv) reads the ORACLE bound (0.8373 ± 0.0109), not the deployable (0.7998 ± 0.0443).** The rival
  gets its best configuration chosen with oracle access; we do not.
- ⭐ **C results are ALWAYS reported BY KIND; pooled is a companion only.** kNN covers `out_of_range` (0.8045) and
  fails `duplicate_factor` (0.3907); LACE is the mirror (0.3971 / 0.9991). **Neither rival covers both kinds and
  ours does — that differentiator is invisible pooled.** ⛔ Not quotable until the rivals are re-scored under A7.
- **LACE whitening exactness gates its configuration:** a seed whose pre-registered selection yields a non-exact
  whitening falls back to the nearest exact configuration, **or** runs the general form with the prior score
  included — both reported, **the stronger quoted**. ⛔ Never-weaken-a-competitor applies to our own selection code.
- **New arm `CLU-wide`** (built by `gen-read-compose`) sits beside the 30-bit read: composed per-well mode vector,
  weights from the **request**, never from features. Report its realised channel width beside `knn_proto`'s.
