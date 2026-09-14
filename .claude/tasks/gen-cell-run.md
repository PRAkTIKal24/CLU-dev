> ⛔ **SUPERSEDED 2026-09-13 by R29's pivot and by `.claude/tasks/gen-cell-harness.md`.** Composition is released
> as a claim, conditional dreaming is retired as the mechanism, and the headline table is now scored by an
> external image classifier. ⛔ Do not spawn this file. Kept for the record only.

# Task: gen-cell-run — run the A/B/C cell and render the kill-rule verdicts

- **Agent:** `results-analyst` · **Output:** ⭐ **`.claude/outputs/gen-clu-phase1.md`** (the phase deliverable) + artefacts under `.claude/outputs/gen-clu-phase1/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · ⭐ **`.claude/outputs/gen-clu-phase1/PREREG.md` — binding, and you do not edit it** · `.claude/tasks/gen-clu-phase1-rulings.md` (whole file binding) · the five build reports.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Repo:** read-only on `chlu/`. Runs existing entry points; **flags defects for the engineer rather than fixing them.**
- ⛔ **PRECONDITION — mechanical.** `.claude/outputs/gen-clu-phase1/PREREG.md` exists **and** contains the line
  `status: filed` **and** contains `hub_approved: true`. ⛔ If `hub_approved` is absent, **stop** — the Hub, Head and
  Advisor review the prereg before the cell runs. Running first would void the pre-registration.
- **Env:** `uv sync --extra gen`. Launch via `scripts/csf3/job_gpu_gen.sh` with `STAGE=cell`; every submit line carries
  `GEN_GIT_COMMIT=$(git rev-parse HEAD)`. **Budget: ≤30 min per dataset × seed × arm on one GPU.**

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *admission* (the C cell) and *isolation* (the B cell).
- **Laundering control:** the **store-removed** arm on B, and the **plain distance threshold** on C. Both run on every
  number; neither is a footnote.
- **Falsifies:** any of the four kill rules in R7. ⭐ **A fired kill rule is a legitimate outcome of this phase.**
- **Does NOT falsify:** losing a margin to **LACE**, which is label-supervised where we are not (R7). Reported from
  phase 1 onward, never phase-killing.

## The one job
Run the pre-registered A/B/C cell — **A** seen combinations → generate · **B** Montero held-out combinations → generate ·
**C** structurally invalid requests → refuse — across **four arms × ≥3 seeds** on **Shapes3D**, and render the kill-rule
verdicts against `PREREG.md`.

## Non-negotiables
- **Arms:** CLU · store-removed · LACE · kNN. **All four decode through the ONE shared frozen decoder** — assert its
  parameter hash for every arm before any number is recorded.
- **Request sets** come from `chlu/data/gen_splits.py` and `c_invalid.json`, byte-identical across arms. ⛔ Never regenerate.
- **Every arm gets the same tuning-trial budget**, and it is reported.
- **≥3 seeds on every number**, with spread. Samples-per-combination is fixed by `PREREG.md`. ⚠ **B is 72 combinations**
  — say plainly how much of the spread is sampler noise versus arm difference.
- **Report all three AUROCs separately** (A-vs-C, **B-vs-C**, class-balanced A∪B-vs-C) plus risk-coverage and AURC.
- **Kill rule (i) is read relative to the decoder ceiling**, not absolutely (R3.2).
- **Sampler audit traces** from `gen-dream-sampler` are carried into the deliverable, not summarised away.

## ⛔ The rule that matters most
**A failed kill rule returns to the Head and the Main Advisor. It does NOT return to more tuning.**
⛔ Do not re-tune, do not re-seed hunting for a pass, do not reinterpret a rule that fired, and do not soften the
wording. ⭐ **A phase that fails a pre-registered kill rule cleanly is a better result than one that passes by
adjustment** — the first is evidence, the second is nothing.

## Honesty rules, restated because this is where they get bent
- **A result exists only if it is on disk.** Every number traces to a file; no number lives only in the report.
- **Anything not run is reported as NOT RUN** — never as a zero, never as a failure.
- Flag-provenance table on every quantitative result: commit hash, seeds, and **all** non-default flags in effect.
- If an arm fails to run, say so with the error. A truthful partial cell beats a confident complete-looking one.

## Acceptance
1. The **A/B/C table per arm**, ≥3 seeds, with spread and the flag-provenance table.
2. **Kill-rule verdicts for all four rules**, each stated as fired / did not fire, with the margin.
3. All three AUROCs + risk-coverage + AURC; per-factor accuracy reported for A and B **separately**.
4. Sampler audit traces carried through; decoder-hash assertion reported per arm.
5. Tuning-trial budget per arm; byte ledger for the kNN arm against the store's.
6. Artifact paths, job ids, git footprint. ⛔ **No paper numbers from this phase** — it is infrastructure plus first evidence.
7. Every deviation from `PREREG.md` listed explicitly, with its reason. ⛔ Silent deviations are the one unrecoverable failure.

## What would prove this spoke wrong
- A verdict that does not follow from the numbers in your own tables. ⭐ Re-derive each verdict from the table
  mechanically, and say which line of `PREREG.md` you read it against.
- An arm that quietly used a different decoder, a regenerated request set, or an unequal tuning budget. Assert all three
  before you record anything.

---

# ⛔ STANDING CONSTRAINT 2026-09-10 — CSF3 CANNOT BE REACHED FROM AN AGENT SESSION
**Its ssh requires 2FA. No spoke has ever submitted a job and none ever will.** ⛔ Do not try, and ⛔ **never report a
cluster run you did not see complete.**
**What you do instead:** run locally on CPU at whatever scale fits, and **emit the exact submit line** the Head will
run — `PAYLOAD_MODULE=chlu.experiments.exp_gen_<stage>`, `ARMS=…`, `GEN_GIT_COMMIT=$(git rev-parse HEAD)`, and every
defaultless axis (`DATASET`, `PRETEXT`, `STAGE`) — plus the realised output paths it will produce. Report every
cluster acceptance item as **NOT RUN** with that line beside it. ⭐ A validated dry run plus a correct submit line is
the *complete* deliverable here; it is not a partial one.

## Environment reminders that have already cost two spokes time
- ⛔ **No `uv sync` inside your worktree** (w6 lesson) — reuse the main venv:
  `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python -m …`, with `CHLU_GEN_DATA_ROOT` set (worktrees have
  no `.claude/`). h5py is in the main venv via the `gen` extra.
- ⚠ **§7.23 will bite any conv module you write.** `test_friction_field` / `test_goldstone` / `test_lattice` /
  `test_twins` / `test_kt` / `test_wormhole` set `jax_enable_x64=True` **at module import** and sort before yours; an
  `eqx.nn.Conv2d` fed an explicitly-float32 array then dies **in the full suite while passing alone**.
  ⭐ **Fix it by casting the input to the module's own parameter dtype** — `chlu/core/gen_bridge.py` is the shipped
  exemplar. ⛔ A float32-pinning pytest fixture is not the fix; it leaves the non-pytest path broken.
- ⚠ **The probe is saturated** — it reads *real* images at **1.0000** on both A and B. Every accuracy loss in this
  phase is the decoder's. ⛔ Never attribute a gap to "a hard probe".

## ⟲ AMENDMENT 2026-09-11 — R23 (Advisor), binding
- **Kill rule (i) reads BALANCED ACCURACY** on the compositional term; **AUROC is a mandatory companion and is never
  the gate statistic.** Floor 0.8×.
- ⛔ **Shared-corner cells (`montero`, `wide-000`) are quoted as a two-class corner and NEVER pooled.** The pooled
  headline runs on the **other 11 cells** (Hub-recomputed: **0.9894 ± 0.0021**, ratio 1.0034, AUROC 0.9993); the
  corner pair is reported alongside (`montero` 1.0000; `wide-000` 0.8542 ± 0.0967, AUROC 1.0000, TPR 0.7083).
- ⭐ **Report TWO ceilings, side by side, on every arm table:** the **decoder ceiling** (what is reachable from real
  features) **and the STORE ceiling** — the in-corner store read, **0.6866** against the **0.9414** real-feature
  ceiling. The 30-bit channel is **not** touched in phase 1; capacity-vs-bypass-risk is a separate later axis.
  ⛔ An arm number quoted against only one of the two ceilings is incomplete.
- ⛔ **The cluster smoke runs before this cell. No exceptions** (R23.6).

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
