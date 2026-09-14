# Task: gen-prereg — commit to the numbers before the cell runs

- **Agent:** `results-analyst` · **Output:** `.claude/outputs/gen-prereg.md` + **`.claude/outputs/gen-clu-phase1/PREREG.md`**
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all, **especially the §5 pre-registration rule**) · **`.claude/tasks/gen-clu-phase1-rulings.md` — the whole file is binding** · the four build reports: `gen-feature-bridge.md`, `gen-store-grouping.md`, `gen-dream-sampler.md`, `gen-certificate.md`, `gen-baselines.md`.
- ⛔ Read no campaign ledger, charter or handover — **EXCEPT `.claude/handover_context.md` §7 Known Issues** (§7 only).
- **Repo:** read-only on `chlu/`. **Zero worktrees, zero branches, zero tracked-code edits.**
- ⛔ **PRECONDITION — mechanical.** All four of these must exist on `main`:
  `chlu/core/gen_store.py` · `chlu/core/gen_dream.py` · `chlu/core/gen_certificate.py` · `chlu/eval/rivals/gen_knn.py`
  (`git cat-file -e main:<path>` for each). ⛔ If any is missing, stop — the prereg cannot predict arms that do not exist.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** *none — the instrument that makes the cell interpretable.* ⛔ **You run nothing and measure nothing.**
- **Laundering control:** N/A.
- **Falsifies:** N/A.

## The one job
Write `PREREG.md` — the point predictions, the thresholds and the protocol — **before `gen-cell-run` executes**, and
end it with the line `status: filed`.

## Why this spoke exists separately
A pre-registered prediction that survives is evidence; one that fails is a finding; **an un-pre-registered agreement is
neither.** The cell cannot pre-register its own predictions credibly once it has begun running, so the prediction is
frozen in a file first and the runner is gated on it.

## What `PREREG.md` must contain
1. **A point prediction for every cell entry** — per-factor probe accuracy on A and on B, **stated as a fraction of the
   bridge's measured `ceiling_B`/`ceiling_A`** (R3.2), for each of the four arms: CLU · store-removed · LACE · kNN.
   ⭐ **Say how each number was derived.** A prediction with no reasoning is a guess wearing a lab coat.
2. **The three refusal AUROCs**, separately: A-vs-C, **B-vs-C**, class-balanced A∪B-vs-C; plus risk-coverage and AURC.
   ⚠ **B is 72 of 480,000 — 0.0150% of A∪B.** Note explicitly what that does to a pooled number.
3. **The four kill rules, verbatim from R7**, each with the numeric threshold and the spread convention that decides it.
   ⚠ Kill rule (i) is read **relative to the ceiling**, not absolutely — carry the amendment and show the original.
   ⚠ **Which AUROC kill rule (iv) binds on is a Head ruling.** If it is still open when you write, record **both**
   candidate forms and mark the choice `PENDING HEAD` rather than picking one.
4. **The sampler declaration**, lifted from `gen-dream-sampler`: family, steps, step size, noise convention, and the
   Metropolis yes/no **with its argument**.
5. **The declared concessions and asymmetries** (R9.3, R9.7): LACE gets dense factor labels and a validation split for
   its OR/NOT constants; our store gets neither. ⭐ **This makes LACE the stronger baseline, which is the direction we
   want** — state it that way, plainly, rather than burying it.
6. **The protocol constants:** seeds (≥3), **samples per held-out combination**, and the **tuning-trial budget per arm,
   equal across arms**. ⛔ With B at 72, samples-per-combination decides how much of the spread is sampler noise — fix
   it here, not later.
7. **What each kill rule would look like if it fired**, in one line each, so nobody re-reads the rule afterwards and
   discovers it was ambiguous.

## ⛔ Boundaries
- ⛔ **Run no experiment. Compute no cell number.** If a prediction needs a quantity, take it from a build report and cite
  the file — do not generate it.
- ⛔ Do not soften, reinterpret or renumber a kill rule. If one now looks wrong, **say so in your report** and leave the
  rule intact in `PREREG.md`. Changing a kill rule is a Head + Advisor decision, never an analyst's.
- ⛔ Do not write `status: filed` until every section above is complete.

## Acceptance
1. `PREREG.md` exists, covers items 1–7, and ends with `status: filed`.
2. Every predicted number carries its derivation.
3. Every quantity borrowed from a build report cites the file it came from. **No number that lives only in this file.**
4. Any still-open Head ruling is marked `PENDING HEAD`, not resolved by you.

## What would prove this spoke wrong
- A prediction you cannot defend, written to fill a row. ⭐ **An honest "we do not know, here is the range and why" is a
  better prereg entry than a confident number chosen to be safe** — a prediction hedged into unfalsifiability is worth
  nothing when the cell lands.

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
