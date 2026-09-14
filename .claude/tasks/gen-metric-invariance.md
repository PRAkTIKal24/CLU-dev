# Task: gen-metric-invariance — prove DCI-C is configuration-invariant, before R1 is allowed to run

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-metric-invariance.md` + artefacts under `.claude/outputs/gen-metric-invariance/`
- **Read first:** `.claude/AGENT_PROTOCOL.md` (all) · **`.claude/tasks/gen-clu-phase1-rulings.md` — R26 ruling 2 is the brief; R22/R25 give the history** · `.claude/outputs/gen-r0-wide-refit.md` **§4.2–§4.4** (why leakage failed) · `chlu/core/gen_store.py::dci_scores`, `mig_score`.
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3faef27` · **Branch:** `agent/experiment-engineer/gen-metric-invariance` · **Worktree:** fresh (`../CHLU-gen-inv`).
- **Env:** main venv, ⛔ no `uv sync`. ⛔ CSF3 unreachable — local CPU only.

## ⭐ DIAL DECLARATION
- **Dial:** *none — instrument qualification.* ⛔ No arm, no rung result, no CLU claim. ⛔ **You do not run R1.**
- **Laundering control:** the **failed** axis. Probe-transfer leakage is scored through the identical check as the
  negative control, so the check is shown to be capable of failing something.
- **Falsifies:** DCI-C fails the invariance check ⇒ **the ladder has no qualified x-axis and returns to Head + Advisor.**
  ⭐ A legitimate outcome. ⛔ Do not go looking for a third candidate metric — that is metric-shopping and it is
  precisely what the criterion exists to prevent.

## The one job
Run the **pre-registered configuration-invariance check** on **DCI-C** and decide, by that check alone, whether it may
serve as the ladder's x-axis.

## ⭐ The criterion, verbatim (R26 ruling 2)
> **The axis must give the same value for the same features under different partitions.**
It was chosen **before any rung result was seen**, and it is the reason DCI-C is promoted rather than shopped for.
⛔ It is the *only* admissible ground for the choice. ⛔ Do not add a criterion that happens to favour DCI-C.

**Why the previous axis failed, so you test for the same disease:** the wide corners fix `scale` and `orientation`, so
probe-transfer leakage scored a **6-class numerator against a 15-class denominator**. Balanced accuracy normalises
*within* a class count, not *across* one — the weaker arm gained more from the easier numerator and the margin
**inverted sign** between configurations (Montero `+0.1337` → wide `−0.0660`, 3/3 seeds, 4.59× spread).

## What to run
1. **Same features, different partitions.** Take one seed's cached features, hold them fixed, and vary only the
   partition: the correlation grouping, the random grouping at the same `K`, and ≥2 further partitions
   (e.g. a permuted-label partition, a coarser/finer one at fixed `K` where the contract allows).
   ⭐ **The features do not move. Only the partition does.** Report DCI-C for each.
2. **Both configurations** — Montero and wide — ≥3 seeds each. ⭐ **The sign of the correlation-minus-random margin
   must agree across configurations.** Report per seed; a single inverted seed is a finding, not noise to average away.
3. **Class-count stress, the specific failure mode:** score DCI-C where the numerator and denominator realise
   **different numbers of classes** (the wide corners fix `scale`/`orientation`). ⭐ **This is the test the old axis
   fails. Show whether DCI-C is immune, and by how much.**
4. **The companion:** chance-corrected balanced accuracy `(BA − 1/c)/(1 − 1/c)`. Run it through the **identical**
   check and report whether the correction actually removes the class-count dependence — ⛔ if it does not, say so;
   the Advisor offered it as a route back to a transfer axis, not as a promise.
5. **The negative control:** raw probe-transfer leakage through the same check. ⭐ **It must FAIL.** If your check
   passes the axis we already know is broken, the check is wrong — **stop and report that**, do not proceed to DCI-C.

## Acceptance
1. A **PASS / FAIL** verdict on DCI-C against the invariance criterion, with the numeric tolerance **pre-registered in
   `PREREG.md` before any measurement**.
2. Sign-agreement of the correlation-minus-random margin across configurations, per seed.
3. The class-count stress result.
4. The chance-corrected companion reported through the identical check.
5. ⭐ The negative control **fails**, demonstrating the check has teeth.
6. Suite green (before/after, main venv); ref verified from the main repo before worktree removal.

## What would prove this spoke wrong
- ⭐ **Your check passes everything.** Then it is not a check. Item 5 is how you find out before we do.
- You averaged over a sign inversion instead of reporting it.
- You tuned the tolerance after seeing DCI-C's numbers. ⛔ It is pre-registered.

---

# ⟲ AMENDMENT 2026-09-13 — you OOM'd a 16 GB machine twice. Resume, do not restart.

## Where you are
⭐ **5 of 6 cells are already written** and are valid: `inv_wide_seed_{0,1,2}`, `inv_montero_seed_{0,1}`.
**Only `inv_montero_seed_2` is missing.** ⛔ **Do not recompute the five that exist** — re-running them costs hours and
risks a third OOM. Resume onto the missing cell only, then aggregate all six.

## Why it happened (Hub-measured, so you do not have to rediscover it)
| quantity | size |
|---|---|
| one full-scope feature cache on disk (`float16`, 480,000 × 2048) | **1.97 GB** |
| the same array upcast to `float32` | **3.93 GB** |
| physical RAM on this machine | **16 GB**, ~38 % free ≈ 6 GB |
The check holds the **same features under several partitions**; two upcast copies plus JAX device arrays exceeds
what is free. ⛔ The problem is the **upcast and the copies**, not the number of partitions.

## The fix, in order of preference
1. ⭐ **Memmap and never materialise the whole array.** `np.load(path, mmap_mode='r')`, then compute in **row
   batches**, upcasting **per batch** only. Peak memory becomes the batch, not 3.93 GB. ⛔ This does **not** change
   the measurement, so no pre-registration is affected. **Try this first.**
2. **A declared stratified subsample**, if (1) is still tight. DCI over 480,000 rows is far past convergence.
   ⛔ This **does** change the measurement, so: declare the row count, and **show convergence** (DCI-C at ~25k vs
   ~50k vs ~100k rows on one cell that already succeeded) before using it for the verdict.
3. ⛔ **NOT acceptable:** dropping to 2 seeds on Montero. The criterion is **sign-agreement across configurations**,
   and the third point is exactly where a sign inversion would show. If you truly cannot get it, report
   `montero seed 2` as **NOT RUN** and say the verdict is weakened — ⛔ never a silent 2-seed number.

## Resumability is now a standing requirement
⭐ Write each cell to its own JSON **as it completes** (you already do — that is why only one cell was lost) and make
the entry point **skip any cell whose JSON already exists**, so a kill at any point costs one cell.
⚠ A detached spoke can keep running and still fail its **last** write, so ⛔ never hold results to the end.

## ⛔ Unchanged
The pre-registered tolerance, the negative control (raw probe-transfer leakage **must FAIL** the check), the
class-count stress, and the chance-corrected companion all stand exactly as filed. ⛔ Do not retune the tolerance
because the run was hard.
