# matched-bytes-frontier — the contested-win experiment

**Agent:** experiment-engineer. **Worktree MANDATORY** (§3.2). Base local `main` (post-w25).
**This is the wave's shot at a PRIMARY claim** (addendum-2 §B2 Candidate 1, §B3.3; both advisors,
Advisor-2 insists).

## ⭐ Why this task exists (read the standing ruling first)
**Head standing ruling, binding program-wide:** *win-by-construction results — where the
incumbents fail the class by construction — are **SUPPLEMENTARY** claims only. The primary claim
requires a genuine win on a task where the baselines also do fairly well.* Split-MNIST's
rehearsal-free win is supplementary for exactly that reason.

**Replay is the strongest anti-forgetting method that exists. It does not fail by construction.**
And w25's own numbers say we are competitive with it on the *forgetting* axis:
CLU BWT **−0.169** vs parametric rehearsal-free **−0.99**; forgetting **0.169 vs ER's 0.264 at
the same item budget with 24.5× fewer floats** (store 6 400 vs raw buffer 156 800).
⭐ **Accuracy at matched memory is laundered; forgetting at matched BYTES is not.** Dominating a
region of that frontier is a contested win on the axis the whole CL field optimises, against its
best methods.

## ⭐ DIAL DECLARATION (protocol §7)
- **Dial:** admission + isolation (the anti-forgetting mechanism), measured on the forgetting axis.
- **Laundering control:** ⚠ **the hard one, and it must be run at matched BYTES, not matched
  items.** kNN-in-φ over a class-balanced ring buffer **given the same byte budget** — i.e. it
  gets ~24.5× more φ keys too. Anything less is a rigged control. Run both forms (same keys +
  ring buffer), at **every budget point**. It may fire again; pre-register that possibility.
- **Falsifies:** CLU fails to dominate **any** region of the forgetting-vs-bytes frontier against
  tuned ER / DER++ / GDumb / iCaRL; or the matched-bytes launder dominates CLU everywhere.
- **Does NOT falsify:** losing on **ACC** at every budget point (the axis here is forgetting —
  report ACC alongside, always, but it is not the claim); losing at very large budgets where
  replay saturates and the store is geometry-bound; sitting below joint/offline.

## What to build

1. **Pin the byte accounting explicitly, before any run.** Floats per stored item for: the CLU
   store (addresses + payloads + amps + active, and **count every array you actually keep**), a
   raw exemplar buffer, a φ-key ring buffer, and any method-specific state (DER++ keeps logits —
   count them). ⛔ **No capacity may be smuggled via uncounted parameters.** Publish the table in
   the report; a referee will check this line first.
2. **Add DER++** to `cl_baselines.py` (tuned per N78: sweep its hyperparameters on seed 0, then
   fix for all seeds). It is the strongest cheap replay baseline and its absence would be noticed.
3. **The budget sweep.** Sweep the **byte** budget across ≥5 points spanning ~1× to ~50× the
   current operating point, and at each point give **every** method the same bytes (so replay
   methods get N items, CLU gets ~24.5N, the φ-ring-buffer launder gets ~24.5N keys). Report
   **forgetting and BWT as functions of bytes**, with ACC carried alongside. ≥3 seeds.
4. ⭐ **Report where the store saturates.** CLU cannot use unlimited items — it is capacity-bound
   by the packing geometry (w25: budget-bound at 200, admitted fraction 0.46–0.97 per task). The
   frontier's honest shape is *"CLU dominates from X to Y bytes and then flattens because the
   address space is full."* **That saturation point is a result, not an embarrassment** — it is
   the capacity law showing up inside a benchmark. Report the admitted fraction and
   `refused_full` at every budget point (N91: per-offered and per-admitted always travel
   together).
5. **Carried item, folded in here (same file, same owner): retune LwF.** w25 reproduces EWC/SI/
   finetune to ≤0.5 pp of published Split-MNIST Class-IL values but LwF at 19.6 vs published
   23.9 (4.3 pp low). It must be retuned before this table is ever published, and on CIFAR it is
   the baseline the entry loses to.

## Scope
**Split-MNIST now** — do not wait for `cl-encoder`. If that task clears its gate this wave, the
frontier extends to Split-CIFAR-10 in w27; note the extension point in your report but do not
build for it speculatively.

## File-ownership split (⚠ parallel-safety)
**You own:** `chlu/experiments/cl_baselines.py` and the byte-accounting / budget-sweep surface of
`chlu/experiments/exp_cl_entry.py`. **You must NOT edit:** `exp_phi_stream.py` or the φ config
surface (that is `cl-encoder`'s), nor `chlu/core/*` (that is `placement-landing`'s and
`r2-excursion-reach`'s). Keep the config additions to the existing `ExperimentClEntryConfig`
group where possible; if you add a group, register it at **all four sites incl. `save_config`**
(the w23 trap — `tests/test_config.py::test_every_group_round_trips_mutated` guards it).

## Deliverable
PREREG **first** (`.claude/outputs/matched-bytes-frontier/PREREG.md`): the byte-accounting table,
the metric definition, the budget grid, your registered prediction for **where** (if anywhere)
CLU dominates, and the registered outcome-readings for each of "dominates a region" / "dominated
everywhere" / "launder fires at matched bytes". Report at
`.claude/outputs/matched-bytes-frontier.md`, standard format, reconciliation list in the first
10 lines, PREREG scorecard. Full `pytest tests/` green, `ruff` clean, atomic commits on
`agent/experiment-engineer/matched-bytes-frontier`. Do not push.
