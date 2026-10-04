# pilot-checkpoint-resume — per-arm checkpointing + eval-block host-memory hygiene

**Campaign 2, wave C2W5 (Head-directed, 2026-08-03). Agent:** experiment-engineer. **Medium.**
Branch `agent/experiment-engineer/pilot-checkpoint-resume` off `main @ 8efc1d8`. No worktree needed
unless another spoke is live. Output: `.claude/outputs/pilot-checkpoint-resume.md`.

## Context (read the 2026-08-02/03 §10 blocks first)

CSF3 attempt 1: all legs trained clu_store 4000 steps (~22 h) then job 18136619 was **host-RAM
oom_killed in the post-training eval block** (kernel oom_kill, NOT device OOM — remat is fine,
device peak ≈ 8.3 GiB). Because `run_pilot` writes its JSON **only at `_finish`**, the crash cost
the entire run. The other legs are expected to die identically. The Head has ruled: **we need
checkpointing.** The Hub will supply the measured `ReqMem`/`MaxRSS` of the dead job when known —
ask if not in the task addendum below.

## Deliverables

1. **Per-arm checkpoint + resume in `run_pilot`:**
   - After EACH arm completes (train + its evals), write (a) the partial record as
     `pilot_{scale}_seed{N}_PARTIAL.json` (atomic: write tmp + rename), and (b) the trained arm's
     weights (`eqx.tree_serialise_leaves`) as `ckpt_{arm}_seed{N}.eqx` in `OUT`.
   - A `--resume` flag (and `RESUME=1` passthrough in `job_gpu_cluformer.sh`): on start, load the
     partial record + any per-arm checkpoints and **skip completed arms entirely** (their rows come
     from the partial record; their models deserialise for the swap table / cross-arm passes).
     Data iterators must fast-forward deterministically so a resumed arm sees the same batches it
     would have (document the guarantee; if exact stream alignment for LATER arms depends on
     nothing carried from earlier arms' training — verify and state it).
   - ⛔ The FINAL JSON must be byte-identical in content-shape to today's (downstream `--aggregate`
     and the analyst read it); the partial/ckpt files are additive artifacts.
2. **Eval-block host-memory hygiene** (the crash site): between the post-training eval phases
   (static → dyneval → blank → anytime curve (5 compiles) → monitors → gradient probe), release
   what can be released — `jax.clear_caches()` between phases where re-use is not needed, drop
   references to donated buffers, and (engineer's judgment) any cheap restructuring that lowers
   peak host RSS. Target: the eval block must fit inside the same allocation that survived 22 h of
   training. Instrument it: log `ru_maxrss` (or `/proc/self/status` VmHWM) at each phase boundary
   so the next crash — if any — is attributable to a named phase.
3. **Timing print (cheap rider, fixes a live erratum):** `train_arm`'s log loop gains the
   `wall_s`/`plan_s` per-25-steps print that `csf3-memory-fit` §7 claimed exists but does not.
4. **Toy bit-identity gate (the csf3-memory-fit pattern, MANDATORY):** two toy end-to-end runs —
   (a) old path vs new path straight through: held-out static/dyneval/blank bpc and all decision
   fields **bitwise identical**; (b) a run killed after arm 1 + `--resume` vs an uninterrupted run:
   the resumed run's REMAINING arms' numbers bitwise-match the uninterrupted run's (or, if stream
   alignment makes that impossible, say so, quantify the drift, and gate on decision-identity).
   Checkpointing is IO-only; if you find yourself changing any math, stop and flag the Hub.
5. Tests + suite green (baseline **1348**), ruff green, report with flag provenance.

## §A20.4 note (carry verbatim)
All six CSF3 legs will rerun on THIS code uniformly; the change set must be decision-inert and the
toy gate is what licenses that claim. The MEM/STORE/SET flag strings do not change.

## Task addendum (Hub, 2026-08-03 — partition facts from the Head, official CSF3 table)
- **gpuA host RAM is 10 GB/core, max 12 cores/GPU ⇒ a 1-GPU job's HARD CEILING is `-c 12` = 120 GB.**
  Attempt 1 (no `--mem`) almost certainly already had the proportional 120 G ⇒ the eval block spiked
  past ~120 GB host RSS. **More memory is NOT available at `-G 1` — footprint reduction is the
  primary fix, not a fallback.**
- ⭐ **Hard budget target: peak host RSS < 100 GB end-to-end** (120 G ceiling minus headroom for the
  8-worker plan pool's variance). The per-phase RSS instrumentation (deliverable 2) is how you prove
  it — report the measured per-phase peaks at toy AND your best projection at pilot geometry.
- Levers, in preference order: (i) cache/buffer hygiene between eval phases (`jax.clear_caches()`,
  dropping executables for one-shot programs like the 5 anytime-curve budgets and the blank-store
  eval); (ii) if needed, `plan_workers` 8→4 as a UNIFORM SET change across all legs (lane-parallel
  is measured decision-identical, so this is licensed — but declare it, and the Hub re-states the
  flag block); (iii) LAST resort, flag to the Hub: `-G 2 -c 24` doubles host RAM to 240 G at the
  cost of an idle GPU and halved concurrency — a submission-side change, not yours to make.
- Measured (sacct, job 18136619): **ReqMem 128760M ≈ 125.7 GB** (the proportional 12-core gpuA
  allocation, as predicted) · **MaxRSS 131682856K ≈ 125.6 GB — the process peaked AT the ceiling
  and was killed there** · Elapsed 23:04:33 (≈ 22.2 h clu_store training + ~45 min into the eval
  block). ⚠ **MaxRSS is truncated by the kill: true eval-block demand is ≥ 125.6 GB, upper bound
  UNKNOWN.** Two consequences for you: (a) training itself ran ~23 h under this ceiling, so the
  steady-state footprint fits — the excess is genuinely the eval block's; (b) since the overshoot
  could be marginal or large, do NOT assume cache hygiene alone suffices — instrument first, then
  cut until the projection clears the < 100 GB budget with the measured phases as evidence.
