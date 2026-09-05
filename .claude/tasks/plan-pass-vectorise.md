# plan-pass-vectorise — RE-POINTED (Hub, 2026-08-01): the lane-parallel controller

**⚠ SUPERSEDED-IN-PART at the C2W5 wave review.** The original target dissolved: the probe measured
that **98.4 % of the plan pass was the eager JAX forward, not the Python controller** (R1), and the
fix — `filter_jit` on the three stages — is **built, merged (`46755fb`), decision-identical, 21.7×
on the plan pass / 5.7× end-to-end.** N196's compute clause is amended accordingly.

**What remains (probe §8.1, priced ~half a day — the original gate condition, met at the boundary;
spawn is a HEAD funding decision):** the **lane-parallel controller**. After the jit, the residual
CPU-serial term is the Python controller: `n_layers × batch` sequential lane-calls, ~1.7–2.4 ms per
lane-chunk ⇒ **≈ 2.6–3.7 s/step of pure Python at pilot scale that no GPU can absorb** (GPU idle
39–79 % on an A100 bracketing; < 30 % after the fix).

**Campaign 2, wave C2W5. Agent:** experiment-engineer. **Small worktree.** Branch
`lane-parallel-controller`. Writes `.claude/outputs/plan-pass-vectorise.md`.

**Read first:** `.claude/outputs/pilot-placement-probe.md` **§8/§8.1 in full** (the measurement +
the priced design) · charter §A18.4 · the `[C2W5]` review §10 entry.

## The work
- `ProcessPoolExecutor` over the **batch axis** in `plan_pass` (lanes are independent — each
  `_controller_plan_for_lane` builds its own controller; **layers stay sequential**, layer l+1
  needs layer l's decisions).
- The two picklability blockers the probe named: the per-lane call returns a live
  `CluControllerV0` (for the monitors) and shares the monitor `registry` for guard counts — both
  become picklable summaries, merged post-join; monitor semantics unchanged (same counts, same
  trip states, asserted in a test).
- **Invariant (unchanged from the original file): decision-replay is the SPEC.** Blocking
  `WritePlan`-equivalence test vs the serial path — bit-identical discrete fields, ≥ 3 seeds — added
  to the suite beside the jit tripwire.
- Measure: per-step Python time before/after at toy and extrapolated pilot counts; the GPU-idle
  brackets re-computed; a one-line delta for the CSF3 job config.

## Rider (Head ruling 3, 2026-08-01 — CSF3 monitoring): fold the §7.27 watch-item into the job
Store destruction must be caught **in-flight, not post-mortem**: add per-seed logging of
**untrained-vs-trained well depth** and the **`q*` payload spread** (the probe's §10 row 10) to the
tier-iii training loop at `monitor_every` cadence, emitted into the run artifact so the CSF3 job
carries it by default. Cheap (the probe's depth-fit code exists); assert it fires in a test. The
CSF3 submission waits on THIS task landing (Head ruling: lane-parallel first, then submit; the run
itself stays committed regardless, A18.4).

## Acceptance
Equivalence + monitor-summary tests green · measured ≥ 4× cut on the Python term at batch 8 (or an
honest null with the profile) · the §7.27 in-flight watch logging landed + tested · report the CSF3
idle-fraction update and the final recommended `sbatch` line (probe §10 base + your delta).

**Ownership:** `chlu/training/train_cluformer.py::plan_pass` + `_controller_plan_for_lane` call
path + your tests. ⛔ Not `monitors.py`, not the factored-store files, not `chlu/core/controller*`.
**Git:** branch + scoped worktree; never push `origin`; `clu-dev` only. Report → Hub, spawn nothing.
