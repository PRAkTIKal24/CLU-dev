# csf3-memory-fit — make the pilot's backward fit an 80 GB A100 (run 1 is DOWN on OOM)

**Campaign 2, wave C2W5 (ops-critical; run 1 crashed at 97.8 GiB vs 80 GB). Agent:**
experiment-engineer. **Worktree 3 of ≤3.** Branch `csf3-memory-fit`. Writes
`.claude/outputs/csf3-memory-fit.md`.

**The failure (verbatim evidence in the Head's log):** XLA wants **97.82 GiB** and auto-remat
cannot get below **76.70 GiB**; OOM at a 3 GiB allocation; the crash site is
`allocation_liveness`'s `filter_value_and_grad(loss_fn)` (`train_cluformer.py:1042` → `:750`) — ⚠
**but the train step differentiates the SAME `loss_fn`, so this is the model's backward memory,
not an instrument quirk.** There is currently NO `jax.checkpoint`/remat anywhere in `blocks.py` or
`train_cluformer.py` (grep-verified): the backward through `lax.scan(step, …)` (`blocks.py:1365`)
keeps every within-chunk intermediate — including the 64+64-step Verlet read over 8 192 atoms —
alive for all 16 chunks × 12 layers × batch 8.

**Read first:** `chlu/core/blocks.py` (`CluStoreBlock.__call__` :1365, `CluStoreCell`, the read
path); `chlu/training/train_cluformer.py` (`loss_fn` :750, `allocation_liveness` :1042, the jitted
stages from `46755fb`, the lane-parallel pool from `b81f487` — do not disturb either);
`exp_cluformer_pilot.py` (TOY/PILOT dicts; the `--set/--mem/--store` override path); the `[C2W5]`
§10 babysitting block (§A20.4 discipline: run-2 must stay same-config-otherwise — your fix must
NOT change the registered config's semantics).

## ⭐ DIAL DECLARATION (protocol §7, C2 form) — echo before your first result
- **Dial:** none — infrastructure/memory. The only claims: (a) it fits, (b) **nothing else
  changed**.
- **Control:** bit-identity at toy — held-out bpc and all 7 `WritePlan` fields identical with the
  remat flags on vs off (the jit tripwire pattern; ≤ float-ULP tolerance declared if XLA
  reassociates).
- **Falsifies:** any decision or bpc change beyond declared tolerance; or a projected pilot-step
  memory still > 72 GB (headroom margin) with all levers on.
- **Does NOT falsify:** compute overhead (remat trades FLOPs for memory — measure and report the
  slowdown; the 12 h wallclock check is part of acceptance).

## The work, in order
1. **Checkpoint the chunk scan** (`blocks.py:1365`): `jax.checkpoint`/`eqx.filter_checkpoint` on
   the scanned `step` — backward then saves only chunk-boundary carries and recomputes the chunk
   interior (incl. the Verlet unroll): activations ÷ ~n_chunks (16) at ≤ 2× chunk-interior compute.
2. **If needed, nest one level:** checkpoint the Verlet integrator's own scan (the read is the
   memory monster: 128 steps × atoms × lanes) and/or per-layer remat. Ladder until the projected
   pilot step fits ≤ 72 GB.
3. **Microbatch/grad-accumulation flag** (`accum_steps`, default 1): splits the batch axis in the
   backward and sums grads — the registered EFFECTIVE batch 8 is preserved exactly (state the
   summation-order caveat); this is the fallback lever if remat alone is short.
4. **Shrink the instruments:** `allocation_liveness` (and any other at-init full-batch grad probe)
   runs on a 1-lane microbatch by default at scale — it is a liveness anchor, not a paper number;
   flag `liveness_lanes=1`.
5. **Memory evidence without a GPU:** you cannot run an A100 from here (pre-registered). Provide
   (a) the activation-memory arithmetic per lever (show the ledger: before ≈ 98 GiB, after each
   lever), and (b) where possible `jax.jit(...).lower(...).compile().memory_analysis()` on CPU at
   reduced dims with the scaling law stated. Honest bracket, clearly labelled ESTIMATE.
6. **Wallclock check:** remat ≤ 2× on the chunk interior + the measured lane-parallel step ⇒
   project steps/s and verify 4 000 steps × 3 seeds fits `-t 12:00:00` per seed; if not, say so
   and stop (the job header's cut-order governs — never the seed count, the swap, dyn-eval, or a
   monitor).
7. **The resubmission line:** run-1's ruled flag line + only your new flags (e.g.
   `SET="… remat_chunks=1 accum_steps=2 liveness_lanes=1"`), zero module edits on the cluster.
   All new flags default to the OLD behaviour (off) so toy history is untouched; the SCALE
   submission turns them on explicitly — recorded verbatim in the artifact's flags block.

## Acceptance
Toy bit-identity green (bpc + 7 plan fields, remat on/off) · the memory ledger with the projected
pilot fit ≤ 72 GB · slowdown measured at toy and projected at pilot vs the 12 h budget · the
resubmission line delivered · tests green (new: the identity test with remat on) · declared
NOT-RUNs (no GPU measurement) never nulls. **Ownership:** `blocks.py` (the scan/read remat hunks) ·
`train_cluformer.py` (accum + instrument flags; do not touch the pool or jit structure beyond
wrapping) · `exp_cluformer_pilot.py` flags · your tests. ⛔ Not `psi_readout.py`
(`psi-payload-residual`'s), not the factored-store/read-fix files. **Git:** branch + scoped
worktree; never push `origin`; `clu-dev` only. Report → Hub, spawn nothing.
