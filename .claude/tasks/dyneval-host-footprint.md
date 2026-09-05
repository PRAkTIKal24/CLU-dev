# dyneval-host-footprint — experiment-engineer

⚠ **GATED: spawn only on Hub/Head GO** — first the `-G 2 -c 24` RESUME resubmission measures
dyneval's true host demand (attempt-3 post-mortem, §10 2026-08-05). If that rerun fits under
~251 GB and lands artifacts, this task retires to a post-wave rider. If it OOMs again — or the
Head prefers the durable fix now — this is the task.

## Context (read first)
- §10 `[C2W5]` 2026-08-05 attempt-3 post-mortem + the banked logs in
  `.claude/outputs/cluformer-pilot/csf3_logs/` (jobs 18200591–594; `depr/` = older attempts).
- The `[rss]` series (now working, hotfix `104ca19`) attributes the host-RAM kill to **inside
  `clu_store/dyneval`**: enter at rss 15.9 GB + children 2.9 GB (peak 22.5), killed at the
  125.7 GB cgroup ceiling ⇒ **`dynamic_eval` alone demands ≥ 107 GB host RAM at pilot scale**,
  with Δpeak ≈ 0 at toy. The eval-block cache hygiene *works* (releases visible in the series);
  nothing stacks — this is one phase's own spike.
- `chlu/training/train_cluformer.py::dynamic_eval` (~l.1104): per batch — `plan_pass` +
  **un-jitted `eqx.filter_value_and_grad(loss_fn)`** (the known heavy pattern,
  `pilot-checkpoint-resume` §4) — × `dyneval_batches=40` × a 3-LR grid = **120 eager backwards**,
  with `opt.update`/`apply_updates` re-materializing the model pytree every iteration.

## Deliverable
1. **Attribute the spike** — on a local synthetic scaling probe (CPU is fine): grow
   `seq_len`/`d_model`/`dyneval_batches` toward pilot shape and record host RSS per dyneval
   iteration. Distinguish the two candidate mechanisms (they have different fixes):
   (a) **per-iteration growth** (a leak: op-cache/trace accumulation across the 120 eager
   backwards) vs (b) **a single compile/trace spike** on the first backward at pilot shape.
   The XLA constant-folding alarms in `clu-tier3-18200595.err` (`_block_chunk_latents` /
   LayerNorm) are a possibly-related breadcrumb — check whether remat chunking at pilot shape
   inflates the traced graph.
2. **Fix by mechanism**, cheapest decision-inert lever first:
   - (a) → in-loop hygiene (periodic `jax.clear_caches()`/`gc.collect()` inside `dynamic_eval`,
     value-inert by construction — same argument as `release_host_memory`), or explicit
     donation/deletion of the dead model pytrees per LR-grid leg.
   - (b) → ⛔ **STOP AND REPORT before jitting anything.** `filter_jit` on the dyneval backward
     is NOT bitwise (float re-association) and dyneval bpc is a MANDATORY published column —
     that lever needs a Hub ruling first (same class as `probe_lanes`). Report the measured
     spike + the projected jitted footprint and wait.
3. **Gates (both mandatory, csf3-memory-fit pattern):**
   - toy bit-identity: old-vs-new `run_pilot` at toy S4, **0 differing leaves** in the final
     JSON (if your fix cannot meet this, see 2(b) — stop and report).
   - a regression test that exercises the dyneval loop ≥ 2 LR-legs × ≥ 3 batches and asserts
     host RSS growth per iteration below a stated bound (marked `slow` if needed).
4. ⛔ **Resume compatibility is a hard constraint:** the four dead legs hold same-config
   journals + `ckpt_{arm}.eqx` on the cluster worth ~16 h training each. Your change must NOT
   add/rename any config key that `load_journal` compares (it refuses foreign-config journals
   by key). If a new knob is unavoidable, it must live outside the journal's config fingerprint
   — demonstrate resume-accept in a test.
5. Suite green (expect 1363 + yours), ruff green, scoped branch, report to
   `.claude/outputs/dyneval-host-footprint.md` with proposed §7/§10 updates.

## Acceptance
Attribution table (mechanism named, numbers from the scaling probe) + a fix whose projected
pilot dyneval footprint fits inside 125.7 GB with stated headroom + both gates green + resume
compatibility demonstrated. If the honest answer is "only a non-bitwise lever fits", a
stop-and-report with the numbers IS acceptance — do not spend the bitwise property without the
ruling.
