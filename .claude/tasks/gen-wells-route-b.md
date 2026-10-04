# Task: gen-wells-route-b — ⭐ THE FIRST CSF3 LONG JOB. Learn where the wells GO, with the store in the loop.

- **Agent:** `experiment-engineer` · **Output:** `.claude/outputs/gen-wells-route-b.md`
- **Read first:** `.claude/AGENT_PROTOCOL.md` · ⭐ `.claude/outputs/gen-dynamical-read/PREREG.md` §10 (the metric) and §12 (cost + cards) · `scripts/csf3/README.md`
- ⛔ No campaign ledger — **EXCEPT `.claude/handover_context.md` §7 Known Issues.**
- **Base:** `main @ 3eb9eb0` or the Hub's tip · **Branch:** `agent/experiment-engineer/gen-wells-route-b` · **Worktree:** `../CHLU-routeb`
- **Env:** main venv. pytest with `CHLU_GEN_DATA_ROOT=/Users/user/Desktop/CHLU/.claude/data/gen`; report **passed / skipped / failed**.
- ⚠ **`chlu/config.py`: ONE new block, APPENDED AT EOF** between `BEGIN`/`END` markers, **no `CHLUConfig` field**.
  ⛔ On conflict take the peer's **whole file** and re-apply your block by its markers, asserting **0 deletions**, then
  verify **behaviour** — instantiate every block and round-trip save/load. ⚠ Some blocks are fields, others are reached
  by accessors; a presence check that assumes fields reports a false loss. **It has, twice.**
- ⚠ **`dec_recipe` defaults to `"baseline"`** — the recipe that FAILED the gate. Winner **`v1_capacity_schedule`**,
  selected explicitly with the **hash asserted**. ⚠ **JAX `split` is prefix-stable** — derive arm keys with `fold_in`.
- ⚠ **`--quick` CLOBBERS explicit CLI flags** in the `exp_gen_*` mains. ⛔ No full suite — 16 GB box.
- ⚠ **Budget ≈ 2 days.** ⛔ **No sub-agents.** If it does not fit, say so and stop — it becomes two spokes.

## ⛔ MECHANICAL PRECONDITION — check this FIRST and STOP if it fails
⛔ **`.claude/outputs/gen-cluster-stage-artefacts/STAGED.json` must exist and contain `"status": "complete"` and
`"checksum_match": true`.** ⛔ **If it does not, stop and report `NOT RUN — precondition absent`.** ⛔ Do not
improvise a staging path; a peer spoke owns that file set and two staging scripts will diverge silently.

## ⛔ ONE INHERITED ITEM YOU MUST CLOSE — it fails AFTER the GPU is allocated
⚠ **Neither `scripts/csf3/env.sh` nor `job_gpu_gen.sh` knows about the artefact root.** Today every submit must
carry five `CHLU_*` variables by hand in `--export`, and ⛔ **an omission fails LATE — inside the loader, after the
scheduler has already given you the GPU.** On a queue that costs a resubmit and a wait, and it will happen.
⭐ **Add the export block to `env.sh`** so a job inherits the root instead of being handed it:
`export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CLU_REPO/artefacts}"` plus the four derived paths
(`CHLU_GEN_DATA_ROOT`, `CHLU_GEN_BRIDGE_ROOT`, `CHLU_GEN_STORE_RUNS`, `CHLU_GEN_DECLIFT_DIR`).
⭐ **Every one keeps the `${VAR:-default}` form**, so an explicit `--export` still wins. ⛔ Do not hardcode a path.
⭐ **Assert the roots exist at job start and die loudly with the missing name** — ⛔ a job must never reach the GPU
only to discover it cannot open an artefact.

## ⛔ THE SEED CACHES ARE NOT SYMMETRIC — assert coverage BEFORE the GPU does work
⚠ **Measured on the staged set:** seed 0's feature cache holds **480,000 rows**; seeds 1 and 2 hold **9,288 each**
(seed 0 was built at `scope="full"`, the others at `scope="used"`). ⭐ That one seed is **1.83 GiB of the 1.95 GiB**
staged. `dtype` is **float16**, `D = 2048`, and ⚠ **`encoder_param_hash` DIFFERS per seed** — they are not
interchangeable.
⇒ ⛔ **A run on seed 1 or 2 that requests rows outside those 9,288 falls into the gather-with-extension re-encode
path, which `gen-dynamical-read` flagged as UNTESTED** — and it would discover this **on the GPU, after the
scheduler has already allocated it.**
⭐ **BINDING: assert at job start that the cache covers the rows this run needs.** Name the seed and the shortfall
and **die immediately.** ⛔ Never let a job reach the GPU to find out it cannot open what it needs.
⭐ Same rule as the artefact root: **every precondition fails at startup or not at all.**
⚠ If the re-encode path IS taken deliberately, say so, and report it as an **exercised untested path**, ⛔ not as a
normal run.

## ⭐ DIAL DECLARATION
- **Dial:** ⭐ **WHERE THE WELLS GO** — well placement learned with the store in the loop, replacing a partition
  inherited from `k`-means. ⛔ Not the read, not the decoder, not the encoder.
- **Laundering control:** ⭐⭐ **D-MATCHED RANDOM PLACEMENT at matched bytes.** Binding, see below.
- **Falsifies:** the claim that our wells are semantic. ⚠ It may show placement was never the lever.

## Why this and not another read
The read resolves **inside** the wells it is handed. ⭐ **Route A takes the wells as given.** This task asks the prior
question: **are the wells in the right place at all** — and if they are not, every read downstream is resolving
inside the wrong box. **The codebook is already the convicted culprit:** the partition costs 9.5 % of `scale` and
**0 %** of `orientation`, while **k-means takes 90.5 % / 100 %.**

## The one job
**Learn the well placement with the store in the loop, under a between-well factorisation + within-well multimodality
objective, and deliver a SUBMIT-READY CSF3 long job.**

## ⛔ Arms, declared here so neither can be swapped in later
1. ⭐ **LABEL-FREE — THE PRIMARY.** No factor labels in the objective, at any point, by any route.
2. **WEAKLY-SUPERVISED — a DECLARED arm.** ⛔ It is reported as a separate row and ⛔ **may never be quoted as the
   label-free result.** If the label-free arm fails and this one works, **that is the finding**, stated in those words.

## ⛔⛔ THE CONTROL THAT CAN KILL THIS — build it BEFORE the objective
⭐ **D-matched RANDOM well placement, matched bytes, same everything else.**
**Every placement number is a MARGIN OVER THE RANDOM-PLACEMENT FLOOR, ⛔ never an absolute.**
⚠ This programme has already been burned here: an **untrained random-init encoder** scored **0.9361 / 0.9865** against
a trained encoder's 0.935 / 0.991 for **1,246× less compute** ⇒ ⛔ **"≥ 0.9 on seen data" measures nothing.**
⛔ **A sharpness number without its random-placement floor beside it is void.**

## The metric — "semantic" means ADDRESS-BOOK SHARPNESS PER FACTOR
Per factor `f`, per well `k`: **normalised negentropy `S_fk = 1 − H(p)/log(n_values_realised(f))`**. `0` uniform, `1` a delta.
⛔⛔ **`n_values_realised` on EVERY row.** Montero's corner pins `shape` to **1 of 4** values and yields a
confident-looking zero — **that has produced a wrong headline in this programme once already.**
**Starting point, measured:** the address book is **diffuse but informative** — mean posterior confidence **0.0508**
at N=64 (**3.3×** chance), **0.0125** at N=512 (**6.4×** chance).

## ⛔ Byte discipline
Learned placement must cost what inherited placement costs. ⭐ **Ship the byte ledger with every configuration and
abort the run on a mismatch** — the existing coupling identity `Σ r(d_k+d_l)·4 = r(K−1)D·4` holds for any partition
of `D`, so ⛔ **a per-seed drift in well sizes may not silently renegotiate the budget.**

## Cluster discipline
⛔⛔ **CSF3 needs 2FA. You cannot reach it.** Emit submit lines; ⭐ **report every cluster item `NOT RUN`, naming the
leg.** `GEN_GIT_COMMIT` is exported **on the login node**. ⛔ Never `-p serial`.
⛔ `git pull` on the cluster serves a stale fork — **`push_repo.sh` is the only path.**
⭐ Partition `-p gpuA`: A100 SXM4 **80 GB**, free at point of use, **batch max `-t 4-0`**, `-G N` required.
⚠ **Login nodes have NO off-campus internet** — anything that resolves or downloads runs **inside a job**.
⚠ `~/scratch` has **no backup and a 3-month auto-cleanup**. `$TMPDIR` is node-local NVMe and is **wiped at job end** —
⛔ **never checkpoint to `$TMPDIR`.**

## ⛔⛔ CHECKPOINTING — STANDING HEAD REQUIREMENT, and an ACCEPTANCE ITEM
⭐⭐ **A cluster run that dies at hour 70 of 96 must resume, ⛔ never restart from scratch.** OOM, a node failure, a
wallclock kill and a lost final write are all the same event to this requirement.
⚠ **MEASURED GAP: `job_gpu_single.sh`, `job_gpu_array_seeds.sh` and `job_gpu_eval.sh` contain ZERO checkpoint or
resume logic** (0 matches for `resume|checkpoint|ckpt|requeue|restart`). ⛔ **Do not inherit that.**
⭐ **REUSE THE PROVEN PATTERN — ⛔ do not invent a second one.** `gen-wells-route-b` follows
`.claude/outputs/pilot-checkpoint-resume.md`: per-unit checkpoint + `--resume`, already verified **BITWISE with zero
differing leaves** on both the straight path and the resume path. See `chlu/experiments/exp_cluformer_pilot.py` and
`tests/test_pilot_checkpoint_resume.py`.
**Binding requirements:**
1. **Time-based checkpoints, not only epoch-based** — an OOM does not wait for an epoch boundary.
2. ⭐ **Atomic writes: temp file + rename.** ⛔ A crash *during* a checkpoint write must not corrupt the last good one.
3. **Keep the last N good checkpoints**, not just one.
4. **Resume verifies** the checkpoint's step and config hash and ⛔ **refuses a checkpoint from a different config.**
5. ⭐ **Checkpoint to `~/scratch`**, ⛔ never `$TMPDIR` (wiped at job end), ⛔ never `$HOME` (quota, backed up).
6. ⭐ **The resume path is instrumented** — it prints what it recovered and from which step.

## Acceptance
1. `PREREG.md` filed **before** any measurement: the objective, both arms, the random-placement floor, and **what
   result would refute semantic placement**.
2. The random-placement floor **built and measured first**, before the learned objective runs.
3. A short local or **L4** verification run showing the path end-to-end. ⛔ **No scored verdict** — this is not the
   long run, and ⛔ no number from it enters any table.
4. Per-factor sharpness with `n_values_realised` on every row, **as a margin over the floor**.
5. Byte ledger asserted in code; run aborts on mismatch.
6. ⭐ **One copy-able block of CSF3 submit lines for the Head**, with expected wall time and the resume command.
7. ⭐⭐ **RESUME DEMONSTRATED BY KILLING IT.** Start the verification run, **`kill -9` it mid-flight**, restart with
   `--resume`, and show it **picks up rather than restarting** — with a **bit-identity check against the
   uninterrupted run**, the same gate `pilot-checkpoint-resume` passed. ⛔ **A resume path that has never survived an
   actual kill is not a resume path.** ⛔ This acceptance item may not be waived for time.
8. Targeted tests green with the data root set, passed/skipped/failed reported.
9. A **cost line**: card, wall hours, dollars. ⚠ **A100-80GB and H200 DO NOT EXIST for this project** — L4 (6 free) or
   **A100-40GB spot** only, and another project holds live capacity in `us-central1`: ⛔ do not evict `tsfm-pair-a`.

## ⛔ FILES YOU OWN — and nothing else
- `chlu/core/gen_wells.py` **(new)** · `chlu/experiments/exp_gen_wells_route_b.py` **(new)**
- `tests/test_gen_wells.py` **(new)** · `scripts/csf3/job_gpu_wells.sh` **(new)**
- ⭐ **`chlu/config.py` — ONE appended EOF block. You are the ONLY spoke holding this file this wave.**
- ⭐ `chlu/cli/experiment_cmd.py` — marker-delimited region only, **0 deletions**
- ⭐ **`scripts/csf3/env.sh` — ONE added export block** (see the requirement below). You are its owner this wave.
- `.claude/outputs/gen-wells-route-b/**`
⛔ **You do NOT own `chlu/core/gen_landscape.py`** — the read is built, merged and frozen for this task. ⛔ You do not
own `chlu/data/gen_rowsplit.py` or `scripts/csf3/stage_artefacts.sh` (a peer spoke's).

## What would prove this spoke wrong
- ⭐⭐ **Learned placement does not beat D-matched RANDOM placement.** Then placement was never the lever, the
  `k`-means partition was not the problem, and ⛔ route B does not go to the cluster. **Report it in those words.**
- ⭐ **Sharpness rises by collapsing wells** — one well eats the mass and the address book looks sharp because there
  is nothing to choose between. **Report well occupancy beside every sharpness number.**
- ⭐ **The label-free objective only works with labels leaking in** through the partition, the bridge, or the atom
  selection. **Assert the label-free path is label-free, and demonstrate the assertion FIRING on a leaky variant.**
- A per-factor number without `n_values_realised`.
- ⭐ **Seeds 1 and 2 silently re-encoded** because nobody checked cache coverage, so the seeds are not comparable and
  nobody noticed until the numbers disagreed.
- ⭐ **The resume path was written but never killed.** Every untested resume path in this programme's history has
  been a resume path that did not work.
