# SPOKE D — What work can actually run on this box  **(READ-ONLY)**

Read `.temp_cl/CONTEXT.md` first.
Report to: **`.temp_cl/reports/D-compute-split.md`**

## Why you exist
The human runs this project on a **Mac** (primary, Zed + ACP + Claude agents) and wants to move **part of
the compute load** to this Windows box so the Mac stays free. Nobody has yet established what this box is
actually good for. Your job is to answer that with evidence from the code, not from vibes.

**The hardware, restated:** 16 cores / 16 logical, 63 GB RAM, 895 GB free, **Intel Arc Pro 140T — no CUDA,
JAX will be CPU-only**. For reference the Mac is Apple Silicon (JAX on Metal or CPU) and the project also
targets **CSF3** (a Slurm cluster with A100 nodes — see `scripts/csf3/`) and **Google Cloud**.

## ⛔ You change nothing, and you run no experiments
Read-only analysis. No installs. No training runs. No benchmarks that take more than a few seconds.
Spoke A is installing Python concurrently — **assume Python may not work yet**; do your analysis by
**reading the code**, not by executing it. If Python happens to become available late in your run you may
do a trivially cheap sanity check, but do not build your report on it.

## Map the surface
1. **The CLI.** `chlu/cli/` has `data_cmd.py`, `eval_cmd.py`, `experiment_cmd.py`, `project_cmd.py`,
   `train_cmd.py`, `utils_cmd.py`, plus `chlu/chlu.py` (entry point `chlu = "chlu.chlu:main"`). Enumerate
   every subcommand and what it does. This is the menu of work that could be split.
2. **The experiment surface.** Read `chlu/experiments/`, `scripts/c3_geometry_sweep.py`,
   `scripts/c3_ladder_plan.py`, `scripts/smoke_c3_*.sh`, and `scripts/cafe/`. What are the actual run
   shapes — sweeps? seed arrays? single long trains?
3. **The CSF3 job scripts** (`scripts/csf3/job_gpu_*.sh`, `job_cpu_smoke.sh`). These encode the real
   resource requirements. Extract from each: GPU vs CPU, requested walltime, memory, array size, and what
   it runs. `job_cpu_smoke.sh` is the strongest single signal for "what is CPU-viable".

## Then classify — this is the deliverable
Produce a table with a row per work class and these columns:
`work class | entry point | needs GPU? | est. cost (CPU-hrs / RAM) | runnable on this Windows box? | verdict`

Verdict must be one of: **RUN HERE** / **RUN ON MAC** / **CSF3/GCP ONLY** / **BLOCKED (state on what)**.

Candidate classes to assess (add any you find):
- unit/integration test suite (117 test files) and coverage runs
- lint/format/typecheck (`ruff`, `black`, `pylint` are in the dev group)
- docs build (`sphinx` is in the dev group; note `docs/**` is gitignored and absent)
- data preparation / dataset download and preprocessing
- small-scale smoke runs and ablations (`smoke_c3_*`)
- geometry/ladder sweeps
- evaluation and metrics over **existing** result artefacts (`chlu/eval/`, vendored TSB-AD)
- plotting / figure generation for the ICLR submission
- full training runs
- anything in `chlu/eval/cafe_model.py` / `scripts/cafe/` (note: `cafe_bench` is a **separate checkout
  kept outside this repo** and is therefore absent here)

## Call out explicitly
- **CPU-only blockers:** grep for anything that requires a CUDA device, `jax.devices("gpu")`, sharding
  across multiple devices, `pmap`, or GPU-only kernels. These are hard NOs on this box.
- **Absent-dependency blockers:** the `eval`/`cafe`/`gen` extras, `cafe_bench`, the `.claude/` bundle,
  datasets. Which work classes die on a missing dependency rather than on compute?
- **The parallelism question:** 16 cores with no GPU is genuinely good for *embarrassingly parallel*
  work — seed arrays, sweeps over small configs, `pytest -n auto`. Identify which run shapes exploit that
  and which are single-threaded-long, where this box is worse than the Mac.
- **The split hazard:** if work runs in two places, results must merge. How does this project record
  results (files? a store? `projects/`?) and what would collide if two machines wrote concurrently?
  This is the question most likely to bite the human, so give it real attention.

## Report shape
Lead with a **3–5 bullet recommendation** — the specific work the human should move to this machine
starting now, and the specific work they should not. Then the table, then the blockers, then numbered
open questions.
