# SPOKE D2 — What work should move to this machine  **(read-only, needs no Python — start immediately)**

Read `.temp_cl/CONTEXT.md` first.
Work in: `/c/Users/PratikJawahar/chlu/chlu-hub-bundle`
Report to: **`C:\Users\PratikJawahar\OneDrive - Forgis\Desktop\CLU-dev\.temp_cl\reports\D2-compute-split.md`**

## Why you exist
The human runs this project on a **Mac** (primary, Zed + ACP + Claude agents) and wants to move **part of
the compute load** onto this Windows box so the Mac stays free. The bundle exists to let an autonomous
"hub" agent run here for ~10 hours on one research question. Nobody has yet established **which** work is
a good fit. Answer that from the code and the charters, not from vibes.

**Hardware, restated:** 16 cores / 16 logical, 63 GB RAM, 895 GB free, **Intel Arc Pro 140T — no CUDA,
JAX will be CPU-only**. The Mac is Apple Silicon. The project also targets **CSF3** (Slurm, A100 nodes —
see `scripts/csf3/`) and **Google Cloud** (`scripts/cloud/`, and the bundle came from
`gs://chlu-cloud-forgisprova/`).

## You change nothing, and you run no experiments
Pure read-and-reason. No installs, no training, no benchmarks. **Assume Python does not work** — Spoke A2
is still installing it. Build your report by *reading*, not executing. If Python happens to appear late,
you may do a trivially cheap sanity check, but do not found your conclusions on it.

## Read these first — they are the actual source of truth
The bundle ships the charters that define what a hub is allowed to do. Read them properly:
- `.claude/autonomous-hub-charter.md` — the AUTONOMOUS HUB CHARTER. Note it carries the pin
  `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`, a section on **MATCHED-BYTES LOOKUP**, and a stated **core
  philosophy**. Summarise what a hub is chartered to do and what it is forbidden from doing.
- `.claude/agents/autonomous-hub.md` — the agent profile. It has a **SECTION 0**, a **YOU NEVER DELETE**
  rule, and a **PRE-REGISTRATION** requirement. §0.2 says the hub's sandbox is a git worktree; §0.3 says
  there is exactly one path a hub may write (`CHLU_HUB_OUTPUTS`).
- `.claude/AGENT_PROTOCOL.md` — the CHLU Spoke-Agent Protocol, including **DIAL DECLARATION**.
- `.claude/handover_context.md` — especially its **Known Issues** section.
- `.claude/tasks/hub-composition.md`, `hub-deletion.md`, `hub-refusal.md` — **these are the three
  candidate research questions already written up.** They are the concrete work on offer. Read all three
  and summarise each in two sentences.
- `README-HUB-BUNDLE.md`.

## Map the surface
1. **The CLI.** `chlu/cli/` (`data_cmd`, `eval_cmd`, `experiment_cmd`, `project_cmd`, `train_cmd`,
   `utils_cmd`) plus entry point `chlu = "chlu.chlu:main"`. Enumerate the subcommands.
2. **The experiment surface.** `chlu/experiments/` (~80 modules), `scripts/`, `scripts/cafe/`,
   `scripts/cloud/`. What are the real run shapes — sweeps, seed arrays, single long trains?
3. **The CSF3 job scripts** (`scripts/csf3/job_gpu_*.sh`, `job_cpu_smoke.sh`) encode actual resource
   requirements. Extract per script: GPU vs CPU, walltime, memory, array size, what it runs.
   `job_cpu_smoke.sh` is the strongest single signal for "what is CPU-viable".
4. **`scripts/cloud/`** — how does this project dispatch to Google Cloud, and could this box act purely as
   an orchestrator that submits and collects rather than computing locally?

## Then classify — this is the deliverable
A table, one row per work class:

`work class | entry point | needs GPU? | est. cost (CPU-hrs / RAM) | runnable here? | verdict`

Verdict is one of: **RUN HERE** / **RUN ON MAC** / **CSF3 or GCP ONLY** / **BLOCKED (on what)**.

Classes to assess (add any you find):
- the three chartered hub tasks (composition / deletion / refusal) — **assess each separately**, they are
  the actual decision
- the test suite and coverage runs
- lint / format (`ruff`, `black`, `pylint` in the dev group)
- evaluation and metrics over the **frozen artefacts already in the bundle** (`chlu/eval/`, the vendored
  TSB-AD under `chlu/eval/_tsb_vendor`)
- plotting / figure generation for the ICLR submission
- data preparation and dataset download
- small-scale smoke runs and ablations
- full training runs
- anything needing `cafe_bench` (note: a **separate checkout kept outside this repo**, so absent here)

## Call out explicitly
- **CPU-only blockers.** Grep for `jax.devices("gpu")`, `pmap`, sharding across devices, CUDA-only
  kernels. Hard NOs on this box.
- **Absent-dependency blockers.** Which classes die on a missing package or dataset rather than on
  compute? The bundle ships `.claude/data/gen/shapes3d/` artefacts — but does it ship the **Shapes3D HDF5
  dataset itself** (`chlu/data/shapes3d.py`, the `gen` extra, h5py)? Check, because that decides whether
  the generative line of work can run here at all.
- **The parallelism argument.** 16 cores with no GPU is genuinely good at *embarrassingly parallel* work —
  seed arrays, sweeps over small configs, `pytest -n auto` (pytest-xdist is in the `test` group). Identify
  which run shapes exploit that, and which are single-threaded-long where this box is simply worse than
  the Mac.
- **The split hazard — give this real attention.** If work runs in two places, results must merge. The
  charter's **YOU NEVER DELETE** rule and the single writable path `CHLU_HUB_OUTPUTS`
  (`.claude/outputs/<hub_slug>/`) shape this. How do results come home? Look for a `--return-pack`
  mechanism (`scripts/bundle_hub_workspace.sh` mentions one) and describe the round trip. What collides if
  the Mac and this box both produce outputs? Note `rsync` is **absent** on this machine, and the return
  path may assume it.

## Report shape
Lead with a **3–5 bullet recommendation**: the specific work to move here starting now, and the specific
work not to. If your recommendation is "run hub task X here", name X and say why it beats the other two.
Then the table, then the blockers, then numbered open questions.
