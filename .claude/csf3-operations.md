# CSF3 OPERATIONS — the complete reference

**Written 2026-09-11. Consolidates `scripts/csf3/README.md`, the five spoke reports that hardened the path (`csf3-runbook`, `csf3-download-race-and-sbatch`, `csf3-memory-fit`, `pilot-checkpoint-resume`, `pilot-ttt-nan-and-d5-wiring`, `gen-csf3-launcher`), and the run-1/2/3 launch packages.**

⚠ **Read this before editing any `scripts/csf3/*` file:** the canonical versions live on `main`. If your checkout sits on an old agent branch, the scripts you see are stale — `git show main:scripts/csf3/<file>` is the truth.

---

## 1. What CSF3 is, and how you reach it

University of Manchester CSF3. Scheduler is **Slurm** (migrated from SGE).

```bash
ssh <uom-it-username>@csf3.itservices.manchester.ac.uk     # username form mabcxyz1, NOT your email
```

- **Off campus you need GlobalProtect VPN.** Without it the hostname does not even resolve. This is why every agent-authored script is marked UNTESTED-ON-CLUSTER — no agent has ever been able to establish the VPN. The Head runs everything first.
- **Login nodes have campus-only internet.** No GitHub, no PyPI. **Compute nodes have full off-campus access.** Every download — wheels, datasets, the uv installer — must therefore happen *inside a job*.

## 2. Hard constraints

| Thing | Limit |
|---|---|
| `gpuA` partition | 19 nodes × 4× A100 SXM4 **80 GB** (sm_80), 48 AMD Milan cores, 512 GB/node, driver 580.126.09 |
| GPUs at once | **≤ 4 per user**, free at point of use, `-G N` required |
| Host cores | **≤ 12 per GPU**, ~10.4 GB RAM per core |
| Wallclock | **`-t` is mandatory** — sbatch rejects jobs without it. Batch max **4-0**; interactive max 1 day |
| Other partitions | `serial` (1 Intel core, 7-0) · `multicore` (AMD Genoa 2–168 cores, 7-0) · `interactive` (6 h) · `gpuL` (L40S 48 GB) · `gpuA40GB`/`gpuH` restricted. V100s removed Oct 2025 |
| `$HOME` | `/mnt/iusers01/<group>/<user>` — small, backed up, keep ≤ 50 GB, **do not run jobs here** |
| `~/scratch` | fast, **no backup, 3-month auto-cleanup**, run jobs here |
| `$TMPDIR` | 1.6 TB node-local NVMe, wiped at job end. We don't currently use it |
| Job arrays | `-a 0-N`, `${SLURM_ARRAY_TASK_ID}`, max task id 250000; `%M` suffix throttles concurrency |

Slurm gives your job **exclusive** use of its assigned GPUs — nothing else shares them.

## 3. The host-RAM ceiling — the constraint that has bitten us most

**The A100's 80 GB is rarely the problem. Host RAM is.**

`-c 12` (the 1-GPU maximum) buys ~120 GB and **there is no `--mem` that buys more**. Job 18136619 was `oom_kill`ed at MaxRSS **125.6 GB** against ReqMem 125.7 GB — 22 h into training, 45 min into the post-training eval block. Device peak at the time was only ~8.3 GiB.

Ways around it, in order:
1. **Reduce footprint.** `remat_chunks=true` cut the projected pilot peak from 97.8 GiB to **8.34 GiB** (conservative) / 59.9 GiB (worst case), and is **bitwise identical** on every decision-bearing quantity (the rematted gradient differs by ‖Δg‖/‖g‖ = 8.67e-10, i.e. 0.7 % of one float32 ULP). `eval_cache_hygiene` (default ON) releases each eval phase's XLA executables before the next compiles, and prints an `[rss]` line at every phase boundary so the next crash is attributable to a named phase.
2. **`-G 2 -c 24`** = 240 GB with one GPU deliberately idle. **This is what runs 1–3 actually used** — 240 GB against a measured **219.08 GB** peak. Dropping back to `-G 1 -c 8` OOM-killed the leg four times.
3. `SET="... plan_workers=4"` — declared fallback, measured decision-identical.

## 4. The environment — why it is built the way it is

**We do NOT `module load libs/cuda/...`.** JAX is installed from pip CUDA wheels (`jax[cuda12]` → `jax-cuda12-plugin[with-cuda]`) which bundle CUDA and cuDNN themselves and need only the node driver (≥ 525.60.13; the nodes run 580.126.09). A CUDA module on `LD_LIBRARY_PATH` can shadow the bundled libs and break JAX.

**Pinned versions** (Python 3.11, laptop parity): `jax==0.9.0`, `jaxlib==0.9.0`, `jax-cuda12-plugin==0.9.0`, `equinox==0.13.4`, `optax==0.2.6`, `diffrax==0.7.0`.

**Build it with `setup_env_job.sh`** — a `serial` batch job, because login nodes can't reach PyPI:
```bash
cd ~/scratch/CHLU && sbatch scripts/csf3/setup_env_job.sh
```
It installs uv to `~/.local/bin`, puts the uv cache and managed CPython **on scratch** (CUDA wheels are ~6 GB — too big for the home quota), then runs `uv sync --frozen --extra cuda --extra eval --python 3.11`. `--frozen` means it installs exactly the rsynced `uv.lock`, giving bit-parity with the laptop. It ends with a CPU-side import check that **asserts JAX is still 0.9.x** — the `eval` extra must never bump it.

Re-running is safe and idempotent; it also repairs a partial venv. If scratch's 3-month cleaner eats the interpreter, just re-run it.

**`uv.lock` is tracked now**, so `git pull` carries it. Historically it was gitignored, which is why `push_repo.sh` exists (rsync was the only way to get it across).

⚠ **Worktree venv hazard (w6):** a fresh `uv sync` inside a worktree can resolve *newer* packages than the main venv — this once moved JAX 0.9.0 → 0.10.2 and flipped a bit-level test. **Reuse the main venv instead:** run `PYTHONPATH=<worktree> /Users/user/Desktop/CHLU/.venv/bin/python …` with cwd in the worktree. If you must sync, use `uv sync --frozen` and report the resolved JAX version.

**`env.sh`** is sourced by every job after `module purge`. It `cd`s to `$CLU_REPO` (because `projects/` is created relative to `$PWD`), activates `.venv`, and sets: `MPLBACKEND=Agg` (no `$DISPLAY` on compute nodes), `SCIKIT_LEARN_DATA`, `PYTHONUNBUFFERED=1`, and the **JAX persistent compilation cache** (`~/scratch/.jax_cache`, min compile time 1 s) so repeat jobs skip XLA compilation. GPU memory preallocation is left at JAX's default ~75 % because the job owns its GPU.

⚠ **JAX cold start on the laptop is pathologically slow** (~20 min for even `chlu --help` or a pytest collect). Budget for it; don't mistake it for a hang.

## 5. The scripts

All in `scripts/csf3/`. Canonical copies on `main`.

| Script | Runs where | What it does |
|---|---|---|
| `env.sh` | cluster, sourced | venv + headless env vars. **Never sbatch it** |
| `setup_env_job.sh` | cluster, `serial` | one-time env build (§4) |
| `push_repo.sh` | **laptop** | rsync repo → `~/scratch/CHLU` |
| `sync_project.sh` | **laptop** | `pull`/`push` of `projects/<name>/` artifacts |
| `job_cpu_smoke.sh` | cluster, `serial` | ~5-min CPU smoke: CLI, Agg plot, core tests |
| `job_gpu_single.sh` | cluster, `gpuA` | one experiment, one A100, GPU preflight |
| `job_gpu_array_seeds.sh` | cluster, `gpuA` | seed-sweep array |
| `job_gpu_cluformer.sh` | cluster, `gpuA` | **the tier-III pilot — runs 1, 2 and 3 all used this** |
| `job_gpu_c3_seeds.sh` | cluster, `gpuA` | **the C3 ladder: (arm × seed) as array tasks** |
| `job_gpu_gen.sh` | cluster, `gpuA` | **the generative pivot (added 2026-09-09)** |
| `job_gpu_cafe.sh`, `job_gpu_eval.sh`, `job_gpu_kt.sh` | cluster, `gpuA` | earlier campaigns |
| `requirements-csf3.txt` | either | pinned pip fallback, generated from `uv.lock` |

## 6. The standard workflow

```bash
# LAPTOP (on campus or VPN)
git push clu-dev main                      # code + uv.lock
# or, if the lock is not tracked in your branch:
export CSF3_USER=<username>; scripts/csf3/push_repo.sh

# CSF3
ssh $CSF3_USER@csf3.itservices.manchester.ac.uk
cd ~/scratch/CHLU && git fetch clu-dev && git checkout main && git pull clu-dev main
mkdir -p logs                              # -o/-e point at logs/, and it must exist
sbatch scripts/csf3/setup_env_job.sh       # once, or after a dependency change

# stage data FIRST, serially, once  (see §7)
sbatch --export=ALL,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_cluformer.sh

# submit the real work (see §8)
# then verify, then walk away:
squeue --me
ls -l logs/<jobname>-*.out | wc -l         # a launch with no log did not happen

# LAPTOP — pull results promptly (scratch has no backup)
rsync -av csf3:~/scratch/CHLU/.claude/outputs/<dir>/ ./.claude/outputs/<dir>/
```

Monitoring a live GPU: `squeue` to find the node, `ssh <node>` (allowed where you have a job), then `module load tools/bintools/nvitop && nvitop`, or `nvidia-smi`.

Post-hoc timing: `sacct -j <jobid> --format=Elapsed,MaxRSS,ReqMem`.

Interactive debugging: `srun -p gpuA -G 1 -n 1 -t 1-0 --pty bash`, then `source scripts/csf3/env.sh`. 24 h cap; don't idle-hold GPUs.

## 7. Data staging — always serial, always first

**Array tasks are hard-blocked from downloading** (`chlu.data.corpora.in_array_task`). This is enforced in code, not advisory.

The reason: `download_file` originally wrote a *shared* `<name>.part`, so N parallel jobs corrupted it or renamed it out from under each other (`sha256 mismatch`, `FileNotFoundError: ….part -> ….parquet`). It now uses a process-unique `mkstemp` temp, verifies sha256 on that temp, then does an atomic same-filesystem `os.replace` — plus a check-final-first short-circuit. Last-writer-wins is safe because every temp is an identical verified file.

Still: **stage once, serially, before any array.**
```bash
sbatch --export=ALL,CORPUS=enwik8,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_c3_seeds.sh
```

## 8. Submitting the real jobs

### 8.1 `job_gpu_cluformer.sh` — the tier-III pilot

Everything is passed by `--export`; nothing is set by editing the module (an edited module is a provenance hole, a flag is recorded verbatim in the artifact's `flags` block).

Knobs: `STAGE` (pilot|toy) · `STG` (s1–s4) · `SEEDS` · `STEPS` · `ARMS` · `OUT` · `STAGE_ONLY` · `MEM` (→ StreamMemoryConfig) · `STORE` (→ CluSystemConfig) · `SET` (→ PilotConfig) · `RESUME` · `D5` · `PREREG_CONT`.

**One seed per job at `-t 4-00:00:00`.** Five arms × three seeds in one job is ~270 h against a 96 h limit — the monolith cannot fit, and one failure loses everything. Per-seed jobs also give failure isolation and let run1/run2 pairs complete early when interleaved.

The actual run-3 shape:
```bash
sbatch --export=ALL,SEEDS="0",STAGE=pilot,STG=s4,D5=1,RESUME=1,OUT="$RUN3_OUT",\
MEM="chunk=64 address_steps=64 read_steps=64 traj_stride=8 psi_hidden=128 write_inner_steps=40 \
write_n_perturb=8 retry_rounds=1 conv_kernel=4 mlp_mult=4 atom_place_radius=0.3 remat_chunks=true \
psi_payload_residual=true psi_residual_source=q_star erosion_partition=true",\
STORE="write_margin=0.6",SET="steps=4000 warmup=200 eval_batches=40 dyneval_batches=40 \
monitor_every=25 plan_workers=8 liveness_lanes=1",\
PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed0_PARTIAL.json flag=memory.erosion_partition prereg=$PREREG_MD" \
--mail-user=$CLU_MAIL -G 2 -c 24 -t 4-00:00:00 --job-name=clu-run3-s0 scripts/csf3/job_gpu_cluformer.sh
```

### 8.2 `job_gpu_c3_seeds.sh` — the C3 ladder

Unit of work is **one (arm, seed) pair = one array task = one A100**. 5 arms × 3 seeds = 15 tasks from one sbatch, each with its own wallclock, failure domain and resumable journal.

```bash
sbatch -a 0-14%4 -t 1-00:00:00 --mail-user=$CLU_MAIL \
  --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=3,SEED_BASE=0,D5=1,SLICES=1 \
  scripts/csf3/job_gpu_c3_seeds.sh
```
⛔ **Each arm needs its own `--out`** (`$OUT_BASE/<arm>_s<seed>/`). `arms` is a PilotConfig field, so a narrowed arm list is a *different config* and the journal check refuses to resume a shared directory. Every task's byte ledger still carries all five arms, so the matched-state-byte table can be assembled from any one of them.

### 8.3 `job_gpu_gen.sh` — the generative pivot

Writes into a **fresh tree no earlier campaign has touched**: `$SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/` with `logs/`, `artifacts/` and `run_manifest.json` (git commit, resolved library versions, every non-default flag, the axes, slurm ids, status).

Axes, all of which **fail loudly if unset**: `DATASET` (shapes3d|dsprites), `PRETEXT` (recon|cls), `STAGE` (encoder|decoder|store|cell|**certificate**). Plus `ARMS` (clu, store_removed, lace, knn), `SEEDS`, `CONCURRENCY` (default 2).

⭐ **The stage DAG is a chain with two leaves** (`certificate` added 2026-09-13, `gen-launcher-fixes`):
```
encoder -> decoder -> store -+-> cell
                             `-> certificate
```
`cell` and `certificate` are **siblings**: both declare `store` as upstream, neither reads the other, and **each owns its own workspace subdirectory**. The certificate is re-materialised from the store's own `store_plan.json` grouping and reads nothing the cell produces; gating it on `cell` would serialise two independent leaves and let a cell failure block a result that does not depend on it. Default payload module `chlu.experiments.exp_gen_certificate`.
⛔ **The pre-stage workaround is banned and is now untestable-by-construction:** `STAGE=cell` + `EXTRA_ARGS=--stage certificate` (what `gen-certificate` §10 had to use) wrote certificate artefacts into `cell/` and corrupted a different stage's provenance directory. Four tests exist solely to stop it returning.

```bash
export GEN_GIT_COMMIT=$(git rev-parse HEAD)      # ON THE LAPTOP — push_repo.sh excludes .git/
DATASET=shapes3d PRETEXT=recon STAGE=encoder PLAN=1 bash scripts/csf3/job_gpu_gen.sh   # print the plan
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-11%2 -t 12:00:00 scripts/csf3/job_gpu_gen.sh
sbatch --dependency=afterok:<jobid> --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=decoder,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-11%2 -t 12:00:00 scripts/csf3/job_gpu_gen.sh
```
The array size is checked against arms × seeds and a partial range is **refused** unless `PARTIAL=1` marks it deliberate. Each stage checks its upstream stage's manifest for `status == "ok"` **for the same arm and seed** before doing any work.

**The certificate leaf.** Every arm (including the laundering controls) is scored inside **one** job so they share a byte-identical context ⇒ the array is over **seeds only**, `ARMS=clu`, `-a 0-2`. ⭐ One literal line per submission (§10); the `--export` value contains spaces (`SEEDS=0 1 2`) so the whole argument is **one quoted shell word** — `sbatch --export` splits on commas only.
```bash
DATASET=shapes3d PRETEXT=recon STAGE=certificate ARMS=clu SEEDS="0 1 2" PLAN=1 bash scripts/csf3/job_gpu_gen.sh
sbatch --export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=certificate,ARMS=clu,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" -a 0-2%2 -t 04:00:00 scripts/csf3/job_gpu_gen.sh
```
Realised paths: `$SCRATCH/CHLU-gen/workspaces/shapes3d_recon/certificate/clu_s{0,1,2}/` — ⭐ note `certificate/`, **not** `cell/`.

⚠ **The line above is not yet runnable for a REAL (non-`SMOKE=1`) certificate job, and the reason is not in the launcher.** `exp_gen_certificate` resolves its inputs independently of the launcher workspace: `bridge_root` defaults to `<repo>/.claude/outputs/gen-feature-bridge/runs` and `store_runs` to `<repo>/.claude/outputs/gen-store-grouping/runs` — and **`push_repo.sh` has `--exclude '.claude/'`** (line 42). On CSF3 neither exists, so the job raises `FileNotFoundError` ("no frozen bridge at …" / "no store plan at …"). ⭐ **Fail-loud, so no wrong number can escape** — but the run yields nothing. Stage those artefacts onto scratch explicitly, then pass `EXTRA_ARGS=--store-runs <…> --bridge-root <…> --data-root <…>` (verified locally to reach the payload's argv intact and to be recorded verbatim in `run_manifest.json`), and add `ALLOW_MISSING_UPSTREAM=1` for as long as no `store` stage has been run through the launcher. ⛔ Also note `exp_gen_encoder`, `exp_gen_decoder` and `exp_gen_cell` **do not exist**; the launcher's `find_spec` preflight aborts on them before any wallclock is spent, so those stages are `SMOKE=1`-only today.

## 9. Resume, journals, and pre-registered continuations

`RESUME=1` picks a run back up from `pilot_<scale>_seed<N>_PARTIAL.json` + `ckpt_<arm>_seed<N>.eqx`. Banked phases are lifted verbatim; a trained-but-not-evaluated arm keeps its weights and re-runs only its evals — exactly what attempt 1 lost (22 h of training, killed 45 min into eval).

⛔ **It refuses to resume a journal written under a different config.** A resubmission must carry byte-identical `MEM`/`STORE`/`SET`. Don't transcribe by hand — regenerate the flags from the journal itself.

⚠ **The fingerprint once retro-invalidated every journal on disk:** `_flag_dict` built two groups from non-default keys only but the `memory` group from a full `asdict(StreamMemoryConfig())`, so *any* newly added field made every old journal unresumable. Fixed: a key the journal predates, sitting at its own field default, is accepted (and printed); anything else still refuses.

**`--d5`** (the anytime curve) is a CLI argument, not a config field, so it does not enter the resume fingerprint — a finished leg can be re-resumed with `RESUME=1 D5=1` to compute only the missing curve. ⛔ Don't also pass `ARMS` on such a re-resume; `arms` *is* a config field.

**`PREREG_CONT`** exempts the state-byte budget check *only*, and only when the config is identical to the named journal except one registered flag. It must be on **every submission and every re-resume** — without it the budget refuses the leg before training starts.

## 10. Rules learned the hard way

- **One literal command line per submission.** Building the argument list in a shell variable and expanding it **does not word-split under zsh** and submits garbage. This bit us live.
- **Verify logs exist before walking away.** `ls -l logs/<name>-*.out | wc -l` against the expected count. A launch that produced no log did not happen.
- **`mkdir -p logs` first** — `-o logs/%x-%j.out` fails if the directory is missing.
- **Pull results promptly.** Scratch has no backup and a 3-month cleanup.
- **Back up outputs before any git operation on the cluster.**
- **Don't assume paths — find them.** `find ~/scratch -name 'pilot_pilot_seed0_PARTIAL.json'`. Local pulled copies have been observed to under-report what's on the machine.
- **Pin sha256 when referencing a journal**, and treat a mismatch as a question about which copy is authoritative, not as a reason to drop the pin.
- **GPU preflight fails fast** — every GPU job asserts `jax.default_backend() == "gpu"` rather than burning wallclock on CPU.
- **`--quick` does not shorten CHLU dynamics training** for exp-a/b (only baselines/exp-c). Budget wallclock accordingly.
- Too small a `-t` gets SIGTERM, 30 s grace, then SIGKILL.

## 11. Provenance and honesty markers

Scripts carry three labels: **[DOCS]** (verified against official CSF3 documentation, with the page's last-modified date), **[LOCAL]** (executed on the laptop), **[UNTESTED-ON-CLUSTER]** (prepared but never submitted by the authoring agent, because no agent can establish the VPN). **CSF3 also requires 2FA, which is unreachable from an agent session** — so no agent has ever submitted anything, and no agent report may mark a cluster item as passing.

**`job_gpu_gen.sh`, as of 2026-09-13 (`gen-launcher-fixes`):**
- **[LOCAL] verified:** `bash -n` clean; `shellcheck` clean (rc=0, run via `uvx --from shellcheck-py shellcheck`, with a deliberately-bad control file confirming the checker actually fires); `tests/test_gen_launcher.py` **39 passed** (25 pre-existing + 14 for the `certificate` stage), main venv, ~70 s. Covered: the unset/invalid-axis aborts, grid/path construction, the array-size guard, the provenance guard, the slurm-stream capture, the full `encoder→decoder→store→cell` chain, and the `certificate` leaf (own path, `store` upstream, own default module, and each pre-existing guard re-asserted **under `STAGE=certificate`**) — all through `GEN_LOCAL=1 SMOKE=1`. Full suite before/after on a quiet machine: **2191 → 2205 passed, 13 skipped, rc=0 both** (Δ = +14 = exactly the new tests; nothing else moved).
- ⛔ **[UNTESTED-ON-CLUSTER] — everything cluster-side.** **No job has ever been submitted and no job id exists** for *any* gen stage: not the `-p serial` plumbing smoke, not encoder/decoder/store/cell, not `certificate`. Nothing about GPU preflight, `env.sh` sourcing on a compute node, the slurm-stream copy in the exit trap, `$SCRATCH` resolution, or wallclock sizing has ever been observed. The `-p serial` smoke line in §8.3 remains the first thing to try.

## 12. What has actually run

- **Runs 1, 2, 3** of the tier-III pilot, all through `job_gpu_cluformer.sh`, `-G 2 -c 24 -t 4-00:00:00`, one seed per job. Run 3 = run 2 + `erosion_partition=true` under `PREREG_CONT`. Artifacts pulled to `.claude/outputs/cluformer-pilot/csf3_outs/run{1,2,3}/`.
- **The C3 ladder has not been launched** — zero arms trained.
- **The generative pivot has STILL not been launched** — zero jobs, zero job ids. `job_gpu_gen.sh` landed 2026-09-09 and gained its `certificate` stage 2026-09-13; **both are cluster-untested** (§11). Every generative-pivot number to date — including the certificate barrier 0.9146 vs naive 0.5630 — was produced **locally on CPU**, not on CSF3. ⛔ **Cluster smoke still gates the cell** (R26 ruling 7): nothing pre-registered runs before it returns.
