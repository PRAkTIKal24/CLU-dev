# gen-csf3-cpu-jobs — experiment-engineer report

**Task + acceptance criterion:** a checkpointed CPU jobscript that runs the gen CPU
experiments on `multicore` under `fatpou01` — accepted on the `-c > 84` refusal, the
missing-artefact-root abort, a local dry run, and **resume demonstrated by killing it**.

**Status: done.** `scripts/csf3/job_cpu_gen.sh` (new, 681 ll), `tests/test_job_cpu_gen.py`
(new, 690 ll): **76 passed, 0 skipped, 0 failed**, three consecutive runs. Every guard
driven to fire. `kill -9` mid-sweep → restart → **2 units SKIPPED, 3 run, 0 redone**.
⛔ **Every CSF3 leg is NOT RUN** — the cluster needs 2FA and is unreachable from here;
named individually in §7.

## ⭐ DIAL DECLARATION (echoed, protocol §7)
- **Dial:** ⛔ **none — infrastructure.** **Laundering control:** n/a. **Falsifies:** nothing.
- ⛔ **No PREREG** — the acceptance is a mechanism, not a measurement. No number here enters
  any table.

## ⚠ RECONCILIATION LIST (first-10-lines rule, protocol §5) — **2 items, both need an owner**
1. **`scripts/csf3/README.md` §2's file table does not list `job_cpu_gen.sh`.** ⛔ I do not
   own that file (task: "read them, follow them, leave them alone"), so it is **unchanged
   and now stale**. One row for the Hub/curator to add.
2. **`exp_gen_atom_mean` does NOT namespace its output dir by `n_val`** (§6.3 below). Two
   `--n-val` legs sharing one `--out` overwrite each other's `seed_<N>/atom_manifest.json`.
   My fingerprint refusal catches it *for jobs launched through this script*; ⛔ nothing
   catches it for a direct `python -m` invocation. Owner needed.

---

## 0. FLAG PROVENANCE

| item | value |
|---|---|
| branch / commit | `agent/experiment-engineer/gen-csf3-cpu-jobs` @ **`d57937c`** (single commit) |
| base | local `main` @ **`7295673`** ⚠ **main moved mid-spoke** (`d79ed46` → `7295673`); rebased onto the named base, see §8 |
| worktree | `../CHLU-cpujobs` (task-directed). Branch ref verified **from the main repo** before removal (protocol §3.2). |
| machine / env | Apple M1 Pro, **CPU**, main venv reused (protocol §4) — ⛔ **no worktree `uv sync`**, so no JAX/dep drift. Python 3.11, ruff from the main venv. |
| ⛔ NOT RUN | **everything on CSF3.** Every cluster leg named in §7. |
| seeds | n/a — no measurement. The demo sweeps use seeds 0–5 as *unit labels*, not as science. |
| new file defaults | `-p multicore`, `-A fatpou01`, `-c 16`, `-t 1-0`, `--requeue`, `--signal=B:USR1@300`, `--open-mode=append`; `SEEDS="0 1 2"`, `ARM=all`, `BUDGETS=""`, `QUICK=0`, **`RESUME=1`**, `FORCE_NEW=0`, `AGGREGATE=1`, `CKPT_EVERY_SECONDS=300`, `KEEP_LAST=5`, `GEN_LOCAL=0`, `DRY=0`, `SMOKE=0`, `PLAN=0` |
| ⛔ files NOT touched | `env.sh`, `job_gpu_wells.sh`, `setup_env_job.sh`, `stage_artefacts.sh`, `chlu/**`, `chlu/config.py`, `README.md` — **zero deletions, zero modifications; the commit is +1371/−0.** |

---

## 1. What I did

| # | deliverable | where | status |
|---|---|---|---|
| 1 | the CPU jobscript, experiment as a **parameter** | `scripts/csf3/job_cpu_gen.sh` | ✅ |
| 2 | `-c > 84` refusal naming `MaxTRESPU`/`AssocGrpCpuLimit` | §2 of the script | ✅ **fired** |
| 3 | `-A gpu-free` refusal | §2 | ✅ **fired** |
| 4 | five `CHLU_*` roots asserted at job start, **each named** | §5 | ✅ **fired** |
| 5 | `JAX_PLATFORMS=cpu` set **and asserted** | §3 + §10 | ✅ |
| 6 | ⛔⛔ checkpoint + resume, **two layers** | §11–13 | ✅ **killed** |
| 7 | tests | `tests/test_job_cpu_gen.py` | ✅ 76 passed |
| 8 | submit block for the Head | §5 below | ✅ |

**⛔ One launcher, not two.** `EXP=atom_mean|baselines` selects the module, its unit stage
and its manifest name; args pass through (`SEEDS`, `BUDGETS`, `ARM`, `QUICK`, `EXTRA_ARGS`).
A test asserts that **`job_cpu_gen.sh` is the only `scripts/csf3/job_*.sh` naming either
experiment** — a second jobscript makes it red.

### 1.1 The two resume layers (and why one is not enough)
- **Layer 1 — the unit ledger (this script).** ⭐ **The unit is a seed.** After a unit exits
  0 *and its manifest is on disk*, it is banked in `$CKPT_DIR/journal.json`:
  **tmp → `fsync` → `os.replace`** (atomic), **time-based rotated snapshots, last N kept**.
  On restart the unit prints `SKIP seed_N (manifest exists)`. ⇒ **a kill costs one unit.**
- **Layer 2 — inside the payload.** `exp_gen_atom_mean.run_seed` already skips budgets
  already in its own manifest (`"[tag] already on disk for this seed — skipped"`,
  `exp_gen_atom_mean.py:807-809`), so **re-entering a half-done unit is cheap too.**
  ⛔ `exp_gen_baselines.run_seed` has **no** such skip — its units are all-or-nothing, which
  is why the seed is the unit and not something coarser.
- **The fingerprint.** The journal carries `exp/dataset/pretext/module/unit_stage/arm/
  budgets/quick/extra_args/out/smoke`; a mismatched resume is **refused, exit 3, naming every
  differing key** — the `pilot-checkpoint-resume` pattern, ⛔ not a second mechanism.
  ⭐ **`SEEDS` is deliberately NOT in it**: narrowing/extending the seed list is how you
  repair a sweep, and the ledger is per-unit precisely so that is safe. Neither are cores /
  wallclock / job id — they cannot change a number.

---

## 2. ⭐⭐ THE KILL TEST (acceptance 3) — `kill -9`, not a simulation

`.claude/outputs/gen-csf3-cpu-jobs/kill_resume.{sh,log}`. 5 units, `kill -9` on the whole
process group ~10 s in, mid-unit-3.

```
attempt-1 exit: 137                          <- SIGKILL, not a clean shutdown
  DONE seed_0 in 4s / DONE seed_1 in 4s
  RUN  seed_2 ... Killed: 9
journal after the kill: units_done = {seed_0, seed_1}     <- seed_2 NOT banked
manifests on disk:      seed_0, seed_1                    <- and nothing half-written

attempt-2 (IDENTICAL command line, RESUME=1 is the default):
  [resume] journal ...: fingerprint OK, 2 unit(s) complete: seed_0 seed_1
  SKIP seed_0 (manifest exists)
  SKIP seed_1 (manifest exists)
  RUN  seed_2 / RUN  seed_3 / RUN  seed_4
  === summary: ran=3 skipped=2 failed=0 ===      <- ⭐ RESUMED, not restarted
attempt-2 exit: 0

attempt-3 (fully complete sweep):  ran=0 skipped=5 failed=0
```
Also driven, in the same log and as tests:
- **fingerprint refusal:** same `CKPT_DIR`, `QUICK=1` → `⛔ REFUSING TO RESUME … quick:
  journal='0' now='1'`, **exit 3**. Parametrised over 6 fields.
- **banked-but-manifest-gone:** `⚠ seed_1 is in the journal but its manifest is GONE —
  re-running` → `ran=1 skipped=4`. ⛔ A journal entry whose artefact vanished (scratch has a
  3-month cleanup and no backup) is **not** a completed unit.
- **exit 0 with no manifest** → `⛔ … wrote no atom_manifest.json — NOT marking it done`.
  ⛔ Banking that would make the hole **permanent across every future resume**.
- **a failed unit** → not banked, the other units still finish, job exits 1, and the
  **aggregate is skipped** ("an aggregate over a partial sweep is a table with a silent hole").
- **SIGTERM** as well as SIGKILL; and `SIGUSR1` prints `RESUME WITH: sbatch …`.

**⚠ HONESTY — what the kill test used.** The payload was `SMOKE=1` (a trivial manifest
writer), **not the real experiment**: there is **no artefact tree on this laptop**
(`artefacts/` does not exist) and a real `chlu` import costs ~20 min of cold start.
⭐ What is being tested is **layer 1, the jobscript's own ledger**, and that is exercised
end-to-end and truly `kill -9`'d. **Layer 2 is pre-existing `chlu/**` code I did not write
and am not permitted to touch**; I verified it by reading `exp_gen_atom_mean.py:803-823`.
⛔ I did **not** run a real gen unit anywhere.

---

## 3. The guards, all fired (acceptance 1 + 4)

`.claude/outputs/gen-csf3-cpu-jobs/guards.{sh,log}`. All aborts **exit 2** (refusal = 3).

| guard | trip | observed |
|---|---|---|
| **`-c` > 84** | 85/96/168/1500 | `⛔ -c 96 exceeds the per-user CPU cap on account 'fatpou01': MaxTRESPU cpu=84. … PENDS FOREVER on AssocGrpCpuLimit` |
| `-c` ≤ 84 | 1/8/32/**84** | accepted — ⭐ 84 is the entitlement, not a number to hide under |
| `-c` nonsense | `0 / -4 / eighty / 84.5` | refused, each by name |
| **account** | `SLURM_JOB_ACCOUNT=gpu-free` | `⛔ 'gpu-free' is the GPU association and carries NO CPU grant …` |
| account | `fatpou01` | accepted |
| **partition** | `serial` | `⛔ partition 'serial' is a ONE-CORE partition` |
| **artefact roots** | each of the 5, individually | `FATAL … gen artefact root(s) MISSING, by name:\n    CHLU_GEN_DECLIFT_DIR -> …` — ⭐ **names the one that is missing and blames no other** |
| artefact roots | `CHLU_CLAUDE_ROOT` gone | all five named (it is their parent — separate test) |
| **ckpt location** | under `$TMPDIR` | `⛔ … which is WIPED AT JOB END` |
| ckpt location | under `$HOME` | `⛔ … (quota'd, backed up). Put it under ~/scratch.` |
| ckpt location | under `$HOME/scratch` | ⭐ **accepted** — `~/scratch` string-starts with `$HOME` but is a different filesystem; the `$HOME` guard must not swallow the one correct location |
| axes | `EXP`/`DATASET`/`PRETEXT` unset, or off-whitelist | refused, naming the axis |

**Header, asserted as directives (comments stripped):** `-p multicore`, `-A fatpou01`,
`--requeue`, `--signal=B:USR1@300`, `--open-mode=append`, `-c 16 (≤84)`, `-t 1-0`.
⭐ **`gpu-free` never appears as a directive value** (it appears only in a "never use this"
comment — the test strips comments, or it would have proved the opposite of what it means).

---

## 4. The dry run (acceptance 2) — and a check the task did not ask for

`dryrun.log` prints the exact per-unit command. ⭐ **But a dry run only proves the script
printed what it meant to — not that the experiment accepts it.** So `argv_check.{py,log}`
(and a test) round-trips every emitted argv through **each module's real
`build_arg_parser`**, extracted by `ast` so nothing imports jax. **6/6 accepted.**

```
exp_gen_atom_mean --dataset shapes3d --pretext recon --stage atoms --seed 0 \
  --out <SWEEP ROOT> --budget b1 --budget b4 --n-val 4096 --tune-n 2048 --allow-knn-proto-move
  -> {'seed': [0], 'budget': ['b1','b4'], 'n_val': 4096, 'tune_n': 2048, 'allow_knn_proto_move': True}

exp_gen_baselines --dataset dsprites --pretext cls --stage baselines --arm knn --seed 0 \
  --out <PER-SEED DIR> --quick --no-plots --n-fit 64
  -> {'seed': 0, 'arm': 'knn', 'n_fit': 64, 'no_plots': True}
```
⭐ **The two `--out` are deliberately different, and this is the easiest thing to get wrong.**
`exp_gen_atom_mean` appends `seed_<N>/` itself (`run_experiment_gen_atom_mean`), so it gets
the **sweep root**; `exp_gen_baselines --stage baselines` passes `--out` **straight to
`run_seed`**, which does not, so it gets the **per-seed dir**. Getting these the same way
round is the difference between a sweep and one repeatedly-overwritten directory.

---

## 5. ⭐ ONE COPY-ABLE SUBMIT BLOCK FOR THE HEAD

```bash
# ── LAPTOP (VPN/campus) ───────────────────────────────────────────────────────
export GEN_GIT_COMMIT=$(git rev-parse HEAD)          # push_repo.sh strips .git
bash scripts/csf3/push_repo.sh                       # ⛔ the ONLY code path; `git pull` on CSF3 serves a stale fork

# ── CSF3 LOGIN NODE ───────────────────────────────────────────────────────────
cd ~/scratch/CHLU && mkdir -p logs
export GEN_GIT_COMMIT=<paste the hash from above>

# 0. dry-run the plan first — units, per-unit command, submit line. Runs nothing.
EXP=atom_mean DATASET=shapes3d PRETEXT=recon PLAN=1 bash scripts/csf3/job_cpu_gen.sh

# 1. atom_mean, 3 seeds, 32 cores          [~4 h laptop @8 cores -> ~1-1.5 h @32; -t 1-0 is 16x headroom]
sbatch -c 32 -t 1-0 --export="ALL,EXP=atom_mean,DATASET=shapes3d,PRETEXT=recon,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" scripts/csf3/job_cpu_gen.sh

# 2. baselines (knn + lace), 3 seeds       [-t 12:00:00; both arms per seed]
sbatch -c 32 -t 12:00:00 --export="ALL,EXP=baselines,DATASET=shapes3d,PRETEXT=recon,SEEDS=0 1 2,ARM=all,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" scripts/csf3/job_cpu_gen.sh

# 3. ⭐ RESUME AFTER ANY DEATH — the IDENTICAL line. RESUME=1 is the default;
#    banked units are skipped, never redone.  (Slurm --requeue does this by itself.)

# 4. pull results back
rsync -av csf3:~/scratch/CHLU/artefacts/outputs/gen-atom-mean-control/runs/shapes3d_recon/ \
          ./.claude/outputs/<consuming-slug>/shapes3d_recon/
```
**Cost: CSF3 CPU is FREE at point of use.** ⭐ `fatpou01` = `GrpTRES cpu=1500`,
`MaxTRESPU cpu=84`/user, **no GPU and therefore uncontended**, while `gpua` (76/76) and
`gpul` (84/84) are at their group caps and we scavenge at priority 5759. `-c 32` leaves room
for **two concurrent jobs** inside the 84 cap; `-c 84` is one job at full entitlement.

⚠ **Wall times are ARITHMETIC FROM THE MEASURED LAPTOP LEG, not measured on CSF3** — the
4 h/8-core figure in the task, scaled by cores, with no allowance for AMD-Genoa-vs-M1 per-core
throughput or for the parts of the sweep that do not parallelise. ⛔ Treat `-t` as headroom,
not as a prediction; the checkpoint is what makes a wrong guess cheap.

---

## 6. Findings

**6.1 ⭐ The `-c` value is the whole point, and 84 is not a number to stay under.** The task's
framing ("84 cores instead of 8") is why `-c` ≤ 84 is *accepted at 84* and tested at 84, not
just below it. Over the cap the failure mode is **not slowness** — `AssocGrpCpuLimit` pends
the job indefinitely, exactly as our GPU jobs are pending on `AssocGrpGRES` now.

**6.2 ⚠ `--requeue` alone would have been a trap.** `job_gpu_wells.sh` carries `--requeue`,
and requeue **re-runs the jobscript from line 1**. Without a unit ledger that is a *restart*,
not a resume — the flag would have looked like protection while silently redoing the sweep.
`--open-mode=append` + the journal is what makes it real.

**6.3 ⛔ `exp_gen_atom_mean` does not namespace `--out` by `n_val`** (reconciliation item 2).
It gained `--n-val`/`--tune-n`/`--allow-knn-proto-move` on `main` **while this spoke was in
flight** (`7a47b0a`), and `run_experiment_gen_atom_mean` still builds `out` from
`{dataset}_{pretext}` only. Two `--n-val` legs sharing one `--out` therefore overwrite each
other's per-seed manifests. Through this script they are **refused** (`extra_args` is in the
fingerprint, and the refusal message says "give this leg its own OUT/CKPT_DIR"); ⛔ a direct
`python -m` invocation is unprotected.

**6.4 A rotation bug I introduced and caught.** Snapshot names were whole-second, so six fast
units inside one second **collided** — the snapshot was overwritten, not rotated, and the ring
silently held fewer than `KEEP_LAST`. It flaked the test once, which is how it surfaced. Names
are now zero-padded nanoseconds (fixed width, so the lexicographic glob sort stays
chronological) and the test is parametrised over `keep ∈ {1,2,4}` with a distinctness
assertion. ⚠ **Reported because a checkpoint ring that quietly holds fewer snapshots than it
claims is exactly the class of bug this rig exists to prevent.**

**6.5 `JAX_PLATFORMS=cpu` does double duty.** It suppresses the benign
`cuInit(0) failed: Unknown CUDA error 303` **at the source** on a GPU-less node (rather than
filtering a scary-looking line downstream), and it makes the preflight assertion meaningful:
⛔ a CPU run must never be able to report GPU numbers. BLAS/OpenMP pools are pinned to the
grant — an unbound numpy/sklearn on a 168-core `multicore` node oversubscribes its cgroup and
runs **slower** than the cores it was given.

---

## 7. ⛔ NOT RUN — every cluster leg, named

**CSF3 requires 2FA and is unreachable from this agent.** None of these were executed:
1. `sbatch` of `job_cpu_gen.sh` under `fatpou01` on `multicore` — **the `#SBATCH` header has
   never been parsed by a real Slurm.** ⭐ Highest-value first check for the Head: `PLAN=1`
   on the login node, then one `-c 32 -t 0-1 QUICK=1` job.
2. The **real** `exp_gen_atom_mean` / `exp_gen_baselines` payload — anywhere. Layer-2 resume
   is read, not run.
3. The **JAX CPU preflight** (§10) — it imports jax; skipped in every local run
   (`GEN_LOCAL=1` ⇒ `SKIP_JAX_PREFLIGHT=1`). ⚠ **Untested code path.**
4. `env.sh` sourcing, `module purge`, venv activation — all `GEN_LOCAL=1`-skipped.
5. The `--requeue` / `--signal=B:USR1@300` **Slurm-side** behaviour. I sent `SIGUSR1` by hand
   and the handler fired; **Slurm sending it 300 s before the wallclock is untested.**
6. Wall-clock and core-scaling on AMD Genoa (see §5's ⚠).

## 8. Git footprint
- **Branch** `agent/experiment-engineer/gen-csf3-cpu-jobs`, **commit `d57937c`** (one).
- **Files:** `scripts/csf3/job_cpu_gen.sh` (+681, new), `tests/test_job_cpu_gen.py` (+690,
  new). **+1371 / −0** — ⛔ **no existing file modified or deleted.**
- **Worktree** `../CHLU-cpujobs`; branch ref confirmed from the **main repo** via
  `git log --oneline main..agent/…` **before** removal (protocol §3.2, w4 precedent).
- ⚠ **`main` advanced mid-spoke, `d79ed46` → `7295673`** (`7a47b0a` + merge `7295673`,
  `gen-atom-mean-resweep`). Rebased onto the **named base (local `main`)**, ⛔ never
  `origin/main` (§3.5 / §7.21). **No conflicts.** Tests re-run green on the rebased tree, and
  the argv round-trip (§4) now covers the three flags that merge added.
- **Commands:** `bash -n` (clean), `ruff check` (clean — ⚠ `ruff format` is **not** enforced
  in this repo: 105/141 existing test files would reformat, so I left formatting alone),
  `pytest tests/test_job_cpu_gen.py -q` → **76 passed, 0 skipped, 0 failed** (×3).
  ⛔ **No full suite** (task-forbidden).
- **Unresolved conflicts:** none.

## 9. Open questions / risks
1. **§7.1 is the only thing between this and working.** One `PLAN=1` + one short `QUICK=1`
   job on the login node discharges most of §7.
2. **Does the Head want `-c 84` (one job, full entitlement) or `-c 32` (three concurrent)?**
   I defaulted the header to 16 and the submit block to 32; the cap is enforced either way.
3. **`~/scratch` auto-cleans at 3 months with no backup** — the journal lives there. A sweep
   left idle past the cleanup loses its ledger; the manifests-on-disk cross-check degrades
   that to a full re-run rather than a silent hole, but it is a re-run.

## Proposed handover updates (for the Hub)
- **§4 / cluster runbook:** `scripts/csf3/job_cpu_gen.sh` is **the** CPU launcher for gen
  sweeps (`EXP=atom_mean|baselines`), on `fatpou01`/`multicore`, ⛔ never `gpu-free`,
  ⛔ never `-c > 84`. Resume is the default; the resume command is the submit command.
  ⛔ Do not add a second CPU jobscript — add an `EXP=` case (a test enforces this).
- **§7 Known Issues — new entry.** **`exp_gen_atom_mean` does not namespace `--out` by
  `n_val`**: two `--n-val` legs sharing one `--out` overwrite each other's
  `seed_<N>/atom_manifest.json`. Jobs launched through `job_cpu_gen.sh` are refused by the
  resume fingerprint; ⛔ a direct `python -m` call is unprotected. **Owner needed.**
- **Curator task (1 line):** add `job_cpu_gen.sh` to `scripts/csf3/README.md` §2's file table
  — ⛔ I was scoped out of that file, so it is stale as of `d57937c`.
- **Standing-rule evidence:** the "cluster runs must checkpoint" rule now has a second
  demonstrated instance (`kill -9`, CPU sweeps) alongside `pilot-checkpoint-resume`. The
  pattern is **per-unit ledger + payload-internal skip**, and ⛔ the unit is only banked when
  its artefact is on disk.
