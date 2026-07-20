#!/bin/bash --login
# scripts/csf3/job_gpu_kt.sh — the Kosterlitz-Thouless confirmation-at-scale
# tranche (Thread-10). One MODE per job; array tasks = seeds (or sweep cells).
#
# WHY THIS RUN EXISTS. The KT physics is already CONFIRMED on the real CLU 2-D
# path on a laptop (`kt-2d-csf3`): the kill criterion passed (an L=8 CLU-Langevin
# torus reproduces reduced-XY rho_s to 2-7%), the Nelson-Kosterlitz 2/pi jump and
# T_KT are measured to <1%, and the memory contrast (1-D degrades, 2-D improves)
# is decisive. EXACTLY TWO quantitative exponents remain laptop-under-resolved,
# and this tranche exists to close ONLY those two:
#
#   (a) 2-D winding survival at L >= 32. Below T_KT tau rises steeply with L
#       (log-log slope +5.0/+3.5 at T/J=0.6/0.7) but the predicted SIGN CHANGE
#       above T_KT was not resolved at L <= 16, where vortex-diffusion traversal
#       (~L^2, positive) masks the negative Arrhenius exponent pi*rho_s/T - 2.
#       -> MODE=winding2d
#   (b) 1-D clean tau ~ 1/N slope -1. The laptop got -0.7 at T/J=1.0.
#       -> MODE=winding1d       ** READ THE WARNING IN §(b) BELOW BEFORE RUNNING **
#
# T_KT = 1.786 * kappa * r*^2 = **0.0893** CLU units at kappa=0.05 (= 0.8929 J,
# J = 2*kappa*r*^2 = 0.10); measured 0.0898. ⚠ NOT "0.1786" (retracted, factor 2).
#
# NO DATASET DOWNLOAD. Confirmed: every KT mode is synthetic / self-generated —
# the lattice is constructed in-process and the initial conditions come from an
# in-process reduced-XY Monte-Carlo warm start. There is no `--download`, no
# dataset cache, and hence NO exposure to the shared-cache race that bit the
# voraus flagship. Parallel tasks touch nothing but their own $OUT.
#
# SETTINGS DISCIPLINE (handover §7.22). float64 + langevin_noise="fdt" +
# newtonian_learned + no governor. These are ASSERTED in-process
# (chlu/experiments/kt/clu_path.py::assert_kt_settings) so a misconfigured run
# dies loudly instead of silently producing garbage: under the repo-default
# "legacy" noise T is NOT in energy units and none of this physics holds.
#
# ---------------------------------------------------------------------------
# LAUNCH RECIPES (sizings below are MEASURED on the dev laptop, not guessed)
# ---------------------------------------------------------------------------
#
# (a) 2-D WINDING SURVIVAL, L>=32 — the sign-change run. **CPU, NOT GPU**:
#     this mode is reduced-XY Model-A Metropolis in numpy and would idle an
#     A100. Submit it to `serial` (1 core/task, 7-day limit, no 4-GPU cap):
#
#       cd ~/scratch/CHLU && mkdir -p logs
#       unset MODE SEED_BASE OUT EXTRA_ARGS      # clear stale values from a prior attempt
#       export EXTRA_ARGS="--l-values 16 24 32 48 64 --tj-values 0.60 0.70 1.00 1.10 1.20 1.30 --nwalk-2d 96"
#       sbatch -p serial -G 0 -c 1 -t 8:00:00 -a 0-2 --mail-user=$CLU_MAIL \
#              --export=ALL,MODE=winding2d,SEED_BASE=700,OUT=$HOME/scratch/clu_kt/w2d \
#              scripts/csf3/job_gpu_kt.sh
#
#     ⚠ THE SPLIT RULE (learned the hard way, 2026-07-20 — three failed launches):
#       * Variables WITHOUT spaces (MODE, SEED_BASE, OUT) go INLINE in --export.
#         Inline assignments OVERRIDE the inherited environment, so a stale value
#         from a previous attempt cannot leak in. `--export=ALL` alone does NOT
#         protect you: it inherits whatever is still exported in your shell, which
#         is how MODE=700 survived a *corrected* relaunch and tripped the guard.
#       * Variables WITH spaces (EXTRA_ARGS) CANNOT go inline: sbatch splits
#         --export on COMMAS and cannot carry an embedded space. Export those in
#         the shell; --export=ALL then carries them through.
#       * `unset` first. Belt and braces against stale state.
#       * Do NOT use multi-line backslash continuations inside the --export value:
#         they mangle on copy-paste (`-bash: syntax error near unexpected token
#         'newline'`), and two-vars-on-one-line collapsed `export MODE=winding2d
#         SEED_BASE=700` into `export MODE=700`.
#     Applies to every recipe in this header.
#
#     Sizing (measured, nwalk=4 probe scaled to nwalk=96): above-T_KT cells are
#     ~free (tau_med 85-97 sweeps at L=32/48/64, seconds/cell); the cost is the
#     below-T_KT cells, which censor at n_max=20000 (~1.8/3.6/5.8 min per cell
#     at L=32/48/64 for nwalk=24, so ~4x that at nwalk=96). Whole grid per seed
#     ~2-3 h => -t 8:00:00 with headroom. 3 seeds = 3 array tasks.
#     ⚠ PRE-REGISTERED, and a real result either way: the nwalk=4 probe already
#     shows tau at T/J=1.10 going 45 (L=16, laptop) -> 89 -> 97 -> 85
#     (L=32/48/64), i.e. **the +1.1 apparent slope FLATTENS to ~0 by L>=32**.
#     That is the L^2 masking dying out on schedule. Whether it goes properly
#     NEGATIVE is what the full-statistics run decides. Predict: slope < +0.5 at
#     T/J=1.10 and slope < 0 at T/J=1.30. Do NOT reinterpret after the fact.
#
# (b) 1-D CLEAN tau ~ 1/N — ⚠ **DO NOT LAUNCH AS ORIGINALLY SCOPED.**
#     The scoped fix ("rerun at lower T") does NOT work, measured on the laptop
#     while sizing this script:
#       * the MSD estimator SATURATES. At T/J=1.0, N=8, seed 31 the SAME run
#         fits rate 2.5e-4 over t<=2500 but 4.0e-5 over t<=50000 (6x drift), and
#         the apparent N-slope collapses 0.39 -> ~0 as the window shortens. The
#         laptop's 3e4-step runs were saturation-dominated, so the reported -0.7
#         is very likely a fit artifact, NOT the xi~1.2 effect it was ascribed to.
#       * lowering T barely helps: rate(N=8) only falls 4.0e-5 -> 3.2e-5 from
#         T/J=1.0 -> 0.5, while the N-slope FLATTENS 0.39 -> 0.15 (wrong way).
#       * root cause: the winding is barely metastable here.
#         E_wind(N=8,w=1) = N*J*(1-cos(2pi/N)) = 0.234 vs T = 0.10 at T/J=1.0,
#         i.e. E/T = 2.3 — the ring simply relaxes; there is no long-lived
#         winding whose lifetime could scale as 1/N.
#     A well-posed window opens only at T/J <= 0.2 (E/T >= 11.7), where MSD is
#     still << 1 after 5e4 steps. So IF the Hub wants (b), run it there, with a
#     diffusive-window fit and many more walkers, and treat it as EXPLORATORY:
#
#       unset MODE SEED_BASE OUT EXTRA_ARGS
#       export EXTRA_ARGS="--tj 0.2 --n-values 8 16 32 64 --walkers 2048 --chunks 2000 --chunk-steps 100 --msd-fit-max 0.3"
#       sbatch -t 12:00:00 -a 0-2 --mail-user=$CLU_MAIL \
#              --export=ALL,MODE=winding1d,SEED_BASE=31,OUT=$HOME/scratch/clu_kt/w1d \
#              scripts/csf3/job_gpu_kt.sh
#
#     (2e5 steps x 2048 walkers x 4 ring sizes; the A100 vmaps the walkers, so
#     walker count is nearly free and is what buys the MSD resolution.)
#     RECOMMENDATION TO THE HUB: treat (b) as an open estimator question first,
#     not a compute question. Cheapest decisive alternative is to drop MSD and
#     measure 1-D winding FIRST-PASSAGE tau directly (same estimator as the 2-D
#     arm, so the 1-D-vs-2-D contrast becomes apples-to-apples).
#
# (c) L=16 CLU<->reduced bridge (hardens the kill criterion beyond L=8) — GPU:
#       unset MODE SEED_BASE OUT EXTRA_ARGS
#       export EXTRA_ARGS="--tj-values 0.70 0.85 1.00"
#       sbatch -t 8:00:00 --mail-user=$CLU_MAIL \
#              --export=ALL,MODE=bridge,OUT=$HOME/scratch/clu_kt/bridge \
#              scripts/csf3/job_gpu_kt.sh
#     (L=8 x 3 temperatures took 145 s on laptop CPU; L=16 is ~4x the sites and
#     the reduced-XY warm start is the serial part. 8 h is ample.)
#
# (d) COLLECT: merge array shards + write summary.json/figures (cheap, CPU):
#       unset MODE SEED_BASE OUT EXTRA_ARGS
#       sbatch -p serial -G 0 -c 1 -t 0:30:00 --dependency=afterany:<JOBID> \
#              --mail-user=$CLU_MAIL \
#              --export=ALL,MODE=postproc,OUT=$HOME/scratch/clu_kt/w2d \
#              scripts/csf3/job_gpu_kt.sh
#
# Partition facts (CSF3 gpu-jobs page, mod. 2026-06-11): gpuA = A100 80GB, free
# at point of use, <=4 GPUs concurrently per user, <=12 host cores/GPU, batch
# wallclock max 4-0. `serial` = 1-core Intel, max 7-0 (partitions page).
#
#SBATCH -p gpuA              # A100 (80GB) partition; CPU modes: override -p serial -G 0
#SBATCH -G 1                 # 1 GPU (override -G 0 for winding2d/reduced/postproc)
#SBATCH -n 1                 # 1 task ...
#SBATCH -c 8                 # ... with 8 cores (<=12/GPU; ~10.4GB RAM/core)
#SBATCH -t 4:00:00           # OVERRIDE PER RUN (sbatch -t ...). Max 4-0 on gpuA.
#SBATCH -a 0-2%3             # tasks 0..2 -> seeds SEED_BASE+0..2; %3 = <=3 concurrent
#                            #   (CSF3 4-GPU/user cap; override: sbatch -a 0-N%M)
#SBATCH --job-name=clu-kt
#SBATCH -o logs/%x-%A_%a.out      # per-task stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%A_%a.err      # per-task stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # mail on end/fail; set the address at submit:
#                                 #   sbatch --mail-user=$CLU_MAIL ... (no addr in-repo)

module purge                 # no CUDA module: jax[cuda12] pip wheels bundle it
set -eo pipefail

# ---- knobs (override with sbatch --export=ALL,VAR=...) ---------------------
MODE="${MODE:-winding2d}"                 # winding1d|winding2d|bridge|reduced|postproc
# Fail LOUDLY on a bad MODE. Without this the `*)` fallthrough below silently
# treats any garbage as a CPU mode, burns the env setup + preflight, and only
# dies later in argparse. Observed 2026-07-20: a mangled paste of
# `export MODE=winding2d SEED_BASE=700` set MODE=700, and three array tasks ran
# to the preflight before failing. Cheap guard, clear message.
case "$MODE" in
  winding1d|winding2d|bridge|reduced|postproc) ;;
  *)
    echo "FATAL: MODE='$MODE' is not a valid KT mode." >&2
    echo "       Expected one of: winding1d winding2d bridge reduced postproc" >&2
    echo "       Set it with ONE export per line, then --export=ALL:" >&2
    echo "         export MODE=winding2d" >&2
    echo "         export SEED_BASE=700" >&2
    exit 2
    ;;
esac
SEED_BASE="${SEED_BASE:-700}"
case "$SEED_BASE" in
  ''|*[!0-9]*) echo "FATAL: SEED_BASE='$SEED_BASE' is not an integer." >&2; exit 2 ;;
esac
OUT="${OUT:-$HOME/scratch/clu_kt/${MODE}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"              # e.g. --l-values 32 48 64 --nwalk-2d 96
# Array task id -> seed. Non-array submissions land on task id 0 => SEED_BASE.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((SEED_BASE + TASK_ID))
# Cell-sharding (optional): set SHARD_CELLS=1 to make the array index select ONE
# cell of the mode's sweep grid instead of one seed. Use for grids too big for
# one job; the default (seed-per-task) mirrors job_gpu_array_seeds.sh.
SHARD_CELLS="${SHARD_CELLS:-0}"

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# GPU-free modes: force the CPU backend so the CUDA plugin doesn't probe/warn.
case "$MODE" in
  winding2d|reduced|postproc) export JAX_PLATFORMS=cpu ;;
esac
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

echo "=== node provenance ==="
hostname
echo "MODE=$MODE SEED=$SEED OUT=$OUT TASK_ID=$TASK_ID SHARD_CELLS=$SHARD_CELLS"
echo "EXTRA_ARGS=$EXTRA_ARGS"

case "$MODE" in
  winding1d|bridge)
    nvidia-smi || true
    echo "=== JAX GPU preflight (fail fast rather than sample on CPU) ==="
    python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY
    ;;
  *)
    echo "=== CPU mode ($MODE): numpy reduced-XY, no GPU requested ==="
    ;;
esac

echo "=== float64 + FDT preflight (handover §7.22) ==="
# Fail BEFORE the run if the physics settings are wrong. The same assertion runs
# in-process per cell; this is the cheap early copy.
python - <<'PY'
import jax
jax.config.update("jax_enable_x64", True)
from chlu.config import get_default_config
from chlu.experiments.kt.clu_path import assert_kt_settings
kt = get_default_config().experiment_kt
assert_kt_settings(kt.langevin_noise, kt.kinetic_mode, use_governor=False)
J = 2 * kt.kappa * kt.rstar ** 2
print(f"OK: float64, noise={kt.langevin_noise}, kinetic={kt.kinetic_mode}, "
      f"kappa={kt.kappa}, J={J:.4f}, T_KT={1.786 * kt.kappa * kt.rstar ** 2:.4f} "
      f"CLU units (= {1.786 / 2:.4f} J)")
PY

mkdir -p "$OUT"
SHARD_FLAG=""
if [ "$SHARD_CELLS" = "1" ]; then
  SHARD_FLAG="--task-id $TASK_ID"
  SEED_FLAG=""
else
  SEED_FLAG="--seed $SEED"
fi

echo "=== run: chlu exp-kt --mode $MODE $SEED_FLAG $SHARD_FLAG --out $OUT $EXTRA_ARGS ==="
# shellcheck disable=SC2086  # EXTRA_ARGS/flags are intentionally word-split
chlu exp-kt --mode "$MODE" $SEED_FLAG $SHARD_FLAG --out "$OUT" $EXTRA_ARGS

echo "=== artifacts ==="
ls -l "$OUT" 2>/dev/null || true
echo "Collect with: sbatch --dependency=afterany:${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} \\"
echo "  --export=ALL,MODE=postproc,OUT=$OUT scripts/csf3/job_gpu_kt.sh"
echo "Pull to laptop with: rsync -avz csf3:${OUT}/ ./kt_${MODE}/"
