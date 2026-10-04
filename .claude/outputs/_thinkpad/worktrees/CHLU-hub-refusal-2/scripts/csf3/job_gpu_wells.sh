#!/bin/bash --login
# scripts/csf3/job_gpu_wells.sh — ⭐ ROUTE B: learn WHERE THE WELLS GO, on one A100.
#
# ⭐ THE FIRST CSF3 LONG JOB for the gen phase, and the first one in this directory that
# can survive its own death: `job_gpu_single.sh`, `job_gpu_array_seeds.sh` and
# `job_gpu_eval.sh` contain ZERO checkpoint or resume logic (0 matches for
# resume|checkpoint|ckpt|requeue|restart).  ⛔ This one does not inherit that.
#
# ⛔⛔ A run that dies at hour 70 of 96 MUST RESUME, never restart from scratch.  OOM, a
# node failure, a wallclock kill and a lost final write are all the same event to that
# requirement.  The machinery is `chlu.core.gen_wells` (time-based checkpoints, atomic
# temp+rename writes, last-N retention, a config-fingerprint refusal and an instrumented
# recovery) and it has been demonstrated by `kill -9` — see
# `.claude/outputs/gen-wells-route-b.md`.
#
# Submit (from ~/scratch/CHLU on the login node):
#   sbatch -t 2-0 scripts/csf3/job_gpu_wells.sh                       # seed 0, scored
#   sbatch -t 2-0 --export=ALL,RESUME=1 scripts/csf3/job_gpu_wells.sh # ⭐ resume it
#   sbatch -t 0-4 --export=ALL,QUICK=1,SCORED=0 scripts/csf3/job_gpu_wells.sh   # plumbing
#
# Partition facts (official CSF3 docs, gpu-jobs page, mod. 2026-06-11):
#   gpuA = A100 SXM4 80GB, FREE AT POINT OF USE, <=4 GPUs concurrently per user,
#   <=12 host cores per GPU (~10.4GB RAM/core), batch wallclock max 4-0, -G N required.
# ⛔ Never `-p serial` for this job.
#
#SBATCH -p gpuA                   # ⛔ never `serial`
#SBATCH -G 1                      # required on gpuA
#SBATCH -n 1
#SBATCH -c 8                      # <=12 cores/GPU
#SBATCH -t 2-0                    # OVERRIDE PER RUN. Max 4-0.
#SBATCH --job-name=clu-wells-routeb
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=append        # ⭐ a resumed job appends to its own log
#SBATCH --requeue                 # ⭐ let Slurm restart it; the checkpoint makes that cheap
#SBATCH --signal=B:USR1@300       # ⭐ 5 min warning before the wallclock kill

module purge                      # no CUDA module: jax[cuda12] wheels bundle CUDA/cuDNN
set -eo pipefail

# ---- knobs (override with sbatch --export=ALL,VAR=...) ----------------------
SEEDS="${SEEDS:-0}"               # ⚠ ONLY seed 0 is cached at scope="full" — see below
ARMS="${ARMS:-random_dmatched correlation learned_label_free learned_weak_sup}"
STEPS="${STEPS:-}"                # placement steps; empty = the config default
RESUME="${RESUME:-0}"
QUICK="${QUICK:-0}"
SCORED="${SCORED:-1}"
ALLOW_REENCODE="${ALLOW_REENCODE:-0}"

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"   # ⭐ exports + ASSERTS the five CHLU_* roots

# ---- ⛔ checkpoints live on ~/scratch --------------------------------------
# ⛔ NEVER $TMPDIR: it is node-local NVMe and is WIPED AT JOB END, so a checkpoint there
#    protects nothing — which is the exact opposite of the point.
# ⛔ NEVER $HOME: small, quota'd and backed up; this writes every few minutes.
# ⚠ ~/scratch has NO BACKUP and a 3-month auto-cleanup: pull results promptly.
CKPT_DIR="${CKPT_DIR:-$HOME/scratch/CHLU/ckpt/gen-wells-route-b}"
OUT_DIR="${OUT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-wells-route-b}"
mkdir -p "$CKPT_DIR" "$OUT_DIR" logs

case "$CKPT_DIR" in
    "${TMPDIR:-/nonexistent-tmpdir}"*)
        echo "FATAL: CKPT_DIR is under \$TMPDIR, which is WIPED AT JOB END."; exit 1 ;;
esac

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
# ⭐ GEN_GIT_COMMIT is exported ON THE LOGIN NODE by push_repo.sh (`git` on the compute
# node sees the rsync'd tree, which is not a checkout).  ⛔ `git pull` on the cluster
# serves a STALE fork — push_repo.sh is the only path.
echo "GEN_GIT_COMMIT=${GEN_GIT_COMMIT:-<unset: export it on the login node>}"
echo "CKPT_DIR=$CKPT_DIR"
echo "OUT_DIR=$OUT_DIR"

echo "=== JAX GPU preflight (fail fast rather than train on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

# ---- ⛔ EVERY PRECONDITION FAILS HERE OR NOT AT ALL -------------------------
# ⚠ MEASURED, not assumed: seed 0's feature cache holds 480,000 rows (scope="full",
#   1.83 GiB of the 1.95 GiB staged) while seeds 1 and 2 hold 9,288 each (scope="used"),
#   and `encoder_param_hash` DIFFERS per seed — they are not interchangeable.  A run on
#   seed 1 or 2 that needs more rows falls into the gather-with-extension re-encode path,
#   which gen-dynamical-read flagged UNTESTED.  ⛔ The harness's own coverage assertion
#   names the seed and the shortfall and dies; this is the same gate, run BEFORE the
#   expensive imports so the failure is instant.
echo "=== artefact + cache preflight (⛔ before any GPU work) ==="
python - <<'PY'
import json, os, sys
from pathlib import Path
root = Path(os.environ["CHLU_CLAUDE_ROOT"])
seeds = [int(s) for s in os.environ.get("SEEDS", "0").split()]
need = int(os.environ.get("N_STORE_TRAIN", "32768"))
bad = []
for s in seeds:
    man = (root / "outputs/gen-feature-bridge/runs/shapes3d_recon"
           / f"seed_{s}" / "feature_cache.json")
    if not man.exists():
        bad.append(f"seed {s}: no feature_cache.json at {man}")
        continue
    m = json.loads(man.read_text())
    print(f"  seed {s}: {m['n_rows']} rows, scope={m['scope']!r}, dtype={m['dtype']}, "
          f"D={m['D']}, encoder {m['encoder_param_hash'][:8]}…")
    if int(m["n_rows"]) < need:
        bad.append(
            f"⛔ SEED {s} CACHE SHORTFALL: holds {m['n_rows']} rows (scope={m['scope']!r}) "
            f"but this run needs {need}. Seed 0 is the only seed at scope='full'. "
            "Run seed 0, lower N_STORE_TRAIN, or set ALLOW_REENCODE=1 to take the "
            "UNTESTED re-encode path deliberately.")
for p in ("outputs/gen-store-grouping/runs/shapes3d_recon/store_plan.json",
          "outputs/gen-decoder-lift/winner/winner_hashes.json"):
    if not (root / p).exists():
        bad.append(f"⛔ missing required artefact: {root / p}")
# ⚠ The dataset itself: build_split reads labels from 3dshapes.h5 and the cache
# re-encode check streams images from it.  It is NOT in the staged set as of
# gen-cluster-stage-artefacts' STAGED.json — this names it at job start rather than
# letting the loader discover it after the scheduler has allocated the GPU.
h5 = Path(os.environ["CHLU_GEN_DATA_ROOT"]) / "shapes3d" / "3dshapes.h5"
if not h5.exists():
    bad.append(f"⛔ missing dataset: {h5} (needed for factor labels and the cache "
               "re-encode check; ⛔ do not improvise a staging path — "
               "gen-cluster-stage-artefacts owns that file set)")
if bad and os.environ.get("ALLOW_REENCODE") != "1":
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("  ⭐ preflight OK")
PY

# ---- ⭐ the wallclock-kill handler -----------------------------------------
# On SIGUSR1 (5 min before the kill) we do NOT try to write a special checkpoint: the
# periodic one is already on disk and atomic.  We just say so loudly, so the log shows
# the resume command rather than leaving the Head to work it out.
_on_term() {
    echo "⚠ [signal] wallclock/preemption signal received at $(date -u +%FT%TZ)."
    echo "  ⭐ The last GOOD checkpoint is in $CKPT_DIR (atomic temp+rename, last-N kept)."
    echo "  ⭐ RESUME WITH:  sbatch -t 2-0 --export=ALL,RESUME=1 scripts/csf3/job_gpu_wells.sh"
    exit 0
}
trap _on_term USR1 TERM

# ---- build the command ------------------------------------------------------
ARGS=(--seeds $SEEDS --arms $ARMS --out "$OUT_DIR"
      --ckpt-dir "$CKPT_DIR" --ckpt-every-seconds "${CKPT_EVERY_SECONDS:-300}")
[ -n "$STEPS" ]            && ARGS+=(--steps "$STEPS")
[ "$RESUME" = "1" ]        && ARGS+=(--resume)
[ "$QUICK" = "1" ]         && ARGS+=(--quick)
[ "$SCORED" = "1" ] && [ "$QUICK" != "1" ] && ARGS+=(--scored)
[ "$ALLOW_REENCODE" = "1" ] && ARGS+=(--allow-reencode)

echo "=== run: python -m chlu.experiments.exp_gen_wells_route_b ${ARGS[*]} ==="
python -m chlu.experiments.exp_gen_wells_route_b "${ARGS[@]}" &
wait $!

echo "=== artefacts ==="
ls -R "$OUT_DIR/runs" 2>/dev/null | head -40 || true
echo "checkpoints kept:"; ls -1 "$CKPT_DIR"/*/ 2>/dev/null | tail -10 || true
echo "⚠ ~/scratch has NO BACKUP and a 3-month auto-cleanup — pull results promptly."
echo "=== gen-wells-route-b job ${SLURM_JOB_ID:-?} DONE ==="
