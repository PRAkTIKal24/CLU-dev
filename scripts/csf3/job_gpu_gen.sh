#!/bin/bash --login
# scripts/csf3/job_gpu_gen.sh — THE launch script for the GENERATIVE PIVOT.
#
# ONE script for all four stages (encoder -> decoder -> store -> cell), sized as
# a Slurm job array over ARMS x SEEDS, one A100 per array task. It reuses the
# existing cluster idiom verbatim (env.sh, module purge, the `%N` throttle,
# push_repo.sh / rsync pull-back) — there is deliberately no second way of
# talking to CSF3.
#
# It writes into a FRESH tree that no C2/C3 run has ever touched:
#
#   $SCRATCH/CHLU-gen/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/
#       logs/              stage.log + a copy of the slurm out/err
#       artifacts/         everything the stage produces
#       run_manifest.json  git commit, jax version, EVERY non-default flag,
#                          the slurm job id -> the provenance record that makes
#                          every later number traceable.
#
# ⛔ It NEVER writes to projects/ and never to an existing workspace path.
#
# ---------------------------------------------------------------------------
# SUBMIT (from ~/scratch/CHLU on a CSF3 login node)
# ---------------------------------------------------------------------------
#   cd ~/scratch/CHLU && mkdir -p logs
#
#   # 0. print the plan: the exact `-a` range, and the path every task writes to
#   DATASET=shapes3d PRETEXT=recon STAGE=encoder PLAN=1 \
#       bash scripts/csf3/job_gpu_gen.sh
#
#   # 1. submit the stage (the -a range MUST match ARMS x SEEDS; PLAN prints it)
#   sbatch --export=ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=encoder,\
#SEEDS="0 1 2",ARMS="clu store_removed lace knn",GEN_GIT_COMMIT=$GEN_GIT_COMMIT \
#          -a 0-11%2 -t 12:00:00 scripts/csf3/job_gpu_gen.sh
#
#   # 2. the next stage, gated on the previous one (per-arm manifests are checked
#   #    in-job as well, so a half-finished upstream fails loudly, not silently):
#   sbatch --dependency=afterok:<jobid> --export=ALL,...,STAGE=decoder ...
#
#   # 3. pull the tree back to the CONSUMING SPOKE's own output dir (never a
#   #    shared one — see `Pull to laptop` at the end of this script)
#
# GEN_GIT_COMMIT: push_repo.sh rsyncs the repo WITHOUT .git, so the cluster
# cannot resolve the commit itself. Export it on the laptop before submitting:
#   export GEN_GIT_COMMIT=$(git rev-parse HEAD)
# A run refuses to start without a resolvable commit — an unprovenanced run is
# worse than no run.
#
# ---------------------------------------------------------------------------
# Partition facts (official CSF3 docs, gpu-jobs page, mod. 2026-06-11):
#   gpuA = A100 80GB, free at point of use, <=4 GPUs concurrently per user,
#   <=12 host cores per GPU (10.4GB RAM/core), batch wallclock max 4-0,
#   `-t` is MANDATORY at submit.
# CONCURRENCY defaults to 2 (not 4) so that two stages/datasets can be in
# flight at once without breaching the 4-GPU cap.
# ---------------------------------------------------------------------------
#
#SBATCH -p gpuA              # A100 (80GB) partition
#SBATCH -G 1                 # 1 GPU per array task
#SBATCH -n 1                 # 1 task ...
#SBATCH -c 8                 # ... with 8 cores per array task (<=12/GPU)
#SBATCH -t 4:00:00           # OVERRIDE PER RUN (sbatch -t ...). Max 4-0.
#SBATCH --job-name=clu-gen
#SBATCH -o logs/%x-%A_%a.out      # per-task stdout -> logs/ (dir must exist)
#SBATCH -e logs/%x-%A_%a.err      # per-task stderr -> logs/ (separate stream)
#SBATCH --mail-type=END,FAIL      # address at submit: sbatch --mail-user=$CLU_MAIL

set -eo pipefail

die() { echo "FATAL[job_gpu_gen]: $*" >&2; exit 2; }
say() { echo "[gen] $*"; }

# ===========================================================================
# 1. AXES.  DATASET / PRETEXT / STAGE have NO safe default and FAIL LOUDLY
#    when unset: this repo has been bitten by an inherited default before, and
#    a silently-defaulted axis is a silently-decided result.
#    SEEDS / ARMS / CONCURRENCY may default (the ARMS default is the COMPLETE
#    arm set, which cannot bias a comparison; a *narrowed* arm list is always
#    explicit and is recorded in the manifest).
# ===========================================================================
[ -n "${DATASET:-}" ] || die "DATASET is unset. No default: set DATASET=shapes3d|dsprites"
[ -n "${PRETEXT:-}" ] || die "PRETEXT is unset. No default: set PRETEXT=recon|cls"
[ -n "${STAGE:-}" ]   || die "STAGE is unset. No default: set STAGE=encoder|decoder|store|cell"

case "$DATASET" in
    shapes3d|dsprites) ;;
    *) die "DATASET='$DATASET' is not one of: shapes3d dsprites" ;;
esac
case "$PRETEXT" in
    # recon is the primary; cls is a diagnostic. The launcher carries the axis
    # and privileges neither.
    recon|cls) ;;
    *) die "PRETEXT='$PRETEXT' is not one of: recon cls" ;;
esac
case "$STAGE" in
    encoder|decoder|store|cell) ;;
    *) die "STAGE='$STAGE' is not one of: encoder decoder store cell" ;;
esac

DEF_SEEDS="0 1 2"
DEF_ARMS="clu store_removed lace knn"
DEF_CONCURRENCY="2"
SEEDS="${SEEDS:-$DEF_SEEDS}"
ARMS="${ARMS:-$DEF_ARMS}"
CONCURRENCY="${CONCURRENCY:-$DEF_CONCURRENCY}"

[ -n "$SEEDS" ] || die "SEEDS is empty"
[ -n "$ARMS" ]  || die "ARMS is empty"

# bash 3.2 compatible (no associative arrays, no ${var,,}).
ARM_ARR=(); for a in $ARMS; do
    case "$a" in
        clu|store_removed|lace|knn) ;;
        *) die "ARM '$a' is not one of: clu store_removed lace knn" ;;
    esac
    ARM_ARR[${#ARM_ARR[@]}]="$a"
done
SEED_ARR=(); for s in $SEEDS; do
    case "$s" in
        ''|*[!0-9]*) die "SEED '$s' is not a non-negative integer" ;;
    esac
    SEED_ARR[${#SEED_ARR[@]}]="$s"
done
case "$CONCURRENCY" in
    ''|*[!0-9]*) die "CONCURRENCY='$CONCURRENCY' is not a positive integer" ;;
esac
[ "$CONCURRENCY" -ge 1 ] || die "CONCURRENCY must be >= 1"
[ "$CONCURRENCY" -le 4 ] || die "CONCURRENCY='$CONCURRENCY' exceeds the CSF3 free-tier 4-GPU cap"

N_ARMS=${#ARM_ARR[@]}
N_SEEDS=${#SEED_ARR[@]}
N_TASKS=$((N_ARMS * N_SEEDS))

# ===========================================================================
# 2. FRESH TREE.  Nothing here is inherited from C2/C3.
# ===========================================================================
GEN_SCRATCH="${SCRATCH:-$HOME/scratch}"
GEN_ROOT="${GEN_ROOT:-$GEN_SCRATCH/CHLU-gen}"
WORKSPACE="$GEN_ROOT/workspaces/${DATASET}_${PRETEXT}"

run_dir_for() {  # $1=arm $2=seed
    echo "$WORKSPACE/$STAGE/${1}_s${2}"
}

# ===========================================================================
# 3. PLAN mode: print the submit line and the whole index -> path map, exit.
# ===========================================================================
if [ "${PLAN:-0}" = "1" ]; then
    echo "=== gen launch plan: ${DATASET}_${PRETEXT} stage=${STAGE} ==="
    echo "arms  (${N_ARMS}): ${ARM_ARR[*]}"
    echo "seeds (${N_SEEDS}): ${SEED_ARR[*]}"
    echo "tasks: ${N_TASKS}   throttle: %${CONCURRENCY}"
    echo "sbatch --export=ALL,DATASET=${DATASET},PRETEXT=${PRETEXT},STAGE=${STAGE},ARMS=\"${ARM_ARR[*]}\",SEEDS=\"${SEED_ARR[*]}\",GEN_GIT_COMMIT=\$(git rev-parse HEAD) -a 0-$((N_TASKS - 1))%${CONCURRENCY} -t <WALLCLOCK> scripts/csf3/job_gpu_gen.sh"
    i=0
    while [ "$i" -lt "$N_TASKS" ]; do
        echo "task ${i}: arm=${ARM_ARR[$((i / N_SEEDS))]} seed=${SEED_ARR[$((i % N_SEEDS))]} -> $(run_dir_for "${ARM_ARR[$((i / N_SEEDS))]}" "${SEED_ARR[$((i % N_SEEDS))]}")"
        i=$((i + 1))
    done
    exit 0
fi

# ===========================================================================
# 4. Which cell of the grid am I?
# ===========================================================================
TASK_ID="${SLURM_ARRAY_TASK_ID:-${GEN_TASK_ID:-}}"
[ -n "$TASK_ID" ] || die "not an array task. Submit with 'sbatch -a 0-$((N_TASKS - 1))%${CONCURRENCY}' (PLAN=1 prints the line), or set GEN_TASK_ID=<i> for a local dry run"
case "$TASK_ID" in
    ''|*[!0-9]*) die "task id '$TASK_ID' is not a non-negative integer" ;;
esac
[ "$TASK_ID" -lt "$N_TASKS" ] || die "array task $TASK_ID is outside the grid (ARMS x SEEDS = $N_TASKS). The -a range does not match the axes."

# A short array means arms or seeds are silently missing from the sweep. That is
# a legitimate thing to do (resubmitting one failed cell) but never by accident.
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" != "$N_TASKS" ] && [ "${PARTIAL:-0}" != "1" ]; then
    die "array has $SLURM_ARRAY_TASK_COUNT tasks but ARMS x SEEDS = $N_TASKS — cells would be silently missing. Fix -a, or set PARTIAL=1 if this is a deliberate re-run."
fi

ARM="${ARM_ARR[$((TASK_ID / N_SEEDS))]}"
SEED="${SEED_ARR[$((TASK_ID % N_SEEDS))]}"
RUN_DIR="$(run_dir_for "$ARM" "$SEED")"

# Hard guard: the gen pivot never writes into an inherited workspace.
case "$RUN_DIR" in
    */CHLU-gen/workspaces/*) ;;
    *) die "refusing to run: RUN_DIR='$RUN_DIR' is not under a CHLU-gen workspace" ;;
esac
case "$RUN_DIR" in
    */projects/*) die "refusing to run: RUN_DIR='$RUN_DIR' points into projects/" ;;
esac

# ===========================================================================
# 5. Cluster environment (skipped for a local plumbing dry run).
# ===========================================================================
GEN_LOCAL="${GEN_LOCAL:-0}"
if [ "$GEN_LOCAL" = "1" ]; then
    export CLU_REPO="${CLU_REPO:-$PWD}"
    say "GEN_LOCAL=1 — skipping module purge / env.sh / GPU preflight (plumbing only)"
else
    if command -v module >/dev/null 2>&1; then module purge; fi
    export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
    # shellcheck disable=SC1091
    source "$CLU_REPO/scripts/csf3/env.sh"
fi
PY_BIN="${GEN_PYTHON:-python3}"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN'"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/artifacts"
MANIFEST="$RUN_DIR/run_manifest.json"
STAGE_LOG="$RUN_DIR/logs/stage.log"

# ===========================================================================
# 6. Provenance: git commit (mandatory), jax version, non-default flags.
# ===========================================================================
GIT_COMMIT="${GEN_GIT_COMMIT:-}"
if [ -z "$GIT_COMMIT" ] && [ -e "$CLU_REPO/.git" ] && command -v git >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$CLU_REPO" rev-parse HEAD 2>/dev/null || true)"
fi
if [ -z "$GIT_COMMIT" ] && [ -f "$CLU_REPO/GIT_COMMIT" ]; then
    GIT_COMMIT="$(cat "$CLU_REPO/GIT_COMMIT")"
fi
[ -n "$GIT_COMMIT" ] || die "cannot resolve the git commit (push_repo.sh excludes .git). Submit with --export=ALL,...,GEN_GIT_COMMIT=\$(git rev-parse HEAD). A run without provenance is not worth its wallclock."

# Every knob, and which of them are non-default. DATASET/PRETEXT/STAGE have no
# default at all, so they are always recorded explicitly as axes.
NON_DEFAULT=""
note_flag() {  # name value default
    if [ "$2" != "$3" ]; then NON_DEFAULT="${NON_DEFAULT}${1}=${2}
"; fi
}
note_flag SEEDS       "$SEEDS"                 "$DEF_SEEDS"
note_flag ARMS        "$ARMS"                  "$DEF_ARMS"
note_flag CONCURRENCY "$CONCURRENCY"           "$DEF_CONCURRENCY"
note_flag GEN_ROOT    "$GEN_ROOT"              "$GEN_SCRATCH/CHLU-gen"
note_flag SMOKE       "${SMOKE:-0}"            "0"
note_flag PARTIAL     "${PARTIAL:-0}"          "0"
note_flag GEN_LOCAL   "$GEN_LOCAL"             "0"
note_flag ALLOW_MISSING_UPSTREAM "${ALLOW_MISSING_UPSTREAM:-0}" "0"
note_flag EXTRA_ARGS  "${EXTRA_ARGS:-}"        ""
note_flag PAYLOAD_MODULE "${PAYLOAD_MODULE:-}" ""

# Default payload entry point per stage (overridable with PAYLOAD_MODULE).
case "$STAGE" in
    encoder) DEF_MODULE="chlu.experiments.exp_gen_encoder" ;;
    decoder) DEF_MODULE="chlu.experiments.exp_gen_decoder" ;;
    store)   DEF_MODULE="chlu.experiments.exp_gen_store" ;;
    cell)    DEF_MODULE="chlu.experiments.exp_gen_cell" ;;
esac
MODULE="${PAYLOAD_MODULE:-$DEF_MODULE}"

write_manifest() {  # $1=status $2=exit_code
    GEN_MF_STATUS="$1" GEN_MF_RC="$2" GEN_MF_PATH="$MANIFEST" \
    GEN_MF_DATASET="$DATASET" GEN_MF_PRETEXT="$PRETEXT" GEN_MF_STAGE="$STAGE" \
    GEN_MF_ARM="$ARM" GEN_MF_SEED="$SEED" GEN_MF_ARMS="$ARMS" GEN_MF_SEEDS="$SEEDS" \
    GEN_MF_CONCURRENCY="$CONCURRENCY" GEN_MF_COMMIT="$GIT_COMMIT" \
    GEN_MF_NONDEFAULT="$NON_DEFAULT" GEN_MF_ROOT="$GEN_ROOT" \
    GEN_MF_WORKSPACE="$WORKSPACE" GEN_MF_RUNDIR="$RUN_DIR" \
    GEN_MF_MODULE="$MODULE" GEN_MF_EXTRA="${EXTRA_ARGS:-}" \
    GEN_MF_SMOKE="${SMOKE:-0}" GEN_MF_STARTED="$STARTED_AT" \
    GEN_MF_TASKID="$TASK_ID" GEN_MF_NTASKS="$N_TASKS" \
    "$PY_BIN" - <<'PY'
import datetime as _dt, json, os, platform, socket

def _v(pkg):
    try:
        # importlib.metadata: reports the RESOLVED version without paying the
        # (very slow) cost of importing jax itself.
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None

e = os.environ.get
nd = [l for l in e("GEN_MF_NONDEFAULT", "").splitlines() if l.strip()]
now = _dt.datetime.now(_dt.timezone.utc).isoformat()
m = {
    "schema": "chlu-gen-run-manifest/1",
    "status": e("GEN_MF_STATUS"),
    "exit_code": int(e("GEN_MF_RC", "0")),
    "started_at": e("GEN_MF_STARTED"),
    "finished_at": now if e("GEN_MF_STATUS") != "running" else None,
    "axes": {
        "dataset": e("GEN_MF_DATASET"),
        "pretext": e("GEN_MF_PRETEXT"),
        "stage": e("GEN_MF_STAGE"),
        "arm": e("GEN_MF_ARM"),
        "seed": int(e("GEN_MF_SEED")),
    },
    "grid": {
        "arms": e("GEN_MF_ARMS", "").split(),
        "seeds": [int(s) for s in e("GEN_MF_SEEDS", "").split()],
        "concurrency": int(e("GEN_MF_CONCURRENCY", "0")),
        "array_task_id": int(e("GEN_MF_TASKID", "0")),
        "n_tasks": int(e("GEN_MF_NTASKS", "0")),
    },
    "git_commit": e("GEN_MF_COMMIT"),
    "versions": {
        "python": platform.python_version(),
        "jax": _v("jax"),
        "jaxlib": _v("jaxlib"),
        "equinox": _v("equinox"),
        "optax": _v("optax"),
        "chlu": _v("chlu"),
    },
    "non_default_flags": nd,
    "payload": {
        "module": e("GEN_MF_MODULE"),
        "extra_args": e("GEN_MF_EXTRA"),
        "smoke": e("GEN_MF_SMOKE") == "1",
    },
    "slurm": {
        "job_id": e("SLURM_JOB_ID"),
        "array_job_id": e("SLURM_ARRAY_JOB_ID"),
        "array_task_id": e("SLURM_ARRAY_TASK_ID"),
        "array_task_count": e("SLURM_ARRAY_TASK_COUNT"),
        "partition": e("SLURM_JOB_PARTITION"),
        "nodelist": e("SLURM_JOB_NODELIST"),
        "gpus": e("SLURM_GPUS"),
        "cuda_visible_devices": e("CUDA_VISIBLE_DEVICES"),
    },
    "paths": {
        "gen_root": e("GEN_MF_ROOT"),
        "workspace": e("GEN_MF_WORKSPACE"),
        "run_dir": e("GEN_MF_RUNDIR"),
        "artifacts": os.path.join(e("GEN_MF_RUNDIR"), "artifacts"),
        "logs": os.path.join(e("GEN_MF_RUNDIR"), "logs"),
    },
    "host": socket.gethostname(),
}
with open(e("GEN_MF_PATH"), "w") as fh:
    json.dump(m, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("[gen] manifest -> " + e("GEN_MF_PATH") + " status=" + str(m["status"]))
PY
}

STARTED_AT="$("$PY_BIN" -c 'import datetime;print(datetime.datetime.now(datetime.timezone.utc).isoformat())')"

on_exit() {
    rc=$?
    if [ "$rc" -eq 0 ]; then write_manifest ok 0; else write_manifest failed "$rc"; fi
    # keep the slurm streams with the run they belong to
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        cp "$CLU_REPO/logs/${SLURM_JOB_NAME:-clu-gen}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}."* \
           "$RUN_DIR/logs/" 2>/dev/null || true
    fi
    return $rc
}
trap on_exit EXIT

say "=== ${DATASET}_${PRETEXT} / ${STAGE} / ${ARM}_s${SEED}  (task ${TASK_ID}/${N_TASKS}) ==="
say "run_dir=$RUN_DIR"
say "commit=$GIT_COMMIT  module=$MODULE  smoke=${SMOKE:-0}"
write_manifest running 0

# ===========================================================================
# 7. Stage dependency: encoder -> decoder -> store -> cell, per arm and seed.
#    The stages are submitted independently; this refuses to run on top of an
#    upstream stage that never finished.
# ===========================================================================
case "$STAGE" in
    encoder) UPSTREAM="" ;;
    decoder) UPSTREAM="encoder" ;;
    store)   UPSTREAM="decoder" ;;
    cell)    UPSTREAM="store" ;;
esac
if [ -n "$UPSTREAM" ] && [ "${ALLOW_MISSING_UPSTREAM:-0}" != "1" ]; then
    UP_MF="$WORKSPACE/$UPSTREAM/${ARM}_s${SEED}/run_manifest.json"
    [ -f "$UP_MF" ] || die "upstream stage '$UPSTREAM' has no manifest at $UP_MF — run it first (or set ALLOW_MISSING_UPSTREAM=1)"
    "$PY_BIN" -c "import json,sys;m=json.load(open(sys.argv[1]));sys.exit(0 if m.get('status')=='ok' else 1)" "$UP_MF" \
        || die "upstream stage '$UPSTREAM' did not finish ok (see $UP_MF)"
    say "upstream ok: $UP_MF"
fi

# ===========================================================================
# 7b. Payload entry point must EXIST before any wallclock is spent on it — a
#     stage whose module is not written yet must abort, never quietly no-op.
#     (Checked here, outside the tee'd payload pipeline, so the message lands
#     on stderr and in the slurm .err stream.)
# ===========================================================================
if [ "${SMOKE:-0}" != "1" ]; then
    "$PY_BIN" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$MODULE') else 1)" 2>/dev/null \
        || die "stage module '$MODULE' does not exist yet. Override with PAYLOAD_MODULE=..., or run SMOKE=1 for a plumbing-only run."
    say "payload module ok: $MODULE"
fi

# ===========================================================================
# 8. Node / GPU provenance + preflight (skipped for a local dry run).
# ===========================================================================
if [ "$GEN_LOCAL" != "1" ]; then
    say "=== node/GPU provenance ==="
    hostname
    nvidia-smi || true
    echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    if [ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]; then
        say "=== JAX GPU preflight (fail fast rather than train on CPU) ==="
        "$PY_BIN" - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY
    fi
fi

# ===========================================================================
# 9. Payload.  SMOKE=1 substitutes a trivial one: this script's deliverable is
#    the PATH PLUMBING AND THE MANIFEST, and those must be testable before any
#    stage module exists.
# ===========================================================================
run_payload() {
    if [ "${SMOKE:-0}" = "1" ]; then
        say "SMOKE=1 — trivial payload (plumbing only, no science)"
        GEN_SMOKE_OUT="$RUN_DIR/artifacts" GEN_SMOKE_ARM="$ARM" GEN_SMOKE_SEED="$SEED" \
        GEN_SMOKE_DATASET="$DATASET" GEN_SMOKE_PRETEXT="$PRETEXT" GEN_SMOKE_STAGE="$STAGE" \
        "$PY_BIN" - <<'PY'
import json, os
e = os.environ.get
out = e("GEN_SMOKE_OUT")
os.makedirs(out, exist_ok=True)
rec = {k: e("GEN_SMOKE_" + k.upper()) for k in
       ("dataset", "pretext", "stage", "arm", "seed")}
rec["payload"] = "smoke"
with open(os.path.join(out, "smoke_artifact.json"), "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("[gen] smoke artifact ->", os.path.join(out, "smoke_artifact.json"))
PY
    else
        # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
        "$PY_BIN" -u -m "$MODULE" \
            --dataset "$DATASET" --pretext "$PRETEXT" --stage "$STAGE" \
            --arm "$ARM" --seed "$SEED" --out "$RUN_DIR/artifacts" ${EXTRA_ARGS:-}
    fi
}
run_payload 2>&1 | tee -a "$STAGE_LOG"

say "=== artifacts ==="
ls -l "$RUN_DIR/artifacts" || true
say "Pull to the CONSUMING SPOKE's own output dir (never a shared one), e.g.:"
say "  rsync -av csf3:$RUN_DIR/ ./.claude/outputs/<your-slug>/${DATASET}_${PRETEXT}/${STAGE}/${ARM}_s${SEED}/"
