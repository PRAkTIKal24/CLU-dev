#!/bin/bash --login
# scripts/csf3/job_cpu_gen.sh — ⭐ THE CPU jobscript for the gen sweeps, on `multicore`
# under the `fatpou01` account.
#
# ⭐ WHY THIS EXISTS.  The gen CPU legs (`exp_gen_atom_mean`, `exp_gen_baselines`) are
# PURE CPU work — both ran `JAX_PLATFORMS=cpu` — and they were being run on the laptop,
# where a four-hour `exp_gen_atom_mean` sweep held a 16 GB machine at 570 % CPU all
# evening and BLOCKED the scored cell arm, because both want the same eight cores.
# Meanwhile:
#
#     association   GrpTRES      MaxTRESPU
#     fatpou01      cpu=1500     ⭐ cpu=84 per user      <- uncontended, NO GPU
#     gpu-free      —            a100_80g=4, l40s=4, gpu=6
#
# `multicore` just gained 1,680 cores in the CSF3/CSF4 merger while BOTH GPU partitions
# sit at their group caps (gpua 76/76, gpul 84/84) and our GPU jobs are on the
# free/scavenger QOS at priority 5759.  ⇒ 84 cores, queued in minutes, instead of 8
# laptop cores blocking the headline.
#
# ⛔⛔ NEVER submit this under `-A gpu-free`.  That is the GPU association; it carries NO
# CPU grant, so the job is rejected or scavenges and the entire point — an UNCONTENDED
# entitlement — is lost.  §2 below refuses it at runtime, by name.
# ⛔⛔ NEVER ask for more than 84 cores.  `MaxTRESPU cpu=84` is a per-user cap: -c 96
# does not run slowly, it PENDS FOREVER on `AssocGrpCpuLimit`, exactly as our GPU jobs
# are pending on `AssocGrpGRES` right now.  §2 refuses that too, naming the limit.
# ⛔ NEVER `-p serial` for this: `serial` is a ONE-CORE partition.
#
# ---------------------------------------------------------------------------
# SUBMIT (from ~/scratch/CHLU on a CSF3 login node; `mkdir -p logs` first)
# ---------------------------------------------------------------------------
#   export GEN_GIT_COMMIT=$(git rev-parse HEAD)          # on the LAPTOP, before push_repo.sh
#
#   # print the plan (units, per-unit command, the submit line) and exit:
#   EXP=atom_mean DATASET=shapes3d PRETEXT=recon PLAN=1 bash scripts/csf3/job_cpu_gen.sh
#
#   sbatch -c 32 -t 1-0 \
#     --export="ALL,EXP=atom_mean,DATASET=shapes3d,PRETEXT=recon,SEEDS=0 1 2,GEN_GIT_COMMIT=$GEN_GIT_COMMIT" \
#     scripts/csf3/job_cpu_gen.sh
#
#   # resume after ANY death (wallclock, requeue, node failure, kill): the SAME line.
#   # RESUME defaults to 1 — completed units are skipped from the journal, never redone.
#
# ---------------------------------------------------------------------------
# ⛔⛔ CHECKPOINT AND RESUME — the standing Head rule.  Two layers, and both matter:
#
#   LAYER 1 (this script, the UNIT LEDGER).  A sweep is many units; the unit here is a
#     SEED.  After each unit exits 0 we record it in an atomically-rewritten journal
#     (tmp + mv) under $CKPT_DIR, with time-based rotated snapshots (last $KEEP_LAST
#     kept).  On resume a unit is skipped when the journal holds it AND its manifest is
#     on disk — the `SKIP … (manifest exists)` lines.  ⇒ ⭐ a kill costs ONE UNIT,
#     never the run.  The journal carries a CONFIG FINGERPRINT and a resume whose
#     fingerprint differs is REFUSED by name (the `pilot-checkpoint-resume` pattern —
#     ⛔ there is deliberately no second resume mechanism in this repo).
#
#   LAYER 2 (inside the payload).  `exp_gen_atom_mean.run_seed` reads its own
#     `atom_manifest.json` and skips budgets already on disk ("[tag] already on disk for
#     this seed — skipped"), so re-entering a HALF-DONE unit is cheap too.
#
#   ⛔ Checkpoints live on ~/scratch.  NEVER $TMPDIR (node-local NVMe, WIPED AT JOB END —
#   a checkpoint there protects nothing).  NEVER $HOME (small, quota'd, backed up).
#   ⚠ ~/scratch has NO BACKUP and a 3-month auto-cleanup: pull results promptly.
# ---------------------------------------------------------------------------
#
# Partition facts [DOCS, /csf3/batch-slurm/partitions/, mod. 2026-04-15]:
#   multicore = AMD Genoa, 2–168 cores per job, wallclock max 7-0, `-t` MANDATORY.
#   serial    = 1 core.  ⛔ not for this.
#
#SBATCH -p multicore              # ⛔ never `serial` (1 core), never a gpu partition
#SBATCH -A fatpou01               # ⛔⛔ never `gpu-free` — that association has NO CPU grant
#SBATCH -n 1
#SBATCH -c 16                     # OVERRIDE PER RUN (sbatch -c N).  ⛔ HARD CAP 84.
#SBATCH -t 1-0                    # OVERRIDE PER RUN (sbatch -t ...).  Max 7-0.
#SBATCH --job-name=clu-cpu-gen
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=append        # ⭐ a resumed job appends to its own log
#SBATCH --requeue                 # ⭐ let Slurm restart it; the unit ledger makes that cheap
#SBATCH --signal=B:USR1@300       # ⭐ 5 min warning before the wallclock kill

set -eo pipefail

die() { echo "FATAL[job_cpu_gen]: $*" >&2; exit 2; }
say() { echo "[cpu-gen] $*"; }

# ===========================================================================
# 0. Knobs.  EXP / DATASET / PRETEXT are AXES and have NO safe default: this
#    repo has been bitten by an inherited default before, and a silently
#    defaulted axis is a silently decided result (job_gpu_gen.sh precedent).
# ===========================================================================
DEF_SEEDS="0 1 2"
DEF_ARM="all"
DEF_BUDGETS=""
DEF_CKPT_EVERY_SECONDS="300"
DEF_KEEP_LAST="5"

EXP="${EXP:-}"
DATASET="${DATASET:-}"
PRETEXT="${PRETEXT:-}"
SEEDS="${SEEDS:-$DEF_SEEDS}"
ARM="${ARM:-$DEF_ARM}"                 # exp_gen_baselines only: knn | lace | all
BUDGETS="${BUDGETS:-$DEF_BUDGETS}"     # exp_gen_atom_mean only; empty = the config's full set
QUICK="${QUICK:-0}"
RESUME="${RESUME:-1}"                  # ⭐ ON by default: resuming a sweep is never wrong
FORCE_NEW="${FORCE_NEW:-0}"            # archive a fingerprint-mismatched journal and start over
AGGREGATE="${AGGREGATE:-1}"            # run the module's aggregate stage after the units
EXTRA_ARGS="${EXTRA_ARGS:-}"
CKPT_EVERY_SECONDS="${CKPT_EVERY_SECONDS:-$DEF_CKPT_EVERY_SECONDS}"
KEEP_LAST="${KEEP_LAST:-$DEF_KEEP_LAST}"
GEN_LOCAL="${GEN_LOCAL:-0}"            # skip module purge / env.sh (local plumbing only)
DRY="${DRY:-0}"                        # print the payload command per unit, run nothing
SMOKE="${SMOKE:-0}"                    # trivial payload: plumbing/resume only, ⛔ NO SCIENCE
SMOKE_SLEEP_S="${SMOKE_SLEEP_S:-0}"    # makes a SMOKE unit killable at a known point
# ⭐ FAULT INJECTION, SMOKE ONLY.  The two branches below ("failed" and "exited 0 but
# wrote nothing") are the ones that keep a silent hole out of a finished table, and an
# untested resume branch has never worked in this repo.  These make them reachable from
# tests/test_job_cpu_gen.py.  ⛔ They do nothing unless SMOKE=1.
SMOKE_FAIL_SEEDS="${SMOKE_FAIL_SEEDS:-}"    # these seeds' smoke payload exits 1
SMOKE_EMPTY_SEEDS="${SMOKE_EMPTY_SEEDS:-}"  # these seeds exit 0 and write NO manifest
PLAN="${PLAN:-0}"
SKIP_JAX_PREFLIGHT="${SKIP_JAX_PREFLIGHT:-0}"

[ -n "$EXP" ]     || die "EXP is unset. No default: set EXP=atom_mean|baselines"
[ -n "$DATASET" ] || die "DATASET is unset. No default: set DATASET=shapes3d|dsprites"
[ -n "$PRETEXT" ] || die "PRETEXT is unset. No default: set PRETEXT=recon|cls"

case "$EXP" in
    atom_mean) MODULE="chlu.experiments.exp_gen_atom_mean"; UNIT_STAGE="atoms"
               MANIFEST_NAME="atom_manifest.json" ;;
    baselines) MODULE="chlu.experiments.exp_gen_baselines"; UNIT_STAGE="baselines"
               MANIFEST_NAME="baselines_manifest.json" ;;
    *) die "EXP='$EXP' is not one of: atom_mean baselines.  ⛔ Do not build a second
       launcher — add the module here, with its unit stage and its manifest name." ;;
esac
case "$DATASET" in shapes3d|dsprites) ;; *) die "DATASET='$DATASET' is not one of: shapes3d dsprites" ;; esac
case "$PRETEXT" in recon|cls) ;;        *) die "PRETEXT='$PRETEXT' is not one of: recon cls" ;; esac
case "$ARM"     in knn|lace|all) ;;     *) die "ARM='$ARM' is not one of: knn lace all" ;; esac

SEED_ARR=(); for s in $SEEDS; do
    case "$s" in ''|*[!0-9]*) die "SEED '$s' is not a non-negative integer" ;; esac
    SEED_ARR[${#SEED_ARR[@]}]="$s"
done
[ ${#SEED_ARR[@]} -gt 0 ] || die "SEEDS is empty — there are no units to run"

# ===========================================================================
# 1. ⛔ THE CORE CAP.  `fatpou01` is MaxTRESPU cpu=84 per user.  Over that the job
#    does not run slowly — it PENDS FOREVER on AssocGrpCpuLimit.  Refuse at
#    submit-time semantics (here), naming the limit, rather than let the Head
#    discover it in `squeue` three days later.
# ===========================================================================
MAX_CORES=84
CORES="${SLURM_CPUS_PER_TASK:-${CORES:-16}}"
case "$CORES" in
    ''|*[!0-9]*) die "core count '$CORES' is not a positive integer (-c N / CORES=N)" ;;
esac
[ "$CORES" -ge 1 ] || die "core count '$CORES' must be >= 1"
if [ "$CORES" -gt "$MAX_CORES" ]; then
    die "-c $CORES exceeds the per-user CPU cap on account 'fatpou01': MaxTRESPU cpu=$MAX_CORES.
       A job over that cap PENDS FOREVER on AssocGrpCpuLimit — it is not a slow job, it is a
       stuck one.  Resubmit with -c <= $MAX_CORES (the group cap is GrpTRES cpu=1500, but the
       PER-USER limit is what binds you)."
fi

# ===========================================================================
# 2. ⛔ THE ACCOUNT AND PARTITION GUARDS.  The #SBATCH directives above say
#    `-A fatpou01 -p multicore`, but a submit-line `-A`/`-p` silently overrides
#    them.  Check what Slurm ACTUALLY gave us.
# ===========================================================================
CPU_ACCOUNT="fatpou01"
if [ -n "${SLURM_JOB_ACCOUNT:-}" ] && [ "$SLURM_JOB_ACCOUNT" != "$CPU_ACCOUNT" ]; then
    die "this job is running under account '$SLURM_JOB_ACCOUNT', not '$CPU_ACCOUNT'.
       ⛔ 'gpu-free' is the GPU association and carries NO CPU grant — under it this job is
       rejected or scavenges, and the whole point (an uncontended CPU entitlement) is lost.
       Resubmit WITHOUT overriding -A, or with -A $CPU_ACCOUNT."
fi
if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    case "$SLURM_JOB_PARTITION" in
        serial) die "partition 'serial' is a ONE-CORE partition — multi-core work here is
       silently serialised.  Resubmit with -p multicore." ;;
        multicore) ;;
        *) say "⚠ partition is '$SLURM_JOB_PARTITION', not 'multicore' — proceeding, but the
       core cap and wallclock max below are multicore's." ;;
    esac
fi

# ===========================================================================
# 3. ⭐ CPU PLATFORM, SET AND ASSERTED.
#    A job that silently finds no GPU and runs anyway is FINE here — it is CPU work.
#    A job that REPORTS GPU numbers from a CPU run is not.  So the platform is pinned,
#    not inferred.  ⚠ On a GPU-less node JAX otherwise logs
#        cuInit(0) failed: Unknown CUDA error 303
#    which is benign but looks exactly like a failure in a log; pinning JAX_PLATFORMS
#    SUPPRESSES it at the source rather than filtering it downstream.
# ===========================================================================
export JAX_PLATFORMS=cpu

# BLAS/OpenMP thread pools follow our core grant, not the node's 168 cores: an
# unbound numpy/sklearn on `multicore` oversubscribes its cgroup and runs SLOWER.
# (JAX's own CPU pool follows the cgroup's visible CPUs and is left alone.)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CORES}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CORES}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CORES}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CORES}"

# ===========================================================================
# 4. Cluster environment.  env.sh exports AND asserts the five CHLU_* roots; we
#    assert them again in §5 so the guard also fires under GEN_LOCAL=1, where
#    env.sh is skipped — a guard that only exists on a machine we cannot reach
#    is a guard nobody has ever seen fire.
# ===========================================================================
if [ "$GEN_LOCAL" = "1" ]; then
    export CLU_REPO="${CLU_REPO:-$PWD}"
    say "GEN_LOCAL=1 — skipping module purge / env.sh / JAX preflight (plumbing only)"
    SKIP_JAX_PREFLIGHT=1
else
    if command -v module >/dev/null 2>&1; then module purge; fi
    export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
    # shellcheck disable=SC1091
    source "$CLU_REPO/scripts/csf3/env.sh"
fi
PY_BIN="${GEN_PYTHON:-python3}"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN'"

# ===========================================================================
# 5. ⭐ ARTEFACT ROOTS, ASSERTED AT JOB START, EACH NAMED.
#    ⛔ Never reach the work to find it cannot open an artefact: on a queue where
#    we waited for the allocation, discovering a missing root an hour in is the
#    expensive failure.  Opt out with CHLU_SKIP_ARTEFACT_CHECK=1 only for a run
#    that genuinely needs no artefacts (SMOKE/DRY plumbing).
# ===========================================================================
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CLU_REPO/artefacts}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"

if [ "${CHLU_SKIP_ARTEFACT_CHECK:-0}" != "1" ]; then
    _missing=""
    for _v in CHLU_CLAUDE_ROOT CHLU_GEN_DATA_ROOT CHLU_GEN_BRIDGE_ROOT \
              CHLU_GEN_STORE_RUNS CHLU_GEN_DECLIFT_DIR; do
        eval "_p=\$$_v"
        [ -e "$_p" ] || _missing="${_missing}
    $_v -> $_p"
    done
    if [ -n "$_missing" ]; then
        echo "FATAL[job_cpu_gen]: gen artefact root(s) MISSING, by name:$_missing" >&2
        echo "       Stage them with scripts/csf3/stage_artefacts.sh (⛔ do not improvise a" >&2
        echo "       staging path), or set CHLU_SKIP_ARTEFACT_CHECK=1 if this run needs none." >&2
        exit 2
    fi
    unset _v _p _missing
    say "artefact roots present: CHLU_CLAUDE_ROOT=$CHLU_CLAUDE_ROOT"
else
    say "⚠ CHLU_SKIP_ARTEFACT_CHECK=1 — artefact roots NOT asserted"
fi

# ===========================================================================
# 6. Paths.  OUT is the sweep root (the module makes seed_<N>/ underneath it);
#    CKPT_DIR holds the unit ledger.
# ===========================================================================
case "$EXP" in
    atom_mean) DEF_OUT="$CHLU_CLAUDE_ROOT/outputs/gen-atom-mean-control/runs/${DATASET}_${PRETEXT}" ;;
    baselines) DEF_OUT="$CHLU_CLAUDE_ROOT/outputs/gen-baselines/runs/${DATASET}_${PRETEXT}" ;;
esac
OUT="${OUT:-$DEF_OUT}"
CKPT_DIR="${CKPT_DIR:-$HOME/scratch/CHLU/ckpt/gen-cpu-${EXP}-${DATASET}_${PRETEXT}}"

# ⛔ $TMPDIR is node-local NVMe and is WIPED AT JOB END: a checkpoint there protects
#    nothing, which is the exact opposite of the point.
case "$CKPT_DIR" in
    "${TMPDIR:-/nonexistent-tmpdir}"*)
        die "CKPT_DIR='$CKPT_DIR' is under \$TMPDIR, which is WIPED AT JOB END." ;;
esac
# ⛔ $HOME is small, quota'd and backed up, and this writes after every unit.
#    ~/scratch is a different filesystem even though its path starts with $HOME.
case "$CKPT_DIR" in
    "${HOME}/scratch"*|"${HOME}/scratch") ;;
    "${HOME}"*) die "CKPT_DIR='$CKPT_DIR' is under \$HOME (quota'd, backed up).
       Put it under ~/scratch." ;;
esac

JOURNAL="$CKPT_DIR/journal.json"

# ===========================================================================
# 7. The per-unit payload command.  ⭐ ONE launcher: the experiment is a
#    parameter, not a copy of this file.
# ===========================================================================
unit_out_for() {   # $1 = seed.  Where THIS UNIT's manifest lands.
    echo "$OUT/seed_$1"
}
payload_args_for() {  # $1 = seed.  Echoes the argv after the module name.
    _s="$1"
    case "$EXP" in
        atom_mean)
            # --out is the SWEEP root: exp_gen_atom_mean appends seed_<N>/ itself.
            printf '%s' "--dataset $DATASET --pretext $PRETEXT --stage $UNIT_STAGE --seed $_s --out $OUT"
            for _b in $BUDGETS; do printf '%s' " --budget $_b"; done
            ;;
        baselines)
            # --out is the PER-SEED dir: exp_gen_baselines' `--stage baselines` passes
            # it straight to run_seed, which does NOT append seed_<N>/.
            printf '%s' "--dataset $DATASET --pretext $PRETEXT --stage $UNIT_STAGE --arm $ARM --seed $_s --out $(unit_out_for "$_s")"
            ;;
    esac
    [ "$QUICK" = "1" ] && printf '%s' " --quick"
    [ -n "$EXTRA_ARGS" ] && printf '%s' " $EXTRA_ARGS"
    printf '\n'
}
aggregate_args() {
    case "$EXP" in
        atom_mean) printf '%s\n' "--dataset $DATASET --pretext $PRETEXT --stage aggregate --out $OUT" ;;
        baselines)
            _runs=""
            for _s in "${SEED_ARR[@]}"; do
                [ -f "$(unit_out_for "$_s")/$MANIFEST_NAME" ] && _runs="$_runs $(unit_out_for "$_s")"
            done
            printf '%s\n' "--dataset $DATASET --pretext $PRETEXT --stage aggregate --out $OUT --runs$_runs" ;;
    esac
}

# ===========================================================================
# 8. PLAN mode: units, per-unit command, the submit line.  Runs nothing.
# ===========================================================================
if [ "$PLAN" = "1" ]; then
    echo "=== cpu-gen launch plan: EXP=$EXP ${DATASET}_${PRETEXT} ==="
    echo "account:    $CPU_ACCOUNT   partition: multicore   cores: $CORES (cap $MAX_CORES)"
    echo "module:     $MODULE"
    echo "out:        $OUT"
    echo "ckpt:       $CKPT_DIR   (journal.json, last $KEEP_LAST snapshots, every ${CKPT_EVERY_SECONDS}s)"
    echo "units (${#SEED_ARR[@]}), one per seed:"
    for s in "${SEED_ARR[@]}"; do
        echo "  seed_$s -> $(unit_out_for "$s")/$MANIFEST_NAME"
        echo "      $PY_BIN -u -m $MODULE $(payload_args_for "$s")"
    done
    echo "aggregate:"
    echo "      $PY_BIN -u -m $MODULE $(aggregate_args)"
    echo "submit:"
    echo "  sbatch -c $CORES -t <WALLCLOCK, max 7-0> --export=\"ALL,EXP=$EXP,DATASET=$DATASET,PRETEXT=$PRETEXT,SEEDS=${SEED_ARR[*]},GEN_GIT_COMMIT=\$GEN_GIT_COMMIT\" scripts/csf3/job_cpu_gen.sh"
    echo "resume (identical line — RESUME=1 is the default):"
    echo "  sbatch -c $CORES -t <WALLCLOCK> --export=\"ALL,EXP=$EXP,DATASET=$DATASET,PRETEXT=$PRETEXT,SEEDS=${SEED_ARR[*]},RESUME=1,GEN_GIT_COMMIT=\$GEN_GIT_COMMIT\" scripts/csf3/job_cpu_gen.sh"
    exit 0
fi

mkdir -p "$CKPT_DIR" "$OUT" logs 2>/dev/null || mkdir -p "$CKPT_DIR" "$OUT"

# ===========================================================================
# 9. Provenance.
# ===========================================================================
GIT_COMMIT="${GEN_GIT_COMMIT:-}"
if [ -z "$GIT_COMMIT" ] && [ -e "$CLU_REPO/.git" ] && command -v git >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "$CLU_REPO" rev-parse HEAD 2>/dev/null || true)"
fi
[ -n "$GIT_COMMIT" ] && [ -z "${GEN_GIT_COMMIT:-}" ] && say "⚠ GEN_GIT_COMMIT was unset; resolved from .git"
say "=== EXP=$EXP ${DATASET}_${PRETEXT} · units=${SEED_ARR[*]} · cores=$CORES ==="
say "job=${SLURM_JOB_ID:-<local>} restart_count=${SLURM_RESTART_COUNT:-0} host=$(hostname)"
say "commit=${GIT_COMMIT:-<unset: export GEN_GIT_COMMIT on the login node>}"
say "JAX_PLATFORMS=$JAX_PLATFORMS  OMP_NUM_THREADS=$OMP_NUM_THREADS"
say "out=$OUT"
say "ckpt=$CKPT_DIR"

# ===========================================================================
# 10. ⭐ The CPU preflight: ASSERT the backend rather than hope for it.
# ===========================================================================
if [ "$SKIP_JAX_PREFLIGHT" != "1" ]; then
    say "=== JAX CPU preflight (assert the platform we claim to be on) ==="
    "$PY_BIN" - <<'PY'
import os, sys
assert os.environ.get("JAX_PLATFORMS") == "cpu", "JAX_PLATFORMS is not pinned to cpu"
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
if jax.default_backend() != "cpu":
    sys.exit("⛔ JAX_PLATFORMS=cpu was set but the backend is "
             f"{jax.default_backend()!r} — refusing: a CPU job must not report GPU numbers.")
PY
fi

# ===========================================================================
# 11. ⭐ THE UNIT LEDGER (the `pilot-checkpoint-resume` pattern, one layer up).
#     journal.json = {fingerprint, fingerprint_fields, units_done:{...}}.
#     Written tmp + os.replace (atomic).  Time-based rotated snapshots, last N.
#     A resume whose fingerprint differs is REFUSED, naming every differing key.
#
#     ⛔ SEEDS is NOT in the fingerprint: narrowing or extending the seed list is
#     the normal way to repair a sweep, and the ledger is per-unit precisely so
#     that is safe.  Nor are CORES / wallclock / job id: they cannot change a
#     number.  Everything that CAN change a number is in it.
# ===========================================================================
FINGERPRINT_FIELDS="exp=$EXP
dataset=$DATASET
pretext=$PRETEXT
module=$MODULE
unit_stage=$UNIT_STAGE
arm=$ARM
budgets=$BUDGETS
quick=$QUICK
extra_args=$EXTRA_ARGS
out=$OUT
smoke=$SMOKE"

journal_py() {  # $1 = op, rest via env
    CPUGEN_JOURNAL="$JOURNAL" CPUGEN_FIELDS="$FINGERPRINT_FIELDS" \
    CPUGEN_OP="$1" CPUGEN_UNIT="${2:-}" CPUGEN_WALL="${3:-0}" \
    CPUGEN_KEEP="$KEEP_LAST" CPUGEN_EVERY="$CKPT_EVERY_SECONDS" \
    CPUGEN_FORCE="$FORCE_NEW" CPUGEN_RESUME="$RESUME" \
    CPUGEN_JOB="${SLURM_JOB_ID:-local}" CPUGEN_COMMIT="${GIT_COMMIT:-}" \
    "$PY_BIN" - <<'PY'
import datetime as dt, glob, hashlib, json, os, sys

e = os.environ.get
p = e("CPUGEN_JOURNAL"); op = e("CPUGEN_OP")
fields = dict(l.split("=", 1) for l in e("CPUGEN_FIELDS", "").splitlines() if "=" in l)
fp = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
now = dt.datetime.now(dt.timezone.utc).isoformat()

def load():
    try:
        with open(p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

def save(rec):
    """Atomic: write a sibling tmp, fsync, os.replace.  A kill during the write
    leaves the PREVIOUS journal intact — never a truncated one."""
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)

#: ⚠ NANOSECONDS, zero-padded, not seconds.  Whole-second names COLLIDE when several
#: short units finish inside one second: the snapshot is overwritten instead of
#: rotated, and the ring silently holds fewer than N.  19 fixed-width digits also keep
#: the lexicographic glob sort chronological, which the retention below relies on.
SNAP_GLOB = p[:-len(".json")] + ".*.json"

def rotate(rec):
    """Time-based snapshot, last-N retention."""
    import time as _t
    every = float(e("CPUGEN_EVERY", "300")); keep = int(e("CPUGEN_KEEP", "5"))
    if keep <= 0 or every < 0:
        return
    snaps = sorted(glob.glob(SNAP_GLOB))
    last = 0.0
    if snaps:
        try:
            last = os.path.getmtime(snaps[-1])
        except OSError:
            last = 0.0
    if _t.time() - last < every:
        return
    snap = p[:-len(".json")] + ".%019d.json" % _t.time_ns()
    tmp = snap + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True); fh.write("\n"); fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, snap)
    for stale in sorted(glob.glob(SNAP_GLOB))[:-keep]:
        try:
            os.unlink(stale)
        except OSError:
            pass

if op == "init":
    prior = load()
    if prior is None:
        save({"schema": "chlu-cpu-gen-journal/1", "fingerprint": fp,
              "fingerprint_fields": fields, "units_done": {}, "created_at": now,
              "updated_at": now, "git_commit": e("CPUGEN_COMMIT")})
        print("[resume] no journal at %s — starting from scratch" % p)
        sys.exit(0)
    if prior.get("fingerprint") != fp:
        old = prior.get("fingerprint_fields", {})
        bad = sorted(k for k in set(old) | set(fields) if old.get(k) != fields.get(k))
        msg = ["⛔ REFUSING TO RESUME: %s was written under a DIFFERENT config." % p]
        for k in bad:
            msg.append("    %s: journal=%r now=%r"
                       % (k, old.get(k, "<absent>"), fields.get(k, "<absent>")))
        msg.append("  A resumed sweep must be the SAME sweep. Give this leg its own OUT/CKPT_DIR,")
        msg.append("  or set FORCE_NEW=1 to archive the old journal and start over.")
        if e("CPUGEN_FORCE") != "1":
            print("\n".join(msg), file=sys.stderr)
            sys.exit(3)
        arch = p + ".superseded." + now.replace(":", "")
        os.replace(p, arch)
        print("\n".join(msg))
        print("[resume] FORCE_NEW=1 — archived to %s, starting over" % arch)
        save({"schema": "chlu-cpu-gen-journal/1", "fingerprint": fp,
              "fingerprint_fields": fields, "units_done": {}, "created_at": now,
              "updated_at": now, "git_commit": e("CPUGEN_COMMIT")})
        sys.exit(0)
    if e("CPUGEN_RESUME") != "1":
        print("[resume] RESUME=0 — journal holds %d unit(s) but every unit will be RE-RUN"
              % len(prior.get("units_done") or {}))
        sys.exit(0)
    done = sorted((prior.get("units_done") or {}).keys())
    print("[resume] journal %s: fingerprint OK, %d unit(s) complete: %s"
          % (p, len(done), " ".join(done) or "[]"))
    sys.exit(0)

if op == "is_done":
    prior = load() or {}
    sys.exit(0 if e("CPUGEN_UNIT") in (prior.get("units_done") or {}) else 1)

if op == "mark":
    rec = load() or {"schema": "chlu-cpu-gen-journal/1", "fingerprint": fp,
                     "fingerprint_fields": fields, "units_done": {}, "created_at": now}
    rec.setdefault("units_done", {})[e("CPUGEN_UNIT")] = {
        "finished_at": now, "wall_s": float(e("CPUGEN_WALL", "0")),
        "job_id": e("CPUGEN_JOB"), "git_commit": e("CPUGEN_COMMIT")}
    rec["updated_at"] = now
    save(rec)
    rotate(rec)
    print("[ckpt] journal -> %s (%d unit(s) done)" % (p, len(rec["units_done"])))
    sys.exit(0)

if op == "unmark":
    rec = load() or {}
    (rec.get("units_done") or {}).pop(e("CPUGEN_UNIT"), None)
    rec["updated_at"] = now
    save(rec)
    sys.exit(0)

sys.exit("unknown journal op %r" % op)
PY
}

if [ "$DRY" != "1" ]; then
    journal_py init || exit $?
fi

# ===========================================================================
# 12. ⭐ The wallclock-kill handler.  We do NOT write a special checkpoint: the
#     journal is already on disk and atomic.  We say so loudly, with the resume
#     line, so the log answers the question rather than leaving it to the Head.
# ===========================================================================
CHILD_PID=""
_on_signal() {
    echo "⚠ [signal] wallclock/preemption signal at $(date -u +%FT%TZ)."
    [ -n "$CHILD_PID" ] && kill -TERM "$CHILD_PID" 2>/dev/null
    echo "  ⭐ Completed units are banked in $JOURNAL (atomic; last $KEEP_LAST snapshots kept)."
    echo "  ⭐ The unit that was in flight is NOT marked done and will re-run (its own"
    echo "     partial manifest lets the payload skip what it already finished)."
    echo "  ⭐ RESUME WITH:  sbatch -c $CORES -t <WALLCLOCK> --export=\"ALL,EXP=$EXP,DATASET=$DATASET,PRETEXT=$PRETEXT,SEEDS=${SEED_ARR[*]},GEN_GIT_COMMIT=\$GEN_GIT_COMMIT\" scripts/csf3/job_cpu_gen.sh"
    exit 0
}
trap _on_signal USR1 TERM

# ===========================================================================
# 13. THE SWEEP.  One unit per seed.  ⭐ A kill costs ONE UNIT.
# ===========================================================================
run_payload() {  # $1 = seed
    _s="$1"
    if [ "$SMOKE" = "1" ]; then
        # ⛔ NO SCIENCE.  This exists so the ledger/resume path is testable — and
        # KILLABLE — without the artefact set and without a 20-minute JAX import.
        say "SMOKE=1 — trivial payload for seed $_s (plumbing only)"
        _ud="$(unit_out_for "$_s")"
        mkdir -p "$_ud"
        if [ "$SMOKE_SLEEP_S" != "0" ]; then sleep "$SMOKE_SLEEP_S"; fi
        for _f in $SMOKE_FAIL_SEEDS; do
            [ "$_f" = "$_s" ] && { say "SMOKE fault injection: seed $_s exits 1"; return 1; }
        done
        for _f in $SMOKE_EMPTY_SEEDS; do
            [ "$_f" = "$_s" ] && { say "SMOKE fault injection: seed $_s exits 0, writes nothing"; return 0; }
        done
        CPUGEN_SMOKE_OUT="$_ud/$MANIFEST_NAME" CPUGEN_SMOKE_SEED="$_s" \
        CPUGEN_SMOKE_EXP="$EXP" CPUGEN_SMOKE_DS="$DATASET" CPUGEN_SMOKE_PT="$PRETEXT" \
        "$PY_BIN" - <<'PY'
import json, os
e = os.environ.get
rec = {"payload": "smoke", "seed": int(e("CPUGEN_SMOKE_SEED")), "exp": e("CPUGEN_SMOKE_EXP"),
       "dataset": e("CPUGEN_SMOKE_DS"), "pretext": e("CPUGEN_SMOKE_PT")}
with open(e("CPUGEN_SMOKE_OUT"), "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True); fh.write("\n")
print("[cpu-gen] smoke manifest ->", e("CPUGEN_SMOKE_OUT"))
PY
        return 0
    fi
    # shellcheck disable=SC2086  # the arg string is intentionally word-split
    "$PY_BIN" -u -m "$MODULE" $(payload_args_for "$_s") &
    CHILD_PID=$!
    wait "$CHILD_PID"
}

N_SKIPPED=0; N_RAN=0; N_FAILED=0; FAILED_UNITS=""
for s in "${SEED_ARR[@]}"; do
    UNIT="seed_$s"
    UNIT_MANIFEST="$(unit_out_for "$s")/$MANIFEST_NAME"

    if [ "$DRY" = "1" ]; then
        echo "[dry] $UNIT -> $UNIT_MANIFEST"
        echo "[dry]   $PY_BIN -u -m $MODULE $(payload_args_for "$s")"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    # ⭐ THE RESUME.  Both conditions, deliberately: the journal says the unit
    # finished AND the artefact is still on disk.  A journal entry whose manifest
    # has been deleted (scratch cleanup, a botched rsync) is NOT a completed unit,
    # and skipping it would silently ship a sweep with a hole in it.
    if [ "$RESUME" = "1" ] && journal_py is_done "$UNIT"; then
        if [ -f "$UNIT_MANIFEST" ]; then
            say "SKIP $UNIT (manifest exists) -> $UNIT_MANIFEST"
            N_SKIPPED=$((N_SKIPPED + 1))
            continue
        fi
        say "⚠ $UNIT is in the journal but its manifest is GONE ($UNIT_MANIFEST) — re-running"
        journal_py unmark "$UNIT"
    fi

    say "RUN  $UNIT  ($MODULE)"
    _t0=$(date +%s)
    set +e
    run_payload "$s"
    _rc=$?
    set -e
    CHILD_PID=""
    _wall=$(( $(date +%s) - _t0 ))
    if [ "$_rc" -eq 0 ]; then
        # ⛔ Mark done only if the payload's OWN artefact landed: exit 0 with no
        # manifest is a unit that did nothing, and banking it would make the hole
        # permanent across every future resume.
        if [ -f "$UNIT_MANIFEST" ]; then
            journal_py mark "$UNIT" "$_wall"
            say "DONE $UNIT in ${_wall}s"
            N_RAN=$((N_RAN + 1))
        else
            say "⛔ $UNIT exited 0 but wrote no $MANIFEST_NAME — NOT marking it done"
            N_FAILED=$((N_FAILED + 1)); FAILED_UNITS="$FAILED_UNITS $UNIT(no-manifest)"
        fi
    else
        # ⛔ NOT RUN, never a silent gap: the sweep continues so one bad seed does
        # not cost the other two, and the exit code below carries the failure.
        say "⛔ $UNIT FAILED rc=$_rc after ${_wall}s — NOT marked done, will re-run on resume"
        N_FAILED=$((N_FAILED + 1)); FAILED_UNITS="$FAILED_UNITS $UNIT(rc=$_rc)"
    fi
done

# ===========================================================================
# 14. Aggregate + summary.
# ===========================================================================
AGG_ARGS="$(aggregate_args)"
case "$EXP:$AGG_ARGS" in
    # ⛔ `--stage aggregate --runs` with NOTHING after it is not an aggregate over zero
    # runs, it is a MALFORMED COMMAND LINE.  Say why, rather than emit it — including in
    # DRY mode, where printing it would advertise a command that cannot run.
    baselines:*--runs) AGG_ARGS=""
                       say "⚠ skipping aggregate: no per-seed manifest on disk to aggregate"
                       AGGREGATE=0 ;;
esac

if [ "$DRY" = "1" ]; then
    if [ -n "$AGG_ARGS" ]; then
        echo "[dry] aggregate: $PY_BIN -u -m $MODULE $AGG_ARGS"
    else
        echo "[dry] aggregate: SKIPPED (nothing on disk to aggregate yet)"
    fi
    echo "[dry] ${N_SKIPPED} unit(s) printed, nothing executed"
    exit 0
fi
if [ "$AGGREGATE" = "1" ] && [ "$N_FAILED" -eq 0 ] && [ "$SMOKE" != "1" ]; then
    say "=== aggregate ==="
    # shellcheck disable=SC2086
    "$PY_BIN" -u -m "$MODULE" $AGG_ARGS || say "⚠ aggregate failed — the per-unit manifests are still on disk"
elif [ "$AGGREGATE" = "1" ] && [ "$N_FAILED" -ne 0 ]; then
    say "⚠ skipping aggregate: $N_FAILED unit(s) did not complete — an aggregate over a
       partial sweep is a table with a silent hole in it.  Resume, then aggregate."
fi

say "=== summary: ran=$N_RAN skipped=$N_SKIPPED failed=$N_FAILED ==="
ls -1 "$OUT"/seed_*/"$MANIFEST_NAME" 2>/dev/null || true
say "journal: $JOURNAL"
say "⚠ ~/scratch has NO BACKUP and a 3-month auto-cleanup — pull results promptly:"
say "  rsync -av csf3:$OUT/ ./.claude/outputs/<your-slug>/${DATASET}_${PRETEXT}/"
if [ "$N_FAILED" -ne 0 ]; then
    say "⛔ FAILED UNITS:$FAILED_UNITS — resume with the same submit line (RESUME=1 default)"
    exit 1
fi
say "=== job_cpu_gen ${SLURM_JOB_ID:-local} DONE ==="
