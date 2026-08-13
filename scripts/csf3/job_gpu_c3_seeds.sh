#!/bin/bash --login
# scripts/csf3/job_gpu_c3_seeds.sh — ⭐ THE C3 LADDER: (arm x seed) AS JOBS.
#
# ⚠ STATUS: UNTESTED-ON-CLUSTER (the authoring agent has no CSF3 credentials;
#   `csf3.itservices.manchester.ac.uk` does not resolve without GlobalProtect).
#   `bash -n` clean and the argument construction is exercised by
#   tests/test_c3_csf3_harness.py, which asserts the emitted command line.
#
# WHY THIS EXISTS ALONGSIDE job_gpu_cluformer.sh
#   job_gpu_cluformer.sh runs `--seeds 0 1 2` INSIDE ONE JOB and loops all five
#   arms inside that. At 26-47 M the scout costs one arm at ~1.5 h (35 % MFU) to
#   ~18 h (3 % MFU); five arms x three seeds inside one job is ~270 h against a
#   96 h hard limit, so the monolith cannot fit and a single failure loses
#   everything. Charter §4 is explicit: MULTI-SEED = SEEDS-AS-JOBS.
#   ⇒ Here the unit of work is ONE (arm, seed) pair = one array task = one A100:
#     5 arms x 3 seeds = 15 tasks from ONE sbatch, each with its own -t budget,
#     its own failure domain, and its own resumable journal.
#
# ⛔ WHY EACH ARM NEEDS ITS OWN --out: `arms` IS a PilotConfig field, so a
#   narrowed arm list is a DIFFERENT config and the journal check (§A20.4) will
#   refuse to resume a shared directory. Each task therefore writes to
#   $OUT_BASE/<arm>_s<seed>/. The byte ledger in EVERY task's artifact still
#   carries all five arms (solve_arms solves every cell regardless), so the
#   matched-state-byte table can be assembled from any one of them.
#
# ⚠ ZSH/BASH ARGUMENT TRAP (task §3.4, and it bit the authoring agent live):
#   building the argument list in a shell variable and expanding it does NOT
#   word-split under zsh, and submits garbage. Every launch below is ONE LITERAL
#   COMMAND LINE. Do not "tidy" it into a variable.
#
# ---------------------------------------------------------------------------
# LAUNCH (the whole ladder, 3 seeds x 5 arms, one command):
#
#   cd ~/scratch/CHLU && mkdir -p logs
#
#   # 0. ⛔ STAGE FIRST, SERIALLY, ONCE. Array tasks are HARD-BLOCKED from
#   #    downloading (chlu.data.corpora.in_array_task); this is not advisory.
#   sbatch --export=ALL,CORPUS=enwik8,STAGE_ONLY=1 -p serial -t 0:30:00 \
#          scripts/csf3/job_gpu_c3_seeds.sh
#
#   # 1. then the ladder: 15 tasks (5 arms x 3 seeds), <=4 concurrent
#   sbatch -a 0-14%4 -t 1-00:00:00 --mail-user=$CLU_MAIL \
#          --export=ALL,CORPUS=enwik8,SCALE=pilot,STG=s4,N_SEEDS=3,SEED_BASE=0,D5=1,SLICES=1 \
#          scripts/csf3/job_gpu_c3_seeds.sh
#
#   # 2. verify every task produced a log BEFORE walking away (§3.4)
#   ls -l logs/clu-c3-*.out | wc -l      # expect 15
#
#   # 3. pull
#   rsync -av csf3:~/scratch/CHLU/.claude/outputs/c3-ladder/ ./.claude/outputs/c3-ladder/
#
# 3 SEEDS IS THE PAPER BAR (charter §5) and it is the DEFAULT here: N_SEEDS=3.
# A different width is `-a 0-<5*N_SEEDS-1>` with N_SEEDS set to match.
# ---------------------------------------------------------------------------
#
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 12                     # 12 cores = ~120 GB host RAM, the 1-GPU ceiling
#SBATCH -t 1-00:00:00             # OVERRIDE PER RUN. Max 4-0 (per-job envelope).
#SBATCH -a 0-14%4                 # 5 arms x 3 seeds; %4 = the free-tier GPU cap
#SBATCH --job-name=clu-c3
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge                      # jax[cuda12] wheels bundle CUDA/cuDNN
set -eo pipefail

CORPUS="${CORPUS:-enwik8}"        # enwik8 | wikitext103  (a CONFIG VALUE)
SCALE="${SCALE:-pilot}"           # pilot (26-47 M) | toy
STG="${STG:-s4}"                  # s1 | s2 | s3 | s4
N_SEEDS="${N_SEEDS:-3}"           # >=3 is the paper bar
SEED_BASE="${SEED_BASE:-0}"
ARM_LIST="${ARM_LIST:-clu_store gru_matched ttt_matched none echo}"
OUT_BASE="${OUT_BASE:-.claude/outputs/c3-ladder}"
STAGE_ONLY="${STAGE_ONLY:-0}"
RESUME="${RESUME:-1}"             # safe on a first submission (no journal => normal run)
D5="${D5:-0}"                     # the anytime curve (pre-registered; nothing else sets it)
SLICES="${SLICES:-1}"             # the within-document retention slices
MEM="${MEM:-}"
STORE="${STORE:-}"
SET="${SET:-}"

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

# ---- step 0: download-once (SERIAL, non-array), then exit -------------------
if [ "$STAGE_ONLY" = "1" ]; then
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "⛔ STAGE_ONLY must run as a SERIAL job, not an array task." >&2
    exit 1
  fi
  echo "=== staging corpus '$CORPUS' (SERIAL, download-once) ==="
  python -c "from chlu.data.corpora import stage_corpus; print('staged', stage_corpus('$CORPUS'))"
  echo "=== stage-only DONE ==="
  exit 0
fi

# ---- resolve THIS task's (arm, seed) ---------------------------------------
# shellcheck disable=SC2206
ARMS=($ARM_LIST)
TID="${SLURM_ARRAY_TASK_ID:-0}"
N_ARMS=${#ARMS[@]}
ARM="${ARMS[$((TID / N_SEEDS))]}"
SEED=$((SEED_BASE + TID % N_SEEDS))
OUT="${OUT_BASE}/${ARM}_s${SEED}"
mkdir -p "$OUT"

echo "=== C3 ladder task ${TID}/$((N_ARMS * N_SEEDS - 1)): arm=${ARM} seed=${SEED} ==="
echo "    corpus=${CORPUS} scale=${SCALE} stage=${STG} out=${OUT}"
hostname
nvidia-smi || true

echo "=== JAX GPU preflight (fail fast rather than burn a 4-day allocation on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock.")
PY

echo "=== corpus cache preflight (array tasks may NOT download) ==="
python - <<PY
from chlu.data.corpora import stage_corpus
print("${CORPUS}:", stage_corpus("${CORPUS}", download=False))
PY

# ---- ⛔ the .eqx PRECONDITION, before any re-resume -------------------------
# Ruling (2) of pilot-ttt-nan-and-d5-wiring's reconciliation list: a journal that
# banks an arm as trained whose checkpoint is missing silently RETRAINS it (~16 h).
# The Python side raises too (resume_require_ckpt), but catching it here costs
# nothing and reports before the GPU is touched.
if [ "$RESUME" = "1" ] && [ -f "$OUT/pilot_${SCALE}_seed${SEED}_PARTIAL.json" ]; then
  N_CKPT=$(find "$OUT" -maxdepth 1 -name "ckpt_*_seed${SEED}.eqx" | wc -l | tr -d ' ')
  echo "=== resume precondition: journal present, ${N_CKPT} .eqx checkpoint(s) in $OUT ==="
  python - <<PY
import json, pathlib, sys
j = json.loads(pathlib.Path("$OUT/pilot_${SCALE}_seed${SEED}_PARTIAL.json").read_text())
banked = sorted((j.get("_journal") or {}).get("trained", {}))
missing = [a for a in banked
           if not pathlib.Path("$OUT", f"ckpt_{a}_seed${SEED}.eqx").exists()]
print("banked-as-trained:", banked or "[]", "| missing .eqx:", missing or "[]")
if missing:
    sys.exit("⛔ refusing to submit: %s banked as trained but its .eqx is gone; "
             "resuming would silently retrain it." % missing)
PY
fi

EXTRA=""
[ -n "$MEM" ] && EXTRA="$EXTRA --mem $MEM"
[ -n "$STORE" ] && EXTRA="$EXTRA --store $STORE"
[ -n "$SET" ] && EXTRA="$EXTRA --set $SET"
[ "$RESUME" = "1" ] && EXTRA="$EXTRA --resume"
[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"
[ "$SLICES" = "1" ] && EXTRA="$EXTRA --slices"
echo "=== overrides === CORPUS='$CORPUS' MEM='$MEM' STORE='$STORE' SET='$SET' RESUME='$RESUME' D5='$D5' SLICES='$SLICES'"

# ONE literal command line. ⛔ Do not refactor into a variable (see the zsh note).
# shellcheck disable=SC2086
python -u -m chlu.experiments.exp_cluformer_pilot \
    --scale "$SCALE" --stage "$STG" --seed "$SEED" --out "$OUT" \
    --corpus "$CORPUS" --arms "$ARM" $EXTRA

echo "=== artifacts ==="
ls -l "$OUT" || true
echo "=== task ${TID} done: arm=${ARM} seed=${SEED} -> ${OUT} ==="
