#!/bin/bash --login
# scripts/csf3/job_gpu_cluformer.sh — the TIER-III PILOT at 26-47 M on one A100.
#
# ⚠ STATUS: NOT SUBMITTED BY THE AUTHORING AGENT. `csf3.itservices.manchester.ac.uk`
#   does not resolve from the laptop without GlobalProtect VPN, which the agent
#   cannot establish (`ssh csf3` -> "Could not resolve hostname"). Everything
#   below is laptop-verified (`bash -n`, and the identical Python entry point is
#   exercised at toy scale) and is labelled UNTESTED-ON-CLUSTER.
#
# Submit (Head):
#   cd ~/scratch/CHLU && mkdir -p logs
#   # 0. ONE-OFF, SERIAL: stage enwik8 (compute nodes have internet, login nodes
#   #    do NOT). Never let N array tasks race the download.
#   sbatch --job-name=clu-stage -t 0:30:00 scripts/csf3/job_gpu_cluformer.sh   # STAGE_ONLY=1 default OFF -> see below
#   sbatch --export=ALL,STAGE_ONLY=1 -p serial -t 0:30:00 scripts/csf3/job_gpu_cluformer.sh
#   # 1. then the three seeds, <=4 concurrent (free-tier policy)
#   sbatch --export=ALL,SEEDS="0 1 2",STAGE=pilot,STG=s4,D5=1 --mail-user=$CLU_MAIL \
#          -t 12:00:00 scripts/csf3/job_gpu_cluformer.sh
#   # ⛔ LAUNCH CHECKLIST: every PRE-REGISTERED phase must appear in the
#   #    artifact's phase list. D5 is `D5=1` and NOTHING else sets it.
#   # 2. pull the artifacts
#   rsync -av csf3:~/scratch/CHLU/.claude/outputs/cluformer-pilot/ ./.claude/outputs/cluformer-pilot/
#
# WALLCLOCK (Head ruling 2026-08-02, supersedes the original "<= 108 A100-hours"
#   declaration): gpuA is free at point of use and allows -t up to 4-0, so the
#   12 h ceiling was self-imposed, NOT a cost. Submit ONE SEED PER JOB at
#   `-t 4-00:00:00` (throughput: 6 single-GPU jobs vs the <=4-concurrent cap;
#   failure isolation; early complete run1/run2 PAIRS when interleaved by seed).
#   ⛔ The former instruction "do not resubmit at a larger wallclock" + its cut
#   order ((1) D5, (2) the TTT arm, (3) depth/width) is WITHDRAWN — it protected
#   a budget that does not exist. NOTHING is cut: never the seed count, the swap
#   control, the dyn-eval column, a monitor, D5, or any arm.
#
# Partition facts (official CSF3 docs, gpu-jobs page): gpuA = A100 80GB, free at
# point of use, <=4 GPUs concurrently per user, <=12 host cores/GPU, max -t 4-0.
#
# HOST RAM (attempt-1 post-mortem, `pilot-checkpoint-resume`): gpuA gives 10 GB
#   of host RAM per core and at most 12 cores per GPU, so a 1-GPU job's HARD
#   CEILING is `-c 12` = ~120 GB and there is NO `--mem` that buys more. Job
#   18136619 was `oom_kill`ed at MaxRSS 125.6 GB against ReqMem 125.7 GB — the
#   HOST, not the A100 (device peak is ~8.3 GiB under `remat_chunks`), 22 h into
#   training and ~45 min into the POST-TRAINING EVAL BLOCK. Two consequences,
#   both handled below: (a) submit with `-c 12`, and (b) footprint reduction is
#   the primary fix, not a fallback — `eval_cache_hygiene` (default ON) releases
#   each eval phase's one-shot XLA executables before the next one compiles, and
#   every phase boundary prints an `[rss]` line so the NEXT crash, if any, is
#   attributable to a named phase rather than to "the eval block".
#   If the `[rss]` peaks still run hot, the declared uniform-SET fallback is
#   `SET="... plan_workers=4"` (lane-parallel is measured decision-identical);
#   `-G 2 -c 24` (240 GB, one idle GPU) is a Head decision, not a script default.
#
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 12:00:00               # OVERRIDE PER RUN. Max 4-0.
#SBATCH --job-name=clu-tier3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=END,FAIL      # address at submit: sbatch --mail-user=$CLU_MAIL

module purge                      # jax[cuda12] wheels bundle CUDA/cuDNN
set -eo pipefail

STAGE="${STAGE:-pilot}"           # pilot (26-47 M) | toy
STG="${STG:-s4}"                  # s1 | s2 | s3 | s4
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-}"                # empty => the config's own step count
ARMS="${ARMS:-}"                  # empty => all five arms
OUT="${OUT:-.claude/outputs/cluformer-pilot}"
STAGE_ONLY="${STAGE_ONLY:-0}"     # 1 => fetch enwik8 and exit (run this FIRST)
# ⭐ `pilot-placement-probe`'s recommendation block goes HERE, as flags -- never
#    by editing the module (an edited module is a provenance hole; a flag is
#    recorded verbatim in the artifact's `flags` block). All three default to
#    EMPTY, so an unmodified submission is bit-identical to the pre-probe one.
#      MEM   -> StreamMemoryConfig  (atom_place_radius, write_inner_steps, ...)
#      STORE -> CluSystemConfig
#      SET   -> top-level PilotConfig (monitor_every, ...)
MEM="${MEM:-}"
STORE="${STORE:-}"
SET="${SET:-}"
# ⭐ `pilot-checkpoint-resume`: RESUME=1 picks the run back up from the journal
#    in $OUT (`pilot_<scale>_seed<N>_PARTIAL.json` + `ckpt_<arm>_seed<N>.eqx`).
#    Banked phases are lifted verbatim; a trained-but-not-evaluated arm keeps its
#    weights and only re-runs its evals — which is exactly attempt 1's loss
#    (22 h of clu_store training, killed 45 min into the post-training eval
#    block). ⛔ It REFUSES to resume a journal written under a different config,
#    so a resubmission must carry byte-identical MEM/STORE/SET.
#    ⚠ Safe to leave at 1 on a first submission: no journal => a normal run.
RESUME="${RESUME:-0}"
# ⛔ `pilot-ttt-nan-and-d5-wiring` DEFECT 2: D5 (the anytime curve) is a
#    PRE-REGISTERED deliverable gated behind `--d5`, and until this line existed
#    NO launch path set it -- so `with_d5=False` on every attempt and no
#    `anytime_curve` key exists in any landed artifact. That looked identical, in
#    the artifact, to the withdrawn cut order having fired; it was plumbing.
#    ⭐ `--d5` is a CLI argument, NOT a PilotConfig field, so it does not enter
#    `rec['flags']` and CANNOT change the resume fingerprint: a FINISHED leg can
#    be re-resumed with `RESUME=1 D5=1` and it will lift every banked phase
#    (including the 219 GB `dyneval`) and compute ONLY the missing
#    `anytime_curve`. ⛔ Do NOT also pass `ARMS` on such a re-resume -- `arms` IS
#    a config field and a narrowed arm list is a DIFFERENT config, which the
#    journal check refuses.
D5="${D5:-0}"
# ⭐⭐ RUN 3's PRE-REGISTERED CONTINUATION. ⛔ Runs 1-2 were submitted with THIS
#    script, and run 3 is a continuation of run 2, so run 3 must use it too --
#    it CANNOT be routed through `job_gpu_c3_seeds.sh`, which narrows `--arms`
#    per array task, and `arms` IS a PilotConfig field => a SECOND differing key
#    => the exemption refuses the leg. Without the passthrough below the
#    state-byte budget refuses run 3 outright (exit 1, before training), and a
#    pre-registered leg behind a flag no launch path sets is indistinguishable
#    from a deliberate cut (§7.33). Set e.g.
#      PREREG_CONT="journal=$RUN2_OUT/pilot_pilot_seed${SEED}_PARTIAL.json \
#                   flag=memory.erosion_partition \
#                   prereg=.claude/outputs/c2w6-anti-erosion/PREREG-LeakAblation.md"
#    ⛔ It exempts the state-byte BUDGET check ONLY, and only if this config is
#    identical to that journal except `memory.erosion_partition`.
#    ⛔ It must be on EVERY submission AND every re-resume.
PREREG_CONT="${PREREG_CONT:-}"    # run 3's pre-registered continuation

export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
# shellcheck disable=SC1091
source "$CLU_REPO/scripts/csf3/env.sh"

# ---- step 0: download-once (serial), then exit ----------------------------
if [ "$STAGE_ONLY" = "1" ]; then
  echo "=== staging enwik8 (SERIAL, download-once) ==="
  python -c "from chlu.data.enwik8 import stage_enwik8; print('staged', stage_enwik8())"
  echo "=== stage-only DONE ==="
  exit 0
fi

echo "=== node/GPU provenance ==="
hostname
nvidia-smi || true
echo "SLURM_GPUS=${SLURM_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "=== JAX GPU preflight (fail fast rather than burn 12 h on CPU) ==="
python - <<'PY'
import jax
print("jax", jax.__version__, "backend:", jax.default_backend(), "devices:", jax.devices())
assert jax.default_backend() == "gpu", (
    "JAX did not initialise the A100 — check the cuda extra install "
    "(setup_env_job.sh) before burning GPU wallclock."
)
PY

echo "=== enwik8 cache preflight (must already be staged by STAGE_ONLY=1) ==="
python - <<'PY'
from chlu.data.enwik8 import stage_enwik8
print("enwik8:", stage_enwik8(download=False))
PY

EXTRA=""
[ -n "$STEPS" ] && EXTRA="$EXTRA --steps $STEPS"
[ -n "$ARMS" ] && EXTRA="$EXTRA --arms $ARMS"
[ -n "$MEM" ] && EXTRA="$EXTRA --mem $MEM"
[ -n "$STORE" ] && EXTRA="$EXTRA --store $STORE"
[ -n "$SET" ] && EXTRA="$EXTRA --set $SET"
[ "$RESUME" = "1" ] && EXTRA="$EXTRA --resume"
[ "$D5" = "1" ] && EXTRA="$EXTRA --d5"
[ -n "$PREREG_CONT" ] && EXTRA="$EXTRA --prereg-continuation $PREREG_CONT"
echo "=== config overrides === MEM='$MEM' STORE='$STORE' SET='$SET' RESUME='$RESUME' D5='$D5' PREREG_CONT='$PREREG_CONT'"

echo "=== tier-iii pilot: scale=$STAGE stage=$STG seeds=$SEEDS ==="
# shellcheck disable=SC2086
python -u -m chlu.experiments.exp_cluformer_pilot \
    --scale "$STAGE" --stage "$STG" --seeds $SEEDS --out "$OUT" $EXTRA

echo "=== artifacts ==="
ls -l "$OUT" || true
echo "Pull to laptop with:"
echo "  rsync -av csf3:$CLU_REPO/$OUT/ ./.claude/outputs/cluformer-pilot/"
