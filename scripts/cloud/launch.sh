#!/usr/bin/env bash
# scripts/cloud/launch.sh -- run one CHLU job on one GCP GPU, and get it back.
#
#   scripts/cloud/launch.sh --card l4 --job smoke1 --steps 200
#   scripts/cloud/launch.sh --card l4 --spot --job pilot1 \
#       --exp cluformer --units '0 1' --exp-args '--stage s1 --scale toy'
#   scripts/cloud/launch.sh --card a100-40 --spot --job cell1 \
#       --exp gen_cell --units '0 1 2' --artefacts gs://<bucket>/artefacts --wait
#   scripts/cloud/launch.sh --card a100-80 --dry-run        # refuses: no quota
#
# Design commitments, in the order they bite:
#   * QUOTA IS CHECKED BEFORE ANYTHING IS CREATED, and a shortfall is refused
#     by name (card + measured limit) rather than submitted and discovered.
#   * THE IMAGE MUST BE DIGEST-PINNED. A tag is refused outright.
#   * ONE LAUNCHER, PARAMETERISED: --exp selects the experiment module from
#     startup.sh's registry. There is never a second launcher per experiment.
#   * SPOT IS THE DEFAULT ASSUMPTION FOR TRAINING and spot VMs get taken away,
#     so the job checkpoints to GCS on a timer and resumes by default. A spot
#     job should be driven by run_until_done.sh, not by hand.
#   * The VM deletes itself on every exit path, plus a --max-hours watchdog.
#   * A cost line is printed at exit.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"
PY="${PYTHON:-python3}"

CARD=l4 SPOT=0 JOB="" GPUS=1 DRY=0 WAIT=0 KEEP_VM=0 FRESH=0
STEPS=400 SIZE=4096 STEP_SLEEP=0 CKPT_EVERY=20 SYNC_EVERY=15 MAX_HOURS=2 SEED=0
IMAGE="${CHLU_IMAGE:-}"
EXTRA_ENV="" DETERMINISTIC=0
# --- the real-experiment knobs ------------------------------------------------
EXP="${CHLU_EXP:-smoke}" EXP_ARGS="" UNITS="0" SKIP_ART=0
ARTEFACTS="${CHLU_ARTEFACTS:-}" DEC_RECIPE="${CHLU_DEC_RECIPE:-v1_capacity_schedule}"

usage() { sed -n '2,36p' "$0"; exit "${1:-0}"; }
die() { echo "FATAL: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --card) CARD="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;
    --exp) EXP="$2"; shift 2 ;;
    --exp-args) EXP_ARGS="$2"; shift 2 ;;
    --units) UNITS="$2"; shift 2 ;;
    --artefacts) ARTEFACTS="$2"; shift 2 ;;
    --dec-recipe) DEC_RECIPE="$2"; shift 2 ;;
    --skip-artefact-check) SKIP_ART=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --spot) SPOT=1; shift ;;
    --on-demand) SPOT=0; shift ;;
    --region) CHLU_REGION="$2"; CHLU_ZONE="${2}-a"; shift 2 ;;
    --zone) CHLU_ZONE="$2"; CHLU_REGION="${2%-*}"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --step-sleep-s) STEP_SLEEP="$2"; shift 2 ;;
    --ckpt-every-s) CKPT_EVERY="$2"; shift 2 ;;
    --sync-every-s) SYNC_EVERY="$2"; shift 2 ;;
    --max-hours) MAX_HOURS="$2"; shift 2 ;;
    --env) EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV;}$2"; shift 2 ;;
    --deterministic) DETERMINISTIC=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --keep-vm) KEEP_VM=1; shift ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown argument '$1' (try --help)" ;;
  esac
done
[ -n "$JOB" ] || die "--job is required (it namespaces gs://${CHLU_BUCKET}/jobs/<job>)"
echo "$JOB" | grep -Eq '^[a-z]([-a-z0-9]*[a-z0-9])?$' \
  || die "--job '$JOB' must be a GCE-legal name: lowercase letters, digits, hyphens"

# --- 0. the EXP must exist in startup.sh's registry ---------------------------
# Asked HERE, on the laptop, rather than discovered on the VM: an unknown EXP is
# a typo, and finding it out after a stockout walk and a 7 GB image pull wastes
# a GPU-minute to report a spelling mistake. startup.sh answers, so the launcher
# and the VM cannot hold two different registries.
EXP_ROW="$(bash "$HERE/startup.sh" --check exp "$EXP")" \
  || die "--exp '$EXP' is not in startup.sh's registry (see the list above).
       ⛔ Do not write a second launcher for a new experiment -- add a row to
       exp_row() in scripts/cloud/startup.sh."
read -r _EXP_MODULE _EXP_OUTMODE EXP_NEEDS_ART <<<"$EXP_ROW"
for u in $UNITS; do
  case "$u" in ''|*[!0-9]*) die "--units '$UNITS': unit '$u' is not a non-negative integer" ;; esac
done
[ -n "$UNITS" ] || die "--units is empty"
# ⛔ The decoder trap, refused at the launcher too: GenBridgeConfig.dec_recipe
# defaults to "baseline", the recipe that FAILED the gate. startup.sh asserts it
# again on the VM against the staged bytes; this is the cheap first line.
[ "$DEC_RECIPE" = "v1_capacity_schedule" ] \
  || die "--dec-recipe '$DEC_RECIPE' is not 'v1_capacity_schedule'. 'baseline' is
       the SUPERSEDED decoder that failed the gate (0.7191, 0/3 seeds); a run
       through it looks entirely normal in the logs and is VOID."
if [ "$EXP_NEEDS_ART" = "1" ] && [ -z "$ARTEFACTS" ] && [ "$SKIP_ART" != "1" ]; then
  die "--exp '$EXP' needs the gen artefact set but no --artefacts gs://... was given.
       Stage it with scripts/cloud/stage_gcs.sh (⛔ do not improvise a staging
       path), or pass --skip-artefact-check if this run genuinely needs none."
fi

# --- 1. quota preflight -------------------------------------------------------
# Deliberately first. Nothing is created, nothing is billed, and a "no" here is
# actionable (it names the metric the Head must raise) in a way a GCE
# QUOTA_EXCEEDED five minutes into a launch is not.
SPOT_FLAG=""; [ "$SPOT" = "1" ] && SPOT_FLAG="--spot"
echo "== preflight: ${CARD} x${GPUS} $([ "$SPOT" = 1 ] && echo spot || echo on-demand) in ${CHLU_REGION}"
"$PY" "$HERE/quota.py" --card "$CARD" --gpus "$GPUS" $SPOT_FLAG \
     --project "$CHLU_PROJECT" --region "$CHLU_REGION" \
  || die "quota preflight refused this launch (see above). Nothing was created."

read -r ACCEL MACHINE _ _ VCPU <<<"$(chlu_card_spec "$CARD")"
RATE="$(chlu_price "$CARD" "$SPOT")"

# --- 2. image must be pinned by digest ---------------------------------------
if [ -z "$IMAGE" ]; then
  [ -f "$HERE/IMAGE_DIGEST" ] || die "no --image and no $HERE/IMAGE_DIGEST -- run scripts/cloud/build_image.sh"
  IMAGE="$(tr -d '[:space:]' < "$HERE/IMAGE_DIGEST")"
fi
case "$IMAGE" in
  *@sha256:*) : ;;
  *) die "image '$IMAGE' is pinned by TAG. Refusing: a tag is a mutable pointer, so a
       re-run would silently get a different environment and we would never know.
       Pass a digest ref, e.g. ${CHLU_AR_HOST}/${CHLU_PROJECT}/${CHLU_AR_REPO}/${CHLU_IMAGE_NAME}@sha256:<...>" ;;
esac

VM="chlu-${JOB}"
REMOTE="gs://${CHLU_BUCKET}/jobs"
ARGS="--steps ${STEPS} --size ${SIZE} --seed ${SEED} --ckpt-every-s ${CKPT_EVERY} --step-sleep-s ${STEP_SLEEP}"
[ "$FRESH" = "1" ] && ARGS="$ARGS --fresh"

# XLA autotuning picks matmul algorithms per process, so the same graph on two
# VMs can differ in the last bits. --deterministic turns that off, which is
# what makes a run comparable BIT-FOR-BIT with another run -- the only way a
# resume can be checked by bytes rather than by eyeball.
if [ "$DETERMINISTIC" = "1" ]; then
  EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV;}XLA_FLAGS=--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
fi

cat <<PLAN
== plan
   job           ${JOB}          vm ${VM}
   card          ${CARD} (${ACCEL}) x${GPUS} on ${MACHINE} [${VCPU} vCPU]
   provisioning  $([ "$SPOT" = 1 ] && echo "SPOT -- expect pre-emption; the job resumes" || echo "on-demand")
   zone          ${CHLU_ZONE} preferred; falls back across ${CHLU_REGION} on a stockout
   boot image    ${CHLU_BOOT_IMAGE} (${CHLU_BOOT_IMAGE_PROJECT})
   run image     ${IMAGE}
   experiment    ${EXP} -> ${_EXP_MODULE} (out_mode ${_EXP_OUTMODE}, artefacts ${EXP_NEEDS_ART})
   units         ${UNITS}
   exp args      ${EXP_ARGS:-<none>}
   artefacts     ${ARTEFACTS:-<none needed>}   dec_recipe ${DEC_RECIPE}
   payload       ${ARGS}   (smoke only)
   checkpoints   ${REMOTE}/${JOB}/ckpt   every ${CKPT_EVERY}s
   sync-back     ${REMOTE}/${JOB}/out    every ${SYNC_EVERY}s (DURING the run, not only at exit)
   determinism   $([ "$DETERMINISTIC" = 1 ] && echo "XLA deterministic ops, autotune off" || echo "default (XLA may autotune; last-bit variation expected across VMs)")
   watchdog      ${MAX_HOURS} h, then the VM deletes itself
   est. cost     \$${RATE}/h x ${MAX_HOURS} h ceiling = \$$("$PY" -c "print(round(${RATE}*${MAX_HOURS},2))")
                 (${CHLU_PRICES_AS_OF})
PLAN

if [ "$DRY" = "1" ]; then echo "== --dry-run: nothing created."; exit 0; fi

# --- 3. create -----------------------------------------------------------------
# NOTE: bash 3.2 (macOS /bin/bash) treats "${PROV[@]}" on an EMPTY array as an
# unbound variable under `set -u`, so the expansion is guarded. Without this the
# script dies on every on-demand launch, on the launcher we actually use.
PROV=(); [ "$SPOT" = "1" ] && PROV=(--provisioning-model=SPOT --instance-termination-action=DELETE)
T0="$(date -u +%s)"

# `--metadata` is a COMMA-separated key=value list, so any value containing a
# comma is silently split into two bogus keys. Experiment arguments routinely
# contain one (`--set a=1,b=2`), so they travel as a file instead.
# ⛔ CLEAR THE PREVIOUS ATTEMPT'S SENTINELS BEFORE CREATING ANYTHING.
# out/EXIT_CODE, out/DONE, out/FAILED and out/PREEMPTED describe ONE attempt.
# The sync-back is additive and never deletes, so they survive into the next
# one -- and then `--wait` returns on a stale EXIT_CODE seconds after create,
# before the VM has even booted, and the supervisor reads a stale FAILED and
# calls a fresh attempt a payload failure. Measured on `gencell2`: attempt 2
# "finished" in 62 s without a VM ever running the payload.
# ⭐ out/result.json is NOT cleared: it is the COMPLETION sentinel, it describes
# the job rather than an attempt, and the supervisor has already checked it.
for _stale in EXIT_CODE DONE FAILED PREEMPTED; do
  gcloud storage rm "${REMOTE}/${JOB}/out/${_stale}" --project="$CHLU_PROJECT" -q 2>/dev/null
done

ARGFILE="$(mktemp)"; printf '%s' "$EXP_ARGS" > "$ARGFILE"
GIT_COMMIT="$(git -C "$HERE/../.." rev-parse HEAD 2>/dev/null || echo unknown)"

# Zone fallback. A STOCKOUT is NOT a quota problem -- the preflight already
# passed, the zone is simply out of that card right now -- and GCE tells us
# which zones do have it. Retrying by hand every time is how a launcher stops
# being used, so it retries here. Quota is regional, so no re-check is needed.
create_in_zone() {
  gcloud compute instances create "$VM" \
    --project="$CHLU_PROJECT" --zone="$1" \
    --machine-type="$MACHINE" \
    --accelerator="type=${ACCEL},count=${GPUS}" \
    --image="$CHLU_BOOT_IMAGE" --image-project="$CHLU_BOOT_IMAGE_PROJECT" \
    --boot-disk-size=200GB --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE --scopes=cloud-platform \
    ${PROV[@]+"${PROV[@]}"} \
    --metadata-from-file=startup-script="$HERE/startup.sh",shutdown-script="$HERE/shutdown.sh",chlu-exp-args="$ARGFILE" \
    --metadata="chlu-image=${IMAGE},chlu-job=${JOB},chlu-remote=${REMOTE},chlu-args=${ARGS},chlu-sync-every-s=${SYNC_EVERY},chlu-keep-vm=${KEEP_VM},chlu-max-hours=${MAX_HOURS},chlu-env=${EXTRA_ENV},chlu-exp=${EXP},chlu-units=${UNITS},chlu-artefacts=${ARTEFACTS},chlu-skip-artefact-check=${SKIP_ART},chlu-dec-recipe=${DEC_RECIPE},chlu-git-commit=${GIT_COMMIT},install-nvidia-driver=True" \
    --labels="owner=chlu,job=${JOB},card=${CARD}" >/dev/null 2>"$ERRF"
}

# Candidate zones: only those that actually OFFER this machine type. Not every
# zone in a region does (us-central1-f has no g2 at all), and asking anyway
# produces an error that looks like a failure rather than a skip.
ZONES=""
for z in $(gcloud compute machine-types list --project="$CHLU_PROJECT" \
             --filter="name=${MACHINE} AND zone~^${CHLU_REGION}-" \
             --format="value(zone)" 2>/dev/null | sort -u); do
  if [ "$z" = "$CHLU_ZONE" ]; then ZONES="$z ${ZONES}"; else ZONES="${ZONES} $z"; fi
done
ZONES="$(echo "${ZONES:-$CHLU_ZONE}" | xargs)"
echo "   candidate zones for ${MACHINE}: ${ZONES}"

ERRF="$(mktemp)"; trap 'rm -f "$ERRF" "$ARGFILE"' EXIT
CREATED_ZONE=""
for z in $ZONES; do
  if create_in_zone "$z"; then CREATED_ZONE="$z"; CHLU_ZONE="$z"; break; fi
  if grep -qiE "does not have enough resources|STOCKOUT|ZONE_RESOURCE_POOL_EXHAUSTED|does not exist in zone" "$ERRF"; then
    echo "   ${z}: no ${CARD} available right now (stock/availability, not quota) -- trying the next zone"
    continue
  fi
  echo "== create failed in ${z}, and not with a stockout:" >&2
  cat "$ERRF" >&2
  exit 1
done
[ -n "$CREATED_ZONE" ] || die "no zone in ${CHLU_REGION} had a free ${CARD}. Tried: ${ZONES}.
       Quota is fine (the preflight passed) -- this is stock. Try again later,
       another region (--region), or --spot/--on-demand (separate pools)."
echo "== created ${VM} in ${CREATED_ZONE}. Follow it with:"
echo "   scripts/cloud/status.sh ${JOB}          # progress + cost so far"
echo "   gcloud compute instances get-serial-port-output ${VM} --zone=${CHLU_ZONE} --project=${CHLU_PROJECT}"

# --- 4. optionally wait, then price the run -----------------------------------
if [ "$WAIT" = "1" ]; then
  echo -n "== waiting"
  while :; do
    if gcloud storage cat "${REMOTE}/${JOB}/out/result.json" --project="$CHLU_PROJECT" >/dev/null 2>&1; then
      echo " COMPLETE (result.json present)"; break
    fi
    if gcloud storage cat "${REMOTE}/${JOB}/out/EXIT_CODE" --project="$CHLU_PROJECT" >/dev/null 2>&1; then
      echo " attempt finished"; break
    fi
    STATE="$(gcloud compute instances describe "$VM" --zone="$CHLU_ZONE" --project="$CHLU_PROJECT" \
             --format='value(status)' 2>/dev/null || echo GONE)"
    if [ "$STATE" = "GONE" ]; then echo " vm gone"; break; fi
    if [ "$STATE" = "TERMINATED" ]; then echo " PRE-EMPTED (status TERMINATED)"; break; fi
    echo -n "."; sleep 20
  done
fi

# --- 5. cost line --------------------------------------------------------------
"$HERE/cost.sh" "$JOB" "$CARD" "$SPOT" "$T0" || true
