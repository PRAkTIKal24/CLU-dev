#!/bin/bash
# scripts/cloud/startup.sh -- GCE startup script for a CHLU GPU job VM.
#
# Uploaded by launch.sh as the `startup-script` metadata key. Every job
# parameter arrives as a separate metadata attribute, so this file is generic
# and never templated.
#
# Ordering here is deliberate and pre-emption-shaped:
#   1. the sync loop starts BEFORE the payload, so a VM that dies during
#      `docker pull` still leaves its log in the bucket;
#   2. the payload checkpoints to GCS itself, so losing the VM loses nothing
#      beyond the current checkpoint interval;
#   3. the VM deletes itself on ANY exit path, so a crashed job does not bill
#      overnight.
set -uo pipefail

LOG=/var/log/chlu-job.log
exec > >(tee -a "$LOG") 2>&1
echo "=== chlu startup $(date -u +%FT%TZ) on $(hostname) ==="

md() { curl -s -f -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null; }
attr() { md "attributes/$1"; }

ZONE="$(md zone | awk -F/ '{print $NF}')"
NAME="$(md name)"
IMAGE="$(attr chlu-image)"
JOB="$(attr chlu-job)"
REMOTE="$(attr chlu-remote)"
ARGS="$(attr chlu-args)"
SYNC_EVERY="$(attr chlu-sync-every-s)"; SYNC_EVERY="${SYNC_EVERY:-15}"
KEEP_VM="$(attr chlu-keep-vm)"; KEEP_VM="${KEEP_VM:-0}"
MAX_HOURS="$(attr chlu-max-hours)"; MAX_HOURS="${MAX_HOURS:-6}"
# Extra container env as `K=V;K=V`. Used for XLA_FLAGS, which is how a run asks
# for deterministic GPU kernels -- a host-side knob on purpose, so changing it
# does not mean rebuilding (and re-digesting) the image.
CHLU_ENV="$(attr chlu-env)"; CHLU_ENV="${CHLU_ENV:-}"

OUT=/var/chlu/out
mkdir -p "$OUT" /var/chlu/jax_cache
echo "job=$JOB image=$IMAGE remote=$REMOTE zone=$ZONE args=[$ARGS]"

# --- the image reference must be digest-pinned --------------------------------
# A tag is a mutable pointer. Accepting one here would make every downstream
# claim about "the image we ran" unverifiable, so refuse rather than warn.
case "$IMAGE" in
  *@sha256:*) : ;;
  *) echo "FATAL: image '$IMAGE' is not pinned by digest"; echo FATAL_NOT_PINNED > "$OUT/FAILED" ;;
esac

# --- 1. sync-back loop, started FIRST ----------------------------------------
# Runs on a timer DURING the job. A pre-empted VM never reaches its last write,
# so an at-exit-only sync produces nothing in exactly the common case.
# `ckpt/` is excluded: the payload publishes each checkpoint to
# $REMOTE/$JOB/ckpt/ itself, atomically, the instant it is written. Letting the
# timer copy it a second time into $REMOTE/$JOB/out/ckpt/ doubles the storage
# and leaves two sets of checkpoints with different retention.
sync_now() {
  gcloud storage cp "$LOG" "$REMOTE/$JOB/out/job.log" -q 2>/dev/null
  # `.*ckpt.*`, not `^ckpt/.*`: gcloud matches the exclude regex against the
  # full path, not the path relative to the source, so the anchored form
  # silently matched nothing (observed: out/ckpt/ filled up anyway).
  gcloud storage rsync -r "$OUT" "$REMOTE/$JOB/out" -x '.*ckpt.*' -q 2>/dev/null
}
( while true; do sync_now; sleep "$SYNC_EVERY"; done ) &
SYNC_PID=$!

finish() {
  rc=$?
  echo "=== finishing rc=$rc $(date -u +%FT%TZ) ==="
  kill "$SYNC_PID" 2>/dev/null
  echo "$rc" > "$OUT/EXIT_CODE"
  date -u +%FT%TZ > "$OUT/FINISHED_AT"
  sync_now
  if [ "$KEEP_VM" != "1" ]; then
    echo "deleting self ($NAME in $ZONE)"
    gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
  else
    echo "--keep-vm set: instance left up. REMEMBER TO DELETE IT."
  fi
}
trap finish EXIT

# --- 2. hard watchdog --------------------------------------------------------
# Independent of everything above: if the job wedges, the VM still dies.
( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")" \
  && echo "WATCHDOG: max-hours reached, shutting down" >> "$LOG" \
  && gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet ) &

# --- 3. GPU preflight --------------------------------------------------------
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; echo "waiting for nvidia driver ($i)"; sleep 5; done
nvidia-smi || { echo "FATAL: no GPU visible on this VM"; echo NO_GPU > "$OUT/FAILED"; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- 4. container runtime -----------------------------------------------------
# The DLVM boot image ships the NVIDIA DRIVER (the hard part) but NOT Docker,
# and therefore not the NVIDIA container toolkit either. Measured on
# common-cu129-ubuntu-2204-nvidia-580-v20260909: `docker: command not found`.
# So we install both here, idempotently.
#
# These two host packages are NOT version-pinned, unlike everything else in
# this path. That is a deliberate line: the host contributes a driver and a
# container runtime, while the image contributes python/jax/CUDA -- and it is
# the latter that determines a result. Pinning the host runtime too would mean
# maintaining a custom boot image for no reproducibility gain.
ensure_docker_gpu() {
  if docker info >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
    echo "[runtime] docker + nvidia runtime already present"; return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  if ! command -v docker >/dev/null 2>&1; then
    for i in 1 2 3; do apt-get update -y && apt-get install -y docker.io && break; sleep 15; done
  fi
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    install -m 0755 -d /usr/share/keyrings
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    for i in 1 2 3; do apt-get update -y && apt-get install -y nvidia-container-toolkit && break; sleep 15; done
  fi
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  systemctl enable --now docker
}
ensure_docker_gpu
command -v docker >/dev/null 2>&1 || { echo "FATAL: no docker after install"; echo NO_DOCKER > "$OUT/FAILED"; exit 4; }
# No separate GPU probe container: the payload runs with --require-gpu and
# fails loudly on a cpu backend, which is the same check one layer later.

# --- 5. image ----------------------------------------------------------------
gcloud auth configure-docker "$(echo "$IMAGE" | cut -d/ -f1)" --quiet
docker pull "$IMAGE" || { echo "FATAL: docker pull failed"; echo PULL_FAILED > "$OUT/FAILED"; exit 3; }
docker image inspect "$IMAGE" --format 'image digests: {{join .RepoDigests ","}}'

# --- 6. run ------------------------------------------------------------------
# --network host: the container reaches the metadata server for both its SA
#   token (gcloud -> GCS) and the `instance/preempted` long-poll.
# /var/chlu mount: checkpoints + outputs land on the host too, so the host sync
#   loop is a second, independent path to the bucket.
set -x
ENVARGS=()
if [ -n "$CHLU_ENV" ]; then
  IFS=';' read -ra _kvs <<< "$CHLU_ENV"
  for kv in "${_kvs[@]}"; do [ -n "$kv" ] && ENVARGS+=(-e "$kv"); done
  echo "[runtime] extra container env: $CHLU_ENV"
fi
docker run --rm --name chlu-job --gpus all --network host --shm-size=1g \
  -v /var/chlu:/var/chlu \
  ${ENVARGS[@]+"${ENVARGS[@]}"} \
  -e CHLU_JOB="$JOB" \
  "$IMAGE" \
  python /opt/chlu/scripts/cloud/payload_smoke.py \
     --job "$JOB" --out "$OUT" --remote "$REMOTE" --require-gpu $ARGS
RC=$?
set +x
echo "payload exited rc=$RC"
[ "$RC" -eq 0 ] && echo OK > "$OUT/DONE" || echo "$RC" > "$OUT/FAILED"
exit "$RC"
