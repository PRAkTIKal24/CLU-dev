#!/bin/bash
# scripts/cloud/startup.sh -- GCE startup script for a CHLU GPU job VM.
#
# ⭐ ONE LAUNCHER, PARAMETERISED. The experiment is a job parameter (`EXP=`),
# never a copy of this file. Adding an arm is one row in the EXP registry in §0
# below; it is never a second startup script, because two launchers drift and
# only one of them gets the next fix.
#
# Uploaded by launch.sh as the `startup-script` metadata key. Every job
# parameter arrives as a separate metadata attribute, so this file is generic
# and never templated.
#
# Ordering here is deliberate and pre-emption-shaped:
#   1. the sync loop starts BEFORE anything else, so a VM that dies during
#      `docker pull` still leaves its log in the bucket;
#   2. ⭐ the artefact roots and the decoder recipe are asserted BEFORE the GPU
#      preflight, the docker install and the image pull. A job must never reach
#      the GPU to discover it cannot open an artefact, and must never produce a
#      normal-looking log through the decoder that failed the gate;
#   3. completed units are banked in a GCS-backed journal, so a relaunched VM
#      resumes the sweep instead of restarting it;
#   4. the VM deletes itself on ANY exit path, so a crashed job does not bill
#      overnight.
#
# Offline self-test (no GCE, no docker, no network):
#   CHLU_JOB_LOG=/tmp/x.log CHLU_OUT_DIR=/tmp/out CHLU_GEN_DECLIFT_DIR=... \
#     bash startup.sh --check roots|recipe
# which exercises the SHIPPED assertions rather than a copy of them.
set -uo pipefail

LOG="${CHLU_JOB_LOG:-/var/log/chlu-job.log}"
OUT="${CHLU_OUT_DIR:-/var/chlu/out}"
VARCHLU="$(dirname "$OUT")"

# ===========================================================================
# 0. ⭐ THE EXP REGISTRY. One row per experiment; nothing else changes.
#
#    module        the python module run as `python -u -m <module>`
#    out_mode      sweep = --out is the sweep root and the module appends its
#                  own seed_<N>/ ; unit = --out is THIS unit's directory.
#                  (Copied from scripts/csf3/job_cpu_gen.sh §7, which measured
#                  the difference the hard way -- exp_gen_baselines does not
#                  append, exp_gen_atom_mean does.)
#    needs_art     whether the five CHLU_* artefact roots are required. An
#                  experiment that generates or downloads its own stream needs
#                  none, and asserting them would refuse a legitimate job.
# ===========================================================================
exp_row() {   # $1 = EXP -> "module out_mode needs_art"
  case "$1" in
    smoke)           echo ":smoke: sweep 0" ;;
    cluformer)       echo "chlu.experiments.exp_cluformer_pilot sweep 0" ;;
    gen_encoder)     echo "chlu.experiments.exp_gen_encoder     sweep 1" ;;
    gen_store)       echo "chlu.experiments.exp_gen_store       sweep 1" ;;
    gen_cell)        echo "chlu.experiments.exp_gen_cell        sweep 1" ;;
    gen_dream)       echo "chlu.experiments.exp_gen_dream       sweep 1" ;;
    gen_certificate) echo "chlu.experiments.exp_gen_certificate sweep 1" ;;
    gen_atom_mean)   echo "chlu.experiments.exp_gen_atom_mean   sweep 1" ;;
    gen_baselines)   echo "chlu.experiments.exp_gen_baselines   unit  1" ;;
    # w8 hub/encoder-loop. out_mode=sweep: the module appends its own seed_<N>/.
    # needs_art=1: it loads THE external classifier (the invariant instrument) and the
    # dataset. One unit = one seed = warm start + every arm's stage 2, intra-unit
    # checkpointed through scripts/cloud/ckpt.py, so a pre-emption costs the steps since
    # the last checkpoint and not the unit.
    w8_encoder_loop) echo "chlu.experiments.exp_w8_encoder_loop sweep 1" ;;
    *) return 1 ;;
  esac
}
EXP_KNOWN="smoke cluformer gen_encoder gen_store gen_cell gen_dream gen_certificate gen_atom_mean gen_baselines w8_encoder_loop"

#: ⛔ THE DECODER TRAP. chlu/config.py's GenBridgeConfig.dec_recipe still
#: defaults to "baseline" -- the recipe that FAILED the gate (0.7191 balanced
#: accuracy, 0/3 seeds) -- deliberately, so Montero's history reproduces. A run
#: decoded through it looks ENTIRELY NORMAL in the logs and is void. So the
#: winner is asserted here by NAME and by HASH before the GPU is touched.
EXPECTED_DEC_RECIPE="v1_capacity_schedule"

# ===========================================================================
# 1. ⭐ The artefact roots -- the five CHLU_* variables, each ${VAR:-default}.
#    Defaults mirror scripts/csf3/stage_artefacts.sh's published layout, so the
#    GCS prefix and the CSF3 scratch tree are the same tree in two places.
# ===========================================================================
export CHLU_ARTEFACT_DIR="${CHLU_ARTEFACT_DIR:-${VARCHLU}/artefacts}"
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CHLU_ARTEFACT_DIR}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"
CHLU_ROOT_VARS="CHLU_CLAUDE_ROOT CHLU_GEN_DATA_ROOT CHLU_GEN_BRIDGE_ROOT CHLU_GEN_STORE_RUNS CHLU_GEN_DECLIFT_DIR"

# ⚠ NOT one of the five, and deliberately not asserted: CHLU_DATA_ROOT is the
# DOWNLOAD CACHE (chlu.data.industrial.base.default_data_root), not a staged
# artefact. Its default is $HOME/.cache, which inside a --rm container is
# thrown away -- so a sweep of N units would download enwik8 N
# times. Pointing it at the host mount makes it one download per VM.
export CHLU_DATA_ROOT="${CHLU_DATA_ROOT:-${VARCHLU}/data}"

assert_roots() {
  # ⛔ Name the missing one. "artefacts missing" an hour into a GPU hour is the
  # expensive failure; "CHLU_GEN_STORE_RUNS -> /var/chlu/artefacts/outputs/..."
  # is a fix.
  # ⚠ Not a bare `-e`, unlike scripts/csf3/job_cpu_gen.sh: this file MAKES
  # $CHLU_ARTEFACT_DIR before the fetch, so testing a directory for existence
  # asks a question we already answered ourselves. An EMPTY root is a failed
  # fetch, not a staged set -- measured on `gate1`, where the four derived roots
  # fired and CHLU_CLAUDE_ROOT vacuously passed.
  local missing="" v p
  for v in $CHLU_ROOT_VARS; do
    eval "p=\$$v"
    if [ -d "$p" ]; then
      [ -n "$(ls -A "$p" 2>/dev/null)" ] || missing="${missing}
    $v -> $p   (EMPTY -- the fetch produced nothing)"
    elif [ ! -e "$p" ]; then
      missing="${missing}
    $v -> $p"
    fi
  done
  if [ -n "$missing" ]; then
    echo "FATAL[startup]: gen artefact root(s) MISSING, by name:$missing" >&2
    echo "       Stage them with scripts/cloud/stage_gcs.sh and pass --artefacts" >&2
    echo "       gs://<bucket>/artefacts (⛔ do not improvise a staging path), or" >&2
    echo "       set CHLU_SKIP_ARTEFACT_CHECK=1 if this run genuinely needs none." >&2
    return 2
  fi
  echo "[artefacts] all five roots present: CHLU_CLAUDE_ROOT=$CHLU_CLAUDE_ROOT"
  return 0
}

# ===========================================================================
# 2. ⭐ The decoder recipe, asserted by NAME *and* by HASH.
#    Two independent checks because either alone is weak: the name catches
#    "the config silently defaulted to baseline", the hash catches "the right
#    name pointing at the wrong bytes". Both read gen-decoder-lift's own
#    winner_hashes.json, which is the authority the experiments use.
# ===========================================================================
assert_dec_recipe() {   # $1 = requested recipe, $2... = units (seeds)
  local want="$1"; shift
  python3 - "$CHLU_GEN_DECLIFT_DIR" "$want" "$EXPECTED_DEC_RECIPE" "$@" <<'PYDEC'
import hashlib, json, os, sys

root, want, expected, units = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
fail = []

# (a) BY NAME -- against the pinned winner, not against "anything but baseline".
if want != expected:
    fail.append(
        f"dec_recipe is {want!r}, not {expected!r}. 'baseline' is the SUPERSEDED "
        "decoder that FAILED the gate (0.7191 balanced accuracy, 0/3 seeds); a run "
        "through it looks entirely normal in the logs and is VOID."
    )

hp = os.path.join(root, "winner_hashes.json")
try:
    with open(hp) as fh:
        hashes = json.load(fh)
except OSError as exc:
    fail.append(f"cannot read the winner hashes at {hp}: {exc}")
    hashes = {}

for u in units:
    key = f"seed_{u}"
    rec = hashes.get(key)
    if rec is None:
        fail.append(f"{key} has no entry in {hp} (have: {', '.join(sorted(hashes)) or 'nothing'})")
        continue
    # (a') BY NAME again, from the artefact rather than from our own flag: the
    #      staged decoder must itself declare the winning recipe.
    got_name = (rec.get("recipe") or {}).get("name")
    if got_name != expected:
        fail.append(f"{key}: staged decoder declares recipe {got_name!r}, expected {expected!r}")
    # (b) BY HASH -- the bytes actually staged.
    pkl = os.path.join(root, f"decoder_v1_s{u}.pkl")
    if not os.path.exists(pkl):
        fail.append(f"{key}: {pkl} is missing (winner_decoder_pattern 'decoder_v1_s{{seed}}.pkl')")
        continue
    h = hashlib.sha256()
    with open(pkl, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    exp_h = rec.get("pkl_sha256")
    if got != exp_h:
        fail.append(f"{key}: {os.path.basename(pkl)} sha256 {got} != winner_hashes {exp_h}")
    else:
        print(f"[decoder] {key}: recipe {got_name} · pkl sha256 {got[:16]}… OK "
              f"(param_hash {str(rec.get('param_hash'))[:16]}…, {rec.get('param_count')} params)")

if fail:
    sys.stdout.flush()
    print("FATAL[startup]: DECODER RECIPE ASSERTION FAILED -- refusing to run.", file=sys.stderr)
    for f in fail:
        print(f"    - {f}", file=sys.stderr)
    print("    ⛔ Every number this job would produce is void. Nothing was run.", file=sys.stderr)
    raise SystemExit(5)
print(f"[decoder] recipe {want!r} asserted by name and by hash for seed(s): {' '.join(units)}")
PYDEC
}

# ===========================================================================
# 3. Offline self-test hook. Runs ONLY the assertion being named, against the
#    environment, so tests exercise the shipped code path. GCE never passes
#    arguments, so this cannot fire on a real VM.
# ===========================================================================
if [ "${1:-}" = "--check" ]; then
  case "${2:-}" in
    roots)  assert_roots; exit $? ;;
    recipe) shift 2; assert_dec_recipe "${CHLU_DEC_RECIPE:-$EXPECTED_DEC_RECIPE}" "$@"; exit $? ;;
    exp)    exp_row "${3:-}" || { echo "unknown EXP '${3:-}'. Known: $EXP_KNOWN" >&2; exit 6; }; exit 0 ;;
    *) echo "usage: startup.sh --check roots|recipe|exp [args]" >&2; exit 64 ;;
  esac
fi

# ===========================================================================
# 4. On a real VM from here down.
# ===========================================================================
mkdir -p "$(dirname "$LOG")" 2>/dev/null
exec > >(tee -a "$LOG") 2>&1
echo "=== chlu startup $(date -u +%FT%TZ) on $(hostname) ==="

md() { curl -s -f --connect-timeout 2 --max-time 10 -H 'Metadata-Flavor: Google' \
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

# --- the job parameters that make this launcher generic ----------------------
EXP="$(attr chlu-exp)"; EXP="${EXP:-smoke}"
EXP_ARGS="$(attr chlu-exp-args)"; EXP_ARGS="${EXP_ARGS:-}"
UNITS="$(attr chlu-units)"; UNITS="${UNITS:-0}"
ARTEFACTS="$(attr chlu-artefacts)"; ARTEFACTS="${ARTEFACTS:-}"
SKIP_ART="$(attr chlu-skip-artefact-check)"; SKIP_ART="${SKIP_ART:-0}"
DEC_RECIPE="$(attr chlu-dec-recipe)"; DEC_RECIPE="${DEC_RECIPE:-$EXPECTED_DEC_RECIPE}"
GIT_COMMIT="$(attr chlu-git-commit)"; GIT_COMMIT="${GIT_COMMIT:-unknown}"

mkdir -p "$OUT" "$VARCHLU/jax_cache" "$VARCHLU/journal" "$VARCHLU/data" "$CHLU_ARTEFACT_DIR"
echo "job=$JOB exp=$EXP units=[$UNITS] image=$IMAGE remote=$REMOTE zone=$ZONE"
echo "exp_args=[$EXP_ARGS] artefacts=[$ARTEFACTS] commit=$GIT_COMMIT"

# --- the image reference must be digest-pinned --------------------------------
# A tag is a mutable pointer. Accepting one here would make every downstream
# claim about "the image we ran" unverifiable, so refuse rather than warn.
case "$IMAGE" in
  *@sha256:*) : ;;
  *) echo "FATAL: image '$IMAGE' is not pinned by digest"; echo FATAL_NOT_PINNED > "$OUT/FAILED" ;;
esac

# --- 5. sync-back loop, started FIRST ----------------------------------------
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

# --- 6. hard watchdog --------------------------------------------------------
# Independent of everything above: if the job wedges, the VM still dies.
( sleep "$(python3 -c "print(int(float('$MAX_HOURS')*3600))")" \
  && echo "WATCHDOG: max-hours reached, shutting down" >> "$LOG" \
  && gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet ) &

# --- 7. resolve the EXP row ---------------------------------------------------
if ! ROW="$(exp_row "$EXP")"; then
  echo "FATAL: unknown EXP '$EXP'. Known: $EXP_KNOWN" >&2
  echo "       ⛔ Do not add a second launcher -- add a row to exp_row() in this file." >&2
  echo UNKNOWN_EXP > "$OUT/FAILED"; exit 6
fi
read -r MODULE OUT_MODE NEEDS_ART <<<"$ROW"
echo "[exp] $EXP -> module=$MODULE out_mode=$OUT_MODE needs_artefacts=$NEEDS_ART"

# --- 8. ⭐ restore this job's outputs BEFORE anything runs ---------------------
# This is what makes a relaunch a RESUME. A new VM has an empty disk; the
# previous attempt's partial outputs (and the experiment's own resume state)
# are in the bucket. Additive rsync, never --delete: an attempt may have
# produced files this one has not yet.
if [ -n "$REMOTE" ]; then
  echo "[resume] restoring $REMOTE/$JOB/out -> $OUT"
  # Same `-x '.*ckpt.*'` as the upload leg, for the same reason and in the other
  # direction: checkpoints live at $REMOTE/$JOB/ckpt and the Checkpointer fetches
  # the one it needs itself. Pulling them into out/ would re-create the duplicate
  # prefix that nothing prunes.
  gcloud storage rsync -r "$REMOTE/$JOB/out" "$OUT" -x '.*ckpt.*' -q 2>/dev/null
  rm -f "$OUT/FAILED" "$OUT/PREEMPTED" "$OUT/EXIT_CODE" "$OUT/DONE"
fi

# --- 9. ⭐ artefacts: fetch, then ASSERT, BEFORE the GPU ----------------------
# Deliberately ahead of the driver wait, the docker install and the 7 GB image
# pull. All three cost minutes; a missing artefact costs nothing to detect.
if [ "$NEEDS_ART" = "1" ] && [ "$SKIP_ART" != "1" ]; then
  if [ -n "$ARTEFACTS" ]; then
    echo "[artefacts] fetching $ARTEFACTS -> $CHLU_ARTEFACT_DIR"
    # ⚠ `cp -r`, deliberately not `rsync`. Every rsync on this path carries
    # `-x '.*ckpt.*'` so the two syncers cannot duplicate checkpoints; applying
    # that filter to the ARTEFACT set would silently drop a staged file whose
    # name happens to contain "ckpt" -- a missing artefact that looks like a
    # successful fetch, which is exactly what verify_gcs.sh exists to catch.
    # A fresh VM starts empty, so there is nothing for rsync to skip anyway.
    gcloud storage cp -r "$ARTEFACTS/*" "$CHLU_ARTEFACT_DIR/" 2>&1 | tail -5
  else
    echo "[artefacts] ⚠ no --artefacts prefix given; asserting the roots as they stand"
  fi
  assert_roots || { echo MISSING_ARTEFACT_ROOTS > "$OUT/FAILED"; exit 2; }
  assert_dec_recipe "$DEC_RECIPE" $UNITS || { echo BAD_DEC_RECIPE > "$OUT/FAILED"; exit 5; }
  NEED_DATASET=1
elif [ "$NEEDS_ART" = "1" ]; then
  echo "[artefacts] ⚠ CHLU_SKIP_ARTEFACT_CHECK=1 -- roots and decoder NOT asserted"
else
  echo "[artefacts] $EXP needs none (it generates or downloads its own stream)"
fi

# ⭐ 9b. THE DATASET. ⛔ Not staged, by `gen-cloud-artefact-bridge`'s measured
#     decision: 3dshapes.h5 comes from a PUBLIC bucket whose owner pays the
#     egress, so a VM download costs $0.00 while staging it would cost storage
#     and a 255 MiB upload. ⭐ Its report calls this "the only place the two
#     platforms differ" and hands the wiring here -- so here it is, and the
#     runner is the reason a gen job finds its dataset at all.
#
#     `download_shapes3d` / `download_dsprites` ARE the production path (the one
#     `chlu data <ds> --download` calls). They assert sha256 AND the exact byte
#     count, so a truncated fetch fails loudly instead of yielding a short file.
#     ⛔ The CLI command is not used because it also BUILDS THE SPLIT, which the
#     arm builds for itself moments later; the download is the part a job needs.
#
#     The dataset is read out of the experiment's own --dataset argument: every
#     gen module takes one, so there is no second place to keep it in sync.
DATASET=""
case " $EXP_ARGS " in
  *" --dataset "*) DATASET="$(printf '%s' "$EXP_ARGS" | sed -n 's/.*--dataset[= ]\([^ ]*\).*/\1/p')" ;;
esac

# --- 10. GPU preflight --------------------------------------------------------
for i in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; echo "waiting for nvidia driver ($i)"; sleep 5; done
nvidia-smi || { echo "FATAL: no GPU visible on this VM"; echo NO_GPU > "$OUT/FAILED"; exit 2; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# --- 11. container runtime ----------------------------------------------------
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

# --- 12. image ----------------------------------------------------------------
gcloud auth configure-docker "$(echo "$IMAGE" | cut -d/ -f1)" --quiet
docker pull "$IMAGE" || { echo "FATAL: docker pull failed"; echo PULL_FAILED > "$OUT/FAILED"; exit 3; }
docker image inspect "$IMAGE" --format 'image digests: {{join .RepoDigests ","}}'
docker run --rm "$IMAGE" python -c \
  'import os;print("[image] built from commit", os.environ.get("CHLU_IMAGE_GIT_COMMIT"), "build", os.environ.get("CHLU_IMAGE_BUILD_ID"))'

# --- 13. container env --------------------------------------------------------
# --network host: the container reaches the metadata server for both its SA
#   token (gcloud -> GCS) and the `instance/preempted` long-poll.
# /var/chlu mount: checkpoints + outputs land on the host too, so the host sync
#   loop is a second, independent path to the bucket.
ENVARGS=()
if [ -n "$CHLU_ENV" ]; then
  IFS=';' read -ra _kvs <<< "$CHLU_ENV"
  for kv in "${_kvs[@]}"; do [ -n "$kv" ] && ENVARGS+=(-e "$kv"); done
  echo "[runtime] extra container env: $CHLU_ENV"
fi
for v in $CHLU_ROOT_VARS CHLU_DATA_ROOT; do eval "ENVARGS+=(-e $v=\$$v)"; done

docker_run() {   # $@ = the command inside the container
  docker run --rm -i --name chlu-job --gpus all --network host --shm-size=1g \
    -v "$VARCHLU:$VARCHLU" \
    ${ENVARGS[@]+"${ENVARGS[@]}"} \
    -e CHLU_JOB="$JOB" -e CHLU_DEC_RECIPE="$DEC_RECIPE" \
    "$IMAGE" "$@"
}

# ===========================================================================
# 14. ⭐ THE UNIT JOURNAL -- scripts/cloud/ckpt.py, REUSED.
#     ⛔ NOT a second checkpoint mechanism. The ledger is one more Checkpointer
#     with its own prefix: it inherits atomic write, GCS publication at save
#     time, keep-last-N across BOTH stores, and -- the part that matters here --
#     the fingerprint refusal, so a relaunch under a changed config is rejected
#     by name instead of silently half-resumed.
#     ⛔ SEEDS are not in the fingerprint: narrowing or extending the unit list
#     is the normal way to repair a sweep, and the ledger is per-unit precisely
#     so that is safe.
# ===========================================================================
JOB_CFG_JSON="$(python3 - "$EXP" "$MODULE" "$OUT_MODE" "$EXP_ARGS" "$DEC_RECIPE" "$IMAGE" <<'PYCFG'
import json, sys
k = ["exp", "module", "out_mode", "exp_args", "dec_recipe", "image"]
print(json.dumps(dict(zip(k, sys.argv[1:])), sort_keys=True))
PYCFG
)"

journal() {   # $1 = init|is_done|mark|list   $2 = unit
  docker run --rm -i --network host -v "$VARCHLU:$VARCHLU" \
    -e CHLU_J_OP="$1" -e CHLU_J_UNIT="${2:-}" -e CHLU_J_CFG="$JOB_CFG_JSON" \
    -e CHLU_J_LOCAL="$VARCHLU/journal" -e CHLU_J_REMOTE="$REMOTE/$JOB/journal" \
    "$IMAGE" python - <<'PYJ'
import json, os, sys
sys.path.insert(0, "/opt/chlu/scripts/cloud")
import numpy as np
from ckpt import Checkpointer, FingerprintMismatch, config_fingerprint

op = os.environ["CHLU_J_OP"]
unit = os.environ.get("CHLU_J_UNIT", "")
cfg = json.loads(os.environ["CHLU_J_CFG"])
fp = config_fingerprint(cfg)
cp = Checkpointer(os.environ["CHLU_J_LOCAL"], fp,
                  remote_uri=(os.environ.get("CHLU_J_REMOTE") or None),
                  every_s=0.0, keep=3)


def read():
    try:
        found = cp.latest()
    except FingerprintMismatch as exc:
        print("FATAL[journal]: %s" % exc, file=sys.stderr)
        print("    ⛔ REFUSING to resume: this VM was launched with a different "
              "config than the one that banked these units.", file=sys.stderr)
        raise SystemExit(3)
    return [] if found is None else json.loads(bytes(found[1]["units"]).decode())


if op == "init":
    done = read()
    print("[journal] fingerprint %s :: %d unit(s) already done: %s"
          % (fp, len(done), " ".join(done) or "[]"))
elif op == "is_done":
    raise SystemExit(0 if unit in read() else 1)
elif op == "list":
    print(json.dumps(read()))
elif op == "mark":
    done = read()
    if unit not in done:
        done.append(unit)
    cp.save(len(done), {"units": np.frombuffer(json.dumps(done).encode(), dtype=np.uint8)})
    print("[journal] unit %s banked (%d done) -> %s" % (unit, len(done), cp.remote_uri))
else:
    raise SystemExit("unknown journal op %r" % op)
PYJ
}

journal init || { echo JOURNAL_REFUSED > "$OUT/FAILED"; exit 7; }

if [ "${NEED_DATASET:-0}" = "1" ] && [ -n "$DATASET" ]; then
  echo "[dataset] fetching '$DATASET' into \$CHLU_GEN_DATA_ROOT=$CHLU_GEN_DATA_ROOT"
  docker_run python - "$DATASET" <<'PYDS' || { echo DATASET_FETCH_FAILED > "$OUT/FAILED"; exit 8; }
import sys

name = sys.argv[1]
if name == "shapes3d":
    from chlu.data.shapes3d import download_shapes3d as fetch
elif name == "dsprites":
    from chlu.data.dsprites import download_dsprites as fetch
else:
    raise SystemExit(f"FATAL[startup]: unknown --dataset {name!r} (shapes3d | dsprites)")

for fname, rec in fetch(None, verify=True).items():
    print(f"[dataset] {name}/{fname}: {rec['bytes']} bytes sha256={rec['sha256']}", flush=True)
PYDS
elif [ "${NEED_DATASET:-0}" = "1" ]; then
  echo "[dataset] ⚠ no --dataset in the experiment args; nothing fetched"
fi

# ===========================================================================
# 15. ⭐ THE SWEEP. One unit per seed. A pre-emption costs ONE UNIT.
# ===========================================================================
unit_out() {   # $1 = unit
  case "$OUT_MODE" in
    unit) echo "$OUT/$EXP/seed_$1" ;;
    *)    echo "$OUT/$EXP" ;;
  esac
}

RC=0
DONE_UNITS=""
SKIPPED=""
T_SWEEP="$(date -u +%s)"

if [ "$EXP" = "smoke" ]; then
  # ⛔ NO SCIENCE. The synthetic payload the enablement spoke accepted on. Kept
  # byte-compatible so its five-run hash-chain evidence still reproduces, and
  # because every piece of this file has to be testable without an artefact set.
  docker_run python /opt/chlu/scripts/cloud/payload_smoke.py \
     --job "$JOB" --out "$OUT" --remote "$REMOTE" --require-gpu $ARGS
  RC=$?
else
  for u in $UNITS; do
    if journal is_done "$u"; then
      echo "[sweep] unit $u already banked -- skipping"
      SKIPPED="$SKIPPED $u"
      continue
    fi
    uo="$(unit_out "$u")"
    mkdir -p "$uo"
    echo "=== [sweep] unit $u -> $uo :: python -u -m $MODULE --seed $u --out $uo $EXP_ARGS"
    t0="$(date -u +%s)"
    docker_run python -u -m "$MODULE" --seed "$u" --out "$uo" $EXP_ARGS
    rc=$?
    t1="$(date -u +%s)"
    if [ "$rc" -ne 0 ]; then
      echo "[sweep] ⛔ unit $u FAILED rc=$rc after $((t1-t0))s -- NOT banked; it re-runs on the next attempt"
      RC=$rc
      break
    fi
    echo "[sweep] unit $u ok in $((t1-t0))s"
    journal mark "$u"
    DONE_UNITS="$DONE_UNITS $u"
    sync_now
  done
fi

# ===========================================================================
# 16. ⭐ result.json -- the launcher's completion sentinel.
#     run_until_done.sh relaunches while this object is ABSENT, so it is written
#     only when every unit is banked. ⛔ It is not a result: the experiment's own
#     manifests are, and they are under out/<exp>/.
# ===========================================================================
if [ "$EXP" != "smoke" ] && [ "$RC" -eq 0 ]; then
  BANKED="$(journal list | tail -1)"
  python3 - "$OUT/result.json" "$JOB" "$EXP" "$MODULE" "$UNITS" "$BANKED" \
           "$GIT_COMMIT" "$IMAGE" "$DEC_RECIPE" "$(( $(date -u +%s) - T_SWEEP ))" <<'PYRES'
import json, os, socket, sys, time
p, job, exp, module, units, banked, commit, image, recipe, wall = sys.argv[1:11]
banked = json.loads(banked)
units = units.split()
rec = {
    "job": job, "exp": exp, "module": module,
    "units_requested": units, "units_banked": banked,
    "complete": sorted(banked) == sorted(units),
    "git_commit": commit, "image": image, "dec_recipe": recipe,
    "sweep_wall_s": int(wall), "hostname": socket.gethostname(),
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "note": "launcher completion sentinel, not a result; the experiment's own "
            "manifests are under out/<exp>/",
}
if not rec["complete"]:
    print("[result] NOT complete: banked %s of %s -- no sentinel written" % (banked, units))
    raise SystemExit(1)
tmp = p + ".tmp"
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=2, sort_keys=True)
os.replace(tmp, p)
print("[result] " + json.dumps(rec, sort_keys=True))
PYRES
fi

echo "payload exited rc=$RC (banked this attempt:${DONE_UNITS:- none}; skipped:${SKIPPED:- none})"
[ "$RC" -eq 0 ] && echo OK > "$OUT/DONE" || echo "$RC" > "$OUT/FAILED"
sync_now
exit "$RC"
