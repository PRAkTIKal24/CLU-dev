#!/usr/bin/env bash
# scripts/cloud/build_image.sh -- build the pinned CHLU CUDA image, print its DIGEST.
#
# The build runs on a throwaway GCE VM, not on the laptop, for three reasons:
#   * the image is ~6 GB of linux/amd64 CUDA wheels; cross-building that under
#     emulation on an arm64 Mac is slow and the push would go over home uplink,
#   * the builder pulls from PyPI and pushes to Artifact Registry entirely
#     inside Google's network,
#   * the laptop's Docker daemon is not a dependency of the cloud path.
#
# The builder is fully unattended (no SSH, no key material): the context goes
# up through GCS, the startup script builds/pushes/logs, and THE VM DELETES
# ITSELF -- a build failure cannot leave an instance billing.
#
# Usage:  scripts/cloud/build_image.sh [--keep-builder]
# Output: the image reference pinned BY DIGEST, written to scripts/cloud/IMAGE_DIGEST
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=scripts/cloud/config.sh
source "$HERE/config.sh"

KEEP_BUILDER=0
[[ "${1:-}" == "--keep-builder" ]] && KEEP_BUILDER=1

BUILD_ID="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BUILDER="chlu-builder-$(echo "$BUILD_ID" | tr '[:upper:]' '[:lower:]')"
STAGE="gs://${CHLU_BUCKET}/build/${BUILD_ID}"
IMAGE="${CHLU_AR_HOST}/${CHLU_PROJECT}/${CHLU_AR_REPO}/${CHLU_IMAGE_NAME}"

echo "== build ${BUILD_ID} (commit ${GIT_COMMIT}) -> ${IMAGE}"

# ---- 1. context -------------------------------------------------------------
# uv.lock is gitignored (CSF3 README §3.2) -- it must be shipped explicitly or
# the build would resolve fresh and silently drift off the laptop's pins.
[[ -f "$REPO/uv.lock" ]] || { echo "FATAL: $REPO/uv.lock missing -- run 'uv lock' first"; exit 1; }
# The lock must be the one that belongs to THIS checkout. `uv sync --frozen`
# does not catch extras-only drift between pyproject.toml and uv.lock -- a lock
# copied in from another branch builds green and yields an image whose
# provenance is a different commit's. (Observed while building this path.)
if git -C "$REPO" ls-files --error-unmatch uv.lock >/dev/null 2>&1 \
   && ! git -C "$REPO" diff --quiet -- uv.lock; then
  echo "FATAL: $REPO/uv.lock differs from the checked-out commit. The image would"
  echo "       record commit ${GIT_COMMIT} but be built from a different lock."
  echo "       Commit the lock, or 'git checkout -- uv.lock', then re-run."
  exit 1
fi
CTX="$(mktemp -d)"; trap 'rm -rf "$CTX"' EXIT
tar -czf "$CTX/context.tar.gz" -C "$REPO" \
    pyproject.toml uv.lock README.md chlu scripts/cloud
echo "   context $(du -h "$CTX/context.tar.gz" | cut -f1)"
gcloud storage cp "$CTX/context.tar.gz" "$STAGE/context.tar.gz" --project="$CHLU_PROJECT" -q

# ---- 2. builder startup script ---------------------------------------------
cat > "$CTX/startup.sh" <<STARTUP
#!/bin/bash
exec > >(tee -a /var/log/chlu-build.log) 2>&1
set -x
ZONE=\$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print \$NF}')
finish() {
  rc=\$?
  gcloud storage cp /var/log/chlu-build.log "${STAGE}/build.log" -q || true
  if [ \$rc -ne 0 ]; then echo "\$rc" > /tmp/FAILED; gcloud storage cp /tmp/FAILED "${STAGE}/FAILED" -q || true; fi
  if [ "${KEEP_BUILDER}" != "1" ]; then
    gcloud compute instances delete "${BUILDER}" --zone="\$ZONE" --quiet || true
  fi
}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update -y && apt-get install -y docker.io && break; sleep 20; done
systemctl enable --now docker

gcloud auth configure-docker "${CHLU_AR_HOST}" --quiet

mkdir -p /build && cd /build
gcloud storage cp "${STAGE}/context.tar.gz" . -q
tar -xzf context.tar.gz

docker build -f scripts/cloud/Dockerfile \
  --build-arg GIT_COMMIT="${GIT_COMMIT}" --build-arg BUILD_ID="${BUILD_ID}" \
  -t "${IMAGE}:${BUILD_ID}" -t "${IMAGE}:latest" .
docker push "${IMAGE}:${BUILD_ID}"
docker push "${IMAGE}:latest"

# The digest is what we actually ship. A tag is a mutable pointer; recording it
# instead of the digest is how a "reproducible" re-run quietly changes.
DIGEST=\$(docker inspect --format='{{index .RepoDigests 0}}' "${IMAGE}:${BUILD_ID}" | awk -F@ '{print \$2}')
echo "\$DIGEST" > /tmp/DIGEST
gcloud storage cp /tmp/DIGEST "${STAGE}/DIGEST" -q
echo "BUILD OK \$DIGEST"
STARTUP

# ---- 3. builder VM ----------------------------------------------------------
gcloud compute instances create "$BUILDER" \
  --project="$CHLU_PROJECT" --zone="$CHLU_BUILDER_ZONE" \
  --machine-type="$CHLU_BUILDER_TYPE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --scopes=cloud-platform --metadata-from-file=startup-script="$CTX/startup.sh" \
  --labels=owner=chlu,role=builder >/dev/null
echo "   builder ${BUILDER} up (${CHLU_BUILDER_TYPE}, ${CHLU_BUILDER_ZONE})"

# ---- 4. wait ----------------------------------------------------------------
echo -n "   building"
for _ in $(seq 1 180); do   # 180 * 20s = 60 min ceiling
  if gcloud storage cat "$STAGE/DIGEST" --project="$CHLU_PROJECT" 2>/dev/null > "$CTX/digest"; then
    DIGEST="$(tr -d '[:space:]' < "$CTX/digest")"
    echo " ok"
    echo "${IMAGE}@${DIGEST}" | tee "$HERE/IMAGE_DIGEST"
    # protocol: agent artefacts live under .claude/, never in the repo root
    mkdir -p "$REPO/.claude/outputs/gen-cloud-enablement"
    gcloud storage cp "$STAGE/build.log" \
      "$REPO/.claude/outputs/gen-cloud-enablement/build-${BUILD_ID}.log" -q 2>/dev/null || true
    exit 0
  fi
  if gcloud storage cat "$STAGE/FAILED" --project="$CHLU_PROJECT" >/dev/null 2>&1; then
    echo " FAILED"; echo "   log: gcloud storage cat $STAGE/build.log"; exit 1
  fi
  echo -n "."; sleep 20
done
echo " TIMEOUT after 60 min"; echo "   log: gcloud storage cat $STAGE/build.log"; exit 1
