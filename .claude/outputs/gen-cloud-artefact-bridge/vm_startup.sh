#!/bin/bash
# One-off startup script for the artefact-bridge VERIFY VM. Scratch, not a repo
# file: scripts/cloud/launch.sh owns job launching and belongs to another agent.
# Everything that needs gcloud runs INSIDE the shipped image (which carries the
# pinned CLI); the host contributes docker and a metadata token, nothing else.
set -uo pipefail
LOG=/var/log/chlu-verify.log
exec > >(tee -a "$LOG") 2>&1
echo "=== chlu artefact-bridge verify $(date -u +%FT%TZ) on $(hostname) ==="

md()   { curl -s -f -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { md "attributes/$1"; }

NAME="$(md name)"; ZONE="$(md zone | awk -F/ '{print $NF}')"
IMAGE="$(attr chlu-image)"; BUCKET="$(attr chlu-bucket)"; PROJECT="$(attr chlu-project)"
KEEP="$(attr chlu-keep-vm)"; KEEP="${KEEP:-1}"
GCS_ROOT="gs://${BUCKET}/artefacts"
echo "image=$IMAGE bucket=$BUCKET zone=$ZONE keep=$KEEP"
case "$IMAGE" in *@sha256:*) : ;; *) echo "FATAL: image is not digest-pinned"; exit 2 ;; esac

mkdir -p /var/chlu

# --- docker (Ubuntu ships neither docker nor gcloud) --------------------------
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update -y && apt-get install -y docker.io && break; sleep 15; done
command -v docker >/dev/null || { echo "FATAL: no docker"; exit 3; }

# --- registry auth straight from the metadata server -------------------------
TOKEN="$(curl -s -f -H 'Metadata-Flavor: Google' \
   'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
   | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')"
printf '%s' "$TOKEN" | docker login -u oauth2accesstoken --password-stdin https://us-central1-docker.pkg.dev
docker pull "$IMAGE" || { echo "FATAL: docker pull failed"; exit 4; }
docker image inspect "$IMAGE" --format 'digest: {{join .RepoDigests ","}}'

RUN=(docker run --rm --network host -v /var/chlu:/var/chlu -e CHLU_VM_NAME="$NAME"
     -e CHLU_PROJECT="$PROJECT" -e CHLU_BUCKET="$BUCKET")

# --- the verify script itself (fetched with the image's own gcloud) ----------
"${RUN[@]}" "$IMAGE" gcloud storage cp "$GCS_ROOT/_verify/verify_gcs.sh" /var/chlu/verify_gcs.sh
chmod +x /var/chlu/verify_gcs.sh
# Mounted OVER its place in the image so it resolves config.sh exactly as it
# would in a checkout -- the script under test is the repo's, unmodified.
RUN+=(-v /var/chlu/verify_gcs.sh:/opt/chlu/scripts/cloud/verify_gcs.sh:ro)

echo "=== PHASE 1: the real verify (fetch + checksum + OPEN + dataset) ==="
"${RUN[@]}" "$IMAGE" bash /opt/chlu/scripts/cloud/verify_gcs.sh --with-dataset --allow-pip
echo "PHASE1_RC=$?"

# ---------------------------------------------------------------------------
# PHASE 2 + 3: the checker must be able to FAIL, both ways, or it proves
# nothing. Exactly the two demos the CSF3 verify was accepted on.
# ---------------------------------------------------------------------------
ART=/var/chlu/artefacts
SA="$ART/outputs/gen-cluster-stage-artefacts"
REL2=outputs/gen-feature-bridge/runs/shapes3d_recon/seed_1/feature_rows.npy
REL3=outputs/gen-read-compose/runs/shapes3d_recon/seed_0/store_s0.pkl

echo "=== PHASE 2: one flipped byte (still loads; sha256 must catch it) ==="
python3 - "$ART/$REL2" <<'PYX'
import sys
p = sys.argv[1]
with open(p, "r+b") as f:
    f.seek(2000); b = f.read(1)[0]
    f.seek(2000); f.write(bytes([b ^ 0xFF]))
print("flipped one byte at offset 2000 of", p)
PYX
"${RUN[@]}" "$IMAGE" bash /opt/chlu/scripts/cloud/verify_gcs.sh \
    --no-fetch --no-upload --out "$SA/VM-FAIL-corrupt-byte.json"
echo "PHASE2_RC=$?  (a ZERO here would mean the checker cannot fail)"

echo "=== PHASE 3: byte-perfect but unloadable (sha256 agrees; the LOADER must catch it) ==="
"${RUN[@]}" "$IMAGE" gcloud storage cp "$GCS_ROOT/$REL2" "$ART/$REL2"   # undo phase 2
python3 - "$ART/$REL3" "$REL3" "$SA/stage_manifest.json" <<'PYX'
import hashlib, json, sys
path, rel, mf = sys.argv[1:4]
data = open(path, "rb").read()
half = data[: len(data) // 2]
open(path, "wb").write(half)
m = json.load(open(mf))
for rec in m["files"]:
    if rec["path"] == rel:
        rec["bytes"] = len(half)
        rec["sha256"] = hashlib.sha256(half).hexdigest()
json.dump(m, open(mf, "w"), indent=2, sort_keys=True)
print("truncated %s to %d bytes and re-hashed it IN the manifest -- the checksum"
      " gate now passes it" % (rel, len(half)))
PYX
"${RUN[@]}" "$IMAGE" bash /opt/chlu/scripts/cloud/verify_gcs.sh \
    --no-fetch --no-upload --out "$SA/VM-FAIL-unloadable.json"
echo "PHASE3_RC=$?  (a ZERO here is the exact failure the CSF3 verify was built to catch)"

for f in VM-FAIL-corrupt-byte.json VM-FAIL-unloadable.json; do
  "${RUN[@]}" "$IMAGE" gcloud storage cp "$SA/$f" "$GCS_ROOT/_verify/$f" || true
done

echo "=== done $(date -u +%FT%TZ) ==="
"${RUN[@]}" -v /var/log:/var/log:ro "$IMAGE" \
    gcloud storage cp /var/log/chlu-verify.log "$GCS_ROOT/_verify/vm-verify.log" || true
if [ "$KEEP" != "1" ]; then
  "${RUN[@]}" "$IMAGE" gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
fi
