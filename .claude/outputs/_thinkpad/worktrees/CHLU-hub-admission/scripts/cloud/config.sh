# scripts/cloud/config.sh -- shared settings for the GCP path. Sourced, not run.
#
# Everything here is overridable from the environment, so a second project or
# region needs no edit to a script.

# --- project / region --------------------------------------------------------
# us-central1 is where every prior job in this project ran, and where the
# measured quota lives. Do NOT change it casually: the L4 and A100-SPOT quota
# below is per-region, and another team holds live L4 capacity here.
export CHLU_PROJECT="${CHLU_PROJECT:-forgisprova}"
export CHLU_REGION="${CHLU_REGION:-us-central1}"
export CHLU_ZONE="${CHLU_ZONE:-us-central1-a}"

# --- artefact locations ------------------------------------------------------
export CHLU_BUCKET="${CHLU_BUCKET:-chlu-cloud-forgisprova}"   # checkpoints + results
export CHLU_AR_REPO="${CHLU_AR_REPO:-chlu}"
export CHLU_AR_HOST="${CHLU_AR_HOST:-${CHLU_REGION}-docker.pkg.dev}"
export CHLU_IMAGE_NAME="${CHLU_IMAGE_NAME:-chlu-cuda}"

# --- builder -----------------------------------------------------------------
export CHLU_BUILDER_TYPE="${CHLU_BUILDER_TYPE:-e2-standard-8}"
export CHLU_BUILDER_ZONE="${CHLU_BUILDER_ZONE:-$CHLU_ZONE}"

# --- GPU boot image ----------------------------------------------------------
# Pinned to an exact image NAME, never a family: an image family is a moving
# pointer, exactly like a docker tag. Driver 580.x here matches CSF3's
# 580.126.09, so the same bundled-CUDA-wheel argument holds unchanged.
# It ships Docker + the NVIDIA container toolkit, so the VM contributes the
# driver and nothing else -- no host CUDA to shadow the wheels.
export CHLU_BOOT_IMAGE="${CHLU_BOOT_IMAGE:-common-cu129-ubuntu-2204-nvidia-580-v20260909}"
export CHLU_BOOT_IMAGE_PROJECT="${CHLU_BOOT_IMAGE_PROJECT:-deeplearning-platform-release}"

# --- card catalogue ----------------------------------------------------------
# The catalogue itself lives in cards.json so that bash (here) and python
# (quota.py) cannot drift apart. These are thin readers, not a second copy.
CHLU_CARDS_JSON="${CHLU_CARDS_JSON:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cards.json}"
export CHLU_CARDS_JSON

chlu_card_spec() {   # $1=card -> "accel machine gpu_metric cpu_metric vcpu"
  "${PYTHON:-python3}" - "$CHLU_CARDS_JSON" "$1" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))["cards"]
if sys.argv[2] not in c:
    sys.exit(f"unknown card {sys.argv[2]!r}; known: {', '.join(sorted(c))}")
s = c[sys.argv[2]]
print(s["accelerator"], s["machine_type"], s["gpu_metric"], s["cpu_metric"], s["vcpu"])
PYEOF
}

chlu_price() {   # $1=card $2=spot(0|1) -> whole-VM USD/hour
  "${PYTHON:-python3}" - "$CHLU_CARDS_JSON" "$1" "$2" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))["cards"].get(sys.argv[2])
print(0 if not c else c["usd_per_hour"]["spot" if sys.argv[3] == "1" else "ondemand"])
PYEOF
}

chlu_cards() { "${PYTHON:-python3}" -c \
  'import json,sys;print(" ".join(sorted(json.load(open(sys.argv[1]))["cards"])))' "$CHLU_CARDS_JSON"; }

# List prices are ESTIMATES from the public price list, captured with cards.json.
# They exist so a run's spend is visible while it runs. They are not an invoice:
# spot rates float, and sustained-use/committed discounts are not modelled.
export CHLU_PRICES_AS_OF="2026-09-15 (public list price, us-central1; estimate, not an invoice)"
