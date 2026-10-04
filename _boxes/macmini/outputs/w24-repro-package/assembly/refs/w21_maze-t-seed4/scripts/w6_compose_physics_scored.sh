#!/usr/bin/env bash
# w6 hub/compose-physics -- the SCORED leg, exactly as PREREG-w6-compose-physics-1.md fixes it.
#
#   scripts/w6_compose_physics_scored.sh <seed> [threads]
#
# ⛔ Nothing about the arm list, the primary or the margin is a command-line choice: they
# are written into this file so a replicator runs the same thing from the PREREG without
# reconstructing a command line.  The leg refuses to start unless the PREREG file and the
# ledger row are already on disk (the check lives in w6_compose_physics_leg.sh).
#
# ⭐ THE RIVAL SET AND THE INCUMBENT ARE NOT RE-RUN.  Their `rows.npz`, the per-seed
# `pca_content.pkl` and the per-seed address book are copied bit-for-bit from wave 5's
# own scored legs into $OUT before this runs (PREREG section 2).  That is stronger than
# re-running them: the settled read is compared against the table wave 5 PUBLISHED, on
# the same corners, by the same objects -- not against a re-derivation of it.
set -euo pipefail

SEED="${1:?seed}"
export THREADS="${2:-3}"
OUT=/c/chlu/outputs/w6-compose-physics/scored
export PREREG=/c/chlu/outputs/w6-compose-physics/PREREG-w6-compose-physics-1.md
export CHLU_GEN_BRIDGE_ROOT_OVERRIDE=/c/chlu/outputs/spoke-a-rivals/cache_generation2
export MAX_SECONDS="${MAX_SECONDS:-2700}"
export PRIMARY=pca_settle
export MARGIN=1.0

# Refuse to run if the copied wave-5 leg is not in place: without it there is no rival
# set and no incumbent on this leg, and a table without the bar is not a result.
LEG="$OUT/test_seed${SEED}"
for need in "$LEG/arms/pca_tensor/rows.npz" "$LEG/arms/rival_mlp_180k/rows.npz" \
            "$LEG/pca_content.pkl" \
            "$LEG/address_pca_prod_R128_lam0.01_sw40_n131072.pkl"; do
  [ -f "$need" ] || { echo "[scored] REFUSING: missing $need" >&2; exit 2; }
done
echo "[scored] wave-5 leg present: rival set, incumbent, content and address book in place"

# Order matters only for what lands first if the leg is killed: the NO-OP PROOF first,
# because if it fails every margin below is uninterpretable; then the ladder from least
# to most relaxation; then the charter-C family.
ARMS=(
  pca_settle_zerostep        # 0 steps -- asserted BIT-IDENTICAL to pca_tensor
  pca_settle_f05_conv        # lambda = 0.05 x certified-safe, converged
  pca_settle_onestep         # ONE step, lambda = certified-safe
  pca_settle                 # THE ARM: converged, lambda = certified-safe
  pca_settle_lookup          # FLOOR, charter C.1/C.4
  pca_settle_removed         # CONTROL, charter C.2
  pca_settle_shuffled        # CONTROL, charter C.2
)

exec bash "$(dirname "$0")/w6_compose_physics_leg.sh" test "$SEED" "$OUT" "${ARMS[@]}"
