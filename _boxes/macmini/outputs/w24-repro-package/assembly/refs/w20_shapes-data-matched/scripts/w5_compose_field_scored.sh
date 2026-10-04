#!/usr/bin/env bash
# w5 hub/compose-field -- the SCORED leg, exactly as PREREG-w5-compose-field-1.md fixes it.
#
#   scripts/w5_compose_field_scored.sh <seed> [threads]
#
# ⛔ Nothing about the arm list, the primary or the margin is a command-line choice here:
# they are written into this file so a replicator runs the same thing from the PREREG
# without reconstructing a command line.  The leg refuses to start unless the PREREG
# file and the ledger row are already on disk (the check lives in w5_compose_field_leg.sh).
set -euo pipefail

SEED="${1:?seed}"
export THREADS="${2:-3}"
OUT=/c/chlu/outputs/w5-compose-field/scored
export PREREG=/c/chlu/outputs/w5-compose-field/PREREG-w5-compose-field-1.md
export CHLU_GEN_BRIDGE_ROOT_OVERRIDE=/c/chlu/outputs/spoke-a-rivals/cache_generation2
export MAX_SECONDS="${MAX_SECONDS:-2700}"
export PRIMARY=pca_tensor
export MARGIN=1.0

# Order matters only for what lands first if the leg is killed: THE ARM, then THE BAR,
# then the floors and controls that can retire it, then the rest of the rival set, then
# the ceilings.  Every stage is resumable at the arm, so a restart re-runs nothing.
ARMS=(
  # -- the arm, and the bar it must clear -------------------------------------
  pca_tensor
  rival_mlp_180k
  # -- the things that can KILL the arm, run early and on purpose -------------
  pca_lookup_argmax          # FLOOR, charter C.1/C.4
  pca_lookup_convex          # FLOOR, charter C.1
  pca_random_refit           # CONTROL: random atoms, address book REFIT
  pca_landscape_removed      # CONTROL, charter C.2
  pca_landscape_shuffled     # CONTROL, charter C.2
  pca_tensor_sum             # ABLATION: the address SUMMED, not multiplied
  # -- the rest of the FULL RIVAL SET ----------------------------------------
  rival_table rival_mlp rival_cont_mlp rival_transformer rival_lstm
  rival_lace rival_poe_ebm
  # -- the wave-3 aligned store: an ABLATION, never a result ------------------
  span_head
  # -- the ceilings.  Every one a DECLARED NON-ARM ---------------------------
  oracle_content_pool        # the arm's own content ceiling
  oracle_span_field          # the task file's literal gate: the field store's atoms
  oracle_content_items       # 64 whole ITEMS: the price of keeping exact deletion
  oracle_content_fine        # same 524,288 B, fine wells: does the byte pin bind?
  oracle_true_feature        # the ceiling of every store at every budget
  oracle_span                # wave-3's 64 fitted modes: w4's own number, replicated
)

exec bash "$(dirname "$0")/w5_compose_field_leg.sh" test "$SEED" "$OUT" "${ARMS[@]}"
