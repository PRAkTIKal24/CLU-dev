#!/bin/bash --login
# =====================================================================
# run_all_msl.sh — ONE-SHOT: submit the full MSL_REMS experiment grid on CSF3
# (events × finetune-modes), then AUTO-AGGREGATE when it finishes.
#
# Data note: train.npy / test.npy hold the RAW telemetry; the event labels
# (PDROP/INV/RHSAT/THRESH/…) are derived at LOAD time from that same telemetry,
# so every event type is already prepped — no extra data step.
#
# Usage (from the repo root on a CSF3 login node, after `uv sync`):
#     bash hpc/csf3/run_all_msl.sh           # full grid (50+30 epochs)
#     SMOKE=1 bash hpc/csf3/run_all_msl.sh   # 1+1-epoch smoke to validate first
# Override the grid: EVENTS="MSL_REMS_INV MSL_REMS_RHSAT" MODES="e2e" bash ...
# Run-4 sp_ft sweep (8 arms × 3 seeds × 3 events = 72 tasks):
#     EVENTS="MSL_REMS_PDROP MSL_REMS_INV MSL_REMS_RHSAT" MODES="sp_ft" bash hpc/csf3/run_all_msl.sh
# 10-seed re-run into a FRESH figure dir (tops the existing workspaces up to 10
# seeds; deterministic, so seeds 0-2 already match -> pass only the 7 new seeds):
#     SEEDS="3 4 5 6 7 8 9" EVENTS="MSL_REMS_PDROP MSL_REMS_RHSAT" \
#       MODES="pred_ft sp_ft e2e" FIGDIR="docs/figures/run5_10seed" bash hpc/csf3/run_all_msl.sh
#
# Knobs (env): SEEDS (space-separated seed list; default "0 1 2"), CONCURRENCY
# (array %N throttle; default 2), FIGDIR (figure output root; default = legacy
# run3/run4 + in-workspace). The SLURM array is sized 8 arms x nseed automatically.
#
# It submits one (8*nseed)-task array per (event,mode) into workspaces/<event>_<mode>/,
# then a CPU aggregation job that depends on all of them (afterany) and renders:
#   A. per-(event,mode) label-efficiency curves  (evaluate_msl_ablations.py)
#   B. event effect / geometry-vs-control lift    (evaluate_cross_event.py --events …)
#   C. finetune-mode comparison (whatever modes ran: pred_ft/sp_ft/e2e, per event;
#      the run-4 3-mode lift figure)             (evaluate_modes.py --events …)
# When FIGDIR is set, B/C and the per-(event,mode) curves write under $FIGDIR/
# (cross_event/<mode>/, modes/, curves/) so a re-run never overwrites older plots.
# =====================================================================
set -euo pipefail

EVENTS="${EVENTS:-MSL_REMS_PDROP MSL_REMS_INV MSL_REMS_RHSAT}"   # control + 2 geometric
# finetune-mode factorial. The three regimes are e2e | pred_ft | sp_ft (run-4
# spatial-adapter finetune). Override MODES to run any subset, e.g. MODES="sp_ft".
MODES="${MODES:-e2e pred_ft sp_ft}"
SEEDS="${SEEDS:-0 1 2}"                 # space-separated; array is sized 8*nseed
CONCURRENCY="${CONCURRENCY:-2}"         # SLURM array %N throttle (your concurrent GPUs)
FIGDIR="${FIGDIR:-}"                    # fresh figure root; empty = legacy run3/run4 paths
# Pretrain variants: space-separated "ws_suffix:pretrain_gate_init" tokens. Default is
# one cold variant. Run-5 bundles warm-gate pretraining alongside cold in ONE launch:
#   VARIANTS=":0.0 _wg:0.1"
VARIANTS="${VARIANTS:-:0.0}"
REPO_DIR="${REPO_DIR:-$HOME/scratch/HEPA}"
PROJECTS="baseline,v1_0_hybrid,v1_1_coords,v1_2_eucbias,v2_1_0_geo_sensorq,v2_1_1_geo_patchq,v3_1_0_se3_sensorq,v3_1_1_se3_patchq"
cd "$REPO_DIR"
NSEED=$(set -- $SEEDS; echo $#)         # seed count -> array size 8*NSEED

# ------------------------------------------------------------------ aggregate
# (invoked by the dependent SLURM job; EVENTS/MODES/REPO_DIR arrive via --export)
if [ "${1:-}" = "--aggregate" ]; then
  R="uv run --no-sync"
  EVCSV=$(echo "$EVENTS" | tr ' ' ',')
  MODECSV=$(echo "$MODES" | tr ' ' ',')
  # Render A/B/C once per pretrain variant, into its own sub-tree (cold / wg / ...),
  # so the warm-gate run never overwrites the cold run's plots.
  for V in $VARIANTS; do
    SUF="${V%%:*}"
    VTAG="${SUF#_}"; VTAG="${VTAG:-cold}"          # "_wg"->"wg", ""->"cold"
    echo "[agg] === variant '${SUF:-<cold>}' (tag=$VTAG) ==="
    echo "[agg] A — per-(event,mode) label-efficiency curves"
    for E in $EVENTS; do for M in $MODES; do
      A_OUT=(); [ -n "$FIGDIR" ] && A_OUT=(--out "$FIGDIR/$VTAG/curves/${E}_${M}_label_efficiency.png")
      $R python scripts/evaluate_msl_ablations.py --workspace "${E}_${M}${SUF}" --projects "$PROJECTS" "${A_OUT[@]}" || true
    done; done
    echo "[agg] B — event effect (geometry vs control), per mode"
    for M in $MODES; do
      # Per-mode out-dir: cross_event_lift.png is mode-agnostic in name, so a per-mode
      # dir stops modes clobbering each other (legacy docs/figures/run3/ when FIGDIR unset).
      B_OUT=(); [ -n "$FIGDIR" ] && B_OUT=(--out-dir "$FIGDIR/$VTAG/cross_event/$M/")
      $R python scripts/evaluate_cross_event.py --events "$EVCSV" --mode "$M" --ws-suffix "$SUF" "${B_OUT[@]}" || true
    done
    echo "[agg] C — finetune-mode comparison (whatever modes ran), per event"
    C_OUT=(); [ -n "$FIGDIR" ] && C_OUT=(--out-dir "$FIGDIR/$VTAG/modes/")
    $R python scripts/evaluate_modes.py --events "$EVCSV" --modes "$MODECSV" --ws-suffix "$SUF" "${C_OUT[@]}" || true
  done
  # Run-5 headline: warm-gate vs cold, paired per (event, mode, arm). Fires for every
  # non-empty variant suffix present (e.g. _wg) against the cold ("") baseline.
  for V in $VARIANTS; do
    SUF="${V%%:*}"; [ -z "$SUF" ] && continue
    VTAG="${SUF#_}"
    echo "[agg] D — warm-vs-cold comparison ('$SUF' vs cold)"
    D_OUT=(); [ -n "$FIGDIR" ] && D_OUT=(--out-dir "$FIGDIR/${VTAG}_vs_cold/")
    $R python scripts/evaluate_modes.py --events "$EVCSV" --modes "$MODECSV" \
        --compare-suffix "$SUF" "${D_OUT[@]}" || true
  done
  echo "[agg] done -> ${FIGDIR:-docs/figures/run3,run4 + workspaces/<event>_<mode>/}"
  exit 0
fi

# --------------------------------------------------------------------- submit
mkdir -p logs
EPOCH_ENV=""; [ "${SMOKE:-0}" = "1" ] && EPOCH_ENV=",PRE_EPOCHS=1,FT_EPOCHS=1"
NTASK=$(( 8 * NSEED ))                   # 8 arms x nseed array tasks per (event,mode)
echo "submitting grid: events=[$EVENTS] modes=[$MODES] seeds=[$SEEDS] variants=[$VARIANTS] (${NTASK} tasks/array, %${CONCURRENCY}) smoke=${SMOKE:-0}"
JIDS=()
# VARIANTS axis: each token is "ws_suffix:pretrain_gate_init". Default ":0.0" is one
# cold variant (empty suffix, no warm). The run-5 warm-gate run adds "_wg:0.1" so a
# single launch submits BOTH cold and warm-gate-pretrained sets into separate trees.
for V in $VARIANTS; do
  SUF="${V%%:*}"; GATE="${V##*:}"
  for E in $EVENTS; do for M in $MODES; do
    # `--array` on the CLI overrides the script's `#SBATCH -a` directive; SEEDS rides
    # along in the env so the task->(arm,seed) map matches the array size.
    jid=$(sbatch --parsable -J "msl_${E}_${M}${SUF}" \
          --array="1-${NTASK}%${CONCURRENCY}" \
          --export="ALL,EVENT=$E,FT_MODE=$M,SEEDS=$SEEDS,WS_SUFFIX=$SUF,PRETRAIN_GATE_INIT=$GATE$EPOCH_ENV" \
          hpc/csf3/msl_cad_event.sbatch)
    echo "  ${E}_${M}${SUF} (gate=$GATE) -> array $jid (${NTASK} tasks)"; JIDS+=("$jid")
  done; done
done
DEP=$(IFS=:; echo "${JIDS[*]}")

# Aggregation: 1 CPU core on the serial partition, fires once ALL arrays finish
# (afterany = aggregate whatever completed). Override AGG_PARTITION if 'serial'
# isn't available to you. Resilient: if this submit fails, run the aggregate
# step by hand later: `bash hpc/csf3/run_all_msl.sh --aggregate`.
if agg=$(sbatch --parsable \
      --dependency="afterany:$DEP" -p "${AGG_PARTITION:-serial}" -n 1 -c 1 -t 0-01:00 \
      -J msl_aggregate -o logs/aggregate.%j.out \
      --export="ALL,REPO_DIR=$REPO_DIR,EVENTS=$EVENTS,MODES=$MODES,FIGDIR=$FIGDIR,VARIANTS=$VARIANTS" \
      --wrap="bash '$REPO_DIR/hpc/csf3/run_all_msl.sh' --aggregate" 2>&1); then
  echo "submitted ${#JIDS[@]} sweep arrays + aggregation job $agg (auto-runs after)."
else
  echo "WARN: aggregation submit failed ($agg)."
  echo "  -> after the arrays finish, aggregate by hand: FIGDIR=$FIGDIR bash hpc/csf3/run_all_msl.sh --aggregate"
fi
echo "watch: squeue -u \$USER   |   results land in workspaces/<event>_<mode>/ and ${FIGDIR:-docs/figures/run3,run4}/"
