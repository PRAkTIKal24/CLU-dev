#!/usr/bin/env bash
# scripts/csf3/stage_artefacts.sh — put the GEN ARTEFACT SET on CSF3 scratch.
# RUN ON THE LAPTOP (on-campus network or GlobalProtect VPN required).
#
#   export CSF3_USER=<uom-it-username>            # e.g. mabcxyz1, NOT the email
#   scripts/csf3/stage_artefacts.sh --dry-run     # plan + manifest, no transfer
#   scripts/csf3/stage_artefacts.sh               # manifest + rsync -> ~/scratch/CHLU/artefacts
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS
# ---------------------------------------------------------------------------
# `push_repo.sh` carries `--exclude '.claude/'` (line 42) and `sync_project.sh`
# only moves `projects/<name>/`. Every gen artefact — the feature bridge, the
# shipped store plan, the winner decoder, the composition store, the external
# classifier — lives under `.claude/outputs/**`, which is neither. ⛔ Before
# this script there was NO path that put a gen artefact on CSF3 scratch.
#
# ⛔ `push_repo.sh` is NOT edited: it is the live code path for a cluster user
# and changing a working sync under them is how a working sync silently stops
# working. This script is additive and orthogonal — code goes by push_repo.sh,
# artefacts go by this.
#
# ---------------------------------------------------------------------------
# THE LAYOUT, AND WHY IT IS A MIRROR OF `.claude/`
# ---------------------------------------------------------------------------
# Every gen loader resolves its default as `<repo>/.claude/<something>`:
#
#   gen_*.claude_root   -> $CHLU_CLAUDE_ROOT      else <repo>/.claude
#   gen_*.bridge_root   -> $CHLU_GEN_BRIDGE_ROOT  else <claude>/outputs/gen-feature-bridge/runs
#   gen_*.store_runs    -> $CHLU_GEN_STORE_RUNS   else <claude>/outputs/gen-store-grouping/runs
#   gen_data.data_root  -> $CHLU_GEN_DATA_ROOT    else <repo>/.claude/data/gen
#
# So the artefacts are staged as a VERBATIM MIRROR of `.claude/` at
# `~/scratch/CHLU/artefacts/`, and ONE variable makes every default resolve:
#
#   export CHLU_CLAUDE_ROOT=$HOME/scratch/CHLU/artefacts
#   export CHLU_GEN_DATA_ROOT=$CHLU_CLAUDE_ROOT/data/gen
#
# (the bridge/store roots then fall out of claude_root; they are exported too
# in the Head's command block because a few configs read the env var directly.)
#
# ---------------------------------------------------------------------------
# ⛔ WHAT IS *NOT* STAGED
# ---------------------------------------------------------------------------
# * The image blobs (`3dshapes.h5`, the dSprites `.npz`/`.hdf5`). They are
#   re-downloadable on a compute node (`chlu data shapes3d --download`;
#   compute nodes have internet, login nodes do not) and CSF3 scratch has NO
#   backup and a 3-month auto-cleanup. ⛔ Never stage a dataset.
# * Anything with no named loader in the five launcher stages — plots, logs,
#   `encoder_manifest.json`, `shared_decoder_hash.json`, `winning_recipe.yaml`.
#   ⚠ An artefact nobody can name a loader for does not go.
#
# ---------------------------------------------------------------------------
# KNOBS
# ---------------------------------------------------------------------------
#   DATASETS="shapes3d"      PRETEXTS="recon"      SEEDS="0 1 2"
#   REMOTE_ART="scratch/CHLU/artefacts"            (relative to $HOME on CSF3)
#   CLAUDE_ROOT=<path>       (default <repo>/.claude)
#   STAGE_PYTHON=python3     (stdlib only — no jax, no venv needed here)
#
# Flags: --dry-run | --manifest-only | --datasets S | --pretexts S | --seeds S
#        --manifest PATH | -h

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
CLAUDE_ROOT="${CLAUDE_ROOT:-$LOCAL_REPO/.claude}"
CSF3_HOST="${CSF3_HOST:-csf3.itservices.manchester.ac.uk}"
REMOTE_ART="${REMOTE_ART:-scratch/CHLU/artefacts}"
PY_BIN="${STAGE_PYTHON:-python3}"

DATASETS="${DATASETS:-shapes3d}"
PRETEXTS="${PRETEXTS:-recon}"
SEEDS="${SEEDS:-0 1 2}"

DRY_RUN=0
MANIFEST_ONLY=0
MANIFEST_OUT=""

die()  { echo "FATAL[stage_artefacts]: $*" >&2; exit 2; }
say()  { echo "[stage] $*"; }

usage() {
    sed -n '2,66p' "$BASH_SOURCE"
    exit "${1:-1}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY_RUN=1 ;;
        --manifest-only) MANIFEST_ONLY=1 ;;
        --datasets)      DATASETS="${2:?--datasets needs a value}"; shift ;;
        --pretexts)      PRETEXTS="${2:?--pretexts needs a value}"; shift ;;
        --seeds)         SEEDS="${2:?--seeds needs a value}"; shift ;;
        --manifest)      MANIFEST_OUT="${2:?--manifest needs a path}"; shift ;;
        -h|--help)       usage 0 ;;
        *)               echo "unknown flag: $1" >&2; usage 1 ;;
    esac
    shift
done

MANIFEST_OUT="${MANIFEST_OUT:-$CLAUDE_ROOT/outputs/gen-cluster-stage-artefacts/stage_manifest.json}"

[ -d "$CLAUDE_ROOT" ] || die "no artefact root at $CLAUDE_ROOT (set CLAUDE_ROOT=…). ⚠ A worktree has no .claude/ of its own."
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN' (stdlib json only)"

for d in $DATASETS; do
    case "$d" in shapes3d|dsprites) ;; *) die "DATASET '$d' is not one of: shapes3d dsprites" ;; esac
done
for p in $PRETEXTS; do
    case "$p" in recon|cls) ;; *) die "PRETEXT '$p' is not one of: recon cls" ;; esac
done
for s in $SEEDS; do
    case "$s" in ''|*[!0-9]*) die "SEED '$s' is not a non-negative integer" ;; esac
done

# ---------------------------------------------------------------------------
# sha256, portably (Linux coreutils vs macOS)
# ---------------------------------------------------------------------------
if command -v sha256sum >/dev/null 2>&1; then
    sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    die "no sha256sum and no shasum — cannot checksum, and an unchecksummed stage is not a stage"
fi

bytes_of() { wc -c < "$1" | tr -d ' '; }

# ---------------------------------------------------------------------------
# ⭐ THE ARTEFACT SET — DERIVED FROM THE LOADERS, NOT FROM A DIRECTORY LISTING
#
# Each row is  <relative-path-under-.claude>  <open_with>  <required>  <loader>
# where `loader` names the production call site that opens the file and
# `open_with` is the mechanical dispatch key `verify_artefacts.sh` uses to
# OPEN it on the far side. ⛔ A file with no loader is not in this table.
#
# The five launcher stages (job_gpu_gen.sh: encoder|decoder|store|cell|
# certificate) and what each of them opens:
#
#   encoder      chlu.experiments.exp_gen_encoder — opens NOTHING under
#                .claude/outputs; it needs only the dataset root. Not staged.
#   decoder      ⚠ chlu.experiments.exp_gen_decoder DOES NOT EXIST (finding).
#   store        chlu.experiments.exp_gen_store
#   cell         chlu.experiments.exp_gen_cell
#   certificate  chlu.experiments.exp_gen_certificate
# ---------------------------------------------------------------------------

TAB="$(printf '\t')"
ROWS_FILE="$(mktemp -t chlu_stage_rows.XXXXXX)"
MISSING_FILE="$(mktemp -t chlu_stage_missing.XXXXXX)"
trap 'rm -f "$ROWS_FILE" "$MISSING_FILE" "$ROWS_FILE.list"' EXIT

add_row() {  # $1=relpath $2=open_with $3=required(0|1) $4=loader
    local rel="$1" how="$2" req="$3" loader="$4"
    if [ ! -f "$CLAUDE_ROOT/$rel" ]; then
        if [ "$req" = "1" ]; then
            printf '%s\t%s\n' "$rel" "$loader" >> "$MISSING_FILE"
        else
            say "  (optional, absent — skipped) $rel"
        fi
        return 0
    fi
    printf '%s%s%s%s%s%s%s\n' "$rel" "$TAB" "$how" "$TAB" "$req" "$TAB" "$loader" >> "$ROWS_FILE"
}

# --- 1. the FEATURE BRIDGE, per (dataset, pretext, seed) --------------------
# The bridge cell is `<dataset>_<pretext>` for the headline Montero condition
# (chlu.experiments.exp_gen_bridge.artifacts_tag; a non-default gen_data.condition
# suffixes the tag and needs its own stage run — pass --pretexts accordingly).
for ds in $DATASETS; do
  for px in $PRETEXTS; do
    TAG="${ds}_${px}"
    for sd in $SEEDS; do
      B="outputs/gen-feature-bridge/runs/$TAG/seed_$sd"
      add_row "$B/bridge_manifest.json" json 1 \
        "chlu.experiments.exp_gen_store.bridge_dir + exp_gen_cell.run_seed + exp_gen_certificate.bridge_dir (json.load; existence is the bridge's own precondition check)"
      add_row "$B/encoder.pkl" eqx_pickle 1 \
        "chlu.utils.checkpoints.load_model via exp_gen_store.fit_seed / exp_gen_cell.run_seed / exp_gen_certificate (param_hash asserted against bridge_manifest.json)"
      add_row "$B/decoder.pkl" eqx_pickle 1 \
        "chlu.utils.checkpoints.load_model via exp_gen_store.fit_seed (assert_decoder_hash) and exp_gen_compose.resolve_decoder when decoder_source='bridge'"
      add_row "$B/probe.pkl" eqx_pickle 1 \
        "chlu.utils.checkpoints.load_model via exp_gen_store.fit_seed / exp_gen_cell.run_seed (probe_param_hash asserted)"
      # ⭐ The cache's two payload files are named BY the cache manifest, not by
      #    this script — reading them off disk is how the file set stays derived
      #    from the loader (chlu.core.gen_bridge.load_feature_cache) rather than
      #    from a guess about the filenames.
      add_row "$B/feature_cache.json" feature_cache 1 \
        "chlu.core.gen_bridge.load_feature_cache (reads rows_file/features_file, verifies features_sha256)"
      if [ -f "$CLAUDE_ROOT/$B/feature_cache.json" ]; then
        for key in rows_file features_file; do
          fn="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
                "$CLAUDE_ROOT/$B/feature_cache.json" "$key")"
          add_row "$B/$fn" npy 1 \
            "chlu.core.gen_bridge.load_feature_cache -> np.load (named by feature_cache.json['$key'])"
        done
      fi
    done

    # --- 2. the SHIPPED STORE (the pinned grouping) -------------------------
    add_row "outputs/gen-store-grouping/runs/$TAG/store_plan.json" store_plan 1 \
      "chlu.core.gen_store.load_store_plan via exp_gen_cell.store_plan_path and exp_gen_certificate.store_runs_dir"
    for sd in $SEEDS; do
      add_row "outputs/gen-store-grouping/runs/$TAG/seed_$sd/store_manifest.json" json 0 \
        "chlu.experiments.exp_gen_certificate.run_seed (the 'shipped' record; optional — absent means no bit-identity assertion)"
    done

    # --- 3. the COMPOSITION STORE the cell loads instead of re-training -----
    for sd in $SEEDS; do
      add_row "outputs/gen-read-compose/runs/$TAG/seed_$sd/store_s$sd.pkl" eqx_pickle 1 \
        "chlu.utils.checkpoints.load_model via exp_gen_cell.resolve_store (gen_cell.store_checkpoint_pattern); ⛔ absent => the cell silently RE-TRAINS a store and its row is no longer comparable"
    done
  done

  # --- 4. the single-sourced C set (derived, ⛔ NOT the dataset) ------------
  add_row "data/gen/$ds/c_invalid.json" c_invalid 1 \
    "chlu.core.gen_certificate.load_c_requests via exp_gen_certificate.c_invalid_path; chlu.experiments.exp_gen_store.load_c_set"
done

# --- 5. the WINNER DECODER (seed-indexed, pretext-independent) --------------
add_row "outputs/gen-decoder-lift/winner/winner_hashes.json" winner_hashes 1 \
  "chlu.core.gen_dream.load_winner_hashes via exp_gen_compose.resolve_decoder"
for sd in $SEEDS; do
  add_row "outputs/gen-decoder-lift/winner/decoder_v1_s$sd.pkl" eqx_pickle 1 \
    "chlu.utils.checkpoints.load_model via exp_gen_compose.resolve_decoder (gen_cell.winner_decoder_pattern; recipe + param hash asserted)"
done

# --- 6. the EXTERNAL CLASSIFIER — the one instrument, and its gate ----------
add_row "outputs/gen-external-classifier/classifier_gate.json" json 1 \
  "chlu.experiments.exp_gen_cell.check_preconditions (⛔ PRECONDITION: top-level 'gate' must read 'PASS')"
for sd in $SEEDS; do
  add_row "outputs/gen-external-classifier/seed_$sd/external_classifier.pkl" eqx_pickle 1 \
    "chlu.utils.checkpoints.load_model via exp_gen_cell.load_external_classifier (param hash asserted against its own manifest)"
  add_row "outputs/gen-external-classifier/seed_$sd/classifier_manifest.json" json 1 \
    "chlu.experiments.exp_gen_cell.load_external_classifier (supplies the asserted param_hash)"
done

# --- 7. the A7 re-scoring: the cell's second precondition + LACE's config ---
add_row "outputs/gen-arms-a7-wiring/refusal_scores.json" json 1 \
  "chlu.experiments.exp_gen_cell.check_preconditions (⛔ PRECONDITION: 'status' must read 'complete')"
for sd in $SEEDS; do
  add_row "outputs/gen-arms-a7-wiring/runs/historical_688128/seed_$sd/baselines_manifest.json" json 1 \
    "chlu.experiments.exp_gen_cell.lace_selection (gen_cell.lace_selection_source/pattern — the INHERITED LACE configuration; re-tuning here would weaken the rival through our own code)"
done

# ---------------------------------------------------------------------------
# Refusals: a missing REQUIRED artefact, and an empty set.
# ---------------------------------------------------------------------------
# ⭐ Emptiness FIRST: nothing found at all is almost always a wrong CLAUDE_ROOT
# (a worktree has no `.claude/` of its own), and naming 40 "missing" files
# buries that. A partial set is the second, different refusal below.
if [ ! -s "$ROWS_FILE" ]; then
    echo "FATAL[stage_artefacts]: the artefact set is EMPTY — refusing to run." >&2
    echo "  root=$CLAUDE_ROOT  datasets='$DATASETS' pretexts='$PRETEXTS' seeds='$SEEDS'" >&2
    echo "  ⚠ Not one of the $( [ -s "$MISSING_FILE" ] && wc -l < "$MISSING_FILE" | tr -d ' ' || echo 0 ) expected artefacts exists there." >&2
    echo "  ⛔ An empty stage is a silent no-op that looks like a success on the far side." >&2
    exit 4
fi
if [ -s "$MISSING_FILE" ]; then
    echo "FATAL[stage_artefacts]: $(wc -l < "$MISSING_FILE" | tr -d ' ') REQUIRED artefact(s) are missing under $CLAUDE_ROOT:" >&2
    while IFS="$TAB" read -r rel loader; do
        echo "  ⛔ $rel" >&2
        echo "       loader: $loader" >&2
    done < "$MISSING_FILE"
    echo "⛔ Refusing to stage a partial set: the leg that opens a missing file dies mid-run on the cluster." >&2
    exit 3
fi

N_FILES="$(wc -l < "$ROWS_FILE" | tr -d ' ')"
say "artefact root : $CLAUDE_ROOT"
say "axes          : datasets='$DATASETS' pretexts='$PRETEXTS' seeds='$SEEDS'"
say "files         : $N_FILES"
say "checksumming  : $N_FILES files (sha256) …"

# ---------------------------------------------------------------------------
# Manifest: every file, size, sha256, its loader, its dispatch key.
# ---------------------------------------------------------------------------
TOTAL=0
HASHED="$ROWS_FILE.hashed"
: > "$HASHED"
while IFS="$TAB" read -r rel how req loader; do
    [ -n "$rel" ] || continue
    f="$CLAUDE_ROOT/$rel"
    b="$(bytes_of "$f")"
    h="$(sha256_of "$f")"
    TOTAL=$((TOTAL + b))
    printf '%s%s%s%s%s%s%s%s%s%s%s\n' "$rel" "$TAB" "$how" "$TAB" "$req" "$TAB" "$b" "$TAB" "$h" "$TAB" "$loader" >> "$HASHED"
done < "$ROWS_FILE"

mkdir -p "$(dirname "$MANIFEST_OUT")"
STAGE_MF_OUT="$MANIFEST_OUT" STAGE_MF_ROOT="$CLAUDE_ROOT" STAGE_MF_REMOTE="$REMOTE_ART" \
STAGE_MF_DATASETS="$DATASETS" STAGE_MF_PRETEXTS="$PRETEXTS" STAGE_MF_SEEDS="$SEEDS" \
STAGE_MF_TOTAL="$TOTAL" STAGE_MF_HOST="$CSF3_HOST" \
"$PY_BIN" - "$HASHED" <<'PY'
import datetime as _dt, json, os, sys

e = os.environ.get
files = []
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        rel, how, req, size, sha, loader = line.split("\t", 5)
        files.append({
            "path": rel,                      # relative to the staged artefact root
            "bytes": int(size),
            "sha256": sha,
            "loader": loader,                 # ⭐ the production call site that opens it
            "open_with": how,                 # dispatch key for verify_artefacts.sh
            "required": req == "1",
        })
files.sort(key=lambda d: d["path"])
m = {
    "schema": "chlu-gen-stage-manifest/1",
    "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "source_root": e("STAGE_MF_ROOT"),
    "remote_root": e("STAGE_MF_REMOTE"),
    "remote_host": e("STAGE_MF_HOST"),
    "axes": {
        "datasets": e("STAGE_MF_DATASETS", "").split(),
        "pretexts": e("STAGE_MF_PRETEXTS", "").split(),
        "seeds": [int(s) for s in e("STAGE_MF_SEEDS", "").split()],
    },
    "n_files": len(files),
    "total_bytes": int(e("STAGE_MF_TOTAL", "0")),
    "env_for_the_job": {
        "CHLU_CLAUDE_ROOT": "$HOME/" + e("STAGE_MF_REMOTE", ""),
        "CHLU_GEN_DATA_ROOT": "$HOME/" + e("STAGE_MF_REMOTE", "") + "/data/gen",
        "CHLU_GEN_BRIDGE_ROOT": "$HOME/" + e("STAGE_MF_REMOTE", "") + "/outputs/gen-feature-bridge/runs",
        "CHLU_GEN_STORE_RUNS": "$HOME/" + e("STAGE_MF_REMOTE", "") + "/outputs/gen-store-grouping/runs",
        "CHLU_GEN_DECLIFT_DIR": "$HOME/" + e("STAGE_MF_REMOTE", "") + "/outputs/gen-decoder-lift/winner",
    },
    "not_staged": {
        "image_blobs": ("3dshapes.h5 / dsprites *.npz,*.hdf5 — re-downloadable on a "
                        "compute node (`chlu data <ds> --download`); ⛔ scratch has no "
                        "backup and a 3-month auto-cleanup, never stage a dataset"),
        "no_named_loader": ("plots, logs, encoder_manifest.json, shared_decoder_hash.json, "
                            "winning_recipe.yaml, factor_index.npz — no loader in the five "
                            "launcher stages opens them"),
    },
    "files": files,
}
with open(e("STAGE_MF_OUT"), "w") as fh:
    json.dump(m, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("[stage] manifest -> %s  (%d files, %d bytes, %.2f GiB)"
      % (e("STAGE_MF_OUT"), len(files), m["total_bytes"], m["total_bytes"] / 2**30))
PY

if [ "$DRY_RUN" = "1" ] || [ "$MANIFEST_ONLY" = "1" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        say "=== DRY RUN — the plan (nothing transferred) ==="
        awk -F"$TAB" '{printf "  %12d  %s\n", $4, $1}' "$HASHED"
        say "would rsync $N_FILES files -> ${CSF3_USER:-<CSF3_USER>}@$CSF3_HOST:$REMOTE_ART/"
        say "then, in the job:  export CHLU_CLAUDE_ROOT=\$HOME/$REMOTE_ART"
    fi
    say "total: $TOTAL bytes ($("$PY_BIN" -c "print('%.2f' % ($TOTAL/2**30))") GiB)"
    say "⚠ CSF3 scratch has NO backup and a 3-month auto-cleanup — everything here is re-stageable from the laptop."
    exit 0
fi

# ---------------------------------------------------------------------------
# Transfer. Idempotent (rsync skips unchanged files) and resumable
# (--partial-dir keeps a half-sent 1.9 GB features.npy across a dropped VPN).
# ---------------------------------------------------------------------------
CSF3_USER="${CSF3_USER:?set CSF3_USER=<uom-it-username> (form mabcxyz1), or pass --dry-run}"
cut -d"$TAB" -f1 "$HASHED" > "$ROWS_FILE.list"
# the manifest travels WITH the set — the verify job reads it on the far side
MF_REL="outputs/gen-cluster-stage-artefacts/stage_manifest.json"
if [ "$MANIFEST_OUT" = "$CLAUDE_ROOT/$MF_REL" ]; then
    echo "$MF_REL" >> "$ROWS_FILE.list"
fi

say "rsync $N_FILES files -> $CSF3_USER@$CSF3_HOST:$REMOTE_ART/"
rsync -av --progress --partial-dir=.rsync-partial \
    --files-from="$ROWS_FILE.list" \
    "$CLAUDE_ROOT/" "$CSF3_USER@$CSF3_HOST:$REMOTE_ART/"

say "staged $TOTAL bytes. ⭐ NOW RUN THE VERIFY JOB — a file that transferred and will not load is a failure, not a success:"
say "  ssh $CSF3_USER@$CSF3_HOST"
say "  cd ~/scratch/CHLU && mkdir -p logs && sbatch scripts/csf3/verify_artefacts.sh"
