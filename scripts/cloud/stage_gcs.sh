#!/usr/bin/env bash
# scripts/cloud/stage_gcs.sh — put the GEN ARTEFACT SET in Cloud Storage.
# RUN ON THE LAPTOP (no VPN, no cluster account — just an authenticated gcloud).
#
#   scripts/cloud/stage_gcs.sh --dry-run      # plan + cost, nothing transferred
#   scripts/cloud/stage_gcs.sh                # upload -> gs://<bucket>/artefacts/
#
# ---------------------------------------------------------------------------
# ⭐ THIS IS A TRANSPORT PORT OF `scripts/csf3/stage_artefacts.sh`. NOTHING ELSE.
# ---------------------------------------------------------------------------
# The hard part — deriving the artefact set FROM THE LOADERS rather than from a
# directory listing, naming the production call site that opens each file, and
# checksumming the lot — is already built, merged and proven over there. This
# script does not repeat one line of it:
#
#   * the FILE SET comes from `stage_manifest.json`, which the CSF3 script
#     writes. ⛔ It is NEVER re-derived here. If the two platforms disagreed
#     about which 44 files constitute "the gen artefact set", a result could
#     not be moved between them, and that is the whole point of having both.
#   * `--refresh-manifest` shells out to `scripts/csf3/stage_artefacts.sh
#     --manifest-only`, i.e. the SAME derivation, so a refresh cannot drift.
#   * the manifest is uploaded BYTE-IDENTICAL alongside the set. ⛔ Its schema
#     is not "improved" for the cloud: `chlu-gen-stage-manifest/1` is read by
#     both `scripts/csf3/verify_artefacts.sh` and `scripts/cloud/verify_gcs.sh`.
#
# ⛔ `scripts/csf3/**` IS NOT EDITED by this spoke. It is a live path for a
# cluster user; rsync-to-CSF3 keeps working exactly as it did.
#
# ---------------------------------------------------------------------------
# THE LAYOUT — the same verbatim mirror of `.claude/`, one level up
# ---------------------------------------------------------------------------
#   CSF3:  ~/scratch/CHLU/artefacts/<relpath>
#   GCP:   gs://$CHLU_BUCKET/artefacts/<relpath>      (this script)
#          -> downloaded by verify_gcs.sh / the job VM to a local root, and
#             ONE variable makes every gen loader default resolve:
#                 export CHLU_CLAUDE_ROOT=<local root>
#
# The five roots the job must export are printed at the end of every run (and
# by `--print-env`). ⛔ This script does not export them itself and does not
# touch `startup.sh` — `gen-cloud-runner` owns that file this wave.
#
# ---------------------------------------------------------------------------
# ⛔ WHAT IS *NOT* UPLOADED — and the 3dshapes.h5 decision
# ---------------------------------------------------------------------------
# `3dshapes.h5` (255 MiB, sha256 0a0f6ed9…be1785) is ⛔ **NOT staged**. The VM
# downloads it, exactly as a CSF3 compute node does:
#
#   * it is free to us either way, but downloading leaves nothing behind:
#     the upstream copy is a PUBLIC bucket (chlu.data.shapes3d.H5_URL ->
#     storage.googleapis.com/3d-shapes), whose egress is billed to its owner,
#     and GCS->GCE reads inside one region are free. Staging would add
#     255 MiB × $0.020/GiB-month of OUR storage, forever, for a file that is
#     one `download_shapes3d()` call away on a machine that has internet.
#   * ⭐ the production download path ASSERTS the checksum itself
#     (`download_shapes3d` -> `download_file(..., sha256=H5_SHA256)` plus an
#     exact byte-count check), so a truncated fetch CANNOT look fine. That is
#     the only property that makes "download it there" safe, and it is the
#     reason this is a decision rather than a shortcut.
#   * ⭐ it keeps the two platforms comparable: neither stages a dataset.
#
#   `verify_gcs.sh --with-dataset` performs that download ON THE VM and both
#   asserts the sha256 and OPENS the file through `open_images()`.
#
# Anything with no named loader (plots, logs, encoder_manifest.json,
# shared_decoder_hash.json, winning_recipe.yaml, factor_index.npz) is not in
# the manifest and therefore not here either. ⚠ An artefact nobody can name a
# loader for does not go.
#
# ---------------------------------------------------------------------------
# KNOBS   (all overridable from the environment)
# ---------------------------------------------------------------------------
#   CHLU_BUCKET / CHLU_PROJECT      — from scripts/cloud/config.sh
#   GCS_ART_PREFIX=artefacts        — object prefix inside the bucket
#   CLAUDE_ROOT=<repo>/.claude      — the local artefact root
#   MANIFEST=<claude>/outputs/gen-cluster-stage-artefacts/stage_manifest.json
#   STAGE_PYTHON=python3            — stdlib only; no jax, no venv needed here
#
# Flags: --dry-run | --refresh-manifest | --manifest PATH | --claude-root PATH
#        --bucket NAME | --prefix P | --print-env | --no-rehash
#        --datasets S | --pretexts S | --seeds S   (only with --refresh-manifest)
#
# Exit codes (⭐ deliberately the CSF3 ones — a wrapper can treat them alike):
#   0 ok · 2 usage/env · 3 PARTIAL set (a manifest file is missing locally)
#   4 EMPTY set · 5 manifest is STALE (a local file no longer matches its hash)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.sh"

CLAUDE_ROOT="${CLAUDE_ROOT:-$LOCAL_REPO/.claude}"
GCS_ART_PREFIX="${GCS_ART_PREFIX:-artefacts}"
PY_BIN="${STAGE_PYTHON:-python3}"
MANIFEST=""
DRY_RUN=0
REFRESH=0
PRINT_ENV_ONLY=0
REHASH=1
DATASETS="" ; PRETEXTS="" ; SEEDS=""

die() { echo "FATAL[stage_gcs]: $*" >&2; exit 2; }
say() { echo "[stage-gcs] $*"; }

usage() { sed -n '2,100p' "$BASH_SOURCE"; exit "${1:-1}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)          DRY_RUN=1 ;;
        --refresh-manifest) REFRESH=1 ;;
        --manifest)         MANIFEST="${2:?--manifest needs a path}"; shift ;;
        --claude-root)      CLAUDE_ROOT="${2:?--claude-root needs a path}"; shift ;;
        --bucket)           CHLU_BUCKET="${2:?--bucket needs a name}"; shift ;;
        --prefix)           GCS_ART_PREFIX="${2:?--prefix needs a value}"; shift ;;
        --datasets)         DATASETS="${2:?--datasets needs a value}"; shift ;;
        --pretexts)         PRETEXTS="${2:?--pretexts needs a value}"; shift ;;
        --seeds)            SEEDS="${2:?--seeds needs a value}"; shift ;;
        --print-env)        PRINT_ENV_ONLY=1 ;;
        --no-rehash)        REHASH=0 ;;
        -h|--help)          usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
    shift
done

GCS_ROOT="gs://${CHLU_BUCKET}/${GCS_ART_PREFIX}"
MANIFEST="${MANIFEST:-$CLAUDE_ROOT/outputs/gen-cluster-stage-artefacts/stage_manifest.json}"
MF_REL="outputs/gen-cluster-stage-artefacts/stage_manifest.json"

# ---------------------------------------------------------------------------
# ⭐ The env block the RUNNING JOB needs. Published here so `gen-cloud-runner`
#    (which owns startup.sh) and this script cannot disagree about the names.
#    ⛔ Names and relative layout are the CSF3 ones verbatim; only the root moves.
# ---------------------------------------------------------------------------
JOB_ART_ROOT="${JOB_ART_ROOT:-/var/chlu/artefacts}"
print_env_block() {
    cat <<ENVEOF
# --- the five gen roots (export these on the job VM; startup.sh owns doing it) ---
export CHLU_CLAUDE_ROOT="${JOB_ART_ROOT}"
export CHLU_GEN_DATA_ROOT="${JOB_ART_ROOT}/data/gen"
export CHLU_GEN_BRIDGE_ROOT="${JOB_ART_ROOT}/outputs/gen-feature-bridge/runs"
export CHLU_GEN_STORE_RUNS="${JOB_ART_ROOT}/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="${JOB_ART_ROOT}/outputs/gen-decoder-lift/winner"
# source of the tree:  ${GCS_ROOT}/   (gcloud storage rsync -r <that> ${JOB_ART_ROOT})
ENVEOF
}
if [ "$PRINT_ENV_ONLY" = "1" ]; then print_env_block; exit 0; fi

command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN' (stdlib json only)"
command -v gcloud   >/dev/null 2>&1 || die "no gcloud on PATH — this path is gcloud-only by design (ckpt.py shells out to it too)"
[ -d "$CLAUDE_ROOT" ] || die "no artefact root at $CLAUDE_ROOT (set --claude-root). ⚠ A worktree has no .claude/ of its own."

# ---------------------------------------------------------------------------
# The manifest. ⛔ Derived by the CSF3 script or not at all.
# ---------------------------------------------------------------------------
if [ "$REFRESH" = "1" ] || [ ! -f "$MANIFEST" ]; then
    CSF3_STAGE="$LOCAL_REPO/scripts/csf3/stage_artefacts.sh"
    [ -f "$CSF3_STAGE" ] || die "no $CSF3_STAGE — the file set is derived THERE; ⛔ this script will not invent one"
    if [ "$REFRESH" = "1" ]; then
        say "refreshing the manifest via the CSF3 derivation (⛔ not a second derivation)"
    else
        say "no manifest at $MANIFEST — deriving one via scripts/csf3/stage_artefacts.sh --manifest-only"
    fi
    REFRESH_ARGS=(--manifest-only --manifest "$MANIFEST")
    [ -n "$DATASETS" ] && REFRESH_ARGS+=(--datasets "$DATASETS")
    [ -n "$PRETEXTS" ] && REFRESH_ARGS+=(--pretexts "$PRETEXTS")
    [ -n "$SEEDS" ]    && REFRESH_ARGS+=(--seeds "$SEEDS")
    # Its refusals ARE our refusals: exit 4 = empty set, exit 3 = partial set,
    # each naming the missing file and the loader that opens it.
    CLAUDE_ROOT="$CLAUDE_ROOT" LOCAL_REPO="$LOCAL_REPO" STAGE_PYTHON="$PY_BIN" \
        bash "$CSF3_STAGE" "${REFRESH_ARGS[@]}"
elif [ -n "$DATASETS$PRETEXTS$SEEDS" ]; then
    die "--datasets/--pretexts/--seeds only mean something with --refresh-manifest (the axes come FROM the manifest)"
fi
[ -f "$MANIFEST" ] || die "no manifest at $MANIFEST"

# ---------------------------------------------------------------------------
# Preflight: read the manifest, confirm every file is here and still matches
# its recorded hash. ⭐ Uploading bytes that disagree with the manifest would
# produce a set that fails the far-side verify AFTER a 2 GiB transfer.
# ---------------------------------------------------------------------------
WORK="$(mktemp -t chlu_stage_gcs.XXXXXX)"
trap 'rm -f "$WORK" "$WORK.remote" "$WORK.plan"' EXIT

set +e
SG_MANIFEST="$MANIFEST" SG_ROOT="$CLAUDE_ROOT" SG_OUT="$WORK" SG_REHASH="$REHASH" \
    "$PY_BIN" - <<'PY'
import hashlib, json, os, sys
from pathlib import Path

e = os.environ.get
root = Path(e("SG_ROOT"))
mf_path = Path(e("SG_MANIFEST"))
rehash = e("SG_REHASH") == "1"
m = json.loads(mf_path.read_text())

if m.get("schema") != "chlu-gen-stage-manifest/1":
    print("FATAL[stage_gcs]: %s is not a chlu-gen-stage-manifest/1 (schema=%r)"
          % (mf_path, m.get("schema")), file=sys.stderr)
    sys.exit(2)

files = m.get("files") or []

# ⭐ EMPTINESS FIRST, as on CSF3: nothing at all is almost always a wrong root,
#    and naming 44 "missing" files buries that.
if not files:
    print("FATAL[stage_gcs]: the artefact set is EMPTY — refusing to stage.", file=sys.stderr)
    print("  manifest=%s  root=%s" % (mf_path, root), file=sys.stderr)
    print("  ⛔ An empty stage is a silent no-op that looks like a success on the far side.",
          file=sys.stderr)
    sys.exit(4)


def sha256_of(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


missing, stale, rows, total = [], [], [], 0
for rec in files:
    p = root / rec["path"]
    if not p.is_file():
        missing.append((rec["path"], rec["loader"]))
        continue
    size = p.stat().st_size
    if size != rec["bytes"]:
        stale.append((rec["path"], "size %d != %d in the manifest" % (size, rec["bytes"])))
        continue
    if rehash:
        got = sha256_of(p)
        if got != rec["sha256"]:
            stale.append((rec["path"], "sha256 %s… != %s… in the manifest"
                          % (got[:16], rec["sha256"][:16])))
            continue
    total += size
    rows.append("\t".join([rec["path"], str(size), rec["sha256"], rec["open_with"]]))

# ⭐ PARTIAL SET: name the file AND the loader that opens it. The leg that
#    opens a missing file dies mid-run on the far side, which is a far more
#    expensive way to learn the same thing.
if missing:
    print("FATAL[stage_gcs]: %d REQUIRED artefact(s) in the manifest are missing under %s:"
          % (len(missing), root), file=sys.stderr)
    for rel, loader in missing:
        print("  ⛔ %s" % rel, file=sys.stderr)
        print("       loader: %s" % loader, file=sys.stderr)
    print("⛔ Refusing to stage a partial set.", file=sys.stderr)
    sys.exit(3)

if stale:
    print("FATAL[stage_gcs]: %d file(s) no longer match the manifest — it is STALE:"
          % len(stale), file=sys.stderr)
    for rel, why in stale:
        print("  ⛔ %s\n       %s" % (rel, why), file=sys.stderr)
    print("⛔ Re-derive it: scripts/cloud/stage_gcs.sh --refresh-manifest", file=sys.stderr)
    sys.exit(5)

Path(e("SG_OUT")).write_text("\n".join(rows) + "\n")
ax = m.get("axes", {})
print("[stage-gcs] manifest      : %s" % mf_path)
print("[stage-gcs] generated_at  : %s" % m.get("generated_at"))
print("[stage-gcs] axes          : datasets=%s pretexts=%s seeds=%s"
      % (ax.get("datasets"), ax.get("pretexts"), ax.get("seeds")))
print("[stage-gcs] files         : %d  (%d bytes, %.2f GiB)%s"
      % (len(rows), total, total / 2**30, "" if rehash else "   [--no-rehash: sizes only]"))
PY
PRE_RC=$?
set -e
[ "$PRE_RC" -eq 0 ] || exit "$PRE_RC"

N_FILES="$(wc -l < "$WORK" | tr -d ' ')"

# ---------------------------------------------------------------------------
# Idempotence: ONE listing of the destination, then skip what is already there
# with the right size and the right sha256. ⭐ GCS does not store sha256, so
# every object carries it as custom metadata — which is also what makes a
# resumed/partial stage safe to re-run.
# ---------------------------------------------------------------------------
say "destination   : $GCS_ROOT/"
say "listing the destination (one call) …"
gcloud storage ls --recursive --json "$GCS_ROOT/**" --project="$CHLU_PROJECT" \
    > "$WORK.remote" 2>/dev/null || echo '[]' > "$WORK.remote"

SG_WORK="$WORK" SG_REMOTE="$WORK.remote" SG_PLAN="$WORK.plan" SG_PREFIX="$GCS_ART_PREFIX" \
    "$PY_BIN" - <<'PY'
import json, os
from pathlib import Path

e = os.environ.get
prefix = e("SG_PREFIX").strip("/")
try:
    listing = json.loads(Path(e("SG_REMOTE")).read_text() or "[]")
except json.JSONDecodeError:
    listing = []

have = {}
for obj in listing:
    md = obj.get("metadata") or {}
    name = md.get("name")
    if not name:
        continue
    rel = name[len(prefix) + 1:] if name.startswith(prefix + "/") else name
    have[rel] = (int(md.get("size", -1)), ((md.get("metadata") or {}).get("sha256")))

plan, skip_n, skip_b, up_n, up_b = [], 0, 0, 0, 0
for line in Path(e("SG_WORK")).read_text().splitlines():
    if not line.strip():
        continue
    rel, size, sha, how = line.split("\t")
    size = int(size)
    if have.get(rel) == (size, sha):
        skip_n += 1
        skip_b += size
        plan.append("\t".join([rel, str(size), sha, how, "SKIP"]))
    else:
        up_n += 1
        up_b += size
        plan.append("\t".join([rel, str(size), sha, how, "UPLOAD"]))

Path(e("SG_PLAN")).write_text("\n".join(plan) + "\n")
print("[stage-gcs] already there : %d files (%d bytes) — skipped" % (skip_n, skip_b))
print("[stage-gcs] to upload     : %d files (%d bytes, %.2f GiB)" % (up_n, up_b, up_b / 2**30))
PY

# ---------------------------------------------------------------------------
# Cost. ⭐ Storage is the only recurring line; the upload itself is free
# (GCP charges nothing for ingress) and a same-region GCS->GCE read is free.
# ---------------------------------------------------------------------------
cost_line() {
    SG_PLAN="$WORK.plan" "$PY_BIN" - <<'PY'
import os
from pathlib import Path
rows = [l for l in Path(os.environ["SG_PLAN"]).read_text().splitlines() if l.strip()]
tot = sum(int(l.split("\t")[1]) for l in rows)
n = len(rows) + 1  # + the manifest object itself
gb = tot / 1e9
print("[stage-gcs] COST: %.3f GB stored -> $%.4f/month (standard, us-central1, $0.020/GB-mo)."
      % (gb, gb * 0.020))
print("[stage-gcs]       upload (ingress) $0.00 · same-region GCS->GCE read $0.00 ·"
      " %d class-A ops $%.4f ($0.005/1000)." % (n, n * 0.005 / 1000))
print("[stage-gcs]       ⛔ 3dshapes.h5 is NOT here: the VM downloads it (free, checksum asserted).")
PY
}

if [ "$DRY_RUN" = "1" ]; then
    say "=== DRY RUN — the plan (nothing transferred) ==="
    awk -F'\t' '{printf "  %-6s %12d  %-14s %s\n", $5, $2, $4, $1}' "$WORK.plan"
    printf '  %-6s %12s  %-14s %s\n' "UPLOAD" "-" "manifest" "$MF_REL  (travels WITH the set)"
    cost_line
    say "then, on the VM:"
    print_env_block | sed 's/^/    /'
    exit 0
fi

# ---------------------------------------------------------------------------
# Transfer. One `gcloud storage cp` per file: the per-object sha256 custom
# metadata is what makes the next run idempotent, and `cp` is resumable for
# the 1.9 GiB features.npy on its own.
# ---------------------------------------------------------------------------
say "uploading $N_FILES files -> $GCS_ROOT/"
i=0
while IFS=$'\t' read -r rel size sha how action; do
    [ -n "${rel:-}" ] || continue
    i=$((i + 1))
    if [ "$action" = "SKIP" ]; then
        printf '  [%2d/%2d] skip   %s\n' "$i" "$N_FILES" "$rel"
        continue
    fi
    printf '  [%2d/%2d] upload %12s  %s\n' "$i" "$N_FILES" "$size" "$rel"
    gcloud storage cp "$CLAUDE_ROOT/$rel" "$GCS_ROOT/$rel" \
        --project="$CHLU_PROJECT" --custom-metadata="sha256=$sha" -q
done < "$WORK.plan"

# ⭐ The manifest travels WITH the set — verify_gcs.sh reads it on the far side,
#    byte-identical to the one the CSF3 verify reads.
say "uploading the manifest -> $GCS_ROOT/$MF_REL"
MF_SHA="$("$PY_BIN" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MANIFEST")"
gcloud storage cp "$MANIFEST" "$GCS_ROOT/$MF_REL" \
    --project="$CHLU_PROJECT" --custom-metadata="sha256=$MF_SHA" -q

cost_line
say "staged. ⭐ NOW RUN THE VERIFY ON A VM — a file that transferred and will not load is a failure,"
say "   not a success, and only the VM can tell you which (see scripts/cloud/verify_gcs.sh's header)."
say "the job's env:"
print_env_block | sed 's/^/    /'
