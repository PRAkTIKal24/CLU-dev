#!/usr/bin/env bash
# scripts/cloud/verify_gcs.sh — prove the GCS-staged gen artefact set is USABLE,
# ⭐⭐ ON A VM. ⛔ A laptop run proves the laptop can open them; the question is
# whether the LOADERS WORK THERE.
#
# ⭐⭐ OPENING IT IS THE TEST. A byte-perfect artefact that its loader rejects is
# still a dead run, so this does TWO things per file and reports both:
#   1. re-checksum (sha256) against `stage_manifest.json`, and
#   2. OPEN it through its REAL production loader (the `open_with` dispatch key
#      in the manifest maps onto the same functions the stage modules call).
# It writes `STAGED-GCS.json` — the schema of `STAGED.json`
# (`chlu-gen-staged/1`), same keys, plus three additive GCP-only keys
# (`gcs_root`, `vm`, `dataset`) so a reader of the CSF3 file is unaffected.
#
# ⛔ THIS IS A TRANSPORT PORT of `scripts/csf3/verify_artefacts.sh`. The loader
#    dispatch table below is that file's, function for function. ⛔ Nothing here
#    re-implements a parse: if the production loader would reject the file,
#    this rejects it.
#
# ---------------------------------------------------------------------------
# HOW IT GETS ONTO A VM (the analogue of the `#SBATCH` header over on CSF3)
# ---------------------------------------------------------------------------
# ⛔ There is no launcher here on purpose: `scripts/cloud/launch.sh` launches
#    GPU JOB VMs and belongs to another agent; a second launch path would
#    diverge from it silently. A verify is a one-off on a CPU VM, and this is
#    the whole of it (the image already carries `chlu`, jax and gcloud):
#
#   source scripts/cloud/config.sh
#   gcloud storage cp scripts/cloud/verify_gcs.sh \
#       "gs://$CHLU_BUCKET/artefacts/_verify/verify_gcs.sh"
#   gcloud compute instances create chlu-verify-gcs \
#     --project="$CHLU_PROJECT" --zone="$CHLU_ZONE" --machine-type=e2-standard-4 \
#     --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
#     --boot-disk-size=60GB --boot-disk-type=pd-balanced \
#     --scopes=cloud-platform --labels=owner=chlu,job=verify-gcs \
#     --metadata=chlu-image="$(cat scripts/cloud/IMAGE_DIGEST)",chlu-bucket="$CHLU_BUCKET" \
#     --metadata-from-file=startup-script=<a 15-line pull-and-run; see the report>
#
# ⚠ A GPU is NOT needed and not wanted: every loader here is numpy/pickle/json
#   on the CPU, and an L4 would cost 6× for the same answer.
#
# ---------------------------------------------------------------------------
# LOCALLY (the failure demo — ⭐ a checker that cannot fail proves nothing)
# ---------------------------------------------------------------------------
#   VERIFY_PYTHON=<repo>/.venv/bin/python bash scripts/cloud/verify_gcs.sh \
#       --no-fetch --art-root <a local copy> --out /tmp/LOCAL-VERIFY.json
#   # then flip one byte in a .npy, and truncate-but-keep-the-sha of a .pkl.
#
# Knobs: GCS_ART_ROOT (default gs://$CHLU_BUCKET/artefacts) · ART_ROOT
#        (default /var/chlu/artefacts) · MANIFEST · STAGED_OUT · VERIFY_PYTHON
#        SKIP_LOAD=1 (checksums only — ⛔ NOT a pass; for triage of a broken env)
# Flags: --gcs-root URL | --art-root PATH | --out PATH | --no-fetch | --no-upload
#        --with-dataset | --allow-pip | --skip-load | -h

set -eo pipefail

die() { echo "FATAL[verify_gcs]: $*" >&2; exit 2; }
say() { echo "[verify-gcs] $*"; }
usage() { sed -n '2,58p' "$BASH_SOURCE"; exit "${1:-1}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.sh"
fi
CHLU_BUCKET="${CHLU_BUCKET:-chlu-cloud-forgisprova}"
GCS_ART_ROOT="${GCS_ART_ROOT:-gs://${CHLU_BUCKET}/${GCS_ART_PREFIX:-artefacts}}"
ART_ROOT="${ART_ROOT:-/var/chlu/artefacts}"
STAGED_OUT="${STAGED_OUT:-}"
FETCH=1
UPLOAD=1
WITH_DATASET=0
ALLOW_PIP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --gcs-root)     GCS_ART_ROOT="${2:?--gcs-root needs a URL}"; shift ;;
        --art-root)     ART_ROOT="${2:?--art-root needs a path}"; shift ;;
        --out)          STAGED_OUT="${2:?--out needs a path}"; shift ;;
        --no-fetch)     FETCH=0 ;;
        --no-upload)    UPLOAD=0 ;;
        --with-dataset) WITH_DATASET=1 ;;
        --allow-pip)    ALLOW_PIP=1 ;;
        --skip-load)    SKIP_LOAD=1 ;;
        -h|--help)      usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
    shift
done

MF_REL="outputs/gen-cluster-stage-artefacts/stage_manifest.json"
MANIFEST="${MANIFEST:-$ART_ROOT/$MF_REL}"
STAGED_OUT="${STAGED_OUT:-$ART_ROOT/outputs/gen-cluster-stage-artefacts/STAGED-GCS.json}"

PY_BIN="${VERIFY_PYTHON:-python3}"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN'"

# ---------------------------------------------------------------------------
# 1. TRANSPORT — the only thing that differs from the CSF3 verify.
#    `rsync -r` is idempotent and resumable, so a VM that died mid-fetch is
#    re-runnable and a re-verify after a re-stage costs one listing.
# ---------------------------------------------------------------------------
if [ "$FETCH" = "1" ]; then
    command -v gcloud >/dev/null 2>&1 || die "no gcloud on PATH (the image ships 585.0.0 at /opt/google-cloud-sdk/bin)"
    mkdir -p "$ART_ROOT"
    say "fetch    : $GCS_ART_ROOT/  ->  $ART_ROOT"
    gcloud storage rsync -r "$GCS_ART_ROOT" "$ART_ROOT" ${CHLU_PROJECT:+--project="$CHLU_PROJECT"}
fi

[ -d "$ART_ROOT" ] || die "no artefact root at $ART_ROOT — run stage_gcs.sh, then re-run without --no-fetch"
[ -f "$MANIFEST" ] || die "no manifest at $MANIFEST — it travels with the set; re-run scripts/cloud/stage_gcs.sh"

# ⭐ The loaders resolve their defaults against these; export them here so the
#    verify proves the SAME resolution a real gen job on this VM will get.
#    ⛔ These five names are the contract with `startup.sh` (gen-cloud-runner).
export CHLU_CLAUDE_ROOT="$ART_ROOT"
export CHLU_GEN_DATA_ROOT="$ART_ROOT/data/gen"
export CHLU_GEN_BRIDGE_ROOT="$ART_ROOT/outputs/gen-feature-bridge/runs"
export CHLU_GEN_STORE_RUNS="$ART_ROOT/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="$ART_ROOT/outputs/gen-decoder-lift/winner"

say "root     : $ART_ROOT"
say "manifest : $MANIFEST"
say "out      : $STAGED_OUT"
[ "$WITH_DATASET" = "1" ] && say "dataset  : 3dshapes.h5 will be DOWNLOADED here (not staged) and opened"

rc=0
VF_ROOT="$ART_ROOT" VF_MANIFEST="$MANIFEST" VF_OUT="$STAGED_OUT" \
VF_SKIP_LOAD="${SKIP_LOAD:-0}" VF_HOST="$(hostname)" VF_GCS="$GCS_ART_ROOT" \
VF_DATASET="$WITH_DATASET" VF_ALLOW_PIP="$ALLOW_PIP" \
"$PY_BIN" - <<'PY' || rc=$?
import datetime as _dt, hashlib, json, os, subprocess, sys, traceback
from pathlib import Path

e = os.environ.get
root = Path(e("VF_ROOT"))
manifest = json.loads(Path(e("VF_MANIFEST")).read_text())
skip_load = e("VF_SKIP_LOAD") == "1"

if manifest.get("schema") != "chlu-gen-stage-manifest/1":
    sys.exit("FATAL: %r is not a chlu-gen-stage-manifest/1" % e("VF_MANIFEST"))


def note(r, msg):
    """Accumulate a verdict on one file. ⛔ Never overwrite an earlier one."""
    r["error"] = msg if not r["error"] else r["error"] + " | " + msg


def sha256_of(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ⭐ THE LOADERS — verbatim from scripts/csf3/verify_artefacts.sh. Each dispatch
#    key calls the SAME function the stage module calls.
# ---------------------------------------------------------------------------
def _load_json(p):
    d = json.loads(Path(p).read_text())
    return {"keys": len(d) if isinstance(d, dict) else None}


def _load_eqx_pickle(p):
    from chlu.utils.checkpoints import load_model
    import jax
    m = load_model(p)
    leaves = [x for x in jax.tree_util.tree_leaves(m)]
    return {"n_leaves": len(leaves)}


def _load_feature_cache(p):
    from chlu.core.gen_bridge import load_feature_cache
    rows, feats, man = load_feature_cache(Path(p).parent, verify=True)
    return {"n_rows": int(rows.size), "feat_shape": list(man["feat_shape"]),
            "scope": man.get("scope"), "dtype": man.get("dtype")}


def _load_npy(p):
    import numpy as np
    a = np.load(p, mmap_mode="r")
    return {"shape": list(a.shape), "dtype": str(a.dtype)}


def _load_store_plan(p):
    from chlu.core.gen_store import load_store_plan
    plan = load_store_plan(p)
    return {"K": int(plan["K"]), "n_modes": int(plan["n_modes"]), "D": int(plan["D"])}


def _load_winner_hashes(p):
    from chlu.core.gen_dream import load_winner_hashes
    h = load_winner_hashes(p)
    return {"seeds": sorted(k for k in h if k.startswith("seed_"))}


def _load_c_invalid(p):
    from chlu.core.gen_certificate import load_c_requests
    names = json.loads(Path(p).read_text())["factors"]["names"]
    recs = load_c_requests(p, names)
    return {"n_records": len(recs), "factors": names}


LOADERS = {
    "json": _load_json,
    "eqx_pickle": _load_eqx_pickle,
    "feature_cache": _load_feature_cache,
    "npy": _load_npy,
    "store_plan": _load_store_plan,
    "winner_hashes": _load_winner_hashes,
    "c_invalid": _load_c_invalid,
}

results, n_ck_bad, n_load_bad, n_missing = [], 0, 0, 0
total = 0
for rec in manifest["files"]:
    rel = rec["path"]
    p = root / rel
    r = {"path": rel, "loader": rec["loader"], "open_with": rec["open_with"],
         "present": p.is_file(), "bytes_expected": rec["bytes"],
         "bytes_actual": None, "checksum_match": False, "loaded": False,
         "detail": None, "error": None,
         "checksum_error": None, "load_error": None}
    if not r["present"]:
        note(r, "MISSING on the far side")
        n_missing += 1
        results.append(r)
        continue
    r["bytes_actual"] = p.stat().st_size
    total += r["bytes_actual"]
    got = sha256_of(p)
    r["sha256_actual"] = got
    r["checksum_match"] = (got == rec["sha256"])
    if not r["checksum_match"]:
        r["checksum_error"] = ("sha256 MISMATCH: %s != %s (expected)"
                               % (got[:16], rec["sha256"][:16]))
        note(r, r["checksum_error"])
        n_ck_bad += 1
    if skip_load:
        r["detail"] = "SKIP_LOAD=1 — ⛔ checksums only, this is NOT a pass"
    else:
        try:
            r["detail"] = LOADERS[rec["open_with"]](p)
            r["loaded"] = True
        except Exception as exc:                       # noqa: BLE001
            r["loaded"] = False
            r["load_error"] = "%s: %s" % (type(exc).__name__, exc)
            # ⛔ APPEND, never replace: a file can fail BOTH checks and the
            # checksum verdict must survive the loader's verdict.
            note(r, r["load_error"])
            r["traceback"] = traceback.format_exc(limit=6)
            n_load_bad += 1
    results.append(r)
    flag = "ok " if (r["checksum_match"] and (r["loaded"] or skip_load)) else "FAIL"
    print("[verify-gcs] %s %-96s %s" % (flag, rel, r["error"] or ""))

# ---------------------------------------------------------------------------
# ⭐ THE DATASET — the one file this platform resolves DIFFERENTLY from CSF3,
#    and the whole of that difference is WHERE it comes from, not what it is.
#    ⛔ Not staged: downloaded here, by the production path, which asserts
#    sha256 AND the exact byte count — so a truncated fetch cannot pass.
#    Then OPENED, because a checksum is not a load.
# ---------------------------------------------------------------------------
dataset = None
if e("VF_DATASET") == "1":
    from chlu.data.shapes3d import (H5_BYTES, H5_NAME, H5_SHA256, H5_URL,
                                    shapes3d_root)
    dataset = {"name": H5_NAME, "url": H5_URL, "sha256_expected": H5_SHA256,
               "bytes_expected": H5_BYTES, "decision": "VM-DOWNLOADED, not staged",
               "downloaded": False, "sha256_match": False, "opened": False,
               "h5py_installed_at_runtime": False, "detail": None, "error": None}
    try:
        from chlu.data.shapes3d import download_shapes3d
        print("[verify-gcs] downloading %s from %s …" % (H5_NAME, H5_URL))
        info = download_shapes3d()          # ⭐ asserts sha256 + byte count itself
        rec = info[H5_NAME]
        dataset["downloaded"] = True
        dataset["path"] = rec["path"]
        dataset["bytes_actual"] = rec["bytes"]
        dataset["sha256_actual"] = rec["sha256"]
        dataset["sha256_match"] = (rec["sha256"] == H5_SHA256 and rec["bytes"] == H5_BYTES)
        if not dataset["sha256_match"]:
            dataset["error"] = "sha256/bytes disagree with chlu.data.shapes3d"
    except Exception as exc:                            # noqa: BLE001
        dataset["error"] = "download: %s: %s" % (type(exc).__name__, exc)

    if dataset["downloaded"]:
        def _open_dataset():
            from chlu.data.shapes3d import load_labels, open_images
            lab = load_labels()
            f = open_images()
            try:
                shp = list(f["images"].shape)
            finally:
                f.close()
            return {"labels_shape": list(lab.shape), "images_shape": shp,
                    "root": str(shapes3d_root())}
        try:
            dataset["detail"] = _open_dataset()
            dataset["opened"] = True
        except ImportError as exc:
            # ⚠ The shipped image is built `uv sync --extra cuda` — h5py lives
            #    in the `gen` extra, so it is NOT in it. That is an IMAGE gap,
            #    reported as such rather than papered over.
            dataset["error"] = "open: ImportError: %s" % exc
            if e("VF_ALLOW_PIP") == "1":
                print("[verify-gcs] --allow-pip: installing h5py>=3.0 (the `gen` extra) …")
                # ⚠ The image's /opt/venv is a `uv sync` venv and has no pip in
                #    it, so uv is tried first and pip is the fallback.
                import shutil
                if shutil.which("uv"):
                    cmd = ["uv", "pip", "install", "--python", sys.executable, "h5py>=3.0"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", "-q", "h5py>=3.0"]
                subprocess.run(cmd, check=True)
                dataset["h5py_installed_at_runtime"] = True
                try:
                    dataset["detail"] = _open_dataset()
                    dataset["opened"] = True
                    dataset["error"] += " | opened after a runtime h5py install"
                except Exception as exc2:               # noqa: BLE001
                    dataset["error"] += " | retry: %s: %s" % (type(exc2).__name__, exc2)
        except Exception as exc:                        # noqa: BLE001
            dataset["error"] = "open: %s: %s" % (type(exc).__name__, exc)
    print("[verify-gcs] %s %s  %s" % ("ok " if (dataset["sha256_match"] and dataset["opened"])
                                      else "FAIL", H5_NAME, dataset["error"] or ""))

all_ck = all(r["checksum_match"] for r in results)
all_ld = all(r["loaded"] for r in results) if not skip_load else False
status = "complete" if (all_ck and all_ld and not n_missing) else "failed"
if dataset is not None and not (dataset["sha256_match"] and dataset["opened"]):
    status = "failed"
out = {
    "schema": "chlu-gen-staged/1",
    "status": status,
    "checksum_match": bool(all_ck and not n_missing),
    "loaders_all_opened": bool(all_ld),
    "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "host": e("VF_HOST"),
    "slurm_job_id": None,                 # ⭐ key kept: this IS STAGED.json's schema
    "artefact_root": str(root),
    "manifest": e("VF_MANIFEST"),
    "manifest_generated_at": manifest.get("generated_at"),
    "axes": manifest.get("axes"),
    "n_files": len(results),
    "n_missing": n_missing,
    "n_checksum_mismatch": n_ck_bad,
    "n_load_failed": n_load_bad,
    "total_bytes_on_disk": total,
    "skip_load": skip_load,
    "files": results,
    # --- additive, GCP-only ---------------------------------------------------
    "gcs_root": e("VF_GCS"),
    "vm": {"hostname": e("VF_HOST"),
           "instance": os.environ.get("CHLU_VM_NAME"),
           "python": sys.version.split()[0]},
    "dataset": dataset,
}
Path(e("VF_OUT")).parent.mkdir(parents=True, exist_ok=True)
with open(e("VF_OUT"), "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True, default=str)
    fh.write("\n")
print("[verify-gcs] -> %s" % e("VF_OUT"))
print("[verify-gcs] status=%s checksum_match=%s  (%d files, %d missing, %d mismatched, %d failed to load)"
      % (status, out["checksum_match"], len(results), n_missing, n_ck_bad, n_load_bad))
sys.exit(0 if status == "complete" else 1)
PY

# The verdict travels back to the bucket whether it passed or failed: a VM that
# deletes itself takes its local filesystem with it.
if [ "$UPLOAD" = "1" ] && command -v gcloud >/dev/null 2>&1; then
    gcloud storage cp "$STAGED_OUT" "$GCS_ART_ROOT/outputs/gen-cluster-stage-artefacts/$(basename "$STAGED_OUT")" \
        ${CHLU_PROJECT:+--project="$CHLU_PROJECT"} -q || say "⚠ could not upload $STAGED_OUT"
fi

if [ "$rc" -eq 0 ]; then
    say "=== gen artefact set VERIFIED ON THIS MACHINE: checksummed AND opened through every real loader ==="
else
    say "=== ⛔ VERIFY FAILED — see $STAGED_OUT. Re-stage the named files; do NOT launch a gen run. ==="
fi
exit "$rc"
