#!/bin/bash --login
# scripts/csf3/verify_artefacts.sh — prove the staged gen artefact set is USABLE.
#
# ⭐⭐ OPENING IT IS THE TEST. A byte-perfect artefact that its loader rejects is
# still a dead run, so this job does TWO things per file and reports both:
#   1. re-checksum (sha256) against `stage_manifest.json`, and
#   2. OPEN it through its REAL production loader (the `open_with` dispatch key
#      in the manifest maps onto the same functions the stage modules call).
# It writes `STAGED.json` with `status`, `checksum_match` and a per-file result.
#
# ---------------------------------------------------------------------------
# ON CSF3 (login node, from ~/scratch/CHLU) — GPU-LESS, ⛔ never `-p serial`:
#   cd ~/scratch/CHLU && mkdir -p logs
#   sbatch scripts/csf3/verify_artefacts.sh
#   # then: cat ~/scratch/CHLU/artefacts/outputs/gen-cluster-stage-artefacts/STAGED.json
#
# LOCALLY (the failure demo — corrupt a byte and watch it report a mismatch):
#   VERIFY_LOCAL=1 ART_ROOT=<...>/.claude VERIFY_PYTHON=<repo>/.venv/bin/python \
#       bash scripts/csf3/verify_artefacts.sh
#
# Knobs: ART_ROOT (default $HOME/scratch/CHLU/artefacts) · MANIFEST · STAGED_OUT
#        VERIFY_LOCAL=1 (skip module purge / env.sh) · VERIFY_PYTHON
#        SKIP_LOAD=1 (checksums only — ⛔ NOT a pass; for triage of a broken env)
#
# Partition facts (official CSF3 docs, partitions page, mod. 2026-04-15):
#   `multicore` = AMD Genoa, 2–168 cores, wallclock max 7-0, `-t` MANDATORY.
#
#SBATCH -p multicore
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH --job-name=clu-gen-verify
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -eo pipefail

die() { echo "FATAL[verify_artefacts]: $*" >&2; exit 2; }
say() { echo "[verify] $*"; }

ART_ROOT="${ART_ROOT:-$HOME/scratch/CHLU/artefacts}"
MANIFEST="${MANIFEST:-$ART_ROOT/outputs/gen-cluster-stage-artefacts/stage_manifest.json}"
STAGED_OUT="${STAGED_OUT:-$ART_ROOT/outputs/gen-cluster-stage-artefacts/STAGED.json}"
VERIFY_LOCAL="${VERIFY_LOCAL:-0}"

[ -d "$ART_ROOT" ] || die "no artefact root at $ART_ROOT — run stage_artefacts.sh first"
[ -f "$MANIFEST" ] || die "no manifest at $MANIFEST — it travels with the set; re-run stage_artefacts.sh"

if [ "$VERIFY_LOCAL" = "1" ]; then
    export CLU_REPO="${CLU_REPO:-$PWD}"
    say "VERIFY_LOCAL=1 — skipping module purge / env.sh"
else
    if command -v module >/dev/null 2>&1; then module purge; fi
    export CLU_REPO="${CLU_REPO:-$HOME/scratch/CHLU}"
    # shellcheck disable=SC1091
    source "$CLU_REPO/scripts/csf3/env.sh"
fi

PY_BIN="${VERIFY_PYTHON:-python3}"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN'"

# ⭐ The loaders resolve their defaults against these; export them here so the
#    verify job proves the SAME resolution a real stage run will get.
export CHLU_CLAUDE_ROOT="$ART_ROOT"
export CHLU_GEN_DATA_ROOT="$ART_ROOT/data/gen"
export CHLU_GEN_BRIDGE_ROOT="$ART_ROOT/outputs/gen-feature-bridge/runs"
export CHLU_GEN_STORE_RUNS="$ART_ROOT/outputs/gen-store-grouping/runs"
export CHLU_GEN_DECLIFT_DIR="$ART_ROOT/outputs/gen-decoder-lift/winner"

say "root     : $ART_ROOT"
say "manifest : $MANIFEST"
say "out      : $STAGED_OUT"

rc=0
VF_ROOT="$ART_ROOT" VF_MANIFEST="$MANIFEST" VF_OUT="$STAGED_OUT" \
VF_SKIP_LOAD="${SKIP_LOAD:-0}" VF_HOST="$(hostname)" \
"$PY_BIN" - <<'PY' || rc=$?
import datetime as _dt, hashlib, json, os, sys, traceback
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
# ⭐ THE LOADERS. Each dispatch key calls the SAME function the stage module
#    calls. ⛔ Nothing here re-implements a parse: if the production loader
#    would reject the file, this rejects it.
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
    # chlu.core.gen_bridge.load_feature_cache takes the DIRECTORY and verifies
    # features_sha256 itself — i.e. the production checksum gate, not ours.
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
    # ⭐ The real loader. Its `factor_names` argument normally comes from the
    #    dataset grid; the dataset is deliberately NOT staged, so the names are
    #    taken from the file's own header — which still exercises the schema
    #    check, the factor-index/name agreement and the record copy. ⚠ The one
    #    thing this cannot catch is a C set built against a DIFFERENT grid; the
    #    stage run itself asserts that against the loaded split.
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
            # checksum verdict must survive the loader's verdict (and vice
            # versa) — one clobbering the other is how a mismatch gets
            # misdiagnosed as a schema problem.
            note(r, r["load_error"])
            r["traceback"] = traceback.format_exc(limit=6)
            n_load_bad += 1
    results.append(r)
    flag = "ok " if (r["checksum_match"] and (r["loaded"] or skip_load)) else "FAIL"
    print("[verify] %s %-96s %s" % (flag, rel, r["error"] or ""))

all_ck = all(r["checksum_match"] for r in results)
all_ld = all(r["loaded"] for r in results) if not skip_load else False
status = "complete" if (all_ck and all_ld and not n_missing) else "failed"
out = {
    "schema": "chlu-gen-staged/1",
    "status": status,
    "checksum_match": bool(all_ck and not n_missing),
    "loaders_all_opened": bool(all_ld),
    "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "host": e("VF_HOST"),
    "slurm_job_id": e("SLURM_JOB_ID"),
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
}
Path(e("VF_OUT")).parent.mkdir(parents=True, exist_ok=True)
with open(e("VF_OUT"), "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True, default=str)
    fh.write("\n")
print("[verify] -> %s" % e("VF_OUT"))
print("[verify] status=%s checksum_match=%s  (%d files, %d missing, %d mismatched, %d failed to load)"
      % (status, out["checksum_match"], len(results), n_missing, n_ck_bad, n_load_bad))
sys.exit(0 if status == "complete" else 1)
PY

if [ "$rc" -eq 0 ]; then
    say "=== gen artefact set VERIFIED: checksummed AND opened through every real loader ==="
else
    say "=== ⛔ VERIFY FAILED — see $STAGED_OUT. Re-stage the named files; do NOT launch a stage run. ==="
fi
exit "$rc"
