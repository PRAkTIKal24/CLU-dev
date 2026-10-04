#!/usr/bin/env bash
# scripts/verify_hub_workspace.sh — prove a hub workspace bundle is USABLE, ON THE FAR SIDE.
#
# ⭐⭐ OPENING IT IS THE TEST (the verify_artefacts.sh lesson): a byte-perfect file that
# its loader rejects is still a dead run. So this does FIVE things and reports all five:
#
#   1. re-checksum every file in BUNDLE_MANIFEST.json (sha256);
#   2. OPEN every artefact through its REAL production loader (the `open_with` dispatch
#      key, the same functions the stage modules call — ⛔ nothing is re-implemented);
#   3. read the DOCS a hub reads, and assert each one's own anchor is present
#      (a truncated charter checksums fine and is useless);
#   4. import `chlu` and ⛔ ASSERT `chlu.__file__` IS UNDER $CHLU_ROOT
#      — ⚠ `python -c` puts the CWD at sys.path[0] AHEAD of PYTHONPATH, so a verify run
#        from the wrong directory will happily import the ORIGINAL machine's checkout and
#        report a portable bundle that is not portable. This assertion is the whole point;
#   5. run ONE targeted test (`tests/test_hub_workspace.py`);
#   6. assert from GIT that the bundle's ROOT TREE IS THE PIN'S TREE — and that the
#      repository exists at all, because the hub's sandbox is a worktree (§0.2) and
#      a plain `git archive` directory can hold none.
#
# It writes `WORKSPACE-READY.json` and exits non-zero unless `status == "ready"`.
#
# ---------------------------------------------------------------------------
#   cd <bundle> && source ./env_hub.sh && ./scripts/verify_hub_workspace.sh
#
# Knobs: CHLU_ROOT (default: this script's repo root) · MANIFEST · READY_OUT
#        VERIFY_PYTHON (default python3) · SKIP_LOAD=1 (checksums only — ⛔ NOT a pass)
#        SKIP_TEST=1 (⛔ NOT a pass) · HUB_SLUG (which task file must be present)
#        CHLU_WORKSPACE_VERIFY_DEPTH (recursion guard; ⛔ set by this script, not by you)
# ---------------------------------------------------------------------------

set -uo pipefail

die() { echo "FATAL[verify_hub_workspace]: $*" >&2; exit 2; }
say() { echo "[verify] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHLU_ROOT="${CHLU_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CHLU_ROOT="$(cd "$CHLU_ROOT" && pwd)" || die "no root at $CHLU_ROOT"

# ⭐ Derive every root from CHLU_ROOT (honouring an explicit override), exactly as
#    env_hub.sh does — so the verify proves the SAME resolution a real hub will get.
export CHLU_ROOT
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CHLU_ROOT/.claude}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_WIDE_BRIDGE_ROOT="${CHLU_GEN_WIDE_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-bridge-wide/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"
export CHLU_HUB_OUTPUTS="${CHLU_HUB_OUTPUTS:-$CHLU_CLAUDE_ROOT/outputs}"

# ---------------------------------------------------------------------------
# ⛔⛔ RECURSION GUARD. This script runs pytest; that pytest file builds bundles and
# runs THIS SCRIPT. ⚠ The guard used to be implicit ("a bundle is not a git repo, so
# the round-trip tests skip there") — and it silently stopped holding the moment the
# bundle became a git repository: the run forked unboundedly and 28 processes had to
# be killed by hand. ⭐ The depth is now explicit, exported, and asserted.
# ---------------------------------------------------------------------------
VERIFY_DEPTH="${CHLU_WORKSPACE_VERIFY_DEPTH:-0}"
case "$VERIFY_DEPTH" in ''|*[!0-9]*) VERIFY_DEPTH=0 ;; esac
if [ "$VERIFY_DEPTH" -ge 2 ]; then
    die "recursion guard: CHLU_WORKSPACE_VERIFY_DEPTH=$VERIFY_DEPTH. ⛔ verify -> pytest -> verify is forking; refusing."
fi

MANIFEST="${MANIFEST:-$CHLU_ROOT/BUNDLE_MANIFEST.json}"
READY_OUT="${READY_OUT:-$CHLU_ROOT/WORKSPACE-READY.json}"
PY_BIN="${VERIFY_PYTHON:-python3}"

[ -f "$MANIFEST" ] || die "no manifest at $MANIFEST — it travels WITH the bundle; re-run bundle_hub_workspace.sh"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN'"

say "root     : $CHLU_ROOT"
say "manifest : $MANIFEST"
say "python   : $PY_BIN"
say "out      : $READY_OUT"

rc=0
VF_ROOT="$CHLU_ROOT" VF_MANIFEST="$MANIFEST" VF_OUT="$READY_OUT" \
VF_SKIP_LOAD="${SKIP_LOAD:-0}" VF_HOST="$(hostname)" VF_HUB="${HUB_SLUG:-}" \
"$PY_BIN" - <<'PY' || rc=$?
import datetime as _dt, hashlib, json, os, platform, sys, traceback
from pathlib import Path

e = os.environ.get
root = Path(e("VF_ROOT")).resolve()
manifest = json.loads(Path(e("VF_MANIFEST")).read_text(encoding="utf-8"))
skip_load = e("VF_SKIP_LOAD") == "1"

if manifest.get("schema") != "chlu-hub-workspace-bundle/1":
    sys.exit("FATAL: %r is not a chlu-hub-workspace-bundle/1" % e("VF_MANIFEST"))

# ---------------------------------------------------------------------------
# ⭐ SUPPLEMENTS ARE VERIFIED TOO. `--supplement-apply` tops a workspace up without
#    touching BUNDLE_MANIFEST.json (⛔ a hub may be running in here), so its rows would
#    otherwise be the one part of the workspace nothing ever opens — which is exactly
#    the hole the §2b omission left in the first place.
#    ⚠ Member paths in a supplement are relative to its `member_root` (`.claude`).
# ---------------------------------------------------------------------------
supplements = []
for sp in sorted(root.glob("SUPPLEMENT_MANIFEST*.json")):
    sm = json.loads(sp.read_text(encoding="utf-8"))
    if sm.get("schema") != "chlu-hub-workspace-supplement/1":
        sys.exit("FATAL: %s is not a chlu-hub-workspace-supplement/1" % sp.name)
    base = sm.get("member_root", ".claude")
    known = {f["path"] for f in manifest["files"]}
    for rec in sm["files"]:
        rec = dict(rec, path="%s/%s" % (base, rec["path"]))
        if rec["path"] not in known:
            manifest["files"].append(rec)
            known.add(rec["path"])
    supplements.append({"manifest": sp.name, "hubs": sm.get("hubs"),
                        "n_files": sm.get("n_files"), "generated_at": sm.get("generated_at")})
if supplements:
    print("[verify] supplements: %s" % ", ".join(s["manifest"] for s in supplements))


def note(r, msg):
    """Accumulate a verdict. ⛔ Never overwrite an earlier one: a file can fail BOTH
    the checksum and the loader, and one clobbering the other is how a corrupted
    transfer gets misdiagnosed as a schema problem."""
    r["error"] = msg if not r["error"] else r["error"] + " | " + msg


def sha256_of(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ⭐ THE LOADERS. Each key calls the SAME function the production module calls.
#    ⛔ Nothing here re-implements a parse: if the real loader would reject the
#    file, this rejects it. (The seven artefact keys are the same set
#    `scripts/csf3/verify_artefacts.sh` dispatches on, so a bundle verified here
#    and a stage verified on CSF3 are making the same statement.)
# ---------------------------------------------------------------------------
def _load_json(p):
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    return {"keys": len(d) if isinstance(d, dict) else None}


def _load_eqx_pickle(p):
    from chlu.utils.checkpoints import load_model
    import jax
    m = load_model(p)
    return {"n_leaves": len(jax.tree_util.tree_leaves(m))}


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
    names = json.loads(Path(p).read_text(encoding="utf-8"))["factors"]["names"]
    return {"n_records": len(load_c_requests(p, names)), "factors": names}


# --- the hub addendum's three keys -----------------------------------------
def _load_query_registry(p):
    # ⭐ The real loader, and it opens every set the registry declares — so a
    #    registry that names a file we failed to ship fails HERE, not on hour 6.
    from chlu.data.gen_splits import load_registered_query_sets
    reg, sets = load_registered_query_sets(p)
    return {"sets": {k: len(v) for k, v in sorted(sets.items())},
            "dataset": reg.get("dataset")}


def _load_control_set(p):
    # ⭐ The real loader `exp_gen_baselines.py:1562` calls, and it is registry-driven:
    #    the set is looked up BY NAME in query_sets.json, so a control whose registered
    #    filename and shipped filename disagree fails here.
    from chlu.core.gen_request import load_registered_control_requests
    d = Path(p).parent
    registry = json.loads((d / "query_sets.json").read_text(encoding="utf-8"))
    name = next((k for k, s in registry.get("sets", {}).items()
                 if s.get("file") == Path(p).name), None)
    if name is None:
        raise KeyError("%s is not named by any set in %s" % (Path(p).name, d / "query_sets.json"))
    sets = load_registered_control_requests(d, registry, [name])
    return {"name": name, "n_records": len(sets[name])}


def _load_npz(p):
    """⭐ A generic ``.npz`` — the shape most §2b hub-requirement artefacts take
    (energy traces, settled states, decoded tuples). ⛔ `np.load` alone is LAZY: a
    truncated member only raises when it is actually read, i.e. on the far side, six
    hours in. So every array is materialised here."""
    import numpy as np
    with np.load(p) as z:
        names = list(z.files)
        shapes = {k: [list(np.asarray(z[k]).shape), str(np.asarray(z[k]).dtype)]
                  for k in names}
    return {"n_arrays": len(names), "arrays": shapes}


def _load_npz_index(p):
    from chlu.data.gen_splits import load_index
    grid, train_rows, test_rows, _ = load_index(p)
    return {"factors": list(grid.names), "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows))}


# --- docs: checksum is not enough; a truncated charter checksums fine -------
DOC_ANCHORS = {
    ".claude/AGENT_PROTOCOL.md": ["CHLU Spoke-Agent Protocol", "DIAL DECLARATION"],
    ".claude/autonomous-hub-charter.md": [
        "AUTONOMOUS HUB CHARTER",
        # ⛔ the pin itself: a charter that does not carry it cannot pin a hub
        "ae907466ba2b5fbec124e8995cf4d7c32a1ab30e",
        # ⛔ §C's floor and §B's philosophy: a charter truncated above either one
        #    licenses exactly the degenerate ten hours the profile warns about
        "MATCHED-BYTES LOOKUP", "core philosophy",
    ],
    ".claude/agents/autonomous-hub.md": [
        "autonomous-hub", "SECTION 0", "YOU NEVER DELETE", "PRE-REGISTRATION",
    ],
    ".claude/handover_context.md": ["Known Issues"],
}


def _load_doc(p):
    # ⭐⭐ `.as_posix()`, ⛔ NOT `str()`. On Windows `Path.relative_to()` yields a path
    #    whose str() is BACKSLASH-separated, while DOC_ANCHORS is keyed with forward
    #    slashes ⇒ the lookup ALWAYS missed, `missing` was ALWAYS [], and every document
    #    returned `loaded: true` having inspected ZERO anchors. ⛔ Check #3 degraded to
    #    "the file is non-empty", and a truncated charter — the exact thing it exists to
    #    catch — reported success. ⛔ The fix normalises the KEY; it does not loosen the
    #    check. (tests/test_hub_workspace.py exercises this source under a Windows path
    #    flavour, because fixing it by inspection is what this defect proves cannot work.)
    rel = Path(p).resolve().relative_to(root).as_posix()
    # ⚠ The encoding is pinned IN THE CODE. The Windows default is cp1252, which turns
    #    every ⭐/⛔/⚠ in these documents into three replacement characters — a silent
    #    corruption of the text the anchors are matched against. ⛔ PYTHONUTF8=1 is a
    #    fine workaround but it is an environment someone has to remember to set.
    txt = Path(p).read_text(encoding="utf-8", errors="replace")
    if not txt.strip():
        raise ValueError("document is empty")
    missing = [a for a in DOC_ANCHORS.get(rel, []) if a not in txt]
    if missing:
        raise ValueError("required anchor(s) absent: %s" % ", ".join(repr(m) for m in missing))
    return {"chars": len(txt), "lines": txt.count("\n") + 1,
            "anchors_checked": len(DOC_ANCHORS.get(rel, []))}


def _load_code(p):
    """⭐ Opening a script means PARSING it. A truncated shell script passes every
    byte check and then dies at line 400 on the far side, six hours in."""
    p = Path(p)
    if p.suffix == ".py":
        src = p.read_text(encoding="utf-8")
        compile(src, str(p), "exec")
        return {"lines": src.count("\n") + 1, "parsed": "python"}
    import subprocess
    r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SyntaxError("bash -n: %s" % (r.stderr.strip() or r.returncode))
    if not os.access(p, os.X_OK):
        raise PermissionError("%s lost its executable bit in transit" % p.name)
    return {"lines": p.read_text(encoding="utf-8").count("\n") + 1, "parsed": "bash -n"}


LOADERS = {
    "code": _load_code,
    "json": _load_json,
    "eqx_pickle": _load_eqx_pickle,
    "feature_cache": _load_feature_cache,
    "npy": _load_npy,
    "store_plan": _load_store_plan,
    "winner_hashes": _load_winner_hashes,
    "c_invalid": _load_c_invalid,
    "query_registry": _load_query_registry,
    "control_set": _load_control_set,
    "npz_index": _load_npz_index,
    "npz": _load_npz,
    "doc": _load_doc,
}

results, n_ck_bad, n_load_bad, n_missing = [], 0, 0, 0
total = 0
for rec in manifest["files"]:
    rel = rec["path"]
    p = root / rel
    r = {"path": rel, "kind": rec.get("kind"), "loader": rec["loader"],
         "open_with": rec["open_with"], "present": p.is_file(),
         "bytes_expected": rec["bytes"], "bytes_actual": None,
         "checksum_match": False, "loaded": False, "detail": None,
         "error": None, "checksum_error": None, "load_error": None}
    if not r["present"]:
        note(r, "MISSING on the far side")
        n_missing += 1
        results.append(r)
        print("[verify] FAIL %-88s %s" % (rel, r["error"]))
        continue
    r["bytes_actual"] = p.stat().st_size
    total += r["bytes_actual"]
    got = sha256_of(p)
    r["sha256_actual"] = got
    r["checksum_match"] = (got == rec["sha256"])
    if not r["checksum_match"]:
        r["checksum_error"] = "sha256 MISMATCH: %s != %s (expected)" % (got[:16], rec["sha256"][:16])
        note(r, r["checksum_error"])
        n_ck_bad += 1
    if skip_load:
        r["detail"] = "SKIP_LOAD=1 — ⛔ checksums only, this is NOT a pass"
    else:
        try:
            r["detail"] = LOADERS[rec["open_with"]](p)
            r["loaded"] = True
        except Exception as exc:                        # noqa: BLE001
            r["loaded"] = False
            r["load_error"] = "%s: %s" % (type(exc).__name__, exc)
            note(r, r["load_error"])
            r["traceback"] = traceback.format_exc(limit=6)
            n_load_bad += 1
    results.append(r)
    flag = "ok  " if (r["checksum_match"] and (r["loaded"] or skip_load)) else "FAIL"
    print("[verify] %s %-88s %s" % (flag, rel, r["error"] or ""))

# ---------------------------------------------------------------------------
# ⭐⭐ THE PORTABILITY ASSERTION. ⛔ Without this, everything above can pass while
# the run is silently reading the ORIGINAL machine's checkout.
# ---------------------------------------------------------------------------
imp = {"ok": False, "chlu_file": None, "under_root": False, "jax": None,
       "error": None, "sys_path_0": sys.path[0]}
if skip_load:
    imp["error"] = "SKIP_LOAD=1 — import NOT attempted"
else:
    try:
        import chlu
        imp["chlu_file"] = str(Path(chlu.__file__).resolve())
        imp["chlu_version"] = getattr(chlu, "__version__", None)
        imp["under_root"] = Path(imp["chlu_file"]).is_relative_to(root)
        try:
            import jax
            imp["jax"] = jax.__version__
            imp["jax_devices"] = [str(d) for d in jax.devices()]
        except Exception as exc:                        # noqa: BLE001
            imp["error"] = "jax: %s: %s" % (type(exc).__name__, exc)
        imp["ok"] = bool(imp["under_root"]) and imp["error"] is None
        if not imp["under_root"]:
            imp["error"] = (
                "⛔ chlu imported from %s, which is OUTSIDE $CHLU_ROOT=%s. "
                "The bundle is NOT being exercised — this run proves nothing about "
                "portability. (`python -c` puts the CWD at sys.path[0] ahead of "
                "PYTHONPATH; run from $CHLU_ROOT, or source env_hub.sh.)"
                % (imp["chlu_file"], root)
            )
    except Exception as exc:                            # noqa: BLE001
        imp["error"] = "%s: %s" % (type(exc).__name__, exc)
        imp["traceback"] = traceback.format_exc(limit=6)
print("[verify] import: chlu=%s under_root=%s jax=%s %s"
      % (imp["chlu_file"], imp["under_root"], imp["jax"], imp["error"] or ""))

# ---------------------------------------------------------------------------
# ⭐ THE CODE IS THE PIN — asserted from git, ⛔ not taken on trust.
#
# The bundle's ROOT commit holds the pin's tree verbatim; every later commit (the
# declared tooling overlay, and whatever the hub itself commits) sits on top. So the
# root commit's tree hash is a stable, tamper-evident statement that the code a hub
# is about to run is the code the charter pinned.
# ---------------------------------------------------------------------------
gitc = {"repo": (root / ".git").exists(), "root_tree": None, "pin_tree": None,
        "matches_pin": None, "head": None, "worktree_possible": False, "error": None}
if gitc["repo"]:
    import subprocess as _sp

    def _g(*a):
        r = _sp.run(["git", "-C", str(root), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    gitc["pin_tree"] = (manifest.get("code") or {}).get("pin_tree")
    base = _g("rev-list", "--max-parents=0", "HEAD")
    gitc["root_commit"] = base
    gitc["root_tree"] = _g("rev-parse", "%s^{tree}" % base) if base else None
    gitc["head"] = _g("rev-parse", "--short", "HEAD")
    gitc["branch"] = _g("rev-parse", "--abbrev-ref", "HEAD")
    if gitc["pin_tree"] and gitc["root_tree"]:
        gitc["matches_pin"] = gitc["root_tree"] == gitc["pin_tree"]
        if not gitc["matches_pin"]:
            gitc["error"] = ("⛔ the bundle's root tree %s is NOT the pin's tree %s — "
                             "the code a hub would run is not the code the charter pinned"
                             % (gitc["root_tree"], gitc["pin_tree"]))
    gitc["worktree_possible"] = base is not None
else:
    gitc["error"] = ("⚠ no git repository — the hub's sandbox is a WORKTREE (agent "
                     "profile §0.2) and it cannot create one here. Rebuild without --no-git.")
print("[verify] git   : root_tree=%s matches_pin=%s head=%s %s"
      % (gitc["root_tree"], gitc["matches_pin"], gitc["head"], gitc["error"] or ""))

# --- the hub's write path must exist and be writable ------------------------
PROBE_NAME = ".workspace-write-probe.json"
hub = e("VF_HUB") or (manifest.get("hubs") or [None])[0]
wr = {"hub_slug": hub, "path": None, "writable": False, "error": None}
if hub:
    d = Path(e("CHLU_HUB_OUTPUTS", str(root / ".claude" / "outputs"))) / hub
    wr["path"] = str(d)
    try:
        d.mkdir(parents=True, exist_ok=True)
        # ⭐ ONE fixed path, rewritten each run. ⚠ A timestamped probe per run
        #    accumulates inside the hub's own results directory and then rides home
        #    in the return tar — observed, and fixed here. `--return-pack` skips
        #    this exact basename by name (⛔ never by a blanket dotfile filter).
        probe = d / PROBE_NAME
        stamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        probe.write_text(json.dumps(
            {"written_by": "verify_hub_workspace.sh", "at": stamp,
             "note": "the verifier's own write probe; ⛔ not a hub artefact"}) + "\n")
        wr["writable"] = json.loads(probe.read_text(encoding="utf-8"))["written_by"] == "verify_hub_workspace.sh"
        wr["probe"] = str(probe)      # ⛔ left in place: §0.1, we never delete
    except Exception as exc:                            # noqa: BLE001
        wr["error"] = "%s: %s" % (type(exc).__name__, exc)
print("[verify] write : %s writable=%s %s" % (wr["path"], wr["writable"], wr["error"] or ""))

all_ck = all(r["checksum_match"] for r in results)
all_ld = all(r["loaded"] for r in results) if not skip_load else False
ok = bool(all_ck and all_ld and not n_missing and imp["ok"]
          and (wr["writable"] or not hub)
          and gitc["matches_pin"] is not False and gitc["repo"])
out = {
    "schema": "chlu-hub-workspace-ready/1",
    "status": "ready" if ok else "NOT-READY",
    "checksum_match": bool(all_ck and not n_missing),
    "loaders_all_opened": bool(all_ld),
    "import_under_root": bool(imp["under_root"]),
    "code_is_the_pin": gitc["matches_pin"],
    "worktree_possible": bool(gitc["worktree_possible"]),
    "hub_output_path_writable": bool(wr["writable"]),
    "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "host": e("VF_HOST"),
    "platform": {"system": platform.system(), "release": platform.release(),
                 "machine": platform.machine(), "python": platform.python_version()},
    "chlu_root": str(root),
    "env": {k: e(k) for k in sorted(os.environ) if k.startswith("CHLU_")},
    "manifest": e("VF_MANIFEST"),
    "manifest_generated_at": manifest.get("generated_at"),
    "pin": manifest.get("pin"),
    "hubs": manifest.get("hubs"),
    "artefact_axes": manifest.get("artefact_axes"),
    "hub_requirements": manifest.get("hub_requirements"),
    "supplements": supplements,
    "n_files": len(results),
    "n_missing": n_missing,
    "n_checksum_mismatch": n_ck_bad,
    "n_load_failed": n_load_bad,
    "total_bytes_on_disk": total,
    "skip_load": skip_load,
    "import_check": imp,
    "git_check": gitc,
    "write_check": wr,
    "targeted_test": "NOT RUN (the shell runs it after this block and rewrites this key)",
    "files": results,
}
Path(e("VF_OUT")).parent.mkdir(parents=True, exist_ok=True)
with open(e("VF_OUT"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, sort_keys=True, default=str)
    fh.write("\n")
print("[verify] -> %s" % e("VF_OUT"))
print("[verify] files=%d missing=%d mismatched=%d unopenable=%d import_ok=%s"
      % (len(results), n_missing, n_ck_bad, n_load_bad, imp["ok"]))
sys.exit(0 if ok else 1)
PY

# ---------------------------------------------------------------------------
# 5. ⭐ ONE TARGETED TEST — ⛔ not the suite. Runs from $CHLU_ROOT so the bundle's
#    own `chlu` is the one under test.
# ---------------------------------------------------------------------------
TEST_FILE="$CHLU_ROOT/tests/test_hub_workspace.py"
TEST_STATUS="NOT RUN"
TEST_RC=0
if [ "${SKIP_TEST:-0}" = "1" ]; then
    TEST_STATUS="SKIPPED (SKIP_TEST=1 — ⛔ NOT a pass)"
elif [ ! -f "$TEST_FILE" ]; then
    TEST_STATUS="NOT RUN: $TEST_FILE is absent"
    rc=1
else
    say "targeted test: $TEST_FILE"
    TEST_LOG="$CHLU_ROOT/.workspace-verify-test.log"
    # ⭐ the depth the child sees; its @needs_bundle tests skip on it
    ( cd "$CHLU_ROOT" && CHLU_ROOT="$CHLU_ROOT" \
      CHLU_WORKSPACE_VERIFY_DEPTH="$((VERIFY_DEPTH + 1))" \
      "$PY_BIN" -m pytest -q "$TEST_FILE" ) 2>&1 | tee "$TEST_LOG"
    TEST_RC="${PIPESTATUS[0]}"
    TEST_SUMMARY="$(grep -Eo '[0-9]+ (passed|failed|skipped|error)[^|]*' "$TEST_LOG" | tail -1)"
    if [ "$TEST_RC" -eq 0 ]; then
        TEST_STATUS="passed: ${TEST_SUMMARY:-ok}"
    else
        TEST_STATUS="FAILED (rc=$TEST_RC): ${TEST_SUMMARY:-see $TEST_LOG}"
        rc=1
    fi
fi
say "targeted test: $TEST_STATUS"

# fold the test verdict into WORKSPACE-READY.json (and downgrade status if it failed)
RD_OUT="$READY_OUT" RD_TEST="$TEST_STATUS" RD_RC="$TEST_RC" "$PY_BIN" - <<'PY'
import json, os
p = os.environ["RD_OUT"]
d = json.load(open(p, encoding="utf-8"))
d["targeted_test"] = os.environ["RD_TEST"]
if os.environ["RD_RC"] != "0" or os.environ["RD_TEST"].startswith(("NOT RUN", "SKIPPED", "FAILED")):
    d["status"] = "NOT-READY"
json.dump(d, open(p, "w", encoding="utf-8"), indent=2, sort_keys=True, default=str)
open(p, "a", encoding="utf-8").write("\n")
print("[verify] status=%s" % d["status"])
PY

if [ "$rc" -eq 0 ]; then
    say "=== ⭐ WORKSPACE READY: checksummed, every artefact OPENED through its real loader,"
    say "    chlu imported FROM THE BUNDLE, the hub's output path written, targeted test green. ==="
else
    say "=== ⛔ NOT READY — see $READY_OUT. ⛔ Do not launch a hub. ==="
fi
exit "$rc"
