#!/usr/bin/env bash
# scripts/bundle_hub_workspace.sh — build a PORTABLE hub workspace, and carry results back.
#
# ⭐ THE PROBLEM THIS SOLVES
#   Three autonomous hubs are to run on a SECOND MACHINE. ⛔ `git push` does not get
#   them there: `.gitignore:22` excludes `.claude/**`, so a push carries NO charter,
#   NO agent profile, NO hub task file and NO artefact. What a hub *reads* is exactly
#   the part git refuses to move.
#
# ⭐ WHAT IT PRODUCES
#   A directory (or tar) that IS a complete `$CHLU_ROOT`:
#
#     <out>/                       <- this is $CHLU_ROOT on the far side
#       chlu/ scripts/ tests/ …    <- the tracked code AT THE PIN (git archive)
#       .claude/AGENT_PROTOCOL.md
#       .claude/autonomous-hub-charter.md
#       .claude/agents/autonomous-hub.md
#       .claude/handover_context.md
#       .claude/tasks/<hub-slug>.md          (one per --hub)
#       .claude/outputs/**                   <- the FROZEN ARTEFACT SET, loader-derived
#       .claude/data/gen/<ds>/…              <- the derived C set + query-set registry
#       BUNDLE_MANIFEST.json                 <- every file: bytes, sha256, loader, open_with
#       env_hub.sh                           <- $CHLU_ROOT -> every CHLU_* root
#       README-HUB-BUNDLE.md                 <- WSL2, the dataset, the return path
#
# ⭐ REUSE, NOT REWRITE. The artefact set is NOT a directory listing: it is produced by
#   `scripts/csf3/stage_artefacts.sh --manifest-only`, whose table is derived FROM THE
#   LOADERS (each row names the production call site that opens the file). ⛔ That script
#   is not edited and not copied — it is invoked.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   build (default):
#     scripts/bundle_hub_workspace.sh --hub hub-generation-table --out /tmp/chlu-hub-test --dry-run
#     scripts/bundle_hub_workspace.sh --hub hub-generation-table --hub hub-refusal --out <dir>
#     scripts/bundle_hub_workspace.sh --hub hub-refusal --out <dir> --tar      # + <dir>.tar.gz
#
#   ⭐ THE RETURN PATH (§5 — a result that is not synced back does not exist):
#     on the FAR side :  scripts/bundle_hub_workspace.sh --return-pack hub-refusal \
#                            --return-out /tmp/return-hub-refusal.tar.gz
#     on THIS machine :  scripts/bundle_hub_workspace.sh --return-ingest /path/return-hub-refusal.tar.gz
#                        (⛔ NEVER overwrites: an existing destination diverts to
#                         `<slug>.returned-<utc>/` and says so.)
#
# Flags: --hub SLUG (repeatable) · --hub-task PATH (repeatable, overrides the lookup)
#        --out DIR · --pin SHA · --src ROOT · --tar · --dry-run · --force
#        --datasets S · --pretexts S · --seeds S   (passed to stage_artefacts.sh)
#        --no-artefacts (docs+code only; ⛔ NOT a runnable hub workspace)
#        --no-git (skip the git repo; ⚠ the hub's mandatory worktree then cannot exist)
#        --return-pack SLUG · --return-out PATH · --return-ingest TAR · --dest DIR
#        -h|--help
#
#   ⭐ THE PER-HUB ARTEFACT REQUIREMENT (§2b). Every artefact a hub's task file names in
#     backticks is a requirement of that hub's bundle, and an unmet one is a REFUSAL that
#     names the path. ⛔ A charter can no longer promise an input the bundle omits.
#     --print-requirements     show the verdict for a task file; build nothing
#     --require PATH           add a requirement       (repeatable)
#     --exclude-require PATH   drop one                (repeatable)
#     --no-charter-requires    ⛔ derive none — the class of omission is open again
#
#   ⭐ THE SUPPLEMENT — top up a workspace a hub may ALREADY BE RUNNING IN, ⛔ without
#     rebuilding 2.1 GB and ⛔ without replacing one byte of it:
#     scripts/bundle_hub_workspace.sh --supplement --hub hub-refusal --supplement-out ~/s.tar.gz
#     scripts/bundle_hub_workspace.sh --supplement-apply ~/s.tar.gz --dest <the workspace>
#     ⛔ apply NEVER overwrites: identical files are kept, differing ones are written
#        beside the original as `<name>.supplement-<utc>` and reported.
#
# ⚠ macOS has no `timeout` and no GNU coreutils by default; everything here is
#   POSIX-ish + `shasum`/`sha256sum` fallback, the same way stage_artefacts.sh does it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "FATAL[bundle_hub_workspace]: $*" >&2; exit 2; }
say() { echo "[bundle] $*"; }

# ---------------------------------------------------------------------------
# ⭐ $CHLU_ROOT — THE SINGLE SOURCE OF TRUTH FOR EVERY PATH.
#
# The macOS checkout is the DEFAULT, so ⛔ nothing on this machine changes behaviour:
# an unset CHLU_ROOT resolves to exactly the paths that were hardcoded before.
# Every CHLU_* root the production code already honours is derived from it, and any
# one of them may still be overridden individually.
#
# (The code side of this already exists — `chlu/data/gen_splits.default_gen_data_root`,
#  `GenCellConfig.claude_root`, `exp_gen_baselines` … all read these env vars. ⛔ No
#  Python is edited here; this is the shell-side single source of truth they read.)
# ---------------------------------------------------------------------------
CHLU_ROOT_DEFAULT="/Users/user/Desktop/CHLU"

chlu_derive_roots() {   # $1 = root; exports the derived CHLU_* roots (respecting overrides)
    local root="$1"
    export CHLU_ROOT="$root"
    export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$root/.claude}"
    export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
    export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
    export CHLU_GEN_WIDE_BRIDGE_ROOT="${CHLU_GEN_WIDE_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-bridge-wide/runs}"
    export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
    export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"
    # ⭐ The one path a hub may WRITE (agent profile §0.3). Absolute, and derived.
    export CHLU_HUB_OUTPUTS="${CHLU_HUB_OUTPUTS:-$CHLU_CLAUDE_ROOT/outputs}"
}

# ---------------------------------------------------------------------------
# sha256, portably (Linux coreutils vs macOS) — same idiom as stage_artefacts.sh
# ---------------------------------------------------------------------------
if command -v sha256sum >/dev/null 2>&1; then
    sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    die "no sha256sum and no shasum — an unchecksummed bundle is not a bundle"
fi
bytes_of() { wc -c < "$1" | tr -d ' '; }

# ---------------------------------------------------------------------------
# ⛔⛔ APPLE DOUBLE ("._*") — 54 OF THEM REACHED THE FAR SIDE AND `git fsck` FAILED.
#
# `refs/heads/._main` is a MALFORMED REF (`badRefName`), `.git/objects/XX/._<sha>` are
# BAD SHA1 FILES, and `._pack-*.idx` SHADOWS the real pack index on every git call.
# ⚠ `git worktree add` is a hub's literal FIRST move (agent profile §0.2) and
# `git bundle create` is how its code comes home — both write refs and read the object
# store. ⛔ This is the hub's critical path, not cosmetics.
#
# ⭐⭐ WHY NOBODY SAW IT ON THIS MACHINE, AND IT IS NOT SUBTLE:
#   macOS `tar` (bsdtar) writes ONE `._<name>` member PER ENTRY — and then HIDES THEM
#   AGAIN on `tar -t` and `tar -x`, because it knows they are its own metadata. GNU tar,
#   on the far side, has no such knowledge and writes every one of them out as an
#   ORDINARY FILE. ⇒ A check run with macOS `tar` sees a clean archive and the far side
#   still gets broken refs and bad objects.
#   ⭐ MEASURED on one bundle, same tarball, two readers:
#       bsdtar `tar -tzf` : 1193 members,    0 AppleDouble
#       python `tarfile`  : 2386 members, 1193 AppleDouble
#   ⛔ So "our local .git has ZERO ._ files" was never the question: they are created
#      IN TRANSIT, by the tar, exactly as reported.
#
# ⭐ THE FIX, AT CREATION — ⛔ never by deleting anything on the far side:
#   1. COPYFILE_DISABLE=1 (+ the legacy COPY_EXTENDED_ATTRIBUTES_DISABLE) and
#      --no-mac-metadata / --no-xattrs: there is then no metadata to emit, and nothing
#      for a later copy onto exFAT/SMB to materialise as a sidecar either. ⭐ Probed,
#      ⛔ not assumed: GNU tar has neither flag, bsdtar has both.
#   2. --exclude: a `._*` or `.DS_Store` ALREADY on disk is never archived. ⚠ Two
#      patterns, because both tars match `--exclude` against the whole member path.
#      ⛔ These are two NAMED patterns, ⛔ never a blanket dotfile filter — a hub's own
#      `.PREREG` or `.checkpoint` must still travel.
#   3. `assert_no_apple_double`: every archive this script writes is read back with
#      PYTHON, which sees what GNU tar will see, and the build REFUSES if any survive.
# ---------------------------------------------------------------------------
TAR_CLEAN_OPTS=()
for _opt in --no-mac-metadata --no-xattrs; do
    if tar "$_opt" --version >/dev/null 2>&1; then TAR_CLEAN_OPTS+=("$_opt"); fi
done
TAR_AD_EXCLUDE=(--exclude '._*' --exclude '*/._*' --exclude '.DS_Store' --exclude '*/.DS_Store')

chlu_tar() {   # ⭐ the ONLY tar in this script, create or extract
    COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
        tar "${TAR_CLEAN_OPTS[@]+"${TAR_CLEAN_OPTS[@]}"}" "${TAR_AD_EXCLUDE[@]}" "$@"
}

assert_no_apple_double() {   # $1 = archive path
    # ⛔ Read with python, ⛔ NOT with `tar -t`: macOS tar hides exactly the members
    #    we are looking for. This is the check the far side would otherwise perform
    #    for us, six hours in, as a `git fsck` failure.
    AD_TAR="$1" "$PY_BIN" - <<'ADPY' || die "the archive $1 carries AppleDouble members — ⛔ refusing to ship a tar that breaks git on the far side"
import os, sys, tarfile
p = os.environ["AD_TAR"]
with tarfile.open(p) as t:
    bad = [n for n in t.getnames() if n.rsplit("/", 1)[-1].startswith("._")]
if bad:
    print("FATAL[bundle_hub_workspace]: %d AppleDouble member(s) in %s, e.g. %s"
          % (len(bad), p, ", ".join(bad[:5])), file=sys.stderr)
    sys.exit(1)
print("[bundle] apple-double: 0 members in %s (read with python, ⛔ not with macOS tar)" % p)
ADPY
}

# ---------------------------------------------------------------------------
# ⭐⭐ 2b. THE PER-HUB ARTEFACT REQUIREMENT — DERIVED FROM THE HUB'S OWN CHARTER.
#
# ⛔ THE DEFECT THIS CLOSES. `hub-refusal`'s charter calls itself "the cheapest
# unclaimed target" because "it is scored off artefacts that already exist" — and
# NEITHER of the two it names was in the bundle. On the far side it was therefore the
# most EXPENSIVE target, not the cheapest, and the hub would have discovered that after
# it had already spent hours of its ten.
#
# ⛔ Hardcoding those two paths would fix one bundle and leave the CLASS open: the next
# charter promises a different input and the same silence happens again.
# ⭐ So the CHARTER IS THE DECLARATION. Every artefact a hub's task file names in
#   backticks becomes a REQUIREMENT of that hub's bundle, and a requirement that cannot
#   be resolved is a REFUSAL that names the path. A charter can no longer promise an
#   input the bundle does not carry, because the bundle is built FROM the promise.
#
# ⭐ THE RULES, stated so a charter author can predict them (`--print-requirements`
#   shows the verdict for any task file without building anything):
#   * a candidate is a BACKTICKED token containing "/", not absolute, no ".." segment,
#     and ending either in a file extension or in "/" (a directory).
#     ⇒ `hub/refusal` (a branch) and `exp_gen_baselines.py:1562` (a call site) are not
#       candidates; `.claude/outputs/gen-read-cell-arm/runs/` and
#       `gen-val-split-fix/refusal_scores.json` are.
#   * ".claude/…" resolves under $CHLU_CLAUDE_ROOT. A bare "slug/…" resolves under
#     $CHLU_CLAUDE_ROOT/outputs/ when its first segment is an existing output slug or
#     has a slug's shape — that is how the charter's own `gen-val-split-fix/…` is found.
#   * ⛔ THE HUB'S OWN WRITE PATH IS NEVER A REQUIREMENT. `.claude/outputs/<slug>/**`
#     and `<slug>.md` are what the hub PRODUCES (agent profile §0.3); requiring them
#     would refuse every bundle, since they do not exist yet.
#   * ⛔ A candidate whose ⛔/⚠/⭐-delimited clause marks it SUPERSEDED / "never quote" /
#     "never ship" is an EXCLUSION, ⛔ not a requirement — and an exclusion WINS over a
#     requirement for the same path. This is how `gen-arms-a7-wiring/refusal_scores.json`
#     stays off the far side while `gen-val-split-fix/refusal_scores.json`, named one
#     clause earlier ON THE SAME LINE, is shipped.
#   * everything else in the charter is ignored, ⛔ silently and on purpose.
#
# ⭐ Escape hatches, because a derivation must never become a cage:
#     --require PATH           an extra requirement (repeatable)
#     --exclude-require PATH   drop one (repeatable)
#     --no-charter-requires    derive nothing; ⛔ then the class is open again, so say so
# ---------------------------------------------------------------------------
derive_hub_requirements() {   # $1=task file  $2=slug  $3=json-summary out ; prints TSV rows
    HR_TASK="$1" HR_SLUG="$2" HR_JSON="$3" HR_CLAUDE="$SRC_CLAUDE" \
    HR_EXTRA="$(printf '%s\n' "${REQUIRE_EXTRA[@]+"${REQUIRE_EXTRA[@]}"}")" \
    HR_DROP="$(printf '%s\n' "${REQUIRE_DROP[@]+"${REQUIRE_DROP[@]}"}")" \
    "$PY_BIN" - <<'REQPY'
import json, os, re, sys
from pathlib import Path

e = os.environ.get
task = Path(e("HR_TASK"))
slug = e("HR_SLUG")
claude = Path(e("HR_CLAUDE"))
text = task.read_text(encoding="utf-8", errors="replace")

BACKTICK = re.compile(r"`([^`\n]+)`")
CLAUSE_SPLIT = re.compile(r"[⛔⚠⭐]+")          # ⛔ ⚠ ⭐
NEGATION = re.compile(r"SUPERSEDED|never quote|never ship|do not ship|do not quote"
                      r"|not shipping|DEPRECATED|superseded", re.I)
HAS_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,6}$")
SLUG_SHAPE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")

#: ⭐ An artefact with no loader cannot be OPENED, and opening it is the test. ⛔ So an
#   unmapped suffix is a REFUSAL, not a silently-shipped byte blob.
BY_SUFFIX = {".json": "json", ".npy": "npy", ".npz": "npz", ".md": "doc", ".txt": "doc",
             ".log": "doc", ".out": "doc", ".csv": "doc", ".py": "code", ".sh": "code",
             ".eqx": "eqx_pickle", ".pkl": "eqx_pickle"}
SKIP_NAMES = {".DS_Store", ".workspace-write-probe.json"}


def candidates(raw):
    """Every backticked token in `raw` that could name an artefact, with its verdict."""
    for line in raw.splitlines():
        for clause in CLAUSE_SPLIT.split(line):
            banned = bool(NEGATION.search(clause))
            for tok in BACKTICK.findall(clause):
                # ⛔ rstrip only: `.strip(".")` ate the leading dot of every
                #    `.claude/…` token and silently turned it into a non-candidate.
                tok = tok.strip().rstrip(",;: ")
                if "/" not in tok or tok.startswith("/") or ".." in Path(tok).parts:
                    continue
                if not (tok.endswith("/") or HAS_SUFFIX.search(tok)):
                    continue                       # `hub/refusal` is a branch, not a path
                yield tok, banned


def resolve(tok):
    """(bundle-relative path, source path) or None when the token is not an artefact."""
    t = tok.rstrip("/")
    if t.startswith(".claude/"):
        return ".claude/" + t[len(".claude/"):], claude / t[len(".claude/"):]
    head = t.split("/", 1)[0]
    if (claude / "outputs" / head).is_dir() or SLUG_SHAPE.match(head):
        return ".claude/outputs/" + t, claude / "outputs" / t
    return None                                    # `scripts/csf3/stage_artefacts.sh`


def is_the_hubs_own_output(rel):
    return rel in (".claude/outputs/%s.md" % slug,) or \
        rel.startswith(".claude/outputs/%s/" % slug) or rel == ".claude/outputs/" + slug


required, excluded, ignored = {}, {}, []
for tok, banned in candidates(text):
    r = resolve(tok)
    if r is None:
        ignored.append(tok)
        continue
    rel, src = r
    if is_the_hubs_own_output(rel):
        ignored.append(tok + "  (the hub's OWN write path — it produces this)")
        continue
    (excluded if banned else required)[rel] = (src, tok)

for extra in filter(None, (e("HR_EXTRA") or "").splitlines()):
    r = resolve(extra if extra.startswith(".claude/") else extra)
    if r is None:
        sys.exit("FATAL[bundle_hub_workspace]: --require %r does not resolve under %s"
                 % (extra, claude))
    required[r[0]] = (r[1], "--require")
for drop in filter(None, (e("HR_DROP") or "").splitlines()):
    r = resolve(drop)
    if r is not None:
        excluded[r[0]] = (r[1], "--exclude-require")

# ⛔ An exclusion WINS. A superseded artefact must never reach the far side, even if
#    some other clause of the same charter also names it.
for rel in list(required):
    if rel in excluded:
        del required[rel]

missing = sorted(rel for rel, (src, _) in required.items() if not (src.is_file() or src.is_dir()))
if missing:
    print("FATAL[bundle_hub_workspace]: ⛔ %s declares %d artefact requirement(s) that do "
          "not exist — ⛔ REFUSING to build a workspace whose charter promises inputs the "
          "bundle cannot carry:" % (task, len(missing)), file=sys.stderr)
    for rel in missing:
        print("  ⛔ MISSING  %-60s (expected at %s)" % (rel, required[rel][0]), file=sys.stderr)
    print("  ⭐ Fix the charter, stage the artefact, or pass --exclude-require <path>.",
          file=sys.stderr)
    sys.exit(3)

rows, resolved, nofit = [], [], []
for rel in sorted(required):
    src, tok = required[rel]
    files = [src] if src.is_file() else sorted(q for q in src.rglob("*") if q.is_file())
    for f in files:
        if f.name in SKIP_NAMES or f.name.startswith("._"):
            continue
        how = BY_SUFFIX.get(f.suffix)
        if how is None:
            nofit.append(str(f))
            continue
        sub = ".claude/" + str(f.relative_to(claude)).replace(os.sep, "/")
        rows.append((sub, str(f), how, "hub-requirement",
                     "declared by %s as `%s`; ⛔ without it that charter's own premise "
                     "('scored off artefacts that already exist') is false on the far side"
                     % (task.name, tok)))
    resolved.append({"token": tok, "path": rel, "is_dir": src.is_dir(), "n_files": len(files)})

if nofit:
    print("FATAL[bundle_hub_workspace]: ⛔ no loader for %d required file(s), e.g. %s — "
          "an artefact that cannot be OPENED cannot be verified, and opening it is the "
          "test. ⭐ Add the suffix to BY_SUFFIX and a loader to verify_hub_workspace.sh."
          % (len(nofit), ", ".join(nofit[:3])), file=sys.stderr)
    sys.exit(3)

for r in rows:
    print("\t".join(r))

if e("HR_JSON"):
    Path(e("HR_JSON")).write_text(json.dumps({
        "hub": slug, "task_file": str(task), "n_files": len(rows),
        "required": resolved,
        "excluded": [{"path": rel, "token": tok, "why": "⛔ marked superseded/never-ship "
                      "in its own clause of the charter"} for rel, (_, tok) in sorted(excluded.items())],
        "ignored_tokens": sorted(set(ignored)),
    }, indent=2) + "\n", encoding="utf-8")
REQPY
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
SRC_ROOT="${CHLU_ROOT:-$CHLU_ROOT_DEFAULT}"
PIN="ae907466ba2b5fbec124e8995cf4d7c32a1ab30e"
OUT=""
DRY_RUN=0
DO_TAR=0
FORCE=0
NO_ARTEFACTS=0
GIT_INIT=1
HUBS=()
HUB_TASKS=()
DATASETS=""
PRETEXTS=""
SEEDS=""
MODE="build"
RETURN_SLUG=""
RETURN_OUT=""
RETURN_TAR=""
DEST=""
REQUIRE_EXTRA=()
REQUIRE_DROP=()
CHARTER_REQUIRES=1
SUPP_OUT=""
SUPP_TAR=""
PY_BIN="${BUNDLE_PYTHON:-python3}"

usage() { sed -n '2,78p' "${BASH_SOURCE[0]}"; exit "${1:-1}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --hub)            HUBS+=("${2:?--hub needs a slug}"); shift ;;
        --hub-task)       HUB_TASKS+=("${2:?--hub-task needs a path}"); shift ;;
        --out)            OUT="${2:?--out needs a dir}"; shift ;;
        --pin)            PIN="${2:?--pin needs a sha}"; shift ;;
        --src)            SRC_ROOT="${2:?--src needs a dir}"; shift ;;
        --datasets)       DATASETS="${2:?}"; shift ;;
        --pretexts)       PRETEXTS="${2:?}"; shift ;;
        --seeds)          SEEDS="${2:?}"; shift ;;
        --tar)            DO_TAR=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        --force)          FORCE=1 ;;
        --no-artefacts)   NO_ARTEFACTS=1 ;;
        --no-git)         GIT_INIT=0 ;;
        --return-pack)    MODE="return-pack"; RETURN_SLUG="${2:?--return-pack needs a hub slug}"; shift ;;
        --return-out)     RETURN_OUT="${2:?--return-out needs a path}"; shift ;;
        --return-ingest)  MODE="return-ingest"; RETURN_TAR="${2:?--return-ingest needs a tar}"; shift ;;
        --dest)           DEST="${2:?--dest needs a dir}"; shift ;;
        --require)        REQUIRE_EXTRA+=("${2:?--require needs a path}"); shift ;;
        --exclude-require) REQUIRE_DROP+=("${2:?--exclude-require needs a path}"); shift ;;
        --no-charter-requires) CHARTER_REQUIRES=0 ;;
        --print-requirements)  MODE="print-requirements" ;;
        --supplement)     MODE="supplement" ;;
        --supplement-out) SUPP_OUT="${2:?--supplement-out needs a path}"; shift ;;
        --supplement-apply) MODE="supplement-apply"; SUPP_TAR="${2:?--supplement-apply needs a tar}"; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
    shift
done

SRC_ROOT="$(cd "$SRC_ROOT" 2>/dev/null && pwd)" || die "no source root at '$SRC_ROOT'"
command -v "$PY_BIN" >/dev/null 2>&1 || die "no python at '$PY_BIN' (stdlib json only)"

# ---------------------------------------------------------------------------
# Shared: resolve one hub's task file (build, --print-requirements and --supplement
# all need it). ⛔ `die` inside $( ) exits only the subshell, so callers check the rc.
# ---------------------------------------------------------------------------
resolve_hub_task() {   # $1=slug $2=index-in-HUBS
    local hub="$1" idx="$2" override="" src
    case "$hub" in */*|..*|"") die "hub slug '$hub' must be a bare name" ;; esac
    if [ "${#HUB_TASKS[@]}" -gt "$idx" ]; then override="${HUB_TASKS[$idx]}"; fi
    if [ -n "$override" ]; then
        [ -f "$override" ] || die "--hub-task '$override' does not exist"
        printf '%s\n' "$override"; return 0
    fi
    src="$SRC_CLAUDE/tasks/$hub.md"
    [ -f "$src" ] || die "no task file at $src.
  ⛔ A hub with no charter of its own is not a hub — it is ten unsupervised hours with no question.
  ⭐ Write .claude/tasks/$hub.md first, or pass --hub-task <path> to point at it."
    printf '%s\n' "$src"
}

# Collect every hub's requirement rows into $1, and their JSON summaries into $2/.
collect_hub_requirements() {   # $1=rows out  $2=json dir
    local out="$1" jdir="$2" i=0 hub taskf
    : > "$out"
    mkdir -p "$jdir"
    [ "$CHARTER_REQUIRES" = "1" ] || {
        say "requirements: ⛔ SKIPPED (--no-charter-requires) — the class of omission this"
        say "              closes is open again for this build. ⚠ Say so in the report."
        return 0
    }
    for hub in "${HUBS[@]}"; do
        taskf="$(resolve_hub_task "$hub" "$i")" || exit 2
        [ -n "$taskf" ] || exit 2
        derive_hub_requirements "$taskf" "$hub" "$jdir/$hub.json" >> "$out" \
            || die "⛔ $hub's charter declares an artefact requirement this machine cannot meet (above). Refusing to build a workspace that would strand it."
        say "requirements: $hub -> $(wc -l < "$out" | tr -d ' ') file(s) so far (from $(basename "$taskf"))"
        i=$((i + 1))
    done
}

# ===========================================================================
# ⭐ MODE: print-requirements — the verdict for a charter, ⛔ building nothing.
# ===========================================================================
if [ "$MODE" = "print-requirements" ]; then
    chlu_derive_roots "$SRC_ROOT"
    SRC_CLAUDE="$CHLU_CLAUDE_ROOT"
    [ -d "$SRC_CLAUDE" ] || die "no .claude at $SRC_CLAUDE — pass --src <primary checkout>"
    [ "${#HUBS[@]}" -gt 0 ] || die "--print-requirements needs at least one --hub SLUG"
    PR_RC=0
    i=0
    for hub in "${HUBS[@]}"; do
        taskf="$(resolve_hub_task "$hub" "$i")" || exit 2
        J="$(mktemp -t chlu_req.XXXXXX)"
        say "=== $hub  ($taskf) ==="
        if derive_hub_requirements "$taskf" "$hub" "$J" >/dev/null; then
            PR_J="$J" "$PY_BIN" - <<'PRPY'
import json, os
d = json.load(open(os.environ["PR_J"], encoding="utf-8"))
print("  files to ship : %d" % d["n_files"])
for r in d["required"]:
    print("  ⭐ REQUIRED  %-58s %s" % (r["path"], "(%d files)" % r["n_files"] if r["is_dir"] else ""))
for x in d["excluded"]:
    print("  ⛔ EXCLUDED  %-58s %s" % (x["path"], x["why"]))
for t in d["ignored_tokens"]:
    print("     ignored   %s" % t)
PRPY
        else
            PR_RC=1
        fi
        rm -f "$J"
        i=$((i + 1))
    done
    exit "$PR_RC"
fi

# ===========================================================================
# ⭐ MODE: supplement — a TOP-UP tarball carrying just a hub's declared artefacts.
# ===========================================================================
# ⚠ A workspace may already be built, verified and HAVE A HUB RUNNING IN IT. Rebuilding
# 2.1 GB to add 210 MB would mean rewriting every file underneath that hub. ⛔ So the
# supplement carries ONLY the §2b requirement rows, and `--supplement-apply` never
# replaces a byte that is already there.
if [ "$MODE" = "supplement" ]; then
    chlu_derive_roots "$SRC_ROOT"
    SRC_CLAUDE="$CHLU_CLAUDE_ROOT"
    [ -d "$SRC_CLAUDE" ] || die "no .claude at $SRC_CLAUDE — pass --src <primary checkout>"
    [ "${#HUBS[@]}" -gt 0 ] || die "--supplement needs --hub SLUG"
    SUPP_OUT="${SUPP_OUT:-$PWD/${HUBS[0]}.supplement.tar.gz}"
    case "$SUPP_OUT" in /*) : ;; *) SUPP_OUT="$PWD/$SUPP_OUT" ;; esac
    if [ -e "$SUPP_OUT" ] && [ "$FORCE" != "1" ]; then
        die "$SUPP_OUT exists — ⛔ refusing to overwrite (use --force)"
    fi

    SUP_ROWS="$(mktemp -t chlu_supp_rows.XXXXXX)"
    SUP_JDIR="$(mktemp -d -t chlu_supp_json.XXXXXX)"
    SUP_STAGE="$(mktemp -d -t chlu_supp_stage.XXXXXX)"
    SUP_LIST="$(mktemp -t chlu_supp_list.XXXXXX)"
    trap 'rm -rf "$SUP_ROWS" "$SUP_JDIR" "$SUP_STAGE" "$SUP_LIST"' EXIT
    collect_hub_requirements "$SUP_ROWS" "$SUP_JDIR"
    [ -s "$SUP_ROWS" ] || die "${HUBS[*]} declares no artefact requirement — ⛔ nothing to supplement"

    SUP_MF="$SUP_STAGE/SUPPLEMENT_MANIFEST.json"
    SP_ROWS="$SUP_ROWS" SP_CLAUDE="$SRC_CLAUDE" SP_HUBS="${HUBS[*]}" SP_MF="$SUP_MF" \
    SP_LIST="$SUP_LIST" SP_HOST="$(hostname)" SP_ROOT="$SRC_ROOT" \
    "$PY_BIN" - <<'SPPY' || die "could not build the supplement manifest"
import datetime as _dt, hashlib, json, os
from pathlib import Path
e = os.environ.get
claude = Path(e("SP_CLAUDE"))
files, seen, total = [], set(), 0
with open(e("SP_ROWS"), encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        rel, src, how, kind, loader = line.split("\t", 4)
        # members are stored RELATIVE TO `.claude/`, so the tar needs no staging copy
        sub = rel[len(".claude/"):]
        if sub in seen:
            continue
        seen.add(sub)
        if sub == "SUPPLEMENT_MANIFEST.json":
            raise SystemExit("FATAL: an artefact named SUPPLEMENT_MANIFEST.json would "
                             "collide with the supplement's own manifest")
        p = Path(src)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        b = p.stat().st_size
        total += b
        files.append({"path": sub, "bytes": b, "sha256": h.hexdigest(),
                      "loader": loader, "open_with": how, "kind": kind, "required": True})
files.sort(key=lambda d: d["path"])
Path(e("SP_LIST")).write_text("".join(f["path"] + "\n" for f in files), encoding="utf-8")
Path(e("SP_MF")).write_text(json.dumps({
    "schema": "chlu-hub-workspace-supplement/1",
    "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "built_on_host": e("SP_HOST"), "built_from_root": e("SP_ROOT"),
    "hubs": e("SP_HUBS", "").split(),
    # ⭐ Every member path is relative to the workspace's `.claude/`. ⛔ Nothing in a
    #    supplement may ever address anything outside it.
    "member_root": ".claude",
    "n_files": len(files), "total_bytes": total, "files": files,
    "why": ("the artefacts these hubs' own charters declare (§2b). ⭐ Applied with "
            "--supplement-apply, which tops a workspace up and ⛔ never replaces a byte."),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("[bundle] supplement: %d file(s), %.3f GiB" % (len(files), total / 2**30))
SPPY

    mkdir -p "$(dirname "$SUPP_OUT")"
    SUPP_TARBALL="${SUPP_OUT%.gz}"
    chlu_tar -cf "$SUPP_TARBALL" -C "$SRC_CLAUDE" -T "$SUP_LIST"
    chlu_tar -rf "$SUPP_TARBALL" -C "$SUP_STAGE" "SUPPLEMENT_MANIFEST.json"
    assert_no_apple_double "$SUPP_TARBALL"
    gzip -c "$SUPP_TARBALL" > "$SUPP_OUT"
    rm -f "$SUPP_TARBALL"
    say "supplement -> $SUPP_OUT  ($(bytes_of "$SUPP_OUT") bytes)"
    say "sha256($(basename "$SUPP_OUT")) = $(sha256_of "$SUPP_OUT")"
    say "⭐ apply it: scripts/bundle_hub_workspace.sh --supplement-apply $SUPP_OUT --dest <workspace>"
    exit 0
fi

# ===========================================================================
# ⭐ MODE: supplement-apply — TOP UP a workspace. ⛔ IT NEVER REPLACES A BYTE.
# ===========================================================================
if [ "$MODE" = "supplement-apply" ]; then
    [ -f "$SUPP_TAR" ] || die "no supplement tar at $SUPP_TAR"
    [ -n "$DEST" ] || die "--supplement-apply needs --dest <the workspace root>"
    [ -d "$DEST" ] || die "--dest '$DEST' is not a directory — ⛔ a supplement tops up an EXISTING workspace"
    DEST="$(cd "$DEST" && pwd)"
    SA_TMP="$(mktemp -d -t chlu_supp_apply.XXXXXX)"
    trap 'rm -rf "$SA_TMP"' EXIT
    chlu_tar -xzf "$SUPP_TAR" -C "$SA_TMP"
    [ -f "$SA_TMP/SUPPLEMENT_MANIFEST.json" ] || die "$SUPP_TAR carries no SUPPLEMENT_MANIFEST.json — ⛔ an undeclared tar is not applied"

    SA_TMP="$SA_TMP" SA_DEST="$DEST" "$PY_BIN" - <<'SAPY'
import datetime as _dt, hashlib, json, os, shutil, sys
from pathlib import Path

tmp = Path(os.environ["SA_TMP"]).resolve()
dest = Path(os.environ["SA_DEST"]).resolve()
mf = json.loads((tmp / "SUPPLEMENT_MANIFEST.json").read_text(encoding="utf-8"))
if mf.get("schema") != "chlu-hub-workspace-supplement/1":
    sys.exit("FATAL[bundle_hub_workspace]: not a chlu-hub-workspace-supplement/1")
root = dest / mf.get("member_root", ".claude")

def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

# --- the same refusals --return-ingest makes: nothing may address outside the set ---
declared = {}
bad = []
for rec in mf["files"]:
    rel = rec["path"]
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        bad.append("%s: traversing path" % rel); continue
    src = tmp / rel
    if not src.is_file():
        bad.append("%s: declared but not in the tar" % rel); continue
    if src.is_symlink():
        bad.append("%s: is a symlink" % rel); continue
    if sha256_of(src) != rec["sha256"]:
        bad.append("%s: sha256 mismatch" % rel); continue
    declared[rel] = rec
for p in sorted(tmp.rglob("*")):
    if not p.is_file():
        continue
    rel = str(p.relative_to(tmp))
    if rel != "SUPPLEMENT_MANIFEST.json" and rel not in declared:
        bad.append("%s: present in the tar but NOT declared in the manifest" % rel)
if bad:
    print("FATAL[bundle_hub_workspace]: ⛔ refusing the supplement:", file=sys.stderr)
    for b in bad:
        print("  ⛔ " + b, file=sys.stderr)
    sys.exit(2)

stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
n_new = n_same = n_kept = 0
for rel, rec in sorted(declared.items()):
    target = root / rel
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp / rel, target)
        n_new += 1
        continue
    if target.is_file() and sha256_of(target) == rec["sha256"]:
        n_same += 1                     # already there, byte-identical: ⭐ a no-op
        continue
    # ⛔⛔ A HUB MAY BE RUNNING IN HERE. We do not replace it; we put ours beside it.
    # ⚠ …but only once. A file that legitimately differs forever (a hub edited it) must
    #    not gain one stamped copy per apply — that is litter accumulating inside a
    #    running hub's own workspace, which is its own way of disturbing it.
    already = [q for q in target.parent.glob(target.name + ".supplement-*")
               if q.is_file() and sha256_of(q) == rec["sha256"]]
    if already:
        n_same += 1
        continue
    beside = target.with_name(target.name + ".supplement-" + stamp)
    shutil.copy2(tmp / rel, beside)
    n_kept += 1
    print("[bundle] ⛔ NOT overwriting %s (differs) -> wrote %s" % (target, beside.name))

# the supplement's own manifest lands beside the bundle's, ⛔ never on top of it
mf_target = dest / "SUPPLEMENT_MANIFEST.json"
if mf_target.exists():
    mf_target = dest / ("SUPPLEMENT_MANIFEST-%s.json" % stamp)
shutil.copy2(tmp / "SUPPLEMENT_MANIFEST.json", mf_target)

print("[bundle] supplement applied to %s" % root)
print("[bundle]   %d new · %d already identical (no-op) · %d differing, KEPT and written beside"
      % (n_new, n_same, n_kept))
print("[bundle]   manifest -> %s" % mf_target)
print("[bundle] ⭐ re-run ./scripts/verify_hub_workspace.sh — it picks the supplement up.")
SAPY
    exit $?
fi

# ===========================================================================
# ⭐ MODE: return-pack — run this ON THE FAR SIDE.
# ===========================================================================
# ⚠ A result that is not synced back does not exist. This packs exactly what a hub
# is allowed to have produced — `.claude/outputs/<slug>/**` and `.claude/outputs/<slug>.md`
# — with a sha per file, and ⛔ nothing else. The ingest side refuses anything outside
# that prefix, so the tar cannot reach any other path on this machine.
if [ "$MODE" = "return-pack" ]; then
    chlu_derive_roots "$SRC_ROOT"
    SLUG="$RETURN_SLUG"
    case "$SLUG" in */*|..*|"") die "hub slug '$SLUG' must be a bare name" ;; esac
    D="$CHLU_HUB_OUTPUTS/$SLUG"
    R="$CHLU_HUB_OUTPUTS/$SLUG.md"
    [ -d "$D" ] || [ -f "$R" ] || die "nothing to return: neither $D/ nor $R exists"
    RETURN_OUT="${RETURN_OUT:-$CHLU_HUB_OUTPUTS/$SLUG.return.tar.gz}"
    if [ -e "$RETURN_OUT" ] && [ "$FORCE" != "1" ]; then
        die "$RETURN_OUT exists — ⛔ refusing to overwrite (use --force)"
    fi

    LIST="$(mktemp -t chlu_return_list.XXXXXX)"
    MF="$(mktemp -t chlu_return_mf.XXXXXX)"
    trap 'rm -f "$LIST" "$MF"' EXIT
    : > "$LIST"
    if [ -f "$R" ]; then echo "$SLUG.md" >> "$LIST"; fi
    if [ -d "$D" ]; then
        # ⛔ The verifier's own write probe is not a hub artefact and does not go
        #    home. Excluded BY NAME — ⛔ never by a blanket dotfile filter, which
        #    would silently drop a hub's own `.PREREG` or `.checkpoint`.
        ( cd "$CHLU_HUB_OUTPUTS" && find "$SLUG" -type f \
              ! -name ".workspace-write-probe.json" \
              ! -name "._*" ! -name ".DS_Store" -print ) >> "$LIST"
    fi
    [ -s "$LIST" ] || die "nothing to return under $D"

    : > "$MF"
    TOTAL=0
    while IFS= read -r rel; do
        f="$CHLU_HUB_OUTPUTS/$rel"
        b="$(bytes_of "$f")"; h="$(sha256_of "$f")"
        TOTAL=$((TOTAL + b))
        printf '%s\t%s\t%s\n' "$rel" "$b" "$h" >> "$MF"
    done < "$LIST"
    N="$(wc -l < "$MF" | tr -d ' ')"

    RM_JSON="$(mktemp -t chlu_return_json.XXXXXX)"
    RP_SLUG="$SLUG" RP_TOTAL="$TOTAL" RP_HOST="$(hostname)" RP_ROOT="$CHLU_ROOT" RP_OUT="$RM_JSON" \
        "$PY_BIN" - "$MF" <<'PY'
import datetime as _dt, json, os, sys
e = os.environ.get
files = []
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line:
        continue
    rel, size, sha = line.split("\t", 2)
    files.append({"path": rel, "bytes": int(size), "sha256": sha})
files.sort(key=lambda d: d["path"])
json.dump({
    "schema": "chlu-hub-return/1",
    "hub_slug": e("RP_SLUG"),
    "packed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "source_host": e("RP_HOST"),
    "source_chlu_root": e("RP_ROOT"),
    "n_files": len(files),
    "total_bytes": int(e("RP_TOTAL", "0")),
    # ⛔ Every path is relative to `.claude/outputs/`; the ingest side asserts the
    # `<slug>` prefix and refuses absolute paths, `..` and symlinks.
    "paths_relative_to": ".claude/outputs/",
    "files": files,
}, open(e("RP_OUT"), "w"), indent=2, sort_keys=True)
PY
    if [ "$DRY_RUN" = "1" ]; then
        say "DRY RUN — would pack $N file(s), $TOTAL bytes -> $RETURN_OUT"
        rm -f "$RM_JSON"; exit 0
    fi
    STAGE="$(mktemp -d -t chlu_return_stage.XXXXXX)"
    cp "$RM_JSON" "$STAGE/RETURN_MANIFEST.json"
    rm -f "$RM_JSON"
    mkdir -p "$(dirname "$RETURN_OUT")"
    # ⚠ Two roots, one archive. bsdtar (macOS) stops parsing options at the first
    # file operand, so `-C a f -C b -T list` silently treats `-T` as a filename.
    # Build in two passes instead — every invocation is options-only.
    chlu_tar -cf  "$STAGE/pack.tar" -C "$CHLU_HUB_OUTPUTS" -T "$LIST"
    chlu_tar -rf  "$STAGE/pack.tar" -C "$STAGE" "RETURN_MANIFEST.json"
    assert_no_apple_double "$STAGE/pack.tar"
    gzip -c  "$STAGE/pack.tar" > "$RETURN_OUT"
    say "packed $N file(s), $TOTAL bytes -> $RETURN_OUT"
    say "sha256($(basename "$RETURN_OUT")) = $(sha256_of "$RETURN_OUT")"
    say "⭐ carry it back, then on the primary machine:"
    say "   scripts/bundle_hub_workspace.sh --return-ingest $RETURN_OUT"
    exit 0
fi

# ===========================================================================
# ⭐ MODE: return-ingest — run this ON THE PRIMARY MACHINE.
# ===========================================================================
# ⛔⛔ THE SAFETY RULE: this NEVER writes over anything. If the destination exists,
# the whole set is diverted to `<slug>.returned-<utc>/` and the divert is printed.
# Every member path is checked before extraction: it must be relative, must not
# contain `..`, must not be a symlink or a device, and must begin with `<slug>`.
if [ "$MODE" = "return-ingest" ]; then
    chlu_derive_roots "$SRC_ROOT"
    [ -f "$RETURN_TAR" ] || die "no return tar at $RETURN_TAR"
    DEST="${DEST:-$CHLU_HUB_OUTPUTS}"
    mkdir -p "$DEST"

    TMP="$(mktemp -d -t chlu_ingest.XXXXXX)"
    trap 'rm -rf "$TMP"' EXIT
    chlu_tar -xzf "$RETURN_TAR" -C "$TMP"
    [ -f "$TMP/RETURN_MANIFEST.json" ] || die "$RETURN_TAR has no RETURN_MANIFEST.json — ⛔ refusing an unmanifested return"

    SLUG="$("$PY_BIN" -c 'import json,sys;print(json.load(open(sys.argv[1]))["hub_slug"])' "$TMP/RETURN_MANIFEST.json")"
    case "$SLUG" in */*|..*|"") die "return manifest hub_slug '$SLUG' is not a bare name" ;; esac

    # --- per-file: path safety + sha match, BEFORE anything is placed ---------
    IG_TMP="$TMP" IG_SLUG="$SLUG" "$PY_BIN" - <<'PY' || die "the return set failed its safety/checksum audit — ⛔ nothing was placed"
import hashlib, json, os, sys
from pathlib import Path
tmp = Path(os.environ["IG_TMP"]); slug = os.environ["IG_SLUG"]
mf = json.loads((tmp / "RETURN_MANIFEST.json").read_text())
if mf.get("schema") != "chlu-hub-return/1":
    sys.exit("FATAL: not a chlu-hub-return/1 manifest")
bad = []
declared = {f["path"] for f in mf["files"]}
for f in mf["files"]:
    rel = f["path"]
    if rel.startswith("/") or ".." in Path(rel).parts:
        bad.append("%s: absolute or traversing path" % rel); continue
    if not (rel == slug + ".md" or rel == slug or rel.startswith(slug + "/")):
        bad.append("%s: outside the hub's own prefix %r" % (rel, slug)); continue
    p = tmp / rel
    if p.is_symlink():
        bad.append("%s: is a symlink" % rel); continue
    if not p.is_file():
        bad.append("%s: declared but not in the tar" % rel); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != f["sha256"]:
        bad.append("%s: sha256 %s != %s declared" % (rel, h[:16], f["sha256"][:16]))
# ⛔ A member in the tar that the manifest does not declare is a smuggled file.
for p in tmp.rglob("*"):
    if p.is_file():
        rel = str(p.relative_to(tmp))
        if rel != "RETURN_MANIFEST.json" and rel not in declared:
            bad.append("%s: present in the tar but NOT declared in the manifest" % rel)
if bad:
    print("⛔ RETURN AUDIT FAILED:", file=sys.stderr)
    for b in bad:
        print("   " + b, file=sys.stderr)
    sys.exit(1)
print("[ingest] audit ok: %d file(s), %d bytes, hub=%s" % (mf["n_files"], mf["total_bytes"], slug))
PY

    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    TARGET_DIR="$DEST/$SLUG"
    TARGET_MD="$DEST/$SLUG.md"
    DIVERTED=0
    if [ -e "$TARGET_DIR" ] || [ -e "$TARGET_MD" ]; then
        TARGET_DIR="$DEST/$SLUG.returned-$STAMP"
        TARGET_MD="$DEST/$SLUG.returned-$STAMP.md"
        DIVERTED=1
        say "⚠ $DEST/$SLUG already exists — ⛔ NOT overwriting."
        say "   diverting the whole return to $TARGET_DIR (and $(basename "$TARGET_MD"))"
    fi
    if [ "$DRY_RUN" = "1" ]; then
        say "DRY RUN — would place into $TARGET_DIR (diverted=$DIVERTED); nothing written"
        exit 0
    fi
    mkdir -p "$TARGET_DIR"
    if [ -d "$TMP/$SLUG" ]; then
        ( cd "$TMP/$SLUG" && chlu_tar -cf - . ) | ( cd "$TARGET_DIR" && chlu_tar -xf - )
    fi
    if [ -f "$TMP/$SLUG.md" ]; then cp "$TMP/$SLUG.md" "$TARGET_MD"; fi
    cp "$TMP/RETURN_MANIFEST.json" "$TARGET_DIR/RETURN_MANIFEST.json"
    say "ingested -> $TARGET_DIR"
    if [ "$DIVERTED" = "1" ]; then
        say "⭐ DIVERTED: the pre-existing $DEST/$SLUG was left untouched. Reconcile by hand."
    fi
    exit 0
fi

# ===========================================================================
# MODE: build
# ===========================================================================
[ -n "$OUT" ] || die "--out DIR is required (the bundle root; it becomes \$CHLU_ROOT on the far side)"
# ⭐ Validate the destination FIRST — ⛔ before deriving anything, checksumming
#    anything, or creating a directory. `--out` inside the repo would bundle the
#    repo into itself.
case "$OUT" in
    /*) OUT_ABS="$OUT" ;;
    *)  OUT_ABS="$PWD/$OUT" ;;
esac
if [ "${OUT_ABS%/}" = "$SRC_ROOT" ]; then
    die "⛔ --out is the source root. That would write the bundle over the repo."
fi
case "$OUT_ABS" in
    "$SRC_ROOT"/*) die "⛔ --out is inside the source root ($SRC_ROOT). Bundle somewhere else." ;;
esac
chlu_derive_roots "$SRC_ROOT"
SRC_CLAUDE="$CHLU_CLAUDE_ROOT"
[ -d "$SRC_CLAUDE" ] || die "no .claude at $SRC_CLAUDE — ⚠ a worktree has none of its own; pass --src <primary checkout>"
command -v git >/dev/null 2>&1 || die "no git"
git -C "$SRC_ROOT" cat-file -e "${PIN}^{commit}" 2>/dev/null || die "pin $PIN is not a commit in $SRC_ROOT"

[ "${#HUBS[@]}" -gt 0 ] || die "no --hub given. ⛔ A bundle without a hub task file is not a hub workspace."

TAB="$(printf '\t')"
ROWS="$(mktemp -t chlu_bundle_rows.XXXXXX)"
STAGE_MF="$(mktemp -t chlu_bundle_stage.XXXXXX).json"
trap 'rm -rf "$ROWS" "$ROWS.hashed" "$ROWS.seen" "$ROWS.reg" "$ROWS.code" "$ROWS.req" "$ROWS.reqjson" "$STAGE_MF"' EXIT
: > "$ROWS"

add_row() {  # $1=relpath(in bundle) $2=abs source $3=open_with $4=kind $5=loader
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >> "$ROWS"
}

# ---------------------------------------------------------------------------
# 1. THE CODE — the tracked tree AT THE PIN. ⛔ Not the working tree: a hub that
#    branches from the pin must see the pin, and the primary checkout is usually
#    sitting on some other agent's branch.
# ---------------------------------------------------------------------------
say "code     : git tree at $PIN"
CODE_LIST="$ROWS.code"
git -C "$SRC_ROOT" ls-tree -r --name-only "$PIN" > "$CODE_LIST"
N_CODE="$(wc -l < "$CODE_LIST" | tr -d ' ')"

# ---------------------------------------------------------------------------
# 1b. ⭐ THE TOOLING OVERLAY — declared, never silent.
#
# ⚠ The pin predates this script. A bundle built from `git archive <pin>` therefore
# contains NO `bundle_hub_workspace.sh`, NO `verify_hub_workspace.sh` and NO
# `tests/test_hub_workspace.py` — so the far side would have nothing to verify or
# return WITH. These three files are overlaid from the tree this script lives in.
#
# ⭐ They are pure plumbing: none of them is imported by `chlu`, none touches the
#    scoring harness the charter freezes (classifier, splits, rival rows, byte
#    ledger, bypass assertions). ⛔ Nothing else is ever overlaid, and every
#    overlaid file is listed in BUNDLE_MANIFEST.json under `kind: "tooling"` with
#    the commit it came from — a hub can see exactly where its workspace deviates
#    from the pin.
# ---------------------------------------------------------------------------
TOOL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOL_COMMIT="$(git -C "$TOOL_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
TOOLING=(
    "scripts/bundle_hub_workspace.sh"
    "scripts/verify_hub_workspace.sh"
    "tests/test_hub_workspace.py"
)
for rel in "${TOOLING[@]}"; do
    [ -f "$TOOL_ROOT/$rel" ] || die "tooling file $TOOL_ROOT/$rel is missing — ⛔ the far side would have no way to verify or return anything"
    add_row "$rel" "$TOOL_ROOT/$rel" code tooling \
        "the workspace plumbing, overlaid on the pin from $TOOL_COMMIT; ⛔ not imported by chlu and ⛔ not part of the frozen scoring harness"
done

# ---------------------------------------------------------------------------
# 2. THE DOCS A HUB ACTUALLY READS — and ⛔ NOTHING ELSE from the archive.
#    The agent profile is explicit: "⛔ Read nothing else from the campaign archive —
#    three campaigns of ledgers exist and every agent that booted on them optimised
#    for the paperwork instead of the result."  So the bundle ships exactly the four
#    files the profile names, plus each hub's own task file.
#    ⛔ No `.claude/scratch/`, ⛔ no foreign slugs' outputs, ⛔ no campaign ledgers.
# ---------------------------------------------------------------------------
DOCS=(
    "AGENT_PROTOCOL.md|the spoke protocol (agent profile line 1)"
    "autonomous-hub-charter.md|the charter: the pin, pre-registration, the philosophy, the floor"
    "agents/autonomous-hub.md|the hub's own agent profile (§0 sandbox, §0.3 allow-list)"
    "handover_context.md|read for §7 Known Issues only (profile line 2)"
)
for spec in "${DOCS[@]}"; do
    rel="${spec%%|*}"; why="${spec#*|}"
    [ -f "$SRC_CLAUDE/$rel" ] || die "missing hub-read doc: $SRC_CLAUDE/$rel"
    add_row ".claude/$rel" "$SRC_CLAUDE/$rel" doc doc "$why"
done

# --- each hub's own task file ---------------------------------------------
i=0
for hub in "${HUBS[@]}"; do
    src="$(resolve_hub_task "$hub" "$i")" || exit 2
    [ -n "$src" ] || exit 2
    add_row ".claude/tasks/$hub.md" "$src" doc doc "the hub's own charter (agent profile line 3)"
    i=$((i + 1))
done

# ---------------------------------------------------------------------------
# 3. ⭐⭐ THE FROZEN ARTEFACT SET — DERIVED FROM THE LOADERS.
#    ⛔ Not a directory listing, and ⛔ not re-implemented here: this shells out to
#    `scripts/csf3/stage_artefacts.sh --manifest-only`, whose table names, per row,
#    the production call site that opens the file. Reusing it is what keeps the two
#    platforms (CSF3 and the second machine) comparable — same rows, same shas.
# ---------------------------------------------------------------------------
if [ "$NO_ARTEFACTS" = "1" ]; then
    say "artefacts: ⛔ SKIPPED (--no-artefacts). This bundle CANNOT run a hub."
else
    STAGE_SH="$SCRIPT_DIR/csf3/stage_artefacts.sh"
    [ -f "$STAGE_SH" ] || die "no $STAGE_SH — the artefact set must be derived from it, ⛔ never re-listed here"
    say "artefacts: deriving via $(basename "$STAGE_SH") --manifest-only"
    STAGE_ARGS=(--manifest-only --manifest "$STAGE_MF")
    if [ -n "$DATASETS" ]; then STAGE_ARGS+=(--datasets "$DATASETS"); fi
    if [ -n "$PRETEXTS" ]; then STAGE_ARGS+=(--pretexts "$PRETEXTS"); fi
    if [ -n "$SEEDS" ];    then STAGE_ARGS+=(--seeds "$SEEDS"); fi
    CLAUDE_ROOT="$SRC_CLAUDE" STAGE_PYTHON="$PY_BIN" bash "$STAGE_SH" "${STAGE_ARGS[@]}" >&2 \
        || die "stage_artefacts.sh refused — its refusal IS the answer (missing required artefact, or wrong root). Fix that first."

    # Every row of the stage manifest, verbatim: path, open_with, loader.
    # ⚠ The absolute source is emitted here, not reconstructed later: the artefact
    # root is $CHLU_CLAUDE_ROOT, which need NOT be `$CHLU_ROOT/.claude` (a worktree
    # has no `.claude/`, so the two are routinely different directories).
    "$PY_BIN" - "$STAGE_MF" "$SRC_CLAUDE" <<'PY' >> "$ROWS"
import json, os, sys
m = json.load(open(sys.argv[1]))
claude = sys.argv[2]
assert m["schema"] == "chlu-gen-stage-manifest/1", m["schema"]
for f in m["files"]:
    print("\t".join([".claude/" + f["path"], os.path.join(claude, f["path"]),
                     f["open_with"], "artefact", f["loader"]]))
PY

    # --- 3b. ⭐ THE HUB ADDENDUM — also loader-derived, also not a listing. ---
    # The stage table is scoped to the FIVE CSF3 launcher stages. A hub aiming at a
    # REFUSAL win runs `exp_gen_baselines`, which opens the query-set REGISTRY and
    # every control set the registry names (`exp_gen_baselines.py:1547` ->
    # `chlu.core.gen_request.load_registered_query_sets`). Those rows are not in the
    # launcher's table, and a hub that discovers this on the far side loses a day.
    # ⭐ The registry is READ to name its own files — the same discipline the stage
    #    script uses for `feature_cache.json`'s `rows_file`/`features_file`.
    DS_LIST="$("$PY_BIN" -c 'import json,sys;print(" ".join(json.load(open(sys.argv[1]))["axes"]["datasets"]))' "$STAGE_MF")"
    for ds in $DS_LIST; do
        REG="$SRC_CLAUDE/data/gen/$ds/query_sets.json"
        if [ ! -f "$REG" ]; then
            say "  ⚠ no query-set registry at data/gen/$ds/query_sets.json — the REFUSAL path"
            say "    (exp_gen_baselines.score_registered_controls) will refuse on the far side."
            continue
        fi
        add_row ".claude/data/gen/$ds/query_sets.json" "$REG" query_registry artefact \
            "chlu.data.gen_splits.load_registered_query_sets via exp_gen_baselines.py:1547 (gen_baselines.query_set_registry_file); ⛔ absent => require_registered_controls raises, or the CONTAMINATED column runs alone"
        # the registry names its own members; ⛔ we do not guess the filenames
        "$PY_BIN" - "$REG" "$ds" "$SRC_CLAUDE" <<'PY' > "$ROWS.reg"
import json, os, sys
reg = json.load(open(sys.argv[1])); ds = sys.argv[2]; claude = sys.argv[3]
def src(fn):
    return os.path.join(claude, "data", "gen", ds, fn)
for name, spec in sorted(reg["sets"].items()):
    if spec.get("file"):
        print("\t".join([
            ".claude/data/gen/%s/%s" % (ds, spec["file"]), src(spec["file"]), "control_set", "artefact",
            "chlu.core.gen_request.load_registered_control_requests via exp_gen_baselines.py:1562 — the %r query set, named BY query_sets.json['sets'][%r]['file']" % (name, name),
        ]))
    elif spec.get("derived_from"):
        print("\t".join([
            ".claude/data/gen/%s/%s" % (ds, spec["derived_from"]), src(spec["derived_from"]),
            "npz_index", "artefact",
            "chlu.data.gen_splits.load_index via load_registered_query_sets — %r is derived_from this file" % (name,),
        ]))
PY
        while IFS="$TAB" read -r rel rsrc how kind loader; do
            [ -n "$rel" ] || continue
            if [ -f "$rsrc" ]; then
                add_row "$rel" "$rsrc" "$how" "$kind" "$loader"
            else
                say "  ⚠ registry names $rel but it is absent at $rsrc — ⛔ not shipping a path that is not there"
            fi
        done < "$ROWS.reg"
        rm -f "$ROWS.reg"
    done
fi

# ---------------------------------------------------------------------------
# ⭐⭐ 3c. THE PER-HUB ARTEFACT REQUIREMENT (§2b), APPLIED.
#    ⛔ This is the block whose absence made `hub-refusal` "the cheapest unclaimed
#    target" on this machine and the most expensive one on the far side. It runs even
#    under --no-artefacts: a charter's own promise is not an optional extra.
#    ⚠ Appended AFTER the launcher table on purpose — first-row-wins dedup then keeps
#    the launcher's verdict for any file both name, which is the row CSF3 also checks.
# ---------------------------------------------------------------------------
REQ_ROWS="$ROWS.req"
REQ_JDIR="$ROWS.reqjson"
collect_hub_requirements "$REQ_ROWS" "$REQ_JDIR"
if [ -s "$REQ_ROWS" ]; then cat "$REQ_ROWS" >> "$ROWS"; fi

# ---------------------------------------------------------------------------
# 4. Resolve every row's source, checksum it, and total it up.
# ---------------------------------------------------------------------------
HASHED="$ROWS.hashed"
SEEN="$ROWS.seen"
: > "$HASHED"
: > "$SEEN"
TOTAL=0
MISSING=0
while IFS="$TAB" read -r rel src how kind loader; do
    [ -n "$rel" ] || continue
    # ⭐ First row wins. `c_invalid.json` is named twice on purpose — once by the
    #    launcher table (`open_with=c_invalid`) and once by the query-set registry
    #    (`open_with=control_set`). Same file, two real loaders; ship it once, and
    #    keep the launcher's verdict because that is the row CSF3 also checks.
    if grep -qxF "$rel" "$SEEN" 2>/dev/null; then continue; fi
    echo "$rel" >> "$SEEN"
    if [ ! -f "$src" ]; then
        echo "⛔ missing: $rel  (expected at $src)" >&2
        MISSING=$((MISSING + 1))
        continue
    fi
    b="$(bytes_of "$src")"; h="$(sha256_of "$src")"
    TOTAL=$((TOTAL + b))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$src" "$how" "$kind" "$b" "$h" "$loader" >> "$HASHED"
done < "$ROWS"
[ "$MISSING" -eq 0 ] || die "$MISSING declared file(s) are missing — ⛔ refusing to ship a partial workspace"

N_CLAUDE="$(wc -l < "$HASHED" | tr -d ' ')"
CODE_BYTES="$(git -C "$SRC_ROOT" archive --format=tar "$PIN" | wc -c | tr -d ' ')"
say "files    : $N_CODE code (tracked @ pin) + $N_CLAUDE .claude"
say "bytes    : code ~$CODE_BYTES (tar) + .claude $TOTAL  = ~$((CODE_BYTES + TOTAL))"
"$PY_BIN" -c "print('[bundle] total    : %.3f GiB' % ((${CODE_BYTES} + ${TOTAL}) / 2**30))"

if [ "$DRY_RUN" = "1" ]; then
    say "=== DRY RUN — the plan (nothing written) ==="
    awk -F"$TAB" '{printf "  %-10s %12d  %s\n", $4, $5, $1}' "$HASHED" | sort -k1,1 -k2,2nr
    say "would write the bundle to $OUT"
    exit 0
fi

# ---------------------------------------------------------------------------
# 5. Write it. Idempotent: a file whose sha already matches is not re-copied.
# ---------------------------------------------------------------------------
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# (re-checked after resolution: a symlinked --out can land inside the repo)
if [ "$OUT" = "$SRC_ROOT" ]; then die "⛔ --out is the source root. That would write the bundle over the repo."; fi
case "$OUT" in "$SRC_ROOT"/*) die "⛔ --out is inside the source root ($SRC_ROOT). Bundle somewhere else." ;; esac

# ⛔⛔ EXTRACT THE CODE ONLY INTO A FRESH BUNDLE.
#
# ⚠ If `$OUT/.git` exists, this bundle has already been built — and a hub may have
# been WORKING in it. Re-extracting `git archive <pin>` over that would overwrite the
# hub's own edits to `chlu/**` with the pin's versions, silently, with no diff to see
# afterwards. ⛔ That is data loss, not a refresh.
# ⭐ The `.claude` set below is still re-checked and repaired on every run; the code is
#    under git, so `git status` in the bundle shows any corruption there.
if [ -d "$OUT/.git" ]; then
    say "code : ⛔ NOT re-extracting — $OUT/.git exists, so a hub may have edits here."
    say "       ⭐ \`git -C $OUT status\` shows the code state. Delete the bundle to rebuild it clean."
else
    say "code -> $OUT (git archive @ $PIN)"
    git -C "$SRC_ROOT" archive --format=tar "$PIN" | ( cd "$OUT" && chlu_tar -xf - )
fi

# ---------------------------------------------------------------------------
# ⭐⭐ MAKE THE BUNDLE A GIT REPOSITORY — and prove the code IS the pin.
#
# ⚠ `git archive` produces a plain directory. The hub's own profile (§0.2) makes a
# WORKTREE its sandbox — "⛔ You never touch main. The Hub merges your branch after
# review." ⛔ In a plain directory a hub can create no branch, no worktree and no
# commit, so its one isolation mechanism does not exist and there is no way to get
# its CODE changes home. (`.claude/outputs` comes home by --return-pack; a modified
# CLU does not.)
#
# ⭐ TWO COMMITS, ON PURPOSE:
#   HEAD~1  = the pin's tree, VERBATIM. Its tree hash is compared against
#             `git rev-parse <PIN>^{tree}` on this machine and recorded in the
#             manifest — a one-line cryptographic proof, not a promise.
#   HEAD    = + the declared tooling overlay, and nothing else.
#             ⭐ `git diff --stat $(git rev-list --max-parents=0 HEAD) HEAD` is the whole
#             deviation from the pin. ⚠ ⛔ NOT `git show HEAD` — a REBUILT bundle adds a
#             second overlay commit and `show` would then display only the last delta.
#
# ⛔ Generated files (the manifest, env_hub.sh, the READMEs, WORKSPACE-READY.json)
#    go in `.git/info/exclude`, ⛔ NOT in `.gitignore` — that file is tracked, at the
#    pin, and is not ours to edit.
# ---------------------------------------------------------------------------
PIN_TREE="$(git -C "$SRC_ROOT" rev-parse "${PIN}^{tree}")"
BUNDLE_BASE_TREE=""
BUNDLE_TREE_MATCH="false"
if [ "$GIT_INIT" = "1" ]; then
    if [ -d "$OUT/.git" ]; then
        say "git  : $OUT/.git already exists — ⛔ leaving it alone (a hub may have committed into it)"
        BUNDLE_BASE_TREE="$(git -C "$OUT" rev-parse 'HEAD~1^{tree}' 2>/dev/null || echo unknown)"
        if [ "$BUNDLE_BASE_TREE" = "$PIN_TREE" ]; then BUNDLE_TREE_MATCH="true"; fi
    else
        say "git  : init at $OUT (base commit = the pin's tree, verbatim)"
        git -C "$OUT" init -q -b main
        {
            echo "# ⭐ generated by bundle_hub_workspace.sh; ⛔ not part of the pin's tree"
            echo "/BUNDLE_MANIFEST.json"
            echo "/WORKSPACE-READY.json"
            echo "/SUPPLEMENT_MANIFEST*.json"
            echo "/env_hub.sh"
            echo "/README-HUB-BUNDLE.md"
            echo "/.workspace-verify-test.log"
            echo "/coverage.xml"
        } >> "$OUT/.git/info/exclude"
        # ⚠ Re-running onto a directory that already holds a previous run's overlay
        #    would fold the overlay INTO the base commit and the tree would no longer
        #    be the pin's. Exclude exactly those tooling paths the pin does not have —
        #    computed from the pin, ⛔ not hardcoded — and let the tree assertion below
        #    catch it if this reasoning is ever wrong.
        BASE_EXCLUDE=()
        for _t in "${TOOLING[@]}"; do
            if ! git -C "$SRC_ROOT" cat-file -e "$PIN:$_t" 2>/dev/null; then
                BASE_EXCLUDE+=(":(exclude)$_t")
            fi
        done
        git -C "$OUT" add -A -- . "${BASE_EXCLUDE[@]+"${BASE_EXCLUDE[@]}"}"
        git -C "$OUT" -c user.name="chlu-bundle" -c user.email="bundle@localhost" \
            commit -q -m "[bundle] the CHLU tree at pin $PIN, verbatim

⛔ Nothing here is ours. This commit exists so the far side has a git repository:
the hub's sandbox is a WORKTREE (agent profile §0.2) and a plain directory has none.
Its tree hash is asserted against \`git rev-parse $PIN^{tree}\` on the build machine."
        BUNDLE_BASE_TREE="$(git -C "$OUT" rev-parse 'HEAD^{tree}')"
        if [ "$BUNDLE_BASE_TREE" = "$PIN_TREE" ]; then
            BUNDLE_TREE_MATCH="true"
            say "git  : ⭐ base tree $BUNDLE_BASE_TREE == pin tree — the code IS the pin"
        else
            say "git  : ⚠⚠ base tree $BUNDLE_BASE_TREE != pin tree $PIN_TREE"
            say "       ⛔ REPORT THIS. The bundle's code is not bit-identical to the pin."
        fi
    fi
fi

N_COPIED=0; N_SKIPPED=0
while IFS="$TAB" read -r rel src how kind b h loader; do
    [ -n "$rel" ] || continue
    dst="$OUT/$rel"
    if [ -f "$dst" ] && [ "$FORCE" != "1" ] && [ "$(sha256_of "$dst")" = "$h" ]; then
        N_SKIPPED=$((N_SKIPPED + 1)); continue
    fi
    mkdir -p "$(dirname "$dst")"
    cp -p "$src" "$dst.part.$$"
    mv -f "$dst.part.$$" "$dst"      # atomic: a half-copied 1.9 GB features.npy never has the real name
    N_COPIED=$((N_COPIED + 1))
done < "$HASHED"
say "copied $N_COPIED, skipped $N_SKIPPED already-identical (idempotent)"

# ⭐ Commit #2: the tooling overlay, alone. (`.claude/**` is excluded by the pin's
#    own .gitignore, so it never enters either commit — the far side's `.claude` is
#    untracked there exactly as it is here.)
if [ "$GIT_INIT" = "1" ] && [ -d "$OUT/.git" ]; then
    if [ -n "$(git -C "$OUT" status --porcelain)" ]; then
        git -C "$OUT" add -A
        git -C "$OUT" -c user.name="chlu-bundle" -c user.email="bundle@localhost" \
            commit -q -m "[bundle] the workspace tooling, overlaid on the pin

⭐ THE WHOLE DEVIATION FROM THE PIN — \`git show --stat HEAD\` lists it.
The pin predates this tooling, so a bundle built from it alone could neither verify
itself nor return a result. ⛔ None of these files is imported by \`chlu\`; ⛔ none is
part of the scoring harness the charter freezes.
Overlaid from $TOOL_COMMIT."
        say "git  : overlay committed — the full deviation from the pin is"
        say "git  :   git diff --stat \$(git rev-list --max-parents=0 HEAD) HEAD"
    else
        say "git  : no overlay delta to commit (already current)"
    fi
    git -C "$OUT" log --oneline -2 | sed 's/^/[bundle] git  : /'
fi

# ---------------------------------------------------------------------------
# 6. env_hub.sh — ⭐ $CHLU_ROOT is the ONLY thing the far side has to set.
# ---------------------------------------------------------------------------
cat > "$OUT/env_hub.sh" <<'ENVEOF'
# env_hub.sh — source this on the far side. ⭐ $CHLU_ROOT is the single source of truth.
#   cd <bundle> && source ./env_hub.sh && ./scripts/verify_hub_workspace.sh
# ⚠ If CHLU_ROOT is unset it defaults to THIS file's directory, so the bundle is
#   correct wherever it is extracted. ⛔ Not one path here names the build machine.
_ENV_HUB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export CHLU_ROOT="${CHLU_ROOT:-$_ENV_HUB_DIR}"
export CHLU_CLAUDE_ROOT="${CHLU_CLAUDE_ROOT:-$CHLU_ROOT/.claude}"
export CHLU_GEN_DATA_ROOT="${CHLU_GEN_DATA_ROOT:-$CHLU_CLAUDE_ROOT/data/gen}"
export CHLU_GEN_BRIDGE_ROOT="${CHLU_GEN_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs}"
export CHLU_GEN_WIDE_BRIDGE_ROOT="${CHLU_GEN_WIDE_BRIDGE_ROOT:-$CHLU_CLAUDE_ROOT/outputs/gen-bridge-wide/runs}"
export CHLU_GEN_STORE_RUNS="${CHLU_GEN_STORE_RUNS:-$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs}"
export CHLU_GEN_DECLIFT_DIR="${CHLU_GEN_DECLIFT_DIR:-$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner}"
# ⭐ The one path a hub may write (agent profile §0.3), derived — ⛔ never hardcoded.
export CHLU_HUB_OUTPUTS="${CHLU_HUB_OUTPUTS:-$CHLU_CLAUDE_ROOT/outputs}"
# ⛔⛔ THE IMPORT TRAP. `python -c` puts the CWD at sys.path[0], AHEAD of PYTHONPATH.
# Run from anywhere but $CHLU_ROOT and you silently import some other checkout's chlu.
# verify_hub_workspace.sh asserts chlu.__file__ for exactly this reason.
export PYTHONPATH="$CHLU_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset _ENV_HUB_DIR
ENVEOF
say "wrote $OUT/env_hub.sh"

# ---------------------------------------------------------------------------
# 7. BUNDLE_MANIFEST.json — same shape as the stage manifest, so the two
#    platforms stay comparable (`path/bytes/sha256/loader/open_with/required`).
# ---------------------------------------------------------------------------
BM_OUT="$OUT/BUNDLE_MANIFEST.json" BM_PIN="$PIN" BM_SRC="$SRC_ROOT" BM_TOTAL="$TOTAL" \
BM_CODEBYTES="$CODE_BYTES" BM_NCODE="$N_CODE" BM_HUBS="${HUBS[*]}" BM_HOST="$(hostname)" \
BM_STAGE_MF="$STAGE_MF" BM_TOOLCOMMIT="$TOOL_COMMIT" BM_PINTREE="$PIN_TREE" \
BM_BASETREE="$BUNDLE_BASE_TREE" BM_TREEMATCH="$BUNDLE_TREE_MATCH" \
BM_REQ_DIR="$REQ_JDIR" \
"$PY_BIN" - "$HASHED" <<'PY'
import datetime as _dt, json, os, sys
e = os.environ.get
files = []
with open(sys.argv[1]) as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        rel, src, how, kind, size, sha, loader = line.split("\t", 6)
        files.append({
            "path": rel, "bytes": int(size), "sha256": sha,
            "loader": loader, "open_with": how, "kind": kind,
            "required": True,
        })
files.sort(key=lambda d: d["path"])

# ---------------------------------------------------------------------------
# ⭐ What each hub's OWN charter declared (§2b), and the SUPERSEDED rows it named.
#
# ⚠ `gen-arms-a7-wiring/refusal_scores.json` is the live case and it is NOT simply
# dropped: the launcher table ships it as `exp_gen_cell.check_preconditions`' own
# precondition, so deleting it would break a different consumer. ⛔ But `hub-refusal`'s
# charter says it must never be quoted AS THE RIVAL ROW — so it travels FLAGGED, in the
# manifest, where the far side is already looking. ⛔ Silence is what this whole task
# is about.
# ---------------------------------------------------------------------------
hub_reqs = {}
_req_dir = e("BM_REQ_DIR") or ""
if os.path.isdir(_req_dir):
    for fn in sorted(os.listdir(_req_dir)):
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(_req_dir, fn), encoding="utf-8"))
            hub_reqs[d["hub"]] = d
superseded = {x["path"]: (hub, x) for hub, d in hub_reqs.items() for x in d["excluded"]}
n_flagged = 0
for f in files:
    if f["path"] in superseded:
        hub, x = superseded[f["path"]]
        f["superseded_for_hub"] = hub
        f["superseded_note"] = (
            "⛔ %s's charter marks this SUPERSEDED — never quote it as the rival row. "
            "It ships only because another consumer opens it (%s)." % (hub, f["loader"][:80])
        )
        n_flagged += 1
if n_flagged:
    print("[bundle] ⚠ %d shipped file(s) flagged SUPERSEDED by a hub charter — see "
          "`superseded_note` in the manifest" % n_flagged)

stage = {}
try:
    stage = json.load(open(e("BM_STAGE_MF")))
except Exception:
    pass
m = {
    "schema": "chlu-hub-workspace-bundle/1",
    "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    "built_on_host": e("BM_HOST"),
    "built_from_root": e("BM_SRC"),
    "pin": e("BM_PIN"),
    "hubs": e("BM_HUBS", "").split(),
    "code": {
        # ⛔ The code is NOT listed file-by-file: it is `git archive <pin>`, and git's
        # own tree hash is a stronger statement than 900 shas. Verified by re-archiving.
        "source": "git archive %s" % e("BM_PIN"),
        "n_tracked_files": int(e("BM_NCODE", "0")),
        "tar_bytes": int(e("BM_CODEBYTES", "0")),
        # ⭐ The proof, not the promise: HEAD~1's tree in the bundle's own git repo.
        "pin_tree": e("BM_PINTREE"),
        "bundle_base_tree": e("BM_BASETREE"),
        "bundle_base_tree_matches_pin": e("BM_TREEMATCH") == "true",
        # ⭐ Where the workspace deviates from the pin, stated out loud.
        "tooling_overlay": {
            "from_commit": e("BM_TOOLCOMMIT"),
            "files": sorted(f["path"] for f in files if f["kind"] == "tooling"),
            "why": ("the pin predates the bundle tooling; without these three files the "
                    "far side could neither verify the workspace nor return a result. "
                    "⛔ None is imported by chlu; none is part of the frozen scoring harness."),
        },
    },
    "n_files": len(files),
    "claude_bytes": int(e("BM_TOTAL", "0")),
    "total_bytes": int(e("BM_TOTAL", "0")) + int(e("BM_CODEBYTES", "0")),
    "artefact_axes": stage.get("axes"),
    # ⭐ What each hub's OWN charter declared, and what was therefore shipped. ⛔ A
    #    charter that names an artefact the bundle cannot carry no longer builds at all.
    "hub_requirements": hub_reqs,
    "env": {
        "CHLU_ROOT": "<the bundle root, wherever it is extracted>",
        "CHLU_CLAUDE_ROOT": "$CHLU_ROOT/.claude",
        "CHLU_GEN_DATA_ROOT": "$CHLU_CLAUDE_ROOT/data/gen",
        "CHLU_GEN_BRIDGE_ROOT": "$CHLU_CLAUDE_ROOT/outputs/gen-feature-bridge/runs",
        "CHLU_GEN_STORE_RUNS": "$CHLU_CLAUDE_ROOT/outputs/gen-store-grouping/runs",
        "CHLU_GEN_DECLIFT_DIR": "$CHLU_CLAUDE_ROOT/outputs/gen-decoder-lift/winner",
        "CHLU_HUB_OUTPUTS": "$CHLU_CLAUDE_ROOT/outputs",
    },
    "not_bundled": {
        "datasets": ("3dshapes.h5 (255 MiB, sha256 0a0f6ed9…be1785) and the dSprites blobs — "
                     "re-downloadable with `chlu data shapes3d --download`; see README-HUB-BUNDLE.md"),
        "scratch": ".claude/scratch/ (11 GB) — ⛔ no loader names any of it",
        "foreign_slugs": ("every other slug's .claude/outputs/** — the agent profile forbids reading "
                          "the campaign archive, so shipping it would be shipping a distraction"),
        "campaign_ledgers": ("advisor-*, c3-handover, claims_matrix, negative_results, papers/ — "
                             "⛔ excluded on purpose (agent profile: 'Read nothing else from the "
                             "campaign archive')"),
    },
    "files": files,
}
json.dump(m, open(e("BM_OUT"), "w"), indent=2, sort_keys=True)
open(e("BM_OUT"), "a").write("\n")
print("[bundle] manifest -> %s  (%d .claude files, %.3f GiB total)"
      % (e("BM_OUT"), len(files), m["total_bytes"] / 2**30))
PY

# ---------------------------------------------------------------------------
# 8. README-HUB-BUNDLE.md — the far side reads this first.
# ---------------------------------------------------------------------------
{
cat <<'RMEOF'
# Hub workspace bundle — read this first

This directory **is** `$CHLU_ROOT`. Everything resolves from it; nothing is hardcoded.

```bash
cd <this directory>
source ./env_hub.sh            # sets CHLU_ROOT + every derived CHLU_* root
./scripts/verify_hub_workspace.sh     # re-checksums, OPENS every artefact, writes WORKSPACE-READY.json
```

⛔ **Do not run a hub until `WORKSPACE-READY.json` says `"status": "ready"`.**

## ⛔⛔ JAX HAS NO NATIVE WINDOWS SUPPORT — use WSL2

`jaxlib` publishes **no `win_amd64` wheel**. `pip install jax jaxlib` on native Windows
resolves nothing installable, and a hub imports `chlu` (hence `jax`) locally *even when its
compute is on GCloud*. ⇒ **On the second machine, run inside WSL2 (Ubuntu), not native Windows.**
Zed connects to WSL2 over its remote/SSH project support; the bundle lives on the WSL2 filesystem
(`/home/<user>/CHLU-hub`), ⛔ **not** under `/mnt/c/…` — the 9p bridge makes the 1.9 GB
`features.npy` read pathologically slowly and breaks `mmap_mode="r"` performance.

```bash
wsl --install -d Ubuntu            # PowerShell, admin, once
wsl                                # then, inside Ubuntu:
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv ~/chlu-venv && source ~/chlu-venv/bin/activate
pip install -e "$CHLU_ROOT"        # or: pip install -r <constraints>; JAX CPU wheels are manylinux
```

## The dataset is NOT in this bundle

It is 255 MiB and re-downloadable. Fetch and **assert the checksum before using it**:

```bash
python -m chlu.cli data shapes3d --download     # or: chlu data shapes3d --download
sha256sum "$CHLU_GEN_DATA_ROOT/shapes3d/3dshapes.h5"
# MUST equal:
# 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785
```

⛔ A different sha means a different grid, and every request index in the C set is then wrong.

Most hub work does **not** need the image blobs at all: the feature caches in
`.claude/outputs/gen-feature-bridge/runs/**` are pre-encoded.

## ⭐ THE RETURN PATH — a result that is not synced back does not exist

A hub writes only to `$CHLU_HUB_OUTPUTS/<hub-slug>/**` and `<hub-slug>.md`. To bring that home:

```bash
# on the SECOND machine, when the hub is done:
./scripts/bundle_hub_workspace.sh --return-pack <hub-slug> --return-out ~/<hub-slug>.return.tar.gz
# carry the tar back (scp / drive / whatever), then on the PRIMARY machine:
scripts/bundle_hub_workspace.sh --return-ingest /path/<hub-slug>.return.tar.gz
```

### …and the hub's CODE comes back by `git bundle`

The results tar carries `.claude/outputs/**` only. A hub **may modify the CLU** — that
work lives on its branch in this repository, and `git push` is forbidden to it.

```bash
# on the SECOND machine (the hub's branch, everything since the pin):
git -C "$CHLU_ROOT" bundle create ~/<hub-slug>.gitbundle agent/autonomous-hub/<hub-slug>
git -C "$CHLU_ROOT" bundle verify ~/<hub-slug>.gitbundle   # "records a complete history"
# on the PRIMARY machine — ⛔ fetch into a namespaced ref; never onto a local branch:
git fetch /path/<hub-slug>.gitbundle 'refs/heads/agent/autonomous-hub/<hub-slug>:refs/remotes/farside/<hub-slug>'
git diff --stat <pin> refs/remotes/farside/<hub-slug>      # the hub's whole code delta
```

⛔⛔ **`git bundle create … main..<branch>` DOES NOT WORK, and it fails only on arrival.**
A range bundle is *thin*: its prerequisite is this repository's own synthetic base
commit, which the primary machine has never seen, so the fetch dies with
`Repository lacks these prerequisite commits`. ⭐ Bundle the branch WHOLE. Measured:
~3 MB, self-contained. ⚠ Take the 3 MB.

⛔ `refs/remotes/farside/*` cannot fast-forward `main` or any local branch — the fetched
history shares no ancestor with it — so the worst case is an unreferenced ref, never a
lost commit. The Hub reviews, then cherry-picks or merges deliberately.

⭐ **The base commit of this repository holds the pin's tree verbatim** — `git log`'s
root commit. `verify_hub_workspace.sh` asserts its tree hash against the manifest, so
"the code is the pin" is checkable in one command, not a promise:

```bash
git -C "$CHLU_ROOT" rev-parse "$(git -C "$CHLU_ROOT" rev-list --max-parents=0 HEAD)^{tree}"
python -c "import json;print(json.load(open('BUNDLE_MANIFEST.json'))['code']['pin_tree'])"
```

```bash
git -C "$CHLU_ROOT" diff --stat "$(git -C "$CHLU_ROOT" rev-list --max-parents=0 HEAD)" HEAD
```
lists the **entire** deviation from the pin: three plumbing files, nothing else (plus
whatever the hub itself has since committed). ⛔ Not `git show --stat HEAD` — a rebuilt
bundle carries a second overlay commit and `show` would display only the last delta.

## Safety of the results ingest

`--return-ingest` **cannot overwrite anything**:
* every member path must be relative, `..`-free, not a symlink, and prefixed with `<hub-slug>`;
* every sha must match `RETURN_MANIFEST.json`, and a file in the tar that the manifest does not
  declare aborts the whole ingest;
* if `.claude/outputs/<hub-slug>` already exists the entire set is diverted to
  `<hub-slug>.returned-<utc>/` and the divert is printed. ⛔ Nothing existing is ever touched.

## What is deliberately NOT here

`.claude/scratch/` · every other slug's outputs · the campaign ledgers · the datasets.
The hub's own profile says: *"⛔ Read nothing else from the campaign archive."* This bundle
enforces that by construction.
RMEOF
} > "$OUT/README-HUB-BUNDLE.md"
say "wrote $OUT/README-HUB-BUNDLE.md"

if [ "$DO_TAR" = "1" ]; then
    TARBALL="$OUT.tar.gz"
    say "tarring -> $TARBALL (this is ~$( "$PY_BIN" -c "print('%.2f' % ((${CODE_BYTES}+${TOTAL})/2**30))") GiB uncompressed)"
    ( cd "$(dirname "$OUT")" && chlu_tar -czf "$TARBALL" "$(basename "$OUT")" )
    assert_no_apple_double "$TARBALL"
    say "sha256($(basename "$TARBALL")) = $(sha256_of "$TARBALL")"
fi

say "=== bundle built at $OUT ==="
say "⭐ NOW PROVE IT: cd $OUT && source ./env_hub.sh && ./scripts/verify_hub_workspace.sh"
