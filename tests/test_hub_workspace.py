"""Tests for the PORTABLE HUB WORKSPACE BUNDLE.

⛔ **There is no second machine here.** Three autonomous hubs are to run on a
Windows + Zed box (under WSL2 — ``jaxlib`` publishes no ``win_amd64`` wheel), and
``git push`` cannot carry what a hub reads: ``.gitignore`` excludes ``.claude/**``,
so a push delivers **no** charter, **no** agent profile, **no** hub task file and
**no** artefact.

What *is* testable, and what this file covers, is the half that decides whether the
far side can work at all:

* ``scripts/bundle_hub_workspace.sh`` builds a bundle that **is** a complete
  ``$CHLU_ROOT`` — code at the pin, the four documents a hub reads, each hub's own
  task file, and the **loader-derived** artefact set (derived by *invoking*
  ``scripts/csf3/stage_artefacts.sh``, ⛔ never by listing a directory);
* ``scripts/verify_hub_workspace.sh`` re-checksums **and opens** every file, asserts
  ``chlu`` is imported **from the bundle**, and **fails** on a missing file, on a
  byte-perfect file its loader rejects, on a document that lost its own anchor, and
  on an import that resolved to some other checkout. ⭐ A checker that cannot fail
  proves nothing, so every one of those failures is asserted here;
* the bundle is a **git repository whose root tree is the pin's tree**, so the hub's
  mandatory worktree (§0.2) exists and "the code is the pin" is checkable, not claimed;
* the **return path** never overwrites anything and refuses a tar that traverses,
  smuggles an undeclared member, or reaches outside the hub's own slug.

The fixture is a synthetic ``.claude/`` mirror — real schemas, tiny payloads — reused
from :mod:`tests.test_stage_artefacts`, so the suite never depends on the 1.95 GiB
shipped artefact set being present.

⛔⛔ The round-trip tests carry ``@needs_bundle``, which skips them whenever
``$CHLU_WORKSPACE_VERIFY_DEPTH`` is set — i.e. when this file is running *underneath*
``verify_hub_workspace.sh``. That is the ONLY thing stopping
``verify`` -> ``pytest`` -> ``verify`` from forking without bound. ⛔ Do not remove it,
and ⛔ do not replace it with an inference about the environment: the previous guard
("a bundle has no ``.git``, so these skip there") stopped holding silently the day the
bundle became a git repository, and the run had to be killed by hand.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PureWindowsPath

import pytest

from tests.test_stage_artefacts import (
    DATASET,
    PRETEXT,
    SEED,
    build_fixture_root,
    _write_json,
)

REPO = Path(__file__).resolve().parents[1]
BUNDLE_SH = REPO / "scripts" / "bundle_hub_workspace.sh"
VERIFY_SH = REPO / "scripts" / "verify_hub_workspace.sh"

#: ⛔ The only absolute path allowed to survive anywhere, and only as a *default*.
LEGACY_ROOT = "/Users/user/Desktop/CHLU"

HUB_SLUG = "hub-test-target"

#: The four documents the agent profile tells a hub to read — and ⛔ nothing else
#: from the campaign archive.
HUB_DOCS = {
    "AGENT_PROTOCOL.md": (
        "# CHLU Spoke-Agent Protocol (shared)\n\n"
        "## 7. Dial declaration\nEvery task file opens with a DIAL DECLARATION block.\n"
    ),
    "autonomous-hub-charter.md": (
        "# AUTONOMOUS HUB CHARTER\n\n"
        "THE PIN - ae907466ba2b5fbec124e8995cf4d7c32a1ab30e\n\n"
        "## B. The core philosophy\nThe store is a LANDSCAPE.\n\n"
        "## C. The floor\nEvery arm must beat the MATCHED-BYTES LOOKUP.\n"
    ),
    "agents/autonomous-hub.md": (
        "---\nname: autonomous-hub\n---\n\n"
        "# SECTION 0 - THE SANDBOX\n## 0.1 YOU NEVER DELETE. ANYTHING. EVER.\n"
        "# 2. PRE-REGISTRATION UNDER AUTONOMY\n"
    ),
    "handover_context.md": "# Handover\n\n## 7. Known Issues & Code-vs-Paper Discrepancies\n",
}


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo_has_git() -> bool:
    try:
        _git("rev-parse", "HEAD")
        return True
    except Exception:  # noqa: BLE001
        return False


#: ⛔⛔ THE RECURSION GUARD, and it is EXPLICIT on purpose.
#:
#: `verify_hub_workspace.sh` runs this file; the tests below build bundles and run
#: `verify_hub_workspace.sh`. ⚠ The guard was once implicit — "a bundle has no `.git`,
#: so these tests skip there". The moment the bundle became a git repository that
#: stopped being true, the run forked unboundedly, and 28 processes had to be killed
#: by hand. ⭐ The depth is now an exported integer; nothing relies on a side effect.
NESTED = os.environ.get("CHLU_WORKSPACE_VERIFY_DEPTH", "0") not in ("", "0")

needs_bundle = pytest.mark.skipif(
    NESTED or not _repo_has_git(),
    reason=("nested under verify_hub_workspace.sh (recursion guard)" if NESTED
            else "no git repository here — this is an extracted bundle, not a checkout"),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def claude_root(tmp_path: Path) -> Path:
    """A synthetic ``.claude/``: the artefact fixture + the four hub documents."""
    root = build_fixture_root(tmp_path / "claude")
    for rel, text in HUB_DOCS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "tasks" / f"{HUB_SLUG}.md").write_text(
        f"# Hub: {HUB_SLUG}\nOne question, ten hours, through the CLU.\n"
    )

    # ⭐ The hub addendum the CSF3 launcher table does not carry: the query-set
    #    registry `exp_gen_baselines` opens for the REFUSAL path, and the control
    #    set the registry *names*. ⛔ The filename is read out of the registry, never
    #    guessed — the same discipline as feature_cache.json's `features_file`.
    dgen = root / "data" / "gen" / DATASET
    _write_json(dgen / "a7_redundant.json", {
        "schema": {"version": 1},
        "records": [
            {"id": "a7-0", "kind": "valid_redundant", "row": 0, "redundant_factor": "scale",
             "request": [{"factor_index": 3, "factor": "scale", "value": 0.0},
                         {"factor_index": 3, "factor": "scale", "value": 0.0}]},
        ],
    })
    _write_json(dgen / "query_sets.json", {
        "dataset": DATASET,
        "schema": {"version": 1},
        "excluded_record_keys": ["kind", "violated_factor", "reason", "row", "redundant_factor"],
        "sets": {
            "a7_redundant": {"file": "a7_redundant.json", "role": "valid", "n_records": 1},
            "c_invalid": {"file": "c_invalid.json", "role": "invalid",
                          "controls": {"duplicate_factor": "a7_redundant"}},
        },
    })
    return root


def run_bundle(*args: str, claude: Path, expect: int = 0,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, CHLU_CLAUDE_ROOT=str(claude), BUNDLE_PYTHON=sys.executable)
    # ⛔ nothing in a test may ever inherit a real artefact root
    for k in ("CHLU_GEN_DATA_ROOT", "CHLU_GEN_BRIDGE_ROOT", "CHLU_GEN_STORE_RUNS",
              "CHLU_GEN_DECLIFT_DIR", "CHLU_GEN_WIDE_BRIDGE_ROOT", "CHLU_HUB_OUTPUTS",
              "CSF3_USER"):
        env.pop(k, None)
    env.update(env_extra or {})
    p = subprocess.run(["bash", str(BUNDLE_SH), *args], capture_output=True, text=True, env=env)
    assert p.returncode == expect, (
        f"rc={p.returncode} (expected {expect})\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    )
    return p


def build_bundle(out: Path, claude: Path, *extra: str) -> Path:
    run_bundle(
        "--src", str(REPO), "--pin", _git("rev-parse", "HEAD"),
        "--out", str(out), "--hub", HUB_SLUG,
        "--datasets", DATASET, "--pretexts", PRETEXT, "--seeds", str(SEED),
        *extra, claude=claude,
    )
    return out


def run_verify(root: Path, expect: int | None = None,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run the verifier against a bundle, targeted test and all.

    ⚠ This re-enters pytest one level down (verify -> pytest -> this file). It
    TERMINATES **only** because the verifier exports ``CHLU_WORKSPACE_VERIFY_DEPTH``
    and every ``@needs_bundle`` test skips on it, so no second bundle is ever built.
    ⛔ Do not remove that guard, and ⛔ do not replace it with an inference.
    """
    env = dict(
        os.environ,
        CHLU_ROOT=str(root),
        VERIFY_PYTHON=sys.executable,
        HUB_SLUG=HUB_SLUG,
    )
    for k in ("CHLU_CLAUDE_ROOT", "CHLU_GEN_DATA_ROOT", "CHLU_GEN_BRIDGE_ROOT",
              "CHLU_GEN_STORE_RUNS", "CHLU_GEN_DECLIFT_DIR", "CHLU_GEN_WIDE_BRIDGE_ROOT",
              "CHLU_HUB_OUTPUTS"):
        env.pop(k, None)
    env.update(env_extra or {})
    p = subprocess.run(
        ["bash", str(root / "scripts" / "verify_hub_workspace.sh")],
        capture_output=True, text=True, env=env, cwd=str(root),
    )
    if expect is not None:
        assert p.returncode == expect, (
            f"rc={p.returncode} (expected {expect})\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p


def ready(root: Path) -> dict:
    return json.loads((root / "WORKSPACE-READY.json").read_text())


# ---------------------------------------------------------------------------
# 1. The scripts themselves  (these run everywhere, including inside a bundle)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [BUNDLE_SH, VERIFY_SH])
def test_script_exists_and_parses(script: Path):
    assert script.is_file(), f"{script} is missing"
    assert os.access(script, os.X_OK), f"{script} is not executable"
    p = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_the_legacy_absolute_path_survives_only_as_a_default():
    """⭐ The macOS path may appear ONCE, as ``CHLU_ROOT_DEFAULT``.

    ⛔ Any other occurrence is a path that will be wrong on the second machine, and
    ⛔ the verifier — which must run *there* — may not mention it at all.
    """
    hits = [ln for ln in BUNDLE_SH.read_text().splitlines()
            if LEGACY_ROOT in ln and not ln.lstrip().startswith("#")]
    assert hits == [f'CHLU_ROOT_DEFAULT="{LEGACY_ROOT}"'], hits
    assert LEGACY_ROOT not in VERIFY_SH.read_text(), (
        "verify_hub_workspace.sh runs on the FAR side; it may not name this machine"
    )


def test_verify_needs_no_gpu_and_no_cluster():
    """⛔ This is plumbing. No GPU, no sbatch, no ssh, no transfer."""
    text = VERIFY_SH.read_text()
    for banned in ("#SBATCH", "sbatch", "nvidia-smi", "ssh ", "rsync"):
        assert banned not in text, f"verify_hub_workspace.sh must not reference {banned!r}"


# ---------------------------------------------------------------------------
# 2. $CHLU_ROOT is the single source of truth
# ---------------------------------------------------------------------------


def test_default_gen_data_root_follows_the_env_var(tmp_path, monkeypatch):
    """The code side of the contract already exists — assert it, do not change it."""
    from chlu.data.gen_splits import default_gen_data_root

    monkeypatch.setenv("CHLU_GEN_DATA_ROOT", str(tmp_path / "elsewhere"))
    assert default_gen_data_root() == tmp_path / "elsewhere"

    monkeypatch.delenv("CHLU_GEN_DATA_ROOT", raising=False)
    import chlu.data.gen_splits as gs

    assert default_gen_data_root() == Path(gs.__file__).resolve().parents[2] / ".claude" / "data" / "gen"


@pytest.mark.skipif("CHLU_ROOT" not in os.environ, reason="not running under a CHLU_ROOT")
def test_chlu_is_imported_from_chlu_root():
    """⛔⛔ The assertion the whole exercise turns on.

    ``python -c`` puts the CWD at ``sys.path[0]``, **ahead of PYTHONPATH**. Without
    this check a "portable" run can quietly import the original machine's checkout
    and pass everything.
    """
    import chlu

    root = Path(os.environ["CHLU_ROOT"]).resolve()
    assert Path(chlu.__file__).resolve().is_relative_to(root), (
        f"chlu came from {chlu.__file__}, outside $CHLU_ROOT={root}"
    )


@needs_bundle
def test_env_hub_sh_derives_every_root_from_wherever_it_lands(tmp_path: Path, claude_root: Path):
    """⭐ The laundering control: extract to a DIFFERENT absolute path and re-derive."""
    b = build_bundle(tmp_path / "b1", claude_root)
    moved = tmp_path / "some" / "other" / "place"
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(b, moved)

    assert LEGACY_ROOT not in (moved / "env_hub.sh").read_text()
    out = subprocess.run(
        ["bash", "-c", f'cd "{moved}" && unset CHLU_ROOT && . ./env_hub.sh && env | grep ^CHLU_'],
        capture_output=True, text=True, check=True,
    ).stdout
    got = dict(ln.split("=", 1) for ln in out.strip().splitlines())
    assert got["CHLU_ROOT"] == str(moved)
    for k, v in got.items():
        assert v.startswith(str(moved)), f"{k}={v} did not follow CHLU_ROOT"
    assert got["CHLU_HUB_OUTPUTS"] == f"{moved}/.claude/outputs"


# ---------------------------------------------------------------------------
# 3. The bundle is the MINIMUM set, and every row names a loader
# ---------------------------------------------------------------------------


@needs_bundle
def test_bundle_ships_the_minimum_set_and_nothing_else(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    m = json.loads((b / "BUNDLE_MANIFEST.json").read_text())
    paths = [f["path"] for f in m["files"]]

    for rel in HUB_DOCS:
        assert f".claude/{rel}" in paths
    assert f".claude/tasks/{HUB_SLUG}.md" in paths

    # ⛔ the three things a 20 GB bundle would have carried
    assert not [p for p in paths if "/scratch/" in p or p.startswith(".claude/scratch")]
    assert not [p for p in paths if p.endswith((".h5", ".hdf5", ".npz")) and "factor_index" not in p]
    assert not (b / ".claude" / "scratch").exists()

    # ⛔ no foreign slug: every shipped output belongs to a slug some loader names
    slugs = {Path(p).parts[2] for p in paths if p.startswith(".claude/outputs/")}
    assert slugs <= {
        "gen-feature-bridge", "gen-store-grouping", "gen-read-compose",
        "gen-decoder-lift", "gen-external-classifier", "gen-arms-a7-wiring",
    }, slugs

    # ⛔ no campaign ledger
    assert not [p for p in paths
                if Path(p).name in {"claims_matrix.md", "negative_results.md",
                                    "research_roadmap.md", "future_work.md"}]


@needs_bundle
def test_every_manifest_entry_names_a_loader(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    m = json.loads((b / "BUNDLE_MANIFEST.json").read_text())
    assert m["schema"] == "chlu-hub-workspace-bundle/1"
    assert m["files"], "an empty bundle is a silent no-op that looks like a success"
    for f in m["files"]:
        assert f["loader"].strip(), f"{f['path']} names no loader"
        assert f["open_with"] in {
            "json", "eqx_pickle", "feature_cache", "npy", "store_plan", "winner_hashes",
            "c_invalid", "query_registry", "control_set", "npz_index", "npz", "doc", "code",
        }, f["open_with"]
        assert (b / f["path"]).is_file()
        assert not Path(f["path"]).is_absolute() and ".." not in Path(f["path"]).parts


@needs_bundle
def test_the_query_set_registry_names_its_own_members(tmp_path: Path, claude_root: Path):
    """⭐ The refusal path's rows are DERIVED by reading the registry, ⛔ not guessed."""
    b = build_bundle(tmp_path / "b", claude_root)
    paths = {f["path"]: f for f in json.loads((b / "BUNDLE_MANIFEST.json").read_text())["files"]}
    reg = f".claude/data/gen/{DATASET}/query_sets.json"
    a7 = f".claude/data/gen/{DATASET}/a7_redundant.json"
    assert reg in paths and paths[reg]["open_with"] == "query_registry"
    assert a7 in paths and paths[a7]["open_with"] == "control_set"
    assert "query_sets.json" in paths[a7]["loader"], "the row must say it was registry-named"
    # ⛔ shipped once, even though two different loaders name it
    assert sum(p.endswith("c_invalid.json") for p in paths) == 1


@needs_bundle
def test_bundle_is_idempotent_and_dry_run_writes_nothing(tmp_path: Path, claude_root: Path):
    out = tmp_path / "b"
    p = run_bundle(
        "--src", str(REPO), "--pin", _git("rev-parse", "HEAD"), "--out", str(out),
        "--hub", HUB_SLUG, "--datasets", DATASET, "--pretexts", PRETEXT,
        "--seeds", str(SEED), "--dry-run", claude=claude_root,
    )
    assert "DRY RUN" in p.stdout
    assert not out.exists() or not any(out.iterdir())

    build_bundle(out, claude_root)
    p2 = run_bundle(
        "--src", str(REPO), "--pin", _git("rev-parse", "HEAD"), "--out", str(out),
        "--hub", HUB_SLUG, "--datasets", DATASET, "--pretexts", PRETEXT,
        "--seeds", str(SEED), claude=claude_root,
    )
    assert "copied 0," in p2.stdout, p2.stdout
    # ⛔ and the second build must NOT have re-extracted the code over it: a hub may
    #    have edits in `chlu/**` by then, and `git archive` would overwrite them.
    assert "NOT re-extracting" in p2.stdout, p2.stdout


@needs_bundle
def test_bundle_refuses_a_hub_with_no_task_file(tmp_path: Path, claude_root: Path):
    """⛔ A hub with no charter is ten unsupervised hours with no question."""
    p = run_bundle(
        "--src", str(REPO), "--pin", _git("rev-parse", "HEAD"),
        "--out", str(tmp_path / "b"), "--hub", "hub-does-not-exist",
        "--datasets", DATASET, "--pretexts", PRETEXT, "--seeds", str(SEED),
        claude=claude_root, expect=2,
    )
    assert "no task file at" in p.stderr


@needs_bundle
def test_bundle_refuses_to_write_into_the_source_root(tmp_path: Path, claude_root: Path):
    p = run_bundle(
        "--src", str(REPO), "--pin", _git("rev-parse", "HEAD"),
        "--out", str(REPO / "chlu"), "--hub", HUB_SLUG,
        claude=claude_root, expect=2,
    )
    assert "inside the source root" in p.stderr or "over the repo" in p.stderr


# ---------------------------------------------------------------------------
# 3b. ⭐ The bundle is a GIT REPOSITORY whose root tree IS the pin's tree
# ---------------------------------------------------------------------------


@needs_bundle
def test_the_bundles_root_commit_tree_is_the_pin_tree(tmp_path: Path, claude_root: Path):
    """⭐ The proof, not the promise.

    ⛔ "the code is the pin" is a claim a tarball cannot support. The bundle's ROOT
    commit holds the pin's tree object itself, so one `rev-parse` settles it.
    """
    pin = _git("rev-parse", "HEAD")
    b = build_bundle(tmp_path / "b", claude_root)
    base = _git("rev-list", "--max-parents=0", "HEAD", cwd=b)
    assert _git("rev-parse", f"{base}^{{tree}}", cwd=b) == _git("rev-parse", f"{pin}^{{tree}}")

    m = json.loads((b / "BUNDLE_MANIFEST.json").read_text())
    assert m["code"]["bundle_base_tree_matches_pin"] is True
    assert m["code"]["pin_tree"] == _git("rev-parse", f"{pin}^{{tree}}")

    # ⭐ and the ENTIRE deviation from the pin lies inside the three DECLARED plumbing
    #    files — ⛔ nothing else moved.
    #    ⚠ SUBSET, not equality: a tooling file the pin already carries byte-identically
    #    contributes no diff, so the delta is a subset of the declaration and never a
    #    superset. (Equality held only while the pin predated all three — it broke the
    #    moment a test pinned a commit that already had them.)
    #    ⛔ Measured root..HEAD, not `git show HEAD`: a REBUILT bundle carries a second
    #    overlay commit and `show` would then report only the last delta.
    declared = set(m["code"]["tooling_overlay"]["files"])
    assert declared == {
        "scripts/bundle_hub_workspace.sh", "scripts/verify_hub_workspace.sh",
        "tests/test_hub_workspace.py",
    }
    changed = set(_git("diff", "--name-only", base, "HEAD", cwd=b).split())
    assert changed <= declared, changed - declared


@needs_bundle
def test_rebuilding_over_an_existing_bundle_keeps_the_base_tree(tmp_path: Path, claude_root: Path):
    """⚠ The ordering hazard: the second build sees the FIRST build's overlay already
    on disk. ⛔ If that were folded into the base commit, "the code is the pin" would
    quietly stop being true."""
    pin_tree = _git("rev-parse", f'{_git("rev-parse", "HEAD")}^{{tree}}')
    out = tmp_path / "b"
    build_bundle(out, claude_root)
    shutil.rmtree(out / ".git")          # simulate a bundle rebuilt from a bare copy
    build_bundle(out, claude_root)
    base = _git("rev-list", "--max-parents=0", "HEAD", cwd=out)
    assert _git("rev-parse", f"{base}^{{tree}}", cwd=out) == pin_tree


@needs_bundle
def test_the_bundle_repo_can_hold_the_hubs_worktree(tmp_path: Path, claude_root: Path):
    """⛔ The hub's sandbox IS a worktree (agent profile §0.2). A `git archive`
    directory can hold none, and then the hub has no isolation and no way home."""
    b = build_bundle(tmp_path / "b", claude_root)
    wt = tmp_path / "hub-wt"
    _git("worktree", "add", "-q", str(wt), "-b", "agent/autonomous-hub/probe", cwd=b)
    assert (wt / "chlu" / "__init__.py").is_file()
    assert "agent/autonomous-hub/probe" in _git("worktree", "list", cwd=b)
    _git("worktree", "remove", str(wt), cwd=b)


@needs_bundle
def test_verify_refuses_a_bundle_with_no_repository(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root, "--no-git")
    assert not (b / ".git").exists()
    p = run_verify(b)
    assert p.returncode != 0
    r = ready(b)
    assert r["status"] == "NOT-READY" and r["worktree_possible"] is False
    assert "WORKTREE" in r["git_check"]["error"]


# ---------------------------------------------------------------------------
# 3c. ⛔⛔ THE ANCHOR CHECK ON A WINDOWS-STYLE PATH
#
# ⭐ Check #3 of `verify_hub_workspace.sh` ("a truncated charter checksums fine and is
#    useless") is the one check whose whole value is that it FAILS. On Windows it did
#    not: `_load_doc` built its `DOC_ANCHORS` key with `str(Path.relative_to(...))`,
#    which is **backslash-separated** there, while `DOC_ANCHORS` is keyed with forward
#    slashes. ⇒ the lookup ALWAYS missed, `missing` was ALWAYS `[]`, and every document
#    returned `loaded: true` having inspected ZERO anchors. ⛔ Check #3 degraded to
#    "the file is non-empty", and a truncated charter — the exact thing it exists to
#    catch — reported success.
#
# ⚠ There is no Windows box here either, so these tests do not *assume* the flavour:
#    they exec the SHIPPED source of `DOC_ANCHORS` and `_load_doc` out of
#    `verify_hub_workspace.sh` with a `Path` that reproduces the two Windows behaviours
#    that matter (backslashed `relative_to`, cp1252 default decode). ⛔ Nothing here
#    re-implements the loader: if the script's text changes, these change with it.
# ---------------------------------------------------------------------------


def _shipped_block(first_line_prefix: str) -> str:
    """The verbatim source of one top-level definition inside the verifier's heredoc."""
    lines = VERIFY_SH.read_text(encoding="utf-8").splitlines()
    start = next((n for n, ln in enumerate(lines)
                  if ln.startswith(first_line_prefix)), None)
    if start is None:
        raise AssertionError(
            f"verify_hub_workspace.sh no longer defines {first_line_prefix!r} — "
            "this test exercises the SHIPPED source, so it must be re-pointed, not deleted"
        )
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if ln[:1] not in ("", " ", "\t", "#", ")", "]", "}"):
            break
        out.append(ln)
    return "\n".join(out)


class _WindowsFlavourPath:
    """A real filesystem path that behaves the way ``pathlib`` does **on Windows**.

    ⭐ Exactly two behaviours are reproduced, because exactly two are the defect:

    * ``relative_to()`` returns a :class:`PureWindowsPath`, so ``str()`` of it is
      **backslash-separated** — what made every ``DOC_ANCHORS`` lookup miss;
    * ``read_text()`` with no explicit ``encoding=`` decodes as **cp1252**, the Windows
      ANSI default — what mangles every ``⭐``/``⛔``/``⚠`` in these documents.

    Everything else is the real file on the real disk.
    """

    def __init__(self, p):
        self._p = Path(p)

    def resolve(self):
        return _WindowsFlavourPath(self._p.resolve())

    def relative_to(self, other):
        other = other._p if isinstance(other, _WindowsFlavourPath) else Path(other)
        return PureWindowsPath(self._p.relative_to(other))

    def read_text(self, encoding=None, errors="strict"):
        data = self._p.read_bytes()
        return data.decode(encoding or "cp1252", errors)

    def __str__(self):
        return str(self._p)

    def __fspath__(self):
        return str(self._p)


def _load_doc_under(path_cls, root: Path):
    """The SHIPPED ``_load_doc``, bound to a chosen ``Path`` class and ``root``."""
    ns: dict = {"Path": path_cls, "root": path_cls(root) if path_cls is not Path else root}
    exec(_shipped_block("DOC_ANCHORS = "), ns)            # noqa: S102 - shipped source
    exec(_shipped_block("def _load_doc("), ns)            # noqa: S102 - shipped source
    return ns["_load_doc"], ns["DOC_ANCHORS"]


@pytest.fixture()
def charter_root(tmp_path: Path) -> Path:
    """A bundle root holding the real charter, and a truncated twin of it."""
    root = tmp_path / "wsroot"
    (root / ".claude").mkdir(parents=True)
    (root / ".claude" / "autonomous-hub-charter.md").write_text(
        HUB_DOCS["autonomous-hub-charter.md"], encoding="utf-8"
    )
    return root


def test_the_anchor_check_still_fires_on_a_windows_style_path(charter_root: Path):
    """⛔⛔ THE DEFECT. A truncated charter must be REFUSED on Windows too.

    Pre-fix this asserted nothing at all: the key was
    ``.claude\\autonomous-hub-charter.md``, ``DOC_ANCHORS.get(rel, [])`` returned ``[]``,
    and the truncated charter loaded cleanly with ``anchors_checked == 0``.
    """
    victim = charter_root / ".claude" / "autonomous-hub-charter.md"
    victim.write_text("# AUTONOMOUS HUB CHARTER\n\n(the rest was lost in transit)\n",
                      encoding="utf-8")

    # --- positive control: on POSIX the check has always worked. If this ever stops
    #     failing, the test below is passing for the wrong reason.
    posix_load, _ = _load_doc_under(Path, charter_root)
    with pytest.raises(ValueError, match="required anchor"):
        posix_load(victim)

    # --- the actual assertion: the SAME refusal under a Windows path flavour.
    win_load, _ = _load_doc_under(_WindowsFlavourPath, charter_root)
    with pytest.raises(ValueError, match="required anchor"):
        win_load(_WindowsFlavourPath(victim))


def test_a_good_charter_reports_the_anchors_it_actually_inspected(charter_root: Path):
    """⭐ The other half: ``anchors_checked`` must be the real count, ⛔ never 0.

    ``loaded: true`` with ``anchors_checked: 0`` is the signature of the silent
    no-op — it is what the far side reported for all four documents.
    """
    good = charter_root / ".claude" / "autonomous-hub-charter.md"
    _, anchors = _load_doc_under(Path, charter_root)
    expected = len(anchors[".claude/autonomous-hub-charter.md"])
    assert expected > 0, "DOC_ANCHORS declares no anchor for the charter"

    win_load, _ = _load_doc_under(_WindowsFlavourPath, charter_root)
    detail = win_load(_WindowsFlavourPath(good))
    assert detail["anchors_checked"] == expected, (
        "check #3 inspected %d of %d anchors — it degraded to 'the file is non-empty'"
        % (detail["anchors_checked"], expected)
    )


def test_documents_are_decoded_as_utf8_regardless_of_the_platform_locale(charter_root: Path):
    """⚠ The sister bug: bare ``read_text()`` takes the platform default — **cp1252**
    on Windows — and every ``⭐``/``⛔``/``⚠`` in these documents becomes mojibake.
    ⭐ The encoding belongs in the CODE; ⛔ ``PYTHONUTF8=1`` is an environment someone
    has to remember to set.
    """
    doc = charter_root / ".claude" / "autonomous-hub-charter.md"
    text = HUB_DOCS["autonomous-hub-charter.md"] + "\n⭐ a starred clause ⛔ and a banned one ⚠\n"
    doc.write_text(text, encoding="utf-8")

    win_load, _ = _load_doc_under(_WindowsFlavourPath, charter_root)
    detail = win_load(_WindowsFlavourPath(doc))
    assert detail["chars"] == len(text), (
        "the document was decoded as cp1252, not utf-8: %d chars read for a %d-char file "
        "(each ⭐/⛔/⚠ became its three raw utf-8 bytes)" % (detail["chars"], len(text))
    )


# ---------------------------------------------------------------------------
# 3d. ⛔⛔ THE TARBALL MUST NOT CARRY APPLE-DOUBLE SIDECARS INTO `.git/`
#
# 54 of them reached the far side. `refs/heads/._main` is a MALFORMED REF
# (`badRefName`), `.git/objects/XX/._<sha>` are BAD SHA1 FILES, and `._pack-*.idx`
# SHADOWS the real pack index on every git call. ⚠ `git worktree add` is a hub's
# literal first move (agent profile §0.2) and `git bundle create` is how its code comes
# home — both write refs and read the object store, so this is the hub's critical path.
#
# ⭐ TWO causes, and both are closed at CREATION — ⛔ neither by deleting anything on
#    the far side:
#   1. sidecars already on disk must never be archived  (test 1 below);
#   2. the resource-fork / xattr payload a sidecar is MATERIALISED from must never
#      travel either — a plain bsdtar stores it in a pax header, and a copy onto a
#      filesystem with no xattr support writes it back out as `._<name>` (test 2).
# ---------------------------------------------------------------------------

#: The exact shapes observed on the far side.
APPLE_DOUBLE_HEADER = b"\x00\x05\x16\x07" + b"\x00" * 22


def _plant_apple_doubles(bundle: Path) -> list[Path]:
    """Reproduce the far side's measured condition inside a freshly built bundle."""
    planted = []
    for rel in (".git/refs/heads/._main",
                ".git/objects/pack/._pack-0123456789abcdef0123456789abcdef01234567.idx"):
        p = bundle / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(APPLE_DOUBLE_HEADER)
        planted.append(p)
    loose = sorted(q for q in (bundle / ".git" / "objects").glob("??/*") if q.is_file())
    assert loose, "the bundle repo has no loose object to shadow"
    p = loose[0].with_name("._" + loose[0].name)
    p.write_bytes(APPLE_DOUBLE_HEADER)
    planted.append(p)
    return planted


def _extract_fresh(tarball: Path, into: Path) -> Path:
    """Extract with Python's ``tarfile`` — ⛔ **NOT** with macOS ``tar``.

    ⭐⭐ THIS IS WHY THE DEFECT SURVIVED EVERY CHECK RUN ON THIS MACHINE. macOS ``tar``
    writes one ``._<name>`` AppleDouble member per entry **and hides them again** on
    ``tar -t`` and ``tar -x``, because it knows they are its own metadata. GNU tar — on
    the far side — has no such knowledge and writes every one of them out as an ordinary
    file. A verification performed with macOS ``tar`` therefore sees a clean archive
    while the far side gets 54 broken refs and objects.

    ⭐ Measured on one bundle, same tarball, two readers:
    ``tar -tzf`` reported 1193 members and 0 AppleDouble; ``tarfile`` reported 2386 and
    1193. ``tarfile`` is the honest reader here, so it is the one this test uses.
    """
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as t:
        t.extractall(into)
    roots = [d for d in into.iterdir() if d.is_dir()]
    assert len(roots) == 1, roots
    return roots[0]


@needs_bundle
def test_the_tarball_never_carries_apple_double_into_git(tmp_path: Path, claude_root: Path):
    """⭐ tar -> extract to a FRESH dir -> `git fsck` passes and the `._` count is 0."""
    b = build_bundle(tmp_path / "b", claude_root)
    _plant_apple_doubles(b)

    # --- the failure being closed, demonstrated on the source ------------------
    broken = subprocess.run(["git", "-C", str(b), "fsck"], capture_output=True, text=True)
    assert broken.returncode != 0, "the plant did not reproduce the far side's condition"
    assert "badRefName" in broken.stderr or "bad sha1 file" in broken.stderr, broken.stderr

    build_bundle(b, claude_root, "--tar")          # idempotent rebuild, + the tarball
    tarball = Path(f"{b}.tar.gz")
    assert tarball.is_file()

    with tarfile.open(tarball) as t:
        members = t.getnames()
    assert any("/.git/" in f"/{m}" for m in members), "the tar carries no .git at all"
    smuggled = [m for m in members if Path(m).name.startswith("._")]
    assert not smuggled, f"{len(smuggled)} AppleDouble member(s) in the tar: {smuggled[:5]}"

    extracted = _extract_fresh(tarball, tmp_path / "fresh")
    on_disk = [str(p.relative_to(extracted)) for p in extracted.rglob("._*")]
    assert not on_disk, f"{len(on_disk)} AppleDouble file(s) on the far side: {on_disk[:5]}"

    fsck = subprocess.run(["git", "-C", str(extracted), "fsck"], capture_output=True, text=True)
    assert fsck.returncode == 0, f"git fsck rc={fsck.returncode}\n{fsck.stdout}\n{fsck.stderr}"


@needs_bundle
@pytest.mark.skipif(sys.platform != "darwin",
                    reason="resource forks are the macOS-only source of `._*`")
def test_the_tarball_strips_the_metadata_a_sidecar_is_made_from(tmp_path: Path,
                                                                claude_root: Path):
    """⛔ Deleting `._*` on the far side is not the fix; neither is excluding them alone.

    A plain macOS tar keeps the resource fork in a pax header, so it *travels* — and the
    next copy onto a filesystem without xattr support writes it back out as `._<name>`.
    ⭐ The archive must carry no such payload at all.
    """
    b = build_bundle(tmp_path / "b", claude_root)
    victim = b / ".claude" / "AGENT_PROTOCOL.md"   # copied only when its sha differs
    with open(f"{victim}/..namedfork/rsrc", "wb") as fh:
        fh.write(b"RSRCFORKDATA")
    assert "com.apple.ResourceFork" in subprocess.run(
        ["xattr", str(victim)], capture_output=True, text=True).stdout

    build_bundle(b, claude_root, "--tar")
    extracted = _extract_fresh(Path(f"{b}.tar.gz"), tmp_path / "fresh")
    got = subprocess.run(["xattr", str(extracted / ".claude" / "AGENT_PROTOCOL.md")],
                         capture_output=True, text=True).stdout
    assert "com.apple.ResourceFork" not in got, (
        "the resource fork survived the archive — a copy onto exFAT/SMB on the far side "
        "will write it out as `._AGENT_PROTOCOL.md`:\n" + got
    )


# ---------------------------------------------------------------------------
# 3e. ⭐⭐ THE PER-HUB ARTEFACT REQUIREMENT, AND THE SUPPLEMENT
#
# ⛔ `hub-refusal`'s charter calls itself "the cheapest unclaimed target" because "it is
# scored off artefacts that already exist" — and NEITHER of the two it names was in the
# bundle. On the far side it was therefore the most expensive target, not the cheapest.
# ⭐ Hardcoding those two paths would fix one bundle and leave the CLASS open, so the
# CHARTER IS THE DECLARATION: the bundle is built from the hub's own promise, and a
# promise the bundle cannot keep is a REFUSAL that names the path.
# ---------------------------------------------------------------------------

#: A charter shaped like a real one: a needed artefact, a SUPERSEDED one named on the
#: SAME LINE, the hub's own write path, and a branch name that is not a path at all.
REQ_HUB = "hub-requirement-probe"


#: ⭐ An artefact NO other mechanism ships — not the launcher table, not the query-set
#: registry. It exists only because this charter names it, which is the whole point:
#: `hub-refusal`'s `gen-read-cell-arm/runs/` was exactly this shape, and exactly absent.
REQ_ONLY_DIR = ".claude/outputs/gen-read-cell-arm/runs"


def _charter(dataset: str, *, missing: bool = False) -> str:
    needed = "gen-nonexistent-slug/refusal_scores.json" if missing else REQ_ONLY_DIR + "/"
    return (
        f"# HUB CHARTER — `hub/{REQ_HUB}`\n"
        f"- **Worktree:** `../CHLU-{REQ_HUB}` · **Branch:** `hub/{REQ_HUB}`\n"
        f"- **Report:** `.claude/outputs/{REQ_HUB}.md` · running log "
        f"`.claude/outputs/{REQ_HUB}/LOG.md`\n"
        f"- **Boot:** `.claude/AGENT_PROTOCOL.md` → `.claude/handover_context.md` §7\n"
        "## ⭐ WHY THIS IS THE CHEAPEST UNCLAIMED TARGET\n"
        f"⭐ It is scored off artefacts that already exist, under `{needed}`.\n"
        f"⚠ The rivals are already re-scored (`.claude/data/gen/{dataset}/a7_redundant.json`"
        ", complete). "
        "⛔ **`gen-arms-a7-wiring/refusal_scores.json` is SUPERSEDED — never quote it.**\n"
        "⭐ The call site is `exp_gen_baselines.py:1562`; the code is in "
        "`scripts/csf3/stage_artefacts.sh`.\n"
    )


@pytest.fixture()
def req_claude_root(claude_root: Path) -> Path:
    """``claude_root`` + the charter-only artefact set, in the shapes those take:
    a ``.json`` report and the ``.npz`` traces that dominate a real read-cell run."""
    import numpy as np

    runs = claude_root / "outputs" / "gen-read-cell-arm" / "runs"
    (runs / "artefacts").mkdir(parents=True, exist_ok=True)
    _write_json(runs / "launch_diagnostic.json", {"schema": {"version": 1}, "still_descending": 0.72})
    np.savez(runs / "artefacts" / "energy_trace.npz",
             energy=np.arange(6, dtype=np.float32), settled=np.zeros((2, 3), dtype=np.float32))
    (claude_root / "tasks" / f"{REQ_HUB}.md").write_text(_charter(DATASET), encoding="utf-8")
    return claude_root


def _requirements(claude: Path, hub: str = REQ_HUB, *extra: str):
    p = run_bundle("--src", str(REPO), "--hub", hub, "--print-requirements", *extra,
                   claude=claude)
    req = [ln.split()[2] for ln in p.stdout.splitlines() if "⭐ REQUIRED" in ln]
    exc = [ln.split()[2] for ln in p.stdout.splitlines() if "⛔ EXCLUDED" in ln]
    return req, exc, p


@needs_bundle
def test_the_charter_is_the_declaration(req_claude_root: Path):
    """⭐ Every artefact the charter names is a requirement — ⛔ and nothing else is."""
    req, exc, p = _requirements(req_claude_root)

    # the artefacts, including the one named WITHOUT a `.claude/` prefix
    assert REQ_ONLY_DIR in req, p.stdout
    assert f".claude/data/gen/{DATASET}/a7_redundant.json" in req, p.stdout
    # the documents the Boot line names
    assert ".claude/AGENT_PROTOCOL.md" in req and ".claude/handover_context.md" in req

    # ⛔ the hub's OWN write path is what it PRODUCES — requiring it would refuse
    #    every bundle, since it does not exist yet (agent profile §0.3)
    assert not [r for r in req if REQ_HUB in r], req
    # ⛔ a branch name and a call site are not artefacts
    assert not [r for r in req if "hub/" in r or "stage_artefacts" in r], req


@needs_bundle
def test_a_superseded_artefact_is_excluded_even_from_the_same_line(req_claude_root: Path):
    """⭐⭐ The clause, not the line, is the unit.

    ``gen-val-split-fix/refusal_scores.json`` (ship it) and
    ``gen-arms-a7-wiring/refusal_scores.json`` (⛔ never quote it) sit on ONE line of the
    real charter, separated only by a ``⛔``. A line-level rule would ship both or
    neither.
    """
    req, exc, p = _requirements(req_claude_root)
    assert ".claude/outputs/gen-arms-a7-wiring/refusal_scores.json" in exc, p.stdout
    assert ".claude/outputs/gen-arms-a7-wiring/refusal_scores.json" not in req
    # …and the one named a clause earlier, on the same line, IS required
    assert f".claude/data/gen/{DATASET}/a7_redundant.json" in req


@needs_bundle
def test_the_bundler_refuses_a_hub_whose_requirement_is_unmet(tmp_path: Path,
                                                              claude_root: Path):
    """⛔⛔ ACCEPTANCE: silent omission is the defect; a loud refusal is the fix."""
    (claude_root / "tasks" / f"{REQ_HUB}.md").write_text(
        _charter(DATASET, missing=True), encoding="utf-8")
    out = tmp_path / "b"
    p = run_bundle("--src", str(REPO), "--pin", _git("rev-parse", "HEAD"),
                   "--out", str(out), "--hub", REQ_HUB, "--datasets", DATASET,
                   "--pretexts", PRETEXT, "--seeds", str(SEED),
                   claude=claude_root, expect=2)
    assert "gen-nonexistent-slug/refusal_scores.json" in p.stderr, p.stderr
    assert "REFUSING" in p.stderr
    assert not out.exists() or not any(out.iterdir()), "it refused but still wrote a bundle"


@needs_bundle
def test_the_required_artefacts_reach_the_bundle_and_are_opened(tmp_path: Path,
                                                               req_claude_root: Path):
    """⭐ End to end: declared -> shipped -> manifest -> OPENED by the real loader."""
    b = build_bundle(tmp_path / "b", req_claude_root, "--hub", REQ_HUB)
    m = json.loads((b / "BUNDLE_MANIFEST.json").read_text())
    paths = {f["path"]: f for f in m["files"]}

    a7 = f".claude/data/gen/{DATASET}/a7_redundant.json"
    assert a7 in paths and (b / a7).is_file()
    # ⭐ the rows NOTHING else would have shipped — the defect, closed
    trace = f"{REQ_ONLY_DIR}/artefacts/energy_trace.npz"
    assert trace in paths and (b / trace).is_file(), sorted(paths)[:5]
    assert paths[trace]["open_with"] == "npz"
    assert f"{REQ_ONLY_DIR}/launch_diagnostic.json" in paths

    reqs = m["hub_requirements"][REQ_HUB]
    assert {r["path"] for r in reqs["required"]} >= {REQ_ONLY_DIR, a7}
    assert [x["path"] for x in reqs["excluded"]] == [
        ".claude/outputs/gen-arms-a7-wiring/refusal_scores.json"]

    # ⚠ The superseded file still SHIPS — the launcher table opens it as
    #    `check_preconditions`' own precondition — but it travels FLAGGED, in the
    #    manifest, where the far side is already looking. ⛔ Not silently.
    sup = paths.get(".claude/outputs/gen-arms-a7-wiring/refusal_scores.json")
    if sup is not None:
        assert sup["superseded_for_hub"] == REQ_HUB
        assert "never quote it as the rival row" in sup["superseded_note"]

    run_verify(b, expect=0, env_extra={"HUB_SLUG": REQ_HUB})
    r = ready(b)
    assert r["status"] == "ready" and r["loaders_all_opened"]


# --- the supplement --------------------------------------------------------


@needs_bundle
def test_the_supplement_tops_up_a_workspace_and_never_replaces_a_byte(
        tmp_path: Path, claude_root: Path, req_claude_root: Path):
    """⭐⭐ A hub may ALREADY BE RUNNING in the workspace being topped up.

    ⛔ So: no 2.1 GB rebuild, no replacement of anything already there. A file that is
    absent is written; one that is byte-identical is a no-op; one that DIFFERS is kept
    and ours is written beside it under a stamped name.
    """
    # a workspace built BEFORE the charter declared its artefacts — i.e. the far side
    (req_claude_root / "tasks" / f"{REQ_HUB}.md").write_text(
        _charter(DATASET), encoding="utf-8")
    b = build_bundle(tmp_path / "b", req_claude_root, "--hub", REQ_HUB,
                     "--no-charter-requires")
    trace = Path(f"{REQ_ONLY_DIR}/artefacts/energy_trace.npz")
    assert not (b / trace).exists(), "the pre-fix workspace should lack the artefact"

    # ⚠ and a hub is at work in here: one shipped file already differs
    hostage = b / ".claude" / "AGENT_PROTOCOL.md"
    hostage.write_text("# the hub edited this\n", encoding="utf-8")

    tar = tmp_path / "supp.tar.gz"
    run_bundle("--src", str(REPO), "--supplement", "--hub", REQ_HUB,
               "--supplement-out", str(tar), claude=req_claude_root)
    assert tar.is_file()
    with tarfile.open(tar) as t:
        names = set(t.getnames())
    assert "SUPPLEMENT_MANIFEST.json" in names
    assert str(trace.relative_to(".claude")) in names, sorted(names)[:8]

    p = run_bundle("--src", str(REPO), "--supplement-apply", str(tar), "--dest", str(b),
                   claude=req_claude_root)

    # ⭐ the missing artefact arrived …
    assert (b / trace).is_file(), p.stdout
    # ⛔ … and the hub's own edit was NOT touched
    assert hostage.read_text(encoding="utf-8") == "# the hub edited this\n"
    assert "NOT overwriting" in p.stdout, p.stdout
    beside = list(hostage.parent.glob("AGENT_PROTOCOL.md.supplement-*"))
    assert beside, list(hostage.parent.iterdir())
    assert "CHLU Spoke-Agent Protocol" in beside[0].read_text(encoding="utf-8")

    # ⭐ and the verifier now OPENS the topped-up rows too — ⛔ a supplement nothing
    #    ever opens is the same hole in a new place.
    sm = json.loads((b / "SUPPLEMENT_MANIFEST.json").read_text())
    assert sm["schema"] == "chlu-hub-workspace-supplement/1"
    assert sm["member_root"] == ".claude"
    hostage.write_text(HUB_DOCS["AGENT_PROTOCOL.md"], encoding="utf-8")   # restore
    run_verify(b, expect=0, env_extra={"HUB_SLUG": REQ_HUB})
    r = ready(b)
    assert r["supplements"] and r["supplements"][0]["n_files"] >= 1
    opened = {f["path"] for f in r["files"] if f["loaded"]}
    assert str(trace) in opened, "the supplement's own rows must be OPENED too"


@needs_bundle
def test_the_supplement_is_idempotent(tmp_path: Path, req_claude_root: Path):
    """⭐ Applying it twice must change nothing the second time — ⛔ no `.supplement-*`
    litter accumulating inside a running hub's workspace."""
    b = build_bundle(tmp_path / "b", req_claude_root, "--hub", REQ_HUB)
    tar = tmp_path / "supp.tar.gz"
    run_bundle("--src", str(REPO), "--supplement", "--hub", REQ_HUB,
               "--supplement-out", str(tar), claude=req_claude_root)
    run_bundle("--src", str(REPO), "--supplement-apply", str(tar), "--dest", str(b),
               claude=req_claude_root)
    p = run_bundle("--src", str(REPO), "--supplement-apply", str(tar), "--dest", str(b),
                   claude=req_claude_root)
    assert "NOT overwriting" not in p.stdout, p.stdout
    assert "0 new" in p.stdout and "0 differing" in p.stdout, p.stdout


# ---------------------------------------------------------------------------
# 4. The verifier PASSES on a good bundle — and FAILS four ways
# ---------------------------------------------------------------------------


@needs_bundle
def test_verify_opens_every_file_through_its_real_loader(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    run_verify(b, expect=0)
    r = ready(b)
    assert r["status"] == "ready"
    assert r["checksum_match"] and r["loaders_all_opened"] and r["import_under_root"]
    assert r["hub_output_path_writable"]
    by = {f["path"]: f for f in r["files"]}
    # a real parse happened, not a stat()
    plan = by[f".claude/outputs/gen-store-grouping/runs/{DATASET}_{PRETEXT}/store_plan.json"]
    assert plan["detail"]["K"] == 2 and plan["detail"]["n_modes"] == 3
    cache = by[f".claude/outputs/gen-feature-bridge/runs/{DATASET}_{PRETEXT}/seed_{SEED}/feature_cache.json"]
    assert cache["detail"]["scope"] == "used"
    assert by[f".claude/data/gen/{DATASET}/query_sets.json"]["detail"]["sets"]["a7_redundant"] == 1


@needs_bundle
def test_verify_fails_on_a_file_that_never_arrived(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    victim = b / ".claude" / "outputs" / "gen-external-classifier" / "classifier_gate.json"
    victim.rename(victim.with_suffix(".json.moved-aside"))  # ⛔ §0.1: moved, never deleted
    p = run_verify(b)
    assert p.returncode != 0
    r = ready(b)
    assert r["status"] == "NOT-READY" and r["n_missing"] == 1
    assert any("MISSING" in (f["error"] or "") for f in r["files"])


@needs_bundle
def test_verify_fails_on_a_byte_perfect_file_its_loader_rejects(tmp_path: Path, claude_root: Path):
    """⭐⭐ The failure a checksum alone cannot see.

    The file is re-written to something *valid JSON* and the manifest's sha256 is
    updated to match, so the checksum leg passes cleanly. ⛔ ``load_store_plan``
    still rejects it — and that is a dead run, not a pass.
    """
    import hashlib

    b = build_bundle(tmp_path / "b", claude_root)
    rel = f".claude/outputs/gen-store-grouping/runs/{DATASET}_{PRETEXT}/store_plan.json"
    victim = b / rel
    victim.write_text(json.dumps({"schema": "chlu-gen-store-plan/1", "K": 2}) + "\n")

    mf_path = b / "BUNDLE_MANIFEST.json"
    mf = json.loads(mf_path.read_text())
    for f in mf["files"]:
        if f["path"] == rel:
            f["sha256"] = hashlib.sha256(victim.read_bytes()).hexdigest()
            f["bytes"] = victim.stat().st_size
    mf_path.write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")

    p = run_verify(b)
    assert p.returncode != 0
    r = ready(b)
    assert r["status"] == "NOT-READY"
    assert r["checksum_match"] is True, "the checksum leg must still PASS — that is the point"
    assert r["n_checksum_mismatch"] == 0 and r["n_load_failed"] >= 1
    bad = next(f for f in r["files"] if f["path"] == rel)
    assert bad["checksum_match"] and not bad["loaded"] and bad["load_error"]


@needs_bundle
def test_verify_fails_on_a_document_that_lost_its_anchor(tmp_path: Path, claude_root: Path):
    """⭐ A charter truncated above the pin checksums fine and pins nothing."""
    import hashlib

    b = build_bundle(tmp_path / "b", claude_root)
    rel = ".claude/autonomous-hub-charter.md"
    victim = b / rel
    victim.write_text("# AUTONOMOUS HUB CHARTER\n\n(the rest was lost in transit)\n")
    mf_path = b / "BUNDLE_MANIFEST.json"
    mf = json.loads(mf_path.read_text())
    for f in mf["files"]:
        if f["path"] == rel:
            f["sha256"] = hashlib.sha256(victim.read_bytes()).hexdigest()
            f["bytes"] = victim.stat().st_size
    mf_path.write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")

    p = run_verify(b)
    assert p.returncode != 0
    bad = next(f for f in ready(b)["files"] if f["path"] == rel)
    assert bad["checksum_match"] and not bad["loaded"]
    assert "ae907466" in bad["load_error"], bad["load_error"]


@needs_bundle
def test_verify_fails_when_chlu_resolves_outside_the_root(tmp_path: Path, claude_root: Path):
    """⛔ The silent way a 'portable' bundle proves nothing at all."""
    b = build_bundle(tmp_path / "b", claude_root)
    # point CHLU_ROOT at a copy that carries no code: chlu must then come from
    # elsewhere, and the verifier must say so rather than pass.
    shell = tmp_path / "shell"
    (shell / "scripts").mkdir(parents=True)
    shutil.copy2(b / "BUNDLE_MANIFEST.json", shell / "BUNDLE_MANIFEST.json")
    shutil.copy2(VERIFY_SH, shell / "scripts" / "verify_hub_workspace.sh")
    for f in json.loads((b / "BUNDLE_MANIFEST.json").read_text())["files"]:
        dst = shell / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(b / f["path"], dst)

    p = run_verify(shell, env_extra={"PYTHONPATH": str(REPO)})
    assert p.returncode != 0
    r = ready(shell)
    assert r["import_under_root"] is False
    assert "OUTSIDE $CHLU_ROOT" in r["import_check"]["error"]


# ---------------------------------------------------------------------------
# 5. ⭐ THE RETURN PATH — and it can never overwrite anything
# ---------------------------------------------------------------------------


@needs_bundle
def test_return_pack_and_ingest_round_trip(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    outdir = b / ".claude" / "outputs" / HUB_SLUG
    (outdir / "runs").mkdir(parents=True)
    (outdir / "LOG.md").write_text("- 2026-09-16 tried X, measured Y\n")
    (outdir / "runs" / "PREREG-1.md").write_text("bar: 0.30\n")
    (b / ".claude" / "outputs" / f"{HUB_SLUG}.md").write_text("# report\n")

    tar = tmp_path / "ret.tar.gz"
    run_bundle("--src", str(b), "--return-pack", HUB_SLUG, "--return-out", str(tar),
               claude=b / ".claude")
    assert tar.is_file()

    dest = tmp_path / "landing"
    run_bundle("--src", str(REPO), "--return-ingest", str(tar), "--dest", str(dest),
               claude=claude_root)
    assert (dest / HUB_SLUG / "LOG.md").read_text().startswith("- 2026-09-16")
    assert (dest / HUB_SLUG / "runs" / "PREREG-1.md").is_file()
    assert (dest / f"{HUB_SLUG}.md").read_text() == "# report\n"
    assert (dest / HUB_SLUG / "RETURN_MANIFEST.json").is_file()


@needs_bundle
def test_return_ingest_never_overwrites_an_existing_result(tmp_path: Path, claude_root: Path):
    """⚠ Two hubs, or one hub twice. ⛔ The second arrival may not eat the first."""
    b = build_bundle(tmp_path / "b", claude_root)
    outdir = b / ".claude" / "outputs" / HUB_SLUG
    outdir.mkdir(parents=True)
    (outdir / "LOG.md").write_text("second run\n")
    tar = tmp_path / "ret.tar.gz"
    run_bundle("--src", str(b), "--return-pack", HUB_SLUG, "--return-out", str(tar),
               claude=b / ".claude")

    dest = tmp_path / "landing"
    (dest / HUB_SLUG).mkdir(parents=True)
    (dest / HUB_SLUG / "LOG.md").write_text("FIRST RUN - PRECIOUS\n")

    p = run_bundle("--src", str(REPO), "--return-ingest", str(tar), "--dest", str(dest),
                   claude=claude_root)
    assert "NOT overwriting" in p.stdout
    assert (dest / HUB_SLUG / "LOG.md").read_text() == "FIRST RUN - PRECIOUS\n"
    diverted = [d for d in dest.iterdir() if d.name.startswith(f"{HUB_SLUG}.returned-")]
    assert diverted, list(dest.iterdir())
    assert (diverted[0] / "LOG.md").read_text() == "second run\n"


@needs_bundle
def test_return_ingest_refuses_a_traversing_or_smuggled_member(tmp_path: Path, claude_root: Path):
    b = build_bundle(tmp_path / "b", claude_root)
    outdir = b / ".claude" / "outputs" / HUB_SLUG
    outdir.mkdir(parents=True)
    (outdir / "LOG.md").write_text("ok\n")
    good = tmp_path / "ret.tar.gz"
    run_bundle("--src", str(b), "--return-pack", HUB_SLUG, "--return-out", str(good),
               claude=b / ".claude")

    # (a) a member the manifest does not declare
    work = tmp_path / "work"
    work.mkdir()
    with tarfile.open(good) as t:
        t.extractall(work)
    (work / HUB_SLUG / "smuggled.txt").write_text("not in the manifest\n")
    bad1 = tmp_path / "bad1.tar.gz"
    with tarfile.open(bad1, "w:gz") as t:
        for p in sorted(work.rglob("*")):
            t.add(p, arcname=str(p.relative_to(work)))
    p = run_bundle("--src", str(REPO), "--return-ingest", str(bad1),
                   "--dest", str(tmp_path / "land1"), claude=claude_root, expect=2)
    assert "NOT declared in the manifest" in p.stderr

    # (b) a declared path that reaches outside the hub's own slug
    mf = json.loads((work / "RETURN_MANIFEST.json").read_text())
    mf["files"].append({"path": "../../handover_context.md", "bytes": 1, "sha256": "0" * 64})
    (work / "RETURN_MANIFEST.json").write_text(json.dumps(mf))
    (work / HUB_SLUG / "smuggled.txt").unlink()
    bad2 = tmp_path / "bad2.tar.gz"
    with tarfile.open(bad2, "w:gz") as t:
        for p_ in sorted(work.rglob("*")):
            t.add(p_, arcname=str(p_.relative_to(work)))
    p = run_bundle("--src", str(REPO), "--return-ingest", str(bad2),
                   "--dest", str(tmp_path / "land2"), claude=claude_root, expect=2)
    assert "traversing path" in p.stderr or "outside the hub" in p.stderr


# ---------------------------------------------------------------------------
# 6. The live workspace (these are the rows that matter ON the far side)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (Path(os.environ.get("CHLU_ROOT", REPO)) / "BUNDLE_MANIFEST.json").is_file(),
    reason="not running inside an extracted bundle",
)
def test_live_bundle_manifest_is_internally_consistent():
    root = Path(os.environ.get("CHLU_ROOT", REPO))
    m = json.loads((root / "BUNDLE_MANIFEST.json").read_text())
    assert m["schema"] == "chlu-hub-workspace-bundle/1"
    assert m["n_files"] == len(m["files"])
    assert m["pin"], "a bundle with no pin cannot pin a hub"
    for f in m["files"]:
        p = root / f["path"]
        assert p.is_file(), f"{f['path']} is declared but absent"
        assert p.stat().st_size == f["bytes"], f["path"]
    for hub in m["hubs"]:
        assert (root / ".claude" / "tasks" / f"{hub}.md").is_file()
