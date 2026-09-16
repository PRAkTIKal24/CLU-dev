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
from pathlib import Path

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
            "c_invalid", "query_registry", "control_set", "npz_index", "doc", "code",
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
