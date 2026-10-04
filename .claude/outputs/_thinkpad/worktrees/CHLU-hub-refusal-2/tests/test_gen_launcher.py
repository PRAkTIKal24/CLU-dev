"""`gen-csf3-launcher` — the ONE generative-pivot launch script, as tests.

``scripts/csf3/job_gpu_gen.sh`` is the single entry point for every stage of
the generative pivot. The stage DAG is a chain with two leaves::

    encoder -> decoder -> store -+-> cell
                                 `-> certificate

It is a shell script, so nothing about it is type-checked and nothing about it
is exercised by the rest of the suite — which is exactly how a launcher "looks
right and has never actually run".

Two things are under test here, and neither is science:

1. **No axis that could decide a result has a silent default.** ``DATASET``,
   ``PRETEXT`` and ``STAGE`` must abort with a non-zero exit and a named
   message when unset, and reject values outside their whitelists. (``SEEDS``,
   ``ARMS`` and ``CONCURRENCY`` *may* default: the ``ARMS`` default is the
   complete arm set, which cannot bias a comparison.)
2. **The path plumbing and the manifest.** Every run lands in the fresh tree
   ``$GEN_ROOT/workspaces/<dataset>_<pretext>/<stage>/<arm>_s<seed>/`` with
   ``logs/``, ``artifacts/`` and a populated ``run_manifest.json`` — the record
   that makes every later number traceable. A run is refused outright if it
   cannot resolve a git commit, or if its target is not under a ``CHLU-gen``
   workspace.

3. **Every stage owns its own workspace subdirectory.** ``certificate`` in
   particular must never land in ``cell/``: the pre-stage workaround
   (``STAGE=cell`` + ``EXTRA_ARGS=--stage certificate``) wrote certificate
   artefacts into a different stage's provenance directory, and the regression
   test below is what keeps it from coming back.

Everything runs through ``GEN_LOCAL=1 SMOKE=1``: no module system, no venv,
no GPU preflight, a trivial payload. CPU, seconds, no JAX import.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "csf3" / "job_gpu_gen.sh"

FAKE_COMMIT = "0" * 40
ALL_ARMS = ["clu", "store_removed", "lace", "knn"]


def run_launcher(env_overrides, expect_ok=None):
    """Run the launcher with a *clean* environment plus ``env_overrides``.

    A clean env matters: ``env -u X X=1`` does not unset anything, so an
    "unset axis" test written the sloppy way silently passes.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "GEN_LOCAL": "1",
        "GEN_GIT_COMMIT": FAKE_COMMIT,
    }
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if expect_ok is True:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    elif expect_ok is False:
        assert proc.returncode != 0, proc.stdout + proc.stderr
    return proc


def smoke_env(root, **kw):
    base = {
        "GEN_ROOT": str(Path(root) / "CHLU-gen"),
        "SMOKE": "1",
        "DATASET": "shapes3d",
        "PRETEXT": "recon",
        "STAGE": "encoder",
        "GEN_TASK_ID": "0",
    }
    base.update(kw)
    return base


def test_script_exists_and_parses():
    assert SCRIPT.is_file(), SCRIPT
    assert os.access(SCRIPT, os.X_OK), "launcher must be executable"
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


# --------------------------------------------------------------------------
# 1. axes with no safe default fail LOUDLY
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["DATASET", "PRETEXT", "STAGE"])
def test_unset_axis_fails_loudly(axis):
    """An unset result-deciding axis aborts; it never falls back to a default."""
    env = {"DATASET": "shapes3d", "PRETEXT": "recon", "STAGE": "encoder", "PLAN": "1"}
    env[axis] = None
    proc = run_launcher(env, expect_ok=False)
    assert f"{axis} is unset" in proc.stderr, proc.stderr
    # and it did NOT get as far as planning anything
    assert "gen launch plan" not in proc.stdout


@pytest.mark.parametrize(
    "axis,bad",
    [("DATASET", "mnist"), ("PRETEXT", "contrastive"), ("STAGE", "readout")],
)
def test_invalid_axis_value_fails_loudly(axis, bad):
    env = {"DATASET": "shapes3d", "PRETEXT": "recon", "STAGE": "encoder", "PLAN": "1"}
    env[axis] = bad
    proc = run_launcher(env, expect_ok=False)
    assert f"{axis}='{bad}'" in proc.stderr, proc.stderr


def test_invalid_arm_and_seed_fail_loudly():
    base = {"DATASET": "dsprites", "PRETEXT": "cls", "STAGE": "store", "PLAN": "1"}
    p = run_launcher({**base, "ARMS": "clu oracle"}, expect_ok=False)
    assert "ARM 'oracle'" in p.stderr
    p = run_launcher({**base, "SEEDS": "0 one"}, expect_ok=False)
    assert "SEED 'one'" in p.stderr
    p = run_launcher({**base, "CONCURRENCY": "9"}, expect_ok=False)
    assert "4-GPU cap" in p.stderr


def test_seeds_arms_concurrency_may_default():
    """The three axes that MAY default do, and default to the complete grid."""
    proc = run_launcher(
        {"DATASET": "shapes3d", "PRETEXT": "recon", "STAGE": "encoder", "PLAN": "1"},
        expect_ok=True,
    )
    assert "arms  (4): clu store_removed lace knn" in proc.stdout
    assert "seeds (3): 0 1 2" in proc.stdout
    assert "tasks: 12   throttle: %2" in proc.stdout


def test_pretext_axis_is_carried_not_privileged():
    """recon is primary and cls is a diagnostic, but the launcher hides neither."""
    for pretext in ("recon", "cls"):
        proc = run_launcher(
            {
                "DATASET": "dsprites",
                "PRETEXT": pretext,
                "STAGE": "cell",
                "PLAN": "1",
            },
            expect_ok=True,
        )
        assert f"dsprites_{pretext}" in proc.stdout


# --------------------------------------------------------------------------
# 2. path construction
# --------------------------------------------------------------------------


def test_plan_enumerates_arms_x_seeds_grid(tmp_path):
    proc = run_launcher(
        {
            "DATASET": "dsprites",
            "PRETEXT": "recon",
            "STAGE": "store",
            "SEEDS": "0 1",
            "ARMS": "clu knn",
            "CONCURRENCY": "3",
            "PLAN": "1",
            "GEN_ROOT": str(tmp_path / "CHLU-gen"),
        },
        expect_ok=True,
    )
    assert "-a 0-3%3" in proc.stdout
    # sbatch --export splits on COMMAS only, so space-containing values are fine
    # provided the whole argument is one shell word: the printed line quotes it.
    assert '--export="ALL,DATASET=dsprites,PRETEXT=recon,STAGE=store,' in proc.stdout
    assert 'ARMS=clu knn,SEEDS=0 1,GEN_GIT_COMMIT=$GEN_GIT_COMMIT"' in proc.stdout
    root = tmp_path / "CHLU-gen" / "workspaces" / "dsprites_recon" / "store"
    expected = [
        (0, "clu", 0),
        (1, "clu", 1),
        (2, "knn", 0),
        (3, "knn", 1),
    ]
    for idx, arm, seed in expected:
        assert f"task {idx}: arm={arm} seed={seed} -> {root / f'{arm}_s{seed}'}" in (
            proc.stdout
        ), proc.stdout


def test_task_id_outside_the_grid_is_refused(tmp_path):
    proc = run_launcher(smoke_env(tmp_path, GEN_TASK_ID="12"), expect_ok=False)
    assert "outside the grid" in proc.stderr


def test_short_array_is_refused_unless_partial(tmp_path):
    """A -a range that does not cover ARMS x SEEDS silently drops cells."""
    env = smoke_env(
        tmp_path, SLURM_ARRAY_TASK_ID="0", SLURM_ARRAY_TASK_COUNT="3", GEN_TASK_ID=None
    )
    proc = run_launcher(env, expect_ok=False)
    assert "cells would be silently missing" in proc.stderr
    run_launcher({**env, "PARTIAL": "1"}, expect_ok=True)


def test_no_array_context_at_all_is_refused(tmp_path):
    proc = run_launcher(smoke_env(tmp_path, GEN_TASK_ID=None), expect_ok=False)
    assert "not an array task" in proc.stderr


def test_refuses_a_run_dir_outside_a_chlu_gen_workspace(tmp_path):
    """The gen pivot never writes into an inherited C2/C3 workspace."""
    proc = run_launcher(
        smoke_env(tmp_path, GEN_ROOT=str(tmp_path / "projects" / "csf3_b1")),
        expect_ok=False,
    )
    assert "not under a CHLU-gen workspace" in proc.stderr
    assert not (tmp_path / "projects").exists()


# --------------------------------------------------------------------------
# 3. the manifest — the deliverable that makes later numbers traceable
# --------------------------------------------------------------------------


def test_smoke_run_builds_the_tree_and_populates_the_manifest(tmp_path):
    env = smoke_env(
        tmp_path,
        GEN_TASK_ID="4",  # arm index 1 (store_removed), seed index 1
        SLURM_JOB_ID="1234567",
        SLURM_ARRAY_JOB_ID="1234560",
        SLURM_ARRAY_TASK_ID="4",
        SLURM_ARRAY_TASK_COUNT="12",
        SLURM_JOB_PARTITION="gpuA",
        CONCURRENCY="2",
    )
    run_launcher(env, expect_ok=True)

    run_dir = (
        tmp_path
        / "CHLU-gen"
        / "workspaces"
        / "shapes3d_recon"
        / "encoder"
        / "store_removed_s1"
    )
    assert (run_dir / "logs" / "stage.log").is_file()
    assert (run_dir / "artifacts" / "smoke_artifact.json").is_file()

    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["schema"] == "chlu-gen-run-manifest/1"
    assert m["status"] == "ok" and m["exit_code"] == 0
    assert m["axes"] == {
        "dataset": "shapes3d",
        "pretext": "recon",
        "stage": "encoder",
        "arm": "store_removed",
        "seed": 1,
    }
    assert m["grid"]["arms"] == ALL_ARMS
    assert m["grid"]["seeds"] == [0, 1, 2]
    assert m["grid"]["n_tasks"] == 12 and m["grid"]["array_task_id"] == 4
    assert m["git_commit"] == FAKE_COMMIT
    assert m["slurm"]["job_id"] == "1234567"
    assert m["slurm"]["array_job_id"] == "1234560"
    assert m["slurm"]["partition"] == "gpuA"
    assert m["paths"]["run_dir"] == str(run_dir)
    assert m["paths"]["artifacts"] == str(run_dir / "artifacts")
    assert m["versions"]["python"]
    assert "jax" in m["versions"]  # None only when jax is absent from the env
    assert m["started_at"] and m["finished_at"]
    # every non-default knob in effect is recorded, and nothing else
    nd = set(m["non_default_flags"])
    assert {"SMOKE=1", "GEN_LOCAL=1", f"GEN_ROOT={tmp_path / 'CHLU-gen'}"} <= nd
    assert not any(f.startswith("ARMS=") for f in nd), nd  # defaulted -> not listed


def test_non_default_arms_and_seeds_are_recorded(tmp_path):
    run_launcher(
        smoke_env(
            tmp_path, ARMS="clu knn", SEEDS="7", GEN_TASK_ID="1", CONCURRENCY="1"
        ),
        expect_ok=True,
    )
    run_dir = (
        tmp_path / "CHLU-gen" / "workspaces" / "shapes3d_recon" / "encoder" / "knn_s7"
    )
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["axes"]["arm"] == "knn" and m["axes"]["seed"] == 7
    nd = set(m["non_default_flags"])
    assert "ARMS=clu knn" in nd and "SEEDS=7" in nd and "CONCURRENCY=1" in nd


def test_run_without_a_resolvable_commit_is_refused(tmp_path):
    """No provenance, no run: the manifest is the point, not a formality."""
    env = smoke_env(tmp_path)
    env["GEN_GIT_COMMIT"] = None
    env["CLU_REPO"] = str(tmp_path / "no_git_here")  # no .git, no GIT_COMMIT stamp
    (tmp_path / "no_git_here").mkdir()
    proc = run_launcher(env, expect_ok=False)
    assert "cannot resolve the git commit" in proc.stderr
    assert not list((tmp_path / "CHLU-gen").rglob("run_manifest.json"))


def test_missing_stage_module_fails_loudly_and_names_the_module(tmp_path):
    """Until a stage module exists, a non-smoke run must abort, not no-op."""
    env = smoke_env(tmp_path, SMOKE="0", PAYLOAD_MODULE="chlu.experiments.no_such_gen")
    proc = run_launcher(env, expect_ok=False)
    assert "chlu.experiments.no_such_gen" in proc.stderr
    assert "does not exist yet" in proc.stderr
    m = json.loads(
        (
            tmp_path
            / "CHLU-gen"
            / "workspaces"
            / "shapes3d_recon"
            / "encoder"
            / "clu_s0"
            / "run_manifest.json"
        ).read_text()
    )
    assert m["status"] == "failed" and m["exit_code"] != 0


# --------------------------------------------------------------------------
# 4. stage dependency order: encoder -> decoder -> store -> cell
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage,upstream",
    [
        ("decoder", "encoder"),
        ("store", "decoder"),
        ("cell", "store"),
        # the certificate is a LEAF off the store, not a child of the cell
        ("certificate", "store"),
    ],
)
def test_stage_refuses_to_run_on_a_missing_upstream(tmp_path, stage, upstream):
    proc = run_launcher(smoke_env(tmp_path, STAGE=stage), expect_ok=False)
    assert f"upstream stage '{upstream}' has no manifest" in proc.stderr
    assert upstream in proc.stderr


def test_full_chain_runs_in_dependency_order(tmp_path):
    ws = tmp_path / "CHLU-gen" / "workspaces" / "shapes3d_recon"
    for stage in ("encoder", "decoder", "store", "cell"):
        run_launcher(smoke_env(tmp_path, STAGE=stage), expect_ok=True)
        m = json.loads((ws / stage / "clu_s0" / "run_manifest.json").read_text())
        assert m["status"] == "ok", (stage, m)
        assert m["axes"]["stage"] == stage
    # a stage whose upstream FAILED is refused too
    (ws / "encoder" / "clu_s0" / "run_manifest.json").write_text(
        json.dumps({"status": "failed"})
    )
    shutil.rmtree(ws / "decoder")
    proc = run_launcher(smoke_env(tmp_path, STAGE="decoder"), expect_ok=False)
    assert "did not finish ok" in proc.stderr


def test_encoder_has_no_upstream_requirement(tmp_path):
    proc = run_launcher(smoke_env(tmp_path, STAGE="encoder"), expect_ok=True)
    # (match on the launcher's own markers, not the bare word: pytest's
    # tmp_path itself contains "upstream" for this test.)
    assert "upstream ok:" not in proc.stdout
    assert "has no manifest" not in proc.stdout + proc.stderr


def test_slurm_streams_are_captured_into_the_run_logs(tmp_path):
    """Slurm writes -o/-e into $CLU_REPO/logs/; the run keeps its own copy.

    ``#SBATCH -o`` is resolved at *submit* time, so it cannot name the per-run
    directory (which depends on the array task id). The exit trap copies the
    streams next to the artifacts they belong to — the only cluster-only
    behaviour in the script, simulated here.
    """
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    (repo / "logs" / "clu-gen-777000_2.out").write_text("stdout stream\n")
    (repo / "logs" / "clu-gen-777000_2.err").write_text("stderr stream\n")
    run_launcher(
        smoke_env(
            tmp_path,
            CLU_REPO=str(repo),
            GEN_TASK_ID=None,
            SLURM_JOB_NAME="clu-gen",
            SLURM_ARRAY_JOB_ID="777000",
            SLURM_ARRAY_TASK_ID="2",
            SLURM_ARRAY_TASK_COUNT="12",
        ),
        expect_ok=True,
    )
    logs = (
        tmp_path
        / "CHLU-gen"
        / "workspaces"
        / "shapes3d_recon"
        / "encoder"
        / "clu_s2"
        / "logs"
    )
    assert (logs / "clu-gen-777000_2.out").read_text() == "stdout stream\n"
    assert (logs / "clu-gen-777000_2.err").read_text() == "stderr stream\n"
    assert (logs / "stage.log").is_file()


# --------------------------------------------------------------------------
# 5. the `certificate` stage — a LEAF off the store, with its own workspace
#
# Before this stage existed the only way to run the certificate on CSF3 was
# ``STAGE=cell`` + ``EXTRA_ARGS=--stage certificate``, which wrote certificate
# artefacts into ``cell/`` and corrupted a different stage's provenance. These
# tests exist to make that unrepresentable.
# --------------------------------------------------------------------------


def _chain_to_store(tmp_path, **kw):
    """Run encoder -> decoder -> store as smoke, so a leaf has an ok upstream."""
    for stage in ("encoder", "decoder", "store"):
        run_launcher(smoke_env(tmp_path, STAGE=stage, **kw), expect_ok=True)


def test_unset_stage_message_names_the_certificate_stage():
    """The 'no default' message has to enumerate every stage that exists."""
    env = {"DATASET": "shapes3d", "PRETEXT": "recon", "STAGE": None, "PLAN": "1"}
    proc = run_launcher(env, expect_ok=False)
    assert "STAGE=encoder|decoder|store|cell|certificate" in proc.stderr, proc.stderr


def test_invalid_stage_message_names_the_certificate_stage():
    env = {"DATASET": "shapes3d", "PRETEXT": "recon", "STAGE": "readout", "PLAN": "1"}
    proc = run_launcher(env, expect_ok=False)
    assert "encoder decoder store cell certificate" in proc.stderr, proc.stderr


def test_certificate_plans_its_own_paths_never_the_cells(tmp_path):
    """Every arm is scored inside one certificate job -> the array is over seeds."""
    proc = run_launcher(
        {
            "DATASET": "shapes3d",
            "PRETEXT": "recon",
            "STAGE": "certificate",
            "ARMS": "clu",
            "SEEDS": "0 1 2",
            "PLAN": "1",
            "GEN_ROOT": str(tmp_path / "CHLU-gen"),
        },
        expect_ok=True,
    )
    assert "-a 0-2%2" in proc.stdout
    assert '--export="ALL,DATASET=shapes3d,PRETEXT=recon,STAGE=certificate,' in (
        proc.stdout
    )
    root = tmp_path / "CHLU-gen" / "workspaces" / "shapes3d_recon" / "certificate"
    for idx, seed in enumerate((0, 1, 2)):
        assert f"task {idx}: arm=clu seed={seed} -> {root / f'clu_s{seed}'}" in (
            proc.stdout
        ), proc.stdout
    # ⛔ the workaround this stage replaces put every one of those under cell/
    assert "/cell/" not in proc.stdout, proc.stdout


def test_certificate_upstream_is_the_store_not_the_cell(tmp_path):
    """A missing cell must NOT block the certificate: they are siblings."""
    _chain_to_store(tmp_path)
    ws = tmp_path / "CHLU-gen" / "workspaces" / "shapes3d_recon"
    assert not (ws / "cell").exists()
    proc = run_launcher(smoke_env(tmp_path, STAGE="certificate"), expect_ok=True)
    assert f"upstream ok: {ws / 'store' / 'clu_s0' / 'run_manifest.json'}" in (
        proc.stdout
    ), proc.stdout


def test_certificate_writes_its_own_workspace_and_leaves_the_cell_alone(tmp_path):
    _chain_to_store(tmp_path)
    ws = tmp_path / "CHLU-gen" / "workspaces" / "shapes3d_recon"
    run_launcher(smoke_env(tmp_path, STAGE="cell"), expect_ok=True)
    run_launcher(smoke_env(tmp_path, STAGE="certificate"), expect_ok=True)

    cert_dir = ws / "certificate" / "clu_s0"
    cell_dir = ws / "cell" / "clu_s0"
    assert cert_dir.is_dir() and cell_dir.is_dir()
    assert cert_dir != cell_dir

    cert = json.loads((cert_dir / "run_manifest.json").read_text())
    cell = json.loads((cell_dir / "run_manifest.json").read_text())
    assert cert["status"] == "ok" and cert["axes"]["stage"] == "certificate"
    assert cell["status"] == "ok" and cell["axes"]["stage"] == "cell"
    assert cert["paths"]["run_dir"] == str(cert_dir)
    assert "/cell/" not in cert["paths"]["run_dir"]
    # the smoke artefact records the stage it was run under, per directory
    for d, stage in ((cert_dir, "certificate"), (cell_dir, "cell")):
        art = json.loads((d / "artifacts" / "smoke_artifact.json").read_text())
        assert art["stage"] == stage, (d, art)


def test_certificate_default_payload_module(tmp_path):
    """Its own module — ⛔ never exp_gen_cell with a --stage override."""
    _chain_to_store(tmp_path)
    run_launcher(smoke_env(tmp_path, STAGE="certificate"), expect_ok=True)
    m = json.loads(
        (
            tmp_path
            / "CHLU-gen"
            / "workspaces"
            / "shapes3d_recon"
            / "certificate"
            / "clu_s0"
            / "run_manifest.json"
        ).read_text()
    )
    assert m["payload"]["module"] == "chlu.experiments.exp_gen_certificate"
    assert m["payload"]["extra_args"] == ""
    assert not any(f.startswith("EXTRA_ARGS=") for f in m["non_default_flags"])


def test_certificate_payload_module_exists_on_disk():
    """Guards the typo class that would otherwise surface after GPU allocation.

    (Checked as a path, not an import: importing the module pulls in JAX, which
    is a ~20-minute cold start on the dev machine and buys nothing here. The
    launcher's own in-job ``find_spec`` preflight does the import-level check.)
    """
    assert (REPO / "chlu" / "experiments" / "exp_gen_certificate.py").is_file()


# --- and every pre-existing guard still fires for the new stage -----------


def test_certificate_stage_keeps_the_commit_guard(tmp_path):
    env = smoke_env(tmp_path, STAGE="certificate")
    env["GEN_GIT_COMMIT"] = None
    env["CLU_REPO"] = str(tmp_path / "no_git_here")
    (tmp_path / "no_git_here").mkdir()
    proc = run_launcher(env, expect_ok=False)
    assert "cannot resolve the git commit" in proc.stderr
    assert not list((tmp_path / "CHLU-gen").rglob("run_manifest.json"))


def test_certificate_stage_keeps_the_workspace_guard(tmp_path):
    proc = run_launcher(
        smoke_env(
            tmp_path,
            STAGE="certificate",
            GEN_ROOT=str(tmp_path / "projects" / "csf3_b1"),
        ),
        expect_ok=False,
    )
    assert "not under a CHLU-gen workspace" in proc.stderr
    assert not (tmp_path / "projects").exists()


def test_certificate_stage_keeps_the_short_array_guard(tmp_path):
    """3 seeds x 1 arm = 3 tasks; a 2-task array drops a seed unless declared."""
    _chain_to_store(tmp_path)
    env = smoke_env(
        tmp_path,
        STAGE="certificate",
        ARMS="clu",
        SEEDS="0 1 2",
        GEN_TASK_ID=None,
        SLURM_ARRAY_TASK_ID="0",
        SLURM_ARRAY_TASK_COUNT="2",
    )
    proc = run_launcher(env, expect_ok=False)
    assert "cells would be silently missing" in proc.stderr
    assert "ARMS x SEEDS = 3" in proc.stderr
    run_launcher({**env, "PARTIAL": "1"}, expect_ok=True)


def test_certificate_stage_keeps_the_grid_bounds_guard(tmp_path):
    proc = run_launcher(
        smoke_env(
            tmp_path, STAGE="certificate", ARMS="clu", SEEDS="0 1 2", GEN_TASK_ID="3"
        ),
        expect_ok=False,
    )
    assert "outside the grid" in proc.stderr


@pytest.mark.parametrize("axis", ["DATASET", "PRETEXT"])
def test_certificate_stage_keeps_the_unset_axis_guards(axis, tmp_path):
    env = smoke_env(tmp_path, STAGE="certificate")
    env[axis] = None
    proc = run_launcher(env, expect_ok=False)
    assert f"{axis} is unset" in proc.stderr, proc.stderr
