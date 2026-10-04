"""``scripts/csf3/job_cpu_gen.sh`` — the CPU jobscript for the gen sweeps, as tests.

⭐ **Why this file exists.** The gen CPU legs were pinning a 16 GB laptop at 570 % CPU —
blocking the scored cell arm — while an association we are not using (``fatpou01``:
``GrpTRES cpu=1500``, ``MaxTRESPU cpu=84``) sits uncontended on ``multicore``. The
jobscript that moves them there is a shell script: nothing about it is type-checked and
nothing else in this suite executes it, which is exactly how a launcher "looks right and
has never actually run" (``test_gen_launcher.py``'s premise, inherited verbatim).

Four things are under test, and **none of them is science**:

1. **The two mistakes that waste the entitlement.** Submitting under ``gpu-free`` (the GPU
   association — it has NO CPU grant, so the job is rejected or scavenges) and asking for
   more than 84 cores (``MaxTRESPU cpu=84`` — over it the job does not run slowly, it
   PENDS FOREVER on ``AssocGrpCpuLimit``). Both must abort, naming the limit.
2. **⛔⛔ CHECKPOINT AND RESUME — the standing Head rule.** A sweep is many units; the unit
   is a seed. A ``kill -9`` mid-sweep must cost ONE UNIT, not the run: the completed units
   are banked in an atomically-rewritten journal and skipped on restart
   (``SKIP … (manifest exists)``). A resume under a different config is REFUSED by name.
   ⛔ An untested resume path has never worked here, so every branch below — including the
   two failure branches that keep a silent hole out of a finished table — is driven.
3. **No axis that could decide a result has a silent default**, and no artefact root is
   discovered missing *after* the scheduler has allocated the cores.
4. **One launcher, two experiments.** ``exp_gen_atom_mean`` and ``exp_gen_baselines`` are
   selected by ``EXP=``, with their args passed through — ⛔ not a second jobscript, and
   ⛔ not one hardcoded experiment.

Everything runs ``GEN_LOCAL=1`` (no module system, no venv) with ``DRY=1`` or ``SMOKE=1``
(a trivial payload). **CPU, seconds, no JAX import.**
"""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "csf3" / "job_cpu_gen.sh"

MAX_CORES = 84
CPU_ACCOUNT = "fatpou01"
GPU_ACCOUNT = "gpu-free"

#: The five roots ``env.sh`` exports from ``$CLU_REPO/artefacts``. The jobscript asserts
#: them itself (rather than leaning on env.sh) so the guard also fires under GEN_LOCAL=1.
ROOTS = {
    "CHLU_CLAUDE_ROOT": "",
    "CHLU_GEN_DATA_ROOT": "data/gen",
    "CHLU_GEN_BRIDGE_ROOT": "outputs/gen-feature-bridge/runs",
    "CHLU_GEN_STORE_RUNS": "outputs/gen-store-grouping/runs",
    "CHLU_GEN_DECLIFT_DIR": "outputs/gen-decoder-lift/winner",
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def stage_artefacts(tmp_path) -> Path:
    """A complete (empty) artefact tree: the roots exist, so §5 passes."""
    root = Path(tmp_path) / "art"
    for sub in ROOTS.values():
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def run_job(overrides, expect_ok=None, timeout=120):
    """Run the jobscript with a *clean* environment plus ``overrides``.

    A clean env matters: ``env -u X X=1`` unsets nothing, so an "unset axis" test
    written the sloppy way silently passes (``test_gen_launcher.py`` precedent).
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "GEN_LOCAL": "1",
    }
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True,
                          text=True, cwd=str(REPO), timeout=timeout)
    if expect_ok is True:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    elif expect_ok is False:
        assert proc.returncode != 0, proc.stdout + proc.stderr
    return proc


def base_env(tmp_path, **kw):
    """A valid atom_mean sweep whose artefact roots are staged."""
    env = {
        "EXP": "atom_mean",
        "DATASET": "shapes3d",
        "PRETEXT": "recon",
        "CHLU_CLAUDE_ROOT": str(stage_artefacts(tmp_path)),
        "CKPT_DIR": str(Path(tmp_path) / "ckpt"),
        "OUT": str(Path(tmp_path) / "out"),
        "AGGREGATE": "0",
    }
    env.update({k: v for k, v in kw.items()})
    return env


def smoke_env(tmp_path, **kw):
    """...with the trivial payload, so units actually complete in milliseconds."""
    return base_env(tmp_path, SMOKE="1", **kw)


def journal(env):
    return json.loads((Path(env["CKPT_DIR"]) / "journal.json").read_text())


def header(script_text):
    """The ``#SBATCH`` block only — directives after the first non-comment line are
    inert, so a test that greps the whole file proves nothing."""
    out = []
    for line in script_text.splitlines():
        if line.startswith("#SBATCH"):
            out.append(line)
    return out


HEADER = header(SCRIPT.read_text())


# ---------------------------------------------------------------------------
# 0. the script itself
# ---------------------------------------------------------------------------
def test_the_script_exists_and_is_syntactically_valid():
    assert SCRIPT.exists(), SCRIPT
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_there_is_exactly_one_cpu_launcher_for_these_two_experiments():
    """⛔ 'Do not build a second launcher.' A future CPU sweep adds an ``EXP=`` case
    here; it does not add a sibling jobscript."""
    csf3 = REPO / "scripts" / "csf3"
    owners = [p.name for p in sorted(csf3.glob("job_*.sh"))
              if "exp_gen_atom_mean" in p.read_text() or "exp_gen_baselines" in p.read_text()]
    assert owners == ["job_cpu_gen.sh"], owners


# ---------------------------------------------------------------------------
# 1. the SBATCH header: the entitlement we are actually claiming
# ---------------------------------------------------------------------------
def test_the_header_claims_the_cpu_account_and_the_multicore_partition():
    assert f"#SBATCH -A {CPU_ACCOUNT}" in "\n".join(HEADER)
    assert "#SBATCH -p multicore" in "\n".join(HEADER)


def directives(lines):
    """The ``#SBATCH`` directives with their trailing ``#`` comments stripped.

    ⚠ The comments deliberately DO name ``gpu-free`` ('never gpu-free'), so a test that
    greps the raw line proves the opposite of what it means to.
    """
    out = []
    for line in lines:
        body = line[len("#SBATCH"):]
        out.append(body.split("#", 1)[0].strip())
    return out


def test_the_header_never_claims_the_gpu_association():
    """⛔ ``gpu-free`` has no CPU grant. A jobscript that asks for it is the failure this
    spoke exists to prevent."""
    for d in directives(HEADER):
        assert GPU_ACCOUNT not in d, d
    assert [d for d in directives(HEADER) if d.startswith("-A ")] == [f"-A {CPU_ACCOUNT}"]


def test_the_header_never_asks_for_the_one_core_serial_partition():
    for d in directives(HEADER):
        assert "-p serial" not in d, d
    assert [d for d in directives(HEADER) if d.startswith("-p ")] == ["-p multicore"]


@pytest.mark.parametrize("directive", ["--requeue", "--signal=B:USR1@300",
                                       "--open-mode=append"])
def test_the_resumability_directives_are_carried_over_from_job_gpu_wells(directive):
    """``--requeue`` (let Slurm restart it) + ``--signal`` (5 min warning) +
    ``--open-mode=append`` (a resumed job appends to its own log) are what make a
    checkpointed job cheap to interrupt."""
    assert any(directive in line for line in HEADER), HEADER


def test_the_header_default_core_count_is_within_the_cap():
    c = [line for line in HEADER if line.startswith("#SBATCH -c ")]
    assert c, HEADER
    assert int(c[0].split()[2]) <= MAX_CORES


def test_the_header_carries_a_wallclock_because_sbatch_rejects_a_job_without_one():
    t = [line for line in HEADER if line.startswith("#SBATCH -t ")]
    assert t, HEADER


# ---------------------------------------------------------------------------
# 2. ⛔ THE CORE CAP.  Over 84 the job PENDS FOREVER on AssocGrpCpuLimit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cores", [85, 96, 168, 1500])
def test_more_than_84_cores_is_refused_and_the_message_names_the_limit(tmp_path, cores):
    proc = run_job(base_env(tmp_path, PLAN="1", SLURM_CPUS_PER_TASK=cores),
                   expect_ok=False)
    err = proc.stdout + proc.stderr
    assert str(MAX_CORES) in err, err
    assert "AssocGrpCpuLimit" in err, err
    assert CPU_ACCOUNT in err, err


@pytest.mark.parametrize("cores", [1, 8, 32, 84])
def test_up_to_and_including_84_cores_is_accepted(tmp_path, cores):
    """⭐ 84 is the entitlement, not a number to stay safely under: the whole point is
    84 cores instead of the laptop's 8."""
    run_job(base_env(tmp_path, PLAN="1", SLURM_CPUS_PER_TASK=cores), expect_ok=True)


@pytest.mark.parametrize("cores", ["0", "-4", "eighty", "84.5"])
def test_a_nonsensical_core_count_is_refused(tmp_path, cores):
    run_job(base_env(tmp_path, PLAN="1", CORES=cores), expect_ok=False)


def test_an_unset_core_count_falls_back_to_the_header_default(tmp_path):
    """``-c`` is a resource knob, not a result axis: unlike EXP/DATASET/PRETEXT it is
    allowed to default (to the ``#SBATCH -c`` value), and Slurm supplies it in a real
    job via ``SLURM_CPUS_PER_TASK``."""
    default = int([d for d in directives(HEADER) if d.startswith("-c ")][0].split()[1])
    proc = run_job(base_env(tmp_path, PLAN="1", CORES=""), expect_ok=True)
    assert f"cores: {default} (cap {MAX_CORES})" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# 3. ⛔ THE ACCOUNT.  A submit-line -A silently overrides the header.
# ---------------------------------------------------------------------------
def test_running_under_the_gpu_association_is_refused_by_name(tmp_path):
    proc = run_job(base_env(tmp_path, PLAN="1", SLURM_JOB_ACCOUNT=GPU_ACCOUNT),
                   expect_ok=False)
    err = proc.stdout + proc.stderr
    assert GPU_ACCOUNT in err and CPU_ACCOUNT in err, err
    assert "no cpu grant" in err.lower(), err


def test_running_under_the_cpu_account_is_accepted(tmp_path):
    run_job(base_env(tmp_path, PLAN="1", SLURM_JOB_ACCOUNT=CPU_ACCOUNT), expect_ok=True)


def test_the_one_core_serial_partition_is_refused(tmp_path):
    proc = run_job(base_env(tmp_path, PLAN="1", SLURM_JOB_PARTITION="serial"),
                   expect_ok=False)
    assert "one-core" in (proc.stdout + proc.stderr).lower()


def test_the_multicore_partition_is_accepted(tmp_path):
    run_job(base_env(tmp_path, PLAN="1", SLURM_JOB_PARTITION="multicore"), expect_ok=True)


# ---------------------------------------------------------------------------
# 4. no silently-defaulted axis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", ["EXP", "DATASET", "PRETEXT"])
def test_an_unset_axis_aborts_naming_itself(tmp_path, axis):
    proc = run_job(base_env(tmp_path, PLAN="1", **{axis: None}), expect_ok=False)
    assert axis in (proc.stdout + proc.stderr)


@pytest.mark.parametrize("axis,value", [("EXP", "exp_gen_cell"), ("DATASET", "mnist"),
                                        ("PRETEXT", "contrastive"), ("ARM", "oracle"),
                                        ("SEEDS", "0 one")])
def test_a_value_outside_the_whitelist_aborts(tmp_path, axis, value):
    run_job(base_env(tmp_path, PLAN="1", **{axis: value}), expect_ok=False)


# ---------------------------------------------------------------------------
# 5. ⭐ artefact roots asserted AT JOB START, each named
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("var", sorted(set(ROOTS) - {"CHLU_CLAUDE_ROOT"}))
def test_a_missing_artefact_root_aborts_at_job_start_naming_THAT_root(tmp_path, var):
    """⛔ Never reach the work to find it cannot open an artefact: on a queue we waited
    in, discovering a missing root an hour later is the expensive failure. And the abort
    must name the ONE that is missing — "something is missing" is not a diagnosis."""
    env = base_env(tmp_path, CHLU_CLAUDE_ROOT=str(stage_artefacts(tmp_path)))
    env[var] = str(Path(tmp_path) / "definitely-not-here" / var)
    proc = run_job(env, expect_ok=False)
    err = proc.stdout + proc.stderr
    assert "MISSING" in err, err
    assert f"{var} -> {env[var]}" in err, err
    # ⛔ the other four are present and must not be blamed
    for other in ROOTS:
        if other != var:
            assert f"    {other} ->" not in err, err


def test_a_missing_claude_root_names_all_five_because_it_is_their_parent(tmp_path):
    """``CHLU_CLAUDE_ROOT`` is the root the other four hang off (``env.sh`` derives them
    from ``$CLU_REPO/artefacts``), so losing it legitimately loses all five."""
    env = base_env(tmp_path, CHLU_CLAUDE_ROOT=str(Path(tmp_path) / "nothing-staged"))
    proc = run_job(env, expect_ok=False)
    err = proc.stdout + proc.stderr
    for var in ROOTS:
        assert f"    {var} ->" in err, err


def test_all_roots_present_passes_the_preflight(tmp_path):
    proc = run_job(base_env(tmp_path, DRY="1"), expect_ok=True)
    assert "artefact roots present" in proc.stdout


def test_the_artefact_check_can_be_opted_out_of_explicitly(tmp_path):
    proc = run_job(base_env(tmp_path, DRY="1", CHLU_CLAUDE_ROOT="/nope",
                            CHLU_SKIP_ARTEFACT_CHECK="1"), expect_ok=True)
    assert "NOT asserted" in proc.stdout


# ---------------------------------------------------------------------------
# 6. ⛔ checkpoints live on ~/scratch
# ---------------------------------------------------------------------------
def test_a_checkpoint_under_TMPDIR_is_refused(tmp_path):
    """$TMPDIR is node-local NVMe and is WIPED AT JOB END: a checkpoint there protects
    nothing, which is the exact opposite of the point."""
    d = Path(tmp_path) / "nodelocal"
    proc = run_job(base_env(tmp_path, PLAN="1", TMPDIR=str(d), CKPT_DIR=str(d / "ckpt")),
                   expect_ok=False)
    assert "WIPED AT JOB END" in (proc.stdout + proc.stderr)


def test_a_checkpoint_under_HOME_is_refused(tmp_path):
    home = os.environ.get("HOME", "/tmp")
    proc = run_job(base_env(tmp_path, PLAN="1", CKPT_DIR=f"{home}/ckpt-should-refuse"),
                   expect_ok=False)
    assert "quota" in (proc.stdout + proc.stderr)


def test_a_checkpoint_under_home_scratch_is_accepted(tmp_path):
    """~/scratch string-starts with $HOME but is a different filesystem — the $HOME
    guard must not swallow the one location that is actually correct."""
    home = os.environ.get("HOME", "/tmp")
    run_job(base_env(tmp_path, PLAN="1", CKPT_DIR=f"{home}/scratch/CHLU/ckpt/x"),
            expect_ok=True)


# ---------------------------------------------------------------------------
# 7. ⭐ JAX_PLATFORMS=cpu, set and asserted
# ---------------------------------------------------------------------------
def test_the_cpu_platform_is_pinned_not_inferred(tmp_path):
    """⚠ On a GPU-less node JAX logs ``cuInit(0) failed: Unknown CUDA error 303``, which
    is benign but looks exactly like a failure. Pinning the platform suppresses it at
    the source. ⛔ And a CPU run must never be able to report GPU numbers."""
    proc = run_job(base_env(tmp_path, DRY="1"), expect_ok=True)
    assert "JAX_PLATFORMS=cpu" in proc.stdout
    body = SCRIPT.read_text()
    assert "export JAX_PLATFORMS=cpu" in body
    assert 'jax.default_backend() != "cpu"' in body


def test_blas_thread_pools_follow_the_core_grant_not_the_node(tmp_path):
    """``multicore`` nodes have up to 168 cores; an unbound numpy/sklearn oversubscribes
    its cgroup and runs SLOWER than the grant it was given."""
    proc = run_job(base_env(tmp_path, DRY="1", SLURM_CPUS_PER_TASK="32"), expect_ok=True)
    assert "OMP_NUM_THREADS=32" in proc.stdout


# ---------------------------------------------------------------------------
# 8. ⭐ ONE launcher, two experiments, args passed through
# ---------------------------------------------------------------------------
def test_atom_mean_is_invoked_with_the_sweep_root_as_out(tmp_path):
    """``exp_gen_atom_mean`` appends ``seed_<N>/`` to ``--out`` itself
    (``run_experiment_gen_atom_mean``), so ``--out`` is the SWEEP ROOT."""
    env = base_env(tmp_path, DRY="1", SEEDS="0 1 2", BUDGETS="b1 b4")
    proc = run_job(env, expect_ok=True)
    for s in (0, 1, 2):
        assert (f"-m chlu.experiments.exp_gen_atom_mean --dataset shapes3d "
                f"--pretext recon --stage atoms --seed {s} --out {env['OUT']} "
                f"--budget b1 --budget b4") in proc.stdout, proc.stdout
    assert "exp_gen_baselines" not in proc.stdout


def test_baselines_is_invoked_with_the_per_seed_dir_as_out(tmp_path):
    """``exp_gen_baselines --stage baselines`` passes ``--out`` STRAIGHT to ``run_seed``,
    which does NOT append ``seed_<N>/`` — so this one gets the per-seed dir. Getting
    these two the same way round is the difference between a sweep and one overwritten
    directory."""
    env = base_env(tmp_path, DRY="1", EXP="baselines", DATASET="dsprites",
                   PRETEXT="cls", SEEDS="0 7", ARM="knn", QUICK="1",
                   EXTRA_ARGS="--no-plots")
    proc = run_job(env, expect_ok=True)
    for s in (0, 7):
        assert (f"-m chlu.experiments.exp_gen_baselines --dataset dsprites "
                f"--pretext cls --stage baselines --arm knn --seed {s} "
                f"--out {env['OUT']}/seed_{s} --quick --no-plots") in proc.stdout, proc.stdout


def test_the_dry_run_executes_nothing(tmp_path):
    env = base_env(tmp_path, DRY="1")
    run_job(env, expect_ok=True)
    assert not (Path(env["CKPT_DIR"]) / "journal.json").exists()
    assert not list(Path(env["OUT"]).glob("seed_*/*.json"))


def test_plan_prints_the_submit_line_with_the_account_and_a_legal_core_count(tmp_path):
    proc = run_job(base_env(tmp_path, PLAN="1", SLURM_CPUS_PER_TASK="48"), expect_ok=True)
    assert "sbatch -c 48" in proc.stdout
    assert CPU_ACCOUNT in proc.stdout
    assert "EXP=atom_mean" in proc.stdout


# ---------------------------------------------------------------------------
# 9. ⛔⛔ CHECKPOINT AND RESUME — the standing Head rule
# ---------------------------------------------------------------------------
def test_a_completed_sweep_reruns_nothing(tmp_path):
    env = smoke_env(tmp_path, SEEDS="0 1 2")
    first = run_job(env, expect_ok=True)
    assert first.stdout.count("DONE seed_") == 3
    again = run_job(env, expect_ok=True)
    assert again.stdout.count("SKIP seed_") == 3, again.stdout
    assert "RUN  seed_" not in again.stdout
    assert "ran=0 skipped=3 failed=0" in again.stdout


def test_an_interrupted_sweep_resumes_at_the_unit_boundary(tmp_path):
    """⭐ The narrow version of the kill test: units 0-1 banked, the sweep re-entered
    with a longer seed list skips them and runs only what is missing."""
    env = smoke_env(tmp_path, SEEDS="0 1")
    run_job(env, expect_ok=True)
    assert sorted(journal(env)["units_done"]) == ["seed_0", "seed_1"]

    env2 = dict(env, SEEDS="0 1 2 3")
    proc = run_job(env2, expect_ok=True)
    assert proc.stdout.count("SKIP seed_") == 2
    assert "RUN  seed_2" in proc.stdout and "RUN  seed_3" in proc.stdout
    assert "ran=2 skipped=2 failed=0" in proc.stdout


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_a_kill_mid_sweep_costs_one_unit_not_the_run(tmp_path, sig):
    """⭐⭐ THE ACCEPTANCE TEST. ⛔ Not waivable: an untested resume path has never
    worked here. Kill the sweep while unit 2 is in flight, restart with the identical
    command line, and assert it SKIPS the banked units and RESUMES — not restarts."""
    env = smoke_env(tmp_path, SEEDS="0 1 2 3 4", SMOKE_SLEEP_S="2",
                    CKPT_EVERY_SECONDS="0")
    shell_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"),
                 "GEN_LOCAL": "1"}
    shell_env.update({k: str(v) for k, v in env.items()})

    proc = subprocess.Popen(["bash", str(SCRIPT)], env=shell_env, cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        j = Path(env["CKPT_DIR"]) / "journal.json"
        if j.exists() and len(json.loads(j.read_text())["units_done"]) >= 2:
            break
        time.sleep(0.2)
    os.killpg(os.getpgid(proc.pid), sig)
    proc.wait(timeout=30)

    banked = sorted(journal(env)["units_done"])
    assert len(banked) >= 2 and len(banked) < 5, banked
    # ⛔ the in-flight unit is NOT banked: its manifest may be half-written
    for unit in banked:
        assert (Path(env["OUT"]) / unit / "atom_manifest.json").exists()

    resumed = run_job(env, expect_ok=True)
    for unit in banked:
        assert f"SKIP {unit} (manifest exists)" in resumed.stdout, resumed.stdout
    assert resumed.stdout.count("SKIP seed_") == len(banked)
    assert resumed.stdout.count("DONE seed_") == 5 - len(banked)
    assert sorted(journal(env)["units_done"]) == [f"seed_{i}" for i in range(5)]


def test_the_journal_write_is_atomic(tmp_path):
    """tmp + ``os.replace``: a kill during the write leaves the PREVIOUS journal intact,
    never a truncated one. Asserted on the code path, and by the absence of leftovers."""
    body = SCRIPT.read_text()
    assert "os.replace(tmp, p)" in body
    assert "os.fsync(fh.fileno())" in body
    env = smoke_env(tmp_path, SEEDS="0 1")
    run_job(env, expect_ok=True)
    assert not list(Path(env["CKPT_DIR"]).glob("*.tmp"))


@pytest.mark.parametrize("keep", [1, 2, 4])
def test_rotated_journal_snapshots_keep_the_last_n(tmp_path, keep):
    """⚠ Six SMOKE units finish inside one second, which is the point: whole-second
    snapshot names COLLIDE, so the snapshot is overwritten instead of rotated and the
    ring silently holds fewer than N. That flaked this test once; the names are
    nanosecond-resolution now and this parametrisation is what pins it."""
    env = smoke_env(tmp_path, SEEDS="0 1 2 3 4 5", CKPT_EVERY_SECONDS="0",
                    KEEP_LAST=str(keep))
    run_job(env, expect_ok=True)
    snaps = sorted(Path(env["CKPT_DIR"]).glob("journal.*.json"))
    assert len(snaps) == keep, snaps
    assert len({s.name for s in snaps}) == keep, snaps
    # every retained snapshot is a whole journal, not a fragment
    for snap in snaps:
        assert json.loads(snap.read_text())["schema"] == "chlu-cpu-gen-journal/1"
    # ...and the newest snapshot is the most complete one (chronological sort order)
    assert len(json.loads(snaps[-1].read_text())["units_done"]) == 6


def test_resume_can_be_turned_off(tmp_path):
    env = smoke_env(tmp_path, SEEDS="0 1")
    run_job(env, expect_ok=True)
    proc = run_job(dict(env, RESUME="0"), expect_ok=True)
    assert "SKIP seed_" not in proc.stdout
    assert proc.stdout.count("DONE seed_") == 2


# ---------------------------------------------------------------------------
# 10. ⭐ the resume REFUSES a config it was not written under
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("QUICK", "1"), ("EXP", "baselines"), ("DATASET", "dsprites"),
    ("PRETEXT", "cls"), ("EXTRA_ARGS", "--n-fit 8"), ("BUDGETS", "b1"),
])
def test_resuming_under_a_different_config_is_refused_naming_the_key(tmp_path, field, value):
    """⭐ The ``pilot-checkpoint-resume`` pattern, one layer up: a resumed sweep must be
    the SAME sweep. Two legs sharing one CKPT_DIR is how a table silently mixes them."""
    env = smoke_env(tmp_path, SEEDS="0")
    run_job(env, expect_ok=True)
    proc = run_job(dict(env, **{field: value}), expect_ok=False)
    err = proc.stdout + proc.stderr
    assert proc.returncode == 3, err
    assert "REFUSING TO RESUME" in err, err
    assert "journal=" in err and "now=" in err, err


def test_the_seed_list_is_deliberately_not_in_the_fingerprint(tmp_path):
    """⛔ Narrowing or extending the seed list is the normal way to repair a sweep, and
    the ledger is per-unit precisely so that is safe. Nor are cores/wallclock/job id:
    they cannot change a number."""
    env = smoke_env(tmp_path, SEEDS="0 1")
    run_job(env, expect_ok=True)
    run_job(dict(env, SEEDS="2", SLURM_CPUS_PER_TASK="64"), expect_ok=True)
    assert sorted(journal(env)["units_done"]) == ["seed_0", "seed_1", "seed_2"]


def test_force_new_archives_the_superseded_journal_rather_than_deleting_it(tmp_path):
    env = smoke_env(tmp_path, SEEDS="0")
    run_job(env, expect_ok=True)
    proc = run_job(dict(env, QUICK="1", FORCE_NEW="1"), expect_ok=True)
    assert "archived to" in proc.stdout
    assert list(Path(env["CKPT_DIR"]).glob("journal.json.superseded.*"))
    assert journal(dict(env, QUICK="1"))["fingerprint_fields"]["quick"] == "1"


# ---------------------------------------------------------------------------
# 11. ⛔ the two branches that keep a silent hole out of a finished table
# ---------------------------------------------------------------------------
def test_a_failed_unit_is_not_banked_and_the_sweep_still_finishes_the_others(tmp_path):
    env = smoke_env(tmp_path, SEEDS="0 1 2", SMOKE_FAIL_SEEDS="1")
    proc = run_job(env, expect_ok=False)
    assert proc.returncode == 1, proc.stdout
    assert "seed_1 FAILED" in proc.stdout
    assert "ran=2 skipped=0 failed=1" in proc.stdout
    assert sorted(journal(env)["units_done"]) == ["seed_0", "seed_2"]
    # and the resume re-runs exactly the one that failed
    fixed = run_job(dict(env, SMOKE_FAIL_SEEDS=""), expect_ok=True)
    assert "RUN  seed_1" in fixed.stdout
    assert fixed.stdout.count("SKIP seed_") == 2


def test_exit_zero_without_a_manifest_is_not_a_completed_unit(tmp_path):
    """⛔ Banking a unit that wrote nothing makes the hole PERMANENT across every future
    resume — the sweep would then never notice it is missing a seed."""
    env = smoke_env(tmp_path, SEEDS="0 1", SMOKE_EMPTY_SEEDS="0")
    proc = run_job(env, expect_ok=False)
    assert "wrote no atom_manifest.json" in proc.stdout
    assert sorted(journal(env)["units_done"]) == ["seed_1"]


def test_a_banked_unit_whose_manifest_vanished_is_rerun_not_skipped(tmp_path):
    """Scratch has a 3-month auto-cleanup and no backup. A journal entry whose artefact
    is gone is NOT a completed unit."""
    env = smoke_env(tmp_path, SEEDS="0 1")
    run_job(env, expect_ok=True)
    (Path(env["OUT"]) / "seed_0" / "atom_manifest.json").unlink()
    proc = run_job(env, expect_ok=True)
    assert "seed_0 is in the journal but its manifest is GONE" in proc.stdout
    assert "RUN  seed_0" in proc.stdout
    assert "SKIP seed_1" in proc.stdout


def test_the_aggregate_is_skipped_when_a_unit_did_not_complete(tmp_path):
    """An aggregate over a partial sweep is a table with a silent hole in it."""
    env = smoke_env(tmp_path, SEEDS="0 1", SMOKE_FAIL_SEEDS="1", AGGREGATE="1")
    proc = run_job(env, expect_ok=False)
    assert "skipping aggregate" in proc.stdout


# ---------------------------------------------------------------------------
# 12. the wallclock-kill handler tells the log how to resume
# ---------------------------------------------------------------------------
def test_the_signal_handler_prints_the_resume_command(tmp_path):
    env = smoke_env(tmp_path, SEEDS="0 1 2", SMOKE_SLEEP_S="3")
    shell_env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"),
                 "GEN_LOCAL": "1"}
    shell_env.update({k: str(v) for k, v in env.items()})
    proc = subprocess.Popen(["bash", str(SCRIPT)], env=shell_env, cwd=str(REPO),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4)
    proc.send_signal(signal.SIGUSR1)
    out, _ = proc.communicate(timeout=30)
    assert "[signal]" in out, out
    assert "RESUME WITH:" in out and "sbatch" in out, out
    assert "EXP=atom_mean" in out, out


# ---------------------------------------------------------------------------
# 13. ⭐⭐ the argv the jobscript emits, parsed by the EXPERIMENT'S OWN PARSER
# ---------------------------------------------------------------------------
# A dry run proves the jobscript prints what it meant to. It does NOT prove the
# experiment accepts it — and a launcher that emits one stale flag fails on the
# cluster, after the queue, having burned the allocation. So the argv is round-tripped
# through each module's real ``build_arg_parser``, extracted with ``ast`` so that
# nothing here imports jax (a chlu import costs ~20 min of cold start on this machine).
#
# ⚠ This is what catches an experiment CLI moving under the launcher: `exp_gen_atom_mean`
# grew --n-val/--tune-n/--allow-knn-proto-move while this spoke was in flight.

def _parser_for(module_rel):
    import argparse as _argparse
    import ast

    tree = ast.parse((REPO / module_rel).read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_arg_parser")
    ns = {"argparse": _argparse, "PRETEXTS": ("recon", "cls")}  # gen_bridge.PRETEXTS
    for node in tree.body:  # module constants the parser's `choices=` reference
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                ns[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    fn.returns = None
    for a in fn.args.args:
        a.annotation = None
    exec(compile(ast.Module([fn], []), "<parser>", "exec"), ns)
    return ns["build_arg_parser"]()


def _emitted_argv(proc_stdout):
    """Every payload/aggregate argv a DRY run printed, module name stripped."""
    out = []
    for line in proc_stdout.splitlines():
        line = line.strip()
        for marker in ("[dry]   python3 -u -m ", "[dry] aggregate: python3 -u -m "):
            if line.startswith(marker):
                parts = line[len(marker):].split()
                out.append((parts[0], parts[1:]))
    return out


@pytest.mark.parametrize("module_rel,overrides", [
    # ⭐ including the three flags exp_gen_atom_mean gained mid-spoke, via EXTRA_ARGS
    ("chlu/experiments/exp_gen_atom_mean.py",
     {"EXP": "atom_mean", "DATASET": "shapes3d", "PRETEXT": "recon", "SEEDS": "0 1 2",
      "BUDGETS": "b1 b4",
      "EXTRA_ARGS": "--n-val 4096 --tune-n 2048 --allow-knn-proto-move"}),
    ("chlu/experiments/exp_gen_baselines.py",
     {"EXP": "baselines", "DATASET": "dsprites", "PRETEXT": "cls", "SEEDS": "0 7",
      "ARM": "knn", "QUICK": "1", "EXTRA_ARGS": "--no-plots --n-fit 64"}),
])
def test_every_emitted_argv_is_accepted_by_that_experiments_real_parser(
        tmp_path, module_rel, overrides):
    proc = run_job(base_env(tmp_path, DRY="1", **overrides), expect_ok=True)
    emitted = _emitted_argv(proc.stdout)
    assert emitted, proc.stdout
    parser = _parser_for(module_rel)
    for module_name, argv in emitted:
        assert module_name.replace(".", "/") + ".py" == module_rel, module_name
        try:
            parser.parse_args(argv)
        except SystemExit:  # argparse exits on an unknown/invalid flag
            pytest.fail(f"{module_name} rejects the emitted argv: {' '.join(argv)}")


def test_the_dry_aggregate_is_never_a_malformed_command_line(tmp_path):
    """``--stage aggregate --runs`` with nothing after it is not an aggregate over zero
    runs; printing it would advertise a command that cannot run."""
    proc = run_job(base_env(tmp_path, DRY="1", EXP="baselines", SEEDS="0"),
                   expect_ok=True)
    assert "[dry] aggregate: SKIPPED" in proc.stdout, proc.stdout
    assert not proc.stdout.rstrip().endswith("--runs")
