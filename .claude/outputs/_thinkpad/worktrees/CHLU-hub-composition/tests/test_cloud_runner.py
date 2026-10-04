"""Tests for the CHLU cloud RUNNER -- the real entry point and the relaunch supervisor.

Companion to ``tests/test_cloud_launch.py`` (which covers quota refusals,
checkpoint semantics and image pinning). This file covers what
``gen-cloud-runner`` added on top:

* the **EXP registry** -- one launcher, parameterised, with the laptop and the
  VM reading the same table;
* the **five ``CHLU_*`` artefact roots**, asserted at job start and *naming* the
  missing one;
* the **decoder-recipe assertion**, by name **and** by hash, because
  ``GenBridgeConfig.dec_recipe`` still defaults to ``"baseline"`` -- the recipe
  that failed the gate -- and a run through it looks entirely normal in the logs;
* the **relaunch supervisor's two hard caps**, both demonstrated *refusing*, and
  its stockout back-off.

Everything here runs offline: no GCP, no GPU, no JAX, no docker. The shell
scripts are exercised through their own ``--check`` / seam hooks, so these are
tests of the shipped code rather than of a copy of it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLOUD = REPO / "scripts" / "cloud"
STARTUP = CLOUD / "startup.sh"
LAUNCH = CLOUD / "launch.sh"
SUPERVISOR = CLOUD / "run_until_done.sh"

#: The five roots, in the order scripts/csf3/stage_artefacts.sh publishes them.
ROOT_VARS = [
    "CHLU_CLAUDE_ROOT",
    "CHLU_GEN_DATA_ROOT",
    "CHLU_GEN_BRIDGE_ROOT",
    "CHLU_GEN_STORE_RUNS",
    "CHLU_GEN_DECLIFT_DIR",
]
WINNER = "v1_capacity_schedule"


def sh(script: Path, *args: str, env: dict | None = None, cwd: Path | None = None):
    e = dict(os.environ)
    # Keep the harness out of the script's way: these are read as job parameters.
    for k in ROOT_VARS + ["CHLU_DEC_RECIPE", "CHLU_EXP", "CHLU_ARTEFACTS"]:
        e.pop(k, None)
    e.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True,
        env=e, cwd=str(cwd or REPO),
    )


# ---------------------------------------------------------------------------
# 1. The EXP registry -- one launcher, parameterised
# ---------------------------------------------------------------------------

KNOWN_EXPS = [
    ("smoke", 0),
    ("cluformer", 0),
    ("gen_encoder", 1),
    ("gen_store", 1),
    ("gen_cell", 1),
    ("gen_dream", 1),
    ("gen_certificate", 1),
    ("gen_atom_mean", 1),
    ("gen_baselines", 1),
]


@pytest.mark.parametrize("exp,needs_art", KNOWN_EXPS)
def test_exp_registry_resolves(exp, needs_art):
    r = sh(STARTUP, "--check", "exp", exp)
    assert r.returncode == 0, r.stderr
    module, out_mode, art = r.stdout.split()
    assert art == str(needs_art)
    assert out_mode in ("sweep", "unit")
    if exp != "smoke":
        assert module.startswith("chlu.experiments.")
        assert (REPO / (module.replace(".", "/") + ".py")).exists(), module


def test_unknown_exp_is_refused_and_lists_the_known_ones():
    r = sh(STARTUP, "--check", "exp", "exp_that_does_not_exist")
    assert r.returncode == 6
    assert "unknown EXP" in r.stderr
    for exp, _ in KNOWN_EXPS:
        assert exp in r.stderr


def test_gen_baselines_out_is_the_unit_dir_not_the_sweep_root():
    """scripts/csf3/job_cpu_gen.sh §7: exp_gen_baselines does NOT append seed_<N>/
    to --out, exp_gen_atom_mean does. Getting this backwards silently writes every
    seed over the same manifest."""
    assert sh(STARTUP, "--check", "exp", "gen_baselines").stdout.split()[1] == "unit"
    assert sh(STARTUP, "--check", "exp", "gen_atom_mean").stdout.split()[1] == "sweep"


# ---------------------------------------------------------------------------
# 2. The five artefact roots
# ---------------------------------------------------------------------------


@pytest.fixture
def artefact_tree(tmp_path: Path) -> Path:
    """A minimal but LAYOUT-FAITHFUL artefact root (the CSF3 relative layout)."""
    root = tmp_path / "artefacts"
    for rel in ("data/gen",
                "outputs/gen-feature-bridge/runs",
                "outputs/gen-store-grouping/runs",
                "outputs/gen-decoder-lift/winner"):
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        # a staged root holds files; an empty one is a failed fetch and is
        # refused (see test_an_EMPTY_root_is_refused_too)
        (d / "staged.marker").write_text("x")
    return root


def _write_winner(root: Path, seeds=(0, 1), recipe: str = WINNER) -> None:
    w = root / "outputs/gen-decoder-lift/winner"
    hashes = {}
    for s in seeds:
        pkl = w / f"decoder_v1_s{s}.pkl"
        pkl.write_bytes(b"not-a-real-decoder-but-the-bytes-are-what-is-hashed-%d" % s)
        hashes[f"seed_{s}"] = {
            "pkl_sha256": hashlib.sha256(pkl.read_bytes()).hexdigest(),
            "param_hash": f"{s:064x}",
            "param_count": 1516227,
            "recipe": {"name": recipe},
        }
    (w / "winner_hashes.json").write_text(json.dumps(hashes, indent=2))


def test_roots_pass_when_all_five_exist(artefact_tree):
    r = sh(STARTUP, "--check", "roots", env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 0, r.stderr
    assert "all five roots present" in r.stdout


@pytest.mark.parametrize("victim", ROOT_VARS)
def test_a_missing_root_is_refused_BY_NAME(artefact_tree, victim):
    """⛔ A job must never reach the GPU to discover it cannot open an artefact --
    and 'artefacts missing' is not a fix; the variable's name and value is."""
    env = {"CHLU_CLAUDE_ROOT": str(artefact_tree), victim: "/nonexistent/chlu-root"}
    r = sh(STARTUP, "--check", "roots", env=env)
    assert r.returncode == 2
    assert victim in r.stderr
    assert "/nonexistent/chlu-root" in r.stderr
    named = [v for v in ROOT_VARS if f"    {v} ->" in r.stderr]
    if victim == "CHLU_CLAUDE_ROOT":
        # the other four DERIVE from it, so breaking it breaks all five -- and
        # the check reports every one of them, not just the first it hits.
        assert named == ROOT_VARS
    else:
        assert named == [victim]


def test_an_EMPTY_root_is_refused_too(artefact_tree, tmp_path):
    """startup.sh makes the artefact dir itself before fetching into it, so a bare
    existence test asks a question it already answered. An empty root is a failed
    fetch. (Measured on `gate1`: the four derived roots fired and CHLU_CLAUDE_ROOT
    vacuously passed.)"""
    empty = tmp_path / "empty"
    empty.mkdir()
    r = sh(STARTUP, "--check", "roots", env={"CHLU_CLAUDE_ROOT": str(empty)})
    assert r.returncode == 2
    assert "CHLU_CLAUDE_ROOT" in r.stderr
    assert "EMPTY" in r.stderr


# ---------------------------------------------------------------------------
# 2b. The dataset: NOT staged, fetched on the VM (the bridge's measured decision)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args,expected", [
    ("--dataset shapes3d --pretext recon --stage cell", "shapes3d"),
    ("--dataset=dsprites --stage cell", "dsprites"),
    ("--stage s1 --scale toy", ""),
])
def test_the_dataset_is_read_out_of_the_experiments_own_argument(args, expected):
    """Every gen module takes --dataset, so there is no second place to keep it
    in sync with."""
    got = subprocess.run(
        ["sed", "-n", r"s/.*--dataset[= ]\([^ ]*\).*/\1/p"],
        input=args, capture_output=True, text=True,
    ).stdout.strip()
    assert got == expected


def test_the_dataset_download_is_the_production_path_not_a_reimplementation():
    """3dshapes.h5 comes from a public bucket whose owner pays egress, so the VM
    downloads it and the sha256 + byte count are asserted by chlu's own loader."""
    src = STARTUP.read_text()
    assert "from chlu.data.shapes3d import download_shapes3d" in src
    assert "from chlu.data.dsprites import download_dsprites" in src
    assert "verify=True" in src
    # ⛔ no hand-rolled fetching of a DATASET anywhere in the launcher. (curl is
    # used, legitimately, for the metadata server and the NVIDIA toolkit repo --
    # so the check is for the dataset's own URLs and filenames, not for curl.)
    for banned in ("storage.googleapis.com/3d-shapes", "dsprites_ndarray", "wget ",
                   "H5_SHA256", "sha256sum"):
        assert banned not in src, banned


def test_the_dataset_is_never_staged_into_the_artefact_prefix():
    """stage_gcs.sh (the bridge's file) stages 44 files and not the dataset; the
    runner must not quietly add a 45th path that only exists on GCP."""
    src = STARTUP.read_text()
    assert "stage_gcs.sh" in src            # named as the staging owner
    assert "gcloud storage cp" in src       # artefacts come down, nothing goes up
    assert "storage cp -r" in src


def test_roots_default_to_the_csf3_relative_layout(artefact_tree):
    """The GCS prefix and the CSF3 scratch tree must be the same tree in two
    places, or a result cannot be moved between the platforms."""
    src = STARTUP.read_text()
    for rel in ("data/gen",
                "outputs/gen-feature-bridge/runs",
                "outputs/gen-store-grouping/runs",
                "outputs/gen-decoder-lift/winner"):
        assert rel in src, rel


# ---------------------------------------------------------------------------
# 3. ⛔ THE DECODER TRAP -- asserted by NAME and by HASH
# ---------------------------------------------------------------------------


def test_recipe_passes_on_a_good_tree(artefact_tree):
    _write_winner(artefact_tree)
    r = sh(STARTUP, "--check", "recipe", "0", "1",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 0, r.stderr + r.stdout
    assert "asserted by name and by hash" in r.stdout


def test_baseline_recipe_is_refused_BY_NAME(artefact_tree):
    _write_winner(artefact_tree)
    r = sh(STARTUP, "--check", "recipe", "0",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree), "CHLU_DEC_RECIPE": "baseline"})
    assert r.returncode == 5
    assert "FAILED the gate" in r.stderr
    assert "VOID" in r.stderr


def test_a_staged_decoder_declaring_the_wrong_recipe_is_refused(artefact_tree):
    """The name is checked twice: once against our own flag, once against what
    the STAGED artefact declares. A correct flag pointing at a baseline decoder
    is exactly the run that looks normal and is void."""
    _write_winner(artefact_tree, recipe="baseline")
    r = sh(STARTUP, "--check", "recipe", "0",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 5
    assert "declares recipe 'baseline'" in r.stderr


def test_one_flipped_byte_is_refused_BY_HASH(artefact_tree):
    _write_winner(artefact_tree)
    pkl = artefact_tree / "outputs/gen-decoder-lift/winner/decoder_v1_s0.pkl"
    b = bytearray(pkl.read_bytes())
    b[0] ^= 0x01
    pkl.write_bytes(bytes(b))
    r = sh(STARTUP, "--check", "recipe", "0", "1",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 5
    assert "sha256" in r.stderr and "!=" in r.stderr
    assert "seed_1" in r.stdout            # the good seed still verified


def test_a_missing_decoder_pkl_names_the_pattern(artefact_tree):
    _write_winner(artefact_tree)
    (artefact_tree / "outputs/gen-decoder-lift/winner/decoder_v1_s1.pkl").unlink()
    r = sh(STARTUP, "--check", "recipe", "1",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 5
    assert "decoder_v1_s{seed}.pkl" in r.stderr


def test_a_seed_absent_from_winner_hashes_is_refused(artefact_tree):
    _write_winner(artefact_tree, seeds=(0,))
    r = sh(STARTUP, "--check", "recipe", "0", "7",
           env={"CHLU_CLAUDE_ROOT": str(artefact_tree)})
    assert r.returncode == 5
    assert "seed_7 has no entry" in r.stderr


def test_the_winner_name_is_pinned_not_merely_not_baseline():
    """'anything but baseline' would accept v2_object_weighted, which LOST."""
    src = STARTUP.read_text()
    assert f'EXPECTED_DEC_RECIPE="{WINNER}"' in src
    assert "v2_object_weighted" not in src


# ---------------------------------------------------------------------------
# 4. launch.sh refuses before anything is created
# ---------------------------------------------------------------------------


def test_launcher_refuses_an_unknown_exp_before_the_quota_preflight():
    r = sh(LAUNCH, "--job", "t", "--exp", "nope", "--dry-run")
    assert r.returncode == 1
    assert "not in startup.sh's registry" in r.stderr
    assert "Do not write a second launcher" in r.stderr
    assert "preflight" not in r.stdout       # nothing was even asked of GCP


def test_launcher_refuses_a_gen_exp_with_no_artefact_prefix():
    r = sh(LAUNCH, "--job", "t", "--exp", "gen_cell", "--units", "0", "--dry-run")
    assert r.returncode == 1
    assert "needs the gen artefact set" in r.stderr
    assert "stage_gcs.sh" in r.stderr


def test_launcher_refuses_the_baseline_decoder():
    r = sh(LAUNCH, "--job", "t", "--exp", "gen_cell", "--artefacts", "gs://b/artefacts",
           "--dec-recipe", "baseline", "--dry-run")
    assert r.returncode == 1
    assert "VOID" in r.stderr


def test_launcher_refuses_a_non_integer_unit():
    r = sh(LAUNCH, "--job", "t", "--exp", "cluformer", "--units", "0 x", "--dry-run")
    assert r.returncode == 1
    assert "is not a non-negative integer" in r.stderr


def test_the_previous_attempts_sentinels_are_cleared_before_a_vm_is_created():
    """out/EXIT_CODE, DONE, FAILED and PREEMPTED describe ONE attempt, and the
    sync-back never deletes. Measured on `gencell2`: --wait returned on a stale
    EXIT_CODE 62 s after create, before the VM had booted, and the supervisor
    read a stale FAILED and called a fresh attempt a payload failure."""
    src = LAUNCH.read_text()
    m = re.search(r"for _stale in ([A-Z_ ]+); do", src)
    assert m, "the stale-sentinel sweep is gone"
    assert set(m.group(1).split()) == {"EXIT_CODE", "DONE", "FAILED", "PREEMPTED"}
    # it must happen BEFORE the instance is created, not after
    assert src.index("for _stale in") < src.index("gcloud compute instances create")
    # ⛔ and it must never clear the COMPLETION sentinel
    assert "result.json" not in m.group(1)
    # `|| true` under set -e: on a job's FIRST attempt none of these exist and
    # `gcloud storage rm` exits non-zero, which killed the launcher four seconds
    # in, having created nothing and printed nothing after the plan.
    body = src[src.index("for _stale in"):src.index("ARGFILE=")]
    assert "|| true" in body


def test_exp_args_travel_by_file_not_in_the_comma_separated_metadata_list():
    """gcloud splits --metadata on commas, so an experiment flag like
    `--set a=1,b=2` would be torn into two bogus metadata keys."""
    src = LAUNCH.read_text()
    assert re.search(r"--metadata-from-file=.*chlu-exp-args=", src)
    assert "chlu-exp-args=${EXP_ARGS}" not in src


def test_launcher_and_vm_share_one_registry():
    """The laptop must not hold a second copy of the EXP table."""
    src = LAUNCH.read_text()
    assert 'startup.sh" --check exp' in src
    for exp, _ in KNOWN_EXPS:
        if exp not in ("smoke",):
            assert f'{exp})' not in src, f"{exp} is hard-coded in launch.sh too"


# ---------------------------------------------------------------------------
# 5. The relaunch supervisor: two hard caps, both refusing
# ---------------------------------------------------------------------------


@pytest.fixture
def seams(tmp_path: Path):
    """Stub launcher / completion probe / VM probe, so a cap can be proved to
    refuse without spending the budget it exists to protect."""
    def _mk(name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text("#!/usr/bin/env bash\n" + body + "\n")
        p.chmod(0o755)
        return p

    return {
        # `== created chlu-...` is the line that means a VM exists and is billing;
        # the supervisor prices an attempt on its presence, not on wall clock.
        "ok": _mk("launch_ok.sh",
                  'echo "fake launch $*"; echo "== created chlu-j in us-central1-a"; exit 0'),
        "stockout": _mk("launch_stockout.sh",
                        'echo "FATAL: no zone in us-central1 had a free l4."; exit 1'),
        "quota": _mk("launch_quota.sh",
                     'echo "[quota] us-central1: REFUSING TO LAUNCH."; exit 1'),
        "nothing": _mk("launch_nothing.sh",
                       'echo "FATAL: image is pinned by TAG. Refusing."; exit 1'),
        "never_done": _mk("never.sh", "exit 1"),
        "always_done": _mk("always.sh", "exit 0"),
        "no_vm": _mk("novm.sh", 'echo ""'),
        # out/FAILED present, out/PREEMPTED absent == the VM ran and the unit
        # failed; the opposite means it was taken away mid-run.
        "obj_failed": _mk("obj_failed.sh", '[ "$2" = "out/FAILED" ]'),
        "obj_preempted": _mk("obj_preempted.sh", '[ "$2" = "out/PREEMPTED" ]'),
        "dir": tmp_path,
    }


def _run_sup(seams, *args, launch="ok", done="never_done", objects=None):
    env = {
        "CHLU_SUPERVISOR_DIR": str(seams["dir"] / "state"),
        "CHLU_NO_SLEEP": "1",
        "CHLU_LAUNCH_CMD": str(seams[launch]),
        "CHLU_DONE_PROBE": str(seams[done]),
        "CHLU_VM_PROBE": str(seams["no_vm"]),
    }
    if objects:
        env["CHLU_OBJECT_PROBE"] = str(seams[objects])
    return sh(SUPERVISOR, *args, env=env)


def test_attempt_cap_refuses_and_reports_attempts_used(seams):
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot",
                 "--max-attempts", "3", "--max-usd", "100")
    assert r.returncode == 4
    assert "attempt cap reached (3 of 3 used)" in r.stdout
    assert "attempts used: 3 of 3" in r.stdout
    state = json.loads((seams["dir"] / "state" / "j.json").read_text())
    assert state["attempts_used"] == 3 and state["status"] == "attempt_cap"


def test_spend_cap_refuses_BEFORE_the_attempt_that_would_breach_it(seams):
    """Checking spend-so-far only would let the cap be breached by exactly the
    attempt that tests it. l4 spot is $0.343/h; a 2 h ceiling is $0.686 > $0.50."""
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot",
                 "--max-attempts", "99", "--max-usd", "0.50", "--max-hours", "2")
    assert r.returncode == 5
    assert "spend cap would be exceeded" in r.stdout
    assert "attempts used: 0 of 99" in r.stdout      # nothing was launched


def test_a_raised_spend_cap_lets_the_same_attempt_through(seams):
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot",
                 "--max-attempts", "1", "--max-usd", "1.00", "--max-hours", "2")
    assert r.returncode == 4                          # stopped by the OTHER cap
    assert "attempts used: 1 of 1" in r.stdout


def test_there_is_no_unbounded_loop(seams):
    """Both caps default to a finite value; neither may be disabled by omission."""
    src = SUPERVISOR.read_text()
    assert re.search(r"MAX_ATTEMPTS=\d+", src)
    assert re.search(r"MAX_USD=[\d.]+", src)
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot")
    assert r.returncode in (4, 5)                     # it stops on its own


def test_stockout_backs_off_exponentially_and_is_charged_nothing(seams):
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", "--max-attempts", "4",
                 "--max-usd", "100", "--backoff-s", "30", "--backoff-max-s", "120",
                 launch="stockout")
    assert r.returncode == 4
    waits = [int(m) for m in re.findall(r"backing off (\d+)s", r.stdout)]
    assert waits[:4] == [30, 60, 120, 120], waits    # doubling, then capped
    state = json.loads((seams["dir"] / "state" / "j.json").read_text())
    assert state["stockouts"] == 4
    assert state["usd_estimated"] == 0.0             # a stockout creates no VM


def test_a_quota_refusal_stops_instead_of_burning_the_attempt_budget(seams):
    """Stock comes back; a quota metric of zero does not. Relaunching against it
    reprints the same refusal until the cap fires."""
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", "--max-attempts", "9",
                 "--max-usd", "100", launch="quota")
    assert r.returncode == 3
    assert "does not" in r.stdout and "change by retrying" in r.stdout
    assert "attempts used: 1 of 9" in r.stdout


def test_an_attempt_that_created_no_vm_is_charged_nothing(seams):
    """A refused launch bills nothing, so charging it would make the spend cap
    refuse for spend that never happened."""
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", "--max-attempts", "2",
                 "--max-usd", "100", launch="nothing")
    assert r.returncode == 4
    assert "created NO VM" in r.stdout
    state = json.loads((seams["dir"] / "state" / "j.json").read_text())
    assert state["usd_estimated"] == 0.0


def test_two_payload_failures_in_a_row_stop_the_loop(seams):
    """A VM that RAN and whose unit failed leaves out/FAILED and no out/PREEMPTED.
    Relaunching that buys the same traceback on a new VM. (Measured on `gencell1`:
    the image was missing h5py.)"""
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", "--max-attempts", "9",
                 "--max-usd", "100", objects="obj_failed")
    assert r.returncode == 6
    assert "failed in the payload twice" in r.stdout
    assert "attempts used: 2 of 9" in r.stdout        # one is not enough


def test_a_preempted_attempt_is_not_mistaken_for_a_payload_failure(seams):
    """out/PREEMPTED present: the VM was taken away. That is the case the whole
    supervisor exists for and must never stop the loop."""
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", "--max-attempts", "4",
                 "--max-usd", "100", objects="obj_preempted")
    assert r.returncode == 4                          # ran to the attempt cap
    assert "Pre-emption." in r.stdout
    assert "failed in the payload twice" not in r.stdout


def test_an_already_complete_job_launches_nothing(seams):
    r = _run_sup(seams, "--job", "j", "--card", "l4", "--spot", done="always_done")
    assert r.returncode == 0
    assert "nothing to do" in r.stdout


def test_a_card_with_no_price_is_refused(seams):
    """Without a rate the spend cap cannot bind, and a loop whose cost cannot be
    computed is an unbounded loop wearing a cap."""
    r = _run_sup(seams, "--job", "j", "--card", "made-up-card", "--spot")
    assert r.returncode == 1
    assert "no price for card" in r.stderr


def test_bad_caps_are_refused(seams):
    assert _run_sup(seams, "--job", "j", "--max-attempts", "0").returncode == 1
    assert _run_sup(seams, "--job", "j", "--max-usd", "0").returncode == 1
    assert _run_sup(seams, "--card", "l4").returncode == 1      # no --job


def test_supervisor_does_not_re_declare_the_launcher_flags():
    """Two copies of one flag list is how the two drift apart; unknown flags are
    passed through to launch.sh verbatim."""
    src = SUPERVISOR.read_text()
    for flag in ("--exp)", "--units)", "--artefacts)", "--exp-args)", "--zone)"):
        assert flag not in src, flag


# ---------------------------------------------------------------------------
# 6. ⛔ ONE checkpoint mechanism, not two
# ---------------------------------------------------------------------------


def _journal_source() -> str:
    """The python program startup.sh feeds to the container for journal ops."""
    src = STARTUP.read_text()
    m = re.search(r"\"\$IMAGE\" python - <<'PYJ'\n(.*?)\nPYJ\n", src, re.S)
    assert m, "the journal heredoc moved -- this test guards it, so fix the test"
    return m.group(1)


def test_the_unit_journal_is_ckpt_py_reused_not_a_second_mechanism():
    body = _journal_source()
    assert "from ckpt import Checkpointer" in body
    assert "config_fingerprint" in body and "FingerprintMismatch" in body
    # no hand-rolled persistence anywhere in the launcher
    src = STARTUP.read_text()
    for banned in ("np.savez", "os.replace(tmp, final)", "def save(", "tempfile.mkstemp"):
        assert banned not in src, f"startup.sh hand-rolls persistence: {banned}"


def test_the_journal_program_compiles():
    compile(_journal_source(), "<journal>", "exec")


def test_ckpt_py_is_untouched_by_this_spoke():
    """`gen-cloud-enablement`'s task said reuse it, it held, and it still holds."""
    r = subprocess.run(
        ["git", "diff", "--name-only", "main...HEAD", "--", "scripts/cloud/ckpt.py"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    if r.returncode != 0:
        pytest.skip("not a git checkout with a `main` to diff against")
    assert r.stdout.strip() == "", "ckpt.py was modified: " + r.stdout


def test_the_journal_fingerprint_excludes_the_unit_list():
    """Narrowing or extending the seed list is the normal way to repair a sweep,
    and the ledger is per-unit precisely so that is safe."""
    src = STARTUP.read_text()
    m = re.search(r"k = \[(.*?)\]", src)
    assert m
    keys = [k.strip().strip('"') for k in m.group(1).split(",")]
    assert "units" not in keys and "seeds" not in keys
    assert {"exp", "module", "dec_recipe", "image"} <= set(keys)


# ---------------------------------------------------------------------------
# 7. The completion sentinel
# ---------------------------------------------------------------------------


def test_the_supervisor_relaunches_on_the_absence_of_result_json():
    sup = SUPERVISOR.read_text()
    assert "out/result.json" in sup
    start = STARTUP.read_text()
    # written only when every requested unit is banked
    assert 'rec["complete"] = ' not in start
    assert 'sorted(banked) == sorted(units)' in start
    assert "no sentinel written" in start


def test_startup_restores_the_output_prefix_before_running():
    """A relaunch is a RESUME only if the previous attempt's outputs come back."""
    src = STARTUP.read_text()
    restore = [ln for ln in src.splitlines()
               if 'rsync -r "$REMOTE/$JOB/out" "$OUT"' in ln]
    assert len(restore) == 1, restore
    # additive: an attempt may hold files this one lacks
    assert "--delete" not in restore[0]
