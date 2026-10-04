"""Tests for the CSF3 gen-artefact staging path.

⛔ **These tests never touch CSF3.** The cluster requires 2FA and no agent can
reach it; everything the transfer leg would do is reported ``NOT RUN`` in
``.claude/outputs/gen-cluster-stage-artefacts.md``. What IS testable — and what
these tests cover — is the half that decides whether a cluster run can work at
all:

* ``scripts/csf3/stage_artefacts.sh`` enumerates the artefact set, refuses an
  empty or a partial one, and writes a manifest in which **every file names the
  loader that opens it**; and
* ``scripts/csf3/verify_artefacts.sh`` re-checksums **and opens** every file,
  and **fails** on a corrupted byte *and* on a byte-perfect file its loader
  rejects. ⭐ A checker that cannot fail proves nothing, so both failures are
  asserted here, not just the happy path.

The fixture is a synthetic mirror of ``.claude/`` — real schemas, tiny payloads —
so the suite never depends on the multi-GB shipped artefacts being present.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
STAGE_SH = REPO / "scripts" / "csf3" / "stage_artefacts.sh"
VERIFY_SH = REPO / "scripts" / "csf3" / "verify_artefacts.sh"

#: The axes the fixture is built for (one seed keeps the tree small).
DATASET, PRETEXT, SEED = "shapes3d", "recon", 1
TAG = f"{DATASET}_{PRETEXT}"

#: ⛔ Mirrors ``stage_artefacts.sh``'s table. A row added there without a row
#: here (or vice versa) fails :func:`test_manifest_is_exactly_the_expected_set`.
EXPECTED_REQUIRED = [
    f"data/gen/{DATASET}/c_invalid.json",
    "outputs/gen-arms-a7-wiring/refusal_scores.json",
    f"outputs/gen-arms-a7-wiring/runs/historical_688128/seed_{SEED}/baselines_manifest.json",
    "outputs/gen-decoder-lift/winner/winner_hashes.json",
    f"outputs/gen-decoder-lift/winner/decoder_v1_s{SEED}.pkl",
    "outputs/gen-external-classifier/classifier_gate.json",
    f"outputs/gen-external-classifier/seed_{SEED}/classifier_manifest.json",
    f"outputs/gen-external-classifier/seed_{SEED}/external_classifier.pkl",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/bridge_manifest.json",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/decoder.pkl",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/encoder.pkl",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/feature_cache.json",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/feature_rows.npy",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/features.npy",
    f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/probe.pkl",
    f"outputs/gen-read-compose/runs/{TAG}/seed_{SEED}/store_s{SEED}.pkl",
    f"outputs/gen-store-grouping/runs/{TAG}/store_plan.json",
]
EXPECTED_OPTIONAL = [
    f"outputs/gen-store-grouping/runs/{TAG}/seed_{SEED}/store_manifest.json",
]


# ---------------------------------------------------------------------------
# Fixture: a synthetic `.claude/` mirror with REAL schemas
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_model(path: Path, n: int = 4) -> Path:
    """A checkpoint in ``chlu.utils.checkpoints.save_model``'s on-disk shape.

    ⭐ Written with :mod:`pickle` directly so the fixture does not need a real
    Equinox module — ``load_model`` calls ``eqx.combine`` on the two halves,
    which works on any matching pytree.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {"w": np.arange(n, dtype=np.float32)}
    static = {"w": None}
    with open(path, "wb") as fh:
        pickle.dump({"params": params, "static": static, "metadata": {}}, fh)
    return path


FACTORS = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]


def build_fixture_root(root: Path, *, n_rows: int = 8, dim: int = 4) -> Path:
    """A complete, loadable artefact set for (shapes3d, recon, seed 1)."""
    bridge = root / "outputs" / "gen-feature-bridge" / "runs" / TAG / f"seed_{SEED}"

    rows = np.arange(n_rows, dtype=np.int64)
    feats = np.zeros((n_rows, dim), dtype=np.float16)
    bridge.mkdir(parents=True, exist_ok=True)
    np.save(bridge / "feature_rows.npy", rows)
    np.save(bridge / "features.npy", feats)

    for name in ("encoder.pkl", "decoder.pkl", "probe.pkl"):
        _write_model(bridge / name)

    _write_json(bridge / "feature_cache.json", {
        "schema": "chlu-gen-feature-cache/1",
        "dataset": DATASET,
        "D": dim,
        "dtype": "float16",
        "feat_shape": [dim],
        "n_rows": n_rows,
        "scope": "used",
        "encoder_param_hash": "0" * 64,
        "rows_file": "feature_rows.npy",
        "rows_sha256": _sha256(bridge / "feature_rows.npy"),
        "features_file": "features.npy",
        "features_sha256": _sha256(bridge / "features.npy"),
        "features_bytes": (bridge / "features.npy").stat().st_size,
    })
    _write_json(bridge / "bridge_manifest.json", {
        "encoder_param_hash": "0" * 64,
        "decoder_param_hash": "1" * 64,
        "probe_param_hash": "2" * 64,
    })

    sg = root / "outputs" / "gen-store-grouping" / "runs" / TAG
    _write_json(sg / "store_plan.json", {
        "schema": "chlu-gen-store-plan/1",
        "dataset": DATASET, "pretext": PRETEXT,
        "K": 2, "n_modes": 3, "D": dim, "feat_shape": [dim],
        "labels": [0, 0, 1, 1], "well_sizes": [2, 2],
        "grouping_method": "correlation",
        "store_bytes": {"store_bytes": 524288},
    })
    _write_json(sg / f"seed_{SEED}" / "store_manifest.json", {"seed": SEED})

    _write_model(root / "outputs" / "gen-read-compose" / "runs" / TAG
                 / f"seed_{SEED}" / f"store_s{SEED}.pkl")

    _write_json(root / "data" / "gen" / DATASET / "c_invalid.json", {
        "schema": {"version": 1},
        "factors": {"names": FACTORS},
        "records": [
            {"id": "c0", "kind": "out_of_range", "violated_factor": "scale",
             "request": [{"factor_index": 3, "factor": "scale", "value": 99.0}]},
            {"id": "c1", "kind": "duplicate_factor", "violated_factor": "shape",
             "request": [{"factor_index": 4, "factor": "shape", "value": 0.0},
                         {"factor_index": 4, "factor": "shape", "value": 1.0}]},
        ],
    })

    win = root / "outputs" / "gen-decoder-lift" / "winner"
    _write_json(win / "winner_hashes.json", {
        f"seed_{SEED}": {"param_hash": "3" * 64,
                         "recipe": {"name": "v1_capacity_schedule"},
                         "encoder_param_hash_it_was_fit_against": "0" * 64},
    })
    _write_model(win / f"decoder_v1_s{SEED}.pkl")

    clf = root / "outputs" / "gen-external-classifier"
    _write_json(clf / "classifier_gate.json", {"gate": "PASS", "floor": 0.9, "n_seeds": 3})
    _write_model(clf / f"seed_{SEED}" / "external_classifier.pkl")
    _write_json(clf / f"seed_{SEED}" / "classifier_manifest.json",
                {"classifier": {"param_hash": "4" * 64, "param_count": 4}})

    a7 = root / "outputs" / "gen-arms-a7-wiring"
    _write_json(a7 / "refusal_scores.json", {"status": "complete", "not_run": []})
    _write_json(a7 / "runs" / "historical_688128" / f"seed_{SEED}" / "baselines_manifest.json",
                {"arms": {"lace": {"tuning": {"selected": {"sigma": 0.1, "form": "full",
                                                           "shrinkage": 0.0}, "budget": 36}}}})
    return root


@pytest.fixture()
def art_root(tmp_path: Path) -> Path:
    return build_fixture_root(tmp_path / "claude")


def run_stage(*args: str, root: Path, expect: int = 0) -> subprocess.CompletedProcess:
    env = dict(os.environ, CLAUDE_ROOT=str(root))
    env.pop("CSF3_USER", None)  # ⛔ nothing here may ever attempt a transfer
    p = subprocess.run(
        ["bash", str(STAGE_SH), "--datasets", DATASET, "--pretexts", PRETEXT,
         "--seeds", str(SEED), *args],
        capture_output=True, text=True, env=env,
    )
    assert p.returncode == expect, f"rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    return p


def stage_manifest(root: Path, out: Path) -> dict:
    run_stage("--manifest-only", "--manifest", str(out), root=root)
    return json.loads(out.read_text())


# ---------------------------------------------------------------------------
# 1. The scripts themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [STAGE_SH, VERIFY_SH])
def test_script_exists_and_parses(script: Path):
    assert script.is_file(), f"{script} is missing"
    assert os.access(script, os.X_OK), f"{script} is not executable"
    p = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_verify_job_is_gpu_less_and_never_serial():
    """⛔ `-p serial` is banned for this program; the verify job needs no GPU."""
    text = VERIFY_SH.read_text()
    assert "#SBATCH -p multicore" in text
    assert "#SBATCH -p serial" not in text
    assert "#SBATCH -G" not in text, "the verify job must not request a GPU"
    assert "#SBATCH -t " in text, "CSF3 rejects a job without -t"


# ---------------------------------------------------------------------------
# 2. Enumeration + manifest
# ---------------------------------------------------------------------------


def test_manifest_is_exactly_the_expected_set(art_root: Path, tmp_path: Path):
    m = stage_manifest(art_root, tmp_path / "stage_manifest.json")
    assert m["schema"] == "chlu-gen-stage-manifest/1"
    got = sorted(f["path"] for f in m["files"])
    assert got == sorted(EXPECTED_REQUIRED + EXPECTED_OPTIONAL)
    assert m["n_files"] == len(got)
    assert m["total_bytes"] == sum(f["bytes"] for f in m["files"]) > 0


def test_every_manifest_entry_names_a_loader(art_root: Path, tmp_path: Path):
    """⛔ Acceptance rule: a file with no loader is not in the manifest."""
    m = stage_manifest(art_root, tmp_path / "stage_manifest.json")
    # the dispatch table keys, read out of verify_artefacts.sh itself
    dispatch = set()
    body = VERIFY_SH.read_text().split("LOADERS = {", 1)[1].split("}", 1)[0]
    for line in body.splitlines():
        s = line.strip()
        if s.startswith('"'):
            dispatch.add(s.split('"')[1])
    assert dispatch, "could not read verify_artefacts.sh's loader dispatch table"

    for f in m["files"]:
        assert f["sha256"] and len(f["sha256"]) == 64, f
        assert f["bytes"] > 0, f
        assert f["loader"].strip(), f"{f['path']} carries no loader"
        assert "." in f["loader"], f"{f['path']}'s loader is not a call site: {f['loader']}"
        assert f["open_with"] in dispatch, (
            f"{f['path']} dispatches to {f['open_with']!r}, which verify_artefacts.sh "
            "cannot open"
        )


def test_manifest_checksums_are_the_real_ones(art_root: Path, tmp_path: Path):
    m = stage_manifest(art_root, tmp_path / "stage_manifest.json")
    for f in m["files"]:
        p = art_root / f["path"]
        assert p.stat().st_size == f["bytes"]
        assert _sha256(p) == f["sha256"]


def test_no_dataset_blob_is_ever_staged(art_root: Path, tmp_path: Path):
    """⛔ Scratch has no backup and a 3-month cleanup — datasets are re-downloaded."""
    m = stage_manifest(art_root, tmp_path / "stage_manifest.json")
    for f in m["files"]:
        assert not f["path"].endswith((".h5", ".hdf5")), f["path"]
        assert "factor_index" not in f["path"], f["path"]
        if f["path"].startswith("data/"):
            assert f["path"].endswith("c_invalid.json"), (
                f"{f['path']} is under data/ but is not the single-sourced C set"
            )


def test_manifest_is_idempotent(art_root: Path, tmp_path: Path):
    a = stage_manifest(art_root, tmp_path / "a.json")
    b = stage_manifest(art_root, tmp_path / "b.json")
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_dry_run_transfers_nothing(art_root: Path, tmp_path: Path):
    out = tmp_path / "stage_manifest.json"
    p = run_stage("--dry-run", "--manifest", str(out), root=art_root)
    assert "DRY RUN" in p.stdout
    assert "rsync" not in p.stdout.replace("would rsync", "")
    assert out.is_file()


# ---------------------------------------------------------------------------
# 3. The two refusals
# ---------------------------------------------------------------------------


def test_refuses_an_empty_file_set(tmp_path: Path):
    empty = tmp_path / "claude"
    empty.mkdir()
    p = run_stage("--dry-run", "--manifest", str(tmp_path / "m.json"),
                  root=empty, expect=4)
    assert "EMPTY" in p.stderr
    assert not (tmp_path / "m.json").exists()


def test_refuses_a_partial_file_set(art_root: Path, tmp_path: Path):
    """A missing REQUIRED artefact is named, with its loader, and nothing stages."""
    victim = art_root / f"outputs/gen-decoder-lift/winner/decoder_v1_s{SEED}.pkl"
    victim.unlink()
    p = run_stage("--dry-run", "--manifest", str(tmp_path / "m.json"),
                  root=art_root, expect=3)
    assert f"decoder_v1_s{SEED}.pkl" in p.stderr
    assert "loader:" in p.stderr
    assert not (tmp_path / "m.json").exists()


def test_an_absent_optional_artefact_is_skipped_not_fatal(art_root: Path, tmp_path: Path):
    (art_root / EXPECTED_OPTIONAL[0]).unlink()
    m = stage_manifest(art_root, tmp_path / "m.json")
    assert sorted(f["path"] for f in m["files"]) == sorted(EXPECTED_REQUIRED)


# ---------------------------------------------------------------------------
# 4. ⭐ The verify job — and, above all, that it CAN FAIL
# ---------------------------------------------------------------------------


def run_verify(root: Path, out: Path, expect: int) -> dict:
    env = dict(os.environ, VERIFY_LOCAL="1", ART_ROOT=str(root),
               STAGED_OUT=str(out), VERIFY_PYTHON=sys.executable,
               CLU_REPO=str(REPO), PYTHONPATH=str(REPO))
    p = subprocess.run(["bash", str(VERIFY_SH)], capture_output=True, text=True, env=env)
    assert p.returncode == expect, f"rc={p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    return json.loads(out.read_text())


def _staged(root: Path, tmp_path: Path, expect: int) -> dict:
    stage_manifest(root, root / "outputs/gen-cluster-stage-artefacts/stage_manifest.json")
    return run_verify(root, tmp_path / "STAGED.json", expect)


def test_verify_opens_every_artefact_through_its_real_loader(art_root: Path, tmp_path: Path):
    staged = _staged(art_root, tmp_path, expect=0)
    assert staged["schema"] == "chlu-gen-staged/1"
    assert staged["status"] == "complete"
    assert staged["checksum_match"] is True
    assert staged["loaders_all_opened"] is True
    assert staged["n_files"] == len(EXPECTED_REQUIRED) + len(EXPECTED_OPTIONAL)
    assert all(f["loaded"] for f in staged["files"])
    # ⭐ opening is the test: the loaders returned real structure, not just bytes
    by_path = {f["path"]: f for f in staged["files"]}
    cache = by_path[f"outputs/gen-feature-bridge/runs/{TAG}/seed_{SEED}/feature_cache.json"]
    assert cache["detail"]["n_rows"] == 8
    plan = by_path[f"outputs/gen-store-grouping/runs/{TAG}/store_plan.json"]
    assert plan["detail"]["K"] == 2 and plan["detail"]["n_modes"] == 3
    cset = by_path[f"data/gen/{DATASET}/c_invalid.json"]
    assert cset["detail"]["n_records"] == 2


def test_verify_fails_on_one_corrupted_byte(art_root: Path, tmp_path: Path):
    """⛔ A checker that cannot fail proves nothing."""
    stage_manifest(art_root, art_root / "outputs/gen-cluster-stage-artefacts/stage_manifest.json")
    victim = art_root / f"outputs/gen-decoder-lift/winner/decoder_v1_s{SEED}.pkl"
    blob = bytearray(victim.read_bytes())
    i = len(blob) // 2
    blob[i] ^= 0xFF          # exactly one byte
    victim.write_bytes(bytes(blob))

    staged = run_verify(art_root, tmp_path / "STAGED.json", expect=1)
    assert staged["status"] == "failed"
    assert staged["checksum_match"] is False
    assert staged["n_checksum_mismatch"] == 1
    bad = [f for f in staged["files"] if not f["checksum_match"]]
    assert len(bad) == 1 and bad[0]["path"] == victim.relative_to(art_root).as_posix()
    assert "MISMATCH" in bad[0]["error"]
    assert "MISMATCH" in bad[0]["checksum_error"]


def test_verify_fails_on_a_byte_perfect_file_its_loader_rejects(art_root: Path, tmp_path: Path):
    """⭐⭐ The failure a checksum alone can NEVER catch.

    The store plan is staged intact and its sha256 matches — but its schema is
    one the production loader refuses. A pure checksum gate would call this a
    complete stage and the cluster would discover it four days in.
    """
    plan_path = art_root / f"outputs/gen-store-grouping/runs/{TAG}/store_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["schema"] = "chlu-gen-store-plan/2"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    staged = _staged(art_root, tmp_path, expect=1)
    assert staged["checksum_match"] is True, "the bytes are perfect — that is the point"
    assert staged["status"] == "failed"
    assert staged["n_load_failed"] == 1
    bad = [f for f in staged["files"] if not f["loaded"]]
    assert len(bad) == 1 and bad[0]["path"].endswith("store_plan.json")
    assert "not a store plan" in bad[0]["error"]


def test_verify_fails_on_a_file_that_never_arrived(art_root: Path, tmp_path: Path):
    stage_manifest(art_root, art_root / "outputs/gen-cluster-stage-artefacts/stage_manifest.json")
    (art_root / f"outputs/gen-external-classifier/seed_{SEED}/external_classifier.pkl").unlink()
    staged = run_verify(art_root, tmp_path / "STAGED.json", expect=1)
    assert staged["status"] == "failed"
    assert staged["n_missing"] == 1
    assert staged["checksum_match"] is False
