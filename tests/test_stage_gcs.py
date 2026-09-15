"""Targeted tests for the GCS artefact bridge (`scripts/cloud/stage_gcs.sh`).

⭐ These drive the SHELL SCRIPT through a **stub `gcloud`** on ``PATH`` that
records every invocation, so the refusals, the plan and the object layout are
tested without touching a real bucket — and without importing jax (a cold JAX
import costs ~20 minutes on this machine and none of these assertions need it).

⛔ The artefact set itself is deliberately NOT re-derived here: the script reads
`stage_manifest.json`, so every test builds a tiny synthetic manifest in the
same schema (`chlu-gen-stage-manifest/1`) and asserts on the transport.
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
STAGE = REPO / "scripts" / "cloud" / "stage_gcs.sh"
VERIFY = REPO / "scripts" / "cloud" / "verify_gcs.sh"

MF_REL = "outputs/gen-cluster-stage-artefacts/stage_manifest.json"

# The stub records `gcloud <args>` one call per line and answers the single
# read-only call the script makes (`storage ls --json`) with an empty listing,
# i.e. "the destination is empty".
GCLOUD_STUB = """#!/usr/bin/env bash
echo "$@" >> "$GCLOUD_LOG"
if [ "$1" = "storage" ] && [ "$2" = "ls" ]; then
  cat "${GCLOUD_LS_JSON:-/dev/null}" 2>/dev/null || echo '[]'
  exit 0
fi
exit 0
"""


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def rig(tmp_path):
    """A fake `.claude` root, a manifest, and a stub gcloud on PATH."""
    root = tmp_path / "claude"
    (root / "outputs" / "gen-cluster-stage-artefacts").mkdir(parents=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "gcloud"
    stub.write_text(GCLOUD_STUB)
    stub.chmod(0o755)
    log = tmp_path / "gcloud.log"

    def write_file(rel: str, payload: bytes) -> dict:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        return {"path": rel, "bytes": len(payload), "sha256": _sha256(payload),
                "loader": "chlu.fake.loader.open_%s" % Path(rel).stem,
                "open_with": "json", "required": True}

    def write_manifest(files, **over):
        m = {"schema": "chlu-gen-stage-manifest/1",
             "generated_at": "2026-09-15T00:00:00+00:00",
             "source_root": str(root), "remote_root": "scratch/CHLU/artefacts",
             "axes": {"datasets": ["shapes3d"], "pretexts": ["recon"], "seeds": [0]},
             "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files),
             "files": files}
        m.update(over)
        (root / MF_REL).write_text(json.dumps(m, indent=2, sort_keys=True))
        return m

    def run(*args, env_extra=None):
        env = dict(os.environ)
        env["PATH"] = "%s:%s" % (bindir, env["PATH"])
        env["CLAUDE_ROOT"] = str(root)
        env["GCLOUD_LOG"] = str(log)
        env["CHLU_BUCKET"] = "test-bucket"
        env["CHLU_PROJECT"] = "test-project"
        env.update(env_extra or {})
        return subprocess.run(["bash", str(STAGE), *args], env=env,
                              capture_output=True, text=True)

    rig = type("Rig", (), {})()
    rig.root, rig.log, rig.bindir = root, log, bindir
    rig.write_file, rig.write_manifest, rig.run = write_file, write_manifest, run
    rig.calls = lambda: (log.read_text().splitlines() if log.exists() else [])
    return rig


# ---------------------------------------------------------------------------
# 1. --dry-run: a real plan, and ⛔ not one byte transferred
# ---------------------------------------------------------------------------
def test_dry_run_plans_everything_and_uploads_nothing(rig):
    files = [rig.write_file("outputs/a/x.json", b'{"k": 1}'),
             rig.write_file("data/gen/shapes3d/c_invalid.json", b'{"c": []}')]
    rig.write_manifest(files)

    r = rig.run("--dry-run")
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert "outputs/a/x.json" in r.stdout
    assert "data/gen/shapes3d/c_invalid.json" in r.stdout
    # the manifest is planned too — it must travel WITH the set
    assert MF_REL in r.stdout
    # ⛔ no `cp` anywhere: the only gcloud call a dry run may make is the listing
    assert not [c for c in rig.calls() if c.startswith("storage cp")], rig.calls()
    assert any(c.startswith("storage ls") for c in rig.calls())


def test_dry_run_reports_a_cost_line(rig):
    rig.write_manifest([rig.write_file("outputs/a/x.json", b"x" * 1000)])
    r = rig.run("--dry-run")
    assert re.search(r"COST: .*GB stored -> \$\d+\.\d+/month", r.stdout), r.stdout
    # ingress is free and the dataset is not here; both must be stated, not implied
    assert "$0.00" in r.stdout
    assert "3dshapes.h5 is NOT here" in r.stdout


# ---------------------------------------------------------------------------
# 2. the two refusals — ⭐ emptiness FIRST, then partial, naming the LOADER
# ---------------------------------------------------------------------------
def test_refuses_an_empty_set(rig):
    rig.write_manifest([])
    r = rig.run("--dry-run")
    assert r.returncode == 4, (r.returncode, r.stdout, r.stderr)
    assert "EMPTY" in r.stderr
    assert "silent no-op" in r.stderr
    assert not [c for c in rig.calls() if c.startswith("storage cp")]


def test_refuses_a_partial_set_naming_the_file_and_its_loader(rig):
    present = rig.write_file("outputs/a/x.json", b'{"k": 1}')
    absent = {"path": "outputs/gen-feature-bridge/runs/shapes3d_recon/seed_0/encoder.pkl",
              "bytes": 17, "sha256": "0" * 64, "open_with": "eqx_pickle", "required": True,
              "loader": "chlu.utils.checkpoints.load_model via exp_gen_store.fit_seed"}
    rig.write_manifest([present, absent])

    r = rig.run("--dry-run")
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    assert absent["path"] in r.stderr           # ⭐ the missing file, by name
    assert "loader:" in r.stderr                # ⭐ and the loader that opens it
    assert "chlu.utils.checkpoints.load_model" in r.stderr
    assert "partial set" in r.stderr
    assert not [c for c in rig.calls() if c.startswith("storage cp")]


def test_empty_beats_partial_when_both_apply(rig):
    """⭐ Nothing at all is almost always a wrong root; naming 44 missing files
    buries that, so emptiness is reported first — the CSF3 ordering."""
    rig.write_manifest([])
    r = rig.run("--dry-run")
    assert r.returncode == 4
    assert "REQUIRED artefact" not in r.stderr


def test_refuses_a_stale_manifest_rather_than_uploading_wrong_bytes(rig):
    f = rig.write_file("outputs/a/x.json", b'{"k": 1}')
    rig.write_manifest([f])
    (rig.root / "outputs/a/x.json").write_bytes(b'{"k": 2}')     # edited after hashing
    r = rig.run("--dry-run")
    assert r.returncode == 5, (r.returncode, r.stderr)
    assert "STALE" in r.stderr
    assert "--refresh-manifest" in r.stderr


def test_rejects_a_foreign_manifest_schema(rig):
    f = rig.write_file("outputs/a/x.json", b"{}")
    rig.write_manifest([f], schema="something-else/9")
    r = rig.run("--dry-run")
    assert r.returncode == 2
    assert "chlu-gen-stage-manifest/1" in r.stderr


# ---------------------------------------------------------------------------
# 3. transport: the layout, the sha256 metadata, and idempotence
# ---------------------------------------------------------------------------
def test_uploads_preserve_the_relative_layout_and_carry_the_sha256(rig):
    f = rig.write_file("outputs/gen-decoder-lift/winner/winner_hashes.json", b'{"seed_0": 1}')
    rig.write_manifest([f])

    r = rig.run()
    assert r.returncode == 0, r.stderr
    cps = [c for c in rig.calls() if c.startswith("storage cp")]
    assert len(cps) == 2, cps                    # the artefact + the manifest
    art = cps[0]
    assert str(rig.root / f["path"]) in art
    # ⭐ same relative layout the loaders expect, one level under the prefix
    assert "gs://test-bucket/artefacts/%s" % f["path"] in art
    assert "--custom-metadata=sha256=%s" % f["sha256"] in art
    assert "gs://test-bucket/artefacts/%s" % MF_REL in cps[1]


def test_second_run_skips_objects_already_there(rig):
    """Idempotent + resumable: a matching size AND sha256 is a skip, so a stage
    interrupted at 1.9 GiB resumes instead of re-uploading."""
    f = rig.write_file("outputs/a/x.json", b'{"k": 1}')
    rig.write_manifest([f])
    listing = [{"url": "gs://test-bucket/artefacts/%s" % f["path"],
                "metadata": {"name": "artefacts/%s" % f["path"],
                             "size": str(f["bytes"]),
                             "metadata": {"sha256": f["sha256"]}}}]
    ls_json = rig.root.parent / "ls.json"
    ls_json.write_text(json.dumps(listing))

    r = rig.run(env_extra={"GCLOUD_LS_JSON": str(ls_json)})
    assert r.returncode == 0, r.stderr
    assert "already there : 1 files" in r.stdout
    cps = [c for c in rig.calls() if c.startswith("storage cp")]
    assert len(cps) == 1, cps                    # the manifest only
    assert MF_REL in cps[0]


def test_a_remote_object_with_the_wrong_sha_is_re_uploaded(rig):
    f = rig.write_file("outputs/a/x.json", b'{"k": 1}')
    rig.write_manifest([f])
    listing = [{"metadata": {"name": "artefacts/%s" % f["path"],
                             "size": str(f["bytes"]),
                             "metadata": {"sha256": "f" * 64}}}]
    ls_json = rig.root.parent / "ls.json"
    ls_json.write_text(json.dumps(listing))

    r = rig.run(env_extra={"GCLOUD_LS_JSON": str(ls_json)})
    assert "to upload     : 1 files" in r.stdout
    assert len([c for c in rig.calls() if c.startswith("storage cp")]) == 2


def test_prefix_and_bucket_are_overridable(rig):
    f = rig.write_file("outputs/a/x.json", b"{}")
    rig.write_manifest([f])
    r = rig.run("--bucket", "other-bucket", "--prefix", "artefacts-v2")
    assert r.returncode == 0, r.stderr
    assert any("gs://other-bucket/artefacts-v2/outputs/a/x.json" in c
               for c in rig.calls() if c.startswith("storage cp"))


# ---------------------------------------------------------------------------
# 4. the contract with `startup.sh` (gen-cloud-runner owns the other half)
# ---------------------------------------------------------------------------
def test_print_env_publishes_exactly_the_five_gen_roots(rig):
    r = rig.run("--print-env")
    assert r.returncode == 0, r.stderr
    exported = set(re.findall(r"^export (CHLU_[A-Z_]+)=", r.stdout, re.M))
    assert exported == {"CHLU_CLAUDE_ROOT", "CHLU_GEN_DATA_ROOT", "CHLU_GEN_BRIDGE_ROOT",
                        "CHLU_GEN_STORE_RUNS", "CHLU_GEN_DECLIFT_DIR"}


def test_the_five_roots_match_the_csf3_manifests_relative_layout(rig):
    """⛔ The names and the RELATIVE paths are CSF3's verbatim — only the root
    moves. If these drift, a result cannot be moved between the platforms."""
    r = rig.run("--print-env", env_extra={"JOB_ART_ROOT": "/ROOT"})
    got = dict(re.findall(r'^export (CHLU_[A-Z_]+)="([^"]+)"', r.stdout, re.M))
    assert got == {
        "CHLU_CLAUDE_ROOT": "/ROOT",
        "CHLU_GEN_DATA_ROOT": "/ROOT/data/gen",
        "CHLU_GEN_BRIDGE_ROOT": "/ROOT/outputs/gen-feature-bridge/runs",
        "CHLU_GEN_STORE_RUNS": "/ROOT/outputs/gen-store-grouping/runs",
        "CHLU_GEN_DECLIFT_DIR": "/ROOT/outputs/gen-decoder-lift/winner",
    }


def test_env_names_agree_with_the_csf3_stage_manifest_block():
    """The CSF3 script writes `env_for_the_job`; the two must name the same
    five variables or the platforms are wired differently."""
    csf3 = (REPO / "scripts" / "csf3" / "stage_artefacts.sh").read_text()
    csf3_names = set(re.findall(r'"(CHLU_[A-Z_]+)": "\$HOME/', csf3))
    cloud = STAGE.read_text()
    cloud_names = set(re.findall(r'^export (CHLU_[A-Z_]+)="\$\{JOB_ART_ROOT\}', cloud, re.M))
    assert csf3_names and csf3_names == cloud_names


# ---------------------------------------------------------------------------
# 5. the dataset decision, pinned to the production constant
# ---------------------------------------------------------------------------
def test_3dshapes_sha256_matches_the_production_loader_constant():
    """⭐ Asserted from `chlu/data/shapes3d.py` by text, not by import: the point
    is that the bridge's documented checksum cannot drift from the loader's."""
    src = (REPO / "chlu" / "data" / "shapes3d.py").read_text()
    sha = re.search(r'H5_SHA256 = "([0-9a-f]{64})"', src).group(1)
    assert sha == "0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785"
    # the decision and the abbreviated hash are stated in the stager's header
    header = STAGE.read_text()
    assert "NOT staged" in header
    assert sha[:8] in header


def test_the_dataset_is_never_uploaded_by_the_stager(rig):
    """⛔ A dataset in the manifest would mean paying storage for a file the VM
    can fetch free; the stager transports the manifest's set and nothing else."""
    f = rig.write_file("outputs/a/x.json", b"{}")
    rig.write_manifest([f])
    rig.run()
    assert not any("3dshapes.h5" in c for c in rig.calls())


# ---------------------------------------------------------------------------
# 6. the verifier's own shape (it runs on a VM; here we only check its contract)
# ---------------------------------------------------------------------------
def test_verify_gcs_refuses_a_missing_manifest(tmp_path):
    env = dict(os.environ)
    env["CHLU_BUCKET"] = "test-bucket"
    r = subprocess.run(["bash", str(VERIFY), "--no-fetch", "--art-root", str(tmp_path)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2
    assert "no manifest" in r.stderr


def test_verify_gcs_ports_every_csf3_loader_key():
    """⛔ The dispatch table is the CSF3 one, key for key: a key that exists
    there and not here is an artefact this platform cannot open."""
    csf3 = (REPO / "scripts" / "csf3" / "verify_artefacts.sh").read_text()
    gcs = VERIFY.read_text()

    def keys(src):
        block = re.search(r"LOADERS = \{(.*?)\n\}", src, re.S).group(1)
        return set(re.findall(r'"(\w+)":', block))

    assert keys(csf3) == keys(gcs)
    # and the loader bodies are ported verbatim, not re-implemented
    for fn in ("_load_feature_cache", "_load_store_plan", "_load_winner_hashes",
               "_load_c_invalid", "_load_eqx_pickle", "_load_npy", "_load_json"):
        assert fn in gcs


def test_verify_gcs_writes_the_staged_json_schema():
    """`STAGED-GCS.json` is `STAGED.json`'s schema — same id, same keys, plus
    three additive GCP-only ones."""
    csf3 = (REPO / "scripts" / "csf3" / "verify_artefacts.sh").read_text()
    gcs = VERIFY.read_text()
    csf3_keys = set(re.findall(r'^    "(\w+)":', csf3, re.M))
    gcs_keys = set(re.findall(r'^    "(\w+)":', gcs, re.M))
    assert "chlu-gen-staged/1" in gcs
    assert csf3_keys <= gcs_keys, csf3_keys - gcs_keys
    assert gcs_keys - csf3_keys == {"gcs_root", "vm", "dataset"}
