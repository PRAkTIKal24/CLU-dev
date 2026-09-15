"""Tests for the GCP cloud path (`scripts/cloud/`).

Everything here is hermetic: no GCP call, no GPU, no network, no JAX import.
The quota logic is driven from a RECORDED `gcloud compute regions describe`
payload for us-central1 (captured 2026-09-15) rather than the live API, so the
refusal behaviour stays pinned even after the real quota changes -- which is
the point, since the refusal is what stops a launch.

What is deliberately NOT tested here: that a pre-empted VM resumes. That claim
cannot be made by a unit test and was made by actually destroying a running
instance; the evidence is in `.claude/outputs/gen-cloud-enablement.md`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

CLOUD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "cloud")
sys.path.insert(0, CLOUD)

import ckpt as ckpt_mod  # noqa: E402
import quota as quota_mod  # noqa: E402


# --------------------------------------------------------------------------
# The measured us-central1 quota, 2026-09-15. Only the metrics the preflight
# consults; `limit`/`usage` are verbatim from the API.
# --------------------------------------------------------------------------
REGION_FIXTURE = {
    "name": "us-central1",
    "quotas": [
        {"metric": "CPUS", "limit": 200.0, "usage": 8.0},
        {"metric": "PREEMPTIBLE_CPUS", "limit": 0.0, "usage": 0.0},
        {"metric": "A2_CPUS", "limit": 12.0, "usage": 0.0},
        {"metric": "NVIDIA_L4_GPUS", "limit": 8.0, "usage": 2.0},
        {"metric": "PREEMPTIBLE_NVIDIA_L4_GPUS", "limit": 8.0, "usage": 0.0},
        {"metric": "NVIDIA_A100_GPUS", "limit": 1.0, "usage": 1.0},
        {"metric": "PREEMPTIBLE_NVIDIA_A100_GPUS", "limit": 16.0, "usage": 0.0},
        {"metric": "NVIDIA_A100_80GB_GPUS", "limit": 0.0, "usage": 0.0},
        {"metric": "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS", "limit": 0.0, "usage": 0.0},
        {"metric": "NVIDIA_T4_GPUS", "limit": 4.0, "usage": 0.0},
        {"metric": "PREEMPTIBLE_NVIDIA_T4_GPUS", "limit": 4.0, "usage": 0.0},
    ],
}


@pytest.fixture()
def quotas():
    return quota_mod.quota_map(REGION_FIXTURE)


# ============================== quota preflight ==============================


def test_l4_on_demand_fits(quotas):
    rows = quota_mod.check("l4", spot=False, gpus=1, quotas=quotas)
    assert all(r["ok"] for r in rows)
    gpu = next(r for r in rows if r["metric"] == "NVIDIA_L4_GPUS")
    assert gpu["free"] == 6.0, "2 of the 8 L4s are held by another team's instances"


def test_spot_consults_the_preemptible_gpu_metric(quotas):
    rows = quota_mod.check("a100-40", spot=True, gpus=1, quotas=quotas)
    assert rows[0]["metric"] == "PREEMPTIBLE_NVIDIA_A100_GPUS"
    # ...and the on-demand A100 is exhausted, so this is not the same question.
    with pytest.raises(quota_mod.QuotaRefusal):
        quota_mod.check("a100-40", spot=False, gpus=1, quotas=quotas)


def test_a100_80gb_is_refused_by_name_with_its_measured_limit(quotas):
    with pytest.raises(quota_mod.QuotaRefusal) as exc:
        quota_mod.check("a100-80", spot=True, gpus=1, quotas=quotas)
    msg = str(exc.value)
    assert "a100-80" in msg
    assert "PREEMPTIBLE_NVIDIA_A100_80GB_GPUS" in msg
    assert "limit = 0" in msg


def test_h100_h200_are_refused_as_unknown_and_the_message_says_why(quotas):
    for card in ("h100", "h200"):
        with pytest.raises(quota_mod.QuotaRefusal) as exc:
            quota_mod.check(card, spot=True, gpus=1, quotas=quotas)
        assert "not offered" in str(exc.value)


def test_cpu_quota_binds_before_gpu_quota_for_a100(quotas):
    """16 spot A100 GPUs, 12 A2_CPUS => concurrency 1, not 16.

    The regression this pins: a GPU-only preflight passes x2 here and the
    launch then fails minutes later with an opaque QUOTA_EXCEEDED.
    """
    quota_mod.check("a100-40", spot=True, gpus=1, quotas=quotas)  # 1 fits
    with pytest.raises(quota_mod.QuotaRefusal) as exc:
        quota_mod.check("a100-40", spot=True, gpus=2, quotas=quotas)
    msg = str(exc.value)
    assert "A2_CPUS" in msg and "24 vCPU" in msg
    assert "caps concurrency at 1" in msg


def test_zero_preemptible_cpu_quota_is_not_treated_as_a_block(quotas):
    """PREEMPTIBLE_CPUS is 0 in this project; GCE then bills spot VMs against
    the regular CPUS quota. Reading the 0 literally would refuse every spot
    launch the project can actually make."""
    rows = quota_mod.check("l4", spot=True, gpus=1, quotas=quotas)
    assert rows[1]["metric"] == "CPUS"
    assert rows[1]["ok"]


def test_a_nonzero_preemptible_cpu_quota_would_bind_instead(quotas):
    q = dict(quotas)
    q["PREEMPTIBLE_CPUS"] = (4.0, 0.0)
    with pytest.raises(quota_mod.QuotaRefusal) as exc:
        quota_mod.check("l4", spot=True, gpus=1, quotas=q)
    assert "PREEMPTIBLE_CPUS" in str(exc.value)


def test_unknown_card_lists_the_known_ones(quotas):
    with pytest.raises(quota_mod.QuotaRefusal) as exc:
        quota_mod.check("v100", spot=False, gpus=1, quotas=quotas)
    assert "a100-40" in str(exc.value) and "l4" in str(exc.value)


# ============================== checkpointing ================================


def _state(step: int) -> dict:
    return {"acc": np.arange(step, step + 8, dtype=np.float32) * 1.5,
            "chain": np.frombuffer(bytes([step % 256]) * 32, dtype=np.uint8),
            "step": np.int64(step)}


def test_roundtrip_is_bit_exact(tmp_path):
    fp = ckpt_mod.config_fingerprint({"seed": 0, "steps": 10})
    cp = ckpt_mod.Checkpointer(str(tmp_path / "ck"), fp, every_s=0.0)
    cp.save(5, _state(5))
    step, got = cp.latest()
    assert step == 5
    for k, v in _state(5).items():
        assert np.array_equal(got[k], np.asarray(v))
        assert got[k].dtype == np.asarray(v).dtype
        # bytes, not just values: a resumed run must continue from the same bits
        assert got[k].tobytes() == np.asarray(v).tobytes()


def test_no_checkpoint_reads_as_a_fresh_run(tmp_path):
    cp = ckpt_mod.Checkpointer(str(tmp_path / "ck"), "abc", every_s=0.0)
    assert cp.latest() is None


def test_resume_refuses_a_config_mismatch_and_names_both(tmp_path):
    d = str(tmp_path / "ck")
    a = ckpt_mod.config_fingerprint({"seed": 0, "steps": 10})
    b = ckpt_mod.config_fingerprint({"seed": 1, "steps": 10})
    ckpt_mod.Checkpointer(d, a, every_s=0.0).save(3, _state(3))
    with pytest.raises(ckpt_mod.FingerprintMismatch) as exc:
        ckpt_mod.Checkpointer(d, b, every_s=0.0).latest()
    assert a in str(exc.value) and b in str(exc.value)
    assert "Refusing to resume" in str(exc.value)


def test_fingerprint_is_order_and_whitespace_stable():
    assert ckpt_mod.config_fingerprint({"a": 1, "b": 2}) == ckpt_mod.config_fingerprint({"b": 2, "a": 1})
    assert ckpt_mod.config_fingerprint({"a": 1}) != ckpt_mod.config_fingerprint({"a": 1.5})


def test_keep_n_prunes_oldest_first(tmp_path):
    d = str(tmp_path / "ck")
    cp = ckpt_mod.Checkpointer(d, "fp", every_s=0.0, keep=2)
    for s in (1, 2, 3, 4):
        cp.save(s, _state(s))
    kept = sorted(n for n in os.listdir(d) if n.startswith("ckpt_"))
    assert kept == ["ckpt_000000000003.npz", "ckpt_000000000004.npz"]
    assert cp.latest()[0] == 4


def test_cadence_is_wall_clock_not_step_count(tmp_path):
    """Pre-emption does not wait for an epoch boundary, so neither may we."""
    cp = ckpt_mod.Checkpointer(str(tmp_path / "ck"), "fp", every_s=3600.0)
    assert cp.maybe_save(1, _state(1)) is True     # first call is always due
    assert cp.maybe_save(2, _state(2)) is False    # timer not elapsed
    assert cp.maybe_save(3, _state(3), force=True) is True   # the pre-emption path
    assert cp.latest()[0] == 3


def test_an_interrupted_write_leaves_the_previous_checkpoint_intact(tmp_path):
    """A process killed mid-save must not destroy the checkpoint it had."""
    d = tmp_path / "ck"
    cp = ckpt_mod.Checkpointer(str(d), "fp", every_s=0.0)
    cp.save(1, _state(1))
    (d / ".ckpt_tmp_halfwritten.npz").write_bytes(b"\x93NUMPY truncated garbage")
    step, got = cp.latest()
    assert step == 1
    assert np.array_equal(got["acc"], _state(1)["acc"])


def test_a_temp_file_is_never_mistaken_for_a_checkpoint(tmp_path):
    d = tmp_path / "ck"
    cp = ckpt_mod.Checkpointer(str(d), "fp", every_s=0.0)
    cp.save(1, _state(1))
    (d / ".ckpt_tmp_x.npz").write_bytes(b"junk")
    cp._prune()
    assert cp.latest()[0] == 1


def test_a_hand_edited_step_is_refused(tmp_path):
    d = tmp_path / "ck"
    cp = ckpt_mod.Checkpointer(str(d), "fp", every_s=0.0)
    cp.save(4, _state(4))
    os.rename(d / "ckpt_000000000004.npz", d / "ckpt_000000000009.npz")
    with pytest.raises(ckpt_mod.FingerprintMismatch) as exc:
        cp.latest()
    assert "corrupt or hand-edited" in str(exc.value)


def test_keep_n_spans_local_and_remote_across_a_resume(tmp_path, monkeypatch):
    """The regression the `preempt-kill` run exposed.

    A resumed process downloads only the ONE checkpoint it restarts from, so a
    prune that looks at the local directory alone never sees -- and never
    deletes -- what the pre-empted process left in the bucket. "Keep last N"
    then holds per process, not per job.
    """
    d = str(tmp_path / "ck")
    cp = ckpt_mod.Checkpointer(d, "fp", remote_uri="gs://bucket/pfx", every_s=0.0, keep=1)

    orphans = [137, 163]          # left in the bucket by the killed process
    removed: list[str] = []
    monkeypatch.setattr(cp, "_publish", lambda path: None)
    monkeypatch.setattr(ckpt_mod, "_gcloud",
                        lambda *a, **k: removed.append(a[-1]) or subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(cp, "_remote_steps", lambda: sorted(set(orphans)))

    cp.save(300, _state(300))
    # keep=1 and the newest is 300, so BOTH bucket orphans must go
    assert any("137" in r for r in removed), "the orphaned remote checkpoint was never pruned"
    assert any("163" in r for r in removed)
    assert not any("300" in r for r in removed), "the newest checkpoint must survive"


def test_discard_clears_everything(tmp_path):
    cp = ckpt_mod.Checkpointer(str(tmp_path / "ck"), "fp", every_s=0.0)
    cp.save(1, _state(1))
    cp.discard()
    assert cp.latest() is None


def test_reserved_meta_key_is_rejected(tmp_path):
    cp = ckpt_mod.Checkpointer(str(tmp_path / "ck"), "fp", every_s=0.0)
    with pytest.raises(ValueError):
        cp.save(1, {"__chlu_meta__": np.zeros(1)})


# ============================ image / launch wiring ==========================


def test_dockerfile_pins_every_base_by_digest():
    """A tag is a mutable pointer: a re-run silently gets a different
    environment and nothing in the result says so."""
    text = open(os.path.join(CLOUD, "Dockerfile")).read()
    refs = [ln.split()[1] for ln in text.splitlines() if ln.startswith("FROM ")]
    refs += [tok.split("=", 1)[1] for ln in text.splitlines() if ln.startswith("COPY --from=")
             for tok in [ln.split()[1]]]
    assert refs, "no image references found -- the test is not looking at the right file"
    for ref in refs:
        assert "@sha256:" in ref, f"{ref} is not digest-pinned"


def test_dockerfile_does_not_add_a_host_cuda():
    """CSF3 README §3.1: JAX's pip CUDA wheels bundle CUDA+cuDNN, and a second
    CUDA on LD_LIBRARY_PATH shadows them and breaks jax."""
    text = open(os.path.join(CLOUD, "Dockerfile")).read()
    # comments are allowed to *discuss* the trap; instructions must not set it
    live = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "nvidia/cuda" not in live
    assert "cuda-toolkit" not in live
    assert "LD_LIBRARY_PATH" not in live
    assert "uv sync --frozen" in live and "--extra cuda" in live


def test_the_gcloud_download_is_pinned_by_version_and_sha256():
    text = open(os.path.join(CLOUD, "Dockerfile")).read()
    assert "ARG GCLOUD_SHA256=" in text
    assert "sha256sum -c -" in text
    assert "${GCLOUD_VERSION}" in text


def test_cards_json_and_config_sh_agree():
    """These were two hand-maintained copies once. They are not any more, and
    this test is what keeps them from becoming two again."""
    cards = quota_mod.load_cards()
    for name, spec in cards["cards"].items():
        out = subprocess.run(
            ["bash", "-c", f'source "{CLOUD}/config.sh"; chlu_card_spec {name}; chlu_price {name} 0; chlu_price {name} 1'],
            capture_output=True, text=True, check=True).stdout.split()
        assert out[0] == spec["accelerator"]
        assert out[1] == spec["machine_type"]
        assert out[2] == spec["gpu_metric"]
        assert out[3] == spec["cpu_metric"]
        assert int(out[4]) == spec["vcpu"]
        assert float(out[5]) == spec["usd_per_hour"]["ondemand"]
        assert float(out[6]) == spec["usd_per_hour"]["spot"]


def _stub_gcloud(tmp_path):
    """A `gcloud` that answers only `compute regions describe`, from the fixture."""
    p = tmp_path / "gcloud"
    p.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1 $2 $3" = "compute regions describe" ]; then\n'
        f"  cat {tmp_path / 'region.json'}\n"
        "  exit 0\nfi\nexit 9\n"
    )
    p.chmod(0o755)
    (tmp_path / "region.json").write_text(json.dumps(REGION_FIXTURE))
    return str(p)


def _launch(tmp_path, *args):
    env = dict(os.environ, GCLOUD_BIN=_stub_gcloud(tmp_path))
    return subprocess.run(["bash", os.path.join(CLOUD, "launch.sh"), *args],
                          capture_output=True, text=True, env=env)


DIGEST_REF = ("us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda@sha256:"
              + "0" * 64)


def test_launch_dry_run_plans_a_valid_job(tmp_path):
    r = _launch(tmp_path, "--card", "l4", "--job", "unit", "--image", DIGEST_REF, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "nothing created" in r.stdout
    assert "sync-back" in r.stdout and "every 15s" in r.stdout
    assert "est. cost" in r.stdout


def test_launch_refuses_a_tag_pinned_image(tmp_path):
    r = _launch(tmp_path, "--card", "l4", "--job", "unit",
                "--image", "us-central1-docker.pkg.dev/forgisprova/chlu/chlu-cuda:latest", "--dry-run")
    assert r.returncode != 0
    assert "pinned by TAG" in r.stderr


def test_launch_refuses_an_unquotaed_card_naming_card_and_limit(tmp_path):
    r = _launch(tmp_path, "--card", "a100-80", "--job", "unit", "--image", DIGEST_REF, "--dry-run")
    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "a100-80" in combined
    assert "NVIDIA_A100_80GB_GPUS" in combined and "limit = 0" in combined
    assert "Nothing was created" in combined


def test_launch_requires_a_gce_legal_job_name(tmp_path):
    r = _launch(tmp_path, "--card", "l4", "--job", "Bad_Name", "--image", DIGEST_REF, "--dry-run")
    assert r.returncode != 0 and "GCE-legal" in r.stderr


def test_launch_rejects_an_unknown_flag_rather_than_ignoring_it(tmp_path):
    r = _launch(tmp_path, "--card", "l4", "--job", "unit", "--typo", "--dry-run")
    assert r.returncode != 0 and "unknown argument" in r.stderr


def test_spot_launch_sets_termination_action_delete():
    """A pre-empted spot VM must not linger STOPPED with its disk billing."""
    text = open(os.path.join(CLOUD, "launch.sh")).read()
    assert "--provisioning-model=SPOT" in text
    assert "--instance-termination-action=DELETE" in text


def test_startup_refuses_an_unpinned_image_and_syncs_before_the_payload():
    text = open(os.path.join(CLOUD, "startup.sh")).read()
    assert "FATAL_NOT_PINNED" in text
    # the sync loop must be started before `docker run`, or a VM that dies
    # during the pull leaves nothing behind at all
    assert text.index("SYNC_PID=$!") < text.index("docker run")
    assert "gcloud compute instances delete" in text, "the VM must delete itself"


def test_shutdown_handler_stops_the_container_so_the_payload_gets_sigterm():
    text = open(os.path.join(CLOUD, "shutdown.sh")).read()
    assert "docker stop" in text
    assert "rsync" in text, "the last write must go to the bucket, not just to disk"


def test_payload_watches_for_preemption_two_independent_ways():
    text = open(os.path.join(CLOUD, "payload_smoke.py")).read()
    assert "instance/preempted" in text and "wait_for_change=true" in text
    assert "signal.SIGTERM" in text
    assert "force=True" in text or "cp.save(step" in text


def test_neither_syncer_duplicates_checkpoints_into_the_output_prefix():
    """The Checkpointer publishes and prunes `<remote>/ckpt/` itself. A copy of
    the same files under `<remote>/out/ckpt/` is pruned by nothing and grows
    without bound across a campaign of pre-emptions.

    Both syncers need the exclusion, and it must be unanchored: gcloud matches
    the regex against the FULL path, so `^ckpt/.*` silently matches nothing --
    which is how the first attempt at this passed review and did nothing.
    """
    host = open(os.path.join(CLOUD, "startup.sh")).read()
    rsync_lines = [ln for ln in host.splitlines()
                   if "storage rsync" in ln and not ln.lstrip().startswith("#")]
    assert rsync_lines, "startup.sh no longer syncs at all"
    for ln in rsync_lines:
        assert "-x '.*ckpt.*'" in ln, f"host sync copies checkpoints twice: {ln.strip()}"

    payload = open(os.path.join(CLOUD, "payload_smoke.py")).read()
    call = payload[payload.index("def _sync("):]
    assert '"rsync"' in call, "the in-container sync is gone"
    assert '"-x", ".*ckpt.*"' in call, "the in-container sync copies checkpoints twice"


def test_payload_syncs_on_every_checkpoint_not_only_at_exit():
    text = open(os.path.join(CLOUD, "payload_smoke.py")).read()
    body = text[text.index("for step in range("):text.index("    result = {")]
    assert "_sync(args.out, remote)" in body, "mid-run sync-back is missing from the loop"
