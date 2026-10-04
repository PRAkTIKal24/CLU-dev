"""Tests for the C2W8 **strong-φ re-price** of the Split-CIFAR null.

What must never break silently here:

  * **the φ term of the byte ledger** (charter §A4.3). A strong encoder that is off
    the ledger is the matched-bytes violation in its most obvious form, so every arm
    must be able to report its own frozen parameter count, that count must match an
    independent hand count, and it must be **the same number on the store arm and on
    both kNN-in-φ laundering arms** — they read through the same φ, so the ledger
    must not silently favour one of them;
  * **the conv arms reach the CL entry end-to-end** under the binding ``task1_only``
    regime and produce a launder row computed in the same φ;
  * **the preset trap** (`cl-encoder` §10): ``--dataset cifar10`` applies its preset
    *after* a project config, so overrides must be applied last or a run silently
    executes on a different φ fit pool than the one that was asked for.

Everything runs on tiny synthetic labelled images at CIFAR shape — no download.
"""

import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.experiments import exp_cl_entry as cle
from chlu.experiments.exp_phi_read_in import build_read_in, read_in_param_floats
from chlu.experiments.phi_encoders import ENCODER_ARMS, module_param_floats


# ---------------------------------------------------------------------------
# tiny synthetic CIFAR-shaped labelled data (10 classes, 3×8×8 images)
# ---------------------------------------------------------------------------
SIDE, CH = 32, 3  # the real CIFAR-10 shape: ``image_shape`` is dataset-keyed
DIM = CH * SIDE * SIDE


def _toy_images(n_per_class=14, seed=0):
    rng = np.random.default_rng(seed)
    proto = rng.random((10, DIM), dtype=np.float32)
    X, y = [], []
    for c in range(10):
        X.append(np.clip(proto[c] + rng.normal(size=(n_per_class, DIM)) * 0.05, 0, 1))
        y.append(np.full(n_per_class, c))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])


def _toy_cfg(arm="randconv", phi_dim=8):
    cfg = get_default_config().experiment_cl_entry
    cfg.dataset = "cifar10"  # drives image_shape; the data is passed in explicitly
    cfg.seeds = [0]
    cfg.phi_arm = arm
    cfg.phi_dim = phi_dim
    cfg.phi_regimes = ["task1_only"]  # the binding regime (PREREG_CL_PHI)
    cfg.n_train_per_task = 14
    cfg.n_test_per_task = 8
    cfg.n_fit_region = 60
    cfg.n_fit_pool = 30
    cfg.memory_items = 12
    cfg.clu_steps = 20
    cfg.rollout_chunk = 32
    cfg.baselines = ["finetune"]
    # ⚠ the MLP backbone, not the CIFAR "cnn" one — a **COST** choice, nothing more.
    # ⛔ The reason this comment used to give is STALE and has been corrected (C2W8
    # pass 3, rider 4c): the x64 dtype bug it described (``ConvNet`` building weights
    # at the ambient JAX dtype against float32 images ⇒
    # `lax.conv_general_dilated requires arguments to have the same dtypes`) was
    # **FIXED** by pass-2 wt3 in `42b781c` — the input is now promoted to the
    # parameter dtype, a no-op at x64-off asserted bit-identically in
    # `tests/test_cl_baselines_x64.py`, which also exercises the CNN forward AND one
    # `_train_task` step under x64 behind a function-scoped fixture. The `"cnn"` path
    # is therefore testable; it is simply slower than this test needs to be.
    cfg.backbone = "mlp"
    cfg.mlp_width = 16
    cfg.mlp_depth = 1
    cfg.baseline_iters = 6
    cfg.tune_baselines = False
    cfg.fisher_samples = 6
    cfg.enc_channels = [4, 8]
    cfg.enc_pool = 2
    cfg.enc_steps = 2
    cfg.enc_batch = 8
    cfg.enc_proj_dim = 8
    return cfg


# ---------------------------------------------------------------------------
# the φ term of the byte ledger (§A4.3)
# ---------------------------------------------------------------------------
def test_pca_read_in_reports_its_own_parameter_count():
    """φ = mean (D) + components (k·D). Hand-counted, not read off the object."""
    cfg = _toy_cfg(arm="pca", phi_dim=8)
    X = _toy_images()[0][0]
    phi, _ = build_read_in("pca", "cifar10", X, X, cfg, seed=0)
    assert read_in_param_floats(phi) == 8 * DIM + DIM


@pytest.mark.parametrize("arm", ENCODER_ARMS)
def test_conv_arms_report_the_frozen_trunk_plus_head(arm):
    """Hand count: conv(3→4) + conv(4→8) + 2 GroupNorms + the PCA head."""
    cfg = _toy_cfg(arm=arm, phi_dim=8)
    X = _toy_images()[0][0]
    phi, prov = build_read_in(arm, "cifar10", X, X, cfg, seed=0)
    conv = (3 * 4 * 9 + 4) + (4 * 8 * 9 + 8)
    norms = 2 * 4 + 2 * 8  # GroupNorm weight+bias per channel
    h_dim = 8 * cfg.enc_pool**2
    head = h_dim + 8 * h_dim + 8  # mean + components + scale
    assert read_in_param_floats(phi) == conv + norms + head
    # the provenance carries the same number (it travels into the results JSON)
    assert prov["phi_param_floats"] == read_in_param_floats(phi)
    # ⛔ the fitting-only parts (projection head / decoder) are NOT carried
    assert module_param_floats(phi.trunk) == conv + norms


def test_a_read_in_that_cannot_be_ledgered_is_refused():
    """A φ off the byte ledger is a hidden capacity increase — it must raise, not 0."""
    with pytest.raises(TypeError):
        read_in_param_floats(object())


# ---------------------------------------------------------------------------
# the conv arms reach the entry, and the launder is ledgered identically
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ["randconv", "pca"])
def test_entry_runs_on_the_arm_and_ledgers_phi_on_every_row(arm, tmp_path):
    cfg_all = get_default_config()
    cfg_all.experiment_cl_entry = _toy_cfg(arm=arm)
    res = cle.run_experiment_cl_entry(
        config=cfg_all, save_dir=str(tmp_path / "plots"), seed=0, items=["entry"],
        data=_toy_images(),
    )
    table = {r["method"]: r for r in res["baseline_table"]}
    assert "clu_entry_task1_only" in table
    launders = [m for m in table if m.startswith("knn_phi_")]
    assert launders, "the kNN-in-φ launder must run on every cell"
    phi_floats = table["clu_entry_task1_only"]["phi_param_floats"]
    assert phi_floats > 0
    # ⭐ the launder reads through the SAME φ ⇒ identical φ bytes on both arms;
    # the ledger must not quietly hand the store a cheaper encoder than its control
    for m in launders:
        assert table[m]["phi_param_floats"] == phi_floats
        assert table[m]["total_floats"] == table[m]["memory_floats"] + phi_floats
    assert (table["clu_entry_task1_only"]["total_floats"]
            == table["clu_entry_task1_only"]["memory_floats"] + phi_floats)
    # the gradient baseline carries its backbone instead, and no φ
    assert table["finetune"]["fixed_state_floats"] > 0
    assert table["finetune"]["phi_param_floats"] == 0


def test_phi_ledger_can_be_switched_off_without_touching_accuracy():
    """The ledger flag is a reporting column, never a physics knob."""
    cfg_all = get_default_config()
    cfg_all.experiment_cl_entry = _toy_cfg(arm="pca")
    cfg_all.experiment_cl_entry.baselines = ["none"]
    stream = cle.build_cl_stream(cfg_all.experiment_cl_entry, 0, data=_toy_images())
    on = cle.run_clu_entry(cfg_all.experiment_cl_entry, stream, "task1_only", 0)
    cfg_all.experiment_cl_entry.count_phi_param_floats = False
    off = cle.run_clu_entry(cfg_all.experiment_cl_entry, stream, "task1_only", 0)
    assert on["phi_param_floats"] > 0 and off["phi_param_floats"] == 0
    assert on["metrics_clu"]["ACC"] == off["metrics_clu"]["ACC"]
    assert on["metrics_knn_ringbuffer"]["ACC"] == off["metrics_knn_ringbuffer"]["ACC"]


# ---------------------------------------------------------------------------
# the preset trap (cl-encoder §10)
# ---------------------------------------------------------------------------
def test_overrides_are_applied_after_the_cifar_preset():
    """⚠ ``apply_cifar10`` overwrites ``n_fit_region``/``n_fit_pool``; overrides last."""
    config = get_default_config()
    config.experiment_cl_entry.n_fit_pool = 6000
    cle.apply_cifar10(config)
    assert config.experiment_cl_entry.n_fit_pool == 3000  # the trap, reproduced
    applied = cle.apply_overrides(config, [
        "n_fit_pool=6000", "n_fit_region=25000", "phi_arm=simclr", "phi_dim=256",
        "enc_steps=8000", "phi_regimes=task1_only", "seeds=0,1,2",
    ])
    cfg = config.experiment_cl_entry
    assert cfg.n_fit_pool == 6000 and cfg.n_fit_region == 25000
    assert cfg.phi_arm == "simclr" and cfg.phi_dim == 256 and cfg.enc_steps == 8000
    assert cfg.phi_regimes == ["task1_only"] and cfg.seeds == [0, 1, 2]
    assert applied["phi_arm"] == "simclr"


def test_unknown_or_malformed_overrides_are_refused():
    config = get_default_config()
    with pytest.raises(ValueError):
        cle.apply_overrides(config, ["clu_gamma=0.9"])  # not on the allow-list
    with pytest.raises(ValueError):
        cle.apply_overrides(config, ["phi_arm"])  # not key=value
