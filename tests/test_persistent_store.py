"""C2W10 — the persistent-store rig: L7's OFF identity, the address block, the
geometry estimator the coverage bug forced, and the end-to-end lifecycle cell.

⛔ Every number produced here is MECHANICS: no VALUE cell, no performance claim,
no verdict. The synthetic regime-switcher is a regression instrument and never a
claim venue (§A14.8).
"""

import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.store_lifecycle import ACTIVE, PROTECTED, TRASH, LifecycleParams
from chlu.experiments.exp_persistent_store import (
    RandomProjectionAddress,
    apply_quick,
    distinct_key_spacing,
    label_to_payload,
    off_identity_check,
    run_cell,
    store_config,
)
from chlu.experiments.stream_sources import make_regime_switcher


def _cfg(**over):
    cfg = get_default_config()
    apply_quick(cfg)
    p = cfg.experiment_persistent_store
    p.lifecycle = True
    p.persistent_store = True
    for k, v in over.items():
        setattr(p, k, v)
    return cfg


# ----------------------------------------------------------- address block
def test_the_address_block_is_unfitted_and_idempotent_on_store_space():
    e = RandomProjectionAddress(6, 4, 1, seed=0)
    X = make_regime_switcher(n_features=6, n_per_stream=8, n_anchors=8,
                             schedule=[0, 1], seed=0).X
    e.fit(X[:8])
    z = np.asarray(e(X[:3]))
    assert z.shape == (3, 5)
    assert np.array_equal(np.asarray(e(z)), z)   # idempotent on store-space
    assert np.allclose(z[:, 4:], 0.0)            # payload channels are zero


def test_the_address_block_refuses_an_ambiguous_feature_dimension():
    with pytest.raises(ValueError, match="idempotence"):
        RandomProjectionAddress(5, 4, 1, seed=0)


def test_phi_parameters_are_on_the_byte_ledger():
    e = RandomProjectionAddress(6, 4, 1, seed=0)
    assert e.n_bytes() == (6 * 4 + 4 + 4 + 1) * 4


def test_the_address_scale_is_fixed_from_the_first_stream_only():
    st = make_regime_switcher(n_features=6, n_per_stream=16, n_anchors=8,
                              schedule=[0, 1, 2], seed=0)
    a = RandomProjectionAddress(6, 4, 1, seed=0).fit(st.stream_slice(0)[0])
    b = RandomProjectionAddress(6, 4, 1, seed=0).fit(st.stream_slice(0)[0])
    assert a.scale == b.scale
    assert np.array_equal(a.keys(st.X), b.keys(st.X))


def test_payload_is_bounded_and_class_separating():
    vals = [label_to_payload(c, 4, 8.0) for c in range(4)]
    assert all(abs(v) <= 0.5 for v in vals)
    assert sorted(vals) == vals
    assert min(np.diff(vals)) >= 0.1   # above payload_tol


# ------------------------------------------------- the geometry estimator
def test_distinct_key_spacing_measures_ITEMS_not_INSTANCES():
    """⚠ The coverage bug, pinned. On a revisiting stream the median NN of the
    instances measures the jitter; sizing ``d_safe`` from it collapses
    ``min_sep``, hence the read's coverage radius, hence the usage proxy — and
    every usage-driven verb then looks unexercised for an instrument reason."""
    st = make_regime_switcher(n_features=8, n_per_stream=96, n_anchors=16,
                              jitter=0.02, schedule=[0, 1], seed=0)
    e = RandomProjectionAddress(8, 12, 1, seed=0).fit(st.stream_slice(0)[0])
    g = distinct_key_spacing(e.keys(st.stream_slice(0)[0]))
    assert g["duplicates_detected"] is True
    assert g["spacing"] > 5.0 * g["median_nn_instances"]
    assert 8 <= g["n_distinct"] <= 16


def test_distinct_key_spacing_is_the_plain_median_on_a_duplicate_free_stream():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(64, 6))
    g = distinct_key_spacing(k)
    assert g["duplicates_detected"] is False
    assert g["spacing"] == g["median_nn_instances"]
    assert g["jump_ratio"] < 2.0


# ------------------------------------------------------------------- L7
def test_l7_off_is_bit_identical_and_parameter_count_identical():
    """⭐ L7: with ``persistent_store=False`` and every verb OFF the rig is
    bit-identical AND parameter-count-identical, and NO ``gamma_phi`` field is
    attached at all (an empty field is not bit-identical — C2W8 K2 fact (i))."""
    out = off_identity_check(_cfg(), seed=0)
    assert out["param_count_identical"]
    assert out["leaves_bitwise_identical"]
    assert out["read_bitwise_identical"]
    assert out["trash_attached"] is False
    assert out["trash_bytes"] == 0


def test_l7_the_store_config_only_attaches_the_trash_region_when_the_verb_is_on():
    p = _cfg().experiment_persistent_store
    assert store_config(p, 0, 0.4, 0.5, lifecycle_on=False).gamma_phi is False
    assert store_config(p, 0, 0.4, 0.5, lifecycle_on=True).gamma_phi is True
    p.trash = False
    assert store_config(p, 0, 0.4, 0.5, lifecycle_on=True).gamma_phi is False


# ------------------------------------------------------- the end-to-end cell
def test_the_cell_runs_every_leg_and_emits_both_depth_forms():
    out = run_cell(_cfg(), seed=0, verbose=False)
    assert out["n_live_max"] >= 1
    assert set(out["lifecycle"]["states"]) == {PROTECTED, ACTIVE, TRASH}
    # L6: no depth curve exists without its netted twin
    assert out["depth_curves"]
    for curve in out["depth_curves"].values():
        assert len(curve["depth_raw"]) == len(curve["depth_netted"])
        assert len(curve["chunk"]) == len(curve["depth_raw"])
        for raw, net, cum in zip(curve["depth_raw"], curve["depth_netted"],
                                 curve["cum_decay_factor"], strict=True):
            assert net >= raw if cum < 1.0 else net == raw
    # the byte ledger names every component, trash included
    assert set(out["bytes"]) >= {"clu_store_bytes", "phi_param_bytes",
                                 "trash_bytes", "gamma_phi_enabled"}
    # the cross-stream record the trash criterion is computed from
    assert out["cross_stream"]["hits_by_stream"]
    assert out["cross_stream"]["first_seen_stream"]
    # and the monitor row is registered and reported by name
    assert "protected_saturation" in out["monitor_trips"]
    assert out["flags"]["venue_note"].startswith("⛔")


def test_the_lifecycle_off_arm_fires_no_verb():
    """The master switch really is one: OFF => no promotion, demotion or trash."""
    out = run_cell(_cfg(), seed=0, lifecycle_on=False, verbose=False)
    ev = out["controller_events"]
    assert ev["promote"] == 0 and ev["demote"] == 0 and ev["trash"] == 0
    assert out["bytes"]["trash_bytes"] == 0
    assert out["bytes"]["gamma_phi_enabled"] is False


def test_lifecycle_params_come_from_the_config_group_unchanged():
    p = _cfg(h_hi=5, d_dwell=9, window=4, f_max=0.5).experiment_persistent_store
    lp = LifecycleParams.from_config(p)
    assert (lp.h_hi, lp.d_dwell, lp.window, lp.f_max) == (5, 9, 4, 0.5)
    assert lp.trash_criterion == "last_k_streams"
