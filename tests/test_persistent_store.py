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


# ------------------------------------------- L5 on the LIVE store path
def _tiny_rig(seed=0, **over):
    import jax

    from chlu.core.clu_system import CluSystemConfig, build_system

    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=4, budget=4, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=0.05, write_steps=20, read_steps=60, address_steps=40,
        n_query_per_item=2, leak=0.0, **over)
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


def _plant_and_rewrite(guard: bool, seed: int = 0):
    """Write an item, then rewrite its own well from a displaced address."""
    import jax

    from chlu.core.store_lifecycle import guarded_rewrite

    sysm = _tiny_rig(seed=seed)
    site = np.array([0.4, 0.0])
    sysm.write_stream([{"item_id": 0, "address": site, "payload": 0.2}])
    p = LifecycleParams(lifecycle=True, refresh_monotonic=guard)
    events = []
    for j, (dx, pay) in enumerate([(0.6, -0.4), (-0.7, 0.45), (0.9, -0.45)]):
        events.append(guarded_rewrite(sysm, 0, site + np.array([dx, 0.0]), pay,
                                      jax.random.PRNGKey(seed * 100 + j), p))
    return events


def test_l5_guard_off_a_destructive_rewrite_reduces_depth_on_the_live_store():
    """⭐ L5's designed negative on the SHIPPED write path, not on replayed
    numbers: with the guard OFF at least one planted rewrite must actually
    reduce the well's depth. If none does, the guard is unfalsifiable here and
    the leg must not ship on this evidence."""
    off = _plant_and_rewrite(guard=False)
    destructive = [e for e in off if e["depth_after"] < e["depth_before"]]
    assert destructive, "no destructive rewrite could be planted: the negative is vacuous"
    for e in destructive:
        assert e["depth_guarded"] == pytest.approx(e["depth_after"], rel=1e-6)
        assert e["depth_guarded"] < e["depth_before"]


def test_l5_the_guard_repairs_the_same_rewrite_on_the_live_store():
    """...and with the guard ON the same events are repaired **up to the declared
    budget**: ``depth_guarded = min(depth_before, depth_after * gain^2)``.

    ⚠ The cap is not a caveat, it is the registered semantics (§A23.2: "rewrites
    refresh/deepen **up to budget**") and it is the SAME cap ``blocks.py`` applies
    (``amp *= f``, ``f <= refresh_max_gain``, so depth scales by ``f^2``). A
    rewrite that destroyed more than ``gain^2 = 16x`` of the depth is restored by
    16x and no further — and the test says which of the two branches it is in
    rather than quietly asserting only the easy one.
    """
    gain = 4.0
    on = _plant_and_rewrite(guard=True)
    repaired = [e for e in on if e["violation"] > 0.5]
    assert repaired, "no violating rewrite on the guarded arm: nothing to repair"
    capped = 0
    for e in repaired:
        assert e["refresh_factor"] > 1.0
        assert e["depth_guarded"] >= e["depth_after"]      # never worse than unguarded
        target = min(e["depth_before"], e["depth_after"] * gain ** 2)
        assert e["depth_guarded"] == pytest.approx(target, rel=1e-5)
        capped += int(e["depth_after"] * gain ** 2 < e["depth_before"])
    # both branches must be reachable; this rig exercises the capped one
    assert capped >= 1


def test_lifecycle_params_come_from_the_config_group_unchanged():
    p = _cfg(h_hi=5, d_dwell=9, window=4, f_max=0.5).experiment_persistent_store
    lp = LifecycleParams.from_config(p)
    assert (lp.h_hi, lp.d_dwell, lp.window, lp.f_max) == (5, 9, 4, 0.5)
    assert lp.trash_criterion == "last_k_streams"
