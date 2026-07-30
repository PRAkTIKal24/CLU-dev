"""Tests for the full-CLU harness (C2W1).

The properties asserted here are the ones that make the harness a test of the
CLU rather than of a lookup table:

* items live in a **learned ``V_theta``** and the read path **never** consults a
  stored payload (perturbing the eval-only codebook payload must not move the
  read by a single bit);
* :meth:`CluSystem.read` returns the **trajectory** as well as ``q*``, so a
  trajectory read-out is a configuration, not a rewrite;
* a masked write is **local in parameter space** (every other item's atoms come
  out bit-identical), which is what makes streaming writes C3-local;
* a per-item lifetime is **physical** (the item's own atom rows shallow) and
  eviction re-draws the freed group from the init distribution rather than
  zeroing it (zeroing starves the next item in that slot *and* leaves a
  membership trace);
* the settle exposes a **fixed-point residual** for an implicit/DEQ gradient.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.clu_system import (
    CluSystem,
    CluSystemConfig,
    LearnedVStore,
    build_system,
    settled_point_psi,
    store_relative_trajectory,
    tail_mean_psi,
)
from chlu.core.memory_potentials import designed_payloads, designed_sites
from chlu.eval.dividend import (
    byte_account,
    dividend,
    same_keys_null,
    settle_deleted_launder,
)


def _tiny_cfg(**kw):
    base = dict(addr_dim=3, capacity=3, write_steps=25, address_steps=60,
                read_steps=80, n_query_per_item=2, atoms_per_item=8,
                min_atoms=48, min_atoms_base=48, seed=0)
    base.update(kw)
    return CluSystemConfig(**base)


def _written_system(cfg=None, n=3):
    cfg = cfg or _tiny_cfg()
    sys_ = build_system(cfg, loud=False)
    sites = np.asarray(designed_sites(cfg.addr_dim, n, R=cfg.ball_radius, seed=0))
    pays = np.asarray(designed_payloads(n, seed=0))
    sys_.write_stream([{"item_id": i, "address": sites[i], "payload": float(pays[i])}
                       for i in range(n)])
    return sys_, sites, pays


# -- config -----------------------------------------------------------------
def test_config_ignores_unknown_keys_and_reports_non_defaults():
    cfg = CluSystemConfig.from_mapping({"capacity": 5, "not_a_field": 1})
    assert cfg.capacity == 5
    assert cfg.as_flag_table() == {"capacity": 5}


def test_atom_count_uses_the_dimension_aware_floor():
    """Scaling the budget with K only starves the write at high d (w23)."""
    assert _tiny_cfg(addr_dim=4, capacity=8).n_atoms >= 8 * 8
    big = CluSystemConfig(addr_dim=8, capacity=4)
    small = CluSystemConfig(addr_dim=2, capacity=4)
    assert big.n_atoms > small.n_atoms


def test_defaults_are_the_shipped_productive_band():
    """Stage 0 must start where 26 waves measured the levers, not at a guess."""
    c = CluSystemConfig()
    assert (c.atoms_per_item, c.atom_width, c.atom_init_scale) == (32, 0.3, 1.0)
    assert (c.gamma_address, c.gamma_read) == (0.05, 0.02)
    assert (c.address_steps, c.read_steps) == (400, 800)
    assert c.confine == 0.05 and c.query_sigma == 0.15


# -- the store --------------------------------------------------------------
def test_items_live_in_a_learned_V_not_in_arrays():
    """The landscape is entirely learned (coercivity only) — no designed part."""
    store = LearnedVStore(_tiny_cfg(), jax.random.PRNGKey(0))
    assert store.V.designed is None
    assert store.V.learned_family == "atoms"
    assert store.n_bytes() > 0


def test_the_read_path_never_consults_a_stored_payload():
    """The payload lives ONLY in ``V_theta``. The codebook's payload column is
    eval/launder/monitor bookkeeping; if the read could see it, the harness would
    be measuring a lookup table."""
    sys_, sites, pays = _written_system()
    q = np.zeros((2, sys_.store.dim), dtype=np.float32)
    q[:, : sys_.cfg.addr_dim] = sites[:2]
    before = np.asarray(sys_.read(q).value)
    for iid in list(sys_._payloads):
        sys_._payloads[iid] = sys_._payloads[iid] * 0 + 7.77  # corrupt the bookkeeping
    after = np.asarray(sys_.read(q).value)
    assert np.array_equal(before, after)


def test_masked_write_is_local_in_parameter_space():
    """Writing item j leaves every other item's atoms BIT-IDENTICAL (C3-local)."""
    cfg = _tiny_cfg()
    sys_ = build_system(cfg, loud=False)
    sites = np.asarray(designed_sites(cfg.addr_dim, 3, R=1.0, seed=0))
    sys_.write_stream([{"item_id": 0, "address": sites[0], "payload": 0.5}])
    m0 = np.asarray(sys_.store.group_rows(0), dtype=bool)
    before = np.asarray(sys_.store.V.learned.centers)[~m0].copy()
    amp_before = np.asarray(sys_.store.V.learned.amp)[~m0].copy()
    sys_.write_stream([{"item_id": 1, "address": sites[1], "payload": -0.5}])
    after = np.asarray(sys_.store.V.learned.centers)[~m0]
    amp_after = np.asarray(sys_.store.V.learned.amp)[~m0]
    # rows outside the WRITTEN group are untouched; group 0's rows are inside ~m0's
    # complement only for the written slot, so compare the intersection explicitly
    m1 = np.asarray(sys_.store.group_rows(1), dtype=bool)
    keep = (~m0) & (~m1)
    assert np.array_equal(before[keep[~m0]], after[keep[~m0]])
    assert np.array_equal(amp_before[keep[~m0]], amp_after[keep[~m0]])


def test_decay_is_physical_and_touches_only_the_items_own_atoms():
    cfg = _tiny_cfg(stage_lifetimes=True, leak=0.3)
    sys_, sites, pays = _written_system(cfg, n=2)
    rows0 = np.asarray(sys_.store.group_rows(0), dtype=bool)
    amp = np.asarray(sys_.store.V.learned.amp).copy()
    sys_.controller.decay(1)
    sys_._sync_decay()
    amp2 = np.asarray(sys_.store.V.learned.amp)
    # DEPTH is amp**2, and a write can drive amp negative; decay scales |amp|
    # toward zero, so the depth is what must fall.
    assert np.all(amp2[rows0] ** 2 < amp[rows0] ** 2 + 1e-12)
    assert np.sum(amp2[rows0] ** 2) < np.sum(amp[rows0] ** 2)


def test_eviction_redraws_the_freed_group_instead_of_zeroing_it():
    """Zeroing starves the next item in that slot AND leaves a membership trace;
    a fresh draw makes an evicted slot indistinguishable from a never-used one."""
    cfg = _tiny_cfg()
    sys_ = build_system(cfg, loud=False)
    sites = np.asarray(designed_sites(cfg.addr_dim, 3, R=1.0, seed=0))
    sys_.write_stream([{"item_id": 0, "address": sites[0], "payload": 0.5}])
    rows = np.asarray(sys_.store.group_rows(0), dtype=bool)
    sys_.write_stream([{"item_id": 0, "delete": True}])
    centers = np.asarray(sys_.store.V.learned.centers)[rows]
    amps = np.asarray(sys_.store.V.learned.amp)[rows]
    assert not np.allclose(centers, 0.0), "freed rows were zeroed, not re-drawn"
    assert np.allclose(amps, cfg.atom_depth_init ** 0.5, atol=1e-6)
    assert np.std(centers) > 0.1  # a genuine scatter, like a never-used slot


# -- the read ---------------------------------------------------------------
def test_read_returns_the_trajectory_and_the_settled_point():
    """⭐ The API decision that makes pillar 1 testable at all."""
    sys_, sites, pays = _written_system()
    q = np.zeros((2, sys_.store.dim), dtype=np.float32)
    q[:, : sys_.cfg.addr_dim] = sites[:2]
    res = sys_.read(q)
    assert res.traj.ndim == 3 and res.traj.shape[0] == 2
    assert res.traj.shape[-1] == 2 * sys_.store.dim
    assert res.traj.shape[1] == res.phase.size
    assert set(np.unique(res.phase)) <= {1, 2}
    assert res.state.q_star.shape == (2, sys_.store.dim)
    assert res.n_steps > 0 and res.residual.shape == (2,)


def test_point_and_trajectory_psi_are_a_configuration_not_a_rewrite():
    sys_, sites, pays = _written_system()
    q = np.zeros((2, sys_.store.dim), dtype=np.float32)
    q[:, : sys_.cfg.addr_dim] = sites[:2]
    res = sys_.read(q)
    v_point = settled_point_psi(sys_.cfg.addr_dim)(res.traj, res.state)
    v_traj = tail_mean_psi(sys_.cfg.addr_dim)(res.traj, res.state)
    assert v_point.shape == v_traj.shape == (2, 1)


def test_store_relative_trajectory_removes_the_query_embedding():
    """doctrine I-2: the raw buffer CONTAINS q0 = phi(x); the launder needs the
    store-relative form or a blank-store psi read is a classifier on phi(x)."""
    sys_, sites, pays = _written_system()
    q = np.zeros((2, sys_.store.dim), dtype=np.float32)
    q[:, : sys_.cfg.addr_dim] = sites[:2]
    res = sys_.read(q)
    rel = np.asarray(store_relative_trajectory(res.traj, res.state))
    q0 = np.asarray(res.state.q0)
    traj = np.asarray(res.traj)
    # every recorded q is expressed relative to the launch point, so a psi over
    # the store-relative buffer cannot read phi(x) off the trajectory directly
    assert np.allclose(rel[:, :, : sys_.store.dim] + q0[:, None, :],
                       traj[:, :, : sys_.store.dim], atol=1e-5)
    assert not np.allclose(rel[:, :, : sys_.store.dim],
                           traj[:, :, : sys_.store.dim], atol=1e-3)


def test_a_read_schedule_must_return_to_the_stored_landscape():
    sys_, _, _ = _written_system()
    q = np.zeros((1, sys_.store.dim), dtype=np.float32)
    with pytest.raises(ValueError):
        sys_.read(q, schedule=[2.0, 1.5])
    sys_.read(q, schedule=[2.0, 1.0])  # returns => allowed


def test_fixed_point_residual_is_small_at_a_settled_point():
    """The DEQ hook: a settle is a fixed point of the damped Verlet map."""
    sys_, sites, pays = _written_system()
    q = np.zeros((1, sys_.store.dim), dtype=np.float32)
    q[:, : sys_.cfg.addr_dim] = sites[:1]
    res = sys_.read(q)
    r = np.asarray(sys_.fixed_point_residual(res.state.q_star[0], res.state.p_star[0]))
    rng = np.random.default_rng(0)
    far = np.asarray(rng.normal(size=(sys_.store.dim,)), dtype=np.float32) * 1.5
    far_r = np.asarray(sys_.fixed_point_residual(far, np.zeros(sys_.store.dim)))
    assert np.linalg.norm(r) < np.linalg.norm(far_r)


def test_fixed_point_residual_is_differentiable():
    sys_, sites, pays = _written_system()
    q = jnp.asarray(np.pad(sites[0], (0, sys_.store.dim - sys_.cfg.addr_dim)),
                    dtype=jnp.float32)
    p = jnp.zeros_like(q)
    g = jax.grad(lambda z: jnp.sum(sys_.fixed_point_residual(z, p) ** 2))(q)
    assert np.all(np.isfinite(np.asarray(g)))


# -- the dividend and its controls -----------------------------------------
def test_settle_deleted_launder_is_argmin_over_the_stores_own_keys():
    keys = np.array([[0.0, 0.0], [1.0, 0.0]])
    pays = np.array([[0.5], [-0.5]])
    q = np.array([[0.1, 0.0], [0.9, 0.0]])
    assert np.allclose(settle_deleted_launder(keys, pays, q), [[0.5], [-0.5]])
    assert np.array_equal(settle_deleted_launder(keys, pays, q, metric="assign"),
                          [0, 1])


def test_same_keys_null_destroys_content_but_not_addresses():
    keys = np.array([[0.0, 0.0], [1.0, 0.0]])
    pays = np.array([[0.5], [-0.5]])
    q = np.array([[0.1, 0.0], [0.9, 0.0]])
    out = same_keys_null(keys, pays, q, np.random.default_rng(1))
    assert set(np.round(out.ravel(), 6)) <= {0.5, -0.5}


def test_dividend_report_carries_its_controls():
    rep = dividend(0.8, 0.9, controls={"blank_store": 0.12, "chance": 0.125})
    assert rep.dividend == pytest.approx(-0.1)
    md = rep.to_markdown()
    assert "blank_store" in md and "-0.1" in md


def test_byte_account_is_honest_about_the_ratio():
    sys_, sites, pays = _written_system()
    ids, centers, p = sys_.codebook()
    ba = byte_account(sys_, centers, p)
    assert ba.full_bytes > ba.launder_bytes  # the learned store is NOT free
    assert not ba.matched()  # and the harness says so rather than hiding it


# -- monitors wired to the system -------------------------------------------
def test_observe_produces_a_reading_for_every_monitor():
    sys_, _, _ = _written_system()
    readings = sys_.observe(stage="test")
    assert len(readings) == len(sys_.registry.monitors)
    assert all(r.band for r in readings)


def test_self_probe_is_label_free_and_reports_acquisition():
    sys_, _, _ = _written_system()
    probe = sys_.self_probe()
    assert 0.0 <= probe["acq"] <= 1.0
    assert probe["n_probed"] > 0
    assert "retention" in probe and "payload_abs" in probe


def test_consolidate_runs_the_certificates_and_the_monitors():
    sys_, _, _ = _written_system()
    rep = sys_.consolidate()
    assert "sep_over_sigma_q" in rep.certificates
    assert rep.readings and rep.self_probe
