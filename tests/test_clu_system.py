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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.clu_system import (
    CluSystemConfig,
    LearnedVStore,
    build_system,
    normalize_write_objective,
    resolve_store_potential_factory,
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


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 for the whole module, restoring the global flag after.

    ⚠ Repo-wide test-isolation hazard (handover §7.2, and it bit this file):
    several test modules enable ``jax_enable_x64`` at MODULE import, so x64 is
    globally ON in a full-suite run even though it is off when this file runs
    alone. The harness stores and reads in float32 by construction, and the
    settle residual it asserts on is a float32 quantity — under x64 the same
    assertion compares a different number and
    ``test_fixed_point_residual_is_small_at_a_settled_point`` failed in the full
    suite while passing in isolation. Same fixture as
    ``test_hopfield_capacity`` / ``test_retry_compute`` / ``test_phi_stream``.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


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


# ==========================================================================
# C2W2 RACE+SEAM FREEZE — the two seams the wave's other branches build on.
#
# ⛔ Both are additive and default-off, and the two "when off it is
# bit-identical" tests below are BLOCKING: if the shipped write or the shipped
# store moved by a single bit, every C2W2 number measured against the
# `endpoint_write` / `gauss` control would be uninterpretable.
# ==========================================================================
def _store_leaves(store):
    import equinox as eqx

    return [np.asarray(x) for x in
            jax.tree_util.tree_leaves(eqx.filter(store.V, eqx.is_inexact_array))]


def test_seam_a_write_objective_defaults_to_the_shipped_write_bit_identically():
    """⛔ SEAM (a) OFF => the written ``V`` is bit-identical to the shipped one."""
    a, _, _ = _written_system()
    cfg = _tiny_cfg()
    b = build_system(cfg, loud=False, write_objective=None)
    sites = np.asarray(designed_sites(cfg.addr_dim, 3, R=cfg.ball_radius, seed=0))
    pays = np.asarray(designed_payloads(3, seed=0))
    b.write_stream([{"item_id": i, "address": sites[i], "payload": float(pays[i])}
                    for i in range(3)])
    for la, lb in zip(_store_leaves(a.store), _store_leaves(b.store), strict=True):
        assert np.array_equal(la, lb)
    # and the empty spec is the same object-level no-op as None
    assert build_system(cfg, loud=False, write_objective={}).write_objective == {}


def test_seam_a_forwards_loss_and_train_kwargs_to_the_writer():
    """The spec reaches ``train_memory_landscape``: fewer steps => shorter history
    and a different landscape, proving the passthrough is live (not decorative)."""
    cfg = _tiny_cfg()
    sys_ = build_system(cfg, loud=False,
                        write_objective={"train_kwargs": {"steps": 3}})
    assert sys_.write_objective == {"train_kwargs": {"steps": 3}}
    site = np.asarray(designed_sites(cfg.addr_dim, 1, R=cfg.ball_radius, seed=0))[0]
    sys_.write_stream([{"item_id": 0, "address": site, "payload": 0.5}])
    ref = build_system(cfg, loud=False)
    ref.write_stream([{"item_id": 0, "address": site, "payload": 0.5}])
    assert not all(np.array_equal(a, b) for a, b in
                   zip(_store_leaves(sys_.store), _store_leaves(ref.store), strict=True))


def test_seam_a_rejects_an_unknown_key_instead_of_dropping_it():
    """A silently-dropped coefficient would report as 'the term is inert' — which
    is exactly the finding the C2W2 gate must not fabricate."""
    with pytest.raises(ValueError, match="unknown write-objective key"):
        normalize_write_objective({"lambda_traj": 1.0})
    assert normalize_write_objective(None) == {}
    assert normalize_write_objective({"loss_kwargs": {}}) == {}


def test_seam_b_store_factory_defaults_to_the_shipped_potential_bit_identically():
    """⛔ SEAM (b) OFF => the store is bit-identical to the shipped one."""
    cfg = _tiny_cfg()
    assert cfg.store_potential_factory is None
    assert CluSystemConfig().as_flag_table() == {}  # the kwargs default is inert
    a = LearnedVStore(cfg, jax.random.PRNGKey(0))
    b = LearnedVStore(cfg, jax.random.PRNGKey(0))
    for la, lb in zip(_store_leaves(a), _store_leaves(b), strict=True):
        assert np.array_equal(la, lb)
    assert type(a.V).__name__ == "DesignFreedomPotential"


def test_seam_b_registers_an_external_store_family_from_config_alone():
    """A factory living in a module ``clu_system.py`` has never heard of is wired
    in by import path — this is how ``ssb-shell-atoms`` registers shell atoms
    without editing a file it does not own."""
    _SEAM_FACTORY_CALLS.clear()
    cfg = _tiny_cfg(
        store_potential_factory="tests.test_clu_system:_factory_for_seam_test",
        store_potential_kwargs={"n_atoms_override": 24},
    )
    store = LearnedVStore(cfg, jax.random.PRNGKey(0))
    assert _SEAM_FACTORY_CALLS == [{"n_atoms_override": 24, "dim": cfg.dim}]
    # the factory's store — NOT the shipped one — is what the system now runs on
    assert store.V.learned.centers.shape[0] == 24
    assert store.group_rows(0).shape[0] == 24
    assert cfg.as_flag_table()["store_potential_kwargs"] == {"n_atoms_override": 24}


#: call log for :func:`_factory_for_seam_test` (the seam is import-path resolved,
#: so the factory must live at module scope).
_SEAM_FACTORY_CALLS: list = []


def _factory_for_seam_test(*, cfg, key, n_atoms_override: int):
    """A stand-in external store family — the seam's contract, minimally.

    ``ssb-shell-atoms``'s shell-atom family plugs in exactly here.
    """
    from chlu.core.memory_potentials import DesignFreedomPotential

    _SEAM_FACTORY_CALLS.append({"n_atoms_override": int(n_atoms_override),
                                "dim": int(cfg.dim)})
    return DesignFreedomPotential(
        rung="free_mlp", dim=cfg.dim, payloads=jnp.zeros((cfg.capacity,)), key=key,
        learned_family="atoms", n_atoms=int(n_atoms_override),
        rbf_init_width=float(cfg.atom_width), confine=float(cfg.confine),
        atom_depth_init=float(cfg.atom_depth_init), atom_groups=cfg.capacity,
        atom_init_scale=float(cfg.atom_init_scale),
    )


def test_seam_b_rejects_a_bad_import_path():
    with pytest.raises(ValueError, match="pkg.module"):
        resolve_store_potential_factory("notapath")
    with pytest.raises(TypeError, match="non-callable"):
        resolve_store_potential_factory("chlu.core.clu_system:WRITE_OBJECTIVE_KEYS")
