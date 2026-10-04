"""Tests for the N-unit SHARDED STORE (``chlu.core.shard_store``, w25).

These pin the five build items the additivity verdict rests on, and each one is a
trap the theorist's §2.3 code-path audit flagged:

  * the shard container is a **product store** — ``CLULattice(units, edges=(),
    couplings=())`` — whose kinetic term is **exactly** separable per unit, and the
    **relativistic wide-unit path is refused** (one global Lorentz factor couples
    every block; §2.3 #6);
  * the **N98 localized atom init** puts group ``j``'s atoms near site ``j`` and,
    with the flag off, leaves the historical construction **bit-identical**;
  * the **global allocator** is a registry: its spacing test runs against the UNION
    of every shard's addresses, and the per-shard control lets the union collapse;
  * ⛔ **R1 is not shipped** (N97: post-settle energy routes at or below chance) and
    asking for it raises; R2 routes exactly on a designed 2-shard store;
  * **per-shard query noise** is ``sigma/sqrt(d_shard)``, never ``sigma/sqrt(N*d)``
    (§2.3 #10 — the fairness trap that would hand the sharded arm free precision).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import get_default_config
from chlu.core.controller import Controller
from chlu.core.lattice import CLULattice
from chlu.core.memory_potentials import (
    AtomDictionaryPotential,
    AtomStorePotential,
    BallRegisterPotential,
    designed_payloads,
    site_separation,
)
from chlu.core.shard_store import (
    ROUTERS,
    ShardedRegistry,
    ShardedStore,
    allocate_sites,
    assert_per_shard_query_noise,
    build_sharded_store,
    r2_scores,
    route_from_scores,
    router_scores,
    shard_partition,
    sharded_two_phase,
    spread_partition,
)
from chlu.experiments.exp_sharded_store import (
    ARMS,
    apply_quick,
    arm_spec,
    build_designed_shards,
    item4_allocator,
    parse_cell,
    score_sharded,
)
from chlu.experiments.goldstone_harness import clu_with_potential


# ---------------------------------------------------------------------------
# Build item 1 -- the container is a PRODUCT store, never a wide relativistic unit
# ---------------------------------------------------------------------------


def _designed_shards(d=2, K=4, n=2, seed=0):
    centers, groups, _ = allocate_sites(d, K, n, R=1.0, seed=seed)
    pay = np.asarray(designed_payloads(K, seed=0))
    Vs = [
        BallRegisterPotential(pay[g], centers[g], R=1.5, w=0.15, b=1.0, kappa=0.1)
        for g in groups
    ]
    return build_sharded_store(Vs, groups, d=d), centers, pay


def test_shard_container_is_a_product_store():
    store, _, _ = _designed_shards()
    assert isinstance(store.lattice, CLULattice)
    assert store.lattice.edges == ()
    assert store.lattice.couplings == ()
    assert store.n_shards == 2


def test_coupled_lattice_is_refused_as_a_store():
    """A 'lattice' in this codebase is by default NOT a set of shards (§2.3 #7)."""
    from chlu.core.lattice import SpringCoupling

    V = BallRegisterPotential(np.zeros(2), np.zeros((2, 2)), R=1.5)
    units = [clu_with_potential(V, dim=3), clu_with_potential(V, dim=3)]
    spring = SpringCoupling(W_i=jnp.eye(3), W_j=jnp.eye(3), kappa=0.1)
    coupled = CLULattice(units, edges=((0, 1),), couplings=(spring,))
    with pytest.raises(ValueError, match="PRODUCT store"):
        ShardedStore(coupled, [[0], [1]], d=2)


def test_relativistic_shards_are_refused():
    """⛔ §2.3 #6: one relativistic unit of dim N*d couples every block through a
    single Lorentz factor (off-block d2T/dp_i dp_j = 8.4e-2, block speed falls 11x
    with hot neighbours). Shards must be lattice units, and the relativistic path is
    refused rather than warned about in a docstring."""
    V = BallRegisterPotential(np.zeros(2), np.zeros((2, 2)), R=1.5)
    with pytest.raises(ValueError, match="relativistic"):
        build_sharded_store([V, V], [[0], [1]], d=2, kinetic_mode="relativistic")


def test_per_unit_kinetic_energy_is_exactly_separable():
    """The whole reason shards are CLULattice units: T_net = sum_r T_r(p_r) with
    ZERO off-block second derivative."""
    store, _, _ = _designed_shards()
    lat = store.lattice
    H = jax.hessian(lambda p: lat.T(p))(jnp.linspace(0.3, 2.0, lat.dim))
    dim = store.unit_dim
    off = np.asarray(H)[:dim, dim:]
    assert np.max(np.abs(off)) == 0.0


# ---------------------------------------------------------------------------
# Build item 2 -- the N98 localized atom init
# ---------------------------------------------------------------------------


def test_localized_atom_init_is_bit_identical_when_disabled():
    """Default construction must be UNCHANGED — the localized draw uses a folded
    key so even the RNG stream of the default path is untouched."""
    k = jax.random.PRNGKey(0)
    a = AtomDictionaryPotential(3, 24, k, n_groups=4)
    b = AtomDictionaryPotential(
        3, 24, k, n_groups=4,
        group_centers=np.zeros((4, 2)), local_radius=0.0,  # radius 0 => disabled
    )
    assert np.array_equal(np.asarray(a.centers), np.asarray(b.centers))
    assert np.array_equal(np.asarray(a.amp), np.asarray(b.amp))


def test_localized_atom_init_places_each_group_near_its_own_site():
    """N98's fix: group j's atoms start in a ball of radius r around site j, in the
    ADDRESS axes only — the payload axis keeps the w23 scatter (basin reach)."""
    k = jax.random.PRNGKey(1)
    sites = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    r = 0.6
    V = AtomDictionaryPotential(
        3, 40, k, n_groups=4, group_centers=sites, local_radius=r
    )
    c = np.asarray(V.centers)
    owner = np.clip(np.searchsorted(
        np.array([round(g * 40 / 4) for g in range(5)]), np.arange(40), side="right"
    ) - 1, 0, 3)
    dist = np.linalg.norm(c[:, :2] - sites[owner], axis=1)
    assert dist.max() <= r + 1e-5
    # every atom is nearest to its OWN site (the property that restores W3)
    d2 = ((c[:, None, :2] - sites[None, :, :]) ** 2).sum(-1)
    assert np.array_equal(np.argmin(d2, axis=1), owner)
    # the payload axis is NOT localized (it still spans the scatter)
    assert np.std(c[:, 2]) > 0.5


def test_localized_init_rejects_a_mismatched_group_count():
    with pytest.raises(ValueError, match="rows for"):
        AtomDictionaryPotential(
            3, 12, jax.random.PRNGKey(0), n_groups=4,
            group_centers=np.zeros((3, 2)), local_radius=0.5,
        )


# ---------------------------------------------------------------------------
# Build item 3 -- the global allocator is a REGISTRY across shards
# ---------------------------------------------------------------------------


def test_global_registry_spacing_test_sees_the_union():
    """The admission gate of shard 0 must refuse a site that is too close to an
    address held by shard 1 — that is the whole content of condition W4."""
    s, d_safe = 0.35, 1.54
    ctls = [
        Controller(AtomStorePotential(dim=3, capacity=8, s=s), d_safe=d_safe,
                   budget=8, n_candidates=0)
        for _ in range(2)
    ]
    reg = ShardedRegistry(ctls, global_alloc=True)
    reg.offer(1, 0, np.array([0.0, 0.0]), 0.5)
    # a near-coincident proposal into the OTHER shard must be refused on spacing
    row = reg.offer(0, 1, np.array([0.05, 0.0]), -0.5)
    assert row["decision"] == "refuse_spacing"
    # with no peer view (per-shard registries) the same proposal is admitted
    ctls2 = [
        Controller(AtomStorePotential(dim=3, capacity=8, s=s), d_safe=d_safe,
                   budget=8, n_candidates=0)
        for _ in range(2)
    ]
    reg2 = ShardedRegistry(ctls2, global_alloc=False)
    reg2.offer(1, 0, np.array([0.0, 0.0]), 0.5)
    assert reg2.offer(0, 1, np.array([0.05, 0.0]), -0.5)["decision"] == "admit"


def test_controller_default_is_unchanged_without_peers():
    """The Controller hook is opt-in: no peer function => historical behaviour."""
    ctl = Controller(AtomStorePotential(dim=3, capacity=4, s=0.35), d_safe=1.54)
    assert ctl.peer_addresses_fn is None
    row = ctl.offer(0, np.array([0.0, 0.0]), 0.5)
    assert row["decision"] == "admit"


def test_allocator_item_global_beats_local_on_union_separation():
    cfg = get_default_config()
    apply_quick(cfg)
    out = item4_allocator(cfg.experiment_designed_mechanism, cfg.experiment_sharded_store)
    g, loc = out["regimes"]["global"], out["regimes"]["local"]
    assert g["union_separation"] >= loc["union_separation"]
    assert g["union_respects_d_safe"]


def test_global_allocation_separates_the_union_better_than_local():
    _, _, g = allocate_sites(2, 16, 4, R=1.0, seed=0, allocation="global")
    _, _, loc = allocate_sites(2, 16, 4, R=1.0, seed=0, allocation="local")
    assert g["union_separation"] > loc["union_separation"]
    # ...while LOCAL flatters the within-shard number: the exact trap Prop L4 warns
    # about (write-side crowding looks solved, read-side discrimination collapses)
    assert loc["within_shard_separation_min"] > g["within_shard_separation_min"]


def test_spread_partition_is_balanced_and_total():
    c = np.asarray(allocate_sites(3, 12, 1, R=1.0, seed=0)[0])
    groups = spread_partition(c, 4)
    assert sorted(np.concatenate(groups).tolist()) == list(range(12))
    assert {len(g) for g in groups} == {3}
    # a spread partition must not be WORSE than contiguous chunking on the
    # within-shard separation (that is the write-side quantity it buys)
    contig = shard_partition(c, 4, strategy="contiguous")
    assert min(site_separation(c[g]) for g in groups) >= min(
        site_separation(c[g]) for g in contig
    ) - 1e-6


# ---------------------------------------------------------------------------
# Build item 4 -- routers R2 / R3 only, plus the abstention deadband
# ---------------------------------------------------------------------------


def test_r1_router_is_not_shipped():
    """⛔ N97: post-settle energy routes AT OR BELOW chance at equal well depth.

    ``RG`` (distance to the registry) is shipped alongside R2/R3 because the
    theorist's own O(1) condition names it ("pre-settle energy, distance to the
    registry, or an explicit tag") — it is declared classical NN indexing (N89).
    """
    assert ROUTERS == ("R2", "R3", "RG")
    store, _, _ = _designed_shards()
    with pytest.raises(ValueError, match="N97"):
        router_scores("R1", store, jnp.zeros((1, 3)))


def test_registry_router_is_exact_and_needs_no_dynamics():
    """RG routes on the addresses the WRITER RECORDED — no rollout, no learned
    parameters. It is exact here because the union separation (0.71-0.91 in the
    swept cells) dwarfs the query jitter."""
    store, centers, _ = _designed_shards(d=2, K=8, n=2)
    Q0 = np.zeros((8, 3), dtype=np.float32)
    Q0[:, :2] = centers + 0.02
    sc = router_scores("RG", store, Q0, centers=centers)
    route, _m, _ab = route_from_scores(sc)
    assert np.mean(route == store.item_shard_map(8)) == 1.0
    with pytest.raises(ValueError, match="registry"):
        router_scores("RG", store, Q0)


def test_r3_is_degenerate_on_a_flat_vacuum_store():
    """⚠ MEASURED w25 (and it contradicts the theorist's R3 = 1.000): on the shipped
    `BallRegisterPotential` — whose confinement is **identically flat inside the
    address ball by design** — R3 (settling displacement) is not a discriminating
    statistic, while R2 (pre-settle energy) discriminates by orders of magnitude.

    The reason is structural. At an item's site the OWNING shard's gradient is ~0
    (it is a minimum) and a FOREIGN shard's gradient is also ~0 (flat vacuum plus an
    exponentially small well tail) — the two displacement scales are comparable, so
    ``argmin_r ||x_final - q||`` is choosing between two near-equal small numbers.
    The ENERGY, by contrast, differs by ~4 orders. Measured consequence at d=4 K=32,
    global allocation: R3 strict 1.000 / 0.549 / 0.236 / 0.228 at N = 1/2/4/8 while
    R2 stays 1.000 throughout. This test pins the mechanism, not the accuracy, so it
    stays cheap.
    """
    store, centers, _ = _designed_shards(d=2, K=8, n=2)
    own, far = store.shard_of(0), 1 - store.shard_of(0)
    q = jnp.asarray(np.concatenate([centers[0], [0.0]]).astype(np.float32))
    v_own = float(store.lattice.units[own].potential_net(q))
    v_far = float(store.lattice.units[far].potential_net(q))
    g = [
        float(jnp.linalg.norm(jax.grad(lambda x, u=u: u.potential_net(x))(q)[:2]))
        for u in (store.lattice.units[own], store.lattice.units[far])
    ]
    # R2 separates by >= 3 orders of magnitude...
    assert abs(v_own) > 1000 * abs(v_far)
    # ...while the forces that drive the displacement are both ~0 and within an
    # order of magnitude of each other => R3 has essentially no signal to use.
    assert max(g) < 1e-2 and max(g) < 20 * max(min(g), 1e-12)


def test_r2_routes_a_designed_two_shard_store_exactly():
    """R2 = argmin_r V_r(q), evaluated WITHOUT running the dynamics."""
    store, centers, _ = _designed_shards(d=2, K=8, n=2)
    Q0 = np.zeros((8, 3), dtype=np.float32)
    Q0[:, :2] = centers
    route, margin, abstain = route_from_scores(r2_scores(store, jnp.asarray(Q0)))
    truth = store.item_shard_map(8)
    assert np.mean(route == truth) == 1.0
    assert not abstain.any()


def test_abstention_deadband_trades_offered_for_answered():
    store, centers, _ = _designed_shards(d=2, K=8, n=2)
    Q0 = np.zeros((8, 3), dtype=np.float32)
    Q0[:, :2] = centers
    s = r2_scores(store, jnp.asarray(Q0))
    _, margin, ab0 = route_from_scores(s, 0.0)
    _, _, ab_big = route_from_scores(s, float(margin.max()) + 1.0)
    assert ab0.sum() == 0
    assert ab_big.all()


# ---------------------------------------------------------------------------
# Build item 5 -- per-shard query noise (the §2.3 #10 fairness trap)
# ---------------------------------------------------------------------------


def test_per_shard_query_noise_is_not_scaled_by_the_joint_dimension():
    sigma, d, n = 0.15, 4, 8
    assert_per_shard_query_noise(sigma, d, n, sigma / np.sqrt(d))  # correct
    with pytest.raises(AssertionError, match="fairness trap"):
        assert_per_shard_query_noise(sigma, d, n, sigma / np.sqrt(d * n))


def test_scoring_asserts_the_query_noise_convention():
    """``score_sharded`` runs the assertion on every cell it measures."""
    cfg = get_default_config()
    apply_quick(cfg)
    dm, ss = cfg.experiment_designed_mechanism, cfg.experiment_sharded_store
    store, centers, payloads, _ = build_designed_shards(2, 4, 2, dm, ss, "global", 0)
    got = score_sharded(store, centers, payloads, dm, ss, 0)
    assert 0.0 <= got["strict_success_rate"] <= 1.0
    assert got["n_shards"] == 2
    assert got["finite"]


# ---------------------------------------------------------------------------
# The read path: ONE joint rollout settles every shard
# ---------------------------------------------------------------------------


def test_joint_rollout_equals_per_shard_rollouts():
    """V is separable, so the joint lattice rollout must reproduce, block by block,
    what each shard would have done alone — this is why the read is O(1) in DEPTH."""
    cfg = get_default_config()
    apply_quick(cfg)
    dm = cfg.experiment_designed_mechanism
    store, centers, _ = _designed_shards(d=2, K=4, n=2)
    Q0 = np.zeros((2, 3), dtype=np.float32)
    Q0[:, :2] = centers[:2]
    P0 = np.zeros_like(Q0)
    addr_x, feat = sharded_two_phase(store, jnp.asarray(Q0), jnp.asarray(P0), dm)
    assert addr_x.shape == (2, 2, 2) and feat.shape == (2, 2, dm.n_subsample)
    for r, unit in enumerate(store.lattice.units):
        tr1 = unit(jnp.asarray(Q0[0]), jnp.asarray(P0[0]), dm.address_steps,
                   dm.dt, dm.gamma_address)
        aq, ap = tr1[-1, :3], tr1[-1, 3:]
        tr2 = unit(aq, ap, dm.read_steps, dm.dt, dm.gamma_read)
        np.testing.assert_allclose(
            np.asarray(tr2[-1, :2]), addr_x[0, r], rtol=1e-4, atol=1e-5
        )


def test_monolithic_arm_reproduces_the_w23_read_path():
    """⭐ The laundering control must be the SAME store the w23/w24 numbers came
    from: at N=1 the sharded read path must agree with ``exp_designed_mechanism``'s
    two-phase read to float32 rounding, or 'sharded beats monolithic' could be an
    artifact of two different readers."""
    from chlu.core.memory_potentials import designed_sites
    from chlu.experiments.exp_designed_mechanism import (
        _two_phase, build_designed_model, make_ball_queries,
    )

    cfg = get_default_config()
    dm = cfg.experiment_designed_mechanism
    dm.n_query_per_item, dm.max_total_queries = 2, 16
    d, K = 4, 8
    centers = designed_sites(d, K, R=dm.R, seed=dm.site_seed)
    pay = designed_payloads(K, seed=dm.payload_seed)
    V = BallRegisterPotential(
        pay, centers, R=dm.R + dm.wall_margin, w=dm.well_width, b=dm.well_depth,
        kappa=dm.payload_kappa, c_conf=dm.c_conf,
    )
    store = build_sharded_store([V], [np.arange(K)], d=d)
    Q0, P0, _ = make_ball_queries(jax.random.PRNGKey(0), centers, 2, dm)
    a1, f1 = _two_phase(build_designed_model(centers, pay, dm), Q0, P0, dm, d)
    a2, f2 = sharded_two_phase(store, Q0, P0, dm)
    np.testing.assert_allclose(a1, a2[:, 0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(f1, f2[:, 0], rtol=0, atol=1e-5)


def test_monolithic_arm_is_the_n_equals_one_case():
    """Both arms travel the SAME read path, so no comparison can be contaminated by
    two different readers."""
    store, centers, payloads = _designed_shards(d=2, K=4, n=1)
    assert store.n_shards == 1
    assert store.shard_items == (tuple(range(4)),)


# ---------------------------------------------------------------------------
# Arm bookkeeping: what "matched" and "per-shard" atom budgets mean
# ---------------------------------------------------------------------------


def test_arm_atom_budgets():
    dm = get_default_config().experiment_designed_mechanism
    d, K, n = 6, 64, 2
    mono_n, mono_a, mono_t = arm_spec("monolithic", d, K, n, dm)
    m_n, m_a, m_t = arm_spec("sharded_matched", d, K, n, dm)
    p_n, p_a, p_t = arm_spec("sharded_per_shard", d, K, n, dm)
    nx_n, nx_a, nx_t = arm_spec("monolithic_nx", d, K, n, dm)
    assert mono_n == 1 and nx_n == 1
    assert m_n == n and p_n == n
    # parameter-matched: the sharded arm may not spend more atoms than the baseline
    assert m_t == mono_t
    # per-shard: each shard gets the budget a monolithic K/N store would get, and
    # its laundering line (monolithic_nx) has EXACTLY the same total
    assert p_t == nx_t
    assert p_a == arm_spec("monolithic", d, K // n, 1, dm)[1]


def test_parse_cell_rejects_indivisible_loads():
    assert parse_cell("6:64:2") == (6, 64, 2)
    with pytest.raises(ValueError, match="divisible"):
        parse_cell("6:65:2")
    with pytest.raises(ValueError, match="d:K:n_shards"):
        parse_cell("6:64")
    assert set(ARMS) == {
        "monolithic", "monolithic_nx", "sharded_matched", "sharded_per_shard"
    }


def test_config_group_is_registered_and_round_trips(tmp_path):
    from chlu.config import load_config, save_config

    cfg = get_default_config()
    cfg.experiment_sharded_store.cells = ["4:16:4"]
    cfg.experiment_sharded_store.abstain_deadband = 0.25
    cfg.experiment_sharded_store.atom_init_local = True
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.experiment_sharded_store.cells == ["4:16:4"]
    assert back.experiment_sharded_store.abstain_deadband == 0.25
    assert back.experiment_sharded_store.atom_init_local is True
