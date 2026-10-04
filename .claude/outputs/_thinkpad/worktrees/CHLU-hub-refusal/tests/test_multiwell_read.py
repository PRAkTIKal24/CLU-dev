"""Tests for the tier-ii multi-well read protocol (charter §A20.3).

The obligations these tests encode are the ones the charter makes blocking:

* **K0** — the launch must address ``F`` distinct wells (bar 0.90), and the
  instrument must reproduce the REFUTED ``P = 4`` protocol's numbers on the same
  family (``orgdiv-null-arms`` §3: 2.20 distinct, 0.05 ``>= F``, 0.41 precision);
* the four §A20.3(c) guards, **each shown firing on a designed negative** (a
  guard that cannot fire is N74's vacuous gate);
* ⛔ no ``argmax`` in anything a fitted reader consumes;
* the launder is recomputed **live** from this task's own launches;
* ``k`` is capacity and is matched across arms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.factored_store import (CatTestConfig, FactoredStore, build_family,
                                      build_phi, place_wells, write_wells)
from chlu.core.multiwell_read import (LaunchHead, MultiWellReadConfig,
                                      aggregate_occupancy, assert_k_matched,
                                      consolidate_wells, descent_weight,
                                      find_wells, fit_readers_mw,
                                      hard_vs_soft_gradient, head_trainable_spec,
                                      launch_only_launder, multiwell_read,
                                      mwr_ledger, query_code, read_stats,
                                      s_effective, score_readers_mw,
                                      soft_occupancy, staging_gradient_probe,
                                      trash_field)


# --------------------------------------------------------------------------
# fixtures — one small written store, shared (the settle is the expensive part)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cell():
    cfg = CatTestConfig(atoms_per_well=4, addr_dim=8, n_wells=24, n_items=32,
                        n_unseen=24, f_subset=4, write_steps=25, address_steps=30,
                        read_steps=40, payload_radius=0.5,
                        atom_payload_init_radius=0.5, s_measured=0.3611)
    mw = MultiWellReadConfig(k_particles=6, payload_ref=0.5, conf_w=0.05,
                             n_probes=48, probe_steps=40, head_steps=2,
                             head_batch=8, head_settle_address=15,
                             head_settle_read=20)
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    fam = build_family(cfg, 0)
    blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(0))
    store, _ = write_wells(blank, cfg, anchors, fam.payloads,
                           jax.random.PRNGKey(1))
    head = LaunchHead(phi, anchors, cfg, mw)
    return dict(cfg=cfg, mw=mw, phi=phi, anchors=anchors, fam=fam, blank=blank,
                store=store, head=head)


# --------------------------------------------------------------------------
# K0 — the pre-condition, and the instrument's own control
# --------------------------------------------------------------------------
def test_k0_head_addresses_F_distinct_wells_and_beats_the_refuted_protocol():
    """⭐ K0: bar 0.90, store-free. The refuted ``P = 4`` read is the control.

    The refuted protocol must reproduce ``orgdiv-null-arms`` §3 on the SAME
    family (2.202 distinct / 0.050 >= F / 0.4106 precision at ``d = 4``); if the
    instrument cannot reproduce the number it is refuting, it measures nothing.
    """
    cfg4 = CatTestConfig(atoms_per_well=32, addr_dim=4)
    phi4 = build_phi(cfg4)
    anc4 = place_wells(phi4, cfg4, cfg4.target_ds * cfg4.s_measured)
    fam4 = build_family(cfg4, 0)
    ind = fam4.indicator(fam4.unseen, cfg4.n_wells)
    c = np.asarray(phi4.set_code(jnp.asarray(ind)))
    rng = np.random.default_rng(1000)
    old = (c[:, None, :] + np.asarray(phi4.offsets)[None]
           + cfg4.query_sigma * rng.normal(size=(len(c), 4, 4)))
    ref = read_stats({"z": old}, anc4, fam4.unseen, cfg4.f_subset)
    assert 2.0 < ref["distinct_wells_raw"] < 2.5      # published 2.202
    assert ref["ge_F_distinct_raw"] < 0.10            # published 0.050
    assert 0.35 < ref["occupancy_precision_raw"] < 0.47  # published 0.4106
    assert ref["exact_set_occupancy_raw"] == 0.0      # published 0.0000/2560

    cfg8 = CatTestConfig(atoms_per_well=32, addr_dim=8, s_measured=0.3611)
    mw = MultiWellReadConfig(k_particles=12)
    phi8 = build_phi(cfg8)
    anc8 = place_wells(phi8, cfg8, cfg8.target_ds * cfg8.s_measured)
    fam8 = build_family(cfg8, 0)
    cn = (np.asarray(phi8.set_code(jnp.asarray(
        fam8.indicator(fam8.unseen, cfg8.n_wells))))
        + cfg8.query_sigma * np.random.default_rng(1000).normal(
            size=(len(fam8.unseen), 8)))
    head = LaunchHead(phi8, anc8, cfg8, mw)
    q0 = np.asarray(jax.vmap(head)(jnp.asarray(cn, jnp.float32))[0])
    new = read_stats({"z": q0}, anc8, fam8.unseen, cfg8.f_subset)
    assert new["ge_F_distinct_raw"] >= 0.90, "K0 bar (charter §A20.3 DIAL)"
    assert new["distinct_wells_raw"] > 4.0 * ref["distinct_wells_raw"]


# --------------------------------------------------------------------------
# (b) the read latent — ⛔ no quantisation anywhere a reader can see
# --------------------------------------------------------------------------
def test_the_latent_a_reader_sees_is_continuous(cell):
    """A finite perturbation of the query must move the latent continuously.

    ``orgdiv-null-arms`` §4 named quantisation as the destructive step; this is
    the property that says the new latent does not do it. ``pi`` is a softmax and
    ``z`` is a settled state, so a small change in the code must produce a small
    change in both — never a jump between codebook entries.
    """
    c = cell
    d = c["cfg"].addr_dim
    z = jnp.asarray(np.random.default_rng(0).normal(size=(3, 5, d)), jnp.float32)
    pi = soft_occupancy(z, c["head"].anchors, 0.25)
    assert pi.shape == (3, 5, c["cfg"].n_wells)
    np.testing.assert_allclose(np.asarray(pi.sum(-1)), 1.0, atol=1e-5)
    pi2 = soft_occupancy(z + 1e-4, c["head"].anchors, 0.25)
    assert float(jnp.abs(pi2 - pi).max()) < 5e-2   # continuous, not a jump
    # and it is differentiable everywhere (an argmax is not)
    g = jax.grad(lambda x: soft_occupancy(x, c["head"].anchors, 0.25).sum())(z)
    assert np.all(np.isfinite(np.asarray(g)))


def test_dedupe_verb_is_a_set_union_not_a_multiset_sum(cell):
    """⭐ Two particles in one well must count ONCE.

    The refuted read summed the ``P`` particles, so a well visited twice was
    added twice while two of ``A(x)``'s wells went unvisited. ``noisy_or``/``max``
    are set unions; ``sum`` is kept only as the designed negative.
    """
    pi = jnp.zeros((1, 3, 4)).at[0, :, 1].set(1.0)      # 3 particles, all well 1
    w = jnp.ones((1, 3))
    assert float(aggregate_occupancy(pi, w, "noisy_or")[0, 1]) == pytest.approx(1.0, abs=1e-4)
    assert float(aggregate_occupancy(pi, w, "max")[0, 1]) == pytest.approx(1.0, abs=1e-6)
    assert float(aggregate_occupancy(pi, w, "sum")[0, 1]) == pytest.approx(3.0, abs=1e-6)


def test_descent_weight_is_finite_at_the_pinned_launch(cell):
    """The launch pins the payload block to 0 and ``grad ||.||`` at 0 is NaN."""
    z = jnp.zeros((2, 3, 4))
    w = descent_weight(z, 0.5)
    assert np.all(np.isfinite(np.asarray(w)))
    g = jax.grad(lambda x: descent_weight(x, 0.5).sum())(z)
    assert np.all(np.isfinite(np.asarray(g)))


def test_read_stats_gated_set_is_what_R1_scores(cell):
    """The raw exact-set statistic is 0 by construction at ``k > F``; R1 uses the
    GATED effective set, which is the thing the read actually asserts."""
    c = cell
    lat = multiwell_read(c["store"], c["head"], c["phi"], c["cfg"], c["mw"],
                         c["fam"].indicator(c["fam"].unseen, c["cfg"].n_wells),
                         jax.random.PRNGKey(2))
    st = read_stats(lat, c["anchors"], c["fam"].unseen, c["cfg"].f_subset)
    assert st["distinct_wells_raw"] > c["cfg"].f_subset
    assert st["exact_set_occupancy_raw"] == 0.0
    assert "exact_set_occupancy_gated" in st
    assert st["distinct_wells_gated"] <= st["distinct_wells_raw"] + 1e-9


# --------------------------------------------------------------------------
# (c) the four guards — each MUST fire on a designed negative
# --------------------------------------------------------------------------
def test_guard1_launder_is_recomputed_live_from_this_tasks_own_launches(cell):
    """⛔ Advisor amendment A1: the bar is THIS head's launches with the store
    deleted — never the old protocol's stale 0.272 reference."""
    c = cell
    ind = c["fam"].indicator(c["fam"].unseen, c["cfg"].n_wells)
    lat = multiwell_read(c["store"], c["head"], c["phi"], c["cfg"], c["mw"], ind,
                         jax.random.PRNGKey(2))
    lau = launch_only_launder(c["head"], c["phi"], c["cfg"], c["mw"], ind,
                              jax.random.PRNGKey(2))
    # the launder IS the read's own launch, bit-identically
    np.testing.assert_array_equal(lau["z"], lat["q0"])
    # ... and it is NOT the settled read (the store did something)
    assert not np.array_equal(lau["z"], lat["z"])
    # designed negative: a read that never settles cannot beat its own launder
    zero = multiwell_read(c["store"], c["head"], c["phi"],
                          CatTestConfig(**{**c["cfg"].as_dict(), "address_steps": 0,
                                           "read_steps": 0}),
                          c["mw"], ind, jax.random.PRNGKey(2))
    np.testing.assert_allclose(zero["z"], lau["z"], atol=1e-6)


def test_guard2_hard_assignments_do_not_backprop(cell):
    """⛔ Guard 2 fires on its designed negative: with a hard (argmax) assignment
    the gradient through the occupancy channel is EXACTLY zero."""
    c = cell
    out = hard_vs_soft_gradient(c["store"], c["head"], c["phi"], c["cfg"], c["mw"],
                                c["fam"], jax.random.PRNGKey(5), n=4)
    assert out["grad_soft"] > 0.0
    assert out["grad_hard"] == 0.0
    assert out["ratio_hard_over_soft"] == 0.0


def test_guard3_staging_the_head_needs_a_written_store(cell):
    """⛔ Guard 3's designed negative: on a blank store the read gradient is
    orders of magnitude weaker than on a written one, so joint (un-staged)
    co-training has nothing to descend."""
    c = cell
    out = staging_gradient_probe(c["blank"], c["store"], c["head"], c["phi"],
                                 c["cfg"], c["mw"], c["fam"],
                                 jax.random.PRNGKey(5), n=4)
    assert out["grad_head_written"] > out["grad_head_blank"]
    assert out["head_ratio_blank_over_written"] < 0.5


def test_guard4_k_is_capacity_and_must_be_matched(cell):
    """⛔ Guard 4: ``k`` is on the byte ledger of every arm and a mismatch RAISES."""
    c = cell
    led_a = mwr_ledger("a", c["cfg"], c["mw"], store_params=10, head=c["head"],
                       phi_bytes=576)
    assert led_a["k_particles"] == c["mw"].k_particles
    assert led_a["read_flops_per_query"] > 0
    assert assert_k_matched([led_a, dict(led_a, arm="b")]) == c["mw"].k_particles
    mw2 = MultiWellReadConfig(k_particles=2 * c["mw"].k_particles)
    led_b = mwr_ledger("b", c["cfg"], mw2, store_params=10, head=c["head"],
                       phi_bytes=576)
    with pytest.raises(ValueError, match="k is capacity"):
        assert_k_matched([led_a, led_b])


# --------------------------------------------------------------------------
# (d) consolidate-to-budget + the trash region's first use
# --------------------------------------------------------------------------
def test_consolidate_merges_to_budget_and_trashes_spurious_wells():
    """Mechanical criteria only; a spurious shallow well is TRASHED, never merged
    into a meaningful one (its centre must not move)."""
    centers = np.array([[0.0, 0.0], [0.05, 0.0], [3.0, 0.0], [9.0, 9.0]])
    depths = np.array([1.0, 0.9, 0.8, 0.001])
    con = consolidate_wells(centers, depths, budget=2, merge_radius=0.5,
                            trash_depth_frac=0.2)
    assert con["n_found"] == 3          # the two near-duplicates merged
    assert con["n_kept"] == 2
    assert con["n_trashed_over_budget"] == 1
    # the deepest kept centre is unmoved by the spurious well
    assert np.linalg.norm(con["kept_centers"][0] - centers[0]) < 0.05
    # and pruning BELOW budget is a controller decision, not an accident
    con2 = consolidate_wells(centers, depths, budget=4, merge_radius=0.5,
                             trash_depth_frac=0.2)
    assert con2["n_trashed_shallow"] == 1
    assert con2["n_kept"] == 2


def test_trash_field_is_exactly_inert_outside_its_horizons():
    """F5 Prop-11 with ``gate='compact'``: ``gamma_phi == 0`` EXACTLY outside, so
    a trash region cannot silently change the read anywhere else."""
    tf = trash_field(np.array([[1.0, 0.0]]), 2, radius=0.3, strength=0.4)
    assert tf is not None and tf.k == 1
    assert float(tf(jnp.array([1.0, 0.0]))) > 0.1
    assert float(tf(jnp.array([5.0, 0.0]))) == 0.0
    assert trash_field(np.zeros((0, 2)), 2, radius=0.3, strength=0.4) is None


def test_s_effective_band_rule():
    cfg = CatTestConfig()
    assert s_effective(cfg.n_wells, cfg) == pytest.approx(
        cfg.n_items * cfg.f_subset / cfg.n_wells)
    assert s_effective(10, cfg) > 16.0      # the cat-test's COLLAPSED regime


def test_find_wells_depth_is_confinement_subtracted(cell):
    """§7.28 (program-wide ruler): without subtracting ``alpha||q||^2`` the bowl
    alone log-fits as a well (1.44x inflation measured on this store)."""
    c = cell
    q, dep = find_wells(c["store"], c["cfg"], c["mw"], jax.random.PRNGKey(3))
    V = np.asarray(jax.vmap(c["store"].V)(jnp.asarray(q)))
    raw = -V
    np.testing.assert_allclose(
        dep, raw + c["cfg"].confine * (q ** 2).sum(-1), rtol=1e-4, atol=1e-5)
    assert not np.allclose(dep, raw)


# --------------------------------------------------------------------------
# (e) learned p0 / the head's contract
# --------------------------------------------------------------------------
def test_head_emits_k_full_particles_and_p0_is_a_switchable_lever(cell):
    c = cell
    code = jnp.asarray(np.random.default_rng(0).normal(size=(c["cfg"].addr_dim,)),
                       jnp.float32)
    q0, p0, mass, gmult, conf = c["head"](code)
    k, dim = c["mw"].k_particles, c["cfg"].dim
    assert q0.shape == (k, dim) and p0.shape == (k, dim)
    assert mass.shape == (k, dim) and gmult.shape == (k,) and conf.shape == (k,)
    # ⛔ anti-decoration: the payload block of the launch is pinned to 0
    np.testing.assert_allclose(np.asarray(q0[:, c["cfg"].addr_dim:]), 0.0, atol=0)
    # confidence is monotone down the slots (overlap-as-confidence)
    assert float(conf[0]) >= float(conf[-1])
    # unconfident particles are heavier and more damped
    assert float(mass[0, 0]) <= float(mass[-1, 0])
    assert float(gmult[0]) <= float(gmult[-1])
    mw_off = MultiWellReadConfig(**{**c["mw"].as_dict(), "learned_p0": False})
    h_off = LaunchHead(c["phi"], c["anchors"], c["cfg"], mw_off)
    assert float(jnp.abs(h_off(code)[1]).max()) == 0.0
    assert float(jnp.abs(p0).max()) > 0.0


def test_head_trainable_spec_excludes_the_frozen_launch_geometry(cell):
    """⛔ ``codes``/``anchors`` are shared byte-identically by every arm; handing
    them to an optimiser would break the ``PhiMismatchError`` contract."""
    c = cell
    spec = head_trainable_spec(c["head"])
    assert spec.codes is False and spec.anchors is False
    assert spec.bias is True and spec.slot_offset is True
    import equinox as eqx
    params, _ = eqx.partition(c["head"], spec)
    assert params.codes is None and params.anchors is None
    assert c["head"].n_params() == c["mw"].k_particles * c["cfg"].addr_dim \
        + c["cfg"].n_wells + 11


def test_query_code_is_one_draw_per_query_and_batch_independent(cell):
    """Registered deviation D7 + guard 1's precondition: the noisy code a query
    gets must not depend on how it was batched."""
    c = cell
    ind = c["fam"].indicator(c["fam"].unseen, c["cfg"].n_wells)
    key = jax.random.PRNGKey(7)
    full = np.asarray(query_code(c["phi"], ind, key, c["cfg"].query_sigma))
    assert full.shape == (len(ind), c["cfg"].addr_dim)
    again = np.asarray(query_code(c["phi"], ind, key, c["cfg"].query_sigma))
    np.testing.assert_array_equal(full, again)
    assert not np.allclose(full, np.asarray(
        query_code(c["phi"], ind, None, c["cfg"].query_sigma)))


# --------------------------------------------------------------------------
# the reader class (D8) — the soft twin has the SAME capacity as the hard one
# --------------------------------------------------------------------------
def test_soft_well_table_matches_the_hard_readers_capacity(cell):
    c = cell
    lat = multiwell_read(c["store"], c["head"], c["phi"], c["cfg"], c["mw"],
                         c["fam"].indicator(c["fam"].seen, c["cfg"].n_wells),
                         jax.random.PRNGKey(2))
    rd = fit_readers_mw(lat, c["fam"].y_seen,
                        anchors=c["anchors"][:, : c["cfg"].addr_dim],
                        well_payloads=c["fam"].payloads)
    assert set(rd) == {"sum_linear", "well_table", "knn", "mlp", "soft_well_table"}
    assert rd["soft_well_table"]["n_params"] == rd["well_table"]["n_params"]
    bound = c["cfg"].n_wells * c["cfg"].payload_dim
    for name, m in rd.items():
        assert int(m.get("n_params", 0)) < bound, name
    sc = score_readers_mw(rd, lat, c["fam"].y_seen, c["fam"].tol)
    assert set(sc) == set(rd)
    assert all(0.0 <= v <= 1.0 for v in sc.values())
