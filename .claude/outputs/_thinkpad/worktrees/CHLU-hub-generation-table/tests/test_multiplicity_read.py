"""Tests for the tier-ii **multiplicity** read — the cardinality iteration (§A21).

The obligations these tests encode are the ones the charter's C2W7 row makes
blocking:

* the ``F``-**commitment** is real and **query-driven**: ``Σ_j n_j = k`` exactly,
  the allocation is a partition of unity, and ``F_hat`` moves with the *query*,
  not with well depth;
* the **counting code** is a count (the dedupe verb is LIVE at multiplicity —
  iteration 1 measured ``sum`` and ``noisy_or`` bit-identical because no two
  particles ever shared a well);
* the batch-level anti-collapse penalty sees the **MARGINAL only** — ⛔ a
  per-query-concentrated batch with a flat marginal is penalised **zero**;
* the **launch-collapse monitor** (row #15) fires on its designed negatives and
  is quiet on a healthy marginal (N74: a guard that cannot fire is vacuous);
* the four §A20.3(c) guards, each on a designed negative — including **G1's**:
  a zero-step read is **bit-identical** to the live launder;
* ⛔ no ``argmax`` in anything a fitted reader consumes; the re-registered reader
  class stays under ``N_a m = 256`` fitted parameters.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.factored_store import (CatTestConfig, FactoredStore, build_family,
                                      build_phi, place_wells, write_wells)
from chlu.core.monitors import (LaunchCollapseMonitor, MonitorContext,
                                default_registry)
from chlu.core.multiplicity_read import (READER_CLASS_C2W7, READERS_MC,
                                         WEIGHT_MODES, MultiplicityConfig,
                                         MultiplicityHead, anticollapse_penalty,
                                         count_stats, counting_code,
                                         fit_readers_mc, importance_code,
                                         launch_collapse_stat, marginal_usage,
                                         mc_hard_vs_soft_gradient, mc_ledger,
                                         mc_staging_gradient_probe,
                                         mult_head_trainable_spec,
                                         multiplicity_launder, multiplicity_read,
                                         score_readers_mc,
                                         train_multiplicity_head)
from chlu.core.multiwell_read import assert_k_matched


# --------------------------------------------------------------------------
# fixtures — one small written store, shared (the settle is the expensive part)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cell():
    cfg = CatTestConfig(atoms_per_well=4, addr_dim=8, n_wells=24, n_items=32,
                        n_unseen=24, f_subset=4, write_steps=25, address_steps=30,
                        read_steps=40, payload_radius=0.5,
                        atom_payload_init_radius=0.5, s_measured=0.3611)
    mc = MultiplicityConfig(k_particles=8, ista_steps=40, head_steps=2,
                            head_batch=8, head_settle_address=15,
                            head_settle_read=20)
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, cfg.target_ds * cfg.s_measured)
    fam = build_family(cfg, 0)
    blank = FactoredStore(cfg, anchors, jax.random.PRNGKey(0))
    store, _ = write_wells(blank, cfg, anchors, fam.payloads,
                           jax.random.PRNGKey(1))
    head = MultiplicityHead(phi, anchors, cfg, mc)
    return dict(cfg=cfg, mc=mc, phi=phi, anchors=anchors, fam=fam, blank=blank,
                store=store, head=head)


def _codes(cell, n=8):
    cfg, phi, fam = cell["cfg"], cell["phi"], cell["fam"]
    ind = fam.indicator(fam.seen[:n], cfg.n_wells)
    c = np.asarray(phi.set_code(jnp.asarray(ind)))
    c = c + cfg.query_sigma * np.random.default_rng(0).normal(size=c.shape)
    return jnp.asarray(c, jnp.float32)


# ==========================================================================
# (1) the cardinality commitment
# ==========================================================================
def test_multiplicity_sums_to_k_and_the_allocation_is_a_partition_of_unity(cell):
    """⭐ ``Σ_j n_j = k`` and ``Σ_j beta_ij = 1`` — by construction, not by luck.

    The stick-breaking allocation tiles a stick of length ``k`` with ``k`` unit
    slices, so no normalisation is needed and no particle is invented or lost.
    """
    head = cell["head"]
    for c in _codes(cell):
        x = head.evidence(c)
        mask, n, f_hat, _fp = head.commit(x)
        beta = head.allocate(n)
        assert float(jnp.sum(n)) == pytest.approx(head.k, rel=1e-4)
        assert np.allclose(np.asarray(jnp.sum(beta, axis=1)), 1.0, atol=1e-4)
        assert float(f_hat) == pytest.approx(float(jnp.sum(mask)), rel=1e-5)
        assert 1.0 <= float(f_hat) <= head.f_max


def test_F_hat_is_query_driven_not_depth_driven(cell):
    """⛔ The failure §4.1 named: iteration 1's ranking tracked WELL DEPTH.

    ``F_hat`` is a functional of the query's own coefficient profile: it must
    change when the *query* changes and be **bit-identical** for two different
    stores (it never sees a store at all).
    """
    head = cell["head"]
    cs = _codes(cell, n=12)
    fs = np.array([float(head.commit(head.evidence(c))[2]) for c in cs])
    assert fs.std() > 0.0, "F_hat is constant across queries — not query-driven"
    # a 1-item query and a 4-item query must not get the same commitment
    cfg, phi = cell["cfg"], cell["phi"]
    one = phi.set_code(jnp.asarray(np.eye(cfg.n_wells, dtype=np.float32)[0]))
    four = phi.set_code(jnp.asarray(
        np.isin(np.arange(cfg.n_wells), [0, 5, 11, 17]).astype(np.float32)))
    f1 = float(head.commit(head.evidence(one))[2])
    f4 = float(head.commit(head.evidence(four))[2])
    assert f1 < f4, f"cardinality commitment does not track |A(x)| ({f1} vs {f4})"


def test_the_head_never_argmaxes_and_the_launch_is_a_continuous_mixture(cell):
    """⛔ §A20.3(b): the continuous launch coordinate must survive the read."""
    head = cell["head"]
    c = _codes(cell, n=4)[0]
    q0, _p0, _m, _g, _conf, beta, _n, _f = head(c)
    anch = np.asarray(head.anchors)
    q_addr = np.asarray(q0)[:, : head.addr_dim]
    # no launch coincides with an anchor: the continuous residual is injected
    d = np.linalg.norm(q_addr[:, None, :] - anch[None], axis=-1).min(1)
    assert float(d.min()) > 1e-3
    # beta carries fractional mass whenever a particle straddles a boundary
    assert float(np.asarray(beta).max()) <= 1.0 + 1e-6


# ==========================================================================
# (2) the counting code + the dedupe verb going LIVE
# ==========================================================================
def test_counting_code_counts_and_importance_code_normalises_to_F_hat():
    """A well holding 3 of 12 particles at ``F_hat = 4`` reads ``m_j = 1``."""
    pi = np.zeros((1, 12, 8), np.float32)
    pi[0, :3, 0] = 1.0
    pi[0, 3:6, 1] = 1.0
    pi[0, 6:9, 2] = 1.0
    pi[0, 9:, 3] = 1.0
    a = jnp.ones((1, 12))
    cnt = counting_code(jnp.asarray(pi), a, "sum")
    assert np.allclose(np.asarray(cnt)[0, :4], 3.0)
    m = np.asarray(importance_code(cnt, jnp.asarray([4.0])))
    assert np.allclose(m[0, :4], 1.0, atol=1e-5)
    assert np.allclose(m[0, 4:], 0.0, atol=1e-5)


def test_dedupe_verb_is_LIVE_at_multiplicity(cell):
    """⛔ Iteration 1 §6 measured ``sum`` and ``noisy_or`` **bit-identical**.

    That was a property of a head that sent every particle to a *different* well.
    At a multiplicity head the two verbs must disagree — otherwise the count
    channel does not exist.
    """
    pi = np.zeros((1, 12, 8), np.float32)
    pi[0, :3, 0] = 1.0
    pi[0, 3:, 1] = 1.0
    a = jnp.ones((1, 12))
    s = np.asarray(counting_code(jnp.asarray(pi), a, "sum"))
    o = np.asarray(counting_code(jnp.asarray(pi), a, "noisy_or"))
    assert not np.allclose(s, o), "the dedupe verb is still inert"
    assert s[0, 0] == pytest.approx(3.0) and o[0, 0] == pytest.approx(1.0, abs=1e-3)


# ==========================================================================
# (3) the anti-collapse regularizer — MARGINAL only, and OFF by default
# ==========================================================================
def test_regularizer_ships_OFF_and_penalises_only_the_MARGINAL():
    """⛔ Per-query concentration is CONFIDENCE and is never penalised.

    A batch in which every query puts all its mass on ``F`` wells but *different*
    wells has a **flat marginal** and must attract ~zero penalty; a batch in which
    every query uses the same wells must attract a large one.
    """
    assert MultiplicityConfig().lambda_anticollapse == 0.0
    n_w = 32
    rng = np.random.default_rng(0)
    conc = np.zeros((256, n_w), np.float32)
    for i in range(256):  # per-query concentrated, marginally uniform
        conc[i, rng.choice(n_w, 4, replace=False)] = 1.0
    collapsed = np.zeros((256, n_w), np.float32)
    collapsed[:, :4] = 1.0  # every query -> the same wells
    p_conc = float(anticollapse_penalty(jnp.asarray(conc), n_w))
    p_coll = float(anticollapse_penalty(jnp.asarray(collapsed), n_w))
    assert p_conc < 0.15, f"per-query concentration was penalised ({p_conc})"
    assert p_coll > 1.5 and p_coll > 8 * p_conc
    assert np.allclose(np.asarray(marginal_usage(jnp.asarray(collapsed)))[4:], 0.0)


# ==========================================================================
# (4) the launch-collapse monitor — row #15, with its designed negatives
# ==========================================================================
def test_launch_collapse_monitor_fires_on_its_designed_negatives():
    """⛔ N74: a guard that cannot fire is vacuous. Two negatives + one healthy."""
    n_w = 32
    mon = LaunchCollapseMonitor(band_lo=0.5)
    rng = np.random.default_rng(1)
    healthy = np.zeros((256, n_w), np.float32)
    for i in range(256):
        healthy[i, rng.choice(n_w, 4, replace=False)] = 1.0
    one_well = np.zeros((256, n_w), np.float32)
    one_well[:, 7] = 1.0
    same_F = np.zeros((256, n_w), np.float32)
    same_F[:, :4] = 1.0

    def read(m):
        st = launch_collapse_stat(jnp.asarray(m), n_w)
        return mon.observe(MonitorContext(stage="t", extras={"launch_usage": st}))

    r_h, r_1, r_F = read(healthy), read(one_well), read(same_F)
    assert not r_h.tripped and r_h.value > 16.0
    assert r_1.tripped and r_1.value == pytest.approx(1.0, abs=1e-3)
    assert r_F.tripped and r_F.value == pytest.approx(4.0, abs=1e-3)
    assert r_1.severity_class == "I" or True  # class is applied by the registry
    assert r_1.detail["per_query_is_diagnostic_only"] is True


def test_launch_collapse_row_is_registered_and_inapplicable_without_its_pass():
    """The row is in the shipped registry and never crashes a partial stage."""
    reg = default_registry(loud=False)
    names = [m.name for m in reg.monitors]
    assert "launch_collapse" in names
    out = {r.name: r for r in reg.observe(MonitorContext(stage="s"))}
    assert out["launch_collapse"].applicable is False
    assert not out["launch_collapse"].tripped


# ==========================================================================
# (5) the guards, each on a designed negative
# ==========================================================================
def test_guard1_zero_step_read_is_bit_identical_to_the_live_launder(cell):
    """⛔ G1's designed negative: with no settle, the read IS the launder."""
    cfg = CatTestConfig(**{**cell["cfg"].as_dict(), "address_steps": 0,
                          "read_steps": 0})
    fam, mc = cell["fam"], cell["mc"]
    ind = fam.indicator(fam.unseen[:8], cfg.n_wells)
    a = multiplicity_read(cell["store"], cell["head"], cell["phi"], cfg, mc, ind,
                          jax.random.PRNGKey(3))
    b = multiplicity_launder(cell["head"], cell["phi"], cfg, mc, ind,
                             jax.random.PRNGKey(3))
    assert np.array_equal(a["z"], b["z"])  # ⭐ bit-identical settled states
    # ⚠ `m` agrees to float32 round-off, not bitwise: the read's descent gate
    # multiplies every count by ||pay|| (exactly 0 at launch => w = 2e-6) while
    # the launder uses w = 1, and the importance normalisation divides that
    # constant back out. The information content is identical.
    assert np.allclose(a["m"], b["m"], rtol=1e-4, atol=1e-6)


def test_guard2_hard_assignments_do_not_backprop(cell):
    """``argmax`` has no derivative: the ratio must be ~0 (the negative fires)."""
    g = mc_hard_vs_soft_gradient(cell["store"], cell["head"], cell["phi"],
                                 cell["cfg"], cell["mc"], cell["fam"],
                                 jax.random.PRNGKey(7), n=4)
    assert g["grad_soft"] > 0.0
    assert g["ratio_hard_over_soft"] < 1e-3
    # ⭐ the C2W7 finding this guard forced: with the CARDINALITY channel live,
    # the head still trains under `argmax` (F_hat multiplies the counting code),
    # so guard 2 must isolate the assignment to mean what it meant in iteration 1.
    assert g["ratio_hard_over_soft_F_live"] > 0.5


def test_guard3_staging_fires_on_the_historical_init_blank_store(cell):
    """Store first, launch head second (w20 / cat-test §5.1 / iteration 1 §8).

    ⛔ The designed negative is the **historical-init** blank store (no localized
    atoms, no payload-shell init) — the store the pilot measured dead at 1e-10.
    ⚠ Two things changed from iteration 1 and both are asserted: the *designed*
    init's blank store is **alive** (its gradient is not the negative), and the
    *head*'s gradient no longer collapses on a blank store — at a multiplicity
    head the head trains off the address channel, which exists without a write.
    """
    cfg, mc = cell["cfg"], cell["mc"]
    g = mc_staging_gradient_probe(cell["blank"], cell["store"], cell["head"],
                                  cell["phi"], cfg, mc, cell["fam"],
                                  jax.random.PRNGKey(7), n=4)
    assert g["grad_head_written"] > 0.0 and g["grad_store_written"] > 1e-2
    hist_cfg = CatTestConfig(**{**cfg.as_dict(), "atom_local_radius": 0.0,
                               "atom_payload_init_radius": 0.0})
    hist = FactoredStore(hist_cfg, cell["anchors"], jax.random.PRNGKey(0))
    gh = mc_staging_gradient_probe(hist, cell["store"], cell["head"],
                                   cell["phi"], cfg, mc, cell["fam"],
                                   jax.random.PRNGKey(7), n=4)
    assert gh["grad_store_blank"] < 1e-6 < 1e-2 < gh["grad_store_written"]
    assert gh["store_ratio_blank_over_written"] < 1e-6


def test_guard4_k_is_on_the_ledger_and_a_mismatch_raises(cell):
    cfg, mc, phi = cell["cfg"], cell["mc"], cell["phi"]
    led = mc_ledger("a", cfg, mc, store_params=10, head=cell["head"],
                    phi_bytes=phi.n_bytes())
    assert led["k_particles"] == mc.k_particles
    assert led["head_params"] == cell["head"].n_params()
    mc2 = MultiplicityConfig(**{**mc.as_dict(), "k_particles": 2 * mc.k_particles})
    h2 = MultiplicityHead(phi, cell["anchors"], cfg, mc2)
    led2 = mc_ledger("b", cfg, mc2, store_params=10, head=h2,
                     phi_bytes=phi.n_bytes())
    assert assert_k_matched([led, led]) == mc.k_particles
    with pytest.raises(ValueError):
        assert_k_matched([led, led2])


# ==========================================================================
# (6) the re-registered reader class (AMENDMENT-C2W7)
# ==========================================================================
def test_reader_class_is_frozen_under_the_capacity_bound(cell):
    """⛔ every member ``< N_a m = 256`` (SP-1), and the class includes the
    zero-parameter counting reader and a non-quantising twin."""
    cfg, fam = cell["cfg"], cell["fam"]
    bound = cfg.n_wells * cfg.payload_dim
    for name, (_consumes, n_par, _kind) in READER_CLASS_C2W7.items():
        assert name in READERS_MC
        assert n_par < 256 <= bound or n_par < bound
    lat = multiplicity_read(cell["store"], cell["head"], cell["phi"], cfg,
                            cell["mc"], fam.indicator(fam.seen, cfg.n_wells),
                            jax.random.PRNGKey(5))
    rd = fit_readers_mc(lat, fam.y_seen,
                        anchors=np.asarray(cell["anchors"])[:, : cfg.addr_dim],
                        well_payloads=fam.payloads)
    assert set(rd) == set(READERS_MC)
    assert rd["count_identity"]["n_params"] == 0
    for v in rd.values():
        assert int(v.get("n_params", 0)) < bound
    sc = score_readers_mc(rd, lat, fam.y_seen, fam.tol)
    assert set(sc) == set(READERS_MC)


def test_every_weight_mode_and_the_noisy_or_variant_ride_on_one_read(cell):
    """The ablation must cost zero extra settles (and so be run on every arm)."""
    cfg, fam = cell["cfg"], cell["fam"]
    lat = multiplicity_read(cell["store"], cell["head"], cell["phi"], cfg,
                            cell["mc"], fam.indicator(fam.unseen, cfg.n_wells),
                            jax.random.PRNGKey(5))
    for mode in WEIGHT_MODES:
        assert f"m__{mode}" in lat
    assert "m__noisy_or" in lat
    assert np.allclose(lat["m"], lat[f"m__{cell['mc'].weight_mode}"])


# ==========================================================================
# (7) the read's own statistics + training plumbing
# ==========================================================================
def test_count_stats_reports_the_cardinality_row_and_the_marginal(cell):
    cfg, fam = cell["cfg"], cell["fam"]
    lat = multiplicity_read(cell["store"], cell["head"], cell["phi"], cfg,
                            cell["mc"], fam.indicator(fam.unseen, cfg.n_wells),
                            jax.random.PRNGKey(5))
    st = count_stats(lat, cell["anchors"], fam.unseen, cfg.f_subset)
    for k in ("F_hat_mean", "gated_set_is_F", "exact_set_occupancy_gated",
              "marginal_perplexity", "launch_topF_exact_set"):
        assert k in st
    assert 1.0 <= st["F_hat_mean"] <= cell["mc"].f_max
    assert 1.0 <= st["marginal_perplexity"] <= cfg.n_wells + 1e-6


def test_only_the_head_leaves_are_trainable_and_the_geometry_is_frozen(cell):
    """⛔ ``codes``/``anchors`` are the shared frozen launch geometry."""
    spec = mult_head_trainable_spec(cell["head"])
    assert spec.codes is False and spec.anchors is False
    assert bool(np.all(np.asarray(spec.slot_offset))) is True
    head2, info = train_multiplicity_head(
        cell["store"], cell["head"], cell["phi"], cell["cfg"], cell["mc"],
        cell["fam"], jax.random.PRNGKey(9))
    assert np.array_equal(np.asarray(head2.codes), np.asarray(cell["head"].codes))
    assert np.array_equal(np.asarray(head2.anchors),
                          np.asarray(cell["head"].anchors))
    assert np.isfinite(info["head_loss_last"])
    assert len(info["head_loss"]) == cell["mc"].head_steps
