"""C2W11 spoke B — the organizer loss package, the set-level psi, the novelty read.

⛔ **Every MECHANICS leg ships with a designed negative that is pytest-asserted.
A leg that cannot fail on the degenerate configuration does not ship** (the
defect class caught three times in C2W8). The negatives here are:

* ``test_coefficient_zero_*`` — the **C2W2 invariant**: every coefficient at 0
  must be **BIT-IDENTICAL** to the shipped objective, including the placement,
  the store, and the RNG stream.
* ``test_m3_designed_negative_planted_permutation`` — same store, same queries,
  **wrong declared targets** ⇒ M3 must score ~0 **while ``any_basin`` stays
  high**. ⛔ ``any_basin`` is reported and is NOT the leg.
* ``test_m3_designed_negative_narrow_wells`` — a rig retrievable at its own
  sites but unreachable from a cue must **FAIL**.
* ``test_m3_designed_negative_zero_drift_table_store`` — ⚠⚠ **THE D2a TRAP**:
  a planted **near-zero-drift, table-like** store must **not** be able to buy a
  pass. Across pass 3's 9 cells ``rho(A1, G-DRIFT) = -0.967``, i.e. the gate's
  score was almost a monotone function of settle-collapse.
* ``test_v2_designed_negative_*`` — the novelty channel on a scrambled store.
  The **registered** negative (permuted payloads) is asserted **as measured**
  and its non-discrimination is asserted explicitly, beside two negatives that
  DO bite (blank store, shuffled labels).
* ``test_designed_write_channel_is_live_and_the_accidental_one_is_dead`` —
  §A28.1's designed-vs-accidental separation, in the form the theorist
  specified (N-a2 (i)/(ii)/(iii)).

Each test is named for the defect it prevents from coming back.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.factored_store import (
    CatTestConfig,
    FactoredStore,
    build_family,
    build_phi,
    place_wells,
    write_store,
)
from chlu.core.feature_launch import build_launch_head
from chlu.core.novelty_read import (
    NOVELTY_FEATURES,
    NoveltyHead,
    auroc,
    collapse_statistic,
    ece,
    novelty_input,
    particle_descriptors,
    psi_input,
)
from chlu.core.psi_readout import ParticleSetPsi, set_psi_param_count
from chlu.experiments.exp_c2w11_organizer import (
    ARMS,
    FROZEN,
    OrganizerConfig,
    _JiggedCodes,
    _settle_points,
    build_organized_cell,
    launch_and_settle,
    load_frozen,
    per_feature_gaddr,
    refine_store,
    train_jig,
)
from chlu.training.losses import (
    C2W11LossCoeffs,
    band_hinge,
    cal_loss,
    organizer_total,
    reach_org_loss,
    refresh_amplitudes,
    shape_loss,
    soft_min,
)


@pytest.fixture(scope="module")
def small():
    """A small but STRUCTURALLY IDENTICAL family (same rules, same read)."""
    return CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=32,
                         atoms_per_well=6, addr_dim=4, payload_dim=6,
                         write_steps=20, address_steps=40, read_steps=40,
                         write_mode="placing", launch_mode="feature_factored",
                         n_particles=3, payload_radius=0.6,
                         atom_payload_init_radius=0.6)


@pytest.fixture(scope="module")
def ocfg():
    return OrganizerConfig(jig_steps=8, refine_steps=4, psi_steps=8,
                           novelty_steps=8, n_train_episodes=1,
                           m3_n_capture_wells=3)


@pytest.fixture(scope="module")
def rig(small, ocfg):
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    cell = build_organized_cell(small, ocfg, fam, 0, "coef0", phi=phi)
    return {"cfg": small, "fam": fam, "phi": phi, "cell": cell}


# ==========================================================================
# ⛔⛔ THE COEFFICIENT-ZERO INVARIANT (the C2W2 invariant)
# ==========================================================================
def test_coefficient_zero_placement_is_bit_identical(small, rig):
    """⛔ ``jig = 0`` ⇒ the placement is bit-identical to spoke A's ``place_wells``.

    Structural, not numerical: the jigged code carrier adds exactly ``0.0``.
    """
    phi = rig["phi"]
    ref = place_wells(phi, small, sep=0.6)
    jig = np.zeros((small.n_wells, small.addr_dim))
    got = place_wells(_JiggedCodes(phi.codes, jig, small.ball_radius), small, sep=0.6)
    assert np.array_equal(ref, got)


def test_coefficient_zero_organizer_draws_no_rng_and_returns_zero(small, rig, ocfg):
    """⛔ At coefficient zero the organizer must not exist: no optimiser, no
    graph, **no RNG draw** (a term that consumes RNG at coefficient zero changes
    the whole run — §1(a)(iii)(1)'s 'usual silent killer')."""
    jig, rep = train_jig(rig["phi"], small, rig["fam"], C2W11LossCoeffs(), ocfg,
                         jax.random.PRNGKey(0), jig_max=1.0, spacing_ruler=0.3)
    assert rep["steps"] == 0
    assert np.array_equal(jig, np.zeros_like(jig))
    assert "OFF" in rep["organizer"]


def test_coefficient_zero_refine_returns_the_store_unchanged(small, rig, ocfg):
    """⛔ M7's own registered negative: the term's coefficient at 0 must leave the
    shipped objective **bit-identical**."""
    cell = rig["cell"]
    out, rep = refine_store(cell["store"], small, ocfg, cell["targets"],
                            C2W11LossCoeffs(), jax.random.PRNGKey(0))
    assert out is cell["store"]
    assert rep["steps"] == 0
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(out.V, eqx.is_inexact_array)),
                    jax.tree_util.tree_leaves(
                        eqx.filter(cell["store"].V, eqx.is_inexact_array)), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_coefficient_zero_total_is_the_shipped_objective_bitwise():
    """``organizer_total`` at all-zero coefficients returns the shipped scalar
    **unchanged, by identity** — the term's graph is never built."""
    shipped = jnp.asarray(1.2345678901234567)
    out = organizer_total(shipped, C2W11LossCoeffs())
    assert out is shipped
    with pytest.raises(KeyError):
        organizer_total(shipped, C2W11LossCoeffs(lambda_shape=1.0))


def test_coefficient_zero_arm_reproduces_the_shipped_store(small, ocfg):
    """⛔ The whole ``coef0`` arm must reproduce spoke A's shipped arm."""
    fam = build_family(small, seed=1)
    phi = build_phi(small)
    a = build_organized_cell(small, ocfg, fam, 1, "coef0", phi=phi)
    b = build_organized_cell(small, ocfg, fam, 1, "coef0", phi=phi)
    assert np.array_equal(a["anchors"], b["anchors"])
    assert ARMS["coef0"].as_flag_table() == {}
    assert not ARMS["coef0"].any_live


# ==========================================================================
# M3 — the designed negatives (both), and THE D2a TRAP
# ==========================================================================
def _m3_inputs(rig, small, ocfg, *, caps=None):
    cell = rig["cell"]
    fam = rig["fam"]
    ind = fam.indicator(fam.unseen, small.n_wells)
    z, q0 = launch_and_settle(cell["store"], cell["head"], small, ind,
                              jax.random.PRNGKey(3))
    picks = np.asarray(jax.vmap(
        lambda i: cell["head"].channels(cell["head"].set_code(i)))(jnp.asarray(ind)))
    if caps is None:
        caps = np.full(small.n_wells, 0.9)
    return z, picks, np.asarray(fam.unseen), cell["anchors"], caps


def test_m3_designed_negative_planted_permutation(rig, small, ocfg):
    """⛔ (a) A planted permutation — same store, same queries, **wrong declared
    targets** — must score ~0 **while ``any_basin`` stays high**."""
    z, picks, subs, anchors, caps = _m3_inputs(rig, small, ocfg)
    good = per_feature_gaddr(z, picks, subs, anchors, caps, small.addr_dim)
    rng = np.random.default_rng(0)
    perm = rng.permutation(small.n_wells)
    bad = per_feature_gaddr(z, perm[picks], perm[subs], anchors, caps, small.addr_dim)
    # ⛔ the negative must FAIL THE LEG's own bar (max(4*chance, chance+2SE)),
    # not an absolute constant: at N_a = 16 chance is 1/16 and 4*chance = 0.25.
    bar = 4.0 / small.n_wells
    assert good["per_feature_gaddr"] >= bar, good
    assert bad["per_feature_gaddr"] < bar, bad
    assert bad["per_feature_gaddr"] < 0.25 * good["per_feature_gaddr"], bad
    assert bad["any_basin"] >= 0.5 * good["any_basin"]
    assert bad["any_basin"] == good["any_basin"], "any_basin must be target-blind"


def test_m3_designed_negative_narrow_wells(rig, small, ocfg):
    """⛔ (b) A **narrow-wells** rig — retrievable at its own sites, unreachable
    from a cue — must FAIL the leg."""
    z, picks, subs, anchors, _ = _m3_inputs(rig, small, ocfg)
    narrow = np.full(small.n_wells, 1e-4)   # basins far narrower than sigma_q
    out = per_feature_gaddr(z, picks, subs, anchors, narrow, small.addr_dim)
    assert out["per_feature_gaddr"] == 0.0, out
    assert out["any_basin"] == 0.0


def test_m3_designed_negative_zero_drift_table_store(rig, small, ocfg):
    """⚠⚠ **THE D2a TRAP.** A planted **near-zero-drift, table-like** store must
    not be able to buy an M3 pass: M3 is measured against the **declared
    targets**, so collapsing every particle onto one site (zero drift, perfect
    table behaviour) scores ~0, not 1.

    Banked motivation: across pass 3's 9 cells ``rho(A1, G-DRIFT) = -0.967`` and
    ``rho(A1, settle<->launder agreement) = +0.933`` — the completed gate's score
    was almost a **monotone function of settle-collapse**. ⛔ Drift -> 0 is never
    a target.
    """
    z, picks, subs, anchors, caps = _m3_inputs(rig, small, ocfg)
    collapsed = np.zeros_like(z)
    collapsed[..., :small.addr_dim] = anchors[0][None, None, :small.addr_dim]
    out = per_feature_gaddr(collapsed, picks, subs, anchors, caps, small.addr_dim)
    assert out["per_feature_gaddr"] <= 0.10, out
    assert out["any_basin"] == 1.0, "the collapsed store IS in a basin — that is " \
                                    "exactly why any_basin cannot be the leg"


# ==========================================================================
# V2 — the novelty channel's designed negatives
# ==========================================================================
def _novelty_rig(rig, small, ocfg, dropped):
    from chlu.experiments.exp_c2w11_organizer import _novelty_episode
    fam = rig["fam"]
    ind = fam.indicator(fam.unseen, small.n_wells)
    return _novelty_episode(rig["cell"], small, ocfg, ind, fam.unseen,
                            np.asarray(dropped), jax.random.PRNGKey(9), 0)


def test_v2_designed_negative_blank_store_reports_chance(rig, small, ocfg):
    """⛔ A store where **nothing** was written cannot tell a known channel from a
    novel one: the AUROC of the depth feature must sit at ~0.5."""
    ep = _novelty_rig(rig, small, ocfg, [0, 1, 2, 3])
    blank = FactoredStore(small, rig["cell"]["anchors"], jax.random.PRNGKey(0),
                          atom_width=rig["cell"]["width"]["atom_width"])
    cell_b = dict(rig["cell"], store=blank)
    from chlu.experiments.exp_c2w11_organizer import _novelty_episode
    fam = rig["fam"]
    eb = _novelty_episode(cell_b, small, ocfg,
                          fam.indicator(fam.unseen, small.n_wells), fam.unseen,
                          np.asarray([0, 1, 2, 3]), jax.random.PRNGKey(9), 0)
    a = auroc(eb["desc"]["residual"].ravel(), eb["novel"].ravel())
    assert abs(a - 0.5) < 0.10, f"blank store must report chance, got {a}"


def test_v2_designed_negative_shuffled_labels_reports_chance(rig, small, ocfg):
    """⛔ The instrument's own null: shuffling ``n_f`` must destroy the AUROC."""
    ep = _novelty_rig(rig, small, ocfg, [0, 1, 2, 3])
    rng = np.random.default_rng(0)
    a = auroc(ep["desc"]["residual"].ravel(), rng.permutation(ep["novel"].ravel()))
    assert abs(a - 0.5) < 0.15, a


def test_v2_registered_negative_permuted_payloads_is_non_discriminating(rig, small,
                                                                        ocfg):
    """⚠ **The REGISTERED negative, asserted AS MEASURED.**

    ``PREREG-C2W11.md`` §5 V2 registers *permuted payloads ⇒ AUROC ~ 0.5*. It is
    written for a **payload-keyed** channel; this spoke's channel is
    **depth-keyed** by the (c)/(e) feature-set contract, and permuting ``v_j``
    leaves every well written and every depth intact. The negative is therefore
    **structurally incapable of producing both outcomes** (§A37's own criterion)
    and this test pins that fact so it cannot be quietly re-read as a pass.
    """
    fam = rig["fam"]
    cell = rig["cell"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(small.n_wells)
    store_p, _ = write_store(
        FactoredStore(small, cell["anchors"], jax.random.PRNGKey(0),
                      atom_width=cell["width"]["atom_width"]),
        small, cell["anchors"], fam.payloads[perm], jax.random.PRNGKey(1),
        atom_width=cell["width"]["atom_width"])
    from chlu.experiments.exp_c2w11_organizer import _novelty_episode
    ep = _novelty_episode(dict(cell, store=store_p), small, ocfg,
                          fam.indicator(fam.unseen, small.n_wells), fam.unseen,
                          np.asarray([0, 1, 2, 3]), jax.random.PRNGKey(9), 0)
    a = auroc(ep["desc"]["residual"].ravel(), ep["novel"].ravel())
    assert a > 0.55, ("the depth channel is payload-blind by construction, so "
                      "the registered permuted-payload negative does NOT bite "
                      f"here; measured {a:.4f}. If this ever drops to 0.5 the "
                      "novelty channel changed and the report must be re-read.")


def test_novelty_head_never_sees_lambda_min():
    """⛔ The (c)/(e) cross-term contract, enforced in code rather than prose: if
    term (c) succeeds, written and unwritten sites become spectrally
    indistinguishable in ``lambda_min``, so ``lambda_min`` is barred as a
    novelty feature."""
    assert "lambda_min" not in NOVELTY_FEATURES
    assert "lambda_2nd" in NOVELTY_FEATURES
    assert "residual" in NOVELTY_FEATURES


# ==========================================================================
# §A28.1 — the DESIGNED write->phi channel vs the ACCIDENTAL leak
# ==========================================================================
def test_designed_write_channel_is_live_and_the_accidental_one_is_dead(rig, small):
    """N-a2 (i)/(ii)/(iii), the theorist's three assertions.

    (i) the designed channel is LIVE · (ii) with the placement stop-gradiented it
    is the ONLY path (bitwise 0) · (iii) the **accidental** channel is DEAD — the
    banked leak is *27 % of layer-0 phi gradient flowing through the write
    whenever ``atom_place_radius > 0``*, and on this substrate ``phi`` is frozen,
    so the prediction (PREREG B18) is that it is structurally absent.
    """
    phi, fam = rig["phi"], rig["fam"]
    codes = jnp.asarray(np.asarray(phi.codes), dtype=jnp.float32)
    ind = jnp.asarray(fam.indicator(fam.seen, small.n_wells))
    jig = jnp.zeros((small.n_wells, small.addr_dim))
    q = jnp.asarray(np.random.default_rng(0).normal(
        size=(len(fam.seen), small.n_particles, small.addr_dim)), dtype=jnp.float32)

    def live(c):
        return reach_org_loss(small.ball_radius * c + jig, q, ind)

    def cut(c):
        return reach_org_loss(jax.lax.stop_gradient(small.ball_radius * c + jig),
                              q, ind)

    g_live = float(jnp.linalg.norm(jax.grad(live)(codes)))
    g_cut = float(jnp.linalg.norm(jax.grad(cut)(codes)))
    assert g_live > 0.0, "N-a2(i): the DESIGNED channel must be live"
    assert g_cut == 0.0, "N-a2(ii): the placement must be the ONLY path, bitwise"


# ==========================================================================
# the package's own units
# ==========================================================================
def test_band_hinge_is_two_sided():
    """⛔ Two-sided is mandatory: below ``d/s = 2.01`` wells merge; at or above
    4.0 the settled-point organization is exactly nearest-centroid VQ."""
    assert float(band_hinge(jnp.asarray(2.7))) == 0.0
    assert float(band_hinge(jnp.asarray(2.0))) > 0.0
    assert float(band_hinge(jnp.asarray(4.0))) > 0.0


def test_refresh_is_monotone_by_parameterisation_not_by_penalty():
    """I1 holds as an **IDENTITY**: a loss that rewards monotonicity can be traded
    away by any other term; a parameterisation cannot."""
    amp = jnp.asarray([0.1, 0.5, 1.0])
    for delta in (0.0, 0.25, 3.0):
        out = refresh_amplitudes(amp, jnp.full_like(amp, delta))
        assert bool(jnp.all(out ** 2 >= amp ** 2 - 1e-12))


def test_shape_loss_refuses_to_reward_an_undug_site():
    """⛔ M8's banked trap: an **undug** site reports ``lambda_min = 2 alpha`` for
    free. ``L_shape`` must be **larger** there than at a dug, capturing site with
    the same spectrum — otherwise the term is minimised by writing nothing."""
    lam = jnp.asarray([[0.105, 1.0, 2.0]])
    dug = float(shape_loss(lam, jnp.asarray([0.30]), jnp.asarray([0.5]),
                           depth_min=0.15, sigma_q=0.15))
    undug = float(shape_loss(lam, jnp.asarray([0.0]), jnp.asarray([0.0]),
                             depth_min=0.15, sigma_q=0.15))
    assert undug > dug


def test_soft_min_tracks_the_true_minimum():
    lam = jnp.asarray([0.11, 3.0, 9.9])
    assert abs(float(soft_min(lam, beta=200.0)) - 0.11) < 1e-3


def test_psi_is_permutation_invariant_and_ledgered(small):
    """⛔ DeepSets pooled sum only — permutation invariance is the property that
    makes it a SET read, and it is asserted, not assumed."""
    psi = ParticleSetPsi(7, 3, jax.random.PRNGKey(0), hidden=8)
    u = jax.random.normal(jax.random.PRNGKey(1), (4, 3, 7))
    w = jax.random.uniform(jax.random.PRNGKey(2), (4, 3))
    a = np.asarray(psi(u, w))
    b = np.asarray(psi(u[:, ::-1], w[:, ::-1]))
    assert np.allclose(a, b, atol=1e-5)
    assert set_psi_param_count(psi) > 0


def test_psi_exceeds_the_sp1_bound_and_that_is_why_k4_at_full_psi_exists():
    """⛔ K7-CAP's ``N_a*m`` bound binds the READER CLASS, not ``psi``. psi is not
    a member of that class; its guard is the **measured** leak (K4 at full psi)
    plus the K8 structural cell. Pinning this stops a future reader from quietly
    inheriting psi's capacity."""
    psi = ParticleSetPsi(17, 8, jax.random.PRNGKey(0), hidden=16)
    assert set_psi_param_count(psi) > FROZEN["bound_Na_times_m"]


def test_frozen_mirror_matches_the_json_when_it_is_reachable():
    """⛔ The payload repair moved ``tol`` 0.47827 -> 0.28696 and every y-scale
    with it. A pre-repair constant is exactly the trap this guards."""
    f = load_frozen()
    assert f["family"]["tol"] == 0.286960063782279
    assert f["family"]["payload_radius"] == 0.6
    assert f["v3_budget_grid"]["points_total_verlet_steps"] == [50, 100, 200, 400,
                                                                800, 1200]


def test_auroc_and_ece_are_correct_on_hand_cases():
    assert auroc(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1])) == 1.0
    assert auroc(np.array([3.0, 2.0, 1.0]), np.array([0, 0, 1])) == 0.0
    assert abs(auroc(np.array([1.0, 1.0]), np.array([0, 1])) - 0.5) < 1e-12
    out = ece(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert out["ece"] == 0.0 and out["brier"] == 0.0


def test_collapse_statistic_reports_the_banked_shape():
    """⭐ *confident ⇒ the ``k`` particles collapse to ``F`` unique wells;
    unfamiliar ⇒ scattered guesses.*"""
    conf = collapse_statistic(np.array([[0, 1, 2, 3]]), 4)
    scat = collapse_statistic(np.array([[0, 0, 0, 0]]), 4)
    assert conf["frac_collapsed_to_F"] == 1.0
    assert scat["frac_collapsed_to_F"] == 0.0


def test_cal_loss_is_the_proper_scoring_rule_not_auroc():
    """⛔ Log-loss is the objective; AUROC is REPORTED and never trained against."""
    logits = jnp.asarray([[10.0, -10.0]])
    assert float(cal_loss(logits, jnp.asarray([[1.0, 0.0]]))) < 1e-3
    assert float(cal_loss(logits, jnp.asarray([[0.0, 1.0]]))) > 5.0
