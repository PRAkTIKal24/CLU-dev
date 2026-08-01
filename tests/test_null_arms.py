"""Tests for the matched-capacity null arms (C2W5 ``orgdiv-null-arms``).

Every test here asserts a property the audit's verdict depends on. Several encode
the *matching* obligations of ``FROZEN-interfaces.md`` — if one of them breaks, an
arm is no longer a matched control and its number is void
(``PhiMismatchError`` precedent), so they are cheap insurance against exactly the
failure mode the frozen document exists to prevent.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.factored_store import (
    CatTestConfig,
    FactoredStore,
    build_family,
    build_phi,
    fit_readers,
    multi_particle_read,
    place_wells,
    reader_bytes,
)
from chlu.core.null_arms import (
    ARMS,
    NullArmGrid,
    fit_code_payloads,
    launch_points,
    n1_apply,
    n1_gradient_placed,
    n2_vq,
    n3_static_geometric,
    n4_keys,
    n4_knn,
    n5_titans,
    phi_decodability_ceiling,
    shuffle_launches,
    store_codebook,
)
from chlu.experiments.exp_null_arms import _grid_configs, _native, seed_setup


@pytest.fixture(scope="module")
def small():
    """Small but STRUCTURALLY IDENTICAL: same rules, same phi, same read."""
    return CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=24,
                         atoms_per_well=6, addr_dim=4, payload_dim=6,
                         write_steps=20, address_steps=20, read_steps=20)


@pytest.fixture(scope="module")
def setup(small):
    return seed_setup(small, 0, n_val=4)


# ==========================================================================
# the matching obligations (FROZEN-interfaces.md)
# ==========================================================================
def test_launch_points_are_bit_identical_to_the_physics_arms_launch(small):
    """⛔ The launch protocol must be the SAME OBJECT on every arm.

    With a zero-step settle, :func:`multi_particle_read` *is* the launch, so the
    null arms' :func:`launch_points` must reproduce it exactly — key folding
    included. A silent drift here (a different key, a different sigma_q) would make
    every arm's number incomparable while looking perfectly healthy.
    """
    cfg = CatTestConfig(**{**small.as_dict(), "address_steps": 0, "read_steps": 0})
    fam = build_family(cfg, seed=0)
    phi = build_phi(cfg)
    anchors = place_wells(phi, cfg, sep=0.8)
    store = FactoredStore(cfg, anchors, jax.random.PRNGKey(0))
    ind = fam.indicator(fam.seen, cfg.n_wells)
    k = jax.random.PRNGKey(2000)
    z = multi_particle_read(store, phi, cfg, ind, k)
    q0 = launch_points(phi, cfg, ind, k)
    assert z.shape == q0.shape
    np.testing.assert_array_equal(np.asarray(z), np.asarray(q0))


def test_phi_is_frozen_across_seeds_and_its_bytes_are_ledgered(small):
    """phi is drawn from ``phi_seed``, never from ``cfg.seed`` (FROZEN §(i))."""
    a, b = build_phi(small), build_phi(small)
    np.testing.assert_array_equal(np.asarray(a.codes), np.asarray(b.codes))
    np.testing.assert_array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    reg = CatTestConfig()  # the registered cell
    assert build_phi(reg).n_bytes() == 576  # (32*4 + 4*4) * 4 B


def test_the_payload_block_of_every_launch_is_pinned_to_zero(setup, small):
    """The anti-decoration guard: nothing hands the read its answer."""
    q0 = setup["q0_s"]
    assert np.abs(q0[..., small.addr_dim:]).max() == 0.0


def test_n1_carries_exactly_the_physics_stores_parameter_count(small):
    """⭐ N1's whole claim to being *the* damaging swap is that its DOF are the
    physics arm's, exactly — not approximately."""
    S = seed_setup(small, 0, n_val=4)
    fit = n1_gradient_placed(small, S["family"], S["anchors"], S["q0_s"],
                             S["family"].y_seen, lr=1e-2, tau=0.2, steps=5, seed=0)
    store = FactoredStore(small, S["anchors"], jax.random.PRNGKey(0))
    assert fit["ledger"]["param_bytes"] == store.n_bytes()
    assert fit["ledger"]["n_params"] == small.n_atoms * (small.dim + 2)


def test_every_z_arm_returns_the_frozen_reader_shape(setup, small):
    """``z`` is ``(B, P, d+m)`` on every arm — the shape the frozen readers take."""
    S, fam = setup, setup["family"]
    B, P, dim = len(fam.seen), small.n_particles, small.dim
    n1 = n1_gradient_placed(small, fam, S["anchors"], S["q0_s"], fam.y_seen,
                            lr=1e-2, tau=0.2, steps=5, seed=0)
    assert n1_apply(n1, S["q0_s"]).shape == (B, P, dim)
    n2 = n2_vq(small, fam, S["q0_s"], fam.y_seen, variant="kmeans", n_codes=8,
               seed=0, restarts=2)
    assert n2["apply"](S["q0_s"]).shape == (B, P, dim)
    n3 = n3_static_geometric(small, fam, S["anchors"], S["q0_s"], fam.y_seen,
                             level="sb", lr=1e-2, steps=5, seed=0)
    assert n3["apply"](S["q0_s"]).shape == (B, P, dim)


def test_readers_stay_under_the_capacity_bound_on_a_null_arm(setup, small):
    """⛔ FROZEN §(ii): a reader with ``>= N_a*m`` fitted params solves the family
    from the SEEN split with NO store (SP-1) — on every arm, including a blank
    one. The bound must hold for the null arms' readers too."""
    S, fam = setup, setup["family"]
    n1 = n1_gradient_placed(small, fam, S["anchors"], S["q0_s"], fam.y_seen,
                            lr=1e-2, tau=0.2, steps=5, seed=0)
    z = n1_apply(n1, S["q0_s"])
    anc, pay = n1["codebook"]
    rd = fit_readers(z, fam.y_seen, anchors=anc, well_payloads=pay, seed=0)
    bound = small.n_wells * small.payload_dim
    for name, n in reader_bytes(rd).items():
        assert n < bound, f"reader {name} has {n} fitted params, bound {bound}"


# ==========================================================================
# the selection protocol (⛔ never on Q_unseen)
# ==========================================================================
def test_the_validation_split_is_rule_4_valid_against_the_training_rows(small):
    """⚠ The naive seen-holdout is NOT the same problem as ``Q_unseen``: two
    written items may share ``F-1`` wells. Selection must not be able to reward
    near-neighbour interpolation, so the validation rows carry rule 4 too."""
    S = seed_setup(small, 0, n_val=4)
    seen = np.asarray(S["family"].seen)
    for i in S["va"]:
        ov = np.array([len(set(seen[i].tolist()) & set(seen[j].tolist()))
                       for j in S["tr"]])
        assert ov.max() <= small.f_subset - 2
    assert set(S["va"].tolist()) & set(S["tr"].tolist()) == set()
    assert len(S["va"]) + len(S["tr"]) == len(seen)


def test_registered_grid_meets_the_committed_budget():
    """prereg §4.3: >= 5 optimiser points x 3 capacity points, per arm."""
    g = NullArmGrid()
    assert len(g.lrs) >= 5
    for arm in ARMS:
        confs = _grid_configs(arm, g)
        assert len(confs) >= 15, f"{arm} grid is {len(confs)} configs"
    assert len(set(c["atoms_per_well"] for c in _grid_configs("N1", g))) == 3
    assert len(set(c["n_codes"] for c in _grid_configs("N2", g))) == 3
    assert len(set(c["level"] for c in _grid_configs("N3", g))) == 3
    assert len(set(c["hidden"] for c in _grid_configs("N5", g))) == 3


def test_a_diverged_fit_loses_selection_instead_of_poisoning_it():
    """A NaN must not silently win (or vanish from) an ``argmax`` over configs."""
    y = np.zeros((4, 3))
    bad = _native(np.full((4, 3), np.nan), y, 0.5)
    good = _native(np.zeros((4, 3)), y, 0.5)
    assert bad["diverged"] and np.isfinite(bad["mse"]) and bad["acc"] == 0.0
    assert max([bad, good], key=lambda r: (r["acc"], -r["mse"])) is good


# ==========================================================================
# the arms themselves
# ==========================================================================
def test_n1_can_fit_the_items_it_was_trained_on(small):
    """⭐ L1, the internal-validity anchor: a verdict of *"no arm clears"* is only
    meaningful if the arms can fit what they saw. This is the optimiser check the
    audit's headline depends on."""
    S = seed_setup(small, 0, n_val=4)
    fam = S["family"]
    fit = n1_gradient_placed(small, fam, S["anchors"], S["q0_s"], fam.y_seen,
                             lr=3e-2, tau=0.2, steps=300, seed=0)
    assert fit["loss_last"] < 0.25 * fit["loss_first"]


def test_fit_code_payloads_recovers_the_payloads_from_a_true_assignment():
    """N2/N3's payload table is a least-squares solve: with the TRUE assignment it
    must return the written ``v_j``, or a null arm would be handicapped by a
    numerical artifact rather than by the family.

    ⚠ It also documents the **identifiability condition** the arms inherit: the
    count design matrix is ``(K x N_a)``, so recovery needs ``K >= N_a`` *and* full
    column rank. Below that the solve is under-determined — it still reproduces
    ``y`` on the fitted rows, but the per-code payloads are not the written ones.
    The registered cell has ``K = 128 > N_a = 32`` and is safely inside.
    """
    rng = np.random.default_rng(0)
    N, m, K, P = 8, 3, 40, 4
    payloads = rng.normal(size=(N, m))
    assign = rng.integers(0, N, size=(K, P))
    M = np.zeros((K, N))
    np.add.at(M, (np.repeat(np.arange(K), P), assign.ravel()), 1.0)
    Y = M @ payloads
    V = fit_code_payloads(assign, Y, N, ridge=1e-9)
    np.testing.assert_allclose(V, payloads, atol=1e-6)

    # under-determined: K < N_a -> y is reproduced, the payloads are NOT recovered
    a2, y2 = assign[:4], Y[:4]
    V2 = fit_code_payloads(a2, y2, N, ridge=1e-6)
    M2 = np.zeros((4, N))
    np.add.at(M2, (np.repeat(np.arange(4), P), a2.ravel()), 1.0)
    np.testing.assert_allclose(M2 @ V2, y2, atol=1e-3)
    assert not np.allclose(V2, payloads, atol=1e-3)


def test_n4_reproduces_the_cat_tests_own_idw_substitute(small):
    """⛔ No drift from ``exp_cat_test``'s inline N4: the arm that beat us in C2W1
    must be *the same* arm here."""
    S = seed_setup(small, 0, n_val=4)
    fam, phi = S["family"], S["phi"]
    code_s = np.asarray(phi.set_code(jnp.asarray(S["ind_s"])))
    code_u = np.asarray(phi.set_code(jnp.asarray(S["ind_u"])))
    for k in (1, 2, 3):
        d_cs = np.linalg.norm(code_u[:, None, :] - code_s[None, :, :], axis=-1)
        idx = np.argsort(d_cs, axis=1)[:, :k]
        w = 1.0 / (np.take_along_axis(d_cs, idx, 1) + 1e-9)
        w /= w.sum(1, keepdims=True)
        ref = (w[..., None] * fam.y_seen[idx]).sum(1)
        mine = n4_knn(small, code_s, fam.y_seen, code_u, k=k, weight="idw")
        np.testing.assert_allclose(mine, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        n4_keys("set_code", phi, small, S["ind_s"], S["q0_s"]), code_s, atol=1e-5)


def test_n5_declares_init_as_parameters_and_deviation_as_state(small):
    """prereg §1's learned-initial-state rule, mechanically enforced."""
    S = seed_setup(small, 0, n_val=4)
    fit = n5_titans(small, np.asarray(S["q0_s"])[..., :small.addr_dim].mean(1),
                    S["family"].y_seen, hidden=8, lr=1e-2, passes=1,
                    pretrain_steps=5, seed=0)
    led = fit["ledger"]
    assert led["init_is"] == "PARAMETERS" and led["deviation_is"] == "STATE"
    assert led["n_state"] == led["n_params"] > 0


def test_the_shuffle_phi_launder_destroys_the_query_and_nothing_else(setup):
    """The laundering control must permute WHOLE launch blocks (capacity, fitting
    and training all preserved) — otherwise it is not a matched launder."""
    q0 = setup["q0_s"]
    sh = shuffle_launches(q0, 0)
    assert sh.shape == q0.shape
    assert not np.array_equal(sh, q0)
    a = np.sort(np.asarray(q0).reshape(len(q0), -1).sum(1))
    b = np.sort(np.asarray(sh).reshape(len(sh), -1).sum(1))
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_the_ceiling_diagnostic_is_out_of_class_and_decodes_the_noiseless_code(small):
    """⛔ Reported, never scored as an arm. It must decode the *exact* set code
    essentially perfectly — that is what makes it a ceiling and not an arm."""
    S = seed_setup(small, 0, n_val=4)
    r = phi_decodability_ceiling(S["phi"], small, S["family"], q0_unseen=S["q0_u"])
    assert r["noiseless_combo_exact"] >= 0.95
    assert r["noiseless_accuracy"] >= 0.95
    assert 0.0 <= r["as_launched_accuracy"] <= 1.0
    assert r["n_combos"] == 560  # C(16, 3)


def test_store_codebook_is_the_arms_own_and_never_the_written_payloads(small):
    """R2's null-arm form reads the NULL ARM's codebook (FROZEN §(ii))."""
    S = seed_setup(small, 0, n_val=4)
    store = FactoredStore(small, S["anchors"], jax.random.PRNGKey(0))
    anc, pay = store_codebook(store.V, small)
    assert anc.shape == (small.n_wells, small.addr_dim)
    assert pay.shape == (small.n_wells, small.payload_dim)
    assert not np.allclose(pay, S["family"].payloads)
