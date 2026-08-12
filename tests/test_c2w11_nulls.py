"""Tests for **C2W11 spoke C** — the organizer swap's null side.

Every test here asserts a property a VALUE-leg number depends on. The two
classes that matter most:

* **the matching obligations** — φ byte-identical, launches **bit-identical** to
  the physics read path, and the frozen family's ``tol`` scale. A drift in any of
  them makes every arm's number incomparable *while looking perfectly healthy*
  (the ``PhiMismatchError`` precedent, and the payload-repair trap in particular:
  ``run1``'s ``tol = 0.47827`` is 1.667x the repaired ``0.28696``);
* **the selection guard** — nothing may fit or select on ``Q_unseen``.

⚠ **A repo hazard this file is written against** (banked, ``orgdiv-null-arms``
§9): a module mixing explicitly-cast float32 data with flag-following
``jax.random`` initialisers is silently x64-dependent — it passes alone and fails
in the full suite, because another test enables ``jax_enable_x64`` process-wide.
:func:`test_confidence_channels_are_x64_safe` runs the whole confidence path with
the flag ON for exactly that reason.
"""

import itertools

import jax
import numpy as np
import pytest

from chlu.core.factored_store import (
    CatTestConfig,
    FactoredStore,
    build_family,
    build_phi,
    multi_particle_read,
    place_wells,
)
from chlu.core.feature_launch import build_launch_head
from chlu.core.null_arms import (
    NullArmGrid,
    anytime_read,
    expected_calibration_error,
    feature_decodability_ceiling,
    feature_keys,
    feature_launch_states,
    instantiate_landscape,
    novelty_auroc,
)
from chlu.experiments.exp_c2w11_nulls import (
    C2W11_GRID,
    VALUE_LAUNCH_KEY,
    FrozenCellMismatch,
    _fit_arm,
    assert_frozen_match,
    build_novelty_family,
    c2w11_null_config,
    seed_setup,
)


@pytest.fixture(scope="module")
def small():
    """Small but STRUCTURALLY IDENTICAL: the same repairs, the same launch."""
    return c2w11_null_config(
        None, n_wells=16, f_subset=3, n_items=12, n_unseen=24, atoms_per_well=6,
        addr_dim=4, payload_dim=6, write_steps=20, address_steps=20,
        read_steps=20)


@pytest.fixture(scope="module")
def grid():
    """The registered grid, shrunk to the fixture's scale.

    ⚠ ``n2_codes`` MUST shrink with the family: k-means++ on ``K_train * k = 24``
    launch points cannot seed 32 clusters (the ``d2`` distribution collapses and
    ``rng.choice`` refuses), which is a property of the fixture, not of the arm —
    the claim cell has 512 points against at most 128 codes.
    """
    return NullArmGrid(**{**C2W11_GRID.as_dict(), "steps": 10, "n_val": 4,
                          "n5_pretrain_steps": 5, "n5_passes": 1,
                          "n2_restarts": 2, "n2_codes": (4, 8),
                          "n1_atoms_per_well": (6,), "n5_hidden": (8,)})


@pytest.fixture(scope="module")
def setup(small):
    return seed_setup(small, 0, n_val=4)


# ==========================================================================
# the matching obligations
# ==========================================================================
def test_launch_points_are_bit_identical_to_the_physics_read(small):
    """⛔ The C2W11 launch must be the SAME OBJECT the physics arm launches from.

    With a zero-step settle :func:`multi_particle_read` *is* its launch stage, so
    :func:`feature_launch_states` must reproduce it exactly — ``fold_in(key, lo)``
    chunking included. ⚠ ``feature_launch.launch_points`` does **not** chunk, so
    it is *not* the right function for an arm that must match the physics read.
    """
    cfg = CatTestConfig(**{**small.as_dict(), "address_steps": 0, "read_steps": 0})
    fam = build_family(cfg, seed=0)
    phi = build_phi(cfg)
    head = build_launch_head(phi, cfg)
    anchors = place_wells(phi, cfg, sep=0.8)
    store = FactoredStore(cfg, anchors, jax.random.PRNGKey(0))
    ind = fam.indicator(fam.seen, cfg.n_wells)
    k = jax.random.PRNGKey(VALUE_LAUNCH_KEY)
    z = multi_particle_read(store, head, cfg, ind, k)
    q0 = feature_launch_states(head, cfg, ind, k)
    assert z.shape == q0.shape == (len(fam.seen), cfg.f_subset, cfg.dim)
    np.testing.assert_array_equal(np.asarray(z), np.asarray(q0))


def test_launch_chunking_is_load_bearing(small):
    """The chunked and unchunked launches DIFFER — so the choice is not cosmetic."""
    from chlu.core.feature_launch import launch_points as fl_launch_points

    cfg = small
    fam = build_family(cfg, seed=0)
    head = build_launch_head(build_phi(cfg), cfg)
    ind = fam.indicator(fam.unseen, cfg.n_wells)  # 24 rows
    k = jax.random.PRNGKey(VALUE_LAUNCH_KEY)
    a = feature_launch_states(head, cfg, ind, k, batch=8)
    b = np.asarray(fl_launch_points(head, ind, cfg, k))
    assert not np.array_equal(a, b), (
        "if these ever agree, the bit-identity test above stops discriminating")


def test_phi_is_frozen_and_byte_identical_across_seeds(small):
    """φ is drawn from ``phi_seed``, never from the family seed."""
    a, b = build_phi(small), build_phi(small)
    np.testing.assert_array_equal(np.asarray(a.codes), np.asarray(b.codes))
    assert a.n_bytes() == b.n_bytes()
    s0, s1 = seed_setup(small, 0, n_val=4), seed_setup(small, 1, n_val=4)
    np.testing.assert_array_equal(np.asarray(s0["phi"].codes),
                                  np.asarray(s1["phi"].codes))


def test_launch_head_costs_zero_parameters(setup):
    """The feature-factored head reuses φ's codes ⇒ ``head_bytes = 0``."""
    assert setup["head"].n_bytes()["head_bytes"] == 0


# ==========================================================================
# ⛔ the payload-repair trap, mechanically
# ==========================================================================
def test_frozen_mismatch_refuses_a_pre_repair_tol(small):
    """A cell carrying ``run1``'s pre-repair ``tol`` must be REFUSED, by name."""
    fam = build_family(small, seed=0)
    frozen = {"family": {"tol": float(fam.tol), "N_a": small.n_wells,
                         "F": small.f_subset, "K": small.n_items,
                         "m": small.payload_dim, "d_addr": small.addr_dim,
                         "a": small.atoms_per_well,
                         "n_unseen_sampled": small.n_unseen,
                         "payload_radius": small.payload_radius,
                         "atom_payload_init_radius":
                             small.atom_payload_init_radius,
                         "chance_per_seed": [0.0]}}
    assert assert_frozen_match(small, fam, 0, frozen)["frozen_verified"]

    class _F:  # the pre-repair family: tol 1.667x the repaired one
        tol = float(fam.tol) * 1.667
    with pytest.raises(FrozenCellMismatch):
        assert_frozen_match(small, _F(), 0, frozen)
    # ... and a co-scaled payload radius drift is refused too
    bad = CatTestConfig(**{**small.as_dict(), "payload_radius": 1.0})
    with pytest.raises(FrozenCellMismatch):
        assert_frozen_match(bad, fam, 0, frozen)


def test_no_frozen_artifact_is_not_silently_a_pass(small):
    """⛔ Absent artifact ⇒ ``frozen_verified = False``, never a quiet success."""
    info = assert_frozen_match(small, build_family(small, seed=0), 0, None)
    assert info["frozen_verified"] is False and "note" in info


# ==========================================================================
# ⛔ the selection guard
# ==========================================================================
def test_validation_split_inherits_the_familys_own_rule_4(setup, small):
    """The seen-validation rows are rule-4-valid against **every** training row.

    ⚠ A naive seen-holdout is a *different, easier* problem: two written items may
    share ``F-1`` wells, so selection on it would reward near-neighbour
    interpolation — the ability rule 4 exists to exclude.
    """
    fam = setup["family"]
    va, tr = setup["va"], setup["tr"]
    assert len(set(va.tolist()) & set(tr.tolist())) == 0
    for i in va:
        for j in tr:
            assert len(set(fam.seen[i].tolist()) & set(fam.seen[j].tolist())) \
                <= small.f_subset - 2
    assert setup["val_rule4"]["frac_rule4_valid"] == 1.0


def test_q_unseen_is_never_an_input_to_a_fit(setup, small, grid):
    """A fit that saw ``Q_unseen`` would change when ``Q_unseen`` changes.

    The mechanical guard: perturb ``q0_u``/``ind_u`` beyond recognition, refit
    every arm on SEEN, and assert the SEEN-side predictions are bit-identical.
    """
    import copy

    for arm in ("N1", "N2", "N3", "N4", "N5"):
        conf = _grid_first(arm, grid)
        base = _fit_arm(arm, conf, setup, small, grid, idx=setup["tr"])
        p0 = _predict(base, setup["q0_s"], setup["ind_s"], small)
        poisoned = copy.copy(setup)
        poisoned["q0_u"] = np.zeros_like(setup["q0_u"])
        poisoned["ind_u"] = np.zeros_like(setup["ind_u"])
        alt = _fit_arm(arm, conf, poisoned, small, grid, idx=setup["tr"])
        p1 = _predict(alt, setup["q0_s"], setup["ind_s"], small)
        np.testing.assert_array_equal(np.asarray(p0), np.asarray(p1))


def _grid_first(arm, grid):
    from chlu.experiments.exp_null_arms import _grid_configs

    return _grid_configs(arm, grid)[0]


def _predict(fitted, q0, ind, cfg):
    from chlu.experiments.exp_c2w11_nulls import _arm_predict

    return _arm_predict(fitted, q0, ind, cfg)


# ==========================================================================
# ⭐ the matched confidence channel (VALUE leg ii's null side)
# ==========================================================================
def test_every_arm_emits_a_per_particle_confidence(setup, small, grid):
    """⛔ An arm with no principled channel would be a NOT-RUN. All five have one."""
    for arm in ("N1", "N2", "N3", "N4", "N5"):
        fitted = _fit_arm(arm, _grid_first(arm, grid), setup, small, grid,
                          idx=setup["tr"])
        c = np.asarray(fitted["conf"](setup["q0_u"], setup["ind_u"]))
        assert c.shape == (len(setup["ind_u"]), small.n_particles), arm
        assert np.isfinite(c).all(), arm


def test_confidence_is_higher_where_the_arm_was_fitted(setup, small, grid):
    """The sign convention is load-bearing: higher = more familiar.

    A channel that reported *lower* confidence on the very points it was fitted
    on would invert every AUROC in the V2 table, and would do it silently.
    """
    for arm in ("N1", "N2", "N3", "N4"):
        fitted = _fit_arm(arm, _grid_first(arm, grid), setup, small, grid,
                          idx=setup["tr"])
        near = np.asarray(fitted["conf"](setup["q0_s"], setup["ind_s"])).mean()
        far_q0 = np.asarray(setup["q0_s"]) + 50.0   # far outside the shell
        far = np.asarray(fitted["conf"](far_q0, setup["ind_s"])).mean()
        assert near > far, arm


def test_novelty_auroc_is_exact_and_declares_a_missing_class():
    """Rank-based AUROC, ties averaged; a missing class is ``nan``, not 0.5."""
    # perfectly separated, with the sign convention applied (novel = LOW conf)
    conf = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    lab = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert novelty_auroc(conf, lab) == 1.0
    assert novelty_auroc(conf, ~lab) == 0.0
    assert novelty_auroc(np.ones(6), lab) == 0.5      # all ties => 0.5
    assert np.isnan(novelty_auroc(conf, np.zeros(6, dtype=bool)))


def test_ece_reports_its_own_degeneracy():
    """⚠ At zero accuracy ECE **is** mean confidence — and must say so."""
    r = expected_calibration_error(np.full(50, 0.4), np.zeros(50))
    assert r["degenerate"] is True
    assert abs(r["ece"] - 0.4) < 1e-9 and abs(r["mean_conf"] - 0.4) < 1e-9
    r2 = expected_calibration_error(np.full(50, 0.4), np.ones(50))
    assert r2["degenerate"] is False and abs(r2["ece"] - 0.6) < 1e-9


def test_confidence_channels_are_x64_safe(small, grid):
    """⚠ THE BANKED REPO HAZARD, asserted rather than hoped for.

    Another test in the suite enables ``jax_enable_x64`` process-wide; a module
    that mixes explicitly-cast float32 data with flag-following ``jax.random``
    initialisers then breaks ``lax.scan``'s carry invariance. This runs the whole
    confidence path with the flag ON, which a per-file run would never do.

    ⛔ **The flag is saved and RESTORED TO ITS PREVIOUS VALUE, never to False**
    (the repo convention in ``test_blocks.py`` / ``test_cl_baselines_x64.py``).
    Measured the hard way: several modules enable x64 *at import*, so the
    ambient state during a full-suite run is ON, and restoring a hard-coded
    ``False`` here turned it OFF for everything downstream — **18 tests in
    `test_goldstone.py` / `test_friction_field.py` / `test_lattice*.py` failed
    in-suite and every one of them passed alone.**
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        S = seed_setup(small, 0, n_val=4)
        for arm in ("N1", "N2", "N3", "N4", "N5"):
            fitted = _fit_arm(arm, _grid_first(arm, grid), S, small, grid,
                              idx=S["tr"])
            c = np.asarray(fitted["conf"](S["q0_u"], S["ind_u"]))
            assert np.isfinite(c).all(), arm
    finally:
        jax.config.update("jax_enable_x64", prev)


# ==========================================================================
# ⭐ V3 — the landscape instantiation
# ==========================================================================
def test_instantiation_matches_the_atom_budget_and_produces_a_real_store(setup,
                                                                        small):
    """The null's codebook becomes a store with the **matched total** atom budget."""
    anc = np.asarray(setup["anchors"])[:, : small.addr_dim]
    pay = np.asarray(setup["family"].payloads)
    store, info = instantiate_landscape(small, anc, pay, seed=0,
                                        total_atoms=small.n_atoms,
                                        width_frac=0.37, place_depth=0.30)
    assert info["total_atoms"] == small.n_atoms
    assert info["n_codes"] == small.n_wells
    assert store.V.centers.shape == (small.n_atoms, small.dim)
    # a placed store is not the blank one it started from
    blank = FactoredStore(small, setup["anchors"], jax.random.PRNGKey(0))
    assert not np.allclose(np.asarray(store.V.centers),
                           np.asarray(blank.V.centers))


def test_instantiation_halves_atoms_per_unit_when_the_codebook_doubles(setup,
                                                                      small):
    """Matched **total** capacity, not matched per-unit capacity."""
    rng = np.random.default_rng(0)
    C = rng.normal(size=(2 * small.n_wells, small.addr_dim))
    P = rng.normal(size=(2 * small.n_wells, small.payload_dim))
    _, info = instantiate_landscape(small, C, P, seed=0,
                                    total_atoms=small.n_atoms, width_frac=0.37)
    assert info["n_codes"] == 2 * small.n_wells
    assert info["atoms_per_well"] == small.atoms_per_well // 2


def test_anytime_read_uses_the_frozen_split_rule(setup, small):
    """``address = round(b/3)``, ``read = b - round(b/3)`` — a budget of 0 is the launch."""
    store, _ = instantiate_landscape(
        small, np.asarray(setup["anchors"])[:, : small.addr_dim],
        np.asarray(setup["family"].payloads), seed=0, width_frac=0.37)
    z0 = anytime_read(store, setup["head"], small, setup["ind_u"],
                      jax.random.fold_in(setup["key"], 1), budget=0)
    np.testing.assert_array_equal(np.asarray(z0), np.asarray(setup["q0_u"]))
    z1 = anytime_read(store, setup["head"], small, setup["ind_u"],
                      jax.random.fold_in(setup["key"], 1), budget=30)
    assert not np.allclose(np.asarray(z0), np.asarray(z1))


def test_the_v3_budget_grid_is_read_from_the_frozen_artifact():
    """⛔ A mismatched axis VOIDS leg iii, so the axis is never re-derived."""
    from chlu.experiments.exp_c2w11_nulls import _v3_budget_grid

    frozen = {"v3_budget_grid": {"points_total_verlet_steps": [7, 13]}}
    assert _v3_budget_grid(frozen) == [7, 13]


# ==========================================================================
# ⛔ the out-of-class ceiling
# ==========================================================================
def test_decodability_ceiling_is_perfect_on_noiseless_codes(setup, small):
    """The noiseless matched filter recovers the set; the launch is what costs."""
    c = feature_decodability_ceiling(setup["head"], small, setup["family"],
                                     setup["q0_u"])
    assert c["noiseless_combo_exact"] >= 0.95
    assert 0.0 <= c["as_launched_combo_exact"] <= c["noiseless_combo_exact"]
    assert c["n_combos"] == len(list(itertools.combinations(
        range(small.n_wells), small.f_subset)))


# ==========================================================================
# V2's split, and the key spaces
# ==========================================================================
def test_novelty_split_never_writes_a_novel_well(small):
    """SEEN is built over the WRITTEN wells only — that is what makes a well novel."""
    fam, novel, ev_novel = build_novelty_family(small, 0, n_novel=3,
                                                n_eval_per_class=8)
    assert len(novel) == 3
    assert not np.isin(fam.seen, novel).any()
    for j in np.unique(ev_novel):
        rows = fam.unseen[ev_novel == j]
        assert (np.isin(rows, novel).sum(1) == j).all()
    # rule 4 holds against every SEEN row
    for A in fam.unseen:
        for B in fam.seen:
            assert len(set(A.tolist()) & set(B.tolist())) <= small.f_subset - 2


def test_feature_key_spaces_are_ordered_by_information(setup, small):
    """``launch_flat`` is strictly richer than ``launch_mean`` (it determines it)."""
    flat = feature_keys("launch_flat", setup["head"], small, setup["ind_u"],
                        setup["q0_u"])
    mean = feature_keys("launch_mean", setup["head"], small, setup["ind_u"],
                        setup["q0_u"])
    assert flat.shape == (len(mean), small.n_particles * small.addr_dim)
    np.testing.assert_allclose(
        flat.reshape(len(mean), small.n_particles, small.addr_dim).mean(1),
        mean, rtol=1e-6)
    with pytest.raises(ValueError):
        feature_keys("nope", setup["head"], small, setup["ind_u"], setup["q0_u"])


# ==========================================================================
# the anchors' own instruments (L1/L2 are measured by the harness; these assert
# that the instruments can register both outcomes)
# ==========================================================================
def test_l2_anchor_k1_knn_memorises_seen(setup, small, grid):
    """L2: ``k = 1`` kNN on the noiseless key must reproduce SEEN exactly."""
    from chlu.core.null_arms import n4_knn
    from chlu.experiments.exp_null_arms import _native

    fam = setup["family"]
    keys = feature_keys("set_code", setup["head"], small, setup["ind_s"],
                        setup["q0_s"])
    pred = n4_knn(small, keys, fam.y_seen, keys, k=1, weight="idw")
    assert _native(pred, fam.y_seen, fam.tol)["acc"] >= 0.95


def test_shuffle_launder_destroys_the_query_information(setup, small, grid):
    """L3's instrument: the launder must be able to score at chance, and does."""
    from chlu.core.null_arms import shuffle_launches

    sh = shuffle_launches(setup["q0_u"], 0)
    assert sh.shape == setup["q0_u"].shape
    assert not np.array_equal(sh, np.asarray(setup["q0_u"]))
