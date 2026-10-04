"""Tests for the factored store / cat test (C2W5 ``orgdiv-cat-test``).

Every test here asserts a property the wave's verdict depends on, and several of
them encode a defect that was found by RUNNING the harness (each is named for the
defect it prevents from coming back).
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
    byte_ratio,
    chance_accuracy,
    effective_s,
    exact_set_accuracy,
    fit_readers,
    min_separation,
    multi_particle_read,
    occupancy_precision,
    place_wells,
    query_identifiability,
    reader_bytes,
    score_curve,
    well_write_loss,
    write_wells,
)


@pytest.fixture(scope="module")
def small():
    """A small but STRUCTURALLY IDENTICAL family (same rules, same read)."""
    return CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=24,
                         atoms_per_well=6, addr_dim=4, payload_dim=6,
                         write_steps=40, address_steps=60, read_steps=60)


# ==========================================================================
# K2 / rule 4 — the family's proof obligations
# ==========================================================================
def test_rule4_overlap_holds_for_every_heldout_query(small):
    """⭐ Rule 4 is a CONSTRUCTION: every unseen A shares <= F-2 wells with every
    stored B, so no single stored row can be the answer."""
    fam = build_family(small, seed=0)
    seen = fam.seen
    for A in fam.unseen:
        ov = np.array([len(set(A.tolist()) & set(B.tolist())) for B in seen])
        assert ov.max() <= small.f_subset - 2
    assert fam.k2["overlap_ok"]
    assert fam.k2["max_overlap"] <= small.f_subset - 2


def test_a_split_with_no_valid_heldout_is_REJECTED_not_repaired():
    """prereg §2.3: a failing split is rejected. At (N_a=8, F=3, K=16) the
    rule-4-valid held-out set is empty and construction must RAISE."""
    cfg = CatTestConfig(n_wells=8, f_subset=3, n_items=16, n_unseen=8)
    with pytest.raises(ValueError, match="EMPTY"):
        build_family(cfg, seed=0)


def test_Na16_at_K128_is_registered_forbidden():
    """prereg §2.4 registers N_a = 16 at K >= 128 as FORBIDDEN, and the code
    refuses it rather than silently producing an empty split."""
    with pytest.raises(ValueError, match="FORBIDDEN"):
        build_family(CatTestConfig(n_wells=16, n_items=128), seed=0)


def test_K2_second_assertion_is_unsatisfiable_at_m_equals_1():
    """⛔ The finding that forced the registered deviation: with a SCALAR payload,
    128 stored y-values crowd the line, so ``min_B ||y(A)-y(B)|| >= tol`` fails on
    almost every held-out query. It is a property of the family, not of a seed."""
    cfg1 = build_family(CatTestConfig(payload_dim=1), seed=0)
    cfg8 = build_family(CatTestConfig(payload_dim=8), seed=0)
    assert cfg1.k2["frac_payload_sep_ok"] < 0.05
    assert cfg8.k2["frac_payload_sep_ok"] == 1.0
    assert not cfg1.k2["payload_sep_ok"]
    assert cfg8.k2["payload_sep_ok"]


def test_insertion_order_is_reshuffled_per_seed(small):
    """Rule 1: ``y`` may not be recoverable from row order."""
    o0 = build_family(small, seed=0).order
    o1 = build_family(small, seed=1).order
    assert not np.array_equal(o0, o1)


# ==========================================================================
# phi — frozen, identical on every arm
# ==========================================================================
def test_phi_is_frozen_across_seeds_and_arms(small):
    """⛔ ``PhiMismatchError`` precedent: re-seeding the family must NOT re-draw
    phi, or the arms are not comparable."""
    a, b = build_phi(small), build_phi(small)
    assert np.array_equal(np.asarray(a.codes), np.asarray(b.codes))
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    assert a.n_bytes() == b.n_bytes() > 0


def test_launch_pins_the_payload_block_to_zero(small):
    """The anti-decoration guard: nothing may hand the read its answer."""
    phi = build_phi(small)
    ind = jnp.zeros((small.n_wells,)).at[jnp.array([0, 1, 2])].set(1.0)
    q0 = phi.launch(ind, jax.random.PRNGKey(0), small.query_sigma)
    assert q0.shape == (small.n_particles, small.dim)
    assert float(jnp.abs(q0[:, small.addr_dim:]).max()) == 0.0


def test_set_code_is_permutation_invariant(small):
    phi = build_phi(small)
    i1 = jnp.zeros((small.n_wells,)).at[jnp.array([1, 5, 9])].set(1.0)
    i2 = jnp.zeros((small.n_wells,)).at[jnp.array([9, 1, 5])].set(1.0)
    assert np.allclose(np.asarray(phi.set_code(i1)), np.asarray(phi.set_code(i2)))


# ==========================================================================
# placement — a pure function of (frozen codes, shared policy params)
# ==========================================================================
def test_placement_is_pure_and_shuffle_invariant(small):
    """prereg §6 rule 6 (C4/C5): placement carries NO per-item table, so it is
    invariant under any permutation of the write stream. The allocation-shuffle
    test is therefore exact, not approximate."""
    phi = build_phi(small)
    u1 = place_wells(phi, small, sep=0.9)
    u2 = place_wells(phi, small, sep=0.9)
    assert np.array_equal(u1, u2)
    assert min_separation(u1) >= 0.9 - 1e-6


# ==========================================================================
# the store, the write, and the reach defect that was found by running it
# ==========================================================================
def test_masked_write_is_local_in_parameter_space(small):
    """Writing well j must leave every other well's atoms BIT-IDENTICAL."""
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    st = FactoredStore(small, u, jax.random.PRNGKey(0))
    before = np.asarray(st.V.centers).copy()
    st2, _ = write_wells(st, small, u, fam.payloads, jax.random.PRNGKey(1),
                         order=np.array([0]))
    after = np.asarray(st2.V.centers)
    rows = np.asarray(st.group_rows(0), dtype=bool)
    assert not np.array_equal(before[rows], after[rows])          # well 0 moved
    assert np.array_equal(before[~rows], after[~rows])            # nothing else did


def test_payload_block_init_is_on_the_target_shell_not_scattered(small):
    """⛔ The defect that made the write inert. With the historical
    ``N(0, init_scale)`` scatter the payload block of an atom starts at radius
    ``sqrt(m)`` from a target at radius 1, i.e. ``exp(-m/2s^2)`` of signal, and
    every well relaxed to the origin (``lambda_min = 2*alpha`` exactly)."""
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    st = FactoredStore(small, u, jax.random.PRNGKey(0))
    r = np.linalg.norm(np.asarray(st.V.centers)[:, small.addr_dim:], axis=1)
    assert np.allclose(r, small.payload_radius, atol=1e-5)
    # ...and with the mechanism OFF the scatter is back and is ~sqrt(m)
    off = FactoredStore(small.__class__(**{**small.as_dict(),
                                           "atom_payload_init_radius": 0.0}),
                        u, jax.random.PRNGKey(0))
    r_off = np.linalg.norm(np.asarray(off.V.centers)[:, small.addr_dim:], axis=1)
    assert float(np.median(r_off)) > 1.5 * small.payload_radius


def test_payloads_are_unit_norm_and_live_only_in_the_store(small):
    fam = build_family(small, seed=0)
    r = np.linalg.norm(fam.payloads, axis=1)
    assert np.allclose(r, small.payload_radius)
    # y is the SUM over the item's wells (registered primary)
    ind = fam.indicator(fam.seen, small.n_wells)
    assert np.allclose(ind @ fam.payloads, fam.y_seen)


def test_endpoint_write_loss_is_reevaluated_on_the_final_store(small):
    """⛔ The per-well losses collected during the loop are STALE: a well written
    early is scored against a landscape the later wells have not been dug into.
    K1 is adjudicated on the re-evaluated endpoint number."""
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    st = FactoredStore(small, u, jax.random.PRNGKey(0))
    st, rep = write_wells(st, small, u, fam.payloads, jax.random.PRNGKey(1))
    assert "endpoint_write_loss" in rep and "per_well_last_loss_mean" in rep
    assert np.isfinite(rep["endpoint_write_loss"])
    assert rep["endpoint_write_loss"] != rep["per_well_last_loss_mean"]


def test_well_write_loss_matches_the_shipped_objective_at_m_equals_1():
    """The generalised objective is the SAME OBJECT as
    ``train_memory.write_loss`` when the payload block is one coordinate."""
    from chlu.training.train_memory import write_loss

    cfg = CatTestConfig(n_wells=4, f_subset=2, n_items=4, addr_dim=2,
                        payload_dim=1, atoms_per_well=4)
    u = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    st = FactoredStore(cfg, u, jax.random.PRNGKey(0))
    tgt = jnp.asarray(np.concatenate([u, np.ones((4, 1))], axis=1),
                      dtype=jnp.float32)
    k = jax.random.PRNGKey(3)
    mine = float(well_write_loss(st.V, tgt, k, addr_dim=2, payload_dim=1,
                                 n_perturb=8, sigma_addr=0.25, sigma_pay=0.6,
                                 margin=0.15, barrier=0.2, crowd_targets=tgt))
    theirs = float(write_loss(st.V, tgt, k, n_perturb=8, sigma_addr=0.25,
                              sigma_pay=0.6, margin=0.15, barrier=0.2,
                              payload_index=2, barrier_pairs="nn",
                              crowd_targets=tgt))
    assert mine == pytest.approx(theirs, rel=1e-5)


# ==========================================================================
# the effective-`s` instrument
# ==========================================================================
def test_effective_s_recovers_a_known_gaussian_width():
    """Zero free parameters: fit ``A exp(-r^2/2s^2)`` to a landscape whose width
    we chose, and get that width back."""
    class Toy:
        def __call__(self, q):
            return -jnp.exp(-jnp.sum(q**2) / (2 * 0.37**2))

    out = effective_s(Toy(), np.zeros(3), s_hint=0.37, n_rays=8, n_r=24,
                      confine=0.0)
    assert out["s"] == pytest.approx(0.37, rel=0.05)
    assert out["r2"] > 0.99


def test_effective_s_returns_nan_on_a_landscape_with_no_well():
    """⛔ The defect this encodes: a pure CONFINEMENT bowl has a monotone radial
    profile and log-fits as a well, so without subtracting ``alpha ||q||^2`` the
    estimator reports a confident width for a landscape with no wells at all —
    and ``s`` is the blocking instrument the whole operating point is set on."""
    class Flat:
        def __call__(self, q):
            return 0.05 * jnp.sum(q**2)

    assert np.isfinite(effective_s(Flat(), np.zeros(3), s_hint=0.3)["s"])   # the trap
    assert not np.isfinite(
        effective_s(Flat(), np.zeros(3), s_hint=0.3, confine=0.05)["s"])


# ==========================================================================
# the reader class — the capacity bound is the metric's precondition
# ==========================================================================
def test_every_reader_carries_fewer_params_than_the_family_has_unknowns(small):
    """⭐ SP-1. The ground truth ``1_A -> y`` has ``N_a * m`` free parameters. A
    reader with at least that many can solve the family from the SEEN split with
    NO STORE, so the capacity bound is what makes rule 2 dischargeable."""
    fam = build_family(small, seed=0)
    z = np.random.default_rng(0).normal(
        size=(len(fam.seen), small.n_particles, small.dim))
    rd = fit_readers(z, fam.y_seen, anchors=np.zeros((small.n_wells, small.addr_dim)),
                     well_payloads=fam.payloads, seed=0)
    budget = small.n_wells * small.payload_dim
    for name, n in reader_bytes(rd).items():
        assert n < budget, f"reader {name} has {n} params, budget {budget}"


def test_SP1_out_of_class_probe_solves_the_family_without_a_store():
    """⛔ Declared out-of-class diagnostic, registered in PREREG SP-1: an OLS fit
    of y on the TRUE indicator (``N_a*m`` dof) recovers every payload exactly and
    generalises perfectly. It is the family's structural ceiling — reported, never
    scored as an arm or as a K4 leg.

    ⭐ The second assertion is the escape and it is the reason ``K`` matters: with
    ``K < N_a`` the system is UNDER-determined and the probe cannot recover ``v``.
    The registered design point has ``K = 128 > N_a = 32``, so it is squarely in
    the solvable regime."""
    cfg = CatTestConfig()  # the registered design point: K = 128 > N_a = 32
    fam = build_family(cfg, seed=0)
    Xs = fam.indicator(fam.seen, cfg.n_wells)
    Xu = fam.indicator(fam.unseen, cfg.n_wells)
    w, *_ = np.linalg.lstsq(Xs, fam.y_seen, rcond=None)
    assert np.abs(w - fam.payloads).max() < 1e-8
    assert exact_set_accuracy(Xu @ w, fam.y_unseen, fam.tol) == pytest.approx(1.0)

    under = CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=24)
    fu = build_family(under, seed=0)
    Xs2 = fu.indicator(fu.seen, under.n_wells)
    w2, *_ = np.linalg.lstsq(Xs2, fu.y_seen, rcond=None)
    assert np.abs(w2 - fu.payloads).max() > 1e-3  # K < N_a: NOT identifiable


def test_query_only_identifiability_is_rank_limited_by_d(small):
    """SP-2's upper squeeze: a linear reader on the set-code can explain at most
    ``~d/N_a`` of ``var(y)``, because the code has rank ``min(d, N_a)``."""
    fam = build_family(small, seed=0)
    q = query_identifiability(build_phi(small), fam, small)
    assert q["r2_unseen"] < 3.0 * q["rank_ceiling_d_over_Na"] + 0.05


def test_score_curve_is_monotone_in_tol(small):
    fam = build_family(small, seed=0)
    z = np.zeros((len(fam.unseen), small.n_particles, small.dim))
    rd = fit_readers(np.zeros((len(fam.seen), small.n_particles, small.dim)),
                     fam.y_seen, seed=0, which=("sum_linear",))
    c = score_curve(rd["sum_linear"], z, fam.y_unseen, fam.tol)
    vals = [c[k] for k in ("x0.25", "x0.5", "x1", "x2", "x4")]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:], strict=False))


# ==========================================================================
# the read — multi-particle, never a single settled point
# ==========================================================================
def test_read_is_multi_particle_and_shaped_per_particle(small):
    """⛔ Theorem O1: composition cannot live in a SINGLE settled point, so the
    read must expose P particles, not their aggregate."""
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    st = FactoredStore(small, u, jax.random.PRNGKey(0))
    ind = fam.indicator(fam.seen[:4], small.n_wells)
    z = multi_particle_read(st, phi, small, ind, jax.random.PRNGKey(2))
    assert z.shape == (4, small.n_particles, small.dim)
    assert small.n_particles >= 4


def test_blank_store_occupancy_precision_is_ABOVE_chance(small):
    """⛔ A finding, encoded as a test, because it changes what the number means.

    Occupancy precision above ``F/N_a`` is **not** evidence that the store did
    anything: the wells are placed at the frozen query codes, so the launch
    geometry alone is a matched filter and an UNWRITTEN store already scores
    above chance. The admissible baseline for occupancy precision is therefore the
    BLANK STORE, never ``F/N_a`` — and that is how the harness reports it."""
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    blank = FactoredStore(small, u, jax.random.PRNGKey(0))
    z = multi_particle_read(blank, phi, small,
                            fam.indicator(fam.unseen, small.n_wells),
                            jax.random.PRNGKey(2))
    prec = occupancy_precision(z, u, fam.unseen)
    assert prec > small.f_subset / small.n_wells


# ==========================================================================
# ledgers
# ==========================================================================
def test_corrected_byte_law_agrees_with_the_prereg_form_at_d4_m1():
    """`harness-debt`'s ``[A(D+2)+d]/(d+m)`` and prereg §5.2's ``1.4A + 0.8`` are
    the SAME law at ``d=4, m=1`` (``D = 5``): both give 5.00x at ``a = 12``."""
    b = byte_ratio(CatTestConfig(addr_dim=4, payload_dim=1, atoms_per_well=12))
    assert b["A_atoms_per_item"] == pytest.approx(3.0)
    assert b["ratio_corrected"] == pytest.approx(5.00, rel=1e-6)
    assert b["ratio_prereg_5p2_form"] == pytest.approx(5.00, rel=1e-6)


def test_chance_is_the_constant_predictor(small):
    fam = build_family(small, seed=0)
    ch = chance_accuracy(fam.y_seen, fam.y_unseen, fam.tol)
    assert 0.0 <= ch <= 1.0


def test_store_bytes_and_phi_bytes_are_both_ledgered(small):
    phi = build_phi(small)
    u = place_wells(phi, small, sep=0.9)
    st = FactoredStore(small, u, jax.random.PRNGKey(0))
    assert st.n_bytes() == small.n_atoms * (small.dim + 2) * 4
    assert phi.n_bytes() == (small.n_wells * small.addr_dim
                             + small.n_particles * small.addr_dim) * 4


# ==========================================================================
# claim-form discipline (prereg §2.6) — enforced mechanically
# ==========================================================================
def test_no_well_is_ever_named_semantically():
    """⛔ 'the fur well' and every semantic naming of a well is forbidden in every
    artifact, including code comments and figure captions."""
    import pathlib

    import chlu.core.factored_store as fs
    import chlu.experiments.exp_cat_test as ct

    import re

    banned = ("fur", "whisker", "feather", "wing", "four-legs", "fourlegs",
              "cat well", "dog well", "crow")
    for mod in (fs, ct):
        text = pathlib.Path(mod.__file__).read_text().lower()
        for w in banned:
            assert not re.search(rf"\b{re.escape(w)}\b", text), \
                f"{w!r} appears in {mod.__file__}"
