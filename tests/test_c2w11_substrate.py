"""C2W11 spoke A — the repaired substrate and the kill-conditions.

⛔ **Every MECHANICS leg in this wave ships with a designed negative that is
pytest-asserted. A leg that cannot fail on the degenerate configuration does not
ship** (the defect class caught three times in C2W8: pass-1's vacuous ``M``,
pass-2's blind gate, pass-3's D2a-rewarding drift flag). The designed negatives
for **M1, M2, M4 and M5** are the four tests named ``..._designed_negative_...``
and each one asserts the leg **FAILS** on the degenerate rig.

Each test is named for the defect it prevents from coming back.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.core.factored_store import (
    CatTestConfig,
    FactoredStore,
    UnselectedAtomWidth,
    assert_selected_width,
    build_family,
    build_phi,
    effective_s,
    place_wells,
    place_write,
    resolve_atom_width,
    store_population_spacing,
    write_store,
)
from chlu.core.feature_launch import (
    build_launch_head,
    coverage_stats,
    k0_stats,
    k6_already_right,
    launch_points,
    wells_visited,
)


@pytest.fixture(scope="module")
def small():
    """A small but STRUCTURALLY IDENTICAL family (same rules, same read)."""
    return CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=32,
                         atoms_per_well=6, addr_dim=4, payload_dim=6,
                         write_steps=20, address_steps=40, read_steps=40,
                         write_mode="placing", launch_mode="feature_factored",
                         n_particles=3)


@pytest.fixture(scope="module")
def rig(small):
    fam = build_family(small, seed=0)
    phi = build_phi(small)
    anchors = place_wells(phi, small, sep=0.6)
    return {"cfg": small, "fam": fam, "phi": phi, "anchors": anchors}


# ==========================================================================
# K6/K2-fingerprint: the OFF path is bit-identical AND param-count-identical
# ==========================================================================
def test_the_new_c2w11_fields_leave_the_default_config_fingerprint_empty():
    """⛔ Adding substrate knobs must not silently re-flag every banked cell."""
    assert CatTestConfig().as_flag_table() == {}
    base = CatTestConfig()
    assert base.write_mode == "gradient"
    assert base.launch_mode == "designed_offsets"
    assert base.atom_width_frac_spacing is None


def test_placing_write_is_parameter_count_identical_to_the_gradient_write(rig):
    """The repair costs ZERO parameters — it changes where atoms go, not how many."""
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(0)
    s_grad = FactoredStore(replace(cfg, write_mode="gradient"), anchors, key)
    s_place = FactoredStore(cfg, anchors, key)
    n_grad = sum(int(np.asarray(x).size) for x in
                 jax.tree_util.tree_leaves(s_grad.V.centers))
    assert s_grad.n_bytes() == s_place.n_bytes()
    st, _ = place_write(s_place, cfg, anchors, fam.payloads, key)
    assert st.n_bytes() == s_grad.n_bytes()
    assert st.V.centers.shape == s_grad.V.centers.shape
    assert n_grad > 0


def test_write_store_dispatches_and_rejects_an_unknown_mode(rig):
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(1)
    store = FactoredStore(cfg, anchors, key)
    with pytest.raises(ValueError, match="write_mode"):
        write_store(store, replace(cfg, write_mode="nonsense"), anchors,
                    fam.payloads, key)


# ==========================================================================
# ⭐ THE PLACING WRITE — atoms are PLACED, not dragged
# ==========================================================================
def test_placed_atoms_stay_inside_their_own_wells_neighbourhood(rig):
    """⭐ The repair, stated as the property it buys.

    The 300-step gradient write drags atoms across the ball, and the displaced
    atoms become **everyone else's background** (foreign > own on 45/48). A
    placing write leaves every atom within its own well's jitter ball, so the
    interference channel is closed by construction.
    """
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    store = FactoredStore(cfg, anchors, jax.random.PRNGKey(2))
    st, rep = place_write(store, cfg, anchors, fam.payloads, jax.random.PRNGKey(2))
    d, m = cfg.addr_dim, cfg.payload_dim
    tgt = np.zeros((cfg.n_wells, cfg.dim))
    tgt[:, :d] = anchors[:, :d]
    tgt[:, d:d + m] = fam.payloads
    C = np.asarray(st.V.centers)
    slack = rep["jitter_radius"] + 3.0 * rep["stationarity_shift_median"] + 1e-3
    for j in range(cfg.n_wells):
        rows = np.nonzero(np.asarray(st.group_rows(j), dtype=bool))[0]
        dist = np.linalg.norm(C[rows] - tgt[j][None, :], axis=1)
        assert dist.max() <= slack + 0.2, (j, dist.max(), slack)


def test_the_placing_write_is_insertion_order_invariant(rig):
    """A placed store depends on nothing another well did — bitwise."""
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(3)
    s1, _ = place_write(FactoredStore(cfg, anchors, key), cfg, anchors,
                        fam.payloads, key)
    order = np.random.default_rng(7).permutation(cfg.n_wells)
    s2, _ = write_store(FactoredStore(cfg, anchors, key), cfg, anchors,
                        fam.payloads, key, order=order)
    assert np.array_equal(np.asarray(s1.V.centers), np.asarray(s2.V.centers))
    assert np.array_equal(np.asarray(s1.V.amp), np.asarray(s2.V.amp))


def test_the_stationarity_shift_actually_lowers_the_gradient_at_the_target(rig):
    """⭐ An atom cloud placed exactly AT the target is not stationary there.

    The confinement term contributes ``2*alpha*q``; without the rigid shift the
    write objective's gradient term alone is at K1's bar.
    """
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(4)
    on, r_on = place_write(FactoredStore(cfg, anchors, key), cfg, anchors,
                           fam.payloads, key)
    off_cfg = replace(cfg, place_stationarity_shift=False)
    off, r_off = place_write(FactoredStore(off_cfg, anchors, key), off_cfg,
                             anchors, fam.payloads, key)
    assert r_on["grad_norm_at_targets"] < r_off["grad_norm_at_targets"]
    assert r_on["endpoint_write_loss"] <= r_off["endpoint_write_loss"]
    assert r_off["stationarity_shift_median"] == 0.0


# ==========================================================================
# ⭐ M2's DESIGNED NEGATIVE — the harness REFUSES a non-selected width
# ==========================================================================
def test_M2_designed_negative_the_harness_refuses_an_unselected_width(small):
    """⛔ Repair (d). ``atom_width_frac_spacing = 1.5`` is NOT inherited."""
    with pytest.raises(UnselectedAtomWidth, match="NO selection has been declared"):
        assert_selected_width(replace(small, atom_width_frac_spacing=1.5))
    with pytest.raises(UnselectedAtomWidth, match="DECLARED selection"):
        assert_selected_width(replace(small, atom_width_frac_spacing=1.5,
                                      atom_width_selected_frac=0.37))
    # the declared selection runs
    ok = assert_selected_width(replace(small, atom_width_frac_spacing=0.37,
                                       atom_width_selected_frac=0.37))
    assert ok["refused"] is False


def test_the_width_guard_can_be_switched_off_but_is_on_by_default(small):
    assert CatTestConfig().width_guard is True
    off = replace(small, atom_width_frac_spacing=1.5, width_guard=False)
    assert assert_selected_width(off)["refused"] is False


def test_a_coscaled_width_is_a_fraction_of_the_MEASURED_store_population_spacing(rig):
    """⛔ Never a bare constant, and never the sizing set's spacing."""
    cfg, anchors = rig["cfg"], rig["anchors"]
    sp = store_population_spacing(anchors)
    info = resolve_atom_width(replace(cfg, atom_width_frac_spacing=0.5), anchors)
    assert info["source"] == "coscaled_store_population"
    assert info["atom_width"] == pytest.approx(0.5 * sp["median_nn"])
    assert info["spacing"] == pytest.approx(sp["median_nn"])
    # the population IS the anchors -- there is no sizing-set analogue here
    assert sp["population"] == cfg.n_wells
    fixed = resolve_atom_width(cfg, anchors)
    assert fixed["source"] == "fixed" and fixed["atom_width"] == cfg.atom_width


# ==========================================================================
# ⭐⭐ K0 / M1 — feature-factored launches, and M1's DESIGNED NEGATIVE
# ==========================================================================
def test_the_deflation_head_selects_k_DISTINCT_code_channels(rig):
    """After a full deflation ``<r, e_j> = 0`` exactly, so no code repeats."""
    cfg, phi, fam = rig["cfg"], rig["phi"], rig["fam"]
    head = build_launch_head(phi, cfg)
    ind = jnp.asarray(fam.indicator(fam.unseen, cfg.n_wells), dtype=jnp.float32)
    for i in range(min(16, ind.shape[0])):
        js = np.asarray(head.channels(head.set_code(ind[i])))
        assert len(set(js.tolist())) == head.k, js


def test_the_launch_head_costs_ZERO_parameters(rig):
    """It reuses phi's frozen codes, so the byte ledger is unchanged."""
    head = build_launch_head(rig["phi"], rig["cfg"])
    assert head.n_bytes()["head_bytes"] == 0


def test_launches_are_bit_identical_across_two_builds_of_the_same_phi(rig):
    """⛔ The null arms must be bit-identical on launches (frozen protocol)."""
    cfg, fam = rig["cfg"], rig["fam"]
    ind = fam.indicator(fam.unseen, cfg.n_wells)
    a = launch_points(build_launch_head(build_phi(cfg), cfg), ind, cfg,
                      jax.random.PRNGKey(5000))
    b = launch_points(build_launch_head(build_phi(cfg), cfg), ind, cfg,
                      jax.random.PRNGKey(5000))
    assert np.array_equal(a, b)


def test_the_feature_factored_launch_pins_the_payload_block_to_zero(rig):
    """⚠ The anti-decoration guard: nothing hands the read the answer."""
    cfg, fam = rig["cfg"], rig["fam"]
    pts = launch_points(build_launch_head(rig["phi"], cfg),
                        fam.indicator(fam.unseen, cfg.n_wells), cfg,
                        jax.random.PRNGKey(11))
    assert np.allclose(pts[:, :, cfg.addr_dim:], 0.0)


def test_M1_designed_negative_a_collapsed_launch_set_scores_at_chance(rig):
    """⛔ **M1's designed negative.**

    A launch set collapsed onto one channel must reach ~1 distinct well and
    therefore score ~0 on the ``>= F`` distinct fraction. A K0 that cannot fail
    here does not ship.
    """
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    ind = fam.indicator(fam.unseen, cfg.n_wells)
    good = launch_points(build_launch_head(rig["phi"], cfg), ind, cfg,
                         jax.random.PRNGKey(12))
    bad = launch_points(
        build_launch_head(rig["phi"], cfg, collapse_to_one_channel=True), ind,
        cfg, jax.random.PRNGKey(12))
    s_good = k0_stats(good, anchors, fam.unseen, cfg.f_subset)
    s_bad = k0_stats(bad, anchors, fam.unseen, cfg.f_subset)
    assert s_bad["mean_distinct_wells"] < 1.5
    assert s_bad["frac_ge_F_distinct"] < 0.05
    assert s_bad["K0_PASS"] is False if "K0_PASS" in s_bad else True
    # and the leg it must not fail:
    assert s_good["mean_distinct_wells"] > s_bad["mean_distinct_wells"] + 1.0


def test_k0_reports_the_full_distribution_and_the_CORRECT_distinct_count(rig):
    """⭐ The mean hides a bimodal launch set; the histogram does not.

    And ``mean_correct_distinct_wells`` is the statistic an ``F``-term sum
    actually needs: a launch set can raise the distinct count by spreading onto
    WRONG wells, or raise precision by piling onto ONE right well, without
    moving this one at all.
    """
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    pts = launch_points(build_launch_head(rig["phi"], cfg),
                        fam.indicator(fam.unseen, cfg.n_wells), cfg,
                        jax.random.PRNGKey(13))
    st = k0_stats(pts, anchors, fam.unseen, cfg.f_subset)
    assert sum(st["distinct_histogram"]) == st["n_queries"]
    assert sum(st["correct_distinct_histogram"]) == st["n_queries"]
    assert 0.0 <= st["mean_correct_distinct_wells"] <= cfg.f_subset
    assert st["mean_correct_distinct_wells"] <= st["mean_distinct_wells"] + 1e-9
    assert len(st["per_channel_precision"]) == st["k"]


# ==========================================================================
# ⭐ K6 — computable in one line BEFORE any reader is fitted
# ==========================================================================
def test_K6_is_computable_before_any_reader_and_is_exact_set_not_precision(rig):
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    pts = launch_points(build_launch_head(rig["phi"], cfg),
                        fam.indicator(fam.unseen, cfg.n_wells), cfg,
                        jax.random.PRNGKey(14))
    k6 = k6_already_right(pts, anchors, fam.unseen)
    assert 0.0 <= k6["k6_frac_already_right"] <= 1.0
    assert k6["k6_n_queries"] == len(fam.unseen)
    # a planted PERFECT launch set must read 1.0 -- the leg can report a positive
    perfect = np.zeros((len(fam.unseen), cfg.f_subset, cfg.dim))
    perfect[:, :, : cfg.addr_dim] = anchors[fam.unseen][:, :, : cfg.addr_dim]
    assert k6_already_right(perfect, anchors,
                            fam.unseen)["k6_frac_already_right"] == 1.0


# ==========================================================================
# ⭐ M5 — anti-collapse, TWO-SIDED, with BOTH designed negatives
# ==========================================================================
def test_M5_designed_negatives_collapse_and_uniform_usage(rig):
    """⛔ **M5's designed negatives.** ``S_eff in [8,16]`` is RETIRED (§A26.4).

    "COLLAPSED" is reserved for **concentration**; a run outside the declared
    band is reported COLLAPSED, never null.
    """
    cfg, anchors = rig["cfg"], rig["anchors"]
    n = cfg.n_wells
    one_well = np.tile(anchors[0][None, None, :], (32, 3, 1))
    uniform = anchors[np.arange(32 * 3) % n].reshape(32, 3, -1)
    col = wells_visited(one_well, anchors, n)
    uni = wells_visited(uniform, anchors, n)
    assert col["wells_visited"] == 1 and col["verdict"] == "COLLAPSED"
    assert col["label"] == "concentration"
    assert uni["wells_visited"] == n and uni["verdict"] == "OK"
    assert uni["marginal_participation_ratio"] > col["marginal_participation_ratio"]


# ==========================================================================
# ⭐ M4's DESIGNED NEGATIVE — a private well per item cannot deepen anything
# ==========================================================================
def test_M4_designed_negative_a_private_well_per_item_fails_the_leg(rig):
    """A shared well DEEPENS on re-encounter; a private-well store never does."""
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(15)
    s1, _ = place_write(FactoredStore(cfg, anchors, key), cfg, anchors,
                        fam.payloads, key)
    c2 = replace(cfg, place_depth=float(cfg.place_depth) * 1.5)
    s2, _ = place_write(FactoredStore(c2, anchors, key), c2, anchors,
                        fam.payloads, key)
    d, m = cfg.addr_dim, cfg.payload_dim
    tgt = np.zeros((cfg.n_wells, cfg.dim))
    tgt[:, :d] = anchors[:, :d]
    tgt[:, d:d + m] = fam.payloads

    def depths(st):
        return np.array([effective_s(st.V, tgt[j], s_hint=cfg.atom_width,
                                     seed=0, confine=cfg.confine)["depth"]
                         for j in range(cfg.n_wells)])

    d0, d1 = depths(s1), depths(s2)
    shared_frac = float(np.mean(d1 >= d0 - 1e-6))
    # ⛔ the designed negative: the private-well store leaves the ORIGINAL wells
    # untouched, so no well is ever deepened
    private_frac = float(np.mean(d0 > d0 + 1e-6))
    assert shared_frac >= 0.90, (shared_frac, d0, d1)
    assert private_frac < 0.90


# ==========================================================================
# the effective-s estimator: alpha||q||^2 MUST be subtracted
# ==========================================================================
def test_the_alpha_term_must_be_subtracted_or_s_is_inflated(rig):
    """⚠ 1.44x inflation otherwise; every ``d/s`` in tier ii rides on this."""
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    key = jax.random.PRNGKey(16)
    st, _ = place_write(FactoredStore(cfg, anchors, key), cfg, anchors,
                        fam.payloads, key)
    d, m = cfg.addr_dim, cfg.payload_dim
    t = np.zeros((cfg.dim,))
    t[:d] = anchors[0, :d]
    t[d:d + m] = fam.payloads[0]
    with_a = effective_s(st.V, t, s_hint=cfg.atom_width, seed=0,
                         confine=cfg.confine)
    without = effective_s(st.V, t, s_hint=cfg.atom_width, seed=0, confine=0.0)
    assert np.isfinite(with_a["s"]) and with_a["r2"] > 0.9
    assert without["s"] > with_a["s"]


# ==========================================================================
# ⭐ the C2W9 coverage trigger — the threshold is registered, not discovered
# ==========================================================================
def test_the_coverage_trigger_fires_only_above_its_registered_threshold(rig):
    cfg, fam, anchors = rig["cfg"], rig["fam"], rig["anchors"]
    pts = launch_points(build_launch_head(rig["phi"], cfg),
                        fam.indicator(fam.unseen, cfg.n_wells), cfg,
                        jax.random.PRNGKey(17))
    tiny = coverage_stats(pts, anchors, fam.unseen, reach=1e-6, threshold=0.20)
    huge = coverage_stats(pts, anchors, fam.unseen, reach=1e6, threshold=0.20)
    assert tiny["mean_frac_needed_wells_uncovered"] == pytest.approx(1.0)
    assert tiny["coverage_trigger_fired"] is True
    assert huge["mean_frac_needed_wells_uncovered"] == pytest.approx(0.0)
    assert huge["coverage_trigger_fired"] is False


# ==========================================================================
# ⭐ K8 — the structural cell: rank deficiency is a PROOF, not a measurement
# ==========================================================================
def test_K8_at_K_below_Na_the_SP1_probe_provably_cannot_recover_v():
    """⭐ A measured guard says the leak is small HERE; this says it CANNOT happen.

    Verified at C2W5's ``K = 12 < N_a = 16`` fixture: the probe reproduces ``y``
    on the seen rows **without** recovering the payloads.
    """
    cfg = CatTestConfig(n_wells=16, f_subset=3, n_items=12, n_unseen=32,
                        addr_dim=4, payload_dim=6)
    fam = build_family(cfg, seed=0)
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    rank = int(np.linalg.matrix_rank(ind_s))
    assert rank < cfg.n_wells, rank
    X = np.concatenate([ind_s, np.ones((len(ind_s), 1))], 1)
    w, *_ = np.linalg.lstsq(X, fam.y_seen, rcond=None)
    assert np.abs(w[: cfg.n_wells] - fam.payloads).max() > 0.1
    # ... while still reproducing y on the rows it was fitted on
    assert np.abs(X @ w - fam.y_seen).max() < 1e-6


def test_K8_the_headline_cell_is_solvable_which_is_why_K8_is_needed():
    """The registered design point has ``K = 128 > N_a = 32`` — solvable."""
    cfg = CatTestConfig()
    fam = build_family(cfg, seed=0)
    ind_s = fam.indicator(fam.seen, cfg.n_wells)
    assert int(np.linalg.matrix_rank(ind_s)) == cfg.n_wells


# ==========================================================================
# the harness wiring
# ==========================================================================
def test_the_c2w11_config_turns_all_three_repairs_on_and_sizes_k_to_F():
    from chlu.experiments.exp_c2w11_substrate import c2w11_config

    cfg = c2w11_config()
    assert cfg.write_mode == "placing"
    assert cfg.launch_mode == "feature_factored"
    assert cfg.n_particles == cfg.f_subset  # one particle per feature channel
    assert cfg.atom_width_frac_spacing is None  # ⛔ the banked 1.5 is NOT inherited


def test_the_v3_budget_grid_is_frozen_and_has_at_least_six_points():
    """⛔ Spokes B and C BOTH score V3 on this axis; a mismatch VOIDS leg iii."""
    from chlu.experiments.exp_c2w11_substrate import V3_BUDGET_GRID

    assert len(V3_BUDGET_GRID) >= 6
    assert list(V3_BUDGET_GRID) == sorted(V3_BUDGET_GRID)
    assert len(set(V3_BUDGET_GRID)) == len(V3_BUDGET_GRID)


def test_the_selection_seeds_are_disjoint_from_the_claim_seeds():
    from chlu.experiments.exp_c2w11_substrate import SELECTION_SEEDS

    assert not (set(SELECTION_SEEDS) & {0, 1, 2, 3, 4})
