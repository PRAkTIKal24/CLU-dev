"""C2W8 stage 1 — the census instrument and its **designed negatives**.

*"A census instrument that cannot see a planted population cannot license a
kill"* (`PREREG-C2W8.md` §5 K1). The two assertions this wave's stage-2 gate
rests on are here:

* a hand-built store with **4 known never-read attractors** must read
  ``P >= 4 / n_live``;
* a hand-built store with **3 known near-duplicate pairs** must read
  ``M >= 3 / n_pairs``.

Plus the rules that keep the instrument honest: the two well populations stay
separate (mechanic 1), the prune leg is usage and not depth (mechanic 2), the
decay netting is exact (B1), and the K1 arithmetic is mechanical.
"""

import jax
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.core.well_lifecycle import (
    UNLOCK_THRESHOLD,
    census,
    designed_decay_factors,
    flatten_unused_groups,
    measure_theta_att,
    mergeable_pairs,
    own_foreign_site_depth,
    plant_item,
    unlock_verdict,
    well_states,
)
from chlu.experiments.usage_telemetry import UsageTelemetry


def _tiny_system(capacity=8, d_safe=0.05, seed=0, **over):
    """A small CluSystem whose atoms are cheap and whose gate admits neighbours."""
    # the two step budgets are defaults an `over` may raise (the close-out's
    # settle-side control needs a settle that has actually settled)
    over.setdefault("read_steps", 60)
    over.setdefault("address_steps", 40)
    cfg = CluSystemConfig(
        addr_dim=2, payload_dim=1, capacity=capacity, atoms_per_item=8,
        min_atoms=32, min_atoms_base=8, min_atoms_c=1.0, seed=seed,
        d_safe_override=float(d_safe), n_query_per_item=2, **over,
    )
    return build_system(cfg, key=jax.random.PRNGKey(seed), loud=False)


def _ring(n, r=0.6):
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


# --------------------------------------------------------------------------
# DESIGNED NEGATIVE 1 (K1): four planted never-read attractors must be seen
# --------------------------------------------------------------------------
def test_census_sees_four_planted_unread_attractors():
    sysm = _tiny_system(capacity=8)
    sites = _ring(8)
    for i in range(8):
        plant_item(sysm, i, sites[i], payload=0.1 * (i - 4), depth=0.8,
                   width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    for i in range(8):
        tel.note_admitted(i, t=i)
    # four wells are read, four are NEVER read -> the planted prunable population
    for i in (0, 1, 2, 3):
        tel.note_read(i, t=10, controller=sysm.controller.allocator)

    out = census(sysm, tel, well_budget=4, n_admitted=8, measure_capture=False)
    assert out["n_live"] == 8
    assert out["P"] >= 4.0 / out["n_live"], out
    # every planted unread well is an attractor, none is counted as eroded
    assert set(out["population_live_attractor_never_read"]) >= {4, 5, 6, 7}
    assert not set(out["population_eroded_not_attractor"]) & {4, 5, 6, 7}
    # and the counters really are item-id keyed and survive the census
    assert sysm.controller.allocator.read_hits == {0: 1, 1: 1, 2: 1, 3: 1}


# --------------------------------------------------------------------------
# DESIGNED NEGATIVE 2 (K1): three planted near-duplicate pairs must be seen
# --------------------------------------------------------------------------
def test_census_sees_three_planted_near_duplicate_pairs():
    sysm = _tiny_system(capacity=8, d_safe=0.001)
    base = _ring(3, r=0.7)
    ids = 0
    for k in range(3):  # three near-duplicate PAIRS: same payload, close centers
        for j in range(2):
            plant_item(sysm, ids, base[k] + np.array([0.004 * j, 0.0]),
                       payload=0.3 * k, depth=0.7, width=0.25, leak=0.01)
            ids += 1
    # two far-apart, distinct-payload wells that must NOT be called mergeable
    plant_item(sysm, 6, np.array([-0.9, 0.9]), payload=-0.9, depth=0.7,
               width=0.25, leak=0.01)
    plant_item(sysm, 7, np.array([0.9, -0.9]), payload=0.9, depth=0.7,
               width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    out = census(sysm, UsageTelemetry(), well_budget=4, n_admitted=8,
                 measure_capture=False)
    n_pairs = out["n_pairs"]
    assert n_pairs == 28
    assert out["M"] >= 3.0 / n_pairs, out
    found = {tuple(sorted((p["item_i"], p["item_j"]))) for p in out["mergeable_pairs"]}
    assert {(0, 1), (2, 3), (4, 5)} <= found
    assert (6, 7) not in found  # far apart AND payload-distinct


# --------------------------------------------------------------------------
# mechanic 1 — an eroded well is NOT prunable, and is counted separately
# --------------------------------------------------------------------------
def test_eroded_well_is_not_prunable_and_is_counted_separately():
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.01)
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.9, width=0.25, leak=0.01)
    plant_item(sysm, 2, sites[2], payload=-0.4, depth=0.0, width=0.25, leak=0.01)
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    states, theta = well_states(sysm, tel, measure_capture=False)
    by_id = {s.item_id: s for s in states}
    assert by_id[2].depth_raw == pytest.approx(0.0, abs=1e-6)
    # theta_att is 0.0 when capture is not measured, so the zero-depth well is
    # excluded by "depth > theta_att" alone: it is eroded, hence not prunable.
    assert not by_id[2].is_attractor
    assert by_id[2].eroded and not by_id[2].prunable
    assert by_id[0].is_attractor and by_id[0].prunable  # unread live attractor

    out = census(sysm, tel, well_budget=2, n_admitted=3, measure_capture=False)
    assert 2 in out["population_eroded_not_attractor"]
    assert 2 not in out["population_live_attractor_never_read"]


# --------------------------------------------------------------------------
# mechanic 2 — the prune leg is USAGE, not depth
# --------------------------------------------------------------------------
def test_prunable_is_usage_not_depth():
    """A deep-but-unread well is prunable; a shallow-but-read one is not.

    ⛔ This is the census-level form of K3 (`PREREG-C2W8.md` §5): if depth could
    substitute for usage here, the prune criterion would be a depth policy
    wearing a usage costume. The stage-2 verb's own K3 assertion rides on top of
    this and does not replace it.
    """
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.01)  # deep
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.05, width=0.25, leak=0.01)  # shallow
    flatten_unused_groups(sysm)

    tel = UsageTelemetry()
    tel.note_admitted(0, 0)
    tel.note_admitted(1, 0)
    for _ in range(5):  # the SHALLOW well is the frequently-read one
        tel.note_read(1, t=1, controller=sysm.controller.allocator)

    states, _ = well_states(sysm, tel, measure_capture=False)
    by_id = {s.item_id: s for s in states}
    assert by_id[0].depth_raw > by_id[1].depth_raw
    assert by_id[0].prunable is True     # deep, never read
    assert by_id[1].prunable is False    # shallow, read 5x


# --------------------------------------------------------------------------
# protection — the leak == 0 cohort is excluded from P
# --------------------------------------------------------------------------
def test_permanent_cohort_is_excluded_from_P():
    sysm = _tiny_system(capacity=4)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25,
               permanent=True, leak=0.0)
    plant_item(sysm, 1, sites[1], payload=0.4, depth=0.9, width=0.25, leak=0.01)
    flatten_unused_groups(sysm)
    out = census(sysm, UsageTelemetry(), well_budget=2, n_admitted=2,
                 measure_capture=False)
    assert 0 in out["population_protected"]
    assert out["n_prunable"] == 1  # only the non-protected unread attractor
    assert out["P"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# B1 — the decay netting is exact, and is reported beside the raw curve
# --------------------------------------------------------------------------
def test_decay_netting_is_exact_against_the_designed_law():
    sysm = _tiny_system(capacity=4, leak=0.05, stage_lifetimes=True)
    sites = _ring(4)
    plant_item(sysm, 0, sites[0], payload=0.0, depth=0.9, width=0.25, leak=0.05)
    flatten_unused_groups(sysm)
    d0, _ = sysm.well_fits()

    n_ticks = 4
    for _ in range(n_ticks):
        sysm.controller.decay(1)
        sysm._sync_decay()

    factors = designed_decay_factors(sysm.controller)
    predicted = float(np.exp(-0.05 * n_ticks))
    assert factors[0] == pytest.approx(predicted, rel=1e-4)

    states, _ = well_states(sysm, UsageTelemetry(), measure_capture=False)
    s0 = states[0]
    assert s0.depth_raw < float(d0[0])                       # the raw curve fell
    assert s0.depth_netted == pytest.approx(s0.depth_raw / factors[0], rel=1e-9)
    assert s0.depth_netted == pytest.approx(float(d0[0]), rel=0.05)  # netting restores it


def test_own_and_foreign_site_depth_split():
    sysm = _tiny_system(capacity=4, d_safe=0.001)
    plant_item(sysm, 0, np.array([0.0, 0.0]), payload=0.0, depth=0.6, width=0.25)
    plant_item(sysm, 1, np.array([0.02, 0.0]), payload=0.0, depth=0.6, width=0.25)
    flatten_unused_groups(sysm)
    z = np.zeros((sysm.store.dim,))
    own, foreign = own_foreign_site_depth(sysm.store, sysm._slot_of(0), z)
    assert own == pytest.approx(0.6, rel=1e-5)     # exactly what was planted
    assert foreign > 0.0                            # the near-duplicate is felt
    assert foreign < own


# --------------------------------------------------------------------------
# theta_att is MEASURED, and K1's arithmetic is mechanical
# --------------------------------------------------------------------------
def test_theta_att_is_measured_not_guessed():
    # two wells whose capture radius is below sigma_q => the floor sits at the
    # deepest of THOSE, not at a constant.
    out = measure_theta_att(depths=[0.9, 0.2, 0.05], capture=[0.4, 0.02, 0.01],
                            sigma_q=0.15)
    assert out["theta_att"] == pytest.approx(0.2)
    assert out["n_capturing"] == 1 and out["n_non_capturing"] == 2
    allcap = measure_theta_att([0.9, 0.8], [0.4, 0.5], 0.15)
    assert allcap["theta_att"] == 0.0


def test_unlock_verdict_is_mechanical():
    assert unlock_verdict([0.2, 0.1, 0.0], [0.0, 0.0, 0.0])["stage2_unlock"] is True
    assert unlock_verdict([0.0, 0.0], [0.09, 0.09])["stage2_unlock"] is True
    v = unlock_verdict([0.0, 0.01, 0.0], [0.0, 0.02, 0.0])
    assert v["stage2_unlock"] is False and v["kill"] is True
    # a single seed above threshold blocks KILL but need not UNLOCK on the mean
    v2 = unlock_verdict([0.0, 0.0, 0.12], [0.0, 0.0, 0.0])
    assert v2["kill"] is False
    assert UNLOCK_THRESHOLD == 0.05


# --------------------------------------------------------------------------
# K2 — the trash region's FIRST USE, and its safety (PREREG-C2W8 §5 K2)
# --------------------------------------------------------------------------
def _two_well_system(gamma_phi=False, seed=0):
    sysm = _tiny_system(capacity=4, seed=seed, gamma_phi=gamma_phi)
    sites = np.array([[0.6, 0.0], [-0.6, 0.0]])
    for i, s in enumerate(sites):
        plant_item(sysm, i, s, payload=0.3 * (2 * i - 1), depth=0.9, width=0.25)
    flatten_unused_groups(sysm)
    q0 = np.zeros((2, sysm.store.dim), dtype=np.float32)
    q0[:, :2] = sites + 0.02
    return sysm, sites, q0


def test_gamma_phi_off_is_bit_identical_and_parameter_count_identical():
    """⛔ OFF must be the pre-build path, exactly — bits AND parameter count.

    ⚠ Even an EMPTY field is not bit-identical: the integrator composes
    `1 - (1 - gamma)(1 - gamma_phi)` and `1 - (1 - g) * 1.0 != g` in floating
    point. OFF therefore means *no field is attached at all*, and this test is
    what pins that.
    """
    import equinox as eqx

    off, _, q0 = _two_well_system(gamma_phi=False)
    on, _, _ = _two_well_system(gamma_phi=True)

    assert off.trash is None and off.model().friction_field is None
    assert on.trash is not None and on.trash.k == 0     # ON but with no holes yet

    # (i) parameter-count-identical: the ON-with-no-holes model must carry no
    # extra learnable leaves, and the OFF model must be the shipped tree.
    def n_params(m):
        return int(sum(np.asarray(x).size
                       for x in jax.tree_util.tree_leaves(
                           eqx.filter(m, eqx.is_inexact_array))))

    assert n_params(off.model()) == n_params(_tiny_system(capacity=4).model())
    assert n_params(on.model()) == n_params(off.model())   # 0 holes = 0 bytes
    assert off.trash_bytes() == 0 and on.trash_bytes() == 0

    # (ii) bit-identical reads against a system built with the flag absent
    ref, _, _ = _two_well_system(gamma_phi=False)
    a = np.asarray(off.read(q0).state.q_star)
    b = np.asarray(ref.read(q0).state.q_star)
    assert np.array_equal(a, b)


def test_gamma_phi_hole_at_a_well_destroys_its_retrievability():
    """K2 designed negative (a): the field must actually do something."""
    base, sites, q0 = _two_well_system(gamma_phi=True)
    before = np.asarray(base.read(q0).state.q_star)
    d_before = np.linalg.norm(before[0, :2] - sites[0])

    base.trash_route(np.concatenate([sites[0], [0.3 * (-1)]]),
                     radius=0.35, strength=0.45)
    assert base.trash.k == 1
    after = np.asarray(base.read(q0).state.q_star)
    d_after = np.linalg.norm(after[0, :2] - sites[0])

    assert not np.array_equal(before, after)
    # the read of the holed well is measurably displaced from its site
    assert d_after > d_before + 1e-4, (d_before, d_after)
    # ...and the holes are on the byte ledger
    assert base.trash_bytes() == 1 * (base.store.dim + 2) * 4
    assert base.n_bytes() > base.store.n_bytes()


def test_gamma_phi_hole_far_from_every_well_leaves_reads_bit_identical():
    """K2 designed negative (b): the field must NOT leak.

    The **compact** gate is exactly zero beyond `r_k`, so a hole outside every
    read's trajectory changes nothing bitwise. A sigmoid gate would leak a tail
    everywhere and this assertion is what would catch it.
    """
    sysm, sites, q0 = _two_well_system(gamma_phi=True)
    before = np.asarray(sysm.read(q0).state.q_star)
    far = np.zeros((sysm.store.dim,))
    far[0] = 25.0                        # far outside the ball and every basin
    sysm.trash_route(far, radius=0.2, strength=0.45)
    after = np.asarray(sysm.read(q0).state.q_star)
    assert np.array_equal(before, after)


def test_trash_route_refuses_when_the_flag_is_off():
    """A verb that silently builds the field would un-ship the OFF guarantee."""
    off, sites, _ = _two_well_system(gamma_phi=False)
    with pytest.raises(RuntimeError, match="gamma_phi"):
        off.trash_route(sites[0])


# --------------------------------------------------------------------------
# the rig end to end, on synthetic labelled data (no MNIST download in tests)
# --------------------------------------------------------------------------
def _toy_data(n_per_class=30, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, dim)) * 3.0
    X = np.concatenate([centers[c] + rng.normal(size=(n_per_class, dim)) * 0.35
                        for c in range(10)]).astype(np.float32)
    y = np.concatenate([np.full(n_per_class, c) for c in range(10)])
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])


def _toy_cfg():
    from chlu.config import get_default_config
    from chlu.experiments.exp_well_lifecycle import apply_quick

    cfg = get_default_config()
    apply_quick(cfg)
    w = cfg.experiment_well_lifecycle
    w.addr_dim = 3
    w.capacity = 4
    w.well_budget = 2
    w.n_offer_per_task = 3
    w.write_steps = 15
    w.read_steps = 40
    w.address_steps = 30
    w.read_batch = 4
    w.capture_dirs = 4
    w.capture_bisect_steps = 3
    w.loo_repeats = 2
    cl = cfg.experiment_cl_entry
    cl.n_tasks = 2
    cl.n_train_per_task = 12
    cl.n_test_per_task = 8
    cl.n_fit_region = 100
    cl.n_fit_pool = 50
    return cfg


def test_census_cell_runs_end_to_end_and_reports_both_curves():
    from chlu.experiments.exp_well_lifecycle import run_census_cell

    cfg = _toy_cfg()
    out = run_census_cell(cfg, seed=0, data=_toy_data(), verbose=False)
    cen = out["census"]
    assert cen["n_live"] >= 1
    assert out["stream"]["n_admitted"] >= 1
    assert 0.0 <= cen["P"] <= 1.0 and 0.0 <= cen["M"] <= 1.0
    # every depth curve exists RAW and NETTED (B1), per well and in aggregate
    assert "depth_raw_median" in cen and "depth_netted_median" in cen
    for wl in cen["wells"]:
        assert "depth_raw" in wl and "depth_netted" in wl and "decay_factor" in wl
    # theta_att is measured on this rig, and both populations are reported
    assert "theta_att" in cen["theta_att_block"]
    assert "population_eroded_not_attractor" in cen
    assert "population_live_attractor_never_read" in cen
    # the launder and the byte ledger ride along, gamma_phi holes included (at 0)
    assert out["bytes"]["knn_launder_bytes"] >= 0
    assert out["bytes"]["gamma_phi_hole_bytes"] == 0
    # the reading is labelled non-promotable, with its reason
    assert out["flags"]["promotable"] is False and out["flags"]["why_not_promotable"]
    # the usage telemetry is item-id keyed and the LOO leg carries its ICC
    assert out["usage"]["key"] == "item_id" and out["usage"]["proxy"] == "read_hits"
    assert "icc_1_1" in out["loo"] or out["loo"]["status"] == "NOT RUN"
    # ⭐⭐ C2W8 close-out: every hardened block rides on the cell
    assert cen["G_DRIFT_two_sided"]["label"] == "MECHANICS"          # item (i)
    assert cen["P_comparability"]["rule"]                            # item (vi.2)
    assert out["geometry"]["d_safe_population"] == "store"           # item (v)
    assert np.isfinite(out["geometry"]["median_nn_store_population"])
    assert out["atom_width_selection"]["selection_source"]           # item (vi.5)
    g = out["g_addr"]
    if isinstance(g.get("A1"), dict):
        assert "margin_in_se_vs_threshold" in g["A1"]                # item (ii)
        assert g["A3"]["in_pass_condition"] is False                 # §A33.1
        assert g["gaddr_spacing_population"] == "codebook"           # item (vi.6)
        # BOTH ratios on every cell, so no cross-arm comparison is made blind
        assert "cue_sigma_over_codebook_spacing" in g
        assert "cue_sigma_over_sizing_spacing" in g
        assert "LAUNCH-SIDE" in g["telemetry_launch_side"]["label"]   # item (iv)
        assert "SETTLE-SIDE" in g["telemetry_settle_side"]["label"]


def test_phi_address_is_idempotent_on_store_space_points():
    """The system re-embeds its own sites; phi must pass those through."""
    from chlu.experiments.exp_well_lifecycle import PhiAddress

    class _Phi:
        def __call__(self, X):
            return np.asarray(X, dtype=np.float32)[:, :3] * 2.0

    emb = PhiAddress(_Phi(), dim=4, addr_dim=3, scale=0.5)
    store_pt = np.arange(4, dtype=np.float32)[None, :]
    assert np.allclose(np.asarray(emb(store_pt)), store_pt)
    img = np.ones((2, 24), dtype=np.float32)
    out = np.asarray(emb(img))
    assert out.shape == (2, 4)
    assert np.allclose(out[:, 3], 0.0)          # payload channels are launched at 0
    assert np.allclose(out[:, :3], 1.0)         # phi * scale


def test_label_to_payload_separates_classes_above_the_merge_threshold():
    from chlu.core.clu_system import CluSystemConfig
    from chlu.experiments.exp_well_lifecycle import label_to_payload

    tol = CluSystemConfig().payload_tol
    gap = abs(label_to_payload(1, 9.0) - label_to_payload(0, 9.0))
    assert gap > tol                                    # different class => NOT mergeable
    assert label_to_payload(3, 9.0) == label_to_payload(3, 9.0)  # same class => distance 0
    assert abs(label_to_payload(9, 9.0)) <= 0.5         # bounded, for the reach certificate


def test_census_survives_x64_the_ordering_hazard_that_bit_this_file():
    """§7.23 regression, **function-scoped** (module-scoped x64 is the hazard).

    Under `jax_enable_x64` a float64 JAX array converts zero-copy, so
    `np.asarray(...)` returns a READ-ONLY view and any in-place write raises.
    `flatten_unused_groups` did exactly that: green alone, red in the full suite
    the moment an x64-enabling module ran first. This test pins the x64 path so
    the next such write is caught in one file rather than in a 55-minute suite.
    """
    from jax import config as jax_config

    was = bool(jax_config.read("jax_enable_x64"))
    jax_config.update("jax_enable_x64", True)
    try:
        sysm = _tiny_system(capacity=4, seed=3)
        plant_item(sysm, 0, np.array([0.5, 0.0]), payload=0.0, depth=0.7, width=0.25)
        flatten_unused_groups(sysm)          # the exact line that failed
        out = census(sysm, UsageTelemetry(), well_budget=1, n_admitted=1,
                     measure_capture=False)
        assert out["n_live"] == 1
        assert np.isfinite(out["depth_raw_median"])
        # ...and the same for the trash region: a float64 field would promote p
        # and break the scan carry, so the field is pinned to float32.
        holed = _tiny_system(capacity=4, seed=3, gamma_phi=True)
        plant_item(holed, 0, np.array([0.5, 0.0]), payload=0.0, depth=0.7, width=0.25)
        holed.trash_route(np.array([5.0, 0.0, 0.0]))
        q0 = np.zeros((1, holed.store.dim), dtype=np.float32)
        q0[:, :2] = np.array([0.52, 0.0])
        assert np.all(np.isfinite(np.asarray(holed.read(q0).state.q_star)))
    finally:
        jax_config.update("jax_enable_x64", was)


def test_mergeable_pairs_use_certificate_radius_and_payload_tol():
    sysm = _tiny_system(capacity=4, d_safe=0.001)
    plant_item(sysm, 0, np.array([0.0, 0.0]), payload=0.0, depth=0.6, width=0.25)
    plant_item(sysm, 1, np.array([0.01, 0.0]), payload=0.0, depth=0.6, width=0.25)
    flatten_unused_groups(sysm)
    states, _ = well_states(sysm, UsageTelemetry(), measure_capture=False)
    pairs, meta = mergeable_pairs(sysm, states)
    assert len(pairs) == 1
    # payload distance beyond the read tolerance disqualifies the same geometry
    pairs2, _ = mergeable_pairs(sysm, states, payload_thresh=-1.0)
    assert pairs2 == []
    # so does a certificate radius below the separation
    pairs3, _ = mergeable_pairs(sysm, states, r_cert=1e-6)
    assert pairs3 == []
    assert meta["r_cert"] > 0.0


# ==========================================================================
# ⭐⭐ C2W8 CLOSE-OUT — the gate-hardening repairs (charter §A32.3 / §A33.1)
#
# Two of these are the task's MANDATORY DESIGNED NEGATIVES:
#   (i)  a planted near-zero-drift, table-like store must FAIL the drift leg;
#   (iv) a store mutated so reads land differently must leave the LAUNCH-side
#        statistic unchanged and move the SETTLE-side one.
# A leg that cannot fail on the degenerate configuration is not a repair.
# ==========================================================================
from chlu.core.well_lifecycle import (  # noqa: E402
    GDRIFT_FLOOR_FRAC_SPACING,
    drift_leg,
)


def _table_like_system(n=4, r=0.9, depth=4.0, width=0.12, capacity=8, **over):
    """A store whose atoms sit EXACTLY at their own sites.

    The relaxed site is then the recorded site to numerical tolerance, i.e.
    ``site_drift -> 0``: the settled point is a deterministic function of the
    stored key. **That is D2a — table-expressible** (§A29.6), the configuration
    intervention §8.2 prohibits, and the pass-2 one-sided rule scored it
    *perfectly*.

    ⚠ The wells are planted **deep and narrow relative to the confinement bowl**
    (depth 4.0, width 0.12, radius 0.9). A shallow planted well still relaxes a
    measurable distance toward the origin, and that residual is the *census
    relaxation's own* floor, not the store's. Measured on this rig:
    ``median site_drift / codebook_spacing = 0.0042``, 2.4x below the registered
    0.01 floor. (Measured for reference on the real rig: the banked pass-3 arm-A
    seed-0 cell sits at **0.0071**, i.e. the floor fires on real cells too — it
    is not a bound only a toy can reach.)
    """
    sysm = _tiny_system(capacity=capacity, d_safe=0.001, **over)
    sites = _ring(n, r=r)
    for i in range(n):
        plant_item(sysm, i, sites[i], payload=0.1 * (i - n / 2.0),
                   depth=float(depth), width=float(width), jitter=0.0)
    flatten_unused_groups(sysm)
    return sysm


# --------------------------------------------------------------------------
# ⛔⛔ DESIGNED NEGATIVE — item (i): drift -> 0 must FAIL
# --------------------------------------------------------------------------
def test_designed_negative_table_like_store_fails_the_two_sided_drift_leg():
    sysm = _table_like_system()
    cen = census(sysm, UsageTelemetry(), measure_capture=False)
    g = cen["G_DRIFT_two_sided"]
    # the store IS the degenerate one: the settle collapses onto the stored key
    assert g["ratio"] < GDRIFT_FLOOR_FRAC_SPACING, g
    # ⛔ the repaired leg FAILS it, and names WHY (D2a, not "cannot address")
    assert g["pass"] is False, g
    assert g["fails_low_D2a_table_expressible"] is True, g
    assert g["fails_high_cannot_address"] is False, g
    # ⭐ and the pass-2 one-sided rule scored this exact store PERFECTLY —
    #    which is the defect, stated as an assertion
    assert g["one_sided_pass2_pass"] is True, g
    assert g["label"] == "MECHANICS"


def test_the_two_sided_drift_leg_still_fails_on_the_high_side():
    """The ceiling is unchanged: drift beyond the key spacing still fails."""
    g = drift_leg([0.9, 1.1, 1.0], codebook_spacing=0.5)
    assert g["fails_high_cannot_address"] is True and g["pass"] is False
    assert g["one_sided_pass2_pass"] is False
    # ... and a healthy store in between PASSES (a leg that cannot pass is as
    # vacuous as one that cannot fail)
    ok = drift_leg([0.10, 0.12, 0.11], codebook_spacing=0.5)
    assert ok["pass"] is True and ok["ratio"] == pytest.approx(0.22, abs=1e-9)


def test_the_drift_floor_is_a_fraction_of_a_MEASURED_spacing_not_a_constant():
    """⛔ 'derive it from a measured quantity, never a bare constant'."""
    a = drift_leg([0.004], codebook_spacing=1.0)
    b = drift_leg([0.004 * 10.0], codebook_spacing=10.0)
    assert a["floor"] == pytest.approx(GDRIFT_FLOOR_FRAC_SPACING)
    assert b["floor"] == pytest.approx(10.0 * GDRIFT_FLOOR_FRAC_SPACING)
    # the same DIMENSIONLESS store gets the same verdict at any scale
    assert a["ratio"] == pytest.approx(b["ratio"])
    assert a["pass"] is b["pass"] is False
    assert a["fails_low_D2a_table_expressible"] is True


# --------------------------------------------------------------------------
# ⛔⛔ DESIGNED NEGATIVE — item (iv): launch-side vs settle-side coverage
# --------------------------------------------------------------------------
def test_designed_negative_launch_coverage_is_store_invariant_settle_is_not():
    """⭐⭐ **The single test that proves the two are different quantities.**

    ``covered`` is computed on ``q0`` against the codebook, so the same φ and the
    same admitted codebook give the same number **whatever the store does** —
    which is why "58 / 62 / 62 unassigned, digit-identical" looked decisive and
    was vacuous (§A31.1). Here the store is mutated so the reads land somewhere
    else entirely, and the launch-side statistic does not move by one bit.
    """
    import equinox as eqx
    import jax.numpy as jnp

    # ⚠ the settle must be given time to actually move: at the tiny default step
    # budget the flattened landscape has not yet pulled the read off its launch
    # point, and the test would pass for the wrong reason.
    sysm = _table_like_system(n=4, capacity=4, read_steps=800, address_steps=400)
    q0 = np.zeros((8, sysm.store.dim), dtype=np.float32)
    q0[:, :2] = _ring(4, r=0.9).repeat(2, axis=0) + 0.02
    before = sysm.read(q0).diagnostics
    # --- mutate the STORE so reads land differently (codebook untouched) ---
    V = eqx.tree_at(lambda t: t.learned.amp, sysm.store.V,
                    jnp.zeros_like(sysm.store.V.learned.amp))
    sysm.store = eqx.tree_at(lambda s: s.V, sysm.store, V)
    after = sysm.read(q0).diagnostics

    launch0 = np.asarray(before["launch_covered"], dtype=bool)
    launch1 = np.asarray(after["launch_covered"], dtype=bool)
    settle0 = np.asarray(before["settle_covered"], dtype=bool)
    settle1 = np.asarray(after["settle_covered"], dtype=bool)
    # ⛔ the launch-side statistic is BIT-IDENTICAL: it never saw the store
    assert np.array_equal(launch0, launch1), (launch0, launch1)
    # ⭐ the settle-side statistic MOVED: it is a property of the store
    assert not np.array_equal(settle0, settle1), (settle0, settle1)
    assert int(settle1.sum()) < int(settle0.sum())
    # and `covered` is the launch-side alias monitor #settle_argmin still needs
    assert np.array_equal(np.asarray(before["covered"], dtype=bool), launch0)


def test_never_read_telemetry_is_gated_on_the_settle_side():
    """``n_never_read`` inherited the launch-point defect; it no longer does."""
    from chlu.experiments.usage_telemetry import attach_reads

    class _Res:
        diagnostics = {"assign_settle": np.array([0, 0, 1]),
                       "launch_covered": np.array([True, True, True]),
                       "covered": np.array([True, True, True]),
                       "settle_covered": np.array([True, False, False])}

    sysm = _tiny_system(capacity=2, d_safe=0.001)
    plant_item(sysm, 0, np.array([0.5, 0.0]), payload=0.0, depth=0.6, width=0.25)
    plant_item(sysm, 1, np.array([-0.5, 0.0]), payload=0.1, depth=0.6, width=0.25)
    tel = UsageTelemetry()
    tel.note_admitted(0, 0)
    tel.note_admitted(1, 0)
    n = attach_reads(sysm, tel, _Res(), 1)
    assert n == 1                       # only the settle-covered read is credited
    assert tel.hits(0) == 1 and tel.hits(1) == 0
    assert tel.n_unassigned == 2        # the two settle-uncovered reads
    s = tel.summary(live_ids=[0, 1])
    assert s["frac_never_read"] == pytest.approx(0.5)
    assert "SETTLE" in s["coverage_side"] and "RETIRED" in s["caption"]


# --------------------------------------------------------------------------
# item (vi.2) — `P` is never emitted without `n_non_capturing`
# --------------------------------------------------------------------------
def test_P_is_never_emitted_without_the_theta_att_degeneracy_qualifier():
    sysm = _table_like_system(capacity=4)
    cen = census(sysm, UsageTelemetry(), measure_capture=False)
    assert "P" in cen and "P_comparability" in cen
    pc = cen["P_comparability"]
    assert set(("n_non_capturing", "theta_att_degenerate",
                "P_comparable_across_arms")) <= set(pc)
    # measure_capture=False => nothing was measured non-capturing => degenerate
    assert pc["n_non_capturing"] == 0
    assert pc["theta_att_degenerate"] is True
    assert pc["P_comparable_across_arms"] is False
    assert "not comparable across arms" in pc["rule"].lower()


# --------------------------------------------------------------------------
# item (v) — d_safe is sized on the STORE population, not the sizing set
# --------------------------------------------------------------------------
def test_d_safe_population_spacing_is_larger_than_the_sizing_set_spacing():
    """⚠ NN spacing grows as the population shrinks: a ~200-key sizing set
    under-states a 16-item store's spacing, which is why monitor #3's 0.000
    refusal rate was **arithmetic, not a finding** (§A31.2)."""
    from chlu.core.soft_certificate import population_median_nn

    rng = np.random.default_rng(0)
    keys = rng.normal(size=(200, 8))
    out = population_median_nn(keys, 16, n_draws=32, seed=0)
    assert out["applicable"] is True
    assert out["median_nn_population"] > out["median_nn_sizing"]
    assert out["ratio_population_over_sizing"] > 1.0
    assert out["n_population"] == 16 and out["n_sizing_keys"] == 200
    # a sizing set no larger than the population: the two coincide, declared
    same = population_median_nn(keys[:8], 16, n_draws=4, seed=0)
    assert same["ratio_population_over_sizing"] == 1.0
    assert "declared" in same["note"]


# --------------------------------------------------------------------------
# item (vi.5) — the census REFUSES to run at a width nobody selected
# --------------------------------------------------------------------------
def test_census_refuses_to_run_at_an_unselected_atom_width():
    """⛔ Designed negative: arm A's banked runs used
    ``atom_width_frac_spacing = 1.5`` while the shipped default is 0.5. A census
    that silently runs at a width nobody selected produces numbers nobody can
    attribute."""
    import dataclasses

    from chlu.experiments import exp_well_lifecycle as ewl

    cfg = _toy_cfg()
    original = ewl.store_config
    try:
        # a width that is neither the census default nor any declared fraction
        ewl.store_config = lambda c, s, d, overrides=None: dataclasses.replace(
            original(c, s, d, overrides=overrides), atom_width=0.777 * float(d))
        with pytest.raises(ewl.UnselectedAtomWidth) as e:
            ewl.run_census_cell(cfg, seed=0, data=_toy_data(), verbose=False)
        assert "REFUSES TO RUN" in str(e.value)
        assert "atom_width_frac_spacing" in str(e.value)
        # ... and DECLARING it lets the identical run through
        cfg2 = _toy_cfg()
        w = cfg2.experiment_well_lifecycle
        w.atom_width_selection = 0.777 * float(w.d_safe_frac)
        out = ewl.run_census_cell(cfg2, seed=0, data=_toy_data(), verbose=False)
        blk = out["atom_width_selection"]
        assert blk["selection_source"] == (
            "experiment_well_lifecycle.atom_width_selection")
        assert blk["atom_width_frac_spacing_effective"] == pytest.approx(
            w.atom_width_selection, rel=1e-6)
    finally:
        ewl.store_config = original


def test_the_width_guard_can_be_switched_off_but_is_on_by_default():
    from chlu.config import get_default_config

    w = get_default_config().experiment_well_lifecycle
    assert w.refuse_unselected_atom_width is True
    assert w.atom_width_selection is None       # resolved from the arm configs
    assert w.d_safe_population == "store"
    assert w.gaddr_spacing_population == "codebook"
    assert w.gdrift_floor_frac_spacing == GDRIFT_FLOOR_FRAC_SPACING
