"""Tests for the C2W1 memory gym (Track 1, the dividend as the sole KPI).

The properties asserted here are the ones that decide whether a gym number means
anything at all:

* the **byte ledger** agrees with its closed form, and the closed form's *floor*
  is architectural — one atom group per item forces ``ratio >= 2.2x`` at
  ``d=4, m=1``, so **matched bytes is unreachable by construction**, not merely
  unachieved (this is the task's hard problem, pinned as a test);
* the ``aggregate`` family's construction guarantee holds — **no target is within
  ``payload_tol`` of any stored payload**, so the arg-min launder cannot be
  accidentally right (without this the family is metric-native in disguise);
* the ``recency`` family's label really is *insertion order*, and the **+0 B
  order-aware substitute** recovers it — the pre-registered laundering;
* every launder is scored by **exactly the same scorer** as the CLU, and
  ``same_keys_null`` provably **coincides** with the settle-deleted launder for an
  index-valued question (a documented degeneracy that must not be silently lost);
* one cell runs end to end and emits all three harness-native controls, the byte
  ledger and the monitor readings — a cell that cannot produce all three does not
  report.
"""

import jax
import numpy as np
import pytest

from chlu.core.clu_system import CluSystemConfig, build_system
from chlu.eval.dividend import (
    byte_account,
    echo_launder,
    fit_shared_metric,
    knn_mean_launder,
    order_aware_launder,
    settle_deleted_launder,
    shared_metric_launder,
)
from chlu.experiments.exp_memory_gym import aggregate, run_cell
from chlu.experiments.memory_gym import (
    FAMILIES,
    PRIMARY_METRIC,
    GymConfig,
    byte_ratio_law,
    gym_config,
    make_gym_stream,
    queries_aggregate,
    queries_recency,
    readout_occupancy,
    score_coord,
    score_index,
    score_value,
)


@pytest.fixture(autouse=True)
def float32_dynamics():
    """Pin float32 for the whole module, restoring the global flag after.

    ⚠ Repo-wide test-isolation hazard (handover §7.2): several test modules enable
    ``jax_enable_x64`` at MODULE import, so x64 is globally ON in a full-suite run
    even though it is off when this file runs alone. The gym stores and reads in
    float32 by construction. Same fixture as ``test_clu_system``.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", was)


def _tiny(family="aggregate", **kw):
    over = dict(
        n_offer=4, capacity=3, budget=3, reference_capacity=3,
        n_query_per_item=2, n_query_per_pair=3, consolidate_every=2,
        min_consolidations=4, n_manifold_launch=4,
        clu_overrides=dict(addr_dim=3, write_steps=20, address_steps=60,
                           read_steps=80, atoms_per_item=8),
    )
    over.update(kw)
    return gym_config(family, "base", seed=0, **over)


# -- configuration ----------------------------------------------------------
def test_families_and_arms_are_declared():
    assert FAMILIES == ("overload", "aggregate", "recency", "manifold")
    assert set(PRIMARY_METRIC) == set(FAMILIES)
    with pytest.raises(ValueError):
        gym_config("not_a_family")
    with pytest.raises(ValueError):
        gym_config("overload", "not_an_arm")


def test_clu_overrides_merge_rather_than_replace():
    """An arm may change one store knob without restating the family's."""
    base = gym_config("aggregate", "base")
    tight = gym_config("aggregate", "tight")
    assert base.clu_overrides["stage_admission"] is True
    # the arm adds ball_radius/d_safe_override and keeps the family's admission
    assert tight.clu_overrides["stage_admission"] is True
    assert tight.clu_overrides["ball_radius"] == 0.45
    assert tight.clu_overrides["d_safe_override"] == 0.32


def test_from_mapping_ignores_unknown_keys():
    g = GymConfig.from_mapping({"n_offer": 11, "not_a_field": 3})
    assert g.n_offer == 11


def test_atom_budget_is_pinned_to_the_reference_capacity():
    """⭐ The overload is in atoms-per-item, not in slots."""
    g1 = gym_config("overload", "base", n_offer=6, capacity=6, reference_capacity=6)
    g3 = gym_config("overload", "base", n_offer=18, capacity=18, reference_capacity=6)
    assert g1.overload_factor == 1.0
    assert g3.overload_factor == 3.0
    # the same total atom budget, three times the items
    assert g1.build_clu().n_atoms == pytest.approx(g3.build_clu().n_atoms, rel=0.1)


# -- the byte ledger (the hard problem, pinned) -----------------------------
@pytest.mark.parametrize("n_spectator", [0, 1])
def test_byte_ratio_law_matches_the_measured_ledger(n_spectator):
    """⛔ **C2W4 regression (theorist C2).** This test used to pass
    ``n_spectator = 0`` literally, and the end-to-end test was parametrised over
    ``aggregate``/``recency`` only, so **no test exercised a spectator dim** —
    which is exactly the geometry in which the shipped closed form was wrong for
    three waves. The spectator case is now parametrised in.
    """
    cfg = CluSystemConfig(addr_dim=4, payload_dim=1, capacity=4, atoms_per_item=8,
                          min_atoms=32, min_atoms_base=32, min_atoms_c=1.0,
                          n_spectator=n_spectator)
    sys_ = build_system(cfg, loud=False)
    keys = np.zeros((4, 4))
    pays = np.zeros((4, 1))
    ba = byte_account(sys_, keys, pays)
    n_atoms = int(sys_.store.V.learned.centers.shape[0])
    predicted = byte_ratio_law(n_atoms / 4, 4, 1, n_spectator)
    assert ba.ratio == pytest.approx(predicted, rel=1e-9)
    assert not ba.matched()
    # ⭐ and structurally, as INTEGERS -- the law is an accounting identity over
    # the store's parameter leaves, not a float coincidence of the sizing
    # convention (`full = 4[N_at(D+2) + K d]`, `launder = 4 K (d+m)`).
    d, m, k = 4, 1, 4
    store_dim = d + m + n_spectator
    assert ba.full_bytes == 4 * (n_atoms * (store_dim + 2) + k * d)
    assert ba.launder_bytes == 4 * k * (d + m)


def test_matched_bytes_is_unreachable_by_construction():
    """⛔ One atom group per item forces ``atoms_per_item >= 1``, hence a hard
    floor on the byte ratio. This is why no gym cell may be quoted as a dividend.
    """
    floor = byte_ratio_law(1.0, 4, 1, 0)
    assert floor == pytest.approx(1.4 + 0.8)
    assert floor > 1.05  # i.e. `ByteAccount.matched(tol=0.05)` can never be True
    # the ratio is independent of K and monotone in the atom budget
    assert byte_ratio_law(8, 4, 1, 0) == pytest.approx(1.4 * 8 + 0.8)
    assert byte_ratio_law(32, 4, 1, 0) > byte_ratio_law(8, 4, 1, 0)


def test_byte_ratio_law_is_correct_on_a_spectator_dim():
    """⛔ **The C2W4 erratum, pinned as a test.** ``byte_ratio_law`` used to
    divide by the *store* dim ``D`` where the launder row is ``(d + m)`` floats;
    the two coincide iff ``n_spectator == 0``, so the defect was invisible to
    every test we had. The corrected law is ``[A(D+2) + d] / (d + m)``.

    Numbers pinned here are the C2W1 ``manifold`` cells (measured **52.00x**,
    published **43.33x**) and the corrected floor (**2.40x**, printed as
    **2.00x**). Both corrections are in the **conservative** direction.
    """
    d, m, nsp = 4, 1, 1
    store_dim = d + m + nsp  # D = 6
    # the four C2W1 `manifold` cells: A = 32 atoms per live item
    assert byte_ratio_law(32, d, m, nsp) == pytest.approx(52.0, rel=0, abs=1e-12)
    assert byte_ratio_law(32, d, m, nsp) == pytest.approx(
        (32 * (store_dim + 2) + d) / (d + m))
    # ...against the pre-erratum value, which is 8.6667 LOWER
    assert 32 * (store_dim + 2.0) / store_dim + d / store_dim == pytest.approx(
        43.3333333, abs=1e-6)
    # ⭐ the floor RISES with a spectator dim: 2.20x -> 2.40x
    assert byte_ratio_law(1.0, d, m, nsp) == pytest.approx(2.4, rel=0, abs=1e-12)
    assert byte_ratio_law(1.0, d, m, 0) == pytest.approx(2.2, rel=0, abs=1e-12)
    # ⭐ bit-identity gate: at n_spectator = 0 the corrected law is the SAME
    # float expression as the pre-erratum one, so the 24 unaffected C2W1 cells
    # re-score bitwise unchanged.
    for A in (3.0, 8.0, 32.0, 341.0, 198 / 17, 192 / 5):
        pre = A * (d + m + 2.0) / (d + m) + d / (d + m)
        assert byte_ratio_law(A, d, m, 0).hex() == pre.hex()


# -- the streams ------------------------------------------------------------
def test_stream_carries_deletion_revisit_and_enough_consolidations():
    g = _tiny("aggregate")
    st = make_gym_stream(g, g.build_clu())
    assert len(st.offered) == g.n_offer
    deletes = [r for r in st.items if r.get("delete")]
    assert len(deletes) == 1
    # the deletion must name an item that was actually offered (a delete of an
    # absent id is a silent no-op that empties the stage of its demand)
    assert deletes[0]["item_id"] in {o["item_id"] for o in st.offered}
    assert any(r.get("item_id", 0) >= 2000 for r in st.items)  # the revisit
    assert len(st.chunks) >= g.min_consolidations  # monitor #6 needs >= 4 windows
    assert list(st.order) == sorted(st.order)


# -- the family constructions ----------------------------------------------
def test_aggregate_targets_are_never_a_stored_payload():
    """The construction guarantee that keeps the family non-metric-native."""
    g = _tiny("aggregate", n_query_per_pair=12)
    ccfg = g.build_clu()
    rng = np.random.default_rng(0)
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pays = np.asarray([[-1.0], [0.0], [1.0]])
    qs = queries_aggregate(g, ccfg, centers, pays, rng)
    assert len(qs) > 0
    d = np.min(np.abs(qs.target[:, None, :] - pays[None, :, :]), axis=(1, 2))
    assert np.all(d >= ccfg.payload_tol)


def test_recency_label_is_insertion_order_and_pairs_differ_in_age():
    g = _tiny("recency", n_query_per_pair=4)
    ccfg = g.build_clu()
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    born = np.asarray([0.0, 1.0, 2.0])
    qs = queries_recency(g, ccfg, centers, born, np.random.default_rng(0))
    pairs = qs.meta["pairs"]
    assert np.all(born[pairs[:, 0]] != born[pairs[:, 1]])
    expected = np.where(born[pairs[:, 0]] > born[pairs[:, 1]],
                        pairs[:, 0], pairs[:, 1])
    assert np.array_equal(qs.label, expected)


# -- the gym-side launders --------------------------------------------------
def test_knn_mean_launder_uniform_and_idw():
    keys = np.asarray([[0.0], [1.0]])
    pays = np.asarray([[0.0], [1.0]])
    q = np.asarray([[0.25]])
    assert knn_mean_launder(keys, pays, q, k=2)[0, 0] == pytest.approx(0.5)
    # IDW recovers the MIXING FRACTION, which is why it is the honest ceiling
    idw = knn_mean_launder(keys, pays, q, k=2, weighting="inverse_distance")[0, 0]
    assert idw == pytest.approx(0.25, abs=1e-6)
    with pytest.raises(ValueError):
        knn_mean_launder(keys, pays, q, weighting="nope")


def test_order_aware_launder_costs_zero_bytes_and_answers_recency():
    """⛔ The pre-registered laundering: a table's ROW ORDER is free information."""
    keys = np.asarray([[0.0], [1.0], [5.0]])
    order = np.asarray([2, 0, 1])  # key 0 is the most recent
    q = np.asarray([[0.5], [0.45]])
    got = order_aware_launder(keys, order, q, k=2)
    assert list(got) == [0, 0]
    oldest = order_aware_launder(keys, order, q, k=2, newest=False)
    assert list(oldest) == [1, 1]


def test_echo_launder_is_a_copy_not_a_view():
    x = np.asarray([[1.0], [2.0]])
    y = echo_launder(x)
    y[0, 0] = 99.0
    assert x[0, 0] == 1.0


def test_fit_shared_metric_is_det1_and_beats_argmin_on_anisotropic_noise():
    """Doctrine I-12's launder, exercised: under an ANISOTROPIC query law the
    fitted shared metric is a genuinely stronger control than plain arg-min."""
    # ⚠ the keys must differ in BOTH coordinates: with an axis-aligned key grid and
    # axis-aligned noise, Euclidean and Mahalanobis arg-min make identical
    # decisions (the common-axis term cancels), so such a case cannot show the
    # difference at all — a trap this test hit on its first spelling.
    rng = np.random.default_rng(0)
    keys = np.asarray([[0.0, 0.0], [1.0, 0.3]])
    pays = np.asarray([[0.0], [1.0]])
    lab = rng.integers(0, 2, size=600)
    noise = rng.normal(size=(600, 2)) * np.asarray([0.7, 0.05])
    q = keys[lab] + noise
    M = fit_shared_metric(keys, q, lab)
    assert np.linalg.det(M) == pytest.approx(1.0, rel=1e-6)
    plain = settle_deleted_launder(keys, pays, q)
    shared = shared_metric_launder(keys, pays, q, M)
    acc_plain = float(np.mean(plain[:, 0] == pays[lab][:, 0]))
    acc_shared = float(np.mean(shared[:, 0] == pays[lab][:, 0]))
    assert acc_shared > acc_plain
    # and under an ISOTROPIC law it must TIE (M ~ I): the honest caveat, asserted
    q_iso = keys[lab] + rng.normal(size=(600, 2)) * 0.2
    M_iso = fit_shared_metric(keys, q_iso, lab)
    same = settle_deleted_launder(keys, pays, q_iso)
    shared_iso = shared_metric_launder(keys, pays, q_iso, M_iso)
    assert float(np.mean(same == shared_iso)) > 0.97


# -- the read-outs and scorers ---------------------------------------------
class _FakeRead:
    def __init__(self, traj, phase):
        self.traj = traj
        self.phase = phase


def test_occupancy_readout_is_a_distribution_and_peaks_at_the_visited_well():
    centers = np.asarray([[0.0, 0.0], [3.0, 0.0]])
    traj = np.zeros((2, 5, 4), dtype=float)
    traj[0, :, :2] = np.asarray([0.0, 0.0])   # sits on well 0
    traj[1, :, :2] = np.asarray([3.0, 0.0])   # sits on well 1
    res = _FakeRead(traj, np.full((5,), 2))
    occ = readout_occupancy(res, centers, radius=0.5)
    assert occ.shape == (2, 2)
    assert np.allclose(occ.sum(axis=1), 1.0)
    assert list(np.argmax(occ, axis=1)) == [0, 1]


def test_scorers_are_sane():
    class _QS:
        kind = "value"
        target = np.asarray([[0.0], [1.0]])
        label = np.asarray([0, 1])
        alphabet = np.asarray([[0.0], [1.0]])
        meta = {}

        def __len__(self):
            return 2

    qs = _QS()
    s = score_value(np.asarray([[0.05], [0.95]]), qs)
    assert s["decode"] == 1.0
    assert s["mae"] == pytest.approx(0.05)
    assert s["neg_mae"] == pytest.approx(-0.05)

    class _QI(_QS):
        kind = "index"
        meta = {"pairs": np.zeros((2, 2), dtype=int)}

    si = score_index(np.asarray([0, 0]), _QI())
    assert si["acc"] == 0.5 and si["chance"] == 0.5

    class _QC(_QS):
        kind = "coord"
        target = np.asarray([[-1.0], [1.0]])

    assert score_coord(np.asarray([[-1.0], [1.0]]), _QC())["r2"] == pytest.approx(1.0)
    assert score_coord(np.asarray([[0.0], [0.0]]), _QC())["r2"] == pytest.approx(0.0)


# -- end to end -------------------------------------------------------------
@pytest.mark.parametrize("family", ["aggregate", "recency", "manifold"])
def test_cell_reports_all_three_harness_native_controls_and_a_byte_ledger(family):
    """A cell that cannot produce all three controls does not report.

    ⛔ **``manifold`` added in C2W4 (theorist C2).** It is the only family with
    ``n_spectator = 1``, and its absence here is why the byte law's denominator
    bug survived three waves of green tests.
    """
    rec = run_cell(family, "base", seed=0, quick=True, loud=False,
                   gym_overrides=dict(clu_overrides=dict(addr_dim=3)))
    assert not rec["degenerate"], rec
    div = rec["dividend"]
    assert div["dividend"] == pytest.approx(div["full"] - div["launder"])
    for control in ("same_keys_null", "blank_store", "blank_aux_decode"):
        assert control in div["controls"]
    led = rec["byte_ledger"]
    assert led["ratio"] == pytest.approx(led["closed_form_ratio"], rel=1e-6)
    assert led["matched"] is False  # unreachable by construction
    assert rec["n_consolidations"] >= 4  # monitor #6's applicability condition
    assert rec["monitors"] and "settle_argmin" in {m["name"] for m in rec["monitors"]}
    # both read variants are recorded, and the annealed one costs zero bytes
    assert rec["read_variants"]["annealed"]["extra_bytes"] == 0
    ag = aggregate([rec])
    key = f"{family}/base"
    assert ag[key]["dividend"]["n"] == 1
    assert "SINGLE SEED" in ag[key]["sign"]


def test_same_keys_null_coincides_with_the_launder_for_an_index_question():
    """A documented degeneracy that must not be silently lost: permuting payloads
    cannot change an arg-min over keys, so for the recency family the same-keys
    null carries no information beyond the settle-deleted launder."""
    rec = run_cell("recency", "base", seed=0, quick=True, loud=False,
                   gym_overrides=dict(clu_overrides=dict(addr_dim=3)))
    assert rec["scores"]["same_keys_null"]["acc"] == pytest.approx(
        rec["scores"]["settle_deleted"]["acc"])
