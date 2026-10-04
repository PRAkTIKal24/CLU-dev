"""Tests for ``chlu.experiments.gen_erasure`` -- hub/erasure's machinery.

These cover the parts that decide whether a number means anything and that a later
edit could silently break: the byte arithmetic, the forbidden deletion targets, the
8-byte gate, the request encoding, the size-matched pools, and the churn metric.
⛔ They do not touch the frozen instrument, so they run anywhere and in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.experiments import gen_erasure as E

CARDS = (10, 10, 10, 8, 4, 15)


class _Grid:
    """The two methods the module actually uses off a FactorGrid."""

    names = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
    cardinalities = CARDS
    values = tuple(np.arange(c, dtype=np.float64) / 10.0 for c in CARDS)

    def factor_indices_of(self, rows):
        rows = np.asarray(rows, np.int64)
        out = np.empty((rows.size, len(CARDS)), np.int64)
        rem = rows.copy()
        for k in range(len(CARDS) - 1, -1, -1):
            out[:, k] = rem % CARDS[k]
            rem //= CARDS[k]
        return out


# ---------------------------------------------------------------------------
# the forbidden deletion targets -- refused in CODE, not by convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", ["scale", "orientation"])
def test_a_physical_default_is_refused_as_a_deletion_target(factor):
    """⛔ Target file §5. A convention is what a later hub breaks by accident."""
    with pytest.raises(ValueError, match="PHYSICAL DEFAULT"):
        E.DeletionTarget(factor=factor, factor_index=3, value_index=4, value=1.0,
                         label="x", why="y")


def test_the_two_shipped_targets_are_neither_scale_nor_orientation():
    assert {t.factor for t in E.VALUE_TARGETS} == {"object_hue", "shape"}
    for t in E.VALUE_TARGETS:
        assert t.factor not in E.FORBIDDEN_DELETION_FACTORS


def test_the_conjunction_cell_is_not_the_cylinder():
    """Target file §4: the wave-2 scored cell is entirely shape = cylinder (1.0)."""
    assert E.CONJUNCTION_CELL["factor"] == "shape"
    assert E.CONJUNCTION_CELL["value_index"] != 1          # not the cylinder
    assert E.CONJUNCTION_CELL["value_index"] != 0          # not the cube (our target)
    assert E.CONJUNCTION_CELL["value_index"] != 2          # not the sphere
    assert len(E.CONJUNCTIONS) >= 3
    for c in E.CONJUNCTIONS:
        assert E.F_SCALE not in (c.factor_a, c.factor_b)
        assert E.F_ORIENTATION not in (c.factor_a, c.factor_b)


# ---------------------------------------------------------------------------
# bytes
# ---------------------------------------------------------------------------


def test_clu_is_exactly_the_budget_and_basis_head_is_the_largest_m_that_fits():
    led = E.assert_byte_budget(E.ErasureConfig(), 2048, CARDS)
    assert led["clu"]["deployed_bytes"] == E.BYTE_BUDGET
    assert led["clu"]["exactly_the_budget"] is True
    assert led["basis_head"]["m"] == led["basis_head"]["m_max_at_budget"]
    assert led["basis_head"]["used_bytes"] <= E.BYTE_BUDGET
    # the control is DECLARED at the full budget -- over-stating it is conservative
    assert led["basis_head"]["deployed_bytes"] == E.BYTE_BUDGET


def test_an_under_resourced_like_for_like_is_refused():
    cfg = E.ErasureConfig(basis_m=10)
    with pytest.raises(AssertionError, match="not the largest m"):
        E.assert_byte_budget(cfg, 2048, CARDS)


def test_the_gate_is_eight_bytes():
    bm = E.ValueBitmap(CARDS, np.ones(sum(CARDS), bool))
    assert bm.present.size == 57
    assert bm.n_bytes() == E.GATE_BYTES == 8


# ---------------------------------------------------------------------------
# the request encoding: singles + ALL pairs + bias
# ---------------------------------------------------------------------------


def test_the_head_carries_the_pair_interaction_block():
    names = E.request_feature_names(CARDS)
    n_single = sum(CARDS)
    n_pair = sum(CARDS[i] * CARDS[j]
                 for i in range(len(CARDS)) for j in range(i + 1, len(CARDS)))
    assert len(names) == n_single + n_pair + 1
    assert sum(1 for n in names if n[0] == "pair") == n_pair
    assert names[-1][0] == "bias"


def test_onehot_fires_one_single_per_factor_and_one_per_pair():
    fi = np.array([[1, 2, 3, 4, 0, 5]])
    oh = E._onehot(fi, CARDS)
    assert oh.shape == (1, len(E.request_feature_names(CARDS)))
    n_pairs = len(CARDS) * (len(CARDS) - 1) // 2
    assert oh.sum() == len(CARDS) + n_pairs + 1


def test_an_unrequested_factor_contributes_to_no_single_and_no_pair():
    full = E._onehot(np.array([[1, 2, 3, 4, 0, 5]]), CARDS).sum()
    one_missing = E._onehot(np.array([[1, 2, 3, 4, 0, -1]]), CARDS).sum()
    # drops its own single and the 5 pairs it took part in
    assert one_missing == full - 1 - (len(CARDS) - 1)


# ---------------------------------------------------------------------------
# the 8-byte gate
# ---------------------------------------------------------------------------


def test_clearing_a_bit_refuses_exactly_the_requests_naming_that_value():
    grid = _Grid()
    rows = np.arange(4000)
    bm = E.ValueBitmap.from_pool(grid, rows)
    gated = bm.clear(E.F_OBJECT_HUE, 5)
    V = np.zeros((3, 6), np.int64)
    V[0, E.F_OBJECT_HUE] = 5
    V[1, E.F_OBJECT_HUE] = 4
    V[2, E.F_OBJECT_HUE] = -1
    served = gated.accepts(V)
    assert served[0] == np.False_      # names the cleared value -> refused
    assert served[1] == np.True_       # names a retained value  -> served
    assert served[2] == np.True_       # does not name it at all -> served
    assert gated.n_bytes() == 8


def test_the_gate_forgets_nothing_only_one_bit_moves():
    grid = _Grid()
    rows = np.arange(4000)
    bm = E.ValueBitmap.from_pool(grid, rows)
    gated = bm.clear(E.F_OBJECT_HUE, 5)
    assert int((bm.present != gated.present).sum()) == 1


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------


def test_deleted_and_placebo_pools_are_the_same_size():
    grid = _Grid()
    cfg = E.ErasureConfig(n_store_train=2000)
    train = np.arange(0, 480_000, 7)[:60_000]
    pools = E.build_live_sets(grid, train, E.TARGET_HUE, cfg, 0)
    assert pools["L_deleted"].size == pools["L_placebo"].size
    assert pools["matched_deletion_size"] is True
    assert pools["n_carrying_target"] > 0


def test_the_deleted_pool_really_carries_none_of_the_value():
    grid = _Grid()
    cfg = E.ErasureConfig(n_store_train=2000)
    train = np.arange(0, 480_000, 7)[:60_000]
    pools = E.build_live_sets(grid, train, E.TARGET_HUE, cfg, 0)
    out = E.assert_value_absent(grid, pools["L_deleted"], E.TARGET_HUE)
    assert out["n_carrying"] == 0
    with pytest.raises(AssertionError, match="still carry"):
        E.assert_value_absent(grid, pools["L_full"], E.TARGET_HUE)


def test_the_placebo_for_a_conjunction_is_drawn_from_inside_the_cell():
    grid = _Grid()
    L = np.arange(480_000)[::11][:40_000]
    conj = E.CONJUNCTIONS[0]
    pools = E.conjunction_pools(grid, L, conj, 0)
    assert pools["size_matched"] is True
    fi = grid.factor_indices_of(pools["L_deleted"])
    assert int(conj.mask(fi).sum()) == 0          # the conjunction is gone
    fi_p = grid.factor_indices_of(pools["L_placebo"])
    assert int(conj.mask(fi_p).sum()) > 0         # the placebo left it alone


# ---------------------------------------------------------------------------
# churn
# ---------------------------------------------------------------------------


def test_identical_predictions_give_churn_exactly_zero():
    """⛔ The null of claim 2. If this is not exactly 0, no churn number is readable."""
    P = np.random.default_rng(0).integers(0, 9, (3, 200, 6)).astype(np.int16)
    out = E.churn_between(P, P)
    assert out["judge_0"]["tuple_changed"] == 0.0
    assert out["any_judge_changed"] == 0.0
    assert out["all_judges_changed"] == 0.0


def test_churn_counts_a_request_once_however_many_factors_moved():
    before = np.zeros((1, 4, 6), np.int16)
    after = before.copy()
    after[0, 0, :] = 1          # request 0: all six factors changed
    after[0, 1, 2] = 1          # request 1: one factor changed
    out = E.churn_between(before, after)
    assert out["judge_0"]["tuple_changed"] == pytest.approx(0.5)
    assert out["judge_0"]["mean_factors_changed"] == pytest.approx((6 + 1) / 4)


# ---------------------------------------------------------------------------
# the §4 gate contrast, computed rather than argued
# ---------------------------------------------------------------------------


def test_a_per_factor_bitmap_cannot_express_a_conjunction_deletion():
    grid = _Grid()
    L = np.arange(480_000)[::11][:40_000]
    out = E.bitmap_cannot_express_a_conjunction(grid, L, E.CONJUNCTIONS[0])
    assert out["bitmap_bits_changed_by_the_deletion"] == 0
    assert out["bitmap_can_express_it"] is False
    assert out["joint_table_n_combinations"] == int(np.prod(CARDS))


# ---------------------------------------------------------------------------
# the adversary's controls exist and are declared
# ---------------------------------------------------------------------------


def test_the_move_set_and_its_calibration_are_declared():
    assert set(E.ADVERSARY_MOVES) >= {"M1_direct_request", "M2_weight_read",
                                      "M3_indirect_request", "M4_span_sweep",
                                      "M5_read_reachable"}
    assert set(E.ADVERSARY_CALIBRATION) == {"sensitivity", "specificity",
                                            "verdict_rule"}


def test_claim1_carries_both_a_sensitivity_and_a_specificity_condition():
    ids = [c[0] for c in E.CLAIM1_CONDITIONS]
    assert "clu_full" in ids            # sensitivity ceiling
    assert "random_span" in ids         # specificity control
    assert "clu_shuffled" in ids        # charter §C.2 landscape-removed
    assert "basis_head_gated" in ids    # the like-for-like
    assert "knn_proto_gated" in ids     # the second control


def test_a_random_span_is_matched_to_the_reference_marginals():
    ref = np.random.default_rng(0).normal(3.0, 2.0, (500, 40)).astype(np.float32)
    span = E.random_span(ref, 64, 0)
    assert span.shape == (64, 40)
    assert abs(float(span.mean()) - 3.0) < 0.5
    assert abs(float(span.std()) - 2.0) < 0.5


def test_the_shuffled_span_keeps_every_marginal_and_destroys_the_joint():
    a = np.random.default_rng(1).normal(0, 1, (32, 50)).astype(np.float32)
    b = E.shuffled_span(a, 0)
    assert b.shape == a.shape
    for j in range(a.shape[1]):
        assert np.allclose(np.sort(a[:, j]), np.sort(b[:, j]))
    assert not np.allclose(a, b)


def test_the_landscape_removed_control_can_veto_a_cell_the_random_span_would_pass():
    """Measured on seed 0, target shape=cube: random_span 0.000 but clu_shuffled 0.344
    against a ceiling of 0.385. Gating on the random span alone would have licensed an
    erasure claim on a target where a content-destroyed store does nearly as well."""
    recs = [
        {"target": "cube", "seed": 0, "condition": "clu_full",
         "best_unanimous_hit": 0.385, "best_move": "M4a"},
        {"target": "cube", "seed": 0, "condition": "random_span",
         "best_unanimous_hit": 0.000, "best_move": "M2"},
        {"target": "cube", "seed": 0, "condition": "clu_shuffled",
         "best_unanimous_hit": 0.344, "best_move": "M4a"},
        {"target": "cube", "seed": 0, "condition": "clu_deleted",
         "best_unanimous_hit": 0.188, "best_move": "M4a"},
        {"target": "cube", "seed": 0, "condition": "knn_proto_gated",
         "best_unanimous_hit": 0.646, "best_move": "M4a"},
    ]
    row = E.claim1_verdict(recs, margin=0.10)["rows"][0]
    assert row["specificity_binding_control"] == "clu_shuffled"
    assert row["specificity_worst_case"] == pytest.approx(0.344)
    assert row["calibrated"] is False
    assert row["verdict"] == "ADVERSARY NOT CALIBRATED"
    assert "clu_deleted" not in row


def test_a_cell_that_fails_calibration_is_not_read_as_an_erasure_result():
    recs = [
        {"target": "t", "seed": 0, "condition": "clu_full", "best_unanimous_hit": 0.10,
         "best_move": "M1"},
        {"target": "t", "seed": 0, "condition": "random_span", "best_unanimous_hit": 0.09,
         "best_move": "M4a"},
        {"target": "t", "seed": 0, "condition": "clu_deleted", "best_unanimous_hit": 0.0,
         "best_move": "M1"},
        {"target": "t", "seed": 0, "condition": "clu_shuffled",
         "best_unanimous_hit": 0.01, "best_move": "M5"},
        {"target": "t", "seed": 0, "condition": "basis_head_gated",
         "best_unanimous_hit": 1.0, "best_move": "M2"},
    ]
    v = E.claim1_verdict(recs, margin=0.10)
    assert v["rows"][0]["calibrated"] is False
    assert v["rows"][0]["verdict"] == "ADVERSARY NOT CALIBRATED"
    assert "clu_deleted" not in v["rows"][0]      # ⛔ no erasure number is exposed


def test_a_calibrated_cell_reports_the_gap_between_gated_and_refitted():
    recs = [
        {"target": "t", "seed": 0, "condition": "clu_full", "best_unanimous_hit": 0.60,
         "best_move": "M1"},
        {"target": "t", "seed": 0, "condition": "random_span", "best_unanimous_hit": 0.05,
         "best_move": "M4a"},
        {"target": "t", "seed": 0, "condition": "clu_deleted", "best_unanimous_hit": 0.02,
         "best_move": "M4a"},
        {"target": "t", "seed": 0, "condition": "basis_head_gated",
         "best_unanimous_hit": 0.95, "best_move": "M2"},
        {"target": "t", "seed": 0, "condition": "clu_shuffled",
         "best_unanimous_hit": 0.03, "best_move": "M5"},
        {"target": "t", "seed": 0, "condition": "knn_proto_gated",
         "best_unanimous_hit": 0.90, "best_move": "M2"},
    ]
    v = E.claim1_verdict(recs, margin=0.10)
    row = v["rows"][0]
    assert row["calibrated"] is True
    assert row["verdict"] == "ERASED vs GATED"
    assert row["gap_gated_minus_clu"] == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# how a move's recovery is READ -- the two rules that decide every headline
# ---------------------------------------------------------------------------


def test_an_interface_move_is_read_AFTER_the_gate():
    """⛔ Crediting the adversary with a request the gate refuses would overstate the
    GATED baseline's leakage, which is the one direction this hub must not err in."""
    assert E.move_recovery(
        {"move": "M1", "unanimous_hit": 0.934, "unanimous_hit_after_gate": 0.0}) == 0.0


def test_an_ungated_interface_move_is_read_at_its_plain_rate():
    assert E.move_recovery({"move": "M1", "unanimous_hit": 0.54}) == pytest.approx(0.54)


def test_the_weight_read_is_read_at_its_TARGETED_rate_when_one_exists():
    """An adversary does not need half the parameters to leak; it needs one, and it
    knows which one. The all-parameter rate has the wrong denominator."""
    assert E.move_recovery({"move": "M2_weight_read", "targeted_hit": 0.75,
                            "unanimous_hit": 0.005}) == pytest.approx(0.75)


def test_the_weight_read_falls_back_when_the_parameters_carry_no_value_labels():
    """`clu`'s atoms are whole train rows indexed by nothing, so it has no targeted
    set and the all-parameter rate is the right and only statistic."""
    assert E.move_recovery({"move": "M2_weight_read", "targeted_hit": None,
                            "unanimous_hit": 0.109}) == pytest.approx(0.109)


def test_an_inapplicable_move_contributes_no_number():
    assert E.move_recovery({"move": "M3", "applicable": False, "why": "x"}) is None
    assert E.move_recovery({"move": "M5", "status": "NOT APPLICABLE"}) is None


def test_offgrid_interpolation_lands_between_the_bracketing_one_hots():
    """The off-grid move must actually be off-grid: at the midpoint the encoding is
    half on each bracketing value, which is what makes it a bracket."""
    grid = _Grid()
    # a DENSE random head: an identity-like W would discard most request features and
    # the test would pass on a fixture that cannot tell the two values apart at all.
    rng = np.random.default_rng(0)
    P = len(E.request_feature_names(CARDS))
    arm = {"cardinalities": CARDS,
           "W": rng.normal(0, 1, (P, 5)).astype(np.float32),
           "B": rng.normal(0, 1, (5, 8)).astype(np.float32)}
    V = np.zeros((1, 6), np.int64)
    mid = float((grid.values[E.F_OBJECT_HUE][4] + grid.values[E.F_OBJECT_HUE][5]) / 2)
    x_mid = E.basis_head_read_offgrid(arm, V, E.F_OBJECT_HUE, mid, grid)
    V4 = V.copy(); V4[:, E.F_OBJECT_HUE] = 4
    V5 = V.copy(); V5[:, E.F_OBJECT_HUE] = 5
    x4 = E.basis_head_read(arm, V4)
    x5 = E.basis_head_read(arm, V5)
    assert np.allclose(x_mid, 0.5 * (x4 + x5), atol=1e-5)
    assert not np.allclose(x_mid, x4)
