"""⭐ Navigation (``gen-navigation``): traversal between the store's committed codes.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_assert_no_feature_tensor_FIRES_when_a_feature_argument_appears`` and
  ``..._FIRES_when_soft_read_is_imported`` — the no-bypass guard is seen to fire, not
  merely seen to pass.  ``gen_store.soft_read`` and ``gen_store.address_codes`` both
  take FEATURES; either on the traversal path would void R20's 30-bit guarantee.
- ``test_mean_deleted_traversal_is_CONSTANT`` — the fair deletion control is
  demonstrably dead: every code reads the same vector, so nothing it produces can be
  smoothness the store supplied.  ⭐ PREREG P6 predicts exactly 0, so a non-zero
  traverse rate there is a plumbing bug and this test is what catches it.
- ``test_a_two_valued_factor_would_score_a_vacuous_monotone_success`` — the
  step-function trap, made explicit: a weakly-monotone path over a 2-valued factor
  is a *switch*, not a traversal, which is why ``min_delta = 2`` is pre-registered
  and why ``n_values_realised`` travels beside every per-factor number (R31, fourth
  strike).
- ``test_select_pairs_is_seeded_and_not_hand_picked`` — the endpoint rule is
  reproducible from the seed alone.

Everything runs on synthetic stores; the real-data path is exercised by the harness
runs reported in ``.claude/outputs/gen-navigation.md``.

⚠ §7.23: other modules enable ``jax_enable_x64`` at import and may sort before this
one; every function under test casts to the **store's own** dtype.
"""

import inspect

import jax
import numpy as np
import pytest

from chlu.config import GenNavigateConfig, gen_navigate_config, get_default_config
from chlu.core import gen_navigate
from chlu.core.gen_navigate import (
    ARMS,
    CONTROL_ARMS,
    STORE_ARM,
    arm_path_features,
    assert_no_feature_tensor,
    enumerate_single_factor_pairs,
    image_smoothness,
    linear_path_features,
    navigation_summary,
    path_channel,
    select_pairs,
    snap_path_codes,
    traversal_metrics,
)
from chlu.core.gen_store import GenStore, read_from_codes

SHIPPED_K, SHIPPED_M, SHIPPED_D = 5, 64, 2048
DESIGN_CONSTANT_BITS = 30.0


def _store(k=4, m=8, dk=6, seed=0):
    wc = tuple(tuple(range(j * dk, (j + 1) * dk)) for j in range(k))
    return GenStore(wc, m, (k * dk,), jax.random.PRNGKey(seed))


def _shipped_store(seed=0):
    edges = np.linspace(0, SHIPPED_D, SHIPPED_K + 1).astype(int)
    wc = tuple(tuple(range(int(a), int(b))) for a, b in zip(edges[:-1], edges[1:], strict=True))
    return GenStore(wc, SHIPPED_M, (SHIPPED_D,), jax.random.PRNGKey(seed))


# ---------------------------------------------------------------------------
# 1. ⛔ THE GUARANTEE — 30 bits, no feature tensor
# ---------------------------------------------------------------------------


def test_the_traversal_path_takes_no_feature_argument():
    st = _store()
    out = assert_no_feature_tensor(st, [0, 1, 2, 3], [4, 5, 6, 7], 4,
                                   decoy_features=np.zeros((3, st.dim), np.float32))
    assert out["no_feature_argument_on_the_path"]
    assert out["soft_read_not_imported"] and out["address_codes_not_imported"]
    assert out["endpoints_exact"]
    assert all(v for k, v in out["numeric_regression_guard"].items()
               if k.endswith("_replacement"))


def test_assert_no_feature_tensor_FIRES_when_a_feature_argument_appears(monkeypatch):
    """⭐ A guard that has never fired has never been tested."""
    st = _store()

    def sneaky(store, code_a, code_b, n_steps=8, features=None):  # noqa: ARG001
        return snap_path_codes(store, code_a, code_b, n_steps)

    monkeypatch.setattr(gen_navigate, "snap_path_codes", sneaky)
    with pytest.raises(AssertionError, match="bypass"):
        assert_no_feature_tensor(st, [0, 1, 2, 3], [4, 5, 6, 7], 4)


def test_assert_no_feature_tensor_FIRES_when_soft_read_is_imported(monkeypatch):
    from chlu.core.gen_store import soft_read

    st = _store()
    monkeypatch.setattr(gen_navigate, "soft_read", soft_read, raising=False)
    with pytest.raises(AssertionError, match="FEATURES"):
        assert_no_feature_tensor(st, [0, 1, 2, 3], [4, 5, 6, 7], 4)


def test_assert_no_feature_tensor_FIRES_when_address_codes_is_imported(monkeypatch):
    """``address_codes`` is the *other* feature-taking function and is banned too."""
    from chlu.core.gen_store import address_codes

    st = _store()
    monkeypatch.setattr(gen_navigate, "address_codes", address_codes, raising=False)
    with pytest.raises(AssertionError, match="FEATURES"):
        assert_no_feature_tensor(st, [0, 1, 2, 3], [4, 5, 6, 7], 4)


def test_every_intermediate_point_of_a_store_arm_is_exactly_30_bits():
    """⛔ R20's declared design constant, checked at the shipped cell's shape."""
    st = _shipped_store()
    for arm in (STORE_ARM, "deleted_mean", "deleted_scramble"):
        ch = path_channel(st, arm, 8, DESIGN_CONSTANT_BITS)
        assert ch["per_point_bits"] == pytest.approx(DESIGN_CONSTANT_BITS)
        assert not ch["exceeds_design_constant"]
    lin = path_channel(st, "linear", 8, DESIGN_CONSTANT_BITS)
    assert lin["exceeds_design_constant"], "the linear control's per-point object is a real vector"
    assert lin["per_point_bits"] == SHIPPED_D * 32
    # ⭐ but the PATH it can carry is still two codes plus the step index
    assert lin["path_information_bits"] == pytest.approx(2 * 30.0 + np.log2(9))
    assert lin["path_information_bits"] == pytest.approx(ch["path_information_bits"])


def test_deleted_arms_carry_zero_informative_bits():
    st = _store()
    for arm in ("deleted_mean", "deleted_scramble"):
        assert path_channel(st, arm, 4)["informative_bits"] == 0.0
    assert path_channel(st, STORE_ARM, 4)["informative_bits"] > 0.0


# ---------------------------------------------------------------------------
# 2. The paths themselves
# ---------------------------------------------------------------------------


def test_snap_path_endpoints_are_exact_and_the_path_is_a_code_sequence():
    st = _store(k=4, m=8)
    a, b = [0, 1, 2, 3], [4, 5, 6, 7]
    codes = snap_path_codes(st, a, b, 8)
    assert codes.shape == (9, 4)
    assert np.array_equal(codes[0], a) and np.array_equal(codes[-1], b)
    assert codes.dtype == np.int64


def test_snap_path_is_deterministic_and_reversible_in_shape():
    st = _store()
    a, b = [0, 1, 2, 3], [4, 5, 6, 7]
    assert np.array_equal(snap_path_codes(st, a, b, 6), snap_path_codes(st, a, b, 6))
    rev = snap_path_codes(st, b, a, 6)
    assert np.array_equal(rev[0], b) and np.array_equal(rev[-1], a)


def test_identical_endpoints_give_a_single_point_path():
    """⭐ The corner's failure mode, in miniature: if the commitment gives both
    requests the same code there is nothing to traverse (PREREG P1/P2)."""
    st = _store()
    codes = snap_path_codes(st, [1, 1, 1, 1], [1, 1, 1, 1], 8)
    assert np.unique(codes, axis=0).shape[0] == 1


def test_linear_control_matches_the_store_read_at_the_endpoints_only():
    st = _store()
    a, b = [0, 1, 2, 3], [4, 5, 6, 7]
    lin = linear_path_features(st, a, b, 4)
    ya = np.asarray(read_from_codes(st, np.asarray(a)[None, :]))[0]
    yb = np.asarray(read_from_codes(st, np.asarray(b)[None, :]))[0]
    assert np.allclose(lin[0], ya) and np.allclose(lin[-1], yb)
    # the interior is off the codebook: it equals no single read
    assert not np.allclose(lin[2], ya) and not np.allclose(lin[2], yb)


def test_mean_deleted_traversal_is_CONSTANT():
    """⭐⭐ The fair deletion control is dead by construction (PREREG P6 = 0.00)."""
    st = _store()
    feats = arm_path_features(st, [0, 1, 2, 3], [4, 5, 6, 7], 8, "deleted_mean")["features"]
    assert np.allclose(feats, feats[0]), "a mean-deleted store must read the same vector for every code"
    sm = image_smoothness(feats)
    assert sm["mean_step_rmse"] == 0.0 and sm["n_zero_steps"] == 8


def test_scramble_deletion_still_MOVES_which_is_what_makes_it_a_real_control():
    st = _store()
    feats = arm_path_features(st, [0, 1, 2, 3], [4, 5, 6, 7], 8, "deleted_scramble",
                              seed=7)["features"]
    assert not np.allclose(feats, feats[0]), (
        "scramble preserves per-well marginals, so the path must still move — that is "
        "exactly why smoothness alone cannot mean 'meaningful'"
    )


def test_arm_path_features_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="arm must be one of"):
        arm_path_features(_store(), [0, 1, 2, 3], [1, 2, 3, 4], 4, "wishful_thinking")


def test_control_arms_are_declared_and_run_on_every_number():
    assert STORE_ARM == "store_snap"
    assert set(CONTROL_ARMS) == {"linear", "deleted_mean", "deleted_scramble"}
    assert ARMS[0] == STORE_ARM and set(ARMS) == {STORE_ARM, *CONTROL_ARMS}


# ---------------------------------------------------------------------------
# 3. Endpoint selection — ⛔ enumerated and seeded, never hand-picked
# ---------------------------------------------------------------------------


def _line(card=5, n_other=3, seed=0):
    """``n_other`` parallel lines through a 2-factor grid: factor 0 varies, 1 is fixed."""
    rows = [[v, o] for o in range(n_other) for v in range(card)]
    return np.asarray(rows, np.int64)


def test_enumerate_single_factor_pairs_finds_only_single_factor_edges():
    R = _line(card=4, n_other=3)
    pairs = enumerate_single_factor_pairs(R, 0, 1)
    assert pairs.shape[0] == 3 * (4 * 3 // 2)
    for i, j in pairs:
        assert R[i, 1] == R[j, 1] and R[i, 0] != R[j, 0]


def test_min_delta_excludes_adjacent_pairs():
    R = _line(card=4, n_other=1)
    assert enumerate_single_factor_pairs(R, 0, 1).shape[0] == 6
    got = enumerate_single_factor_pairs(R, 0, 2)
    assert got.shape[0] == 3
    for i, j in got:
        assert abs(int(R[i, 0]) - int(R[j, 0])) >= 2


def test_select_pairs_is_seeded_and_not_hand_picked():
    R = _line(card=6, n_other=4)
    a = select_pairs(R, 0, 5, np.random.default_rng(11), 2)["pairs"]
    b = select_pairs(R, 0, 5, np.random.default_rng(11), 2)["pairs"]
    c = select_pairs(R, 0, 5, np.random.default_rng(12), 2)["pairs"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_select_pairs_falls_back_to_delta_one_and_SAYS_SO():
    """⭐ A 2-valued factor has no ``|Δ| >= 2`` pair; the fallback must be flagged so
    the monotonicity statistic never reads it (the step-function trap)."""
    R = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], np.int64)
    got = select_pairs(R, 0, 4, np.random.default_rng(0), 2)
    assert got["qualifies_monotone"] is False
    assert got["min_delta_used"] == 1
    assert got["n_values_realised"] == 2


def test_a_one_valued_factor_yields_NOT_APPLICABLE_never_a_zero():
    """⛔ R31, fourth strike: Montero's corner pins ``shape`` to ONE value and a
    factor with one realised value produces a confident-looking zero."""
    R = np.asarray([[0, 0], [0, 1], [0, 2]], np.int64)
    got = select_pairs(R, 0, 4, np.random.default_rng(0), 2)
    assert got["pairs"].shape == (0, 2)
    assert got["n_values_realised"] == 1
    assert "NOT APPLICABLE" in got["note"]


# ---------------------------------------------------------------------------
# 4. The metrics — and the traps they are written around
# ---------------------------------------------------------------------------


def _readings(trace, others, n_factors=3):
    R = np.zeros((len(trace), n_factors), np.int64)
    R[:, 0] = trace
    for j, col in enumerate(others, start=1):
        R[:, j] = col
    return R


def test_a_clean_monotone_traversal_scores_TRAVERSES():
    R = _readings([0, 0, 1, 1, 2, 2, 3, 3, 4], [[7] * 9, [3] * 9])
    m = traversal_metrics(R, 0, 0, 4, [7, 3], True)
    assert m["monotone"] and m["visits_intermediate"] and m["endpoints_read_ok"]
    assert m["off_target_stability"] == 1.0
    assert m["TRAVERSES"]


def test_a_STEP_FUNCTION_is_weakly_monotone_but_does_NOT_traverse():
    """⭐⭐ The single most dangerous false positive: a path that jumps straight from
    one endpoint to the other is weakly monotone and would score as success."""
    R = _readings([0, 0, 0, 0, 4, 4, 4, 4, 4], [[7] * 9, [3] * 9])
    m = traversal_metrics(R, 0, 0, 4, [7, 3], True)
    assert m["monotone"], "a step function IS weakly monotone"
    assert not m["visits_intermediate"]
    assert not m["TRAVERSES"]
    assert m["clean_switch"]


def test_a_two_valued_factor_would_score_a_vacuous_monotone_success():
    """⭐ Why ``min_delta = 2`` is pre-registered: with values {0, 1} there is no
    intermediate value to visit, so *every* correct path is a step function."""
    R = _readings([0, 0, 0, 1, 1, 1, 1, 1, 1], [[7] * 9, [3] * 9])
    vacuous = traversal_metrics(R, 0, 0, 1, [7, 3], qualifies_monotone=True)
    assert vacuous["monotone"] and not vacuous["visits_intermediate"]
    honest = traversal_metrics(R, 0, 0, 1, [7, 3], qualifies_monotone=False)
    assert honest["TRAVERSES"] is False


def test_a_traversal_that_changes_everything_does_NOT_traverse():
    """⭐ Off-target stability is a gate, not a garnish."""
    drift = [7, 7, 6, 5, 4, 3, 2, 1, 0]
    R = _readings([0, 0, 1, 1, 2, 2, 3, 3, 4], [drift, [3] * 9])
    m = traversal_metrics(R, 0, 0, 4, [7, 3], True)
    assert m["monotone"] and m["visits_intermediate"]
    assert m["off_target_stability"] < 0.9
    assert not m["TRAVERSES"]


def test_non_monotone_wandering_does_not_traverse():
    R = _readings([0, 3, 1, 4, 2, 2, 3, 3, 4], [[7] * 9, [3] * 9])
    m = traversal_metrics(R, 0, 0, 4, [7, 3], True)
    assert not m["monotone"] and not m["TRAVERSES"]


def test_wrong_endpoints_do_not_traverse():
    R = _readings([1, 1, 2, 2, 3, 3, 3, 3, 3], [[7] * 9, [3] * 9])
    m = traversal_metrics(R, 0, 0, 4, [7, 3], True)
    assert not m["endpoints_read_ok"] and not m["TRAVERSES"]


def test_image_smoothness_prices_a_jump():
    even = np.stack([np.full((4,), float(t)) for t in range(5)])
    jumpy = np.stack([np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.full((4,), 4.0)])
    assert image_smoothness(even)["jump_ratio"] == pytest.approx(1.0)
    assert image_smoothness(jumpy)["jump_ratio"] > 3.0
    assert image_smoothness(even)["path_efficiency"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. ⭐ The pre-registered summary statistic
# ---------------------------------------------------------------------------


def _rec(store_t, lin_t, del_t=False, qual=True, stab=1.0):
    def arm(t):
        return {"TRAVERSES": bool(t), "qualifies_monotone": bool(qual),
                "off_target_stability": float(stab)}
    return {"arms": {STORE_ARM: arm(store_t), "linear": arm(lin_t),
                     "deleted_mean": arm(del_t), "deleted_scramble": arm(False)}}


def test_NAV_counts_only_strict_wins_over_BOTH_controls():
    recs = [_rec(True, False), _rec(True, True), _rec(False, True), _rec(True, False)]
    s = navigation_summary(recs)
    assert s["NAV"] == pytest.approx(0.5)
    assert s["traverse_rate"][STORE_ARM] == pytest.approx(0.75)
    assert s["traverse_rate"]["linear"] == pytest.approx(0.5)


def test_the_verdict_fires_when_interpolation_does_as_well():
    """⛔ The pre-registered headline: if the control ties or wins, say so plainly."""
    s = navigation_summary([_rec(False, True), _rec(True, True)])
    assert s["linear_traverses_as_well_or_better"]
    assert "ADDS NOTHING" in s["verdict"]
    s2 = navigation_summary([_rec(True, False), _rec(True, False)])
    assert not s2["linear_traverses_as_well_or_better"]


def test_a_TIE_AT_ZERO_is_reported_as_neither_arm_traversing():
    """⛔ "interpolation does it too" and "nobody does it" are different findings and
    must not share a sentence — the honest failure mode of this whole spoke."""
    s = navigation_summary([_rec(False, False), _rec(False, False)])
    assert s["both_arms_at_zero"]
    assert "NEITHER ARM TRAVERSES" in s["verdict"]
    assert "ADDS NOTHING" not in s["verdict"]
    # ⭐ but the pre-registered comparison flag still reads the rule verbatim
    assert s["linear_traverses_as_well_or_better"]


def test_non_qualifying_pairs_are_excluded_from_the_headline():
    s = navigation_summary([_rec(True, False, qual=False), _rec(True, False, qual=True)])
    assert s["n_qualifying_pairs"] == 1
    assert s["n_pairs_total"] == 2


def test_no_qualifying_pair_is_reported_as_NOT_RUN_never_as_zero():
    s = navigation_summary([_rec(True, False, qual=False)])
    assert s["n_qualifying_pairs"] == 0
    assert "NOT RUN" in s["verdict"]


# ---------------------------------------------------------------------------
# 6. Config block
# ---------------------------------------------------------------------------


def test_config_block_defaults_are_the_preregistered_ones():
    c = GenNavigateConfig()
    assert c.n_steps == 8 and c.min_delta == 2 and c.n_pairs_per_factor == 8
    assert c.stability_floor == 0.9
    assert c.design_constant_bits == 30.0
    assert c.require_gate_pass is True
    assert c.expected_decoder_recipe == "v1_capacity_schedule", (
        "⛔ dec_recipe still defaults to 'baseline', the recipe that FAILED the gate"
    )
    assert list(c.pools) == ["CORNER", "GRID"]


def test_gen_navigate_config_memoises_and_is_not_a_CHLUConfig_field():
    cfg = get_default_config()
    a = gen_navigate_config(cfg)
    b = gen_navigate_config(cfg)
    assert a is b
    # ⛔ the block is reached through the accessor precisely so the merge anchor at
    # the file tail stays one contiguous region with zero house wiring sites
    assert "gen_navigate" not in {f for f in getattr(type(cfg), "__dataclass_fields__", {})}


def test_experiment_module_never_imports_a_feature_taking_read():
    from chlu.experiments import exp_gen_navigate as exp

    for banned in ("soft_read", "address_codes", "store_read"):
        assert banned not in vars(exp), (
            f"{banned} takes the FEATURES; importing it into the navigation harness "
            "would void the 30-bit no-bypass guarantee"
        )
    assert list(inspect.signature(exp.run_seed).parameters) == [
        "seed", "out_dir", "config", "log"]
