"""Tests for the FB4 gate — the attention arm and the pre-registered rule.

⛔ The two constants (``0.95`` and the 2-SE attention leg) are **pre-registered
and non-tunable**; a test asserts their values so a later edit cannot silently
move the gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from chlu.eval.fb4_gate import (
    ATTENTION_SE_LEGS,
    ATTENTION_TEMPERATURE_BYTES,
    METRIC_MAX,
    SATURATION_THRESHOLD,
    attention_logits,
    attention_read,
    attention_weights,
    family_saturated,
    fb4_verdict,
    fit_attention_temperature,
    saturation,
)


# --------------------------------------------------------------------------
# the pre-registered constants
# --------------------------------------------------------------------------
def test_prereg_constants_are_frozen():
    assert SATURATION_THRESHOLD == 0.95
    assert ATTENTION_SE_LEGS == 2.0
    assert ATTENTION_TEMPERATURE_BYTES == 4
    assert METRIC_MAX == {"decode": 1.0, "acc": 1.0, "r2": 1.0, "neg_mae": 0.0}


# --------------------------------------------------------------------------
# the attention arm
# --------------------------------------------------------------------------
def test_attention_weights_are_a_simplex():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(5, 4))
    q = rng.normal(size=(7, 4))
    w = attention_weights(k, q, temperature=0.3)
    assert w.shape == (7, 5)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.all(w >= 0.0)


def test_attention_read_is_a_convex_combination_of_stored_values():
    """The arm reads the table and nothing else: its output is inside the hull."""
    rng = np.random.default_rng(1)
    k = rng.normal(size=(6, 3))
    v = rng.normal(size=(6, 1))
    q = rng.normal(size=(9, 3))
    pred = attention_read(k, v, q, temperature=0.5, kind="value")
    assert pred.shape == (9, 1)
    assert np.all(pred >= v.min() - 1e-9) and np.all(pred <= v.max() + 1e-9)


def test_attention_at_low_temperature_is_the_dot_product_argmax():
    rng = np.random.default_rng(2)
    k = rng.normal(size=(5, 4))
    v = np.arange(5, dtype=float)[:, None]
    q = rng.normal(size=(11, 4))
    pred = attention_read(k, v, q, temperature=1e-3, kind="value").ravel()
    hard = v[np.argmax(attention_logits(k, q), axis=1)].ravel()
    assert np.allclose(pred, hard)


def test_attention_on_an_all_zero_value_column_reads_zero():
    """⭐ The manifold family: a table stores ONE point per item and the spectator
    coordinate it stored is the written one (zero), so **any** convex combination
    of it is zero. The arm is not handicapped — that is the table's own answer."""
    rng = np.random.default_rng(3)
    k = rng.normal(size=(4, 3))
    pred = attention_read(k, np.zeros((4, 1)), rng.normal(size=(10, 3)),
                          temperature=0.7, kind="coord")
    assert np.allclose(pred, 0.0)


def test_attention_index_arm_is_temperature_independent():
    """A positive scalar cannot reorder two logits — reported, not hidden."""
    rng = np.random.default_rng(4)
    k = rng.normal(size=(6, 4))
    q = rng.normal(size=(12, 4))
    pairs = np.stack([np.zeros(12, dtype=int), np.ones(12, dtype=int) * 3], axis=1)
    a = attention_read(k, None, q, temperature=0.01, kind="index", pairs=pairs)
    b = attention_read(k, None, q, temperature=50.0, kind="index", pairs=pairs)
    assert np.array_equal(a, b)
    assert set(np.unique(a).tolist()) <= {0, 3}


def test_fit_temperature_reports_a_degenerate_fit_as_degenerate():
    rng = np.random.default_rng(5)
    k = rng.normal(size=(5, 3))
    q = rng.normal(size=(8, 3))
    out = fit_attention_temperature(k, None, q, lambda p: 0.5, kind="index",
                                    pairs=np.stack([np.zeros(8, dtype=int),
                                                    np.ones(8, dtype=int)], axis=1))
    assert out["degenerate"] is True
    assert out["bytes"] == ATTENTION_TEMPERATURE_BYTES


def test_fit_temperature_picks_the_grid_maximum():
    rng = np.random.default_rng(6)
    k = rng.normal(size=(5, 3))
    v = rng.normal(size=(5, 1))
    q = rng.normal(size=(20, 3))
    tgt = attention_read(k, v, q, temperature=0.5, kind="value")
    out = fit_attention_temperature(
        k, v, q, lambda p: -float(np.mean(np.abs(p - tgt))), kind="value")
    assert 0.3 <= out["tau"] <= 0.8
    assert out["fit_score"] > -1e-2


def test_attention_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        attention_logits(np.zeros((3, 4)), np.zeros((5, 2)))


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def test_saturation_is_floor_normalised_on_both_metric_signs():
    # decode-like: M = 1
    assert saturation(1.0, 0.1667, 1.0) == pytest.approx(1.0)
    assert saturation(0.7083, 0.1667, 1.0) == pytest.approx(0.65, abs=1e-3)
    # neg_mae-like: M = 0, both terms negative
    assert saturation(-0.2081, -0.4221, 0.0) == pytest.approx(0.5070, abs=1e-3)
    assert np.isnan(saturation(0.5, 1.0, 1.0 + 1e-15))


def test_a_family_needs_BOTH_legs_to_saturate():
    # ceiling leg passes, attention leg fails (attention clearly beats the sub)
    r = family_saturated("f", "acc", sub_seeds=[0.98, 0.98, 0.98],
                         attn_seeds=[1.0, 1.0, 1.0], blank_seeds=[0.0, 0.0, 0.0])
    assert r.S == pytest.approx(0.98)
    assert r.detail["leg_ceiling(S>=0.95)"] is True
    assert r.detail["leg_attention(sub>=attn-2SE)"] is False
    assert r.saturated is False
    # attention leg passes, ceiling leg fails
    r2 = family_saturated("f", "acc", sub_seeds=[0.5, 0.5, 0.5],
                          attn_seeds=[0.1, 0.1, 0.1], blank_seeds=[0.0, 0.0, 0.0])
    assert r2.saturated is False
    # both pass
    r3 = family_saturated("f", "acc", sub_seeds=[1.0, 1.0, 1.0],
                          attn_seeds=[0.5, 0.5, 0.5], blank_seeds=[0.0, 0.0, 0.0])
    assert r3.saturated is True


def test_attention_leg_uses_the_PAIRED_se():
    """Both arms are read off the same store at the same seed, so the leg's SE is
    the SE of the per-seed difference — not of either arm alone."""
    r = family_saturated("f", "acc", sub_seeds=[0.5, 0.7, 0.9],
                         attn_seeds=[0.52, 0.72, 0.92], blank_seeds=[0.0, 0.0, 0.0])
    assert r.se_paired == pytest.approx(0.0, abs=1e-12)   # constant difference
    assert r.detail["se_sub"] > 0.1                       # but each arm varies
    assert r.saturated is False  # sub < attn and the paired SE gives it no slack


def test_verdict_fires_only_when_all_four_saturate():
    def row(name, sub, attn, blank):
        return family_saturated(name, "acc", sub_seeds=[sub] * 3,
                                attn_seeds=[attn] * 3, blank_seeds=[blank] * 3)

    all4 = [row(n, 1.0, 0.5, 0.0)
            for n in ("overload", "aggregate", "recency", "manifold")]
    assert fb4_verdict(all4)["verdict"] == "FIRES"
    assert fb4_verdict(all4)["surviving_families"] == []

    three = all4[:3] + [row("manifold", 0.2, 0.1, 0.0)]
    v = fb4_verdict(three)
    assert v["verdict"].startswith("PARTIAL")
    assert v["surviving_families"] == ["manifold"]

    none = [row(n, 0.2, 0.1, 0.0)
            for n in ("overload", "aggregate", "recency", "manifold")]
    assert fb4_verdict(none)["verdict"] == "CLEARS"


def test_manifold_only_partial_is_reported_as_CLEARS_not_news():
    """A partial of exactly {manifold} is the PREDICTED outcome (echo = 1.0000 at
    +0 B by construction), so the gate must not dress it up as a firing."""
    def row(name, sub, attn, blank):
        return family_saturated(name, "acc", sub_seeds=[sub] * 3,
                                attn_seeds=[attn] * 3, blank_seeds=[blank] * 3)

    rows = [row(n, 0.2, 0.1, 0.0) for n in ("overload", "aggregate", "recency")]
    rows.append(row("manifold", 1.0, 0.5, 0.0))
    v = fb4_verdict(rows)
    assert v["verdict"].startswith("CLEARS")
    assert v["saturated"] == ["manifold"]
    assert v["surviving_families"] == ["aggregate", "overload", "recency"]


def test_a_three_family_run_can_never_fire_the_gate():
    def row(name):
        return family_saturated(name, "acc", sub_seeds=[1.0] * 3,
                                attn_seeds=[0.5] * 3, blank_seeds=[0.0] * 3)

    v = fb4_verdict([row(n) for n in ("overload", "aggregate", "recency")])
    assert v["verdict"] != "FIRES"


# --------------------------------------------------------------------------
# the runner's own helpers (no store is built here)
# --------------------------------------------------------------------------
def test_order_aware_pair_substitute_is_exact_by_construction():
    """⭐ Once every arm answers the SAME 2-way question, a reader of the table's
    row order answers it exactly at +0 B. That is what FB4 is built to detect."""
    from chlu.experiments.exp_fb4_gate import order_aware_pair_launder

    born = np.array([3.0, 0.0, 2.0, 1.0])
    pairs = np.array([[0, 1], [2, 3], [1, 3], [0, 2]])
    pred = order_aware_pair_launder(born, pairs)
    assert np.array_equal(pred, np.array([0, 2, 3, 0]))


def test_zero_byte_candidate_set_includes_the_launder_itself():
    from chlu.experiments.exp_fb4_gate import zero_byte_candidates

    sc = {"settle_deleted": 0.4, "same_keys_null": 0.4, "knn2_idw_+0B": 0.7}
    got = zero_byte_candidates("aggregate", sc)
    assert got == ["settle_deleted", "knn2_idw_+0B"]   # the null is NOT a reader


def test_phi_ledger_row_is_identical_across_arms_and_hashes_the_queries():
    from chlu.experiments.exp_fb4_gate import _assert_identical_phi_rows, phi_ledger_row
    from chlu.experiments.memory_gym import QuerySet

    qs = QuerySet(q0=np.ones((4, 3)), keys=np.ones((4, 2)), target=np.zeros((4, 1)),
                  label=np.zeros(4, dtype=int), alphabet=None, kind="value")
    row = phi_ledger_row(qs)
    assert row["phi_bytes"] == 0
    assert _assert_identical_phi_rows({"full": row, "attention": dict(row)}) == row["phi_id"]

    other = QuerySet(q0=np.zeros((4, 3)), keys=np.ones((4, 2)),
                     target=np.zeros((4, 1)), label=np.zeros(4, dtype=int),
                     alphabet=None, kind="value")
    from chlu.core.psi_readout import PhiMismatchError

    with pytest.raises(PhiMismatchError):
        _assert_identical_phi_rows({"full": row, "attention": phi_ledger_row(other)})
