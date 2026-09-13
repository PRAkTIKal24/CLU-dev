"""Composition through the READ (``gen-read-compose``): the four-arm cell path.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_clu_wide_is_request_derived_while_soft_read_is_FEATURE_derived`` — the
  Hub's trap, reproduced: :func:`chlu.core.gen_store.soft_read` computes exactly the
  arithmetic ``CLU-wide`` needs but takes the FEATURES, so using it would be an
  encoder → decoder path and would void the programme's no-bypass guarantee.  The
  test shows ``soft_read``'s output **moving** with the features while
  :func:`chlu.core.gen_compose.clu_wide_vector`'s is bit-identical.
- ``test_assert_request_only_FIRES_when_a_feature_argument_appears`` and
  ``..._FIRES_when_soft_read_is_imported`` — the guard is seen to fire, not merely
  seen to pass.
- ``test_store_removed_read_carries_ZERO_information`` — the laundering control is
  demonstrably dead: under ``mean`` deletion every request renders the same vector,
  so any surviving accuracy is accuracy the store never supplied.
- ``test_metric_invariance_check_is_not_vacuous`` — the R26 standing-rule check
  only means something if the re-partition actually changes *something*; raw
  accuracy must move while balanced accuracy does not.
- ``test_falsifier_reports_a_DEAD_read_as_dead`` — the spoke's own falsifier fires
  when the read and its ablation coincide.

Everything runs on synthetic tensors; the real-data path is exercised by the
harness runs reported in ``.claude/outputs/gen-read-compose.md``.

⚠ §7.23: several test modules enable ``jax_enable_x64`` at **module import** and may
sort before this one.  Every function under test casts to the **store's own** dtype
(the ``gen_store`` convention), which is why these pass in the full suite and not
only in isolation.  ⛔ A float32-pinning fixture is not the fix.
"""

import inspect

import jax
import numpy as np
import pytest

from chlu.config import GenComposeConfig, get_default_config, load_config, save_config
from chlu.core.gen_compose import (
    ARMS,
    COMPOSITIONAL_FACTOR,
    address_book_bytes,
    assert_request_only,
    channel_ledger,
    clu_wide_vector,
    code_agreement,
    compose_codes,
    deployed_byte_ledger,
    metric_invariance_check,
    read_compose,
    request_posterior,
    store_removed_read,
)
from chlu.core.gen_dream import fit_address_book
from chlu.core.gen_store import GenStore, soft_read, store_bytes
from chlu.experiments.exp_gen_compose import seed_falsifier

#: The shipped Shapes3D/recon cell, for the byte and channel ledgers.
SHIPPED_K, SHIPPED_M, SHIPPED_D = 5, 64, 2048
STORE_BYTES = 524_288       # n_modes * D * 4
COUPLING_BYTES = 163_840    # K(K-1)/2 * M^2 * 4 — ⛔ NOT deployed on this path (R26)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _store(k=4, m=8, dk=6, seed=0):
    wc = tuple(tuple(range(j * dk, (j + 1) * dk)) for j in range(k))
    return GenStore(wc, m, (k * dk,), jax.random.PRNGKey(seed))


def _shipped_store(seed=0):
    """The shipped Shapes3D/recon cell: 5 wells PARTITIONING D = 2048, M = 64.

    ⚠ ``D`` is not divisible by ``K``, and that matters: ``store_bytes`` is
    ``M·D·4`` **independently of K**, so an evenly-chopped fixture would quietly
    miss the published 524,288 B by the remainder.
    """
    edges = np.linspace(0, SHIPPED_D, SHIPPED_K + 1).astype(int)
    wc = tuple(tuple(range(int(a), int(b))) for a, b in zip(edges[:-1], edges[1:], strict=True))
    return GenStore(wc, SHIPPED_M, (SHIPPED_D,), jax.random.PRNGKey(seed))


def _book(k=4, m=8, n=400, cards=(3, 4), seed=0):
    """An address book fitted on codes that genuinely depend on the factors."""
    rng = np.random.default_rng(seed)
    Y = np.stack([rng.integers(0, c, size=n) for c in cards], axis=1)
    codes = np.stack(
        [(Y[:, j % len(cards)] * 2 + j) % m for j in range(k)], axis=1
    ).astype(np.int64)
    return fit_address_book(codes, Y, cards), Y, codes


# ---------------------------------------------------------------------------
# ⛔ 1. THE TRAP: request-derived weights vs feature-derived ones
# ---------------------------------------------------------------------------


def test_clu_wide_is_request_derived_while_soft_read_is_FEATURE_derived():
    """⭐ The single easiest way to void the structural claim, reproduced.

    ``soft_read(store, X, tau)`` computes ``sum_m pi_m · mode_m`` — *exactly* the
    composed per-well mode vector this arm needs — but derives ``pi`` from ``X``.
    ⛔ That is an encoder → decoder path.  ``clu_wide_vector`` takes the posterior,
    and ``request_posterior`` derives it from the request alone, so replacing the
    features must leave it **bit-identical** while ``soft_read`` moves.
    """
    st = _store()
    book, Y, _ = _book()
    P = request_posterior(book, Y[:16])
    wide = np.asarray(clu_wide_vector(st, P))

    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(16, st.dim)).astype(np.float32)
    X2 = 37.0 * rng.normal(size=(16, st.dim)).astype(np.float32)
    s1 = np.asarray(soft_read(st, X1, 0.5)).reshape(16, -1)
    s2 = np.asarray(soft_read(st, X2, 0.5)).reshape(16, -1)

    # the trap moves with the features …
    assert not np.allclose(s1, s2)
    # … the arm does not: it never saw them
    assert np.array_equal(wide, np.asarray(clu_wide_vector(st, request_posterior(book, Y[:16]))))
    assert "X" not in inspect.signature(clu_wide_vector).parameters


def test_assert_request_only_passes_on_the_shipped_path():
    st = _store()
    book, Y, _ = _book()
    rng = np.random.default_rng(1)
    out = assert_request_only(
        st, book, Y[:32], decoy_features=rng.normal(size=(32, st.dim)).astype(np.float32)
    )
    assert out["signature_ok"] and out["soft_read_not_imported"]
    assert out["no_feature_argument_on_the_path"]
    assert out["numeric_regression_guard"]["read_bit_identical_under_feature_replacement"]
    assert out["bits_per_read"] == pytest.approx(4 * np.log2(8))


def test_assert_request_only_FIRES_when_a_feature_argument_appears(monkeypatch):
    """⛔ The guard must fire on the edit that would open a bypass."""
    import chlu.core.gen_compose as gc

    def leaky_clu_wide_vector(store, posterior, features):  # noqa: ARG001
        return clu_wide_vector(store, posterior)

    monkeypatch.setattr(gc, "clu_wide_vector", leaky_clu_wide_vector)
    st, (book, Y, _) = _store(), _book()
    with pytest.raises(AssertionError, match="bypass"):
        gc.assert_request_only(st, book, Y[:8])


def test_assert_request_only_FIRES_when_soft_read_is_imported(monkeypatch):
    """⛔ Importing the feature-taking twin is caught **by name**, before use."""
    import chlu.core.gen_compose as gc

    monkeypatch.setattr(gc, "soft_read", soft_read, raising=False)
    st, (book, Y, _) = _store(), _book()
    with pytest.raises(AssertionError, match="soft_read"):
        gc.assert_request_only(st, book, Y[:8])


# ---------------------------------------------------------------------------
# 2. The composition path itself
# ---------------------------------------------------------------------------


def test_request_posterior_is_a_normalised_softmax_and_codes_are_its_argmax():
    book, Y, _ = _book()
    P = request_posterior(book, Y[:50])
    assert P.shape == (50, book.n_wells, book.n_modes)
    assert np.allclose(P.sum(axis=-1), 1.0, atol=1e-5)
    c = compose_codes(P)
    assert np.array_equal(c["code"], np.argmax(P, axis=-1))
    assert c["confidence"].shape == (50, book.n_wells)
    assert 0.0 <= c["mean_confidence"] <= 1.0


def test_the_read_arm_recovers_the_stores_own_modes_and_nothing_else():
    """⭐ The read is a pure codebook gather — the structural no-bypass guarantee."""
    st = _store()
    book, Y, _ = _book()
    codes = compose_codes(request_posterior(book, Y[:20]))["code"]
    y = np.asarray(read_compose(st, codes)).reshape(20, -1)
    for j, (w, m) in enumerate(zip(st.well_channels, st.modes, strict=True)):
        assert np.array_equal(y[:, list(w)], np.asarray(m)[codes[:, j]])


def test_store_removed_read_carries_ZERO_information():
    """⭐ The laundering control is demonstrably dead, not merely 'ablated'."""
    st = _store()
    book, Y, _ = _book()
    codes = compose_codes(request_posterior(book, Y[:24]))["code"]
    assert codes.std(axis=0).sum() > 0  # the codes really do vary across requests
    y = np.asarray(store_removed_read(st, codes, "mean")).reshape(24, -1)
    assert np.unique(y, axis=0).shape[0] == 1
    # ⛔ and it is NOT the same thing as the live read
    assert not np.allclose(y, np.asarray(read_compose(st, codes)).reshape(24, -1))


def test_clu_wide_lies_in_the_span_of_the_stores_modes():
    """⭐ CLU-wide is wider than 30 bits but still speaks the store's vocabulary."""
    st = _store()
    book, Y, _ = _book()
    P = request_posterior(book, Y[:12])
    wide = np.asarray(clu_wide_vector(st, P))
    for j, (w, m) in enumerate(zip(st.well_channels, st.modes, strict=True)):
        assert np.allclose(wide[:, list(w)], P[:, j, :] @ np.asarray(m), atol=1e-5)
    # a one-hot posterior degenerates EXACTLY to the read
    one_hot = np.zeros_like(P)
    codes = compose_codes(P)["code"]
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            one_hot[i, j, codes[i, j]] = 1.0
    assert np.allclose(
        np.asarray(clu_wide_vector(st, one_hot)),
        np.asarray(read_compose(st, codes)), atol=1e-6,
    )


def test_code_agreement_is_a_diagnostic_that_separates_agreement_from_disagreement():
    a = np.array([[1, 2, 3], [4, 5, 6]], np.int64)
    assert code_agreement(a, a)["exact_set"] == 1.0
    assert code_agreement(a, a + 1)["exact_set"] == 0.0
    half = a.copy()
    half[0, 0] = 9
    assert code_agreement(a, half)["exact_set"] == 0.5
    assert code_agreement(a, half)["per_well_agreement"][0] == 0.5


# ---------------------------------------------------------------------------
# 3. ⭐ The ledgers — the byte target REVERTS, and the channel is auditable
# ---------------------------------------------------------------------------


def test_deployed_byte_ledger_is_524288_with_the_coupling_EXCLUDED():
    """⛔ R26: dreaming is off the critical path, so its 163,840 B PMI field leaves
    the deployed store and ``gen-baselines``' 688,128 B target does NOT apply."""
    st = _shipped_store()
    led = deployed_byte_ledger(st)
    assert led["deployed_bytes"] == store_bytes(st)["store_bytes"]
    assert led["coupling_bytes_excluded"] == COUPLING_BYTES
    assert led["superseded_target_do_not_inherit"] == 688_128
    assert led["deployed_bytes"] + led["coupling_bytes_excluded"] == 688_128
    assert "R26" in led["coupling_excluded_reason"]


def test_the_shipped_cell_ledger_is_exactly_the_published_524288_bytes():
    st = _shipped_store()
    assert deployed_byte_ledger(st)["deployed_bytes"] == STORE_BYTES
    assert store_bytes(st)["bits_per_read"] == 30.0  # R20's design constant


def test_channel_ledger_keeps_byte_matched_and_channel_matched_apart():
    st = _shipped_store()
    book, Y, _ = _book(k=SHIPPED_K, m=SHIPPED_M, cards=(3, 4))
    chan = channel_ledger(st, request_posterior(book, Y[:8]), proto_dim=SHIPPED_D)
    assert chan["read"]["bits"] == 30.0
    assert chan["clu_wide"]["n_real_values"] == SHIPPED_K * SHIPPED_M
    assert chan["clu_wide"]["degrees_of_freedom"] == SHIPPED_K * (SHIPPED_M - 1)
    # ⛔ the prototype arm's channel is the whole feature vector
    assert chan["proto"]["n_real_values"] == SHIPPED_D
    assert chan["proto"]["bits_float32_upper_bound"] > chan["read"]["bits"]
    assert chan["reference_rows"]["knn_proto_in_corner_shape_balacc"] == 0.8884
    assert chan["reference_rows"]["store_ceiling_in_corner_shape_balacc"] == 0.6866


def test_the_addressing_head_is_reported_beside_the_store_never_inside_it():
    st = _store()
    book, _, _ = _book()
    led = deployed_byte_ledger(st, book)
    head = address_book_bytes(book)
    assert head["bytes"] > 0
    assert led["deployed_bytes"] == store_bytes(st)["store_bytes"]  # head excluded
    assert led["addressing_head"]["bytes"] == head["bytes"]


# ---------------------------------------------------------------------------
# 4. ⚠ The pre-registered metric invariance check (R26 standing rule)
# ---------------------------------------------------------------------------


def _probs(n_pos=20, n_neg=60, tpr=0.9, tnr=0.6, target=1, c=3, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.full((n_pos, c), 0.1)
    pos[: int(round(tpr * n_pos)), target] = 0.9
    pos[int(round(tpr * n_pos)):, (target + 1) % c] = 0.9
    neg = np.full((n_neg, c), 0.1)
    neg[: int(round(tnr * n_neg)), (target + 1) % c] = 0.9
    neg[int(round(tnr * n_neg)):, target] = 0.9
    pos += 1e-3 * rng.random(pos.shape)
    neg += 1e-3 * rng.random(neg.shape)
    return pos, neg, target


def test_metric_invariance_check_passes_for_the_gate_statistic():
    pos, neg, t = _probs()
    out = metric_invariance_check(pos, neg, t)
    assert out["invariant"]
    assert out["delta_balanced_accuracy"] < 1e-12
    assert out["delta_auroc"] < 1e-12


def test_metric_invariance_check_is_not_vacuous():
    """⭐ The re-partition must really change the class balance — otherwise the
    check passes for a statistic that was never at risk."""
    pos, neg, t = _probs(n_pos=40, n_neg=40, tpr=0.95, tnr=0.55)
    out = metric_invariance_check(pos, neg, t)
    assert out["repartitioned"]["n_neg"] == 2 * out["base"]["n_neg"]
    assert abs(out["delta_raw_accuracy"]) > 0.05     # ⛔ raw accuracy MOVES
    assert out["delta_balanced_accuracy"] == 0.0     # ⭐ the gate statistic does not


def test_metric_invariance_check_reports_its_monte_carlo_companion_separately():
    pos, neg, t = _probs()
    out = metric_invariance_check(pos, neg, t, subsample_frac=0.5, seed=3)
    assert out["subsample_leg"]["n_neg"] == 30
    assert "never the gate" in out["subsample_leg"]["note"]


# ---------------------------------------------------------------------------
# 5. ⭐ The spoke's own falsifier
# ---------------------------------------------------------------------------


def _arm_row(balacc, b_shape=0.5, a_agg=0.5):
    return {
        "in_corner_discrimination": {"balanced_accuracy": balacc, "auroc": balacc},
        "composition_A": {"aggregate": a_agg, "per_factor": {COMPOSITIONAL_FACTOR: b_shape}},
        "composition_B": {"aggregate": a_agg, "per_factor": {COMPOSITIONAL_FACTOR: b_shape}},
    }


def test_falsifier_reports_a_DEAD_read_as_dead():
    """⛔ read == store_removed is the outcome that kills the claim; it must be
    visible as a zero delta, not hidden inside an aggregate."""
    man = {"arms": {"read": _arm_row(0.5), "store_removed": _arm_row(0.5)}}
    f = seed_falsifier(man)
    assert f["available"] and f["delta_in_corner_balacc"] == 0.0


def test_falsifier_reports_a_live_read_as_live():
    man = {"arms": {"read": _arm_row(0.69, 0.8), "store_removed": _arm_row(0.50, 0.0)}}
    f = seed_falsifier(man)
    assert f["delta_in_corner_balacc"] == pytest.approx(0.19)
    assert f["delta_B_shape_accuracy"] == pytest.approx(0.8)


def test_falsifier_refuses_to_report_without_its_control():
    assert seed_falsifier({"arms": {"read": _arm_row(0.7)}})["available"] is False


# ---------------------------------------------------------------------------
# 6. Config wiring (the three house sites) + the decoder trap
# ---------------------------------------------------------------------------


def test_config_defaults_select_the_WINNING_decoder_recipe_not_the_baseline():
    """⛔ ``dec_recipe`` defaults to ``'baseline'`` — the decoder that FAILED the
    gate.  This spoke must select the winner explicitly."""
    g = get_default_config().gen_compose
    assert g.expected_decoder_recipe == "v1_capacity_schedule"
    assert g.decoder_source == "winner"
    assert g.expect_deployed_store_bytes == STORE_BYTES
    assert tuple(g.arms) == ARMS
    assert g.grouping_source == "per_seed"       # ⛔ never the plan's labels (R26.5)


def test_gen_compose_config_survives_a_save_load_roundtrip(tmp_path):
    config = get_default_config()
    config.gen_compose.n_eval_a = 77
    config.gen_compose.arms = ["read", "store_removed"]
    p = tmp_path / "config.yaml"
    save_config(config, p)
    back = load_config(p)
    assert back.gen_compose.n_eval_a == 77
    assert back.gen_compose.arms == ["read", "store_removed"]
    assert isinstance(back.gen_compose, GenComposeConfig)


def test_the_declared_reference_rows_are_the_ruled_ones():
    """⛔ R27: both rows go in every table — the byte-matched prototype arm and the
    channel-matched kNN — so neither can be quoted without the other."""
    g = get_default_config().gen_compose
    assert g.knn_proto_reference_balacc == 0.8884
    assert g.knn_channel_matched_reference_balacc == 0.5206
    assert g.store_ceiling_balacc == 0.6866
