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


# ---------------------------------------------------------------------------
# ⭐ 6. THE COMMITMENT DIAL (``gen-commitment-collapse``, R29 pivot item 1)
# ---------------------------------------------------------------------------
#
# ⭐ The load-bearing tests here are again the ones that can fail on purpose:
# ``test_sampling_buys_distinct_codes_with_NOISE`` shows the distinct-code
# statistic being trivially maximised by a rule that carries no information — which
# is exactly why the accuracy number must always be reported beside it — and
# ``test_multiplicity_EXCEEDS_the_30_bit_design_constant`` shows the channel ledger
# refusing to let a wider channel pass as a free win (R20).

from chlu.core.gen_compose import (  # noqa: E402
    ABSTAIN,
    code_overlap,
    commit,
    commitment_channel,
    commitment_read,
    distinct_codes,
    factor_evidence,
    posterior_diagnostics,
    temper_posterior,
)
from chlu.core.gen_store import read_from_codes  # noqa: E402
from chlu.experiments.exp_gen_compose import config_key, parse_config_key  # noqa: E402


def test_temperature_CANNOT_move_the_shipped_argmax_commitment():
    """⭐ P0, and it is analytic: ``softmax`` is monotone, so ``argmax`` is τ-blind.

    ⛔ If this ever fails, the tempering is buggy — it is **not** a finding about the
    dial.  The sweep measures this rather than assuming it.
    """
    book, Y, _ = _book(n=200)
    P = request_posterior(book, Y)
    ref = commit(P, "argmax", tau=1.0)
    for tau in (0.05, 0.25, 0.5, 2.0, 8.0, 64.0):
        got = commit(P, "argmax", tau=tau)
        assert np.array_equal(got["code"], ref["code"])
        assert got["n_distinct"] == ref["n_distinct"]


def test_commit_argmax_IS_the_shipped_compose_codes():
    """The baseline arm must be the shipped one, not a look-alike."""
    book, Y, _ = _book(n=128)
    P = request_posterior(book, Y)
    assert np.array_equal(commit(P, "argmax")["code"], compose_codes(P)["code"])
    assert commit(P, "argmax")["n_distinct"] == compose_codes(P)["n_distinct_code_configs"]


def test_pure_pointer_commitment_reads_through_the_SHIPPED_read_from_codes():
    """⛔ The no-bypass guarantee lives in ``read_from_codes``' signature, so the
    argmax/sample rules must route through it and not through a re-implementation."""
    st = _store()
    book, Y, _ = _book(n=64)
    P = request_posterior(book, Y)
    for rule in ("argmax", "sample"):
        com = commit(P, rule, seed=3)
        assert com["pure_pointer"]
        assert np.array_equal(
            np.asarray(commitment_read(st, com)),
            np.asarray(read_from_codes(st, com["code"])),
        )
    # k=1 multiplicity is the same object, shape included
    com1 = commit(P, "topk", k=1)
    assert com1["code"].shape == (Y.shape[0], st.n_wells)
    assert np.array_equal(com1["code"], commit(P, "argmax")["code"])


def test_margin_abstain_fills_a_well_with_its_MEAN_mode():
    """⭐ An abstained well is on-manifold ("not asked sharply enough"), ⛔ not zeroed —
    a zeroed well would score the deletion control instead of the refusal."""
    st = _store()
    book, Y, _ = _book(n=48)
    P = request_posterior(book, Y)
    com = commit(P, "margin", margin=1.01)          # nothing can clear it
    assert com["abstain_rate"] == 1.0
    assert (com["code"] == ABSTAIN).all()
    out = np.asarray(commitment_read(st, com)).reshape(Y.shape[0], -1)
    for j, (w, m) in enumerate(zip(st.well_channels, st.modes, strict=True)):
        mean_mode = np.asarray(m).mean(axis=0)
        assert np.allclose(out[:, list(w)], mean_mode[None, :], atol=1e-5)
    # ⛔ and a margin of 0 is the shipped argmax again
    zero = commit(P, "margin", margin=0.0)
    assert zero["abstain_rate"] == 0.0
    assert np.array_equal(zero["code"], commit(P, "argmax")["code"])


def test_multiplicity_EXCEEDS_the_30_bit_design_constant():
    """⛔ R20 made ``K·log2(M) = 30`` bits a DECLARED design constant: a rule that
    widens it is a declared change to the claim, and the ledger must say so."""
    st = _shipped_store()
    P = np.asarray(np.random.default_rng(0).dirichlet(np.ones(SHIPPED_M), size=(32, SHIPPED_K)))
    base = commitment_channel(st, commit(P, "argmax"))
    assert base["pointer_bits"] == pytest.approx(30.0)
    assert not base["exceeds_design_constant"]
    for k in (2, 4):
        ch = commitment_channel(st, commit(P, "topk", k=k))
        assert ch["pointer_bits"] > 30.0
        assert ch["exceeds_design_constant"]
    # abstention is a real extra symbol per well (M+1), and is reported as such
    ab = commitment_channel(st, commit(P, "margin", margin=0.99))
    assert ab["symbols_per_well"] == SHIPPED_M + 1
    assert ab["pointer_bits"] > 30.0


def test_sampling_buys_distinct_codes_with_NOISE():
    """⭐⭐ The failure mode the whole task is shaped to catch, demonstrated.

    On a posterior that commits every request to the SAME code, sampling produces
    many distinct codes while carrying **no** extra information about the request.
    ⛔ This is why a distinct count is never reported without its accuracy.
    """
    n, K, M = 72, 5, 64
    P = np.full((n, K, M), 0.002, np.float64)
    P[:, :, 7] = 1.0 - 0.002 * (M - 1)
    P /= P.sum(-1, keepdims=True)
    assert commit(P, "argmax")["n_distinct"] == 1          # the collapse, synthesised
    hot = commit(P, "sample", tau=8.0, seed=0)             # ⛔ pure noise
    assert hot["n_distinct"] > 40
    # …and the "information" is a function of the SEED, not of the request:
    other = commit(P, "sample", tau=8.0, seed=1)
    assert not np.array_equal(hot["code"], other["code"])


def test_posterior_diagnostics_flags_a_DEGENERATE_posterior():
    """⭐ The diagnosis leg's gate: if top-1 ≈ 1.0 no stopping rule can move the code,
    and the budget must not be spent on a sweep that cannot move."""
    n, K, M = 16, 4, 8
    hard = np.zeros((n, K, M)); hard[:, :, 0] = 1.0
    d = posterior_diagnostics(hard)
    assert d["degenerate"] and d["mean_entropy_bits_per_well"] == pytest.approx(0.0)
    soft = np.full((n, K, M), 1.0 / M)
    d2 = posterior_diagnostics(soft)
    assert not d2["degenerate"]
    assert d2["mean_entropy_bits_per_well"] == pytest.approx(np.log2(M))
    assert d2["mean_top1_minus_top2_margin"] == pytest.approx(0.0)


def test_factor_evidence_names_the_COMMON_MODE_factor():
    """⭐ The mechanical hypothesis, as an instrument that fires.

    A factor held CONSTANT across a request set contributes identical evidence to
    every request in it: large share, zero across-request variance.  ⛔ That is the
    shape of a colour corner drowning the factors that actually vary.
    """
    book, Y, _ = _book(n=300, cards=(3, 4))
    corner = Y[:64].copy()
    corner[:, 0] = 1                      # the "colour corner": factor 0 pinned
    ev = factor_evidence(book, corner)
    assert ev["across_request_std_per_factor"][0] == pytest.approx(0.0, abs=1e-12)
    assert ev["across_request_std_per_factor"][1] > 0.0
    assert ev["n_common_mode_factors"] == 1
    assert 0.0 < ev["common_mode_share"] < 1.0
    # ⛔ on a set where nothing is pinned, nothing is common mode
    assert factor_evidence(book, Y[:64])["n_common_mode_factors"] == 0


def test_code_overlap_counts_TUPLES_not_rows():
    a = np.array([[1, 2], [1, 2], [3, 4]])
    b = np.array([[3, 4], [5, 6]])
    ov = code_overlap(a, b)
    assert ov["n_distinct_a"] == 2 and ov["n_distinct_b"] == 2
    assert ov["n_intersection"] == 1
    assert ov["fraction_of_b_in_a"] == pytest.approx(0.5)


def test_distinct_codes_is_over_the_WHOLE_tuple():
    c = np.array([[1, 1], [1, 2], [1, 1]])
    assert distinct_codes(c) == 2
    assert distinct_codes(np.array([[[1, 2], [3, 4]], [[1, 2], [3, 5]]])) == 2


def test_temper_posterior_renormalises_and_refuses_nonsense():
    P = np.asarray(np.random.default_rng(0).dirichlet(np.ones(6), size=(5, 3)))
    Q = temper_posterior(P, 0.3)
    assert np.allclose(Q.sum(-1), 1.0)
    assert np.array_equal(np.argmax(Q, -1), np.argmax(P, -1))
    with pytest.raises(ValueError):
        temper_posterior(P, 0.0)


def test_config_key_roundtrips():
    for cc in ({"rule": "argmax", "tau": 1.0, "margin": 0.0, "k": 1},
               {"rule": "topk", "tau": 0.5, "margin": 0.0, "k": 4},
               {"rule": "margin", "tau": 2.0, "margin": 0.25, "k": 1}):
        assert parse_config_key(config_key(cc)) == cc
    with pytest.raises(ValueError):
        parse_config_key("nonsense|tau=1")


def test_factor_leverage_separates_OUTVOTED_from_ABSENT():
    """⭐⭐ The diagnosis instrument, shown to distinguish the two diseases.

    A factor whose conditionals are flat contributes nothing and can reach exactly
    one code on its own; a factor with real conditionals reaches several.  ⛔ Only
    this comparison tells "the address book never learned it" apart from "the other
    factors outvote it", and they have completely different fixes.
    """
    from chlu.core.gen_compose import factor_leverage

    rng = np.random.default_rng(0)
    n, k, m = 600, 4, 8
    cards = (4, 3)
    Y = np.stack([rng.integers(0, c, size=n) for c in cards], axis=1)
    # factor 0 determines every well's code; factor 1 is independent noise
    codes = np.stack([(Y[:, 0] * 2 + j) % m for j in range(k)], axis=1).astype(np.int64)
    book = fit_address_book(codes, Y, cards)
    lev = factor_leverage(book, Y[:64])["per_factor"]
    assert lev[0]["code_flip_rate"] > 0.9
    assert lev[0]["n_codes_from_factor_alone"] == 4
    assert lev[1]["code_flip_rate"] == pytest.approx(0.0)
    assert lev[1]["n_codes_from_factor_alone"] == 1
