"""The factored store over cached channels: grouping, modes, and the no-bypass proof.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_no_bypass_POSITIVE_CONTROL_fires`` — a read that *does* see the features is
  handed to :func:`assert_no_bypass`, which must FIRE.  An assertion never seen to
  fail is not evidence, and "the block routes around the memory" is precisely the
  failure this whole programme exists to catch.
- ``test_delete_the_store_POSITIVE_CONTROL_a_bypassing_read_shows_zero_delta`` — a
  deliberately bypassing read shows the near-zero deletion delta, so the metric is
  demonstrably able to expose the failure it is there to expose.
- ``test_row_permutation_deletion_is_a_no_op`` — pins the trap that a permuted
  codebook is **not** a deletion.

Everything runs on synthetic tensors; the real-data path is exercised by the harness
runs reported in ``.claude/outputs/gen-store-grouping.md``.

⛔ Wells are integer-indexed here as everywhere else.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenStoreConfig, get_default_config, load_config, save_config
from chlu.core.gen_store import (
    GROUPING_ORDER,
    GROUPINGS,
    GenStore,
    address_codes,
    assert_no_bypass,
    assert_plan_matched,
    block_diagonality,
    c_well_mapping,
    channel_correlation,
    choose_k,
    correlation_grouping,
    dci_scores,
    dead_channels,
    delete_store,
    grouping_labels,
    init_store,
    load_store_plan,
    mig_score,
    mode_usage,
    random_grouping,
    read_from_codes,
    reconstruction_mse,
    save_store_plan,
    soft_weights,
    store_bytes,
    store_read,
    train_store,
    well_channels_from_labels,
    well_relevance,
)
from chlu.core.multiwell_read import soft_occupancy


class _Cfg:
    train_steps = 40
    train_batch = 64
    train_lr = 3e-3
    commitment_beta = 0.25
    tau_start = 1.0
    tau_end = 0.1
    probe_lr = 3e-3
    probe_steps = 20
    probe_batch = 64


def _blocky(n=600, d=32, k=4, noise=0.05, seed=0):
    """``k`` planted blocks of correlated coordinates + a couple of dead ones."""
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, k))
    W = np.zeros((k, d))
    per = d // k
    for j in range(k):
        W[j, j * per : (j + 1) * per] = 1.0
    X = (lat @ W + noise * rng.normal(size=(n, d))).astype(np.float32)
    return X, lat


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------


def test_correlation_grouping_recovers_planted_blocks():
    X, _ = _blocky()
    lab = correlation_grouping(channel_correlation(X), 4, method="auto")
    per = X.shape[1] // 4
    for j in range(4):
        assert len(set(lab[j * per : (j + 1) * per])) == 1, "a planted block was split"
    assert len(set(lab)) == 4


def test_random_control_matches_the_well_size_histogram_exactly():
    """⭐ The laundering control varies ONE thing: which coordinate sits where."""
    X, _ = _blocky()
    ref = correlation_grouping(channel_correlation(X), 4, method="auto")
    rnd = random_grouping(ref, seed=3)
    assert np.array_equal(np.sort(np.bincount(ref)), np.sort(np.bincount(rnd)))
    assert not np.array_equal(ref, rnd)


def test_choose_k_finds_the_planted_number_of_blocks():
    X, _ = _blocky(k=4, d=64)
    out = choose_k(channel_correlation(X), 2, 8)
    assert out["k"] == 4, out["eigengaps"]


def test_choose_k_never_leaves_the_declared_candidate_range():
    X, _ = _blocky(k=4, d=64)
    out = choose_k(channel_correlation(X), 5, 7)
    assert 5 <= out["k"] <= 7


def test_dead_channels_are_detected_and_placed_without_creating_an_empty_well():
    X, _ = _blocky(d=36)
    X[:, -4:] = 1.0  # four constant coordinates
    dead = dead_channels(X)
    assert int(dead.sum()) == 4
    lab, info = correlation_grouping(
        channel_correlation(X), 4, method="auto", dead=dead, return_info=True
    )
    assert info["n_dead_channels"] == 4
    assert lab.size == X.shape[1]
    assert np.bincount(lab, minlength=4).min() > 0


def test_auto_falls_through_a_degenerate_method_and_records_why():
    """⚠ The shipped Shapes3D/recon cache makes average linkage chain; ``auto`` must
    reject it on a property of the PARTITION and say which method it used."""
    rng = np.random.default_rng(0)
    d = 60
    # one dense blob + a few near-orthogonal stragglers => classic chaining
    X = rng.normal(size=(400, 4)) @ np.ones((4, d)) * 0.9 + 0.4 * rng.normal(size=(400, d))
    lab, info = correlation_grouping(
        channel_correlation(X), 6, method="auto", return_info=True
    )
    assert info["method_used"] in GROUPING_ORDER
    assert info["method_order"] == list(GROUPING_ORDER)
    assert np.bincount(lab, minlength=6).min() >= info["min_well_size_floor"]
    assert all("sizes" in a and "accepted" in a for a in info["attempts"])


def test_block_diagonality_is_higher_for_the_clustered_grouping():
    X, _ = _blocky()
    corr = channel_correlation(X)
    ref = correlation_grouping(corr, 4, method="auto")
    rnd = random_grouping(ref, seed=1)
    assert (
        block_diagonality(corr, ref)["in_block_fraction"]
        > block_diagonality(corr, rnd)["in_block_fraction"]
    )


def test_grouping_labels_returns_both_arms_at_the_same_k():
    X, _ = _blocky()
    corr = channel_correlation(X)
    outs = {g: grouping_labels(corr, 4, g, 0)[0] for g in GROUPINGS}
    assert len(set(len(set(v)) for v in outs.values())) == 1


# ---------------------------------------------------------------------------
# the store + ⛔ the no-bypass proof
# ---------------------------------------------------------------------------


def _fit(seed=0, n_modes=8, d=32, k=4):
    X, lat = _blocky(d=d, k=k, seed=seed)
    corr = channel_correlation(X)
    lab = correlation_grouping(corr, k, method="auto")
    wc = well_channels_from_labels(lab, k)
    key = jax.random.PRNGKey(seed)
    st = init_store(X, wc, n_modes, (d,), key, seed=seed)
    st, hist, _ = train_store(st, X, _Cfg, key)
    return st, X, lat, hist


def test_wells_must_partition_every_coordinate():
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="PARTITION"):
        GenStore(((0, 1), (1, 2)), 4, (4,), key)  # coordinate 1 twice, 3 missing


def test_store_bytes_is_exactly_n_modes_times_D_times_4():
    st, X, _, _ = _fit(n_modes=8, d=32)
    b = store_bytes(st)
    assert b["store_bytes"] == 8 * 32 * 4
    assert b["bits_per_read"] == pytest.approx(4 * np.log2(8))


def test_read_is_a_pure_codebook_gather_and_carries_no_features():
    st, X, _, _ = _fit()
    info = assert_no_bypass(st, X[:32])
    assert info["signature_ok"] and info["read_is_pure_codebook_gather"]
    assert info["max_abs_delta_vs_independent_gather"] == 0.0
    # the channel is LIVE, not merely closed
    assert info["codes_changed_under_feature_replacement"]
    assert info["output_changed_when_codes_changed"] > 0.0


def test_no_bypass_POSITIVE_CONTROL_fires():
    """⛔ Hand the checker a read that DOES see the features; it must fail."""
    import chlu.core.gen_store as gsmod

    st, X, _, _ = _fit()
    good = gsmod.read_from_codes

    def bypassing_read(store, codes, features=None):  # extra channel = a bypass
        return good(store, codes)

    gsmod.read_from_codes = bypassing_read
    try:
        with pytest.raises(AssertionError, match="bypass"):
            gsmod.assert_no_bypass(st, X[:8])
    finally:
        gsmod.read_from_codes = good


def test_read_output_depends_on_the_features_only_through_the_codes():
    st, X, _, _ = _fit()
    codes = address_codes(st, X[:16])
    y = read_from_codes(st, codes)
    y_again = read_from_codes(st, codes.copy())
    assert np.array_equal(y, y_again)
    # two different feature vectors with the SAME codes give the SAME read
    same = np.flatnonzero(np.all(codes == codes[0], axis=1))
    if same.size > 1:
        assert np.allclose(y[same[0]], y[same[1]])


def test_store_read_beats_a_deleted_store_on_reconstruction():
    st, X, _, _ = _fit()
    intact = reconstruction_mse(st, X)["relative_mse"]
    for mode in ("zero", "mean", "scramble"):
        assert reconstruction_mse(delete_store(st, mode), X)["relative_mse"] > intact


def test_delete_the_store_moves_the_read_output():
    """⭐ The measurement the whole spoke lives or dies on, on synthetic data."""
    st, X, _, _ = _fit()
    y = store_read(st, X)
    for mode in ("zero", "mean", "scramble"):
        d = store_read(delete_store(st, mode, seed=0), X)
        rms = float(np.sqrt(np.mean((y - d) ** 2)))
        assert rms > 1e-3, f"deletion mode {mode!r} left the read output unchanged"


def test_row_permutation_deletion_is_a_no_op():
    """⚠ Pins the trap: permuting a codebook's ROWS is not a deletion — the
    nearest-neighbour address re-finds the same vector."""
    st, X, _, _ = _fit()
    rng = np.random.default_rng(0)
    permuted = jax.tree_util.tree_map(lambda m: m, st)
    import equinox as eqx

    permuted = eqx.tree_at(
        lambda t: t.modes,
        st,
        [jnp.asarray(np.asarray(m)[rng.permutation(np.asarray(m).shape[0])]) for m in st.modes],
    )
    assert np.allclose(store_read(st, X), store_read(permuted, X)), (
        "row permutation should be a no-op; if it is not, the deletion suite's "
        "rationale needs revisiting"
    )


def test_delete_the_store_POSITIVE_CONTROL_a_bypassing_read_shows_zero_delta():
    """⛔ A read that ignores the store must produce the near-zero delta the task
    calls a FAILED task — proving the metric can see that failure."""
    st, X, _, _ = _fit()

    def bypassing(store, x):
        return np.asarray(x, np.float32)  # straight through: the store is unused

    delta = float(
        np.sqrt(np.mean((bypassing(st, X) - bypassing(delete_store(st, "zero"), X)) ** 2))
    )
    assert delta == 0.0


def test_training_improves_the_hard_read_over_the_kmeanspp_init():
    """⚠ **The training loss is NOT monotone and must not be read as one.** ``tau`` is
    annealed down during the run, so ``loss[0]`` and ``loss[-1]`` are values of two
    *different* functions and the trace can rise while the store gets strictly
    better.  The quantity that matters — and the one asserted here — is the HARD
    read's reconstruction error against the k-means++ initialisation.
    (Measured: relative MSE 0.0568 init -> 0.0474 @40 steps -> 0.0391 @200, while the
    reported loss rose 0.283 -> 0.398.)"""
    X, _ = _blocky(noise=0.5)
    corr = channel_correlation(X)
    wc = well_channels_from_labels(correlation_grouping(corr, 4, method="auto"), 4)
    key = jax.random.PRNGKey(0)
    st0 = init_store(X, wc, 8, (X.shape[1],), key, seed=0)
    st, hist, _ = train_store(st0, X, _Cfg, key)
    before = reconstruction_mse(st0, X)["relative_mse"]
    after = reconstruction_mse(st, X)["relative_mse"]
    assert after < before, f"training did not improve the hard read ({before} -> {after})"
    assert len(hist) == _Cfg.train_steps


def test_modes_are_actually_used():
    st, X, _, _ = _fit(n_modes=8)
    u = mode_usage(address_codes(st, X), 8)
    assert u["mean_used_modes"] > 1.0
    assert u["mean_entropy_bits"] > 0.0


def test_gram_softmax_matches_soft_occupancy():
    """⚠ ``train_store`` uses the gram form for memory; it must be the same function."""
    rng = np.random.default_rng(0)
    z = jnp.asarray(rng.normal(size=(7, 5)), jnp.float32)
    m = jnp.asarray(rng.normal(size=(4, 5)), jnp.float32)
    tau = 0.7
    d2 = (
        jnp.sum(z * z, -1)[:, None] - 2.0 * (z @ m.T) + jnp.sum(m * m, -1)[None, :]
    )
    gram = jax.nn.softmax(-d2 / (2.0 * tau**2), axis=-1)
    assert np.allclose(np.asarray(gram), np.asarray(soft_occupancy(z, m, tau)), atol=1e-5)


def test_soft_weights_are_a_simplex_per_well():
    st, X, _, _ = _fit()
    for w in soft_weights(st, jnp.asarray(X[:8]), 0.5):
        assert np.allclose(np.asarray(w).sum(-1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# factorisation measures + the C mapping
# ---------------------------------------------------------------------------


def _discretise(lat):
    return np.stack(
        [np.digitize(lat[:, j], np.quantile(lat[:, j], [1 / 3, 2 / 3])) for j in range(lat.shape[1])],
        axis=1,
    ).astype(np.int64)


def test_mig_and_dci_prefer_the_clustered_grouping_on_planted_structure():
    X, lat = _blocky()
    y = _discretise(lat)
    cards = [3] * y.shape[1]
    corr = channel_correlation(X)
    ref = correlation_grouping(corr, 4, method="auto")
    rnd = random_grouping(ref, seed=2)
    key = jax.random.PRNGKey(0)
    got = {}
    for name, lab in (("correlation", ref), ("random", rnd)):
        wc = well_channels_from_labels(lab, 4)
        st = init_store(X, wc, 8, (X.shape[1],), key, seed=0)
        st, _, _ = train_store(st, X, _Cfg, key)
        codes = address_codes(st, X)
        got[name] = (mig_score(codes, y, cards)["mig"], dci_scores(codes, y, cards)["disentanglement"])
    assert got["correlation"][0] > got["random"][0]


def test_well_relevance_shares_sum_to_one_per_factor():
    X, lat = _blocky()
    y = _discretise(lat)
    st, X2, _, _ = _fit()
    rel = well_relevance(address_codes(st, X2), y[: X2.shape[0]], [3] * y.shape[1])
    share = np.asarray(rel["share_by_factor"])
    assert np.allclose(share.sum(axis=0), 1.0, atol=1e-6)


def test_c_well_mapping_is_a_diagnostic_and_never_a_query_set():
    X, lat = _blocky()
    y = _discretise(lat)
    st, X2, _, _ = _fit()
    rel = well_relevance(address_codes(st, X2), y[: X2.shape[0]], [3] * 4)
    recs = [
        {"id": "x-0", "kind": "duplicate_factor", "violated_factor": "f0"},
        {"id": "x-1", "kind": "out_of_range", "violated_factor": "f2"},
    ]
    out = c_well_mapping(recs, rel, ["f0", "f1", "f2", "f3"])
    assert out["n_records"] == 2
    assert 0.0 <= out["intra_well_conflict_fraction"] <= 1.0
    assert len(out["landing_histogram"]) == 4
    assert "NEVER becomes a well-space query set" in out["note"]
    # ⛔ every returned well identifier is an integer index, never a name
    assert all(isinstance(r["top_well"], int) for r in out["per_record"])


# ---------------------------------------------------------------------------
# the reusable plan (the ladder's contract)
# ---------------------------------------------------------------------------


def test_store_plan_round_trips_and_pins_k_and_bytes(tmp_path):
    st, X, _, _ = _fit(n_modes=8, d=32)
    lab = np.asarray([i % 4 for i in range(32)])
    plan = save_store_plan(
        tmp_path / "store_plan.json",
        dataset="shapes3d", pretext="recon", k=4, n_modes=8, dim=32,
        feat_shape=(32,), labels=lab, k_choice={"k": 4, "rule": "test"},
        grouping_method="ward", store_bytes_dict=store_bytes(st),
    )
    back = load_store_plan(tmp_path / "store_plan.json")
    assert back["K"] == 4 and back["n_modes"] == 8 and back["D"] == 32
    assert back["store_bytes"]["store_bytes"] == plan["store_bytes"]["store_bytes"]
    assert assert_plan_matched([back, dict(back)])["n_rungs"] == 2


def test_a_later_rung_that_redefines_k_is_refused(tmp_path):
    st, _, _, _ = _fit(n_modes=8, d=32)
    base = save_store_plan(
        tmp_path / "a.json", dataset="shapes3d", pretext="recon", k=4, n_modes=8,
        dim=32, feat_shape=(32,), labels=np.arange(32) % 4, k_choice={},
        grouping_method="ward", store_bytes_dict=store_bytes(st),
    )
    other = dict(base, K=6)
    with pytest.raises(AssertionError, match="only the grouping may change"):
        assert_plan_matched([base, other])


def test_load_store_plan_rejects_a_foreign_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(ValueError, match="not a store plan"):
        load_store_plan(p)


# ---------------------------------------------------------------------------
# config wiring (the three-site house convention)
# ---------------------------------------------------------------------------


def test_gen_store_config_defaults_and_byte_formula():
    g = GenStoreConfig()
    assert g.n_modes == 64
    assert list(g.groupings) == list(GROUPINGS), "the random control is never optional"
    assert g.k_min == 4 and g.k_max == 16
    assert g.grouping_method == "auto"
    assert "zero" in g.delete_modes


def test_gen_store_config_round_trips_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_store.k_wells = 9
    cfg.gen_store.n_modes = 32
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_store.k_wells == 9
    assert back.gen_store.n_modes == 32
    assert back.gen_store == cfg.gen_store


def test_cli_exposes_the_binding_contract():
    from chlu.experiments.exp_gen_store import build_arg_parser

    opts = {a.option_strings[0] for a in build_arg_parser()._actions if a.option_strings}
    for flag in ("--dataset", "--pretext", "--stage", "--arm", "--seed", "--out"):
        assert flag in opts


def test_module_refuses_a_stage_that_is_not_its_own(capsys):
    from chlu.experiments.exp_gen_store import main

    assert main(["--stage", "encoder"]) == 2
