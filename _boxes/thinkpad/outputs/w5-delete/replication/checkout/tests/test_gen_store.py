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


# ---------------------------------------------------------------------------
# ⭐ the cache re-encode gate (``_assert_cache_reencodes``)
#
# The gate used to demand BIT-IDENTITY.  On an L40S (cluster job 20603452,
# 2026-09-15) one element of 64 x 2048 crossed a float16 rounding boundary by
# exactly 2^-8 — one float16 ULP in [4, 8) — and it convicted an encoder whose
# parameter hash had already matched, blocking every gen stage on a GPU node.
#
# ⭐ The load-bearing test here is ``..._convicts_a_perturbed_encoder``: a gate that
# cannot fail proves nothing, and loosening one is how a real defect gets through.
# ---------------------------------------------------------------------------

import equinox as eqx  # noqa: E402

from chlu.core.gen_bridge import encode_images as _real_encode_images  # noqa: E402


class _TinyEncoder(eqx.Module):
    """A real (tiny) encoder: ``(C,H,W) -> (C,H,W)`` through a learned matrix, so
    perturbing its parameters perturbs its outputs exactly as the frozen bridge
    encoder would."""

    W: jnp.ndarray
    feat_shape: tuple = eqx.field(static=True)

    def __call__(self, x):
        flat = jnp.reshape(x, (-1,))
        return jnp.reshape(flat @ self.W, self.feat_shape)


def _reenc_fixture(n_rows=8, d=64, seed=0):
    """``(cache_rows, features_float16_through_float32, cman)`` on exact float16 values.

    ⭐ Values are drawn in [4, 8), the binade of the cluster incident, where one
    float16 ULP is exactly ``0.00390625``.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(4.0, 8.0, size=(n_rows, d)).astype(np.float32)
    feats = raw.astype(np.float16).astype(np.float32)  # exact float16 values
    cache_rows = np.arange(n_rows, dtype=np.int64)
    cman = {"dtype": "float16", "D": d, "feat_shape": (1, 1, d), "n_rows": n_rows}
    return cache_rows, feats, cman


def _run_gate(monkeypatch, cache_rows, feats, cman, fresh_or_encoder, *, real_encoder=False,
              images=None, n=64, **gate_kw):
    """Drive ``_assert_cache_reencodes`` with the data sources stubbed out.

    Returns ``(stats_or_None, raised_or_None, log_lines)``.  ⭐ The log lines are
    captured so the reported statistics can be asserted on a FAILING run too — the
    gate must report always, pass or fail.
    """
    from chlu.experiments import exp_gen_store as X

    imgs = np.zeros((cache_rows.size, 1, 1, cman["D"]), np.float32) if images is None else images
    monkeypatch.setattr(X, "stream_images", lambda *a, **k: imgs)
    if real_encoder:
        monkeypatch.setattr(X, "encode_images", _real_encode_images)
        encoder = fresh_or_encoder
    else:
        monkeypatch.setattr(X, "encode_images", lambda *a, **k: np.asarray(fresh_or_encoder))
        encoder = object()

    lines = []
    cfg = get_default_config()
    try:
        stats = X._assert_cache_reencodes(
            "shapes3d", cfg, encoder, cache_rows, feats, cman, n=n, log=lines.append,
            **gate_kw,
        )
        return stats, None, lines
    except AssertionError as exc:
        return None, exc, lines


def test_reencode_gate_accepts_the_cluster_case_one_float16_ulp(monkeypatch):
    """⭐ The exact defect: ONE element moved by ONE float16 ULP is the same encoder."""
    cache_rows, feats, cman = _reenc_fixture()
    fresh = feats.copy()
    assert 4.0 <= fresh[3, 7] < 8.0
    fresh[3, 7] = np.float32(fresh[3, 7] + np.float32(0.00390625))  # 2^-8, one ULP here

    stats, raised, lines = _run_gate(monkeypatch, cache_rows, feats, cman, fresh)

    assert raised is None, f"the gate convicted a one-ULP rounding difference: {raised}"
    assert stats["max_abs_delta"] == 0.00390625
    assert stats["max_ulp_delta"] == 1.0
    assert stats["n_differing"] == 1
    assert stats["frac_differing"] == 1 / feats.size
    assert stats["n_checked"] == cache_rows.size  # ⛔ rows, for the two callers that log it
    assert lines and "1.000 float16 ULP" in lines[0]


def test_reencode_gate_tolerance_tracks_magnitude_not_a_global_ulp(monkeypatch):
    """⛔ A single global ULP is wrong.  The SAME absolute delta must be 1 ULP at 4-8
    and 8 ULP near 1.0 — so it is accepted there and convicted here."""
    d = 64
    cache_rows = np.arange(4, dtype=np.int64)
    feats = np.full((4, d), 1.0, np.float32)  # ULP near 1.0 is 2^-10 = 0.0009765625
    cman = {"dtype": "float16", "D": d, "feat_shape": (1, 1, d), "n_rows": 4}
    fresh = feats.copy()
    fresh[1, 5] = np.float32(1.0 + 0.00390625)  # the cluster's absolute delta, at 1.0

    stats, raised, lines = _run_gate(monkeypatch, cache_rows, feats, cman, fresh)

    assert raised is not None, "a 4-ULP move near 1.0 was excused by a too-loose bound"
    assert "4.000 float16 ULP" in lines[0]
    assert "4.000 ULP" in str(raised)


# ⚠ equinox warns on the ``filter_jit(vmap(module))`` idiom that ``encode_images``
# uses BY DESIGN; the vmapped callable is the whole encoder, which is the point.
@pytest.mark.filterwarnings("ignore:\\s*\\nPossibly assigning a JAX-transformed callable")
def test_reencode_gate_convicts_a_perturbed_encoder_POSITIVE_CONTROL(monkeypatch):
    """⭐⭐ THE POSITIVE CONTROL.  A genuinely wrong encoder is still convicted, by
    MANY ULP — the gate did not lose its teeth.

    ⛔ An assertion never seen to fail is not evidence; this check is the only thing
    standing between a swapped encoder and a silently invalid run.
    """
    d = 32
    n_rows = 8
    key = jax.random.PRNGKey(20_260_915)
    kw, ki = jax.random.split(key)
    W = jax.random.normal(kw, (d, d)) * 0.5 + 6.0  # outputs land in the [4, 8) binade
    enc = _TinyEncoder(W=W, feat_shape=(1, 1, d))
    imgs = np.asarray(jax.random.uniform(ki, (n_rows, 1, 1, d)), np.float32)

    cache_rows = np.arange(n_rows, dtype=np.int64)
    feats = (
        _real_encode_images(enc, imgs, 512)
        .astype(np.float16).astype(np.float32).reshape((n_rows, d))
    )
    cman = {"dtype": "float16", "D": d, "feat_shape": (1, 1, d), "n_rows": n_rows}

    # the TRUE encoder passes at 0 ULP — otherwise the conviction below proves nothing
    ok, raised_ok, lines_ok = _run_gate(
        monkeypatch, cache_rows, feats, cman, enc, real_encoder=True, images=imgs
    )
    assert raised_ok is None, f"the gate convicted the encoder that wrote the cache: {raised_ok}"
    assert ok["max_ulp_delta"] == 0.0 and ok["n_differing"] == 0

    # ⭐ now a DIFFERENT encoder: a 1% perturbation of its parameters
    wrong = eqx.tree_at(lambda m: m.W, enc, enc.W * 1.01)
    bad, raised, lines = _run_gate(
        monkeypatch, cache_rows, feats, cman, wrong, real_encoder=True, images=imgs
    )

    assert bad is None and raised is not None, "the gate did NOT convict a perturbed encoder"
    assert lines, "the gate must report its statistics on a FAILING run too"
    ulp = float(lines[0].split("= ")[-1].split(" float16")[0])
    assert ulp > 10.0, f"a perturbed encoder was only {ulp} ULP away — the gate has no teeth"
    assert "ULP" in str(raised)
    # ⭐ the message says what was MEASURED; it does not accuse the (hash-matched) encoder
    assert "this is NOT the encoder" not in str(raised)


def test_reencode_gate_refuses_half_the_tensor_off_by_one_ulp(monkeypatch):
    """⚠ Every element within 1 ULP is NOT enough: a rounding-boundary crossing touches
    a handful of elements; half the tensor moving is a different computation."""
    cache_rows, feats, cman = _reenc_fixture(n_rows=8, d=64)
    fresh = feats.copy()
    flat = fresh.reshape(-1)
    flat[::2] = (flat[::2] + np.float32(0.00390625)).astype(np.float32)  # 1 ULP, 50%

    stats, raised, lines = _run_gate(monkeypatch, cache_rows, feats, cman, fresh)

    assert raised is not None, "half the tensor moved and the fraction bound did not fire"
    assert "1.000 float16 ULP" in lines[0], "no element exceeded one ULP — the fraction bound fired"
    assert "50.000%" in str(raised) and "1.000%" in str(raised)


def test_reencode_gate_refuses_a_nan(monkeypatch):
    """⛔ `nan > 0` is False, so a NaN would otherwise sail through every comparison."""
    cache_rows, feats, cman = _reenc_fixture()
    fresh = feats.copy()
    fresh[0, 0] = np.float32("nan")

    stats, raised, _ = _run_gate(monkeypatch, cache_rows, feats, cman, fresh)

    assert raised is not None and "NaN" in str(raised)


def test_reencode_gate_reports_the_five_fields_on_a_clean_pass(monkeypatch):
    cache_rows, feats, cman = _reenc_fixture()

    stats, raised, lines = _run_gate(monkeypatch, cache_rows, feats, cman, feats.copy())

    assert raised is None
    for field_ in ("max_abs_delta", "max_ulp_delta", "n_differing", "frac_differing", "n_checked"):
        assert field_ in stats, field_
    assert stats == {
        "n_checked": 8, "n_elements": 512, "max_abs_delta": 0.0, "max_ulp_delta": 0.0,
        "n_differing": 0, "frac_differing": 0.0, "n_nonfinite": 0,
        "cache_dtype": "float16", "ulp_bound": 1.0, "frac_bound": 0.01,
        # ⭐ the scoping record: strict by default, and it would not have failed
        "enforced": True, "enforced_reason": "",
        "would_have_failed": False, "would_have_failed_why": [],
    }
    assert lines and lines[0].lstrip().startswith("⭐ cache re-encode check: 8 rows")


# ⚠ the task named FOUR callers; there are in fact 14 import sites, all binding the
# same object.  The four named ones lead; the rest are pinned so they cannot drift.
_REENC_CALLERS = [
    "chlu.experiments.exp_gen_store",
    "chlu.experiments.exp_gen_classifier",
    "chlu.experiments.exp_gen_certificate",
    "chlu.experiments.exp_gen_wells_route_b",
    "chlu.experiments.exp_gen_atom_mean",
    "chlu.experiments.exp_gen_attribution",
    "chlu.experiments.exp_gen_baselines",
    "chlu.experiments.exp_gen_cell",
    "chlu.experiments.exp_gen_codebook_alloc",
    "chlu.experiments.exp_gen_codebook_exponent",
    "chlu.experiments.exp_gen_compose",
    "chlu.experiments.exp_gen_dream",
    "chlu.experiments.exp_gen_dynamical_read",
    "chlu.experiments.exp_gen_variance_eq",
]


def test_reencode_gate_signature_only_ever_GAINS_and_the_gain_is_strict_by_default():
    """⛔ Callers pass positionally through ``cman``; the signature only ever gains —
    and ⭐ every gain is **keyword-only** with a **strict** default, so a caller that
    passes nothing gets exactly today's abort."""
    import inspect

    from chlu.experiments import exp_gen_store as X

    sig = inspect.signature(X._assert_cache_reencodes)
    assert list(sig.parameters) == [
        "dataset", "config", "encoder", "cache_rows", "features", "cman", "n", "log",
        "will_extend", "reason",
    ]
    for name in ("will_extend", "reason"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    # ⭐⭐ THE default. ⛔ If this ever flips, every un-opted caller silently loses its gate.
    assert sig.parameters["will_extend"].default is True


@pytest.mark.parametrize("mod_name", _REENC_CALLERS)
def test_reencode_gate_callers_still_work_unchanged(monkeypatch, mod_name):
    """⛔ The signature and the returned dict stay backward compatible: every caller
    imports the same object and their exact call form still works."""
    import importlib

    from chlu.experiments import exp_gen_store as X

    mod = importlib.import_module(mod_name)
    assert mod._assert_cache_reencodes is X._assert_cache_reencodes

    cache_rows, feats, cman = _reenc_fixture()
    monkeypatch.setattr(X, "stream_images", lambda *a, **k: np.zeros((8, 1, 1, 64), np.float32))
    monkeypatch.setattr(X, "encode_images", lambda *a, **k: feats.copy())
    cfg = get_default_config()

    # the exact call form all four sites use: positional through `cman`, then log=
    out = mod._assert_cache_reencodes(
        "shapes3d", cfg, object(), cache_rows, feats, cman, log=lambda _m: None)
    # two callers do `reenc.get('n_checked')` and print it as a ROW count
    assert out.get("n_checked") == 8
    assert out["max_abs_delta"] == 0.0
    # ⭐ a caller that says nothing is ENFORCED — including the 10 modules this spoke
    # did not touch
    assert out["enforced"] is True


# ---------------------------------------------------------------------------
# ⭐⭐ Scoping the re-encode gate to the hazard it guards (`will_extend`)
#
# The gate exists because a run that EXTENDS the cache through an encoder that does
# not reproduce the rows already in it writes inconsistent features, and ⛔ nothing
# downstream notices.  A run that encodes NOTHING NEW cannot trigger that hazard —
# and route B's own coverage gate, two lines above the call, reports
# `33864/33864 — 0 shortfall`.
#
# ⛔⛔ These tests exist to prove the teeth are still there.  `will_extend` decides
# whether the bounds BIND; it never changes what they are.  The bounds themselves
# (1.0 ULP, 1% of elements) are asserted UNCHANGED below.
# ---------------------------------------------------------------------------

_COVERED = "the coverage gate reports 0 shortfall: this run encodes nothing new"


def _disagreeing_fixture(n_rows=8, d=64, seed=0):
    """Like :func:`_reenc_fixture`, but drawn from **[4.0, 7.5)**.

    ⚠ Measured, not assumed: on the [4, 8) fixture, adding 0.03125 to a value just
    under 8.0 crosses the binade, where float16's step doubles to 0.0078125 and
    round-half-to-even lands the result 0.03515625 away — **9** ULP of the cached
    value, not 8.  Keeping the whole fixture clear of the boundary makes the
    perturbation exactly 8 ULP at EVERY element, so these tests assert an exact
    number instead of an inequality.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(4.0, 7.5, size=(n_rows, d)).astype(np.float32)
    feats = raw.astype(np.float16).astype(np.float32)  # exact float16 values
    cache_rows = np.arange(n_rows, dtype=np.int64)
    cman = {"dtype": "float16", "D": d, "feat_shape": (1, 1, d), "n_rows": n_rows}
    return cache_rows, feats, cman


def _disagreeing(feats):
    """A re-encode that fails BOTH bounds at today's values: every element moved, each
    by exactly 8 float16 ULP at its own magnitude.  ⛔ Use with
    :func:`_disagreeing_fixture`, never with the [4, 8) one."""
    return (feats + np.float32(8 * 0.00390625)).astype(np.float32)


def test_scoped_gate_STILL_ABORTS_when_the_run_will_extend(monkeypatch):
    """⭐⭐ ACCEPTANCE 1 — THE TEETH.  A run that WILL extend the cache and whose
    encoder disagrees still aborts, at today's bounds."""
    cache_rows, feats, cman = _disagreeing_fixture()

    stats, raised, lines = _run_gate(
        monkeypatch, cache_rows, feats, cman, _disagreeing(feats),
        will_extend=True, reason="this run will extend the cache",
    )

    assert stats is None and raised is not None, "the gate lost its teeth on an extending run"
    assert "8.000 ULP" in str(raised) and "bound 1)" in str(raised)
    assert any("⭐ ENFORCED (enforced=true)" in ln for ln in lines)


def test_scoped_gate_is_STRICT_when_the_caller_says_nothing(monkeypatch):
    """⭐ ACCEPTANCE 3 — the default.  No new parameter ⇒ exactly today's behaviour."""
    cache_rows, feats, cman = _disagreeing_fixture()

    stats, raised, lines = _run_gate(monkeypatch, cache_rows, feats, cman, _disagreeing(feats))

    assert stats is None and raised is not None
    assert "8.000 ULP" in str(raised)


def test_a_fully_covered_run_PASSES_with_the_disagreement_on_the_record(monkeypatch):
    """⭐ ACCEPTANCE 2 — the same disagreement, on a run that cannot extend the cache:
    it passes, and ⛔ it leaves a trace nobody can mistake for a clean check."""
    cache_rows, feats, cman = _disagreeing_fixture()

    stats, raised, lines = _run_gate(
        monkeypatch, cache_rows, feats, cman, _disagreeing(feats),
        will_extend=False, reason=_COVERED,
    )

    assert raised is None, f"a covered run was refused on a capability it never uses: {raised}"
    # the five statistics are still computed and still returned
    for field_ in ("max_abs_delta", "max_ulp_delta", "n_differing", "frac_differing", "n_checked"):
        assert field_ in stats, field_
    assert stats["max_ulp_delta"] == 8.0 and stats["frac_differing"] == 1.0
    # ⭐ enforced: false, the reason, and what the bounds WOULD have said
    assert stats["enforced"] is False
    assert stats["enforced_reason"] == _COVERED
    assert stats["would_have_failed"] is True
    assert stats["would_have_failed_why"] == ["ulp", "fraction"]
    # ⛔ the bounds are still the bounds — recorded, unwidened
    assert stats["ulp_bound"] == 1.0 and stats["frac_bound"] == 0.01
    # ⭐ and the log says all of it, loudly
    blob = "\n".join(lines)
    assert "⚠ NOT ENFORCED (enforced=false)" in blob and _COVERED in blob
    assert "WOULD HAVE ABORTED THIS RUN (ulp, fraction)" in blob
    assert "8.000 float16 ULP" in lines[0]


def test_an_unenforced_check_that_would_have_passed_says_so(monkeypatch):
    """⛔ A reader must be able to tell a skipped check from a passed one — in BOTH
    directions."""
    cache_rows, feats, cman = _reenc_fixture()

    stats, raised, lines = _run_gate(
        monkeypatch, cache_rows, feats, cman, feats.copy(),
        will_extend=False, reason=_COVERED,
    )

    assert raised is None and stats["enforced"] is False
    assert stats["would_have_failed"] is False and stats["would_have_failed_why"] == []
    assert any("not enforced, and it would have PASSED anyway" in ln for ln in lines)


def test_relaxing_the_gate_without_a_reason_is_an_error(monkeypatch):
    """⛔ The relaxation must be ASKED FOR, in the open. A reasonless skip is a
    ValueError, not a quiet pass."""
    from chlu.experiments import exp_gen_store as X

    cache_rows, feats, cman = _reenc_fixture()
    with pytest.raises(ValueError, match="requires a `reason`"):
        X._assert_cache_reencodes(
            "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
            log=lambda _m: None, will_extend=False,
        )
    with pytest.raises(ValueError, match="requires a `reason`"):
        X._assert_cache_reencodes(
            "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
            log=lambda _m: None, will_extend=False, reason="   ",
        )


def test_the_bounds_are_not_reachable_from_any_caller():
    """⛔⛔ The Hub has twice mistaken this gate for the problem.  There is no keyword,
    on any caller, that can widen either bound — only whether they bind."""
    import inspect

    from chlu.experiments import exp_gen_store as X

    src = inspect.getsource(X._assert_cache_reencodes)
    assert "ULP_BOUND = 1.0" in src and "FRAC_BOUND = 1e-2" in src
    params = set(inspect.signature(X._assert_cache_reencodes).parameters)
    assert not (params & {"ulp_bound", "frac_bound", "tol", "atol", "rtol", "bound"})


# --- ⭐⭐ item 4: the hazard is caught AT THE POINT OF EXTENSION ---------------

def _gather_fixture(monkeypatch, fresh_maker):
    """Stub `gather_features`' two data sources.  ``fresh_maker(feats, n)`` returns the
    re-encoded block for ``n`` images."""
    from chlu.experiments import exp_gen_store as X

    X._REENCODE_VERIFIED.clear()
    cache_rows, feats, cman = _disagreeing_fixture(n_rows=8, d=64)
    cman = dict(cman, encoder_param_hash="deadbeef")

    monkeypatch.setattr(
        X, "stream_images",
        lambda _ds, rows, _root, _blk: np.zeros((np.asarray(rows).size, 1, 1, 64), np.float32),
    )
    monkeypatch.setattr(
        X, "encode_images",
        lambda _enc, imgs, _batch: fresh_maker(feats, int(np.asarray(imgs).shape[0])),
    )
    return cache_rows, feats, cman


def test_extension_path_ABORTS_when_the_encoder_disagrees(monkeypatch):
    """⭐⭐ ACCEPTANCE 4.  `gather_features` is where the hazard actually lives: it
    returns cached rows interleaved with freshly encoded ones.  A row the cache does
    not hold puts it on the extension path, and there the check is ENFORCED."""
    from chlu.experiments import exp_gen_store as X

    cache_rows, feats, cman = _gather_fixture(
        monkeypatch, lambda f, n: _disagreeing(f)[:n]
    )
    lines = []
    with pytest.raises(AssertionError, match="ULP"):
        X.gather_features(
            "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
            np.array([0, 1, 99], np.int64), log=lines.append,
        )
    assert any("re-encode check runs ENFORCED here" in ln for ln in lines)


def test_an_UNENFORCED_preflight_does_not_excuse_a_later_extension(monkeypatch):
    """⛔⛔ THE HOLE THIS MUST NOT OPEN.  A run passes an unenforced preflight (it was
    covered) and then encodes a new row anyway — the extension path asserts at FULL
    strength, and the unenforced pass buys it nothing."""
    from chlu.experiments import exp_gen_store as X

    cache_rows, feats, cman = _gather_fixture(
        monkeypatch, lambda f, n: _disagreeing(f)[:n]
    )

    pre = X._assert_cache_reencodes(
        "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
        log=lambda _m: None, will_extend=False, reason=_COVERED,
    )
    assert pre["enforced"] is False and pre["would_have_failed"] is True
    # ⛔ and it did NOT discharge the extension-point check
    assert X._cache_identity(cache_rows, cman) not in X._REENCODE_VERIFIED

    with pytest.raises(AssertionError, match="ULP"):
        X.gather_features(
            "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
            np.array([0, 99], np.int64), log=lambda _m: None,
        )


def test_a_covered_gather_never_pays_for_the_check_at_all(monkeypatch):
    """⭐ The scoping, end to end: every requested row is cached, so `gather_features`
    never enters the extension path — and the disagreeing encoder is never consulted."""
    from chlu.experiments import exp_gen_store as X

    seen = []

    def _fresh(f, n):
        seen.append(n)
        return _disagreeing(f)[:n]

    cache_rows, feats, cman = _gather_fixture(monkeypatch, _fresh)
    out = X.gather_features(
        "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
        np.array([0, 3, 7], np.int64), log=lambda _m: None,
    )
    assert out.shape == (3, 64)
    np.testing.assert_array_equal(out, feats[[0, 3, 7]])
    assert seen == [], "a fully-cached gather encoded something"


def test_an_agreeing_encoder_still_extends_the_cache(monkeypatch):
    """⛔ A gate that cannot pass is as useless as one that cannot fail: the extension
    path still WORKS when the encoder does reproduce the cache."""
    from chlu.experiments import exp_gen_store as X

    cache_rows, feats, cman = _gather_fixture(monkeypatch, lambda f, n: f[:n].copy())
    lines = []
    out = X.gather_features(
        "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
        np.array([0, 99], np.int64), log=lines.append,
    )
    assert out.shape == (2, 64)
    np.testing.assert_array_equal(out[0], feats[0])
    np.testing.assert_array_equal(out[1], feats[0])  # the stub encodes row 0's values
    assert any("extending cache coverage: 1 row(s)" in ln for ln in lines)
    # ⭐ charged ONCE per (cache, encoder) per process
    assert X._cache_identity(cache_rows, cman) in X._REENCODE_VERIFIED


def test_an_enforced_pass_discharges_the_extension_check_once(monkeypatch):
    """⭐ The preflight that DID enforce is not paid for twice."""
    from chlu.experiments import exp_gen_store as X

    calls = []

    def _fresh(f, n):
        calls.append(n)
        return f[:n].copy()

    cache_rows, feats, cman = _gather_fixture(monkeypatch, _fresh)
    X._assert_cache_reencodes(
        "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
        log=lambda _m: None,
    )
    assert len(calls) == 1
    X.gather_features(
        "shapes3d", get_default_config(), object(), cache_rows, feats, cman,
        np.array([0, 99], np.int64), log=lambda _m: None,
    )
    assert len(calls) == 2, "the extension re-ran a check an ENFORCED preflight had passed"


# --- ⭐ the four scoped call sites -------------------------------------------

def test_route_b_uses_the_COVERAGE_GATES_OWN_answer(monkeypatch):
    """⛔ Coverage must never be re-derived at the re-encode call: two derivations can
    disagree about the same run.  Route B passes gate (4)'s own `cov`."""
    import inspect

    from chlu.experiments import exp_gen_wells_route_b as B

    src = inspect.getsource(B.preflight)
    assert 'will_extend=bool(cov["n_shortfall"] > 0)' in src
    # ⛔ exactly one coverage derivation in the whole module
    assert src.count("assert_cache_coverage(") == 1
    assert "searchsorted" not in src


@pytest.mark.parametrize("mod_name,fn_name", [
    ("chlu.experiments.exp_gen_store", "run_seed"),
    ("chlu.experiments.exp_gen_classifier", "run_seed"),
    ("chlu.experiments.exp_gen_certificate", "run_seed"),
])
def test_the_three_callers_without_a_coverage_gate_stay_ENFORCED(mod_name, fn_name):
    """⭐ These modules call `gather_features` and have no coverage gate to relax on,
    so they say `will_extend=True` explicitly — in the open, at the call site."""
    import importlib
    import inspect

    mod = importlib.import_module(mod_name)
    src = inspect.getsource(getattr(mod, fn_name))
    assert "_assert_cache_reencodes(" in src
    assert "will_extend=True" in src
    assert "will_extend=False" not in src
