"""The frozen bridge: encoder, feature cache, THE one shared decoder, probe, gate.

⭐ The load-bearing tests here are the two that can *fail on purpose*:

- ``test_no_leak_POSITIVE_CONTROL_fires`` — the R3.3 assertion is corrupted (a
  known held-out combination is injected into the decoder's training rows) and
  must FIRE, naming the row.  An assertion never seen to fail is not evidence.
- ``test_assert_decoder_hash_catches_a_retrained_decoder`` — an arm that
  silently retrains the shared decoder must be caught, because that single act
  invalidates every cross-arm comparison in the cell.

Everything else runs on synthetic tensors and a fake HDF5 handle, so the file
needs no dataset on disk.  The real-data path is exercised by the harness runs
reported in ``.claude/outputs/gen-feature-bridge.md``.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenBridgeConfig, get_default_config, load_config, save_config
from chlu.core.gen_bridge import (
    ENCODE_PRECISION,
    WHITENING_D_LIMIT,
    GenDecoder,
    GenEncoder,
    FactorProbe,
    assert_decoder_hash,
    assert_no_holdout_leak,
    build_feature_cache,
    encode_images,
    encode_precision_record,
    features_for_rows,
    load_feature_cache,
    param_count,
    param_hash,
    stream_images,
    train_decoder,
    train_encoder,
    train_probe,
    whitening_feasibility,
)
from chlu.data.gen_splits import FactorGrid, GenSplit, HoldoutViolation
from chlu.eval.gen_probe import (
    DEFAULT_CEILING_FLOOR,
    factor_labels,
    gate_verdict,
    measure_ceiling,
    probe_accuracy,
    summarize_per_factor,
    summarize_seeds,
)
from chlu.experiments.exp_gen_bridge import (
    _default_encoder_dir,
    apply_quick,
    build_arg_parser,
    cache_rows_for,
    main,
    select_rows,
)

# ---------------------------------------------------------------------------
# fixtures — a toy 3 x 4 x 5 grid and a tiny config (no data files needed)
# ---------------------------------------------------------------------------


def _toy_grid():
    return FactorGrid(
        names=("a", "b", "c"),
        values=(
            np.arange(3, dtype=float),
            np.arange(4, dtype=float),
            np.arange(5, dtype=float),
        ),
        dataset="toy",
    )


def _toy_predicate(values):
    """Held out: a >= 2 & b >= 3 & c >= 4 (one cell of the 60-cell grid)."""
    v = np.asarray(values)
    return (v[:, 0] >= 2) & (v[:, 1] >= 3) & (v[:, 2] >= 4)


def _toy_split():
    grid = _toy_grid()
    rows = np.arange(grid.n_rows, dtype=np.int64)
    mask = _toy_predicate(grid.values_of(rows))
    return GenSplit(
        dataset="toy",
        condition="recomb2element",
        variant="leave1out",
        grid=grid,
        train_rows=rows[~mask],
        test_rows=rows[mask],
        test_predicate=_toy_predicate,
    )


def _tiny_cfg(**over):
    cfg = GenBridgeConfig()
    cfg.enc_channels = [4, 8]
    cfg.dec_channels = [8, 4]
    cfg.probe_channels = [4, 8]
    cfg.enc_groups = cfg.dec_groups = cfg.probe_groups = 2
    cfg.enc_steps = cfg.dec_steps = cfg.probe_steps = 2
    cfg.enc_batch = cfg.dec_batch = cfg.probe_batch = 4
    cfg.n_train = 8
    cfg.n_eval_a = 4
    cfg.eval_batch = 4
    cfg.cache_block = 8
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_images(n=8, c=1, s=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, c, s, s), dtype=np.float64).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. the encoder, D, and the LACE whitening constraint
# ---------------------------------------------------------------------------


def test_default_encoder_D_is_under_the_lace_whitening_limit():
    """⭐ D is a design constraint, not a free choice: LACE needs a D x D whitening."""
    cfg = GenBridgeConfig()
    enc = GenEncoder(
        (3, 64, 64),
        cfg.enc_channels,
        cfg.enc_kernel,
        cfg.enc_stride,
        cfg.enc_groups,
        jax.random.PRNGKey(0),
    )
    assert enc.feat_shape == (128, 4, 4)
    assert enc.feature_dim == 2048
    verdict = whitening_feasibility(enc.feature_dim, 8192)
    assert verdict["full_whitening_feasible"] is True
    assert verdict["D"] <= WHITENING_D_LIMIT
    assert verdict["shrinkage_mandatory"] is False
    # ⭐ Head ruling: our choices never weaken a competitor — option 1 is recorded
    assert verdict["lace_option"].startswith("1 —")


def test_whitening_flags_shrinkage_when_D_exceeds_N():
    v = whitening_feasibility(2048, 512)
    assert v["shrinkage_mandatory"] is True
    big = whitening_feasibility(WHITENING_D_LIMIT + 1, 10**6)
    assert big["full_whitening_feasible"] is False
    assert big["lace_option"].startswith("2 —")
    assert "HEAD CALL" in big["note"]


def test_encoder_feat_shape_is_measured_not_assumed():
    """A kernel/stride that does not halve exactly must still report the true D."""
    enc = GenEncoder((1, 28, 28), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    out = enc(jnp.zeros((1, 28, 28)))
    assert out.dtype == enc.param_dtype  # ⚠ §7.23: dtype comes from the module
    assert tuple(out.shape) == enc.feat_shape
    assert enc.feature_dim == int(np.prod(out.shape))


def test_decoder_renders_back_to_the_image_shape():
    enc = GenEncoder((3, 32, 32), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    dec = GenDecoder(enc.feat_shape, (3, 32, 32), [8, 4], 2, jax.random.PRNGKey(1))
    y = dec(enc(jnp.zeros((3, 32, 32))))
    assert tuple(y.shape) == (3, 32, 32)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0  # sigmoid range


def test_decoder_refuses_a_feature_map_that_cannot_reach_the_image():
    with pytest.raises(ValueError, match="does not reach"):
        GenDecoder((8, 3, 3), (3, 32, 32), [8, 4], 2, jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# 2. hashing — the "one decoder, forever" guarantee
# ---------------------------------------------------------------------------


def test_param_hash_is_deterministic_and_sensitive():
    dec = GenDecoder((8, 8, 8), (1, 32, 32), [8, 4], 2, jax.random.PRNGKey(0))
    h1, h2 = param_hash(dec), param_hash(dec)
    assert h1 == h2 and len(h1) == 64
    bumped = eqx.tree_at(lambda m: m.convs[0].weight, dec, dec.convs[0].weight + 1e-3)
    assert param_hash(bumped) != h1
    assert param_count(dec) > 0


def test_assert_decoder_hash_catches_a_retrained_decoder():
    """⭐ An arm that silently retrains the shared decoder invalidates the cell."""
    dec = GenDecoder((8, 8, 8), (1, 32, 32), [8, 4], 2, jax.random.PRNGKey(0))
    other = GenDecoder((8, 8, 8), (1, 32, 32), [8, 4], 2, jax.random.PRNGKey(7))
    good = param_hash(dec)
    assert assert_decoder_hash(dec, good) == good
    with pytest.raises(AssertionError, match="NOT the shared frozen"):
        assert_decoder_hash(other, good)


# ---------------------------------------------------------------------------
# 3. R3.3 — the no-leak assertion, and its POSITIVE CONTROL
# ---------------------------------------------------------------------------


def test_no_leak_passes_and_counts_the_excluded_images():
    split = _toy_split()
    rng = np.random.default_rng(0)
    used = np.sort(rng.permutation(split.train_rows)[:20])
    rep = assert_no_holdout_leak(split, used)
    assert rep["n_train_rows_used"] == 20
    assert rep["n_checked"] == 20
    assert rep["n_holdout_excluded"] == split.n_test
    assert rep["n_holdout_leaked"] == 0


def test_no_leak_POSITIVE_CONTROL_fires():
    """⭐ Inject a known held-out row into the decoder's training rows: must FIRE."""
    split = _toy_split()
    offender = int(split.test_rows[0])
    used = np.sort(np.append(split.train_rows[:10], offender))
    with pytest.raises(HoldoutViolation) as exc:
        assert_no_holdout_leak(split, used)
    assert str(offender) in str(exc.value)


# ---------------------------------------------------------------------------
# 4. row selection: deterministic, arm-independent, disjoint
# ---------------------------------------------------------------------------


def test_select_rows_is_deterministic_disjoint_and_keeps_all_of_B():
    split = _toy_split()
    cfg = _tiny_cfg(n_train=20, n_eval_a=10)
    r1 = select_rows(split, cfg, 3)
    r2 = select_rows(split, cfg, 3)
    r3 = select_rows(split, cfg, 4)
    assert np.array_equal(r1["train"], r2["train"])  # same seed -> same rows
    assert not np.array_equal(r1["train"], r3["train"])  # different seed -> different
    assert np.intersect1d(r1["train"], r1["eval_a"]).size == 0
    assert np.intersect1d(r1["train"], r1["eval_b"]).size == 0
    assert np.array_equal(np.sort(r1["eval_b"]), np.sort(split.test_rows))
    # ⭐ ceiling_A is measured on rows the fitting stages never saw
    assert np.isin(r1["eval_a"], split.train_rows).all()


def test_cache_rows_full_covers_the_whole_grid():
    split = _toy_split()
    cfg = _tiny_cfg(n_train=20, n_eval_a=10)
    rows = select_rows(split, cfg, 0)
    full = cache_rows_for(split, rows, "full")
    used = cache_rows_for(split, rows, "used")
    assert full.size == split.n_total
    assert used.size == 20 + 10 + split.n_test
    assert np.isin(used, full).all()
    with pytest.raises(ValueError):
        cache_rows_for(split, rows, "nonsense")


# ---------------------------------------------------------------------------
# 5. the image stream — ⛔ never materialise the dataset
# ---------------------------------------------------------------------------


class _FakeH5:
    """Minimal stand-in for an h5py File: records the slices that were read."""

    def __init__(self, arr, key):
        self._d = {key: arr}
        self.reads = []

    def __getitem__(self, k):
        return _FakeDataset(self._d[k], self.reads)

    def close(self):
        pass


class _FakeDataset:
    def __init__(self, arr, reads):
        self._a = arr
        self._reads = reads
        self.shape = arr.shape

    def __getitem__(self, sl):
        self._reads.append(sl)
        return self._a[sl]


def test_stream_images_reads_contiguous_blocks_and_preserves_order():
    n = 40
    arr = (np.arange(n * 8 * 8 * 3).reshape(n, 8, 8, 3) % 251).astype(np.uint8)
    h = _FakeH5(arr, "images")
    rows = np.array([31, 2, 17], dtype=np.int64)
    out = stream_images("shapes3d", rows, block=8, handle=h)
    assert out.shape == (3, 3, 8, 8)
    assert out.dtype == np.float32
    for i, r in enumerate(rows):  # row order preserved, values scaled to [0, 1]
        want = np.transpose(arr[r], (2, 0, 1)).astype(np.float32) / 255.0
        assert np.allclose(out[i], want)
    # ⛔ every read is a bounded contiguous slice, never the whole array
    assert all(isinstance(s, slice) and (s.stop - s.start) <= 8 for s in h.reads)
    # blocks with no requested row are skipped entirely (3 rows -> <= 3 reads)
    assert len(h.reads) <= 3


def test_stream_images_handles_single_channel_and_empty_requests():
    arr = (np.arange(10 * 4 * 4).reshape(10, 4, 4) % 255).astype(np.uint8)
    h = _FakeH5(arr, "imgs")
    out = stream_images("dsprites", np.array([9, 0]), block=4, handle=h)
    assert out.shape == (2, 1, 4, 4)
    assert stream_images("dsprites", np.array([], dtype=np.int64), handle=h).shape == (
        0,
        1,
        64,
        64,
    )


def test_stream_images_rejects_an_out_of_range_row():
    arr = np.zeros((4, 4, 4, 3), np.uint8)
    with pytest.raises(IndexError):
        stream_images("shapes3d", np.array([9]), block=2, handle=_FakeH5(arr, "images"))


# ---------------------------------------------------------------------------
# 6. the feature cache: checksum, gather, and the too-narrow-scope error
# ---------------------------------------------------------------------------


def _cache_fixture(tmp_path, monkeypatch, n=12):
    arr = (np.arange(n * 8 * 8 * 3).reshape(n, 8, 8, 3) % 251).astype(np.uint8)
    fake = _FakeH5(arr, "images")

    class _Mod:
        @staticmethod
        def open_images(root=None):
            return fake

    monkeypatch.setattr("chlu.core.gen_bridge.dataset_module", lambda d: _Mod)
    enc = GenEncoder((3, 8, 8), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    cfg = _tiny_cfg(cache_block=4, cache_dtype="float16")
    rows = np.arange(n, dtype=np.int64)
    man = build_feature_cache(enc, "shapes3d", rows, tmp_path, cfg, scope="full")
    return enc, cfg, man


def test_feature_cache_roundtrips_with_a_checksum(tmp_path, monkeypatch):
    enc, _, man = _cache_fixture(tmp_path, monkeypatch)
    assert man["n_rows"] == 12
    assert man["D"] == int(np.prod(enc.feat_shape))
    assert man["encoder_param_hash"] == param_hash(enc)
    rows, feats, man2 = load_feature_cache(tmp_path)
    assert man2["features_sha256"] == man["features_sha256"]
    assert feats.shape == (12,) + enc.feat_shape
    got = features_for_rows(rows, feats, np.array([7, 1]))
    assert np.allclose(got[0], np.asarray(feats[7], np.float32))
    assert np.allclose(got[1], np.asarray(feats[1], np.float32))


def test_feature_cache_load_detects_corruption(tmp_path, monkeypatch):
    _cache_fixture(tmp_path, monkeypatch)
    raw = bytearray((tmp_path / "features.npy").read_bytes())
    raw[-1] ^= 0xFF
    (tmp_path / "features.npy").write_bytes(bytes(raw))
    with pytest.raises(AssertionError, match="checksum mismatch"):
        load_feature_cache(tmp_path)


def test_features_for_rows_refuses_a_row_the_cache_does_not_hold(tmp_path, monkeypatch):
    _cache_fixture(tmp_path, monkeypatch)
    rows, feats, _ = load_feature_cache(tmp_path)
    with pytest.raises(KeyError, match="not in the feature cache"):
        features_for_rows(rows, feats, np.array([999]))


# ---------------------------------------------------------------------------
# 6b. ⭐ the encode PRECISION — the frozen encoder has to compute the same
#     features on GPU as on the CPU that wrote the cache
# ---------------------------------------------------------------------------


def _capture_encode_jaxpr(enc, imgs):
    """The jaxpr of the function ``encode_images`` actually hands to ``filter_jit``.

    ⭐ Not a reconstruction: ``filter_jit`` is intercepted so the captured callable
    is the shipped one, and the precision is read off the real ``conv`` equations.
    ``eqx.nn.Conv2d`` passes **no** ``precision`` to
    :func:`jax.lax.conv_general_dilated`, so every conv here resolves
    ``precision=None`` against ``jax_default_matmul_precision`` at TRACE time —
    which is the moment this test exists to pin.
    """
    captured = {}
    real_filter_jit = eqx.filter_jit

    def capture(fn, *a, **kw):
        captured["fn"] = fn
        return real_filter_jit(fn, *a, **kw)

    eqx.filter_jit = capture
    try:
        encode_images(enc, imgs, 2, log=lambda *_: None)
    finally:
        eqx.filter_jit = real_filter_jit
    jaxpr = jax.make_jaxpr(captured["fn"])(jnp.asarray(imgs[:2]))
    return [
        e.params.get("precision")
        for e in jaxpr.jaxpr.eqns
        if "conv_general_dilated" in str(e.primitive)
    ]


def test_encode_path_convolutions_carry_the_requested_precision():
    """⭐⭐ The load-bearing test: DELETE the precision context in
    ``encode_images`` and every entry below becomes ``None``, i.e. XLA's default,
    i.e. **TF32 on Ampere/Ada** — the defect measured on an L40S (2026-09-16:
    ``max |delta| = 2^-8 = 4181 float16 ULP`` on 20.85 % of elements, encoder
    parameter hash MATCHING).  ⛔ It is not enough that the encoder is frozen; the
    arithmetic has to be pinned too."""
    enc = GenEncoder((3, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    imgs = np.zeros((2, 3, 16, 16), np.float32)
    precisions = _capture_encode_jaxpr(enc, imgs)
    want = jax.lax.Precision(ENCODE_PRECISION)
    assert want is jax.lax.Precision.HIGHEST
    assert len(precisions) == 2  # one per encoder block
    assert all(p == (want, want) for p in precisions), precisions


def test_encode_precision_does_not_move_the_numbers_on_cpu():
    """⭐ The laundering control, pinned as a test.  ``Precision`` is a no-op on
    CPU backends — and it must stay one, because the whole feature cache was
    written on CPU and every ``features_sha256`` in the programme depends on it.
    ⛔ If this ever fails on the laptop, the fix has changed the science."""
    enc = GenEncoder((3, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    imgs = np.asarray(np.random.default_rng(0).random((6, 3, 16, 16)), np.float32)
    fixed = encode_images(enc, imgs, 4, log=lambda *_: None)
    loose = np.concatenate(
        [
            np.asarray(
                eqx.filter_jit(jax.vmap(enc))(jnp.asarray(imgs[i : i + 4])), np.float32
            )
            for i in range(0, 6, 4)
        ]
    )
    assert fixed.tobytes() == loose.tobytes()


def test_encode_precision_record_names_the_precision_backend_and_device():
    """⛔ A silent precision is how the defect hid: the hash matched and nothing
    in the record said which arithmetic ran."""
    rec = encode_precision_record()
    assert rec["requested"] == ENCODE_PRECISION
    assert rec["resolved"] == "HIGHEST"
    assert rec["backend"] == jax.devices()[0].platform
    assert rec["device_kind"] == jax.devices()[0].device_kind
    assert rec["n_devices"] == len(jax.devices())
    assert rec["jax_version"] == jax.__version__


def test_encode_images_announces_the_precision_once_per_process(monkeypatch):
    monkeypatch.setattr("chlu.core.gen_bridge._PRECISION_ANNOUNCED", set())
    lines = []
    enc = GenEncoder((3, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
    imgs = np.zeros((2, 3, 16, 16), np.float32)
    encode_images(enc, imgs, 2, log=lines.append)
    encode_images(enc, imgs, 2, log=lines.append)
    assert len(lines) == 1, lines  # ⚠ a 300-block cache build logs it once
    assert "encode precision" in lines[0]
    assert "resolved=HIGHEST" in lines[0]
    assert f"backend={jax.devices()[0].platform}" in lines[0]
    assert f"device={jax.devices()[0].device_kind}" in lines[0]


def test_feature_cache_manifest_records_which_arithmetic_wrote_it(tmp_path, monkeypatch):
    """⭐ A cache written under one precision and extended under another is a
    different computation with a matching ``encoder_param_hash``."""
    _, _, man = _cache_fixture(tmp_path, monkeypatch)
    rec = man["encode_precision"]
    assert rec["resolved"] == "HIGHEST"
    assert rec["backend"] and rec["device_kind"]
    on_disk = json.loads((tmp_path / "feature_cache.json").read_text())
    assert on_disk["encode_precision"] == rec


# ---------------------------------------------------------------------------
# 7. training paths run and stay finite (tiny, 2 steps)
# ---------------------------------------------------------------------------


def test_recon_pretext_trains_and_the_shared_decoder_is_fit_on_frozen_features():
    cfg = _tiny_cfg()
    imgs = _tiny_images(n=8, c=1, s=16)
    enc, aux, hist = train_encoder(imgs, "recon", cfg, jax.random.PRNGKey(0))
    assert len(hist) == 2 and all(np.isfinite(h) for h in hist)
    assert isinstance(aux, GenDecoder)  # the pretext decoder — discarded downstream
    before = param_hash(enc)
    dec, dhist = train_decoder(enc, imgs, cfg, jax.random.PRNGKey(1))
    assert all(np.isfinite(h) for h in dhist)
    # ⭐ the encoder is FROZEN while the shared decoder is fit
    assert param_hash(enc) == before
    assert param_hash(dec) != param_hash(aux)


def test_cls_pretext_needs_labels_and_trains():
    cfg = _tiny_cfg()
    imgs = _tiny_images(n=8, c=1, s=16)
    labels = np.stack([np.arange(8) % 3, np.arange(8) % 4], axis=1)
    with pytest.raises(ValueError, match="needs labels"):
        train_encoder(imgs, "cls", cfg, jax.random.PRNGKey(0))
    enc, heads, hist = train_encoder(
        imgs, "cls", cfg, jax.random.PRNGKey(0), labels=labels, cardinalities=(3, 4)
    )
    assert len(heads.linears) == 2
    assert all(np.isfinite(h) for h in hist)


def test_unknown_pretext_is_refused():
    with pytest.raises(ValueError, match="pretext must be"):
        train_encoder(_tiny_images(4, 1, 16), "vibes", _tiny_cfg(), jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# 8. the probe and the ceiling measurement
# ---------------------------------------------------------------------------


class _PerfectProbe(eqx.Module):
    """A probe whose prediction is read straight off the image's first pixels."""

    heads: object

    def __call__(self, x):
        v = x.reshape(-1)
        return [jnp.array([1.0 - v[0], v[0]]), jnp.array([1.0 - v[1], v[1]])]


def test_probe_accuracy_reports_per_factor_aggregate_and_exact_match():
    probe = _PerfectProbe(heads=None)
    imgs = np.zeros((4, 1, 4, 4), np.float32)
    imgs[:, 0, 0, 0] = [1, 1, 0, 0]  # factor 0 label
    imgs[:, 0, 0, 1] = [1, 0, 1, 0]  # factor 1 label
    labels = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], np.int32)
    acc = probe_accuracy(probe, imgs, labels, ("f0", "f1"), batch=2)
    assert acc["per_factor"] == {"f0": 1.0, "f1": 1.0}
    assert acc["aggregate"] == 1.0 and acc["exact_match"] == 1.0
    wrong = probe_accuracy(probe, imgs, labels[::-1], ("f0", "f1"), batch=2)
    assert wrong["aggregate"] < 1.0


def test_measure_ceiling_runs_end_to_end_on_tiny_tensors():
    cfg = _tiny_cfg()
    imgs = _tiny_images(n=8, c=1, s=16)
    labels = np.stack([np.arange(8) % 3, np.arange(8) % 4], axis=1)
    enc, _, _ = train_encoder(imgs, "recon", cfg, jax.random.PRNGKey(0))
    dec, _ = train_decoder(enc, imgs, cfg, jax.random.PRNGKey(1))
    probe, _ = train_probe(imgs, labels, (3, 4), cfg, jax.random.PRNGKey(2))
    feats = encode_images(enc, imgs, 4)
    out = measure_ceiling(dec, probe, feats, labels, ("f0", "f1"), 4)
    assert set(out["per_factor"]) == {"f0", "f1"}
    assert 0.0 <= out["aggregate"] <= 1.0
    assert isinstance(probe, FactorProbe)


def test_factor_labels_match_the_grid_bijection():
    grid = _toy_grid()
    rows = np.array([0, 1, 59], np.int64)
    lab = factor_labels(grid, rows)
    assert lab.shape == (3, 3)
    assert lab[0].tolist() == [0, 0, 0]
    assert lab[2].tolist() == [2, 3, 4]


# ---------------------------------------------------------------------------
# 9. ⭐ the gate itself
# ---------------------------------------------------------------------------


def test_gate_proceeds_exactly_at_the_floor_and_stops_below_it():
    assert DEFAULT_CEILING_FLOOR == 0.8
    g = gate_verdict(0.90, 0.72)  # exactly 0.8 x
    assert g["verdict"] == "PROCEED" and g["ratio"] == pytest.approx(0.8)
    below = gate_verdict(0.90, 0.7199)
    assert below["verdict"] == "STOP"
    assert "return to Head and Advisor" in below["note"]
    assert "Do not tune the decoder" in below["note"]


def test_gate_with_a_dead_ceiling_A_stops_rather_than_dividing_by_zero():
    g = gate_verdict(0.0, 0.0)
    assert g["verdict"] == "STOP" and np.isnan(g["ratio"])


def test_gate_floor_is_configurable_but_defaults_to_the_prereg_value():
    assert gate_verdict(1.0, 0.5, floor=0.4)["verdict"] == "PROCEED"
    assert gate_verdict(1.0, 0.5)["verdict"] == "STOP"
    assert GenBridgeConfig().ceiling_floor == 0.8


def test_seed_summaries_report_spread():
    recs = [{"aggregate": 0.8}, {"aggregate": 0.9}, {"aggregate": 1.0}]
    s = summarize_seeds(recs, "aggregate")
    assert s["n_seeds"] == 3 and s["mean"] == pytest.approx(0.9)
    assert s["std"] > 0 and s["min"] == 0.8 and s["max"] == 1.0
    pf = summarize_per_factor(
        [{"per_factor": {"f": 0.4}}, {"per_factor": {"f": 0.6}}], ("f",)
    )
    assert pf["f"]["mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 10. config wiring (house three-site rule) and quick mode
# ---------------------------------------------------------------------------


def test_gen_bridge_config_roundtrips_through_yaml(tmp_path):
    config = get_default_config()
    config.gen_bridge.n_train = 123
    config.gen_bridge.cache_scope = "used"
    path = tmp_path / "config.yaml"
    save_config(config, path)
    assert "gen_bridge" in json.dumps(
        {k: str(v) for k, v in [("t", path.read_text())]}
    )
    back = load_config(path)
    assert back.gen_bridge.n_train == 123
    assert back.gen_bridge.cache_scope == "used"
    assert back.gen_bridge.ceiling_floor == 0.8


def test_apply_quick_shrinks_everything_and_narrows_the_cache():
    config = apply_quick(get_default_config())
    gb = config.gen_bridge
    assert gb.enc_steps == gb.quick_enc_steps < GenBridgeConfig().enc_steps
    assert gb.n_train == gb.quick_n_train < GenBridgeConfig().n_train
    assert gb.cache_scope == "used"
    assert gb.seeds == [0]


# ---------------------------------------------------------------------------
# 11. ⛔ the binding CLI contract from the CSF3 launcher
# ---------------------------------------------------------------------------


def test_cli_accepts_exactly_the_launcher_contract_flags():
    args = build_arg_parser().parse_args(
        [
            "--dataset", "shapes3d",
            "--pretext", "recon",
            "--stage", "encoder",
            "--arm", "clu",
            "--seed", "2",
            "--out", "/tmp/x",
        ]
    )
    assert (args.dataset, args.pretext, args.stage, args.arm, args.seed) == (
        "shapes3d", "recon", "encoder", "clu", 2
    )
    assert args.out == "/tmp/x"


def test_cli_refuses_the_stages_that_belong_to_other_spokes(tmp_path):
    for stage in ("store", "cell"):
        rc = main(
            ["--dataset", "shapes3d", "--pretext", "recon", "--stage", stage,
             "--arm", "clu", "--seed", "0", "--out", str(tmp_path)]
        )
        assert rc == 2  # ⛔ refuses to no-op rather than reporting success


def test_decoder_stage_finds_the_encoder_in_the_launcher_layout(tmp_path):
    """job_gpu_gen.sh writes <ws>/<stage>/<arm>_s<seed>/artifacts — the decoder
    stage must resolve its sibling encoder dir, or it silently trains a SECOND
    decoder against a re-initialised encoder."""
    ws = tmp_path / "shapes3d_recon"
    enc = ws / "encoder" / "clu_s2" / "artifacts"
    dec = ws / "decoder" / "clu_s2" / "artifacts"
    enc.mkdir(parents=True)
    dec.mkdir(parents=True)
    assert _default_encoder_dir(dec, "clu", 2) == dec  # no encoder.pkl yet -> self
    (enc / "encoder.pkl").write_bytes(b"x")
    assert _default_encoder_dir(dec, "clu", 2) == enc


@pytest.mark.parametrize("x64", [False, True])
def test_every_module_reads_a_float32_input_under_x64(x64):
    """⚠ §7.23 regression guard. A module-scoped ``jax_enable_x64`` fixture
    elsewhere in the suite makes eqx build float64 weights; before the fix these
    modules crashed on a float32 image with 'lhs and rhs must have the same
    element type'. The failure only appeared in a full-suite run, never in
    isolation, which is exactly why it is pinned here."""
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", x64)
        enc = GenEncoder((1, 16, 16), [4, 8], 4, 2, 2, jax.random.PRNGKey(0))
        dec = GenDecoder(enc.feat_shape, (1, 16, 16), [8, 4], 2, jax.random.PRNGKey(1))
        probe = FactorProbe((1, 16, 16), [4], 2, (3,), jax.random.PRNGKey(2))
        x32 = np.zeros((1, 16, 16), np.float32)
        z = enc(x32)
        assert z.shape == enc.feat_shape
        assert dec(np.asarray(z, np.float32)).shape == (1, 16, 16)
        assert probe(x32)[0].shape == (3,)
        assert (z.dtype == jnp.float64) == x64
    finally:
        jax.config.update("jax_enable_x64", prev)
