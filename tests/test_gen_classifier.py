"""The **external** factor classifier — the instrument, and the assertions that gate it.

⭐ Three groups of load-bearing tests, in the order the task's acceptance items ask
for them:

1. **The instrument is not a second copy of the probe.**  Different depth, different
   kernel, different activation, a hidden projection the probe does not have, and —
   the one that actually matters — a **disjoint** training draw from the rows the
   bridge fitted its decoder and probe on.  ⛔ Architecture alone is not
   independence; these tests only pin what is mechanically checkable.
2. ⭐ **Every no-leak assertion is shown ABLE TO FIRE** on a deliberately seeded
   violation (`gen-wide-split` precedent).  An assertion that has never fired is a
   comment.
3. ⭐ **The gate is stop-and-return and cannot be softened**: ``PASS`` requires
   *every* seed, a 2/3 pass is a ``FAIL``, and the file a downstream spoke reads
   carries the verdict at a fixed top-level key.

Everything runs on synthetic tensors except the render-leak bit-identity test, which
``importorskip``s h5py exactly as ``test_gen_wide_split.py`` does.
"""

import json

import jax
import numpy as np
import pytest

from chlu.config import (
    GenClassifierConfig,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_bridge import FactorProbe, param_count, param_hash
from chlu.data.gen_splits import FactorGrid, GenSplit, HoldoutViolation
from chlu.eval.gen_classifier import (
    DEFAULT_GATE_FLOOR,
    GATE_FACTOR,
    ExternalFactorCNN,
    LineageLeak,
    RenderLeak,
    assert_lineage_disjoint,
    assert_no_render_leak,
    classifier_rows,
    corner_rows,
    gate_verdict,
    instrument_gap,
    no_leak_report,
    train_external_classifier,
)
from chlu.eval.gen_probe import factor_probs, predict_factors
from chlu.experiments.exp_gen_classifier import (
    STAGES,
    _config_from_args,
    apply_quick,
    build_arg_parser,
    read_both,
)

CARDS = (10, 10, 10, 8, 4, 15)
IN_SHAPE = (3, 32, 32)


def _cfg():
    c = GenClassifierConfig()
    c.clf_channels = [8, 16]
    c.clf_hidden = 16
    c.clf_steps = 3
    c.clf_batch = 4
    c.clf_checkpoint_every = 2
    return c


def _model(key=0, cfg=None, in_shape=IN_SHAPE):
    cfg = cfg or _cfg()
    return ExternalFactorCNN(
        in_shape, cfg.clf_channels, cfg.clf_kernel, cfg.clf_stride,
        cfg.clf_groups, cfg.clf_hidden, CARDS, jax.random.PRNGKey(key),
    )


# ---------------------------------------------------------------------------
# 1. The instrument reads images and is scored by the SAME functions as the probe
# ---------------------------------------------------------------------------


def test_it_emits_one_logit_vector_per_factor():
    out = _model()(np.zeros(IN_SHAPE, np.float32))
    assert [int(o.shape[-1]) for o in out] == list(CARDS)


def test_the_same_scoring_functions_read_both_instruments():
    """⭐ A gap between the two can never be a difference in the scoring code."""
    ext = _model()
    probe = FactorProbe(IN_SHAPE, [8, 16], 8, CARDS, jax.random.PRNGKey(1))
    imgs = np.zeros((5,) + IN_SHAPE, np.float32)
    labels = np.zeros((5, len(CARDS)), np.int32)
    both = read_both(ext, probe, imgs, labels, [f"f{i}" for i in range(len(CARDS))], 4)
    assert set(both) == {"external", "probe"}
    for r in both.values():
        assert r["n"] == 5 and len(r["per_factor"]) == len(CARDS)
    assert predict_factors(ext, imgs, 4).shape == (5, len(CARDS))
    assert factor_probs(ext, imgs, 4, 4).shape == (5, CARDS[4])


@pytest.mark.parametrize("x64", [False, True])
def test_it_reads_a_float32_input_under_x64(x64):
    """⚠ §7.23: x64 flipped at module import must not break the new path."""
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", x64)
        out = _model()(np.zeros(IN_SHAPE, np.float32))
        assert len(out) == len(CARDS)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_it_is_architecturally_distinct_from_the_shipped_probe():
    """⛔ Not independence — but a shipped default that silently converged on the
    probe's own architecture would make leg 3 vacuous, so it is pinned."""
    d = GenClassifierConfig()
    b = get_default_config().gen_bridge
    assert list(d.clf_channels) != list(b.probe_channels)
    assert len(d.clf_channels) == len(b.probe_channels) + 1
    assert d.clf_kernel != b.enc_kernel or d.clf_kernel != 4
    assert d.clf_hidden > 0  # the probe has no hidden projection at all
    ext = _model()
    assert hasattr(ext, "proj") and not hasattr(
        FactorProbe(IN_SHAPE, [8, 16], 8, CARDS, jax.random.PRNGKey(1)), "proj"
    )


def test_the_key_chain_is_never_the_bridges():
    """⛔ ``seed * 10_007 + 20_260_910`` is the bridge's; reusing it would make the
    instrument's init a function of the decoder's."""
    d = GenClassifierConfig()
    for seed in (0, 1, 2):
        assert seed * d.row_seed_mult + d.row_seed_offset != seed * 10_007 + 20_260_910


def test_training_runs_and_checkpoints_every_epoch():
    cfg = _cfg()
    imgs = np.random.default_rng(0).random((8,) + IN_SHAPE).astype(np.float32)
    labels = np.zeros((8, len(CARDS)), np.int32)
    seen = []
    model, hist, _ = train_external_classifier(
        imgs, labels, CARDS, cfg, jax.random.PRNGKey(0),
        on_epoch=lambda step, m, loss: seen.append(int(step)),
    )
    assert len(hist) == cfg.clf_steps and seen == [2, 3]
    assert param_count(model) > 0 and len(param_hash(model)) == 64


# ---------------------------------------------------------------------------
# 2. ⭐ Rows — and every assertion SHOWN ABLE TO FIRE
# ---------------------------------------------------------------------------


def _toy_split(n_per=(4, 4, 4, 4, 4, 4)):
    """A 6-factor grid shaped like Shapes3D, with a corner-style held-out predicate."""
    values = tuple(np.linspace(0.0, 1.0, n) for n in n_per[:3]) + (
        np.linspace(0.75, 1.25, n_per[3]),
        np.arange(n_per[4], dtype=float),          # shape: 1.0 == cylinder
        np.linspace(-30.0, 30.0, n_per[5]),
    )
    grid = FactorGrid(names=("floor_hue", "wall_hue", "object_hue", "scale", "shape",
                             "orientation"), values=values, dataset="toy")

    def predicate(v):
        v = np.asarray(v)
        return (
            (v[:, 0] >= 0.8) & (v[:, 1] >= 0.8) & (v[:, 2] >= 0.8)
            & (v[:, 3] >= 1.1) & (v[:, 4] == 1.0) & (v[:, 5] > 20)
        )

    rows = np.arange(grid.n_rows, dtype=np.int64)
    mask = predicate(grid.values_of(rows))
    return GenSplit("toy", "recomb2element", "leave1out", grid,
                    rows[~mask], rows[mask], predicate), grid.values_of(rows)


def _rows(seed=0):
    split, values = _toy_split()
    cfg = _cfg()
    cfg.n_clf_train = 32
    bridge = get_default_config().gen_bridge
    bridge.n_train, bridge.n_eval_a = 64, 16
    return split, values, cfg, bridge, classifier_rows(split, values, cfg, bridge, seed)


def test_the_corner_predicate_is_the_one_the_evaluation_sets_use():
    split, values = _toy_split()
    corner = corner_rows(values)
    assert corner.size > 0
    # every held-out row lives inside the corner: the corner is a strict superset
    assert np.intersect1d(corner, split.test_rows).size == split.test_rows.size


def test_the_training_draw_is_disjoint_from_everything_it_is_scored_on():
    split, _, _, _, rows = _rows()
    clf = rows["clf_train"]
    for name in ("bridge_train", "eval_a", "eval_b", "corner"):
        assert np.intersect1d(clf, rows[name]).size == 0
    assert np.intersect1d(clf, split.test_rows).size == 0
    assert np.all(np.isin(clf, split.train_rows))


def test_the_draw_is_deterministic_in_the_seed_and_moves_with_it():
    _, _, _, _, a = _rows(0)
    _, _, _, _, b = _rows(0)
    _, _, _, _, c = _rows(1)
    assert np.array_equal(a["clf_train"], b["clf_train"])
    assert not np.array_equal(a["clf_train"], c["clf_train"])


def test_lineage_assertion_fires_on_a_seeded_violation():
    """⭐ Deliberately smuggle one bridge row into the draw."""
    _, _, _, _, rows = _rows()
    assert assert_lineage_disjoint(rows["clf_train"], rows)["n_overlap_bridge_train"] == 0
    dirty = np.sort(np.concatenate([rows["clf_train"][:-1], rows["bridge_train"][:1]]))
    with pytest.raises(LineageLeak, match="bridge_train"):
        assert_lineage_disjoint(dirty, rows)


def test_holdout_assertion_fires_on_a_seeded_violation():
    """⭐ R3.3's positive control: one held-out combination in the training rows."""
    split, _, _, _, rows = _rows()
    clean = no_leak_report(split, rows["clf_train"], rows)
    assert clean["holdout"]["n_holdout_leaked"] == 0
    assert clean["holdout"]["n_holdout_excluded"] == split.test_rows.size
    dirty = np.sort(np.concatenate([rows["clf_train"], split.test_rows[:1]]))
    with pytest.raises(HoldoutViolation):
        no_leak_report(split, dirty, rows)


def test_render_leak_assertion_fires_on_a_shape_mismatch():
    with pytest.raises(RenderLeak, match="training images for"):
        assert_no_render_leak(np.zeros((2, 3, 4, 4), np.float32), np.arange(3), "shapes3d")


def test_render_leak_assertion_is_bit_identity_against_the_dataset():
    """⭐ The one that makes ⛔'never saw a render' a measurement: perturb one pixel
    of one real image and the check must refuse the tensor."""
    pytest.importorskip("h5py")
    from chlu.core.gen_bridge import stream_images

    from chlu.data.shapes3d import H5_NAME, shapes3d_root

    root = None  # -> $CHLU_GEN_DATA_ROOT/shapes3d, exactly as the experiment resolves it
    if not (shapes3d_root(root) / H5_NAME).exists():
        pytest.skip("Shapes3D not on disk")
    rows = np.arange(16, dtype=np.int64)
    imgs = stream_images("shapes3d", rows, root, 8192)
    ok = assert_no_render_leak(imgs, rows, "shapes3d", root, n_sample=16)
    assert ok["max_abs_delta"] == 0.0 and ok["n_render_rows"] == 0
    dirty = imgs.copy()
    dirty[3, 0, 0, 0] = np.float32(dirty[3, 0, 0, 0] + 0.5)
    with pytest.raises(RenderLeak, match="bit-identical"):
        assert_no_render_leak(dirty, rows, "shapes3d", root, n_sample=16)


# ---------------------------------------------------------------------------
# 3. ⭐ The gate — stop-and-return, and it cannot be softened
# ---------------------------------------------------------------------------


def test_pass_requires_every_seed_not_the_mean():
    """⛔ A 2/3 pass whose MEAN clears the floor is still a FAIL."""
    per_seed = {"0": 0.99, "1": 0.99, "2": 0.90}
    v = gate_verdict(per_seed, 0.95)
    assert v["gate"] == "FAIL" and v["mean"] > 0.95 and v["min"] == 0.90
    assert "STOP" in v["note"]
    assert gate_verdict({"0": 0.96, "1": 0.95, "2": 0.99}, 0.95)["gate"] == "PASS"


def test_the_floor_and_the_factor_are_the_pre_registered_ones():
    assert DEFAULT_GATE_FLOOR == 0.95 and GATE_FACTOR == "shape"
    d = GenClassifierConfig()
    assert d.gate_floor == DEFAULT_GATE_FLOOR and d.gate_factor == GATE_FACTOR


def test_the_verdict_carries_the_key_a_downstream_spoke_gates_on(tmp_path):
    v = gate_verdict({"0": 0.99}, 0.95)
    p = tmp_path / "classifier_gate.json"
    p.write_text(json.dumps(v))
    assert json.loads(p.read_text())["gate"] in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# 4. ⭐⭐ Leg 3 — the gap is reported, its materiality is a RULE, not a judgement
# ---------------------------------------------------------------------------


def test_a_small_gap_is_not_material_and_selects_H_AGREE():
    g = instrument_gap({"0": 0.94, "1": 0.93, "2": 0.95},
                       {"0": 0.95, "1": 0.94, "2": 0.96}, 0.05, 2.0)
    assert not g["material"] and g["hypothesis"] == "H_AGREE"
    assert g["mean_delta"] == pytest.approx(-0.01)


def test_a_large_negative_gap_selects_H_OPTIMISTIC():
    g = instrument_gap({"0": 0.70, "1": 0.71, "2": 0.72},
                       {"0": 0.94, "1": 0.94, "2": 0.94}, 0.05, 2.0)
    assert g["material"] and g["hypothesis"] == "H_OPTIMISTIC"


def test_a_large_positive_gap_is_a_finding_too():
    g = instrument_gap({"0": 0.99, "1": 0.99, "2": 0.99},
                       {"0": 0.80, "1": 0.80, "2": 0.80}, 0.05, 2.0)
    assert g["material"] and g["hypothesis"] == "H_PROBE_PESSIMISTIC"


def test_a_gap_smaller_than_its_own_spread_is_not_material():
    """⛔ Clearing 0.05 absolutely is not enough; it must also beat the seed spread."""
    g = instrument_gap({"0": 0.50, "1": 0.99, "2": 0.70},
                       {"0": 0.80, "1": 0.80, "2": 0.80}, 0.05, 2.0)
    assert abs(g["mean_delta"]) >= 0.05 and not g["material"]
    assert g["hypothesis"] == "H_AGREE"


# ---------------------------------------------------------------------------
# 5. Config / CLI plumbing — ⛔ and the standing decoder trap
# ---------------------------------------------------------------------------


def test_the_decoder_trap_is_closed_by_default():
    """⛔ ``GenBridgeConfig.dec_recipe`` still defaults to the FAILED baseline; this
    block must never inherit it."""
    c = get_default_config()
    assert c.gen_bridge.dec_recipe == "baseline"
    assert c.gen_classifier.decoder_source == "winner"
    assert c.gen_classifier.expected_decoder_recipe == "v1_capacity_schedule"


def test_defaults_are_the_reported_ones():
    d = get_default_config().gen_classifier
    assert d.seeds == [0, 1, 2] and d.n_clf_train == 8192
    # ⭐ matched to the bridge's own budget on purpose: same data budget, disjoint rows
    assert d.n_clf_train == get_default_config().gen_bridge.n_train
    assert d.gap_abs_threshold == 0.05 and d.gap_spread_mult == 2.0
    assert d.probe_reference_balacc == 0.9414 and d.wide_reference_balacc == 0.9894


def test_the_new_block_round_trips_through_yaml(tmp_path):
    c = get_default_config()
    c.gen_classifier.clf_steps = 77
    c.gen_classifier.gate_floor = 0.97
    p = tmp_path / "config.yaml"
    save_config(c, p)
    back = load_config(p)
    assert back.gen_classifier.clf_steps == 77 and back.gen_classifier.gate_floor == 0.97
    assert back.gen_bridge.dec_recipe == c.gen_bridge.dec_recipe


def test_quick_mode_is_plumbing_only():
    c = apply_quick(get_default_config())
    q = c.gen_classifier
    assert q.seeds == [0] and q.clf_steps < 100 and q.n_clf_train < 8192
    assert q.verify_cache_checksum is False


def test_cli_overrides_reach_the_config():
    args = build_arg_parser().parse_args(
        ["--out", "x", "--stage", "fit", "--seed", "1", "--steps", "9",
         "--n-train", "64", "--no-plots"]
    )
    assert args.stage in STAGES
    c = _config_from_args(args).gen_classifier
    assert c.clf_steps == 9 and c.n_clf_train == 64 and c.make_plots is False
