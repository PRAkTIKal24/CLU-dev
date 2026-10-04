"""The WIDE configuration and the **discrimination** read (R17 / R20).

⭐ Two load-bearing tests, pulling in opposite directions:

1. **Balanced accuracy cannot go degenerate.**  Montero's held-out cell realises
   exactly ONE shape, so a probe that answers "cylinder" to everything in that
   colour corner scores a *detection* rate of 1.000 on it.
   ``test_a_constant_answer_scores_exactly_one_half`` is the test that makes the
   difference between detection and discrimination mechanical rather than
   editorial — it is the whole content of R20's correction.
2. **Montero's configuration must be unchanged** by the existence of a second
   condition: same split, same rows, same artifacts directory, same default
   behaviour end to end.  The wide condition is *additional*, never a
   modification — and it is a SEPARATE full configuration, because artifacts
   frozen under one condition have seen the other's held-out rows.

Everything runs on synthetic tensors except the two tests that need Shapes3D on
disk, which ``importorskip`` h5py exactly as ``test_gen_wide_split.py`` does.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chlu.config import GenBridgeConfig, get_default_config, load_config, save_config
from chlu.core.gen_bridge import FactorProbe
from chlu.data.gen_splits import (
    FactorCell,
    FactorGrid,
    HoldoutViolation,
    build_cell_split,
)
from chlu.eval.gen_probe import (
    DEFAULT_DISC_ABS_FLOOR,
    DEFAULT_DISC_FLOOR,
    DEFAULT_DISC_MIN_NEGATIVES,
    balanced_accuracy,
    binary_auroc,
    discrimination_metrics,
    discrimination_verdict,
    factor_probs,
    multiclass_balanced_accuracy,
    pool_discrimination,
    predict_factors,
)
from chlu.experiments.exp_gen_bridge import (
    DEFAULT_CONDITION,
    artifacts_tag,
    build_arg_parser,
    _config_from_args,
    condition_of,
    load_split_and_cells,
    per_cell_no_leak,
)


def _onehot(n, c, cls):
    p = np.full((n, c), 0.01)
    p[:, cls] = 1.0 - 0.01 * (c - 1)
    return p


# ---------------------------------------------------------------------------
# 1. ⭐ detection vs discrimination — the degeneracy R20 corrected
# ---------------------------------------------------------------------------


def test_a_constant_answer_scores_exactly_one_half():
    """⭐ THE test.  A probe that answers ``target`` to every image in the corner
    has a **detection** rate of 1.000 and a **balanced accuracy** of exactly
    0.5.  ⛔ The two may never appear in the same column."""
    pos, neg = _onehot(72, 4, 1), _onehot(216, 4, 1)
    m = discrimination_metrics(pos, neg, target=1)
    assert m["detection_rate"] == 1.0 and m["tpr"] == 1.0
    assert m["tnr"] == 0.0
    assert m["balanced_accuracy"] == 0.5
    assert m["auroc"] == 0.5  # every score identical -> ties average to chance


def test_detection_rate_is_labelled_as_tpr_and_is_not_the_headline():
    m = discrimination_metrics(_onehot(10, 3, 0), _onehot(10, 3, 1), target=0)
    assert m["detection_rate"] == m["tpr"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_perfect_and_inverted_discrimination():
    perfect = discrimination_metrics(_onehot(8, 3, 2), _onehot(9, 3, 0), target=2)
    assert perfect["balanced_accuracy"] == 1.0 and perfect["auroc"] == 1.0
    # fully inverted: no positive is called ``target`` and EVERY negative is
    inverted = discrimination_metrics(_onehot(8, 3, 0), _onehot(9, 3, 2), target=2)
    assert (inverted["tpr"], inverted["tnr"]) == (0.0, 0.0)
    assert inverted["balanced_accuracy"] == 0.0
    assert inverted["auroc"] == 0.0


def test_balanced_accuracy_is_the_mean_of_the_two_rates():
    assert balanced_accuracy(0.9, 0.7) == pytest.approx(0.8)


def test_metrics_carry_their_counts_and_target():
    m = discrimination_metrics(_onehot(5, 4, 3), _onehot(7, 4, 0), target=3)
    assert (m["n_pos"], m["n_neg"], m["target"]) == (5, 7, 3)


def test_too_few_negatives_is_flagged_not_silently_pooled():
    """⛔ A balanced accuracy over 3 negatives is not a measurement."""
    m = discrimination_metrics(
        _onehot(72, 4, 1),
        _onehot(3, 4, 0),
        target=1,
        min_negatives=DEFAULT_DISC_MIN_NEGATIVES,
    )
    assert m["sufficient"] is False
    assert pool_discrimination([m])["n_cells"] == 0
    assert pool_discrimination([m])["n_cells_dropped"] == 1


def test_mismatched_class_counts_are_refused():
    with pytest.raises(ValueError):
        discrimination_metrics(_onehot(4, 3, 0), _onehot(4, 5, 0), target=0)


# ---------------------------------------------------------------------------
# 2. AUROC, by hand
# ---------------------------------------------------------------------------


def test_auroc_perfect_inverted_and_tied():
    assert binary_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert binary_auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert binary_auroc([0.5] * 6, [1, 1, 1, 0, 0, 0]) == 0.5


def test_auroc_matches_the_hand_computed_rank_statistic():
    # scores 0.4(+) 0.3(-) 0.2(+) 0.1(-): pairs (+>-) = 0.4>0.3, 0.4>0.1, 0.2>0.1
    assert binary_auroc([0.4, 0.3, 0.2, 0.1], [1, 0, 1, 0]) == pytest.approx(3 / 4)


def test_auroc_handles_ties_with_average_ranks():
    # one tied pair contributes 0.5
    assert binary_auroc([0.5, 0.5], [1, 0]) == 0.5


def test_auroc_of_a_single_class_is_nan_not_zero():
    assert np.isnan(binary_auroc([0.1, 0.2], [1, 1]))
    assert np.isnan(
        discrimination_metrics(_onehot(4, 3, 0), np.zeros((0, 3)), 0)["auroc"]
    )


def test_auroc_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        binary_auroc([0.1, 0.2], [1])


# ---------------------------------------------------------------------------
# 3. the 4-way read is a DIFFERENT quantity
# ---------------------------------------------------------------------------


def test_multiclass_balanced_accuracy_is_macro_recall_not_accuracy():
    """⛔ 99 easy items of one class must not hide a class that is never right."""
    y_true = np.array([0] * 99 + [1])
    y_pred = np.array([0] * 99 + [0])
    out = multiclass_balanced_accuracy(y_true, y_pred, 4)
    assert out["accuracy"] == pytest.approx(0.99)
    assert out["balanced_accuracy"] == pytest.approx(0.5)
    assert out["classes_present"] == [0, 1]


def test_multiclass_ignores_classes_absent_from_the_truth():
    out = multiclass_balanced_accuracy([2, 2], [2, 2], 4)
    assert out["balanced_accuracy"] == 1.0 and out["classes_present"] == [2]


# ---------------------------------------------------------------------------
# 4. pooling and the pre-registered verdict rule
# ---------------------------------------------------------------------------


def test_pool_is_a_macro_mean_over_cells():
    cells = [
        {"balanced_accuracy": 0.8, "sufficient": True},
        {"balanced_accuracy": 1.0, "sufficient": True},
    ]
    p = pool_discrimination(cells)
    assert (p["macro_mean"], p["min"], p["max"], p["n_cells"]) == (0.9, 0.8, 1.0, 2)


def test_verdict_needs_both_floors():
    """⭐ The rule pre-registered before the run: absolute AND ratio."""
    assert discrimination_verdict(0.93, 0.96)["verdict"] == "PASS"
    # a high ratio against a weak normaliser must not pass on its own
    assert discrimination_verdict(0.60, 0.62)["verdict"] == "FAIL"
    # a strong absolute value with a collapsed ratio must not pass either
    assert discrimination_verdict(0.76, 0.99)["verdict"] == "FAIL"


def test_verdict_floors_come_from_the_config_block():
    cfg = GenBridgeConfig()
    assert (cfg.disc_floor, cfg.disc_abs_floor) == (
        DEFAULT_DISC_FLOOR,
        DEFAULT_DISC_ABS_FLOOR,
    )
    assert cfg.disc_min_negatives == DEFAULT_DISC_MIN_NEGATIVES
    v = discrimination_verdict(0.9, 0.9, cfg.disc_floor, cfg.disc_abs_floor)
    assert (v["floor"], v["abs_floor"]) == (0.8, 0.75)


def test_verdict_of_a_zero_normaliser_is_a_fail_not_a_crash():
    assert discrimination_verdict(0.9, 0.0)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# 5. the probe's probabilities agree with its argmax
# ---------------------------------------------------------------------------


def _probe(cards=(2, 3, 4), seed=0):
    return FactorProbe((3, 16, 16), (4, 8), 2, cards, jax.random.PRNGKey(seed))


def test_factor_probs_argmax_is_the_predict_factors_column():
    """⛔ AUROC must be read off the SAME head the balanced accuracy argmaxes."""
    probe = _probe()
    imgs = np.random.default_rng(0).random((7, 3, 16, 16)).astype(np.float32)
    preds = predict_factors(probe, imgs, batch=4)
    for j in (0, 1, 2):
        p = factor_probs(probe, imgs, j, batch=4)
        assert p.shape == (7, probe.heads.linears[j].out_features)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(np.argmax(p, axis=1), preds[:, j])


def test_factor_probs_on_an_empty_batch():
    assert factor_probs(_probe(), np.zeros((0, 3, 16, 16), np.float32), 0).shape[0] == 0


@pytest.mark.parametrize("x64", [False, True])
def test_factor_probs_reads_a_float32_input_under_x64(x64):
    """⚠ §7.23: x64 flipped at module import must not break the new path."""
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", x64)
        probe = _probe(seed=1)
        p = factor_probs(probe, np.zeros((3, 3, 16, 16), np.float32), 1, batch=2)
        assert p.shape == (3, 3) and np.isfinite(p).all()
        assert isinstance(eqx.filter_jit(jax.vmap(probe)), object)
        assert float(jnp.sum(jnp.asarray(p))) == pytest.approx(3.0, abs=1e-4)
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# 6. the condition passthrough (⛔ Montero's behaviour must not move)
# ---------------------------------------------------------------------------


def test_the_default_condition_is_monteros():
    assert DEFAULT_CONDITION == "recomb2element"
    assert condition_of(get_default_config()) == DEFAULT_CONDITION


def test_artifacts_tag_is_back_compatible_for_the_headline_and_separate_otherwise():
    """⛔ Two conditions must never write into the same run directory."""
    assert artifacts_tag("shapes3d", "recon", DEFAULT_CONDITION) == "shapes3d_recon"
    assert (
        artifacts_tag("shapes3d", "recon", "recomb2element_wide")
        == "shapes3d_recon_recomb2element_wide"
    )


def test_condition_flag_defaults_to_leaving_the_config_alone():
    args = build_arg_parser().parse_args([])
    assert args.condition is None
    assert condition_of(_config_from_args(args)) == DEFAULT_CONDITION


def test_condition_flag_is_threaded_into_gen_data():
    args = build_arg_parser().parse_args(["--condition", "recomb2element_wide"])
    assert condition_of(_config_from_args(args)) == "recomb2element_wide"


def test_an_unknown_condition_is_refused_loudly():
    cfg = get_default_config()
    cfg.gen_data.condition = "recomb2element_widee"
    with pytest.raises(ValueError, match="unknown condition"):
        load_split_and_cells("shapes3d", cfg, log=lambda *a: None)


def test_new_disc_knobs_roundtrip_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_bridge.disc_floor = 0.71
    cfg.gen_bridge.disc_abs_floor = 0.66
    cfg.gen_bridge.disc_min_negatives = 5
    p = tmp_path / "config.yaml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.gen_bridge.disc_floor == 0.71
    assert back.gen_bridge.disc_abs_floor == 0.66
    assert back.gen_bridge.disc_min_negatives == 5


def test_defaults_preserve_the_shipped_bridge_behaviour():
    """⛔ Adding the discrimination knobs must not move the bridge's defaults."""
    d = GenBridgeConfig()
    assert (d.cache_scope, d.dec_recipe, d.ceiling_floor) == ("full", "baseline", 0.8)
    assert d.seeds == [0, 1, 2] and d.n_train == 8192 and d.n_eval_a == 1024


# ---------------------------------------------------------------------------
# 7. ⭐ the per-cell no-leak assertion — and its positive control
# ---------------------------------------------------------------------------


def _toy():
    grid = FactorGrid(
        names=("a", "b"),
        values=(np.arange(4, dtype=float), np.arange(3, dtype=float)),
    )
    cells = [
        FactorCell(
            "c0", {"a": (0.0,), "b": (0.0,)}, meta={"object_bg_hue_contrast": 0.1}
        ),
        FactorCell(
            "c1", {"a": (3.0,), "b": (2.0,)}, meta={"object_bg_hue_contrast": 0.4}
        ),
    ]
    split = build_cell_split(
        grid, cells, dataset="toy", condition="toy_wide", variant="v"
    )
    return grid, cells, split


def test_per_cell_no_leak_counts_every_cell():
    grid, cells, split = _toy()
    out = per_cell_no_leak(split, cells, split.train_rows)
    assert [c["cell_id"] for c in out] == ["c0", "c1"]
    assert [c["n_rows"] for c in out] == [1, 1]
    assert sum(c["n_leaked"] for c in out) == 0
    assert out[1]["contrast"] == 0.4


def test_per_cell_no_leak_FIRES_for_every_cell_in_turn():
    """⭐ The positive control.  An OR-predicate whose second disjunct is wrong
    passes a one-row check; each cell is corrupted in turn and must be named."""
    grid, cells, split = _toy()
    for cell in cells:
        bad = np.concatenate([split.train_rows, cell.rows(grid)])
        with pytest.raises(HoldoutViolation, match=cell.cell_id):
            per_cell_no_leak(split, cells, bad)


# ---------------------------------------------------------------------------
# 8. real Shapes3D: Montero's configuration is untouched by the wide one
# ---------------------------------------------------------------------------

pytest.importorskip("h5py", reason="Shapes3D access needs h5py")


def _has_shapes3d():
    from chlu.data import shapes3d as s3d

    try:
        return s3d.shapes3d_root(None).joinpath("3dshapes.h5").exists()
    except Exception:  # pragma: no cover - environment without the dataset
        return False


needs_data = pytest.mark.skipif(not _has_shapes3d(), reason="Shapes3D h5 not present")


@needs_data
def test_monteros_split_is_what_the_bridge_still_loads_by_default():
    """⛔ If this moves, every number in gen-feature-bridge / gen-decoder-lift
    stops describing the configuration they were measured on."""
    from chlu.data import shapes3d as s3d

    cfg = get_default_config()
    split, cells = load_split_and_cells("shapes3d", cfg, log=lambda *a: None)
    ref, _ = s3d.build_split(None)
    assert cells is None
    assert split.condition == "recomb2element"
    assert (split.n_test, split.n_train) == (72, 479928)
    assert np.array_equal(split.test_rows, ref.test_rows)
    assert np.array_equal(split.train_rows, ref.train_rows)


@needs_data
def test_the_wide_condition_is_a_separate_configuration_with_its_own_cells():
    cfg = get_default_config()
    cfg.gen_data.condition = "recomb2element_wide"
    split, cells = load_split_and_cells("shapes3d", cfg, log=lambda *a: None)
    assert len(cells) == 13 and split.n_test == 936
    assert all(c.rows(split.grid).size == 72 for c in cells)
    # ⛔ the wide hold-out is a strict SUPERSET of Montero's 72
    from chlu.data import shapes3d as s3d

    ref, _ = s3d.build_split(None)
    assert np.setdiff1d(ref.test_rows, split.test_rows).size == 0


@needs_data
def test_selected_fitting_rows_touch_no_wide_cell():
    """⭐ The exclusion that makes the wide ceiling a measurement of
    generalisation rather than of memorisation."""
    from chlu.experiments.exp_gen_bridge import select_rows

    cfg = get_default_config()
    cfg.gen_data.condition = "recomb2element_wide"
    split, cells = load_split_and_cells("shapes3d", cfg, log=lambda *a: None)
    rows = select_rows(split, cfg.gen_bridge, seed=0)
    out = per_cell_no_leak(split, cells, rows["train"])
    assert sum(c["n_leaked"] for c in out) == 0
    assert sum(c["n_excluded"] for c in out) == 936
