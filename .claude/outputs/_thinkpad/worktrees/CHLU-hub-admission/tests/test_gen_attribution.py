"""Three-point attribution: the probe, the two capacity bounds, and the verdict rule.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_random_code_control_is_at_chance_POSITIVE_CONTROL`` — the control that the
  whole code-level conclusion rests on.  If a probe on a 320-wide one-hot of *pure
  noise* scored above chance on held-out rows, every "the store drops it" reading
  would be probe capacity, ⛔ not store content.  The paired
  ``..._but_memorises_its_train_rows`` test shows the same probe **can** fit the
  noise in-sample, so the near-chance test number is a real generalisation floor and
  not an under-trained probe.
- ``test_lookup_bound_beats_a_linear_head_on_an_XOR_code`` — the capacity bound must
  actually be more expressive than the linear probe, or it bounds nothing.  A
  two-well parity code is linearly unreadable and lookup-readable.
- ``test_per_factor_rows_refuses_a_bare_number_on_a_one_valued_factor`` — R31's
  fourth strike, enforced structurally rather than by discipline.
- ``test_verdict_is_in_between_and_is_NOT_rounded`` — the in-between case is the one
  a tired reader rounds; it must come back in those words.

Everything runs on synthetic tensors; the real-data path is exercised by the harness
run reported in ``.claude/outputs/gen-loss-attribution.md``.
"""

import jax
import numpy as np
import pytest

from chlu.config import GenAttributionConfig, gen_attribution_config, get_default_config
from chlu.eval.gen_attribution import (
    CONTROL_POINT,
    POINTS,
    attribution_verdict,
    chance_rates,
    lookup_upper_bound,
    majority_baseline,
    mi_table,
    n_values_realised,
    onehot_codes,
    per_factor_rows,
    probe_point,
    random_codes,
    retention,
    summarize_across_seeds,
)

NAMES = ["floor_hue", "scale"]
CARDS = [4, 8]


def _probe_cfg(steps=300):
    c = GenAttributionConfig()
    c.probe_steps = int(steps)
    c.probe_batch = 128
    return c


# ---------------------------------------------------------------------------
# 1. Code representations
# ---------------------------------------------------------------------------


def test_onehot_is_one_hot_per_well_and_the_right_width():
    codes = np.array([[0, 3], [2, 3], [1, 0]], np.int64)
    oh = onehot_codes(codes, 4)
    assert oh.shape == (3, 8)
    assert np.all(oh.sum(axis=1) == 2)  # one hot per well, K = 2
    assert oh[0, 0] == 1.0 and oh[0, 4 + 3] == 1.0
    assert oh[2, 1] == 1.0 and oh[2, 4 + 0] == 1.0


def test_onehot_refuses_an_out_of_range_code():
    with pytest.raises(ValueError, match="out of range"):
        onehot_codes(np.array([[0, 9]], np.int64), 4)


def test_random_codes_are_reproducible_and_in_range():
    a = random_codes(50, 5, 64, seed=7)
    b = random_codes(50, 5, 64, seed=7)
    c = random_codes(50, 5, 64, seed=8)
    assert a.shape == (50, 5)
    assert a.min() >= 0 and a.max() < 64
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# 2. ⛔ The probe and ⭐ THE CONTROL THE SPOKE LIVES OR DIES ON
# ---------------------------------------------------------------------------


def _noise_problem(n_train=1024, n_test=512, seed=0):
    """Labels independent of the code — the only honest zero-information setup."""
    rng = np.random.default_rng(seed)
    ytr = np.stack([rng.integers(0, c, n_train) for c in CARDS], axis=1).astype(np.int32)
    yte = np.stack([rng.integers(0, c, n_test) for c in CARDS], axis=1).astype(np.int32)
    ctr = random_codes(n_train, 5, 16, seed=11)
    cte = random_codes(n_test, 5, 16, seed=12)
    return ytr, yte, ctr, cte


def test_random_code_control_is_at_chance_POSITIVE_CONTROL():
    """⭐ A probe on a same-width code carrying NO information must not beat chance.

    ⛔ If this ever fails, the code-level probe is reading its own capacity and every
    "the store drops it" conclusion drawn from it is unsupported.
    """
    ytr, yte, ctr, cte = _noise_problem()
    out = probe_point(
        CONTROL_POINT, onehot_codes(ctr, 16), ytr,
        {"TEST": (onehot_codes(cte, 16), yte)}, CARDS, NAMES, _probe_cfg(),
        jax.random.PRNGKey(0),
    )
    acc = out["eval"]["TEST"]["per_factor"]
    maj = majority_baseline(ytr, yte, NAMES)
    for nm in NAMES:
        assert acc[nm] <= maj[nm] + 0.08, (nm, acc[nm], maj[nm])


def test_random_code_control_at_chance_but_memorises_its_train_rows():
    """⭐ The companion: the SAME probe fits the noise in-sample.

    Without this the near-chance test number above could just be an under-trained
    probe, which is exactly the artefact the control exists to exclude.
    """
    ytr, yte, ctr, cte = _noise_problem(n_train=256, n_test=256)
    out = probe_point(
        CONTROL_POINT, onehot_codes(ctr, 16), ytr,
        {"TEST": (onehot_codes(cte, 16), yte)}, CARDS, NAMES, _probe_cfg(800),
        jax.random.PRNGKey(0),
    )
    assert out["train"]["aggregate"] > out["eval"]["TEST"]["aggregate"] + 0.15
    assert out["loss_last"] < out["loss_first"]


def test_the_same_probe_hyperparameters_reach_every_point():
    """⛔ A different probe at each stage would measure the probes, not the pipeline."""
    ytr, yte, ctr, cte = _noise_problem(n_train=256, n_test=128)
    cfg = _probe_cfg(120)
    feats_tr = np.asarray(onehot_codes(ctr, 16), np.float32)
    feats_te = np.asarray(onehot_codes(cte, 16), np.float32)
    a = probe_point("encoder", feats_tr, ytr, {"T": (feats_te, yte)}, CARDS, NAMES,
                    cfg, jax.random.PRNGKey(3))
    b = probe_point("code", feats_tr[:, :32], ytr, {"T": (feats_te[:, :32], yte)},
                    CARDS, NAMES, cfg, jax.random.PRNGKey(3))
    for rec in (a, b):
        assert rec["probe_steps"] == cfg.probe_steps
        assert rec["probe_batch"] == cfg.probe_batch
        assert rec["probe_lr"] == cfg.probe_lr
    assert a["input_dim"] == 80 and b["input_dim"] == 32


def test_probe_reads_a_factor_that_IS_in_the_code():
    """⭐ The positive control for the instrument: a code that carries the factor."""
    rng = np.random.default_rng(5)
    y = np.stack([rng.integers(0, c, 800) for c in CARDS], axis=1).astype(np.int32)
    # well 0 == floor_hue exactly; well 1 is noise
    codes = np.stack([y[:, 0], rng.integers(0, 8, 800)], axis=1)
    oh = onehot_codes(codes, 8)
    out = probe_point("code", oh[:600], y[:600], {"T": (oh[600:], y[600:])},
                      CARDS, NAMES, _probe_cfg(600), jax.random.PRNGKey(1))
    assert out["eval"]["T"]["per_factor"]["floor_hue"] > 0.95
    assert out["eval"]["T"]["per_factor"]["scale"] < 0.30


# ---------------------------------------------------------------------------
# 3. ⭐ The capacity bound — it must actually bound something
# ---------------------------------------------------------------------------


def test_lookup_bound_beats_a_linear_head_on_an_XOR_code():
    """A parity-of-two-wells label is unreadable by an additive linear head on the
    one-hot and trivially readable by a tuple lookup.  ⭐ That gap is the reason the
    bound exists: it makes the probe-capacity objection measurable instead of
    arguable."""
    rng = np.random.default_rng(2)
    n = 2400
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    y = np.stack([(a ^ b), rng.integers(0, 8, n)], axis=1).astype(np.int32)
    codes = np.stack([a, b], axis=1)
    oh = onehot_codes(codes, 2)
    lin = probe_point("code", oh[:1800], y[:1800], {"T": (oh[1800:], y[1800:])},
                      [2, 8], NAMES, _probe_cfg(600), jax.random.PRNGKey(4))
    lut = lookup_upper_bound(codes[:1800], y[:1800], codes[1800:], y[1800:], NAMES)
    assert lin["eval"]["T"]["per_factor"]["floor_hue"] < 0.60
    assert lut["per_factor"]["floor_hue"] > 0.95
    assert lut["coverage"] == 1.0


def test_lookup_bound_on_an_unseen_tuple_falls_back_to_the_global_majority():
    ytr = np.array([[0, 0], [0, 1], [0, 2]], np.int32)
    yte = np.array([[0, 0]], np.int32)
    ctr = np.array([[1, 1], [1, 1], [1, 1]], np.int64)
    cte = np.array([[5, 5]], np.int64)
    out = lookup_upper_bound(ctr, ytr, cte, yte, NAMES)
    assert out["coverage"] == 0.0
    assert out["n_distinct_tuples_train"] == 1
    assert out["per_factor"]["floor_hue"] == 1.0  # global majority is 0


# ---------------------------------------------------------------------------
# 4. ⛔ n_values_realised — R31's fourth strike, enforced structurally
# ---------------------------------------------------------------------------


def test_n_values_realised_counts_the_realised_values_not_the_cardinality():
    y = np.array([[0, 3], [0, 4], [0, 3]], np.int64)
    nvr = n_values_realised(y, NAMES, CARDS)
    assert nvr["floor_hue"]["n_values_realised"] == 1
    assert nvr["floor_hue"]["cardinality"] == 4
    assert nvr["scale"]["n_values_realised"] == 2


def test_per_factor_rows_refuses_a_bare_number_on_a_one_valued_factor():
    """⚠ Montero's corner pins ``shape`` to ONE value and yields a confident-looking
    number.  The row must be stamped, ⛔ never emitted bare."""
    y = np.array([[0, 3], [0, 4]], np.int64)
    nvr = n_values_realised(y, NAMES, CARDS)
    acc = {"per_factor": {"floor_hue": 1.0, "scale": 0.5}, "aggregate": 0.75}
    rows = per_factor_rows(acc, nvr, NAMES)
    assert rows["floor_hue"]["degenerate"] is True
    assert "NOT APPLICABLE" in rows["floor_hue"]["status"]
    assert rows["scale"]["degenerate"] is False
    assert "status" not in rows["scale"]


# ---------------------------------------------------------------------------
# 5. Retention, the free instrument, and the verdict rule
# ---------------------------------------------------------------------------


def test_retention_is_nan_when_the_encoder_is_at_chance():
    assert np.isnan(retention(0.2, 0.125, 0.125))
    assert retention(0.5, 0.9, 0.1) == pytest.approx(0.5)
    assert retention(0.9, 0.9, 0.1) == pytest.approx(1.0)


def test_mi_table_reports_a_flat_row_for_a_factor_the_code_ignores():
    """⭐ The free instrument: a flat row IS the collapse, seen from the address
    book's own side."""
    rng = np.random.default_rng(3)
    n = 4000
    y = np.stack([rng.integers(0, 4, n), rng.integers(0, 8, n)], axis=1).astype(np.int64)
    codes = np.stack([y[:, 0], y[:, 0]], axis=1)  # carries floor_hue, ignores scale
    tab = mi_table(codes, y, CARDS, NAMES)
    assert tab["per_factor"]["floor_hue"]["max_fraction_of_entropy"] > 0.9
    assert tab["per_factor"]["scale"]["max_fraction_of_entropy"] < 0.1
    assert len(tab["mi_bits"]) == 2


def test_verdict_world_A_requires_BOTH_clauses():
    v = attribution_verdict(0.95, 0.13, 0.125, 0.13, 0.128)
    assert v["verdict"] == "World A"
    # encoder has it, but the code does NOT collapse -> ⛔ not World A
    v2 = attribution_verdict(0.95, 0.70, 0.125, 0.13, 0.128)
    assert v2["verdict"] == "in-between"


def test_verdict_world_B_fires_when_the_encoder_lacks_it():
    v = attribution_verdict(0.30, 0.13, 0.125, 0.13, 0.128)
    assert v["verdict"] == "World B"
    assert v["encoder_lacks_it"] is True
    assert v["world_b_threshold"] == pytest.approx(0.5125)


def test_verdict_is_in_between_and_is_NOT_rounded():
    """⛔ The case the pre-registration says must be reported in those words.

    Both sub-cases are in-between and ⛔ neither is rounded: the encoder sits above
    the World-B threshold but below the 0.90 bar, so World A is unavailable however
    flat the code is.  Only the ``leaning`` field moves.
    """
    # code within slack of the no-information floor -> leans World A
    v = attribution_verdict(0.75, 0.16, 0.125, 0.13, 0.128)
    assert v["verdict"] == "in-between"
    assert v["leaning"] == "World A"
    assert v["encoder_has_it"] is False and v["encoder_lacks_it"] is False
    # code ABOVE the floor by more than the slack -> ⛔ leans neither way
    v2 = attribution_verdict(0.75, 0.20, 0.125, 0.13, 0.128)
    assert v2["verdict"] == "in-between"
    assert v2["code_at_chance"] is False
    assert v2["leaning"] == "neither"


def test_verdict_code_floor_is_the_LARGER_of_majority_and_random_code():
    """⛔ On an unbalanced row set the majority rate, not 1/C, is the no-information
    floor; the random-code control can be higher still."""
    v = attribution_verdict(0.95, 0.42, 0.125, 0.40, 0.128)
    assert v["code_floor_used"] == pytest.approx(0.40)
    assert v["verdict"] == "World A"
    v2 = attribution_verdict(0.95, 0.42, 0.125, 0.13, 0.40)
    assert v2["code_floor_used"] == pytest.approx(0.40)


def test_chance_and_majority_are_different_quantities():
    ch = chance_rates(CARDS, NAMES)
    assert ch["floor_hue"] == pytest.approx(0.25)
    ytr = np.array([[0, 0], [0, 1], [0, 2], [1, 3]], np.int64)
    yte = np.array([[0, 0], [0, 1], [1, 2]], np.int64)
    maj = majority_baseline(ytr, yte, NAMES)
    assert maj["floor_hue"] == pytest.approx(2 / 3)


def test_summarize_declares_ddof_zero():
    """⚠ R31 carried a convention clash forward; the summary declares its own."""
    s = summarize_across_seeds([0.1, 0.2, 0.3])
    assert s["ddof"] == 0
    assert s["std"] == pytest.approx(np.std([0.1, 0.2, 0.3]))
    assert s["n_seeds"] == 3
    s2 = summarize_across_seeds([0.1, float("nan")])
    assert s2["n_seeds"] == 1


# ---------------------------------------------------------------------------
# 6. Config block
# ---------------------------------------------------------------------------


def test_attribution_config_is_memoised_on_the_house_config():
    cfg = get_default_config()
    a = gen_attribution_config(cfg)
    b = gen_attribution_config(cfg)
    assert a is b
    assert a.seeds == [0, 1, 2]
    assert a.verdict_factors == ["scale", "orientation"]


def test_attribution_config_overrides_apply_and_none_is_ignored():
    cfg = get_default_config()
    a = gen_attribution_config(cfg, probe_steps=77, probe_batch=None)
    assert a.probe_steps == 77
    assert a.probe_batch == GenAttributionConfig().probe_batch


def test_points_tuple_is_the_pipeline_order():
    assert POINTS == ("encoder", "well", "code")
    assert CONTROL_POINT == "code_random"
