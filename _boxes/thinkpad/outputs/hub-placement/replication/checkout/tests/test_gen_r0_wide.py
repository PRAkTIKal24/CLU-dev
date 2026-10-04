"""R0 re-fitted on the WIDE cache (R24): the grouping-agreement metric and THE GATE.

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_read_margin_gate_POSITIVE_CONTROL_fires_on_a_divergence`` — a wide/Montero
  margin pair that genuinely diverges must be called a **PLUMBING FAULT**.  A gate
  never seen to fire is not a gate, and "the two feature caches are not the same
  encoder's" is exactly the failure R24 rider 3 exists to catch.
- ``test_the_wide_entry_point_is_the_shipped_path_bit_for_bit`` — ⛔ the wide
  entry point must be a pure wrapper: Montero's R0 fit (grouping **and** trained
  store) has to come out **bit-identical** through the new call, or the
  compositional-results table and the ladder stop being comparable.
- ``test_fit_grouping_from_plan_pins_k_and_never_re_derives_it`` — ``K`` is a
  ladder constant; a rung that re-derives it to make its own grouping fit better
  is refused.

Everything runs on synthetic tensors; the real-data path is exercised by the
harness runs reported in ``.claude/outputs/gen-r0-wide-refit.md``.

⛔ Wells are integer-indexed here as everywhere else.
"""

import jax
import numpy as np
import pytest

from chlu.config import GenR0WideConfig, get_default_config, load_config, save_config
from chlu.core.gen_store import (
    adjusted_rand_index,
    assert_plan_matched,
    channel_correlation,
    choose_k,
    dead_channels,
    fit_grouping_from_plan,
    grouping_agreement,
    grouping_labels,
    init_store,
    read_margin_gate,
    save_store_plan,
    store_bytes,
    train_store,
    well_channels_from_labels,
)


class _Cfg:
    train_steps = 25
    train_batch = 64
    train_lr = 3e-3
    commitment_beta = 0.25
    tau_start = 1.0
    tau_end = 0.1


def _blocky(n=600, d=32, k=4, noise=0.05, seed=0):
    """``k`` planted blocks of correlated coordinates (the shipped test fixture)."""
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, k))
    W = np.zeros((k, d))
    per = d // k
    for j in range(k):
        W[j, j * per : (j + 1) * per] = 1.0
    return (lat @ W + noise * rng.normal(size=(n, d))).astype(np.float32)


def _plan(k=4, d=32, n_modes=8, labels=None):
    return {
        "schema": "chlu-gen-store-plan/1",
        "K": int(k),
        "n_modes": int(n_modes),
        "D": int(d),
        "feat_shape": [int(d)],
        "labels": list(range(int(d))) if labels is None else [int(x) for x in labels],
        "store_bytes": {"store_bytes": int(n_modes) * int(d) * 4},
    }


# ---------------------------------------------------------------------------
# the agreement metric (R24 rider 1)
# ---------------------------------------------------------------------------


def test_adjusted_rand_index_matches_sklearn():
    """⛔ Implemented here rather than imported, so it is checked against the
    reference implementation instead of trusted."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(0)
    for frac in (0.0, 0.3, 0.6, 0.9):
        a = rng.integers(0, 5, 500)
        b = np.where(rng.random(500) < frac, a, rng.integers(0, 5, 500))
        assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)


def test_adjusted_rand_index_is_invariant_to_relabelling_the_wells():
    """⭐ Wells carry integer indices and nothing else, so the agreement measure
    must not care which index a well was given."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 400)
    assert adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert adjusted_rand_index(a, (a + 3) % 5) == pytest.approx(1.0)


def test_adjusted_rand_index_is_about_zero_for_independent_partitions():
    rng = np.random.default_rng(2)
    vals = [adjusted_rand_index(rng.integers(0, 5, 2048), rng.integers(0, 5, 2048)) for _ in range(5)]
    assert max(abs(v) for v in vals) < 0.02


def test_grouping_agreement_counts_the_channels_that_moved():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, 2048)
    b = a.copy()
    moved = rng.choice(2048, size=200, replace=False)
    b[moved] = (b[moved] + 1) % 5
    out = grouping_agreement(a, b, "montero", "wide")
    assert out["n_channels"] == 2048
    assert out["n_wells_a"] == out["n_wells_b"] == 5
    assert out["n_channels_moved"] == 200
    assert out["best_permutation_agreement"] == pytest.approx(1848 / 2048)
    assert 0.0 < out["adjusted_rand_index"] < 1.0
    assert sum(sum(r) for r in out["contingency"]) == 2048
    assert out["name_a"] == "montero" and out["name_b"] == "wide"


def test_grouping_agreement_refuses_partitions_of_different_widths():
    with pytest.raises(ValueError, match="different numbers of coordinates"):
        grouping_agreement(np.zeros(10, int), np.zeros(11, int))


# ---------------------------------------------------------------------------
# the wide-condition entry point
# ---------------------------------------------------------------------------


def test_the_wide_entry_point_is_the_shipped_path_bit_for_bit():
    """⛔ ACCEPTANCE 6: Montero R0 must reproduce bit-identically through the new
    entry point — the grouping **and** the store trained on top of it."""
    X = _blocky(n=400, d=32, k=4, seed=7)
    plan = _plan(k=4, d=32)
    for grouping in ("correlation", "random"):
        shipped = grouping_labels(
            channel_correlation(X), 4, grouping, 3, "auto",
            dead=dead_channels(X), min_well_frac=0.10,
        )[0]
        got = fit_grouping_from_plan(X, plan, grouping=grouping, seed=3)["labels"]
        assert np.array_equal(np.asarray(got), np.asarray(shipped))

        wc = well_channels_from_labels(shipped, 4)
        outs = []
        for _ in range(2):
            key = jax.random.PRNGKey(11)
            ik, tk = jax.random.split(key)
            st = init_store(X, wc, 8, (32,), ik, seed=3)
            st, _, _ = train_store(st, X, _Cfg(), tk, log=None)
            outs.append(np.concatenate([np.asarray(m).ravel() for m in st.modes]))
        assert float(np.max(np.abs(outs[0] - outs[1]))) == 0.0


def test_fit_grouping_from_plan_pins_k_and_never_re_derives_it():
    """⛔ ``K`` is a ladder constant.  The plan says 6; the data says 4; the plan wins."""
    X = _blocky(n=600, d=48, k=4, seed=5)
    assert choose_k(channel_correlation(X), 3, 8, dead=dead_channels(X))["k"] == 4
    out = fit_grouping_from_plan(X, _plan(k=6, d=48), seed=0)
    assert out["K"] == 6
    assert len(out["well_sizes"]) == 6
    assert min(out["well_sizes"]) > 0
    assert "never re-derived" in out["K_source"]


def test_fit_grouping_from_plan_refuses_a_cache_of_a_different_width():
    X = _blocky(n=200, d=32, seed=6)
    with pytest.raises(ValueError, match="not comparable"):
        fit_grouping_from_plan(X, _plan(k=4, d=64))


def test_fit_grouping_from_plan_reports_agreement_against_a_reference():
    X = _blocky(n=400, d=32, k=4, seed=8)
    ref = fit_grouping_from_plan(X, _plan(), seed=0)["labels"]
    out = fit_grouping_from_plan(X, _plan(), seed=0, reference_labels=ref)
    assert out["agreement_vs_reference"]["adjusted_rand_index"] == pytest.approx(1.0)
    assert out["agreement_vs_reference"]["n_channels_moved"] == 0


def test_the_entry_points_random_arm_keeps_the_well_size_histogram():
    """⭐ The laundering control is the same ``K`` AND the same well sizes — only
    which channel sits in which well may move."""
    X = _blocky(n=400, d=32, k=4, seed=9)
    plan = _plan()
    a = fit_grouping_from_plan(X, plan, grouping="correlation", seed=2)
    b = fit_grouping_from_plan(X, plan, grouping="random", seed=2)
    assert sorted(a["well_sizes"]) == sorted(b["well_sizes"])
    assert not np.array_equal(np.asarray(a["labels"]), np.asarray(b["labels"]))
    assert abs(grouping_agreement(a["labels"], b["labels"])["adjusted_rand_index"]) < 0.15


def test_a_wide_rung_is_a_matched_rung_when_only_the_grouping_moved(tmp_path):
    """⛔ ACCEPTANCE 2: ``K`` and the byte budget are unchanged; only ``labels`` move."""
    X = _blocky(n=300, d=32, k=4, seed=4)
    wc = well_channels_from_labels(np.arange(32) % 4, 4)
    st = init_store(X, wc, 8, (32,), jax.random.PRNGKey(0), seed=0)
    montero = save_store_plan(
        tmp_path / "store_plan.json", dataset="shapes3d", pretext="recon", k=4,
        n_modes=8, dim=32, feat_shape=(32,), labels=np.arange(32) % 4,
        k_choice={"k": 4, "rule": "test"}, grouping_method="ward",
        store_bytes_dict=store_bytes(st),
    )
    wide = dict(montero, labels=list(np.roll(np.arange(32) % 4, 5)), rung="R0-wide")
    matched = assert_plan_matched([montero, wide])
    assert matched["K"] == 4 and matched["store_bytes"] == 8 * 32 * 4
    with pytest.raises(AssertionError, match="only the grouping may change"):
        assert_plan_matched([montero, dict(wide, K=6)])


# ---------------------------------------------------------------------------
# ⛔ THE GATE (R24 rider 3)
# ---------------------------------------------------------------------------


def test_read_margin_gate_passes_a_consistent_pair():
    g = read_margin_gate([0.1377, 0.1602, 0.1533], [0.1481, 0.1496, 0.1536])
    assert g["verdict"] == "CONSISTENT" and g["consistent"] is True
    assert g["delta"] == pytest.approx(0.1504 - 0.15043333, abs=1e-3)
    assert g["delta_in_pooled_spreads"] < 3.0


def test_read_margin_gate_POSITIVE_CONTROL_fires_on_a_divergence():
    """⛔ The falsifier: two caches that are NOT the same encoder's.  The gate must
    call it a plumbing fault and STOP, never report it as a result."""
    g = read_margin_gate([0.52, 0.50, 0.51], [0.1377, 0.1602, 0.1533])
    assert g["verdict"] == "PLUMBING FAULT — STOP"
    assert g["consistent"] is False
    assert g["delta_in_pooled_spreads"] > 3.0
    assert "never a property of the split" in g["note"]


def test_read_margin_gate_tolerance_is_a_declared_knob_not_a_magic_number():
    margins_w, margins_m = [0.20, 0.21, 0.19], [0.15, 0.16, 0.14]
    assert read_margin_gate(margins_w, margins_m, tolerance_sigmas=10.0)["consistent"]
    assert not read_margin_gate(margins_w, margins_m, tolerance_sigmas=0.5)["consistent"]


def test_read_margin_gate_cannot_divide_by_zero_on_an_exact_reproduction():
    g = read_margin_gate([0.15, 0.15], [0.15, 0.15])
    assert np.isfinite(g["delta_in_pooled_spreads"])
    assert g["verdict"] == "CONSISTENT"


def test_read_margin_gate_refuses_an_empty_side():
    with pytest.raises(ValueError, match="at least one seed"):
        read_margin_gate([], [0.15])


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_gen_r0_wide_defaults_select_the_decoder_lift_WINNER_not_the_baseline():
    """⛔ THE DECODER TRAP: ``gen_bridge.dec_recipe`` still defaults to the
    SUPERSEDED ``"baseline"`` (0.7191, failed the gate) so Montero's history
    reproduces.  ⭐ The wide re-fit must select the winner explicitly."""
    cfg = get_default_config()
    assert cfg.gen_bridge.dec_recipe == "baseline"
    assert cfg.gen_r0_wide.dec_recipe == "v1_capacity_schedule"
    assert cfg.gen_r0_wide.assert_dec_recipe is True


def test_gen_r0_wide_defaults_pin_the_condition_and_the_shared_corner_pair():
    w = GenR0WideConfig()
    assert w.condition == "recomb2element_wide"
    assert w.shared_corner_cells == ["montero", "wide-000"]
    assert w.exclude_shared_corner_from_pooled is True
    assert w.gate_tolerance_sigmas == 3.0
    assert w.underpowered_spread_multiple == 3.0
    assert w.seeds == [0, 1, 2]


def test_gen_r0_wide_config_round_trips_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_r0_wide.gate_tolerance_sigmas = 2.5
    cfg.gen_r0_wide.condition = "recomb2element_wide"
    cfg.gen_r0_wide.shared_corner_cells = ["montero"]
    save_config(cfg, tmp_path / "c.yaml")
    back = load_config(tmp_path / "c.yaml")
    assert back.gen_r0_wide.gate_tolerance_sigmas == 2.5
    assert back.gen_r0_wide.shared_corner_cells == ["montero"]
    assert back.gen_r0_wide.dec_recipe == "v1_capacity_schedule"


def test_the_wide_block_leaves_the_shipped_store_config_untouched():
    """⛔ Additive only: nothing in this spoke may move R0's shipped fit."""
    g = get_default_config().gen_store
    assert g.k_wells is None and g.n_modes == 64 and g.grouping_method == "auto"
    assert g.groupings == ["correlation", "random"]
    assert g.train_steps == 1200 and g.n_store_train == 32768 and g.n_corr == 16384
