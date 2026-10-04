"""⭐⭐ §11b — is the codebook exponent set by the FACTOR MANIFOLD?

⭐ The load-bearing tests here are the ones that can **fail on purpose**:

- ``test_the_shared_grid_gate_RAISES_on_a_different_range`` — the task's own named
  falsifier (*"your dSprites sweep covers a different ``M`` range than Shapes3D's;
  exponents are only comparable over comparable ranges"*).  A gate that never raises
  gates nothing, so it is shown raising.
- ``test_the_rule_is_NON_OVERLAPPING_not_merely_the_means`` — the pre-registered
  directional rule is *every* dSprites seed below *every* Shapes3D seed.  ⛔ A
  mean-only rule would call an overlapping pair a win, and this test exhibits
  exactly such a pair being called **NO**.
- ``test_a_NO_says_the_account_is_REFUTED`` — ⛔ the falsifier must be *reportable*:
  §11b says a non-steeper exponent refutes the factor-manifold account, and the
  harness has to say so in those words rather than emitting a bare ``False``.
- ``test_the_verdict_REFUSES_on_fewer_than_three_seeds`` — a ``--quick`` smoke must
  not be able to emit a directional answer.
- ``test_fit_log_log_recovers_a_planted_exponent`` — the fit that the whole spoke
  reports is shown recovering a slope it was given.
- ``test_apply_quick_keeps_the_grid_ENDPOINTS`` — the smoke exercises the real gate.

Everything runs on synthetic tensors except the tests marked ``needs_dsprites``.
⚠ Run with ``CHLU_GEN_DATA_ROOT`` set or the real-data tests **silently skip**.
"""

import numpy as np
import pytest

from chlu.config import (
    GenCodebookExponentConfig,
    gen_codebook_exponent_config,
    get_default_config,
    load_config,
    save_config,
)
from chlu.experiments.exp_gen_codebook_exponent import (
    T_CRIT_2DF,
    apply_quick,
    assert_shared_grid,
    fit_log_log,
    fit_seed_exponent,
    select_rows,
    summarize,
)

M_GRID = [4, 8, 16, 32, 64, 128, 186]


# ---------------------------------------------------------------------------
# 1. The shared-grid gate — ⛔ the task's named falsifier
# ---------------------------------------------------------------------------


def test_assert_shared_grid_accepts_the_declared_grid():
    out = assert_shared_grid(M_GRID)
    assert out["m_range"] == [4, 186]
    assert out["n_points"] == 7
    assert out["shared_by_both_datasets"] is True


@pytest.mark.parametrize("bad", [[4, 8, 16, 32, 64], [8, 16, 186], [16, 64, 256]])
def test_the_shared_grid_gate_RAISES_on_a_different_range(bad):
    """⛔ *Exponents are only comparable over comparable ranges.*"""
    with pytest.raises(AssertionError, match="NOT comparable"):
        assert_shared_grid(bad)


def test_the_shared_grid_gate_rejects_a_degenerate_grid():
    with pytest.raises(ValueError, match="distinct values"):
        assert_shared_grid([4, 186])
    with pytest.raises(ValueError, match="distinct values"):
        assert_shared_grid([4, 4, 186])


def test_the_default_grid_is_the_one_the_PREREG_declares():
    assert list(GenCodebookExponentConfig().m_grid) == M_GRID
    assert GenCodebookExponentConfig().k_wells == 5
    assert list(GenCodebookExponentConfig().datasets) == ["shapes3d", "dsprites"]


# ---------------------------------------------------------------------------
# 2. The fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [-0.2631, -0.400, -0.10])
def test_fit_log_log_recovers_a_planted_exponent(planted):
    M = np.asarray(M_GRID, np.float64)
    eps = 0.6 * M**planted
    f = fit_log_log(M, eps)
    assert f["status"] == "ok"
    assert f["fitted_exponent"] == pytest.approx(planted, abs=1e-9)
    assert f["r_squared"] == pytest.approx(1.0, abs=1e-9)
    assert f["level_at_M64"] == pytest.approx(0.6 * 64.0**planted, rel=1e-9)


def test_fit_log_log_declares_too_few_points_rather_than_inventing_a_slope():
    f = fit_log_log([64.0, 64.0], [0.3, 0.2])
    assert f["status"] == "too few distinct M"
    assert np.isnan(f["fitted_exponent"])


def _sweep(per_well_exponents, levels=None, m_grid=M_GRID):
    """A synthetic sweep whose wells carry planted exponents."""
    levels = levels or [0.6] * len(per_well_exponents)
    return [
        {
            "n_modes": int(m),
            "per_well_relative_distortion": [lv * m**a for a, lv in
                                             zip(per_well_exponents, levels, strict=True)],
            "per_well_total_variance": [1.0] * len(per_well_exponents),
            "variance_weighted_relative_distortion": float(
                np.mean([lv * m**a for a, lv in
                         zip(per_well_exponents, levels, strict=True)])),
            "all_modes_used": True,
        }
        for m in m_grid
    ]


def test_fit_seed_exponent_is_the_MEAN_over_wells():
    planted = [-0.20, -0.25, -0.30, -0.35, -0.40]
    out = fit_seed_exponent(_sweep(planted), pr=[4.0] * 5)
    assert [w["fitted_exponent"] for w in out["per_well"]] == pytest.approx(planted, abs=1e-9)
    assert out["seed_exponent"] == pytest.approx(float(np.mean(planted)), abs=1e-9)
    assert out["n_wells"] == 5
    assert out["n_monotone_wells"] == 5
    # the law's own -2/PR travels beside every well so the two can be compared
    assert out["per_well"][0]["law_exponent_minus_2_over_PR"] == pytest.approx(-0.5)


def test_the_monotone_flag_can_be_FALSE():
    """⛔ A flag that is always True measures nothing."""
    sweep = _sweep([-0.3] * 2)
    sweep[3]["per_well_relative_distortion"][0] = 9.9  # a bump at M = 32
    out = fit_seed_exponent(sweep, pr=[4.0, 4.0])
    assert out["per_well"][0]["monotone_decreasing"] is False
    assert out["per_well"][1]["monotone_decreasing"] is True
    assert out["n_monotone_wells"] == 1


# ---------------------------------------------------------------------------
# 3. The pre-registered directional rule
# ---------------------------------------------------------------------------


def _summary(shapes3d_seeds, dsprites_seeds, m_grid=M_GRID):
    cfg = get_default_config()
    c = gen_codebook_exponent_config(cfg)
    c.seeds = list(range(len(shapes3d_seeds)))
    records = {}
    for ds, exps in (("shapes3d", shapes3d_seeds), ("dsprites", dsprites_seeds)):
        recs = []
        for s, e in enumerate(exps):
            sweep = _sweep([e] * 5, m_grid=m_grid)
            recs.append({
                "dataset": ds, "seed": s, "n_factors": 6 if ds == "shapes3d" else 5,
                "factor_names": [f"f{i}" for i in range(6 if ds == "shapes3d" else 5)],
                "realised_split": {"n_total": 1, "n_test": 1, "n_train": 0},
                "n_values_realised": {"SPREAD": {f"f{i}": {"n_values_realised": 3}
                                                 for i in range(6)}},
                "partition": {"well_sizes": [1] * 5},
                "shipped_partition_check": {"bit_identical": None},
                "sweep": sweep,
                "fit": fit_seed_exponent(sweep, pr=[4.0] * 5),
                "wall_clock_s": 0.0,
            })
        records[ds] = recs
    return summarize(records, cfg)


def test_the_rule_answers_YES_when_every_seed_separates():
    out = _summary([-0.26, -0.25, -0.27], [-0.34, -0.33, -0.35])
    cmp_ = out["comparison"]
    assert cmp_["is_the_dsprites_exponent_STEEPER_non_overlapping"] == "YES"
    assert cmp_["seed_sets_non_overlapping"] is True
    assert "SURVIVES" in cmp_["reading"]


def test_the_rule_is_NON_OVERLAPPING_not_merely_the_means():
    """⛔ dSprites is steeper ON AVERAGE but the seed sets interleave ⇒ **NO**."""
    out = _summary([-0.26, -0.33, -0.27], [-0.34, -0.28, -0.35])
    cmp_ = out["comparison"]
    assert cmp_["means_order_as_predicted"] is True
    assert cmp_["seed_sets_non_overlapping"] is False
    assert cmp_["is_the_dsprites_exponent_STEEPER_non_overlapping"] == "NO"


def test_a_NO_says_the_account_is_REFUTED():
    """⭐ §11b: a non-steeper exponent REFUTES the factor-manifold account."""
    out = _summary([-0.30, -0.31, -0.29], [-0.20, -0.21, -0.19])
    cmp_ = out["comparison"]
    assert cmp_["is_the_dsprites_exponent_STEEPER_non_overlapping"] == "NO"
    assert "REFUTED" in cmp_["reading"]
    assert cmp_["delta_mean_steeper_minus_reference"] > 0


def test_the_verdict_REFUSES_on_fewer_than_three_seeds():
    """⛔ A ``--quick`` smoke may not emit a directional answer."""
    out = _summary([-0.26], [-0.40])
    cmp_ = out["comparison"]
    assert cmp_["is_the_dsprites_exponent_STEEPER_non_overlapping"] == "⛔ NOT ANSWERABLE"
    assert "NOT ANSWERABLE" in cmp_["reading"]
    assert cmp_["n_seeds"] == {"shapes3d": 1, "dsprites": 1}


def test_the_companions_are_a_DISTANCE_table_and_neither_is_promoted():
    out = _summary([-0.26, -0.25, -0.27], [-0.32, -0.31, -0.33])
    comp = out["comparison"]["companions"]
    assert comp["values"] == {"pure_factor_count": -0.400, "carried_offset": -0.316}
    assert comp["closer"] == "carried_offset"
    assert "Neither companion may be" in comp["note"]
    # ⛔ the word "prediction" appears only inside the prohibition, never as a label
    assert "the prediction" in comp["note"]


def test_the_historical_number_is_labelled_NEVER_the_comparator():
    out = _summary([-0.26, -0.25, -0.27], [-0.34, -0.33, -0.35])
    ref = out["reference_exponent_shapes3d_HISTORICAL"]
    assert ref["value"] == pytest.approx(-0.2631)
    assert "NEVER the comparator" in ref["role"]
    # ⛔ and it is not what the rule compares against
    assert out["comparison"]["shapes3d_per_seed"] == pytest.approx(
        [-0.26, -0.25, -0.27], abs=1e-9)


def test_the_seed_CI_uses_the_declared_t_quantile():
    out = _summary([-0.20, -0.30, -0.40], [-0.34, -0.33, -0.35])
    v = out["datasets"]["shapes3d"]["exponent_seed_level"]
    sd = float(np.std([-0.20, -0.30, -0.40], ddof=1))
    assert v["mean"] == pytest.approx(-0.30)
    assert v["ci95_half_width"] == pytest.approx(T_CRIT_2DF * sd / np.sqrt(3))
    assert T_CRIT_2DF == pytest.approx(4.302652729911275)


def test_the_M64_row_is_a_DECLARED_ABSENCE_when_64_is_off_the_grid():
    out = _summary([-0.26] * 3, [-0.34] * 3, m_grid=[4, 16, 186])
    row = out["datasets"]["dsprites"]["eps_at_M64_variance_weighted"]
    assert row["n"] == 0
    assert "NOT MEASURED" in row["status"]


# ---------------------------------------------------------------------------
# 4. Config / quick
# ---------------------------------------------------------------------------


def test_the_block_is_memoised_and_is_not_a_CHLUConfig_field():
    cfg = get_default_config()
    a = gen_codebook_exponent_config(cfg)
    b = gen_codebook_exponent_config(cfg)
    assert a is b
    assert "gen_codebook_exponent" not in {
        f.name for f in cfg.__dataclass_fields__.values()}


def test_the_new_block_does_not_disturb_config_round_trip(tmp_path):
    cfg = get_default_config()
    gen_codebook_exponent_config(cfg).k_wells = 7
    p = tmp_path / "c.yaml"
    save_config(cfg, p)
    back = load_config(p)
    # ⛔ the block is deliberately NOT serialised (no house wiring site); the shipped
    # fields must still round-trip untouched.
    assert back.gen_store.n_modes == cfg.gen_store.n_modes
    assert gen_codebook_exponent_config(back).k_wells == 5


def test_apply_quick_keeps_the_grid_ENDPOINTS_and_disables_only_the_partition_assert():
    cfg = get_default_config()
    c = gen_codebook_exponent_config(cfg)
    assert c.assert_partition_matches_shipped is True
    apply_quick(cfg)
    assert min(c.m_grid) == 4 and max(c.m_grid) == 186
    assert c.assert_grid_is_shared is True
    assert assert_shared_grid(c.m_grid)["m_range"] == [4, 186]
    # ⛔ the smoke shrinks n_corr and re-draws store_train, so the shipped-partition
    # check CANNOT hold there — and it must still hold in production.
    assert c.assert_partition_matches_shipped is False
    assert GenCodebookExponentConfig().assert_partition_matches_shipped is True
    assert cfg.gen_store.train_steps == c.quick_store_train_steps


def test_the_two_bridges_are_declared_separately():
    """⚠ Shapes3D's bridge and dSprites' bridge live in different directories."""
    roots = GenCodebookExponentConfig().bridge_root_by_dataset
    assert roots["shapes3d"].endswith("gen-feature-bridge/runs")
    assert roots["dsprites"].endswith("gen-bridge-dsprites/runs")
    assert roots["shapes3d"] != roots["dsprites"]


# ---------------------------------------------------------------------------
# 5. Rows
# ---------------------------------------------------------------------------


class _Split:
    """Minimal stand-in: ``select_store_rows`` reads only these two arrays."""

    def __init__(self, n=200_000, n_test=6_292):
        self.test_rows = np.arange(n_test, dtype=np.int64)
        self.train_rows = np.arange(n_test, n, dtype=np.int64)


def test_SPREAD_is_the_allocation_harness_slice_and_excludes_every_fitted_row():
    """⛔ ``SPREAD`` is ``take[n_spread_train:]``: dropping the discarded draw would
    hand this harness a DIFFERENT held-out set and the Shapes3D control would no
    longer be measured on the rows the published ε was measured on."""
    cfg = get_default_config()
    c = gen_codebook_exponent_config(cfg)
    split = _Split()
    rows = select_rows(cfg, split, seed=0)
    assert rows["SPREAD"].size == c.n_spread_test
    assert rows["n_spread_train_drawn_and_discarded"] == c.n_spread_train
    assert np.intersect1d(rows["SPREAD"], rows["store_train"]).size == 0
    assert np.intersect1d(rows["SPREAD"], split.test_rows).size == 0
    assert np.intersect1d(rows["store_train"], split.test_rows).size == 0
    assert np.all(np.diff(rows["SPREAD"]) > 0)

    # ⭐ the reference slice, recomputed here rather than trusted
    pool = np.setdiff1d(split.train_rows, rows["store_train"])
    rng = np.random.default_rng(0 * c.row_seed_mult + c.row_seed_offset)
    take = rng.permutation(pool)[: c.n_spread_train + c.n_spread_test]
    assert np.array_equal(rows["SPREAD"], np.sort(take[c.n_spread_train:]))


def test_the_rows_are_seed_dependent():
    cfg = get_default_config()
    split = _Split()
    a = select_rows(cfg, split, seed=0)["SPREAD"]
    b = select_rows(cfg, split, seed=1)["SPREAD"]
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 6. The dSprites anchors — ⛔ a wrong split voids every number downstream
# ---------------------------------------------------------------------------

pytest.importorskip("h5py", reason="dSprites image access needs h5py")

from chlu.data import dsprites as dsp  # noqa: E402

_NPZ = dsp.dsprites_root() / dsp.NPZ_NAME
needs_dsprites = pytest.mark.skipif(
    not _NPZ.exists(), reason=f"dSprites npz not at {_NPZ} (set CHLU_GEN_DATA_ROOT)"
)


@needs_dsprites
def test_the_realised_split_this_harness_runs_on_is_737280_6292_730988():
    """⭐ Acceptance 2, measured on the real file through the harness's own loader."""
    split = dsp.build_split()[0]
    assert (split.n_total, split.n_test, split.n_train) == (737_280, 6_292, 730_988)
    assert len(split.grid.names) == 5  # ⭐ five factors — the whole point of §11b
    assert tuple(split.grid.cardinalities) == (3, 6, 40, 32, 32)


@needs_dsprites
def test_the_angle_constants_are_STILL_swapped():
    """⛔ T2: ``a120 = 4π/3``, ``a240 = 2π/3``.  "Fixing" them inverts the band."""
    assert dsp.a120 == pytest.approx(4 * np.pi / 3)
    assert dsp.a240 == pytest.approx(2 * np.pi / 3)
    assert dsp.a120 > dsp.a240
