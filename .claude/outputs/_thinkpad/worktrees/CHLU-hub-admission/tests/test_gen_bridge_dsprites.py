"""dSprites — the SECOND dataset: the traps, the populations, and the gate.

⭐ Three groups of load-bearing tests:

1. **The traps.**  The dSprites split has four documented ways to fail silently
   (T1 colour-column slice, T2 the deliberately swapped angle constants, T3 the
   float boundaries, T4 ``latent_classes`` vs ``latent_values``), and every one of
   them still *typechecks*.  The tests here pin the realised counts against R11
   (**737,280 / 6,292 / 730,988**) and assert the corner mask is the held-out
   filter with exactly one clause removed — so the verbatim upstream filter stays
   verbatim and the new mask cannot drift from it.
2. **The pixel scale**, which is a trap the loader did not have a name for:
   dSprites' ``imgs`` is a **binary mask stored as uint8 with values {0, 1}**, so
   the shipped ``/255`` produced an essentially black dataset that trained and
   evaluated without a single error.  ⛔ Shapes3D's divisor must not move.
3. **The Shapes3D path is bit-identical.**  Every number in this phase rests on
   it; these changes are additive and that has to be mechanical, not editorial.

Everything runs on synthetic tensors except the tests marked ``needs_dsprites``.
⚠ Run with ``CHLU_GEN_DATA_ROOT`` set or the real-data tests **silently skip**.
"""

import numpy as np
import pytest

from chlu.config import (
    GenBridgeConfig,
    GenBridgeDspritesConfig,
    get_default_config,
    load_config,
    save_config,
)
from chlu.core.gen_bridge import (
    DECODER_RECIPES,
    IMAGE_KEYS,
    IMAGE_SHAPES,
    UINT8_SCALES,
    resolve_decoder_cfg,
)
from chlu.data import dsprites as dsp
from chlu.eval.gen_probe import discrimination_metrics, discrimination_verdict
from chlu.experiments.exp_gen_bridge import (
    apply_dsprites_profile,
    apply_quick,
    build_arg_parser,
    dsprites_populations,
    run_discrimination_stage,
)


class _FakeSplit:
    def __init__(self, test_rows, grid=None):
        self.test_rows = np.asarray(test_rows, np.int64)
        self.grid = grid


def _toy_latents(
    n_shapes=3, scales=(0.5, 0.7), rots=(0.0, np.pi), txs=(0.0, 0.9), tys=(0.0, 0.9)
):
    """Small exhaustive (values, classes) grid in dSprites' post-slice layout."""
    vals, cls = [], []
    for si in range(n_shapes):
        for sc in scales:
            for ro in rots:
                for x in txs:
                    for y in tys:
                        vals.append([float(si + 1), sc, ro, x, y])
                        cls.append(
                            [
                                si,
                                scales.index(sc),
                                rots.index(ro),
                                txs.index(x),
                                tys.index(y),
                            ]
                        )
    return np.asarray(vals, np.float64), np.asarray(cls, np.int64)


# ---------------------------------------------------------------------------
# 1. the traps
# ---------------------------------------------------------------------------


def test_T2_angle_constants_are_swapped_and_must_stay_that_way():
    """⛔ ``a120`` is 240° and ``a240`` is 120°.  'Fixing' it inverts the band."""
    assert np.isclose(np.degrees(dsp.a120), 240.0)
    assert np.isclose(np.degrees(dsp.a240), 120.0)
    # the clause therefore admits [120, 240], not the empty set it looks like
    band = np.radians([90.0, 130.0, 180.0, 235.0, 250.0])
    admitted = (band < dsp.a120) & (band > dsp.a240)
    assert admitted.tolist() == [False, True, True, True, False]


def test_T1_corner_mask_rejects_the_raw_six_column_arrays():
    """A 6-column (pre-slice) array must raise, never silently select nothing."""
    raw = np.zeros((4, 6), np.float64)
    with pytest.raises(ValueError, match="5 columns"):
        dsp.montero_corner_mask(raw)
    with pytest.raises(ValueError, match="5 columns"):
        dsp.montero_position_corner_mask(raw)


def test_corner_mask_is_the_test_filter_with_exactly_one_clause_removed():
    """⭐ The new mask cannot drift from the verbatim filter — on synthetic data."""
    v, c = _toy_latents(
        rots=(0.0, np.radians(180.0)), txs=(0.0, 0.9), tys=(0.0, 0.9), scales=(0.5, 0.7)
    )
    lhs = dsp.montero_test_mask(v, c)
    rhs = (c[:, dsp.shp] == dsp.HELDOUT_SHAPE_CLASS) & dsp.montero_corner_mask(v)
    assert np.array_equal(lhs, rhs)
    assert lhs.sum() > 0  # the toy grid actually exercises the cell


def test_position_corner_is_a_superset_of_the_full_corner():
    v, _ = _toy_latents()
    assert np.all(dsp.montero_position_corner_mask(v) | ~dsp.montero_corner_mask(v))


# ---------------------------------------------------------------------------
# 2. the pixel scale
# ---------------------------------------------------------------------------


def test_uint8_scale_is_per_dataset_and_shapes3d_is_unchanged():
    """⛔ Shapes3D stays at 255.0; dSprites' binary mask must not be /255."""
    assert UINT8_SCALES["shapes3d"] == 255.0
    assert UINT8_SCALES["dsprites"] == 1.0
    assert set(UINT8_SCALES) == set(IMAGE_SHAPES) == set(IMAGE_KEYS)


# ---------------------------------------------------------------------------
# 3. the populations (R16's matched-seen + R20's in-corner metric)
# ---------------------------------------------------------------------------


def _pops_on_toy(fitted=(), seed=0, **kw):
    v, c = _toy_latents(**kw)
    test = np.where(dsp.montero_test_mask(v, c))[0]
    dcfg = GenBridgeDspritesConfig(
        max_positives=0, max_negatives=0, max_seen=0, min_negatives=1
    )
    return (
        dsprites_populations(
            _FakeSplit(test), v, c, np.asarray(fitted, np.int64), dcfg, seed
        ),
        v,
        c,
        test,
    )


def test_populations_are_disjoint_from_heldout_and_from_the_fitted_rows():
    pops, v, c, test = _pops_on_toy()
    assert np.array_equal(pops["positives"], np.sort(test))
    assert np.intersect1d(pops["negatives"], test).size == 0
    assert np.intersect1d(pops["seen"], test).size == 0
    assert np.intersect1d(pops["negatives"], pops["seen"]).size == 0
    # negatives are in the corner and are NOT the target shape
    assert np.all(dsp.montero_corner_mask(v)[pops["negatives"]])
    assert np.all(c[pops["negatives"], dsp.shp] != dsp.HELDOUT_SHAPE_CLASS)
    # seen positives ARE the target shape, in the position corner, outside the cell
    assert np.all(c[pops["seen"], dsp.shp] == dsp.HELDOUT_SHAPE_CLASS)
    assert np.all(dsp.montero_position_corner_mask(v)[pops["seen"]])
    assert not np.any(dsp.montero_corner_mask(v)[pops["seen"]])


def test_populations_POSITIVE_CONTROL_a_fitted_negative_raises():
    """⭐ The exclusion assertion must be shown ABLE TO FIRE, not merely to pass."""
    pops, v, c, test = _pops_on_toy()
    smuggled = int(pops["negatives"][0])
    dcfg = GenBridgeDspritesConfig(
        max_positives=0, max_negatives=0, max_seen=0, min_negatives=1
    )
    # a fitted row is excluded by construction, so the pool simply shrinks …
    ok, _, _, _ = _pops_on_toy(fitted=(smuggled,))
    assert smuggled not in ok["negatives"].tolist()
    # … and a held-out row inside the fitted set is an R3.3 leak
    with pytest.raises(AssertionError, match="R3.3 leak"):
        dsprites_populations(
            _FakeSplit(test), v, c, np.asarray([int(test[0])], np.int64), dcfg, 0
        )


def test_populations_refuse_too_few_negatives():
    v, c = _toy_latents()
    test = np.where(dsp.montero_test_mask(v, c))[0]
    corner = np.where(dsp.montero_corner_mask(v) & (c[:, dsp.shp] != 1))[0]
    dcfg = GenBridgeDspritesConfig(
        max_positives=0, max_negatives=0, max_seen=0, min_negatives=10_000
    )
    with pytest.raises(AssertionError, match="not a measurement"):
        dsprites_populations(_FakeSplit(test), v, c, np.zeros(0, np.int64), dcfg, 0)
    assert corner.size < 10_000


def test_population_caps_are_seeded_and_reproducible():
    v, c = _toy_latents(
        scales=(0.5, 0.7, 0.8), rots=(0.0, np.radians(150.0), np.radians(200.0))
    )
    test = np.where(dsp.montero_test_mask(v, c))[0]
    dcfg = GenBridgeDspritesConfig(
        max_positives=0, max_negatives=3, max_seen=0, min_negatives=1
    )
    a = dsprites_populations(_FakeSplit(test), v, c, np.zeros(0, np.int64), dcfg, 7)
    b = dsprites_populations(_FakeSplit(test), v, c, np.zeros(0, np.int64), dcfg, 7)
    d = dsprites_populations(_FakeSplit(test), v, c, np.zeros(0, np.int64), dcfg, 8)
    assert a["negatives"].size == 3
    assert np.array_equal(a["negatives"], b["negatives"])
    assert not np.array_equal(a["negatives"], d["negatives"])


def test_the_gate_statistic_is_balanced_accuracy_not_detection():
    """⛔ One-class cell: 'answer ellipse to everything' must score exactly 0.5."""
    const = np.tile(np.array([[0.01, 0.98, 0.01]]), (40, 1))
    m = discrimination_metrics(const, const, target=dsp.HELDOUT_SHAPE_CLASS)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["detection_rate"] == pytest.approx(1.0)  # ⛔ the degenerate number
    assert discrimination_verdict(m["balanced_accuracy"], 0.98)["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# 4. the profile: the WINNING recipe, pinned and asserted
# ---------------------------------------------------------------------------


def test_profile_selects_the_recipe_that_PASSED_not_the_one_that_failed():
    cfg = get_default_config()
    assert cfg.gen_bridge.dec_recipe == "baseline"  # ⛔ the global default
    apply_dsprites_profile(cfg)
    assert cfg.gen_bridge.dec_recipe == "v1_capacity_schedule"
    rec = resolve_decoder_cfg(cfg.gen_bridge)
    assert rec.name == "v1_capacity_schedule"
    assert rec.steps == DECODER_RECIPES["v1_capacity_schedule"]["dec_steps"] == 1800
    assert rec.lr_schedule == "cosine"
    assert rec.loss == "mse"  # ⛔ NOT object_weighted (it LOST, 0.537)
    assert rec.channels == (256, 128, 64)


def test_profile_does_not_drag_the_1800_step_schedule_into_quick_mode():
    cfg = apply_dsprites_profile(apply_quick(get_default_config()), quick=True)
    assert resolve_decoder_cfg(cfg.gen_bridge).steps == cfg.gen_bridge.quick_dec_steps


def test_profile_leaves_the_shapes3d_defaults_untouched():
    """⛔ The Shapes3D path must be bit-identical before and after."""
    before = GenBridgeConfig()
    cfg = get_default_config()
    apply_dsprites_profile(cfg)  # dSprites only — mutates its own config object
    fresh = get_default_config().gen_bridge
    assert fresh.dec_recipe == before.dec_recipe == "baseline"
    assert fresh.cache_block == before.cache_block == 8192
    assert resolve_decoder_cfg(fresh).as_dict() == resolve_decoder_cfg(before).as_dict()


def test_discrimination_stage_refuses_shapes3d():
    with pytest.raises(NotImplementedError, match="Return to the Hub"):
        run_discrimination_stage("shapes3d", "recon", 0, ".", get_default_config())


def test_cli_exposes_the_discrimination_stage():
    args = build_arg_parser().parse_args(
        ["--dataset", "dsprites", "--stage", "discrimination", "--seed", "1"]
    )
    assert args.stage == "discrimination"
    assert args.dataset == "dsprites"


def test_config_block_round_trips_through_yaml(tmp_path):
    cfg = get_default_config()
    cfg.gen_bridge_dsprites.max_negatives = 123
    cfg.gen_bridge_dsprites.disc_floor = 0.85
    path = tmp_path / "c.yaml"
    save_config(cfg, path)
    back = load_config(path)
    assert back.gen_bridge_dsprites.max_negatives == 123
    assert back.gen_bridge_dsprites.disc_floor == 0.85
    assert back.gen_bridge_dsprites.expected_decoder_recipe == "v1_capacity_schedule"


# ---------------------------------------------------------------------------
# 5. real data (⚠ needs CHLU_GEN_DATA_ROOT; otherwise these SKIP silently)
# ---------------------------------------------------------------------------

pytest.importorskip("h5py", reason="dSprites image access needs h5py")

_NPZ = dsp.dsprites_root() / dsp.NPZ_NAME
_H5 = dsp.dsprites_root() / dsp.HDF5_NAME
needs_dsprites = pytest.mark.skipif(
    not _NPZ.exists(), reason=f"dSprites npz not at {_NPZ} (set CHLU_GEN_DATA_ROOT)"
)
needs_dsprites_images = pytest.mark.skipif(
    not _H5.exists(), reason=f"dSprites hdf5 not at {_H5} (set CHLU_GEN_DATA_ROOT)"
)


@needs_dsprites
def test_real_realised_counts_match_R11():
    """⭐ 737,280 / 6,292 / 730,988 — if this moves, every number downstream is void."""
    lv, lc = dsp.load_latents()
    m = dsp.montero_test_mask(lv, lc)
    assert lv.shape == (737280, 5)
    assert int(m.sum()) == dsp.N_TEST_REALISED == 6292
    assert int((~m).sum()) == 730988
    assert np.array_equal(
        m, (lc[:, dsp.shp] == dsp.HELDOUT_SHAPE_CLASS) & dsp.montero_corner_mask(lv)
    )


@needs_dsprites
def test_real_corner_and_matched_seen_pool_sizes():
    """The matched-seen corner: 18,876 total = 3 x 6,292, one per shape."""
    lv, lc = dsp.load_latents()
    corner = dsp.montero_corner_mask(lv)
    assert int(corner.sum()) == 18876
    for k in (0, 1, 2):
        assert int((corner & (lc[:, dsp.shp] == k)).sum()) == 6292
    assert int(
        (dsp.montero_position_corner_mask(lv) & (lc[:, dsp.shp] == 1)).sum()
    ) == (22748 + 6292)


@needs_dsprites_images
def test_real_images_are_binary_and_are_not_divided_by_255():
    """⛔ The trap: /255 on a {0,1} uint8 mask yields a max pixel of 0.0039."""
    from chlu.core.gen_bridge import stream_images

    rows = np.array([0, 100_000, 400_000, 737_279], np.int64)
    imgs = stream_images("dsprites", rows)
    assert imgs.shape == (4, 1, 64, 64)
    assert set(np.unique(imgs).tolist()) <= {0.0, 1.0}
    assert float(imgs.max()) == 1.0
    assert float(imgs.mean()) > 0.001  # a /255 bug lands here at ~1e-5


@needs_dsprites
def test_real_populations_and_the_no_leak_count():
    """⭐ R3.3 on the real split, with the realised pool sizes reported."""
    split, lv, lc = dsp.build_split()
    dcfg = GenBridgeDspritesConfig(max_positives=0, max_negatives=0, max_seen=0)
    pops = dsprites_populations(split, lv, lc, np.zeros(0, np.int64), dcfg, 0)
    assert pops["positives"].size == 6292
    assert pops["pool_sizes"]["negatives_pool"] == 12584
    assert pops["pool_sizes"]["seen_pool"] == 22748
    assert pops["pool_sizes"]["corner_total"] == 18876
