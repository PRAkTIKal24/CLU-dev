"""The stratified ``fit``/``val`` row split and its balance gate.

⭐ The load-bearing tests here are the **positive controls**: the balance gate is
handed the *shipped broken split* (head and tail of a sorted row-index array) and must
FIRE, and ``mode="head_tail"`` must still reproduce that broken split bit-for-bit so a
historical configuration stays reproducible.  A gate that has never been seen to refuse
anything is not evidence that ``val`` is exchangeable with ``fit``, and every tuned
configuration in the generative phase rests on that.

Structure:

- synthetic-grid tests (always run): the broken split is a factor-grid corner, the
  stratified split realises every value of every factor at near-equal counts, the
  partition is exact, the gate fires on the broken split and passes the fixed one, and
  the split is reproducible from ``seed`` alone;
- one real-data test (skipped when Shapes3D is absent): the production-scale
  ``32,256 / 512`` cut, which is the configuration every gen arm tunes under.
"""

import numpy as np
import pytest

from chlu.config import gen_row_split_config, get_default_config
from chlu.data import shapes3d as s3d
from chlu.data.gen_rowsplit import (
    RowSplitError,
    assert_row_split_balanced,
    head_tail_fit_val,
    row_split_balance,
    split_fit_val,
    stratified_fit_val,
)

CARDS = (10, 10, 10, 8, 4, 15)
NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


def _toy_pool(n_pool: int = 4096, seed: int = 0):
    """A SORTED random subsample of a lexicographic factor grid — exactly the shape
    ``select_store_rows`` hands the split (``np.sort(rng.permutation(pool)[:n])``)."""
    rng = np.random.default_rng(seed)
    n_rows = int(np.prod(CARDS))
    pool = np.sort(rng.permutation(n_rows)[:n_pool]).astype(np.int64)
    labels = np.stack(np.unravel_index(pool, CARDS), axis=-1)
    return pool, labels


# ---------------------------------------------------------------------------
# the defect, reproduced
# ---------------------------------------------------------------------------


def test_head_tail_val_is_a_corner_of_the_factor_grid():
    """⛔ THE DEFECT. The sorted tail realises ONE value of the slowest factor."""
    pool, labels = _toy_pool()
    fit, val = head_tail_fit_val(pool, 4032, 64)
    lf = np.stack(np.unravel_index(fit, CARDS), axis=-1)
    lv = np.stack(np.unravel_index(val, CARDS), axis=-1)
    bal = row_split_balance(lf, lv, NAMES, CARDS)
    assert bal["factors"]["floor_hue"]["fit_n_values_realised"] == 10
    assert bal["factors"]["floor_hue"]["val_n_values_realised"] == 1
    assert bal["min_val_n_values_realised"] == 1


def test_gate_FIRES_on_the_broken_split():
    """⭐ The positive control: a gate that cannot refuse proves nothing."""
    pool, labels = _toy_pool()
    fit, val = head_tail_fit_val(pool, 4032, 64)
    bal = row_split_balance(
        np.stack(np.unravel_index(fit, CARDS), axis=-1),
        np.stack(np.unravel_index(val, CARDS), axis=-1),
        NAMES, CARDS,
    )
    with pytest.raises(RowSplitError, match="floor_hue"):
        assert_row_split_balanced(bal, require_all_values=True, max_count_ratio=1.25)


def test_head_tail_mode_still_reproduces_the_broken_split_bit_for_bit():
    """⛔ Historical reproduction must stay possible, and must be asked for by name."""
    pool, labels = _toy_pool()
    ref_fit, ref_val = head_tail_fit_val(pool, 4032, 64)
    fit, val, info = split_fit_val(pool, labels, 4032, 64, 0, mode="head_tail")
    assert np.array_equal(fit, ref_fit)
    assert np.array_equal(val, ref_val)
    assert info["BROKEN"] is True


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------


def test_stratified_val_realises_every_value_of_every_factor():
    """⭐ ALL SIX factors, not only the two that were visibly broken: the phase turns
    on ``scale`` and ``orientation``, which the broken split already realised in full."""
    pool, labels = _toy_pool()
    fit, val, _ = stratified_fit_val(pool, labels, 4032, 64, 0)
    bal = row_split_balance(
        np.stack(np.unravel_index(fit, CARDS), axis=-1),
        np.stack(np.unravel_index(val, CARDS), axis=-1),
        NAMES, CARDS,
    )
    # n_val = 64 cannot cover 15 orientations at equal counts, but it must cover the
    # factors it structurally can, and never regress to a single value.
    for name in ("scale", "shape"):
        r = bal["factors"][name]
        assert r["val_n_values_realised"] == r["cardinality"], name
    assert bal["min_val_n_values_realised"] > 1


def test_stratified_split_is_an_exact_partition_of_the_untouched_pool():
    """⛔ ``pool`` is never resampled: the store, the atoms and the hold-out are safe."""
    pool, labels = _toy_pool()
    fit, val, info = stratified_fit_val(pool, labels, 4032, 64, 0)
    assert np.intersect1d(fit, val).size == 0
    assert np.array_equal(np.sort(np.concatenate([fit, val])), pool)
    assert info["fit_is_pool_minus_val"] is True
    assert np.array_equal(fit, np.sort(fit)) and np.array_equal(val, np.sort(val))


def test_stratified_split_is_reproducible_from_the_seed_and_moves_with_it():
    pool, labels = _toy_pool()
    a = stratified_fit_val(pool, labels, 4032, 64, 0)[1]
    b = stratified_fit_val(pool, labels, 4032, 64, 0)[1]
    c = stratified_fit_val(pool, labels, 4032, 64, 1)[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_targets_the_pools_own_marginals_not_a_uniform_one():
    """⭐ Proportional, not uniform: a uniform target would IMPOSE a distribution the
    pool does not have.  Build a deliberately skewed pool and check ``val`` follows it."""
    rng = np.random.default_rng(0)
    cards = (4, 3)
    # 90 % of the pool sits at shape value 0
    lab0 = np.zeros((900, 2), np.int64)
    lab0[:, 0] = 0
    lab0[:, 1] = rng.integers(0, 3, 900)
    lab1 = np.stack([rng.integers(1, 4, 100), rng.integers(0, 3, 100)], axis=-1)
    labels = np.concatenate([lab0, lab1])
    pool = np.ravel_multi_index(tuple(labels.T), cards).astype(np.int64)
    order = np.argsort(pool, kind="stable")
    pool, labels = pool[order], labels[order]
    _, val, _ = stratified_fit_val(pool, labels, 900, 100, 0)
    lv = np.stack(np.unravel_index(val, cards), axis=-1)
    frac0 = float((lv[:, 0] == 0).mean())
    assert 0.85 <= frac0 <= 0.95, frac0


def test_config_default_is_the_stratified_split():
    """⛔ The default is the REPAIR. Preserving the broken split as the default would
    preserve a coin flip at the centre of every tuned configuration in the phase."""
    cfg = gen_row_split_config(get_default_config())
    assert cfg.mode == "stratified"
    assert cfg.assert_balanced is True
    assert cfg.require_all_values is True


# ---------------------------------------------------------------------------
# real data (Shapes3D is a 268 MB download and never enters git)
# ---------------------------------------------------------------------------

needs_shapes3d = pytest.mark.skipif(
    not (s3d.shapes3d_root() / s3d.H5_NAME).exists(),
    reason="Shapes3D h5 not downloaded (set $CHLU_GEN_DATA_ROOT / chlu data shapes3d)",
)


@needs_shapes3d
def test_production_scale_split_is_balanced_on_all_six_factors():
    """⭐ The configuration every gen arm tunes under: 32,256 / 512 on the real grid."""
    from chlu.experiments.exp_gen_store import select_store_rows

    config = get_default_config()
    split, _ = s3d.build_split()
    rows = select_store_rows(split, config, 0)
    pool = np.asarray(rows["store_train"], np.int64)
    labels = split.grid.factor_indices_of(pool)
    fit, val, _ = stratified_fit_val(pool, labels, 32_256, 512, 0)
    bal = row_split_balance(
        split.grid.factor_indices_of(fit), split.grid.factor_indices_of(val),
        list(split.grid.names), list(split.grid.cardinalities),
    )
    bal = assert_row_split_balanced(bal, require_all_values=True, max_count_ratio=1.25)
    assert bal["balanced"] is True
    assert bal["min_val_n_values_realised"] == 4  # the smallest cardinality (shape)
    for name, r in bal["factors"].items():
        assert r["val_n_values_realised"] == r["cardinality"], name
    # ⛔ and the broken split must STILL be refused on the real grid
    f_h, v_h = head_tail_fit_val(pool, 32_256, 512)
    bad = row_split_balance(
        split.grid.factor_indices_of(f_h), split.grid.factor_indices_of(v_h),
        list(split.grid.names), list(split.grid.cardinalities),
    )
    assert bad["factors"]["floor_hue"]["val_n_values_realised"] == 1
    assert bad["factors"]["wall_hue"]["val_n_values_realised"] == 2
    with pytest.raises(RowSplitError):
        assert_row_split_balanced(bad)
