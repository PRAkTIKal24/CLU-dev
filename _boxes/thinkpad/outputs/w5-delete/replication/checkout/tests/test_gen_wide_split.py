"""The widened SECONDARY hold-out (``recomb2element_wide``, R17).

⭐ Two load-bearing tests, and they pull in opposite directions:

1. **The absence assertion must FIRE** on the wide split's positive control —
   a wide condition whose assertion has never been seen to fail is worth no more
   than an unchecked claim (the same standard ``test_gen_data_splits.py``
   applies to Montero's split).
2. **Montero's headline split must be BIT-IDENTICAL** after the wide condition
   exists: same 72 rows, same emitted ``factor_index.npz`` / ``c_invalid.json``
   checksums.  The wide split is *additional*, never a modification.

⛔ Not tested here, because it is not a property of code: the frozen encoder /
decoder trained under Montero's split have SEEN the rows this condition holds
out.  The wide condition is a second full configuration, not a re-scoring of the
shipped artifacts.
"""

import numpy as np
import pytest

from chlu.data import shapes3d as s3d
from chlu.data.gen_splits import (
    FactorCell,
    FactorGrid,
    HoldoutViolation,
    assert_cells_disjoint,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    build_c_invalid,
    build_cell_split,
    cells_test_predicate,
    save_index,
    validate_c_invalid,
    write_json_with_checksum,
)

#: ⭐ Pre-registered in `.claude/outputs/gen-wide-split/PREREG.md` BEFORE the
#: builder was first run: the contrast label of a cell whose object-hue band is
#: offset by ``s`` grid steps from its background band, in units of hue.
PREREGISTERED_CONTRAST_LADDER = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.45}

#: Montero's cell shape, per factor: hue bands 2/2/2, scale corner 3, one shape,
#: orientation corner 3.  Every wide cell must have exactly these widths.
MONTERO_CELL_WIDTHS = (2, 2, 2, 3, 1, 3)


# ---------------------------------------------------------------------------
# synthetic fixtures (no data files needed)
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


def _synthetic_shapes3d_labels():
    """The exact Shapes3D factor grid, without the 268 MB file."""
    vals = [
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.75, 1.25, 8),
        np.arange(4, dtype=float),
        np.linspace(-30.0, 30.0, 15),
    ]
    return np.stack([g.ravel() for g in np.meshgrid(*vals, indexing="ij")], axis=-1)


# ---------------------------------------------------------------------------
# FactorCell mechanics
# ---------------------------------------------------------------------------


def test_cell_rows_agree_with_the_cell_mask():
    """``rows()`` (via the bijection) and ``mask()`` (over values) must agree."""
    grid = _toy_grid()
    cell = FactorCell("c0", {"a": (1.0, 2.0), "c": (4.0,)})
    all_rows = np.arange(grid.n_rows)
    by_mask = all_rows[cell.mask(grid, grid.values_of(all_rows))]
    assert np.array_equal(cell.rows(grid), by_mask)
    # 2 values of 'a' x all 4 of the unconstrained 'b' x 1 value of 'c'
    assert cell.rows(grid).size == 2 * 4 * 1
    assert cell.n_rows_nominal == 2 * 1  # nominal counts CONSTRAINED factors only


def test_cell_rejects_an_off_grid_value():
    grid = _toy_grid()
    with pytest.raises(ValueError, match="not on the grid"):
        FactorCell("bad", {"a": (99.0,)}).rows(grid)


def test_assert_cells_disjoint_FIRES_on_overlapping_cells():
    """Overlapping cells would double-count rows in the per-cell table."""
    grid = _toy_grid()
    a = FactorCell("a", {"a": (1.0, 2.0)})
    b = FactorCell("b", {"a": (2.0,)})
    assert assert_cells_disjoint(grid, [a]) == a.rows(grid).size
    with pytest.raises(HoldoutViolation, match="double-count"):
        assert_cells_disjoint(grid, [a, b])


def test_build_cell_split_partitions_the_grid_and_ors_the_cells():
    grid = _toy_grid()
    cells = [
        FactorCell("c0", {"a": (0.0,)}),
        FactorCell("c1", {"a": (2.0,), "b": (3.0,)}),
    ]
    split = build_cell_split(grid, cells, "toy", "wide", "v")
    assert assert_index_partitions_grid(split) == grid.n_rows
    assert split.n_test == 1 * 4 * 5 + 1 * 1 * 5
    pred = cells_test_predicate(grid, cells)
    assert int(pred(grid.values_of(split.test_rows)).sum()) == split.n_test
    assert not pred(grid.values_of(split.train_rows)).any()


# ---------------------------------------------------------------------------
# ⛔ Montero's headline split is untouched
# ---------------------------------------------------------------------------


def test_montero_cell_reproduces_the_montero_mask_exactly():
    """The cell representation of Montero's split is checked, not assumed."""
    labels = _synthetic_shapes3d_labels()
    grid = s3d.build_grid(labels)
    rows_from_mask = np.flatnonzero(s3d.montero_test_mask(labels))
    assert np.array_equal(s3d.montero_cell(grid).rows(grid), rows_from_mask)
    assert rows_from_mask.size == s3d.N_TEST_REALISED == 72


def test_montero_split_is_unchanged_by_the_wide_condition(tmp_path):
    """⭐ Acceptance item 4: 72 rows and a bit-identical emitted index.

    The pinned checksum is the one shipped at base `fcdca43`
    (`.claude/data/gen/shapes3d/factor_index.npz`); it must survive the wide
    condition being added.
    """
    labels = _synthetic_shapes3d_labels()
    split, _ = s3d.build_split(labels=labels)
    assert (split.n_total, split.n_test, split.n_train) == (480000, 72, 479928)
    assert split.condition == "recomb2element"
    _, sha = save_index(split, tmp_path / "factor_index.npz")
    assert sha == "5f1fce25c40cceae6de2d00fd2145c4cdc73dcc38cee7f6545cbcaea3ce78f49"
    c_set = build_c_invalid(
        split.grid, n_per_kind=max(1, -(-split.n_test // 2)), seed=0
    )
    _, c_sha = write_json_with_checksum(c_set, tmp_path / "c_invalid.json")
    assert c_sha == "b886da239e97b19e9001a88ce04fded329dda4884d301f78c3097ea6cc66ba49"


# ---------------------------------------------------------------------------
# the wide condition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide():
    labels = _synthetic_shapes3d_labels()
    split, _, cells = s3d.build_wide_split(labels=labels, n_cells=12, seed=0)
    return split, cells


def test_wide_cells_are_structurally_montero_shaped(wide):
    """Every cell is a conjunction over all six factors, of Montero's size."""
    split, cells = wide
    assert len(cells) == 13  # 12 additional + Montero's own cell
    assert cells[0].cell_id == "montero"
    for cell in cells:
        assert set(cell.values) == set(s3d.FACTOR_NAMES)
        widths = tuple(len(set(cell.values[n])) for n in s3d.FACTOR_NAMES)
        assert widths == MONTERO_CELL_WIDTHS
        assert cell.rows(split.grid).size == 72  # realised, not nominal


def test_wide_realised_counts(wide):
    """⛔ Realised, never adjusted: 13 x 72 = 936 held out of 480,000."""
    split, cells = wide
    assert split.condition == s3d.WIDE_CONDITION
    assert split.n_test == 936
    assert split.n_train == 479064
    assert assert_index_partitions_grid(split) == 480000
    assert sum(c.rows(split.grid).size for c in cells) == split.n_test  # disjoint


def test_wide_contrast_labels_match_the_PREREGISTERED_ladder(wide):
    """⭐ Acceptance item 2: measured contrast, spanning the range."""
    split, cells = wide
    for cell in cells[1:]:
        s = cell.meta["contrast_step"]
        assert cell.meta["object_bg_hue_contrast"] == pytest.approx(
            PREREGISTERED_CONTRAST_LADDER[s], abs=1e-12
        )
    labels = [c.meta["object_bg_hue_contrast"] for c in cells]
    montero = cells[0].meta["object_bg_hue_contrast"]
    assert montero == pytest.approx(0.05, abs=1e-12)  # the bottom rung
    # ⛔ the falsifier: cells clustered in Montero's photometric corner
    assert max(labels) - min(labels) >= 0.3
    assert max(labels) >= 0.9 * cells[0].meta["max_possible"]
    assert sum(abs(x - montero) > 0.05 for x in labels) >= len(labels) // 2


def test_wide_split_holdout_assertion_and_POSITIVE_CONTROL(wide):
    """⭐ The criterion this spoke lives on: the assertion must FIRE."""
    split, _ = wide
    assert assert_holdout_disjoint(split) == split.n_train == 479064

    injected = int(split.test_rows[-1])
    corrupted = split.with_train_rows(
        np.sort(np.concatenate([split.train_rows, np.array([injected])]))
    )
    with pytest.raises(HoldoutViolation) as exc:
        assert_holdout_disjoint(corrupted)
    assert str(injected) in str(exc.value)


def test_wide_positive_control_fires_for_every_cell(wide):
    """One injected row PER CELL — a single-cell check could hide 12 vacuous cells."""
    split, cells = wide
    for cell in cells:
        row = int(cell.rows(split.grid)[0])
        corrupted = split.with_train_rows(
            np.sort(np.concatenate([split.train_rows, np.array([row])]))
        )
        with pytest.raises(HoldoutViolation) as exc:
            assert_holdout_disjoint(corrupted)
        assert str(row) in str(exc.value)


def test_wide_split_is_deterministic_and_seed_sensitive():
    labels = _synthetic_shapes3d_labels()
    a, _, _ = s3d.build_wide_split(labels=labels, n_cells=6, seed=0)
    b, _, _ = s3d.build_wide_split(labels=labels, n_cells=6, seed=0)
    c, _, _ = s3d.build_wide_split(labels=labels, n_cells=6, seed=1)
    assert np.array_equal(a.test_rows, b.test_rows)
    assert not np.array_equal(a.test_rows, c.test_rows)
    assert a.n_test == b.n_test == c.n_test == 7 * 72


def test_the_additional_cells_are_always_beyond_monteros_72(wide):
    """``include_montero`` moves NO cell — it only carries Montero's own.

    Montero's rows are reserved before the draws either way, so the additional
    cells are bit-identical between the two settings and are always disjoint
    from Montero's 72.
    """
    with_m_split, with_m_cells = wide
    labels = _synthetic_shapes3d_labels()
    split, _, cells = s3d.build_wide_split(
        labels=labels, n_cells=12, seed=0, include_montero=False
    )
    assert len(cells) == 12
    assert split.n_test == 864
    montero_rows = np.flatnonzero(s3d.montero_test_mask(labels))
    assert np.intersect1d(split.test_rows, montero_rows).size == 0
    assert np.array_equal(
        split.test_rows, np.setdiff1d(with_m_split.test_rows, montero_rows)
    )
    assert [c.values for c in cells] == [c.values for c in with_m_cells[1:]]


def test_wide_metadata_records_the_rule_verbatim(wide):
    split, cells = wide
    meta = s3d.wide_metadata(split, cells, seed=0)
    assert meta["rule"] == s3d.WIDE_RULE
    assert "REJECTED" in meta["rule"] and "default_rng(seed)" in meta["rule"]
    assert meta["n_test_realised"] == 936
    assert meta["montero_headline_untouched"]["n_test"] == 72
    assert "SEEN these rows" in meta["frozen_artifact_warning"]
    assert [c["n_rows_realised"] for c in meta["cells"]] == [72] * 13


def test_wide_c_set_is_sized_to_its_own_B(wide):
    """|C| ~= |B| for the wide condition, by the same derivation as Montero's."""
    split, _ = wide
    n_per_kind = max(1, -(-split.n_test // 2))
    c_set = build_c_invalid(split.grid, n_per_kind=n_per_kind, seed=0)
    assert validate_c_invalid(c_set, split.grid) == 936 == split.n_test


def test_wide_condition_is_rejected_for_an_unknown_name():
    with pytest.raises(ValueError, match="unknown Shapes3D condition"):
        s3d.load_shapes3d(condition="not_a_condition")


# ---------------------------------------------------------------------------
# real-data tests (the 268 MB file never enters git; skip when absent)
# ---------------------------------------------------------------------------

pytest.importorskip("h5py", reason="Shapes3D access needs h5py")
_SHAPES3D_H5 = s3d.shapes3d_root() / s3d.H5_NAME
needs_shapes3d = pytest.mark.skipif(
    not _SHAPES3D_H5.exists(),
    reason=f"Shapes3D h5 not at {_SHAPES3D_H5} (chlu data shapes3d --download)",
)


@needs_shapes3d
def test_real_wide_split_counts_and_POSITIVE_CONTROL():
    labels = s3d.load_labels()
    split, _, cells = s3d.build_wide_split(labels=labels, n_cells=12, seed=0)
    assert (split.n_test, split.n_train) == (936, 479064)
    assert assert_holdout_disjoint(split) == 479064

    injected = int(split.test_rows[0])
    with pytest.raises(HoldoutViolation) as exc:
        assert_holdout_disjoint(
            split.with_train_rows(
                np.sort(np.concatenate([split.train_rows, np.array([injected])]))
            )
        )
    assert str(injected) in str(exc.value)
    # the real labels place the cells exactly where the synthetic grid does
    assert [c.meta["object_bg_hue_contrast"] for c in cells[1:]] == pytest.approx(
        [PREREGISTERED_CONTRAST_LADDER[c.meta["contrast_step"]] for c in cells[1:]],
        abs=1e-12,
    )


@needs_shapes3d
def test_real_montero_split_still_72_after_the_wide_condition():
    """⛔ Run this before believing anything else about the wide split."""
    split, labels = s3d.build_split()
    assert split.n_test == 72
    held = labels[split.test_rows]
    assert set(np.unique(held[:, s3d.shp])) == {1.0}
    assert np.all(held[:, s3d.rot] > 20) and np.all(held[:, s3d.scl] >= 1.1)
