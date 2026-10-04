"""Factor grids, the Montero recomb-to-element splits, and the hold-out assertion.

⭐ The load-bearing test here is the **positive control**: the hold-out
assertion is deliberately corrupted (a known held-out combination is injected
into the train index) and must FIRE.  An assertion that has never been seen to
fail is not evidence that the held-out combinations are absent from training,
and every downstream generative number rests on that.

Structure:

- Synthetic-grid tests (always run): bijection, assertion fires, C-set validity,
  and the two dSprites constant traps (T1 column slice, T2 swapped angle names).
- Real-data tests (skipped when the datasets are absent, since they are 19 MB /
  268 MB downloads that never enter git): realised counts, empirical axis-order
  decode, and the positive control on the REAL split.
"""

import json

import numpy as np
import pytest

from chlu.data import dsprites as dsp
from chlu.data import shapes3d as s3d
from chlu.data.gen_splits import (
    FactorGrid,
    GenSplit,
    HoldoutViolation,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_non_empty_test,
    build_c_invalid,
    factor_grid_from_latents,
    load_index,
    save_index,
    validate_c_invalid,
)

# ---------------------------------------------------------------------------
# synthetic fixtures (no data files needed)
# ---------------------------------------------------------------------------


def _toy_grid():
    """3 x 4 x 5 exhaustive grid; the 'held-out' cell is a>=2 & b>=3 & c>=4."""
    return FactorGrid(
        names=("a", "b", "c"),
        values=(
            np.arange(3, dtype=float),
            np.arange(4, dtype=float),
            np.arange(5, dtype=float),
        ),
        dataset="toy",
    )


def _toy_predicate(vals):
    return (vals[:, 0] >= 2) & (vals[:, 1] >= 3) & (vals[:, 2] >= 4)


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


# ---------------------------------------------------------------------------
# factor grid + bijection
# ---------------------------------------------------------------------------


def test_toy_grid_bijection_roundtrips():
    grid = _toy_grid()
    assert grid.n_rows == 60
    rows = np.arange(60, dtype=np.int64)
    assert np.array_equal(grid.rows_of(grid.factor_indices_of(rows)), rows)


def test_verify_bijection_catches_permuted_axes():
    """A permuted axis order must be caught, not silently accepted."""
    grid = _toy_grid()
    rows = np.arange(grid.n_rows, dtype=np.int64)
    good = grid.values_of(rows)
    assert grid.verify_bijection(good) == grid.n_rows
    # swap two factor columns -> the same values, a different assignment
    bad = good[:, [1, 0, 2]]
    bad_grid = FactorGrid(
        names=("b", "a", "c"),
        values=(grid.values[1], grid.values[0], grid.values[2]),
        dataset="toy",
    )
    with pytest.raises(AssertionError, match="bijection broken"):
        bad_grid.verify_bijection(good)
    # the permuted grid verifies against its OWN row-major enumeration
    assert bad_grid.verify_bijection(bad_grid.values_of(rows)) == grid.n_rows
    assert not np.array_equal(bad_grid.values_of(rows), bad)


def test_factor_grid_from_latents_requires_exhaustive_grid():
    grid = _toy_grid()
    vals = grid.values_of(np.arange(grid.n_rows, dtype=np.int64))
    factor_grid_from_latents(vals, ("a", "b", "c"), "toy")  # ok
    with pytest.raises(ValueError, match="not exhaustive"):
        factor_grid_from_latents(vals[:-1], ("a", "b", "c"), "toy")


# ---------------------------------------------------------------------------
# ⭐ the hold-out assertion and its positive control (synthetic)
# ---------------------------------------------------------------------------


def test_holdout_assertion_passes_on_clean_split():
    split = _toy_split()
    assert split.n_test == 1 * 1 * 1  # a=2, b=3, c=4
    assert assert_holdout_disjoint(split) == split.n_train == 59


def test_holdout_assertion_FIRES_on_injected_holdout_row():
    """⭐ POSITIVE CONTROL (synthetic): inject a held-out row -> must raise."""
    split = _toy_split()
    corrupted = split.with_train_rows(
        np.sort(np.concatenate([split.train_rows, split.test_rows]))
    )
    with pytest.raises(HoldoutViolation, match="satisfy the held-out predicate"):
        assert_holdout_disjoint(corrupted)


def test_holdout_assertion_fires_with_small_chunks_too():
    """Chunked iteration must not skip the offending row."""
    split = _toy_split()
    corrupted = split.with_train_rows(
        np.sort(np.concatenate([split.train_rows, split.test_rows]))
    )
    with pytest.raises(HoldoutViolation):
        assert_holdout_disjoint(corrupted, chunk=7)


def test_non_empty_assertion_fires_on_empty_test_set():
    """T1's failure mode: a filter that matches nothing must not pass silently."""
    split = _toy_split()
    empty = GenSplit(
        dataset="toy",
        condition="recomb2element",
        variant="leave1out",
        grid=split.grid,
        train_rows=np.arange(split.grid.n_rows, dtype=np.int64),
        test_rows=np.array([], dtype=np.int64),
        test_predicate=_toy_predicate,
    )
    with pytest.raises(HoldoutViolation, match="EMPTY"):
        assert_non_empty_test(empty)
    with pytest.raises(HoldoutViolation, match="expected"):
        assert_non_empty_test(split, expected=999)


def test_partition_assertion_fires_on_a_lost_row():
    """A split that silently drops rows must not pass as a partition."""
    split = _toy_split()
    assert assert_index_partitions_grid(split) == split.grid.n_rows
    with pytest.raises(HoldoutViolation, match="do not "):
        assert_index_partitions_grid(split.with_train_rows(split.train_rows[:10]))
    # a duplicated row is also not a partition
    dup = np.concatenate([split.train_rows, split.train_rows[:1]])
    with pytest.raises(HoldoutViolation, match="do not "):
        assert_index_partitions_grid(split.with_train_rows(dup))


# ---------------------------------------------------------------------------
# the C set (structurally invalid factor requests)
# ---------------------------------------------------------------------------


def test_c_invalid_is_structurally_invalid_and_arm_agnostic():
    grid = _toy_grid()
    payload = build_c_invalid(grid, n_per_kind=6, seed=0)
    assert validate_c_invalid(payload, grid) == 12
    kinds = {r["kind"] for r in payload["records"]}
    assert kinds == {"duplicate_factor", "out_of_range"}
    # ⛔ factor space only: the emitted records must name nothing but factors,
    # values and the violation kind — no well, mode, cluster, arm or encoder.
    blob = json.dumps(
        {"records": payload["records"], "factors": payload["factors"]}
    ).lower()
    for word in ("well", "mode", "cluster", "arm", "encoder", "clu", "latent"):
        assert word not in blob, f"leaked model-internal vocabulary: {word!r}"


def test_c_invalid_is_deterministic_given_seed():
    grid = _toy_grid()
    a = build_c_invalid(grid, n_per_kind=4, seed=3)
    b = build_c_invalid(grid, n_per_kind=4, seed=3)
    c = build_c_invalid(grid, n_per_kind=4, seed=4)
    assert a == b
    assert a != c


def test_validate_c_invalid_rejects_a_satisfiable_request():
    """The C-set validator must itself be capable of failing."""
    grid = _toy_grid()
    payload = build_c_invalid(grid, n_per_kind=2, seed=0)
    payload["records"][0]["request"] = [
        {"factor": "a", "factor_index": 0, "value": 0.0},
        {"factor": "b", "factor_index": 1, "value": 0.0},
        {"factor": "c", "factor_index": 2, "value": 0.0},
    ]
    with pytest.raises(AssertionError):
        validate_c_invalid(payload, grid)


# ---------------------------------------------------------------------------
# index persistence
# ---------------------------------------------------------------------------


def test_save_index_roundtrips_and_is_checksum_stable(tmp_path):
    split = _toy_split()
    p1, sha1 = save_index(split, tmp_path / "a.npz")
    p2, sha2 = save_index(split, tmp_path / "b.npz")
    assert sha1 == sha2, "index checksum must be reproducible across runs"
    grid, train, test, meta = load_index(p1)
    assert grid.names == split.grid.names
    assert np.array_equal(train, split.train_rows)
    assert np.array_equal(test, split.test_rows)
    assert meta["n_test"] == split.n_test


# ---------------------------------------------------------------------------
# the dSprites constant traps — guarded so nobody "repairs" them
# ---------------------------------------------------------------------------


def test_dsprites_angle_constants_are_the_upstream_swapped_ones():
    """⛔ T2: a120 is 240 deg and a240 is 120 deg upstream. Do not 'fix'."""
    assert dsp.a120 == pytest.approx(4 * np.pi / 3)
    assert dsp.a240 == pytest.approx(2 * np.pi / 3)
    assert dsp.a90 == pytest.approx(np.pi / 2)
    assert dsp.a180 == pytest.approx(np.pi)
    # the effective band is 120 deg < theta < 240 deg
    assert dsp.a240 < np.pi < dsp.a120


def test_dsprites_filter_rejects_the_unsliced_six_column_array():
    """⛔ T1: applying the filter to the raw 6-column npz must not run silently."""
    fake = np.zeros((10, 6))
    with pytest.raises(ValueError, match="post colour slice"):
        dsp.montero_test_mask(fake, fake.astype(int))


def test_shapes3d_filter_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="6 label columns"):
        s3d.montero_test_mask(np.zeros((10, 5)))


def test_shapes3d_filter_selects_72_on_a_synthetic_grid():
    """The filter's arithmetic, checked without the 268 MB file."""
    vals = [
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.0, 0.9, 10),
        np.linspace(0.75, 1.25, 8),
        np.arange(4, dtype=float),
        np.linspace(-30.0, 30.0, 15),
    ]
    labels = np.stack([g.ravel() for g in np.meshgrid(*vals, indexing="ij")], axis=-1)
    assert len(labels) == 480000
    assert int(s3d.montero_test_mask(labels).sum()) == 72


# ---------------------------------------------------------------------------
# real-data tests (datasets are never committed; skip when absent)
# ---------------------------------------------------------------------------

h5py = pytest.importorskip("h5py", reason="Shapes3D/dSprites image access needs h5py")

_DSPRITES_NPZ = dsp.dsprites_root() / dsp.NPZ_NAME
_DSPRITES_H5 = dsp.dsprites_root() / dsp.HDF5_NAME
_SHAPES3D_H5 = s3d.shapes3d_root() / s3d.H5_NAME

needs_dsprites = pytest.mark.skipif(
    not _DSPRITES_NPZ.exists(),
    reason=f"dSprites npz not at {_DSPRITES_NPZ} (chlu data dsprites --download)",
)
needs_dsprites_images = pytest.mark.skipif(
    not _DSPRITES_H5.exists(), reason="dSprites hdf5 mirror not downloaded"
)
needs_shapes3d = pytest.mark.skipif(
    not _SHAPES3D_H5.exists(),
    reason=f"Shapes3D h5 not at {_SHAPES3D_H5} (chlu data shapes3d --download)",
)


@needs_dsprites
def test_real_dsprites_realised_counts_and_float_boundary():
    """Realised n_test is 6292, decided by float64 round-off (T3), not 5808."""
    split, lv, lc = dsp.build_split()
    assert split.n_total == 737280
    assert split.n_test == dsp.N_TEST_REALISED == 6292
    assert split.n_train == 730988
    rot_grid = split.grid.values[split.grid.index_of("orientation")]
    # index 13 is admitted (stored value strictly > 2*pi/3), index 26 is not.
    assert rot_grid[13] > dsp.a240
    assert not (rot_grid[26] < dsp.a120)
    assert int(((rot_grid > dsp.a240) & (rot_grid < dsp.a120)).sum()) == 13
    scale_grid = split.grid.values[split.grid.index_of("scale")]
    assert int((scale_grid > 0.6).sum()) == 4  # stored 0.6 is not > 0.6


@needs_dsprites
def test_real_dsprites_holdout_assertion_and_POSITIVE_CONTROL():
    """⭐ The criterion this spoke lives or dies on, on the real split."""
    split, _, _ = dsp.build_split()
    assert assert_holdout_disjoint(split) == split.n_train

    injected = int(split.test_rows[0])
    corrupted = split.with_train_rows(
        np.sort(np.concatenate([split.train_rows, np.array([injected])]))
    )
    with pytest.raises(HoldoutViolation) as exc:
        assert_holdout_disjoint(corrupted)
    assert str(injected) in str(exc.value)


@needs_dsprites
def test_real_dsprites_shape_class_is_ellipse():
    """T5: recomb-to-element indexes latent_CLASSES, so ==1 is the ELLIPSE."""
    _, lv, lc = dsp.build_split()
    ell = lc[:, dsp.shp] == 1
    assert np.all(lv[ell, dsp.shp] == 2.0)


@needs_dsprites_images
def test_real_dsprites_axis_order_decoded_from_images():
    """Empirical axis check: decode known rows and verify each factor's effect."""
    with dsp.open_images() as f:
        imgs = f["imgs"]
        sizes = (1, 3, 6, 40, 32, 32)

        def row(color, shape, scale, orient, px, py):
            return int(
                np.ravel_multi_index((color, shape, scale, orient, px, py), sizes)
            )

        def com(r):
            im = np.asarray(imgs[r])
            ys, xs = np.nonzero(im)
            return float(xs.mean()), float(ys.mean()), int(im.sum())

        # posX (factor 4 raw / 3 post-slice): centre-of-mass x rises, y fixed
        cxs = [com(row(0, 0, 3, 0, px, 16)) for px in (10, 16, 24)]
        assert cxs[0][0] < cxs[1][0] < cxs[2][0]
        assert cxs[0][1] == cxs[1][1] == cxs[2][1]
        # posY (factor 5 raw / 4 post-slice): y rises, x fixed
        cys = [com(row(0, 0, 3, 0, 16, py)) for py in (10, 16, 24)]
        assert cys[0][1] < cys[1][1] < cys[2][1]
        assert cys[0][0] == cys[1][0] == cys[2][0]
        # scale (factor 2 raw / 1 post-slice): area strictly increases
        areas = [com(row(0, 0, s, 0, 16, 16))[2] for s in range(6)]
        assert areas == sorted(areas) and areas[0] < areas[-1]
        # orientation (factor 3 raw / 2 post-slice): index 39 == 2*pi wraps to 0
        heart0 = np.asarray(imgs[row(0, 2, 3, 0, 16, 16)])
        heart39 = np.asarray(imgs[row(0, 2, 3, 39, 16, 16)])
        heart10 = np.asarray(imgs[row(0, 2, 3, 10, 16, 16)])
        assert np.array_equal(heart0, heart39)
        assert not np.array_equal(heart0, heart10)


@needs_shapes3d
def test_real_shapes3d_realised_count_is_72():
    """⛔ T4: any number other than 72 is a stop-and-report, not an adjustment."""
    split, labels = s3d.build_split()
    assert split.n_total == 480000
    assert split.n_test == s3d.N_TEST_REALISED == 72
    assert split.n_train == 479928
    held = labels[split.test_rows]
    assert set(np.unique(held[:, s3d.shp])) == {1.0}  # cylinder only
    assert np.all(held[:, s3d.rot] > 20)
    assert np.all(held[:, s3d.scl] >= 1.1)


@needs_shapes3d
def test_real_shapes3d_holdout_assertion_and_POSITIVE_CONTROL():
    """⭐ The criterion this spoke lives or dies on, on the real split."""
    split, _ = s3d.build_split()
    assert assert_holdout_disjoint(split) == split.n_train == 479928

    injected = int(split.test_rows[-1])
    corrupted = split.with_train_rows(
        np.sort(np.concatenate([split.train_rows, np.array([injected])]))
    )
    with pytest.raises(HoldoutViolation) as exc:
        assert_holdout_disjoint(corrupted)
    assert str(injected) in str(exc.value)


@needs_shapes3d
def test_real_shapes3d_axis_order_decoded_from_images():
    """Empirical axis check: each factor's change must localise where it should.

    floor_hue -> bottom rows only; wall_hue -> upper rows, never the bottom;
    object_hue -> the central object block only.
    """
    with s3d.open_images() as f:
        images = f["images"]
        sizes = (10, 10, 10, 8, 4, 15)
        base = [3, 6, 1, 2, 0, 7]

        def change_map(k, n):
            ims = []
            for v in range(n):
                idx = list(base)
                idx[k] = v
                r = int(np.ravel_multi_index(tuple(idx), sizes))
                ims.append(np.asarray(images[r], dtype=np.float64) / 255.0)
            return np.stack(ims).std(0).mean(-1)  # (64, 64)

        floor = change_map(0, 10)
        wall = change_map(1, 10)
        obj = change_map(2, 10)
        # floor_hue changes the bottom of the frame and nothing in the top half
        assert floor[32:].sum() > 100 * floor[:32].sum()
        # wall_hue changes the upper frame and (essentially) not the last rows
        assert wall[:40].sum() > 100 * wall[56:].sum()
        # object_hue changes only the central object block
        centre = obj[28:56, 16:48].sum()
        assert centre > 0.9 * obj.sum()


@needs_shapes3d
@needs_dsprites
def test_real_c_invalid_files_are_valid(tmp_path):
    """The emitted C sets validate against their own grids."""
    for build in (s3d.build_split, dsp.build_split):
        out = build()
        split = out[0]
        payload = build_c_invalid(split.grid, n_per_kind=4, seed=0)
        assert validate_c_invalid(payload, split.grid) == 8
        for rec in payload["records"]:
            for term in rec["request"]:
                assert term["factor"] in split.grid.names
