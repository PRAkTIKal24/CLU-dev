"""dSprites loader + the Montero recombination-to-element split.

Source of truth for the split is the **official implementation**
``mmrl/disent-and-gen@bff35a7570d08fdef0d10c21b093b9a24d98dde6``, file
``scripts/configs/datasplits.py`` (class ``Dsprites``, ``leave1out``).  The
paper's prose contradicts the code for this dataset and contradicts *itself* on
rotation; the code wins and is reproduced verbatim below.  See
``.claude/outputs/gen-montero-splits-spec/montero_splits.json``.

Three traps live in this file; all three fail silently if "fixed":

T1 — colour-column slice.  ``load_sprites`` in the official repo does
     ``latent_values = data_zip['latents_values'][:, 1:]`` at load time, so the
     filter's index constants ``shp, scl, rot, tx, ty = 0, 1, 2, 3, 4`` are
     valid ONLY post-slice.  Applied to the raw 6-column npz the filter selects
     **nothing** (colour class is always 0, so ``== 1`` is empty) and the
     held-out set is silently empty.
T2 — the angle constants are bound to SWAPPED values upstream (see below).
T3 — two thresholds land exactly on grid values, so the realised count is
     decided by float64 round-off.  We never regenerate the grid; the filter is
     evaluated against the actual stored array.

Data: ``dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`` (latents, authoritative
for the split) and the matching ``.hdf5`` (identical grid, but with lazy
per-row image access — the npz would need ~3 GB of RAM to decompress ``imgs``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gen_splits import (
    FactorGrid,
    GenSplit,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_non_empty_test,
    default_gen_data_root,
    factor_grid_from_latents,
    file_sha256,
)

DATASET = "dsprites"

NPZ_NAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
HDF5_NAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"
NPZ_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/master/" + NPZ_NAME
HDF5_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/master/" + HDF5_NAME
# Upstream publishes no checksum; these were computed on first download
# (2026-09-09) and are the integrity anchor from here on.
NPZ_SHA256 = "857dc001287e8efa226e0025fac6b29e5745c621b380b0d3363e4e33f4e28fa2"
HDF5_SHA256 = "729fa22e884c9c500e1f17415d580746b4c5894fbbb780a1319bbcfa39a745bb"
NPZ_BYTES = 19318135
HDF5_BYTES = 28024106
LICENCE = "zlib/libpng licence (google-deepmind/dsprites-dataset)"
CITATION = "Matthey, Higgins, Hassabis, Lerchner (2017), dSprites"

#: Factor names AFTER the colour-column slice (T1).  Verified empirically
#: against the raw file — see ``tests/test_gen_data_splits.py``.
FACTOR_NAMES = ("shape", "scale", "orientation", "posX", "posY")
RAW_FACTOR_NAMES = ("color",) + FACTOR_NAMES
CARDINALITIES = (3, 6, 40, 32, 32)
N_IMAGES = 737280

# --- verbatim from scripts/configs/datasplits.py ---------------------------
# Post-slice column indices (T1): valid only for the 5-column arrays.
shp, scl, rot, tx, ty = 0, 1, 2, 3, 4

# ⛔ T2 — DO NOT "FIX" THIS LINE.  Reproduced verbatim from upstream.  The
# constant NAMES are bound to SWAPPED values: a120 = 4*pi/3 = 240 degrees and
# a240 = 2*pi/3 = 120 degrees.  Read naively, the clause
# ``(rot < a120) & (rot > a240)`` looks impossible; with the actual bound values
# it means 120 deg < theta < 240 deg, which is what the paper's section 2.1
# describes.  Renaming or swapping these inverts the held-out rotation band.
a90, a120, a180, a240 = np.pi / 2, 4 * np.pi / 3, np.pi, 2 * np.pi / 3
# ---------------------------------------------------------------------------

#: Realised held-out size, measured against the actual npz (T3).  The nominal
#: arithmetic predicts 5808 (12 rotation values); the stored float64
#: linspace(0, 2*pi, 40)[13] = 2.0943951023931957 is *strictly greater* than
#: 2*pi/3 = 2.0943951023931953, so index 13 is admitted (13 values) while index
#: 26 = 4.188790204786391 is NOT less than 4*pi/3 = 4.1887902047863905 and is
#: excluded.  1 * 4 * 13 * 11 * 11 = 6292.
N_TEST_REALISED = 6292
N_TRAIN_REALISED = N_IMAGES - N_TEST_REALISED


def dsprites_root(root: str | Path | None = None) -> Path:
    return Path(root).expanduser() if root else default_gen_data_root() / DATASET


def montero_test_mask(
    latent_values: np.ndarray, latent_classes: np.ndarray
) -> np.ndarray:
    """The verbatim recombination-to-element / leave1out test filter.

    Args:
        latent_values: (N, 5) float array, **post colour slice** (T1).
        latent_classes: (N, 5) int array, post colour slice.

    Returns:
        (N,) boolean mask of held-out rows.
    """
    latent_values = np.asarray(latent_values)
    latent_classes = np.asarray(latent_classes)
    if latent_values.shape[1] != 5 or latent_classes.shape[1] != 5:
        raise ValueError(
            "dSprites filters expect 5 columns (post colour slice, T1); got "
            f"values{latent_values.shape} classes{latent_classes.shape}"
        )
    return (
        (latent_classes[:, shp] == 1)  # ellipse (latent_CLASSES, not values — T5)
        & (latent_values[:, scl] > 0.6)
        & (latent_values[:, rot] > 0.0)
        & (latent_values[:, rot] < a120)  # < 240 deg  (T2)
        & (latent_values[:, rot] > a240)  # > 120 deg  (T2)
        & (latent_values[:, tx] > 0.66)
        & (latent_values[:, ty] > 0.66)
    )


#: The value the held-out cell pins ``shape`` to: class 1 = ellipse (T4 — the
#: filter reads ``latent_CLASSES`` for shape and ``latent_values`` for the rest).
HELDOUT_SHAPE_CLASS = 1


def montero_corner_mask(latent_values: np.ndarray) -> np.ndarray:
    """⭐ The held-out cell's conjunction with the ``shape`` clause RELAXED.

    This is the *corner* R16's **matched-seen** control lives in, and R20's
    in-corner discrimination metric needs: same scale band, same rotation band
    (T2 — the constants stay swapped), same position quadrant, **any shape**.

    ⛔ Every threshold below is the same literal as :func:`montero_test_mask`'s,
    and ``tests/test_gen_bridge_dsprites.py`` asserts on the REAL file that
    ``montero_test_mask(v, c) == (c[:, shp] == 1) & montero_corner_mask(v)``
    exactly — so the verbatim filter stays verbatim and this stays honest.
    """
    latent_values = np.asarray(latent_values)
    if latent_values.shape[1] != 5:
        raise ValueError(
            "dSprites filters expect 5 columns (post colour slice, T1); got "
            f"values{latent_values.shape}"
        )
    return (
        (latent_values[:, scl] > 0.6)
        & (latent_values[:, rot] > 0.0)
        & (latent_values[:, rot] < a120)  # < 240 deg  (T2)
        & (latent_values[:, rot] > a240)  # > 120 deg  (T2)
        & (latent_values[:, tx] > 0.66)
        & (latent_values[:, ty] > 0.66)
    )


def montero_position_corner_mask(latent_values: np.ndarray) -> np.ndarray:
    """Only the POSITION clauses of the cell (``tx > 0.66 & ty > 0.66``).

    The **seen normaliser** for the in-corner read keeps this context clause and
    relaxes scale + rotation — the dSprites analogue of ``gen-bridge-wide``'s
    "the cell's hue bands + its own shape, but outside the cell".  dSprites has
    no photometric clause at all, so position is the only context the cell
    carries beyond the two factors that are relaxed.
    """
    latent_values = np.asarray(latent_values)
    if latent_values.shape[1] != 5:
        raise ValueError(
            "dSprites filters expect 5 columns (post colour slice, T1); got "
            f"values{latent_values.shape}"
        )
    return (latent_values[:, tx] > 0.66) & (latent_values[:, ty] > 0.66)


def _predicate_from_values(latent_values: np.ndarray) -> np.ndarray:
    """Test predicate over reconstructed factor **values** only.

    The hold-out assertion reconstructs values from the grid, so it has no
    ``latent_classes``.  Shape class ``c`` and shape value ``v`` satisfy
    ``v == c + 1`` in dSprites (values 1.0/2.0/3.0 vs classes 0/1/2), so
    ``latent_classes[:, shp] == 1`` is exactly ``latent_values[:, shp] == 2.0``
    (ellipse).  ``_check_class_value_correspondence`` asserts this at load time
    rather than trusting it.
    """
    latent_values = np.asarray(latent_values)
    return (
        (latent_values[:, shp] == 2.0)  # ellipse
        & (latent_values[:, scl] > 0.6)
        & (latent_values[:, rot] > 0.0)
        & (latent_values[:, rot] < a120)
        & (latent_values[:, rot] > a240)
        & (latent_values[:, tx] > 0.66)
        & (latent_values[:, ty] > 0.66)
    )


def _check_class_value_correspondence(
    latent_values: np.ndarray, latent_classes: np.ndarray
) -> None:
    """Assert shape value == shape class + 1 so the two predicates coincide."""
    if not np.array_equal(
        latent_values[:, shp], latent_classes[:, shp].astype(np.float64) + 1.0
    ):
        raise AssertionError(
            "dSprites: shape value != shape class + 1; the value-space hold-out "
            "predicate would not match the verbatim class-space filter"
        )


def download_dsprites(
    root: str | Path | None = None,
    with_images: bool = True,
    verify: bool = True,
) -> dict:
    """Fetch the npz (and optionally the hdf5) into ``root``; verify checksums.

    Returns a provenance dict (path, url, bytes, sha256, licence) per file.
    """
    from .industrial.base import download_file  # local import: keeps import cheap

    root = dsprites_root(root)
    root.mkdir(parents=True, exist_ok=True)
    wanted = [(NPZ_NAME, NPZ_URL, NPZ_SHA256, NPZ_BYTES)]
    if with_images:
        wanted.append((HDF5_NAME, HDF5_URL, HDF5_SHA256, HDF5_BYTES))

    prov = {}
    for name, url, sha, nbytes in wanted:
        dest = root / name
        download_file(url, dest, sha256=sha if verify else None)
        got = dest.stat().st_size
        if got != nbytes:
            raise IOError(f"{name}: expected {nbytes} bytes, got {got}")
        prov[name] = {
            "path": str(dest),
            "url": url,
            "bytes": got,
            "sha256": file_sha256(dest) if verify else sha,
            "licence": LICENCE,
            "citation": CITATION,
        }
    return prov


def load_latents(root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load the dSprites latents, **colour column already sliced off** (T1).

    Returns:
        (latent_values, latent_classes), each (737280, 5).
    """
    path = dsprites_root(root) / NPZ_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `chlu data dsprites --download` "
            f"(source: {NPZ_URL})"
        )
    # allow_pickle is required only for the 'metadata' member, which we skip.
    with np.load(path, allow_pickle=True, encoding="latin1") as z:
        latent_values = np.asarray(z["latents_values"])[:, 1:]  # ⛔ T1
        latent_classes = np.asarray(z["latents_classes"])[:, 1:]  # ⛔ T1
    return latent_values, latent_classes


def open_images(root: str | Path | None = None):
    """Open the hdf5 ``imgs`` dataset for lazy per-row access (returns h5py File).

    The npz stores ``imgs`` compressed; decompressing it costs ~3 GB of RAM, so
    image access goes through the hdf5 mirror (verified row-identical to the
    npz at load time by :func:`build_split` when ``check_images`` is set).
    """
    h5py = _require_h5py()
    path = dsprites_root(root) / HDF5_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `chlu data dsprites --download` "
            f"(source: {HDF5_URL})"
        )
    return h5py.File(path, "r")


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "dSprites/Shapes3D image access requires h5py: `uv pip install h5py`"
        ) from exc


def build_grid(latent_values: np.ndarray) -> FactorGrid:
    """Exhaustive factor grid from the realised (post-slice) latent values."""
    return factor_grid_from_latents(latent_values, FACTOR_NAMES, DATASET)


def build_split(
    root: str | Path | None = None,
    expected_n_test: int | None = N_TEST_REALISED,
    verify_bijection: bool = True,
) -> tuple[GenSplit, np.ndarray, np.ndarray]:
    """Build the Montero recomb-to-element split and run the load-time checks.

    Order matters: the non-empty assertion (T1) runs **first**, before the
    bijection check and before the hold-out assertion.

    Returns:
        (split, latent_values, latent_classes).
    """
    latent_values, latent_classes = load_latents(root)
    _check_class_value_correspondence(latent_values, latent_classes)

    test_mask = montero_test_mask(latent_values, latent_classes)
    all_rows = np.arange(len(latent_values), dtype=np.int64)
    grid = build_grid(latent_values)
    split = GenSplit(
        dataset=DATASET,
        condition="recomb2element",
        variant="leave1out",
        grid=grid,
        train_rows=all_rows[~test_mask],
        test_rows=all_rows[test_mask],
        test_predicate=_predicate_from_values,
    )

    # 1. T1 first: an empty held-out set makes every later check vacuous.
    assert_non_empty_test(split, expected=expected_n_test)
    # 2. train + test must partition the grid: nothing lost, nothing doubled.
    assert_index_partitions_grid(split)
    # 3. axis order / grid ordering.
    if verify_bijection:
        grid.verify_bijection(latent_values)
    return split, latent_values, latent_classes


def load_dsprites(
    root: str | Path | None = None,
    quick: bool = False,
    quick_rows: int = 4096,
    seed: int = 42,
    holdout_assert: bool = True,
) -> dict:
    """Config-driven entry point: split + factor grid + assertion report.

    Args:
        root: dataset directory (default ``$CHLU_GEN_DATA_ROOT``/dsprites).
        quick: subsample the *returned* train row index (the hold-out assertion
            still runs over the FULL train split — it is a mask, and cheap).
        quick_rows: size of the ``--quick`` train subsample.
        seed: subsample seed (threaded from ``config.project.seed``).
        holdout_assert: run :func:`assert_holdout_disjoint`.

    Returns:
        dict with ``split``, ``grid``, ``latent_values``, ``latent_classes``,
        ``n_checked``, ``quick_train_rows``.
    """
    from .gen_splits import _quick_subsample

    split, latent_values, latent_classes = build_split(root)
    n_checked = assert_holdout_disjoint(split) if holdout_assert else 0
    quick_train = (
        _quick_subsample(split.train_rows, quick_rows, seed)
        if quick
        else split.train_rows
    )
    return {
        "split": split,
        "grid": split.grid,
        "latent_values": latent_values,
        "latent_classes": latent_classes,
        "n_checked": n_checked,
        "quick_train_rows": quick_train,
    }
