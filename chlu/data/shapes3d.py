"""Shapes3D (3D Shapes) loader + the Montero recombination-to-element split.

Source of truth is the official implementation
``mmrl/disent-and-gen@bff35a7570d08fdef0d10c21b093b9a24d98dde6``,
``scripts/configs/datasplits.py`` (class ``Shapes3D``, ``leave1out``).  The
paper's prose rule is a loose gloss that is partly *not executable*
(``object-scale = 1`` is not in the scale grid) and partly inverted
(``object-orientation = 0`` vs the code's ``> 20``); the code wins.

Unlike dSprites there is **no column slice**: all six label columns are used in
order, and every term reads the float64 ``labels`` array.  The float-boundary
hazard is LOW here, but the filter is still evaluated against the actual file
rather than a regenerated grid.

Held-out size is **72 of 480,000** (T4).  Any other number means something is
wrong with the data or the filter — it is not to be reconciled by adjusting
either.

Data: ``3dshapes.h5`` (268 MB, single HDF5 with ``images`` and ``labels``).
``labels`` is 23 MB and is read whole; ``images`` (1.7 GB uncompressed) is
**never** loaded whole — access it lazily through :func:`open_images`.
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

DATASET = "shapes3d"

H5_NAME = "3dshapes.h5"
H5_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
GCS_PATH = "gs://3d-shapes/3dshapes.h5"
# Upstream publishes no checksum; computed on first download (2026-09-09).
H5_SHA256 = "0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785"
H5_BYTES = 267573662
LICENCE = "Apache License 2.0 (google-deepmind/3d-shapes)"
CITATION = "Burgess & Kim (2018), 3D Shapes Dataset"

FACTOR_NAMES = (
    "floor_hue",
    "wall_hue",
    "object_hue",
    "scale",
    "shape",
    "orientation",
)
CARDINALITIES = (10, 10, 10, 8, 4, 15)
N_IMAGES = 480000

# --- verbatim from scripts/configs/datasplits.py ---------------------------
fh, wh, oh, scl, shp, rot = 0, 1, 2, 3, 4, 5
# ---------------------------------------------------------------------------

#: Realised held-out size, measured against the actual h5 labels (T4).
#: 2 (floor_hue) * 2 (wall_hue) * 2 (object_hue) * 3 (scale) * 1 (cylinder)
#: * 3 (orientation) = 72.
N_TEST_REALISED = 72
N_TRAIN_REALISED = N_IMAGES - N_TEST_REALISED


def shapes3d_root(root: str | Path | None = None) -> Path:
    return Path(root).expanduser() if root else default_gen_data_root() / DATASET


def montero_test_mask(latent_values: np.ndarray) -> np.ndarray:
    """The verbatim recombination-to-element / leave1out test filter.

    Args:
        latent_values: (N, 6) float ``labels`` array (no column is dropped).

    Returns:
        (N,) boolean mask of held-out rows.  ``shp == 1`` on the raw label
        values is the CYLINDER (T5: the same literal means a different shape in
        the dSprites condition, which indexes ``latent_classes``).
    """
    latent_values = np.asarray(latent_values)
    if latent_values.shape[1] != 6:
        raise ValueError(
            f"Shapes3D filter expects 6 label columns, got {latent_values.shape}"
        )
    return (
        (latent_values[:, oh] >= 0.8)
        & (latent_values[:, wh] >= 0.8)
        & (latent_values[:, fh] >= 0.8)
        & (latent_values[:, scl] >= 1.1)
        & (latent_values[:, shp] == 1)  # cylinder
        & (latent_values[:, rot] > 20)
    )


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Shapes3D is a single HDF5 file and requires h5py: `uv pip install h5py`"
        ) from exc


def download_shapes3d(root: str | Path | None = None, verify: bool = True) -> dict:
    """Fetch ``3dshapes.h5`` into ``root`` and verify its checksum."""
    from .industrial.base import download_file

    root = shapes3d_root(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / H5_NAME
    download_file(H5_URL, dest, sha256=H5_SHA256 if verify else None, timeout=600)
    got = dest.stat().st_size
    if got != H5_BYTES:
        raise IOError(f"{H5_NAME}: expected {H5_BYTES} bytes, got {got}")
    return {
        H5_NAME: {
            "path": str(dest),
            "url": H5_URL,
            "gcs": GCS_PATH,
            "bytes": got,
            "sha256": file_sha256(dest) if verify else H5_SHA256,
            "licence": LICENCE,
            "citation": CITATION,
        }
    }


def load_labels(root: str | Path | None = None) -> np.ndarray:
    """Read the (480000, 6) float64 ``labels`` array (23 MB; images not touched)."""
    h5py = _require_h5py()
    path = shapes3d_root(root) / H5_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `chlu data shapes3d --download` (source: {H5_URL})"
        )
    with h5py.File(path, "r") as f:
        return np.asarray(f["labels"][:])


def open_images(root: str | Path | None = None):
    """Open ``3dshapes.h5`` for lazy image access; caller closes the h5py File.

    ⛔ Do not do ``f['images'][:]`` — that materialises 1.7 GB.  Slice the rows
    you need (``f['images'][rows]`` with a sorted index) instead.
    """
    h5py = _require_h5py()
    path = shapes3d_root(root) / H5_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `chlu data shapes3d --download`"
        )
    return h5py.File(path, "r")


def build_grid(labels: np.ndarray) -> FactorGrid:
    """Exhaustive factor grid from the realised label values."""
    return factor_grid_from_latents(labels, FACTOR_NAMES, DATASET)


def build_split(
    root: str | Path | None = None,
    expected_n_test: int | None = N_TEST_REALISED,
    verify_bijection: bool = True,
    labels: np.ndarray | None = None,
) -> tuple[GenSplit, np.ndarray]:
    """Build the Montero recomb-to-element split and run the load-time checks.

    Returns:
        (split, labels).
    """
    if labels is None:
        labels = load_labels(root)
    test_mask = montero_test_mask(labels)
    all_rows = np.arange(len(labels), dtype=np.int64)
    grid = build_grid(labels)
    split = GenSplit(
        dataset=DATASET,
        condition="recomb2element",
        variant="leave1out",
        grid=grid,
        train_rows=all_rows[~test_mask],
        test_rows=all_rows[test_mask],
        test_predicate=montero_test_mask,
    )
    assert_non_empty_test(split, expected=expected_n_test)
    assert_index_partitions_grid(split)
    if verify_bijection:
        grid.verify_bijection(labels)
    return split, labels


def load_shapes3d(
    root: str | Path | None = None,
    quick: bool = False,
    quick_rows: int = 4096,
    seed: int = 42,
    holdout_assert: bool = True,
) -> dict:
    """Config-driven entry point: split + factor grid + assertion report.

    See :func:`chlu.data.dsprites.load_dsprites` for argument semantics; the
    hold-out assertion always runs over the FULL train split even under
    ``--quick``.
    """
    from .gen_splits import _quick_subsample

    split, labels = build_split(root)
    n_checked = assert_holdout_disjoint(split) if holdout_assert else 0
    quick_train = (
        _quick_subsample(split.train_rows, quick_rows, seed)
        if quick
        else split.train_rows
    )
    return {
        "split": split,
        "grid": split.grid,
        "labels": labels,
        "n_checked": n_checked,
        "quick_train_rows": quick_train,
    }
