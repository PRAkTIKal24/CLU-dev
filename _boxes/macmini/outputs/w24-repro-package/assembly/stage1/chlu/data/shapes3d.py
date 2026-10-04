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
    FactorCell,
    FactorGrid,
    GenSplit,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_non_empty_test,
    build_cell_split,
    default_gen_data_root,
    factor_grid_from_latents,
    file_sha256,
)
DATASET = "shapes3d"
H5_NAME = "3dshapes.h5"
H5_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
GCS_PATH = "gs://3d-shapes/3dshapes.h5"
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
fh, wh, oh, scl, shp, rot = 0, 1, 2, 3, 4, 5
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


WIDE_CONDITION = "recomb2element_wide"
WIDE_VARIANT = "leave1out_multicell"
WIDE_RULE = (
    "recomb2element_wide/v1 (Shapes3D). Each held-out CELL is a conjunction of the "
    "SAME KIND as Montero's: a band of B consecutive values (circular) on each of "
    "floor_hue / wall_hue / object_hue, a corner of S consecutive values of scale, "
    "exactly ONE shape, and a corner of R consecutive values of orientation. "
    "B, S and R are DERIVED from Montero's own cell (B=|{hue>=0.8}|, "
    "S=|{scale>=1.1}|, R=|{orientation>20}|), so every cell has Montero's shape and "
    "size. Cell j (0-based, j = 0..n_cells-1) is: "
    "contrast step s_j = j mod 6; shape_j = grid shape value at index (j mod n_shapes); "
    "scale corner = 'high' if (j // 2) % 2 == 0 else 'low'; "
    "orientation corner = 'high' if (j // 3) % 2 == 0 else 'low'; "
    "floor band start f_j = rng.integers(n_hue) with rng = numpy.random.default_rng(seed) "
    "drawn once per candidate, in cell order; wall band = floor band (as in Montero's "
    "cell, where all three hues sit in one band); object band start = (f_j + s_j) mod "
    "n_hue. A candidate whose rows intersect an already-accepted cell (or Montero's "
    "cell, when include_montero) is REJECTED and f_j is redrawn from the same rng. "
    "When include_montero, Montero's cell is carried first, unmodified, with cell_id "
    "'montero'. The hue ring is circular with period derived from the grid "
    "(max(values) - min(values) + step)."
)
WIDE_CONTRAST_STEPS = (0, 1, 2, 3, 4, 5)
WIDE_MAX_DRAWS = 10_000
def hue_period(values: np.ndarray) -> float:
    """Circular period of a hue axis, derived from the grid (Shapes3D: 1.0)."""
    values = np.asarray(values, dtype=float)
    step = float(np.min(np.diff(values))) if values.size >= 2 else 1.0
    return float(values[-1] - values[0] + step)


def hue_circular_distance(a, b, period: float = 1.0):
    """Circular distance on the hue ring; in [0, period/2]."""
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % period
    return np.minimum(d, period - d)


def montero_cell(grid: FactorGrid) -> FactorCell:
    """Montero's held-out set expressed as a :class:`FactorCell`.

    Built from the same inequalities as :func:`montero_test_mask`, so the two are
    checkable against each other — and they are checked
    (``test_montero_cell_reproduces_the_montero_mask``).  ⛔ This does not change
    the Montero split; it only lets the wide condition carry Montero's cell.
    """
    v = {name: grid.values[grid.index_of(name)] for name in FACTOR_NAMES}
    return FactorCell(
        cell_id="montero",
        values={
            "floor_hue": tuple(float(x) for x in v["floor_hue"][v["floor_hue"] >= 0.8]),
            "wall_hue": tuple(float(x) for x in v["wall_hue"][v["wall_hue"] >= 0.8]),
            "object_hue": tuple(
                float(x) for x in v["object_hue"][v["object_hue"] >= 0.8]
            ),
            "scale": tuple(float(x) for x in v["scale"][v["scale"] >= 1.1]),
            "shape": (1.0,),
            "orientation": tuple(
                float(x) for x in v["orientation"][v["orientation"] > 20]
            ),
        },
        meta={"source": "Montero et al., recomb2element/leave1out, verbatim"},
    )


def cell_contrast(grid: FactorGrid, cell: FactorCell) -> dict:
    """Photometric **contrast label** of a cell, measured from its realised rows.

    ⛔ A measured quantity, not a judgement: the mean over the cell's rows of the
    circular hue distance from the object to the two background surfaces,
    ``0.5 * (d(object, floor) + d(object, wall))``.  Montero's cell puts all
    three hues in one band, which is why its label sits at the bottom of the
    range — that collapse is the whole reason the wide split exists (R15).
    """
    vals = grid.values_of(cell.rows(grid))
    period = hue_period(grid.values[grid.index_of("object_hue")])
    d_floor = hue_circular_distance(vals[:, oh], vals[:, fh], period)
    d_wall = hue_circular_distance(vals[:, oh], vals[:, wh], period)
    per_row = 0.5 * (d_floor + d_wall)
    return {
        "object_bg_hue_contrast": float(per_row.mean()),
        "object_bg_hue_contrast_min": float(per_row.min()),
        "object_bg_hue_contrast_max": float(per_row.max()),
        "object_bg_hue_contrast_nearest": float(np.minimum(d_floor, d_wall).mean()),
        "max_possible": float(period / 2.0),
        "n_rows_realised": int(len(vals)),
    }


def _band(values: np.ndarray, start: int, width: int) -> tuple[float, ...]:
    """``width`` consecutive grid values from ``start``, wrapping (circular hue)."""
    idx = (np.arange(start, start + width) % values.size).astype(int)
    return tuple(float(x) for x in np.sort(values[idx]))


def _corner(values: np.ndarray, side: str, width: int) -> tuple[float, ...]:
    """The top (``'high'``) or bottom (``'low'``) ``width`` values of an axis."""
    return tuple(
        float(x) for x in (values[-width:] if side == "high" else values[:width])
    )


def build_wide_cells(
    grid: FactorGrid,
    n_cells: int,
    seed: int,
    include_montero: bool = True,
) -> list[FactorCell]:
    """The wide condition's cells, deterministic given ``(n_cells, seed)``.

    See :data:`WIDE_RULE` for the rule verbatim.  Band and corner widths are
    derived from Montero's cell so each wide cell is structurally identical to
    it; only its *placement* in factor space moves — deliberately across the
    hue/contrast range, which is what lets composition be separated from
    rendering (R15's photometric finding).
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")
    ref = montero_cell(grid)
    band_w = len(ref.values["floor_hue"])
    scale_w = len(ref.values["scale"])
    rot_w = len(ref.values["orientation"])

    hues = grid.values[grid.index_of("floor_hue")]
    scales = grid.values[grid.index_of("scale")]
    shapes = grid.values[grid.index_of("shape")]
    rots = grid.values[grid.index_of("orientation")]

    rng = np.random.default_rng(seed)
    cells: list[FactorCell] = [ref] if include_montero else []
    # ⭐ Montero's rows are excluded from the ADDITIONAL cells either way, so the
    # additional cells are bit-identical whichever way ``include_montero`` is
    # set: the flag only decides whether Montero's own cell is carried, never
    # where the other cells land, and the extra cells are always "beyond
    # Montero's 72" as the task requires.
    reserved = set(ref.rows(grid).tolist())
    n_rejected = 0

    for j in range(n_cells):
        s_j = WIDE_CONTRAST_STEPS[j % len(WIDE_CONTRAST_STEPS)]
        shape_j = float(shapes[j % shapes.size])
        scale_side = "high" if (j // 2) % 2 == 0 else "low"
        rot_side = "high" if (j // 3) % 2 == 0 else "low"
        taken = reserved | {row for c in cells for row in c.rows(grid).tolist()}

        for attempt in range(WIDE_MAX_DRAWS):
            f_j = int(rng.integers(hues.size))
            bg = _band(hues, f_j, band_w)
            obj = _band(hues, f_j + s_j, band_w)
            cand = FactorCell(
                cell_id=f"wide-{j:03d}",
                values={
                    "floor_hue": bg,
                    "wall_hue": bg,
                    "object_hue": obj,
                    "scale": _corner(scales, scale_side, scale_w),
                    "shape": (shape_j,),
                    "orientation": _corner(rots, rot_side, rot_w),
                },
                meta={
                    "contrast_step": int(s_j),
                    "floor_band_start": f_j,
                    "object_band_start": int((f_j + s_j) % hues.size),
                    "scale_corner": scale_side,
                    "orientation_corner": rot_side,
                    "shape_value": shape_j,
                    "draws_used": attempt + 1,
                },
            )
            if not (taken & set(cand.rows(grid).tolist())):
                break
            n_rejected += 1
        else:  # pragma: no cover - 10k rejections means the rule is exhausted
            raise RuntimeError(
                f"cell {j}: {WIDE_MAX_DRAWS} draws all intersected an accepted cell"
            )
        cells.append(cand)

    for cell in cells:
        cell.meta.update(cell_contrast(grid, cell))
    if cells:
        cells[0].meta["n_candidates_rejected_total"] = n_rejected
    return cells


def build_wide_split(
    root: str | Path | None = None,
    n_cells: int = 12,
    seed: int = 0,
    include_montero: bool = True,
    verify_bijection: bool = True,
    labels: np.ndarray | None = None,
) -> tuple[GenSplit, np.ndarray, list[FactorCell]]:
    """Build the widened SECONDARY split; return ``(split, labels, cells)``.

    ⛔ Realised counts only: nothing here asserts a target ``n_test``.  The
    caller reports what the grid yields.
    """
    if labels is None:
        labels = load_labels(root)
    grid = build_grid(labels)
    cells = build_wide_cells(
        grid, n_cells=n_cells, seed=seed, include_montero=include_montero
    )
    split = build_cell_split(
        grid, cells, dataset=DATASET, condition=WIDE_CONDITION, variant=WIDE_VARIANT
    )
    assert_non_empty_test(split)
    assert_index_partitions_grid(split)
    if verify_bijection:
        grid.verify_bijection(labels)
    return split, labels, cells


def wide_metadata(split: GenSplit, cells: list[FactorCell], seed: int) -> dict:
    """Emitted metadata: the rule verbatim, per-cell conjunctions, realised counts."""
    return {
        "dataset": DATASET,
        "condition": WIDE_CONDITION,
        "variant": WIDE_VARIANT,
        "rule": WIDE_RULE,
        "seed": int(seed),
        "n_cells": len(cells),
        "n_total": split.n_total,
        "n_test_realised": split.n_test,
        "n_train_realised": split.n_train,
        "montero_headline_untouched": {
            "condition": "recomb2element",
            "n_test": N_TEST_REALISED,
            "note": (
                "⛔ Montero's split is the headline and is unchanged; the wide "
                "condition is reported SEPARATELY and never pooled with it."
            ),
        },
        "frozen_artifact_warning": (
            "⛔ The encoder/decoder/store frozen under `recomb2element` were "
            "trained on rows this condition holds out. They have SEEN these rows "
            "and must not be scored on them; the wide condition requires a full "
            "re-run of the stack."
        ),
        "cells": [c.to_dict(split.grid) for c in cells],
    }


def load_shapes3d(
    root: str | Path | None = None,
    quick: bool = False,
    quick_rows: int = 4096,
    seed: int = 42,
    holdout_assert: bool = True,
    condition: str = "recomb2element",
    wide_n_cells: int = 12,
    wide_seed: int = 0,
    wide_include_montero: bool = True,
) -> dict:
    """Config-driven entry point: split + factor grid + assertion report.

    See :func:`chlu.data.dsprites.load_dsprites` for argument semantics; the
    hold-out assertion always runs over the FULL train split even under
    ``--quick``.

    ``condition`` selects ``"recomb2element"`` (Montero's headline split,
    default, unchanged) or ``"recomb2element_wide"`` (the widened SECONDARY
    condition, R17).  ⛔ The two are separate configurations and their numbers
    are never pooled.
    """
    from .gen_splits import _quick_subsample

    cells = None
    if condition == WIDE_CONDITION:
        split, labels, cells = build_wide_split(
            root,
            n_cells=wide_n_cells,
            seed=wide_seed,
            include_montero=wide_include_montero,
        )
    elif condition == "recomb2element":
        split, labels = build_split(root)
    else:
        raise ValueError(
            f"unknown Shapes3D condition {condition!r}; expected 'recomb2element' "
            f"or {WIDE_CONDITION!r}"
        )
    n_checked = assert_holdout_disjoint(split) if holdout_assert else 0
    quick_train = (
        _quick_subsample(split.train_rows, quick_rows, seed)
        if quick
        else split.train_rows
    )
    out = {
        "split": split,
        "grid": split.grid,
        "labels": labels,
        "n_checked": n_checked,
        "quick_train_rows": quick_train,
    }
    if cells is not None:
        out["cells"] = cells
        out["wide_metadata"] = wide_metadata(split, cells, wide_seed)
    return out


