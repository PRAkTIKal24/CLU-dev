"""⭐ The k-WAY COMPOSITION LADDER — the middle of the value-hold-out ladder.

⛔ **Read §0 before reading a single number produced by this module.**

0. THE TWO LAYERS, and why confusing them silently voids the ladder
-------------------------------------------------------------------
A **cell** (:class:`~chlu.data.gen_splits.FactorCell`) is what is removed from the
store's **fit pool**.  A **corner** is a Montero-shaped probe box **inside** a cell —
it is the **unit of the paired estimand** (``exp_gen_factor_wells.paired_estimand``).
They are not the same object and the two existing ends do not relate them the same way:

=========  ==========================  ==========  ===========
row        family                      cells       corners
=========  ==========================  ==========  ===========
row 3      ``pair-*``   (k = 2)        3           60  (tiled)
row 4      ``nway-*``   (k = 6)        49          49  (1 : 1)
=========  ==========================  ==========  ===========

⭐ **The >= 49 bar of w3-hub-common §3 is a bar on CORNERS.**  Row 3 clears it with 3
cells because ``exp_gen_factor_wells.pair_subcorners`` **tiles** each cell over the
factors the cell leaves free, using *the same widths the n-way builder uses*.  This
module generalises exactly that: pin ``k`` factors, tile the other ``6 - k``.

1. The coordinate system — five UNITS, not six factors
-------------------------------------------------------
Every corner in rows 3 and 4 is the same shape: a band of ``hue_band`` consecutive
floor hues **and the identical band of wall hues**, a band of ``hue_band`` consecutive
object hues, a corner of ``scale_corner`` consecutive scales, exactly ONE shape, and a
corner of ``orientation_corner`` consecutive orientations.  Both ends tie ``wall_hue``
to ``floor_hue`` (``build_nway_cells`` uses one ``bg`` for both; ``pair_subcorners``
assigns one ``band`` to both).  So the ladder's coordinates are

``floorwall``    ⭐ **ATOMIC — it pins TWO factors.**  Disjoint width-2 floor bands.
``object_hue``   disjoint width-2 bands over the hues no other family reserved.
``scale``        {high, low} corner.
``shape``        one retained shape.
``orientation``  {high, low} corner.

⛔ **Consequence, and it is a real constraint on the instrument, not a preference:**
because ``floorwall`` is atomic, **no strictly nested chain of pinned-axis sets passes
through every one of k = 2,3,4,5,6.**  From a k=4 set of four single factors the only
unit left to add is ``floorwall``, which lands on k=6.  :data:`LADDER_AXES` therefore
branches: ``A2 < A3 < A4`` and ``A3 < A5 < A6``, with ``A4`` NOT a subset of ``A5``.
:data:`LADDER_AXES_NESTED_ALT` is the alternative that *is* nested — it unties
``wall_hue`` at k=5 — at the cost of corners whose floor band and wall band differ,
which ⛔ **is not the geometry rows 3 and 4 use**.  Both are emitted; the primary is
the tied one; the hub chooses with both numbers in front of it.

2. ⭐ Scale and orientation
----------------------------
``scale`` and ``orientation`` stay **INELIGIBLE as VALUE hold-outs**
(:mod:`chlu.data.gen_value_holdout` §2 — they are physical defaults the model degrades
*towards*).  They are **eligible as pinned axes inside a k-way box, exactly as in row
4**, and ⛔ nowhere else.  This module never constructs a ``value-scale`` or a
``value-orientation`` cell and has no code path that could.

3. ⭐ The check that can KILL the ladder is written and run BEFORE the ladder
-----------------------------------------------------------------------------
:func:`ladder_capacity_report` is pure combinatorics over the coordinate option sets
and needs no cells built.  :func:`assert_corners_disjoint` is the row-level proof, run
on every rung.  :func:`assert_k6_matches_row4` and :func:`assert_k2_matches_row3` are
the end-reproduction assertions.  ⛔ If k=2 disagrees with row 3, **this builder is
wrong, not row 3** — that is why E2 compares row sets and ids, not a summary.

4. ⛔ Declared confound, reported never corrected here
-------------------------------------------------------
A lower-k cell pins fewer factors => covers more grid rows => removes more rows from
the fit pool.  The ladder confounds *composition depth* with *held-out fraction*.
:func:`ladder_capacity_report` reports ``held_out_rows`` and ``held_out_fraction`` per
rung and the minimal cell budget that still clears 49.  ⛔ This module does not pick a
correction; ``FINDINGS.md`` names the control that would remove it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .gen_splits import (
    FactorCell,
    FactorGrid,
    GenSplit,
    assert_cells_disjoint,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_non_empty_test,
    build_cell_split,
    write_json_with_checksum,
)
from .gen_value_holdout import (
    VALUE_HOLDOUT_SCOPE,
    ValueHoldoutConfig,
    _consecutive_runs,
    _disjoint_windows,
    build_nway_cells,
    build_pair_cells,
    build_value_cells,
)

__all__ = [
    "CELL_NESTING",
    "COORD_FACTORS",
    "CORNER_BAR",
    "KWAY_CONDITION",
    "KWAY_WIDTHS",
    "LADDER_AXES",
    "LADDER_AXES_NESTED_ALT",
    "ROW3_TILE_WIDTHS",
    "TILE_NESTING",
    "KwayLadderConfig",
    "assert_corners_disjoint",
    "assert_corners_inside_holdout",
    "assert_k2_matches_row3",
    "assert_k6_matches_row4",
    "assert_row3_widths_match_pins",
    "build_kway_cells",
    "build_kway_condition_cells",
    "build_kway_split",
    "coordinate_options",
    "corner_cell",
    "fit_pool_feasibility",
    "emit_ladder_ready",
    "k_of_axes",
    "kway_corners",
    "ladder_capacity_report",
    "row_matching_feasibility",
    "row3_subcorners",
]

KWAY_CONDITION = "kway_ladder"

#: ⭐ w3-hub-common §3.  The bar is on CORNERS.  ⛔ Not a tunable.
CORNER_BAR = 49

#: ⭐ The widths ``build_nway_cells`` uses (its own defaults) and the widths
#: ``exp_gen_factor_wells.PINS`` uses for row 3's tiling.  They are the same three
#: numbers, and the ladder inherits them verbatim.
KWAY_WIDTHS = {"hue_band": 2, "scale_corner": 3, "orientation_corner": 3}

#: Coordinate UNIT -> the factors it pins.  ``floorwall`` pins TWO (§1).
COORD_FACTORS: dict[str, tuple[str, ...]] = {
    "floorwall": ("floor_hue", "wall_hue"),
    "floor_only": ("floor_hue",),
    "wall_only": ("wall_hue",),
    "object_hue": ("object_hue",),
    "scale": ("scale",),
    "shape": ("shape",),
    "orientation": ("orientation",),
}

#: ⭐ Cell enumeration nesting, **verbatim from** ``build_nway_cells``'s odometer
#: (``for fi ... for rc ... for sc ... for oi ... for si``) — shape varies fastest.
#: Keeping this order is what makes the k=6 rung reproduce row 4 cell-for-cell.
CELL_NESTING = ("floorwall", "orientation", "scale", "object_hue", "shape")

#: ⭐ Within-cell TILE nesting, **verbatim from**
#: ``exp_gen_factor_wells.pair_subcorners`` (``for fi ... for scn ... for rcn``).
#: ``object_hue`` and ``shape`` are pinned at every rung, so they never appear here.
#: Keeping this order is what makes the k=2 rung reproduce row 3 corner-for-corner.
TILE_NESTING = ("floorwall", "scale", "orientation")

#: ⭐ Row 3's tiling widths, **verbatim from** ``exp_gen_factor_wells.PINS``'s
#: ``pair_tile_hue_band`` / ``pair_tile_scale_corner`` / ``pair_tile_orientation_corner``.
#:
#: ⛔ Written out as literals on purpose.  E2 is a check that **this module** still
#: agrees with the hub's row 3; if its reference side read ``cfg.widths`` the check
#: would drift with the thing it is checking and would pass unconditionally.
#: :func:`assert_row3_widths_match_pins` re-ties these literals to the runner's pins.
ROW3_TILE_WIDTHS = {"hue_band": 2, "scale_corner": 3, "orientation_corner": 3}

#: ⭐ THE PRIMARY LADDER.  Every corner is geometrically identical to a row-3/row-4
#: corner (floor band == wall band).  ⛔ ``A4`` is not a subset of ``A5`` — see §1;
#: that is forced by ``floorwall`` being atomic, not chosen.
LADDER_AXES: dict[int, tuple[str, ...]] = {
    2: ("object_hue", "shape"),
    3: ("object_hue", "shape", "scale"),
    4: ("object_hue", "shape", "scale", "orientation"),
    5: ("object_hue", "shape", "scale", "floorwall"),
    6: ("object_hue", "shape", "scale", "orientation", "floorwall"),
}

#: The nested alternative for k=5 only: ``wall_hue`` is UNTIED and becomes a free,
#: tiled coordinate, so ``A4 < A5n < A6``.  ⛔ Its corners have ``floor_hue`` band !=
#: ``wall_hue`` band in 4 of every 5 tiles, which is **not** rows 3/4 geometry.
LADDER_AXES_NESTED_ALT: dict[int, tuple[str, ...]] = {
    5: ("object_hue", "shape", "scale", "orientation", "floor_only"),
}


def k_of_axes(axes: Sequence[str]) -> int:
    """How many of the six FACTORS a pinned-unit set pins.  ``floorwall`` counts 2."""
    return sum(len(COORD_FACTORS[u]) for u in axes)


@dataclass(frozen=True)
class KwayLadderConfig:
    """Configuration of the ladder.

    ⛔ The value-hold-out context is **inherited, never re-derived**, so a rung cannot
    silently disagree with rows 3–5 about which hue is held out, which shape is held
    out, or which hues the ``pair-*`` cells reserved.
    """

    #: The value-hold-out config the rungs are built against.
    value_cfg: ValueHoldoutConfig = field(default_factory=ValueHoldoutConfig)
    #: Montero widths.  Default = :data:`KWAY_WIDTHS` = the n-way builder's own.
    widths: dict = field(default_factory=lambda: dict(KWAY_WIDTHS))
    #: ⭐ ``None`` => emit the FULL coordinate sweep for the rung (every box).  A full
    #: sweep is balanced over every coordinate option; a truncated one is not, and at
    #: the small budgets the odometer would hand back only one ``scale`` corner.
    n_cells: Optional[int] = None
    #: Which ladder to build from: ``"primary"`` (tied floor/wall) or ``"nested_alt"``.
    variant: str = "primary"

    def axes_for(self, k: int) -> tuple[str, ...]:
        if self.variant == "nested_alt" and k in LADDER_AXES_NESTED_ALT:
            return LADDER_AXES_NESTED_ALT[k]
        if k not in LADDER_AXES:
            raise ValueError(f"no ladder rung for k={k}; have {sorted(LADDER_AXES)}")
        return LADDER_AXES[k]

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["value_cfg"] = self.value_cfg.to_dict()
        return d


# ---------------------------------------------------------------------------
# 1. the coordinate options — the whole combinatorial basis of the ladder
# ---------------------------------------------------------------------------


def coordinate_options(
    grid: FactorGrid,
    cfg: KwayLadderConfig,
    pair_cells: Sequence[FactorCell] = (),
) -> dict[str, list[dict]]:
    """Per-unit option lists.  ⛔ Each unit's options are pairwise DISJOINT by
    construction, which is the whole disjointness argument (§3).

    The hue / scale / orientation option sets are computed **the same way**
    ``build_nway_cells`` computes them — same helpers, same reservations, same
    high/low corners — so the k=6 rung is that builder's own slot list.
    """
    vc = cfg.value_cfg
    band_w = int(cfg.widths.get("hue_band", 2))
    scale_w = int(cfg.widths.get("scale_corner", 3))
    rot_w = int(cfg.widths.get("orientation_corner", 3))

    hue_f, shape_f = vc.held_out_hue_factor, vc.held_out_shape_factor
    hue_vals = grid.values[grid.index_of(hue_f)]
    floor_vals = grid.values[grid.index_of("floor_hue")]
    wall_vals = grid.values[grid.index_of("wall_hue")]
    scale_vals = grid.values[grid.index_of("scale")]
    rot_vals = grid.values[grid.index_of("orientation")]

    # ⛔ identical reservation rule to build_nway_cells: the held-out hue, plus every
    # hue a pair cell owns (a pair cell is unconstrained on the other four factors and
    # would otherwise swallow a k-way box whole).
    reserved = {float(vc.held_out_hue_value)}
    for c in pair_cells:
        reserved.update(float(v) for v in c.values.get(hue_f, ()))
    allowed_hue_idx = np.array(
        [i for i, v in enumerate(hue_vals) if float(v) not in reserved], dtype=np.int64
    )
    obj_windows = _disjoint_windows(_consecutive_runs(allowed_hue_idx, band_w))
    floor_windows = _disjoint_windows(
        _consecutive_runs(np.arange(floor_vals.size, dtype=np.int64), band_w)
    )
    wall_windows = _disjoint_windows(
        _consecutive_runs(np.arange(wall_vals.size, dtype=np.int64), band_w)
    )
    if not obj_windows or not floor_windows:
        raise ValueError(
            f"no {band_w}-wide band survives the reserved {hue_f} values "
            f"{sorted(reserved)} — the ladder has no coordinate basis"
        )
    shapes = [
        float(v)
        for v in grid.values[grid.index_of(shape_f)]
        if float(v) != float(vc.held_out_shape_value)
    ]

    opts: dict[str, list[dict]] = {}
    opts["floorwall"] = [
        {
            "label": f"f{int(w[0])}",
            "values": {
                "floor_hue": tuple(float(x) for x in floor_vals[w]),
                "wall_hue": tuple(float(x) for x in floor_vals[w]),
            },
            "meta": {"floor_band_start_index": int(w[0])},
        }
        for w in floor_windows
    ]
    # nested_alt only: floor pinned alone, wall becomes its own free (tiled) unit.
    opts["floor_only"] = [
        {
            "label": f"f{int(w[0])}",
            "values": {"floor_hue": tuple(float(x) for x in floor_vals[w])},
            "meta": {"floor_band_start_index": int(w[0])},
        }
        for w in floor_windows
    ]
    opts["wall_only"] = [
        {
            "label": f"w{int(w[0])}",
            "values": {"wall_hue": tuple(float(x) for x in wall_vals[w])},
            "meta": {"wall_band_start_index": int(w[0])},
        }
        for w in wall_windows
    ]
    opts["object_hue"] = [
        {
            "label": f"o{int(w[0])}",
            "values": {hue_f: tuple(float(x) for x in hue_vals[w])},
            "meta": {"object_band_start_index": int(w[0])},
        }
        for w in obj_windows
    ]
    opts["scale"] = [
        {
            "label": "h",
            "values": {"scale": tuple(float(x) for x in scale_vals[-scale_w:])},
            "meta": {"scale_corner": "high"},
        },
        {
            "label": "l",
            "values": {"scale": tuple(float(x) for x in scale_vals[:scale_w])},
            "meta": {"scale_corner": "low"},
        },
    ]
    opts["orientation"] = [
        {
            "label": "h",
            "values": {"orientation": tuple(float(x) for x in rot_vals[-rot_w:])},
            "meta": {"orientation_corner": "high"},
        },
        {
            "label": "l",
            "values": {"orientation": tuple(float(x) for x in rot_vals[:rot_w])},
            "meta": {"orientation_corner": "low"},
        },
    ]
    opts["shape"] = [
        {"label": f"s{i}", "values": {shape_f: (s,)}, "meta": {"shape_value": s}}
        for i, s in enumerate(shapes)
    ]
    return opts


def _cell_nesting(axes: Sequence[str]) -> list[str]:
    """Pinned units in ``build_nway_cells``'s odometer order (slowest first)."""
    nesting = [u for u in CELL_NESTING if u in axes]
    if "floor_only" in axes:
        nesting = ["floor_only", *nesting]
    return nesting


def _free_units(axes: Sequence[str]) -> tuple[str, ...]:
    """The TILING units: every corner coordinate the rung's box leaves unpinned.

    ⛔ ``wall_only`` is a tiling unit **only** under ``nested_alt``, where the box pins
    ``floor_only``; under the primary ladder ``floorwall`` is one unit and there is no
    free wall coordinate.
    """
    pinned = set(axes)
    if "floor_only" in pinned:
        return ("wall_only", *(u for u in ("scale", "orientation") if u not in pinned))
    return tuple(u for u in TILE_NESTING if u not in pinned)


def _product(nesting: Sequence[str], opts: dict[str, list[dict]]) -> list[dict]:
    """Odometer over ``nesting``: the FIRST name varies SLOWEST, the last fastest."""
    combos: list[dict] = [{}]
    for unit in nesting:
        combos = [{**c, unit: o} for c in combos for o in opts[unit]]
    return combos


def _merge(chosen: dict[str, dict]) -> tuple[dict, dict]:
    values: dict[str, tuple[float, ...]] = {}
    meta: dict = {}
    for opt in chosen.values():
        values.update(opt["values"])
        meta.update(opt["meta"])
    return values, meta


# ---------------------------------------------------------------------------
# 2. the cells and the corners
# ---------------------------------------------------------------------------


def build_kway_cells(
    grid: FactorGrid,
    cfg: KwayLadderConfig,
    k: int,
    pair_cells: Sequence[FactorCell] = (),
) -> list[FactorCell]:
    """The rung's CELLS: ``k`` factors pinned to a Montero-shaped box, ``6 - k`` free.

    ⭐ ``k = 2`` returns the **shipped** ``pair-*`` cells verbatim
    (:func:`~chlu.data.gen_value_holdout.build_pair_cells`) — row 3 is an existing end
    and this builder does not get a second opinion about it.
    ⭐ ``k = 6`` returns cells whose ``values`` equal ``build_nway_cells``'s, in the
    same order; :func:`assert_k6_matches_row4` proves it.
    """
    if k == 2 and cfg.variant == "primary":
        return list(build_pair_cells(grid, cfg.value_cfg))
    axes = cfg.axes_for(k)
    opts = coordinate_options(grid, cfg, pair_cells)
    nesting = _cell_nesting(axes)
    combos = _product(nesting, opts)
    n = len(combos) if cfg.n_cells is None else int(cfg.n_cells)
    if n > len(combos):
        raise ValueError(
            f"k={k}: n_cells={n} exceeds the {len(combos)} structurally-disjoint boxes "
            f"available over the pinned units {tuple(axes)} "
            f"({' x '.join(f'{u}:{len(opts[u])}' for u in nesting)})"
        )
    cells: list[FactorCell] = []
    for j, chosen in enumerate(combos[:n]):
        values, meta = _merge(chosen)
        cells.append(
            FactorCell(
                cell_id=f"kway{k}-{j:03d}",
                values=values,
                meta={
                    "family": f"kway{k}",
                    "k": int(k),
                    "pinned_units": list(axes),
                    "pinned_factors": sorted(values),
                    "free_factors": sorted(set(grid.names) - set(values)),
                    "widths": dict(cfg.widths),
                    "variant": cfg.variant,
                    **meta,
                    "note": (
                        f"{k} of 6 factors jointly pinned to a Montero-shaped box, the "
                        f"other {6 - k} unconstrained; the corner geometry is rows 3/4's"
                    ),
                },
            )
        )
    return cells


def corner_cell(
    cell: FactorCell, tile_values: dict, corner_id: str, tile_meta: dict
) -> FactorCell:
    """A corner expressed as a :class:`FactorCell`, so it gets ``rows()`` for free."""
    return FactorCell(
        cell_id=corner_id,
        values={**cell.values, **tile_values},
        meta={**cell.meta, **tile_meta, "parent": cell.cell_id, "is_corner": True},
    )


def kway_corners(
    grid: FactorGrid,
    cfg: KwayLadderConfig,
    k: int,
    cells: Sequence[FactorCell],
    pair_cells: Sequence[FactorCell] = (),
) -> list[FactorCell]:
    """Tile each cell over the factors it leaves free — ⭐ the ESTIMAND's unit.

    The tiling widths are the n-way builder's widths and the tile nesting is
    :data:`TILE_NESTING`, both taken verbatim from
    ``exp_gen_factor_wells.pair_subcorners``, so ``k = 2`` reproduces row 3's 60
    sub-corners exactly (:func:`assert_k2_matches_row3`).
    """
    axes = cfg.axes_for(k)
    opts = coordinate_options(grid, cfg, pair_cells)
    free = _free_units(axes)
    tiles = _product(free, opts) if free else [{}]
    out: list[FactorCell] = []
    for cell in cells:
        for chosen in tiles:
            values, meta = _merge(chosen)
            parts: list[str] = []
            fw = chosen.get("floorwall") or chosen.get("wall_only")
            if fw is not None:
                parts.append(fw["label"])
            letters = "".join(
                chosen[u]["label"] for u in ("scale", "orientation") if u in chosen
            )
            if letters:
                parts.append(letters)
            cid = cell.cell_id if not parts else f"{cell.cell_id}/{'-'.join(parts)}"
            corner = corner_cell(cell, values, cid, {**meta, "tiled_units": list(free)})
            if corner.rows(grid).size == 0:
                continue
            out.append(corner)
    return out


def row3_subcorners(grid: FactorGrid, cfg: KwayLadderConfig) -> list[FactorCell]:
    """Row 3's 60 sub-corners, rebuilt by ``exp_gen_factor_wells.pair_subcorners``'s
    own arithmetic.

    ⛔ Deliberately written out longhand: E2 must compare against **the runner's**
    logic, not against a second call into this module's own tiler.
    """
    vc = cfg.value_cfg
    # ⛔ ROW3_TILE_WIDTHS, never cfg.widths — see that constant's note.
    bw = int(ROW3_TILE_WIDTHS["hue_band"])
    sw = int(ROW3_TILE_WIDTHS["scale_corner"])
    rw = int(ROW3_TILE_WIDTHS["orientation_corner"])
    fh = grid.values[grid.index_of("floor_hue")]
    sc = grid.values[grid.index_of("scale")]
    ro = grid.values[grid.index_of("orientation")]
    floor_windows = [np.arange(i, i + bw) for i in range(0, fh.size - bw + 1, bw)]
    out: list[FactorCell] = []
    for c in build_pair_cells(grid, vc):
        for win in floor_windows:
            for scn in ("high", "low"):
                for rcn in ("high", "low"):
                    vals = dict(c.values)
                    band = tuple(float(x) for x in fh[win])
                    vals["floor_hue"] = band
                    vals["wall_hue"] = band
                    vals["scale"] = tuple(
                        float(x) for x in (sc[-sw:] if scn == "high" else sc[:sw])
                    )
                    vals["orientation"] = tuple(
                        float(x) for x in (ro[-rw:] if rcn == "high" else ro[:rw])
                    )
                    cid = f"{c.cell_id}/f{int(win[0])}-{scn[0]}{rcn[0]}"
                    cell = FactorCell(cell_id=cid, values=vals, meta={"family": "pair"})
                    if cell.rows(grid).size == 0:
                        continue
                    out.append(cell)
    return out


# ---------------------------------------------------------------------------
# 3. ⛔ THE CHECKS THAT CAN KILL THE LADDER — written and run before the ladder
# ---------------------------------------------------------------------------


def assert_corners_disjoint(
    grid: FactorGrid, corners: Sequence[FactorCell], label: str
) -> dict:
    """Row-level proof of pairwise disjointness.

    ⛔ The structural argument (a product of pairwise-disjoint coordinate options) is
    the *reason*; this is the *proof*, and only the proof is reported.
    """
    if not corners:
        raise AssertionError(f"{label}: no corners")
    rows = [c.rows(grid) for c in corners]
    cat = np.concatenate(rows)
    uniq = np.unique(cat)
    if uniq.size != cat.size:
        raise AssertionError(
            f"{label}: the corners OVERLAP — {cat.size - uniq.size} shared row(s); the "
            "paired estimand assumes disjointness and would double-count"
        )
    return {
        "n_corners": len(corners),
        "n_rows": int(cat.size),
        "pairwise_disjoint": True,
        "rows_per_corner": sorted({int(r.size) for r in rows}),
    }


def assert_corners_inside_holdout(
    grid: FactorGrid, corners: Sequence[FactorCell], test_rows: np.ndarray, label: str
) -> int:
    """Every corner row must be in the rung's own hold-out, ⛔ never in the fit pool."""
    cat = np.unique(np.concatenate([c.rows(grid) for c in corners]))
    stray = np.setdiff1d(cat, np.asarray(test_rows, np.int64))
    if stray.size:
        raise AssertionError(
            f"{label}: {stray.size} corner rows are NOT held out "
            f"(e.g. {stray[:8].tolist()}) — the corner is partly in the fit pool"
        )
    return int(cat.size)


def assert_row3_widths_match_pins() -> dict:
    """Re-tie :data:`ROW3_TILE_WIDTHS` to ``exp_gen_factor_wells.PINS``.

    ⛔ Imported lazily and inside the function: :mod:`chlu.data` must not depend on
    :mod:`chlu.experiments` at import time.  If the runner is not importable the check
    is a **DECLARED NOT-RUN with its reason**, ⛔ never a silent pass.
    """
    try:
        from ..experiments.exp_gen_factor_wells import PINS  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import environment only
        return {
            "assertion": "row3_tile_widths_equal_PINS",
            "status": "NOT_RUN",
            "reason": f"exp_gen_factor_wells not importable: {exc!r}",
            "literals": dict(ROW3_TILE_WIDTHS),
        }
    pins = {
        "hue_band": int(PINS["pair_tile_hue_band"]),
        "scale_corner": int(PINS["pair_tile_scale_corner"]),
        "orientation_corner": int(PINS["pair_tile_orientation_corner"]),
    }
    if pins != dict(ROW3_TILE_WIDTHS):
        raise AssertionError(
            f"ROW3_TILE_WIDTHS {dict(ROW3_TILE_WIDTHS)} != the runner's pins {pins} — "
            "⛔ row 3 has moved and E2's reference side is stale"
        )
    return {
        "assertion": "row3_tile_widths_equal_PINS",
        "status": "PASS",
        "widths": pins,
        "reference": "chlu.experiments.exp_gen_factor_wells.PINS['pair_tile_*']",
    }


def assert_k6_matches_row4(grid: FactorGrid, cfg: KwayLadderConfig) -> dict:
    """⭐ E1: the k=6 rung IS ``build_nway_cells``'s slot list, cell for cell, in order."""
    import dataclasses as _dc

    pair = build_pair_cells(grid, cfg.value_cfg)
    mine = build_kway_cells(grid, cfg, 6, pair_cells=pair)
    vc = _dc.replace(cfg.value_cfg, n_nway_cells=len(mine))
    theirs = build_nway_cells(grid, vc, pair_cells=pair, montero_widths=dict(cfg.widths))
    if len(mine) != len(theirs):
        raise AssertionError(
            f"E1: k=6 built {len(mine)} cells, build_nway_cells built {len(theirs)}"
        )
    for a, b in zip(mine, theirs, strict=True):
        va = {f: tuple(float(x) for x in v) for f, v in a.values.items()}
        vb = {f: tuple(float(x) for x in v) for f, v in b.values.items()}
        if va != vb:
            raise AssertionError(
                f"E1: {a.cell_id} != {b.cell_id}\n  kway: {va}\n  nway: {vb}\n"
                "⛔ the k=6 end does not reproduce row 4 — the ladder is off-instrument"
            )
    return {
        "assertion": "E1_k6_equals_row4_nway_cells",
        "n_cells_compared": len(mine),
        "cell_values_identical": True,
        "order_identical": True,
        "reference": "chlu.data.gen_value_holdout.build_nway_cells",
    }


def assert_k2_matches_row3(grid: FactorGrid, cfg: KwayLadderConfig) -> dict:
    """⭐ E2: the k=2 rung IS row 3's tiled sub-corners — ids AND row sets.

    ⛔ If this fails the builder is wrong, not row 3.
    """
    pair = build_pair_cells(grid, cfg.value_cfg)
    cells = build_kway_cells(grid, cfg, 2, pair_cells=pair)
    mine = kway_corners(grid, cfg, 2, cells, pair_cells=pair)
    theirs = row3_subcorners(grid, cfg)
    if len(mine) != len(theirs):
        raise AssertionError(
            f"E2: k=2 built {len(mine)} corners, row 3 has {len(theirs)}"
        )
    for a, b in zip(mine, theirs, strict=True):
        if a.cell_id != b.cell_id:
            raise AssertionError(f"E2: corner id {a.cell_id!r} != row 3's {b.cell_id!r}")
        ra, rb = a.rows(grid), b.rows(grid)
        if not np.array_equal(ra, rb):
            raise AssertionError(
                f"E2: corner {a.cell_id} covers {ra.size} rows, row 3's covers "
                f"{rb.size} — ⛔ the builder is wrong, not row 3"
            )
    return {
        "assertion": "E2_k2_equals_row3_pair_subcorners",
        "n_corners_compared": len(mine),
        "corner_ids_identical": True,
        "corner_row_sets_identical": True,
        "reference": "chlu.experiments.exp_gen_factor_wells.pair_subcorners",
    }


def ladder_capacity_report(
    grid: FactorGrid,
    cfg: Optional[KwayLadderConfig] = None,
    ks: Sequence[int] = (2, 3, 4, 5, 6),
) -> dict:
    """⛔ THE KILL CHECK.

    Pure combinatorics over the coordinate options — it materialises one probe cell per
    rung for the row budget and nothing else, so it runs **before** the ladder and can
    stop it.  A rung whose ``max_corners < CORNER_BAR`` is reported ``NOT_RUN`` with the
    count it reached; ⛔ it is never padded and never reported as zero.
    """
    cfg = cfg or KwayLadderConfig()
    pair = build_pair_cells(grid, cfg.value_cfg)
    opts = coordinate_options(grid, cfg, pair)
    n_grid = int(grid.n_rows)
    rungs: dict[str, dict] = {}
    for k in ks:
        axes = cfg.axes_for(k)
        free = _free_units(axes)
        if k == 2 and cfg.variant == "primary":
            n_cells_cap = int(cfg.value_cfg.n_pair_cells)
            option_counts = {"pair_cells (build_pair_cells)": n_cells_cap}
            cell_rows = int(pair[0].rows(grid).size)
        else:
            nesting = _cell_nesting(axes)
            option_counts = {u: len(opts[u]) for u in nesting}
            n_cells_cap = int(np.prod([len(opts[u]) for u in nesting]))
            probe_cfg = KwayLadderConfig(
                value_cfg=cfg.value_cfg, widths=cfg.widths, n_cells=1,
                variant=cfg.variant,
            )
            cell_rows = int(
                build_kway_cells(grid, probe_cfg, k, pair_cells=pair)[0].rows(grid).size
            )
        tiles = int(np.prod([len(opts[u]) for u in free])) if free else 1
        max_corners = n_cells_cap * tiles
        min_cells = int(-(-CORNER_BAR // tiles))
        pinned_factors = sorted({f for u in axes for f in COORD_FACTORS[u]})
        rec = {
            "k": int(k),
            "pinned_units": list(axes),
            "pinned_factors": pinned_factors,
            "free_factors": sorted(set(grid.names) - set(pinned_factors)),
            "tiling_units": list(free),
            "cell_option_counts": option_counts,
            "n_cells_capacity": n_cells_cap,
            "corners_per_cell": tiles,
            "max_corners": max_corners,
            "bar": CORNER_BAR,
            "clears_bar": bool(max_corners >= CORNER_BAR),
            "status": "PASS" if max_corners >= CORNER_BAR else "NOT_RUN",
            "rows_per_cell": cell_rows,
            "held_out_rows_full_sweep": cell_rows * n_cells_cap,
            "held_out_fraction_full_sweep": round(cell_rows * n_cells_cap / n_grid, 6),
            "min_cells_for_bar": min_cells,
            "min_cells_corners": min_cells * tiles,
            "min_cells_held_out_rows": cell_rows * min_cells,
            "min_cells_held_out_fraction": round(cell_rows * min_cells / n_grid, 6),
        }
        if max_corners < CORNER_BAR:
            rec["not_run_reason"] = (
                f"only {max_corners} pairwise-disjoint corners exist for k={k} "
                f"({n_cells_cap} boxes x {tiles} tiles); the bar is {CORNER_BAR}, and "
                "⛔ padding with overlapping corners would break the paired estimand"
            )
        rungs[str(k)] = rec
    return {
        "variant": cfg.variant,
        "widths": dict(cfg.widths),
        "grid_rows": n_grid,
        "grid_cardinalities": dict(zip(grid.names, grid.cardinalities, strict=True)),
        "coordinate_option_counts": {u: len(v) for u, v in opts.items()},
        "bar": CORNER_BAR,
        "rungs": rungs,
    }


#: ``exp_gen_factor_wells.PINS`` — the fit pool the hub will draw on every rung.
#: ⛔ Quoted here so a rung whose train index is too small to serve it is caught by
#: the kill check and not by the hub three hours into a run.
PIN_N_POOL = 131_072
PIN_N_STORE_TRAIN = 32_768


def row_matching_feasibility(cap: dict) -> dict:
    """⛔ Can the ladder clear 49 corners per rung AND hold out the same row count?

    The two demands pull against each other: a lower-k box is bigger, so the smallest
    budget that clears the bar removes far more rows, while the highest budget a
    higher-k rung can reach is capped by its own capacity.  A common held-out row
    level exists only if ``max(min_rows_to_clear_bar) <= min(max_rows_at_capacity)``.
    This function reports the two sides and the verdict — ⛔ it does not pick a fix.
    """
    rungs = {k: r for k, r in cap["rungs"].items() if r["status"] == "PASS"}
    floor = {k: r["min_cells_held_out_rows"] for k, r in rungs.items()}
    ceil_ = {k: r["held_out_rows_full_sweep"] for k, r in rungs.items()}
    lo = max(floor.values()) if floor else 0
    hi = min(ceil_.values()) if ceil_ else 0
    return {
        "question": (
            "is there a common held-out ROW budget at which every rung still has >= "
            f"{CORNER_BAR} pairwise-disjoint corners?"
        ),
        "min_rows_to_clear_bar_per_rung": floor,
        "max_rows_at_full_capacity_per_rung": ceil_,
        "required_floor": lo,
        "available_ceiling": hi,
        "feasible": bool(lo <= hi),
        "verdict": (
            f"NO — the rung that needs the most rows to reach {CORNER_BAR} corners "
            f"needs {lo:,}, and the tightest rung can hold out at most {hi:,}. "
            "⛔ The ladder CANNOT be row-matched across k while every rung clears the "
            "bar; the confound must be handled by a control, not by a budget."
            if lo > hi
            else f"YES — any common budget in [{lo:,}, {hi:,}] rows works."
        ),
    }


def fit_pool_feasibility(train_rows: int) -> dict:
    """Does this rung's train index still serve the pinned fit pool?

    ⭐ The hub's ``fit_pool`` draws ``min(PIN_N_POOL, train.size)`` rows, so as long as
    every rung clears ``PIN_N_POOL`` the store is fit on the **same number of rows at
    every rung** and the depth/hold-out-fraction confound does not act through pool
    *size*.  It still acts through pool *coverage*.  ⛔ A rung that falls below it
    silently changes the fit pool and is not comparable.
    """
    return {
        "train_rows": int(train_rows),
        "n_pool_pin": PIN_N_POOL,
        "n_store_train_pin": PIN_N_STORE_TRAIN,
        "pool_drawn": int(min(PIN_N_POOL, train_rows)),
        "headroom_rows": int(train_rows - PIN_N_POOL),
        "serves_pinned_pool": bool(train_rows >= PIN_N_POOL),
    }


# ---------------------------------------------------------------------------
# 4. the split
# ---------------------------------------------------------------------------


def build_kway_condition_cells(
    grid: FactorGrid, cfg: KwayLadderConfig, k: int
) -> tuple[list[FactorCell], list[FactorCell], list[FactorCell]]:
    """``(value cells, pair cells, kway cells)`` — the rung's full condition.

    The value and pair cells are carried unchanged, so a rung's split is the
    value-hold-out condition with the ``nway-*`` family swapped for ``kway{k}-*``:
    rows 1 / 2 / 5 / 7 stay measurable on the same rung, against the same scope
    statement.
    """
    value = build_value_cells(grid, cfg.value_cfg)
    pair = build_pair_cells(grid, cfg.value_cfg)
    kway = [] if k == 2 else build_kway_cells(grid, cfg, k, pair_cells=pair)
    return value, pair, kway


def build_kway_split(
    grid: FactorGrid, cfg: KwayLadderConfig, k: int
) -> tuple[GenSplit, list[FactorCell], list[FactorCell]]:
    """Build the rung's split, run gen_splits' own load-time proofs, tile the corners.

    Returns ``(split, all_cells, corners)``.
    """
    value, pair, kway = build_kway_condition_cells(grid, cfg, k)
    cells = [*value, *pair, *kway]
    assert_cells_disjoint(grid, cells)
    split = build_cell_split(
        grid,
        cells,
        dataset=cfg.value_cfg.dataset,
        condition=KWAY_CONDITION,
        variant=f"k{k}_{cfg.variant}",
    )
    assert_non_empty_test(split)
    assert_holdout_disjoint(split)
    assert_index_partitions_grid(split)
    corners = kway_corners(grid, cfg, k, kway or pair, pair_cells=pair)
    return split, cells, corners


# ---------------------------------------------------------------------------
# 5. LADDER-READY.json
# ---------------------------------------------------------------------------


def emit_ladder_ready(
    grid: FactorGrid,
    out_dir: str | Path,
    cfg: Optional[KwayLadderConfig] = None,
    ks: Sequence[int] = (2, 3, 4, 5, 6),
    alt_ks: Sequence[int] = (5,),
    filename: str = "LADDER-READY.json",
    log=print,
) -> dict:
    """⭐ Kill check first, then the ends, then every rung; write ``LADDER-READY.json``."""
    cfg = cfg or KwayLadderConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("KILL CHECK first (ladder_capacity_report) - before a single cell is built")
    cap = ladder_capacity_report(grid, cfg, ks)
    for kk, r in cap["rungs"].items():
        log(
            f"  k={kk:>2}  boxes={r['n_cells_capacity']:>4} x tiles={r['corners_per_cell']:>3}"
            f"  = max_corners={r['max_corners']:>5}   bar={CORNER_BAR}   {r['status']}"
        )

    log("END REPRODUCTION")
    pinw = assert_row3_widths_match_pins()
    log(f"  row-3 tile widths vs the runner's PINS: {pinw['status']}")
    e1 = assert_k6_matches_row4(grid, cfg)
    log(f"  E1 ok: {e1['n_cells_compared']} k=6 cells == build_nway_cells, in order")
    e2 = assert_k2_matches_row3(grid, cfg)
    log(f"  E2 ok: {e2['n_corners_compared']} k=2 corners == row 3's pair_subcorners")

    families: dict[str, dict] = {}
    for k in ks:
        r = cap["rungs"][str(k)]
        if r["status"] == "NOT_RUN":
            families[f"k{k}"] = {
                "family": f"kway{k}-*",
                "k": k,
                "status": "NOT_RUN",
                "corners_reached": r["max_corners"],
                "bar": CORNER_BAR,
                "reason": r["not_run_reason"],
            }
            log(f"  k={k}: NOT_RUN ({r['max_corners']} corners < {CORNER_BAR})")
            continue
        split, cells, corners = build_kway_split(grid, cfg, k)
        dis = assert_corners_disjoint(grid, corners, f"k{k}")
        n_in = assert_corners_inside_holdout(grid, corners, split.test_rows, f"k{k}")
        n_cells = len([c for c in cells if c.meta.get("family", "").startswith("kway")])
        if k == 2:
            n_cells = int(cfg.value_cfg.n_pair_cells)
        families[f"k{k}"] = {
            "family": "pair-*" if k == 2 else ("nway-*" if k == 6 else f"kway{k}-*"),
            "display_name": {2: "row 3 (existing end)", 6: "row 4 (existing end)"}.get(
                k, f"k={k} (NEW)"
            ),
            "k": k,
            "status": "PASS",
            "role": {
                2: "existing end — reproduced, not rebuilt (E2)",
                6: "existing end — reproduced, not rebuilt (E1)",
            }.get(k, "NEW — built by w12/kway-ladder-harness"),
            "pinned_factors": r["pinned_factors"],
            "free_factors": r["free_factors"],
            "tiling_units": r["tiling_units"],
            "widths": dict(cfg.widths),
            "n_cells": n_cells,
            "n_corners": dis["n_corners"],
            "bar": CORNER_BAR,
            "clears_bar": bool(dis["n_corners"] >= CORNER_BAR),
            "disjointness_check": {
                "test": "np.unique(concat(corner_rows)).size == concat(corner_rows).size",
                "n_corner_rows": dis["n_rows"],
                "pairwise_disjoint": dis["pairwise_disjoint"],
                "rows_per_corner": dis["rows_per_corner"],
            },
            "all_corner_rows_held_out": True,
            "n_corner_rows_in_holdout": n_in,
            "held_out_rows_total": int(split.test_rows.size),
            "held_out_fraction_total": round(
                float(split.test_rows.size) / grid.n_rows, 6
            ),
            "fit_pool_rows": int(split.train_rows.size),
            "fit_pool_feasibility": fit_pool_feasibility(int(split.train_rows.size)),
            "corner_ids_head": [c.cell_id for c in corners[:4]],
            "corner_ids_tail": [c.cell_id for c in corners[-2:]],
        }
        log(
            f"  k={k}: {n_cells} cells -> {dis['n_corners']} corners, disjoint, "
            f"{dis['rows_per_corner']} rows/corner, hold-out {split.test_rows.size:,} rows"
        )

    alt: dict[str, dict] = {}
    if alt_ks:
        alt_cfg = KwayLadderConfig(
            value_cfg=cfg.value_cfg, widths=cfg.widths, n_cells=cfg.n_cells,
            variant="nested_alt",
        )
        alt_cap = ladder_capacity_report(grid, alt_cfg, alt_ks)
        log("NESTED ALTERNATIVE (wall_hue untied - NOT rows 3/4 geometry)")
        for k in alt_ks:
            r = alt_cap["rungs"][str(k)]
            if r["status"] == "NOT_RUN":
                alt[f"k{k}"] = {"status": "NOT_RUN", "corners_reached": r["max_corners"],
                                "reason": r["not_run_reason"]}
                continue
            split, cells, corners = build_kway_split(grid, alt_cfg, k)
            dis = assert_corners_disjoint(grid, corners, f"alt-k{k}")
            n_in = assert_corners_inside_holdout(
                grid, corners, split.test_rows, f"alt-k{k}"
            )
            alt[f"k{k}"] = {
                "family": f"kway{k}-*",
                "variant": "nested_alt",
                "k": k,
                "status": "PASS",
                "pinned_factors": r["pinned_factors"],
                "free_factors": r["free_factors"],
                "n_cells": len(
                    [c for c in cells if c.meta.get("family", "").startswith("kway")]
                ),
                "n_corners": dis["n_corners"],
                "clears_bar": bool(dis["n_corners"] >= CORNER_BAR),
                "disjointness_check": {
                    "pairwise_disjoint": dis["pairwise_disjoint"],
                    "n_corner_rows": dis["n_rows"],
                    "rows_per_corner": dis["rows_per_corner"],
                },
                "all_corner_rows_held_out": True,
                "n_corner_rows_in_holdout": n_in,
                "held_out_rows_total": int(split.test_rows.size),
                "held_out_fraction_total": round(
                    float(split.test_rows.size) / grid.n_rows, 6
                ),
                "geometry_matches_rows_3_4": False,
                "geometry_note": (
                    "floor band != wall band in 4 of every 5 tiles; rows 3 and 4 always "
                    "tie them — ⛔ use only if the hub declares the deviation"
                ),
            }
            log(
                f"  alt k={k}: {alt[f'k{k}']['n_cells']} cells -> {dis['n_corners']} "
                f"corners, disjoint, hold-out {split.test_rows.size:,} rows"
            )
        alt["capacity"] = alt_cap

    payload = {
        "condition": KWAY_CONDITION,
        "built_by": "w12/kway-ladder-harness",
        "scored_runs_spent": 0,
        "dataset": cfg.value_cfg.dataset,
        "grid": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "n_rows": int(grid.n_rows),
            "grid_dataset_id": grid.dataset,
        },
        "scope": VALUE_HOLDOUT_SCOPE,
        "bar": {
            "corners_per_k": CORNER_BAR,
            "source": "w3-hub-common section 3 — the multi-corner paired estimand",
            "declared_in": "PREREG.md (2026-09-21), before any count existed",
            "unit": "corner",
        },
        "estimand": {
            "function": "chlu.experiments.exp_gen_factor_wells.paired_estimand",
            "unit": "corner",
            "statistic": "mean paired per-corner difference (arm - control)",
            "metric_rows_3_4_and_ladder": "distinct_valid_novel_tuples",
            "margin": 1.0,
            "margin_name": "DECORATION_MARGIN_NOVELTY",
            "accuracy_margin": 0.02,
            "accuracy_margin_name": "DECORATION_MARGIN_ACCURACY",
            "bootstrap": {"n_resamples": 10000, "rng_seed": 20260919},
            "note": "⛔ the ladder introduces no new metric and no new margin",
        },
        "geometry": {
            "widths": dict(cfg.widths),
            "wall_tied_to_floor": True,
            "note": (
                "every corner is a Montero box: a floor band == wall band of 2, an "
                "object_hue band, a scale corner of 3, one shape, an orientation "
                "corner of 3 — the geometry of rows 3 and 4"
            ),
            "object_hue_band_width_row3": 1,
            "object_hue_band_width_row4_and_ladder": int(cfg.widths["hue_band"]),
            "object_hue_width_note": (
                "⚠ the two EXISTING ends already disagree here: row 3 pins ONE object "
                "hue (36-row corners), row 4 a band of 2 (72-row corners). The ladder "
                "follows row 4, the end it is anchored to. ⛔ Not introduced by w12."
            ),
        },
        "ineligibility": {
            "value_holdout_ineligible": ["scale", "orientation"],
            "kway_box_eligible": ["scale", "orientation"],
            "note": (
                "scale and orientation are physical defaults (gen_value_holdout section "
                "2) and stay INELIGIBLE as VALUE hold-outs; they are eligible INSIDE a "
                "k-way box exactly as in row 4 — ⛔ not widened"
            ),
        },
        "end_reproduction": {
            "E1_k6_vs_row4": e1,
            "E2_k2_vs_row3": e2,
            "E2_reference_widths_vs_PINS": pinw,
        },
        "kill_check": cap,
        "row_matching_feasibility": row_matching_feasibility(cap),
        "families": families,
        "nested_alternative": alt,
        "nesting_caveat": (
            "floor_hue and wall_hue are TIED in rows 3 and 4, so the pinned-unit sets "
            "cannot be strictly nested through every k: A2 < A3 < A4 and A3 < A5 < A6, "
            "but A4 is not a subset of A5. LADDER_AXES_NESTED_ALT unties wall_hue at "
            "k=5 and IS nested, at the cost of corners whose floor band != wall band."
        ),
        "confound_declared": (
            "a lower-k box pins fewer factors, so it removes more rows from the fit "
            "pool; the ladder confounds composition depth with held-out fraction. The "
            "per-rung held_out_fraction_total is reported for every rung and the "
            "min_cells_* fields give the smallest budget that still clears the bar."
        ),
        "config": cfg.to_dict(),
    }
    path, sha = write_json_with_checksum(payload, out_dir / filename)
    log(f"wrote {path}  sha256={sha[:16]}...")
    payload["_path"] = str(path)
    payload["_sha256"] = sha
    return payload
