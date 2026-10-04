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
from typing import Optional, Sequence
import numpy as np
from .gen_splits import FactorCell, FactorGrid, GenSplit, assert_cells_disjoint, assert_holdout_disjoint, assert_index_partitions_grid, assert_non_empty_test, build_cell_split
from .gen_value_holdout import ValueHoldoutConfig, _consecutive_runs, _disjoint_windows, build_pair_cells, build_value_cells
KWAY_CONDITION = "kway_ladder"
KWAY_WIDTHS = {"hue_band": 2, "scale_corner": 3, "orientation_corner": 3}
CELL_NESTING = ("floorwall", "orientation", "scale", "object_hue", "shape")
TILE_NESTING = ("floorwall", "scale", "orientation")
LADDER_AXES: dict[int, tuple[str, ...]] = {
    2: ("object_hue", "shape"),
    3: ("object_hue", "shape", "scale"),
    4: ("object_hue", "shape", "scale", "orientation"),
    5: ("object_hue", "shape", "scale", "floorwall"),
    6: ("object_hue", "shape", "scale", "orientation", "floorwall"),
}
LADDER_AXES_NESTED_ALT: dict[int, tuple[str, ...]] = {
    5: ("object_hue", "shape", "scale", "orientation", "floor_only"),
}
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


