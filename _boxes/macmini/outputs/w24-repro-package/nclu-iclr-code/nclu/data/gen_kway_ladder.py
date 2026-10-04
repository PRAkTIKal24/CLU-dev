"""gen kway ladder."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, Sequence
import numpy as np
from nclu.data.gen_splits import (
    FactorCell,
    FactorGrid,
    GenSplit,
    assert_cells_disjoint,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_non_empty_test,
    build_cell_split,
)
from nclu.data.gen_value_holdout import (
    ValueHoldoutConfig,
    _consecutive_runs,
    _disjoint_windows,
    build_pair_cells,
    build_value_cells,
)

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
    5: ("object_hue", "shape", "scale", "orientation", "floor_only")
}


@dataclass(frozen=True)
class KwayLadderConfig:
    value_cfg: ValueHoldoutConfig = field(default_factory=ValueHoldoutConfig)
    widths: dict = field(default_factory=lambda: dict(KWAY_WIDTHS))
    n_cells: Optional[int] = None
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
    grid: FactorGrid, cfg: KwayLadderConfig, pair_cells: Sequence[FactorCell] = ()
) -> dict[str, list[dict]]:
    vc = cfg.value_cfg
    band_w = int(cfg.widths.get("hue_band", 2))
    scale_w = int(cfg.widths.get("scale_corner", 3))
    rot_w = int(cfg.widths.get("orientation_corner", 3))
    hue_f, shape_f = (vc.held_out_hue_factor, vc.held_out_shape_factor)
    hue_vals = grid.values[grid.index_of(hue_f)]
    floor_vals = grid.values[grid.index_of("floor_hue")]
    wall_vals = grid.values[grid.index_of("wall_hue")]
    scale_vals = grid.values[grid.index_of("scale")]
    rot_vals = grid.values[grid.index_of("orientation")]
    reserved = {float(vc.held_out_hue_value)}
    for c in pair_cells:
        reserved.update((float(v) for v in c.values.get(hue_f, ())))
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
            f"no {band_w}-wide band survives the reserved {hue_f} values {sorted(reserved)} — the ladder has no coordinate basis"
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
                "floor_hue": tuple((float(x) for x in floor_vals[w])),
                "wall_hue": tuple((float(x) for x in floor_vals[w])),
            },
            "meta": {"floor_band_start_index": int(w[0])},
        }
        for w in floor_windows
    ]
    opts["floor_only"] = [
        {
            "label": f"f{int(w[0])}",
            "values": {"floor_hue": tuple((float(x) for x in floor_vals[w]))},
            "meta": {"floor_band_start_index": int(w[0])},
        }
        for w in floor_windows
    ]
    opts["wall_only"] = [
        {
            "label": f"w{int(w[0])}",
            "values": {"wall_hue": tuple((float(x) for x in wall_vals[w]))},
            "meta": {"wall_band_start_index": int(w[0])},
        }
        for w in wall_windows
    ]
    opts["object_hue"] = [
        {
            "label": f"o{int(w[0])}",
            "values": {hue_f: tuple((float(x) for x in hue_vals[w]))},
            "meta": {"object_band_start_index": int(w[0])},
        }
        for w in obj_windows
    ]
    opts["scale"] = [
        {
            "label": "h",
            "values": {"scale": tuple((float(x) for x in scale_vals[-scale_w:]))},
            "meta": {"scale_corner": "high"},
        },
        {
            "label": "l",
            "values": {"scale": tuple((float(x) for x in scale_vals[:scale_w]))},
            "meta": {"scale_corner": "low"},
        },
    ]
    opts["orientation"] = [
        {
            "label": "h",
            "values": {"orientation": tuple((float(x) for x in rot_vals[-rot_w:]))},
            "meta": {"orientation_corner": "high"},
        },
        {
            "label": "l",
            "values": {"orientation": tuple((float(x) for x in rot_vals[:rot_w]))},
            "meta": {"orientation_corner": "low"},
        },
    ]
    opts["shape"] = [
        {"label": f"s{i}", "values": {shape_f: (s,)}, "meta": {"shape_value": s}}
        for i, s in enumerate(shapes)
    ]
    return opts


def _cell_nesting(axes: Sequence[str]) -> list[str]:
    nesting = [u for u in CELL_NESTING if u in axes]
    if "floor_only" in axes:
        nesting = ["floor_only", *nesting]
    return nesting


def _free_units(axes: Sequence[str]) -> tuple[str, ...]:
    pinned = set(axes)
    if "floor_only" in pinned:
        return ("wall_only", *(u for u in ("scale", "orientation") if u not in pinned))
    return tuple((u for u in TILE_NESTING if u not in pinned))


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
    return (values, meta)


def build_kway_cells(
    grid: FactorGrid, cfg: KwayLadderConfig, k: int, pair_cells: Sequence[FactorCell] = ()
) -> list[FactorCell]:
    if k == 2 and cfg.variant == "primary":
        return list(build_pair_cells(grid, cfg.value_cfg))
    axes = cfg.axes_for(k)
    opts = coordinate_options(grid, cfg, pair_cells)
    nesting = _cell_nesting(axes)
    combos = _product(nesting, opts)
    n = len(combos) if cfg.n_cells is None else int(cfg.n_cells)
    if n > len(combos):
        raise ValueError(
            f"k={k}: n_cells={n} exceeds the {len(combos)} structurally-disjoint boxes available over the pinned units {tuple(axes)} ({' x '.join((f'{u}:{len(opts[u])}' for u in nesting))})"
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
                    "note": f"{k} of 6 factors jointly pinned to a Montero-shaped box, the other {6 - k} unconstrained; the corner geometry is rows 3/4's",
                },
            )
        )
    return cells


def corner_cell(cell: FactorCell, tile_values: dict, corner_id: str, tile_meta: dict) -> FactorCell:
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
            letters = "".join((chosen[u]["label"] for u in ("scale", "orientation") if u in chosen))
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
    return (value, pair, kway)


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
    return (split, cells, corners)
