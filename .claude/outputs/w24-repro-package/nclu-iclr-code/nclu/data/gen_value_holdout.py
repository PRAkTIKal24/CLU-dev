"""gen value holdout."""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np
from nclu.data.gen_splits import (
    TERM_KEYS,
    FactorCell,
    FactorGrid,
    GenSplit,
    HoldoutViolation,
    assert_cells_disjoint,
    assert_holdout_disjoint,
    assert_index_partitions_grid,
    assert_no_model_detector_at_chance,
    assert_non_empty_test,
    audit_query_sets,
    auroc,
    build_a7_redundant,
    build_c_invalid,
    build_cell_split,
    save_index,
    select_a7_rows,
    validate_a7_redundant,
    validate_c_invalid,
    write_json_with_checksum,
    write_query_set_index,
)

__all__ = [
    "HELD_OUT_HUE_RATIONALE",
    "HELD_OUT_SHAPE_RATIONALE",
    "ROW1_SEEN_RATIONALE",
    "ROW_EVIDENCE_RANKING",
    "VALUE_HOLDOUT_CONDITION",
    "VALUE_HOLDOUT_SCOPE",
    "VALUE_HOLDOUT_VARIANT",
    "ValueHoldoutConfig",
    "assert_value_absent_from_train",
    "build_nway_cells",
    "build_pair_cells",
    "assert_row1_matches_row5",
    "build_row1_seen",
    "build_row2_unseen",
    "build_row5_held_out",
    "build_value_cells",
    "build_value_holdout_cells",
    "build_value_holdout_split",
    "emit_value_holdout",
    "grid_reach_auroc",
    "grid_reach_features",
    "main",
    "row1_positive_control",
    "validate_row1",
    "validate_row2",
    "validate_row5",
    "value_holdout_metadata",
]
VALUE_HOLDOUT_CONDITION = "value_holdout"
VALUE_HOLDOUT_VARIANT = "value_cells_plus_combination_cells"
ROW_EVIDENCE_RANKING = {
    "r7_deletion": {
        "evidential": True,
        "rank": 1,
        "why": "⭐ THE strongest instrument. Deleting a value from the store does not move the factor grid, so the no-model rule's delta_reject is exactly 0.0. A changed refusal after a deletion, with NO retraining, is not producible by any syntactic rule.",
    },
    "r5_comp_refusal": {
        "evidential": True,
        "rank": 2,
        "why": "⭐ Evidential UNDER THE AMENDED PAIRING ONLY: the accept-leg is row 1, a STORED on-grid request. Both halves are well-formed on-grid requests and differ only in whether the store holds the value. ⛔ With row 2 as the accept-leg it was free (grid look-up, 1.0000, zero bytes).",
    },
    "r6_invalid_refusal": {
        "evidential": False,
        "rank": 3,
        "why": "⛔ SANITY CHECK ONLY, and it can never be anything else: out-of-range is decidable from the factor grid alone. Kept as the degenerate-arm guard; ⛔ a row-6 number is NOT evidence that the store computes a barrier.",
    },
}
VALUE_HOLDOUT_SCOPE = "SCOPE OF THE VALUE HOLD-OUT (⛔ quote no row without this). The hold-out is from the STORE'S FIT POOL, not from the encoder, the decoder or the judge. Those three are frozen, hash-pinned and read-only (protocol §A.4) and were fitted on a train split containing EVERY hue and EVERY shape; re-fitting them on a value hold-out would void every round-2 number and is out of scope. Therefore 'never trained on' means here: THE STORE NEVER HELD A ROW CARRYING THAT VALUE. The decoder can still render the value and the judge can still recognise it — which is what makes rows 2/5/6/7 scorable at all. ⛔ Read end-to-end, 'never seen' is false and the claim dies with it."
HELD_OUT_HUE_RATIONALE = {
    "factor": "object_hue",
    "value": 0.5,
    "grid_index": 5,
    "why": "The interior mid-point of the 10-value hue ring. (a) It is the object's own photometric property — the thing the request names and the judge scores — unlike floor_hue/wall_hue, which are background. (b) Hue is CIRCULAR, so no hue value is a physical default the model degrades towards; the §3.1 hazard that rules out scale and orientation does not apply to it. (c) It sits four grid steps from Montero's retained band (object_hue >= 0.8), so the value hole does not sit inside the region the headline and wide COMBINATION conditions already spend. (d) It is interior, so BOTH of its grid neighbours are retained and the hole is two-sided — a hole at 0.0 or 0.9 would sit on the realised range boundary, where build_c_invalid's out_of_range records are also placed, and rows 5 and 6 would become adjacent in value space.",
    "rejected": {
        "0.8 / 0.9": "Montero's held-out hue band. A value hold-out placed there is entangled with the headline combination condition and its 72 rows.",
        "0.0 / 0.9 (as range endpoints)": "one-sided in the realised range: the nearest off-grid values are exactly where out_of_range (row 6) lives.",
        "scale / orientation (any value)": "⛔ ineligible per task §3.1: round 2 measured both as PHYSICAL DEFAULTS the model degrades towards, so a held-out default is unremovable for reasons that have nothing to do with the store.",
    },
}
HELD_OUT_SHAPE_RATIONALE = {
    "factor": "shape",
    "value": 0.0,
    "grid_index": 0,
    "label": "cube",
    "why": "The cube is the shape most distant from the retained set under the only degradation mode we have measured. Shape is CATEGORICAL — it cannot be interpolated — so 'held out entirely' is only weakened by a retained NEAR-NEIGHBOUR, and the cube has none among {cylinder, sphere, oblong}: it is the one shape with flat faces and hard edges.",
    "rejected": {
        "2.0 (sphere)": "⛔ the shape hazard of §3.1, exactly: a degraded render is a rounded blob, so the SPHERE is the shape the model degrades TOWARDS. A held-out sphere would be unremovable for rendering reasons, not store reasons — the same argument that rules out the median scale and the null pose.",
        "1.0 (cylinder)": "Montero's held-out shape, and the wide condition's cell-0 shape. Already entangled with both combination conditions.",
        "3.0 (oblong/capsule)": "visually intermediate between the cylinder and the sphere, both retained. A retained near-neighbour weakens 'held out entirely' — refusal could be declined on interpolation grounds, and the row would stop being a refusal row.",
    },
}
ROW1_SEEN_RATIONALE = {
    "object_hue": {
        "value": 0.4,
        "grid_index": 4,
        "mirrors": {"value": 0.5, "grid_index": 5},
        "boundary_distance": 4,
        "mirrored_boundary_distance": 4,
        "why": "the retained hue whose grid index sits at the SAME distance from the realised range boundary as the held-out index 5, so an unsigned boundary/interior rule is blind to the difference",
        "rejected": {
            "a uniform draw over all 9 retained hues": "⛔ row 5 holds its hue CONSTANT; a control whose hue varies is separable by a transductive constant-detector with no model in it",
            "0.0 / 0.9": "boundary distance 0 against the held-out value's 4 — an unsigned boundary rule would separate the legs for free",
        },
    },
    "shape": {
        "value": 3.0,
        "grid_index": 3,
        "label": "oblong/capsule",
        "mirrors": {"value": 0.0, "grid_index": 0, "label": "cube"},
        "boundary_distance": 0,
        "mirrored_boundary_distance": 0,
        "why": "the only retained shape at the SAME boundary distance (0) as the held-out cube, i.e. the other end of the shape axis. ⚠ HELD_OUT_SHAPE_RATIONALE rejects 3.0 as a HOLD-OUT (a retained near-neighbour would weaken refusal); that argument does not apply to an ACCEPT-leg, whose job is to be a value the store demonstrably holds.",
        "rejected": {
            "1.0 (cylinder) / 2.0 (sphere)": "boundary distance 1 against the held-out cube's 0 — an unsigned boundary rule would separate the legs for free"
        },
    },
}
PAIR_AXES = ("object_hue", "shape")


@dataclass(frozen=True)
class ValueHoldoutConfig:
    dataset: str = "shapes3d"
    held_out_hue_factor: str = "object_hue"
    held_out_hue_value: float = 0.5
    held_out_shape_factor: str = "shape"
    held_out_shape_value: float = 0.0
    n_pair_cells: int = 3
    n_nway_cells: int = 49
    row1_seen_hue_value: float = 0.4
    row1_seen_shape_value: float = 3.0
    row1_n_per_kind: int = 256
    row2_n_requests: int = 512
    row5_n_per_kind: int = 256
    c_invalid_n_per_kind: int = 256
    a7_n_requests: int = 512
    pair_seed: int = 0
    nway_seed: int = 0
    row1_seed: int = 0
    row2_seed: int = 0
    row5_seed: int = 0
    c_invalid_seed: int = 0
    a7_seed: int = 0
    detector_tolerance: float = 0.05
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Optional[dict] = None) -> "ValueHoldoutConfig":
        mapping = dict(mapping or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"ValueHoldoutConfig: unknown key(s) {unknown}; known keys are {sorted(known)}"
            )
        return cls(**mapping)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _grid_index_of_value(grid: FactorGrid, factor: str, value: float) -> int:
    vals = grid.values[grid.index_of(factor)]
    hit = np.flatnonzero(vals == float(value))
    if hit.size != 1:
        raise ValueError(
            f"value {value!r} is not a grid value of factor {factor!r} (realised values: {vals.tolist()})"
        )
    return int(hit[0])


def _retained_values(grid: FactorGrid, factor: str, held_out: float) -> np.ndarray:
    vals = grid.values[grid.index_of(factor)]
    return vals[vals != float(held_out)]


def build_value_cells(grid: FactorGrid, cfg: ValueHoldoutConfig) -> list[FactorCell]:
    _grid_index_of_value(grid, cfg.held_out_hue_factor, cfg.held_out_hue_value)
    _grid_index_of_value(grid, cfg.held_out_shape_factor, cfg.held_out_shape_value)
    retained_hues = _retained_values(grid, cfg.held_out_hue_factor, cfg.held_out_hue_value)
    hue_cell = FactorCell(
        cell_id=f"value-{cfg.held_out_hue_factor}",
        values={cfg.held_out_hue_factor: (float(cfg.held_out_hue_value),)},
        meta={
            "family": "value",
            "held_out_factor": cfg.held_out_hue_factor,
            "held_out_value": float(cfg.held_out_hue_value),
            "rationale": HELD_OUT_HUE_RATIONALE,
            "note": "every other factor is UNCONSTRAINED — the value is held out across every shape, scale and pose, which is what makes it a VALUE hold-out and not a combination hold-out",
        },
    )
    shape_cell = FactorCell(
        cell_id=f"value-{cfg.held_out_shape_factor}",
        values={
            cfg.held_out_shape_factor: (float(cfg.held_out_shape_value),),
            cfg.held_out_hue_factor: tuple((float(x) for x in retained_hues)),
        },
        meta={
            "family": "value",
            "held_out_factor": cfg.held_out_shape_factor,
            "held_out_value": float(cfg.held_out_shape_value),
            "rationale": HELD_OUT_SHAPE_RATIONALE,
            "note": f"the {cfg.held_out_hue_factor} constraint carves out ONLY the rows the {cfg.held_out_hue_factor} cell already owns, so the two cells are disjoint; their union is every row carrying either held-out value",
        },
    )
    return [hue_cell, shape_cell]


def build_pair_cells(grid: FactorGrid, cfg: ValueHoldoutConfig) -> list[FactorCell]:
    hues = _retained_values(grid, cfg.held_out_hue_factor, cfg.held_out_hue_value)
    shapes = _retained_values(grid, cfg.held_out_shape_factor, cfg.held_out_shape_value)
    n = int(cfg.n_pair_cells)
    if n < 0:
        raise ValueError(f"n_pair_cells must be >= 0, got {n}")
    if n > min(hues.size, shapes.size):
        raise ValueError(
            f"n_pair_cells={n} exceeds the number of distinct hues ({hues.size}) or shapes ({shapes.size}) available — two pair cells sharing BOTH axes would not be disjoint"
        )
    stride = max(1, hues.size // max(n, 1))
    cells: list[FactorCell] = []
    for j in range(n):
        h = float(hues[j * stride % hues.size])
        s = float(shapes[j % shapes.size])
        cells.append(
            FactorCell(
                cell_id=f"pair-{j:03d}",
                values={cfg.held_out_hue_factor: (h,), cfg.held_out_shape_factor: (s,)},
                meta={
                    "family": "pair",
                    "axes": list(PAIR_AXES),
                    "pair": {cfg.held_out_hue_factor: h, cfg.held_out_shape_factor: s},
                    "note": "both values are RETAINED elsewhere in the grid; only their co-occurrence is held out — the row-3 request",
                },
            )
        )
    return cells


def _consecutive_runs(idx: np.ndarray, width: int) -> list[np.ndarray]:
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    out: list[np.ndarray] = []
    for start in range(idx.size):
        window = idx[start : start + width]
        if window.size == width and int(window[-1] - window[0]) == width - 1:
            out.append(window)
    return out


def _disjoint_windows(windows: Sequence[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    taken: set[int] = set()
    for w in windows:
        s = set(w.tolist())
        if not s & taken:
            kept.append(w)
            taken |= s
    return kept


def build_nway_cells(
    grid: FactorGrid,
    cfg: ValueHoldoutConfig,
    pair_cells: Sequence[FactorCell] = (),
    montero_widths: Optional[dict] = None,
) -> list[FactorCell]:
    widths = dict(montero_widths or {})
    band_w = int(widths.get("hue_band", 2))
    scale_w = int(widths.get("scale_corner", 3))
    rot_w = int(widths.get("orientation_corner", 3))
    hue_f = cfg.held_out_hue_factor
    shape_f = cfg.held_out_shape_factor
    hue_vals = grid.values[grid.index_of(hue_f)]
    floor_vals = grid.values[grid.index_of("floor_hue")]
    scale_vals = grid.values[grid.index_of("scale")]
    rot_vals = grid.values[grid.index_of("orientation")]
    reserved_hues = {cfg.held_out_hue_value}
    for c in pair_cells:
        reserved_hues.update((float(v) for v in c.values.get(hue_f, ())))
    allowed_hue_idx = np.array(
        [k for k, v in enumerate(hue_vals) if float(v) not in reserved_hues], dtype=np.int64
    )
    obj_windows = _disjoint_windows(_consecutive_runs(allowed_hue_idx, band_w))
    floor_windows = _disjoint_windows(
        _consecutive_runs(np.arange(floor_vals.size, dtype=np.int64), band_w)
    )
    shapes = _retained_values(grid, shape_f, cfg.held_out_shape_value)
    scale_corners = {
        "high": tuple((float(x) for x in scale_vals[-scale_w:])),
        "low": tuple((float(x) for x in scale_vals[:scale_w])),
    }
    rot_corners = {
        "high": tuple((float(x) for x in rot_vals[-rot_w:])),
        "low": tuple((float(x) for x in rot_vals[:rot_w])),
    }
    if not obj_windows or not floor_windows:
        raise ValueError(
            f"no {band_w}-wide band of retained {hue_f} values survives the reserved values {sorted(reserved_hues)} — lower n_pair_cells or the band width"
        )
    capacity = len(shapes) * len(obj_windows) * len(floor_windows) * 2 * 2
    n = int(cfg.n_nway_cells)
    if n < 0:
        raise ValueError(f"n_nway_cells must be >= 0, got {n}")
    if n > capacity:
        raise ValueError(
            f"n_nway_cells={n} exceeds the {capacity} structurally-disjoint slots available ({len(shapes)} shapes x {len(obj_windows)} object bands x {len(floor_windows)} floor bands x 2 scale corners x 2 pose corners)"
        )
    slots = [
        (si, oi, fi, sc, rc)
        for fi in range(len(floor_windows))
        for rc in ("high", "low")
        for sc in ("high", "low")
        for oi in range(len(obj_windows))
        for si in range(len(shapes))
    ]
    cells: list[FactorCell] = []
    for j, (si, oi, fi, sc, rc) in enumerate(slots[:n]):
        bg = tuple((float(x) for x in floor_vals[floor_windows[fi]]))
        obj = tuple((float(x) for x in hue_vals[obj_windows[oi]]))
        cells.append(
            FactorCell(
                cell_id=f"nway-{j:03d}",
                values={
                    "floor_hue": bg,
                    "wall_hue": bg,
                    hue_f: obj,
                    "scale": scale_corners[sc],
                    shape_f: (float(shapes[si]),),
                    "orientation": rot_corners[rc],
                },
                meta={
                    "family": "nway",
                    "hue_band_width": band_w,
                    "scale_corner": sc,
                    "orientation_corner": rc,
                    "shape_value": float(shapes[si]),
                    "object_band_start_index": int(obj_windows[oi][0]),
                    "floor_band_start_index": int(floor_windows[fi][0]),
                    "note": "Montero-shaped: a round-3 corner is comparable with a round-2 corner because only the PLACEMENT differs",
                },
            )
        )
    return cells


def build_value_holdout_cells(grid: FactorGrid, cfg: ValueHoldoutConfig) -> list[FactorCell]:
    """All cells of the condition, in emission order: value, then pair, then n-way."""
    value = build_value_cells(grid, cfg)
    pair = build_pair_cells(grid, cfg)
    nway = build_nway_cells(grid, cfg, pair_cells=pair)
    return [*value, *pair, *nway]


def build_value_holdout_split(
    grid: FactorGrid, cfg: ValueHoldoutConfig
) -> tuple[GenSplit, list[FactorCell]]:
    """Build the split (train = the exact complement) and run the load-time checks."""
    cells = build_value_holdout_cells(grid, cfg)
    split = build_cell_split(
        grid,
        cells,
        dataset=grid.dataset or cfg.dataset,
        condition=VALUE_HOLDOUT_CONDITION,
        variant=VALUE_HOLDOUT_VARIANT,
    )
    assert_non_empty_test(split)
    assert_index_partitions_grid(split)
    return (split, cells)


def assert_value_absent_from_train(
    split: GenSplit, factor: str, value: float, chunk: int = 1 << 20
) -> int:
    k = split.grid.index_of(factor)
    checked = 0
    for start in range(0, split.n_train, chunk):
        rows = split.train_rows[start : start + chunk]
        vals = split.grid.values_of(rows)[:, k]
        hits = vals == float(value)
        if hits.any():
            i = int(np.argmax(hits))
            raise HoldoutViolation(
                f"{split.dataset}/{split.condition}: {int(hits.sum())} train row(s) carry the held-out value {factor}={value}; first is row {int(rows[i])} — the store's fit pool would SEE the value"
            )
        checked += int(rows.size)
    return checked


ROW1_SCHEMA = {
    "version": 1,
    "description": "Row 1, SEEN: an ON-GRID request whose EVERY value is stored — the request denotes a TRAIN row, so the store's fit pool literally contains it. The correct answer is ACCEPT. ⭐: this is row 5's accept-leg. Both halves of the refusal pair are now well-formed on-grid requests that look identical to any syntactic rule and differ ONLY in whether the store holds the value.",
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'seen_hue' | 'seen_shape'",
        "request": "list of {factor, factor_index, value} terms; every factor once",
        "violated_factor": "⚠ NOTHING IS VIOLATED. The key name is fixed by the pin's EXCLUDED_RECORD_KEYS allowlist and is carried so row 1's record-key SET is identical to row 5's — otherwise `record_key_count` becomes a new giveaway. Here it names the AXIS: the factor carrying the mirrored, STORED value.",
        "reason": "statement of which stored value the axis factor carries",
        "row": "the TRAIN row this request denotes — the store held it",
    },
    "note": "⛔ Structurally matched to row5_held_out by construction: same term count, same factor order, same id format, the SAME record-key set, the same per-kind record count, and the axis value held CONSTANT within a kind exactly as row 5 does.",
    "scope": VALUE_HOLDOUT_SCOPE,
}


def build_row1_seen(grid: FactorGrid, split: GenSplit, cfg: ValueHoldoutConfig) -> dict:
    rng = np.random.default_rng(int(cfg.row1_seed))
    in_train = np.zeros(grid.n_rows, dtype=bool)
    in_train[split.train_rows] = True
    records: list[dict] = []
    held_out = {
        cfg.held_out_hue_factor: float(cfg.held_out_hue_value),
        cfg.held_out_shape_factor: float(cfg.held_out_shape_value),
    }
    plan = (
        ("seen_hue", cfg.held_out_hue_factor, float(cfg.row1_seen_hue_value)),
        ("seen_shape", cfg.held_out_shape_factor, float(cfg.row1_seen_shape_value)),
    )
    for kind, factor, value in plan:
        if value == held_out[factor]:
            raise ValueError(
                f"row 1/{kind}: {factor}={value} IS the held-out value — the accept-leg would be a refusal item"
            )
        _grid_index_of_value(grid, factor, value)
        k = grid.index_of(factor)
        vals = split.grid.values_of(split.train_rows)[:, k]
        pool = np.asarray(split.train_rows)[vals == value]
        if pool.size == 0:
            raise ValueError(
                f"row 1/{kind}: no TRAIN row carries {factor}={value} — the value is not in the store's fit pool and cannot be an accept-leg"
            )
        n = int(cfg.row1_n_per_kind)
        take = np.sort(rng.choice(pool, size=n, replace=pool.size < n))
        for i, row in enumerate(take.tolist()):
            if not in_train[row]:
                raise ValueError(f"row 1/{kind}: row {row} is not in the fit pool")
            rv = grid.values_of(np.array([row]))[0].tolist()
            records.append(
                {
                    "id": f"{grid.dataset}-row1-{kind}-{i:04d}",
                    "kind": kind,
                    "request": _terms(grid, rv),
                    "violated_factor": factor,
                    "reason": f"{factor}={value} is ON the grid AND the STORE held rows carrying it; this request denotes a train row, so every value it names is stored (see the condition's scope statement)",
                    "row": int(row),
                }
            )
    return {
        "schema": ROW1_SCHEMA,
        "scope": VALUE_HOLDOUT_SCOPE,
        "dataset": grid.dataset,
        "condition": VALUE_HOLDOUT_CONDITION,
        "seed": int(cfg.row1_seed),
        "n_per_kind": int(cfg.row1_n_per_kind),
        "seen": {
            cfg.held_out_hue_factor: float(cfg.row1_seen_hue_value),
            cfg.held_out_shape_factor: float(cfg.row1_seen_shape_value),
        },
        "mirrors_held_out": held_out,
        "rationale": ROW1_SEEN_RATIONALE,
        "factors": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "ranges": [[float(v[0]), float(v[-1])] for v in grid.values],
        },
        "records": records,
    }


def validate_row1(payload: dict, grid: FactorGrid, split: GenSplit) -> int:
    records = payload["records"]
    if not records:
        raise AssertionError("row 1 set is empty")
    in_train = np.zeros(grid.n_rows, dtype=bool)
    in_train[split.train_rows] = True
    n_f = grid.n_factors
    for rec in records:
        req = rec["request"]
        if len(req) != n_f:
            raise AssertionError(f"{rec['id']}: {len(req)} terms, expected {n_f}")
        vals = []
        for k, term in enumerate(req):
            if tuple(term.keys()) != TERM_KEYS:
                raise AssertionError(f"{rec['id']}: term keys {tuple(term.keys())}")
            if int(term["factor_index"]) != k or grid.names[k] != term["factor"]:
                raise AssertionError(f"{rec['id']}: factor/index mismatch at {k}")
            v = float(term["value"])
            if not np.any(grid.values[k] == v):
                raise AssertionError(
                    f"{rec['id']}: value {v} of {term['factor']!r} is OFF-GRID — row 1 is a STORED on-grid request, not a row-2 interpolant"
                )
            vals.append(v)
        row = int(rec["row"])
        if not np.array_equal(grid.values_of(np.array([row]))[0], np.array(vals)):
            raise AssertionError(f"{rec['id']}: request does not denote its own row")
        if not in_train[row]:
            raise AssertionError(
                f"{rec['id']}: row {row} is NOT in the store's fit pool — the accept-leg would be scored on a request the store never held"
            )
        axis = rec["violated_factor"]
        if float(vals[grid.index_of(axis)]) != float(payload["seen"][axis]):
            raise AssertionError(
                f"{rec['id']}: axis {axis!r} does not carry the declared stored value"
            )
        if float(vals[grid.index_of(axis)]) == float(payload["mirrors_held_out"][axis]):
            raise AssertionError(f"{rec['id']}: axis {axis!r} carries the HELD-OUT value")
    return len(records)


def assert_row1_matches_row5(row1: dict, row5: dict) -> dict:
    r1, r5 = (row1["records"], row5["records"])
    facts = {}
    if len(r1) != len(r5):
        raise AssertionError(f"row 1 has {len(r1)} records, row 5 has {len(r5)}")
    facts["n_records"] = len(r1)
    for name, recs in (("row1", r1), ("row5", r5)):
        facts[f"{name}_record_keys"] = sorted({k for r in recs for k in r})
        facts[f"{name}_n_kinds"] = len({r["kind"] for r in recs})
        facts[f"{name}_term_counts"] = sorted({len(r["request"]) for r in recs})
        facts[f"{name}_id_token_counts"] = sorted({len(str(r["id"]).split("-")) for r in recs})
        facts[f"{name}_record_key_counts"] = sorted({len(r) for r in recs})
    for prop in ("record_keys", "n_kinds", "term_counts", "id_token_counts", "record_key_counts"):
        a, b = (facts[f"row1_{prop}"], facts[f"row5_{prop}"])
        if a != b:
            raise AssertionError(
                f"structural mismatch on {prop}: row 1 {a!r} vs row 5 {b!r} — a control that differs from the set it controls is a NEW giveaway, not a control"
            )
    return facts


def row1_positive_control(payload: dict, grid: FactorGrid, split: GenSplit) -> str:
    if not split.test_rows.size:
        raise AssertionError("no test row exists to inject")
    bad_row = int(split.test_rows[0])
    rv = grid.values_of(np.array([bad_row]))[0].tolist()
    injected = {
        **payload,
        "records": [
            {**payload["records"][0], "request": _terms(grid, rv), "row": bad_row},
            *payload["records"][1:],
        ],
    }
    try:
        validate_row1(injected, grid, split)
    except AssertionError as exc:
        return str(exc)
    raise AssertionError(
        f"⛔ POSITIVE CONTROL FAILED: validate_row1 accepted an injected TEST row ({bad_row}) as a row-1 accept-leg record. The validator does not check the one thing it exists to check."
    )


ROW2_SCHEMA = {
    "version": 1,
    "description": "Row 2, UNSEEN-VALID: a request whose object_hue is OFF-GRID, strictly between two RETAINED, grid-adjacent hue values, with the other terms lifted verbatim from a TRAIN row. In-basin interpolation: both bracketing rows are in train, which validate_row2 checks. The correct answer is ACCEPT.",
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'unseen_valid'",
        "request": "list of {factor, factor_index, value} terms; every factor named once, one term OFF-GRID and strictly inside the realised range",
        "reason": "which two stored values the off-grid term is between",
    },
    "note": "⛔ The pair bracketing the row-5 hole is NOT grid-adjacent and is excluded by construction, so a row-2 request can never land on the row-5 value.",
}
ROW5_SCHEMA = {
    "version": 1,
    "description": "Row 5, COMPOSITIONAL REFUSAL: a request naming a value that is ON the grid — the image exists and the frozen decoder renders it and the frozen judge recognises it — but that the STORE never held. The correct answer is REFUSE.",
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'held_out_hue' | 'held_out_shape'",
        "request": "list of {factor, factor_index, value} terms; every factor once",
        "violated_factor": "the factor carrying the held-out value",
        "reason": "statement of which value the store never held",
        "row": "the dataset row this request denotes — it EXISTS, it was never stored",
    },
    "scope": VALUE_HOLDOUT_SCOPE,
}


def _terms(grid: FactorGrid, values: Sequence[float]) -> list[dict]:
    """The ``n_factors`` terms for a factor-value tuple, in :data:`TERM_KEYS` order."""
    return [
        dict(zip(TERM_KEYS, (grid.names[k], k, float(values[k])), strict=True))
        for k in range(grid.n_factors)
    ]


def retained_adjacent_pairs(
    grid: FactorGrid, factor: str, held_out: float
) -> list[tuple[int, int]]:
    vals = grid.values[grid.index_of(factor)]
    keep = vals != float(held_out)
    return [(k, k + 1) for k in range(vals.size - 1) if keep[k] and keep[k + 1]]


def build_row2_unseen(grid: FactorGrid, split: GenSplit, cfg: ValueHoldoutConfig) -> dict:
    """Row 2: off-grid hue midpoints of retained, grid-adjacent pairs, on TRAIN rows.

    The five non-hue terms are lifted from a train row, and the hue term is replaced by
    the midpoint of a retained adjacent pair **whose two bracketing rows are also in
    train**.  That last clause is the operational content of "in-basin": the request
    sits between two combinations the store actually holds.
    """
    hue_f = cfg.held_out_hue_factor
    k_hue = grid.index_of(hue_f)
    hue_vals = grid.values[k_hue]
    pairs = retained_adjacent_pairs(grid, hue_f, cfg.held_out_hue_value)
    if not pairs:
        raise ValueError(f"no grid-adjacent retained pair of {hue_f} exists")
    in_train = np.zeros(grid.n_rows, dtype=bool)
    in_train[split.train_rows] = True
    rng = np.random.default_rng(int(cfg.row2_seed))
    n_want = int(cfg.row2_n_requests)
    records: list[dict] = []
    pool = np.asarray(split.train_rows, dtype=np.int64)
    order = rng.permutation(pool.size)
    attempts = 0
    for t in order:
        if len(records) >= n_want:
            break
        attempts += 1
        row = int(pool[int(t)])
        fi = grid.factor_indices_of(np.array([row]))[0]
        lo, hi = pairs[int(rng.integers(len(pairs)))]
        brackets = []
        for k in (lo, hi):
            fi_b = fi.copy()
            fi_b[k_hue] = k
            brackets.append(int(grid.rows_of(fi_b[None, :])[0]))
        if not all((in_train[b] for b in brackets)):
            continue
        mid = 0.5 * (float(hue_vals[lo]) + float(hue_vals[hi]))
        if np.any(hue_vals == mid):
            raise ValueError(
                f"the midpoint {mid} of ({hue_vals[lo]}, {hue_vals[hi]}) is ON the grid — it would not be an unseen value"
            )
        vals = grid.values_of(np.array([row]))[0].tolist()
        vals[k_hue] = mid
        records.append(
            {
                "id": f"{grid.dataset}-row2-{len(records):04d}",
                "kind": "unseen_valid",
                "request": _terms(grid, vals),
                "reason": f"{hue_f} requested at {mid}, strictly between the stored values {float(hue_vals[lo])} and {float(hue_vals[hi])}; both bracketing rows ({brackets[0]}, {brackets[1]}) are in the store's fit pool",
            }
        )
    if len(records) < n_want:
        raise ValueError(
            f"row 2: only {len(records)} of {n_want} requests could be built from {attempts} train rows — the in-basin bracket condition is not satisfiable at this size"
        )
    return {
        "schema": ROW2_SCHEMA,
        "scope": VALUE_HOLDOUT_SCOPE,
        "dataset": grid.dataset,
        "condition": VALUE_HOLDOUT_CONDITION,
        "seed": int(cfg.row2_seed),
        "n_requests": len(records),
        "factors": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "ranges": [[float(v[0]), float(v[-1])] for v in grid.values],
        },
        "records": records,
    }


def build_row5_held_out(grid: FactorGrid, split: GenSplit, cfg: ValueHoldoutConfig) -> dict:
    """Row 5: on-grid requests naming a value the store never held (two kinds)."""
    rng = np.random.default_rng(int(cfg.row5_seed))
    test_set = np.zeros(grid.n_rows, dtype=bool)
    test_set[split.test_rows] = True
    records: list[dict] = []
    plan = (
        ("held_out_hue", cfg.held_out_hue_factor, float(cfg.held_out_hue_value)),
        ("held_out_shape", cfg.held_out_shape_factor, float(cfg.held_out_shape_value)),
    )
    for kind, factor, value in plan:
        k = grid.index_of(factor)
        vals = split.grid.values_of(split.test_rows)[:, k]
        pool = np.asarray(split.test_rows)[vals == value]
        if pool.size == 0:
            raise ValueError(
                f"row 5/{kind}: no held-out row carries {factor}={value} — the value cell did not do what it claims"
            )
        n = int(cfg.row5_n_per_kind)
        take = np.sort(rng.choice(pool, size=n, replace=pool.size < n))
        for i, row in enumerate(take.tolist()):
            if not test_set[row]:
                raise ValueError(f"row 5/{kind}: row {row} is not held out")
            rv = grid.values_of(np.array([row]))[0].tolist()
            records.append(
                {
                    "id": f"{grid.dataset}-row5-{kind}-{i:04d}",
                    "kind": kind,
                    "request": _terms(grid, rv),
                    "violated_factor": factor,
                    "reason": f"{factor}={value} is ON the grid and the frozen decoder renders it and the frozen judge recognises it, but the STORE never held a row carrying it (see the condition's scope statement)",
                    "row": int(row),
                }
            )
    return {
        "schema": ROW5_SCHEMA,
        "scope": VALUE_HOLDOUT_SCOPE,
        "dataset": grid.dataset,
        "condition": VALUE_HOLDOUT_CONDITION,
        "seed": int(cfg.row5_seed),
        "n_per_kind": int(cfg.row5_n_per_kind),
        "held_out": {
            cfg.held_out_hue_factor: float(cfg.held_out_hue_value),
            cfg.held_out_shape_factor: float(cfg.held_out_shape_value),
        },
        "factors": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "ranges": [[float(v[0]), float(v[-1])] for v in grid.values],
        },
        "records": records,
    }


def validate_row2(payload: dict, grid: FactorGrid, split: GenSplit) -> int:
    records = payload["records"]
    if not records:
        raise AssertionError("row 2 set is empty")
    in_train = np.zeros(grid.n_rows, dtype=bool)
    in_train[split.train_rows] = True
    n_f = grid.n_factors
    for rec in records:
        req = rec["request"]
        if len(req) != n_f:
            raise AssertionError(f"{rec['id']}: {len(req)} terms, expected {n_f}")
        off = []
        for term in req:
            if tuple(term.keys()) != TERM_KEYS:
                raise AssertionError(f"{rec['id']}: term keys {tuple(term.keys())}")
            k = int(term["factor_index"])
            if grid.names[k] != term["factor"]:
                raise AssertionError(f"{rec['id']}: factor/index mismatch")
            vals = grid.values[k]
            v = float(term["value"])
            if not np.any(vals == v):
                if not vals[0] < v < vals[-1]:
                    raise AssertionError(
                        f"{rec['id']}: off-grid value {v} of {term['factor']!r} is OUTSIDE the realised range — that is a row-6 record"
                    )
                lo = int(np.searchsorted(vals, v)) - 1
                off.append((k, lo, v))
        if len(off) != 1:
            raise AssertionError(f"{rec['id']}: {len(off)} off-grid terms, expected exactly 1")
        k, lo, v = off[0]
        if abs(v - 0.5 * (float(vals[lo]) + float(vals[lo + 1]))) > 0:
            pass
        fi = np.array(
            [
                int(np.flatnonzero(grid.values[j] == float(req[j]["value"]))[0]) if j != k else 0
                for j in range(n_f)
            ],
            dtype=np.int64,
        )
        for b in (lo, lo + 1):
            fi_b = fi.copy()
            fi_b[k] = b
            row = int(grid.rows_of(fi_b[None, :])[0])
            if not in_train[row]:
                raise AssertionError(
                    f"{rec['id']}: bracketing row {row} (at {grid.names[k]}={float(grid.values[k][b])}) is NOT in the store's fit pool — the request is not in-basin and cannot be scored as unseen-VALID"
                )
    return len(records)


def validate_row5(payload: dict, grid: FactorGrid, split: GenSplit) -> int:
    """Assert every row-5 record is ON-grid, denotes its own row, and is HELD OUT."""
    records = payload["records"]
    if not records:
        raise AssertionError("row 5 set is empty")
    test_set = np.zeros(grid.n_rows, dtype=bool)
    test_set[split.test_rows] = True
    n_f = grid.n_factors
    for rec in records:
        req = rec["request"]
        if len(req) != n_f:
            raise AssertionError(f"{rec['id']}: {len(req)} terms, expected {n_f}")
        vals = []
        for k, term in enumerate(req):
            if tuple(term.keys()) != TERM_KEYS:
                raise AssertionError(f"{rec['id']}: term keys {tuple(term.keys())}")
            if int(term["factor_index"]) != k or grid.names[k] != term["factor"]:
                raise AssertionError(f"{rec['id']}: factor/index mismatch at {k}")
            v = float(term["value"])
            if not np.any(grid.values[k] == v):
                raise AssertionError(
                    f"{rec['id']}: value {v} of {term['factor']!r} is OFF-GRID — row 5 is an ON-GRID value the store never held, not an invalid request"
                )
            vals.append(v)
        row = int(rec["row"])
        if not np.array_equal(grid.values_of(np.array([row]))[0], np.array(vals)):
            raise AssertionError(f"{rec['id']}: request does not denote its own row")
        if not test_set[row]:
            raise AssertionError(
                f"{rec['id']}: row {row} is IN the store's fit pool — the value was stored and the record is not a refusal item"
            )
        vf = rec["violated_factor"]
        if float(vals[grid.index_of(vf)]) not in (float(payload["held_out"][vf]),):
            raise AssertionError(
                f"{rec['id']}: violated_factor {vf!r} does not carry the held-out value"
            )
    return len(records)


def grid_reach_features(record: dict, grid: FactorGrid) -> dict[str, float]:
    n_off = 0
    worst = 0.0
    for term in record["request"]:
        k = int(term["factor_index"])
        v = float(term["value"])
        vals = grid.values[k]
        if not np.any(vals == v):
            n_off += 1
            if v < vals[0]:
                worst = max(worst, float(vals[0] - v))
            elif v > vals[-1]:
                worst = max(worst, float(v - vals[-1]))
    return {"n_offgrid_terms": float(n_off), "max_distance_outside_range": float(worst)}


def grid_reach_auroc(
    invalid_records: Sequence[dict], control_records: Sequence[dict], grid: FactorGrid
) -> dict[str, float]:
    def table(recs):
        cols: dict[str, list[float]] = {}
        for r in recs:
            for k, v in grid_reach_features(r, grid).items():
                cols.setdefault(k, []).append(v)
        return cols

    pos, neg = (table(invalid_records), table(control_records))
    return {k: float(max(auroc(pos[k], neg[k]), 1.0 - auroc(pos[k], neg[k]))) for k in sorted(pos)}


EXTRA_QUERY_SETS: dict[str, dict] = {
    "row1_seen": {
        "role": "valid",
        "file": "row1_seen.json",
        "builder": "nclu.data.gen_value_holdout.build_row1_seen",
        "n_terms": "n_factors",
        "description": "row 1 (seen): an ON-GRID request whose EVERY value is STORED — it denotes a train row. ⭐: this is the accept-leg of BOTH refusal rows. It is structurally matched to row5_held_out (same record-key set, same term count, same id shape, same per-kind count, axis value constant per kind), so the two differ only in whether the store holds the value.",
    },
    "row2_unseen": {
        "role": "valid",
        "file": "row2_unseen.json",
        "builder": "nclu.data.gen_value_holdout.build_row2_unseen",
        "n_terms": "n_factors",
        "description": "row 2 (unseen-valid): an OFF-GRID hue strictly between two RETAINED, grid-adjacent stored values. The correct answer is ACCEPT, which is what makes it the control the refusal rows are scored against.",
    },
    "row5_held_out": {
        "role": "invalid",
        "file": "row5_held_out.json",
        "builder": "nclu.data.gen_value_holdout.build_row5_held_out",
        "n_terms": "n_factors",
        "controls": {"held_out_hue": "row1_seen", "held_out_shape": "row1_seen"},
        "description": "row 5 (compositional refusal): an ON-GRID value the STORE never held. ⛔ 'invalid' here means 'the correct answer is refuse', NOT 'unsatisfiable' — the image exists and the frozen judge recognises it.",
    },
}
ROW6_CONTROLS = {"duplicate_factor": "a7_redundant", "out_of_range": "row1_seen"}


def value_holdout_metadata(
    split: GenSplit, cells: Sequence[FactorCell], cfg: ValueHoldoutConfig
) -> dict:
    """The condition's metadata: scope, chosen values, rationale, realised counts."""
    by_family: dict[str, list[FactorCell]] = {}
    for c in cells:
        by_family.setdefault(str(c.meta.get("family", "?")), []).append(c)
    return {
        "scope": VALUE_HOLDOUT_SCOPE,
        "dataset": split.dataset,
        "condition": VALUE_HOLDOUT_CONDITION,
        "variant": VALUE_HOLDOUT_VARIANT,
        "row_evidence_ranking": ROW_EVIDENCE_RANKING,
        "row1_seen_rationale": ROW1_SEEN_RATIONALE,
        "config": cfg.to_dict(),
        "held_out_values": {
            cfg.held_out_hue_factor: float(cfg.held_out_hue_value),
            cfg.held_out_shape_factor: float(cfg.held_out_shape_value),
        },
        "rationale": {"hue": HELD_OUT_HUE_RATIONALE, "shape": HELD_OUT_SHAPE_RATIONALE},
        "n_total": split.n_total,
        "n_train_realised": split.n_train,
        "n_test_realised": split.n_test,
        "rows_by_family": {
            k: int(sum((c.rows(split.grid).size for c in v))) for k, v in by_family.items()
        },
        "n_cells_by_family": {k: len(v) for k, v in by_family.items()},
        "cells": [c.to_dict(split.grid) for c in cells],
        "frozen_artifact_warning": "⛔ The encoder/decoder/judge frozen under `recomb2element` were fitted on rows this condition holds out. That is DELIBERATE and is the scope statement above: the hold-out is from the STORE's fit pool only. ⛔ Do not quote a row from this condition as an end-to-end 'never seen' result.",
    }


def emit_value_holdout(
    grid: FactorGrid,
    out_dir: str | Path,
    cfg: Optional[ValueHoldoutConfig] = None,
    run_detector: bool = True,
) -> dict:
    cfg = cfg or ValueHoldoutConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split, cells = build_value_holdout_split(grid, cfg)
    assertions = {
        "assert_non_empty_test": assert_non_empty_test(split),
        "assert_index_partitions_grid": assert_index_partitions_grid(split),
        "assert_cells_disjoint": assert_cells_disjoint(grid, cells),
        "assert_holdout_disjoint": assert_holdout_disjoint(split),
        f"assert_value_absent_from_train[{cfg.held_out_hue_factor}]": assert_value_absent_from_train(
            split, cfg.held_out_hue_factor, cfg.held_out_hue_value
        ),
        f"assert_value_absent_from_train[{cfg.held_out_shape_factor}]": assert_value_absent_from_train(
            split, cfg.held_out_shape_factor, cfg.held_out_shape_value
        ),
    }
    if assertions["assert_holdout_disjoint"] != split.n_train:
        raise HoldoutViolation(
            f"assert_holdout_disjoint checked {assertions['assert_holdout_disjoint']} rows but the split has {split.n_train} train rows — the check was partial"
        )
    index_path, index_sha = save_index(
        split, out_dir / "factor_index.npz", extra={"scope": VALUE_HOLDOUT_SCOPE}
    )
    meta = value_holdout_metadata(split, cells, cfg)
    meta_path, meta_sha = write_json_with_checksum(meta, out_dir / "value_cells.json")
    c_set = build_c_invalid(
        grid, n_per_kind=int(cfg.c_invalid_n_per_kind), seed=int(cfg.c_invalid_seed)
    )
    n_c = validate_c_invalid(c_set, grid)
    c_path, c_sha = write_json_with_checksum(c_set, out_dir / "c_invalid.json")
    a7_rows = select_a7_rows(split.train_rows, int(cfg.a7_n_requests), seed=int(cfg.a7_seed))
    a7_set = build_a7_redundant(grid, a7_rows, seed=int(cfg.a7_seed))
    n_a7 = validate_a7_redundant(a7_set, grid)
    a7_path, a7_sha = write_json_with_checksum(a7_set, out_dir / "a7_redundant.json")
    row1 = build_row1_seen(grid, split, cfg)
    n_row1 = validate_row1(row1, grid, split)
    row1_control_msg = row1_positive_control(row1, grid, split)
    row1_path, row1_sha = write_json_with_checksum(row1, out_dir / "row1_seen.json")
    row2 = build_row2_unseen(grid, split, cfg)
    n_row2 = validate_row2(row2, grid, split)
    row2_path, row2_sha = write_json_with_checksum(row2, out_dir / "row2_unseen.json")
    row5 = build_row5_held_out(grid, split, cfg)
    n_row5 = validate_row5(row5, grid, split)
    row1_match = assert_row1_matches_row5(row1, row5)
    row5_path, row5_sha = write_json_with_checksum(row5, out_dir / "row5_held_out.json")
    emitted = {
        "base": {"derived_from_sha256": index_sha, "n_records": int(split.n_train + split.n_test)},
        "c_invalid": {"sha256": c_sha, "n_records": int(n_c)},
        "a7_redundant": {"sha256": a7_sha, "n_records": int(n_a7)},
    }
    reg_path, _ = write_query_set_index(out_dir, split.dataset, emitted)
    payload = json.loads(Path(reg_path).read_text())
    payload["condition"] = VALUE_HOLDOUT_CONDITION
    payload["scope"] = VALUE_HOLDOUT_SCOPE
    payload["sets"]["c_invalid"]["controls"] = dict(ROW6_CONTROLS)
    payload["sets"]["c_invalid"]["control_note"] = (
        "⭐: out_of_range is re-controlled onto ROW 1 (the accept side of the refusal pair, which is now row 1 for BOTH refusal rows); duplicate_factor keeps a7_redundant because it carries n_factors+1 terms and a 6-term control would restore the term-count giveaway."
    )
    payload["row_evidence_ranking"] = ROW_EVIDENCE_RANKING
    payload["sets"]["row1_seen"] = {
        **EXTRA_QUERY_SETS["row1_seen"],
        "sha256": row1_sha,
        "n_records": int(n_row1),
    }
    payload["sets"]["row2_unseen"] = {
        **EXTRA_QUERY_SETS["row2_unseen"],
        "sha256": row2_sha,
        "n_records": int(n_row2),
    }
    payload["sets"]["row5_held_out"] = {
        **EXTRA_QUERY_SETS["row5_held_out"],
        "sha256": row5_sha,
        "n_records": int(n_row5),
    }
    reg_path, reg_sha = write_json_with_checksum(payload, out_dir / "query_sets.json")
    report = {
        "scope": VALUE_HOLDOUT_SCOPE,
        "dataset": split.dataset,
        "condition": VALUE_HOLDOUT_CONDITION,
        "config": cfg.to_dict(),
        "held_out_values": meta["held_out_values"],
        "counts": {
            "n_total": split.n_total,
            "n_train_realised": split.n_train,
            "n_test_realised": split.n_test,
            "n_cells_by_family": meta["n_cells_by_family"],
            "rows_by_family": meta["rows_by_family"],
            "held_out_fraction": float(split.n_test / split.n_total),
        },
        "assertions": assertions,
        "row_evidence_ranking": ROW_EVIDENCE_RANKING,
        "row1_positive_control": {
            "injected": "the first test row, as the first row-1 record",
            "validate_row1_raised": row1_control_msg,
        },
        "row1_structural_match_to_row5": row1_match,
        "artifacts": {
            "factor_index.npz": {"path": str(index_path), "sha256": index_sha},
            "value_cells.json": {"path": str(meta_path), "sha256": meta_sha},
            "c_invalid.json": {"path": str(c_path), "sha256": c_sha, "n_records": int(n_c)},
            "a7_redundant.json": {"path": str(a7_path), "sha256": a7_sha, "n_records": int(n_a7)},
            "row1_seen.json": {
                "path": str(row1_path),
                "sha256": row1_sha,
                "n_records": int(n_row1),
            },
            "row2_unseen.json": {
                "path": str(row2_path),
                "sha256": row2_sha,
                "n_records": int(n_row2),
            },
            "row5_held_out.json": {
                "path": str(row5_path),
                "sha256": row5_sha,
                "n_records": int(n_row5),
            },
            "query_sets.json": {"path": str(reg_path), "sha256": reg_sha},
        },
    }
    if run_detector:
        report["detector"] = audit_query_sets(reg_path, tolerance=float(cfg.detector_tolerance))
        report["grid_reach_diagnostic"] = {
            "note": "⛔ REPORTED, NEVER ASSERTED. A bare grid look-up decides row 6's out_of_range half (the pin measures the same statistic at 1.0000 and declares it); it decides NOTHING on row 5 vs row 1, both of which are all-on-grid. ⭐: 'row5_vs_row2_SUPERSEDED' is retained because it is the FINDING — 1.0000 on n_offgrid_terms is why row 2 could not be row 5's accept-leg.",
            "row5_vs_row1": grid_reach_auroc(row5["records"], row1["records"], grid),
            "row5_vs_row2_SUPERSEDED": grid_reach_auroc(row5["records"], row2["records"], grid),
            "row6_out_of_range_vs_row1": grid_reach_auroc(
                [r for r in c_set["records"] if r["kind"] == "out_of_range"], row1["records"], grid
            ),
            "row6_duplicate_factor_vs_a7": grid_reach_auroc(
                [r for r in c_set["records"] if r["kind"] == "duplicate_factor"],
                a7_set["records"],
                grid,
            ),
        }
        report["detector_passed"] = not report["detector"]["violations"]
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Emit the value-hold-out condition.")
    p.add_argument("--data-root", default=None, help="dataset root (default: the pin's)")
    p.add_argument("--out-dir", default=None, help="default: <root>/value_holdout")
    p.add_argument("--report", default=None, help="write the JSON report here")
    p.add_argument("--n-nway-cells", type=int, default=None)
    p.add_argument("--n-pair-cells", type=int, default=None)
    p.add_argument("--no-detector", action="store_true")
    args = p.parse_args(argv)
    from nclu.data import shapes3d as s3d

    overrides: dict[str, Any] = {}
    if args.n_nway_cells is not None:
        overrides["n_nway_cells"] = args.n_nway_cells
    if args.n_pair_cells is not None:
        overrides["n_pair_cells"] = args.n_pair_cells
    cfg = ValueHoldoutConfig.from_mapping(overrides)
    root = Path(args.data_root) if args.data_root else s3d.shapes3d_root()
    out_dir = Path(args.out_dir) if args.out_dir else root / VALUE_HOLDOUT_CONDITION
    labels = s3d.load_labels(root)
    grid = s3d.build_grid(labels)
    grid.verify_bijection(labels)
    report = emit_value_holdout(grid, out_dir, cfg, run_detector=not args.no_detector)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text + "\n")
    print(text)
    if report.get("detector") is not None:
        try:
            assert_no_model_detector_at_chance(
                out_dir / "query_sets.json", tolerance=float(cfg.detector_tolerance)
            )
        except AssertionError as exc:
            print(str(exc))
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
