"""⭐ The VALUE hold-out condition: whole factor VALUES the store never held.

⛔ **Read §1 before reading a single number produced by this module.**

1. THE SCOPE STATEMENT — it travels with every artifact
-------------------------------------------------------
The hold-out is from the **store's fit pool**, ⛔ not from the encoder, the decoder
or the judge.  Those three are frozen, hash-pinned and read-only (charter §A.4) and
were fitted on a train split containing **every hue and every shape**.  Re-fitting
them on a value hold-out would void every wave-2 number and is not in scope.

⇒ "never trained on" in wave 3 means: **the store never held a row carrying that
value.**  The decoder can still render it and the judge can still recognise it —
which is precisely what makes rows 2 / 5 / 6 / 7 scorable at all.

:data:`VALUE_HOLDOUT_SCOPE` is that paragraph as a string, and
:func:`emit_value_holdout` writes it into every artifact it produces, so a hub
cannot quote a row without it.  A referee who is not told this reads "never seen"
as end-to-end and the claim dies.

2. What the condition holds out, and why those values
-----------------------------------------------------
Wave 2 held out **combinations**.  Every row of the taxonomy that turns on a value
the model has never stored — rows 2, 5, 6, 7 — is unmeasurable against a combination
hold-out.  This module closes that gap by emitting **three families of cells**:

``value-object_hue``  one **whole** ``object_hue`` value, across every shape, scale
                      and pose.  The row-5 hue.
``value-shape``       one **whole** ``shape`` value (minus the rows the hue cell
                      already took — see §4).  The row-5 shape.
``pair-*``            2-way cells — one ``object_hue`` **and** one ``shape``, each
                      individually retained, never co-occurring.  Rows 3.
``nway-*``            Montero-shaped 6-factor conjunctions over the values that
                      remain.  Rows 4, and the wave-3 multi-corner estimand.

⭐ **The chosen values, and the rejected ones, recorded before the build** — see
:data:`HELD_OUT_HUE_RATIONALE` and :data:`HELD_OUT_SHAPE_RATIONALE`.  ⚠ ``scale`` and
``orientation`` are ⛔ ineligible: wave 2 measured both as **physical defaults** the
model degrades *towards* (median ``scale`` index = the default object size, median
``orientation`` index = the null pose), so a held-out default is unremovable for
reasons that have nothing to do with the store.

3. ⛔ Nothing in :mod:`chlu.data.gen_splits` is reimplemented here
------------------------------------------------------------------
``FactorCell`` with a factor **absent ⇒ unconstrained** already expresses a value
hold-out.  ``build_cell_split`` already makes train the exact complement.
``assert_holdout_disjoint`` / ``assert_index_partitions_grid`` /
``assert_non_empty_test`` / ``assert_cells_disjoint`` / ``save_index`` /
``write_json_with_checksum`` are the proofs, and :func:`emit_value_holdout` runs
**all** of them and records their returned counts.

4. ⚠ Why ``value-shape`` carves the hue out of itself
------------------------------------------------------
``{shape: s*}`` and ``{object_hue: h*}`` **intersect** — the rows with both — and
``assert_cells_disjoint`` would (correctly) fire.  So ``value-shape`` is written as
``{shape: s*, object_hue: everything except h*}``.  The **union** of the two cells is
still "every row carrying ``h*``" ∪ "every row carrying ``s*``", which is what the
hold-out means; only the book-keeping of which cell owns the overlap changes.
:func:`assert_value_absent_from_train` checks the property *directly*, on the train
index, so it fires for exactly one reason and does not depend on that argument.

5. The query sets (rows 2, 5, 6) and the no-model detector
-----------------------------------------------------------
Row 2 is an **off-grid hue strictly between two RETAINED, grid-adjacent values**,
with the other five terms lifted from a train row — and :func:`validate_row2` checks
that **both bracketing rows are in train**, which is what "in-basin" means
mechanically.  ⛔ The pair ``(0.4, 0.6)`` that brackets a hole at ``0.5`` is *not*
grid-adjacent and is therefore excluded by construction, so a row-2 request can never
land on the row-5 value.

Row 5 is an **on-grid** value the store never held.  Row 6 is ``build_c_invalid``'s
``out_of_range`` / ``duplicate_factor`` records — ⛔ the pin's format, not a second one.

All three are registered in ``query_sets.json`` so the pin's
``assert_no_model_detector_at_chance`` reaches them from disk.  Every invalid kind
names its structural control:

=========================  ===================  ==========================================
kind                       control              why
=========================  ===================  ==========================================
``held_out_hue``           ``row2_unseen``      the separation that IS the claim
``held_out_shape``         ``row2_unseen``      same
``out_of_range``           ``row2_unseen``      same; 6 terms on both sides
``duplicate_factor``       ``a7_redundant``     ⛔ 7 terms — a 6-term control would hand
                                                back the term-count giveaway R26.1 killed
=========================  ===================  ==========================================

⭐ :func:`grid_reach_auroc` is a **declared extra diagnostic**, reported and ⛔ never
asserted on: it asks whether a bare grid look-up decides a row.  It does for row 6's
``out_of_range`` half (that IS what out-of-range means — the pin says so in
``definitional_features``) and it does **not** for row 5, which is the whole reason
row 5 is the load-bearing refusal row and row 6 is not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .gen_splits import (
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
    write_query_set_registry,
)

__all__ = [
    "HELD_OUT_HUE_RATIONALE",
    "HELD_OUT_SHAPE_RATIONALE",
    "VALUE_HOLDOUT_CONDITION",
    "VALUE_HOLDOUT_SCOPE",
    "VALUE_HOLDOUT_VARIANT",
    "ValueHoldoutConfig",
    "assert_value_absent_from_train",
    "build_nway_cells",
    "build_pair_cells",
    "build_row2_unseen",
    "build_row5_held_out",
    "build_value_cells",
    "build_value_holdout_cells",
    "build_value_holdout_split",
    "emit_value_holdout",
    "grid_reach_auroc",
    "grid_reach_features",
    "main",
    "validate_row2",
    "validate_row5",
    "value_holdout_metadata",
]

VALUE_HOLDOUT_CONDITION = "value_holdout"
VALUE_HOLDOUT_VARIANT = "value_cells_plus_combination_cells"

#: ⛔ §2 of the task, verbatim.  Written into EVERY artifact this module emits and
#: inherited by every scoreboard row (:mod:`chlu.experiments.gen_scoreboard`).
VALUE_HOLDOUT_SCOPE = (
    "SCOPE OF THE VALUE HOLD-OUT (⛔ quote no row without this). The hold-out is from "
    "the STORE'S FIT POOL, not from the encoder, the decoder or the judge. Those three "
    "are frozen, hash-pinned and read-only (charter §A.4) and were fitted on a train "
    "split containing EVERY hue and EVERY shape; re-fitting them on a value hold-out "
    "would void every wave-2 number and is out of scope. Therefore 'never trained on' "
    "means here: THE STORE NEVER HELD A ROW CARRYING THAT VALUE. The decoder can still "
    "render the value and the judge can still recognise it — which is what makes rows "
    "2/5/6/7 scorable at all. ⛔ Read end-to-end, 'never seen' is false and the claim "
    "dies with it."
)

#: ⭐ Chosen BEFORE the build (task §3.1), with the rejected candidates recorded.
HELD_OUT_HUE_RATIONALE = {
    "factor": "object_hue",
    "value": 0.5,
    "grid_index": 5,
    "why": (
        "The interior mid-point of the 10-value hue ring. (a) It is the object's own "
        "photometric property — the thing the request names and the judge scores — "
        "unlike floor_hue/wall_hue, which are background. (b) Hue is CIRCULAR, so no "
        "hue value is a physical default the model degrades towards; the §3.1 hazard "
        "that rules out scale and orientation does not apply to it. (c) It sits four "
        "grid steps from Montero's retained band (object_hue >= 0.8), so the value "
        "hole does not sit inside the region the headline and wide COMBINATION "
        "conditions already spend. (d) It is interior, so BOTH of its grid neighbours "
        "are retained and the hole is two-sided — a hole at 0.0 or 0.9 would sit on "
        "the realised range boundary, where build_c_invalid's out_of_range records "
        "are also placed, and rows 5 and 6 would become adjacent in value space."
    ),
    "rejected": {
        "0.8 / 0.9": (
            "Montero's held-out hue band. A value hold-out placed there is entangled "
            "with the headline combination condition and its 72 rows."
        ),
        "0.0 / 0.9 (as range endpoints)": (
            "one-sided in the realised range: the nearest off-grid values are exactly "
            "where out_of_range (row 6) lives."
        ),
        "scale / orientation (any value)": (
            "⛔ ineligible per task §3.1: wave 2 measured both as PHYSICAL DEFAULTS the "
            "model degrades towards, so a held-out default is unremovable for reasons "
            "that have nothing to do with the store."
        ),
    },
}

HELD_OUT_SHAPE_RATIONALE = {
    "factor": "shape",
    "value": 0.0,
    "grid_index": 0,
    "label": "cube",
    "why": (
        "The cube is the shape most distant from the retained set under the only "
        "degradation mode we have measured. Shape is CATEGORICAL — it cannot be "
        "interpolated — so 'held out entirely' is only weakened by a retained "
        "NEAR-NEIGHBOUR, and the cube has none among {cylinder, sphere, oblong}: it "
        "is the one shape with flat faces and hard edges."
    ),
    "rejected": {
        "2.0 (sphere)": (
            "⛔ the shape hazard of §3.1, exactly: a degraded render is a rounded blob, "
            "so the SPHERE is the shape the model degrades TOWARDS. A held-out sphere "
            "would be unremovable for rendering reasons, not store reasons — the same "
            "argument that rules out the median scale and the null pose."
        ),
        "1.0 (cylinder)": (
            "Montero's held-out shape, and the wide condition's cell-0 shape. Already "
            "entangled with both combination conditions."
        ),
        "3.0 (oblong/capsule)": (
            "visually intermediate between the cylinder and the sphere, both retained. "
            "A retained near-neighbour weakens 'held out entirely' — refusal could be "
            "declined on interpolation grounds, and the row would stop being a refusal "
            "row."
        ),
    },
}

#: ⭐ The 2-way (row 3) cells are placed on the SAME two axes the value hold-out uses,
#: so "unseen PAIR of seen values" (row 3) and "unseen VALUE" (row 5) differ in exactly
#: one thing and the contrast between the rows is clean.
PAIR_AXES = ("object_hue", "shape")


@dataclass(frozen=True)
class ValueHoldoutConfig:
    """Configuration for the value-hold-out condition.

    ⛔ **Deliberately not a block in** :mod:`chlu.config`.  Its same-anchor merges have
    cost this programme real time; the precedent for putting the dataclass next to the
    code that reads it is ``StreamMemoryConfig`` in :mod:`chlu.core.blocks` and
    ``PsiSpec``, both of which say so in their own docstrings.  Override with
    :meth:`from_mapping`.

    ⭐ The four query sets are **class-balanced at 512 records each** (row 2: 512;
    row 5: 256 + 256; row 6: 256 + 256; A7: 512).  AUROC is insensitive to imbalance,
    but a balanced set makes the audit cheap and makes a miscount visible.
    """

    #: The dataset module's name.  Only ``shapes3d`` has ``object_hue`` and ``shape``.
    dataset: str = "shapes3d"
    held_out_hue_factor: str = "object_hue"
    held_out_hue_value: float = 0.5
    held_out_shape_factor: str = "shape"
    held_out_shape_value: float = 0.0
    #: Row-3 cells: one (hue, shape) pair each, both values individually retained.
    n_pair_cells: int = 3
    #: Row-4 cells: Montero-shaped 6-factor conjunctions. ⭐ >= 49 is the wave-3
    #: multi-corner estimand (w3-hub-common §3); the default meets it exactly.
    n_nway_cells: int = 49
    #: Records per query set.
    row2_n_requests: int = 512
    row5_n_per_kind: int = 256
    c_invalid_n_per_kind: int = 256
    a7_n_requests: int = 512
    #: Seeds.  ⛔ Every draw in this module is seeded; nothing is tuned to a count.
    pair_seed: int = 0
    nway_seed: int = 0
    row2_seed: int = 0
    row5_seed: int = 0
    c_invalid_seed: int = 0
    a7_seed: int = 0
    #: ⛔ The pin's tolerance (``tests/test_gen_c_set_a7.py``: ``DETECTOR_TOLERANCE``).
    #: Raising it is not a way to make the audit pass.
    detector_tolerance: float = 0.05
    #: Extra provenance carried verbatim into the metadata.
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Optional[dict] = None) -> "ValueHoldoutConfig":
        """Build from a plain mapping, ignoring nothing silently.

        ⛔ An unknown key raises: a typo'd override that is quietly dropped is how a
        run ends up not being the run its metadata claims.
        """
        mapping = dict(mapping or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"ValueHoldoutConfig: unknown key(s) {unknown}; known keys are "
                f"{sorted(known)}"
            )
        return cls(**mapping)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# the cells
# ---------------------------------------------------------------------------


def _grid_index_of_value(grid: FactorGrid, factor: str, value: float) -> int:
    """Grid index of ``value`` on ``factor``.  ⛔ Exact float equality, as everywhere."""
    vals = grid.values[grid.index_of(factor)]
    hit = np.flatnonzero(vals == float(value))
    if hit.size != 1:
        raise ValueError(
            f"value {value!r} is not a grid value of factor {factor!r} "
            f"(realised values: {vals.tolist()})"
        )
    return int(hit[0])


def _retained_values(grid: FactorGrid, factor: str, held_out: float) -> np.ndarray:
    vals = grid.values[grid.index_of(factor)]
    return vals[vals != float(held_out)]


def build_value_cells(grid: FactorGrid, cfg: ValueHoldoutConfig) -> list[FactorCell]:
    """The two VALUE cells: one whole hue, one whole shape.

    ⚠ See module §4: the shape cell excludes the held-out hue so the two cells are
    disjoint; their **union** is still every row carrying either value, which is what
    :func:`assert_value_absent_from_train` checks directly.
    """
    _grid_index_of_value(grid, cfg.held_out_hue_factor, cfg.held_out_hue_value)
    _grid_index_of_value(grid, cfg.held_out_shape_factor, cfg.held_out_shape_value)
    retained_hues = _retained_values(
        grid, cfg.held_out_hue_factor, cfg.held_out_hue_value
    )
    hue_cell = FactorCell(
        cell_id=f"value-{cfg.held_out_hue_factor}",
        values={cfg.held_out_hue_factor: (float(cfg.held_out_hue_value),)},
        meta={
            "family": "value",
            "held_out_factor": cfg.held_out_hue_factor,
            "held_out_value": float(cfg.held_out_hue_value),
            "rationale": HELD_OUT_HUE_RATIONALE,
            "note": (
                "every other factor is UNCONSTRAINED — the value is held out across "
                "every shape, scale and pose, which is what makes it a VALUE hold-out "
                "and not a combination hold-out"
            ),
        },
    )
    shape_cell = FactorCell(
        cell_id=f"value-{cfg.held_out_shape_factor}",
        values={
            cfg.held_out_shape_factor: (float(cfg.held_out_shape_value),),
            # ⚠ NOT a weakening of the shape hold-out: the rows carved out here are
            # exactly the rows the hue cell already holds out.  See module §4.
            cfg.held_out_hue_factor: tuple(float(x) for x in retained_hues),
        },
        meta={
            "family": "value",
            "held_out_factor": cfg.held_out_shape_factor,
            "held_out_value": float(cfg.held_out_shape_value),
            "rationale": HELD_OUT_SHAPE_RATIONALE,
            "note": (
                f"the {cfg.held_out_hue_factor} constraint carves out ONLY the rows the "
                f"{cfg.held_out_hue_factor} cell already owns, so the two cells are "
                "disjoint; their union is every row carrying either held-out value"
            ),
        },
    )
    return [hue_cell, shape_cell]


def build_pair_cells(grid: FactorGrid, cfg: ValueHoldoutConfig) -> list[FactorCell]:
    """Row-3 cells: an unseen **PAIR** of individually-retained values.

    Each cell constrains exactly the two :data:`PAIR_AXES` — ``{object_hue: h,
    shape: s}`` — and leaves the other four factors free.  Both ``h`` and ``s`` are
    retained, so the model has stored rows carrying each of them; it has never stored
    a row carrying **both**.  That is the 2-way compositional request.

    Deterministic: the hues are taken at an even stride over the retained hue ring
    starting from the first retained value, and the shapes cycle over the retained
    shapes, so no two cells share a hue **or** a shape.
    """
    hues = _retained_values(grid, cfg.held_out_hue_factor, cfg.held_out_hue_value)
    shapes = _retained_values(grid, cfg.held_out_shape_factor, cfg.held_out_shape_value)
    n = int(cfg.n_pair_cells)
    if n < 0:
        raise ValueError(f"n_pair_cells must be >= 0, got {n}")
    if n > min(hues.size, shapes.size):
        raise ValueError(
            f"n_pair_cells={n} exceeds the number of distinct hues ({hues.size}) or "
            f"shapes ({shapes.size}) available — two pair cells sharing BOTH axes "
            "would not be disjoint"
        )
    stride = max(1, hues.size // max(n, 1))
    cells: list[FactorCell] = []
    for j in range(n):
        h = float(hues[(j * stride) % hues.size])
        s = float(shapes[j % shapes.size])
        cells.append(
            FactorCell(
                cell_id=f"pair-{j:03d}",
                values={
                    cfg.held_out_hue_factor: (h,),
                    cfg.held_out_shape_factor: (s,),
                },
                meta={
                    "family": "pair",
                    "axes": list(PAIR_AXES),
                    "pair": {cfg.held_out_hue_factor: h, cfg.held_out_shape_factor: s},
                    "note": (
                        "both values are RETAINED elsewhere in the grid; only their "
                        "co-occurrence is held out — the row-3 request"
                    ),
                },
            )
        )
    return cells


def _consecutive_runs(idx: np.ndarray, width: int) -> list[np.ndarray]:
    """All windows of ``width`` **consecutive grid indices** entirely inside ``idx``."""
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    out: list[np.ndarray] = []
    for start in range(idx.size):
        window = idx[start : start + width]
        if window.size == width and int(window[-1] - window[0]) == width - 1:
            out.append(window)
    return out


def _disjoint_windows(windows: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Greedily keep a pairwise-DISJOINT subset of ``windows`` (first-fit, in order)."""
    kept: list[np.ndarray] = []
    taken: set[int] = set()
    for w in windows:
        s = set(w.tolist())
        if not (s & taken):
            kept.append(w)
            taken |= s
    return kept


def build_nway_cells(
    grid: FactorGrid,
    cfg: ValueHoldoutConfig,
    pair_cells: Sequence[FactorCell] = (),
    montero_widths: Optional[dict] = None,
) -> list[FactorCell]:
    """Row-4 cells: Montero-shaped 6-factor conjunctions over the values that REMAIN.

    Each cell is structurally identical to Montero's held-out cell — a band of ``B``
    consecutive values on each of ``floor_hue`` / ``wall_hue`` / ``object_hue``, a
    corner of ``S`` consecutive ``scale`` values, exactly ONE ``shape``, and a corner of
    ``R`` consecutive ``orientation`` values — so a wave-3 corner is comparable with a
    wave-2 corner.  Only its **placement** moves, and it may not move onto:

    - the held-out hue or the held-out shape (that is the value cells' territory), or
    - any hue reserved by a :func:`build_pair_cells` cell (which is unconstrained on
      the other four factors and would otherwise swallow an n-way cell whole).

    ⭐ **Disjointness is structural, not rejection-sampled.**  The cells are enumerated
    over a product of *pairwise disjoint* coordinate options — disjoint object-hue
    windows x disjoint floor-hue windows x {high, low} scale corner x {high, low}
    orientation corner x one shape — so any two distinct slots differ in at least one
    coordinate whose value-sets do not intersect.  ``assert_cells_disjoint`` proves it
    anyway; it is not asked to *make* it true.
    """
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
        reserved_hues.update(float(v) for v in c.values.get(hue_f, ()))
    allowed_hue_idx = np.array(
        [k for k, v in enumerate(hue_vals) if float(v) not in reserved_hues],
        dtype=np.int64,
    )
    obj_windows = _disjoint_windows(_consecutive_runs(allowed_hue_idx, band_w))
    floor_windows = _disjoint_windows(
        _consecutive_runs(np.arange(floor_vals.size, dtype=np.int64), band_w)
    )
    shapes = _retained_values(grid, shape_f, cfg.held_out_shape_value)
    scale_corners = {
        "high": tuple(float(x) for x in scale_vals[-scale_w:]),
        "low": tuple(float(x) for x in scale_vals[:scale_w]),
    }
    rot_corners = {
        "high": tuple(float(x) for x in rot_vals[-rot_w:]),
        "low": tuple(float(x) for x in rot_vals[:rot_w]),
    }
    if not obj_windows or not floor_windows:
        raise ValueError(
            f"no {band_w}-wide band of retained {hue_f} values survives the reserved "
            f"values {sorted(reserved_hues)} — lower n_pair_cells or the band width"
        )

    capacity = (
        len(shapes) * len(obj_windows) * len(floor_windows) * 2 * 2
    )
    n = int(cfg.n_nway_cells)
    if n < 0:
        raise ValueError(f"n_nway_cells must be >= 0, got {n}")
    if n > capacity:
        raise ValueError(
            f"n_nway_cells={n} exceeds the {capacity} structurally-disjoint slots "
            f"available ({len(shapes)} shapes x {len(obj_windows)} object bands x "
            f"{len(floor_windows)} floor bands x 2 scale corners x 2 pose corners)"
        )

    # ⭐ Odometer order: shape varies fastest, so the first N cells spread across the
    # shapes, the hue ring and both corners rather than piling into one corner.
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
        bg = tuple(float(x) for x in floor_vals[floor_windows[fi]])
        obj = tuple(float(x) for x in hue_vals[obj_windows[oi]])
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
                    "note": (
                        "Montero-shaped: a wave-3 corner is comparable with a wave-2 "
                        "corner because only the PLACEMENT differs"
                    ),
                },
            )
        )
    return cells


def build_value_holdout_cells(
    grid: FactorGrid, cfg: ValueHoldoutConfig
) -> list[FactorCell]:
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
    return split, cells


def assert_value_absent_from_train(
    split: GenSplit, factor: str, value: float, chunk: int = 1 << 20
) -> int:
    """⭐ The value hold-out property itself, checked **directly on the train index**.

    ``assert_holdout_disjoint`` says "no train row is in any cell".  This says the
    thing the taxonomy actually claims — "the store's fit pool contains no row
    carrying this value" — and it says it by reconstructing the train rows' factor
    values through the grid bijection, never by re-reading the mask that built the
    split.  Kept separate so it fires for exactly one reason.

    Returns:
        The number of train rows checked.
    """
    k = split.grid.index_of(factor)
    checked = 0
    for start in range(0, split.n_train, chunk):
        rows = split.train_rows[start : start + chunk]
        vals = split.grid.values_of(rows)[:, k]
        hits = vals == float(value)
        if hits.any():
            i = int(np.argmax(hits))
            raise HoldoutViolation(
                f"{split.dataset}/{split.condition}: {int(hits.sum())} train row(s) "
                f"carry the held-out value {factor}={value}; first is row "
                f"{int(rows[i])} — the store's fit pool would SEE the value"
            )
        checked += int(rows.size)
    return checked


# ---------------------------------------------------------------------------
# the query sets: row 2 (unseen-valid) and row 5 (held-out value)
# ---------------------------------------------------------------------------

ROW2_SCHEMA = {
    "version": 1,
    "description": (
        "Row 2, UNSEEN-VALID: a request whose object_hue is OFF-GRID, strictly between "
        "two RETAINED, grid-adjacent hue values, with the other terms lifted verbatim "
        "from a TRAIN row. In-basin interpolation: both bracketing rows are in train, "
        "which validate_row2 checks. The correct answer is ACCEPT."
    ),
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'unseen_valid'",
        "request": (
            "list of {factor, factor_index, value} terms; every factor named once, one "
            "term OFF-GRID and strictly inside the realised range"
        ),
        "reason": "which two stored values the off-grid term is between",
    },
    "note": (
        "⛔ The pair bracketing the row-5 hole is NOT grid-adjacent and is excluded by "
        "construction, so a row-2 request can never land on the row-5 value."
    ),
}

ROW5_SCHEMA = {
    "version": 1,
    "description": (
        "Row 5, COMPOSITIONAL REFUSAL: a request naming a value that is ON the grid — "
        "the image exists and the frozen decoder renders it and the frozen judge "
        "recognises it — but that the STORE never held. The correct answer is REFUSE."
    ),
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
    """Grid-index pairs ``(k, k+1)`` whose BOTH values are retained.

    ⛔ The pair bracketing the hole (e.g. ``(0.4, 0.6)`` around a hole at ``0.5``) is
    not grid-adjacent and is therefore absent from this list — which is exactly why a
    row-2 midpoint can never coincide with the row-5 value.
    """
    vals = grid.values[grid.index_of(factor)]
    keep = vals != float(held_out)
    return [(k, k + 1) for k in range(vals.size - 1) if keep[k] and keep[k + 1]]


def build_row2_unseen(
    grid: FactorGrid, split: GenSplit, cfg: ValueHoldoutConfig
) -> dict:
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
    # ⭐ Draw without replacement from a shuffled train sample, so a request is never
    # emitted twice and the file is deterministic given (row2_seed, n).
    order = rng.permutation(pool.size)
    attempts = 0
    for t in order:
        if len(records) >= n_want:
            break
        attempts += 1
        row = int(pool[int(t)])
        fi = grid.factor_indices_of(np.array([row]))[0]
        lo, hi = pairs[int(rng.integers(len(pairs)))]
        # both brackets must be STORED with the same other five factors
        brackets = []
        for k in (lo, hi):
            fi_b = fi.copy()
            fi_b[k_hue] = k
            brackets.append(int(grid.rows_of(fi_b[None, :])[0]))
        if not all(in_train[b] for b in brackets):
            continue
        mid = 0.5 * (float(hue_vals[lo]) + float(hue_vals[hi]))
        if np.any(hue_vals == mid):
            raise ValueError(
                f"the midpoint {mid} of ({hue_vals[lo]}, {hue_vals[hi]}) is ON the "
                "grid — it would not be an unseen value"
            )
        vals = grid.values_of(np.array([row]))[0].tolist()
        vals[k_hue] = mid
        records.append(
            {
                "id": f"{grid.dataset}-row2-{len(records):04d}",
                "kind": "unseen_valid",
                "request": _terms(grid, vals),
                "reason": (
                    f"{hue_f} requested at {mid}, strictly between the stored values "
                    f"{float(hue_vals[lo])} and {float(hue_vals[hi])}; both bracketing "
                    f"rows ({brackets[0]}, {brackets[1]}) are in the store's fit pool"
                ),
            }
        )
    if len(records) < n_want:
        raise ValueError(
            f"row 2: only {len(records)} of {n_want} requests could be built from "
            f"{attempts} train rows — the in-basin bracket condition is not satisfiable "
            "at this size"
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


def build_row5_held_out(
    grid: FactorGrid, split: GenSplit, cfg: ValueHoldoutConfig
) -> dict:
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
                f"row 5/{kind}: no held-out row carries {factor}={value} — the value "
                "cell did not do what it claims"
            )
        n = int(cfg.row5_n_per_kind)
        take = np.sort(rng.choice(pool, size=n, replace=pool.size < n))
        for i, row in enumerate(take.tolist()):
            if not test_set[row]:  # pragma: no cover - defended, not expected
                raise ValueError(f"row 5/{kind}: row {row} is not held out")
            rv = grid.values_of(np.array([row]))[0].tolist()
            records.append(
                {
                    "id": f"{grid.dataset}-row5-{kind}-{i:04d}",
                    "kind": kind,
                    "request": _terms(grid, rv),
                    "violated_factor": factor,
                    "reason": (
                        f"{factor}={value} is ON the grid and the frozen decoder renders "
                        "it and the frozen judge recognises it, but the STORE never held "
                        "a row carrying it (see the condition's scope statement)"
                    ),
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
    """Assert every row-2 record is off-grid, in-range, and IN-BASIN; return the count.

    A row-2 set that had drifted on-grid would silently become row 1, and the refusal
    pair would then be scored against a request the store has literally seen.
    """
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
                if not (vals[0] < v < vals[-1]):
                    raise AssertionError(
                        f"{rec['id']}: off-grid value {v} of {term['factor']!r} is "
                        "OUTSIDE the realised range — that is a row-6 record"
                    )
                lo = int(np.searchsorted(vals, v)) - 1
                off.append((k, lo, v))
        if len(off) != 1:
            raise AssertionError(
                f"{rec['id']}: {len(off)} off-grid terms, expected exactly 1"
            )
        k, lo, v = off[0]
        if abs(v - 0.5 * (float(vals[lo]) + float(vals[lo + 1]))) > 0:
            pass  # midpoint is a convention, not a requirement; bracketing is
        fi = np.array(
            [
                int(np.flatnonzero(grid.values[j] == float(req[j]["value"]))[0])
                if j != k
                else 0
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
                    f"{rec['id']}: bracketing row {row} (at {grid.names[k]}="
                    f"{float(grid.values[k][b])}) is NOT in the store's fit pool — the "
                    "request is not in-basin and cannot be scored as unseen-VALID"
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
                    f"{rec['id']}: value {v} of {term['factor']!r} is OFF-GRID — row 5 "
                    "is an ON-GRID value the store never held, not an invalid request"
                )
            vals.append(v)
        row = int(rec["row"])
        if not np.array_equal(grid.values_of(np.array([row]))[0], np.array(vals)):
            raise AssertionError(f"{rec['id']}: request does not denote its own row")
        if not test_set[row]:
            raise AssertionError(
                f"{rec['id']}: row {row} is IN the store's fit pool — the value was "
                "stored and the record is not a refusal item"
            )
        vf = rec["violated_factor"]
        if float(vals[grid.index_of(vf)]) not in (
            float(payload["held_out"][vf]),
        ):
            raise AssertionError(
                f"{rec['id']}: violated_factor {vf!r} does not carry the held-out value"
            )
    return len(records)


# ---------------------------------------------------------------------------
# ⭐ the declared EXTRA diagnostic: does a bare grid look-up decide the row?
# ---------------------------------------------------------------------------


def grid_reach_features(record: dict, grid: FactorGrid) -> dict[str, float]:
    """"Reach": how far outside the grid the request's values sit.

    ⛔ **Reported, never asserted on** — the same status the pin gives
    ``definitional_features``.  This is the statistic that decides ``out_of_range`` with
    no model in it (the pin measures it at AUROC 1.0000 and says so), and the point of
    computing it here is the CONTRAST: it decides row 6's ``out_of_range`` half and it
    does **not** decide row 5, whose values are all on the grid.  A syntactic rule can
    refuse row 6 for free; only the store's own geometry can refuse row 5.
    """
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
    return {
        "n_offgrid_terms": float(n_off),
        "max_distance_outside_range": float(worst),
    }


def grid_reach_auroc(
    invalid_records: Sequence[dict],
    control_records: Sequence[dict],
    grid: FactorGrid,
) -> dict[str, float]:
    """Orientation-free AUROC of :func:`grid_reach_features`.  ⛔ Reported, not asserted."""
    def table(recs):
        cols: dict[str, list[float]] = {}
        for r in recs:
            for k, v in grid_reach_features(r, grid).items():
                cols.setdefault(k, []).append(v)
        return cols

    pos, neg = table(invalid_records), table(control_records)
    return {k: float(max(auroc(pos[k], neg[k]), 1.0 - auroc(pos[k], neg[k]))) for k in sorted(pos)}


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------

#: ⭐ The registry entries for the sets the PIN does not know about.  They are merged
#: into the pin's ``query_sets.json`` (⛔ ``QUERY_SET_REGISTRY`` is in a read-only file
#: and is not edited); ``audit_query_sets`` reads the FILE, so the merge is what makes
#: rows 2/5 reachable by ``assert_no_model_detector_at_chance``.
EXTRA_QUERY_SETS: dict[str, dict] = {
    "row2_unseen": {
        "role": "valid",
        "file": "row2_unseen.json",
        "builder": "chlu.data.gen_value_holdout.build_row2_unseen",
        "n_terms": "n_factors",
        "description": (
            "row 2 (unseen-valid): an OFF-GRID hue strictly between two RETAINED, "
            "grid-adjacent stored values. The correct answer is ACCEPT, which is what "
            "makes it the control the refusal rows are scored against."
        ),
    },
    "row5_held_out": {
        "role": "invalid",
        "file": "row5_held_out.json",
        "builder": "chlu.data.gen_value_holdout.build_row5_held_out",
        "n_terms": "n_factors",
        # ⭐ Both kinds are controlled by row 2: that pairing IS the claim — row 2
        # accept vs row 5 refuse, from requests that are structurally identical.
        "controls": {
            "held_out_hue": "row2_unseen",
            "held_out_shape": "row2_unseen",
        },
        "description": (
            "row 5 (compositional refusal): an ON-GRID value the STORE never held. "
            "⛔ 'invalid' here means 'the correct answer is refuse', NOT 'unsatisfiable' "
            "— the image exists and the frozen judge recognises it."
        ),
    },
}

#: ⭐ Row 6's control map, overridden onto the pin's ``c_invalid`` entry.  The
#: ``out_of_range`` half is scored against ROW 2 (both 6 terms) because row 2 is the
#: accept side of the refusal pair.  ⛔ ``duplicate_factor`` keeps ``a7_redundant``:
#: it carries ``n_factors + 1`` terms and a 6-term control would hand back the exact
#: term-count giveaway R26 ruling 1 killed.
ROW6_CONTROLS = {"duplicate_factor": "a7_redundant", "out_of_range": "row2_unseen"}


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
        "config": cfg.to_dict(),
        "held_out_values": {
            cfg.held_out_hue_factor: float(cfg.held_out_hue_value),
            cfg.held_out_shape_factor: float(cfg.held_out_shape_value),
        },
        "rationale": {
            "hue": HELD_OUT_HUE_RATIONALE,
            "shape": HELD_OUT_SHAPE_RATIONALE,
        },
        "n_total": split.n_total,
        "n_train_realised": split.n_train,
        "n_test_realised": split.n_test,
        "rows_by_family": {
            k: int(sum(c.rows(split.grid).size for c in v)) for k, v in by_family.items()
        },
        "n_cells_by_family": {k: len(v) for k, v in by_family.items()},
        "cells": [c.to_dict(split.grid) for c in cells],
        "frozen_artifact_warning": (
            "⛔ The encoder/decoder/judge frozen under `recomb2element` were fitted on "
            "rows this condition holds out. That is DELIBERATE and is the scope "
            "statement above: the hold-out is from the STORE's fit pool only. ⛔ Do not "
            "quote a row from this condition as an end-to-end 'never seen' result."
        ),
    }


def emit_value_holdout(
    grid: FactorGrid,
    out_dir: str | Path,
    cfg: Optional[ValueHoldoutConfig] = None,
    run_detector: bool = True,
) -> dict:
    """⭐ Build the condition, run **every** proof the pin provides, and write it out.

    Emits, into ``out_dir``: ``factor_index.npz`` (the split index),
    ``value_cells.json`` (the metadata, carrying the scope statement),
    ``c_invalid.json`` (row 6), ``a7_redundant.json`` (row 6's term-count control),
    ``row2_unseen.json`` (row 2), ``row5_held_out.json`` (row 5) and
    ``query_sets.json`` (the registry the detector audit discovers).

    Returns a report: every checksum, every realised count, every assertion's returned
    count, and the no-model detector's per-feature AUROC table — ⛔ returned whether it
    passes or fails, with ``report["detector"]["violations"]`` non-empty iff it failed.
    """
    cfg = cfg or ValueHoldoutConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split, cells = build_value_holdout_split(grid, cfg)

    # -- the four pinned proofs, each recording its RETURNED count ------------
    assertions = {
        "assert_non_empty_test": assert_non_empty_test(split),
        "assert_index_partitions_grid": assert_index_partitions_grid(split),
        "assert_cells_disjoint": assert_cells_disjoint(grid, cells),
        "assert_holdout_disjoint": assert_holdout_disjoint(split),
        # ⭐ the value-hold-out property itself, checked directly (see the docstring)
        f"assert_value_absent_from_train[{cfg.held_out_hue_factor}]":
            assert_value_absent_from_train(
                split, cfg.held_out_hue_factor, cfg.held_out_hue_value
            ),
        f"assert_value_absent_from_train[{cfg.held_out_shape_factor}]":
            assert_value_absent_from_train(
                split, cfg.held_out_shape_factor, cfg.held_out_shape_value
            ),
    }
    if assertions["assert_holdout_disjoint"] != split.n_train:
        raise HoldoutViolation(
            f"assert_holdout_disjoint checked {assertions['assert_holdout_disjoint']} "
            f"rows but the split has {split.n_train} train rows — the check was partial"
        )

    index_path, index_sha = save_index(
        split, out_dir / "factor_index.npz", extra={"scope": VALUE_HOLDOUT_SCOPE}
    )
    meta = value_holdout_metadata(split, cells, cfg)
    meta_path, meta_sha = write_json_with_checksum(meta, out_dir / "value_cells.json")

    # -- row 6 (the pin's C set) and its mandatory term-count control ---------
    c_set = build_c_invalid(
        grid, n_per_kind=int(cfg.c_invalid_n_per_kind), seed=int(cfg.c_invalid_seed)
    )
    n_c = validate_c_invalid(c_set, grid)
    c_path, c_sha = write_json_with_checksum(c_set, out_dir / "c_invalid.json")

    a7_rows = select_a7_rows(
        split.train_rows, int(cfg.a7_n_requests), seed=int(cfg.a7_seed)
    )
    a7_set = build_a7_redundant(grid, a7_rows, seed=int(cfg.a7_seed))
    n_a7 = validate_a7_redundant(a7_set, grid)
    a7_path, a7_sha = write_json_with_checksum(a7_set, out_dir / "a7_redundant.json")

    # -- rows 2 and 5 --------------------------------------------------------
    row2 = build_row2_unseen(grid, split, cfg)
    n_row2 = validate_row2(row2, grid, split)
    row2_path, row2_sha = write_json_with_checksum(row2, out_dir / "row2_unseen.json")

    row5 = build_row5_held_out(grid, split, cfg)
    n_row5 = validate_row5(row5, grid, split)
    row5_path, row5_sha = write_json_with_checksum(row5, out_dir / "row5_held_out.json")

    # -- the registry: the pin's three sets + ours, with row 6 re-controlled --
    emitted = {
        "base": {
            "derived_from_sha256": index_sha,
            "n_records": int(split.n_train + split.n_test),
        },
        "c_invalid": {"sha256": c_sha, "n_records": int(n_c)},
        "a7_redundant": {"sha256": a7_sha, "n_records": int(n_a7)},
    }
    reg_path, _ = write_query_set_registry(out_dir, split.dataset, emitted)
    payload = json.loads(Path(reg_path).read_text())
    payload["condition"] = VALUE_HOLDOUT_CONDITION
    payload["scope"] = VALUE_HOLDOUT_SCOPE
    payload["sets"]["c_invalid"]["controls"] = dict(ROW6_CONTROLS)
    payload["sets"]["c_invalid"]["control_note"] = (
        "⭐ out_of_range is re-controlled onto ROW 2 (the accept side of the refusal "
        "pair); duplicate_factor keeps a7_redundant because it carries n_factors+1 "
        "terms and a 6-term control would restore the R26.1 term-count giveaway."
    )
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
        "artifacts": {
            "factor_index.npz": {"path": str(index_path), "sha256": index_sha},
            "value_cells.json": {"path": str(meta_path), "sha256": meta_sha},
            "c_invalid.json": {
                "path": str(c_path), "sha256": c_sha, "n_records": int(n_c)
            },
            "a7_redundant.json": {
                "path": str(a7_path), "sha256": a7_sha, "n_records": int(n_a7)
            },
            "row2_unseen.json": {
                "path": str(row2_path), "sha256": row2_sha, "n_records": int(n_row2)
            },
            "row5_held_out.json": {
                "path": str(row5_path), "sha256": row5_sha, "n_records": int(n_row5)
            },
            "query_sets.json": {"path": str(reg_path), "sha256": reg_sha},
        },
    }

    if run_detector:
        # ⛔ audit_query_sets does NOT raise — the numbers are reported either way,
        # which is the task's instruction.  The assert_* wrapper is called after, so a
        # failure still stops the emitter, but only once the table is in the report.
        report["detector"] = audit_query_sets(
            reg_path, tolerance=float(cfg.detector_tolerance)
        )
        report["grid_reach_diagnostic"] = {
            "note": (
                "⛔ REPORTED, NEVER ASSERTED. A bare grid look-up decides row 6's "
                "out_of_range half (the pin measures the same statistic at 1.0000 and "
                "declares it); it decides NOTHING on row 5, whose values are all "
                "on-grid. That asymmetry is why row 5 is the load-bearing refusal row."
            ),
            "row5_vs_row2": grid_reach_auroc(
                row5["records"], row2["records"], grid
            ),
            "row6_out_of_range_vs_row2": grid_reach_auroc(
                [r for r in c_set["records"] if r["kind"] == "out_of_range"],
                row2["records"],
                grid,
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
    """``python -m chlu.data.gen_value_holdout`` — build and emit the condition.

    ⛔ A separate entry point on purpose: adding a branch to ``chlu data shapes3d``
    would mean editing the CLI and the config block, and this spoke is additive-only.
    """
    p = argparse.ArgumentParser(description="Emit the value-hold-out condition.")
    p.add_argument("--data-root", default=None, help="dataset root (default: the pin's)")
    p.add_argument("--out-dir", default=None, help="default: <root>/value_holdout")
    p.add_argument("--report", default=None, help="write the JSON report here")
    p.add_argument("--n-nway-cells", type=int, default=None)
    p.add_argument("--n-pair-cells", type=int, default=None)
    p.add_argument("--no-detector", action="store_true")
    args = p.parse_args(argv)

    from . import shapes3d as s3d  # noqa: PLC0415 - keeps h5py off the import path

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
        # ⭐ The NAMED instrument is run on the shipped artifact — ⛔ but only after the
        # table has been printed: §9 requires the per-feature AUROCs to reach the Head
        # whether they pass or fail, and an exception before the print would eat them.
        try:
            assert_no_model_detector_at_chance(
                out_dir / "query_sets.json", tolerance=float(cfg.detector_tolerance)
            )
        except AssertionError as exc:
            print(str(exc))
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
