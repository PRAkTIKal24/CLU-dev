"""Exhaustive factor grids, the Montero recombination-to-element splits, and the
hold-out assertion that guards them (generative B cell).

This module is **dataset-agnostic**: it holds the factor-grid abstraction, the
split container, the hold-out assertion, and the structurally-invalid ("C") set
builder.  The two concrete datasets live in :mod:`chlu.data.dsprites` and
:mod:`chlu.data.shapes3d`, which supply the raw factor arrays and the verbatim
Montero filter.

Why the assertion is the load-bearing part
------------------------------------------
Every downstream generative claim reads "this factor combination was never seen
in training".  If :func:`assert_holdout_disjoint` is vacuous — wrong axis order,
wrong array compared, empty held-out set — it passes silently and every number
built on it is void.  So the assertion here is deliberately built to be
*capable of failing*:

- it reconstructs each train row's factor values **from the grid via the
  row<->factor bijection**, rather than re-reading the cached boolean mask that
  produced the split (re-checking a mask against itself proves nothing);
- it re-evaluates the dataset's own test predicate on those reconstructed
  values;
- it raises :class:`HoldoutViolation` on any hit and returns the number of
  combinations actually checked, so a silently empty check is visible.

``tests/test_gen_data_splits.py`` exercises the positive control: inject a
held-out row into the train index and show the assertion fires.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FactorCell",
    "FactorGrid",
    "GenSplit",
    "HoldoutViolation",
    "assert_cells_disjoint",
    "assert_holdout_disjoint",
    "assert_index_partitions_grid",
    "build_c_invalid",
    "build_cell_split",
    "cells_test_predicate",
    "default_gen_data_root",
    "file_sha256",
    "write_json_with_checksum",
]


class HoldoutViolation(AssertionError):
    """A train row satisfies the held-out (test) predicate."""


def default_gen_data_root() -> Path:
    """Root for the generative datasets: ``$CHLU_GEN_DATA_ROOT`` else ``<repo>/.claude/data/gen``.

    The datasets are large and **never** enter git; ``.claude/`` is gitignored,
    which is why it is the default home.  Set ``CHLU_GEN_DATA_ROOT`` when
    running from a worktree (which has no ``.claude/``) or on a cluster.
    """
    env = os.environ.get("CHLU_GEN_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / ".claude" / "data" / "gen"


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    """sha256 of a file, streamed (the datasets are 19 MB - 268 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_json_with_checksum(payload: dict, path: str | Path) -> tuple[Path, str]:
    """Write ``payload`` as sorted, newline-terminated JSON; return (path, sha256).

    Sorted keys + fixed separators make the checksum reproducible across runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return path, hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# factor grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorGrid:
    """An exhaustive product grid of generative factors + the row bijection.

    Both dSprites and Shapes3D enumerate **every** factor combination exactly
    once, in C (row-major) order over the factor axes, so a factor-index tuple
    and a dataset row index are in bijection.  ``values[k]`` is the sorted
    array of realised values for factor ``k`` — taken from the *file*, never
    regenerated with ``linspace``, because the split thresholds sit on grid
    values and the realised float64 decides membership (see the dSprites
    rotation boundary in :mod:`chlu.data.dsprites`).

    Attributes:
        names: ordered factor names.
        values: per-factor sorted realised value arrays (float64).
        dataset: dataset id, for provenance.
    """

    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    dataset: str = ""

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"{len(self.names)} factor names for {len(self.values)} value arrays"
            )
        for name, vals in zip(self.names, self.values, strict=True):
            if vals.ndim != 1 or vals.size == 0:
                raise ValueError(
                    f"factor {name!r}: values must be a non-empty 1-D array"
                )
            if not np.all(np.diff(vals) > 0):
                raise ValueError(f"factor {name!r}: values must be strictly increasing")

    # -- shape ---------------------------------------------------------------
    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(int(v.size) for v in self.values)

    @property
    def n_factors(self) -> int:
        return len(self.names)

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.cardinalities))

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    # -- bijection -----------------------------------------------------------
    def rows_of(self, factor_indices: np.ndarray) -> np.ndarray:
        """(N, F) factor-index tuples -> (N,) dataset row indices."""
        fi = np.atleast_2d(np.asarray(factor_indices, dtype=np.int64))
        return np.ravel_multi_index(tuple(fi.T), self.cardinalities).astype(np.int64)

    def factor_indices_of(self, rows: np.ndarray) -> np.ndarray:
        """(N,) dataset row indices -> (N, F) factor-index tuples."""
        rows = np.asarray(rows, dtype=np.int64)
        return np.stack(np.unravel_index(rows, self.cardinalities), axis=-1)

    def values_of(self, rows: np.ndarray) -> np.ndarray:
        """(N,) row indices -> (N, F) realised factor **values**.

        This is the reconstruction path used by the hold-out assertion: it goes
        row -> factor indices -> grid values, and never touches the cached
        latent array, so a corrupted train index cannot smuggle a held-out
        combination past the check.
        """
        fi = self.factor_indices_of(rows)
        return np.stack(
            [self.values[k][fi[:, k]] for k in range(self.n_factors)], axis=-1
        )

    # -- verification --------------------------------------------------------
    def verify_bijection(
        self, latent_values: np.ndarray, n_check: int | None = None
    ) -> int:
        """Assert ``values_of(row) == latent_values[row]`` exactly; return rows checked.

        This is what catches an off-by-one in the axis order: if the grid axes
        were permuted, the reconstructed values would disagree with the file for
        almost every row.  Exact float equality is correct here — the grid
        values are lifted verbatim from the same array.
        """
        latent_values = np.asarray(latent_values)
        if latent_values.shape[1] != self.n_factors:
            raise ValueError(
                f"latent array has {latent_values.shape[1]} columns, grid has "
                f"{self.n_factors} factors"
            )
        rows = (
            np.arange(len(latent_values))
            if n_check is None
            else np.linspace(0, len(latent_values) - 1, n_check).astype(np.int64)
        )
        recon = self.values_of(rows)
        if not np.array_equal(recon, latent_values[rows]):
            bad = int(np.argmax(np.any(recon != latent_values[rows], axis=1)))
            raise AssertionError(
                f"factor-grid bijection broken at row {int(rows[bad])}: "
                f"grid says {recon[bad]}, file says {latent_values[rows][bad]} "
                "(axis order or grid ordering is wrong)"
            )
        return int(rows.size)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "names": list(self.names),
            "cardinalities": list(self.cardinalities),
            "n_rows": self.n_rows,
            "values": [v.tolist() for v in self.values],
        }


# ---------------------------------------------------------------------------
# split container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenSplit:
    """A deterministic Montero split over an exhaustive factor grid.

    The Montero splits carry **no seed and no sampling**: train is the exact
    boolean complement of test over the full grid.  ``test_predicate`` is the
    dataset's verbatim filter, re-callable on reconstructed factor values —
    that is what makes the hold-out assertion independent of ``test_mask``.
    """

    dataset: str
    condition: str
    variant: str
    grid: FactorGrid
    train_rows: np.ndarray
    test_rows: np.ndarray
    test_predicate: Callable[[np.ndarray], np.ndarray]

    @property
    def n_total(self) -> int:
        return self.grid.n_rows

    @property
    def n_train(self) -> int:
        return int(self.train_rows.size)

    @property
    def n_test(self) -> int:
        return int(self.test_rows.size)

    def with_train_rows(self, train_rows: np.ndarray) -> "GenSplit":
        """Copy with a different train index (used by the positive control)."""
        return GenSplit(
            dataset=self.dataset,
            condition=self.condition,
            variant=self.variant,
            grid=self.grid,
            train_rows=np.asarray(train_rows, dtype=np.int64),
            test_rows=self.test_rows,
            test_predicate=self.test_predicate,
        )

    def summary(self) -> dict:
        return {
            "dataset": self.dataset,
            "condition": self.condition,
            "variant": self.variant,
            "n_total": self.n_total,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def assert_non_empty_test(split: GenSplit, expected: int | None = None) -> int:
    """Assert the held-out set is non-empty (and of size ``expected`` if given).

    ⚠ Runs **before** any other check.  The dSprites filter applied to the raw
    6-column npz (i.e. forgetting the colour-column slice) selects *nothing*;
    that failure mode is silent and would make every "unseen combination"
    trivially satisfied.
    """
    if split.n_test == 0:
        raise HoldoutViolation(
            f"{split.dataset}/{split.condition}: held-out set is EMPTY — the filter "
            "matched no rows (check the factor column offset)"
        )
    if expected is not None and split.n_test != expected:
        raise HoldoutViolation(
            f"{split.dataset}/{split.condition}: held-out set has {split.n_test} "
            f"rows, expected {expected}"
        )
    return split.n_test


def assert_holdout_disjoint(split: GenSplit, chunk: int = 1 << 20) -> int:
    """Assert **no** train row satisfies the held-out predicate; return rows checked.

    Reconstructs each train row's factor values from the grid bijection and
    re-evaluates the dataset's own test predicate on them.  Raises
    :class:`HoldoutViolation` naming the offending row and its factor tuple.

    Returns:
        Number of train combinations checked (0 would itself be a red flag —
        callers should compare it against ``split.n_train``).
    """
    checked = 0
    for start in range(0, split.n_train, chunk):
        rows = split.train_rows[start : start + chunk]
        hits = np.asarray(split.test_predicate(split.grid.values_of(rows)))
        if hits.any():
            i = int(np.argmax(hits))
            row = int(rows[i])
            fi = split.grid.factor_indices_of(np.array([row]))[0]
            vals = split.grid.values_of(np.array([row]))[0]
            raise HoldoutViolation(
                f"{split.dataset}/{split.condition}: {int(hits.sum())} train row(s) "
                f"satisfy the held-out predicate; first is row {row} "
                f"(factor indices {fi.tolist()}, values {vals.tolist()})"
            )
        checked += int(rows.size)
    return checked


def assert_index_partitions_grid(split: GenSplit) -> int:
    """Assert train and test are a *partition* of the grid; return the grid size.

    Complements :func:`assert_holdout_disjoint`, which only says "no train row
    is held out".  This says "and nothing was lost or double-counted": the two
    index arrays must be duplicate-free, disjoint, and together cover every row
    exactly once.  Kept separate so each check fires for exactly one reason.
    """
    train, test = split.train_rows, split.test_rows
    union = np.concatenate([train, test])
    n = split.grid.n_rows
    if union.size != n or np.unique(union).size != n:
        raise HoldoutViolation(
            f"{split.dataset}: train({train.size}) + test({test.size}) do not "
            f"partition the {n}-cell grid (unique rows: {np.unique(union).size})"
        )
    return n


# ---------------------------------------------------------------------------
# held-out CELLS: the widened secondary condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorCell:
    """One held-out **cell**: a conjunction over factors, an allowed value-set each.

    Montero's held-out set is exactly one such conjunction (three hue bands x a
    scale corner x one shape x an orientation corner).  A widened *secondary*
    condition is a **union of several cells of the same kind**, which is what
    keeps the two conditions comparable: the wide split is not "more rows",
    it is "more cells of Montero's shape, placed elsewhere in factor space".

    ⛔ The values must be **grid values, verbatim** — membership is decided by
    exact float equality, the same convention :meth:`FactorGrid.verify_bijection`
    uses, because both sides are lifted from the same array.

    Attributes:
        cell_id: stable id, unique within a condition.
        values: factor name -> allowed values.  A factor that is absent is
            unconstrained (all of its values are held out).
        meta: measured / descriptive labels (e.g. a photometric contrast label).
            Carried into the emitted metadata verbatim; never read by the split.
    """

    cell_id: str
    values: dict[str, tuple[float, ...]]
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            object.__setattr__(self, "meta", {})

    def factor_index_sets(self, grid: FactorGrid) -> list[np.ndarray]:
        """Per-factor arrays of allowed **grid indices** (all of them if unconstrained)."""
        out: list[np.ndarray] = []
        for k, name in enumerate(grid.names):
            allowed = self.values.get(name)
            if allowed is None:
                out.append(np.arange(grid.cardinalities[k], dtype=np.int64))
                continue
            allowed_arr = np.asarray(allowed, dtype=float)
            sel = np.flatnonzero(np.isin(grid.values[k], allowed_arr))
            if sel.size != np.unique(allowed_arr).size:
                missing = sorted(
                    set(allowed_arr.tolist()) - set(grid.values[k].tolist())
                )
                raise ValueError(
                    f"cell {self.cell_id!r}: factor {name!r} value(s) {missing} are "
                    "not on the grid"
                )
            out.append(sel.astype(np.int64))
        return out

    def mask(self, grid: FactorGrid, factor_values: np.ndarray) -> np.ndarray:
        """(N, F) factor **values** -> (N,) boolean membership of this cell."""
        vals = np.asarray(factor_values)
        if vals.ndim != 2 or vals.shape[1] != grid.n_factors:
            raise ValueError(
                f"cell {self.cell_id!r}: expected (N, {grid.n_factors}) values, "
                f"got {vals.shape}"
            )
        hit = np.ones(len(vals), dtype=bool)
        for name, allowed in self.values.items():
            k = grid.index_of(name)
            hit &= np.isin(vals[:, k], np.asarray(allowed, dtype=float))
        return hit

    def rows(self, grid: FactorGrid) -> np.ndarray:
        """Sorted dataset row indices covered by this cell (via the bijection)."""
        idx = self.factor_index_sets(grid)
        mesh = np.meshgrid(*idx, indexing="ij")
        fi = np.stack([m.ravel() for m in mesh], axis=-1)
        return np.sort(grid.rows_of(fi))

    @property
    def n_rows_nominal(self) -> int:
        """Product of the constrained set sizes — the *predicted* count.

        ⛔ Never substituted for the realised count: callers report
        ``rows(grid).size``, which is what the grid actually yields.
        """
        return int(np.prod([len(set(v)) for v in self.values.values()]))

    def to_dict(self, grid: FactorGrid | None = None) -> dict:
        d = {
            "cell_id": self.cell_id,
            "values": {k: [float(x) for x in v] for k, v in self.values.items()},
            "meta": self.meta,
        }
        if grid is not None:
            d["n_rows_realised"] = int(self.rows(grid).size)
        return d


def cells_test_predicate(
    grid: FactorGrid, cells: Sequence[FactorCell]
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the held-out predicate of a multi-cell condition: the OR of the cells.

    Returned as a plain callable over reconstructed factor **values**, so
    :func:`assert_holdout_disjoint` treats a wide condition exactly as it treats
    Montero's: re-evaluate the predicate on values rebuilt from the bijection,
    never re-read the mask that produced the split.
    """
    if not cells:
        raise ValueError("a condition needs at least one cell")

    def predicate(factor_values: np.ndarray) -> np.ndarray:
        vals = np.asarray(factor_values)
        hit = np.zeros(len(vals), dtype=bool)
        for cell in cells:
            hit |= cell.mask(grid, vals)
        return hit

    return predicate


def assert_cells_disjoint(grid: FactorGrid, cells: Sequence[FactorCell]) -> int:
    """Assert no two cells share a row; return the pooled held-out row count.

    Overlapping cells would double-count rows in the per-cell table while the
    pooled count silently disagreed — so this fires rather than de-duplicating.
    """
    seen: dict[int, str] = {}
    for cell in cells:
        for row in cell.rows(grid).tolist():
            if row in seen:
                raise HoldoutViolation(
                    f"cells {seen[row]!r} and {cell.cell_id!r} both contain row "
                    f"{row} — the per-cell counts would double-count it"
                )
            seen[row] = cell.cell_id
    return len(seen)


def build_cell_split(
    grid: FactorGrid,
    cells: Sequence[FactorCell],
    dataset: str,
    condition: str,
    variant: str,
) -> GenSplit:
    """Split the grid on a union of :class:`FactorCell`s (train = exact complement)."""
    assert_cells_disjoint(grid, cells)
    test_rows = np.sort(
        np.concatenate([cell.rows(grid) for cell in cells]).astype(np.int64)
    )
    all_rows = np.arange(grid.n_rows, dtype=np.int64)
    train_rows = np.setdiff1d(all_rows, test_rows, assume_unique=True)
    return GenSplit(
        dataset=dataset,
        condition=condition,
        variant=variant,
        grid=grid,
        train_rows=train_rows,
        test_rows=test_rows,
        test_predicate=cells_test_predicate(grid, cells),
    )


# ---------------------------------------------------------------------------
# the C set: structurally invalid factor requests
# ---------------------------------------------------------------------------

C_INVALID_SCHEMA = {
    "version": 1,
    "description": (
        "Structurally invalid factor requests, defined in FACTOR SPACE ONLY. "
        "Identical for every arm: no well, mode, cluster or any other "
        "model-internal structure is referenced, so arms without such structure "
        "consume the same file."
    ),
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'duplicate_factor' | 'out_of_range'",
        "request": (
            "list of {factor, factor_index, value} terms; a well-formed request "
            "names each factor at most once with a value drawn from that "
            "factor's grid"
        ),
        "violated_factor": "name of the factor whose constraint is violated",
        "reason": "human-readable statement of why the request cannot be satisfied",
    },
    "kinds": {
        "duplicate_factor": (
            "two different values of the SAME factor requested simultaneously — "
            "unsatisfiable because each image has exactly one value per factor"
        ),
        "out_of_range": (
            "a value outside that factor's realised range — unsatisfiable "
            "because the grid is exhaustive and closed"
        ),
    },
}


def build_c_invalid(
    grid: FactorGrid,
    n_per_kind: int = 16,
    seed: int = 0,
) -> dict:
    """Build the C set of structurally invalid factor requests from the grid alone.

    (a) ``duplicate_factor``: two different values of the same factor at once.
    (b) ``out_of_range``: a value strictly outside that factor's realised range.

    Deterministic given ``(grid, n_per_kind, seed)``.  Requests are emitted as
    complete factor assignments (one value per factor) plus the offending extra
    or replaced term, so an arm can feed them to a conditional generator without
    knowing anything about the model.
    """
    rng = np.random.default_rng(seed)
    n_f = grid.n_factors
    records: list[dict] = []

    def _base_request(rng: np.random.Generator) -> list[dict]:
        return [
            {
                "factor": grid.names[k],
                "factor_index": k,
                "value": float(grid.values[k][rng.integers(grid.cardinalities[k])]),
            }
            for k in range(n_f)
        ]

    # (a) duplicate factor: append a second, different value for one factor.
    #     Only factors with cardinality >= 2 can host this violation.
    eligible = [k for k in range(n_f) if grid.cardinalities[k] >= 2]
    if not eligible:
        raise ValueError("no factor has >= 2 values; cannot build duplicate requests")
    for i in range(n_per_kind):
        k = eligible[i % len(eligible)]
        req = _base_request(rng)
        v0 = req[k]["value"]
        others = grid.values[k][grid.values[k] != v0]
        v1 = float(others[rng.integers(others.size)])
        req.append({"factor": grid.names[k], "factor_index": k, "value": v1})
        records.append(
            {
                "id": f"{grid.dataset}-dup-{i:03d}",
                "kind": "duplicate_factor",
                "request": req,
                "violated_factor": grid.names[k],
                "reason": (
                    f"factor {grid.names[k]!r} is requested at both {v0} and {v1}; "
                    "an image has exactly one value per factor"
                ),
            }
        )

    # (b) out of range: replace one term with a value strictly outside the grid.
    for i in range(n_per_kind):
        k = i % n_f
        vals = grid.values[k]
        step = float(np.min(np.diff(vals))) if vals.size >= 2 else 1.0
        below = bool(i % 2)
        bad = (
            float(vals[0] - step * (1 + rng.integers(1, 4)))
            if below
            else float(vals[-1] + step * (1 + rng.integers(1, 4)))
        )
        req = _base_request(rng)
        req[k]["value"] = bad
        records.append(
            {
                "id": f"{grid.dataset}-oor-{i:03d}",
                "kind": "out_of_range",
                "request": req,
                "violated_factor": grid.names[k],
                "reason": (
                    f"factor {grid.names[k]!r} requested at {bad}, outside its "
                    f"realised range [{float(vals[0])}, {float(vals[-1])}]"
                ),
            }
        )

    return {
        "schema": C_INVALID_SCHEMA,
        "dataset": grid.dataset,
        "seed": int(seed),
        "n_per_kind": int(n_per_kind),
        "factors": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "ranges": [[float(v[0]), float(v[-1])] for v in grid.values],
        },
        "records": records,
    }


def validate_c_invalid(payload: dict, grid: FactorGrid) -> int:
    """Assert every C record really is structurally invalid; return record count.

    A C set that accidentally contained a *satisfiable* request would be as
    vacuous as a hold-out assertion that cannot fire, so this is checked rather
    than assumed.
    """
    records = payload["records"]
    if not records:
        raise AssertionError("C set is empty")
    for rec in records:
        req = rec["request"]
        seen: dict[str, set] = {}
        for term in req:
            seen.setdefault(term["factor"], set()).add(term["value"])
        dup = any(len(v) > 1 for v in seen.values())
        oor = False
        for term in req:
            k = int(term["factor_index"])
            if not np.any(grid.values[k] == term["value"]):
                oor = True
        if rec["kind"] == "duplicate_factor" and not dup:
            raise AssertionError(f"{rec['id']}: claims duplicate_factor but has none")
        if rec["kind"] == "out_of_range" and not oor:
            raise AssertionError(
                f"{rec['id']}: claims out_of_range but all values are on-grid"
            )
        if not (dup or oor):
            raise AssertionError(f"{rec['id']}: request is satisfiable — not a C item")
    return len(records)


# ---------------------------------------------------------------------------
# index persistence
# ---------------------------------------------------------------------------


def save_index(
    split: GenSplit,
    path: str | Path,
    extra: dict | None = None,
) -> tuple[Path, str]:
    """Persist the derived factor-grid index + split as ``.npz``; return (path, sha256).

    The index is a *derived* artifact (rebuildable from the raw file), but its
    checksum is recorded so a stale or hand-edited index is detectable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_rows": split.train_rows,
        "test_rows": split.test_rows,
        "cardinalities": np.asarray(split.grid.cardinalities, dtype=np.int64),
        "factor_names": np.asarray(split.grid.names),
        "meta": np.asarray(
            json.dumps({**split.summary(), **(extra or {})}, sort_keys=True)
        ),
    }
    for k, name in enumerate(split.grid.names):
        payload[f"values__{name}"] = split.grid.values[k]
    # deterministic bytes: np.savez writes members in insertion order
    with open(path, "wb") as f:
        np.savez(f, **payload)
    return path, file_sha256(path)


def load_index(path: str | Path) -> tuple[FactorGrid, np.ndarray, np.ndarray, dict]:
    """Inverse of :func:`save_index` -> (grid, train_rows, test_rows, meta)."""
    with np.load(path, allow_pickle=False) as z:
        names = tuple(str(n) for n in z["factor_names"])
        meta = json.loads(str(z["meta"]))
        grid = FactorGrid(
            names=names,
            values=tuple(z[f"values__{n}"] for n in names),
            dataset=meta.get("dataset", ""),
        )
        return grid, z["train_rows"], z["test_rows"], meta


def _quick_subsample(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic subsample used by ``--quick`` (sorted, so order is stable)."""
    if n <= 0 or rows.size <= n:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=n, replace=False))


def factor_grid_from_latents(
    latent_values: np.ndarray,
    names: Sequence[str],
    dataset: str,
) -> FactorGrid:
    """Build the grid by taking the **realised** unique values of each column.

    ⛔ Deliberately not ``linspace``: for dSprites the rotation thresholds land
    on grid values and the float64 round-off in the stored array — not the
    nominal value — decides membership.
    """
    latent_values = np.asarray(latent_values)
    if latent_values.shape[1] != len(names):
        raise ValueError(
            f"{latent_values.shape[1]} columns for {len(names)} factor names"
        )
    values = tuple(np.unique(latent_values[:, k]) for k in range(len(names)))
    grid = FactorGrid(names=tuple(names), values=values, dataset=dataset)
    if grid.n_rows != len(latent_values):
        raise ValueError(
            f"{dataset}: factor grid has {grid.n_rows} cells but the file has "
            f"{len(latent_values)} rows — the grid is not exhaustive"
        )
    return grid
