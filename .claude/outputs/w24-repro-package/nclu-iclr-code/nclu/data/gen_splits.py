"""gen splits."""

from __future__ import annotations
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np


class HoldoutViolation(AssertionError):
    """A train row satisfies the held-out (test) predicate."""


def default_gen_data_root() -> Path:
    """Dataset root (``NCLU_DATA_ROOT``, default ``<package>/data``)."""
    from nclu.paths import data_root

    return data_root()


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
    return (path, hashlib.sha256(text.encode()).hexdigest())


@dataclass(frozen=True)
class FactorGrid:
    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    dataset: str = ""

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(f"{len(self.names)} factor names for {len(self.values)} value arrays")
        for name, vals in zip(self.names, self.values, strict=True):
            if vals.ndim != 1 or vals.size == 0:
                raise ValueError(f"factor {name!r}: values must be a non-empty 1-D array")
            if not np.all(np.diff(vals) > 0):
                raise ValueError(f"factor {name!r}: values must be strictly increasing")

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple((int(v.size) for v in self.values))

    @property
    def n_factors(self) -> int:
        return len(self.names)

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.cardinalities))

    def index_of(self, name: str) -> int:
        return self.names.index(name)

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
        return np.stack([self.values[k][fi[:, k]] for k in range(self.n_factors)], axis=-1)

    def verify_bijection(self, latent_values: np.ndarray, n_check: int | None = None) -> int:
        """Assert ``values_of(row) == latent_values[row]`` exactly; return rows checked.

        This is what catches an off-by-one in the axis order: if the grid axes
        were permuted, the reconstructed values would disagree with the file for
        almost every row.  Exact float equality is correct here — the grid
        values are lifted verbatim from the same array.
        """
        latent_values = np.asarray(latent_values)
        if latent_values.shape[1] != self.n_factors:
            raise ValueError(
                f"latent array has {latent_values.shape[1]} columns, grid has {self.n_factors} factors"
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
                f"factor-grid bijection broken at row {int(rows[bad])}: grid says {recon[bad]}, file says {latent_values[rows][bad]} (axis order or grid ordering is wrong)"
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
    if split.n_test == 0:
        raise HoldoutViolation(
            f"{split.dataset}/{split.condition}: held-out set is EMPTY — the filter matched no rows (check the factor column offset)"
        )
    if expected is not None and split.n_test != expected:
        raise HoldoutViolation(
            f"{split.dataset}/{split.condition}: held-out set has {split.n_test} rows, expected {expected}"
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
                f"{split.dataset}/{split.condition}: {int(hits.sum())} train row(s) satisfy the held-out predicate; first is row {row} (factor indices {fi.tolist()}, values {vals.tolist()})"
            )
        checked += int(rows.size)
    return checked


def assert_index_partitions_grid(split: GenSplit) -> int:
    train, test = (split.train_rows, split.test_rows)
    union = np.concatenate([train, test])
    n = split.grid.n_rows
    if union.size != n or np.unique(union).size != n:
        raise HoldoutViolation(
            f"{split.dataset}: train({train.size}) + test({test.size}) do not partition the {n}-cell grid (unique rows: {np.unique(union).size})"
        )
    return n


@dataclass(frozen=True)
class FactorCell:
    cell_id: str
    values: dict[str, tuple[float, ...]]
    meta: dict = None

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
                missing = sorted(set(allowed_arr.tolist()) - set(grid.values[k].tolist()))
                raise ValueError(
                    f"cell {self.cell_id!r}: factor {name!r} value(s) {missing} are not on the grid"
                )
            out.append(sel.astype(np.int64))
        return out

    def mask(self, grid: FactorGrid, factor_values: np.ndarray) -> np.ndarray:
        """(N, F) factor **values** -> (N,) boolean membership of this cell."""
        vals = np.asarray(factor_values)
        if vals.ndim != 2 or vals.shape[1] != grid.n_factors:
            raise ValueError(
                f"cell {self.cell_id!r}: expected (N, {grid.n_factors}) values, got {vals.shape}"
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
    seen: dict[int, str] = {}
    for cell in cells:
        for row in cell.rows(grid).tolist():
            if row in seen:
                raise HoldoutViolation(
                    f"cells {seen[row]!r} and {cell.cell_id!r} both contain row {row} — the per-cell counts would double-count it"
                )
            seen[row] = cell.cell_id
    return len(seen)


def build_cell_split(
    grid: FactorGrid, cells: Sequence[FactorCell], dataset: str, condition: str, variant: str
) -> GenSplit:
    """Split the grid on a union of :class:`FactorCell`s (train = exact complement)."""
    assert_cells_disjoint(grid, cells)
    test_rows = np.sort(np.concatenate([cell.rows(grid) for cell in cells]).astype(np.int64))
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


C_INVALID_SCHEMA = {
    "version": 1,
    "description": "Structurally invalid factor requests, defined in FACTOR SPACE ONLY. Identical for every arm: no well, mode, cluster or any other model-internal structure is referenced, so arms without such structure consume the same file.",
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'duplicate_factor' | 'out_of_range'",
        "request": "list of {factor, factor_index, value} terms; a well-formed request names each factor at most once with a value drawn from that factor's grid",
        "violated_factor": "name of the factor whose constraint is violated",
        "reason": "human-readable statement of why the request cannot be satisfied",
    },
    "kinds": {
        "duplicate_factor": "two different values of the SAME factor requested simultaneously — unsatisfiable because each image has exactly one value per factor",
        "out_of_range": "a value outside that factor's realised range — unsatisfiable because the grid is exhaustive and closed",
    },
}


def build_c_invalid(grid: FactorGrid, n_per_kind: int = 16, seed: int = 0) -> dict:
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
                "reason": f"factor {grid.names[k]!r} is requested at both {v0} and {v1}; an image has exactly one value per factor",
            }
        )
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
                "reason": f"factor {grid.names[k]!r} requested at {bad}, outside its realised range [{float(vals[0])}, {float(vals[-1])}]",
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
        dup = any((len(v) > 1 for v in seen.values()))
        oor = False
        for term in req:
            k = int(term["factor_index"])
            if not np.any(grid.values[k] == term["value"]):
                oor = True
        if rec["kind"] == "duplicate_factor" and (not dup):
            raise AssertionError(f"{rec['id']}: claims duplicate_factor but has none")
        if rec["kind"] == "out_of_range" and (not oor):
            raise AssertionError(f"{rec['id']}: claims out_of_range but all values are on-grid")
        if not (dup or oor):
            raise AssertionError(f"{rec['id']}: request is satisfiable — not a C item")
    return len(records)


TERM_KEYS = ("factor", "factor_index", "value")
A7_SCHEMA = {
    "version": 1,
    "description": "VALID factor requests carrying one REDUNDANT term (a factor repeated at the value it already has).  The structural control for the C set's 'duplicate_factor' records: same term count, same term ordering, same repeated-factor pattern, and still satisfiable.  Built from valid requests alone -- no invalid example is used.",
    "record": {
        "id": "stable string id, unique within the file",
        "kind": "'valid_redundant'",
        "request": "list of {factor, factor_index, value} terms; every factor named once plus ONE repeat of a single factor at its own value",
        "redundant_factor": "name of the repeated factor",
        "row": "dataset row this request denotes (it is satisfiable, and this is it)",
    },
    "convention": "|A7| = |A|: the construction is a 1:1 map over valid requests.  The shipped file instantiates A as a deterministic TRAIN-row sample.",
}


def _grid_terms(grid: FactorGrid, row: int) -> list[dict]:
    """The ``n_factors`` terms of the request denoting ``row`` (grid values, verbatim)."""
    vals = grid.values_of(np.array([int(row)], dtype=np.int64))[0]
    return [
        {"factor": grid.names[k], "factor_index": k, "value": float(vals[k])}
        for k in range(grid.n_factors)
    ]


def build_base_requests(grid: FactorGrid, rows: Sequence[int], source: str = "base") -> list[dict]:
    """The plain ``n_factors``-term valid requests for ``rows`` -- the ``base`` control.

    Derived from ``factor_index.npz`` rather than stored: a valid request *is* a row,
    so persisting them would duplicate the index.  Emitted in the same record shape as
    every other query set so the detector consumes all of them identically.
    """
    out = []
    for i, r in enumerate(np.asarray(rows, dtype=np.int64).tolist()):
        out.append(
            {
                "id": f"{grid.dataset}-{source}-{i:06d}",
                "kind": "valid",
                "request": _grid_terms(grid, r),
                "row": int(r),
            }
        )
    return out


def select_a7_rows(train_rows: np.ndarray, n_requests: int, seed: int = 0) -> np.ndarray:
    train_rows = np.asarray(train_rows, dtype=np.int64)
    n = int(n_requests)
    if n <= 0:
        raise ValueError(f"a7_n_requests must be >= 1, got {n}")
    if train_rows.size == 0:
        raise ValueError("no train rows to build A7 from")
    rng = np.random.default_rng(int(seed))
    replace = train_rows.size < n
    return np.sort(rng.choice(train_rows, size=n, replace=replace))


def build_a7_redundant(grid: FactorGrid, rows: Sequence[int], seed: int = 0) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    n_f = grid.n_factors
    records: list[dict] = []
    for i, r in enumerate(rows.tolist()):
        terms = _grid_terms(grid, r)
        j = int(i % n_f)
        terms.append(dict(terms[j]))
        records.append(
            {
                "id": f"{grid.dataset}-a7-{i:03d}",
                "kind": "valid_redundant",
                "request": terms,
                "redundant_factor": grid.names[j],
                "row": int(r),
            }
        )
    return {
        "schema": A7_SCHEMA,
        "dataset": grid.dataset,
        "seed": int(seed),
        "n_requests": len(records),
        "factors": {
            "names": list(grid.names),
            "cardinalities": list(grid.cardinalities),
            "ranges": [[float(v[0]), float(v[-1])] for v in grid.values],
        },
        "records": records,
    }


def validate_a7_redundant(payload: dict, grid: FactorGrid) -> int:
    """Assert every A7 record is SATISFIABLE and ``n_factors + 1`` terms long.

    An A7 whose "redundant" term carried a *different* value would be a C record with
    the wrong label -- it would make the control harder than the thing it controls and
    every margin computed against it would be wrong in our favour.  So it is checked.
    """
    records = payload["records"]
    if not records:
        raise AssertionError("A7 set is empty")
    n_f = grid.n_factors
    for rec in records:
        req = rec["request"]
        if len(req) != n_f + 1:
            raise AssertionError(
                f"{rec['id']}: {len(req)} terms, expected {n_f + 1} (the whole point of A7 is the term count)"
            )
        by_factor: dict[str, set] = {}
        for term in req:
            if tuple(term.keys()) != TERM_KEYS:
                raise AssertionError(f"{rec['id']}: term keys {tuple(term.keys())} != {TERM_KEYS}")
            k = int(term["factor_index"])
            if grid.names[k] != term["factor"]:
                raise AssertionError(
                    f"{rec['id']}: factor {term['factor']!r} at index {k} (grid says {grid.names[k]!r})"
                )
            if not np.any(grid.values[k] == term["value"]):
                raise AssertionError(
                    f"{rec['id']}: value {term['value']} of factor {term['factor']!r} is OFF-GRID -- A7 must be satisfiable"
                )
            by_factor.setdefault(term["factor"], set()).add(term["value"])
        if len(by_factor) != n_f:
            raise AssertionError(f"{rec['id']}: names {len(by_factor)} factors, expected {n_f}")
        conflicting = [f for f, v in by_factor.items() if len(v) > 1]
        if conflicting:
            raise AssertionError(
                f"{rec['id']}: factor(s) {conflicting} requested at two DIFFERENT values -- that is a C record, not a valid control"
            )
        row = rec.get("row")
        if row is not None:
            want = grid.values_of(np.array([int(row)], dtype=np.int64))[0]
            got = np.array([by_factor[grid.names[k]].pop() for k in range(n_f)])
            if not np.array_equal(got, want):
                raise AssertionError(f"{rec['id']}: request does not denote its own row {row}")
    return len(records)


ASSERTION_RECORD_KEYS = ("id", "request")
EXCLUDED_RECORD_KEYS = ("kind", "violated_factor", "reason", "row", "redundant_factor")
QUERY_SET_INDEX: dict[str, dict] = {
    "base": {
        "role": "valid",
        "derived_from": "factor_index.npz",
        "builder": "nclu.data.gen_splits.build_base_requests",
        "n_terms": "n_factors",
        "description": "A and B: one term per factor, the plain valid request.",
    },
    "a7_redundant": {
        "role": "valid",
        "file": "a7_redundant.json",
        "builder": "nclu.data.gen_splits.build_a7_redundant",
        "n_terms": "n_factors + 1",
        "description": "valid requests carrying one redundant (same-value) term.",
    },
    "c_invalid": {
        "role": "invalid",
        "file": "c_invalid.json",
        "builder": "nclu.data.gen_splits.build_c_invalid",
        "controls": {"duplicate_factor": "a7_redundant", "out_of_range": "base"},
        "description": "structurally invalid requests (the C set).",
    },
}


def write_query_set_index(
    out_dir: str | Path, dataset: str, emitted: dict[str, dict]
) -> tuple[Path, str]:
    out_dir = Path(out_dir)
    sets = {}
    for name, spec in QUERY_SET_INDEX.items():
        entry = {k: v for k, v in spec.items()}
        entry.update(emitted.get(name, {}))
        sets[name] = entry
    payload = {
        "schema": {
            "version": 1,
            "description": "The registered query sets of this dataset directory.  Every invalid set names, per kind, the valid set that is its STRUCTURAL control; the no-model detector test discovers this file and audits the invalid sets against those controls.",
        },
        "dataset": dataset,
        "excluded_record_keys": list(EXCLUDED_RECORD_KEYS),
        "sets": sets,
    }
    return write_json_with_checksum(payload, out_dir / "query_sets.json")


def load_registered_query_sets(
    index_path: str | Path, max_base: int = 2048, base_seed: int = 0
) -> tuple[dict, dict[str, list[dict]]]:
    index_path = Path(index_path)
    index_doc = json.loads(index_path.read_text())
    d = index_path.parent
    sets: dict[str, list[dict]] = {}
    for name, spec in index_doc["sets"].items():
        if spec.get("file"):
            p = d / spec["file"]
            if not p.exists():
                raise FileNotFoundError(
                    f"{index_path}: query set {name!r} is registered at {p} but the file is missing -- the audit cannot be run"
                )
            sets[name] = json.loads(p.read_text())["records"]
        elif spec.get("derived_from") == "factor_index.npz":
            grid, train_rows, test_rows, _ = load_index(d / "factor_index.npz")
            pool = np.concatenate(
                [np.asarray(train_rows, np.int64), np.asarray(test_rows, np.int64)]
            )
            rng = np.random.default_rng(int(base_seed))
            if pool.size > int(max_base):
                pool = np.sort(rng.choice(pool, size=int(max_base), replace=False))
            sets[name] = build_base_requests(grid, pool, source=name)
        else:
            raise ValueError(
                f"{index_path}: query set {name!r} declares neither a file nor a derivation -- the detector cannot reach it"
            )
    return (index_doc, sets)


MASKED_VALUE = 0.0


def mask_request(request: Sequence[dict]) -> list[dict]:
    """The request with every factor VALUE replaced by :data:`MASKED_VALUE`."""
    return [{k: MASKED_VALUE if k == "value" else t[k] for k in t} for t in request]


def structural_features(record: dict) -> dict[str, float]:
    """Every structural property of a record we can reach WITHOUT its factor values.

    Adversarial by intent: term count is only the giveaway we already found, so this
    also reaches term ordering, the repeated-factor pattern and its position, the
    factor-index arithmetic, and the length of the value-masked serialisation.
    """
    req = list(record["request"])
    idx = [int(t["factor_index"]) for t in req]
    n = len(idx)
    counts: dict[int, int] = {}
    for j in idx:
        counts[j] = counts.get(j, 0) + 1
    first_dup = -1
    for p, j in enumerate(idx):
        if idx.index(j) != p:
            first_dup = p
            break
    inversions = sum((1 for a in range(n) for b in range(a + 1, n) if idx[a] > idx[b]))
    masked = json.dumps(mask_request(req))
    return {
        "n_terms": float(n),
        "n_distinct_factors": float(len(counts)),
        "max_factor_multiplicity": float(max(counts.values())) if counts else 0.0,
        "n_redundant_terms": float(n - len(counts)),
        "is_factor_order_sorted": float(idx == sorted(idx)),
        "n_order_inversions": float(inversions),
        "first_duplicate_position": float(first_dup),
        "sum_factor_indices": float(sum(idx)),
        "first_factor_index": float(idx[0]) if idx else -1.0,
        "last_factor_index": float(idx[-1]) if idx else -1.0,
        "masked_request_json_len": float(len(masked)),
    }


def diagnostic_features(record: dict) -> dict[str, float]:
    rid = str(record.get("id", ""))
    tail = rid.rsplit("-", 1)[-1]
    return {
        "id_numeric_suffix": float(tail) if tail.isdigit() else -1.0,
        "id_token_count": float(len(rid.split("-"))),
        "record_key_count": float(len(record)),
    }


def definitional_features(record: dict) -> dict[str, float]:
    req = list(record["request"])
    by_factor: dict[int, set] = {}
    for t in req:
        by_factor.setdefault(int(t["factor_index"]), set()).add(t["value"])
    return {
        "n_distinct_values": float(len({t["value"] for t in req})),
        "has_conflicting_repeat": float(any((len(v) > 1 for v in by_factor.values()))),
    }


FEATURE_VIEWS = {
    "structural": structural_features,
    "diagnostic": diagnostic_features,
    "definitional": definitional_features,
}


def auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """AUROC of ``pos`` over ``neg`` by mid-ranks (a tied feature gives exactly 0.5)."""
    pos = np.asarray(list(pos_scores), dtype=float)
    neg = np.asarray(list(neg_scores), dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUROC needs a non-empty class on both sides")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=float)
    sorted_v = allv[order]
    i = 0
    while i < sorted_v.size:
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _feature_table(records: Sequence[dict], fn) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    for rec in records:
        for k, v in fn(rec).items():
            cols.setdefault(k, []).append(v)
    return cols


def no_model_detector(
    invalid_records: Sequence[dict], valid_records: Sequence[dict], view: str = "structural"
) -> dict[str, float]:
    fn = FEATURE_VIEWS[view]
    pos = _feature_table(invalid_records, fn)
    neg = _feature_table(valid_records, fn)
    out = {}
    for k in sorted(pos):
        a = auroc(pos[k], neg[k])
        out[k] = float(max(a, 1.0 - a))
    return out


def audit_query_sets(index_path: str | Path, tolerance: float, max_base: int = 2048) -> dict:
    index_doc, sets = load_registered_query_sets(index_path, max_base=max_base)
    violations: list[str] = []
    allowed = set(ASSERTION_RECORD_KEYS) | set(EXCLUDED_RECORD_KEYS)
    for name, recs in sets.items():
        for rec in recs:
            extra = set(rec) - allowed
            if extra:
                violations.append(
                    f"{name}/{rec.get('id')}: record key(s) {sorted(extra)} are not in the declared allowlist {sorted(allowed)} -- a new field is a candidate giveaway and must be declared before it ships"
                )
                break
            bad_term = next((t for t in rec["request"] if tuple(t.keys()) != TERM_KEYS), None)
            if bad_term is not None:
                violations.append(
                    f"{name}/{rec.get('id')}: term keys {tuple(bad_term.keys())} != {TERM_KEYS} -- key order is itself a giveaway"
                )
                break
    report = {
        "index": str(index_path),
        "dataset": index_doc.get("dataset", ""),
        "tolerance": float(tolerance),
        "n_records": {k: len(v) for k, v in sets.items()},
        "per_kind": {},
        "uncontrolled_vs_base": {},
        "diagnostics_vs_base": {},
        "definitional_vs_base": {},
        "pooled": {},
        "pooled_uncontrolled_vs_base": {},
        "violations": violations,
    }
    for name, spec in index_doc["sets"].items():
        if spec.get("role") != "invalid":
            continue
        controls = spec.get("controls") or {}
        by_kind: dict[str, list[dict]] = {}
        for rec in sets[name]:
            by_kind.setdefault(str(rec.get("kind", "unknown")), []).append(rec)
        pooled_dir: dict[str, float] = {}
        pooled_unc_dir: dict[str, float] = {}
        n_total = sum((len(v) for v in by_kind.values()))
        for kind, recs in sorted(by_kind.items()):
            control = controls.get(kind)
            if control is None:
                violations.append(
                    f"{name}: kind {kind!r} has NO declared structural control in {index_path} -- every invalid kind needs one"
                )
                continue
            if control not in sets:
                violations.append(
                    f"{name}: kind {kind!r} declares control {control!r}, which is not a registered query set"
                )
                continue
            key = f"{name}[{kind}]"
            feats = no_model_detector(recs, sets[control])
            worst = max(feats, key=lambda k: feats[k])
            report["per_kind"][key] = {
                "control": control,
                "n_invalid": len(recs),
                "n_control": len(sets[control]),
                "features": feats,
                "worst_feature": worst,
                "max_auroc": feats[worst],
            }
            if feats[worst] > 0.5 + tolerance:
                violations.append(
                    f"{key} vs {control}: no-model detector reaches AUROC {feats[worst]:.4f} on {worst!r} (tolerance {0.5 + tolerance:.4f}) -- a structural giveaway separates this set with no model in it"
                )
            if "base" in sets:
                unc = no_model_detector(recs, sets["base"])
                w = max(unc, key=lambda k: unc[k])
                report["uncontrolled_vs_base"][key] = {
                    "features": unc,
                    "worst_feature": w,
                    "max_auroc": unc[w],
                }
                report["diagnostics_vs_base"][key] = no_model_detector(
                    recs, sets["base"], view="diagnostic"
                )
                report["definitional_vs_base"][key] = no_model_detector(
                    recs, sets["base"], view="definitional"
                )
            w_k = len(recs) / n_total if n_total else 0.0
            for f, a in no_model_detector_directional(recs, sets[control]).items():
                pooled_dir[f] = pooled_dir.get(f, 0.0) + w_k * a
            if "base" in sets:
                for f, a in no_model_detector_directional(recs, sets["base"]).items():
                    pooled_unc_dir[f] = pooled_unc_dir.get(f, 0.0) + w_k * a
        if pooled_unc_dir:
            pu = {f: float(max(a, 1.0 - a)) for f, a in pooled_unc_dir.items()}
            w = max(pu, key=lambda k: pu[k])
            report["pooled_uncontrolled_vs_base"][name] = {
                "features": pu,
                "worst_feature": w,
                "max_auroc": pu[w],
            }
        if pooled_dir:
            pooled = {f: float(max(a, 1.0 - a)) for f, a in pooled_dir.items()}
            worst = max(pooled, key=lambda k: pooled[k])
            report["pooled"][name] = {
                "features": pooled,
                "worst_feature": worst,
                "max_auroc": pooled[worst],
            }
            if pooled[worst] > 0.5 + tolerance:
                violations.append(
                    f"{name} pooled: no-model detector reaches AUROC {pooled[worst]:.4f} on {worst!r} (tolerance {0.5 + tolerance:.4f})"
                )
    return report


def no_model_detector_directional(
    invalid_records: Sequence[dict], valid_records: Sequence[dict]
) -> dict[str, float]:
    """Signed per-feature AUROC (used for stratified pooling, where sign matters)."""
    pos = _feature_table(invalid_records, structural_features)
    neg = _feature_table(valid_records, structural_features)
    return {k: auroc(pos[k], neg[k]) for k in sorted(pos)}


def assert_no_model_detector_at_chance(
    index_path: str | Path, tolerance: float, max_base: int = 2048
) -> dict:
    report = audit_query_sets(index_path, tolerance, max_base=max_base)
    if report["violations"]:
        raise AssertionError(
            f"no-model detector audit FAILED for {index_path}:\n  "
            + "\n  ".join(report["violations"])
        )
    return report


def save_index(split: GenSplit, path: str | Path, extra: dict | None = None) -> tuple[Path, str]:
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
        "meta": np.asarray(json.dumps({**split.summary(), **(extra or {})}, sort_keys=True)),
    }
    for k, name in enumerate(split.grid.names):
        payload[f"values__{name}"] = split.grid.values[k]
    with open(path, "wb") as f:
        np.savez(f, **payload)
    return (path, file_sha256(path))


def load_index(path: str | Path) -> tuple[FactorGrid, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        names = tuple((str(n) for n in z["factor_names"]))
        meta = json.loads(str(z["meta"]))
        grid = FactorGrid(
            names=names,
            values=tuple((z[f"values__{n}"] for n in names)),
            dataset=meta.get("dataset", ""),
        )
        return (grid, z["train_rows"], z["test_rows"], meta)


def _quick_subsample(rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic subsample used by ``--quick`` (sorted, so order is stable)."""
    if n <= 0 or rows.size <= n:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=n, replace=False))


def factor_grid_from_latents(
    latent_values: np.ndarray, names: Sequence[str], dataset: str
) -> FactorGrid:
    latent_values = np.asarray(latent_values)
    if latent_values.shape[1] != len(names):
        raise ValueError(f"{latent_values.shape[1]} columns for {len(names)} factor names")
    values = tuple((np.unique(latent_values[:, k]) for k in range(len(names))))
    grid = FactorGrid(names=tuple(names), values=values, dataset=dataset)
    if grid.n_rows != len(latent_values):
        raise ValueError(
            f"{dataset}: factor grid has {grid.n_rows} cells but the file has {len(latent_values)} rows — the grid is not exhaustive"
        )
    return grid
