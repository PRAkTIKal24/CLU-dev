"""The VALUE hold-out condition: the cells, the proofs, and the query sets.

Why this file exists
--------------------
Wave 2 held out **combinations**.  Rows 2, 5, 6 and 7 of the Head's taxonomy turn on a
value the store has never held, and against a combination hold-out they are not
measurable at all.  ``chlu.data.gen_value_holdout`` closes that gap — and every test
below exists because the corresponding failure would be **silent**:

- a value cell that quietly left rows carrying the held-out value in train would make
  every row-5 refusal a refusal of something the store had seen;
- a row-2 request that drifted on-grid would silently become row 1, and the refusal
  pair would then be scored against a request the store has literally seen;
- a row-2 midpoint that landed on the held-out value would make rows 2 and 5 the same
  request with two different correct answers.

⛔ The no-model detector is run here at the pin's own tolerance, on the artifact
discovered from disk, exactly as ``tests/test_gen_c_set_a7.py`` does — and the
**positive control** (``test_detector_fires_...``) proves the audit can fail.
"""

import json

import numpy as np
import pytest

from chlu.data import gen_splits as G
from chlu.data import gen_value_holdout as V

# ⛔ The pin's tolerance (tests/test_gen_c_set_a7.py). Raising it is not a way to pass.
DETECTOR_TOLERANCE = 0.05

CARDS = (10, 10, 10, 8, 4, 15)
NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")


@pytest.fixture(scope="module")
def grid():
    """A Shapes3D-shaped grid with Shapes3D's realised value ranges, built offline.

    ⭐ Same cardinalities, same factor names and the same 0.1-spaced hue ring as the
    real ``3dshapes.h5``, so the cell arithmetic under test is the shipped arithmetic —
    without a 268 MB download in the suite.
    """
    values = (
        np.arange(10) / 10.0,
        np.arange(10) / 10.0,
        np.arange(10) / 10.0,
        np.linspace(0.75, 1.25, 8),
        np.arange(4, dtype=float),
        np.linspace(-30.0, 30.0, 15),
    )
    return G.FactorGrid(names=NAMES, values=values, dataset="toy3d")


@pytest.fixture(scope="module")
def cfg():
    return V.ValueHoldoutConfig(dataset="toy3d")


@pytest.fixture(scope="module")
def split_and_cells(grid, cfg):
    return V.build_value_holdout_split(grid, cfg)


@pytest.fixture(scope="module")
def emitted(tmp_path_factory, grid, cfg):
    d = tmp_path_factory.mktemp("value_holdout")
    return d, V.emit_value_holdout(grid, d, cfg)


# ---------------------------------------------------------------------------
# 1. the cells
# ---------------------------------------------------------------------------


def test_value_cells_hold_out_WHOLE_values_not_combinations(grid, cfg):
    """The hue cell constrains ONE factor; everything else is unconstrained."""
    hue_cell, shape_cell = V.build_value_cells(grid, cfg)
    assert set(hue_cell.values) == {"object_hue"}
    assert hue_cell.values["object_hue"] == (0.5,)
    # ⚠ the shape cell carries the hue carve-out (module §4) and nothing else
    assert set(shape_cell.values) == {"shape", "object_hue"}
    assert shape_cell.values["shape"] == (0.0,)
    assert 0.5 not in shape_cell.values["object_hue"]
    assert len(shape_cell.values["object_hue"]) == 9


def test_the_union_of_the_value_cells_is_every_row_carrying_either_value(grid, cfg):
    """⭐ The carve-out of §4 moves book-keeping, ⛔ not the hold-out itself."""
    rows = set()
    for c in V.build_value_cells(grid, cfg):
        rows |= set(c.rows(grid).tolist())
    vals = grid.values_of(np.array(sorted(rows)))
    assert np.all((vals[:, 2] == 0.5) | (vals[:, 4] == 0.0))
    # and nothing carrying either value is left out
    all_vals = grid.values_of(np.arange(grid.n_rows))
    want = int(np.sum((all_vals[:, 2] == 0.5) | (all_vals[:, 4] == 0.0)))
    assert len(rows) == want == 156_000


def test_every_cell_family_is_pairwise_disjoint(grid, cfg, split_and_cells):
    """``assert_cells_disjoint`` is the proof, and it returns the pooled count."""
    _, cells = split_and_cells
    n = G.assert_cells_disjoint(grid, cells)
    assert n == sum(c.rows(grid).size for c in cells)


def test_combination_cells_avoid_the_held_out_values(grid, cfg, split_and_cells):
    """⛔ A combination cell that used a held-out value would not be a COMBINATION cell."""
    _, cells = split_and_cells
    for c in cells:
        if c.meta["family"] == "value":
            continue
        assert 0.5 not in c.values.get("object_hue", ())
        assert 0.0 not in c.values.get("shape", ())


def test_nway_cells_have_monteros_shape_and_size(grid, cfg, split_and_cells):
    """⭐ 72 rows per cell — Montero's cell exactly, so a wave-3 corner is comparable."""
    _, cells = split_and_cells
    nway = [c for c in cells if c.meta["family"] == "nway"]
    assert len(nway) == 49  # ⭐ the w3-hub-common §3 multi-corner bar
    assert {int(c.rows(grid).size) for c in nway} == {72}


def test_pair_cells_are_two_way_over_individually_retained_values(grid, cfg):
    """Row 3: both values are stored elsewhere; only the CO-OCCURRENCE is held out."""
    split, _ = V.build_value_holdout_split(grid, cfg)
    train_vals = split.grid.values_of(split.train_rows)
    for c in V.build_pair_cells(grid, cfg):
        assert set(c.values) == {"object_hue", "shape"}
        h = c.values["object_hue"][0]
        s = c.values["shape"][0]
        assert np.any(train_vals[:, 2] == h), "the hue must be stored elsewhere"
        assert np.any(train_vals[:, 4] == s), "the shape must be stored elsewhere"
        assert not np.any((train_vals[:, 2] == h) & (train_vals[:, 4] == s))


def test_a_value_cell_that_overlaps_is_caught(grid, cfg):
    """⛔ POSITIVE CONTROL for §4: drop the carve-out and the disjointness check fires."""
    bad = [
        G.FactorCell("value-object_hue", {"object_hue": (0.5,)}),
        G.FactorCell("value-shape", {"shape": (0.0,)}),  # ⛔ no hue carve-out
    ]
    with pytest.raises(G.HoldoutViolation, match="both contain row"):
        G.assert_cells_disjoint(grid, bad)


# ---------------------------------------------------------------------------
# 2. the proofs
# ---------------------------------------------------------------------------


def test_the_split_partitions_the_grid_and_train_is_the_exact_complement(
    split_and_cells,
):
    split, _ = split_and_cells
    assert G.assert_index_partitions_grid(split) == 480_000
    assert G.assert_non_empty_test(split) == split.n_test
    assert G.assert_holdout_disjoint(split) == split.n_train
    assert split.n_train + split.n_test == 480_000


@pytest.mark.parametrize("factor,value", [("object_hue", 0.5), ("shape", 0.0)])
def test_no_train_row_carries_the_held_out_value(split_and_cells, factor, value):
    """⭐ The hold-out property itself, checked directly on the train index."""
    split, _ = split_and_cells
    assert V.assert_value_absent_from_train(split, factor, value) == split.n_train


@pytest.mark.parametrize("factor,value", [("object_hue", 0.5), ("shape", 0.0)])
def test_the_value_assertion_fires_when_a_held_out_row_is_injected(
    split_and_cells, grid, factor, value
):
    """⛔ POSITIVE CONTROL: a check that cannot fail proves nothing."""
    split, _ = split_and_cells
    smuggled = int(split.test_rows[
        np.argmax(grid.values_of(split.test_rows)[:, grid.index_of(factor)] == value)
    ])
    bad = split.with_train_rows(np.append(split.train_rows, smuggled))
    with pytest.raises(G.HoldoutViolation, match="carry the held-out value"):
        V.assert_value_absent_from_train(bad, factor, value)


def test_the_store_fit_pool_is_large_enough_at_the_shipped_scale(split_and_cells):
    """§9: report the realised counts — and they must support the shipped fit.

    The shipped generative cell fits the store on a 32,768-row sample of train.
    """
    split, _ = split_and_cells
    assert split.n_train == 284_472
    assert split.n_train >= 32_768 * 4, "the fit pool must not be the binding constraint"


# ---------------------------------------------------------------------------
# 3. the query sets
# ---------------------------------------------------------------------------


def test_row2_is_offgrid_in_range_and_in_basin(grid, cfg, split_and_cells):
    split, _ = split_and_cells
    payload = V.build_row2_unseen(grid, split, cfg)
    assert V.validate_row2(payload, grid, split) == cfg.row2_n_requests


def test_row2_can_never_land_on_the_held_out_value(grid, cfg, split_and_cells):
    """⛔ The collision that would make rows 2 and 5 the same request.

    ``(0.4, 0.6)`` brackets the hole at ``0.5`` but is NOT grid-adjacent, so it is
    absent from the pair list by construction — this pins that.
    """
    pairs = V.retained_adjacent_pairs(grid, "object_hue", 0.5)
    hue = grid.values[2]
    mids = {0.5 * (float(hue[a]) + float(hue[b])) for a, b in pairs}
    assert 0.5 not in mids
    assert (4, 6) not in pairs and (4, 5) not in pairs and (5, 6) not in pairs
    split, _ = split_and_cells
    payload = V.build_row2_unseen(grid, split, cfg)
    for rec in payload["records"]:
        assert rec["request"][2]["value"] != 0.5


def test_row2_rejects_a_request_whose_bracket_is_not_stored(grid, cfg, split_and_cells):
    """⛔ POSITIVE CONTROL: 'in-basin' is checked, not asserted."""
    split, _ = split_and_cells
    payload = V.build_row2_unseen(grid, split, cfg)
    rec = json.loads(json.dumps(payload["records"][0]))
    rec["request"][2]["value"] = 0.45  # brackets 0.4 and the HELD-OUT 0.5
    payload = {**payload, "records": [rec]}
    with pytest.raises(AssertionError, match="not in-basin"):
        V.validate_row2(payload, grid, split)


def test_row5_is_on_grid_denotes_its_row_and_is_held_out(grid, cfg, split_and_cells):
    split, _ = split_and_cells
    payload = V.build_row5_held_out(grid, split, cfg)
    assert V.validate_row5(payload, grid, split) == 2 * cfg.row5_n_per_kind
    kinds = {r["kind"] for r in payload["records"]}
    assert kinds == {"held_out_hue", "held_out_shape"}
    for rec in payload["records"]:
        k = grid.index_of(rec["violated_factor"])
        assert np.any(grid.values[k] == rec["request"][k]["value"]), "must be ON-grid"


def test_row5_rejects_a_request_the_store_actually_holds(grid, cfg, split_and_cells):
    """⛔ POSITIVE CONTROL: a row-5 record drawn from TRAIN is not a refusal item."""
    split, _ = split_and_cells
    payload = V.build_row5_held_out(grid, split, cfg)
    row = int(split.train_rows[0])
    vals = grid.values_of(np.array([row]))[0].tolist()
    rec = dict(payload["records"][0])
    rec["request"] = V._terms(grid, vals)
    rec["row"] = row
    with pytest.raises(AssertionError, match="IN the store's fit pool"):
        V.validate_row5({**payload, "records": [rec]}, grid, split)


def test_rows_2_5_and_6_are_all_n_factor_requests_in_term_key_order(emitted):
    """⛔ Term key ORDER is itself a giveaway; the pin says so and this pins it."""
    d, _ = emitted
    for name in ("row2_unseen.json", "row5_held_out.json"):
        for rec in json.loads((d / name).read_text())["records"]:
            assert len(rec["request"]) == 6
            for term in rec["request"]:
                assert tuple(term.keys()) == G.TERM_KEYS


def test_every_record_key_is_inside_the_pins_declared_allowlist(emitted):
    """A new field is a candidate giveaway and must be declared before it ships."""
    d, _ = emitted
    allowed = set(G.ASSERTION_RECORD_KEYS) | set(G.EXCLUDED_RECORD_KEYS)
    for name in ("row2_unseen.json", "row5_held_out.json", "c_invalid.json"):
        for rec in json.loads((d / name).read_text())["records"]:
            assert set(rec) <= allowed, f"{name}: {sorted(set(rec) - allowed)}"


# ---------------------------------------------------------------------------
# 4. ⛔⛔ the no-model detector
# ---------------------------------------------------------------------------


def test_the_no_model_detector_is_at_chance_on_rows_2_5_and_6(emitted):
    """⛔⛔ The task's falsifier, at the pin's tolerance, on the artifact from disk."""
    d, report = emitted
    audit = G.assert_no_model_detector_at_chance(
        d / "query_sets.json", tolerance=DETECTOR_TOLERANCE
    )
    assert set(audit["per_kind"]) == {
        "c_invalid[duplicate_factor]",
        "c_invalid[out_of_range]",
        "row5_held_out[held_out_hue]",
        "row5_held_out[held_out_shape]",
    }
    for key, entry in audit["per_kind"].items():
        assert entry["max_auroc"] <= 0.5 + DETECTOR_TOLERANCE, key
    assert report["detector_passed"] is True


def test_rows_5_and_6_are_both_controlled_by_row_2(emitted):
    """⭐ The separation that IS the claim: row 2 accept vs rows 5/6 refuse."""
    d, _ = emitted
    reg = json.loads((d / "query_sets.json").read_text())
    assert reg["sets"]["row5_held_out"]["controls"] == {
        "held_out_hue": "row2_unseen",
        "held_out_shape": "row2_unseen",
    }
    assert reg["sets"]["c_invalid"]["controls"]["out_of_range"] == "row2_unseen"
    # ⛔ duplicate_factor keeps the 7-term control: R26 ruling 1.
    assert reg["sets"]["c_invalid"]["controls"]["duplicate_factor"] == "a7_redundant"


def test_the_detector_fires_when_a_structural_giveaway_is_injected(emitted, tmp_path):
    """⛔ POSITIVE CONTROL: an audit that cannot fail proves nothing.

    Append one redundant term to every row-5 record — the exact giveaway R26 ruling 1
    killed on the C set — and the audit must find it.
    """
    d, _ = emitted
    for name in (
        "factor_index.npz", "query_sets.json", "row2_unseen.json",
        "c_invalid.json", "a7_redundant.json",
    ):
        (tmp_path / name).write_bytes((d / name).read_bytes())
    payload = json.loads((d / "row5_held_out.json").read_text())
    for rec in payload["records"]:
        rec["request"].append(dict(rec["request"][0]))
    (tmp_path / "row5_held_out.json").write_text(json.dumps(payload))
    report = G.audit_query_sets(tmp_path / "query_sets.json", DETECTOR_TOLERANCE)
    assert report["violations"]
    assert report["per_kind"]["row5_held_out[held_out_hue]"]["max_auroc"] == 1.0
    with pytest.raises(AssertionError, match="no-model detector audit FAILED"):
        G.assert_no_model_detector_at_chance(
            tmp_path / "query_sets.json", DETECTOR_TOLERANCE
        )


def test_the_grid_reach_diagnostic_separates_row_5_from_row_2(emitted):
    """⛔⛔ REPORTED, NOT ASSERTED — and it is the spoke's headline finding.

    Row 2 is OFF-grid by definition and row 5 is ON-grid by definition, so counting
    off-grid terms separates them perfectly with no model in it.  ⛔ This is a property
    of the taxonomy's own definitions, not of the query sets, and it is why the refusal
    rows need the laundering control in ``gen_scoreboard``.
    """
    _, report = emitted
    reach = report["grid_reach_diagnostic"]
    assert reach["row5_vs_row2"]["n_offgrid_terms"] == 1.0
    assert reach["row6_out_of_range_vs_row2"]["max_distance_outside_range"] == 1.0
    # ⭐ but the STRUCTURAL view — the asserted one — stays at chance
    assert report["detector_passed"] is True


# ---------------------------------------------------------------------------
# 5. the emitted artifact
# ---------------------------------------------------------------------------


def test_every_artifact_is_on_disk_with_its_checksum(emitted):
    d, report = emitted
    for name, rec in report["artifacts"].items():
        assert (d / name).exists(), name
        assert len(rec["sha256"]) == 64
    assert report["artifacts"]["factor_index.npz"]["sha256"] == G.file_sha256(
        d / "factor_index.npz"
    )


def test_the_scope_statement_is_a_FIELD_of_every_artifact(emitted):
    """⛔ §2: a hub cannot quote a row without it, so it is not only in prose."""
    d, report = emitted
    assert report["scope"] == V.VALUE_HOLDOUT_SCOPE
    assert "STORE'S FIT POOL" in V.VALUE_HOLDOUT_SCOPE
    for name in ("value_cells.json", "row2_unseen.json", "row5_held_out.json",
                 "query_sets.json"):
        assert json.loads((d / name).read_text())["scope"] == V.VALUE_HOLDOUT_SCOPE
    _, train, _, meta = G.load_index(d / "factor_index.npz")
    assert meta["scope"] == V.VALUE_HOLDOUT_SCOPE


def test_the_metadata_records_the_chosen_values_and_the_REJECTED_ones(emitted):
    """§9: 'record the rejected choice' — in the artifact, not only in a report."""
    d, _ = emitted
    meta = json.loads((d / "value_cells.json").read_text())
    assert meta["held_out_values"] == {"object_hue": 0.5, "shape": 0.0}
    assert "2.0 (sphere)" in meta["rationale"]["shape"]["rejected"]
    assert "degrades TOWARDS" in meta["rationale"]["shape"]["rejected"]["2.0 (sphere)"]
    assert "scale / orientation (any value)" in meta["rationale"]["hue"]["rejected"]


def test_realised_counts_are_reported_and_the_assertions_returned_them(emitted):
    _, report = emitted
    a = report["assertions"]
    assert a["assert_holdout_disjoint"] == report["counts"]["n_train_realised"]
    assert a["assert_cells_disjoint"] == report["counts"]["n_test_realised"]
    assert a["assert_index_partitions_grid"] == 480_000
    assert report["counts"]["rows_by_family"] == {
        "value": 156_000, "pair": 36_000, "nway": 3_528
    }


def test_emission_is_deterministic(grid, cfg, tmp_path, emitted):
    """Same config, same bytes — a checksum that moves run to run records nothing."""
    d, report = emitted
    again = V.emit_value_holdout(grid, tmp_path / "again", cfg, run_detector=False)
    for name in ("row2_unseen.json", "row5_held_out.json", "c_invalid.json",
                 "a7_redundant.json", "value_cells.json", "factor_index.npz"):
        assert again["artifacts"][name]["sha256"] == report["artifacts"][name]["sha256"]


def test_config_override_is_by_mapping_and_a_typo_raises(cfg):
    """⛔ §5: the dataclass lives next to the code, ⛔ not in chlu/config.py."""
    assert V.ValueHoldoutConfig.from_mapping({"n_nway_cells": 12}).n_nway_cells == 12
    with pytest.raises(ValueError, match="unknown key"):
        V.ValueHoldoutConfig.from_mapping({"n_nway_cell": 12})


def test_asking_for_more_cells_than_the_grid_can_hold_raises(grid):
    """⛔ Never silently truncated: a short table is a different experiment."""
    with pytest.raises(ValueError, match="structurally-disjoint slots"):
        V.build_value_holdout_cells(
            grid, V.ValueHoldoutConfig(n_nway_cells=10_000)
        )
