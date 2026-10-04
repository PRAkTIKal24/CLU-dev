"""The k-WAY COMPOSITION LADDER — the kill check, the two ends, and the rungs.

Why this file exists
--------------------
Every test below exists because the corresponding failure would be **silent** and
would make the ladder measure nothing:

- corners that **overlap** would double-count rows in a paired estimand that assumes
  disjointness, and the ladder's per-rung means would drift for a book-keeping reason;
- a k=2 rung that disagreed with row 3 would put the whole ladder off the instrument
  the hub already ships — ⛔ and the builder, not row 3, would be the thing that is
  wrong;
- a k=6 rung that disagreed with row 4 would do the same at the other end;
- a rung that could not reach 49 corners but was reported as a number rather than a
  **DECLARED NOT-RUN with the count it reached** would look like a measurement;
- ``scale`` or ``orientation`` leaking into a **VALUE** cell would quietly widen a
  hold-out the programme has ruled ineligible.

⛔ The positive controls (``test_..._fires_...``) prove the checks can fail.
"""

import json

import numpy as np
import pytest

from chlu.data import gen_kway_ladder as K
from chlu.data import gen_splits as G
from chlu.data import gen_value_holdout as V

NAMES = ("floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation")
LADDER_KS = (3, 4, 5)


@pytest.fixture(scope="module")
def grid():
    """The Shapes3D-shaped offline grid — same fixture as tests/test_gen_value_holdout."""
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
    return K.KwayLadderConfig(value_cfg=V.ValueHoldoutConfig(dataset="toy3d"))


@pytest.fixture(scope="module")
def cap(grid, cfg):
    return K.ladder_capacity_report(grid, cfg)


@pytest.fixture(scope="module")
def built(grid, cfg):
    """(split, cells, corners) per rung — built once, the slow part of the suite."""
    return {k: K.build_kway_split(grid, cfg, k) for k in (2, *LADDER_KS, 6)}


# ---------------------------------------------------------------------------
# 1. the kill check — it runs first and it can say NO
# ---------------------------------------------------------------------------


def test_kill_check_needs_no_cells_and_reports_every_rung(cap):
    assert set(cap["rungs"]) == {"2", "3", "4", "5", "6"}
    for k, r in cap["rungs"].items():
        assert r["bar"] == K.CORNER_BAR == 49
        assert r["max_corners"] == r["n_cells_capacity"] * r["corners_per_cell"]
        assert r["status"] in {"PASS", "NOT_RUN"}


def test_every_ladder_rung_clears_the_49_corner_bar(cap):
    """⭐ THE ACCEPTANCE CRITERION, on the shipped coordinate basis."""
    for k in LADDER_KS:
        r = cap["rungs"][str(k)]
        assert r["status"] == "PASS", r.get("not_run_reason")
        assert r["max_corners"] >= K.CORNER_BAR


def test_kill_check_declares_NOT_RUN_with_the_count_it_reached(grid):
    """⛔ Positive control: a basis too thin for 49 must yield NOT_RUN **and a count**,
    ⛔ never a zero, never a silent pass."""
    thin = G.FactorGrid(
        names=NAMES,
        values=(
            np.arange(4) / 10.0,  # floor_hue: 2 disjoint width-2 bands
            np.arange(4) / 10.0,
            np.arange(10) / 10.0,
            np.linspace(0.75, 1.25, 8),
            np.arange(2, dtype=float),  # one retained shape
            np.linspace(-30.0, 30.0, 15),
        ),
        dataset="thin3d",
    )
    cfg = K.KwayLadderConfig(
        value_cfg=V.ValueHoldoutConfig(dataset="thin3d", n_pair_cells=1)
    )
    rep = K.ladder_capacity_report(thin, cfg, ks=(3,))
    r = rep["rungs"]["3"]
    assert r["status"] == "NOT_RUN"
    assert 0 < r["max_corners"] < K.CORNER_BAR  # a COUNT, not a zero
    assert str(r["max_corners"]) in r["not_run_reason"]


# ---------------------------------------------------------------------------
# 2. the two ends — ⛔ if k=2 disagrees with row 3 the BUILDER is wrong
# ---------------------------------------------------------------------------


def test_k6_reproduces_row4_cell_for_cell_in_order(grid, cfg):
    rep = K.assert_k6_matches_row4(grid, cfg)
    assert rep["cell_values_identical"] and rep["order_identical"]
    assert rep["n_cells_compared"] == 120


def test_k6_reproduces_the_SHIPPED_49_nway_cells_specifically(grid, cfg):
    """The shipped row 4 is ``n_nway_cells = 49``; the rung's first 49 must be those."""
    pair = V.build_pair_cells(grid, cfg.value_cfg)
    shipped = V.build_nway_cells(grid, cfg.value_cfg, pair_cells=pair)
    assert len(shipped) == 49
    mine = K.build_kway_cells(grid, cfg, 6, pair_cells=pair)
    for a, b in zip(mine[:49], shipped, strict=True):
        assert {f: tuple(v) for f, v in a.values.items()} == {
            f: tuple(v) for f, v in b.values.items()
        }


def test_k2_reproduces_row3_subcorners_ids_and_rows(grid, cfg):
    rep = K.assert_k2_matches_row3(grid, cfg)
    assert rep["corner_ids_identical"] and rep["corner_row_sets_identical"]
    assert rep["n_corners_compared"] == 60  # the hub's row-3 count


def test_k2_end_reproduction_fires_when_the_tiling_widths_drift(grid, cfg):
    """⛔ Positive control: E2 must FAIL if the builder stops using row 3's widths."""
    import dataclasses as _dc

    drifted = _dc.replace(cfg, widths={**cfg.widths, "scale_corner": 2})
    with pytest.raises(AssertionError):
        K.assert_k2_matches_row3(grid, drifted)


# ---------------------------------------------------------------------------
# 3. the rungs — disjointness is PROVED on rows, not argued
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", LADDER_KS)
def test_rung_corners_are_pairwise_disjoint_on_rows(grid, built, k):
    _, _, corners = built[k]
    rep = K.assert_corners_disjoint(grid, corners, f"k{k}")
    assert rep["pairwise_disjoint"]
    assert rep["n_corners"] >= K.CORNER_BAR


@pytest.mark.parametrize("k", LADDER_KS)
def test_rung_corners_are_all_inside_the_rungs_own_holdout(grid, built, k):
    split, _, corners = built[k]
    n = K.assert_corners_inside_holdout(grid, corners, split.test_rows, f"k{k}")
    assert n == sum(int(c.rows(grid).size) for c in corners)


def test_disjointness_check_fires_on_overlapping_corners(grid, built):
    """⛔ Positive control: the proof must be able to fail."""
    _, _, corners = built[3]
    with pytest.raises(AssertionError, match="OVERLAP"):
        K.assert_corners_disjoint(grid, [corners[0], corners[0]], "dup")


def test_inside_holdout_check_fires_on_a_fit_pool_corner(grid, built):
    """⛔ Positive control: a corner sitting in the fit pool must be caught."""
    split, _, corners = built[3]
    with pytest.raises(AssertionError, match="NOT held out"):
        K.assert_corners_inside_holdout(
            grid, corners, split.test_rows[:10], "truncated"
        )


@pytest.mark.parametrize("k", LADDER_KS)
def test_every_rung_corner_is_the_row4_montero_shape(grid, built, k):
    """⭐ 72 rows: 2 floor x 2 wall x 2 object hues x 3 scales x 1 shape x 3 poses —
    the realised size the shapes3d module records as ``N_TEST_REALISED``."""
    _, _, corners = built[k]
    assert {int(c.rows(grid).size) for c in corners} == {72}


@pytest.mark.parametrize("k", LADDER_KS)
def test_rung_pins_exactly_k_factors_and_leaves_the_rest_free(grid, cfg, k):
    axes = cfg.axes_for(k)
    assert K.k_of_axes(axes) == k
    pair = V.build_pair_cells(grid, cfg.value_cfg)
    for cell in K.build_kway_cells(grid, cfg, k, pair_cells=pair):
        assert len(cell.values) == k
        assert len(set(grid.names) - set(cell.values)) == 6 - k


# ---------------------------------------------------------------------------
# 4. ⛔ the non-negotiables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", LADDER_KS)
def test_scale_and_orientation_never_become_a_VALUE_cell(grid, cfg, k):
    """⛔ Ineligible as VALUE hold-outs, eligible INSIDE a k-way box — exactly row 4."""
    value, pair, kway = K.build_kway_condition_cells(grid, cfg, k)
    for cell in value:
        assert not ({"scale", "orientation"} & set(cell.values)), cell.cell_id
    for cell in pair:
        assert not ({"scale", "orientation"} & set(cell.values)), cell.cell_id
    # ...and eligible inside the box: k >= 3 pins scale, exactly as row 4 does
    assert all("scale" in c.values for c in kway)


@pytest.mark.parametrize("k", LADDER_KS)
def test_kway_boxes_never_touch_a_held_out_or_pair_reserved_value(grid, cfg, k):
    pair = V.build_pair_cells(grid, cfg.value_cfg)
    reserved = {cfg.value_cfg.held_out_hue_value} | {
        float(v) for c in pair for v in c.values["object_hue"]
    }
    for cell in K.build_kway_cells(grid, cfg, k, pair_cells=pair):
        assert not (set(cell.values["object_hue"]) & reserved)
        assert cfg.value_cfg.held_out_shape_value not in cell.values["shape"]


@pytest.mark.parametrize("k", LADDER_KS)
def test_the_whole_rung_condition_is_disjoint_so_rows_1_2_5_7_survive(grid, built, k):
    """The rung swaps ``nway-*`` for ``kway{k}-*`` and leaves the value and pair cells
    standing, so the other taxonomy rows stay measurable on the same split."""
    _, cells, _ = built[k]
    assert G.assert_cells_disjoint(grid, cells) > 0
    fams = {c.meta["family"] for c in cells}
    assert {"value", "pair", f"kway{k}"} <= fams


@pytest.mark.parametrize("k", LADDER_KS)
def test_rung_still_serves_the_pinned_fit_pool(grid, built, k):
    """⛔ A rung whose train index cannot serve ``PINS['n_pool']`` silently changes the
    fit pool and is not comparable with the others."""
    split, _, _ = built[k]
    fp = K.fit_pool_feasibility(int(split.train_rows.size))
    assert fp["serves_pinned_pool"], fp


# ---------------------------------------------------------------------------
# 5. the emitted artifact
# ---------------------------------------------------------------------------


def test_ladder_ready_json_names_every_family_its_count_and_its_check(tmp_path, grid, cfg):
    payload = K.emit_ladder_ready(grid, tmp_path, cfg, log=lambda *a, **kw: None)
    on_disk = json.loads((tmp_path / "LADDER-READY.json").read_text(encoding="utf-8"))
    assert on_disk["families"].keys() == payload["families"].keys()
    assert on_disk["scored_runs_spent"] == 0
    for k in LADDER_KS:
        fam = on_disk["families"][f"k{k}"]
        assert fam["family"] == f"kway{k}-*"
        assert fam["n_corners"] >= K.CORNER_BAR
        assert fam["disjointness_check"]["pairwise_disjoint"] is True
        assert fam["all_corner_rows_held_out"] is True
    # the estimand is inherited, ⛔ not redefined
    assert on_disk["estimand"]["metric_rows_3_4_and_ladder"] == "distinct_valid_novel_tuples"
    assert on_disk["estimand"]["margin"] == 1.0
    assert on_disk["estimand"]["unit"] == "corner"
    # both end reproductions are recorded in the artifact itself
    assert on_disk["end_reproduction"]["E1_k6_vs_row4"]["cell_values_identical"]
    assert on_disk["end_reproduction"]["E2_k2_vs_row3"]["corner_row_sets_identical"]


def test_row_matching_is_reported_as_infeasible_with_both_sides_of_the_number(cap):
    """The confound is REPORTED with its arithmetic, ⛔ not corrected here."""
    rep = K.row_matching_feasibility(cap)
    assert rep["feasible"] is False
    assert rep["required_floor"] > rep["available_ceiling"]
    assert set(rep["min_rows_to_clear_bar_per_rung"]) == {"2", "3", "4", "5", "6"}
