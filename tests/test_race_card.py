"""Tests for the C2W2 race card (:mod:`chlu.eval.race`).

The race card is the **only** thing that makes two agents' numbers on two
branches comparable, and it is what the C2W2 gate is evaluated on. So the
properties asserted here are the ones that keep the gate both **fireable** and
**ungameable**:

* the dividend is **derived**, never stored — a cell cannot carry a number
  inconsistent with its own arms;
* an unconverged / saddled write is **inadmissible** and casts no <=0 vote
  (letting an unwritten store vote would fire B' on noise);
* **every** excluded cell is returned with its reason — silent filtering is the
  named failure mode (admissibility quietly gutting coverage until B' can never
  fire), so ``score_family`` must surface it;
* a family with zero admissible cells **abstains** (neither blocks B' nor
  supports "proceed");
* an inert term with **no perturbing anchor** in its grid grades
  ``under_powered_grid``, not a <=0 vote — *"a term that never moves anything at
  any tested setting hasn't been asked; it's been whispered at"*;
* clearing 2 SE while **losing to the +0 B substitute** is a ``weak_proceed``
  (charter §A6), and the signed margin is recorded, not argued;
* the trajectory launder firing makes the cell inadmissible (the psi is reading
  ``phi(x)``, not the store);
* the gate arithmetic is the charter's: sample sd ``ddof=1``, ``SE = sd/sqrt(n)``,
  "clears" iff ``mean - 2*SE > 0``.
"""

import math

import numpy as np
import pytest

from chlu.eval.race import (
    GRADE_ABSTAIN,
    GRADE_LE_ZERO_VOTE,
    GRADE_PROCEED,
    GRADE_UNDER_POWERED,
    GRADE_WEAK_PROCEED,
    RACE_SCHEMA_VERSION,
    ByteLedger,
    RaceCell,
    TrajectoryLaunder,
    coverage_table,
    gate_summary,
    load_cells,
    make_cell,
    save_cells,
    score_card,
    score_family,
    verdicts_to_markdown,
)

_LIVE = {"passed": True, "coefficient": 1.0, "value": 0.3, "baseline": 0.13,
         "bar": 0.2, "perturbing_anchor": True, "grid": (0.01, 0.1, 1.0)}
_OK_WRITE = {"steps": 300, "final_loss": 1e-3, "lambda_min_min": 0.02,
             "converged": True}


def _cell(seed=0, full=0.6, launder=0.5, sub=0.4, arm="traj_write",
          family="aggregate", route="route1", write=None, liveness=None, **kw):
    return make_cell(route, arm, family, seed, "decode", full=full,
                     settle_deleted_launder=launder, same_keys_null=0.2,
                     blank=0.13, plus_zero_byte_substitute=sub,
                     write=dict(write or _OK_WRITE),
                     liveness=dict(liveness or _LIVE), **kw)


# -- the cell ---------------------------------------------------------------
def test_dividend_and_substitute_margin_are_derived_not_stored():
    c = _cell(full=0.62, launder=0.50, sub=0.70)
    assert c.dividend == pytest.approx(0.12)
    assert c.substitute_margin == pytest.approx(-0.08)
    assert "dividend" not in {f for f in vars(c)}  # derived, so never inconsistent


def test_route_aliases_normalise_so_both_branches_are_comparable():
    assert _cell(route="route2_shell_atoms").route == "route2"
    assert _cell(route="R2").route == "route2"
    with pytest.raises(ValueError, match="unknown route"):
        _cell(route="route3")


def test_cell_round_trips_through_json_with_nan_as_null(tmp_path):
    c = _cell()
    c.full = float("nan")
    p = save_cells(tmp_path / "card.json", [c])
    txt = p.read_text()
    assert "NaN" not in txt and '"full": null' in txt
    back = load_cells(p)[0]
    assert math.isnan(back.full) and back.arm == c.arm
    assert back.write.steps == 300 and back.liveness.perturbing_anchor
    assert back.schema_version == RACE_SCHEMA_VERSION


# -- admissibility (gate ruling (i)) ----------------------------------------
def test_an_unconverged_write_is_inadmissible_and_says_why():
    c = _cell(write={"steps": 300, "final_loss": 0.22, "lambda_min_min": -0.21,
                     "converged": False})
    assert c.gate_admissible is False
    assert "write_not_converged" in c.exclusion_reason


def test_a_saddled_write_is_inadmissible_even_when_the_loss_converged():
    """The gym's multi-target ridge write: loss fine, ``lambda_min = -0.5946``."""
    c = _cell(write={"steps": 300, "final_loss": 1e-4,
                     "lambda_min_min": -0.5946, "converged": True})
    assert c.gate_admissible is False
    assert "lambda_min<0" in c.exclusion_reason


def test_the_trajectory_launder_firing_makes_the_cell_inadmissible():
    tl = TrajectoryLaunder(full=0.40, q0_only=0.31, endpoints=0.30,
                           blank_store=0.29, chance=0.125, bar=0.20)
    assert tl.fired() and tl.leak == pytest.approx(0.11)
    c = _cell(trajectory_launder=tl)
    assert c.gate_admissible is False
    assert "trajectory_launder_fired" in c.exclusion_reason


def test_a_clean_trajectory_launder_does_not_fire_at_the_c2w1_numbers():
    """C2W1 measured ``q0_only`` 0.129 vs chance 0.125 — that REFUTED the leak."""
    tl = TrajectoryLaunder(full=0.30, q0_only=0.129, endpoints=0.28,
                           blank_store=0.148, chance=0.125, bar=0.20)
    assert not tl.fired()
    assert tl.over_endpoints == pytest.approx(0.02)
    assert _cell(trajectory_launder=tl).gate_admissible is True


# -- the scorer -------------------------------------------------------------
def test_gate_arithmetic_is_sample_sd_ddof1_over_sqrt_n():
    vals = [0.30, 0.20, 0.40]
    cells = [_cell(seed=i, full=0.5 + v, launder=0.5) for i, v in enumerate(vals)]
    v = score_family(cells)
    assert v.n_admissible == 3 and v.seeds == [0, 1, 2]
    assert v.dividend_mean == pytest.approx(np.mean(vals))
    assert v.dividend_sd == pytest.approx(np.std(vals, ddof=1))
    assert v.dividend_se == pytest.approx(np.std(vals, ddof=1) / math.sqrt(3))
    assert v.clears_two_se is True  # 0.30 - 2*0.0577 > 0


def test_a_dividend_inside_two_se_does_not_clear():
    vals = [0.05, -0.04, 0.02]
    cells = [_cell(seed=i, full=0.5 + v, launder=0.5) for i, v in enumerate(vals)]
    v = score_family(cells)
    assert v.clears_two_se is False
    assert v.grade == GRADE_LE_ZERO_VOTE and v.votes_le_zero


def test_excluded_cells_are_reported_with_their_reason_never_filtered_silently():
    bad = {"steps": 300, "final_loss": 0.24, "lambda_min_min": -1.20,
           "converged": False}
    cells = [_cell(seed=0), _cell(seed=1, write=bad), _cell(seed=2, write=bad)]
    v = score_family(cells)
    assert (v.n_cells, v.n_admissible) == (3, 1)
    assert v.admissible_coverage == pytest.approx(1 / 3)
    assert {e["seed"] for e in v.excluded} == {1, 2}
    assert all(e["reason"] for e in v.excluded)


def test_zero_admissible_cells_abstains_and_casts_no_vote():
    bad = {"steps": 300, "final_loss": 0.20, "lambda_min_min": -0.9,
           "converged": False}
    v = score_family([_cell(seed=i, write=bad) for i in range(3)], escalated=True)
    assert v.grade == GRADE_ABSTAIN
    assert v.votes_le_zero is False
    assert v.escalated and v.admissible_coverage == 0.0


def test_an_inert_term_with_no_perturbing_anchor_is_an_under_powered_grid():
    """⛔ Without the anchor, 'the term does nothing' is not a legitimate vote."""
    dead = {"passed": False, "coefficient": 1e-3, "value": 0.13, "baseline": 0.13,
            "bar": 0.20, "perturbing_anchor": False, "grid": (1e-3,)}
    v = score_family([_cell(seed=i, full=0.5, launder=0.5, liveness=dead)
                      for i in range(3)])
    assert v.grade == GRADE_UNDER_POWERED and v.votes_le_zero is False
    # ...and WITH the anchor the same inert result DOES vote
    anchored = dict(dead, perturbing_anchor=True)
    v2 = score_family([_cell(seed=i, full=0.5, launder=0.5, liveness=anchored)
                       for i in range(3)])
    assert v2.grade == GRADE_LE_ZERO_VOTE and v2.votes_le_zero


def test_clearing_two_se_while_losing_to_the_plus_zero_byte_substitute_is_weak():
    """Charter §A6, pre-registered before adjudication."""
    strong = [_cell(seed=i, full=0.80 + 0.01 * i, launder=0.50, sub=0.40)
              for i in range(3)]
    assert score_family(strong).grade == GRADE_PROCEED
    weak = [_cell(seed=i, full=0.80 + 0.01 * i, launder=0.50, sub=0.95)
            for i in range(3)]
    v = score_family(weak)
    assert v.clears_two_se and v.grade == GRADE_WEAK_PROCEED
    assert v.substitute_margin_mean < 0


def test_byte_ledger_flags_the_architectural_ratio():
    """⛔ ``ratio >= 2.20`` is architectural (gym PREREG-B1) — never a
    byte-matched dividend."""
    led = ByteLedger(full=2200, launder=1000)
    assert led.ratio == pytest.approx(2.2) and led.architectural
    assert not led.matched
    assert ByteLedger(full=1000, launder=1000).matched


# -- the card ---------------------------------------------------------------
def test_score_card_groups_by_route_arm_family_and_summarises_coverage():
    cells = (
        [_cell(seed=i, arm="endpoint_write") for i in range(3)]
        + [_cell(seed=i, arm="traj_write", family="overload") for i in range(3)]
        + [_cell(seed=0, route="route2", arm="shell", family="manifold")]
    )
    verdicts = score_card(cells)
    assert len(verdicts) == 3
    cov = coverage_table(verdicts)
    assert set(cov) == {"aggregate", "overload", "manifold"}
    assert cov["overload"]["coverage"] == 1.0
    g = gate_summary(verdicts)
    assert g["routes_present"] == ["route1", "route2"]
    assert g["schema_version"] == RACE_SCHEMA_VERSION
    assert "arithmetic only" in g["note"]
    md = verdicts_to_markdown(verdicts)
    assert "| route |" in md and "endpoint_write" in md


def test_gate_summary_separates_abstentions_from_le_zero_votes():
    bad = {"steps": 300, "final_loss": 0.2, "lambda_min_min": -0.5,
           "converged": False}
    cells = (
        [_cell(seed=i, full=0.5, launder=0.5, family="aggregate") for i in range(3)]
        + [_cell(seed=i, family="recency", write=bad) for i in range(3)]
        + [_cell(seed=i, full=0.9 + 0.005 * i, launder=0.5, family="manifold")
           for i in range(3)]
    )
    g = gate_summary(score_card(cells))
    assert g["any_family_clears"] is True
    assert any("manifold" in k for k in g["cleared_two_se"])
    assert any("aggregate" in k for k in g["le_zero_votes"])
    assert any("recency" in k for k in g["abstained"])
    assert not set(g["abstained"]) & set(g["le_zero_votes"])


def test_explicit_gate_admissible_false_still_carries_a_reason():
    c = RaceCell(route="route1", arm="a", family="overload", seed=0,
                 metric_name="decode", full=0.5, settle_deleted_launder=0.5,
                 gate_admissible=False)
    c.resolve_admissibility()
    assert c.exclusion_reason == "marked_inadmissible"
